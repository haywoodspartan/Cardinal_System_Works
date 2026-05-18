#pragma once

// =============================================================================
// Cardinal — Networking: server-authoritative state replication.
//
// Sits on top of the transport (loopback for SP, UDP for MMO — the
// SAME netcode path, the cornerstone of this engine's net design). The
// server gathers the authoritative state of the entities it owns and
// broadcasts a compact snapshot each tick on the Unreliable channel
// (newest-wins: snapshots are disposable, never stall the sim); clients
// decode the newest snapshot and hand the host the authoritative states
// to apply onto local proxies.
//
// Replicator itself is pure serialize/transport — it knows nothing
// about the scene graph. The host owns the entity↔RepState mapping
// (gather on the server, apply on the client), so this stays a thin,
// testable layer that works identically over loopback and UDP.
//
// Snapshot wire layout (little-endian, matches the pack/cook format
// convention): [u32 magic 'RPL1'][u32 seq][u16 count][rec×count].
// Wire record = u32 id + 3×f32 pos + 3×i16 rot-euler + 3×u16 scale
// = 28 bytes (was 40 — rotation/scale are quantized; position keeps
// full f32 since the world is unbounded). RepState in memory stays
// all-f32, so engine code is unaffected — only the wire shrinks 30%.
// Disposable + seq-gated: every snapshot is self-contained (no acked
// baseline), so a reordered/duplicate/lost datagram never applies a
// stale frame and never desyncs — quantization preserves that.
// =============================================================================

#include <cardinal/net/net.hpp>          // Transport / NetEvent / Channel
#include <cardinal/scene/math.hpp>       // scene::Vec3
#include <cardinal/core/types.hpp>       // foundation vocab
#include <cardinal/core/containers.hpp>  // cardinal::vector

namespace cardinal::net {

// One replicated entity's authoritative transform. The host maps its
// own entity ids ⇄ this; Replicator only (de)serialises it.
struct RepState {
    u32                   id{0};
    cardinal::scene::Vec3 position{0, 0, 0};
    cardinal::scene::Vec3 rotation_euler{0, 0, 0};   // radians
    cardinal::scene::Vec3 scale{1, 1, 1};
};

// Lifecycle events ride the ReliableOrdered channel — the complement to
// the disposable Unreliable snapshot stream. Snapshots answer "where is
// everything now" (lossy, newest-wins); events answer "what entities
// exist" (guaranteed, in order). Without this a freshly-snapshotted id
// has unknown identity (mesh/material/archetype), and a removed id just
// stops appearing — the proxy would freeze forever. `archetype` is
// host-defined (e.g. a mesh/material kind enum); the Replicator never
// interprets it. Spawn carries the initial transform; Despawn ignores
// it. Reliable+ordered ⇒ no seq-gating/dedup here (the channel is the
// guarantee; the loopback honours it and the link-sim never perturbs
// ReliableOrdered).
enum class RepEventKind : u8 { Spawn = 1, Despawn = 2 };

struct RepEvent {
    RepEventKind kind{RepEventKind::Spawn};
    u32          id{0};
    u32          archetype{0};   // host-defined; 0 for Despawn
    RepState     state;          // initial transform (Spawn only)
};

class Replicator {
public:
    explicit Replicator(Transport& transport) : t_(transport) {}

    // SERVER: serialise `states` into one snapshot and broadcast it on
    // the Unreliable channel to every connected peer. Call once per
    // network tick with the current authoritative states.
    void server_broadcast(const cardinal::vector<RepState>& states);

    // CLIENT: scan polled transport events for replication snapshots,
    // keep only the newest (by seq), decode it into `out`. Returns the
    // number of states written (0 if no fresher snapshot this poll).
    // Non-replication events in `events` are ignored (left for the
    // host's own message handling).
    cardinal::usize client_ingest(const cardinal::vector<NetEvent>& events,
                                  cardinal::vector<RepState>& out);

    // ---- Client-side interpolation -----------------------------------
    //
    // Snapping straight to the newest snapshot is choppy when snapshots
    // arrive slower than the frame rate (or with jitter). The standard
    // fix: buffer recent snapshots stamped with their arrival time, and
    // RENDER slightly in the past — between the two snapshots bracketing
    // (now - delay) — so motion is smooth and resilient to a missed or
    // reordered packet.
    //
    // The Replicator stays clock-free: the host passes `now` (monotonic
    // seconds, any epoch) so there's no <chrono> dependency here and
    // tests can drive a deterministic clock.
    //
    //   each poll:  client_buffer(events, now);
    //   each frame: client_sample(now - interp_delay, out);
    //
    // interp_delay ≈ 1.5–2 snapshot intervals (e.g. 0.1s at 10–20 Hz).
    // client_buffer/sample and client_ingest are independent paths;
    // pick snap (ingest) OR interpolation (buffer+sample) per use.
    cardinal::usize client_buffer(const cardinal::vector<NetEvent>& events,
                                  double now);
    cardinal::usize client_sample(double render_time,
                                  cardinal::vector<RepState>& out);

    // ---- Reliable lifecycle events -----------------------------------
    //
    // SERVER: serialise `evs` into one datagram on the ReliableOrdered
    // channel. Call when entities spawn/despawn (not every tick — these
    // are rare relative to snapshots). Empty `evs` is a no-op.
    //
    // CLIENT: decode lifecycle events from polled transport events into
    // `out` (appended, in arrival order). Returns the number appended.
    // Snapshot datagrams in `events` are ignored (the snapshot paths
    // handle those); the two streams are independent and composable —
    // a host typically runs client_events + (client_ingest OR
    // client_buffer/sample) each poll. Host policy owns idempotency
    // (re-Spawn an existing id / Despawn an unknown id).
    void server_events(const cardinal::vector<RepEvent>& evs);
    cardinal::usize client_events(const cardinal::vector<NetEvent>& events,
                                  cardinal::vector<RepEvent>& out);

    u64 snapshots_sent() const noexcept { return sent_; }
    u64 snapshots_recv() const noexcept { return recv_; }
    u32 last_seq()       const noexcept { return last_seq_; }
    // Byte size of the most recent broadcast datagram (header +
    // quantized records) — lets a host show real wire bandwidth.
    u32 last_snapshot_bytes() const noexcept { return snap_bytes_; }
    u64 events_sent() const noexcept { return evt_sent_; }
    u64 events_recv() const noexcept { return evt_recv_; }
    cardinal::usize history_size() const noexcept {
        return history_.size();
    }

private:
    // One buffered snapshot: sequence, arrival time, decoded states.
    struct Snap {
        u32                   seq{0};
        double                t{0.0};
        cardinal::vector<RepState> states;
    };
    // Decode every replication snapshot in `events` newer than the last
    // buffered seq, append to history (kept sorted by seq, capped).
    cardinal::usize decode_into_history_(const cardinal::vector<NetEvent>& events,
                                         double now);

    Transport& t_;
    u32 out_seq_{0};      // server: next snapshot sequence
    u32 last_seq_{0};     // client: highest snapshot seq applied (snap path)
    u32 buf_seq_{0};      // client: highest snapshot seq buffered (interp)
    u64 sent_{0};
    u64 recv_{0};
    u32 snap_bytes_{0};   // size of the last broadcast datagram
    u64 evt_sent_{0};     // lifecycle events sent / received (counts)
    u64 evt_recv_{0};

    static constexpr cardinal::usize kHistoryMax = 16;
    cardinal::vector<Snap> history_;   // ascending seq, ≤ kHistoryMax
};

}  // namespace cardinal::net
