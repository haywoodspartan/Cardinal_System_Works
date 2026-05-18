#pragma once

// =============================================================================
// Cardinal — Networking (transport layer).
//
// Greenfield, perf-first, SP→MMO trajectory. The cornerstone design
// decision baked in here: singleplayer and multiplayer run the SAME
// netcode path. SP uses an in-process LoopbackTransport (no sockets,
// deterministic); MMO swaps in a UDP transport — gameplay/replication
// code above this layer never branches on "is multiplayer". This is the
// Quake/Source model and it's why a transport abstraction (not raw
// sockets) is the right foundation.
//
// Model: authoritative client-server. Channels are the standard game-net
// pair — Unreliable (state snapshots: drop, don't stall) and Reliable-
// Ordered (events/RPCs: must arrive, in order). TCP is intentionally
// absent (head-of-line blocking is wrong for games).
//
// This first landing ships the interface + a complete LoopbackTransport.
// The UDP backend (Winsock/BSD sockets, hand-rolled reliability over
// UDP — same no-deps ethos as the hand-rolled physics/audio) and the
// snapshot/delta replication layer are follow-on arcs. Nothing consumes
// this yet, so it is a pure, build-clean, zero-behaviour-change add.
// =============================================================================

#include <cardinal/core/types.hpp>   // foundation: unique_ptr / u32 / …

#include <vector>

namespace cardinal::net {

// 0 is never a valid peer — it's the "no peer" sentinel.
using PeerId = u32;
inline constexpr PeerId kInvalidPeer = 0;

enum class Channel : u32 {
    Unreliable     = 0,   // snapshots / state — newest wins, loss tolerated
    ReliableOrdered = 1,  // events / RPCs — guaranteed, in order
};

enum class NetEventKind : u32 {
    Connected,            // a peer (dis)appeared — `peer` is valid
    Disconnected,
    Message,              // `data` carries the payload on `channel`
};

struct NetEvent {
    NetEventKind    kind{NetEventKind::Message};
    PeerId          peer{kInvalidPeer};
    Channel         channel{Channel::Unreliable};
    std::vector<u8> data;
};

// Artificial adverse link conditions for the in-process loopback. A
// perfect loopback can't exercise (or prove) the netcode's loss /
// jitter / reorder tolerance — interpolation, seq-gating, disposable
// snapshots all exist for adversity that loopback never produces. This
// injects it deterministically: time is counted in poll()s (the
// loopback is clock-free by design — no <chrono>, reproducible from
// `seed`). Affects ONLY Unreliable Messages (snapshots); Reliable-
// Ordered keeps its guarantee and Connected/Disconnected are never
// perturbed. All-zero (the default) is a perfect link → byte-for-byte
// the legacy behaviour, so nothing changes unless explicitly enabled.
struct NetConditions {
    u32   latency_polls{0};      // delivery delay in poll()s (0 = none)
    u32   jitter_polls{0};       // ± uniform spread on that delay
    float loss{0.0f};            // Unreliable drop probability, [0,1]
    u32   seed{0x9E3779B9u};     // PRNG seed (deterministic/reproducible)
};

// ---------------------------------------------------------------------------
// Transport — one network endpoint (a server OR a client). Gameplay
// pumps it once per tick: send()/broadcast() to push, poll() to drain
// inbound events. Concrete kinds are created via the factories; callers
// hold the base pointer so the same code drives loopback or UDP.
// ---------------------------------------------------------------------------
class Transport {
public:
    virtual ~Transport() = default;
    Transport(const Transport&)            = delete;
    Transport& operator=(const Transport&) = delete;

    // In-process, socket-free, deterministic. Used for singleplayer and
    // for replication/gameplay tests before the UDP backend exists.
    static cardinal::unique_ptr<Transport> create_loopback();
    // Real UDP transport — stub (returns nullptr) until the backend
    // arc lands; the call site already handles "no transport" so this
    // is safe to ship now.
    static cardinal::unique_ptr<Transport> create_udp();

    // Role setup. listen() = become a server; connect() = become a
    // client and join `host:port`. For loopback the address args are
    // ignored — connect() synthesises an immediate Connected event.
    virtual bool listen(u16 port)                       = 0;
    virtual bool connect(const char* host, u16 port)    = 0;

    // Push. send() targets one peer; broadcast() hits every peer.
    virtual void send(PeerId to, Channel ch,
                       const void* data, u32 size)      = 0;
    virtual void broadcast(Channel ch,
                           const void* data, u32 size)  = 0;

    // Drain every event since the last poll into `out` (appended).
    // Returns the number appended.
    virtual usize poll(std::vector<NetEvent>& out)      = 0;

    virtual void  disconnect(PeerId peer)               = 0;
    virtual bool  is_server()  const noexcept           = 0;
    virtual usize peer_count() const noexcept           = 0;

    // Inject artificial link conditions (loopback only). Default no-op:
    // real transports have real conditions, and transports that don't
    // model them simply ignore it. Safe to call live, every frame.
    virtual void  set_conditions(const NetConditions&) noexcept {}

protected:
    Transport() = default;
};

}  // namespace cardinal::net
