// =============================================================================
// Cardinal — Networking: Replicator (snapshot serialize / ingest).
//
// Pure little-endian byte packing over the transport's Unreliable
// channel. Zero-dep, same ethos as the rest of net. Seq-gated on the
// client so a reordered/duplicate datagram can never apply an older
// frame than one already shown.
// =============================================================================

#include <cardinal/net/replication.hpp>

#include <cardinal/core/std/cstring.hpp>      // cardinal::memcpy
#include <cardinal/core/std/utility.hpp>      // cardinal::move
#include <cardinal/core/std/containers.hpp>   // cardinal::vector

namespace cardinal::net {

namespace {

// 'RPL1' — bumped from 'RPL0' when state went quantized (28 B/entity).
// A stale mixed build sees the magic mismatch and drops the datagram as
// foreign rather than misparsing 40-byte records as 28-byte ones.
constexpr u32 kSnapMagic = 0x52504C31;   // 'RPL1'
// Lifecycle-event datagram tag (ReliableOrdered). Distinct from the
// snapshot magic so a client can tell the two streams apart even if a
// transport ever blurs the channel id.
constexpr u32 kEvtMagic  = 0x52504531;   // 'RPE1'

// Quantization ranges. Position stays full f32 (the world is
// unbounded — can't pick a safe fixed scale). Rotation-euler is
// bounded ±π; kRotRange > π so in-range values never clamp, ~1e-4 rad
// resolution. Scale is small and non-negative; [0,16] at ~2.4e-4
// resolution. Both far exceed any visible threshold for streamed
// transforms — lossy on the wire, lossless to the eye.
constexpr float kRotRange   = 3.2f;     // signed, > π
constexpr float kScaleRange = 16.0f;    // unsigned, [0, kScaleRange]

template <class T>
void put(cardinal::vector<u8>& b, const T& v) {
    const auto* p = reinterpret_cast<const u8*>(&v);
    b.insert(b.end(), p, p + sizeof(T));
}
template <class T>
bool get(const u8*& p, const u8* end, T& out) {
    if (p + sizeof(T) > end) return false;
    cardinal::memcpy(&out, p, sizeof(T));
    p += sizeof(T);
    return true;
}

// Symmetric round-to-nearest with no <cmath> dependency (foundation
// discipline — the only native header here is the pre-existing
// <cstring>). [-range,range] ⇄ i16, [0,range] ⇄ u16.
i16 enc_s16(float v, float range) {
    float n = v / range;
    if (n >  1.0f) n =  1.0f;
    if (n < -1.0f) n = -1.0f;
    const float q = n * 32767.0f;
    return static_cast<i16>(q >= 0.0f ? q + 0.5f : q - 0.5f);
}
float dec_s16(i16 q, float range) {
    return (static_cast<float>(q) / 32767.0f) * range;
}
u16 enc_u16(float v, float range) {
    float n = v / range;
    if (n > 1.0f) n = 1.0f;
    if (n < 0.0f) n = 0.0f;
    return static_cast<u16>(n * 65535.0f + 0.5f);
}
float dec_u16(u16 q, float range) {
    return (static_cast<float>(q) / 65535.0f) * range;
}

// Pure-arithmetic isfinite without <cmath> (foundation discipline above).
// finite ↔ (v - v) == 0.0f:
//   finite  → 0 - 0 = 0     → true
//   NaN     → NaN - NaN = NaN → NaN == 0 is false
//   ±Inf    → Inf - Inf = NaN → false
inline float finite_or_zero(float v) noexcept {
    return ((v - v) == 0.0f) ? v : 0.0f;
}

// Wire record: u32 id + 3×f32 pos + 3×i16 rot + 3×u16 scale = 28 B.
// Every float is sanitized here so the wire NEVER carries NaN/±Inf:
//   * raw f32 position would survive transit unmodified and poison the
//     client's lerp3 (`a + (b - a) * alpha` with ±Inf → NaN-vector,
//     teleporting every networked proxy to NaN-land);
//   * enc_s16 / enc_u16 had NaN-blind clamps (NaN > 1.0f and NaN < -1.0f
//     both false), so a NaN rotation/scale reached `static_cast<intN>
//     (NaN * 32767/65535)` — UNDEFINED BEHAVIOR for the float→int cast,
//     producing implementation-defined garbage on the wire.
// Sanitize at the encoder so every downstream consumer (this client and
// any future replicator) sees finite values.
void put_state(cardinal::vector<u8>& b, const RepState& s) {
    put(b, s.id);
    put(b, finite_or_zero(s.position.x));
    put(b, finite_or_zero(s.position.y));
    put(b, finite_or_zero(s.position.z));
    put(b, enc_s16(finite_or_zero(s.rotation_euler.x), kRotRange));
    put(b, enc_s16(finite_or_zero(s.rotation_euler.y), kRotRange));
    put(b, enc_s16(finite_or_zero(s.rotation_euler.z), kRotRange));
    put(b, enc_u16(finite_or_zero(s.scale.x), kScaleRange));
    put(b, enc_u16(finite_or_zero(s.scale.y), kScaleRange));
    put(b, enc_u16(finite_or_zero(s.scale.z), kScaleRange));
}
bool get_state(const u8*& p, const u8* end, RepState& s) {
    i16 rx = 0, ry = 0, rz = 0;
    u16 sx = 0, sy = 0, sz = 0;
    if (!(get(p, end, s.id) &&
          get(p, end, s.position.x) && get(p, end, s.position.y) &&
          get(p, end, s.position.z) &&
          get(p, end, rx) && get(p, end, ry) && get(p, end, rz) &&
          get(p, end, sx) && get(p, end, sy) && get(p, end, sz)))
        return false;
    s.rotation_euler.x = dec_s16(rx, kRotRange);
    s.rotation_euler.y = dec_s16(ry, kRotRange);
    s.rotation_euler.z = dec_s16(rz, kRotRange);
    s.scale.x = dec_u16(sx, kScaleRange);
    s.scale.y = dec_u16(sy, kScaleRange);
    s.scale.z = dec_u16(sz, kScaleRange);
    return true;
}

// Event record: u8 kind + u32 id + u32 archetype + 28 B state = 37 B.
// Fixed-width (the transform is sent even for Despawn) keeps the parser
// trivial and robust; lifecycle events are rare, the few spare bytes
// are irrelevant next to the snapshot stream.
void put_event(cardinal::vector<u8>& b, const RepEvent& ev) {
    put(b, static_cast<u8>(ev.kind));
    put(b, ev.id);
    put(b, ev.archetype);
    put_state(b, ev.state);
}
bool get_event(const u8*& p, const u8* end, RepEvent& ev) {
    u8 k = 0;
    if (!get(p, end, k) || !get(p, end, ev.id) ||
        !get(p, end, ev.archetype) || !get_state(p, end, ev.state))
        return false;
    ev.kind = (k == static_cast<u8>(RepEventKind::Despawn))
                  ? RepEventKind::Despawn
                  : RepEventKind::Spawn;
    return true;
}

}  // namespace

void Replicator::server_broadcast(const cardinal::vector<RepState>& states) {
    cardinal::vector<u8> buf;
    buf.reserve(10 + states.size() * 28);   // 10 B header + 28 B/entity
    const u32 seq   = ++out_seq_;
    const u16 count = static_cast<u16>(
        states.size() > 0xFFFF ? 0xFFFF : states.size());
    put(buf, kSnapMagic);
    put(buf, seq);
    put(buf, count);
    for (u16 i = 0; i < count; ++i) put_state(buf, states[i]);

    snap_bytes_ = static_cast<u32>(buf.size());
    t_.broadcast(Channel::Unreliable, buf.data(), snap_bytes_);
    ++sent_;
}

void Replicator::server_events(const cardinal::vector<RepEvent>& evs) {
    if (evs.empty()) return;                      // nothing to send
    cardinal::vector<u8> buf;
    buf.reserve(6 + evs.size() * 37);             // 6 B header + 37 B/evt
    const u16 count = static_cast<u16>(
        evs.size() > 0xFFFF ? 0xFFFF : evs.size());
    put(buf, kEvtMagic);
    put(buf, count);
    for (u16 i = 0; i < count; ++i) put_event(buf, evs[i]);

    // ReliableOrdered: guaranteed, in order — the channel IS the
    // delivery contract, so there's no seq/ack bookkeeping here.
    t_.broadcast(Channel::ReliableOrdered, buf.data(),
                 static_cast<u32>(buf.size()));
    evt_sent_ += count;
}

cardinal::usize Replicator::client_events(
        const cardinal::vector<NetEvent>& events, cardinal::vector<RepEvent>& out) {
    const cardinal::usize before = out.size();
    for (const auto& e : events) {
        if (e.kind != NetEventKind::Message) continue;
        if (e.data.size() < 6) continue;          // need magic + count
        const u8* p   = e.data.data();
        const u8* end = p + e.data.size();
        u32 magic = 0;
        u16 count = 0;
        if (!get(p, end, magic) || !get(p, end, count)) continue;
        if (magic != kEvtMagic) continue;         // snapshot / foreign
        for (u16 i = 0; i < count; ++i) {
            RepEvent ev;
            if (!get_event(p, end, ev)) break;    // truncated datagram
            out.push_back(ev);
        }
    }
    const cardinal::usize n = out.size() - before;
    evt_recv_ += static_cast<u64>(n);
    return n;
}

cardinal::usize Replicator::client_ingest(
        const cardinal::vector<NetEvent>& events, cardinal::vector<RepState>& out) {
    // Pick the newest valid snapshot across this poll (Unreliable can
    // reorder/duplicate — only the freshest frame matters).
    const NetEvent* best = nullptr;
    u32 best_seq = last_seq_;
    for (const auto& e : events) {
        if (e.kind != NetEventKind::Message) continue;
        if (e.data.size() < 10) continue;
        const u8* p = e.data.data();
        u32 magic = 0, seq = 0;
        cardinal::memcpy(&magic, p, 4);
        cardinal::memcpy(&seq,   p + 4, 4);
        if (magic != kSnapMagic) continue;          // host's own traffic
        // Strictly newer than the last applied (and the best so far).
        if (seq > best_seq) { best_seq = seq; best = &e; }
    }
    if (best == nullptr) return 0;

    const u8* p   = best->data.data();
    const u8* end = p + best->data.size();
    u32 magic = 0, seq = 0;
    u16 count = 0;
    if (!get(p, end, magic) || !get(p, end, seq) || !get(p, end, count))
        return 0;
    out.clear();
    out.reserve(count);
    for (u16 i = 0; i < count; ++i) {
        RepState s;
        if (!get_state(p, end, s)) break;           // truncated datagram
        out.push_back(s);
    }
    last_seq_ = seq;
    ++recv_;
    return out.size();
}

namespace {

// Decode one snapshot datagram → (seq, states). false if not a valid
// replication snapshot (foreign traffic / truncated).
bool decode_snapshot(const NetEvent& e, u32& seq,
                     cardinal::vector<RepState>& states) {
    if (e.kind != NetEventKind::Message || e.data.size() < 10) return false;
    const u8* p   = e.data.data();
    const u8* end = p + e.data.size();
    u32 magic = 0;
    u16 count = 0;
    if (!get(p, end, magic) || !get(p, end, seq) || !get(p, end, count))
        return false;
    if (magic != kSnapMagic) return false;
    states.clear();
    states.reserve(count);
    for (u16 i = 0; i < count; ++i) {
        RepState s;
        if (!get_state(p, end, s)) break;     // truncated → take what we got
        states.push_back(s);
    }
    return true;
}

cardinal::scene::Vec3 lerp3(const cardinal::scene::Vec3& a,
                            const cardinal::scene::Vec3& b, float t) {
    return { a.x + (b.x - a.x) * t,
             a.y + (b.y - a.y) * t,
             a.z + (b.z - a.z) * t };
}

}  // namespace

cardinal::usize Replicator::decode_into_history_(
        const cardinal::vector<NetEvent>& events, double now) {
    cardinal::usize buffered = 0;
    for (const auto& e : events) {
        u32 seq = 0;
        cardinal::vector<RepState> states;
        if (!decode_snapshot(e, seq, states)) continue;
        if (seq <= buf_seq_) continue;        // stale / duplicate
        // Insert keeping history_ ascending by seq.
        Snap snap;
        snap.seq    = seq;
        snap.t      = now;
        snap.states = cardinal::move(states);
        auto it = history_.begin();
        while (it != history_.end() && it->seq < seq) ++it;
        history_.insert(it, cardinal::move(snap));
        buf_seq_ = seq;
        ++recv_;
        ++buffered;
    }
    // Cap: drop oldest so the buffer can't grow unbounded.
    while (history_.size() > kHistoryMax) history_.erase(history_.begin());
    return buffered;
}

cardinal::usize Replicator::client_buffer(
        const cardinal::vector<NetEvent>& events, double now) {
    return decode_into_history_(events, now);
}

cardinal::usize Replicator::client_sample(
        double render_time, cardinal::vector<RepState>& out) {
    out.clear();
    if (history_.empty()) return 0;

    // Clamp before the oldest / after the newest buffered snapshot
    // (no extrapolation — stable under packet loss).
    if (render_time <= history_.front().t) {
        out = history_.front().states;
        return out.size();
    }
    // A non-finite render_time (NaN from a bad host clock / interp_delay)
    // makes BOTH ordered clamp guards false — NaN compares unordered — so
    // with one buffered snapshot the a/b pair below reads history_[1] OUT
    // OF BOUNDS, and with >=2 it lerps with alpha = NaN, teleporting every
    // proxy to NaN. (+/-Inf is ordered and already handled by these
    // clamps.) Route NaN into the "past newest" clamp: hold the freshest
    // buffered snapshot — stable, in-gamut, never OOB. (x != x iff NaN;
    // this TU is deliberately <cmath>-free.)
    if (render_time != render_time || render_time >= history_.back().t) {
        out = history_.back().states;
        return out.size();
    }
    // Find adjacent pair a (t ≤ render_time) and b (t > render_time).
    cardinal::usize bi = 1;
    while (bi < history_.size() && history_[bi].t <= render_time) ++bi;
    const Snap& a = history_[bi - 1];
    const Snap& b = history_[bi];
    const double span = b.t - a.t;
    const float  alpha = (span > 1e-9)
        ? static_cast<float>((render_time - a.t) / span) : 0.0f;

    // b defines membership; lerp from a's matching id when present.
    out.reserve(b.states.size());
    for (const RepState& sb : b.states) {
        const RepState* sa = nullptr;
        for (const RepState& c : a.states)
            if (c.id == sb.id) { sa = &c; break; }
        RepState r = sb;
        if (sa) {
            // Component-wise lerp. Euler-lerp is fine for the small
            // per-frame deltas typical of streamed transforms; quat
            // slerp is a later refinement if large spins need it.
            r.position       = lerp3(sa->position,       sb.position,       alpha);
            r.rotation_euler = lerp3(sa->rotation_euler, sb.rotation_euler, alpha);
            r.scale          = lerp3(sa->scale,          sb.scale,          alpha);
        }
        out.push_back(r);
    }
    return out.size();
}

}  // namespace cardinal::net
