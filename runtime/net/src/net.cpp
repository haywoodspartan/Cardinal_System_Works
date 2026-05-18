// =============================================================================
// Cardinal — Networking: LoopbackTransport + UDP factory stub.
//
// LoopbackTransport is the singleplayer / test path: no sockets, no
// threads, fully deterministic, pumped from the game thread. It models
// one local endpoint wired to a single virtual peer (the in-proc
// "other side"). connect() synthesises an immediate Connected; bytes
// sent to the virtual peer echo back on the next poll() as inbound
// Messages — exactly the shape real netcode sees, so gameplay /
// replication can be built + tested against this before the UDP
// backend lands.
//
// It can also inject deterministic adverse conditions (NetConditions:
// latency / jitter / loss, counted in poll()s, hand-rolled PRNG) so the
// loss-tolerant netcode above it — snapshot interpolation, seq-gating —
// is actually exercised and demonstrable. Off by default: a perfect
// link, byte-for-byte the legacy behaviour.
//
// A true two-instance loopback broker (separate server & client objects
// sharing queues) and the real UDP transport are follow-on arcs.
// =============================================================================

#include <cardinal/net/net.hpp>
#include <cardinal/core/log.hpp>

#include <cardinal/core/cstring.hpp>      // cardinal::memcpy
#include <cardinal/core/containers.hpp>   // cardinal::deque/vector
#include <cardinal/core/utility.hpp>      // cardinal::move

namespace cardinal::net {

namespace {

constexpr PeerId kVirtualPeer = 1;   // the in-proc "other side"

class LoopbackTransport final : public Transport {
public:
    bool listen(u16 /*port*/) override {
        is_server_ = true;
        // A loopback server has its single virtual client present
        // immediately — SP runs server+client in one process.
        connected_ = true;
        inbox_.push_back(make_evt(NetEventKind::Connected, kVirtualPeer,
                                  Channel::ReliableOrdered, nullptr, 0));
        return true;
    }

    bool connect(const char* /*host*/, u16 /*port*/) override {
        is_server_ = false;
        connected_ = true;
        inbox_.push_back(make_evt(NetEventKind::Connected, kVirtualPeer,
                                  Channel::ReliableOrdered, nullptr, 0));
        return true;
    }

    void send(PeerId to, Channel ch, const void* data, u32 size) override {
        if (!connected_ || to != kVirtualPeer) return;
        NetEvent e = make_evt(NetEventKind::Message, kVirtualPeer,
                              ch, data, size);
        // Perfect link (default) OR a channel that must not be
        // perturbed → straight back, deterministic and in-order,
        // exactly as the legacy loopback. ReliableOrdered keeps its
        // guarantee; only Unreliable models adverse conditions.
        if (!conditions_active() || ch != Channel::Unreliable) {
            inbox_.push_back(cardinal::move(e));
            return;
        }
        // Unreliable under injected conditions: roll loss, then queue
        // with a randomised poll-count delay (jitter naturally yields
        // reordering — which is the point: it exercises seq-gating).
        if (cond_.loss > 0.0f && rng_unit() < cond_.loss) return;  // dropped
        int d = static_cast<int>(cond_.latency_polls) +
                rng_jitter(cond_.jitter_polls);
        if (d < 0) d = 0;
        delayed_.push_back(Delayed{cardinal::move(e), static_cast<u32>(d)});
    }

    void broadcast(Channel ch, const void* data, u32 size) override {
        if (connected_) send(kVirtualPeer, ch, data, size);
    }

    usize poll(cardinal::vector<NetEvent>& out) override {
        const usize before = out.size();
        // Release Unreliable messages whose delay has elapsed; tick the
        // rest down one poll. delayed_ is empty unless conditions are
        // active, so the default path is unchanged.
        if (!delayed_.empty()) {
            cardinal::vector<Delayed> keep;
            keep.reserve(delayed_.size());
            for (auto& d : delayed_) {
                if (d.left == 0) out.push_back(cardinal::move(d.evt));
                else { --d.left; keep.push_back(cardinal::move(d)); }
            }
            delayed_.swap(keep);
        }
        while (!inbox_.empty()) {
            out.push_back(cardinal::move(inbox_.front()));
            inbox_.pop_front();
        }
        return out.size() - before;
    }

    void disconnect(PeerId peer) override {
        if (!connected_ || peer != kVirtualPeer) return;
        connected_ = false;
        inbox_.push_back(make_evt(NetEventKind::Disconnected, kVirtualPeer,
                                  Channel::ReliableOrdered, nullptr, 0));
    }

    bool  is_server()  const noexcept override { return is_server_; }
    usize peer_count() const noexcept override { return connected_ ? 1u : 0u; }

    void set_conditions(const NetConditions& c) noexcept override {
        cond_ = c;
        // (Re)seed only when the seed actually changes, so calling this
        // every frame from a UI doesn't correlate the jitter/loss draws.
        if (c.seed != seeded_) {
            rng_ = c.seed ? c.seed : 0x9E3779B9u;
            seeded_ = c.seed;
        }
    }

private:
    static NetEvent make_evt(NetEventKind k, PeerId p, Channel ch,
                             const void* data, u32 size) {
        NetEvent e;
        e.kind    = k;
        e.peer    = p;
        e.channel = ch;
        if (data != nullptr && size > 0) {
            e.data.resize(size);
            cardinal::memcpy(e.data.data(), data, size);
        }
        return e;
    }

    // One Unreliable message held back `left` more poll()s.
    struct Delayed { NetEvent evt; u32 left{0}; };

    bool conditions_active() const noexcept {
        return cond_.latency_polls != 0 || cond_.jitter_polls != 0 ||
               cond_.loss > 0.0f;
    }
    // Hand-rolled xorshift32 — deterministic, seedable, no <random>
    // (same no-deps ethos as the rest of net; foundation discipline).
    u32 rng_next() noexcept {
        u32 x = rng_;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        return rng_ = x;
    }
    float rng_unit() noexcept {       // [0, 1)
        return static_cast<float>(rng_next() >> 8) *
               (1.0f / 16777216.0f);
    }
    int rng_jitter(u32 j) noexcept {  // uniform in [-j, +j]
        if (j == 0) return 0;
        const u32 span = 2u * j + 1u;
        return static_cast<int>(rng_next() % span) - static_cast<int>(j);
    }

    cardinal::deque<NetEvent> inbox_;
    cardinal::vector<Delayed> delayed_;          // empty unless conditions on
    NetConditions        cond_;
    u32                  rng_{0x9E3779B9u};
    u32                  seeded_{0};        // last applied cond_.seed
    bool                 is_server_{false};
    bool                 connected_{false};
};

}  // namespace

cardinal::unique_ptr<Transport> Transport::create_loopback() {
    return cardinal::make_unique<LoopbackTransport>();
}

// Transport::create_udp() lives in net_udp.cpp — Winsock impl on
// Windows, nullptr stub elsewhere (same TU-split as audio_wasapi.cpp).

}  // namespace cardinal::net
