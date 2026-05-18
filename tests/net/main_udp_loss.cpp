// =============================================================================
// Cardinal — UDP reliability-under-loss integration test.
//
// The one netcode path nothing else covers: the UDP transport's
// cumulative-ack + retransmit + reorder + dedupe machinery, exercised
// under REAL injected packet loss on both endpoints. Loopback locks
// the loss *semantics* deterministically but has no reliability layer;
// the happy-path UDP test proves ordering on a clean link. This proves
// the hard guarantee: with ~35% of datagrams (data AND acks) vanishing,
// ReliableOrdered still arrives EXACTLY and IN ORDER via retransmit,
// while Unreliable simply degrades.
//
// The UDP retransmit timer is wall-clock (0.10s) — correct for a real
// socket. So this test lets real time pass by busy-pumping the non-
// blocking sockets and bounding on cardinal::hal's monotonic clock (no
// sleeps, no native time headers — the engine's own facility). Skips
// clean (exit 0) where create_udp() is null or sockets are sandboxed.
// =============================================================================

#include <cardinal/net/net.hpp>
#include <cardinal/net/replication.hpp>
#include <cardinal/core/log.hpp>
#include <cardinal/core/hal.hpp>

#include <vector>

namespace {

using cardinal::net::Channel;
using cardinal::net::NetEvent;
using cardinal::net::NetEventKind;
using cardinal::net::NetConditions;
using cardinal::net::RepState;
using cardinal::net::RepEvent;
using cardinal::net::RepEventKind;
using cardinal::net::Replicator;
using cardinal::net::Transport;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("udploss", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

int count_msg(const std::vector<NetEvent>& v, Channel ch) {
    int n = 0;
    for (const NetEvent& e : v)
        if (e.kind == NetEventKind::Message && e.channel == ch) ++n;
    return n;
}
int count_kind(const std::vector<NetEvent>& v, NetEventKind k) {
    int n = 0;
    for (const NetEvent& e : v) if (e.kind == k) ++n;
    return n;
}

// Busy-pump both endpoints until `pred` or the wall-time budget. The
// reliability retransmit is wall-clock gated, so real time must pass;
// the monotonic clock both bounds the loop and lets those timers fire.
template <class Pred>
bool pump_for(Transport& s, Transport& c,
              std::vector<NetEvent>& se, std::vector<NetEvent>& ce,
              Pred pred, double budget_s) {
    const double t0 = cardinal::hal::mono_now_seconds();
    for (;;) {
        s.poll(se);
        c.poll(ce);
        if (pred()) return true;
        if (cardinal::hal::mono_now_seconds() - t0 > budget_s)
            return pred();
    }
}

}  // namespace

int main() {
    auto server = Transport::create_udp();
    auto client = Transport::create_udp();
    if (!server || !client) {
        cardinal::log::infof("udploss", "SKIP  no UDP backend");
        return 0;
    }

    cardinal::u16 port = 0;
    for (cardinal::u16 p = 47680; p < 47760; ++p)
        if (server->listen(p)) { port = p; break; }
    if (port == 0) {
        cardinal::log::infof("udploss", "SKIP  no bindable port");
        return 0;
    }
    if (!client->connect("127.0.0.1", port)) {
        cardinal::log::infof("udploss", "SKIP  connect() unavailable");
        return 0;
    }

    // ---- handshake on a CLEAN link (Connect isn't retransmitted) ----
    std::vector<NetEvent> hse, hce;
    const bool hs = pump_for(*server, *client, hse, hce,
        [&] {
            return count_kind(hse, NetEventKind::Connected) >= 1 &&
                   count_kind(hce, NetEventKind::Connected) >= 1;
        }, 3.0);
    CHECK(hs);
    CHECK(server->peer_count() == sz(1));
    CHECK(client->peer_count() == sz(1));
    if (g_fail != 0) {                       // no point continuing
        cardinal::log::errorf("udploss", "%d/%d FAILED (handshake)",
                              g_fail, g_checks);
        return 1;
    }

    // ---- degrade the link: ~35% loss, both directions --------------
    NetConditions sc; sc.loss = 0.35f; sc.seed = 0x00C0FFEEu;
    NetConditions cc; cc.loss = 0.35f; cc.seed = 0x0000BEEFu;
    server->set_conditions(sc);
    client->set_conditions(cc);

    Replicator sr(*server);
    Replicator cr(*client);

    // ---- ReliableOrdered must survive intact + in order ------------
    {
        std::vector<RepEvent> a(1);
        a[0].kind = RepEventKind::Spawn;   a[0].id = 1u;
        std::vector<RepEvent> b(3);
        b[0].kind = RepEventKind::Spawn;   b[0].id = 2u;
        b[1].kind = RepEventKind::Spawn;   b[1].id = 3u;
        b[2].kind = RepEventKind::Spawn;   b[2].id = 4u;
        std::vector<RepEvent> d(2);
        d[0].kind = RepEventKind::Despawn; d[0].id = 1u;
        d[1].kind = RepEventKind::Spawn;   d[1].id = 5u;
        sr.server_events(a);
        sr.server_events(b);
        sr.server_events(d);                 // 3 datagrams, 6 events

        std::vector<NetEvent> se, ce;
        std::vector<RepEvent> evs;
        const bool got = pump_for(*server, *client, se, ce,
            [&] {
                evs.clear();
                cr.client_events(ce, evs);   // transport already dedupes
                return evs.size() >= sz(6);
            }, 8.0);
        CHECK(got);
        CHECK(evs.size() == sz(6));
        if (evs.size() == sz(6)) {
            CHECK(evs[0].kind == RepEventKind::Spawn   && evs[0].id == 1u);
            CHECK(evs[1].kind == RepEventKind::Spawn   && evs[1].id == 2u);
            CHECK(evs[2].kind == RepEventKind::Spawn   && evs[2].id == 3u);
            CHECK(evs[3].kind == RepEventKind::Spawn   && evs[3].id == 4u);
            CHECK(evs[4].kind == RepEventKind::Despawn && evs[4].id == 1u);
            CHECK(evs[5].kind == RepEventKind::Spawn   && evs[5].id == 5u);
        }
    }

    // ---- Unreliable should visibly degrade on the same link --------
    {
        const int kSent = 40;
        for (int i = 0; i < kSent; ++i) {
            std::vector<RepState> in(1);
            in[0].id = 9u;
            in[0].position =
                cardinal::scene::Vec3{static_cast<float>(i), 0.0f, 0.0f};
            sr.server_broadcast(in);
        }
        std::vector<NetEvent> se, ce;
        // Brief drain — no retransmit for Unreliable, so whatever the
        // lossy link ate is simply gone.
        pump_for(*server, *client, se, ce, [] { return false; }, 0.5);
        const int got = count_msg(ce, Channel::Unreliable);
        // Some got through, but the ~35%-loss link dropped a chunk:
        // the contrast with the exact reliable stream above is the
        // whole point.
        CHECK(got > 0);
        CHECK(got < kSent);
    }

    if (g_fail == 0) {
        cardinal::log::infof("udploss",
            "OK  %d checks passed (udp/%u, ~35%% loss both ways)",
            g_checks, static_cast<unsigned>(port));
        return 0;
    }
    cardinal::log::errorf("udploss", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
