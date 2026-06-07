// =============================================================================
// Cardinal — UDP transport integration test (real localhost sockets).
//
// The deterministic suite (main.cpp) proves the Replicator + the
// in-proc loopback. This one proves the OTHER half of the thesis: the
// exact same Replicator, driven over the real Winsock UDP backend
// between a server and a client on 127.0.0.1 — "swap the factory ⇒
// MMO", verified end to end.
//
// Source-level platform-agnostic: it touches only the Transport
// abstraction, so it compiles everywhere and SKIPS clean (exit 0)
// wherever create_udp() returns null (non-Windows) or the sandbox
// forbids binding a socket. A completed handshake but wrong behaviour
// is a hard failure; a missing socket layer is not.
//
// Scope: handshake/Connected, Unreliable snapshot via the Replicator,
// ReliableOrdered exact + in-order across multiple sends, and clean
// disconnect. Inducing real packet loss (to exercise the UDP
// retransmit/dedupe timer specifically) needs a lossy-link layer on
// the UDP transport — a scoped follow-on; the reliable ORDERING/EXACT
// logic is already exercised here, and loss semantics are locked
// deterministically on the loopback.
// =============================================================================

#include <cardinal/net/net.hpp>
#include <cardinal/net/replication.hpp>
#include <cardinal/core/diag/log.hpp>

#include <vector>

namespace {

using cardinal::net::Channel;
using cardinal::net::NetEvent;
using cardinal::net::NetEventKind;
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
        cardinal::log::errorf("netudp", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool approx(float a, float b, float eps) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= eps;
}
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

int count_kind(const std::vector<NetEvent>& v, NetEventKind k) {
    int n = 0;
    for (const NetEvent& e : v) if (e.kind == k) ++n;
    return n;
}

// Spin both endpoints (non-blocking polls, no sleeps) until `pred` or
// the bounded cap. Localhost UDP delivers in microseconds, so the cap
// is just a safety net; the predicate exits the instant it's met.
template <class Pred>
bool pump_until(Transport& s, Transport& c,
                std::vector<NetEvent>& se, std::vector<NetEvent>& ce,
                Pred pred, int cap) {
    for (int i = 0; i < cap; ++i) {
        s.poll(se);
        c.poll(ce);
        if (pred()) return true;
    }
    return pred();
}

}  // namespace

int main() {
    auto server = Transport::create_udp();
    auto client = Transport::create_udp();
    if (!server || !client) {
        cardinal::log::infof("netudp",
            "SKIP  no UDP backend on this platform");
        return 0;
    }

    // Grab the first free port in a small private range (listen() ==
    // bind(); false on failure). If none bind, the environment forbids
    // sockets — skip rather than false-fail.
    cardinal::u16 port = 0;
    for (cardinal::u16 p = 47600; p < 47680; ++p) {
        if (server->listen(p)) { port = p; break; }
    }
    if (port == 0) {
        cardinal::log::infof("netudp",
            "SKIP  could not bind a localhost UDP port (sandboxed)");
        return 0;
    }

    if (!client->connect("127.0.0.1", port)) {
        cardinal::log::infof("netudp",
            "SKIP  client connect() unavailable in this environment");
        return 0;
    }

    constexpr int kCap = 20000;

    // ---- 1. handshake -----------------------------------------------
    std::vector<NetEvent> se, ce;
    const bool hs = pump_until(*server, *client, se, ce,
        [&] {
            return count_kind(se, NetEventKind::Connected) >= 1 &&
                   count_kind(ce, NetEventKind::Connected) >= 1;
        }, kCap);
    CHECK(hs);
    CHECK(server->peer_count() == sz(1));
    CHECK(client->peer_count() == sz(1));

    // Server peer-id as the client sees it (target for disconnect).
    cardinal::u32 server_pid = 0u;
    for (const NetEvent& e : ce)
        if (e.kind == NetEventKind::Connected) { server_pid = e.peer; break; }
    CHECK(server_pid != 0u);

    // ---- 2. Unreliable snapshot through the Replicator --------------
    Replicator sr(*server);   // authoritative side
    Replicator cr(*client);   // receiving side
    {
        RepState s;
        s.id       = 7u;
        s.position = cardinal::scene::Vec3{12.5f, -3.0f, 8.25f};
        s.scale    = cardinal::scene::Vec3{2.0f, 2.0f, 2.0f};
        std::vector<RepState> in{s};
        sr.server_broadcast(in);

        std::vector<NetEvent> se2, ce2;
        std::vector<RepState> out;
        const bool got = pump_until(*server, *client, se2, ce2,
            [&] {
                out.clear();
                cr.client_ingest(ce2, out);
                return !out.empty();
            }, kCap);
        CHECK(got);
        CHECK(out.size() == sz(1));
        if (out.size() == sz(1)) {
            CHECK(out[0].id == 7u);
            CHECK(out[0].position.x == 12.5f);   // pos is raw f32
            CHECK(out[0].position.y == -3.0f);
            CHECK(out[0].position.z == 8.25f);
            CHECK(approx(out[0].scale.x, 2.0f, 1.0e-3f));
        }
    }

    // ---- 3. ReliableOrdered: exact + in order across sends ----------
    {
        std::vector<RepEvent> a(1);
        a[0].kind = RepEventKind::Spawn;  a[0].id = 10u; a[0].archetype = 1u;
        std::vector<RepEvent> b(2);
        b[0].kind = RepEventKind::Spawn;  b[0].id = 11u; b[0].archetype = 2u;
        b[1].kind = RepEventKind::Spawn;  b[1].id = 12u; b[1].archetype = 3u;
        std::vector<RepEvent> c(1);
        c[0].kind = RepEventKind::Despawn; c[0].id = 10u;
        sr.server_events(a);
        sr.server_events(b);
        sr.server_events(c);

        std::vector<NetEvent> se3, ce3;
        std::vector<RepEvent> evs;
        const bool got = pump_until(*server, *client, se3, ce3,
            [&] {
                evs.clear();
                cr.client_events(ce3, evs);
                return evs.size() >= sz(4);
            }, kCap);
        CHECK(got);
        CHECK(evs.size() == sz(4));
        if (evs.size() == sz(4)) {
            CHECK(evs[0].kind == RepEventKind::Spawn   && evs[0].id == 10u);
            CHECK(evs[1].kind == RepEventKind::Spawn   && evs[1].id == 11u);
            CHECK(evs[2].kind == RepEventKind::Spawn   && evs[2].id == 12u);
            CHECK(evs[3].kind == RepEventKind::Despawn && evs[3].id == 10u);
        }
    }

    // ---- 4. clean disconnect ----------------------------------------
    {
        client->disconnect(server_pid);
        std::vector<NetEvent> se4, ce4;
        const bool gone = pump_until(*server, *client, se4, ce4,
            [&] {
                return count_kind(se4, NetEventKind::Disconnected) >= 1;
            }, kCap);
        CHECK(gone);
        CHECK(server->peer_count() == sz(0));
        CHECK(client->peer_count() == sz(0));
    }

    if (g_fail == 0) {
        cardinal::log::infof("netudp",
            "OK  %d checks passed (udp/%u, localhost)",
            g_checks, static_cast<unsigned>(port));
        return 0;
    }
    cardinal::log::errorf("netudp", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
