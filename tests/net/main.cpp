// =============================================================================
// Cardinal — deterministic networking regression suite.
//
// Locks the five networking arcs the engine shipped, end to end, with
// ZERO test dependencies (same no-deps ethos as net/physics/audio): a
// ~20-line CHECK harness, the in-process loopback transport, and the
// real Replicator. Everything here is deterministic by construction —
// the netcode was deliberately built clock-free (host passes `now`)
// and the loopback adversity sim is poll-counted + seeded, precisely so
// this suite is reproducible and CI-safe.
//
//   1. quantization round-trip ....... rot i16 / scale u16 / pos f32
//   2. seq-gating under reorder ...... newest-wins, never regress
//   3. client interpolation .......... lerp midpoint + clamp ends
//   4. loss-sim determinism .......... same seed ⇒ same drops; the
//                                      zero-loss and max-loss invariants
//   5. reliable lifecycle ............ exact + ordered even when EVERY
//                                      Unreliable snapshot is dropped
//
// Exit code 0 = all pass, 1 = one or more failed (CTest reads this).
// =============================================================================

#include <cardinal/net/net.hpp>
#include <cardinal/net/replication.hpp>
#include <cardinal/core/log.hpp>

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
        cardinal::log::errorf("nettest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

// abs/approx without <cmath> (foundation discipline; all-float so no
// double-promotion either).
bool approx(float a, float b, float eps) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= eps;
}

cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

// Fresh loopback endpoint with the synthetic Connected event drained,
// so Message accounting starts clean.
cardinal::unique_ptr<Transport> fresh() {
    auto t = Transport::create_loopback();
    t->listen(0);
    std::vector<NetEvent> drain;
    t->poll(drain);
    return t;
}

RepState mk(cardinal::u32 id, float x) {
    RepState s;
    s.id       = id;
    s.position = cardinal::scene::Vec3{x, 0.0f, 0.0f};
    return s;
}

// ---- 1. quantization round-trip -------------------------------------
void test_quantization() {
    auto t = fresh();
    Replicator repl(*t);

    RepState s;
    s.id             = 42u;
    s.position       = cardinal::scene::Vec3{1.5f, -2.25f, 100.125f};
    s.rotation_euler = cardinal::scene::Vec3{0.5f, -1.0f, 3.0f};
    s.scale          = cardinal::scene::Vec3{1.0f, 2.0f, 0.5f};

    std::vector<RepState> in{s};
    repl.server_broadcast(in);
    // 10 B header + exactly one 28 B quantized record.
    CHECK(repl.last_snapshot_bytes() == 38u);

    std::vector<NetEvent> ev;
    t->poll(ev);
    std::vector<RepState> out;
    repl.client_ingest(ev, out);

    CHECK(out.size() == sz(1));
    if (out.size() == sz(1)) {
        const RepState& r = out[0];
        CHECK(r.id == 42u);
        // Position is sent as raw f32 — must be bit-exact.
        CHECK(r.position.x == 1.5f);
        CHECK(r.position.y == -2.25f);
        CHECK(r.position.z == 100.125f);
        // Rotation i16 over +/-3.2 rad: < 1e-4 rad resolution.
        CHECK(approx(r.rotation_euler.x,  0.5f, 5.0e-4f));
        CHECK(approx(r.rotation_euler.y, -1.0f, 5.0e-4f));
        CHECK(approx(r.rotation_euler.z,  3.0f, 5.0e-4f));
        // Scale u16 over [0,16]: < 2.5e-4 resolution.
        CHECK(approx(r.scale.x, 1.0f, 1.0e-3f));
        CHECK(approx(r.scale.y, 2.0f, 1.0e-3f));
        CHECK(approx(r.scale.z, 0.5f, 1.0e-3f));
    }
}

// ---- 2. seq-gating under reorder ------------------------------------
void test_seq_gating_reorder() {
    auto t = fresh();
    Replicator repl(*t);

    // Latency + jitter ⇒ datagrams genuinely reorder across polls.
    NetConditions c;
    c.latency_polls = 4u;
    c.jitter_polls  = 3u;
    c.loss          = 0.0f;          // nothing lost — only reordered
    c.seed          = 12345u;
    t->set_conditions(c);

    const cardinal::u32 kSends = 30u;
    float applied = -1.0f;           // last value written to the proxy
    bool  monotone = true;

    for (cardinal::u32 i = 1u; i <= kSends; ++i) {
        std::vector<RepState> in{mk(1u, static_cast<float>(i))};
        repl.server_broadcast(in);
        std::vector<NetEvent> ev;
        t->poll(ev);
        std::vector<RepState> out;
        repl.client_ingest(ev, out);
        if (!out.empty()) {
            const float v = out[0].position.x;
            if (v + 1.0e-4f < applied) monotone = false;  // regressed!
            applied = v;
        }
    }
    // Drain the still-in-flight (delayed) datagrams.
    for (cardinal::u32 d = 0u; d < 40u; ++d) {
        std::vector<NetEvent> ev;
        t->poll(ev);
        std::vector<RepState> out;
        repl.client_ingest(ev, out);
        if (!out.empty()) {
            const float v = out[0].position.x;
            if (v + 1.0e-4f < applied) monotone = false;
            applied = v;
        }
    }
    // Never applied an older frame, and the newest (30) is what stuck.
    CHECK(monotone);
    CHECK(applied == static_cast<float>(kSends));
}

// ---- 3. client interpolation ----------------------------------------
void test_interpolation() {
    auto t = fresh();
    Replicator repl(*t);

    // Snapshot A @ x=0, buffered at t=100; B @ x=10, buffered at t=101.
    {
        std::vector<RepState> in{mk(7u, 0.0f)};
        repl.server_broadcast(in);
        std::vector<NetEvent> ev;
        t->poll(ev);
        repl.client_buffer(ev, 100.0);
    }
    {
        std::vector<RepState> in{mk(7u, 10.0f)};
        repl.server_broadcast(in);
        std::vector<NetEvent> ev;
        t->poll(ev);
        repl.client_buffer(ev, 101.0);
    }
    CHECK(repl.history_size() == sz(2));

    std::vector<RepState> out;
    // Midpoint in time ⇒ midpoint in space.
    repl.client_sample(100.5, out);
    CHECK(out.size() == sz(1));
    if (out.size() == sz(1)) CHECK(approx(out[0].position.x, 5.0f, 1.0e-3f));
    // Before the oldest ⇒ clamp to front (no extrapolation).
    repl.client_sample(99.0, out);
    if (out.size() == sz(1)) CHECK(approx(out[0].position.x, 0.0f, 1.0e-3f));
    // After the newest ⇒ clamp to back.
    repl.client_sample(102.0, out);
    if (out.size() == sz(1)) CHECK(approx(out[0].position.x, 10.0f, 1.0e-3f));
}

// ---- 3b. client interpolation: non-finite render_time --------------
// Regression: a NaN render_time defeats BOTH ordered clamp guards (NaN
// compares unordered). With one buffered snapshot the a/b pair read
// history_[1] OUT OF BOUNDS (UB / crash); with >=2 it lerps alpha=NaN
// → every proxy teleports to NaN. NaN must hold the newest snapshot
// (like "past newest"); +/-Inf is ordered and already clamps. NaN/Inf
// laundered through volatile (foundation discipline: no <cmath>).
double nan_d() { volatile double z = 0.0; return z / z; }    // 0/0 = NaN
double inf_d() { volatile double z = 0.0; return 1.0 / z; }   // 1/0 = +Inf

void test_interpolation_nonfinite() {
    // --- two snapshots: NaN must NOT produce NaN transforms ---
    {
        auto t = fresh();
        Replicator repl(*t);
        { std::vector<RepState> in{mk(7u, 0.0f)};  repl.server_broadcast(in);
          std::vector<NetEvent> ev; t->poll(ev); repl.client_buffer(ev, 100.0); }
        { std::vector<RepState> in{mk(7u, 10.0f)}; repl.server_broadcast(in);
          std::vector<NetEvent> ev; t->poll(ev); repl.client_buffer(ev, 101.0); }
        CHECK(repl.history_size() == sz(2));

        std::vector<RepState> out;
        const cardinal::usize n = repl.client_sample(nan_d(), out);
        CHECK(n == sz(1));
        if (out.size() == sz(1)) {
            const float x = out[0].position.x;
            CHECK(x == x);                        // finite, not NaN
            CHECK(approx(x, 10.0f, 1.0e-3f));     // held newest snapshot
        }
        repl.client_sample(inf_d(), out);          // +Inf → newest
        if (out.size() == sz(1)) CHECK(approx(out[0].position.x, 10.0f, 1.0e-3f));
        repl.client_sample(-inf_d(), out);         // -Inf → oldest
        if (out.size() == sz(1)) CHECK(approx(out[0].position.x, 0.0f, 1.0e-3f));
    }
    // --- one snapshot: NaN must not read history_[1] OUT OF BOUNDS ---
    {
        auto t = fresh();
        Replicator repl(*t);
        { std::vector<RepState> in{mk(3u, 4.0f)}; repl.server_broadcast(in);
          std::vector<NetEvent> ev; t->poll(ev); repl.client_buffer(ev, 50.0); }
        CHECK(repl.history_size() == sz(1));
        std::vector<RepState> out;
        const cardinal::usize n = repl.client_sample(nan_d(), out);
        CHECK(n == sz(1));                          // no OOB; returns the snap
        if (out.size() == sz(1)) {
            const float x = out[0].position.x;
            CHECK(out[0].id == 3u);
            CHECK(x == x && approx(x, 4.0f, 1.0e-3f));
        }
    }
}

// ---- 4. loss-sim determinism + invariants ---------------------------
// One run: `iters` ticks, each sending one Unreliable snapshot and one
// ReliableOrdered lifecycle event; returns how many of each the client
// actually saw.
void run_loss(cardinal::u32 seed, float loss, cardinal::u32 iters,
              cardinal::u32& unrel_out, cardinal::u32& rel_out) {
    auto t = fresh();
    Replicator repl(*t);
    NetConditions c;
    c.latency_polls = 0u;
    c.jitter_polls  = 0u;
    c.loss          = loss;
    c.seed          = seed;
    t->set_conditions(c);

    cardinal::u32 unrel = 0u;
    cardinal::u32 rel   = 0u;
    for (cardinal::u32 k = 0u; k < iters; ++k) {
        std::vector<RepState> in{mk(1u, static_cast<float>(k))};
        repl.server_broadcast(in);
        std::vector<RepEvent> evs{RepEvent{}};
        evs[0].kind = RepEventKind::Spawn;
        evs[0].id   = 1u;
        repl.server_events(evs);

        std::vector<NetEvent> ev;
        t->poll(ev);
        for (const NetEvent& e : ev) {
            if (e.kind != NetEventKind::Message) continue;
            if (e.channel == Channel::Unreliable)      ++unrel;
            else if (e.channel == Channel::ReliableOrdered) ++rel;
        }
    }
    unrel_out = unrel;
    rel_out   = rel;
}

void test_loss_sim() {
    // Determinism: identical seed ⇒ identical drop pattern.
    cardinal::u32 u1 = 0u, r1 = 0u, u2 = 0u, r2 = 0u;
    run_loss(777u, 0.5f, 200u, u1, r1);
    run_loss(777u, 0.5f, 200u, u2, r2);
    CHECK(u1 == u2);
    CHECK(r1 == r2);
    // A lossy link actually loses some (but not all) snapshots.
    CHECK(u1 > 0u);
    CHECK(u1 < 200u);

    // Zero loss ⇒ a perfect link: nothing dropped (legacy behaviour).
    cardinal::u32 uz = 0u, rz = 0u;
    run_loss(0xABCDEFu, 0.0f, 150u, uz, rz);
    CHECK(uz == 150u);
    CHECK(rz == 150u);

    // Total loss ⇒ every Unreliable snapshot gone, but ReliableOrdered
    // is NEVER perturbed by the sim — the channel contract holds.
    cardinal::u32 um = 0u, rm = 0u;
    run_loss(0xBEEFu, 1.0f, 120u, um, rm);
    CHECK(um == 0u);
    CHECK(rm == 120u);
}

// ---- 5. reliable lifecycle exactness --------------------------------
void test_lifecycle() {
    auto t = fresh();
    Replicator repl(*t);

    // Even with the Unreliable channel a black hole, lifecycle events
    // (ReliableOrdered) must arrive exactly and in order.
    NetConditions c;
    c.loss = 1.0f;
    c.seed = 99u;
    t->set_conditions(c);

    std::vector<RepEvent> evs(3);
    evs[0].kind = RepEventKind::Spawn;
    evs[0].id   = 10u;
    evs[0].archetype = 2u;
    evs[0].state.position = cardinal::scene::Vec3{3.0f, 0.0f, 0.0f};
    evs[1].kind = RepEventKind::Spawn;
    evs[1].id   = 11u;
    evs[1].archetype = 5u;
    evs[2].kind = RepEventKind::Despawn;
    evs[2].id   = 10u;

    // A decoy snapshot that the total-loss link must swallow.
    std::vector<RepState> snap{mk(10u, 0.0f)};
    repl.server_broadcast(snap);
    repl.server_events(evs);

    std::vector<NetEvent> ev;
    t->poll(ev);
    std::vector<RepEvent> out;
    repl.client_events(ev, out);

    CHECK(out.size() == sz(3));
    if (out.size() == sz(3)) {
        CHECK(out[0].kind == RepEventKind::Spawn);
        CHECK(out[0].id == 10u);
        CHECK(out[0].archetype == 2u);
        CHECK(approx(out[0].state.position.x, 3.0f, 1.0e-3f));
        CHECK(out[1].kind == RepEventKind::Spawn);
        CHECK(out[1].id == 11u);
        CHECK(out[1].archetype == 5u);
        CHECK(out[2].kind == RepEventKind::Despawn);
        CHECK(out[2].id == 10u);
    }
    CHECK(repl.events_sent() == 3u);
    CHECK(repl.events_recv() == 3u);

    // Snapshot path saw nothing (decoy dropped by the 100%-loss link).
    std::vector<RepState> rx;
    repl.client_ingest(ev, rx);
    CHECK(rx.empty());
}

}  // namespace

int main() {
    test_quantization();
    test_seq_gating_reorder();
    test_interpolation();
    test_interpolation_nonfinite();
    test_loss_sim();
    test_lifecycle();

    if (g_fail == 0) {
        cardinal::log::infof("nettest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("nettest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
