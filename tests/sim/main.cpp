// =============================================================================
// Cardinal — deterministic simulation-orchestrator regression suite.
//
// SimWorld::tick is the fixed-timestep heartbeat behind all gameplay /
// replay determinism. Pinned, with default parallel_handlers=false so
// dispatch is sequential (no JobSystem dependency → fully pure):
//
//   * real_dt clamp [0, max_real_dt]; time-scale (incl. negative→0);
//   * the physics accumulator: carry-over remainder across ticks and a
//     hard sub-step cap (spiral-of-death guard) — fixed_dt = 0.10 so the
//     math is hand-exact;
//   * pause / resume / single-step (exactly one advancing tick then
//     re-frozen) / start_paused;
//   * strict tick-group order PreUpdate→PrePhysics→[physics]→PostPhysics
//     →Update→LateUpdate→Render, Render running even while paused, and
//     the quirk that TickGroup::Physics user handlers are NEVER invoked
//     (integrate_physics_ replaces that slot);
//   * monotonic handler ids, remove, stats.
//
// Exit 0 = all pass.
// =============================================================================

#include <cardinal/sim/sim.hpp>
#include <cardinal/core/diag/log.hpp>

#include <string>
#include <vector>

namespace {

namespace sm = cardinal::sim;
using TG = sm::TickGroup;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("simtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-4f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
bool apd(double a, double b, double e = 1e-5) {
    const double d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }
bool streq(const char* a, const char* b) { return std::string(a) == b; }
// File-level volatile-launder of qNaN — avoids <cmath>/<limits>.
float nan_f() { volatile float z = 0.0f; return z / z; }

sm::SimDesc accum_desc() {
    sm::SimDesc d;
    d.fixed_dt     = 0.10f;     // hand-exact accumulator math
    d.max_real_dt  = 1.0f;
    d.max_substeps = 4u;
    return d;
}

// ---- tick_group_name ----------------------------------------------
void test_group_names() {
    CHECK(streq(sm::tick_group_name(TG::PreUpdate),   "PreUpdate"));
    CHECK(streq(sm::tick_group_name(TG::PrePhysics),  "PrePhysics"));
    CHECK(streq(sm::tick_group_name(TG::Physics),     "Physics"));
    CHECK(streq(sm::tick_group_name(TG::PostPhysics), "PostPhysics"));
    CHECK(streq(sm::tick_group_name(TG::Update),      "Update"));
    CHECK(streq(sm::tick_group_name(TG::LateUpdate),  "LateUpdate"));
    CHECK(streq(sm::tick_group_name(TG::Render),      "Render"));
    CHECK(streq(sm::tick_group_name(TG::Count),       "(count)"));
    CHECK(streq(sm::tick_group_name(static_cast<TG>(99u)), "?"));
}

// ---- accumulator: carry-over + sub-step cap + clamps --------------
void test_accumulator() {
    // Carry-over: half-fixed real_dt ⇒ substeps cadence 0,1,0,1.
    {
        sm::SimWorld w(accum_desc());
        w.tick(0.05f);
        CHECK(w.stats().physics_substeps_last == static_cast<cardinal::u64>(0));
        w.tick(0.05f);                                   // accum 0.10 → 1
        CHECK(w.stats().physics_substeps_last == static_cast<cardinal::u64>(1));
        w.tick(0.05f);
        CHECK(w.stats().physics_substeps_last == static_cast<cardinal::u64>(0));
        w.tick(0.05f);
        CHECK(w.stats().physics_substeps_last == static_cast<cardinal::u64>(1));
        CHECK(w.stats().total_ticks == static_cast<cardinal::u64>(4));
    }
    // Exactly fixed_dt every tick ⇒ exactly one substep.
    {
        sm::SimWorld w(accum_desc());
        w.tick(0.10f);
        CHECK(w.stats().physics_substeps_last == static_cast<cardinal::u64>(1));
        w.tick(0.10f);
        CHECK(w.stats().physics_substeps_last == static_cast<cardinal::u64>(1));
    }
    // 3×fixed in one tick (under the cap) ⇒ exactly 3 substeps.
    {
        sm::SimWorld w(accum_desc());
        w.tick(0.30f);
        CHECK(w.stats().physics_substeps_last == static_cast<cardinal::u64>(3));
    }
    // Spiral-of-death guard: big real_dt is capped at max_substeps.
    {
        sm::SimWorld w(accum_desc());
        w.tick(0.45f);                                   // 4.5 steps → cap 4
        CHECK(w.stats().physics_substeps_last == static_cast<cardinal::u64>(4));
    }
    // real_dt clamped to max_real_dt (1.0) before anything else.
    {
        sm::SimWorld w(accum_desc());
        const float ret = w.tick(1.0e9f);
        CHECK(ap(ret, 1.0f));                             // scaled by ts 1
        CHECK(apd(w.stats().real_time_seconds, 1.0));     // clamped sum
        CHECK(w.stats().physics_substeps_last == static_cast<cardinal::u64>(4));
    }
    // Negative real_dt → 0: advancing tick, but no time, no substeps.
    {
        sm::SimWorld w(accum_desc());
        const float ret = w.tick(-0.5f);
        CHECK(ap(ret, 0.0f));
        CHECK(apd(w.stats().real_time_seconds, 0.0));
        CHECK(w.stats().physics_substeps_last == static_cast<cardinal::u64>(0));
        CHECK(w.stats().total_ticks == static_cast<cardinal::u64>(1)); // advanced
    }
}

// ---- non-finite real_dt must not poison the physics accumulator ---
// Regression: tick()'s clamps were `real_dt < 0` and `real_dt > max`;
// BOTH are false for NaN (ordered compares with NaN are false), so a
// NaN real_dt reached `physics_accum_ +=`, poisoning it forever
// (NaN + x = NaN ⇒ `physics_accum_ >= fixed_dt` always false ⇒
// fixed-step physics never substeps again) and turning
// stats_.real_time_seconds permanently NaN. A non-finite real_dt must
// behave exactly like the existing negative→0 case. NaN/Inf via
// volatile launder (no <cmath>/<limits>).
void test_nonfinite_dt() {
    auto nanf = []{ volatile float z = 0.0f; return z / z; };    // 0/0 → NaN
    auto inff = []{ volatile float z = 0.0f; return 1.0f / z; };  // 1/0 → +Inf

    // (a) NaN real_dt → 0 (advancing tick, no time, no substep) AND the
    //     accumulator stays pristine: later valid ticks still substep.
    {
        sm::SimWorld w(accum_desc());
        const float ret = w.tick(nanf());
        CHECK(ap(ret, 0.0f));                                  // scaled_dt 0
        CHECK(apd(w.stats().real_time_seconds, 0.0));          // finite, not NaN
        CHECK(w.stats().physics_substeps_last == static_cast<cardinal::u64>(0));
        CHECK(w.stats().total_ticks == static_cast<cardinal::u64>(1));
        w.tick(0.10f);
        CHECK(w.stats().physics_substeps_last == static_cast<cardinal::u64>(1));
        CHECK(apd(w.stats().real_time_seconds, 0.10));
        w.tick(0.10f);
        CHECK(w.stats().physics_substeps_last == static_cast<cardinal::u64>(1));
    }
    // (b) A NaN mid-run must not freeze an already-running sim.
    {
        sm::SimWorld w(accum_desc());
        w.tick(0.10f); w.tick(0.10f);
        const cardinal::u64 ticks_mid = w.stats().total_ticks;
        w.tick(nanf());                                        // poison attempt
        w.tick(0.10f);
        CHECK(w.stats().physics_substeps_last == static_cast<cardinal::u64>(1));
        CHECK(w.stats().total_ticks == ticks_mid + 2u);
        CHECK(apd(w.stats().real_time_seconds, 0.30));         // .1+.1+0+.1
    }
    // (c) +Inf still clamped to max_real_dt (existing behaviour, the fix
    //     must not regress it).
    {
        sm::SimWorld w(accum_desc());
        const float ret = w.tick(inff());                      // → 1.0
        CHECK(ap(ret, 1.0f));
        CHECK(apd(w.stats().real_time_seconds, 1.0));
        CHECK(w.stats().physics_substeps_last == static_cast<cardinal::u64>(4));
    }
}

// ---- pause / single-step / start_paused ---------------------------
void test_pause_step() {
    sm::SimWorld w;                                       // default desc
    int uc = 0, rc = 0;
    w.add_handler(TG::Update, [&](float){ ++uc; });
    w.add_handler(TG::Render, [&](float){ ++rc; });

    w.tick(1.0f / 60.0f);                                 // advance
    CHECK(uc == 1 && rc == 1);
    CHECK(w.stats().total_ticks == static_cast<cardinal::u64>(1));
    CHECK(!w.paused());

    w.pause();
    CHECK(w.paused());
    const float r = w.tick(1.0f / 60.0f);                 // frozen
    CHECK(ap(r, 0.0f));
    CHECK(uc == 1);                                       // Update skipped
    CHECK(rc == 2);                                       // Render still runs
    CHECK(w.stats().total_ticks == static_cast<cardinal::u64>(1));
    const double rt_after_pause = w.stats().real_time_seconds;
    CHECK(rt_after_pause > 0.0);                          // real time advances

    w.step_one_frame();
    CHECK(w.paused());                                    // step keeps paused
    w.tick(1.0f / 60.0f);                                 // the one step
    CHECK(uc == 2);
    CHECK(w.stats().total_ticks == static_cast<cardinal::u64>(2));
    w.tick(1.0f / 60.0f);                                 // re-frozen
    CHECK(uc == 2);
    CHECK(w.stats().total_ticks == static_cast<cardinal::u64>(2));

    w.resume();
    CHECK(!w.paused());
    w.tick(1.0f / 60.0f);
    CHECK(uc == 3);
    CHECK(w.stats().total_ticks == static_cast<cardinal::u64>(3));

    // start_paused.
    sm::SimDesc sp; sp.start_paused = true;
    sm::SimWorld w2(sp);
    CHECK(w2.paused());
    w2.tick(1.0f / 60.0f);
    CHECK(w2.stats().total_ticks == static_cast<cardinal::u64>(0));
}

// ---- time scale (incl. negative clamp) ----------------------------
void test_time_scale() {
    sm::SimWorld w;
    float last_dt = -1.0f;
    w.add_handler(TG::Update, [&](float dt){ last_dt = dt; });

    CHECK(ap(w.time_scale(), 1.0f));
    w.set_time_scale(2.0f);
    CHECK(ap(w.time_scale(), 2.0f));
    CHECK(ap(w.tick(0.01f), 0.02f));
    CHECK(ap(last_dt, 0.02f));

    w.set_time_scale(0.5f);
    CHECK(ap(w.tick(0.02f), 0.01f));
    CHECK(ap(last_dt, 0.01f));

    // Negative scale clamps to 0: still an advancing tick, but dt 0.
    w.set_time_scale(-3.0f);
    CHECK(ap(w.time_scale(), 0.0f));
    const cardinal::u64 t0 = w.stats().total_ticks;
    CHECK(ap(w.tick(0.05f), 0.0f));
    CHECK(ap(last_dt, 0.0f));
    CHECK(w.stats().total_ticks == t0 + static_cast<cardinal::u64>(1));
    CHECK(ap(w.stats().time_scale, 0.0f));

    // NaN / ±Inf scale ALSO clamps to 0 — the previous `s < 0.0f`
    // ordered compare was NaN-blind AND +Inf-blind. Without this fix,
    // scaled_dt = real_dt * time_scale_ poisons physics_accum_ to NaN
    // (physics never substeps again) or +Inf (max_substeps every
    // frame). nan_f() is the volatile-launder helper at top-of-file.
    w.set_time_scale(nan_f());
    CHECK(ap(w.time_scale(), 0.0f));
    CHECK(ap(w.tick(0.05f), 0.0f));
    // Reset to normal then test +Inf and -Inf.
    w.set_time_scale(1.0f);
    CHECK(ap(w.time_scale(), 1.0f));
    volatile float big = 1.0f; for (int i = 0; i < 16; ++i) big *= 1e30f;  // +Inf
    w.set_time_scale(big);
    CHECK(ap(w.time_scale(), 0.0f));
    w.set_time_scale(-big);
    CHECK(ap(w.time_scale(), 0.0f));
}

// ---- handlers: ids, group order, Physics-never, remove ------------
void test_handlers() {
    sm::SimWorld w;                                       // not paused
    std::vector<std::string> seq;

    const auto h1 = w.add_handler(TG::PreUpdate,   [&](float){ seq.push_back("A"); });
    const auto h2 = w.add_handler(TG::PrePhysics,  [&](float){ seq.push_back("B"); });
    const auto h3 = w.add_handler(TG::PostPhysics, [&](float){ seq.push_back("C"); });
    const auto h4 = w.add_handler(TG::Update,      [&](float){ seq.push_back("D"); });
    const auto h5 = w.add_handler(TG::LateUpdate,  [&](float){ seq.push_back("E"); });
    const auto h6 = w.add_handler(TG::Render,      [&](float){ seq.push_back("F"); });
    const auto h7 = w.add_handler(TG::Physics,     [&](float){ seq.push_back("X"); });
    CHECK(h1 == 1u && h2 == 2u && h3 == 3u && h4 == 4u);
    CHECK(h5 == 5u && h6 == 6u && h7 == 7u);              // monotonic from 1

    w.tick(1.0f / 60.0f);
    CHECK(seq.size() == sz(6));
    CHECK(seq.size() == sz(6) &&
          seq[0]=="A" && seq[1]=="B" && seq[2]=="C" &&
          seq[3]=="D" && seq[4]=="E" && seq[5]=="F");
    // The Physics user-handler is NEVER invoked (engine-internal slot).
    bool saw_x = false;
    for (const auto& s : seq) if (s == "X") saw_x = true;
    CHECK(!saw_x);

    // Two handlers in one group run in insertion order.
    w.add_handler(TG::Update, [&](float){ seq.push_back("D2"); });
    seq.clear();
    w.tick(1.0f / 60.0f);
    CHECK(seq.size() == sz(7));
    CHECK(seq[3] == "D" && seq[4] == "D2");

    // remove_handler drops just that one; unknown id is a safe no-op.
    w.remove_handler(h4);                                  // removes "D"
    w.remove_handler(9999u);                               // no-op
    seq.clear();
    w.tick(1.0f / 60.0f);
    bool saw_d = false, saw_d2 = false;
    for (const auto& s : seq) { if (s=="D") saw_d=true; if (s=="D2") saw_d2=true; }
    CHECK(!saw_d && saw_d2);

    // Render runs even when paused; the rest don't.
    w.pause();
    seq.clear();
    w.tick(1.0f / 60.0f);
    CHECK(seq.size() == sz(1) && seq[0] == "F");
}

// ---- handler that mutates handlers_ during tick must not UAF ------
// Same range-for-over-mutating-vector UAF class as f3ed9c1 (actor::
// World::tick) and 5057580 (game::Game::apply_lifecycle_ /
// broadcast_begin_play_). A handler that calls add_handler / remove_
// handler reaches into the SAME handlers_[g] vector the dispatch loop
// is walking via range-for; a sufficiently-large add_handler causes
// vector realloc → range-for dangles into freed old buffer → UAF
// crash on the next s.fn deref. Test: a single handler that batches
// many add_handler calls into the SAME group forces realloc; the
// dispatch loop must complete without crashing and the added handlers
// must defer to the NEXT frame (matching the actor/game spawn-during-
// tick contract).
void test_handler_add_during_tick() {
    sm::SimWorld w;
    int orig_count = 0;
    int added_count = 0;
    bool first_ran = false;

    // The triggering handler spawns 64 new handlers into the SAME
    // group on its first invocation. 64 push_back's into a small-
    // initial-capacity vector will realloc at least once, dangling
    // any range-for over the live list.
    w.add_handler(TG::Update, [&](float) {
        ++orig_count;
        if (!first_ran) {
            first_ran = true;
            for (int i = 0; i < 64; ++i) {
                w.add_handler(TG::Update, [&](float) { ++added_count; });
            }
        }
    });

    // Tick 1: orig runs once, spawns 64 deferred handlers. None of the
    // added handlers fire this frame (matches the actor/game contract).
    w.tick(1.0f / 60.0f);
    CHECK(orig_count == 1);
    CHECK(added_count == 0);                  // deferred to next frame

    // Tick 2: orig runs again (first_ran=true so no more adds), and
    // the 64 deferred handlers all fire.
    w.tick(1.0f / 60.0f);
    CHECK(orig_count == 2);
    CHECK(added_count == 64);
}

void test_handler_remove_during_tick() {
    sm::SimWorld w;
    int a_count = 0;
    int b_count = 0;
    sm::SimWorld::HandlerId hb = 0;

    // A removes B mid-tick. Pre-fix the iterator might invalidate
    // after the erase; post-fix the snapshot copy means B still fires
    // ONCE in this frame (defensible — it was on the snapshot when
    // the tick started), then never again.
    w.add_handler(TG::Update, [&](float) {
        ++a_count;
        w.remove_handler(hb);   // remove B (which has already been added below)
    });
    hb = w.add_handler(TG::Update, [&](float) { ++b_count; });

    w.tick(1.0f / 60.0f);
    CHECK(a_count == 1);
    // Snapshot semantics: B was in the snapshot at tick start; A's
    // remove_handler mutates the live list but the snapshot still
    // holds B's function copy → B fires once more this frame.
    CHECK(b_count == 1);

    // Tick 2: B is removed from live list, NOT in next snapshot.
    w.tick(1.0f / 60.0f);
    CHECK(a_count == 2);
    CHECK(b_count == 1);                      // didn't fire again
}

// ---- stats + desc -------------------------------------------------
void test_stats_desc() {
    sm::SimWorld w;
    sm::SimStats s0 = w.stats();
    CHECK(s0.total_ticks == static_cast<cardinal::u64>(0));
    CHECK(s0.physics_substeps_last == static_cast<cardinal::u64>(0));
    CHECK(apd(s0.sim_time_seconds, 0.0) && apd(s0.real_time_seconds, 0.0));
    CHECK(ap(s0.time_scale, 1.0f));
    CHECK(!s0.paused && !s0.single_step_armed);

    // desc() echoes the construction descriptor (defaults here).
    CHECK(ap(w.desc().fixed_dt, 1.0f / 60.0f));
    CHECK(w.desc().max_substeps == 4u);
    CHECK(ap(w.desc().max_real_dt, 0.10f));
    CHECK(!w.desc().parallel_handlers);

    // sim_time accrues scaled_dt on advance; real_time accrues clamped
    // real_dt always; single_step_armed stays false throughout.
    w.tick(0.02f);
    w.tick(0.02f);
    sm::SimStats s1 = w.stats();
    CHECK(s1.total_ticks == static_cast<cardinal::u64>(2));
    CHECK(apd(s1.sim_time_seconds,  0.04));
    CHECK(apd(s1.real_time_seconds, 0.04));
    CHECK(!s1.single_step_armed);

    w.step_one_frame();
    CHECK(!w.stats().single_step_armed);                  // never set true
    w.tick(0.02f);                                        // consumes the step
    CHECK(!w.stats().single_step_armed);
}

}  // namespace

int main() {
    test_group_names();
    test_accumulator();
    test_nonfinite_dt();
    test_pause_step();
    test_time_scale();
    test_handlers();
    test_handler_add_during_tick();
    test_handler_remove_during_tick();
    test_stats_desc();

    if (g_fail == 0) {
        cardinal::log::infof("simtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("simtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
