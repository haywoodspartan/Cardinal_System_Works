// =============================================================================
// Cardinal — deterministic CPU particle-sim regression suite.
//
// Emitter::tick is the VFX backbone: a fractional rate-accumulator
// spawn, semi-implicit-Euler + drag integration, age/lifetime cull,
// colour-over-life — all driven by a splitmix32 RNG seeded from
// desc.rng_seed. A regression silently breaks every effect (wrong
// counts, NaN drift, immortal particles) or, worse, breaks REPLAY
// determinism (same seed + inputs must reproduce bit-for-bit). Pure
// CPU + headless. Spawn counts + closed-form integration + two-System
// bit-identical determinism are pinned. Exit 0 = all pass.
// =============================================================================

#include <cardinal/particles/particles.hpp>
#include <cardinal/core/diag/log.hpp>

#include <limits>

namespace {

namespace pa = cardinal::particles;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("ptest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-3f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

// Exact, all-fields particle equality (determinism cross-check).
bool peq(const pa::Particle& a, const pa::Particle& b) {
    return a.position.x == b.position.x && a.position.y == b.position.y &&
           a.position.z == b.position.z &&
           a.velocity.x == b.velocity.x && a.velocity.y == b.velocity.y &&
           a.velocity.z == b.velocity.z &&
           a.acceleration.x == b.acceleration.x &&
           a.acceleration.y == b.acceleration.y &&
           a.acceleration.z == b.acceleration.z &&
           a.size == b.size && a.age == b.age &&
           a.lifetime == b.lifetime && a.color_rgba == b.color_rgba;
}

// Zero-variance desc → spawned particles are RNG-independent (rand_in
// returns lo when hi==lo), so integration is exactly predictable.
pa::EmitterDesc fixed_desc() {
    pa::EmitterDesc d;
    d.name = "fx";
    d.origin = cardinal::scene::Vec3{0.0f, 0.0f, 0.0f};
    d.velocity_min = cardinal::scene::Vec3{0.0f, 0.0f, 0.0f};
    d.velocity_max = cardinal::scene::Vec3{0.0f, 0.0f, 0.0f};
    d.size_min = 0.1f;  d.size_max = 0.1f;
    d.lifetime_min = 100.0f; d.lifetime_max = 100.0f;   // ~never culls
    d.gravity = cardinal::scene::Vec3{0.0f, 0.0f, 0.0f};
    d.drag = 0.0f;
    d.max_particles = 100000u;
    d.emitting = true;
    return d;
}

// ---- spawn accumulator: rate-independent, exact counts -------------
void test_spawn() {
    {   // 30/s for 2 sim-seconds at 1/60 → exactly 60 spawned.
        pa::EmitterDesc d = fixed_desc();
        d.rate_per_second = 30.0f;
        pa::Emitter e(d);
        for (int i = 0; i < 120; ++i) e.tick(1.0f / 60.0f);
        CHECK(e.total_spawned() == static_cast<cardinal::u64>(60));
        CHECK(e.live_count() == sz(60));            // lifetime huge → no cull
    }
    {   // emitting=false → nothing spawns.
        pa::EmitterDesc d = fixed_desc();
        d.emitting = false;
        pa::Emitter e(d);
        for (int i = 0; i < 100; ++i) e.tick(0.016f);
        CHECK(e.total_spawned() == static_cast<cardinal::u64>(0));
        CHECK(e.live_count() == sz(0));
    }
    {   // rate 0 → nothing; dt<=0 → tick is a no-op.
        pa::EmitterDesc d = fixed_desc();
        d.rate_per_second = 0.0f;
        pa::Emitter e(d);
        e.tick(1.0f);
        CHECK(e.total_spawned() == static_cast<cardinal::u64>(0));
        d.rate_per_second = 100.0f;
        pa::Emitter e2(d);
        e2.tick(0.0f);  e2.tick(-1.0f);
        CHECK(e2.total_spawned() == static_cast<cardinal::u64>(0));
        e2.tick(0.05f);                              // 100*0.05 = 5
        CHECK(e2.total_spawned() == static_cast<cardinal::u64>(5));
        const cardinal::u64 before = e2.total_spawned();
        e2.tick(0.0f);                               // no-op: unchanged
        CHECK(e2.total_spawned() == before);
    }
}

// ---- max_particles cap ---------------------------------------------
void test_cap() {
    pa::EmitterDesc d = fixed_desc();
    d.rate_per_second = 1.0e6f;
    d.max_particles   = 10u;
    pa::Emitter e(d);
    for (int i = 0; i < 8; ++i) {
        e.tick(0.1f);
        CHECK(e.live_count() <= sz(10));            // never exceeds the cap
    }
    CHECK(e.live_count() == sz(10));                // saturated
    // spawn_one_ early-returns (no ++total) at the cap; nothing dies
    // (huge lifetime) so the counter plateaus exactly at the cap.
    CHECK(e.total_spawned() == static_cast<cardinal::u64>(10));
}

// ---- semi-implicit Euler + drag, closed form -----------------------
void test_integration() {
    {   // gravity only, drag 0: v=k·g·dt, y=g·dt²·k(k+1)/2 (k=50).
        pa::EmitterDesc d = fixed_desc();
        d.rate_per_second = 100.0f;                  // dt 0.01 → 1 spawn
        d.gravity = cardinal::scene::Vec3{0.0f, -10.0f, 0.0f};
        pa::Emitter e(d);
        e.tick(0.01f);                               // spawn + integrate #1
        e.set_emitting(false);
        for (int i = 0; i < 49; ++i) e.tick(0.01f);  // integrate #2..#50
        CHECK(e.live_count() == sz(1));
        const pa::Particle& p = e.particles()[0];
        CHECK(ap(p.velocity.y, -5.0f, 1e-3f));       // 50·-10·0.01
        CHECK(ap(p.position.y, -1.275f, 5e-3f));     // -10·1e-4·1275
        CHECK(ap(p.age, 0.50f, 1e-4f));
        CHECK(ap(p.position.x, 0.0f) && ap(p.position.z, 0.0f));
        CHECK(ap(p.velocity.x, 0.0f) && ap(p.velocity.z, 0.0f));
    }
    {   // one drag step: damp = clamp(1 - drag·dt).
        pa::EmitterDesc d = fixed_desc();
        d.rate_per_second = 100.0f;
        d.velocity_min = cardinal::scene::Vec3{10.0f, 0.0f, 0.0f};
        d.velocity_max = cardinal::scene::Vec3{10.0f, 0.0f, 0.0f};
        d.drag = 0.5f;
        pa::Emitter e(d);
        e.tick(0.01f);                               // spawn + 1 integrate
        e.set_emitting(false);
        const pa::Particle& p = e.particles()[0];
        CHECK(ap(p.velocity.x, 9.95f, 1e-4f));       // 10 · (1-0.5·0.01)
        CHECK(ap(p.position.x, 0.0995f, 1e-4f));     // 9.95 · 0.01
    }
}

// ---- cull at age >= lifetime (incl. lifetime <= 0 edge) ------------
void test_cull() {
    {
        // lifetime 0.035 sits strictly BETWEEN tick multiples (0.03 / 0.04)
        // so FP accumulation noise (~1e-8 over 4 adds of 0.01f) can never
        // straddle the cull boundary — margin is ±0.005 either side.
        pa::EmitterDesc d = fixed_desc();
        d.rate_per_second = 100.0f;                  // dt 0.01 → 1/tick
        d.lifetime_min = 0.035f; d.lifetime_max = 0.035f;
        pa::Emitter e(d);
        e.tick(0.01f);                               // spawn; age ≈0.01
        e.set_emitting(false);
        CHECK(e.live_count() == sz(1));
        for (int i = 0; i < 2; ++i) e.tick(0.01f);   // age ≈0.03 (<0.035)
        CHECK(e.live_count() == sz(1));              // not yet culled
        e.tick(0.01f);                               // age ≈0.04 (≥0.035)
        CHECK(e.live_count() == sz(0));              // culled
        CHECK(e.total_spawned() == static_cast<cardinal::u64>(1));
    }
    {   // lifetime 0 → first tick: t forced to 1, age>=0 → instant cull.
        pa::EmitterDesc d = fixed_desc();
        d.rate_per_second = 100.0f;
        d.lifetime_min = 0.0f; d.lifetime_max = 0.0f;
        pa::Emitter e(d);
        e.tick(0.01f);
        CHECK(e.live_count() == sz(0));
        CHECK(e.total_spawned() == static_cast<cardinal::u64>(1));
    }
}

// ---- colour over life (interior blend + near-end) ------------------
void test_color() {
    pa::EmitterDesc d = fixed_desc();
    d.rate_per_second = 2.0f;                        // dt 0.5 → 1 spawn
    d.lifetime_min = 1.0f; d.lifetime_max = 1.0f;
    d.start_rgba = 0x00000000u;
    d.end_rgba   = 0x00000064u;                      // R byte 0..100
    pa::Emitter e(d);
    e.tick(0.5f);                                    // age 0.5 → t 0.5
    e.set_emitting(false);
    CHECK(e.live_count() == sz(1));
    CHECK(e.particles()[0].color_rgba == 0x00000032u);  // blend(0,100,.5)=50
    e.tick(0.49f);                                   // age 0.99 → t 0.99
    CHECK(e.live_count() == sz(1));                  // not yet culled
    CHECK(e.particles()[0].color_rgba == 0x00000063u);  // (u8)(100·0.99)=99
}

// ---- determinism: same seed + inputs ⇒ bit-identical ---------------
void test_determinism() {
    pa::EmitterDesc d;                               // RANDOMISED ranges
    d.name = "rng";
    d.rng_seed = 24680u;
    d.rate_per_second = 50.0f;
    d.lifetime_min = 1.0f;  d.lifetime_max = 2.0f;
    d.size_min = 0.05f;     d.size_max = 0.30f;
    d.velocity_min = cardinal::scene::Vec3{-1.0f, 1.0f, -1.0f};
    d.velocity_max = cardinal::scene::Vec3{ 1.0f, 4.0f,  1.0f};
    d.gravity = cardinal::scene::Vec3{0.0f, -9.81f, 0.0f};
    d.drag = 0.05f;
    d.max_particles = 512u;

    pa::Emitter a(d), b(d);
    for (int i = 0; i < 80; ++i) { a.tick(1.0f/60.0f); b.tick(1.0f/60.0f); }
    CHECK(a.total_spawned() == b.total_spawned());
    CHECK(a.live_count() == b.live_count());
    CHECK(a.live_count() > sz(0));                   // it actually ran
    bool identical = (a.particles().size() == b.particles().size());
    if (identical)
        for (cardinal::usize k = 0; k < a.particles().size(); ++k)
            if (!peq(a.particles()[k], b.particles()[k])) identical = false;
    CHECK(identical);                                // replay-deterministic

    // System::tick just forwards — same desc ⇒ same pool as a standalone.
    auto sys = pa::System::create();
    pa::Emitter* se = sys->add(d);
    pa::Emitter standalone(d);
    for (int i = 0; i < 40; ++i) { sys->tick(1.0f/60.0f);
                                   standalone.tick(1.0f/60.0f); }
    bool same = (se->particles().size() == standalone.particles().size());
    if (same)
        for (cardinal::usize k = 0; k < se->particles().size(); ++k)
            if (!peq(se->particles()[k], standalone.particles()[k]))
                same = false;
    CHECK(same);
}

// ---- System registry + stats ---------------------------------------
void test_system() {
    auto sys = pa::System::create();
    CHECK(sys->emitters().empty());
    pa::System::Stats s0 = sys->stats();
    CHECK(s0.emitters_active == 0u && s0.particles_alive == 0u &&
          s0.particles_total_spawned == static_cast<cardinal::u64>(0));

    pa::EmitterDesc d = fixed_desc();
    d.rate_per_second = 100.0f;
    pa::Emitter* e1 = sys->add(d);
    d.name = "fx2";
    pa::Emitter* e2 = sys->add(d);
    CHECK(e1 != nullptr && e2 != nullptr);
    CHECK(sys->emitters().size() == sz(2));
    CHECK(e1->desc().name == "fx" && e2->desc().name == "fx2");

    for (int i = 0; i < 3; ++i) sys->tick(0.01f);    // 1 spawn/emitter/tick
    pa::System::Stats s1 = sys->stats();
    CHECK(s1.emitters_active == 2u);
    cardinal::u32 alive = static_cast<cardinal::u32>(
        e1->live_count() + e2->live_count());
    CHECK(s1.particles_alive == alive);
    CHECK(s1.particles_total_spawned ==
          e1->total_spawned() + e2->total_spawned());

    sys->remove(e1);
    CHECK(sys->emitters().size() == sz(1));
    sys->clear();
    CHECK(sys->emitters().empty());
    pa::System::Stats s2 = sys->stats();
    CHECK(s2.emitters_active == 0u && s2.particles_alive == 0u);
}

// ---- clear() removes live particles but is NOT an emitter reset -----
void test_clear_semantics() {
    pa::EmitterDesc d = fixed_desc();
    d.rate_per_second = 100.0f;
    pa::Emitter e(d);
    for (int i = 0; i < 5; ++i) e.tick(0.01f);
    CHECK(e.total_spawned() == static_cast<cardinal::u64>(5));
    CHECK(e.live_count() == sz(5));
    e.clear();
    CHECK(e.live_count() == sz(0));
    CHECK(e.total_spawned() == static_cast<cardinal::u64>(5));  // NOT reset
    e.tick(0.01f);                                   // RNG/counter continue
    CHECK(e.total_spawned() == static_cast<cardinal::u64>(6));
    CHECK(e.live_count() == sz(1));
}

// ---- non-finite dt must be rejected (NaN / +Inf robustness) --------
// Regression: `if (dt <= 0.0f) return;` lets NaN through (NaN<=0 is
// false) and +Inf through (Inf<=0 is false). spawn_accum_ then goes
// non-finite; static_cast<u32>(non-finite) is UB (in practice a ~2e9
// "integer indefinite" → multi-billion-iteration spawn loop); the
// accumulator stays NaN forever (NaN-x=NaN) so every LATER valid tick
// is poisoned too; and age+=dt makes `age >= lifetime` false → an
// immortal NaN particle. tick() is noexcept: a non-finite dt must be
// the exact same no-op as dt<=0, leaving state pristine for the next
// valid tick.
void test_nonfinite_dt() {
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    const float inf  = std::numeric_limits<float>::infinity();

    {   // NaN dt: no-op; the subsequent VALID tick proves the spawn
        // accumulator was never poisoned (exactly 5 spawn, not ~2e9).
        pa::EmitterDesc d = fixed_desc();
        d.rate_per_second = 100.0f;
        pa::Emitter e(d);
        e.tick(qnan);
        CHECK(e.total_spawned() == static_cast<cardinal::u64>(0));
        CHECK(e.live_count() == sz(0));
        e.tick(0.05f);                               // 100 * 0.05 = 5
        CHECK(e.total_spawned() == static_cast<cardinal::u64>(5));
        CHECK(e.live_count() == sz(5));
    }
    {   // +Inf dt: same contract.
        pa::EmitterDesc d = fixed_desc();
        d.rate_per_second = 100.0f;
        pa::Emitter e(d);
        e.tick(inf);
        CHECK(e.total_spawned() == static_cast<cardinal::u64>(0));
        CHECK(e.live_count() == sz(0));
        e.tick(0.05f);
        CHECK(e.total_spawned() == static_cast<cardinal::u64>(5));
    }
    {   // A live particle survives a NaN tick with FINITE age (no NaN
        // drift) and still culls normally afterwards (not immortal).
        pa::EmitterDesc d = fixed_desc();
        d.rate_per_second = 100.0f;
        d.lifetime_min = 0.025f; d.lifetime_max = 0.025f;
        pa::Emitter e(d);
        e.tick(0.01f);                               // spawn 1, age≈0.01
        e.set_emitting(false);
        CHECK(e.live_count() == sz(1));
        e.tick(qnan);                                // no-op
        CHECK(e.live_count() == sz(1));
        const pa::Particle& p = e.particles()[0];
        CHECK(ap(p.age, 0.01f, 1e-4f));              // age unchanged
        CHECK(p.age == p.age);                        // finite (not NaN)
        e.tick(0.01f); e.tick(0.01f);                // age≈0.03 ≥ 0.025
        CHECK(e.live_count() == sz(0));               // culled normally
    }
    {   // NaN drag: previously poisoned EVERY particle simultaneously
        // every tick via `damp = clamp(1.0f - NaN*dt, 0, 1)` = NaN
        // (clamp is NaN-passthrough) → velocity *= NaN → position
        // accumulator NaN. The drag_safe fallback in tick() routes NaN
        // → 0 effective drag; velocities keep their finite values.
        pa::EmitterDesc d = fixed_desc();
        d.rate_per_second = 100.0f;
        d.velocity_min = cardinal::scene::Vec3{1.0f, 0.0f, 0.0f};
        d.velocity_max = cardinal::scene::Vec3{1.0f, 0.0f, 0.0f};
        d.drag = qnan;                                // POISON
        pa::Emitter e(d);
        e.tick(0.05f);                               // spawn ~5 with vel=1
        CHECK(e.live_count() >= sz(1));
        e.tick(0.05f);                               // integrate one step
        // Every live particle must still have finite velocity AND
        // position — proves drag NaN didn't poison the integrator.
        for (const pa::Particle& p : e.particles()) {
            CHECK(p.velocity.x == p.velocity.x);
            CHECK(p.position.x == p.position.x);
            CHECK((p.position.x - p.position.x) == 0.0f);   // also not ±Inf
        }
    }
    {   // NaN spawn-time fields (origin / gravity / velocity_min/max /
        // size_min/max): the fz sanitizer in spawn_one_ defaults each
        // non-finite component to 0 BEFORE storing in the particle.
        // Verify by inspecting freshly-spawned particles and tracking
        // them through several integrator ticks — every component of
        // position/velocity/acceleration/size must stay finite.
        pa::EmitterDesc d = fixed_desc();
        d.rate_per_second = 100.0f;
        d.lifetime_min = 5.0f; d.lifetime_max = 5.0f;     // long enough to integrate
        // POISON every Vec3 / scalar spawn field except start_rgba (u32).
        d.origin       = cardinal::scene::Vec3{ qnan,  inf, -inf };
        d.gravity      = cardinal::scene::Vec3{ qnan, qnan, qnan };
        d.velocity_min = cardinal::scene::Vec3{ qnan, qnan, qnan };
        d.velocity_max = cardinal::scene::Vec3{ qnan, qnan, qnan };
        d.size_min = qnan; d.size_max = qnan;
        pa::Emitter e(d);
        e.tick(0.05f);                                    // spawn ~5
        CHECK(e.live_count() >= sz(1));
        // Every freshly-spawned particle must have FINITE state across
        // position / velocity / acceleration / size.
        auto fin = [](float v) { return v == v && (v - v) == 0.0f; };
        for (const pa::Particle& p : e.particles()) {
            CHECK(fin(p.position.x));     CHECK(fin(p.position.y));     CHECK(fin(p.position.z));
            CHECK(fin(p.velocity.x));     CHECK(fin(p.velocity.y));     CHECK(fin(p.velocity.z));
            CHECK(fin(p.acceleration.x)); CHECK(fin(p.acceleration.y)); CHECK(fin(p.acceleration.z));
            CHECK(fin(p.size));
            CHECK(fin(p.lifetime));
            // Sanitization defaulted each non-finite to 0.
            CHECK(ap(p.position.x, 0.0f)); CHECK(ap(p.position.y, 0.0f)); CHECK(ap(p.position.z, 0.0f));
            CHECK(ap(p.velocity.x, 0.0f));
            CHECK(ap(p.acceleration.x, 0.0f));
            CHECK(ap(p.size, 0.0f));
        }
        // Integrate several ticks — persistent state must stay finite.
        for (int i = 0; i < 50; ++i) e.tick(0.01f);
        for (const pa::Particle& p : e.particles()) {
            CHECK(fin(p.position.x)); CHECK(fin(p.position.y)); CHECK(fin(p.position.z));
            CHECK(fin(p.velocity.x)); CHECK(fin(p.velocity.y)); CHECK(fin(p.velocity.z));
        }
    }
    {   // NaN lifetime_min/max: the spawn rolls a NaN p.lifetime via
        // rand_in(lo, hi) when lo or hi is NaN. Pre-fix `p.age >=
        // p.lifetime` is unordered-false → IMMORTAL particle, live_
        // grows unbounded across ticks (memory leak). The
        // !isfinite(p.lifetime) clause in the kill check reaps them
        // on first encounter.
        pa::EmitterDesc d = fixed_desc();
        d.rate_per_second = 100.0f;
        d.lifetime_min = qnan; d.lifetime_max = qnan;   // POISON
        pa::Emitter e(d);
        e.tick(0.05f);                                  // spawn ~5
        // First tick spawned them but they're newly created (age==0)
        // — they should die on the SECOND tick when the kill check
        // sees !isfinite(lifetime). Actually they die on the first
        // tick's kill pass since !isfinite fires regardless of age.
        CHECK(e.live_count() == sz(0));                 // not immortal
        // Confirm they don't leak across many ticks either.
        for (int i = 0; i < 100; ++i) e.tick(0.05f);
        CHECK(e.live_count() == sz(0));
    }
}

// ---- shape sampling: every spawned particle must lie inside the
//      authored shape volume / surface. Use a high spawn count + a
//      zero-variance velocity so position == shape offset directly. ---
void test_shape_point() {
    // Point shape (the default) must spawn EXACTLY at origin — this is
    // the back-compat invariant: every existing call site that uses
    // default-constructed EmitterDesc gets the same per-frame state as
    // before the shape extension landed.
    pa::EmitterDesc d = fixed_desc();
    d.rate_per_second = 200.0f;
    d.shape = pa::EmitterShape::Point;   // explicit even though it's default
    pa::Emitter e(d);
    e.tick(0.5f);                        // spawn ~100
    CHECK(e.live_count() > sz(50));
    for (const pa::Particle& p : e.particles()) {
        CHECK(p.position.x == 0.0f);
        CHECK(p.position.y == 0.0f);
        CHECK(p.position.z == 0.0f);
    }
}

void test_shape_sphere() {
    pa::EmitterDesc d = fixed_desc();
    d.rate_per_second = 500.0f;
    d.shape           = pa::EmitterShape::Sphere;
    d.shape_extent    = cardinal::scene::Vec3{2.0f, 0.0f, 0.0f};  // radius 2
    pa::Emitter e(d);
    e.tick(1.0f);                        // spawn ~500
    CHECK(e.live_count() > sz(200));
    bool any_inside = false;
    bool any_y_nonzero = false, any_x_nonzero = false;
    for (const pa::Particle& p : e.particles()) {
        const float r2 = p.position.x*p.position.x +
                         p.position.y*p.position.y +
                         p.position.z*p.position.z;
        CHECK(r2 <= 4.0f + 1e-3f);       // inside r=2 ball
        if (r2 > 0.0f) any_inside = true;
        if (p.position.y != 0.0f) any_y_nonzero = true;
        if (p.position.x != 0.0f) any_x_nonzero = true;
    }
    CHECK(any_inside);                   // distribution is non-degenerate
    CHECK(any_y_nonzero);                // Sphere covers all 3 axes
    CHECK(any_x_nonzero);
}

void test_shape_box() {
    pa::EmitterDesc d = fixed_desc();
    d.rate_per_second = 500.0f;
    d.shape           = pa::EmitterShape::Box;
    d.shape_extent    = cardinal::scene::Vec3{1.0f, 2.0f, 3.0f};  // half-extents
    pa::Emitter e(d);
    e.tick(1.0f);
    CHECK(e.live_count() > sz(200));
    bool y_range = false, z_range = false;
    for (const pa::Particle& p : e.particles()) {
        CHECK(p.position.x >= -1.0f - 1e-3f && p.position.x <= 1.0f + 1e-3f);
        CHECK(p.position.y >= -2.0f - 1e-3f && p.position.y <= 2.0f + 1e-3f);
        CHECK(p.position.z >= -3.0f - 1e-3f && p.position.z <= 3.0f + 1e-3f);
        if (p.position.y > 1.0f || p.position.y < -1.0f) y_range = true;
        if (p.position.z > 2.0f || p.position.z < -2.0f) z_range = true;
    }
    CHECK(y_range);                      // box actually uses its Y extent
    CHECK(z_range);                      // and its Z extent
}

void test_shape_disk() {
    pa::EmitterDesc d = fixed_desc();
    d.rate_per_second = 500.0f;
    d.shape           = pa::EmitterShape::Disk;
    d.shape_extent    = cardinal::scene::Vec3{1.5f, 0.0f, 0.0f};  // radius 1.5
    pa::Emitter e(d);
    e.tick(1.0f);
    CHECK(e.live_count() > sz(200));
    for (const pa::Particle& p : e.particles()) {
        CHECK(p.position.y == 0.0f);     // Disk is XZ-plane only
        const float r2 = p.position.x*p.position.x + p.position.z*p.position.z;
        CHECK(r2 <= 2.25f + 1e-3f);      // inside r=1.5 disk
    }
}

void test_shape_cone() {
    pa::EmitterDesc d = fixed_desc();
    d.rate_per_second = 500.0f;
    d.shape           = pa::EmitterShape::Cone;
    d.shape_extent    = cardinal::scene::Vec3{0.0f, 4.0f, 0.0f};   // height 4
    d.shape_angle_deg = 30.0f;           // half-angle 30° → max-base radius = 4*tan(30) ≈ 2.31
    pa::Emitter e(d);
    e.tick(1.0f);
    CHECK(e.live_count() > sz(200));
    for (const pa::Particle& p : e.particles()) {
        // Apex at y=0, base at y=4. y must be in [0,4].
        CHECK(p.position.y >= 0.0f - 1e-3f);
        CHECK(p.position.y <= 4.0f + 1e-3f);
        // Radius constraint: at height y, max radius = y * tan(30°).
        const float max_r = p.position.y * 0.5774f + 1e-3f;
        const float r     = (p.position.x*p.position.x + p.position.z*p.position.z);
        CHECK(r <= max_r * max_r);
    }
}

void test_shape_origin_offset() {
    // Spawn at (10, 20, 30) with Sphere of radius 1 — every particle's
    // distance from the origin offset must be ≤ 1.
    pa::EmitterDesc d = fixed_desc();
    d.rate_per_second = 200.0f;
    d.origin          = cardinal::scene::Vec3{10.0f, 20.0f, 30.0f};
    d.shape           = pa::EmitterShape::Sphere;
    d.shape_extent    = cardinal::scene::Vec3{1.0f, 0.0f, 0.0f};
    pa::Emitter e(d);
    e.tick(0.5f);
    CHECK(e.live_count() > sz(50));
    for (const pa::Particle& p : e.particles()) {
        const float dx = p.position.x - 10.0f;
        const float dy = p.position.y - 20.0f;
        const float dz = p.position.z - 30.0f;
        CHECK(dx*dx + dy*dy + dz*dz <= 1.0f + 1e-3f);
    }
}

void test_shape_nan_extent_safe() {
    // Poisoned shape_extent / shape_angle: every spawned position must
    // still be finite. The fz at every desc read site is what keeps the
    // persistent particle state defined.
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    auto fin = [](float v) { return v == v && (v - v) == 0.0f; };
    {
        pa::EmitterDesc d = fixed_desc();
        d.rate_per_second = 100.0f;
        d.shape           = pa::EmitterShape::Sphere;
        d.shape_extent    = cardinal::scene::Vec3{qnan, qnan, qnan};
        pa::Emitter e(d);
        e.tick(0.5f);
        CHECK(e.live_count() >= sz(1));
        for (const pa::Particle& p : e.particles()) {
            CHECK(fin(p.position.x)); CHECK(fin(p.position.y)); CHECK(fin(p.position.z));
        }
    }
    {
        pa::EmitterDesc d = fixed_desc();
        d.rate_per_second = 100.0f;
        d.shape           = pa::EmitterShape::Cone;
        d.shape_extent    = cardinal::scene::Vec3{0.0f, 4.0f, 0.0f};
        d.shape_angle_deg = qnan;        // POISON
        pa::Emitter e(d);
        e.tick(0.5f);
        CHECK(e.live_count() >= sz(1));
        for (const pa::Particle& p : e.particles()) {
            CHECK(fin(p.position.x)); CHECK(fin(p.position.y)); CHECK(fin(p.position.z));
        }
    }
}

// ---- burst mode: burst_count particles every burst_interval seconds,
//      rate_per_second ignored. ----------------------------------------
void test_burst_basic() {
    pa::EmitterDesc d = fixed_desc();
    d.rate_per_second = 1000.0f;          // must be IGNORED
    d.mode            = pa::EmitterMode::Burst;
    d.burst_count     = 10;
    d.burst_interval  = 0.5f;
    pa::Emitter e(d);
    // First sub-interval tick: no burst yet (0.1 < 0.5).
    e.tick(0.1f);
    CHECK(e.live_count() == sz(0));
    // Crossing the interval boundary: exactly one burst fires.
    e.tick(0.5f);                         // accum = 0.6 → 1 burst → 10
    CHECK(e.live_count() == sz(10));
    // Another full interval → second burst.
    e.tick(0.5f);                         // accum = 0.1 + 0.5 = 0.6 → 1 burst
    CHECK(e.live_count() == sz(20));
}

void test_burst_multi_per_tick() {
    // Long hitch crosses multiple intervals — every owed burst fires.
    pa::EmitterDesc d = fixed_desc();
    d.mode            = pa::EmitterMode::Burst;
    d.burst_count     = 5;
    d.burst_interval  = 0.1f;
    pa::Emitter e(d);
    e.tick(0.5f);                         // 5 intervals → 5 bursts = 25
    CHECK(e.live_count() == sz(25));
}

void test_burst_count_zero() {
    // Burst count of 0 is degenerate but defined — no particles spawn.
    pa::EmitterDesc d = fixed_desc();
    d.mode            = pa::EmitterMode::Burst;
    d.burst_count     = 0;
    d.burst_interval  = 0.1f;
    pa::Emitter e(d);
    for (int i = 0; i < 20; ++i) e.tick(0.1f);
    CHECK(e.live_count() == sz(0));
}

void test_burst_nan_interval_safe() {
    // Poisoned burst_interval must not produce non-finite burst_accum_
    // or spawn an unbounded loop in one tick. The dt-fallback collapses
    // a NaN/non-positive interval to "one burst per tick".
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    pa::EmitterDesc d = fixed_desc();
    d.mode            = pa::EmitterMode::Burst;
    d.burst_count     = 3;
    d.burst_interval  = qnan;             // POISON
    pa::Emitter e(d);
    e.tick(0.1f);                         // expect exactly one burst → 3
    CHECK(e.live_count() == sz(3));
    e.tick(0.1f);                         // and another
    CHECK(e.live_count() == sz(6));
}

void test_burst_not_emitting() {
    // emitting=false halts bursts immediately; turning emitting back on
    // doesn't fire a backlog (burst_accum is paused while emitting=false
    // — except dt isn't accumulated either, so no surprise mega-burst).
    pa::EmitterDesc d = fixed_desc();
    d.mode            = pa::EmitterMode::Burst;
    d.burst_count     = 4;
    d.burst_interval  = 0.1f;
    d.emitting        = false;
    pa::Emitter e(d);
    for (int i = 0; i < 10; ++i) e.tick(0.1f);
    CHECK(e.live_count() == sz(0));
    e.set_emitting(true);
    e.tick(0.1f);                         // first burst after enabling
    CHECK(e.live_count() == sz(4));
}

void test_burst_determinism() {
    // Two emitters with same desc + same tick sequence must produce
    // identical particles (Burst mode also bit-for-bit deterministic).
    pa::EmitterDesc d = fixed_desc();
    d.mode            = pa::EmitterMode::Burst;
    d.burst_count     = 7;
    d.burst_interval  = 0.2f;
    d.shape           = pa::EmitterShape::Sphere;
    d.shape_extent    = cardinal::scene::Vec3{1.0f, 0.0f, 0.0f};
    pa::Emitter a(d);
    pa::Emitter b(d);
    for (int i = 0; i < 30; ++i) {
        a.tick(0.05f);
        b.tick(0.05f);
    }
    CHECK(a.live_count() == b.live_count());
    CHECK(a.total_spawned() == b.total_spawned());
    CHECK(a.live_count() > sz(0));
    for (cardinal::usize i = 0; i < a.particles().size(); ++i) {
        CHECK(peq(a.particles()[i], b.particles()[i]));
    }
}

void test_burst_clear_resets_accum() {
    // clear() must reset burst_accum_ too, otherwise the next tick
    // after a clear can fire a partial-interval burst on the spot.
    pa::EmitterDesc d = fixed_desc();
    d.mode            = pa::EmitterMode::Burst;
    d.burst_count     = 5;
    d.burst_interval  = 1.0f;
    pa::Emitter e(d);
    e.tick(0.9f);                         // accum = 0.9 (no burst yet)
    CHECK(e.live_count() == sz(0));
    e.clear();                            // wipes accum to 0
    e.tick(0.5f);                         // 0.5 < 1.0 → no burst
    CHECK(e.live_count() == sz(0));
}

}  // namespace

int main() {
    test_spawn();
    test_cap();
    test_integration();
    test_cull();
    test_color();
    test_determinism();
    test_system();
    test_clear_semantics();
    test_nonfinite_dt();
    // ---- shape + mode extensions
    test_shape_point();
    test_shape_sphere();
    test_shape_box();
    test_shape_disk();
    test_shape_cone();
    test_shape_origin_offset();
    test_shape_nan_extent_safe();
    test_burst_basic();
    test_burst_multi_per_tick();
    test_burst_count_zero();
    test_burst_nan_interval_safe();
    test_burst_not_emitting();
    test_burst_determinism();
    test_burst_clear_resets_accum();

    if (g_fail == 0) {
        cardinal::log::infof("ptest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("ptest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
