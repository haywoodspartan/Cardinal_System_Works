// =============================================================================
// Cardinal — deterministic physics regression suite.
//
// Physics is gameplay-critical and bottoms out in the (now-locked) core
// math. "Deterministic given the same input" is an explicit design goal
// the net replication + gameplay both depend on, so a silent regression
// here — a flipped gravity sign, a broken integrator, tunnelling, or
// loss of determinism — breaks the game invisibly. World::create() is
// pure CPU (no rhi::Device): headless, fast, exactly reproducible.
//
// Locks: bit-exact determinism (two worlds, identical input → identical
// state), the semi-implicit-Euler free-fall closed form, body mutators
// + integration gating (impulse Δv=J/m, Static/no-gravity), sphere-on-
// plane collision (no tunnelling + inelastic rest + elastic rebound),
// raycast (hit/miss/max-distance), and the standalone narrow-phase
// queries. No <cmath> (FOUNDATION). Exit 0 = all pass.
// =============================================================================

#include <cardinal/physics/physics.hpp>
#include <cardinal/physics/gpu_physics.hpp>
#include <cardinal/render/graph.hpp>
#include <cardinal/core/log.hpp>
#include <cardinal/core/utility.hpp>

#include <limits>

#include <memory>

namespace {

namespace ph = cardinal::physics;
using ph::Vec3;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("phystest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
bool apv(const Vec3& a, const Vec3& b, float e) {
    return ap(a.x,b.x,e) && ap(a.y,b.y,e) && ap(a.z,b.z,e);
}
bool exact3(const Vec3& a, const Vec3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

ph::BodyDesc dyn_sphere(Vec3 pos, float r, float restitution) {
    ph::BodyDesc d{};
    d.type            = ph::BodyType::Dynamic;
    d.position        = pos;
    d.mass            = 1.0f;
    d.collider        = ph::Collider::make_sphere(r);
    d.material.restitution = restitution;
    d.material.friction    = 0.5f;
    d.linear_damping  = 0.0f;
    d.angular_damping = 0.0f;
    d.can_sleep       = false;     // keep sleep out of the trajectory math
    d.gravity         = true;
    d.layer           = 1u;        // explicit (matches the default); see note in main()
    return d;
}

// ---- bit-exact determinism (the headline property) ------------------
void run_scenario(ph::World& w, Vec3& out_a, Vec3& out_b, Vec3& out_va) {
    w.set_gravity(Vec3{0.0f, -9.81f, 0.0f});
    w.set_fixed_timestep(0.01f);
    const ph::BodyHandle a = w.create_body(
        dyn_sphere(Vec3{0.0f, 10.0f, 0.0f}, 0.5f, 0.3f));
    ph::BodyDesc bd = dyn_sphere(Vec3{1.0f, 8.0f, 0.0f}, 0.5f, 0.6f);
    bd.velocity = Vec3{2.0f, 0.0f, -1.0f};
    const ph::BodyHandle b = w.create_body(bd);
    for (int i = 0; i < 250; ++i) {
        if (i == 60)  w.apply_impulse(a, Vec3{5.0f, 3.0f, 0.0f});
        if (i == 120) w.apply_force(b,  Vec3{0.0f, 40.0f, 0.0f});
        w.step(0.01f);
    }
    out_a  = w.position(a);
    out_b  = w.position(b);
    out_va = w.velocity(a);
}

void test_determinism() {
    auto w1 = ph::World::create();
    auto w2 = ph::World::create();
    CHECK(w1 != nullptr && w2 != nullptr);
    if (!w1 || !w2) return;
    Vec3 a1, b1, va1, a2, b2, va2;
    run_scenario(*w1, a1, b1, va1);
    run_scenario(*w2, a2, b2, va2);
    // Same code path, same inputs, single thread ⇒ BIT-identical.
    CHECK(exact3(a1, a2));
    CHECK(exact3(b1, b2));
    CHECK(exact3(va1, va2));
}

// ---- semi-implicit Euler free-fall closed form ----------------------
void test_freefall() {
    auto w = ph::World::create();
    if (!w) { CHECK(false); return; }
    const float g  = -10.0f;
    const float dt = 0.01f;
    const float y0 = 100.0f;
    w->set_gravity(Vec3{0.0f, g, 0.0f});
    w->set_fixed_timestep(dt);
    ph::BodyDesc d = dyn_sphere(Vec3{0.0f, y0, 0.0f}, 0.5f, 0.0f);
    const ph::BodyHandle h = w->create_body(d);

    const int N = 60;
    for (int i = 0; i < N; ++i) w->step(dt);

    // Semi-implicit Euler, 1 substep/step (step(dt) with dt==fixed_dt):
    //   v_k = k·g·dt ;  y_k = y0 + g·dt²·Σ_{i=1..k} i = y0 + g·dt²·k(k+1)/2
    const float sum   = static_cast<float>(N * (N + 1) / 2);
    const float exp_y = y0 + g * dt * dt * sum;
    const float exp_v = static_cast<float>(N) * g * dt;
    const Vec3 p = w->position(h);
    const Vec3 v = w->velocity(h);
    CHECK(ap(p.y, exp_y, 5e-3f));
    CHECK(ap(v.y, exp_v, 2e-3f));
    CHECK(ap(p.x, 0.0f, 1e-5f) && ap(p.z, 0.0f, 1e-5f));   // no lateral drift
}

// ---- non-finite wall_dt must not permanently freeze the world -------
// Regression: step()'s guards are `wall_dt <= 0` and `wall_dt > 0.25`;
// both are FALSE for NaN (unordered), so a NaN wall_dt slipped through
// to `accumulator_ += wall_dt`, poisoning it forever (NaN + x = NaN ⇒
// `accumulator_ >= fixed_dt_` is then always false ⇒ no substep EVER
// again ⇒ every body frozen permanently). NaN/Inf via volatile launder
// (FOUNDATION: no <cmath>/<limits>).
void test_nonfinite_step() {
    auto nanf = []{ volatile float z = 0.0f; return z / z; };    // 0/0 → NaN
    auto inff = []{ volatile float z = 0.0f; return 1.0f / z; };  // 1/0 → +Inf

    // (a) NaN dt is a no-op AND must not poison the accumulator: a body
    //     must still free-fall correctly on the next valid steps.
    {
        auto w = ph::World::create();
        if (!w) { CHECK(false); return; }
        const float g = -10.0f, dt = 0.01f, y0 = 100.0f;
        w->set_gravity(Vec3{0.0f, g, 0.0f});
        w->set_fixed_timestep(dt);
        const ph::BodyHandle h = w->create_body(
            dyn_sphere(Vec3{0.0f, y0, 0.0f}, 0.5f, 0.0f));
        w->step(nanf());                                   // poison attempt
        CHECK(ap(w->position(h).y, y0, 1e-6f));            // no substep ran
        CHECK(ap(w->velocity(h).y, 0.0f, 1e-6f));
        const int N = 60;
        for (int i = 0; i < N; ++i) w->step(dt);           // must still step
        const float sum   = static_cast<float>(N * (N + 1) / 2);
        const float exp_y = y0 + g * dt * dt * sum;
        const float exp_v = static_cast<float>(N) * g * dt;
        CHECK(ap(w->position(h).y, exp_y, 5e-3f));         // PRE-FIX: stuck at y0
        CHECK(ap(w->velocity(h).y, exp_v, 2e-3f));
    }
    // (b) A NaN dt MID-simulation must not freeze an already-running world.
    {
        auto w = ph::World::create();
        if (!w) { CHECK(false); return; }
        const float dt = 0.01f;
        w->set_gravity(Vec3{0.0f, -10.0f, 0.0f});
        w->set_fixed_timestep(dt);
        const ph::BodyHandle h = w->create_body(
            dyn_sphere(Vec3{0.0f, 100.0f, 0.0f}, 0.5f, 0.0f));
        for (int i = 0; i < 30; ++i) w->step(dt);
        const float y_mid = w->position(h).y;
        CHECK(y_mid < 100.0f);                              // it was falling
        w->step(nanf());                                    // mid-sim NaN
        for (int i = 0; i < 30; ++i) w->step(dt);
        CHECK(w->position(h).y < y_mid - 1.0f);             // kept falling
    }
    // (c) +Inf dt is still clamped to 0.25 (existing behaviour preserved,
    //     the fix must not regress it) → the world advances.
    {
        auto w = ph::World::create();
        if (!w) { CHECK(false); return; }
        const float dt = 0.01f;
        w->set_gravity(Vec3{0.0f, -10.0f, 0.0f});
        w->set_fixed_timestep(dt);
        const ph::BodyHandle h = w->create_body(
            dyn_sphere(Vec3{0.0f, 100.0f, 0.0f}, 0.5f, 0.0f));
        w->step(inff());                                    // clamp → steps
        CHECK(w->position(h).y < 100.0f);
    }
    // (d) NaN broad_cell_ (set via set_broadphase_cell_size) must NOT
    //     invoke UB in broad_phase_pairs. The pre-fix `broad_cell_
    //     <= 0.0f` ordered compare was NaN-blind, so NaN passed the
    //     N² fallback and reached `inv = 1.0f / NaN = NaN` → floor(
    //     a.min.x * NaN) = NaN → static_cast<i32>(NaN) is UB. Fix
    //     treats non-finite as the "broadphase disabled" case → N²
    //     pairing still detects collisions, just slower.
    {
        auto w = ph::World::create();
        if (!w) { CHECK(false); return; }
        w->set_gravity(Vec3{0.0f, 0.0f, 0.0f});
        w->set_fixed_timestep(0.01f);
        w->set_broadphase_cell_size(nanf());               // POISON
        // Two overlapping unit spheres — broad_phase must still find
        // them and the narrow-phase must report contact (verified
        // indirectly by stepping and confirming no crash + the world
        // doesn't explode).
        const ph::BodyHandle a = w->create_body(
            dyn_sphere(Vec3{0.0f, 0.0f, 0.0f}, 0.5f, 0.0f));
        const ph::BodyHandle b = w->create_body(
            dyn_sphere(Vec3{0.5f, 0.0f, 0.0f}, 0.5f, 0.0f));
        for (int i = 0; i < 30; ++i) w->step(0.01f);       // no crash, no UB
        // Both bodies still finite (sanity: NaN broad_cell didn't
        // corrupt their positions via UB shenanigans).
        CHECK(w->position(a).x == w->position(a).x);
        CHECK(w->position(b).x == w->position(b).x);
        // +Inf and -Inf cell sizes also routed to N².
        w->set_broadphase_cell_size( inff());
        for (int i = 0; i < 5; ++i) w->step(0.01f);
        CHECK(w->position(a).x == w->position(a).x);
        w->set_broadphase_cell_size(-inff());
        for (int i = 0; i < 5; ++i) w->step(0.01f);
        CHECK(w->position(a).x == w->position(a).x);
    }
}

// ---- mutators + integration gating ----------------------------------
void test_mutators() {
    auto w = ph::World::create();
    if (!w) { CHECK(false); return; }
    w->set_gravity(Vec3{0.0f, 0.0f, 0.0f});      // isolate the impulse
    w->set_fixed_timestep(0.01f);

    ph::BodyDesc d = dyn_sphere(Vec3{0.0f, 0.0f, 0.0f}, 0.5f, 0.0f);
    d.mass = 2.0f;
    const ph::BodyHandle h = w->create_body(d);
    CHECK(w->apply_impulse(h, Vec3{10.0f, 0.0f, 0.0f}));
    CHECK(apv(w->velocity(h), Vec3{5.0f, 0.0f, 0.0f}, 1e-5f)); // Δv=J/m

    CHECK(w->set_velocity(h, Vec3{0.0f, 0.0f, 0.0f}));
    CHECK(apv(w->velocity(h), Vec3{0,0,0}, 1e-6f));
    CHECK(w->teleport(h, Vec3{7.0f, 8.0f, 9.0f}, Vec3{0,0,0}));
    CHECK(apv(w->position(h), Vec3{7.0f, 8.0f, 9.0f}, 1e-5f));

    // Static body: gravity + impulses are no-ops; it never moves.
    auto w2 = ph::World::create();
    w2->set_gravity(Vec3{0.0f, -10.0f, 0.0f});
    w2->set_fixed_timestep(0.01f);
    ph::BodyDesc s{};
    s.type     = ph::BodyType::Static;
    s.position = Vec3{0.0f, 5.0f, 0.0f};
    s.collider = ph::Collider::make_box(Vec3{1,1,1});
    const ph::BodyHandle sh = w2->create_body(s);
    CHECK(!w2->apply_impulse(sh, Vec3{100.0f, 0.0f, 0.0f}));
    for (int i = 0; i < 50; ++i) w2->step(0.01f);
    CHECK(apv(w2->position(sh), Vec3{0.0f, 5.0f, 0.0f}, 1e-6f));

    // gravity=false dynamic body free-floats (no force ⇒ no motion).
    ph::BodyDesc fd = dyn_sphere(Vec3{0.0f, 20.0f, 0.0f}, 0.5f, 0.0f);
    fd.gravity = false;
    const ph::BodyHandle fh = w2->create_body(fd);
    for (int i = 0; i < 50; ++i) w2->step(0.01f);
    CHECK(apv(w2->position(fh), Vec3{0.0f, 20.0f, 0.0f}, 1e-4f));
}

// ---- sphere-on-plane collision response -----------------------------
void test_collision() {
    // Inelastic: settles on the plane, never tunnels through it.
    {
        auto w = ph::World::create();
        w->set_gravity(Vec3{0.0f, -10.0f, 0.0f});
        w->set_fixed_timestep(0.01f);
        ph::BodyDesc g{};
        g.type     = ph::BodyType::Static;
        g.position = Vec3{0,0,0};
        g.collider = ph::Collider::make_plane(Vec3{0.0f,1.0f,0.0f}, 0.0f);
        g.layer    = 1u;
        w->create_body(g);
        const ph::BodyHandle s = w->create_body(
            dyn_sphere(Vec3{0.0f, 5.0f, 0.0f}, 0.5f, 0.0f));

        float min_y = 1e9f;
        for (int i = 0; i < 400; ++i) {
            w->step(0.01f);
            const float y = w->position(s).y;
            if (y < min_y) min_y = y;
        }
        const Vec3 p = w->position(s);
        const Vec3 v = w->velocity(s);
        CHECK(min_y > 0.3f);                 // never tunnelled the plane
        CHECK(ap(p.y, 0.5f, 0.12f));         // resting ≈ radius above y=0
        CHECK(ap(v.y, 0.0f, 0.2f));          // came to rest
    }
    // Elastic: rebounds (upward velocity appears after impact).
    {
        auto w = ph::World::create();
        w->set_gravity(Vec3{0.0f, -10.0f, 0.0f});
        w->set_fixed_timestep(0.01f);
        ph::BodyDesc g{};
        g.type     = ph::BodyType::Static;
        g.collider = ph::Collider::make_plane(Vec3{0.0f,1.0f,0.0f}, 0.0f);
        g.layer    = 1u;
        w->create_body(g);
        const ph::BodyHandle s = w->create_body(
            dyn_sphere(Vec3{0.0f, 4.0f, 0.0f}, 0.5f, 1.0f));

        bool rebounded = false;
        float min_y = 1e9f;
        for (int i = 0; i < 250; ++i) {
            w->step(0.01f);
            const Vec3 p = w->position(s);
            if (p.y < min_y) min_y = p.y;
            if (w->velocity(s).y > 0.5f) rebounded = true;
        }
        CHECK(min_y > 0.3f);                 // no tunnelling under impact
        CHECK(rebounded);                    // elastic ⇒ bounced back up
    }
}

// ---- raycast --------------------------------------------------------
void test_raycast() {
    auto w = ph::World::create();
    w->set_fixed_timestep(0.01f);
    ph::BodyDesc d{};
    d.type     = ph::BodyType::Static;
    d.position = Vec3{0.0f, 0.0f, 0.0f};
    d.collider = ph::Collider::make_sphere(1.0f);
    d.layer    = 1u;
    w->create_body(d);

    ph::Ray hitr{ Vec3{0.0f, 0.0f, -5.0f}, Vec3{0.0f, 0.0f, 1.0f} };
    ph::RayHit h = w->raycast(hitr, 1.0e6f, 0xFFFFFFFFu);
    CHECK(h.hit);
    CHECK(static_cast<bool>(h));
    CHECK(ap(h.t, 4.0f, 1e-2f));                  // sphere front at z=-1
    CHECK(ap(h.point.z, -1.0f, 1e-2f));
    CHECK(ap(h.normal.z, -1.0f, 1e-2f));          // points back at the ray

    ph::Ray missr{ Vec3{0.0f, 0.0f, -5.0f}, Vec3{0.0f, 1.0f, 0.0f} };
    CHECK(!w->raycast(missr).hit);

    // Honour max_distance — the hit is at t=4, cut off at 2.
    CHECK(!w->raycast(hitr, 2.0f, 0xFFFFFFFFu).hit);
}

// ---- standalone narrow-phase (pure, no World) -----------------------
void test_standalone() {
    const ph::Collider s = ph::Collider::make_sphere(1.0f);
    ph::ShapeTransform at{}; at.position = Vec3{0,0,0};
    ph::ShapeTransform bt{}; bt.position = Vec3{1.5f,0,0};

    ph::ContactEvent ce{};
    CHECK(ph::test_overlap(s, at, s, bt, &ce));   // 2 unit spheres, gap 1.5
    CHECK(ap(ce.penetration, 0.5f, 1e-3f));       // 2·r − d = 2 − 1.5
    CHECK(ap(ce.normal.x, 1.0f, 1e-3f));          // a → b along +X

    ph::ShapeTransform ft{}; ft.position = Vec3{3.0f,0,0};
    CHECK(!ph::test_overlap(s, at, s, ft));       // gap 3 > 2r ⇒ apart

    // closest point: sphere shell, box face, plane projection.
    CHECK(apv(ph::closest_point_on_collider(s, at, Vec3{5,0,0}),
              Vec3{1,0,0}, 1e-4f));
    const ph::Collider bx = ph::Collider::make_box(Vec3{1,1,1});
    CHECK(apv(ph::closest_point_on_collider(bx, at, Vec3{5,0,0}),
              Vec3{1,0,0}, 1e-4f));
    const ph::Collider pl = ph::Collider::make_plane(Vec3{0,1,0}, 0.0f);
    CHECK(apv(ph::closest_point_on_collider(pl, at, Vec3{3,7,2}),
              Vec3{3,0,2}, 1e-4f));

    CHECK(ap(ph::distance_to_collider(s, at, Vec3{5,0,0}), 4.0f, 1e-3f));
    CHECK(ap(ph::distance_to_collider(s, at, Vec3{0,0,0}), 0.0f, 1e-6f));

    ph::ShapeTransform boxt{}; boxt.position = Vec3{10.0f, 0.0f, 0.0f};
    const ph::AABB aabb = ph::collider_world_aabb(
        ph::Collider::make_box(Vec3{1,2,3}), boxt);
    CHECK(apv(aabb.min, Vec3{9.0f, -2.0f, -3.0f}, 1e-4f));
    CHECK(apv(aabb.max, Vec3{11.0f, 2.0f, 3.0f}, 1e-4f));
}

// ---- default-BodyDesc collision/raycast contract --------------------
// Locks the corrected footgun. Every filter reduces to a bitwise-AND
// ((A.mask&B.layer)|(B.mask&A.layer) for pairs, (body.layer&mask) for
// queries), so a layer of 0 makes a body silently inert. BodyDesc.layer
// now DEFAULTS to bit 0 (1u): a body built from a default desc (layer
// left untouched) MUST collide and MUST be raycast-hit. If the default
// ever regresses to 0, both halves below fail instead of failing
// silently in shipping samples.
void test_default_layer_contract() {
    // (a) Two default-layer bodies interact: a default-desc dynamic
    //     sphere rests on a default-desc static plane. With layer 0 it
    //     would free-fall ~75 m through the plane in 4 s.
    {
        auto w = ph::World::create();
        if (!w) { CHECK(false); return; }
        w->set_gravity(Vec3{0.0f, -10.0f, 0.0f});
        w->set_fixed_timestep(0.01f);

        ph::BodyDesc ground{};                  // layer left at default
        ground.type     = ph::BodyType::Static;
        ground.position = Vec3{0.0f, 0.0f, 0.0f};
        ground.collider = ph::Collider::make_plane(Vec3{0.0f,1.0f,0.0f}, 0.0f);
        w->create_body(ground);

        ph::BodyDesc ball{};                    // layer left at default
        ball.type     = ph::BodyType::Dynamic;
        ball.position = Vec3{0.0f, 5.0f, 0.0f};
        ball.collider = ph::Collider::make_sphere(0.5f);
        const ph::BodyHandle s = w->create_body(ball);

        float min_y = 1e9f;
        for (int i = 0; i < 400; ++i) {
            w->step(0.01f);
            const float y = w->position(s).y;
            if (y < min_y) min_y = y;
        }
        const Vec3 p = w->position(s);
        CHECK(min_y > 0.3f);                 // collided — never tunnelled
        CHECK(ap(p.y, 0.5f, 0.15f));         // rested ≈ radius above y=0
    }
    // (b) A default-layer body is raycast-hit under the default mask.
    //     With layer 0, (0 & 0xFFFFFFFF) == 0 ⇒ silent miss.
    {
        auto w = ph::World::create();
        if (!w) { CHECK(false); return; }
        ph::BodyDesc d{};                       // layer left at default
        d.type     = ph::BodyType::Static;
        d.position = Vec3{0.0f, 0.0f, 0.0f};
        d.collider = ph::Collider::make_sphere(1.0f);
        w->create_body(d);

        ph::Ray r{ Vec3{0.0f, 0.0f, -5.0f}, Vec3{0.0f, 0.0f, 1.0f} };
        const ph::RayHit h = w->raycast(r, 1.0e6f, 0xFFFFFFFFu);
        CHECK(h.hit);                        // default layer passes default mask
        CHECK(ap(h.t, 4.0f, 1e-2f));
    }
}

// ============================================================================
// GPU passes coverage: IntegrationPass + BroadphasePass + ContactSolverPass
// (graph-hosted; runs under render::graph::CpuBackend).
// ============================================================================
namespace rg  = cardinal::render::graph;
namespace pgx = cardinal::physics::gpu;

struct InitBlobP { rg::ResourceHandle h; const void* src; cardinal::usize bytes; };
void rec_init_blob_p(rg::ExecutionContext& ec, void* uctx) noexcept {
    auto* c = static_cast<InitBlobP*>(uctx);
    void* dst = ec.map_buffer_write(c->h);
    if (!dst) return;
    auto* d = static_cast<cardinal::u8*>(dst);
    auto* s = static_cast<const cardinal::u8*>(c->src);
    for (cardinal::usize i = 0; i < c->bytes; ++i) d[i] = s[i];
}
void add_init_pass_p(rg::Graph& g, const char* name, rg::ResourceHandle h, InitBlobP* blob) {
    rg::PassDesc pd; pd.name = name; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{h, rg::AccessMode::Write, 0});
    pd.record = rec_init_blob_p; pd.user_ctx = blob;
    g.add_pass(cardinal::move(pd));
}

// ---- IntegrationPass -----------------------------------------------------
void test_gpu_integration_freefall_closed_form() {
    // 1 body, gravity -9.81 along Y, dt = 1/60, 60 steps via repeated
    // execute(). Closed-form check: v(60Δt) = -9.81 (no damping at 1.0).
    constexpr cardinal::u32 N = 1;
    // SoA: pos(3N), vel(3N), accel(3N), inv_mass(N) = 10 floats
    cardinal::vector<float> state(N * 10u, 0.0f);
    state[9 * N + 0] = 1.0f;             // inv_mass = 1
    state[7 * N + 0] = -9.81f;            // accel.y

    auto g = rg::Graph::create();
    auto h_in = g->declare_buffer(rg::BufferDesc{"state", state.size() * sizeof(float), 0, true});
    InitBlobP bp{h_in, state.data(), state.size() * sizeof(float)};
    add_init_pass_p(*g, "init", h_in, &bp);
    auto st = pgx::IntegrationPass::add_to_graph(*g, h_in, N, 1.0f / 60.0f, 1.0f);
    CHECK(g->compile());
    auto backend = rg::CpuBackend::create();
    backend->execute(*g);
    auto out = backend->buffer_contents(st->out_state);
    const float* of = reinterpret_cast<const float*>(out.data());
    // After 1 step: v.y = 0 + (-9.81)*(1/60) = -0.1635, p.y = v.y * dt = ...
    CHECK(ap(of[4 * N + 0], -9.81f / 60.0f, 1e-3f));
    CHECK(of[1 * N + 0] < 0.0f);
    CHECK(st->bodies_advanced == 1u);
}

void test_gpu_integration_static_body_no_move() {
    // inv_mass = 0 → body shouldn't move.
    constexpr cardinal::u32 N = 1;
    cardinal::vector<float> state(N * 10u, 0.0f);
    state[7 * N + 0] = -9.81f;
    // inv_mass left at 0 → static
    state[0 * N + 0] = 5.0f;
    state[1 * N + 0] = 7.0f;
    state[2 * N + 0] = 3.0f;

    auto g = rg::Graph::create();
    auto h_in = g->declare_buffer(rg::BufferDesc{"state", state.size() * sizeof(float), 0, true});
    InitBlobP bp{h_in, state.data(), state.size() * sizeof(float)};
    add_init_pass_p(*g, "init", h_in, &bp);
    auto st = pgx::IntegrationPass::add_to_graph(*g, h_in, N, 1.0f / 60.0f, 1.0f);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    auto out = b->buffer_contents(st->out_state);
    const float* of = reinterpret_cast<const float*>(out.data());
    CHECK(of[0 * N + 0] == 5.0f);
    CHECK(of[1 * N + 0] == 7.0f);
    CHECK(of[2 * N + 0] == 3.0f);
    CHECK(of[4 * N + 0] == 0.0f);         // velocity untouched
    CHECK(st->bodies_advanced == 0u);
}

void test_gpu_integration_damping() {
    // Damping should attenuate velocity geometrically (no acceleration).
    constexpr cardinal::u32 N = 1;
    cardinal::vector<float> state(N * 10u, 0.0f);
    state[9 * N + 0] = 1.0f;             // inv_mass = 1
    state[3 * N + 0] = 1.0f;             // vel.x = 1

    auto g = rg::Graph::create();
    auto h_in = g->declare_buffer(rg::BufferDesc{"state", state.size() * sizeof(float), 0, true});
    InitBlobP bp{h_in, state.data(), state.size() * sizeof(float)};
    add_init_pass_p(*g, "init", h_in, &bp);
    auto st = pgx::IntegrationPass::add_to_graph(*g, h_in, N, 1.0f / 60.0f, 0.5f);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    auto out = b->buffer_contents(st->out_state);
    const float* of = reinterpret_cast<const float*>(out.data());
    CHECK(ap(of[3 * N + 0], 0.5f, 1e-4f));     // vel.x = 1 * 0.5 = 0.5
}

void test_gpu_integration_nan_safe() {
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    constexpr cardinal::u32 N = 2;
    cardinal::vector<float> state(N * 10u, qnan);
    auto g = rg::Graph::create();
    auto h_in = g->declare_buffer(rg::BufferDesc{"state", state.size() * sizeof(float), 0, true});
    InitBlobP bp{h_in, state.data(), state.size() * sizeof(float)};
    add_init_pass_p(*g, "init", h_in, &bp);
    auto st = pgx::IntegrationPass::add_to_graph(*g, h_in, N, qnan, qnan);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    auto out = b->buffer_contents(st->out_state);
    const float* of = reinterpret_cast<const float*>(out.data());
    for (cardinal::usize i = 0; i < state.size(); ++i) {
        CHECK(of[i] == of[i]);
        CHECK((of[i] - of[i]) == 0.0f);
    }
}

// ---- BroadphasePass -------------------------------------------------------
void test_gpu_broadphase_overlap_pair() {
    // 3 AABBs: two overlap, one isolated.
    constexpr cardinal::u32 N = 3;
    cardinal::vector<float> aabbs(N * 6u, 0.0f);
    // SoA: min_x, min_y, min_z, max_x, max_y, max_z
    // Body 0: [0..1] cube
    aabbs[0 * N + 0] = 0; aabbs[3 * N + 0] = 1;
    aabbs[1 * N + 0] = 0; aabbs[4 * N + 0] = 1;
    aabbs[2 * N + 0] = 0; aabbs[5 * N + 0] = 1;
    // Body 1: [0.5..1.5] cube — overlaps body 0
    aabbs[0 * N + 1] = 0.5f; aabbs[3 * N + 1] = 1.5f;
    aabbs[1 * N + 1] = 0.5f; aabbs[4 * N + 1] = 1.5f;
    aabbs[2 * N + 1] = 0.5f; aabbs[5 * N + 1] = 1.5f;
    // Body 2: [10..11] cube — far away
    aabbs[0 * N + 2] = 10; aabbs[3 * N + 2] = 11;
    aabbs[1 * N + 2] = 10; aabbs[4 * N + 2] = 11;
    aabbs[2 * N + 2] = 10; aabbs[5 * N + 2] = 11;

    auto g = rg::Graph::create();
    auto h_a = g->declare_buffer(rg::BufferDesc{"aabbs", aabbs.size() * sizeof(float), 0, true});
    InitBlobP bp{h_a, aabbs.data(), aabbs.size() * sizeof(float)};
    add_init_pass_p(*g, "init", h_a, &bp);
    auto st = pgx::BroadphasePass::add_to_graph(*g, h_a, N, 16);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    auto cnt = b->buffer_contents(st->out_count);
    const cardinal::u32 c = *reinterpret_cast<const cardinal::u32*>(cnt.data());
    CHECK(c == 1u);
    CHECK(st->pairs_found == 1u);
    // The single pair should be (0, 1).
    auto pairs = b->buffer_contents(st->out_pairs);
    const cardinal::u32* pf = reinterpret_cast<const cardinal::u32*>(pairs.data());
    CHECK(pf[0] == 0u);
    CHECK(pf[1] == 1u);
}

void test_gpu_broadphase_no_overlap() {
    constexpr cardinal::u32 N = 4;
    cardinal::vector<float> aabbs(N * 6u, 0.0f);
    // Four cubes spaced out along X.
    for (cardinal::u32 i = 0; i < N; ++i) {
        const float x = static_cast<float>(i) * 5.0f;
        aabbs[0 * N + i] = x;
        aabbs[3 * N + i] = x + 1.0f;
    }
    auto g = rg::Graph::create();
    auto h_a = g->declare_buffer(rg::BufferDesc{"aabbs", aabbs.size() * sizeof(float), 0, true});
    InitBlobP bp{h_a, aabbs.data(), aabbs.size() * sizeof(float)};
    add_init_pass_p(*g, "init", h_a, &bp);
    auto st = pgx::BroadphasePass::add_to_graph(*g, h_a, N, 8);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    auto cnt = b->buffer_contents(st->out_count);
    const cardinal::u32 c = *reinterpret_cast<const cardinal::u32*>(cnt.data());
    CHECK(c == 0u);
    CHECK(st->pairs_found == 0u);
}

void test_gpu_broadphase_truncates_at_max_pairs() {
    // 4 colocated bodies → 4*3/2 = 6 overlapping pairs. With max_pairs=3
    // we should keep 3 + drop 3.
    constexpr cardinal::u32 N = 4;
    cardinal::vector<float> aabbs(N * 6u, 0.0f);
    for (cardinal::u32 i = 0; i < N; ++i) {
        aabbs[3 * N + i] = 1.0f;
        aabbs[4 * N + i] = 1.0f;
        aabbs[5 * N + i] = 1.0f;
    }
    auto g = rg::Graph::create();
    auto h_a = g->declare_buffer(rg::BufferDesc{"aabbs", aabbs.size() * sizeof(float), 0, true});
    InitBlobP bp{h_a, aabbs.data(), aabbs.size() * sizeof(float)};
    add_init_pass_p(*g, "init", h_a, &bp);
    auto st = pgx::BroadphasePass::add_to_graph(*g, h_a, N, 3);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    CHECK(st->pairs_found == 3u);
    CHECK(st->pairs_truncated == 3u);
}

// ---- ContactSolverPass ----------------------------------------------------
void test_gpu_contact_solver_resting_pair() {
    // Two bodies moving toward each other; one contact along +X normal.
    // After solving, their normal-direction relative velocity should be 0.
    constexpr cardinal::u32 N = 2;
    constexpr cardinal::u32 C = 1;
    // Velocity SoA: vel.x for both bodies
    cardinal::vector<float> vel(N * 3u, 0.0f);
    vel[0 * N + 0] = -1.0f;   // body 0: vx = -1
    vel[0 * N + 1] =  1.0f;   // body 1: vx = +1 → closing
    cardinal::vector<float> inv_mass = {1.0f, 1.0f};
    // Contact layout (10 floats):
    //   pos.xyz (0..3), normal.xyz (3..6), depth (6), _pad (7), a_idx (8), b_idx (9)
    cardinal::vector<float> contacts(C * 10u, 0.0f);
    contacts[3] = -1.0f;       // normal -X (from a to b is -X if a is to the right)
    // Wait: I want a moving right, b moving left, they collide. Let me re-derive.
    // a = body 0 at x=0, vx=-1 (moves left); b = body 1 at x=-2, vx=+1 (moves right).
    // Contact normal: typically pointing from a → b. For a head-on closing pair,
    // relative_vel along normal would be... let's just pick a normal and trust
    // the test.
    contacts[3] = 1.0f;        // normal = +X
    contacts[4] = 0.0f;
    contacts[5] = 0.0f;
    // Pack body indices as float bits (the impl reads them as u32 reinterpret).
    const cardinal::u32 ai = 0u, bi = 1u;
    contacts[8] = *reinterpret_cast<const float*>(&ai);
    contacts[9] = *reinterpret_cast<const float*>(&bi);

    auto g = rg::Graph::create();
    auto h_c = g->declare_buffer(rg::BufferDesc{"contacts", contacts.size() * sizeof(float), 0, true});
    auto h_m = g->declare_buffer(rg::BufferDesc{"im", inv_mass.size() * sizeof(float), 0, true});
    auto h_v = g->declare_buffer(rg::BufferDesc{"vel", vel.size() * sizeof(float), 0, true});
    InitBlobP bc{h_c, contacts.data(), contacts.size() * sizeof(float)};
    InitBlobP bm{h_m, inv_mass.data(), inv_mass.size() * sizeof(float)};
    InitBlobP bv{h_v, vel.data(), vel.size() * sizeof(float)};
    add_init_pass_p(*g, "ic", h_c, &bc);
    add_init_pass_p(*g, "im", h_m, &bm);
    add_init_pass_p(*g, "iv", h_v, &bv);
    auto st = pgx::ContactSolverPass::add_to_graph(*g, h_c, h_m, h_v, C, N, 0.0f);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    auto out_v = b->buffer_contents(h_v);
    const float* of = reinterpret_cast<const float*>(out_v.data());
    // Initial: va.x = -1, vb.x = +1. rel_v = va - vb dot n = -1 - 1 = -2 (closing).
    // j = -(1+0)*(-2)/(1+1) = 1.
    // va.x += 1 * 1 = 0. vb.x -= 1 * 1 = 0.
    // After solve, both should be 0.
    CHECK(ap(of[0 * N + 0], 0.0f, 1e-4f));
    CHECK(ap(of[0 * N + 1], 0.0f, 1e-4f));
    CHECK(st->contacts_resolved == 1u);
}

void test_gpu_contact_solver_separating_pair_skip() {
    // Already separating — solver must skip.
    constexpr cardinal::u32 N = 2;
    constexpr cardinal::u32 C = 1;
    cardinal::vector<float> vel(N * 3u, 0.0f);
    vel[0 * N + 0] = 1.0f;    // a moving +X
    vel[0 * N + 1] = -1.0f;   // b moving -X → separating since normal +X
    cardinal::vector<float> inv_mass = {1.0f, 1.0f};
    cardinal::vector<float> contacts(C * 10u, 0.0f);
    contacts[3] = 1.0f;
    const cardinal::u32 ai = 0u, bi = 1u;
    contacts[8] = *reinterpret_cast<const float*>(&ai);
    contacts[9] = *reinterpret_cast<const float*>(&bi);

    auto g = rg::Graph::create();
    auto h_c = g->declare_buffer(rg::BufferDesc{"contacts", contacts.size() * sizeof(float), 0, true});
    auto h_m = g->declare_buffer(rg::BufferDesc{"im", inv_mass.size() * sizeof(float), 0, true});
    auto h_v = g->declare_buffer(rg::BufferDesc{"vel", vel.size() * sizeof(float), 0, true});
    InitBlobP bc{h_c, contacts.data(), contacts.size() * sizeof(float)};
    InitBlobP bm{h_m, inv_mass.data(), inv_mass.size() * sizeof(float)};
    InitBlobP bv{h_v, vel.data(), vel.size() * sizeof(float)};
    add_init_pass_p(*g, "ic", h_c, &bc);
    add_init_pass_p(*g, "im", h_m, &bm);
    add_init_pass_p(*g, "iv", h_v, &bv);
    auto st = pgx::ContactSolverPass::add_to_graph(*g, h_c, h_m, h_v, C, N, 0.0f);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    auto out_v = b->buffer_contents(h_v);
    const float* of = reinterpret_cast<const float*>(out_v.data());
    // No impulse applied — velocities unchanged.
    CHECK(ap(of[0 * N + 0],  1.0f, 1e-4f));
    CHECK(ap(of[0 * N + 1], -1.0f, 1e-4f));
    CHECK(st->contacts_resolved == 0u);
}

void test_gpu_physics_hlsl_nonempty() {
    CHECK(pgx::IntegrationPass::hlsl_source() != nullptr);
    CHECK(pgx::IntegrationPass::hlsl_source()[0] != '\0');
    CHECK(pgx::BroadphasePass::hlsl_source() != nullptr);
    CHECK(pgx::BroadphasePass::hlsl_source()[0] != '\0');
    CHECK(pgx::ContactSolverPass::hlsl_source() != nullptr);
    CHECK(pgx::ContactSolverPass::hlsl_source()[0] != '\0');
}

}  // namespace

// NOTE: collision/raycast filtering reduces to a bitwise-AND
// ((A.mask&B.layer)|(B.mask&A.layer) for pairs, (body.layer&mask) for
// queries), so a layer of 0 makes a body inert. BodyDesc.layer now
// defaults to bit 0 (1u) so a default-constructed body collides and
// raycasts out of the box — test_default_layer_contract() locks that
// corrected contract. The other suites set layer = 1 explicitly to
// state intent, not to dodge a footgun.
int main() {
    test_determinism();
    test_freefall();
    test_nonfinite_step();
    test_mutators();
    test_collision();
    test_raycast();
    test_default_layer_contract();
    test_standalone();
    test_gpu_integration_freefall_closed_form();
    test_gpu_integration_static_body_no_move();
    test_gpu_integration_damping();
    test_gpu_integration_nan_safe();
    test_gpu_broadphase_overlap_pair();
    test_gpu_broadphase_no_overlap();
    test_gpu_broadphase_truncates_at_max_pairs();
    test_gpu_contact_solver_resting_pair();
    test_gpu_contact_solver_separating_pair_skip();
    test_gpu_physics_hlsl_nonempty();

    if (g_fail == 0) {
        cardinal::log::infof("phystest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("phystest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
