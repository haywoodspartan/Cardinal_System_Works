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
#include <cardinal/core/log.hpp>

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
    test_mutators();
    test_collision();
    test_raycast();
    test_default_layer_contract();
    test_standalone();

    if (g_fail == 0) {
        cardinal::log::infof("phystest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("phystest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
