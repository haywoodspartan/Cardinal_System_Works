// =============================================================================
// Cardinal — deterministic fly-camera regression suite.
//
// FlyCamera::tick drives the viewport camera from synthetic FlyInput.
// A regression silently breaks navigation: inverted look, positional
// drift, a missing pitch clamp (gimbal flip), broken sprint/scroll, or
// input leaking while a panel is unhovered. The integration is pure CPU
// and fully deterministic. Every test vector uses an axis-aligned
// camera and yaw/pitch ∈ {0, ±pi/2, pi, pi/4} so the expected result is
// an exact closed form, a sync→forward round-trip, or a unit-length
// invariant — no <cmath> in the test (the engine does the trig; we pin
// the outcome). Exit 0 = all pass.
// =============================================================================

#include <cardinal/scene/fly_camera.hpp>
#include <cardinal/core/log.hpp>

namespace {

namespace sc = cardinal::scene;
using Vec3   = sc::Vec3;

constexpr float kPi      = 3.14159265f;
constexpr float kHalfPi  = 1.57079633f;
constexpr float kQuartPi = 0.78539816f;
constexpr float kDiag    = 2.82842712f;   // 4 / sqrt(2) = 2*sqrt(2)

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("flytest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-3f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
bool apv(const Vec3& v, float x, float y, float z, float e = 1e-3f) {
    return ap(v.x, x, e) && ap(v.y, y, e) && ap(v.z, z, e);
}
float len2(const Vec3& v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
// volatile-launder a qNaN without dragging <cmath>/<limits> in.
float nan_f() { volatile float z = 0.0f; return z / z; }

Vec3 v3(float x, float y, float z) { return Vec3{ x, y, z }; }

sc::Camera cam_at(Vec3 pos, Vec3 tgt) {
    sc::Camera c;
    c.position = pos;
    c.target   = tgt;
    return c;
}

// ---- sync_from: forward direction → yaw / pitch --------------------
void test_sync_from() {
    sc::FlyCamera fc;

    {   // Looking -Z (canonical "forward") → yaw 0, pitch 0.
        auto c = cam_at(v3(0,0,0), v3(0,0,-1));
        fc.sync_from(c);
        CHECK(ap(fc.yaw_rad(), 0.0f));
        CHECK(ap(fc.pitch_rad(), 0.0f));
    }
    {   // +X → yaw +pi/2 ; -X → yaw -pi/2 ; +Z → yaw pi.
        sc::FlyCamera a, b, d;
        auto cx = cam_at(v3(0,0,0), v3(1,0,0));   a.sync_from(cx);
        auto cmx= cam_at(v3(0,0,0), v3(-1,0,0));  b.sync_from(cmx);
        auto cz = cam_at(v3(0,0,0), v3(0,0,1));   d.sync_from(cz);
        CHECK(ap(a.yaw_rad(),  kHalfPi));
        CHECK(ap(a.pitch_rad(), 0.0f));
        CHECK(ap(b.yaw_rad(), -kHalfPi));
        CHECK(ap(d.yaw_rad(),  kPi));
        CHECK(ap(d.pitch_rad(), 0.0f));
    }
    {   // Straight up / down → pitch ±pi/2 (asin(±1)).
        sc::FlyCamera u, dn;
        auto cu = cam_at(v3(0,0,0), v3(0, 1,0));  u.sync_from(cu);
        auto cd = cam_at(v3(0,0,0), v3(0,-1,0));  dn.sync_from(cd);
        CHECK(ap(u.pitch_rad(),  kHalfPi));
        CHECK(ap(dn.pitch_rad(), -kHalfPi));
    }
    {   // 45° up-and-forward → pitch pi/4, yaw 0 (normalises internally).
        auto c = cam_at(v3(0,0,0), v3(0, 1, -1));
        fc.sync_from(c);
        CHECK(ap(fc.pitch_rad(), kQuartPi));
        CHECK(ap(fc.yaw_rad(),   0.0f));
    }
    {   // Non-unit target is normalised before the angle extraction.
        sc::FlyCamera f2;
        auto c = cam_at(v3(0,0,0), v3(0,0,-100));
        f2.sync_from(c);
        CHECK(ap(f2.yaw_rad(), 0.0f) && ap(f2.pitch_rad(), 0.0f));
    }
}

// ---- gating: accept_input + dt<=0 + first-tick auto-sync ----------
void test_gating() {
    {   // Fresh (uninitialised) + accept_input=false: tick still
        // auto-syncs yaw/pitch from the camera, but moves nothing.
        sc::FlyCamera fc;
        auto c = cam_at(v3(0,0,0), v3(1,0,0));
        sc::FlyInput in{};
        in.accept_input = false;
        in.forward = true;
        fc.tick(c, in, 1.0f);
        CHECK(ap(fc.yaw_rad(), kHalfPi));     // synced from +X target
        CHECK(apv(c.position, 0,0,0));        // not moved
        CHECK(apv(c.target,   1,0,0));        // target NOT overwritten
    }
    {   // accept_input=true, no movement: target round-trips to
        //  position + forward(yaw,pitch). +X sync ⇒ forward (1,0,0).
        sc::FlyCamera fc;
        auto c = cam_at(v3(0,0,0), v3(1,0,0));
        sc::FlyInput in{}; in.accept_input = true;
        fc.tick(c, in, 1.0f);
        CHECK(apv(c.position, 0,0,0));
        CHECK(apv(c.target, 1,0,0));          // sync→forward round-trip
        CHECK(apv(c.up, 0,1,0));
    }
    {   // dt <= 0 ⇒ no positional move even with W held; orientation
        //  (target/up) still refreshed.
        sc::FlyCamera fc;
        auto c = cam_at(v3(0,0,0), v3(0,0,-1));
        fc.sync_from(c);
        sc::FlyInput in{}; in.accept_input = true; in.forward = true;
        fc.tick(c, in, 0.0f);
        CHECK(apv(c.position, 0,0,0));
        CHECK(apv(c.target, 0,0,-1));
        fc.tick(c, in, -5.0f);                // negative clamps to 0 too
        CHECK(apv(c.position, 0,0,0));
        // Non-finite dt MUST clamp to 0 same as negatives — without the
        // isfinite guard, NaN passed the `dt <= 0.0f` ordered compare
        // (NaN <= 0 is false) and flowed into cam.position += move * v,
        // teleporting the camera permanently to NaN-land; +Inf did the
        // same to ±Inf.
        fc.tick(c, in, nan_f());
        CHECK(apv(c.position, 0,0,0));        // not poisoned to NaN
        volatile float big = 1.0f;
        for (int i = 0; i < 16; ++i) big *= 1e30f;   // +Inf
        fc.tick(c, in, big);
        CHECK(apv(c.position, 0,0,0));
        fc.tick(c, in, -big);
        CHECK(apv(c.position, 0,0,0));
    }
    {   // accept_input=false AFTER init: tick is a complete no-op
        //  (position AND target untouched — returns before both).
        sc::FlyCamera fc;
        auto c0 = cam_at(v3(0,0,0), v3(0,0,-1));
        fc.sync_from(c0);                     // initialised_ = true
        auto c = cam_at(v3(5,5,5), v3(9,9,9));
        sc::FlyInput in{}; in.accept_input = false; in.forward = true;
        fc.tick(c, in, 1.0f);
        CHECK(apv(c.position, 5,5,5));
        CHECK(apv(c.target,   9,9,9));
        CHECK(ap(fc.yaw_rad(), 0.0f));        // unchanged (synced earlier)
    }
}

// ---- movement: WASDQE, sprint, diagonal normalise, cancel ---------
void test_move() {
    auto fresh = [](sc::FlyCamera& fc, sc::Camera& c) {
        fc = sc::FlyCamera{};                 // reset tunables + state
        c = cam_at(v3(0,0,0), v3(0,0,-1));    //  (speed must not leak
        fc.sync_from(c);                      //   between sub-cases)
    };
    sc::FlyCamera fc; sc::Camera c;
    sc::FlyInput base{}; base.accept_input = true;

    {   // W: forward = (0,0,-1); speed 4, dt 1 ⇒ z = -4.
        fresh(fc, c); auto in = base; in.forward = true;
        fc.tick(c, in, 1.0f);
        CHECK(apv(c.position, 0,0,-4));
        CHECK(apv(c.target,   0,0,-5));       // pos + forward
    }
    {   fresh(fc, c); auto in = base; in.backward = true;
        fc.tick(c, in, 1.0f);  CHECK(apv(c.position, 0,0,4)); }
    {   // D: right = (1,0,0).
        fresh(fc, c); auto in = base; in.right = true;
        fc.tick(c, in, 1.0f);  CHECK(apv(c.position, 4,0,0)); }
    {   fresh(fc, c); auto in = base; in.left = true;
        fc.tick(c, in, 1.0f);  CHECK(apv(c.position, -4,0,0)); }
    {   // E / Q move along world-up regardless of look.
        fresh(fc, c); auto in = base; in.up = true;
        fc.tick(c, in, 1.0f);  CHECK(apv(c.position, 0,4,0)); }
    {   fresh(fc, c); auto in = base; in.down = true;
        fc.tick(c, in, 1.0f);  CHECK(apv(c.position, 0,-4,0)); }

    {   // Sprint multiplies speed (4 * 4 = 16).
        fresh(fc, c); auto in = base; in.forward = true; in.sprint = true;
        fc.tick(c, in, 1.0f);  CHECK(apv(c.position, 0,0,-16)); }

    {   // speed + dt scale linearly (10 * 0.5 = 5).
        fresh(fc, c); fc.speed = 10.0f;
        auto in = base; in.forward = true;
        fc.tick(c, in, 0.5f);  CHECK(apv(c.position, 0,0,-5)); }

    {   // Diagonal is NORMALISED: |displacement| == speed*dt, not larger.
        fresh(fc, c); auto in = base; in.forward = true; in.right = true;
        fc.tick(c, in, 1.0f);
        CHECK(apv(c.position, kDiag, 0.0f, -kDiag, 2e-3f));
        CHECK(ap(len2(c.position), 16.0f, 1e-2f));   // 4^2, not 32
    }
    {   // Opposite inputs cancel ⇒ zero move, no NaN from normalise.
        fresh(fc, c); auto in = base; in.forward = true; in.backward = true;
        fc.tick(c, in, 1.0f);
        CHECK(apv(c.position, 0,0,0));
        CHECK(apv(c.target, 0,0,-1));
    }
    {   // No movement input ⇒ position held, orientation still written.
        fresh(fc, c); auto in = base;
        fc.tick(c, in, 1.0f);
        CHECK(apv(c.position, 0,0,0));
        CHECK(apv(c.target, 0,0,-1));
    }
}

// ---- look: yaw unbounded, pitch clamped ---------------------------
void test_look() {
    {   // dx → yaw += dx*sens ; dy → pitch -= dy*sens.
        sc::FlyCamera fc;
        auto c = cam_at(v3(0,0,0), v3(0,0,-1));
        fc.sync_from(c);
        sc::FlyInput in{}; in.accept_input = true; in.look = true;
        in.mouse_dx = 100.0f;                 // 100 * 0.0035 = 0.35
        fc.tick(c, in, 1.0f);
        CHECK(ap(fc.yaw_rad(), 0.35f));
        CHECK(ap(fc.pitch_rad(), 0.0f));

        in.mouse_dx = 0.0f; in.mouse_dy = 200.0f;   // pitch -= 0.7
        fc.tick(c, in, 1.0f);
        CHECK(ap(fc.yaw_rad(), 0.35f));       // dx 0 ⇒ yaw unchanged
        CHECK(ap(fc.pitch_rad(), -0.7f));
    }
    {   // Pitch clamps to [min,max]; huge dy can't flip the camera.
        sc::FlyCamera fc;
        auto c = cam_at(v3(0,0,0), v3(0,0,-1));
        fc.sync_from(c);
        sc::FlyInput in{}; in.accept_input = true; in.look = true;
        in.mouse_dy = 100000.0f;              // pitch → -inf, clamp min
        fc.tick(c, in, 1.0f);
        CHECK(ap(fc.pitch_rad(), -1.553f, 2e-3f));
        in.mouse_dy = -100000.0f;             // pitch → +inf, clamp max
        fc.tick(c, in, 1.0f);
        CHECK(ap(fc.pitch_rad(), 1.553f, 2e-3f));
        // forward stays a unit vector at the clamp extreme.
        CHECK(ap(len2(c.target - c.position), 1.0f, 1e-2f));
    }
    {   // Yaw is FREE — accumulates with no wrap to [-pi,pi].
        sc::FlyCamera fc;
        auto c = cam_at(v3(0,0,0), v3(0,0,-1));
        fc.sync_from(c);
        sc::FlyInput in{}; in.accept_input = true; in.look = true;
        in.mouse_dx = 100000.0f;              // 100000 * 0.0035 = 350
        fc.tick(c, in, 1.0f);
        CHECK(ap(fc.yaw_rad(), 350.0f, 1e-1f));
    }
    {   // look=false ⇒ mouse delta ignored entirely.
        sc::FlyCamera fc;
        auto c = cam_at(v3(0,0,0), v3(0,0,-1));
        fc.sync_from(c);
        sc::FlyInput in{}; in.accept_input = true; in.look = false;
        in.mouse_dx = 500.0f; in.mouse_dy = 500.0f;
        fc.tick(c, in, 1.0f);
        CHECK(ap(fc.yaw_rad(), 0.0f));
        CHECK(ap(fc.pitch_rad(), 0.0f));
    }
}

// ---- scroll: multiplicative speed adjust, clamped -----------------
void test_scroll() {
    auto run = [](float scroll, float dt, bool accept,
                  float start_speed) -> float {
        sc::FlyCamera fc;
        fc.speed = start_speed;
        auto c = cam_at(v3(0,0,0), v3(0,0,-1));
        fc.sync_from(c);
        sc::FlyInput in{}; in.accept_input = accept; in.scroll = scroll;
        fc.tick(c, in, dt);
        return fc.speed;
    };

    CHECK(ap(run(0.0f,    1.0f, true, 4.0f), 4.0f));      // no scroll
    CHECK(ap(run(0.0005f, 1.0f, true, 4.0f), 4.0f));      // below threshold
    CHECK(ap(run(1.0f,    1.0f, true, 4.0f), 4.8f));      // *1.2
    CHECK(ap(run(2.0f,    1.0f, true, 4.0f), 5.76f, 1e-2f)); // *1.2^2
    CHECK(ap(run(-1.0f,   1.0f, true, 4.0f), 3.3333f, 2e-3f)); // /1.2
    CHECK(ap(run(200.0f,  1.0f, true, 4.0f), 200.0f));    // clamp to max
    CHECK(ap(run(-200.0f, 1.0f, true, 4.0f), 0.25f));     // clamp to min
    CHECK(ap(run(1.0f,    0.0f, true, 4.0f), 4.8f));      // applies even dt=0
    CHECK(ap(run(10.0f,   1.0f, false, 4.0f), 4.0f));     // gated by accept
}

// ---- FlyCamera owns orientation after init (no re-sync) -----------
void test_no_resync() {
    sc::FlyCamera fc;
    auto c = cam_at(v3(0,0,0), v3(0,0,-1));
    fc.sync_from(c);                          // yaw 0, pitch 0
    CHECK(ap(fc.yaw_rad(), 0.0f));

    // Externally slew the camera target as if some other system moved it.
    c.target = v3(1, 0, 0);
    sc::FlyInput in{}; in.accept_input = true;   // no move, no look
    fc.tick(c, in, 1.0f);

    // tick must NOT re-derive yaw from the new target; instead it
    // overwrites target from its own (unchanged) yaw/pitch.
    CHECK(ap(fc.yaw_rad(), 0.0f));
    CHECK(apv(c.target, 0,0,-1));             // forced back to forward(0,0)
}

}  // namespace

int main() {
    test_sync_from();
    test_gating();
    test_move();
    test_look();
    test_scroll();
    test_no_resync();

    if (g_fail == 0) {
        cardinal::log::infof("flytest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("flytest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
