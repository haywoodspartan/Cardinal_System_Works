// =============================================================================
// Cardinal — deterministic sky time-of-day regression suite.
//
// Sky::set_hour → recompute_state_ blends the phase keys into the
// scene's lighting (sun_dir / sun_color / ambient / zenith / horizon).
// serial_world only round-trips the KEYS; this pins the interpolation
// that actually produces light. A regression mis-lights the whole
// scene (wrong sun angle / colour / day-night blend) and is invisible
// until it ships. Pure CPU, headless, fully deterministic. sun_dir is
// pinned exactly at the cardinal hours (0/6/12/18 → exact trig, no
// <cmath> needed) and as a unit-length invariant elsewhere. Exit 0 =
// all pass.
// =============================================================================

#include <cardinal/sky/sky.hpp>
#include <cardinal/core/log.hpp>

namespace {

namespace sk = cardinal::sky;

// 1/sqrt(1.01) and 0.1/sqrt(1.01) — sun_dir = normalize(±1, ±1, -0.1).
constexpr float kU = 0.99503719f;
constexpr float kZ = 0.09950372f;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("skytest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-3f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
bool apv(const cardinal::scene::Vec3& v, float x, float y, float z,
         float e = 1e-3f) {
    return ap(v.x, x, e) && ap(v.y, y, e) && ap(v.z, z, e);
}
float vlen(const cardinal::scene::Vec3& v) {
    return (v.x*v.x + v.y*v.y + v.z*v.z);   // squared — compare to 1
}
// volatile-launder a qNaN without dragging <cmath>/<limits> into the
// test (per top-of-file: <cmath> deliberately avoided).
float nan_f() { volatile float z = 0.0f; return z / z; }

// ---- set_hour normalisation ----------------------------------------
void test_set_hour_norm() {
    sk::Sky s;
    s.set_hour(25.0f);   CHECK(ap(s.state().hour, 1.0f));
    s.set_hour(-1.0f);   CHECK(ap(s.state().hour, 23.0f));
    s.set_hour(24.0f);   CHECK(ap(s.state().hour, 0.0f));
    s.set_hour(36.0f);   CHECK(ap(s.state().hour, 12.0f));
    s.set_hour(13.5f);   CHECK(ap(s.state().hour, 13.5f));
    s.set_hour(-25.0f);  CHECK(ap(s.state().hour, 23.0f));   // -25→-1→23
}

// ---- key-exact + midpoint + wrap-segment interpolation -------------
void test_state_interp() {
    sk::Sky s;
    // h==12 lands exactly on the @12 key (segment [7,12], t=1).
    s.set_hour(12.0f);
    CHECK(apv(s.state().zenith,    0.30f, 0.55f, 0.95f));
    CHECK(apv(s.state().horizon,   0.65f, 0.78f, 0.92f));
    CHECK(apv(s.state().sun_color, 1.00f, 0.97f, 0.92f));
    CHECK(ap (s.state().sun_intensity, 1.20f));
    CHECK(apv(s.state().ambient,   0.20f, 0.20f, 0.22f));

    s.set_hour(7.0f);
    CHECK(ap(s.state().sun_intensity, 0.85f));
    CHECK(apv(s.state().zenith, 0.30f, 0.45f, 0.78f));

    s.set_hour(0.0f);
    CHECK(ap(s.state().sun_intensity, 0.00f));
    CHECK(apv(s.state().zenith, 0.02f, 0.02f, 0.06f));

    // Midpoint of [7,12]: t = 0.5.
    s.set_hour(9.5f);
    CHECK(ap(s.state().sun_intensity, 0.85f + (1.20f - 0.85f) * 0.5f));
    CHECK(apv(s.state().zenith, 0.30f, 0.50f, 0.865f));

    // Midpoint of [0,5]: t = 0.5.
    s.set_hour(2.5f);
    CHECK(ap(s.state().sun_intensity, 0.05f));
    CHECK(apv(s.state().zenith, 0.06f, 0.06f, 0.12f));

    // Wrap segment (h > last key 23.99): blends @23.99 → @0; both keys
    // hold identical values, so the result is that stable value and the
    // wrap path is exercised without trig.
    s.set_hour(23.995f);
    CHECK(ap(s.state().sun_intensity, 0.00f));
    CHECK(apv(s.state().zenith, 0.02f, 0.02f, 0.06f));
}

// ---- sun_dir: exact at cardinal hours + unit-length elsewhere ------
void test_sun_dir() {
    sk::Sky s;
    // angle = ((hour-6)/12)*π ; to_sun = (cos,sin,0.1) ; dir = -to_sun.
    s.set_hour(6.0f);    // angle 0     → to_sun (1,0,.1)
    CHECK(apv(s.state().sun_dir, -kU, 0.0f, -kZ));
    s.set_hour(12.0f);   // angle π/2   → to_sun (0,1,.1)  (overhead)
    CHECK(apv(s.state().sun_dir, 0.0f, -kU, -kZ));
    s.set_hour(18.0f);   // angle π     → to_sun (-1,0,.1)
    CHECK(apv(s.state().sun_dir, kU, 0.0f, -kZ));
    s.set_hour(0.0f);    // angle -π/2  → to_sun (0,-1,.1) (below horizon)
    CHECK(apv(s.state().sun_dir, 0.0f, kU, -kZ));

    // Always a unit vector regardless of hour.
    const float hs[5] = { 3.0f, 8.5f, 15.0f, 21.0f, 23.9f };
    for (float h : hs) {
        s.set_hour(h);
        CHECK(ap(vlen(s.state().sun_dir), 1.0f, 1e-4f));
    }
}

// ---- day-length ⇄ time-scale round-trip ----------------------------
void test_day_length() {
    sk::Sky s;
    s.set_day_length_seconds(60.0f);
    CHECK(ap(s.day_length_seconds(), 60.0f, 1e-2f));
    CHECK(ap(s.time_scale(), 24.0f / 60.0f));            // 0.4 h/s
    s.set_day_length_seconds(120.0f);
    CHECK(ap(s.day_length_seconds(), 120.0f, 1e-2f));
    CHECK(ap(s.time_scale(), 0.2f));
    s.set_day_length_seconds(24.0f);
    CHECK(ap(s.time_scale(), 1.0f));
    CHECK(ap(s.day_length_seconds(), 24.0f, 1e-2f));
}

// ---- tick: advance / frozen / wrap ---------------------------------
void test_tick() {
    sk::Sky s;
    s.set_hour(10.0f);
    s.set_day_length_seconds(24.0f);                     // 1 hour / real-sec
    s.tick(2.0f);                                        // 10 → 12
    CHECK(ap(s.state().hour, 12.0f));
    CHECK(ap(s.state().sun_intensity, 1.20f));           // key @12

    s.set_frozen(true);
    s.tick(100.0f);                                      // no-op while frozen
    CHECK(ap(s.state().hour, 12.0f));

    s.set_frozen(false);
    s.tick(13.0f);                                       // 12+13=25 → fmod → 1
    CHECK(ap(s.state().hour, 1.0f));

    s.set_day_length_seconds(2.0f);                      // 12 hours / real-sec
    s.set_hour(0.0f);
    s.tick(0.5f);                                        // 0 + 0.5*12 = 6
    CHECK(ap(s.state().hour, 6.0f));
    CHECK(apv(s.state().sun_dir, -kU, 0.0f, -kZ));       // cardinal hour 6
}

// ---- non-finite real_dt must NOT poison state_.hour ----------------
// Without the guard, `state_.hour += NaN * time_scale_` makes hour NaN.
// The 24-wrap guards are both false for NaN, and recompute_state_()
// then drives EVERY derived field (zenith / horizon / sun_color /
// sun_intensity / sun_dir) to NaN via cardinal::clamp NaN-passthrough
// + lerp. Sky goes black/undefined forever (hour is poisoned). With
// the guard, the bad frame is dropped — state survives intact.
void test_nonfinite_dt() {
    sk::Sky s;
    s.set_hour(10.0f);
    s.set_day_length_seconds(24.0f);                     // 1 hour / real-sec
    s.tick(2.0f);                                        // 10 → 12
    CHECK(ap(s.state().hour, 12.0f));
    const float intensity_pre = s.state().sun_intensity;

    // NaN frame: drop. Hour and all derived state must be unchanged.
    s.tick(nan_f());
    CHECK(ap(s.state().hour, 12.0f));
    CHECK(ap(s.state().sun_intensity, intensity_pre));
    // sun_dir must remain unit-length (would be NaN-vector if poisoned).
    CHECK(ap(vlen(s.state().sun_dir), 1.0f, 1e-3f));

    // Subsequent finite tick resumes advancing the clock normally.
    s.tick(1.0f);                                        // 12 → 13
    CHECK(ap(s.state().hour, 13.0f));
}

// ---- fresh Sky defaults --------------------------------------------
void test_defaults() {
    sk::Sky s;                                           // ctor: hour 12
    CHECK(ap(s.state().hour, 12.0f));
    CHECK(ap(s.state().sun_intensity, 1.20f));
    CHECK(apv(s.state().zenith, 0.30f, 0.55f, 0.95f));
    CHECK(s.keys().size() == static_cast<cardinal::usize>(8));
    CHECK(ap(s.keys().front().hour, 0.0f));
    CHECK(ap(s.keys().back().hour, 23.99f, 1e-3f));
    CHECK(!s.frozen());
}

}  // namespace

int main() {
    test_set_hour_norm();
    test_state_interp();
    test_sun_dir();
    test_day_length();
    test_tick();
    test_nonfinite_dt();
    test_defaults();

    if (g_fail == 0) {
        cardinal::log::infof("skytest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("skytest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
