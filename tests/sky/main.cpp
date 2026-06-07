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
#include <cardinal/core/diag/log.hpp>

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

// ---- set_hour / set_day_length_seconds must reject non-finite -----
// Same SECOND-INGRESS NaN-passthrough class as sim::set_time_scale
// (70a9324). The original `if (h < 0.0f)` wrap-up guard was NaN-blind,
// and `if (s < 0.001f) s = 0.001f` was NaN-blind AND +Inf-blind. NaN
// set_hour → state_.hour permanently NaN → recompute_state_() poisons
// every derived field (zenith/horizon/sun_color/sun_intensity/sun_dir)
// via cardinal::clamp NaN-passthrough. NaN set_day_length_seconds →
// time_scale_ = 24/NaN = NaN → every Sky::tick with FINITE real_dt
// still produces NaN hour (real_dt * NaN = NaN), bypassing 6ac4418's
// tick-ingress guard.
void test_setter_nonfinite() {
    sk::Sky s;
    // Establish a known-good state, then feed NaN/+Inf/-Inf set_hour.
    s.set_hour(12.0f);
    CHECK(ap(s.state().hour, 12.0f));
    const float intensity_pre = s.state().sun_intensity;

    s.set_hour(nan_f());
    // NaN routed to the default-noon fallback — recompute_state_ runs
    // on a finite hour, so derived state stays sane.
    CHECK(ap(s.state().hour, 12.0f));
    CHECK(ap(s.state().sun_intensity, intensity_pre));
    CHECK(ap(vlen(s.state().sun_dir), 1.0f, 1e-3f));

    volatile float big = 1.0f; for (int i = 0; i < 16; ++i) big *= 1e30f;  // +Inf
    s.set_hour( big);
    CHECK(ap(s.state().hour, 12.0f));
    s.set_hour(-big);
    CHECK(ap(s.state().hour, 12.0f));

    // set_day_length_seconds: NaN→0.001 min, +Inf→0.001 min. After
    // either, time_scale_ must be finite so subsequent ticks update
    // hour normally. After the NaN call, run a finite tick and verify
    // hour advances (would stay at 12 forever if time_scale_ were NaN).
    s.set_hour(10.0f);
    s.set_day_length_seconds(nan_f());           // → s=0.001, time_scale_=24000
    CHECK(ap(s.day_length_seconds(), 0.001f, 1e-5f));
    s.tick(1.0f / 24000.0f);                     // expect hour += 24000 * (1/24000) = 1
    CHECK(ap(s.state().hour, 11.0f, 1e-2f));     // crossed the threshold

    s.set_day_length_seconds(24.0f);             // restore sane
    s.set_day_length_seconds(big);               // +Inf → clamp
    CHECK(ap(s.day_length_seconds(), 0.001f, 1e-5f));
}

// ---- SkyKey direct-field NaN must NOT poison state_ -----------------
// SkyKey fields are public mutable via Sky::keys(). NaN in any field
// (hour, zenith, horizon, sun_color, sun_intensity, ambient) reaches
// recompute_state_'s lerp_vec (NaN-passthrough) → state_ poisoned →
// renderer reads NaN. NaN hour additionally defeats the segment
// search (`h >= NaN` unordered-false), forcing wrap-around with NaN
// span and NaN t. The safe_key sanitizer coerces non-finite to 0
// BEFORE the segment search and lerp.
void test_key_field_nonfinite() {
    sk::Sky s;
    s.set_hour(12.0f);
    const auto& state = s.state();
    // Sanity: known-good state.
    CHECK(state.hour == state.hour);
    CHECK(state.sun_intensity == state.sun_intensity);

    // Poison the @12 key (which is the active segment for hour 12.0).
    // Find it; if our linear search finds a key with hour==12.0 use it,
    // else poison the back key.
    auto& keys = s.keys();
    CHECK(!keys.empty());
    sk::SkyKey* target = nullptr;
    for (auto& k : keys) if (ap(k.hour, 12.0f)) { target = &k; break; }
    if (!target) target = &keys.back();
    target->sun_intensity = nan_f();
    target->zenith        = cardinal::scene::Vec3{ nan_f(), 0.0f, 0.0f };
    target->sun_color     = cardinal::scene::Vec3{ 0.0f, nan_f(), 0.0f };
    target->ambient       = cardinal::scene::Vec3{ 0.0f, 0.0f, nan_f() };

    // Triggering recompute via set_hour or tick: state_ must stay
    // finite end-to-end. (set_hour to the SAME 12.0 still re-runs
    // recompute_state_ from sky.cpp:96.)
    s.set_hour(12.0f);
    CHECK(state.sun_intensity == state.sun_intensity);    // not NaN
    CHECK(state.zenith.x == state.zenith.x);
    CHECK(state.zenith.y == state.zenith.y);
    CHECK(state.zenith.z == state.zenith.z);
    CHECK(state.horizon.x == state.horizon.x);
    CHECK(state.sun_color.x == state.sun_color.x);
    CHECK(state.sun_color.y == state.sun_color.y);
    CHECK(state.sun_color.z == state.sun_color.z);
    CHECK(state.ambient.z == state.ambient.z);
    // sun_dir must remain unit-length (would be NaN-vector if poisoned).
    CHECK(ap(vlen(s.state().sun_dir), 1.0f, 1e-3f));

    // Also poison hour itself — same sanitization must apply.
    target->hour = nan_f();
    s.set_hour(12.0f);
    CHECK(state.hour == state.hour);
    CHECK(state.sun_intensity == state.sun_intensity);

    // User's stored bad values are preserved (sanitize-at-use, not
    // at storage).
    CHECK(target->sun_intensity != target->sun_intensity);
    CHECK(target->hour != target->hour);
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

// ---- sort_keys must be SWO-safe under NaN hours --------------------
// `[](a,b){ return a.hour < b.hour; }` violates strict-weak-ordering
// when any hour is NaN: NaN compares unordered, so (NaN, x) is treated
// as "equivalent" while (x, y) with x<y is properly ordered — breaks
// transitivity of equivalence. std::sort with a SWO-violating
// comparator is UB; on MSVC's introsort the partition can infinite-
// loop (editor hang) or index OOB (heap corruption). The bad-data
// path is corrupt .sky save (sscanf("%f", ...) accepts "nan"/"inf"
// verbatim → SkyKey.hour = NaN) reaching sort_keys via load_sky:457
// BEFORE any recompute. The UI "Sort by hour" button hits the same
// path after live-edit. Sort completes (no hang/crash), all finite
// hours appear in ascending order, and the NaN hour is sunk to the
// end of the vector as a single equivalence class.
void test_sort_keys_nonfinite() {
    sk::Sky s;
    auto& keys = s.keys();

    // Inject a NaN-hour key in the MIDDLE of the defaults (which are
    // pre-sorted 0..23.99). The middle position maximizes the chance
    // of partition-pathology on a SWO-violating comparator — a leading
    // or trailing NaN can sometimes survive partitioning by luck.
    sk::SkyKey bad{};
    bad.hour = nan_f();
    keys.insert(keys.begin() + (keys.size() / 2), bad);

    // The UB call: with the buggy comparator, this may infinite-loop
    // or scribble OOB. With the SWO-safe comparator, completes in O(n
    // log n) and produces a well-defined permutation.
    s.sort_keys();

    // Finite hours must still be ascending. Walk the prefix until we
    // hit the NaN tail. `x == x` is true for finite (and ±Inf, which
    // we don't inject) but false for NaN — the canonical NaN check
    // without dragging <cmath> into this test (per top-of-file).
    float prev = -1e9f;
    cardinal::usize finite_count = 0;
    for (const auto& k : keys) {
        if (!(k.hour == k.hour)) break;       // NaN ends the run
        CHECK(k.hour >= prev);
        prev = k.hour;
        ++finite_count;
    }
    // Tail is all NaN (just our one injected here).
    for (cardinal::usize i = finite_count; i < keys.size(); ++i) {
        CHECK(!(keys[i].hour == keys[i].hour));
    }
    CHECK(finite_count == keys.size() - 1);

    // recompute_state_ (via set_hour) must still produce a finite,
    // unit-length sun_dir — the safe_key sanitizer handles the NaN
    // tail at read time, defended by sort_keys's NaN-to-end placement.
    s.set_hour(12.0f);
    CHECK(s.state().hour == s.state().hour);
    CHECK(ap(vlen(s.state().sun_dir), 1.0f, 1e-3f));
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
    test_setter_nonfinite();
    test_key_field_nonfinite();
    test_sort_keys_nonfinite();
    test_defaults();

    if (g_fail == 0) {
        cardinal::log::infof("skytest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("skytest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
