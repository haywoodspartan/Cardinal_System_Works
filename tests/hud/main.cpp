// =============================================================================
// Cardinal — deterministic HUD regression suite.
//
// Hud is a state machine: API calls append primitives to prims_ and
// toasts to toasts_; render() walks prims_ into an ImGui draw list at
// frame end. This suite EXERCISES THE STATE side only (no draw list
// needed) via prim_count() / toast_count() accessors.
//
// Coverage:
//   bar          — non-finite fill_fraction must not silently skip
//                  the foreground (NaN passed both < / > 1 ordered
//                  compares unordered-false → foreground rect SKIPPED
//                  → user sees an empty bar with no explanation).
//   damage_flash — non-finite intensity must not flow into
//                  static_cast<u8>(c.a * intensity) — float→int cast
//                  on a NaN source is UNDEFINED BEHAVIOR.
//   push_toast   — non-finite ToastOpts::duration_s must not poison
//                  Toast::remaining (NaN remaining is never reaped by
//                  `remaining <= 0.0f` → IMMORTAL toast).
//   tick_toasts  — non-finite dt must not poison every toast's
//                  remaining via `t.remaining -= NaN = NaN`.
//
// Pure CPU, headless, fully deterministic. Exit 0 = all pass.
// =============================================================================

#include <cardinal/hud/hud.hpp>
#include <cardinal/core/diag/log.hpp>

namespace {

namespace h = cardinal::hud;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("hudtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-4f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

// Volatile-launder NaN / Inf — no <cmath>/<limits> (foundation style).
float nan_f() { volatile float z = 0.0f; return z / z; }
float inf_f() { volatile float z = 0.0f; return 1.0f / z; }

// ---- bar: finite fill_fraction renders bg + fg + border = 3 prims --
// (zero fill renders bg + border = 2). Non-finite fill MUST render
// the same as zero (bg + border, no foreground rect skipped silently).
void test_bar_fill_fraction() {
    h::Hud hud;
    hud.begin_frame(800, 600);

    // Finite >0: 3 prims.
    hud.bar(0, 0, 100, 10, 0.5f, {0,0,0,255}, {255,255,255,255});
    CHECK(hud.prim_count() == sz(3));

    // Finite ==0: 2 prims (no fg).
    hud.begin_frame(800, 600);
    hud.bar(0, 0, 100, 10, 0.0f, {0,0,0,255}, {255,255,255,255});
    CHECK(hud.prim_count() == sz(2));

    // NaN fill_fraction: with the isfinite guard, behaves as 0 → 2 prims.
    // (Without the guard, the bar drew 0 prims because the foreground
    // check `> 0.0f` is also NaN-blind. THAT was the original mystery.)
    hud.begin_frame(800, 600);
    hud.bar(0, 0, 100, 10, nan_f(), {0,0,0,255}, {255,255,255,255});
    CHECK(hud.prim_count() == sz(2));   // bg + border, no fg

    // ±Inf fill_fraction: clamped to [0,1] by the ordered compares
    // AFTER the isfinite guard (+Inf isn't finite so it hits the
    // isfinite branch first → 0).
    hud.begin_frame(800, 600);
    hud.bar(0, 0, 100, 10, inf_f(), {0,0,0,255}, {255,255,255,255});
    CHECK(hud.prim_count() == sz(2));   // sanitized to 0 → no fg

    hud.begin_frame(800, 600);
    hud.bar(0, 0, 100, 10, -inf_f(), {0,0,0,255}, {255,255,255,255});
    CHECK(hud.prim_count() == sz(2));

    // Sanity: 1.0 (full) still renders 3.
    hud.begin_frame(800, 600);
    hud.bar(0, 0, 100, 10, 1.0f, {0,0,0,255}, {255,255,255,255});
    CHECK(hud.prim_count() == sz(3));
}

// ---- damage_flash: NaN intensity → UB float→u8 cast pre-fix.
// Post-fix: non-finite is rejected up-front (no prim pushed).
void test_damage_flash() {
    h::Hud hud;
    hud.begin_frame(800, 600);

    // Finite >0 → 1 prim (vignette).
    hud.damage_flash(0.5f, {255,0,0,255});
    CHECK(hud.prim_count() == sz(1));

    // Finite <=0 → 0 prims (existing short-circuit).
    hud.begin_frame(800, 600);
    hud.damage_flash(0.0f, {255,0,0,255});
    CHECK(hud.prim_count() == sz(0));
    hud.damage_flash(-1.0f, {255,0,0,255});
    CHECK(hud.prim_count() == sz(0));

    // NaN / ±Inf → 0 prims (rejected by the new isfinite guard).
    // The float→u8 UB cast is NEVER reached.
    hud.begin_frame(800, 600);
    hud.damage_flash(nan_f(), {255,0,0,255});
    CHECK(hud.prim_count() == sz(0));
    hud.damage_flash( inf_f(), {255,0,0,255});
    CHECK(hud.prim_count() == sz(0));
    hud.damage_flash(-inf_f(), {255,0,0,255});
    CHECK(hud.prim_count() == sz(0));

    // Finite >1 — the new clamp prevents u8 overflow but still renders.
    hud.begin_frame(800, 600);
    hud.damage_flash(100.0f, {255,0,0,255});      // huge but finite
    CHECK(hud.prim_count() == sz(1));             // accepted, clamped to 1.0
}

// ---- push_toast: NaN duration_s → IMMORTAL toast pre-fix.
// Post-fix: sanitized to the struct default (2.5s) at registration.
void test_push_toast_duration() {
    h::Hud hud;

    // Finite duration: toast lives until tick_toasts reaps it.
    h::Hud::ToastOpts opts;
    opts.duration_s = 1.0f;
    hud.push_toast("hi", opts);
    CHECK(hud.toast_count() == sz(1));
    hud.tick_toasts(0.5f);
    CHECK(hud.toast_count() == sz(1));            // still alive
    hud.tick_toasts(0.6f);                        // 0.5+0.6=1.1 ≥ 1.0
    CHECK(hud.toast_count() == sz(0));            // reaped

    // NaN duration: must NOT be immortal. Sanitized to 2.5s default.
    h::Hud::ToastOpts bad;
    bad.duration_s = nan_f();
    hud.push_toast("nan", bad);
    CHECK(hud.toast_count() == sz(1));
    // After 1.0s: still alive (sanitized 2.5 > 1.0).
    hud.tick_toasts(1.0f);
    CHECK(hud.toast_count() == sz(1));
    // After 1.6s more: 2.6 ≥ 2.5 → reaped (NOT immortal).
    hud.tick_toasts(1.6f);
    CHECK(hud.toast_count() == sz(0));

    // +Inf duration: also sanitized.
    h::Hud::ToastOpts inf_opts;
    inf_opts.duration_s = inf_f();
    hud.push_toast("inf", inf_opts);
    hud.tick_toasts(3.0f);                        // > 2.5 default
    CHECK(hud.toast_count() == sz(0));
}

// ---- tick_toasts: NaN dt must NOT poison any toast's remaining.
void test_tick_toasts_dt() {
    h::Hud hud;
    h::Hud::ToastOpts opts;
    opts.duration_s = 1.0f;
    hud.push_toast("a", opts);
    hud.push_toast("b", opts);
    CHECK(hud.toast_count() == sz(2));

    // NaN dt: skip the bad frame, no remaining poisoned. Toasts still
    // alive AND still on-schedule to reap at 1.0s of FINITE elapsed.
    hud.tick_toasts(nan_f());
    CHECK(hud.toast_count() == sz(2));
    hud.tick_toasts( inf_f());
    CHECK(hud.toast_count() == sz(2));

    // Finite ticks reap on schedule (proves remaining wasn't poisoned).
    hud.tick_toasts(0.5f);
    CHECK(hud.toast_count() == sz(2));
    hud.tick_toasts(0.6f);                        // 1.1s elapsed
    CHECK(hud.toast_count() == sz(0));
}

}  // namespace

int main() {
    test_bar_fill_fraction();
    test_damage_flash();
    test_push_toast_duration();
    test_tick_toasts_dt();

    if (g_fail == 0) {
        cardinal::log::infof("hudtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("hudtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
