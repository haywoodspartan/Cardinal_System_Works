// =============================================================================
// Cardinal — deterministic frame-pacer regression suite.
//
// FramePacer arbitrates the engine's frame budget. Every render context
// (the main loop = id 0, each editor viewport = 1..N) registers an FPS
// quota; per frame the pacer applies the STRICTEST cap = the minimum
// non-zero contributed cap across all contexts. A cap <= 0 means
// "uncapped" and contributes nothing; no contributing context → no cap.
// A regression here silently mis-paces the whole engine: a too-low cap
// stutters every viewport, a lost cap pins CPU/GPU at 100% (thermal /
// battery). With no OS window bound the foreground probe always reports
// foreground, so every context contributes its fps_foreground value —
// that path + the limit registry are pure and fully deterministic and
// are what this suite pins. The OS sleep DURATION and the real Win32
// foreground/background flip are wall-clock / environment dependent and
// deliberately not asserted (only the uncapped no-sleep return, which
// is Clock-independent, is checked). Exit 0 = all pass.
// =============================================================================

#include <cardinal/core/frame_pacer.hpp>
#include <cardinal/core/log.hpp>

#include <limits>

namespace {

namespace co = cardinal::core;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("pacertest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

// Exact field compare. Every cap the pacer stores or selects is a
// verbatim copy of an input literal (no arithmetic anywhere in the
// registry or the min-selection), so bit-exact == is correct here and
// catches any silent rescale/round regression.
bool fl_eq(const co::FrameLimit& l, float fg, float bg) {
    return l.fps_foreground == fg && l.fps_background == bg;
}

// The min-selection is timing-independent, but reading effective_fps_cap()
// requires a wait_for_next_frame() call which may sleep up to one budget
// (1e9 / cap ns). The caps below are deliberately large so that budget is
// tens of microseconds at most — the suite stays sub-millisecond while
// still exercising the real selection path. The *values* are still pinned
// exactly; only their absolute magnitude is "unrealistic".

// ---- fresh pacer: documented initial state -------------------------
void test_fresh_state() {
    auto p = co::FramePacer::create();
    CHECK(p != nullptr);
    CHECK(p->is_foreground() == true);          // foreground_{true}
    CHECK(p->effective_fps_cap() == 0.0f);      // effective_cap_{0.0f}
    // Unset context → default FrameLimit{0 fg, 30 bg}.
    CHECK(fl_eq(p->limit(co::kPaceMainLoop), 0.0f, 30.0f));
    CHECK(fl_eq(p->limit(99u), 0.0f, 30.0f));
    CHECK(co::kPaceMainLoop == 0u);             // documented main-loop id
}

// ---- limit registry: set / get / overwrite / clear / independence --
void test_registry() {
    auto p = co::FramePacer::create();

    p->set_limit(0u, co::FrameLimit{120.0f, 45.0f});
    CHECK(fl_eq(p->limit(0u), 120.0f, 45.0f));
    CHECK(fl_eq(p->limit(1u), 0.0f, 30.0f));        // others untouched

    p->set_limit(1u, co::FrameLimit{0.0f, 15.0f});
    CHECK(fl_eq(p->limit(1u), 0.0f, 15.0f));
    CHECK(fl_eq(p->limit(0u), 120.0f, 45.0f));      // independent

    // Overwrite same id (no accumulation — last write wins).
    p->set_limit(0u, co::FrameLimit{0.0f, 0.0f});
    CHECK(fl_eq(p->limit(0u), 0.0f, 0.0f));

    // Negative is stored verbatim (selection treats <=0 as uncapped;
    // that behaviour is pinned in test_effective_cap, not here).
    p->set_limit(2u, co::FrameLimit{-5.0f, -1.0f});
    CHECK(fl_eq(p->limit(2u), -5.0f, -1.0f));

    // Clear → that id returns to default; siblings keep their values.
    p->clear_limit(0u);
    CHECK(fl_eq(p->limit(0u), 0.0f, 30.0f));        // back to default
    CHECK(fl_eq(p->limit(1u), 0.0f, 15.0f));        // sibling intact
    CHECK(fl_eq(p->limit(2u), -5.0f, -1.0f));

    // Clearing a never-set id is a no-op (must not throw / corrupt).
    p->clear_limit(7777u);
    CHECK(fl_eq(p->limit(7777u), 0.0f, 30.0f));
    CHECK(fl_eq(p->limit(1u), 0.0f, 15.0f));        // unrelated untouched
}

// ---- effective cap = min non-zero contributed (foreground path) ----
void test_effective_cap() {
    auto p = co::FramePacer::create();   // no window → always foreground

    // No contexts at all → uncapped: wait does not sleep, cap stays 0.
    CHECK(p->wait_for_next_frame() == static_cast<cardinal::u64>(0));
    CHECK(p->effective_fps_cap() == 0.0f);
    CHECK(p->is_foreground() == true);

    // One capped context → that cap.
    p->set_limit(0u, co::FrameLimit{30000.0f, 30.0f});
    p->wait_for_next_frame();
    CHECK(p->effective_fps_cap() == 30000.0f);

    // A stricter context lowers the effective cap.
    p->set_limit(1u, co::FrameLimit{12000.0f, 30.0f});
    p->wait_for_next_frame();
    CHECK(p->effective_fps_cap() == 12000.0f);      // min(30000,12000)

    // A looser context does NOT raise it.
    p->set_limit(2u, co::FrameLimit{90000.0f, 30.0f});
    p->wait_for_next_frame();
    CHECK(p->effective_fps_cap() == 12000.0f);      // still the min

    // fps_foreground == 0 → uncapped, contributes nothing.
    p->set_limit(3u, co::FrameLimit{0.0f, 30.0f});
    p->wait_for_next_frame();
    CHECK(p->effective_fps_cap() == 12000.0f);

    // Negative fps_foreground → also treated as uncapped (c <= 0 skip).
    p->set_limit(4u, co::FrameLimit{-10.0f, 30.0f});
    p->wait_for_next_frame();
    CHECK(p->effective_fps_cap() == 12000.0f);

    // Drop the strictest context → min recomputes upward.
    p->clear_limit(1u);
    p->wait_for_next_frame();
    CHECK(p->effective_fps_cap() == 30000.0f);      // min(30000,90000)

    // Drop everything contributing → uncapped again, no sleep.
    p->clear_limit(0u);
    p->clear_limit(2u);
    p->clear_limit(3u);
    p->clear_limit(4u);
    CHECK(p->wait_for_next_frame() == static_cast<cardinal::u64>(0));
    CHECK(p->effective_fps_cap() == 0.0f);
}

// ---- a NaN cap must be ignored (uncapped), never UB ----------------
// Regression: the old `c <= 0` filter let NaN through (NaN<=0 is
// false); NaN then hit static_cast<long long>(1e9 / NaN) — float->int
// of NaN is UB — and poisoned the public effective_fps_cap(). NaN is
// not a usable cap and must behave exactly like 0 / negative.
void test_nan_cap_uncapped() {
    const float nan_f = std::numeric_limits<float>::quiet_NaN();
    {   // sole context's foreground cap is NaN → uncapped, no sleep.
        auto p = co::FramePacer::create();
        p->set_limit(0u, co::FrameLimit{nan_f, 30.0f});
        CHECK(p->wait_for_next_frame() == static_cast<cardinal::u64>(0));
        CHECK(p->effective_fps_cap() == 0.0f);      // NaN skipped, not propagated
    }
    {   // NaN coexists with a valid cap → valid wins; NaN neither lowers
        // nor corrupts the effective cap.
        auto p = co::FramePacer::create();
        p->set_limit(0u, co::FrameLimit{nan_f,     30.0f});
        p->set_limit(1u, co::FrameLimit{20000.0f,  30.0f});
        p->wait_for_next_frame();
        CHECK(p->effective_fps_cap() == 20000.0f);
    }
}

// ---- uncapped ⇒ wait_for_next_frame returns exactly 0 (no sleep) ---
void test_uncapped_no_sleep() {
    {   // both fields zero on the only context.
        auto p = co::FramePacer::create();
        p->set_limit(co::kPaceMainLoop, co::FrameLimit{0.0f, 0.0f});
        CHECK(p->wait_for_next_frame() == static_cast<cardinal::u64>(0));
        CHECK(p->effective_fps_cap() == 0.0f);
    }
    {   // no limit registered at all.
        auto p = co::FramePacer::create();
        CHECK(p->wait_for_next_frame() == static_cast<cardinal::u64>(0));
        CHECK(p->effective_fps_cap() == 0.0f);
        // Repeated calls stay a no-op (idempotent uncapped path).
        CHECK(p->wait_for_next_frame() == static_cast<cardinal::u64>(0));
        CHECK(p->effective_fps_cap() == 0.0f);
    }
}

// ---- foreground stable when no window is bound ---------------------
void test_foreground_stable() {
    auto p = co::FramePacer::create();
    p->set_window(nullptr);                  // explicit "no window"
    for (int i = 0; i < 5; ++i) {
        p->wait_for_next_frame();
        CHECK(p->is_foreground() == true);   // nullptr hwnd → always FG
    }
    // Still foreground with an active cap in play.
    p->set_limit(0u, co::FrameLimit{50000.0f, 30.0f});
    p->wait_for_next_frame();
    CHECK(p->is_foreground() == true);
    CHECK(p->effective_fps_cap() == 50000.0f);
}

}  // namespace

int main() {
    test_fresh_state();
    test_registry();
    test_effective_cap();
    test_nan_cap_uncapped();
    test_uncapped_no_sleep();
    test_foreground_stable();

    if (g_fail == 0) {
        cardinal::log::infof("pacertest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("pacertest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
