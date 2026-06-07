// =============================================================================
// Cardinal — deterministic tessellation-factor regression suite.
//
// render::tess is header-only pure float math (the CPU twin of
// cardinal_tessellation.hlsli). This suite pins:
//
//   * clamp_factor      — the hardware [1,64] clamp incl. both endpoints;
//   * factor_distance   — f = scale / max(d, 0.001); the 0.001 floor
//     means distance 0 / negative can't divide-by-zero and instead
//     saturates to max; custom min/max window honoured;
//   * factor_edge       — f = edge_px / target; target<1 is forced to 1
//     (no blow-up); custom min/max window honoured;
//   * phong_blend_weight— 1-|n.n| clamped to [0,1]: 0 when normals
//     coincide (n.n = ±1), 1 when perpendicular, symmetric in sign,
//     out-of-domain |n.n|>1 clamps to 0;
//   * quality_scale     — Off/Low/Medium/High = 1/8/16/32 and the
//     out-of-range fallback = 1; composes with factor_distance.
//
// Every output is verified to land inside [1,64]. Pure, deterministic.
// Exit 0 = all pass.
// =============================================================================

#include <cardinal/render/tess.hpp>
#include <cardinal/core/diag/log.hpp>

namespace {

namespace ts = cardinal::render::tess;
using ts::Quality;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("tesstest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(double a, double b, double eps = 1e-4) {
    const double d = (a > b) ? (a - b) : (b - a);
    return d <= eps;
}
bool in_hw_range(float f) { return f >= 1.0f && f <= 64.0f; }
Quality q(int v) { return static_cast<Quality>(static_cast<cardinal::u32>(v)); }

// ---- clamp_factor: the [1,64] hardware clamp ----------------------
void test_clamp_factor() {
    CHECK(ap(ts::clamp_factor(0.5f), 1.0));      // below floor
    CHECK(ap(ts::clamp_factor(0.0f), 1.0));
    CHECK(ap(ts::clamp_factor(-9.0f), 1.0));
    CHECK(ap(ts::clamp_factor(1.0f), 1.0));      // exact floor stays
    CHECK(ap(ts::clamp_factor(30.0f), 30.0));    // interior untouched
    CHECK(ap(ts::clamp_factor(64.0f), 64.0));    // exact ceil stays
    CHECK(ap(ts::clamp_factor(100.0f), 64.0));   // above ceil
    CHECK(ap(ts::clamp_factor(1e9f), 64.0));
    CHECK(ap(ts::kMaxFactor, 64.0) && ap(ts::kMinFactor, 1.0));
}

// ---- factor_distance: scale/d with a 0.001 distance floor ---------
void test_factor_distance() {
    CHECK(ap(ts::factor_distance(2.0f, 8.0f), 4.0));     // 8/2
    CHECK(ap(ts::factor_distance(8.0f, 8.0f), 1.0));     // 8/8
    CHECK(ap(ts::factor_distance(10.0f, 5.0f), 1.0));    // 0.5 -> clamp lo
    CHECK(ap(ts::factor_distance(1.0f, 100.0f), 64.0));  // 100 -> clamp hi

    // distance <= 0.001 floored to 0.001 -> no div-by-zero, saturates.
    CHECK(ap(ts::factor_distance(0.0f, 1.0f), 64.0));    // 1/0.001=1000
    CHECK(ap(ts::factor_distance(-5.0f, 1.0f), 64.0));
    CHECK(ap(ts::factor_distance(0.0005f, 1.0f), 64.0));

    // custom [min,max] window.
    CHECK(ap(ts::factor_distance(2.0f, 8.0f, 5.0f, 50.0f), 5.0));  // 4->5
    CHECK(ap(ts::factor_distance(1.0f, 1000.0f, 1.0f, 30.0f), 30.0));
    CHECK(ap(ts::factor_distance(4.0f, 16.0f, 2.0f, 8.0f), 4.0));  // mid

    CHECK(in_hw_range(ts::factor_distance(0.0f, 999.0f)));
    CHECK(in_hw_range(ts::factor_distance(123.0f, 0.01f)));
}

// ---- factor_edge: edge_px/target with target<1 guard --------------
void test_factor_edge() {
    CHECK(ap(ts::factor_edge(160.0f, 16.0f), 10.0));
    CHECK(ap(ts::factor_edge(64.0f), 4.0));              // default target 16
    CHECK(ap(ts::factor_edge(48.0f), 3.0));
    CHECK(ap(ts::factor_edge(8.0f, 16.0f), 1.0));        // 0.5 -> clamp lo
    CHECK(ap(ts::factor_edge(4000.0f, 16.0f), 64.0));    // -> clamp hi

    // target_pixels_per_segment < 1 is forced up to 1 (no blow-up).
    CHECK(ap(ts::factor_edge(10.0f, 0.5f), 10.0));       // target->1
    CHECK(ap(ts::factor_edge(32.0f, 0.0f), 32.0));
    CHECK(ap(ts::factor_edge(20.0f, -4.0f), 20.0));

    // custom [min,max] window.
    CHECK(ap(ts::factor_edge(160.0f, 16.0f, 1.0f, 8.0f), 8.0));   // 10->8
    CHECK(ap(ts::factor_edge(8.0f, 16.0f, 2.0f, 64.0f), 2.0));    // .5->2

    CHECK(in_hw_range(ts::factor_edge(1.0f)));
    CHECK(in_hw_range(ts::factor_edge(99999.0f)));
}

// ---- phong_blend_weight: 1-|n.n| clamped, sign-symmetric ----------
void test_phong_blend() {
    CHECK(ap(ts::phong_blend_weight(1.0f), 0.0));    // coincident -> flat
    CHECK(ap(ts::phong_blend_weight(-1.0f), 0.0));   // anti-parallel too
    CHECK(ap(ts::phong_blend_weight(0.0f), 1.0));    // perpendicular -> full
    CHECK(ap(ts::phong_blend_weight(0.5f), 0.5));
    CHECK(ap(ts::phong_blend_weight(-0.5f), 0.5));   // symmetric in sign
    CHECK(ap(ts::phong_blend_weight(0.25f), 0.75));
    CHECK(ap(ts::phong_blend_weight(-0.75f), 0.25));
    // out-of-domain |n.n| > 1 -> negative k -> clamped to 0.
    CHECK(ap(ts::phong_blend_weight(2.0f), 0.0));
    CHECK(ap(ts::phong_blend_weight(-3.0f), 0.0));
    // result is always a valid [0,1] weight.
    CHECK(ts::phong_blend_weight(0.3f) >= 0.0f
       && ts::phong_blend_weight(0.3f) <= 1.0f);
}

// ---- quality_scale: Off/Low/Medium/High + fallback ----------------
void test_quality_scale() {
    CHECK(ap(ts::quality_scale(Quality::Off),    1.0));
    CHECK(ap(ts::quality_scale(Quality::Low),    8.0));
    CHECK(ap(ts::quality_scale(Quality::Medium), 16.0));
    CHECK(ap(ts::quality_scale(Quality::High),   32.0));
    CHECK(ap(ts::quality_scale(q(99)), 1.0));        // out-of-range -> 1

    // enum ordinals are the stable knob order Off<Low<Medium<High.
    CHECK(static_cast<cardinal::u32>(Quality::Off)    == 0u);
    CHECK(static_cast<cardinal::u32>(Quality::Low)    == 1u);
    CHECK(static_cast<cardinal::u32>(Quality::Medium) == 2u);
    CHECK(static_cast<cardinal::u32>(Quality::High)   == 3u);

    // composes with factor_distance: quality picks the scale.
    CHECK(ap(ts::factor_distance(2.0f, ts::quality_scale(Quality::High)),
             16.0));                                  // 32/2
    CHECK(ap(ts::factor_distance(0.5f, ts::quality_scale(Quality::Off)),
             2.0));                                   // 1/0.5
    CHECK(ap(ts::factor_distance(1.0f, ts::quality_scale(Quality::Low)),
             8.0));                                   // 8/1
}

}  // namespace

int main() {
    test_clamp_factor();
    test_factor_distance();
    test_factor_edge();
    test_phong_blend();
    test_quality_scale();

    if (g_fail == 0) {
        cardinal::log::infof("tesstest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("tesstest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
