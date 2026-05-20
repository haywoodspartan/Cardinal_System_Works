// =============================================================================
// Cardinal — deterministic sculpt/paint-brush regression suite.
//
// edit::brush is pure float math with no hidden state. This suite pins:
//
//   * weight_at — distance<0 clamps to center; distance>=radius and
//     radius<=0 return a HARD 0; None=1, Linear=1-t, Smooth=smoothstep
//     of (1-t), Gaussian=exp(-3t²). Rational falloffs are exact at
//     t = 0 / 0.25 / 0.5 / 0.75; Gaussian within an epsilon;
//   * stamp_height_grid — null/zero-dim/cell<=0/radius<=0 and the
//     fully-off-grid clamp all return {any=false}; the returned AABB is
//     the CLAMPED CANDIDATE box (not the tight touched box) whenever any
//     cell took weight; Add accumulates, Subtract decrements, Set snaps
//     to target (w=1, dt_strength>=1), Erase moves half-way per the
//     0.5·dt_strength factor, Smooth replaces a sample with its live
//     4-neighbour average (boundary clamps to self); dt<=0 zeroes the
//     delta but still reports the AABB;
//   * stamp_generic — weight·strength is forwarded only when w>0
//     (distance>=radius samples are skipped), null enumerate/apply are
//     no-ops.
//
// Pure, deterministic. Exit 0 = all pass.
// =============================================================================

#include <cardinal/edit/brush.hpp>
#include <cardinal/core/log.hpp>

#include <functional>
#include <vector>

namespace {

namespace br = cardinal::edit::brush;
using br::Brush;
using br::Falloff;
using br::Mode;
using cardinal::u32;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("brushtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(double a, double b, double eps = 1e-4) {
    const double d = (a > b) ? (a - b) : (b - a);
    return d <= eps;
}

Brush mk(Falloff f, Mode m, double radius, double strength) {
    Brush b;
    b.falloff      = f;
    b.mode         = m;
    b.radius_world = static_cast<float>(radius);
    b.strength     = static_cast<float>(strength);
    return b;
}

// ---- weight_at: the shared falloff kernel -------------------------
void test_weight_at() {
    // radius 2: center & negative distance => t=0 => all curves == 1.
    Brush n = mk(Falloff::None,     Mode::Add, 2.0, 1.0);
    Brush l = mk(Falloff::Linear,   Mode::Add, 2.0, 1.0);
    Brush s = mk(Falloff::Smooth,   Mode::Add, 2.0, 1.0);
    Brush g = mk(Falloff::Gaussian, Mode::Add, 2.0, 1.0);

    CHECK(ap(br::weight_at(0.0f,  n), 1.0));
    CHECK(ap(br::weight_at(0.0f,  l), 1.0));
    CHECK(ap(br::weight_at(0.0f,  s), 1.0));
    CHECK(ap(br::weight_at(0.0f,  g), 1.0));
    CHECK(ap(br::weight_at(-5.0f, l), 1.0));      // distance<0 -> 0 -> t=0
    CHECK(ap(br::weight_at(-5.0f, s), 1.0));

    // t = 0.5  (distance 1, radius 2)
    CHECK(ap(br::weight_at(1.0f, n), 1.0));        // None is flat
    CHECK(ap(br::weight_at(1.0f, l), 0.5));        // 1 - 0.5
    CHECK(ap(br::weight_at(1.0f, s), 0.5));        // s=.5 -> .25*(3-1)=.5
    CHECK(ap(br::weight_at(1.0f, g), 0.47236655, 1e-3)); // exp(-0.75)

    // t = 0.75 (distance 1.5, radius 2)
    CHECK(ap(br::weight_at(1.5f, l), 0.25));
    CHECK(ap(br::weight_at(1.5f, s), 0.15625));    // s=.25 -> .0625*2.5
    CHECK(ap(br::weight_at(1.5f, g), 0.18498038, 1e-3)); // exp(-1.6875)

    // t = 0.25 (distance 1, radius 4)
    Brush l4 = mk(Falloff::Linear, Mode::Add, 4.0, 1.0);
    Brush s4 = mk(Falloff::Smooth, Mode::Add, 4.0, 1.0);
    Brush g4 = mk(Falloff::Gaussian, Mode::Add, 4.0, 1.0);
    CHECK(ap(br::weight_at(1.0f, l4), 0.75));
    CHECK(ap(br::weight_at(1.0f, s4), 0.84375));   // s=.75 -> .5625*1.5
    CHECK(ap(br::weight_at(1.0f, g4), 0.82902911, 1e-3)); // exp(-0.1875)

    // hard cutoff: distance >= radius -> exactly 0 for every falloff.
    CHECK(br::weight_at(2.0f, n) == 0.0f);
    CHECK(br::weight_at(2.0f, l) == 0.0f);
    CHECK(br::weight_at(2.0f, s) == 0.0f);
    CHECK(br::weight_at(2.0f, g) == 0.0f);
    CHECK(br::weight_at(9.0f, s) == 0.0f);

    // radius <= 0 -> always 0.
    Brush r0 = mk(Falloff::None, Mode::Add, 0.0, 1.0);
    Brush rn = mk(Falloff::None, Mode::Add, -3.0, 1.0);
    CHECK(br::weight_at(0.0f, r0) == 0.0f);
    CHECK(br::weight_at(0.0f, rn) == 0.0f);
}

// ---- stamp_height_grid: guards return an empty AABB ---------------
void test_stamp_guards() {
    Brush b = mk(Falloff::None, Mode::Add, 1.0, 1.0);
    float h[4] = {0,0,0,0};

    auto a0 = br::stamp_height_grid(nullptr, 2u, 2u, 1.0f, 0,0, 0,0, 1.0f, b);
    CHECK(!a0.any);
    auto a1 = br::stamp_height_grid(h, 0u, 2u, 1.0f, 0,0, 0,0, 1.0f, b);
    CHECK(!a1.any);
    auto a2 = br::stamp_height_grid(h, 2u, 0u, 1.0f, 0,0, 0,0, 1.0f, b);
    CHECK(!a2.any);
    auto a3 = br::stamp_height_grid(h, 2u, 2u, 0.0f, 0,0, 0,0, 1.0f, b);
    CHECK(!a3.any);                                   // cell <= 0
    Brush rz = mk(Falloff::None, Mode::Add, 0.0, 1.0);
    auto a4 = br::stamp_height_grid(h, 2u, 2u, 1.0f, 0,0, 0,0, 1.0f, rz);
    CHECK(!a4.any);                                   // radius <= 0
    for (int i = 0; i < 4; ++i) CHECK(h[i] == 0.0f);  // never written

    // Brush AABB entirely off the grid -> clamp gives lo>hi -> empty.
    float g2[16] = {0};
    auto off = br::stamp_height_grid(g2, 4u, 4u, 1.0f, 0,0,
                                     100.0f, 100.0f, 1.0f, b);
    CHECK(!off.any);
    for (int i = 0; i < 16; ++i) CHECK(g2[i] == 0.0f);
}

// ---- stamp_height_grid + weight_at MUST NOT invoke UB on NaN ----
// Same float→int UB-cast pattern as world / mesh_ops / tex_ops / scene /
// physics (ad55e96 sweep). The `cell <= 0.0f` and `radius <= 0.0f`
// ordered guards were NaN-blind. NaN cell or radius (or stamp_x/y
// from a degenerate ray-cast in the editor brush input) flowed into:
//   static_cast<int>(floor((NaN ...) / NaN_cell))   — UB
// weight_at's `distance_world < 0.0f` was likewise NaN-blind, allowing
// NaN to reach `exp(-3 * NaN * NaN) = NaN` weight → NaN heightmap.
void test_stamp_nonfinite() {
    volatile float z = 0.0f;
    const float qnan = z / z;
    const float inf  = 1.0f / z;

    Brush b = mk(Falloff::None, Mode::Add, 1.0, 1.0);
    float h[4] = {0,0,0,0};

    // NaN cell → empty AABB (same contract as cell<=0).
    auto n_cell = br::stamp_height_grid(h, 2u, 2u, qnan, 0,0, 0,0, 1.0f, b);
    CHECK(!n_cell.any);
    // +Inf / -Inf cell → same.
    auto i_cell = br::stamp_height_grid(h, 2u, 2u,  inf, 0,0, 0,0, 1.0f, b);
    CHECK(!i_cell.any);
    auto ni_cell = br::stamp_height_grid(h, 2u, 2u, -inf, 0,0, 0,0, 1.0f, b);
    CHECK(!ni_cell.any);

    // NaN radius → empty.
    Brush nr = mk(Falloff::None, Mode::Add, qnan, 1.0);
    auto n_rad = br::stamp_height_grid(h, 2u, 2u, 1.0f, 0,0, 0,0, 1.0f, nr);
    CHECK(!n_rad.any);

    // NaN stamp position → empty.
    auto n_stx = br::stamp_height_grid(h, 2u, 2u, 1.0f, 0,0, qnan,0, 1.0f, b);
    CHECK(!n_stx.any);
    auto n_sty = br::stamp_height_grid(h, 2u, 2u, 1.0f, 0,0, 0,qnan, 1.0f, b);
    CHECK(!n_sty.any);

    // NaN origin → empty.
    auto n_orx = br::stamp_height_grid(h, 2u, 2u, 1.0f, qnan,0, 0,0, 1.0f, b);
    CHECK(!n_orx.any);

    // No write happened on any of the above.
    for (int i = 0; i < 4; ++i) CHECK(h[i] == 0.0f);

    // weight_at(NaN distance) → 0 (out-of-radius semantic) — must not
    // poison the heightmap via NaN weight.
    Brush w = mk(Falloff::Gaussian, Mode::Add, 1.0, 1.0);
    CHECK(br::weight_at(qnan, w) == 0.0f);
    CHECK(br::weight_at( inf, w) == 0.0f);
    CHECK(br::weight_at(-inf, w) == 0.0f);
    // Finite-valid still works (sanity).
    const float wfin = br::weight_at(0.0f, w);
    CHECK(wfin > 0.0f && wfin <= 1.0f);
}

// ---- stamp_height_grid: Add + the candidate-box AABB contract -----
void test_stamp_add_aabb() {
    // 5x5, cell 1, origin (0,0). None falloff, radius 1.5, strength 2,
    // dt 0.5 -> dt_strength = 1.0. Stamp centered on cell (2,2).
    float g[25];
    for (int i = 0; i < 25; ++i) g[i] = 0.0f;
    Brush b = mk(Falloff::None, Mode::Add, 1.5, 2.0);

    auto r = br::stamp_height_grid(g, 5u, 5u, 1.0f, 0,0, 2.0f,2.0f, 0.5f, b);
    CHECK(r.any);
    // AABB is the CLAMPED CANDIDATE box (floor/ceil of world±radius),
    // not the tight set of >0 cells: gx in [floor(0.5),ceil(3.5)] =
    // [0,4] clamped to [0,4].
    CHECK(r.x0 == 0u && r.y0 == 0u && r.x1 == 4u && r.y1 == 4u);

    auto at = [&](int x, int y) { return g[y * 5 + x]; };
    // None => weight 1 for every cell with distance < 1.5 of (2,2):
    // the 3x3 block (1..3,1..3) incl. the sqrt(2)~1.414 diagonals.
    CHECK(ap(at(2,2), 1.0));      // d=0
    CHECK(ap(at(1,2), 1.0));      // d=1
    CHECK(ap(at(1,1), 1.0));      // d=1.414 < 1.5
    CHECK(ap(at(3,3), 1.0));
    CHECK(ap(at(0,0), 0.0));      // d=2.83 -> w=0, untouched
    CHECK(ap(at(0,2), 0.0));      // d=2    -> w=0
    CHECK(ap(at(4,4), 0.0));

    // A second identical stamp accumulates (Add is additive).
    br::stamp_height_grid(g, 5u, 5u, 1.0f, 0,0, 2.0f,2.0f, 0.5f, b);
    CHECK(ap(at(2,2), 2.0));
    CHECK(ap(at(1,1), 2.0));
}

// ---- stamp_height_grid: Subtract / Set / Erase exact deltas -------
void test_stamp_modes() {
    // Single-cell stamps on a 1x1 grid (cell 1, stamp on the cell) so
    // weight is exactly 1 (None) and the math is closed.

    // Subtract: h(10) -= w(1) * dt_strength(3*0.5=1.5) -> 8.5
    {
        float h[1] = {10.0f};
        Brush b = mk(Falloff::None, Mode::Subtract, 0.6, 3.0);
        auto r = br::stamp_height_grid(h, 1u,1u, 1.0f, 0,0, 0,0, 0.5f, b);
        CHECK(r.any && r.x0 == 0u && r.x1 == 0u);
        CHECK(ap(h[0], 8.5));
    }
    // Set: w=1, dt_strength = 4*1 = 4, min(1,4)=1 -> h := target exactly.
    {
        float h[1] = {2.0f};
        Brush b = mk(Falloff::None, Mode::Set, 0.6, 4.0);
        b.target_value = 7.0f;
        br::stamp_height_grid(h, 1u,1u, 1.0f, 0,0, 0,0, 1.0f, b);
        CHECK(ap(h[0], 7.0));
    }
    // Erase: h += (target-h) * w * 0.5 * dt_strength.
    // start 10, target 0, strength 1, dt 1 -> 10 + (-10)*1*0.5*1 = 5.
    {
        float h[1] = {10.0f};
        Brush b = mk(Falloff::None, Mode::Erase, 0.6, 1.0);
        b.target_value = 0.0f;
        br::stamp_height_grid(h, 1u,1u, 1.0f, 0,0, 0,0, 1.0f, b);
        CHECK(ap(h[0], 5.0));
    }
    // dt <= 0 -> dt_strength 0 -> Add delta is 0, but AABB still reports.
    {
        float h[1] = {3.0f};
        Brush b = mk(Falloff::None, Mode::Add, 0.6, 9.0);
        auto r = br::stamp_height_grid(h, 1u,1u, 1.0f, 0,0, 0,0, 0.0f, b);
        CHECK(r.any);                                 // weight>0 -> any
        CHECK(ap(h[0], 3.0));                          // delta zeroed
        auto r2 = br::stamp_height_grid(h, 1u,1u, 1.0f, 0,0, 0,0, -5.0f, b);
        CHECK(r2.any && ap(h[0], 3.0));               // negative dt same
    }
}

// ---- stamp_height_grid: Smooth = live 4-neighbour average ---------
void test_stamp_smooth() {
    // 3x3, cell 1, origin (0,0). Field: center huge, ring known.
    //   layout idx = y*3 + x
    //   (1,0)=4  (0,1)=8  (2,1)=12  (1,2)=16  center(1,1)=100
    float g[9] = {
        0.0f, 4.0f, 0.0f,      // y=0: (0,0)(1,0)(2,0)
        8.0f, 100.0f, 12.0f,   // y=1: (0,1)(1,1)(2,1)
        0.0f, 16.0f, 0.0f      // y=2: (0,2)(1,2)(2,2)
    };
    // tiny radius so ONLY the center cell takes weight (None, w=1),
    // dt_strength = 5 -> min(1,5)=1 -> center := 4-neighbour average.
    Brush b = mk(Falloff::None, Mode::Smooth, 0.5, 5.0);
    auto r = br::stamp_height_grid(g, 3u,3u, 1.0f, 0,0, 1.0f,1.0f, 1.0f, b);
    CHECK(r.any);
    // avg of N(1,0)=4, S(1,2)=16, E(2,1)=12, W(0,1)=8 -> 40*0.25 = 10.
    CHECK(ap(g[1*3 + 1], 10.0));
    // ring untouched (their weight was 0).
    CHECK(ap(g[0*3 + 1], 4.0));
    CHECK(ap(g[1*3 + 0], 8.0));
    CHECK(ap(g[1*3 + 2], 12.0));
    CHECK(ap(g[2*3 + 1], 16.0));

    // Boundary clamp-to-self at corner (0,0): n & w fold back to self.
    float c[9] = {
        20.0f, 2.0f, 0.0f,
        6.0f,  0.0f, 0.0f,
        0.0f,  0.0f, 0.0f
    };
    Brush cb = mk(Falloff::None, Mode::Smooth, 0.5, 5.0);
    br::stamp_height_grid(c, 3u,3u, 1.0f, 0,0, 0.0f,0.0f, 1.0f, cb);
    // (0,0): N=self(20), W=self(20), S=(0,1)=6, E=(1,0)=2
    // avg = (20 + 6 + 2 + 20) * 0.25 = 48 * 0.25 = 12.
    CHECK(ap(c[0], 12.0));
}

// ---- stamp_generic: weight*strength, skip-zero, null guards -------
void test_stamp_generic() {
    Brush b = mk(Falloff::Linear, Mode::Add, 2.0, 3.0);  // strength 3

    std::vector<u32>   idxs;
    std::vector<float> wts;
    auto apply = [&](u32 i, float w) { idxs.push_back(i); wts.push_back(w); };

    auto enumerate = [&](const std::function<void(u32,float)>& touch) {
        touch(0u,  0.0f);    // t=0   -> w=1   -> apply 0, 1*3=3
        touch(1u,  1.0f);    // t=0.5 -> w=0.5 -> apply 1, 0.5*3=1.5
        touch(2u,  2.0f);    // d>=r  -> w=0   -> SKIPPED
        touch(3u, -7.0f);    // d<0   -> w=1   -> apply 3, 3
        touch(4u,  1.5f);    // t=.75 -> w=.25 -> apply 4, 0.75
    };
    br::stamp_generic(b, enumerate, apply);

    CHECK(idxs.size() == 4u);                          // idx 2 skipped
    CHECK(idxs[0]==0u && idxs[1]==1u && idxs[2]==3u && idxs[3]==4u);
    CHECK(ap(wts[0], 3.0));
    CHECK(ap(wts[1], 1.5));
    CHECK(ap(wts[2], 3.0));
    CHECK(ap(wts[3], 0.75));

    // null enumerate / null apply -> no-op (apply must never fire).
    int fired = 0;
    auto count_apply = [&](u32, float) { ++fired; };
    br::stamp_generic(b, nullptr, count_apply);
    br::stamp_generic(b, enumerate, nullptr);
    CHECK(fired == 0);
}

}  // namespace

int main() {
    test_weight_at();
    test_stamp_guards();
    test_stamp_nonfinite();
    test_stamp_add_aabb();
    test_stamp_modes();
    test_stamp_smooth();
    test_stamp_generic();

    if (g_fail == 0) {
        cardinal::log::infof("brushtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("brushtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
