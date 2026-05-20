// =============================================================================
// Cardinal — deterministic CPU procedural-texture regression suite.
//
// edit::tex_ops emits/edits RGBA8 buffers with pure fixed-point + integer
// hash math — no RNG state — so every byte is reproducible. This suite
// pins:
//
//   * solid           — size = w·h·4, every pixel == the fill color;
//                        get_px/put_px round-trip + (y·w+x)·4 indexing;
//   * checker         — exact cell at sample points; cells_x/y==0 -> 1
//                        (whole image collapses to color `a`);
//   * gradient_linear — X / Y / Diagonal endpoints + the rounded
//                        midpoint (t=0.5 -> 128); degenerate w==1 / h==1;
//   * noise_value     — grayscale + a==255, byte-identical determinism,
//                        seed-sensitive, scale<=0 == scale 1, and the
//                        clean contrast==0 -> flat (127,127,127,255);
//   * noise_fractal   — grayscale + a==255, determinism, octaves 0==1,
//                        base_scale<=0 == 1;
//   * voronoi_cells   — a==255, determinism, site_count 0==1, and
//                        site_count==1 -> a single uniform color;
//   * to_grayscale    — BT.601 weights (R->76 / G->149 / B->29), r==g==b,
//                        alpha preserved;
//   * invert          — 255-c exact, alpha preserved, self-inverse;
//   * levels          — identity is a no-op, gamma>1 lifts midtones,
//                        white<=black collapses to a hard step, gamma<=0
//                        -> 1, endpoints stay 0/255;
//   * channel_swap    — identity / bgra / argb / rrrr, invalid char->0,
//                        len<4 and nullptr are no-ops;
//   * compose_alpha   — opaque src replaces, transparent src is a no-op,
//                        half-alpha blends toward src.
//
// Closed-form pixels are exact; hash-noise is pinned by determinism +
// structure (re-deriving the hash chain by hand would just duplicate the
// impl). Pure, deterministic. Exit 0 = all pass.
// =============================================================================

#include <cardinal/edit/tex_ops.hpp>
#include <cardinal/core/log.hpp>

#include <vector>

namespace {

namespace tx = cardinal::edit::tex_ops;
using tx::Color;
using cardinal::u8;
using cardinal::u32;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("textest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

bool ceq(Color a, Color b) {
    return a.r == b.r && a.g == b.g && a.b == b.b && a.a == b.a;
}
int idiff(int a, int b) { return (a > b) ? (a - b) : (b - a); }
bool cnear(Color a, Color b, int tol) {
    return idiff(a.r, b.r) <= tol && idiff(a.g, b.g) <= tol
        && idiff(a.b, b.b) <= tol && idiff(a.a, b.a) <= tol;
}
// Whole-buffer byte equality (determinism contract).
bool buf_eq(const std::vector<u8>& a, const std::vector<u8>& b) {
    return a.size() == b.size() && a == b;
}
Color C(int r, int g, int b, int a) {
    return Color{ static_cast<u8>(r), static_cast<u8>(g),
                  static_cast<u8>(b), static_cast<u8>(a) };
}

// ---- solid + get/put pixel indexing -------------------------------
void test_solid_pixels() {
    auto img = tx::solid(4u, 3u, C(10, 20, 30, 40));
    CHECK(img.size() == sz(4 * 3 * 4));
    bool all = true;
    for (u32 y = 0; y < 3; ++y)
        for (u32 x = 0; x < 4; ++x)
            if (!ceq(tx::get_px(img, 4u, x, y), C(10, 20, 30, 40))) all = false;
    CHECK(all);

    // put_px writes exactly the (y*w+x)*4 slot; get_px reads it back.
    tx::put_px(img, 4u, 2u, 1u, C(200, 150, 100, 250));
    const cardinal::usize off = (static_cast<cardinal::usize>(1) * 4 + 2) * 4;
    CHECK(img[off + 0] == 200 && img[off + 1] == 150
       && img[off + 2] == 100 && img[off + 3] == 250);
    CHECK(ceq(tx::get_px(img, 4u, 2u, 1u), C(200, 150, 100, 250)));
    // neighbours untouched
    CHECK(ceq(tx::get_px(img, 4u, 1u, 1u), C(10, 20, 30, 40)));
    CHECK(ceq(tx::get_px(img, 4u, 3u, 1u), C(10, 20, 30, 40)));

    CHECK(tx::solid(1u, 1u, C(0, 0, 0, 0)).size() == sz(4));
}

// ---- checker: exact cell pattern + zero-count clamp ---------------
void test_checker() {
    const Color a = C(255, 0, 0, 255);     // red
    const Color b = C(0, 0, 255, 255);     // blue
    auto img = tx::checker(4u, 4u, 2u, 2u, a, b);
    CHECK(img.size() == sz(4 * 4 * 4));
    // cell index = (coord*cells)/dim : x<2 ->0, x>=2 ->1 (same for y).
    CHECK(ceq(tx::get_px(img, 4u, 0u, 0u), a));   // (0+0)&1=0
    CHECK(ceq(tx::get_px(img, 4u, 1u, 0u), a));
    CHECK(ceq(tx::get_px(img, 4u, 2u, 0u), b));   // (1+0)&1=1
    CHECK(ceq(tx::get_px(img, 4u, 0u, 2u), b));   // (0+1)&1=1
    CHECK(ceq(tx::get_px(img, 4u, 2u, 2u), a));   // (1+1)&1=0
    CHECK(ceq(tx::get_px(img, 4u, 3u, 3u), a));
    CHECK(ceq(tx::get_px(img, 4u, 1u, 3u), b));

    // cells_x/y == 0 clamps to 1 -> single cell index 0 -> all `a`.
    auto z = tx::checker(4u, 4u, 0u, 0u, a, b);
    bool alla = true;
    for (u32 y = 0; y < 4; ++y)
        for (u32 x = 0; x < 4; ++x)
            if (!ceq(tx::get_px(z, 4u, x, y), a)) alla = false;
    CHECK(alla);
}

// ---- gradient_linear: endpoints, rounded midpoint, degenerate -----
void test_gradient() {
    const Color a = C(0, 0, 0, 0);
    const Color b = C(255, 255, 255, 255);

    auto gx = tx::gradient_linear(3u, 1u, a, b, tx::Axis::X);
    CHECK(ceq(tx::get_px(gx, 3u, 0u, 0u), C(0, 0, 0, 0)));
    CHECK(ceq(tx::get_px(gx, 3u, 1u, 0u), C(128, 128, 128, 128))); // t=.5
    CHECK(ceq(tx::get_px(gx, 3u, 2u, 0u), C(255, 255, 255, 255)));

    auto gy = tx::gradient_linear(1u, 3u, a, b, tx::Axis::Y);
    CHECK(ceq(tx::get_px(gy, 1u, 0u, 0u), C(0, 0, 0, 0)));
    CHECK(ceq(tx::get_px(gy, 1u, 0u, 1u), C(128, 128, 128, 128)));
    CHECK(ceq(tx::get_px(gy, 1u, 0u, 2u), C(255, 255, 255, 255)));

    auto gd = tx::gradient_linear(2u, 2u, a, b, tx::Axis::Diagonal);
    CHECK(ceq(tx::get_px(gd, 2u, 0u, 0u), C(0, 0, 0, 0)));       // t=0
    CHECK(ceq(tx::get_px(gd, 2u, 1u, 0u), C(128, 128, 128, 128)));// t=.5
    CHECK(ceq(tx::get_px(gd, 2u, 0u, 1u), C(128, 128, 128, 128)));
    CHECK(ceq(tx::get_px(gd, 2u, 1u, 1u), C(255, 255, 255, 255)));// t=1

    // Single column with X axis -> t forced to 0 -> all == a.
    auto deg = tx::gradient_linear(1u, 4u, a, b, tx::Axis::X);
    for (u32 y = 0; y < 4; ++y)
        CHECK(ceq(tx::get_px(deg, 1u, 0u, y), C(0, 0, 0, 0)));
}

// ---- noise_value: structure + determinism + clean contrast=0 ------
void test_noise_value() {
    auto n1 = tx::noise_value(16u, 16u, 1337u, 8.0f, 1.0f);
    auto n2 = tx::noise_value(16u, 16u, 1337u, 8.0f, 1.0f);
    CHECK(n1.size() == sz(16 * 16 * 4));
    CHECK(buf_eq(n1, n2));                              // deterministic

    bool gray = true;
    for (u32 y = 0; y < 16; ++y)
        for (u32 x = 0; x < 16; ++x) {
            Color c = tx::get_px(n1, 16u, x, y);
            if (c.r != c.g || c.g != c.b || c.a != 255) gray = false;
        }
    CHECK(gray);

    auto seedB = tx::noise_value(16u, 16u, 9001u, 8.0f, 1.0f);
    CHECK(!buf_eq(n1, seedB));                          // seed-sensitive

    // scale <= 0 is clamped to 1.0 internally.
    CHECK(buf_eq(tx::noise_value(8u, 8u, 7u, 0.0f, 1.0f),
                 tx::noise_value(8u, 8u, 7u, 1.0f, 1.0f)));

    // contrast 0 -> every sample collapses to 0.5 -> (u8)(127.5)=127.
    auto flat = tx::noise_value(8u, 8u, 42u, 8.0f, 0.0f);
    for (u32 y = 0; y < 8; ++y)
        for (u32 x = 0; x < 8; ++x)
            CHECK(ceq(tx::get_px(flat, 8u, x, y), C(127, 127, 127, 255)));
}

// ---- noise_fractal: structure + determinism + arg clamps ----------
void test_noise_fractal() {
    auto f1 = tx::noise_fractal(16u, 16u, 1337u, 4.0f, 4u, 0.5f);
    auto f2 = tx::noise_fractal(16u, 16u, 1337u, 4.0f, 4u, 0.5f);
    CHECK(f1.size() == sz(16 * 16 * 4));
    CHECK(buf_eq(f1, f2));

    bool gray = true;
    for (u32 y = 0; y < 16; ++y)
        for (u32 x = 0; x < 16; ++x) {
            Color c = tx::get_px(f1, 16u, x, y);
            if (c.r != c.g || c.g != c.b || c.a != 255) gray = false;
        }
    CHECK(gray);

    // octaves 0 clamps to 1.
    CHECK(buf_eq(tx::noise_fractal(8u, 8u, 5u, 4.0f, 0u, 0.5f),
                 tx::noise_fractal(8u, 8u, 5u, 4.0f, 1u, 0.5f)));
    // base_scale <= 0 clamps to 1.0.
    CHECK(buf_eq(tx::noise_fractal(8u, 8u, 5u, 0.0f, 3u, 0.5f),
                 tx::noise_fractal(8u, 8u, 5u, 1.0f, 3u, 0.5f)));
}

// ---- voronoi_cells: determinism, count clamp, single-site uniform -
void test_voronoi() {
    auto v1 = tx::voronoi_cells(16u, 16u, 1337u, 32u);
    auto v2 = tx::voronoi_cells(16u, 16u, 1337u, 32u);
    CHECK(v1.size() == sz(16 * 16 * 4));
    CHECK(buf_eq(v1, v2));
    for (u32 y = 0; y < 16; ++y)
        for (u32 x = 0; x < 16; ++x)
            CHECK(tx::get_px(v1, 16u, x, y).a == 255);

    // site_count 0 clamps to 1.
    CHECK(buf_eq(tx::voronoi_cells(8u, 8u, 3u, 0u),
                 tx::voronoi_cells(8u, 8u, 3u, 1u)));
    // a single site -> every pixel is that site's color.
    auto one = tx::voronoi_cells(8u, 8u, 3u, 1u);
    Color first = tx::get_px(one, 8u, 0u, 0u);
    bool uniform = true;
    for (u32 y = 0; y < 8; ++y)
        for (u32 x = 0; x < 8; ++x)
            if (!ceq(tx::get_px(one, 8u, x, y), first)) uniform = false;
    CHECK(uniform);
}

// ---- to_grayscale: BT.601 weights, r==g==b, alpha kept ------------
void test_grayscale() {
    auto red = tx::solid(2u, 2u, C(255, 0, 0, 200));
    tx::to_grayscale(red, 2u, 2u);
    CHECK(ceq(tx::get_px(red, 2u, 0u, 0u), C(76, 76, 76, 200)));  // .299*255

    auto grn = tx::solid(2u, 2u, C(0, 255, 0, 123));
    tx::to_grayscale(grn, 2u, 2u);
    CHECK(ceq(tx::get_px(grn, 2u, 1u, 1u), C(149, 149, 149, 123)));// .587*255

    auto blu = tx::solid(2u, 2u, C(0, 0, 255, 7));
    tx::to_grayscale(blu, 2u, 2u);
    CHECK(ceq(tx::get_px(blu, 2u, 0u, 1u), C(29, 29, 29, 7)));    // .114*255
}

// ---- invert: exact complement, alpha kept, self-inverse ----------
void test_invert() {
    auto img = tx::solid(3u, 2u, C(10, 20, 30, 40));
    tx::invert(img, 3u, 2u);
    CHECK(ceq(tx::get_px(img, 3u, 0u, 0u), C(245, 235, 225, 40)));
    tx::invert(img, 3u, 2u);                           // invert again
    CHECK(ceq(tx::get_px(img, 3u, 2u, 1u), C(10, 20, 30, 40)));   // back
}

// ---- levels: identity, gamma lift, hard step, clamps -------------
void test_levels() {
    auto id = tx::solid(2u, 2u, C(3, 128, 250, 77));
    tx::levels(id, 2u, 2u, 0.0f, 1.0f, 1.0f);          // identity
    CHECK(cnear(tx::get_px(id, 2u, 0u, 0u), C(3, 128, 250, 77), 1));

    // gamma > 1 lifts a midtone (sqrt(f) >= f on [0,1]); endpoints fixed.
    auto g = tx::solid(2u, 2u, C(0, 64, 255, 255));
    tx::levels(g, 2u, 2u, 0.0f, 1.0f, 2.0f);
    Color gc = tx::get_px(g, 2u, 0u, 0u);
    CHECK(gc.r == 0 && gc.b == 255);                   // endpoints pinned
    CHECK(gc.g > 64);                                  // midtone lifted
    CHECK(gc.a == 255);

    // white_pt <= black_pt -> tiny window -> hard black/white step.
    auto step = tx::solid(2u, 2u, C(0, 0, 0, 255));
    tx::levels(step, 2u, 2u, 0.5f, 0.5f, 1.0f);
    CHECK(tx::get_px(step, 2u, 0u, 0u).r == 0);        // below window -> 0
    auto hi = tx::solid(2u, 2u, C(255, 255, 255, 255));
    tx::levels(hi, 2u, 2u, 0.5f, 0.5f, 1.0f);
    CHECK(tx::get_px(hi, 2u, 0u, 0u).r == 255);        // above window -> 255

    // gamma <= 0 falls back to 1.0 (== identity here).
    auto gz = tx::solid(2u, 2u, C(40, 90, 200, 255));
    tx::levels(gz, 2u, 2u, 0.0f, 1.0f, -3.0f);
    CHECK(cnear(tx::get_px(gz, 2u, 0u, 0u), C(40, 90, 200, 255), 1));
}

// ---- channel_swap: reorder, invalid char, no-op guards -----------
void test_channel_swap() {
    auto base = tx::solid(2u, 2u, C(10, 20, 30, 40));

    auto idn = base;
    tx::channel_swap(idn, 2u, 2u, "rgba");
    CHECK(ceq(tx::get_px(idn, 2u, 0u, 0u), C(10, 20, 30, 40)));

    auto bg = base;
    tx::channel_swap(bg, 2u, 2u, "bgra");              // swap R/B
    CHECK(ceq(tx::get_px(bg, 2u, 1u, 1u), C(30, 20, 10, 40)));

    auto ar = base;
    tx::channel_swap(ar, 2u, 2u, "argb");
    CHECK(ceq(tx::get_px(ar, 2u, 0u, 1u), C(40, 10, 20, 30)));

    auto rr = base;
    tx::channel_swap(rr, 2u, 2u, "rrrr");
    CHECK(ceq(tx::get_px(rr, 2u, 0u, 0u), C(10, 10, 10, 10)));

    auto bad = base;
    tx::channel_swap(bad, 2u, 2u, "xywz");             // invalid -> 0
    CHECK(ceq(tx::get_px(bad, 2u, 0u, 0u), C(0, 0, 0, 0)));

    auto shortl = base;
    tx::channel_swap(shortl, 2u, 2u, "rgb");           // len<4 -> no-op
    CHECK(ceq(tx::get_px(shortl, 2u, 0u, 0u), C(10, 20, 30, 40)));

    auto nul = base;
    tx::channel_swap(nul, 2u, 2u, nullptr);            // null -> no-op
    CHECK(ceq(tx::get_px(nul, 2u, 0u, 0u), C(10, 20, 30, 40)));
}

// ---- compose_alpha: opaque replace, transparent no-op, blend -----
void test_compose() {
    // opaque src (a=255) fully replaces dst.
    auto dst = tx::solid(2u, 2u, C(0, 0, 0, 255));
    auto so  = tx::solid(2u, 2u, C(100, 150, 200, 255));
    tx::compose_alpha(dst, so, 2u, 2u);
    CHECK(ceq(tx::get_px(dst, 2u, 0u, 0u), C(100, 150, 200, 255)));

    // transparent src (a=0) leaves dst rgb unchanged.
    auto dst2 = tx::solid(2u, 2u, C(50, 60, 70, 200));
    auto st   = tx::solid(2u, 2u, C(255, 255, 255, 0));
    tx::compose_alpha(dst2, st, 2u, 2u);
    Color k = tx::get_px(dst2, 2u, 0u, 0u);
    CHECK(k.r == 50 && k.g == 60 && k.b == 70);
    CHECK(k.a >= 199 && k.a <= 200);                   // a = d.a*1

    // half alpha blends toward src: ~ (200*0.502) = 100.
    auto dst3 = tx::solid(2u, 2u, C(0, 0, 0, 255));
    auto sh   = tx::solid(2u, 2u, C(200, 200, 200, 128));
    tx::compose_alpha(dst3, sh, 2u, 2u);
    CHECK(cnear(tx::get_px(dst3, 2u, 0u, 0u), C(100, 100, 100, 254), 1));
}

// ---- Non-finite inputs MUST NOT produce UB float→u8 casts ---------
// fclamp_u8's two ordered compares (`f < 0.0f`, `f > 255.0f`) are
// NaN-blind — NaN falls through to `static_cast<u8>(NaN + 0.5f)`
// which is UNDEFINED BEHAVIOR for the float→int cast. Multiple
// callers can produce NaN: noise_value via NaN contrast, levels via
// NaN gamma/black_pt/white_pt (cardinal::pow propagates NaN), and
// noise_fractal via NaN persistence (sum/norm = NaN). Fix at the
// chokepoint (fclamp_u8) + route the bare cast in noise_value
// through fclamp_u8 — every output byte must be defined regardless
// of finite-or-not input parameters.
void test_nonfinite_inputs() {
    volatile float z = 0.0f;
    const float qnan = z / z;          // 0/0 = NaN
    const float inf  = 1.0f / z;       // 1/0 = +Inf

    // noise_value with NaN contrast — output must be all-finite u8.
    auto a = tx::noise_value(8u, 8u, 1337u, 4.0f, qnan);
    CHECK(a.size() == sz(8 * 8 * 4));
    // Every byte is u8 by type — but the SANITIZED behaviour is that
    // the per-pixel n*255 path mapped NaN → 0. Verify the image is
    // all-black (255 for alpha).
    for (u32 y = 0; y < 8; ++y)
        for (u32 x = 0; x < 8; ++x)
            CHECK(ceq(tx::get_px(a, 8u, x, y), C(0, 0, 0, 255)));

    // noise_value with +Inf contrast — also routes through clamp →
    // fclamp_u8, defined output.
    auto b = tx::noise_value(8u, 8u, 1337u, 4.0f, inf);
    CHECK(b.size() == sz(8 * 8 * 4));   // no crash, no UB

    // noise_fractal with NaN persistence — `amp *= NaN` makes amp NaN
    // after one octave; sum/norm = NaN/NaN = NaN; fclamp_u8(NaN*255)
    // must NOT invoke UB.
    auto c = tx::noise_fractal(8u, 8u, 1337u, 4.0f, 3u, qnan);
    CHECK(c.size() == sz(8 * 8 * 4));

    // levels with NaN gamma — `pow(f, 1/gamma)` cascades NaN into
    // fclamp_u8. Must NOT crash, must produce defined output.
    auto img = tx::solid(4u, 4u, C(128, 128, 128, 255));
    tx::levels(img, 4u, 4u, /*black*/0.0f, /*white*/1.0f, qnan);
    CHECK(img.size() == sz(4 * 4 * 4));

    // levels with NaN black_pt + NaN white_pt — the `white <= black`
    // guard at the function head is NaN-blind so NaN flows through;
    // the inner `(f - black) / (white - black)` is NaN; clamp passes
    // through; fclamp_u8 catches it.
    auto img2 = tx::solid(4u, 4u, C(128, 128, 128, 255));
    tx::levels(img2, 4u, 4u, qnan, qnan, 1.0f);
    CHECK(img2.size() == sz(4 * 4 * 4));
}

}  // namespace

int main() {
    test_solid_pixels();
    test_checker();
    test_gradient();
    test_noise_value();
    test_noise_fractal();
    test_voronoi();
    test_grayscale();
    test_invert();
    test_levels();
    test_channel_swap();
    test_compose();
    test_nonfinite_inputs();

    if (g_fail == 0) {
        cardinal::log::infof("textest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("textest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
