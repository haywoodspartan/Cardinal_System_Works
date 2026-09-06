// =============================================================================
// Cardinal — cui_rhi native renderer geometry regression suite.
//
// Renderer::build_geometry (DrawList -> colored-quad vertex stream) is the part
// of the native UI renderer that previously only ran live on the GPU. It's pure
// + device-free, so we pin it here: vertex counts, corner positions, packed
// color, and the embedded-font text expansion. Headless. Exit 0 = all pass.
// =============================================================================

#include <cardinal/cui_rhi/renderer.hpp>
#include <cardinal/cui/draw.hpp>
#include <cardinal/core/diag/log.hpp>

namespace {

namespace cr  = cardinal::cui_rhi;
namespace cui = cardinal::cui;

int g_checks = 0, g_fail = 0;
void check_impl(bool ok, const char* e, int l) {
    ++g_checks;
    if (!ok) { ++g_fail; cardinal::log::errorf("crtest", "FAIL  L%d  %s", l, e); }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)
bool ap(float a, float b, float e = 1e-3f) { float d = a > b ? a - b : b - a; return d <= e; }

// A filled rect -> one quad (two triangles = 6 verts), top-left first.
void test_rect() {
    cui::DrawList dl;
    dl.rect_filled(cui::Rect{ {10.0f, 20.0f}, {100.0f, 40.0f} }, cui::Color{255, 0, 0, 255});
    cardinal::vector<cr::Vertex> v;
    cr::Renderer::build_geometry(dl, v);
    CHECK(v.size() == 6u);
    CHECK(ap(v[0].x, 10.0f) && ap(v[0].y, 20.0f));               // top-left corner
    CHECK(v[0].rgba == cr::pack_color(cui::Color{255, 0, 0, 255}));
}

// A line -> one oriented quad (6 verts). Text -> a quad per set font pixel.
void test_line_and_text() {
    cui::DrawList dl;
    dl.line({0.0f, 0.0f}, {10.0f, 0.0f}, cui::Color{255, 255, 255, 255}, 2.0f);
    cardinal::vector<cr::Vertex> v;
    cr::Renderer::build_geometry(dl, v);
    CHECK(v.size() == 6u);

    // 'A' in the 8x8 font has 18 set pixels -> 18 quads -> 108 verts.
    cui::DrawList dt;
    dt.text({0.0f, 0.0f}, "A", cui::Color{255, 255, 255, 255}, 16.0f);
    cardinal::vector<cr::Vertex> vt;
    cr::Renderer::build_geometry(dt, vt);
    CHECK(vt.size() == 18u * 6u);

    // Empty string emits nothing.
    cui::DrawList de;
    de.text({0.0f, 0.0f}, "", cui::Color{}, 16.0f);
    cardinal::vector<cr::Vertex> ve;
    cr::Renderer::build_geometry(de, ve);
    CHECK(ve.empty());

    // A space is blank (no set pixels) -> no quads.
    cui::DrawList ds;
    ds.text({0.0f, 0.0f}, " ", cui::Color{}, 16.0f);
    cardinal::vector<cr::Vertex> vs;
    cr::Renderer::build_geometry(ds, vs);
    CHECK(vs.empty());
}

// pack_color maps R,G,B,A into ascending bytes (R8G8B8A8_UNORM).
void test_pack_color() {
    const cardinal::u32 c = cr::pack_color(cui::Color{0x11, 0x22, 0x33, 0x44});
    CHECK((c & 0xFFu) == 0x11u);
    CHECK(((c >> 8)  & 0xFFu) == 0x22u);
    CHECK(((c >> 16) & 0xFFu) == 0x33u);
    CHECK(((c >> 24) & 0xFFu) == 0x44u);
}

// Software clipping: the native pipeline has no scissor, so build_geometry
// clips every primitive to its DrawCmd clip rect.
void test_clip() {
    // Filled rect partially inside -> clipped quad bounds.
    cui::DrawList dl;
    dl.push_clip(cui::Rect{ {0.0f, 0.0f}, {50.0f, 50.0f} });
    dl.rect_filled(cui::Rect{ {10.0f, 20.0f}, {100.0f, 40.0f} }, cui::Color{255, 0, 0, 255});
    cardinal::vector<cr::Vertex> v;
    cr::Renderer::build_geometry(dl, v);
    CHECK(v.size() == 6u);
    float maxx = -1.0f, maxy = -1.0f, minx = 1.0e9f, miny = 1.0e9f;
    for (const auto& q : v) {
        if (q.x > maxx) maxx = q.x; if (q.y > maxy) maxy = q.y;
        if (q.x < minx) minx = q.x; if (q.y < miny) miny = q.y;
    }
    CHECK(ap(minx, 10.0f) && ap(miny, 20.0f) && ap(maxx, 50.0f) && ap(maxy, 50.0f));

    // Fully outside -> dropped. Stroke outside -> dropped too.
    cui::DrawList d2;
    d2.push_clip(cui::Rect{ {0.0f, 0.0f}, {50.0f, 50.0f} });
    d2.rect_filled(cui::Rect{ {100.0f, 100.0f}, {10.0f, 10.0f} }, cui::Color{});
    d2.rect_stroke(cui::Rect{ {60.0f, 60.0f}, {10.0f, 10.0f} }, cui::Color{}, 1.0f);
    cardinal::vector<cr::Vertex> v2;
    cr::Renderer::build_geometry(d2, v2);
    CHECK(v2.empty());

    // Line crossing the clip -> trimmed to the clip's x range.
    cui::DrawList d3;
    d3.push_clip(cui::Rect{ {0.0f, 0.0f}, {50.0f, 50.0f} });
    d3.line({-10.0f, 25.0f}, {200.0f, 25.0f}, cui::Color{}, 2.0f);
    d3.line({100.0f, 100.0f}, {200.0f, 120.0f}, cui::Color{}, 2.0f);   // fully outside
    cardinal::vector<cr::Vertex> v3;
    cr::Renderer::build_geometry(d3, v3);
    CHECK(v3.size() == 6u);
    float lminx = 1.0e9f, lmaxx = -1.0f;
    for (const auto& q : v3) { if (q.x < lminx) lminx = q.x; if (q.x > lmaxx) lmaxx = q.x; }
    CHECK(ap(lminx, 0.0f) && ap(lmaxx, 50.0f));

    // Text: a glyph half inside emits fewer pixel quads than the full 18; a
    // glyph entirely outside emits none.
    cui::DrawList d4;
    d4.push_clip(cui::Rect{ {0.0f, 0.0f}, {8.0f, 8.0f} });
    d4.text({0.0f, 0.0f}, "A", cui::Color{}, 16.0f);
    cardinal::vector<cr::Vertex> v4;
    cr::Renderer::build_geometry(d4, v4);
    CHECK(!v4.empty() && v4.size() < 18u * 6u);
    cui::DrawList d5;
    d5.push_clip(cui::Rect{ {0.0f, 0.0f}, {8.0f, 8.0f} });
    d5.text({100.0f, 0.0f}, "A", cui::Color{}, 16.0f);
    cardinal::vector<cr::Vertex> v5;
    cr::Renderer::build_geometry(d5, v5);
    CHECK(v5.empty());

    // Nested clips intersect; popping restores the outer one.
    cui::DrawList d6;
    d6.push_clip(cui::Rect{ {0.0f, 0.0f}, {50.0f, 50.0f} });
    d6.push_clip(cui::Rect{ {25.0f, 0.0f}, {100.0f, 100.0f} });
    d6.rect_filled(cui::Rect{ {0.0f, 0.0f}, {100.0f, 100.0f} }, cui::Color{});
    d6.pop_clip();
    d6.rect_filled(cui::Rect{ {0.0f, 0.0f}, {100.0f, 100.0f} }, cui::Color{});
    cardinal::vector<cr::Vertex> v6;
    cr::Renderer::build_geometry(d6, v6);
    CHECK(v6.size() == 12u);
    CHECK(ap(v6[0].x, 25.0f) && ap(v6[2].x, 50.0f));   // inner: [25,50]
    CHECK(ap(v6[6].x, 0.0f)  && ap(v6[8].x, 50.0f));   // outer: [0,50]
}

}  // namespace

int main() {
    test_rect();
    test_line_and_text();
    test_pack_color();
    test_clip();
    if (g_fail == 0) {
        cardinal::log::infof("crtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("crtest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
