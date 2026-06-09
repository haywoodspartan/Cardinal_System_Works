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

}  // namespace

int main() {
    test_rect();
    test_line_and_text();
    test_pack_color();
    if (g_fail == 0) {
        cardinal::log::infof("crtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("crtest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
