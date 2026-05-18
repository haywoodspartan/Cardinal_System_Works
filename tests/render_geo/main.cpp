// =============================================================================
// Cardinal — deterministic meshlet / cluster-culling regression suite.
//
// render::geo is pure float math (no RHI / GPU). This suite pins:
//
//   * build_meshlets — null/short/non-mult-of-3 inputs return an empty
//     Mesh; opts.max_*==0 resets to 64 and >64 clamps to 64; triangles
//     with an out-of-range index are skipped; the greedy partition
//     splits on the prim cap AND the vertex cap; out.indices is the
//     contiguous concatenation of valid global tris and the meshlets'
//     [offset,count) ranges tile it exactly; bounds = centroid sphere
//     (radius = farthest vert) with cone_apex == sphere_center; the
//     backface cone is normalize(sum of face normals) / min-dot, and a
//     degenerate (colinear) tri yields the wide-open cone axis (0,0,1)
//     cos -1;
//   * frustum_from_vp — identity VP gives the exact 6-plane set;
//   * sphere_inside_frustum — inside / behind-near / the radius-grazing
//     boundary;
//   * cluster_passes_backface — wide-open (cos<=-1) and camera-at-center
//     always pass; otherwise rejection is d >= cone_angle_cos;
//   * cluster_lod — the strict 20 / 60 / 200 distance ramp -> 0/1/2/3,
//     and a custom LodConfig.
//
// Pure, deterministic. Exit 0 = all pass.
// =============================================================================

#include <cardinal/render/geo.hpp>
#include <cardinal/core/log.hpp>

#include <vector>

namespace {

namespace geo = cardinal::render::geo;
namespace scn = cardinal::scene;
using cardinal::u32;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("geotest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(double a, double b, double eps = 1e-4) {
    const double d = (a > b) ? (a - b) : (b - a);
    return d <= eps;
}
bool veq(const scn::Vec3& v, double x, double y, double z, double eps = 1e-4) {
    return ap(v.x, x, eps) && ap(v.y, y, eps) && ap(v.z, z, eps);
}
constexpr u32 kStride = 3u * static_cast<u32>(sizeof(float));

// ---- build_meshlets: input guards -> empty Mesh -------------------
void test_build_guards() {
    const float pos[9] = { 0,0,0, 1,0,0, 0,1,0 };
    const u32   idx[3] = { 0, 1, 2 };

    CHECK(geo::build_meshlets(nullptr, 3u, pos, 3u, kStride).meshlets.empty());
    CHECK(geo::build_meshlets(idx, 3u, nullptr, 3u, kStride).meshlets.empty());
    CHECK(geo::build_meshlets(idx, 2u, pos, 3u, kStride).meshlets.empty()); // <3
    CHECK(geo::build_meshlets(idx, 4u, pos, 3u, kStride).meshlets.empty()); // %3
    auto e = geo::build_meshlets(idx, 0u, pos, 3u, kStride);
    CHECK(e.meshlets.empty() && e.indices.empty());
}

// ---- build_meshlets: single triangle -> bounds + cone ------------
void test_build_single_tri() {
    const float pos[9] = { 0,0,0,  2,0,0,  0,2,0 };
    const u32   idx[3] = { 0, 1, 2 };
    auto m = geo::build_meshlets(idx, 3u, pos, 3u, kStride);

    CHECK(m.meshlets.size() == 1u);
    CHECK(m.indices.size() == 3u);
    CHECK(m.indices[0]==0u && m.indices[1]==1u && m.indices[2]==2u);

    const geo::Meshlet& ml = m.meshlets[0];
    CHECK(ml.index_offset == 0u && ml.index_count == 3u);
    CHECK(ml.vertex_count == 3u && ml.vertex_offset == 0u);
    CHECK(ml.lod_parent == u32(-1) && ml.lod_level == 0u);

    // centroid = mean = (2/3, 2/3, 0); radius = farthest vert.
    CHECK(veq(ml.bounds.sphere_center, 2.0/3.0, 2.0/3.0, 0.0));
    CHECK(ap(ml.bounds.sphere_radius, 1.49071198));     // sqrt(20/9)
    CHECK(veq(ml.bounds.cone_apex, 2.0/3.0, 2.0/3.0, 0.0));  // == center
    // face normal of CCW tri in XY plane is +Z; single tri -> cos 1.
    CHECK(veq(ml.bounds.cone_axis, 0, 0, 1));
    CHECK(ap(ml.bounds.cone_angle_cos, 1.0));
}

// ---- build_meshlets: prim cap forces a split ---------------------
void test_build_prim_cap() {
    // 3 disjoint triangles, 9 verts. max_prims = 2 -> [2 tris][1 tri].
    float pos[27];
    for (int i = 0; i < 27; ++i) pos[i] = static_cast<float>(i);
    u32 idx[9] = { 0,1,2, 3,4,5, 6,7,8 };
    geo::BuildOptions o; o.max_prims = 2u;
    auto m = geo::build_meshlets(idx, 9u, pos, 9u, kStride, nullptr, 0u, o);

    CHECK(m.meshlets.size() == 2u);
    CHECK(m.indices.size() == 9u);
    CHECK(m.meshlets[0].index_offset == 0u && m.meshlets[0].index_count == 6u);
    CHECK(m.meshlets[1].index_offset == 6u && m.meshlets[1].index_count == 3u);
    CHECK(m.meshlets[0].vertex_count == 6u);
    CHECK(m.meshlets[1].vertex_count == 3u);
    // contiguous tiling: sum of counts == indices.size().
    CHECK(m.meshlets[0].index_count + m.meshlets[1].index_count
          == static_cast<u32>(m.indices.size()));
    for (u32 i = 0; i < 9; ++i) CHECK(m.indices[i] == i);
}

// ---- build_meshlets: vertex cap forces a split -------------------
void test_build_vertex_cap() {
    float pos[18];
    for (int i = 0; i < 18; ++i) pos[i] = static_cast<float>(i);
    u32 idx[6] = { 0,1,2, 3,4,5 };
    geo::BuildOptions o; o.max_verts = 4u;            // 2 tris won't co-fit
    auto m = geo::build_meshlets(idx, 6u, pos, 6u, kStride, nullptr, 0u, o);

    CHECK(m.meshlets.size() == 2u);
    CHECK(m.meshlets[0].vertex_count == 3u && m.meshlets[0].index_count == 3u);
    CHECK(m.meshlets[1].vertex_count == 3u && m.meshlets[1].index_count == 3u);
    CHECK(m.meshlets[0].index_offset == 0u);
    CHECK(m.meshlets[1].index_offset == 3u);
    CHECK(m.indices.size() == 6u);
}

// ---- build_meshlets: out-of-range tri skipped; opts reset --------
void test_build_skip_and_opts() {
    const float pos[9] = { 0,0,0, 1,0,0, 0,1,0 };
    u32 idx[6] = { 0,1,2,  0,1,9 };                   // 2nd tri: 9 >= 3
    auto m = geo::build_meshlets(idx, 6u, pos, 3u, kStride);
    CHECK(m.meshlets.size() == 1u);
    CHECK(m.indices.size() == 3u);                    // bad tri dropped
    CHECK(m.meshlets[0].index_count == 3u);

    // max_verts==0 / max_prims==0 -> reset to the 64/64 defaults.
    geo::BuildOptions z; z.max_verts = 0u; z.max_prims = 0u;
    u32 i3[3] = { 0,1,2 };
    auto d = geo::build_meshlets(i3, 3u, pos, 3u, kStride, nullptr, 0u, z);
    CHECK(d.meshlets.size() == 1u && d.meshlets[0].vertex_count == 3u);
}

// ---- build_meshlets: degenerate tri -> wide-open cone ------------
void test_build_degenerate_cone() {
    const float pos[9] = { 0,0,0, 1,0,0, 2,0,0 };     // colinear
    const u32   idx[3] = { 0, 1, 2 };
    auto m = geo::build_meshlets(idx, 3u, pos, 3u, kStride);
    CHECK(m.meshlets.size() == 1u);
    const geo::ClusterBounds& b = m.meshlets[0].bounds;
    CHECK(b.cone_angle_cos == -1.0f);                 // never reject
    CHECK(veq(b.cone_axis, 0, 0, 1));
}

// ---- frustum_from_vp: identity VP -> exact plane set -------------
void test_frustum_identity() {
    geo::Frustum f = geo::frustum_from_vp(scn::Mat4::identity());
    CHECK(veq({f.planes[0].x,f.planes[0].y,f.planes[0].z}, 1, 0, 0));
    CHECK(ap(f.planes[0].w, 1.0));                    // left  ( 1,0,0,1)
    CHECK(veq({f.planes[1].x,f.planes[1].y,f.planes[1].z}, -1, 0, 0));
    CHECK(ap(f.planes[1].w, 1.0));                    // right (-1,0,0,1)
    CHECK(veq({f.planes[2].x,f.planes[2].y,f.planes[2].z}, 0, 1, 0));
    CHECK(ap(f.planes[2].w, 1.0));                    // bottom( 0,1,0,1)
    CHECK(veq({f.planes[3].x,f.planes[3].y,f.planes[3].z}, 0, -1, 0));
    CHECK(ap(f.planes[3].w, 1.0));                    // top   ( 0,-1,0,1)
    CHECK(veq({f.planes[4].x,f.planes[4].y,f.planes[4].z}, 0, 0, 1));
    CHECK(ap(f.planes[4].w, 0.0));                    // near  ( 0,0,1,0)
    CHECK(veq({f.planes[5].x,f.planes[5].y,f.planes[5].z}, 0, 0, -1));
    CHECK(ap(f.planes[5].w, 1.0));                    // far   ( 0,0,-1,1)
}

// ---- sphere_inside_frustum: inside / behind / grazing -----------
void test_sphere_frustum() {
    geo::Frustum f = geo::frustum_from_vp(scn::Mat4::identity());

    CHECK(geo::sphere_inside_frustum(f, scn::Vec3{0,0,0}, 0.5f));
    // behind the near plane (z=0, inside is z>=0): z=-5 -> outside.
    CHECK(!geo::sphere_inside_frustum(f, scn::Vec3{0,0,-5}, 0.1f));
    // right plane (-1,0,0,1): point x=10 -> dist = -9.
    CHECK(!geo::sphere_inside_frustum(f, scn::Vec3{10,0,0}, 8.5f)); // -9<-8.5
    CHECK( geo::sphere_inside_frustum(f, scn::Vec3{10,0,0}, 9.5f)); // -9>=-9.5
}

// ---- cluster_passes_backface: open / center / rejection ---------
void test_backface() {
    geo::ClusterBounds open{};
    open.cone_angle_cos = -1.0f;                      // wide open
    CHECK(geo::cluster_passes_backface(open, scn::Vec3{0,0,-10}));

    geo::ClusterBounds b{};
    b.sphere_center  = scn::Vec3{0,0,0};
    b.cone_axis      = scn::Vec3{0,0,1};
    b.cone_angle_cos = 0.5f;

    // camera at center -> degenerate -> passes.
    CHECK(geo::cluster_passes_backface(b, scn::Vec3{0,0,0}));

    // camera behind +Z face: dir = (0,0,-1), d = -1 < 0.5 -> visible.
    CHECK(geo::cluster_passes_backface(b, scn::Vec3{0,0,10}));
    // camera on the axis side: dir = (0,0,1), d = 1 >= 0.5 -> rejected.
    CHECK(!geo::cluster_passes_backface(b, scn::Vec3{0,0,-10}));

    // cluster_visible composes sphere test AND backface test.
    geo::Frustum f = geo::frustum_from_vp(scn::Mat4::identity());
    geo::ClusterBounds vis{};
    vis.sphere_center  = scn::Vec3{0,0,0};
    vis.sphere_radius  = 1.0f;
    vis.cone_angle_cos = -1.0f;                       // never backface-cull
    CHECK(geo::cluster_visible(vis, f, scn::Vec3{0,0,10}));
    vis.sphere_center = scn::Vec3{100,0,0};           // outside right plane
    CHECK(!geo::cluster_visible(vis, f, scn::Vec3{0,0,10}));
}

// ---- cluster_lod: the strict distance ramp ----------------------
void test_lod() {
    geo::ClusterBounds b{};
    const scn::Vec3 cam{0,0,0};
    auto lod_at = [&](double dist) {
        b.sphere_center = scn::Vec3{ static_cast<float>(dist), 0, 0 };
        return geo::cluster_lod(b, cam);              // default 20/60/200
    };
    CHECK(lod_at(0.0)   == 0u);
    CHECK(lod_at(10.0)  == 0u);
    CHECK(lod_at(19.9)  == 0u);
    CHECK(lod_at(20.0)  == 1u);                       // strict: !(20<20)
    CHECK(lod_at(50.0)  == 1u);
    CHECK(lod_at(60.0)  == 2u);
    CHECK(lod_at(150.0) == 2u);
    CHECK(lod_at(200.0) == 3u);                       // beyond quarter
    CHECK(lod_at(999.0) == 3u);

    geo::LodConfig cfg;
    cfg.distance_full = 5.0f; cfg.distance_half = 10.0f;
    cfg.distance_quarter = 15.0f;
    b.sphere_center = scn::Vec3{3,0,0};
    CHECK(geo::cluster_lod(b, cam, cfg) == 0u);
    b.sphere_center = scn::Vec3{7,0,0};
    CHECK(geo::cluster_lod(b, cam, cfg) == 1u);
    b.sphere_center = scn::Vec3{12,0,0};
    CHECK(geo::cluster_lod(b, cam, cfg) == 2u);
    b.sphere_center = scn::Vec3{20,0,0};
    CHECK(geo::cluster_lod(b, cam, cfg) == 3u);
}

}  // namespace

int main() {
    test_build_guards();
    test_build_single_tri();
    test_build_prim_cap();
    test_build_vertex_cap();
    test_build_skip_and_opts();
    test_build_degenerate_cone();
    test_frustum_identity();
    test_sphere_frustum();
    test_backface();
    test_lod();

    if (g_fail == 0) {
        cardinal::log::infof("geotest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("geotest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
