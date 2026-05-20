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
using cardinal::usize;

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

// ---- subdivide: counts, watertightness, PN placement --------------
void test_subdivide() {
    geo::TriMesh in;
    in.positions = { {0,0,0}, {1,0,0}, {0,1,0} };
    in.indices   = { 0u,1u,2u };

    auto l0 = geo::subdivide(in, geo::SubdivOptions{0u, 0.0f});
    CHECK(l0.positions.size() == 3u && l0.indices.size() == 3u);   // passthrough

    auto l1 = geo::subdivide(in, geo::SubdivOptions{1u, 0.0f});
    CHECK(l1.positions.size() == 6u);     // 3 corners + 3 edge mids
    CHECK(l1.indices.size()   == 12u);    // 4 sub-tris
    CHECK(veq(l1.positions[3], 0.5, 0.0, 0.0));   // mid(0,1)
    CHECK(veq(l1.positions[4], 0.5, 0.5, 0.0));   // mid(1,2)
    CHECK(veq(l1.positions[5], 0.0, 0.5, 0.0));   // mid(2,0)

    auto l2 = geo::subdivide(in, geo::SubdivOptions{2u, 0.0f});
    CHECK(l2.indices.size() == 48u);      // 4^2 tris * 3

    // Watertight: a 2-tri quad shares the diagonal midpoint exactly once.
    geo::TriMesh q;
    q.positions = { {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0} };
    q.indices   = { 0u,1u,2u,  0u,2u,3u };
    auto qs = geo::subdivide(q, geo::SubdivOptions{1u, 0.0f});
    CHECK(qs.positions.size() == 9u);     // 4 corners + 5 unique edges
    CHECK(qs.indices.size()   == 24u);

    // PN-triangle: a sphere-like edge (normals fanned along the edge)
    // bows midpoint(0,1) to z = -0.12 at full blend. blend 0 stays at the
    // linear midpoint; blend 0.5 lands exactly halfway (lerp wiring).
    geo::TriMesh c;
    c.positions = { {0,0,0}, {1,0,0}, {0,1,0} };
    c.normals   = { {0.6f,0,0.8f}, {-0.6f,0,0.8f}, {0,0,1} };
    c.indices   = { 0u,1u,2u };
    auto flat = geo::subdivide(c, geo::SubdivOptions{1u, 0.0f});
    CHECK(veq(flat.positions[3], 0.5, 0.0, 0.0));            // == linear
    auto bent = geo::subdivide(c, geo::SubdivOptions{1u, 1.0f});
    CHECK(ap(bent.positions[3].x, 0.5) && ap(bent.positions[3].z, -0.12));
    auto half = geo::subdivide(c, geo::SubdivOptions{1u, 0.5f});
    CHECK(ap(half.positions[3].z, -0.06));                   // lerp(lin,pn,.5)
    CHECK(bent.normals.size() == bent.positions.size());

    // Odd index count → rejected (defensive).
    geo::TriMesh bad; bad.positions = { {0,0,0} }; bad.indices = { 0u,0u };
    CHECK(geo::subdivide(bad, geo::SubdivOptions{1u,0.0f}).indices.empty());
}

// ---- subdiv_level_for_factor: ceil(log2) ramp, clamped ------------
void test_subdiv_level() {
    CHECK(geo::subdiv_level_for_factor(0.5f)        == 0u);
    CHECK(geo::subdiv_level_for_factor(1.0f)        == 0u);
    CHECK(geo::subdiv_level_for_factor(2.0f)        == 1u);
    CHECK(geo::subdiv_level_for_factor(3.0f)        == 2u);
    CHECK(geo::subdiv_level_for_factor(4.0f)        == 2u);
    CHECK(geo::subdiv_level_for_factor(8.0f)        == 3u);
    CHECK(geo::subdiv_level_for_factor(64.0f, 4u)   == 4u);   // clamp
    CHECK(geo::subdiv_level_for_factor(1000.0f, 6u) == 6u);   // clamp

    // Non-finite tess_factor must NOT invoke UB on the float→u32 cast
    // (log2(NaN) = NaN, ceil(NaN) = NaN, static_cast<u32>(NaN) is UB
    // per [conv.fpint]p1). Same UB-cast class fixed across world,
    // mesh_ops, tex_ops, scene, physics, brush, vgeom. Treat NaN/±Inf
    // as "no subdivision needed" — the factor==1.0 contract.
    volatile float z = 0.0f;
    const float qnan = z / z;
    const float inf  = 1.0f / z;
    CHECK(geo::subdiv_level_for_factor(qnan)         == 0u);  // no-UB, no-subdiv
    CHECK(geo::subdiv_level_for_factor( inf)         == 0u);
    CHECK(geo::subdiv_level_for_factor(-inf)         == 0u);
}

// ---- cluster DAG + perfect LOD cut --------------------------------
void test_cluster_dag() {
    const int N = 8;                                  // 81 verts, 128 tris
    std::vector<float> pos;
    for (int y = 0; y <= N; ++y)
        for (int x = 0; x <= N; ++x) {
            pos.push_back(static_cast<float>(x));
            pos.push_back(static_cast<float>(y));
            pos.push_back(0.0f);
        }
    auto vid = [&](int x, int y){ return static_cast<u32>(y*(N+1)+x); };
    std::vector<u32> idx;
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            idx.push_back(vid(x,  y  )); idx.push_back(vid(x+1,y  )); idx.push_back(vid(x+1,y+1));
            idx.push_back(vid(x,  y  )); idx.push_back(vid(x+1,y+1)); idx.push_back(vid(x,  y+1));
        }
    const u32 vcount = static_cast<u32>(pos.size() / 3);
    auto base = geo::build_meshlets(idx.data(), static_cast<u32>(idx.size()),
                                    pos.data(), vcount, kStride);
    CHECK(base.meshlets.size() >= 2u);

    auto dag = geo::build_cluster_dag(base, pos.data(), vcount, kStride);
    CHECK(dag.root != u32(-1));
    CHECK(dag.level_count >= 2u);
    CHECK(dag.nodes[dag.root].child_count > 0u);

    u32 l0 = 0;
    for (const auto& n : dag.nodes) if (n.lod_level == 0u) ++l0;
    CHECK(l0 == static_cast<u32>(base.meshlets.size()));

    bool mono = true, ranges_ok = true, idx_ok = true;
    for (const auto& n : dag.nodes) {
        if ((n.index_count % 3u) != 0u) ranges_ok = false;
        if (static_cast<usize>(n.index_offset) + n.index_count
              > dag.indices.size()) ranges_ok = false;
        else for (u32 i = 0; i < n.index_count; ++i)
            if (dag.indices[n.index_offset + i] >= dag.positions.size())
                idx_ok = false;
        for (u32 k = 0; k < n.child_count; ++k) {
            const u32 cid = dag.child_ids[n.first_child + k];
            if (n.error + 1e-3f < dag.nodes[cid].error) mono = false;
        }
    }
    CHECK(mono); CHECK(ranges_ok); CHECK(idx_ok);

    auto dag2 = geo::build_cluster_dag(base, pos.data(), vcount, kStride);
    CHECK(dag2.nodes.size() == dag.nodes.size());
    CHECK(dag2.root == dag.root && dag2.level_count == dag.level_count);

    // Perfect LOD cut — far + huge budget → just the root.
    geo::LodCutParams cam{};
    cam.camera_pos         = scn::Vec3{4.0f, 4.0f, 1000.0f};
    cam.proj_scale         = 1.0f;
    cam.error_threshold_px = 1e9f;
    auto coarse = geo::select_lod_cut(dag, cam);
    CHECK(coarse.size() == 1u && coarse[0] == dag.root);

    // Tight budget + huge projection → refine down to leaves.
    cam.proj_scale         = 1e6f;
    cam.error_threshold_px = 1e-9f;
    auto fine = geo::select_lod_cut(dag, cam);
    CHECK(fine.size() >= coarse.size());
    bool ids_ok = true, reached_leaf = false;
    for (u32 id : fine) {
        if (id >= dag.nodes.size()) ids_ok = false;
        else if (dag.nodes[id].child_count == 0u) reached_leaf = true;
    }
    CHECK(ids_ok); CHECK(reached_leaf);
    auto fine2 = geo::select_lod_cut(dag, cam);
    CHECK(fine2.size() == fine.size());
}

// ---- build_cluster_dag must NOT UB on NaN vertex positions ----------
// build_cluster_dag's internals reach two float→int casts that the
// subdiv_level_for_factor fix (c420695) missed:
//
//   * simplify_cluster's axis_cell lambda: `int c = static_cast<int>
//     ((v - lo) / e * gridN)` where (v, lo, e) come from gpos / its
//     min-max range. A NaN gpos[i].x propagates into mn/mx via
//     cardinal::min/max (NaN-blind → returns the NaN operand), then
//     into ext, then into the cast — `static_cast<int>(NaN)` is UB
//     per [conv.fpint]p1.
//   * morton3's q lambda: same shape, but reached via
//     build_cluster_dag's per-level sort (line 597) which sorts by
//     morton3(node.bounds.sphere_center, ...). A NaN sphere_center
//     hits the same UB cast.
//
// Realistic ingress: a Mesh asset whose vertex stream has any NaN
// component (decode_mesh accepts what asset.cpp:113 reads via
// rd_f → bit_cast<float> — a corrupt blob with the bit pattern of
// NaN passes through verbatim). Without the fix this test would
// UB on the cast; with the fix the build runs and produces a
// non-empty DAG (the NaN vertex sinks into "cell 0" and "bucket 0"
// like any other defined collapse in this UB-cast family).
void test_cluster_dag_nan_positions() {
    // Reuse the test_cluster_dag grid layout, but poison one vertex's
    // X component with qNaN.
    const int N = 8;
    std::vector<float> pos;
    pos.reserve(static_cast<usize>((N+1)*(N+1)*3));
    volatile float z = 0.0f;
    const float qnan = z / z;
    for (int y = 0; y <= N; ++y)
        for (int x = 0; x <= N; ++x) {
            pos.push_back(static_cast<float>(x));
            pos.push_back(static_cast<float>(y));
            pos.push_back(0.0f);
        }
    // Poison the middle vertex. Putting NaN in the interior of the
    // grid maximises the chance that axis_cell / morton3 see the
    // NaN during the BVH-like partition steps.
    const usize mid = static_cast<usize>(((N/2) * (N+1) + (N/2)) * 3);
    pos[mid] = qnan;

    auto vid = [&](int x, int y){ return static_cast<u32>(y*(N+1)+x); };
    std::vector<u32> idx;
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            idx.push_back(vid(x,  y  )); idx.push_back(vid(x+1,y  )); idx.push_back(vid(x+1,y+1));
            idx.push_back(vid(x,  y  )); idx.push_back(vid(x+1,y+1)); idx.push_back(vid(x,  y+1));
        }
    const u32 vcount = static_cast<u32>(pos.size() / 3);
    auto base = geo::build_meshlets(idx.data(), static_cast<u32>(idx.size()),
                                    pos.data(), vcount, kStride);
    // build_meshlets itself reaches axis_cell via simplify_cluster
    // only inside build_cluster_dag's lower levels — the leaf
    // meshlet build doesn't simplify. So the meshlet build should
    // still succeed even with a NaN vertex (the bounds sphere may
    // have NaN center/radius, but no UB cast on that path).
    CHECK(!base.meshlets.empty());

    // The actual UB sites — axis_cell + morton3 — are reached here.
    // Without the fix this is UB (cast of NaN to int). With the fix
    // it returns a coherent DAG. Don't assert on the exact node count
    // (NaN data legitimately steers clustering); just that the build
    // terminates and produces a valid root.
    auto dag = geo::build_cluster_dag(base, pos.data(), vcount, kStride);
    CHECK(dag.root != u32(-1));
    CHECK(!dag.nodes.empty());
    // Index ranges must remain in-bounds.
    bool ranges_ok = true;
    for (const auto& n : dag.nodes) {
        if (static_cast<usize>(n.index_offset) + n.index_count
              > dag.indices.size()) ranges_ok = false;
    }
    CHECK(ranges_ok);
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
    test_subdivide();
    test_subdiv_level();
    test_cluster_dag();
    test_cluster_dag_nan_positions();

    if (g_fail == 0) {
        cardinal::log::infof("geotest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("geotest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
