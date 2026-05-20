// =============================================================================
// Cardinal — deterministic geometry regression suite.
//
// core::geom is the shared CPU spatial toolbox: physics broadphase,
// renderer frustum culling, partition, navmesh, AI cones. A silent
// regression makes visible objects vanish (cull), collisions miss
// (BVH/overlap), or raycasts lie — all invisible until shipped. Pure +
// deterministic, built on the now-locked core math. No <cmath> (one
// local kPi literal; angles at exact values). The BVH/frustum/raycast
// traversals are checked against an independent brute-force oracle so
// the test can't share a bug with the implementation. Exit 0 = pass.
// =============================================================================

#include <cardinal/core/geom.hpp>
#include <cardinal/core/math.hpp>
#include <cardinal/core/log.hpp>

#include <vector>

namespace {

namespace g = cardinal::core::geom;
using cardinal::core::Vec3;
using cardinal::core::Mat4;
using cardinal::u32;

constexpr float kPi = 3.14159265358979323846f;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("geomtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-4f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
bool apv(const Vec3& a, const Vec3& b, float e = 1e-4f) {
    return ap(a.x,b.x,e) && ap(a.y,b.y,e) && ap(a.z,b.z,e);
}

// ---- AABB -----------------------------------------------------------
void test_aabb() {
    const g::AABB e = g::AABB::make_empty();
    CHECK(e.empty());

    g::AABB b = g::AABB::from_center_extent(Vec3{0,0,0}, Vec3{1,2,3});
    CHECK(apv(b.min, Vec3{-1,-2,-3}));
    CHECK(apv(b.max, Vec3{ 1, 2, 3}));
    CHECK(apv(b.center(), Vec3{0,0,0}));
    CHECK(apv(b.extent(), Vec3{1,2,3}));
    CHECK(apv(b.size(),   Vec3{2,4,6}));
    CHECK(ap(b.volume(), 48.0f));                 // 2·4·6
    CHECK(ap(b.surface_area(), 88.0f));           // 2(8+24+12)
    CHECK(!b.empty());

    CHECK(b.contains(Vec3{0,0,0}));
    CHECK(!b.contains(Vec3{0,0,9}));
    CHECK(b.contains(g::AABB::from_center_extent(Vec3{0,0,0}, Vec3{0.5f,0.5f,0.5f})));
    CHECK(!b.contains(g::AABB::from_center_extent(Vec3{0,0,0}, Vec3{10,10,10})));

    g::AABB o1 = g::AABB::from_center_extent(Vec3{0,0,0}, Vec3{1,1,1});
    g::AABB o2 = g::AABB::from_center_extent(Vec3{1.5f,0,0}, Vec3{1,1,1});
    g::AABB o3 = g::AABB::from_center_extent(Vec3{50,0,0}, Vec3{1,1,1});
    CHECK(o1.intersects(o2));                     // overlap on X
    CHECK(!o1.intersects(o3));                    // disjoint

    g::AABB grow = g::AABB::from_center_extent(Vec3{0,0,0}, Vec3{1,1,1});
    grow.expand(Vec3{5,0,0});
    CHECK(grow.contains(Vec3{5,0,0}));
    CHECK(ap(grow.max.x, 5.0f));
    grow.expand(g::AABB::from_center_extent(Vec3{0,0,-9}, Vec3{1,1,1}));
    CHECK(grow.min.z <= -10.0f);
    g::AABB inf = g::AABB::from_center_extent(Vec3{0,0,0}, Vec3{1,1,1});
    inf.inflate(2.0f);
    CHECK(apv(inf.min, Vec3{-3,-3,-3}) && apv(inf.max, Vec3{3,3,3}));
}

// ---- OBB ------------------------------------------------------------
void test_obb() {
    g::OBB ax{};
    ax.center = Vec3{5,0,0};
    ax.half_extents = Vec3{1,2,3};                // default axes ⇒ AABB
    g::AABB wa = ax.world_aabb();
    CHECK(apv(wa.min, Vec3{4,-2,-3}));
    CHECK(apv(wa.max, Vec3{6, 2, 3}));
    CHECK(ax.contains(Vec3{5,0,0}));
    CHECK(!ax.contains(Vec3{100,0,0}));

    // 45° about Y ⇒ the X/Z footprint inflates by √2.
    const float s = 0.70710678f;
    g::OBB rot{};
    rot.center = Vec3{0,0,0};
    rot.half_extents = Vec3{1,1,1};
    rot.axis_x = Vec3{ s, 0.0f, -s};
    rot.axis_y = Vec3{ 0.0f, 1.0f, 0.0f};
    rot.axis_z = Vec3{ s, 0.0f,  s};
    g::AABB rw = rot.world_aabb();
    CHECK(ap(rw.max.x, 1.41421356f, 2e-3f));
    CHECK(ap(rw.max.z, 1.41421356f, 2e-3f));
    CHECK(ap(rw.max.y, 1.0f, 1e-4f));
}

// ---- Sphere / Capsule / Plane / Triangle / Polygon2D ----------------
void test_simple_primitives() {
    g::Sphere sp{ Vec3{0,0,0}, 2.0f };
    CHECK(sp.contains(Vec3{1,0,0}));
    CHECK(!sp.contains(Vec3{3,0,0}));
    CHECK(sp.intersects(g::Sphere{ Vec3{3,0,0}, 1.5f }));   // gap 3 < 3.5
    CHECK(!sp.intersects(g::Sphere{ Vec3{10,0,0}, 1.0f }));
    CHECK(sp.intersects(g::AABB::from_center_extent(Vec3{3,0,0}, Vec3{1,1,1})));
    CHECK(!sp.intersects(g::AABB::from_center_extent(Vec3{50,0,0}, Vec3{1,1,1})));

    g::Capsule cap{ Vec3{0,0,0}, Vec3{0,3,0}, 0.5f };
    g::AABB cb = cap.world_aabb();
    CHECK(apv(cb.min, Vec3{-0.5f,-0.5f,-0.5f}, 1e-4f));
    CHECK(apv(cb.max, Vec3{ 0.5f, 3.5f, 0.5f}, 1e-4f));

    g::Plane pl{ Vec3{0,1,0}, 0.0f };
    CHECK(ap(pl.distance_to(Vec3{0,5,0}),  5.0f));
    CHECK(ap(pl.distance_to(Vec3{0,-2,0}), -2.0f));          // signed

    g::Triangle tri{ Vec3{0,0,0}, Vec3{1,0,0}, Vec3{0,1,0} };
    CHECK(apv(tri.normal(), Vec3{0,0,1}, 1e-4f));
    CHECK(ap(tri.area(), 0.5f));
    CHECK(apv(tri.centroid(), Vec3{1.0f/3.0f, 1.0f/3.0f, 0.0f}, 1e-4f));

    g::Polygon2D sq;
    sq.xs = { 0.0f, 1.0f, 1.0f, 0.0f };
    sq.ys = { 0.0f, 0.0f, 1.0f, 1.0f };
    CHECK(sq.contains(0.5f, 0.5f));
    CHECK(!sq.contains(2.0f, 2.0f));
    CHECK(ap(sq.area(), 1.0f, 1e-4f));
}

// ---- Frustum culling (the renderer keystone) ------------------------
void test_frustum() {
    const Mat4 view = Mat4::look_at(Vec3{0,0,5}, Vec3{0,0,0}, Vec3{0,1,0});
    const Mat4 proj = Mat4::perspective(kPi / 3.0f, 1.0f, 0.1f, 100.0f);
    const g::Frustum fr = g::Frustum::from_view_proj(proj * view);

    // Convention-free behavioural contract: things in front of the
    // camera near the look-at point are kept; things off to the side /
    // behind are culled.
    CHECK(fr.contains(Vec3{0,0,0}));               // look-at point
    CHECK(!fr.contains(Vec3{0,0,50}));             // behind the camera
    CHECK(!fr.contains(Vec3{1000,0,0}));           // far off-axis

    CHECK(fr.intersects(g::Sphere{ Vec3{0,0,0}, 1.0f }));
    CHECK(fr.intersects(g::Sphere{ Vec3{0,0,0}, 1000.0f }));
    CHECK(!fr.intersects(g::Sphere{ Vec3{1000,0,0}, 1.0f }));

    CHECK(fr.intersects(g::AABB::from_center_extent(Vec3{0,0,0}, Vec3{1,1,1})));
    CHECK(!fr.intersects(
        g::AABB::from_center_extent(Vec3{0,0,500}, Vec3{1,1,1})));
}

// ---- raycasts -------------------------------------------------------
void test_raycasts() {
    const g::AABB box = g::AABB::from_center_extent(Vec3{0,0,0}, Vec3{1,1,1});
    g::Hit h = g::raycast_aabb(
        g::Ray{ Vec3{0,0,-5}, Vec3{0,0,1} }, box, 1e6f);
    CHECK(h.hit && static_cast<bool>(h));
    CHECK(ap(h.t, 4.0f, 1e-3f));                   // enters at z = -1
    CHECK(ap(h.point.z, -1.0f, 1e-3f));
    CHECK(!g::raycast_aabb(
        g::Ray{ Vec3{5,5,-5}, Vec3{0,1,0} }, box).hit);
    CHECK(!g::raycast_aabb(
        g::Ray{ Vec3{0,0,-5}, Vec3{0,0,1} }, box, 2.0f).hit);  // max_t

    g::Hit hs = g::raycast_sphere(
        g::Ray{ Vec3{0,0,-5}, Vec3{0,0,1} }, g::Sphere{ Vec3{0,0,0}, 1.0f });
    CHECK(hs.hit && ap(hs.t, 4.0f, 1e-3f));
    CHECK(!g::raycast_sphere(
        g::Ray{ Vec3{0,5,-5}, Vec3{0,0,1} }, g::Sphere{ Vec3{0,0,0}, 1.0f }).hit);
    // Origin INSIDE the sphere must still hit — at the exit point.
    // Regression: the near-root-only test returned a false MISS for any
    // ray cast from within a sphere (AI line-of-sight / projectiles /
    // picking from inside a volume), inconsistent with raycast_aabb.
    g::Hit hin = g::raycast_sphere(
        g::Ray{ Vec3{0,0,0}, Vec3{0,0,1} }, g::Sphere{ Vec3{0,0,0}, 1.0f });
    CHECK(hin.hit && ap(hin.t, 1.0f, 1e-3f));      // exits at z = +1
    CHECK(ap(hin.point.z, 1.0f, 1e-3f));
    // Off-centre interior origin also hits the exit.
    g::Hit hin2 = g::raycast_sphere(
        g::Ray{ Vec3{0.25f,0,0}, Vec3{1,0,0} }, g::Sphere{ Vec3{0,0,0}, 1.0f });
    CHECK(hin2.hit && ap(hin2.t, 0.75f, 1e-3f));   // exits at x = +1

    g::Triangle tri{ Vec3{-1,-1,0}, Vec3{1,-1,0}, Vec3{0,1,0} };
    g::Hit ht = g::raycast_triangle(
        g::Ray{ Vec3{0,0,-5}, Vec3{0,0,1} }, tri);
    CHECK(ht.hit && ap(ht.t, 5.0f, 1e-3f));
    CHECK(apv(ht.point, Vec3{0,0,0}, 1e-3f));
    CHECK(!g::raycast_triangle(
        g::Ray{ Vec3{9,9,-5}, Vec3{0,0,1} }, tri).hit);

    g::Capsule cap{ Vec3{0,0,0}, Vec3{0,3,0}, 0.5f };
    g::Hit hc = g::raycast_capsule(
        g::Ray{ Vec3{-5,1,0}, Vec3{1,0,0} }, cap);
    CHECK(hc.hit && ap(hc.t, 4.5f, 1e-2f));        // cylinder face at x=-0.5
}

// ---- AabbBvh vs brute-force oracle ----------------------------------
void test_bvh() {
    std::vector<g::AABB> boxes;
    std::vector<u32>     ids;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                const Vec3 c{ static_cast<float>(i*4-4),
                              static_cast<float>(j*4-4),
                              static_cast<float>(k*4-4) };
                boxes.push_back(
                    g::AABB::from_center_extent(c, Vec3{1,1,1}));
                ids.push_back(static_cast<u32>(boxes.size() - 1));
            }
    const u32 N = static_cast<u32>(boxes.size());   // 27

    g::AabbBvh bvh;
    bvh.build(boxes, ids);
    CHECK(bvh.primitive_count() == N);
    CHECK(bvh.node_count() >= 1u);
    const g::AABB rt = bvh.root_aabb();
    CHECK(rt.contains(Vec3{-4,-4,-4}) && rt.contains(Vec3{4,4,4}));

    // presence arrays over [0,N) — unsigned char (vector<bool> has no
    // contiguous bool storage). 1 = present.
    // A BVH is a broadphase accelerator: it must enumerate a SUPERSET
    // of the truly-overlapping prims (never miss one — a miss is a
    // dropped collision / popped object), and may conservatively
    // over-report leaf prims for the caller to precisely re-test (same
    // contract as physics broadphase → narrowphase). So the invariant
    // under test is strictly "no false negatives".
    auto run_check = [&](const std::vector<unsigned char>& bvh_set,
                         const std::vector<unsigned char>& brute_set) {
        bool miss = false;
        for (u32 i = 0; i < N; ++i)
            if (brute_set[i] && !bvh_set[i]) miss = true;      // CRITICAL
        CHECK(!miss);
    };

    // AABB query overlapping exactly the (i=0,j=0,k=0) corner box.
    {
        const g::AABB q =
            g::AABB::from_center_extent(Vec3{-4,-4,-4}, Vec3{1.5f,1.5f,1.5f});
        std::vector<unsigned char> bv(N, 0), br(N, 0);
        bvh.traverse_aabb(q, [&](u32 id, const g::AABB&) {
            bv[id] = 1; return true; });
        for (u32 i = 0; i < N; ++i)
            if (q.intersects(boxes[i])) br[i] = 1;
        run_check(bv, br);
        int cnt = 0; for (u32 i=0;i<N;++i) if (br[i]) ++cnt;
        CHECK(cnt == 1);                              // just the corner box
    }
    // Sphere query at the origin, radius 2 (reaches only the centre box).
    {
        const g::Sphere q{ Vec3{0,0,0}, 2.0f };
        std::vector<unsigned char> bv(N, 0), br(N, 0);
        bvh.traverse_sphere(q, [&](u32 id, const g::AABB&) {
            bv[id] = 1; return true; });
        for (u32 i = 0; i < N; ++i)
            if (q.intersects(boxes[i])) br[i] = 1;
        run_check(bv, br);
    }
    // Ray straight down the X axis through the y=z=0 row.
    {
        const g::Ray q{ Vec3{-100,0,0}, Vec3{1,0,0} };
        std::vector<unsigned char> bv(N, 0), br(N, 0);
        bvh.traverse_ray(q, 1e6f, [&](u32 id, const g::AABB&, float) {
            bv[id] = 1; return true; });
        for (u32 i = 0; i < N; ++i)
            if (g::raycast_aabb(q, boxes[i], 1e6f).hit) br[i] = 1;
        run_check(bv, br);
        int cnt = 0; for (u32 i=0;i<N;++i) if (br[i]) ++cnt;
        CHECK(cnt == 3);                              // x = -4, 0, 4
    }
    // Frustum traversal vs Frustum::intersects oracle.
    {
        const Mat4 view = Mat4::look_at(Vec3{0,0,40}, Vec3{0,0,0},
                                        Vec3{0,1,0});
        const Mat4 proj = Mat4::perspective(kPi/3.0f, 1.0f, 0.1f, 200.0f);
        const g::Frustum fr = g::Frustum::from_view_proj(proj * view);
        std::vector<unsigned char> bv(N, 0), br(N, 0);
        bvh.traverse_frustum(fr, [&](u32 id, const g::AABB&) {
            bv[id] = 1; return true; });
        for (u32 i = 0; i < N; ++i)
            if (fr.intersects(boxes[i])) br[i] = 1;
        // BVH must never MISS a visible box (false-cull = popping).
        for (u32 i = 0; i < N; ++i) CHECK(!(br[i] && !bv[i]));
    }

    bvh.clear();
    CHECK(bvh.primitive_count() == 0u);
    CHECK(bvh.node_count() == 0u);
}

// ---- AabbBvh::build must NOT UB on AABBs with NaN components -------
// build_recursive_'s std::nth_element compared centroids via the
// NaN-blind `va < vb`: NaN unordered to everything → (NaN, x) is
// "equivalent" under the predicate while (x, y) with x<y orders
// strictly → transitivity-of-equivalence broken. std::nth_element
// with a SWO-violating comparator is UB — same shape as the
// std::sort family (sky 4ff85a8 / level 4b08e0c / anim+ui
// 23937c5). Realistic ingress: AabbBvh is publicly exposed in
// cardinal::core::geom; a caller cooking a BVH over imported
// geometry whose source file's bit pattern happens to be a NaN
// float reaches this partition with NaN centers. Without the fix
// build can hang or scribble OOB; with the NaN-safe SWO it
// terminates with NaN-center boxes clustered at the tail.
void test_bvh_nan_boxes() {
    std::vector<g::AABB> boxes;
    // 16 finite boxes spread along the X axis. Pass empty `ids` to
    // AabbBvh::build so prim_ids_[i] = i (default identity mapping) —
    // keeps the brute-force-vs-BVH oracle's index space identical to
    // the BVH's reported leaf ids.
    for (int i = 0; i < 16; ++i) {
        const Vec3 c{ static_cast<float>(i),
                      static_cast<float>(i & 1),
                      0.0f };
        boxes.push_back(g::AABB::from_center_extent(c, Vec3{0.5f, 0.5f, 0.5f}));
    }
    // Sprinkle a NaN box in the middle of the vector — worst position
    // for partition pathology under a SWO-violating comparator.
    volatile float z = 0.0f;
    const float qnan = z / z;
    g::AABB nan_box;
    nan_box.min = Vec3{ qnan, 0.0f, 0.0f };
    nan_box.max = Vec3{ qnan + 1.0f, 1.0f, 1.0f };
    boxes.insert(boxes.begin() + 8, nan_box);
    const u32 N = static_cast<u32>(boxes.size());    // 17

    // The UB call. Without the fix, std::nth_element either hangs
    // (introsort partition fails to make progress) or scribbles OOB.
    g::AabbBvh bvh;
    bvh.build(boxes, {});                            // identity ids
    CHECK(bvh.primitive_count() == N);
    CHECK(bvh.node_count() >= 1u);

    // A finite query box overlapping the first few finite boxes must
    // still be reachable from a coherent tree — verify the BVH does
    // not falsely cull legitimate FINITE hits. The NaN box's
    // intersects() check is a spurious false-positive (AABB::
    // intersects uses `min.x > o.max.x` etc. which are NaN-blind so
    // every comparison is FALSE → the whole !(...) chain returns
    // TRUE) — that's an oracle artifact, not a real BVH miss, so
    // exclude the NaN slot from the no-false-negative invariant.
    const g::AABB q =
        g::AABB::from_center_extent(Vec3{1.0f, 0.5f, 0.0f}, Vec3{0.6f, 0.6f, 0.6f});
    std::vector<unsigned char> bv(N, 0), br(N, 0);
    bvh.traverse_aabb(q, [&](u32 id, const g::AABB&) {
        bv[id] = 1; return true; });
    // `x == x` is true for finite and ±Inf, false only for NaN — the
    // canonical NaN check without dragging <cmath> into this test
    // (per top-of-file: <cmath> deliberately avoided). We only inject
    // NaN (not Inf), so this is sufficient.
    auto box_is_finite = [](const g::AABB& b) {
        return b.min.x == b.min.x && b.min.y == b.min.y && b.min.z == b.min.z &&
               b.max.x == b.max.x && b.max.y == b.max.y && b.max.z == b.max.z;
    };
    for (u32 i = 0; i < N; ++i)
        if (box_is_finite(boxes[i]) && q.intersects(boxes[i])) br[i] = 1;
    // BVH must enumerate every brute-force FINITE hit (no false
    // negative on the boxes that genuinely intersect q).
    for (u32 i = 0; i < N; ++i)
        if (br[i]) CHECK(bv[i]);
    // Sanity: brute force found at least one finite hit (boxes 0, 1, 2
    // straddle q's X=[0.4, 1.6] range).
    u32 finite_hits = 0;
    for (u32 i = 0; i < N; ++i) finite_hits += br[i];
    CHECK(finite_hits >= 1u);
}

// ---- DynamicMesh ----------------------------------------------------
void test_dynamic_mesh() {
    g::DynamicMesh m;
    const u32 v0 = m.add_vertex(Vec3{0,0,0});
    const u32 v1 = m.add_vertex(Vec3{2,0,0});
    const u32 v2 = m.add_vertex(Vec3{0,2,0});
    m.add_triangle(v0, v1, v2);
    CHECK(m.vertex_count() == 3u);
    CHECK(m.triangle_count() == 1u);
    const g::AABB bnd = m.bounds();
    CHECK(apv(bnd.min, Vec3{0,0,0}, 1e-4f));
    CHECK(apv(bnd.max, Vec3{2,2,0}, 1e-4f));

    m.subdivide_once();                              // 1 tri → 4
    CHECK(m.triangle_count() == 4u);

    g::DynamicMesh w;
    const u32 a = w.add_vertex(Vec3{0,0,0});
    const u32 b = w.add_vertex(Vec3{0.0001f,0,0});   // ~coincident
    const u32 c = w.add_vertex(Vec3{5,0,0});
    w.add_triangle(a, b, c);
    w.weld_vertices(0.01f);
    CHECK(w.vertex_count() < 3u);                    // a,b collapsed
}

}  // namespace

int main() {
    test_aabb();
    test_obb();
    test_simple_primitives();
    test_frustum();
    test_raycasts();
    test_bvh();
    test_bvh_nan_boxes();
    test_dynamic_mesh();

    if (g_fail == 0) {
        cardinal::log::infof("geomtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("geomtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
