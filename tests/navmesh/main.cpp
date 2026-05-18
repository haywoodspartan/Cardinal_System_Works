// =============================================================================
// Cardinal — deterministic polygon-navmesh regression suite.
//
// Pipeline: triangle soup → build_from_triangles (auto edge-shared
// adjacency) → recompute_centroids → nearest_poly (centroid-nearest) →
// PathQuery (polygon A* over centroids) → funnel string-pulling to
// straight waypoints. A regression silently breaks agent navigation.
// Pure CPU, fully deterministic. This suite pins construction-GUARANTEED
// invariants — exact shared-edge neighbour indices, exact centroids,
// centroid-nearest queries, A* found/cost^2/poly_visited on tiny hand-
// traceable meshes, and the funnel's UNIVERSAL contract (out.front()==
// start, out.back()==goal, finite, count>=2) — rather than fragile
// hand-predicted intermediate waypoint coordinates. Exit 0 = all pass.
// =============================================================================

#include <cardinal/navmesh/navmesh.hpp>
#include <cardinal/core/log.hpp>

#include <vector>

namespace {

namespace nm = cardinal::navmesh;
using Vec3   = cardinal::scene::Vec3;
using cardinal::u32;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("nmtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-4f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
bool apv(const Vec3& v, float x, float y, float z, float e = 1e-4f) {
    return ap(v.x, x, e) && ap(v.y, y, e) && ap(v.z, z, e);
}
bool fin(float v) { return (v == v) && (v < 3.0e38f) && (v > -3.0e38f); }
bool finv(const Vec3& v) { return fin(v.x) && fin(v.y) && fin(v.z); }
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

// double params so plain integer literals widen without C4244.
Vec3 v3(double x, double y, double z) {
    return Vec3{ static_cast<float>(x),
                 static_cast<float>(y),
                 static_cast<float>(z) };
}

// The canonical 3x3 quad split into 2 triangles in the XZ plane.
//   V0(0,0,0) V1(3,0,0) V2(3,0,3) V3(0,0,3)
//   tri0 = (0,1,2)  centroid (2,0,1)
//   tri1 = (0,2,3)  centroid (1,0,2)  — shares edge (0,2) with tri0
nm::Mesh quad_mesh() {
    nm::Mesh m;
    std::vector<Vec3> verts = {
        v3(0,0,0), v3(3,0,0), v3(3,0,3), v3(0,0,3) };
    std::vector<u32> idx = { 0,1,2, 0,2,3 };
    m.build_from_triangles(verts, idx);
    return m;
}

// ---- build_from_triangles: adjacency + centroids + containers -----
void test_build() {
    CHECK(nm::kInvalidPoly == 0xFFFFFFFFu);

    {   // Single triangle: one poly, no neighbours.
        nm::Mesh m;
        std::vector<Vec3> v = { v3(0,0,0), v3(1,0,0), v3(0,0,1) };
        std::vector<u32> idx = { 0,1,2 };
        CHECK(m.empty());
        m.build_from_triangles(v, idx);
        CHECK(!m.empty());
        CHECK(m.polys.size() == sz(1));
        CHECK(m.vertices.size() == sz(3));
        const nm::Poly& p = m.polys[0];
        CHECK(p.neighbours[0] == nm::kInvalidPoly);
        CHECK(p.neighbours[1] == nm::kInvalidPoly);
        CHECK(p.neighbours[2] == nm::kInvalidPoly);
        CHECK(apv(p.centroid, 1.0f / 3.0f, 0.0f, 1.0f / 3.0f));
        m.clear();
        CHECK(m.empty() && m.polys.empty() && m.vertices.empty());
    }
    {   // Two triangles sharing edge (0,2): exact neighbour wiring.
        nm::Mesh m = quad_mesh();
        CHECK(m.polys.size() == sz(2));
        // tri0 verts {0,1,2}: shared edge is slot 2 (edge (2,0)).
        CHECK(m.polys[0].neighbours[0] == nm::kInvalidPoly);
        CHECK(m.polys[0].neighbours[1] == nm::kInvalidPoly);
        CHECK(m.polys[0].neighbours[2] == 1u);
        // tri1 verts {0,2,3}: shared edge is slot 0 (edge (0,2)).
        CHECK(m.polys[1].neighbours[0] == 0u);
        CHECK(m.polys[1].neighbours[1] == nm::kInvalidPoly);
        CHECK(m.polys[1].neighbours[2] == nm::kInvalidPoly);
        // Centroids are the exact vertex averages.
        CHECK(apv(m.polys[0].centroid, 2.0f, 0.0f, 1.0f));
        CHECK(apv(m.polys[1].centroid, 1.0f, 0.0f, 2.0f));
    }
}

// ---- recompute_centroids tracks moved vertices --------------------
void test_recompute() {
    nm::Mesh m = quad_mesh();
    CHECK(apv(m.polys[0].centroid, 2.0f, 0.0f, 1.0f));
    m.vertices[1] = v3(6,0,0);                 // move V1 (3,0,0)→(6,0,0)
    m.recompute_centroids();
    // tri0 = V0(0,0,0) V1(6,0,0) V2(3,0,3) → ((0+6+3)/3,0,(0+0+3)/3)
    CHECK(apv(m.polys[0].centroid, 3.0f, 0.0f, 1.0f));
    CHECK(apv(m.polys[1].centroid, 1.0f, 0.0f, 2.0f));   // unaffected
}

// ---- nearest_poly: centroid-nearest, empty → kInvalidPoly ---------
void test_nearest() {
    {
        nm::Mesh empty;
        CHECK(empty.nearest_poly(v3(0,0,0)) == nm::kInvalidPoly);
    }
    nm::Mesh m = quad_mesh();                   // c0=(2,0,1) c1=(1,0,2)
    CHECK(m.nearest_poly(v3(2,0,1)) == 0u);     // exactly on c0
    CHECK(m.nearest_poly(v3(1,0,2)) == 1u);     // exactly on c1
    CHECK(m.nearest_poly(v3(2.4,0,1.1)) == 0u); // closest to c0
    CHECK(m.nearest_poly(v3(0.8,0,2.2)) == 1u); // closest to c1
}

// ---- A* + funnel on the 2-triangle mesh ---------------------------
void test_path_basic() {
    nm::Mesh m = quad_mesh();
    nm::PathQuery q;
    std::vector<Vec3> wp;
    const Vec3 start = v3(2,0,1);               // == c0 → poly 0
    const Vec3 goal  = v3(1,0,2);               // == c1 → poly 1
    nm::PathStats s = q.find_path(m, start, goal, wp);

    CHECK(s.found);
    // Cost = centroid distance c0→c1 = sqrt(2); pin its square (no sqrt).
    CHECK(ap(s.poly_path_cost * s.poly_path_cost, 2.0f, 1e-2f));
    CHECK(s.poly_visited == 2u);                // start poly + goal poly
    CHECK(s.waypoint_count == static_cast<u32>(wp.size()));
    CHECK(wp.size() >= sz(2));
    // Funnel universal contract: endpoints are exactly start / goal.
    CHECK(apv(wp.front(), 2.0f, 0.0f, 1.0f));
    CHECK(apv(wp.back(),  1.0f, 0.0f, 2.0f));
    bool all_fin = true;
    for (const auto& w : wp) if (!finv(w)) all_fin = false;
    CHECK(all_fin);
}

// ---- start & goal in the SAME poly --------------------------------
void test_path_same_poly() {
    nm::Mesh m = quad_mesh();
    nm::PathQuery q;
    std::vector<Vec3> wp;
    const Vec3 start = v3(2.0,0,1.0);           // → poly 0
    const Vec3 goal  = v3(2.1,0,1.0);           // also nearest poly 0
    nm::PathStats s = q.find_path(m, start, goal, wp);

    CHECK(s.found);
    CHECK(ap(s.poly_path_cost, 0.0f));          // single-poly path, g=0
    CHECK(s.poly_visited == 1u);                // only the start poly
    CHECK(wp.size() == sz(2));                  // just [start, goal]
    CHECK(apv(wp.front(), 2.0f, 0.0f, 1.0f));
    CHECK(apv(wp.back(),  2.1f, 0.0f, 1.0f));
    CHECK(s.waypoint_count == 2u);
}

// ---- multi-hop corridor (4 triangles) -----------------------------
void test_path_corridor() {
    // Grid x∈{0,1,2}, z∈{0,1}; v(c,r) = r*3 + c.
    nm::Mesh m;
    std::vector<Vec3> v = {
        v3(0,0,0), v3(1,0,0), v3(2,0,0),        // row z=0  (v0,v1,v2)
        v3(0,0,1), v3(1,0,1), v3(2,0,1) };      // row z=1  (v3,v4,v5)
    std::vector<u32> idx = {
        0,1,4,   0,4,3,                          // quad 0 → triA, triB
        1,2,5,   1,5,4 };                        // quad 1 → triC, triD
    m.build_from_triangles(v, idx);
    CHECK(m.polys.size() == sz(4));

    nm::PathQuery q;
    std::vector<Vec3> wp;
    const Vec3 start = v3(0.3,0,0.3);            // near triA
    const Vec3 goal  = v3(1.9,0,0.2);            // near triC (far end)
    nm::PathStats s = q.find_path(m, start, goal, wp);

    CHECK(s.found);
    CHECK(s.poly_path_cost > 0.0f);
    CHECK(s.poly_visited >= 2u);
    CHECK(wp.size() >= sz(2));
    CHECK(s.waypoint_count == static_cast<u32>(wp.size()));
    CHECK(apv(wp.front(), 0.3f, 0.0f, 0.3f));    // exact start
    CHECK(apv(wp.back(),  1.9f, 0.0f, 0.2f));    // exact goal
    bool all_fin = true;
    for (const auto& w : wp) if (!finv(w)) all_fin = false;
    CHECK(all_fin);
}

// ---- failure paths: empty mesh + disconnected components ----------
void test_path_fail() {
    {   // Empty mesh → no path, no waypoints.
        nm::Mesh empty;
        nm::PathQuery q;
        std::vector<Vec3> wp;
        nm::PathStats s = q.find_path(empty, v3(0,0,0), v3(1,0,0), wp);
        CHECK(!s.found);
        CHECK(wp.empty());
        CHECK(s.waypoint_count == 0u);
        CHECK(s.poly_visited == 0u);
    }
    {   // Two disjoint triangles (no shared edge) → goal unreachable.
        nm::Mesh m;
        std::vector<Vec3> v = {
            v3(0,0,0),  v3(1,0,0),  v3(0,0,1),     // tri0 idx 0,1,2
            v3(10,0,10),v3(11,0,10),v3(10,0,11) }; // tri1 idx 3,4,5
        std::vector<u32> idx = { 0,1,2, 3,4,5 };
        m.build_from_triangles(v, idx);
        CHECK(m.polys.size() == sz(2));
        CHECK(m.polys[0].neighbours[0] == nm::kInvalidPoly);
        CHECK(m.polys[1].neighbours[0] == nm::kInvalidPoly);

        nm::PathQuery q;
        std::vector<Vec3> wp;
        // start near tri0 centroid, goal near tri1 centroid.
        nm::PathStats s = q.find_path(m, v3(0.33,0,0.33),
                                         v3(10.33,0,10.33), wp);
        CHECK(!s.found);
        CHECK(wp.empty());
        CHECK(s.poly_visited == 1u);             // only the start poly popped
    }
}

}  // namespace

int main() {
    test_build();
    test_recompute();
    test_nearest();
    test_path_basic();
    test_path_same_poly();
    test_path_corridor();
    test_path_fail();

    if (g_fail == 0) {
        cardinal::log::infof("nmtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("nmtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
