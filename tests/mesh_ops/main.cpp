// =============================================================================
// Cardinal — deterministic CPU mesh-authoring regression suite.
//
// edit::mesh_ops is the level editor's quick-edit toolbox. Every op
// returns a NEW std::vector<Vertex> (3 verts/triangle) and must NEVER
// mutate its input. This suite pins:
//
//   * primitive generators — exact vertex/triangle COUNTS, the segment
//     clamps (sphere<4, cyl/cone/torus/disk<3, plane sub<1), AABB extents
//     (size*0.5), uniform per-primitive color, and the analytic surface
//     area of the flat ones (box=6·s², plane=s², disk=½n·sin(2π/n));
//   * subdivide — 1 tri -> 4 with EXACT midpoint positions/colors and the
//     a,ab,ca / ab,b,bc / ca,bc,c / ab,bc,ca fan order; levels=0 is a
//     pass-through; level n multiplies count by 4ⁿ;
//   * mirror — axis component negated on BOTH position and normal, winding
//     reversed (a,c,b), and the axis>=2 -> Z fallthrough;
//   * decimate_cluster — cell_size<=0 returns the input verbatim; a tri is
//     dropped when any two of its verts collapse into one grid cell;
//   * smooth_laplacian — empty in/empty out, lambda=0 / iterations=0 are
//     position-preserving, count invariant;
//   * recompute_normals — flat face normal from the winding, smooth =
//     normalized sum over position-equivalent verts, empty-safe;
//   * bake_transform — identity / translation / non-uniform scale on
//     positions, normals renormalized (translation leaves them alone);
//   * tint — component-wise color multiply; append — concatenation;
//   * stats — count, AABB, area; empty mesh is all-zero;
//   * the no-mutation contract — inputs are byte-stable across every op.
//
// Counts are integer-exact; floats use an epsilon. Transcendental
// primitives are pinned by count + radius/extent invariants, not trig.
// Pure, deterministic. Exit 0 = all pass.
// =============================================================================

#include <cardinal/edit/mesh_ops.hpp>
#include <cardinal/scene/scene.hpp>   // full cardinal::scene::Vertex definition
#include <cardinal/core/log.hpp>

#include <vector>

namespace {

namespace mo = cardinal::edit::mesh_ops;
using mo::Vertex;
using mo::Vec3;
using Mat4 = cardinal::scene::Mat4;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("meshtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

bool ap(double a, double b, double eps = 1e-4) {
    const double d = (a > b) ? (a - b) : (b - a);
    return d <= eps;
}
bool veq(const Vec3& v, double x, double y, double z, double eps = 1e-4) {
    return ap(v.x, x, eps) && ap(v.y, y, eps) && ap(v.z, z, eps);
}
double vlen(const Vec3& v) {
    const double d = static_cast<double>(v.x) * v.x
                   + static_cast<double>(v.y) * v.y
                   + static_cast<double>(v.z) * v.z;
    // crude sqrt: Newton from a coarse guess (no <cmath> in test code).
    if (d <= 0.0) return 0.0;
    double g = d;
    for (int i = 0; i < 40; ++i) g = 0.5 * (g + d / g);
    return g;
}
bool vsame(const Vec3& a, const Vec3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
// Byte-stable equality of a whole buffer (no-mutation contract).
bool buf_same(const std::vector<Vertex>& a, const std::vector<Vertex>& b) {
    if (a.size() != b.size()) return false;
    for (cardinal::usize i = 0; i < a.size(); ++i) {
        if (!vsame(a[i].position, b[i].position)) return false;
        if (!vsame(a[i].normal,   b[i].normal))   return false;
        if (!vsame(a[i].color,    b[i].color))    return false;
    }
    return true;
}

Vertex vtx(double px, double py, double pz,
           double nx, double ny, double nz,
           double cx, double cy, double cz) {
    Vertex v;
    v.position = Vec3{ static_cast<float>(px), static_cast<float>(py), static_cast<float>(pz) };
    v.normal   = Vec3{ static_cast<float>(nx), static_cast<float>(ny), static_cast<float>(nz) };
    v.color    = Vec3{ static_cast<float>(cx), static_cast<float>(cy), static_cast<float>(cz) };
    return v;
}

// ---- primitive generators: counts, clamps, extents, colors --------
void test_primitives() {
    // box: 6 faces * 2 tris * 3 verts = 36; AABB = [-s/2, s/2]^3.
    {
        auto b = mo::make_box(2.0f);
        CHECK(b.size() == sz(36));
        auto s = mo::stats(b);
        CHECK(s.vert_count == 36u && s.tri_count == 12u);
        CHECK(veq(s.aabb_min, -1, -1, -1));
        CHECK(veq(s.aabb_max,  1,  1,  1));
        CHECK(ap(s.surface_area, 24.0));         // 6 * (2*2)
        for (const auto& v : b) CHECK(veq(v.color, 0.7, 0.7, 0.7));
        auto b1 = mo::make_box();                // default size 1
        CHECK(ap(mo::stats(b1).surface_area, 6.0));
        CHECK(veq(mo::stats(b1).aabb_min, -0.5, -0.5, -0.5));
    }
    // plane: N*N*6 verts; sub clamps to >=1; flat -> area = size^2.
    {
        auto p = mo::make_plane(1.0f, 1);
        CHECK(p.size() == sz(6));
        CHECK(ap(mo::stats(p).surface_area, 1.0));
        for (const auto& v : p) {
            CHECK(veq(v.normal, 0, 1, 0));
            CHECK(veq(v.color, 0.5, 0.5, 0.5));
            CHECK(ap(v.position.y, 0.0));
        }
        CHECK(mo::make_plane(1.0f, 0).size() == sz(6));   // sub 0 -> 1
        CHECK(mo::make_plane(1.0f, 2).size() == sz(24));   // 2*2*6
        auto p3 = mo::make_plane(2.0f, 3);
        CHECK(p3.size() == sz(54));                        // 3*3*6
        CHECK(ap(mo::stats(p3).surface_area, 4.0));        // 2*2
        CHECK(veq(mo::stats(p3).aabb_min, -1, 0, -1));
        CHECK(veq(mo::stats(p3).aabb_max,  1, 0,  1));
    }
    // sphere: segments<4 clamps; verts = segments*(segments/2)*6.
    {
        auto s4 = mo::make_sphere(1.0f, 4);
        CHECK(s4.size() == sz(48));               // 4*2*6
        CHECK(mo::make_sphere(1.0f, 2).size() == sz(48));   // clamp 2->4
        auto sd = mo::make_sphere(3.0f, 24);
        CHECK(sd.size() == sz(1728));             // 24*12*6
        for (const auto& v : sd) CHECK(ap(vlen(v.position), 3.0, 1e-3));
        CHECK(mo::stats(sd).tri_count == 576u);
    }
    // cylinder: segments<3 clamps; verts = segments*12; y in [-h,h].
    {
        auto c3 = mo::make_cylinder(1.0f, 2.0f, 3);
        CHECK(c3.size() == sz(36));
        CHECK(mo::make_cylinder(1.0f, 2.0f, 1).size() == sz(36)); // clamp
        auto cd = mo::make_cylinder(1.0f, 4.0f, 24);
        CHECK(cd.size() == sz(288));
        CHECK(ap(mo::stats(cd).aabb_min.y, -2.0));
        CHECK(ap(mo::stats(cd).aabb_max.y,  2.0));
    }
    // cone: segments<3 clamps; verts = segments*6; apex at +h.
    {
        auto k3 = mo::make_cone(1.0f, 2.0f, 3);
        CHECK(k3.size() == sz(18));
        CHECK(mo::make_cone(1.0f, 2.0f, 2).size() == sz(18));     // clamp
        auto kd = mo::make_cone(1.0f, 2.0f, 24);
        CHECK(kd.size() == sz(144));
        CHECK(ap(mo::stats(kd).aabb_max.y, 1.0));   // apex y = h
        CHECK(ap(mo::stats(kd).aabb_min.y, -1.0));
    }
    // torus: maj/min < 3 clamp; verts = maj*min*6.
    {
        CHECK(mo::make_torus(1.0f, 0.3f, 3, 3).size()  == sz(54));
        CHECK(mo::make_torus(1.0f, 0.3f, 2, 2).size()  == sz(54)); // clamp
        CHECK(mo::make_torus(1.0f, 0.3f, 32, 16).size()== sz(3072));
    }
    // disk: segments<3 clamps; verts = segments*3; flat at y=0.
    {
        auto d3 = mo::make_disk(1.0f, 3);
        CHECK(d3.size() == sz(9));
        CHECK(mo::make_disk(1.0f, 2).size() == sz(9));             // clamp
        auto d4 = mo::make_disk(1.0f, 4);
        CHECK(d4.size() == sz(12));
        for (const auto& v : d4) {
            CHECK(veq(v.normal, 0, 1, 0));
            CHECK(ap(v.position.y, 0.0));
        }
        // 4 isoceles tris inscribed in unit circle: 4 * 0.5 = 2.0.
        CHECK(ap(mo::stats(d4).surface_area, 2.0, 1e-3));
        CHECK(mo::make_disk(1.0f, 32).size() == sz(96));
    }
}

// ---- subdivide: exact midpoint split + fan order ------------------
void test_subdivide() {
    std::vector<Vertex> tri;
    tri.push_back(vtx(0,0,0, 0,0,1, 1,1,1));   // a
    tri.push_back(vtx(4,0,0, 0,0,1, 1,1,1));   // b
    tri.push_back(vtx(0,4,0, 0,0,1, 1,1,1));   // c

    CHECK(mo::subdivide(tri, 0).size() == sz(3));   // levels 0 = pass-through
    CHECK(buf_same(mo::subdivide(tri, 0), tri));

    auto s = mo::subdivide(tri, 1);
    CHECK(s.size() == sz(12));                       // 1 tri -> 4
    // midpoints: ab=(2,0,0) bc=(2,2,0) ca=(0,2,0)
    CHECK(veq(s[0].position, 0, 0, 0));   // a
    CHECK(veq(s[1].position, 2, 0, 0));   // ab
    CHECK(veq(s[2].position, 0, 2, 0));   // ca
    CHECK(veq(s[3].position, 2, 0, 0));   // ab
    CHECK(veq(s[4].position, 4, 0, 0));   // b
    CHECK(veq(s[5].position, 2, 2, 0));   // bc
    CHECK(veq(s[6].position, 0, 2, 0));   // ca
    CHECK(veq(s[7].position, 2, 2, 0));   // bc
    CHECK(veq(s[8].position, 0, 4, 0));   // c
    CHECK(veq(s[9].position, 2, 0, 0));   // ab  (centre tri)
    CHECK(veq(s[10].position, 2, 2, 0));  // bc
    CHECK(veq(s[11].position, 0, 2, 0));  // ca
    for (const auto& v : s) {
        CHECK(veq(v.normal, 0, 0, 1));    // normalized sum of equal normals
        CHECK(veq(v.color, 1, 1, 1));     // midpoint of equal colors
    }
    CHECK(mo::subdivide(tri, 2).size() == sz(48));   // 4^2 * 3
    auto box = mo::make_box(1.0f);
    CHECK(mo::subdivide(box, 1).size() == sz(144));  // 36 * 4
}

// ---- mirror: negate axis on pos+normal, reverse winding -----------
void test_mirror() {
    std::vector<Vertex> tri;
    tri.push_back(vtx(1,2,3, 1,0,0, 0,0,0));   // a
    tri.push_back(vtx(4,5,6, 1,0,0, 0,0,0));   // b
    tri.push_back(vtx(7,8,9, 1,0,0, 0,0,0));   // c

    auto mx = mo::mirror(tri, 0);                    // X
    CHECK(mx.size() == sz(3));
    CHECK(veq(mx[0].position, -1, 2, 3));            // a' (winding: a,c,b)
    CHECK(veq(mx[1].position, -7, 8, 9));            // c'
    CHECK(veq(mx[2].position, -4, 5, 6));            // b'
    CHECK(veq(mx[0].normal,  -1, 0, 0));             // normal X negated too

    auto my = mo::mirror(tri, 1);                    // Y
    CHECK(veq(my[0].position, 1, -2, 3));
    CHECK(veq(my[1].position, 7, -8, 9));

    auto mz = mo::mirror(tri, 2);                    // Z
    CHECK(veq(mz[0].position, 1, 2, -3));
    auto mfall = mo::mirror(tri, 5);                 // axis>=2 -> Z
    CHECK(veq(mfall[0].position, 1, 2, -3));
    CHECK(buf_same(mz, mfall));
}

// ---- decimate_cluster: cell<=0 passthrough, cell-collapse drop ----
void test_decimate() {
    std::vector<Vertex> tri;
    tri.push_back(vtx(0.5,0.5,0.5, 0,1,0, 1,1,1));
    tri.push_back(vtx(5.5,0.5,0.5, 0,1,0, 1,1,1));
    tri.push_back(vtx(0.5,5.5,0.5, 0,1,0, 1,1,1));

    CHECK(buf_same(mo::decimate_cluster(tri, 0.0f), tri));   // cell 0 -> in
    CHECK(buf_same(mo::decimate_cluster(tri, -1.0f), tri));  // cell <0 -> in

    // 3 distinct cells @ cell_size 1 -> triangle survives.
    auto keep = mo::decimate_cluster(tri, 1.0f);
    CHECK(keep.size() == sz(3));

    // all 3 verts in one cell -> dropped.
    std::vector<Vertex> same;
    same.push_back(vtx(0.1,0.1,0.1, 0,1,0, 1,1,1));
    same.push_back(vtx(0.2,0.2,0.2, 0,1,0, 1,1,1));
    same.push_back(vtx(0.3,0.3,0.3, 0,1,0, 1,1,1));
    CHECK(mo::decimate_cluster(same, 1.0f).size() == sz(0));

    // two verts share a cell -> dropped.
    std::vector<Vertex> two;
    two.push_back(vtx(0.1,0,0, 0,1,0, 1,1,1));
    two.push_back(vtx(0.2,0,0, 0,1,0, 1,1,1));
    two.push_back(vtx(5.5,0,0, 0,1,0, 1,1,1));
    CHECK(mo::decimate_cluster(two, 1.0f).size() == sz(0));
}

// ---- smooth_laplacian: empty/identity invariants ------------------
void test_smooth() {
    std::vector<Vertex> empty;
    CHECK(mo::smooth_laplacian(empty, 3, 0.5f).empty());

    auto box = mo::make_box(2.0f);
    auto s0 = mo::smooth_laplacian(box, 0, 0.5f);     // 0 iters: pos fixed
    CHECK(s0.size() == box.size());
    for (cardinal::usize i = 0; i < box.size(); ++i)
        CHECK(vsame(s0[i].position, box[i].position));

    auto sl0 = mo::smooth_laplacian(box, 5, 0.0f);    // lambda 0: pos fixed
    CHECK(sl0.size() == box.size());
    for (cardinal::usize i = 0; i < box.size(); ++i)
        CHECK(vsame(sl0[i].position, box[i].position));

    // A real sweep keeps the count and stays within the original AABB
    // (Laplacian relaxation is strictly contractive for a closed mesh).
    auto sm = mo::smooth_laplacian(box, 3, 0.5f);
    CHECK(sm.size() == box.size());
    auto bs = mo::stats(box), ss = mo::stats(sm);
    CHECK(ss.aabb_min.x >= bs.aabb_min.x - 1e-3f);
    CHECK(ss.aabb_max.x <= bs.aabb_max.x + 1e-3f);
}

// ---- recompute_normals: flat winding + smooth average -------------
void test_normals() {
    std::vector<Vertex> empty;
    CHECK(mo::recompute_normals(empty, true).empty());
    CHECK(mo::recompute_normals(empty, false).empty());

    std::vector<Vertex> tri;                          // CCW in XY
    tri.push_back(vtx(0,0,0, 9,9,9, 0,0,0));
    tri.push_back(vtx(1,0,0, 9,9,9, 0,0,0));
    tri.push_back(vtx(0,1,0, 9,9,9, 0,0,0));
    auto f = mo::recompute_normals(tri, false);
    CHECK(f.size() == sz(3));
    for (const auto& v : f) CHECK(veq(v.normal, 0, 0, 1));   // +Z

    std::vector<Vertex> rev;                          // reversed winding
    rev.push_back(vtx(0,0,0, 0,0,0, 0,0,0));
    rev.push_back(vtx(0,1,0, 0,0,0, 0,0,0));
    rev.push_back(vtx(1,0,0, 0,0,0, 0,0,0));
    auto fr = mo::recompute_normals(rev, false);
    for (const auto& v : fr) CHECK(veq(v.normal, 0, 0, -1));  // -Z

    // smooth on a box: count preserved and every normal is unit length.
    // (The exact corner value depends on each face's two-triangle fan
    // diagonal — an impl detail of make_box — so we don't pin it here;
    // the clean averaging contract is exercised by the roof below.)
    auto box = mo::make_box(2.0f);
    auto sm = mo::recompute_normals(box, true);
    CHECK(sm.size() == box.size());
    for (const auto& v : sm) CHECK(ap(vlen(v.normal), 1.0, 1e-3));

    // Controlled smooth average: two single triangles, face normals +Z
    // and -X, sharing the verts at (0,0,0) and (0,1,0). Each shared vert
    // is touched once per face, so its smoothed normal is exactly
    // normalize(+Z + -X) = (-0.7071, 0, 0.7071); unshared verts keep
    // their lone face normal.
    std::vector<Vertex> roof;
    roof.push_back(vtx(0,0,0, 0,0,0, 0,0,0));   // [0] A  fnA = +Z
    roof.push_back(vtx(1,0,0, 0,0,0, 0,0,0));   // [1] A
    roof.push_back(vtx(0,1,0, 0,0,0, 0,0,0));   // [2] A
    roof.push_back(vtx(0,0,0, 0,0,0, 0,0,0));   // [3] B  fnB = -X
    roof.push_back(vtx(0,0,1, 0,0,0, 0,0,0));   // [4] B
    roof.push_back(vtx(0,1,0, 0,0,0, 0,0,0));   // [5] B
    auto rs = mo::recompute_normals(roof, true);
    CHECK(rs.size() == sz(6));
    CHECK(veq(rs[0].normal, -0.70711, 0, 0.70711, 1e-3));  // shared (0,0,0)
    CHECK(veq(rs[5].normal, -0.70711, 0, 0.70711, 1e-3));  // shared (0,1,0)
    CHECK(veq(rs[1].normal, 0, 0, 1, 1e-3));               // lone +Z
    CHECK(veq(rs[4].normal, -1, 0, 0, 1e-3));              // lone -X
}

// ---- bake_transform: identity / translate / scale -----------------
void test_bake() {
    std::vector<Vertex> v;
    v.push_back(vtx(1,2,3, 1,0,0, 0.2,0.3,0.4));
    v.push_back(vtx(1,1,1, 0,1,0, 0.2,0.3,0.4));

    auto id = mo::bake_transform(v, Mat4::identity());
    CHECK(id.size() == sz(2));
    CHECK(veq(id[0].position, 1, 2, 3));
    CHECK(veq(id[0].normal, 1, 0, 0));
    CHECK(veq(id[0].color, 0.2, 0.3, 0.4));            // color untouched

    auto tr = mo::bake_transform(v, Mat4::translation(Vec3{10,20,30}));
    CHECK(veq(tr[0].position, 11, 22, 33));
    CHECK(veq(tr[1].position, 11, 21, 31));
    CHECK(veq(tr[0].normal, 1, 0, 0));                 // dir unaffected by T

    auto sc = mo::bake_transform(v, Mat4::scaling(Vec3{2,3,4}));
    CHECK(veq(sc[0].position, 2, 6, 12));
    CHECK(veq(sc[1].position, 2, 3, 4));
    CHECK(veq(sc[0].normal, 1, 0, 0));                 // (2,0,0) renorm
    CHECK(veq(sc[1].normal, 0, 1, 0));                 // (0,3,0) renorm
}

// ---- tint + append ------------------------------------------------
void test_tint_append() {
    auto p = mo::make_plane(1.0f, 1);                  // color (0.5,0.5,0.5)
    auto t = mo::tint(p, Vec3{2.0f, 1.0f, 0.0f});
    CHECK(t.size() == p.size());
    for (const auto& v : t) CHECK(veq(v.color, 1.0, 0.5, 0.0));
    auto t1 = mo::tint(p, Vec3{1,1,1});
    for (const auto& v : t1) CHECK(veq(v.color, 0.5, 0.5, 0.5));

    auto box = mo::make_box(1.0f);
    auto cat = mo::append(box, p);
    CHECK(cat.size() == box.size() + p.size());        // 36 + 6
    for (cardinal::usize i = 0; i < box.size(); ++i)
        CHECK(vsame(cat[i].position, box[i].position));
    for (cardinal::usize i = 0; i < p.size(); ++i)
        CHECK(vsame(cat[box.size()+i].position, p[i].position));

    std::vector<Vertex> empty;
    CHECK(mo::append(empty, box).size() == box.size());
    CHECK(mo::append(box, empty).size() == box.size());
}

// ---- stats: empty zero, triangle area, AABB -----------------------
void test_stats() {
    std::vector<Vertex> empty;
    auto e = mo::stats(empty);
    CHECK(e.vert_count == 0u && e.tri_count == 0u);
    CHECK(veq(e.aabb_min, 0, 0, 0) && veq(e.aabb_max, 0, 0, 0));
    CHECK(ap(e.surface_area, 0.0));

    std::vector<Vertex> tri;                            // 3-4-5 right tri
    tri.push_back(vtx(0,0,0, 0,0,1, 0,0,0));
    tri.push_back(vtx(3,0,0, 0,0,1, 0,0,0));
    tri.push_back(vtx(0,4,0, 0,0,1, 0,0,0));
    auto s = mo::stats(tri);
    CHECK(s.vert_count == 3u && s.tri_count == 1u);
    CHECK(veq(s.aabb_min, 0, 0, 0));
    CHECK(veq(s.aabb_max, 3, 4, 0));
    CHECK(ap(s.surface_area, 6.0));                     // 0.5 * 3 * 4
}

// ---- no-mutation contract: inputs byte-stable across every op -----
void test_no_mutation() {
    auto src   = mo::make_box(2.0f);
    auto guard = src;                                   // independent copy

    mo::subdivide(src, 2);
    mo::mirror(src, 1);
    mo::decimate_cluster(src, 0.5f);
    mo::smooth_laplacian(src, 4, 0.5f);
    mo::recompute_normals(src, true);
    mo::recompute_normals(src, false);
    mo::bake_transform(src, Mat4::scaling(Vec3{3,3,3}));
    mo::tint(src, Vec3{9,9,9});
    mo::append(src, src);
    mo::stats(src);

    CHECK(buf_same(src, guard));                        // src untouched
}

}  // namespace

int main() {
    test_primitives();
    test_subdivide();
    test_mirror();
    test_decimate();
    test_smooth();
    test_normals();
    test_bake();
    test_tint_append();
    test_stats();
    test_no_mutation();

    if (g_fail == 0) {
        cardinal::log::infof("meshtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("meshtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
