// =============================================================================
// Cardinal — deterministic math regression suite (the keystone).
//
// Every transform, camera matrix, physics integration step and net
// interpolation lerp bottoms out in cardinal::core math (re-exported as
// cardinal::scene). A silent regression here corrupts the entire
// engine — including the binary LAYOUT invariants the RHI relies on to
// upload Vec3/Mat4 straight into GPU uniform buffers and that modules
// rely on for zero-copy type-punning at their boundaries.
//
// Pure, header-only, deterministic. No <cmath>/<cstddef> in the test
// (FOUNDATION discipline): angles use scene::kPi at values with exact
// trig (0, π/2, π), magnitudes use Pythagorean triples, and matrix
// behaviour is checked via convention-free PROPERTIES (associativity,
// M·M⁻¹ = I, geometric mappings) so the test can't bake in a wrong
// convention. Exit 0 = all pass.
// =============================================================================

#include <cardinal/scene/math.hpp>
#include <cardinal/core/log.hpp>

namespace {

namespace co = cardinal::core;
namespace sc = cardinal::scene;
using co::Vec2; using co::Vec3; using co::Vec4;
using co::Quat; using co::Mat3; using co::Mat4;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("mathtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-5f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
bool apv(const Vec3& a, const Vec3& b, float e = 1e-5f) {
    return ap(a.x, b.x, e) && ap(a.y, b.y, e) && ap(a.z, b.z, e);
}
bool mat4_is_identity(const Mat4& m, float e) {
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            if (!ap(m.m[c][r], (c == r) ? 1.0f : 0.0f, e)) return false;
    return true;
}
bool mat4_eq(const Mat4& a, const Mat4& b, float e) {
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            if (!ap(a.m[c][r], b.m[c][r], e)) return false;
    return true;
}
long long byte_off(const void* member, const void* base) {
    return static_cast<const char*>(member) -
           static_cast<const char*>(base);
}

// Launder a value through volatile so the optimiser can't constant-fold
// it. Used to build the degenerate test rays at *runtime*: a literal
// {1,0,0} direction would let MSVC specialise the inlined
// ray_plane_y_intersect with a constant-0 divisor and (correctly, for
// that dead path) raise C4723 — an artefact of the test, not a header
// defect, since real callers unproject rays from runtime matrices.
float rt(float v) { volatile float x = v; return x; }

// ---- vectors --------------------------------------------------------
void test_vectors() {
    const Vec3 a{1, 2, 3}, b{4, 5, 6};
    CHECK(apv(a + b, {5, 7, 9}));
    CHECK(apv(b - a, {3, 3, 3}));
    CHECK(apv(a * 2.0f, {2, 4, 6}));
    CHECK(apv(-a, {-1, -2, -3}));
    Vec3 c = a; c += b; CHECK(apv(c, {5, 7, 9}));
    c -= b; CHECK(apv(c, a));
    c *= 3.0f; CHECK(apv(c, {3, 6, 9}));

    CHECK(ap(co::dot(a, b), 32.0f));                 // 4+10+18
    CHECK(ap(co::dot(Vec3{1,0,0}, Vec3{0,1,0}), 0.0f));
    // Right-handed cross: x × y = +z.
    CHECK(apv(co::cross({1,0,0}, {0,1,0}), {0,0,1}));
    CHECK(apv(co::cross({0,1,0}, {1,0,0}), {0,0,-1}));
    CHECK(ap(co::length(Vec3{3,4,0}), 5.0f));        // 3-4-5
    CHECK(ap(co::length(co::normalize(Vec3{0,3,0})), 1.0f));
    CHECK(apv(co::normalize(Vec3{0,0,0}), {0,0,0})); // zero ⇒ zero (safe)
    CHECK(apv(co::lerp(Vec3{0,0,0}, Vec3{10,20,30}, 0.5f), {5,10,15}));
    CHECK(apv(co::lerp(a, b, 0.0f), a));
    CHECK(apv(co::lerp(a, b, 1.0f), b));

    CHECK(ap(co::dot(Vec2{1,2}, Vec2{3,4}), 11.0f));
    CHECK(ap(co::length(Vec2{3,4}), 5.0f));
    CHECK(ap(co::dot(Vec4{1,2,3,4}, Vec4{1,1,1,1}), 10.0f));
}

// ---- LAYOUT invariants (GPU upload / cross-module type-pun) ---------
void test_layout() {
    CHECK(sizeof(Vec2) == 8u);
    CHECK(sizeof(Vec3) == 12u);     // NOT padded to 16 — std140 packing
    CHECK(sizeof(Vec4) == 16u);
    CHECK(sizeof(Quat) == 16u);
    CHECK(sizeof(Mat3) == 36u);     // 3×3 f32, row-major
    CHECK(sizeof(Mat4) == 64u);     // 4×4 f32, column-major

    Vec3 v{};
    CHECK(byte_off(&v.x, &v) == 0);
    CHECK(byte_off(&v.y, &v) == 4);
    CHECK(byte_off(&v.z, &v) == 8);
    Vec4 q4{};
    CHECK(byte_off(&q4.w, &q4) == 12);
    Quat qq{};
    CHECK(byte_off(&qq.x, &qq) == 0);
    CHECK(byte_off(&qq.w, &qq) == 12);   // xyzw order

    // scene:: aliases must BE the core types (zero-copy boundary).
    sc::Vec3 sv{1, 2, 3};
    const co::Vec3* punned = &sv;        // compiles ⇒ same type
    CHECK(ap(punned->y, 2.0f));
}

// ---- quaternion -----------------------------------------------------
void test_quat() {
    const Quat I = Quat::identity();
    CHECK(ap(I.x,0) && ap(I.y,0) && ap(I.z,0) && ap(I.w,1));
    CHECK(apv(co::rotate(I, Vec3{7, -3, 2}), {7, -3, 2}));

    const Quat qa{0.1f, 0.2f, 0.3f, 0.9f};
    const Quat ia = I * qa;              // identity is the unit
    CHECK(ap(ia.x,qa.x) && ap(ia.y,qa.y) && ap(ia.z,qa.z) && ap(ia.w,qa.w));

    // +90° about +Z sends +X → +Y (right-handed).
    const Quat rz = co::axis_angle(Vec3{0,0,1}, sc::kPi * 0.5f);
    CHECK(apv(co::rotate(rz, Vec3{1,0,0}), {0,1,0}, 1e-5f));
    // 180° about +Y sends +X → −X.
    const Quat ry = co::axis_angle(Vec3{0,1,0}, sc::kPi);
    CHECK(apv(co::rotate(ry, Vec3{1,0,0}), {-1,0,0}, 1e-5f));

    // quat_to_mat3 agrees with rotate(); identity ⇒ identity.
    const Mat3 mi = co::quat_to_mat3(I);
    CHECK(ap(mi.m[0][0],1) && ap(mi.m[1][1],1) && ap(mi.m[2][2],1));
    CHECK(ap(mi.m[0][1],0) && ap(mi.m[2][0],0));
    const Mat3 mz = co::quat_to_mat3(rz);
    CHECK(apv(mz * Vec3{1,0,0}, co::rotate(rz, Vec3{1,0,0}), 1e-5f));

    const Quat n = co::normalize(Quat{0,0,0,5});
    CHECK(ap(n.w, 1.0f) && ap(n.x, 0.0f));
    CHECK(ap(co::length(co::normalize(Vec3{2,0,0})), 1.0f));

    // integrate() stays unit-norm and ≈ a small axis_angle step.
    const Quat step = co::integrate(I, Vec3{0,0,1}, 0.01f);
    CHECK(ap(step.x*step.x + step.y*step.y +
             step.z*step.z + step.w*step.w, 1.0f, 1e-4f));
    const Quat ref = co::axis_angle(Vec3{0,0,1}, 0.01f);
    CHECK(apv(co::rotate(step, Vec3{1,0,0}),
              co::rotate(ref,  Vec3{1,0,0}), 1e-3f));

    // euler(XYZ) <-> quat (the canonical helpers the editor gizmo uses).
    const Quat ze = co::quat_from_euler_xyz(Vec3{0,0,0});
    CHECK(ap(ze.x,0) && ap(ze.y,0) && ap(ze.z,0) && ap(ze.w,1));
    // +90° about Z from euler sends +X → +Y.
    const Quat ez = co::quat_from_euler_xyz(Vec3{0,0,sc::kPi*0.5f});
    CHECK(apv(co::rotate(ez, Vec3{1,0,0}), {0,1,0}, 1e-5f));
    // Single-axis euler == axis_angle about that axis.
    const Quat ay = co::quat_from_euler_xyz(Vec3{0, 0.7f, 0});
    CHECK(apv(co::rotate(ay, Vec3{1,0,0}),
              co::rotate(co::axis_angle(Vec3{0,1,0}, 0.7f), Vec3{1,0,0}),
              1e-5f));
    // Round-trip euler → quat → euler for representative orientations
    // (away from the gimbal pole) recovers the original angles.
    const Vec3 samples[4] = { {0.3f,-0.4f,0.5f}, {1.1f,0.2f,-0.9f},
                              {-0.6f,0.8f,0.25f}, {0.0f,1.3f,0.0f} };
    for (const Vec3& e : samples) {
        // Compare via the rotation effect (euler triples aren't unique)
        // — original and round-tripped must rotate a probe identically.
        const Quat q_src = co::quat_from_euler_xyz(e);
        const Quat q_rt  = co::quat_from_euler_xyz(
            co::quat_to_euler_xyz(q_src));
        CHECK(apv(co::rotate(q_src, Vec3{1,2,3}),
                  co::rotate(q_rt,  Vec3{1,2,3}), 1e-3f));
    }
}

// ---- Mat3 -----------------------------------------------------------
void test_mat3() {
    const Mat3 I = Mat3::identity();
    CHECK(apv(I * Vec3{3,-4,5}, {3,-4,5}));
    const Mat3 d = Mat3::diagonal(2, 3, 4);
    CHECK(apv(d * Vec3{1,1,1}, {2,3,4}));
    CHECK(apv(Mat3::zero() * Vec3{9,9,9}, {0,0,0}));

    // (A·B)·v == A·(B·v).
    const Mat3 A = co::quat_to_mat3(co::axis_angle(Vec3{0,0,1}, 0.7f));
    const Mat3 B = Mat3::diagonal(1.5f, 2.0f, 0.5f);
    const Vec3 v{1, -2, 3};
    CHECK(apv((A * B) * v, A * (B * v), 1e-4f));
    // transpose is an involution; identity is symmetric.
    CHECK(apv(A.transposed().transposed() * v, A * v, 1e-5f));
}

// ---- Mat4 -----------------------------------------------------------
void test_mat4() {
    const Mat4 I = Mat4::identity();
    CHECK(mat4_is_identity(I, 1e-6f));
    const Vec4 p{2, 3, 4, 1};
    Vec4 ip = I * p;
    CHECK(ap(ip.x,2) && ap(ip.y,3) && ap(ip.z,4) && ap(ip.w,1));

    // Translation moves POINTS (w=1), leaves DIRECTIONS (w=0) alone.
    const Mat4 T = Mat4::translation(Vec3{5, 6, 7});
    const Vec4 tp = T * Vec4{1, 2, 3, 1};
    CHECK(ap(tp.x,6) && ap(tp.y,8) && ap(tp.z,10) && ap(tp.w,1));
    const Vec4 td = T * Vec4{1, 2, 3, 0};
    CHECK(ap(td.x,1) && ap(td.y,2) && ap(td.z,3));

    const Mat4 S = Mat4::scaling(Vec3{2, 3, 4});
    const Vec4 sp = S * Vec4{1, 1, 1, 1};
    CHECK(ap(sp.x,2) && ap(sp.y,3) && ap(sp.z,4));

    // rotation_xyz(0) == identity; +90° about Z sends +X → +Y.
    CHECK(mat4_is_identity(Mat4::rotation_xyz(Vec3{0,0,0}), 1e-6f));
    const Mat4 Rz = Mat4::rotation_xyz(Vec3{0, 0, sc::kPi * 0.5f});
    const Vec4 rx = Rz * Vec4{1, 0, 0, 0};
    CHECK(ap(rx.x,0,1e-5f) && ap(rx.y,1,1e-5f) && ap(rx.z,0,1e-5f));

    // Convention-free: (A·B)·v == A·(B·v); identity is neutral.
    const Mat4 A = Mat4::translation(Vec3{1,2,3});
    const Mat4 B = Mat4::scaling(Vec3{2,2,2});
    const Vec4 v{1, 1, 1, 1};
    const Vec4 ab1 = (A * B) * v;
    const Vec4 ab2 = A * (B * v);
    CHECK(ap(ab1.x,ab2.x) && ap(ab1.y,ab2.y) &&
          ap(ab1.z,ab2.z) && ap(ab1.w,ab2.w));
    CHECK(mat4_eq(A * I, A, 1e-5f));        // identity is neutral
    CHECK(mat4_eq(I * A, A, 1e-5f));

    // M · M⁻¹ ≈ I for a full TRS, a view, and a projection.
    const Mat4 M = Mat4::translation(Vec3{3,-2,7}) *
                   Mat4::rotation_xyz(Vec3{0.3f, -0.7f, 1.1f}) *
                   Mat4::scaling(Vec3{1.5f, 0.5f, 2.0f});
    CHECK(mat4_is_identity(M * M.inverse(), 1e-3f));
    CHECK(mat4_is_identity(M.inverse() * M, 1e-3f));

    const Mat4 V = Mat4::look_at(Vec3{0,0,5}, Vec3{0,0,0}, Vec3{0,1,0});
    CHECK(mat4_is_identity(V * V.inverse(), 1e-3f));
    const Mat4 P = Mat4::perspective(sc::kPi / 3.0f, 16.0f/9.0f,
                                     0.1f, 100.0f);
    CHECK(mat4_is_identity(P * P.inverse(), 1e-2f));

    // Singular matrix ⇒ documented identity fallback.
    Mat4 zero{};
    CHECK(mat4_is_identity(zero.inverse(), 1e-6f));

    // Perspective structural invariants (no <cmath> needed).
    CHECK(ap(P.m[2][3], -1.0f));
    CHECK(ap(P.m[3][3],  0.0f));
    CHECK(ap(P.m[0][0] * (16.0f/9.0f), P.m[1][1], 1e-4f)); // m00=f/asp, m11=f
    CHECK(P.m[2][2] < 0.0f && P.m[3][2] < 0.0f);

    // look_at: eye → origin; target lands on the view axis (x≈y≈0).
    const Vec4 ve = V * Vec4{0,0,5,1};
    CHECK(ap(ve.x,0,1e-5f) && ap(ve.y,0,1e-5f) && ap(ve.z,0,1e-5f));
    const Vec4 vt = V * Vec4{0,0,0,1};
    CHECK(ap(vt.x,0,1e-5f) && ap(vt.y,0,1e-5f));
}

// ---- scene Ray helpers ---------------------------------------------
void test_ray() {
    sc::Ray down{ Vec3{0,10,0}, Vec3{rt(0.0f), rt(-1.0f), rt(0.0f)} };
    Vec3 hit{};
    CHECK(sc::ray_plane_y_intersect(down, 0.0f, &hit));
    CHECK(apv(hit, {0,0,0}, 1e-4f));
    // Parallel (dy==0) ⇒ no hit; runtime-sourced so the guard is the
    // thing under test, not the optimiser.
    sc::Ray side{ Vec3{0,10,0}, Vec3{rt(1.0f), rt(0.0f), rt(0.0f)} };
    CHECK(!sc::ray_plane_y_intersect(side, 0.0f, nullptr));
    // Wrong-way (heading away from the plane) ⇒ no hit.
    sc::Ray up{ Vec3{0,10,0}, Vec3{rt(0.0f), rt(1.0f), rt(0.0f)} };
    CHECK(!sc::ray_plane_y_intersect(up, 0.0f, nullptr));

    sc::Ray fwd{ Vec3{0,0,-5}, Vec3{0,0,1} };
    float t = -1.0f;
    CHECK(sc::ray_sphere_intersect(fwd, Vec3{0,0,0}, 1.0f, &t));
    CHECK(ap(t, 4.0f, 1e-4f));            // hits near side at z=-1
    sc::Ray miss{ Vec3{0,0,-5}, Vec3{0,1,0} };
    CHECK(!sc::ray_sphere_intersect(miss, Vec3{0,0,0}, 1.0f, nullptr));

    // unproject: centre NDC ray from a +Z camera points into the scene.
    const Mat4 view = Mat4::look_at(Vec3{0,0,5}, Vec3{0,0,0},
                                    Vec3{0,1,0});
    const Mat4 proj = Mat4::perspective(sc::kPi/3.0f, 1.0f, 0.1f, 100.0f);
    sc::Ray r = sc::unproject_ndc_ray(0.0f, 0.0f, view, proj);
    CHECK(ap(co::length(r.direction), 1.0f, 1e-3f));
    CHECK(r.direction.z < 0.0f);          // away from camera, into scene
}

// The viewport pick must unproject with the SAME aspect the scene was
// RENDERED with (the committed RTT size), not a different panel aspect.
// Using a mismatched aspect skews the ray -> the placed object lands at a
// constant offset from the cursor (the release-build placement bug). This
// pins the invariant in pure math: matching aspect hits the target point,
// mismatched aspect misses it by a clear margin.
void test_unproject_aspect_invariant() {
    const Vec3 cam{ 0.0f, 0.0f, 5.0f };
    const Mat4 view = Mat4::look_at(cam, Vec3{0,0,0}, Vec3{0,1,0});
    const float aspect_render = 16.0f / 9.0f;
    const Mat4 proj_r = Mat4::perspective(sc::kPi / 3.0f, aspect_render, 0.1f, 100.0f);

    // An OFF-CENTRE world point — aspect only matters away from the centre axis.
    const Vec3 P{ 1.0f, 0.6f, 0.0f };
    const Vec4 clip = proj_r * (view * Vec4{ P.x, P.y, P.z, 1.0f });
    const float ndc_x = clip.x / clip.w;
    const float ndc_y = clip.y / clip.w;

    // Perpendicular distance from a point to a (unit-direction) ray.
    auto point_ray_dist = [](const Vec3& pt, const sc::Ray& ray) {
        const Vec3  w  = pt - ray.origin;
        const float tt = co::dot(w, ray.direction);
        const Vec3  cp = ray.origin + ray.direction * tt;
        return co::length(pt - cp);
    };

    // MATCHING aspect (render == unproject): the ray passes through P.
    const sc::Ray ok = sc::unproject_ndc_ray(ndc_x, ndc_y, view, proj_r);
    CHECK(point_ray_dist(P, ok) < 1e-2f);

    // MISMATCHED aspect (the bug): the ray misses P by a clear margin.
    const Mat4 proj_w = Mat4::perspective(sc::kPi / 3.0f, 1.0f, 0.1f, 100.0f);
    const sc::Ray bad = sc::unproject_ndc_ray(ndc_x, ndc_y, view, proj_w);
    CHECK(point_ray_dist(P, bad) > 0.05f);
}

// Mat4::ortho + the directional shadow-map light VP. The light VP centres an
// ortho box on a point (the camera target) and texel-snaps it; the regression
// the test pins: an OFF-ORIGIN focus must map INSIDE the shadow box (the old
// world-origin box left off-origin objects unshadowed), and a caster above a
// ground point must reproject to a SMALLER depth (so the PCF compare shadows
// the ground). Convention-free: checked via geometric mappings.
void test_directional_shadow_vp() {
    auto approx = [](float a, float b, float e) { return cardinal::abs(a - b) < e; };

    // ---- Mat4::ortho maps the box to the [-1,1]² × [0,1] clip cube --------
    const Mat4 o = Mat4::ortho(-2.0f, 2.0f, -2.0f, 2.0f, 0.0f, 10.0f);
    auto ondc = [&](float x, float y, float z) {
        const Vec4 c = o * Vec4{ x, y, z, 1.0f };
        return Vec3{ c.x / c.w, c.y / c.w, c.z / c.w };
    };
    const Vec3 cen = ondc(0.0f, 0.0f, -5.0f);   // box centre → NDC origin, mid-depth
    CHECK(approx(cen.x, 0.0f, 1e-5f) && approx(cen.y, 0.0f, 1e-5f));
    CHECK(approx(cen.z, 0.5f, 1e-5f));
    const Vec3 corner = ondc(2.0f, 2.0f, -10.0f);  // far top-right corner → (1,1,1)
    CHECK(approx(corner.x, 1.0f, 1e-5f) && approx(corner.y, 1.0f, 1e-5f));
    CHECK(approx(corner.z, 1.0f, 1e-5f));

    // ---- Off-origin directional light VP ---------------------------------
    const Vec3 sun{ 0.0f, -1.0f, 0.0f };          // straight-down sun
    const Vec3 focus{ 50.0f, 0.0f, -30.0f };      // OFF the world origin
    const float half = 32.0f;
    const Mat4 vp = sc::directional_light_vp(sun, focus, half, 0.05f, 200.0f, 2048u);
    auto lndc = [&](const Vec3& p) {
        const Vec4 c = vp * Vec4{ p.x, p.y, p.z, 1.0f };
        const float iw = (cardinal::abs(c.w) > 1e-6f) ? 1.0f / c.w : 1.0f;
        return Vec3{ c.x * iw, c.y * iw, c.z * iw };
    };
    // The focus point must land INSIDE the shadow box (the bug: off-origin
    // points fell outside the world-origin box and read as unshadowed).
    const Vec3 nf = lndc(focus);
    CHECK(cardinal::abs(nf.x) <= 1.0f && cardinal::abs(nf.y) <= 1.0f);
    CHECK(nf.z >= 0.0f && nf.z <= 1.0f);
    // Texel-snap keeps the focus very near (but not exactly at) box centre.
    CHECK(cardinal::abs(nf.x) < 0.05f && cardinal::abs(nf.y) < 0.05f);

    // A caster 5 units ABOVE the focus reprojects to the same XY but a
    // SMALLER depth (nearer the top-down sun) → the PCF compare shadows the
    // ground beneath it.
    const Vec3 occ = lndc(Vec3{ focus.x, focus.y + 5.0f, focus.z });
    CHECK(approx(occ.x, nf.x, 1e-3f) && approx(occ.y, nf.y, 1e-3f));
    CHECK(occ.z < nf.z);

    // A point well outside the box (80 units east, box half=32) falls outside
    // [-1,1] → correctly reads as lit (no false shadow at the map edge).
    const Vec3 out = lndc(Vec3{ focus.x + 80.0f, focus.y, focus.z });
    CHECK(cardinal::abs(out.x) > 1.0f);
}

}  // namespace

int main() {
    test_vectors();
    test_layout();
    test_quat();
    test_mat3();
    test_mat4();
    test_ray();
    test_unproject_aspect_invariant();
    test_directional_shadow_vp();

    if (g_fail == 0) {
        cardinal::log::infof("mathtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("mathtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
