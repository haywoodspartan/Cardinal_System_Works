// =============================================================================
// Cardinal — Physics engine.
//
// Real impulse-based rigid-body dynamics with rotation:
//   • Shapes: Sphere / OBB / Plane / Capsule
//   • Linear + angular dynamics (mass, inertia tensor, force, torque)
//   • Sleeping (auto-park idle bodies)
//   • Sensor / trigger volumes
//   • Distance joints
//   • Uniform-grid spatial broad phase + N² fallback
//   • Sequential-impulse solver with friction + Baumgarte position correction
//   • Single-shape + multi-hit raycasts, sphere cast, overlap queries
//
// Architecture mirrors a small commercial engine (e.g. Box3D / cyclone).
// Single-file implementation; under ~1000 lines including comments because
// the broad-phase / narrow-phase / solver / integrator are deliberately
// minimal — fewer features per pass beats a thousand-line solver per pass.
// =============================================================================
#include <cardinal/physics/physics.hpp>

#include <cardinal/core/log.hpp>
#include <cardinal/core/id_gen.hpp>
#include <cardinal/core/simd_math.hpp>
#include <cardinal/core/typedefs.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace cardinal::physics {

// Note: Vec3/Quat/Mat3 math (length, normalize, rotate, axis_angle,
// integrate, quat_to_mat3) lives in <cardinal/core/math.hpp> as inline
// constexpr/noexcept functions — physics::physics.hpp re-exports them
// via `using` so this file calls them unqualified just like before.

namespace {
inline Vec3 vmin(const Vec3& a, const Vec3& b) noexcept {
    return { std::min(a.x,b.x), std::min(a.y,b.y), std::min(a.z,b.z) };
}
inline Vec3 vmax(const Vec3& a, const Vec3& b) noexcept {
    return { std::max(a.x,b.x), std::max(a.y,b.y), std::max(a.z,b.z) };
}
inline f32 clamp(f32 x, f32 lo, f32 hi) noexcept {
    return x < lo ? lo : (x > hi ? hi : x);
}
// Closest point on segment [a,b] to point p.
inline Vec3 closest_point_on_segment(const Vec3& a, const Vec3& b,
                                     const Vec3& p) noexcept {
    const Vec3 ab = b - a;
    const f32  l2 = dot(ab, ab);
    if (l2 < 1e-12f) return a;
    f32 t = dot(p - a, ab) / l2;
    t = clamp(t, 0.0f, 1.0f);
    return a + ab * t;
}
}  // namespace

Collider Collider::make_sphere(f32 r) noexcept {
    Collider c{}; c.kind = ShapeKind::Sphere; c.sphere.radius = r; return c;
}
Collider Collider::make_box(Vec3 he) noexcept {
    Collider c{}; c.kind = ShapeKind::Box;    c.box.half_extents = he; return c;
}
Collider Collider::make_plane(Vec3 n, f32 d) noexcept {
    Collider c{}; c.kind = ShapeKind::Plane;  c.plane.normal = normalize(n); c.plane.d = d; return c;
}
Collider Collider::make_capsule(f32 r, f32 hh) noexcept {
    Collider c{}; c.kind = ShapeKind::Capsule;
    c.capsule.radius = r; c.capsule.half_height = hh; return c;
}

// =============================================================================
// Internal Body
// =============================================================================
struct Body {
    BodyType  type{BodyType::Dynamic};
    Vec3      position{};
    Vec3      velocity{};
    Vec3      force_accum{};
    f32       inv_mass{1.0f};

    Quat      orientation{Quat::identity()};
    Vec3      angular_velocity{};
    Vec3      torque_accum{};
    Mat3      inv_inertia_local{Mat3::identity()};   // body frame
    Mat3      inv_inertia_world{Mat3::identity()};   // refreshed per substep

    Collider  collider{};
    Material  material{};
    bool      gravity{true};
    u32       layer{0};
    u32       mask {0xFFFFFFFFu};
    bool      live{false};
    bool      is_sensor{false};

    f32       linear_damping{0.0f};
    f32       angular_damping{0.05f};

    bool      can_sleep{true};
    bool      sleeping{false};
    f32       sleep_timer{0.0f};
    f32       linear_sleep_threshold{0.10f};
    f32       angular_sleep_threshold{0.20f};
    f32       sleep_time_seconds{0.5f};
};

// Compute the inverse inertia tensor for the given collider + mass (body
// local frame). Standard formulas — sphere (2/5 m r²), box (1/12 m
// (a²+b²) per axis), capsule (treat as cylinder + 2 hemispheres lumped),
// plane (infinite — zero rotation, leave 0).
static Mat3 compute_inv_inertia_local(const Collider& c, f32 mass) noexcept {
    if (mass <= 0.0f) return Mat3::zero();
    switch (c.kind) {
        case ShapeKind::Sphere: {
            const f32 r = c.sphere.radius;
            const f32 i = (2.0f / 5.0f) * mass * r * r;
            return Mat3::diagonal(1.0f/i, 1.0f/i, 1.0f/i);
        }
        case ShapeKind::Box: {
            const Vec3 he = c.box.half_extents;
            const f32 wx = (2.0f * he.x), wy = (2.0f * he.y), wz = (2.0f * he.z);
            const f32 ix = (1.0f / 12.0f) * mass * (wy*wy + wz*wz);
            const f32 iy = (1.0f / 12.0f) * mass * (wx*wx + wz*wz);
            const f32 iz = (1.0f / 12.0f) * mass * (wx*wx + wy*wy);
            return Mat3::diagonal(1.0f/ix, 1.0f/iy, 1.0f/iz);
        }
        case ShapeKind::Capsule: {
            // Approx: cylinder of length 2*hh + sphere caps lumped together.
            const f32 r = c.capsule.radius;
            const f32 h = c.capsule.half_height * 2.0f;
            const f32 iy = 0.5f * mass * r * r;                // around long axis
            const f32 ix = (1.0f/12.0f) * mass * (3.0f*r*r + h*h);
            return Mat3::diagonal(1.0f/ix, 1.0f/iy, 1.0f/ix);
        }
        case ShapeKind::Plane:
        default:
            return Mat3::zero();   // infinite mass → zero inv inertia
    }
}

// Refresh inv_inertia_world = R * I^-1_local * R^T from the current orientation.
static void refresh_world_inertia(Body& b) noexcept {
    if (b.type != BodyType::Dynamic) { b.inv_inertia_world = Mat3::zero(); return; }
    const Mat3 R = quat_to_mat3(b.orientation);
    b.inv_inertia_world = R * b.inv_inertia_local * R.transposed();
}

// AABB for a body's collider in world space. Capsules + rotated boxes use
// the conservative axis-aligned envelope of the rotated extents.
static AABB body_aabb(const Body& b) noexcept {
    AABB out{};
    switch (b.collider.kind) {
        case ShapeKind::Sphere: {
            const f32 r = b.collider.sphere.radius;
            out.min = b.position - Vec3{r,r,r};
            out.max = b.position + Vec3{r,r,r};
            break;
        }
        case ShapeKind::Box: {
            const Vec3& he = b.collider.box.half_extents;
            // Rotated AABB envelope: |R| * he (where |R| is component-wise
            // absolute value of the rotation matrix).
            const Mat3 R = quat_to_mat3(b.orientation);
            Vec3 e{
                std::fabs(R.m[0][0])*he.x + std::fabs(R.m[0][1])*he.y + std::fabs(R.m[0][2])*he.z,
                std::fabs(R.m[1][0])*he.x + std::fabs(R.m[1][1])*he.y + std::fabs(R.m[1][2])*he.z,
                std::fabs(R.m[2][0])*he.x + std::fabs(R.m[2][1])*he.y + std::fabs(R.m[2][2])*he.z,
            };
            out.min = b.position - e;
            out.max = b.position + e;
            break;
        }
        case ShapeKind::Capsule: {
            const f32 r = b.collider.capsule.radius;
            const f32 hh = b.collider.capsule.half_height;
            // Long axis = body's local Y rotated into world.
            const Vec3 axis = rotate(b.orientation, Vec3{0, 1, 0}) * hh;
            const Vec3 a = b.position + axis;
            const Vec3 c = b.position - axis;
            out.min = vmin(a, c) - Vec3{r,r,r};
            out.max = vmax(a, c) + Vec3{r,r,r};
            break;
        }
        case ShapeKind::Plane:
            out.min = Vec3{ -1e9f, -1e9f, -1e9f };
            out.max = Vec3{  1e9f,  1e9f,  1e9f };
            break;
    }
    return out;
}

static bool aabb_overlap(const AABB& a, const AABB& b) noexcept {
    return (a.min.x <= b.max.x && a.max.x >= b.min.x)
        && (a.min.y <= b.max.y && a.max.y >= b.min.y)
        && (a.min.z <= b.max.z && a.max.z >= b.min.z);
}

// =============================================================================
// Capsule helpers — grab the world-space line segment of a capsule body.
// =============================================================================
namespace {
inline void capsule_segment(const Body& b, Vec3& a, Vec3& c) noexcept {
    const Vec3 axis = rotate(b.orientation, Vec3{0, 1, 0}) * b.collider.capsule.half_height;
    a = b.position + axis;
    c = b.position - axis;
}
}  // namespace

// =============================================================================
// Narrow-phase contact tests (out.normal points from a → b).
// =============================================================================
static bool sphere_sphere(const Body& a, const Body& b, ContactEvent& out) {
    const f32 ra = a.collider.sphere.radius;
    const f32 rb = b.collider.sphere.radius;
    const Vec3 d = b.position - a.position;
    const f32 dist2 = dot(d, d);
    const f32 r = ra + rb;
    if (dist2 >= r * r) return false;
    const f32 dist = std::sqrt(dist2);
    out.normal      = (dist > 1e-6f) ? d * (1.0f / dist) : Vec3{0, 1, 0};
    out.penetration = r - dist;
    out.point       = a.position + out.normal * (ra - out.penetration * 0.5f);
    return true;
}

static bool sphere_plane(const Body& s, const Body& p, ContactEvent& out) {
    const f32 r  = s.collider.sphere.radius;
    const Vec3 n = p.collider.plane.normal;
    const f32 dist = dot(n, s.position) + p.collider.plane.d;
    if (dist >= r) return false;
    out.normal      = -n;
    out.penetration = r - dist;
    out.point       = s.position - n * dist;
    return true;
}

// Sphere vs OBB: transform sphere into box-local frame, clamp, transform back.
static bool sphere_box(const Body& s, const Body& bx, ContactEvent& out) {
    const f32 r   = s.collider.sphere.radius;
    const Vec3 he = bx.collider.box.half_extents;
    const Mat3 R  = quat_to_mat3(bx.orientation);
    const Mat3 Rt = R.transposed();
    const Vec3 d  = s.position - bx.position;
    const Vec3 d_local = Rt * d;
    const Vec3 c_local{ clamp(d_local.x, -he.x, he.x),
                        clamp(d_local.y, -he.y, he.y),
                        clamp(d_local.z, -he.z, he.z) };
    const Vec3 diff_local = d_local - c_local;
    const f32  d2 = dot(diff_local, diff_local);
    if (d2 >= r * r) return false;
    const Vec3 c_world = bx.position + R * c_local;
    const Vec3 diff_world = R * diff_local;
    const f32  dist = std::sqrt(d2);
    out.normal      = (dist > 1e-6f) ? diff_world * (-1.0f / dist) : Vec3{0, 1, 0};
    out.penetration = (dist > 1e-6f) ? (r - dist) : r;
    out.point       = c_world;
    return true;
}

// OBB vs OBB via Separating Axis Theorem (15 axes: 3+3+9). Returns the
// minimum penetration axis as the contact normal.
static bool box_box(const Body& a, const Body& b, ContactEvent& out) {
    const Mat3 Ra = quat_to_mat3(a.orientation);
    const Mat3 Rb = quat_to_mat3(b.orientation);
    const Vec3 ax_a[3] = {
        { Ra.m[0][0], Ra.m[1][0], Ra.m[2][0] },
        { Ra.m[0][1], Ra.m[1][1], Ra.m[2][1] },
        { Ra.m[0][2], Ra.m[1][2], Ra.m[2][2] },
    };
    const Vec3 ax_b[3] = {
        { Rb.m[0][0], Rb.m[1][0], Rb.m[2][0] },
        { Rb.m[0][1], Rb.m[1][1], Rb.m[2][1] },
        { Rb.m[0][2], Rb.m[1][2], Rb.m[2][2] },
    };
    const Vec3 he_a = a.collider.box.half_extents;
    const Vec3 he_b = b.collider.box.half_extents;
    const Vec3 t   = b.position - a.position;

    f32  best_overlap = 1e30f;
    Vec3 best_axis{0, 1, 0};

    auto try_axis = [&](Vec3 L) -> bool {
        const f32 ll = length(L);
        if (ll < 1e-6f) return true;   // degenerate axis — skip
        L = L * (1.0f / ll);
        const f32 ra = std::fabs(dot(ax_a[0], L)) * he_a.x
                     + std::fabs(dot(ax_a[1], L)) * he_a.y
                     + std::fabs(dot(ax_a[2], L)) * he_a.z;
        const f32 rb = std::fabs(dot(ax_b[0], L)) * he_b.x
                     + std::fabs(dot(ax_b[1], L)) * he_b.y
                     + std::fabs(dot(ax_b[2], L)) * he_b.z;
        const f32 dist = std::fabs(dot(t, L));
        const f32 overlap = (ra + rb) - dist;
        if (overlap < 0.0f) return false;     // separating axis found
        if (overlap < best_overlap) {
            best_overlap = overlap;
            best_axis    = (dot(t, L) < 0.0f) ? -L : L;
        }
        return true;
    };

    for (int i = 0; i < 3; ++i) if (!try_axis(ax_a[i])) return false;
    for (int i = 0; i < 3; ++i) if (!try_axis(ax_b[i])) return false;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (!try_axis(cross(ax_a[i], ax_b[j]))) return false;

    out.normal      = best_axis;
    out.penetration = best_overlap;
    out.point       = (a.position + b.position) * 0.5f;
    return true;
}

static bool box_plane(const Body& bx, const Body& p, ContactEvent& out) {
    const Vec3 he = bx.collider.box.half_extents;
    const Mat3 R  = quat_to_mat3(bx.orientation);
    const Vec3 n  = p.collider.plane.normal;
    // Project box's 3 axes onto plane normal.
    const Vec3 ax[3] = {
        { R.m[0][0], R.m[1][0], R.m[2][0] },
        { R.m[0][1], R.m[1][1], R.m[2][1] },
        { R.m[0][2], R.m[1][2], R.m[2][2] },
    };
    const f32  proj = std::fabs(dot(ax[0], n)) * he.x
                    + std::fabs(dot(ax[1], n)) * he.y
                    + std::fabs(dot(ax[2], n)) * he.z;
    const f32  dist = dot(n, bx.position) + p.collider.plane.d;
    if (dist >= proj) return false;
    out.normal      = -n;
    out.penetration = proj - dist;
    out.point       = bx.position - n * dist;
    return true;
}

// Capsule helpers — closest point between segment + segment, and
// segment + point. Reused by all capsule narrow-phase pairs.
namespace {
inline void closest_segment_segment(const Vec3& a0, const Vec3& a1,
                                    const Vec3& b0, const Vec3& b1,
                                    Vec3& cpa, Vec3& cpb) noexcept
{
    const Vec3 da = a1 - a0;
    const Vec3 db = b1 - b0;
    const Vec3 r  = a0 - b0;
    const f32  a  = dot(da, da);
    const f32  e  = dot(db, db);
    const f32  f  = dot(db, r);
    f32 s = 0, t = 0;
    if (a < 1e-6f && e < 1e-6f) { cpa = a0; cpb = b0; return; }
    if (a < 1e-6f) { s = 0; t = clamp(f / e, 0.0f, 1.0f); }
    else {
        const f32 c = dot(da, r);
        if (e < 1e-6f) { t = 0; s = clamp(-c / a, 0.0f, 1.0f); }
        else {
            const f32 b = dot(da, db);
            const f32 denom = a*e - b*b;
            s = (denom != 0) ? clamp((b*f - c*e) / denom, 0.0f, 1.0f) : 0.0f;
            t = (b*s + f) / e;
            if (t < 0) { t = 0; s = clamp(-c / a, 0.0f, 1.0f); }
            else if (t > 1) { t = 1; s = clamp((b - c) / a, 0.0f, 1.0f); }
        }
    }
    cpa = a0 + da * s;
    cpb = b0 + db * t;
}
}  // namespace

static bool sphere_capsule(const Body& s, const Body& cap, ContactEvent& out) {
    const f32 sr = s.collider.sphere.radius;
    const f32 cr = cap.collider.capsule.radius;
    Vec3 a0, a1; capsule_segment(cap, a0, a1);
    const Vec3 cp = closest_point_on_segment(a0, a1, s.position);
    const Vec3 d  = s.position - cp;
    const f32  d2 = dot(d, d);
    const f32  r  = sr + cr;
    if (d2 >= r * r) return false;
    const f32 dist = std::sqrt(d2);
    out.normal      = (dist > 1e-6f) ? d * (-1.0f / dist) : Vec3{0, 1, 0};
    out.penetration = r - dist;
    out.point       = cp + (d * (sr / std::max(dist, 1e-6f)));
    return true;
}

static bool capsule_plane(const Body& cap, const Body& p, ContactEvent& out) {
    Vec3 a0, a1; capsule_segment(cap, a0, a1);
    const f32 r  = cap.collider.capsule.radius;
    const Vec3 n = p.collider.plane.normal;
    const f32 d0 = dot(n, a0) + p.collider.plane.d;
    const f32 d1 = dot(n, a1) + p.collider.plane.d;
    const f32 dist = std::min(d0, d1);
    if (dist >= r) return false;
    out.normal      = -n;
    out.penetration = r - dist;
    out.point       = ((d0 < d1) ? a0 : a1) - n * dist;
    return true;
}

static bool capsule_capsule(const Body& a, const Body& b, ContactEvent& out) {
    const f32 ra = a.collider.capsule.radius;
    const f32 rb = b.collider.capsule.radius;
    Vec3 a0, a1; capsule_segment(a, a0, a1);
    Vec3 b0, b1; capsule_segment(b, b0, b1);
    Vec3 cpa, cpb;
    closest_segment_segment(a0, a1, b0, b1, cpa, cpb);
    const Vec3 d  = cpb - cpa;
    const f32  d2 = dot(d, d);
    const f32  r  = ra + rb;
    if (d2 >= r * r) return false;
    const f32 dist = std::sqrt(d2);
    out.normal      = (dist > 1e-6f) ? d * (1.0f / dist) : Vec3{0, 1, 0};
    out.penetration = r - dist;
    out.point       = cpa + out.normal * ra;
    return true;
}

static bool capsule_box(const Body& cap, const Body& bx, ContactEvent& out) {
    // Approximate: sample the capsule axis at its closest point to the box
    // centre, then use sphere-vs-box. Acceptable for short capsules; an
    // analytic capsule-OBB lands when needed.
    Vec3 a0, a1; capsule_segment(cap, a0, a1);
    const Vec3 cp = closest_point_on_segment(a0, a1, bx.position);
    Body proxy = cap;
    proxy.collider = Collider::make_sphere(cap.collider.capsule.radius);
    proxy.position = cp;
    return sphere_box(proxy, bx, out);
}

// Narrow-phase dispatch (handles symmetric pairs by swapping + flipping normal).
static bool narrow_test(const Body& a, const Body& b, ContactEvent& out) {
    using K = ShapeKind;
    const auto ka = a.collider.kind, kb = b.collider.kind;
    if (ka == K::Sphere  && kb == K::Sphere ) return sphere_sphere (a, b, out);
    if (ka == K::Sphere  && kb == K::Plane  ) return sphere_plane  (a, b, out);
    if (ka == K::Sphere  && kb == K::Box    ) return sphere_box    (a, b, out);
    if (ka == K::Sphere  && kb == K::Capsule) return sphere_capsule(a, b, out);
    if (ka == K::Box     && kb == K::Box    ) return box_box       (a, b, out);
    if (ka == K::Box     && kb == K::Plane  ) return box_plane     (a, b, out);
    if (ka == K::Capsule && kb == K::Plane  ) return capsule_plane (a, b, out);
    if (ka == K::Capsule && kb == K::Capsule) return capsule_capsule(a, b, out);
    if (ka == K::Capsule && kb == K::Box    ) return capsule_box   (a, b, out);
    // Symmetric pairs — swap and flip normal.
    auto sym = [&](bool (*f)(const Body&, const Body&, ContactEvent&)) {
        const bool r = f(b, a, out); out.normal = -out.normal; return r;
    };
    if (ka == K::Plane   && kb == K::Sphere ) return sym(sphere_plane);
    if (ka == K::Box     && kb == K::Sphere ) return sym(sphere_box);
    if (ka == K::Capsule && kb == K::Sphere ) return sym(sphere_capsule);
    if (ka == K::Plane   && kb == K::Box    ) return sym(box_plane);
    if (ka == K::Plane   && kb == K::Capsule) return sym(capsule_plane);
    if (ka == K::Box     && kb == K::Capsule) return sym(capsule_box);
    return false;   // plane-plane: both infinite, skip
}

// =============================================================================
// Ray intersections.
// =============================================================================
static bool ray_sphere(const Ray& r, const Body& s, f32& t_out) {
    const Vec3 oc = r.origin - s.position;
    const f32  R  = s.collider.sphere.radius;
    const f32  b  = dot(oc, r.direction);
    const f32  c  = dot(oc, oc) - R * R;
    const f32  disc = b * b - c;
    if (disc < 0.0f) return false;
    const f32 sq = std::sqrt(disc);
    const f32 t0 = -b - sq;
    const f32 t1 = -b + sq;
    const f32 t  = (t0 > 0.0f) ? t0 : ((t1 > 0.0f) ? t1 : -1.0f);
    if (t < 0.0f) return false;
    t_out = t;
    return true;
}
static bool ray_plane(const Ray& r, const Body& p, f32& t_out) {
    const Vec3 n = p.collider.plane.normal;
    const f32  denom = dot(n, r.direction);
    if (std::fabs(denom) < 1e-6f) return false;
    const f32 t = -(dot(n, r.origin) + p.collider.plane.d) / denom;
    if (t < 0.0f) return false;
    t_out = t;
    return true;
}
// Ray vs OBB — transform ray to box-local then run AABB slab test.
static bool ray_box(const Ray& r, const Body& b, f32& t_out) {
    const Mat3 Rt = quat_to_mat3(b.orientation).transposed();
    const Vec3 ro_local = Rt * (r.origin - b.position);
    const Vec3 rd_local = Rt * r.direction;
    const Vec3 he       = b.collider.box.half_extents;
    f32 tmin = -1e30f, tmax = 1e30f;
    for (int i = 0; i < 3; ++i) {
        const f32 ro = (&ro_local.x)[i];
        const f32 rd = (&rd_local.x)[i];
        const f32 lo = -(&he.x)[i];
        const f32 hi =  (&he.x)[i];
        if (std::fabs(rd) < 1e-6f) {
            if (ro < lo || ro > hi) return false;
        } else {
            const f32 inv = 1.0f / rd;
            f32 t1 = (lo - ro) * inv;
            f32 t2 = (hi - ro) * inv;
            if (t1 > t2) std::swap(t1, t2);
            if (t1 > tmin) tmin = t1;
            if (t2 < tmax) tmax = t2;
            if (tmin > tmax) return false;
        }
    }
    if (tmax < 0.0f) return false;
    t_out = (tmin > 0.0f) ? tmin : tmax;
    return true;
}
// Ray vs capsule — line-line distance test, accept when distance ≤ radius.
static bool ray_capsule(const Ray& r, const Body& cap, f32& t_out) {
    Vec3 a0, a1; capsule_segment(cap, a0, a1);
    // Intersection of ray with infinite cylinder around segment.
    const Vec3 ba = a1 - a0;
    const Vec3 oa = r.origin - a0;
    const f32  baba = dot(ba, ba);
    const f32  bard = dot(ba, r.direction);
    const f32  baoa = dot(ba, oa);
    const f32  rdoa = dot(r.direction, oa);
    const f32  oaoa = dot(oa, oa);
    const f32  R2   = cap.collider.capsule.radius * cap.collider.capsule.radius;
    const f32  a    = baba - bard * bard;
    const f32  b    = baba * rdoa - baoa * bard;
    const f32  c    = baba * oaoa - baoa * baoa - R2 * baba;
    const f32  h    = b * b - a * c;
    if (h < 0.0f) return false;
    const f32 t = (-b - std::sqrt(h)) / std::max(a, 1e-6f);
    const f32 y = baoa + t * bard;
    if (y > 0.0f && y < baba) {
        if (t < 0.0f) return false;
        t_out = t; return true;
    }
    // Hemisphere caps.
    const Vec3 oc = (y <= 0.0f) ? oa : (r.origin - a1);
    const f32 b2 = dot(oc, r.direction);
    const f32 c2 = dot(oc, oc) - R2;
    const f32 h2 = b2 * b2 - c2;
    if (h2 < 0.0f) return false;
    const f32 t2 = -b2 - std::sqrt(h2);
    if (t2 < 0.0f) return false;
    t_out = t2; return true;
}

// =============================================================================
// Joints
// =============================================================================
struct DistanceJoint {
    BodyHandle a, b;
    Vec3       la, lb;
    f32        rest;
    f32        stiffness;
    bool       break_on_overforce{false};
    f32        break_force{1e9f};
    bool       live{true};
};

// =============================================================================
// World implementation
// =============================================================================
class WorldImpl final : public World {
public:
    void set_gravity(Vec3 g) noexcept override { gravity_ = g; }
    Vec3 gravity() const noexcept    override { return gravity_; }
    void set_fixed_timestep(f32 dt) noexcept override {
        fixed_dt_ = std::max(1e-4f, dt);
    }
    f32  fixed_timestep() const noexcept    override { return fixed_dt_; }
    void set_solver_iterations(u32 n) noexcept override { solver_iters_ = std::max<u32>(1, n); }
    u32  solver_iterations() const noexcept    override { return solver_iters_; }

    BodyHandle create_body(const BodyDesc& desc) override {
        const auto h = pool_.allocate();
        if (h.index >= bodies_.size()) bodies_.resize(h.index + 1);
        Body& b      = bodies_[h.index];
        b.type       = desc.type;
        b.position   = desc.position;
        b.velocity   = desc.velocity;
        b.force_accum= {};
        b.collider   = desc.collider;
        b.material   = desc.material;
        b.gravity    = desc.gravity && desc.type == BodyType::Dynamic;
        b.layer      = desc.layer;
        b.mask       = desc.mask;
        b.live       = true;
        b.inv_mass   = (desc.type == BodyType::Dynamic && desc.mass > 0.0f)
                       ? (1.0f / desc.mass) : 0.0f;
        b.is_sensor  = desc.is_sensor;

        b.orientation       = normalize(desc.orientation);
        b.angular_velocity  = desc.angular_velocity;
        b.torque_accum      = {};
        b.linear_damping    = desc.linear_damping;
        b.angular_damping   = desc.angular_damping;
        b.can_sleep         = desc.can_sleep;
        b.linear_sleep_threshold  = desc.linear_sleep_threshold;
        b.angular_sleep_threshold = desc.angular_sleep_threshold;
        b.sleep_time_seconds      = desc.sleep_time_seconds;
        b.sleeping          = false;
        b.sleep_timer       = 0.0f;
        b.inv_inertia_local = (desc.type == BodyType::Dynamic && desc.mass > 0.0f)
                              ? compute_inv_inertia_local(desc.collider, desc.mass)
                              : Mat3::zero();
        refresh_world_inertia(b);
        return h;
    }

    bool destroy_body(BodyHandle h) override {
        if (!pool_.alive(h)) return false;
        bodies_[h.index].live = false;
        pool_.free(h);
        return true;
    }

    usize body_count() const noexcept override { return pool_.live_count(); }

    bool set_position (BodyHandle h, Vec3 p) override {
        if (!pool_.alive(h)) return false; bodies_[h.index].position = p;
        wake_body(bodies_[h.index]); return true;
    }
    bool set_velocity (BodyHandle h, Vec3 v) override {
        if (!pool_.alive(h)) return false; bodies_[h.index].velocity = v;
        wake_body(bodies_[h.index]); return true;
    }
    bool apply_impulse(BodyHandle h, Vec3 j) override {
        if (!pool_.alive(h)) return false;
        Body& b = bodies_[h.index];
        if (b.inv_mass <= 0.0f) return false;
        b.velocity += j * b.inv_mass;
        wake_body(b); return true;
    }
    bool apply_force(BodyHandle h, Vec3 f) override {
        if (!pool_.alive(h)) return false;
        bodies_[h.index].force_accum += f;
        wake_body(bodies_[h.index]); return true;
    }
    bool teleport(BodyHandle h, Vec3 p, Vec3 v) override {
        if (!pool_.alive(h)) return false;
        Body& b = bodies_[h.index]; b.position = p; b.velocity = v; b.force_accum = {};
        wake_body(b); return true;
    }

    bool set_orientation(BodyHandle h, Quat q) override {
        if (!pool_.alive(h)) return false; bodies_[h.index].orientation = normalize(q);
        wake_body(bodies_[h.index]); return true;
    }
    bool set_angular_velocity(BodyHandle h, Vec3 w) override {
        if (!pool_.alive(h)) return false; bodies_[h.index].angular_velocity = w;
        wake_body(bodies_[h.index]); return true;
    }
    bool apply_torque(BodyHandle h, Vec3 tau) override {
        if (!pool_.alive(h)) return false; bodies_[h.index].torque_accum += tau;
        wake_body(bodies_[h.index]); return true;
    }
    bool apply_angular_impulse(BodyHandle h, Vec3 j_ang) override {
        if (!pool_.alive(h)) return false;
        Body& b = bodies_[h.index];
        if (b.type != BodyType::Dynamic) return false;
        b.angular_velocity += b.inv_inertia_world * j_ang;
        wake_body(b); return true;
    }
    bool apply_force_at_point(BodyHandle h, Vec3 f, Vec3 p) override {
        if (!pool_.alive(h)) return false;
        Body& b = bodies_[h.index];
        if (b.type != BodyType::Dynamic) return false;
        b.force_accum  += f;
        b.torque_accum += cross(p - b.position, f);
        wake_body(b); return true;
    }

    bool is_sleeping(BodyHandle h) const override {
        return pool_.alive(h) && bodies_[h.index].sleeping;
    }
    bool wake(BodyHandle h) override {
        if (!pool_.alive(h)) return false; wake_body(bodies_[h.index]); return true;
    }
    bool sleep(BodyHandle h) override {
        if (!pool_.alive(h)) return false; bodies_[h.index].sleeping = true;
        bodies_[h.index].velocity = {}; bodies_[h.index].angular_velocity = {};
        return true;
    }

    Vec3 position(BodyHandle h) const override {
        return pool_.alive(h) ? bodies_[h.index].position : Vec3{};
    }
    Vec3 velocity(BodyHandle h) const override {
        return pool_.alive(h) ? bodies_[h.index].velocity : Vec3{};
    }
    Quat orientation(BodyHandle h) const override {
        return pool_.alive(h) ? bodies_[h.index].orientation : Quat::identity();
    }
    Vec3 angular_velocity(BodyHandle h) const override {
        return pool_.alive(h) ? bodies_[h.index].angular_velocity : Vec3{};
    }
    BodyType type(BodyHandle h) const override {
        return pool_.alive(h) ? bodies_[h.index].type : BodyType::Static;
    }
    AABB world_aabb(BodyHandle h) const override {
        return pool_.alive(h) ? body_aabb(bodies_[h.index]) : AABB{};
    }

    // -- Joints --
    JointHandle create_distance_joint(const DistanceJointDesc& d) override {
        const auto h = joint_pool_.allocate();
        if (h.index >= joints_.size()) joints_.resize(h.index + 1);
        DistanceJoint& j = joints_[h.index];
        j.a = d.a; j.b = d.b; j.la = d.local_anchor_a; j.lb = d.local_anchor_b;
        j.rest = d.rest_length; j.stiffness = clamp(d.stiffness, 0.0f, 1.0f);
        j.break_on_overforce = d.break_on_overforce;
        j.break_force = d.break_force;
        j.live = true;
        return h;
    }
    bool destroy_joint(JointHandle h) override {
        if (!joint_pool_.alive(h)) return false;
        joints_[h.index].live = false;
        joint_pool_.free(h);
        return true;
    }
    usize joint_count() const noexcept override { return joint_pool_.live_count(); }

    // -- Broad-phase config --
    void set_broadphase_cell_size(f32 cell) noexcept override { broad_cell_ = std::max(0.0f, cell); }
    f32  broadphase_cell_size() const noexcept override { return broad_cell_; }

    // -- Time scale (slow-mo / fast-forward / pause) --
    void set_time_scale(f32 s) noexcept override { time_scale_ = std::max(0.0f, s); }
    f32  time_scale() const noexcept override    { return time_scale_; }

    // -- Force fields --
    FieldHandle add_force_field(const ForceField& f) override {
        const auto h = field_pool_.allocate();
        if (h.index >= fields_.size()) fields_.resize(h.index + 1);
        FieldRecord& r = fields_[h.index];
        r.f    = f;
        r.live = true;
        return h;
    }
    bool remove_force_field(FieldHandle h) override {
        if (!field_pool_.alive(h)) return false;
        fields_[h.index].live = false;
        field_pool_.free(h);
        return true;
    }
    usize force_field_count() const noexcept override { return field_pool_.live_count(); }

    void step(f32 wall_dt) override {
        if (wall_dt <= 0.0f) return;
        // Apply the global time-scale knob — used for slow-mo, fast-forward,
        // pause without touching the renderer's wall-clock dt.
        wall_dt *= time_scale_;
        if (wall_dt <= 0.0f) return;
        if (wall_dt > 0.25f) wall_dt = 0.25f;
        accumulator_ += wall_dt;
        u32 sub = 0;
        while (accumulator_ >= fixed_dt_ && sub < kMaxSubsteps_) {
            substep(fixed_dt_);
            accumulator_ -= fixed_dt_;
            ++sub;
        }
    }

    const std::vector<ContactEvent>& last_contacts() const noexcept override {
        return contacts_;
    }

    RayHit raycast(const Ray& r, f32 max_distance, u32 layer_mask) const override {
        RayHit best{}; best.t = max_distance;
        for (u32 i = 0; i < bodies_.size(); ++i) {
            const Body& b = bodies_[i];
            if (!b.live) continue;
            if ((b.layer & layer_mask) == 0) continue;
            f32 t = 0.0f; bool hit = false;
            switch (b.collider.kind) {
                case ShapeKind::Sphere : hit = ray_sphere (r, b, t); break;
                case ShapeKind::Plane  : hit = ray_plane  (r, b, t); break;
                case ShapeKind::Box    : hit = ray_box    (r, b, t); break;
                case ShapeKind::Capsule: hit = ray_capsule(r, b, t); break;
            }
            if (!hit || t >= best.t) continue;
            best.hit  = true;
            best.t    = t;
            best.body = handle_for_index(i);
            best.point = Vec3{ r.origin.x + r.direction.x * t,
                               r.origin.y + r.direction.y * t,
                               r.origin.z + r.direction.z * t };
            best.normal = (b.collider.kind == ShapeKind::Plane)
                ? b.collider.plane.normal
                : normalize(best.point - b.position);
        }
        return best;
    }

    usize raycast_all(const Ray& r, f32 max_distance, u32 layer_mask,
                      const RaycastCallback& cb) const override {
        usize count = 0;
        for (u32 i = 0; i < bodies_.size(); ++i) {
            const Body& b = bodies_[i];
            if (!b.live) continue;
            if ((b.layer & layer_mask) == 0) continue;
            f32 t = 0.0f; bool hit = false;
            switch (b.collider.kind) {
                case ShapeKind::Sphere : hit = ray_sphere (r, b, t); break;
                case ShapeKind::Plane  : hit = ray_plane  (r, b, t); break;
                case ShapeKind::Box    : hit = ray_box    (r, b, t); break;
                case ShapeKind::Capsule: hit = ray_capsule(r, b, t); break;
            }
            if (!hit || t >= max_distance) continue;
            RayHit rh{};
            rh.hit   = true;
            rh.t     = t;
            rh.point = Vec3{ r.origin.x + r.direction.x * t,
                             r.origin.y + r.direction.y * t,
                             r.origin.z + r.direction.z * t };
            rh.normal = (b.collider.kind == ShapeKind::Plane)
                ? b.collider.plane.normal
                : normalize(rh.point - b.position);
            rh.body  = handle_for_index(i);
            ++count;
            if (cb && !cb(rh)) break;
        }
        return count;
    }

    RayHit sphere_cast(const Ray& r, f32 radius, f32 max_distance,
                       u32 layer_mask) const override {
        RayHit best{}; best.t = max_distance;
        if (radius <= 0.0f) return raycast(r, max_distance, layer_mask);
        for (u32 i = 0; i < bodies_.size(); ++i) {
            const Body& b = bodies_[i];
            if (!b.live) continue;
            if ((b.layer & layer_mask) == 0) continue;
            f32 t = 0.0f; bool hit = false;
            switch (b.collider.kind) {
                case ShapeKind::Sphere: {
                    Body inflated = b;
                    inflated.collider.sphere.radius = b.collider.sphere.radius + radius;
                    hit = ray_sphere(r, inflated, t);
                    break;
                }
                case ShapeKind::Plane: {
                    Body inflated = b;
                    inflated.collider.plane.d = b.collider.plane.d - radius;
                    hit = ray_plane(r, inflated, t);
                    break;
                }
                case ShapeKind::Box: {
                    Body inflated = b;
                    inflated.collider.box.half_extents += Vec3{ radius, radius, radius };
                    hit = ray_box(r, inflated, t);
                    break;
                }
                case ShapeKind::Capsule: {
                    Body inflated = b;
                    inflated.collider.capsule.radius = b.collider.capsule.radius + radius;
                    hit = ray_capsule(r, inflated, t);
                    break;
                }
            }
            if (!hit || t >= best.t) continue;
            best.hit   = true;
            best.t     = t;
            best.point = Vec3{ r.origin.x + r.direction.x * t,
                               r.origin.y + r.direction.y * t,
                               r.origin.z + r.direction.z * t };
            best.normal = (b.collider.kind == ShapeKind::Plane)
                ? b.collider.plane.normal
                : normalize(best.point - b.position);
            best.body = handle_for_index(i);
        }
        return best;
    }

    usize overlap_sphere(const Vec3& center, f32 radius, u32 mask,
                         const OverlapCallback& cb) const override
    { Body p{}; p.position = center; p.collider = Collider::make_sphere(radius); p.live = true;
      return overlap_with_probe(p, mask, cb); }
    usize overlap_aabb(const AABB& v, u32 mask, const OverlapCallback& cb) const override
    { Body p{}; p.position = v.center(); p.collider = Collider::make_box(v.extent()); p.live = true;
      return overlap_with_probe(p, mask, cb); }
    usize overlap_box(const Vec3& center, const Vec3& he, u32 mask,
                      const OverlapCallback& cb) const override
    { Body p{}; p.position = center; p.collider = Collider::make_box(he); p.live = true;
      return overlap_with_probe(p, mask, cb); }

    Vec3 closest_point_on_body(BodyHandle h, const Vec3& q) const override {
        if (!pool_.alive(h)) return q;
        const Body& b = bodies_[h.index];
        ShapeTransform st{ b.position };
        return closest_point_on_collider(b.collider, st, q);
    }

    void set_contact_callback(ContactCallback cb) override {
        contact_cb_ = std::move(cb);
    }

private:
    static constexpr u32 kMaxSubsteps_ = 8;

    BodyHandle handle_for_index(u32 idx) const noexcept {
        for (u32 gen = 1; gen < 0x80000u; ++gen) {
            BodyHandle h{ idx, gen };
            if (pool_.alive(h)) return h;
        }
        return BodyHandle{};
    }
    void wake_body(Body& b) noexcept { b.sleeping = false; b.sleep_timer = 0.0f; }

    usize overlap_with_probe(const Body& probe, u32 layer_mask,
                             const OverlapCallback& cb) const
    {
        const AABB pa = body_aabb(probe);
        usize count = 0;
        for (u32 i = 0; i < bodies_.size(); ++i) {
            const Body& B = bodies_[i];
            if (!B.live) continue;
            if ((B.layer & layer_mask) == 0) continue;
            if (!aabb_overlap(pa, body_aabb(B))) continue;
            ContactEvent dummy{};
            if (!narrow_test(probe, B, dummy)) continue;
            ++count;
            if (cb && !cb(handle_for_index(i))) break;
        }
        return count;
    }

    // Spatial-hash broad-phase: bucket bodies by AABB cell range, emit
    // unique pairs within each cell. cell_size <= 0 → fall back to N²
    // pair test (correct, just slower for large N).
    void broad_phase_pairs(std::vector<std::pair<u32, u32>>& out) {
        out.clear();
        if (broad_cell_ <= 0.0f) {
            for (u32 i = 0; i < bodies_.size(); ++i) {
                if (!bodies_[i].live) continue;
                const AABB ai = body_aabb(bodies_[i]);
                for (u32 j = i + 1; j < bodies_.size(); ++j) {
                    if (!bodies_[j].live) continue;
                    if (!aabb_overlap(ai, body_aabb(bodies_[j]))) continue;
                    out.push_back({ i, j });
                }
            }
            return;
        }
        // Hash cell coord (i32, i32, i32) → list of body indices.
        struct Cell { i32 x, y, z; };
        struct CellHash {
            usize operator()(const Cell& c) const noexcept {
                u64 h = static_cast<u32>(c.x) * 0x9E3779B97F4A7C15ull;
                h ^= static_cast<u32>(c.y) * 0xBF58476D1CE4E5B9ull;
                h ^= static_cast<u32>(c.z) * 0x94D049BB133111EBull;
                return static_cast<usize>(h ^ (h >> 32));
            }
        };
        struct CellEq {
            bool operator()(const Cell& a, const Cell& b) const noexcept {
                return a.x == b.x && a.y == b.y && a.z == b.z;
            }
        };
        std::unordered_map<Cell, std::vector<u32>, CellHash, CellEq> grid;
        const f32 inv = 1.0f / broad_cell_;
        for (u32 i = 0; i < bodies_.size(); ++i) {
            if (!bodies_[i].live) continue;
            const AABB a = body_aabb(bodies_[i]);
            // Plane bodies have huge AABB → skip insertion; pair them
            // against every dynamic separately.
            if (bodies_[i].collider.kind == ShapeKind::Plane) continue;
            const i32 x0 = static_cast<i32>(std::floor(a.min.x * inv));
            const i32 x1 = static_cast<i32>(std::floor(a.max.x * inv));
            const i32 y0 = static_cast<i32>(std::floor(a.min.y * inv));
            const i32 y1 = static_cast<i32>(std::floor(a.max.y * inv));
            const i32 z0 = static_cast<i32>(std::floor(a.min.z * inv));
            const i32 z1 = static_cast<i32>(std::floor(a.max.z * inv));
            for (i32 z = z0; z <= z1; ++z)
                for (i32 y = y0; y <= y1; ++y)
                    for (i32 x = x0; x <= x1; ++x)
                        grid[{x, y, z}].push_back(i);
        }
        std::unordered_set<u64> seen;
        seen.reserve(grid.size() * 4);
        auto pack = [](u32 a, u32 b) -> u64 {
            if (a > b) std::swap(a, b);
            return (static_cast<u64>(a) << 32) | b;
        };
        for (const auto& kv : grid) {
            const auto& v = kv.second;
            for (size_t i = 0; i < v.size(); ++i) {
                for (size_t j = i + 1; j < v.size(); ++j) {
                    const u32 ia = v[i], ib = v[j];
                    const u64 k = pack(ia, ib);
                    if (!seen.insert(k).second) continue;
                    if (!aabb_overlap(body_aabb(bodies_[ia]), body_aabb(bodies_[ib]))) continue;
                    out.push_back({ ia, ib });
                }
            }
        }
        // Pair every plane against every non-plane.
        for (u32 i = 0; i < bodies_.size(); ++i) {
            if (!bodies_[i].live) continue;
            if (bodies_[i].collider.kind != ShapeKind::Plane) continue;
            for (u32 j = 0; j < bodies_.size(); ++j) {
                if (j == i || !bodies_[j].live) continue;
                if (bodies_[j].collider.kind == ShapeKind::Plane) continue;
                out.push_back({ std::min(i, j), std::max(i, j) });
            }
        }
    }

    // Single fixed-dt physics tick.
    void substep(f32 dt) {
        contacts_.clear();

        // Refresh world inertia from latest orientation (cheap; ~8 mul/body).
        for (auto& b : bodies_) {
            if (!b.live || b.type != BodyType::Dynamic || b.sleeping) continue;
            refresh_world_inertia(b);
        }

        // 1. Integrate forces.
        // 1a. Apply user-defined force fields (gravity wells, drag zones,
        //     vortex, anti-gravity bubbles) BEFORE gravity / accumulator
        //     so the visible motion order is "fields → world gravity → user".
        apply_force_fields(dt);

        for (auto& b : bodies_) {
            if (!b.live || b.type != BodyType::Dynamic || b.sleeping) continue;
            if (b.gravity) b.velocity += gravity_ * dt;
            if (b.inv_mass > 0.0f) {
                b.velocity         += b.force_accum * (b.inv_mass * dt);
                b.angular_velocity += b.inv_inertia_world * b.torque_accum * dt;
            }
            b.force_accum  = {};
            b.torque_accum = {};
            // Damping (exponential per-step approx).
            if (b.linear_damping > 0.0f)
                b.velocity         *= std::max(0.0f, 1.0f - b.linear_damping  * dt);
            if (b.angular_damping > 0.0f)
                b.angular_velocity *= std::max(0.0f, 1.0f - b.angular_damping * dt);
        }

        // 2. Broad phase.
        broad_phase_pairs(broad_pairs_);

        // 3. Narrow phase.
        for (const auto& p : broad_pairs_) {
            const Body& A = bodies_[p.first];
            const Body& B = bodies_[p.second];
            if ((A.mask & B.layer) == 0 && (B.mask & A.layer) == 0) continue;
            if (A.type == BodyType::Static && B.type == BodyType::Static) continue;
            ContactEvent ev{};
            if (!narrow_test(A, B, ev)) continue;
            ev.a = handle_for_index(p.first);
            ev.b = handle_for_index(p.second);
            contacts_.push_back(ev);
            // Wake either side on contact (unless both are sensor-pairs).
            if (!(A.is_sensor && B.is_sensor)) {
                if (bodies_[p.first].sleeping || bodies_[p.second].sleeping) {
                    wake_body(bodies_[p.first]);
                    wake_body(bodies_[p.second]);
                }
            }
        }

        // 4. Solve — sequential impulses with full angular dynamics.
        for (u32 it = 0; it < solver_iters_; ++it) {
            for (auto& ev : contacts_) {
                Body& A = bodies_[ev.a.index];
                Body& B = bodies_[ev.b.index];
                if (A.is_sensor || B.is_sensor) continue;

                const Vec3 ra = ev.point - A.position;
                const Vec3 rb = ev.point - B.position;
                // Velocity at contact point.
                const Vec3 va = A.velocity + cross(A.angular_velocity, ra);
                const Vec3 vb = B.velocity + cross(B.angular_velocity, rb);
                const Vec3 rv = vb - va;
                const f32  vn = dot(rv, ev.normal);
                if (vn > 0.0f) continue;     // already separating

                // Effective mass along normal.
                const Vec3 ra_x_n = cross(ra, ev.normal);
                const Vec3 rb_x_n = cross(rb, ev.normal);
                const f32  inv_eff =
                    A.inv_mass + B.inv_mass
                    + dot(ra_x_n, A.inv_inertia_world * ra_x_n)
                    + dot(rb_x_n, B.inv_inertia_world * rb_x_n);
                if (inv_eff <= 0.0f) continue;

                const f32 e  = std::min(A.material.restitution, B.material.restitution);
                const f32 jn = -(1.0f + e) * vn / inv_eff;
                const Vec3 imp = ev.normal * jn;
                A.velocity         -= imp * A.inv_mass;
                A.angular_velocity -= A.inv_inertia_world * cross(ra, imp);
                B.velocity         += imp * B.inv_mass;
                B.angular_velocity += B.inv_inertia_world * cross(rb, imp);

                // Friction along the contact tangent.
                const Vec3 t_dir = rv - ev.normal * vn;
                const f32  t_len = length(t_dir);
                if (t_len > 1e-4f) {
                    const Vec3 t = t_dir * (1.0f / t_len);
                    const Vec3 ra_x_t = cross(ra, t);
                    const Vec3 rb_x_t = cross(rb, t);
                    const f32  inv_eff_t =
                        A.inv_mass + B.inv_mass
                        + dot(ra_x_t, A.inv_inertia_world * ra_x_t)
                        + dot(rb_x_t, B.inv_inertia_world * rb_x_t);
                    if (inv_eff_t > 0.0f) {
                        const f32 jt = -dot(rv, t) / inv_eff_t;
                        const f32 mu = std::sqrt(A.material.friction * B.material.friction);
                        const f32 jt_clamp = std::max(-jn * mu, std::min(jt, jn * mu));
                        const Vec3 ti = t * jt_clamp;
                        A.velocity         -= ti * A.inv_mass;
                        A.angular_velocity -= A.inv_inertia_world * cross(ra, ti);
                        B.velocity         += ti * B.inv_mass;
                        B.angular_velocity += B.inv_inertia_world * cross(rb, ti);
                    }
                }
            }

            // 4b. Distance-joint impulses (sequential w/ contacts).
            for (auto& j : joints_) {
                if (!j.live) continue;
                if (!pool_.alive(j.a) || !pool_.alive(j.b)) { j.live = false; continue; }
                Body& A = bodies_[j.a.index];
                Body& B = bodies_[j.b.index];
                const Vec3 pa = A.position + rotate(A.orientation, j.la);
                const Vec3 pb = B.position + rotate(B.orientation, j.lb);
                Vec3 dir = pb - pa;
                const f32 len = length(dir);
                if (len < 1e-6f) continue;
                dir = dir * (1.0f / len);
                const f32 c = len - j.rest;          // constraint error
                const Vec3 ra = pa - A.position;
                const Vec3 rb = pb - B.position;
                const Vec3 va = A.velocity + cross(A.angular_velocity, ra);
                const Vec3 vb = B.velocity + cross(B.angular_velocity, rb);
                const Vec3 rv = vb - va;
                const f32  vn = dot(rv, dir);
                const Vec3 ra_x_n = cross(ra, dir);
                const Vec3 rb_x_n = cross(rb, dir);
                const f32  inv_eff =
                    A.inv_mass + B.inv_mass
                    + dot(ra_x_n, A.inv_inertia_world * ra_x_n)
                    + dot(rb_x_n, B.inv_inertia_world * rb_x_n);
                if (inv_eff <= 0.0f) continue;
                // Bias from positional error → soft constraint via stiffness.
                const f32 bias  = -0.2f * c / dt;
                const f32 j_imp = j.stiffness * (-(vn + bias)) / inv_eff;
                const Vec3 imp  = dir * j_imp;
                if (j.break_on_overforce && std::fabs(j_imp / dt) > j.break_force) {
                    j.live = false;
                    continue;
                }
                A.velocity         -= imp * A.inv_mass;
                A.angular_velocity -= A.inv_inertia_world * cross(ra, imp);
                B.velocity         += imp * B.inv_mass;
                B.angular_velocity += B.inv_inertia_world * cross(rb, imp);
            }
        }

        // 4c. Position correction (Baumgarte).
        constexpr f32 kBaumgarte = 0.2f;
        constexpr f32 kSlop      = 0.005f;
        for (const auto& ev : contacts_) {
            Body& A = bodies_[ev.a.index];
            Body& B = bodies_[ev.b.index];
            if (A.is_sensor || B.is_sensor) continue;
            const f32 inv_sum = A.inv_mass + B.inv_mass;
            if (inv_sum <= 0.0f) continue;
            const f32 corr = std::max(ev.penetration - kSlop, 0.0f) / inv_sum * kBaumgarte;
            const Vec3 push = ev.normal * corr;
            A.position -= push * A.inv_mass;
            B.position += push * B.inv_mass;
        }

        // 5. Integrate motion + sleep tracking.
        // Position integration — split into a SIMD-friendly batch path
        // (axpy: position += velocity * dt) plus a scalar pass for the
        // branchy bookkeeping (snap-to-zero, sleep tracking, quaternion
        // orientation integrate). The SIMD path runs once over a flat
        // contiguous SoA stage of {x, y, z, x, y, z, ...} for every
        // live, awake, dynamic body — three floats per body. A 1000-body
        // world becomes one vec_axpy_f32 call over 3000 floats, which
        // the AVX-512 backend tears through in ~16 lanes per cycle.
        //
        // Below threshold (a few dozen bodies) the gather/scatter wash
        // costs more than they save — skip the SoA path and run the
        // legacy scalar loop. Threshold is conservative; benchmark on
        // real worlds and tune.
        constexpr usize kSimdIntegrateThreshold = 32;

        // Phase 1 — collect indices of bodies that participate in the
        // SIMD integration. Branchy filter so it stays scalar.
        static thread_local cardinal::UInt32Vec live_idx;
        live_idx.clear();
        live_idx.reserve(bodies_.size());
        for (u32 i = 0; i < bodies_.size(); ++i) {
            const Body& b = bodies_[i];
            if (!b.live) continue;
            if (b.type == BodyType::Static) continue;
            if (b.sleeping) continue;
            live_idx.push_back(i);
        }

        if (live_idx.size() >= kSimdIntegrateThreshold) {
            // Phase 2A — gather positions + velocities into SoA.
            static thread_local cardinal::FloatVec pos_soa, vel_soa;
            pos_soa.resize(live_idx.size() * 3);
            vel_soa.resize(live_idx.size() * 3);
            for (usize k = 0; k < live_idx.size(); ++k) {
                const Body& b = bodies_[live_idx[k]];
                pos_soa[k*3 + 0] = b.position.x;
                pos_soa[k*3 + 1] = b.position.y;
                pos_soa[k*3 + 2] = b.position.z;
                vel_soa[k*3 + 0] = b.velocity.x;
                vel_soa[k*3 + 1] = b.velocity.y;
                vel_soa[k*3 + 2] = b.velocity.z;
            }
            // Phase 2B — pos += vel * dt as one big vector op. Runtime-
            // dispatched: AVX-512 on Zen 5 / Sapphire Rapids, AVX2+FMA
            // on Skylake/Zen 2, AVX on Sandy Bridge, SSE4.2 on Westmere,
            // scalar on whatever else. Same call; CPU dictates which.
            cardinal::core::simd::vec_axpy_f32(
                pos_soa.data(), dt, vel_soa.data(),
                pos_soa.size());
            // Phase 2C — scatter positions back.
            for (usize k = 0; k < live_idx.size(); ++k) {
                Body& b = bodies_[live_idx[k]];
                b.position.x = pos_soa[k*3 + 0];
                b.position.y = pos_soa[k*3 + 1];
                b.position.z = pos_soa[k*3 + 2];
            }
        }

        // Phase 3 — branchy per-body work. Snap-to-zero, sleep tracking,
        // and quaternion integration aren't SIMD-friendly without per-
        // lane masking; the per-body cost here is dwarfed by the
        // collision phases above. When we took the SIMD path, position
        // was already updated — skip the `b.position += ...` line by
        // checking `simd_position_done`.
        const bool simd_position_done = live_idx.size() >= kSimdIntegrateThreshold;
        for (auto& b : bodies_) {
            if (!b.live) continue;
            if (b.type == BodyType::Static) continue;
            if (b.sleeping) continue;
            const f32 lv2 = dot(b.velocity, b.velocity);
            const f32 av2 = dot(b.angular_velocity, b.angular_velocity);
            if (lv2 < 1e-6f) b.velocity         = {};
            if (av2 < 1e-6f) b.angular_velocity = {};
            if (!simd_position_done) {
                b.position    += b.velocity * dt;
            }
            b.orientation  = integrate(b.orientation, b.angular_velocity, dt);

            if (b.can_sleep && b.type == BodyType::Dynamic) {
                const f32 lt2 = b.linear_sleep_threshold * b.linear_sleep_threshold;
                const f32 at2 = b.angular_sleep_threshold * b.angular_sleep_threshold;
                if (lv2 < lt2 && av2 < at2) {
                    b.sleep_timer += dt;
                    if (b.sleep_timer >= b.sleep_time_seconds) {
                        b.sleeping = true;
                        b.velocity = b.angular_velocity = {};
                    }
                } else {
                    b.sleep_timer = 0.0f;
                }
            }
        }

        // 6. Broadcast contacts.
        if (contact_cb_) for (const auto& ev : contacts_) contact_cb_(ev);
    }

    // Apply every active force field to dynamic bodies. Called from the
    // top of substep before gravity / accumulated forces integrate.
    void apply_force_fields(f32 dt) {
        if (field_pool_.live_count() == 0) return;
        for (auto& b : bodies_) {
            if (!b.live || b.type != BodyType::Dynamic || b.sleeping) continue;
            for (const auto& fr : fields_) {
                if (!fr.live) continue;
                if ((fr.f.layer_mask & b.layer) == 0
                 && (fr.f.layer_mask & b.mask) == 0) continue;
                switch (fr.f.kind) {
                    case FieldKind::UniformAccel: {
                        // F = m·a, but we treat strength·direction as
                        // acceleration directly (independent of mass).
                        b.velocity += fr.f.direction * fr.f.strength * dt;
                        break;
                    }
                    case FieldKind::Radial: {
                        const Vec3 d = fr.f.centre - b.position;
                        const f32 r2 = dot(d, d);
                        if (fr.f.radius > 0.0f && r2 > fr.f.radius * fr.f.radius) break;
                        const f32 r = std::sqrt(std::max(r2, 1e-6f));
                        const Vec3 n = d * (1.0f / r);
                        const f32 fall = (fr.f.falloff_exp <= 0.0f)
                            ? 1.0f
                            : std::pow(std::max(r, 1e-3f), -fr.f.falloff_exp);
                        b.velocity += n * (fr.f.strength * fall * dt);
                        break;
                    }
                    case FieldKind::Drag: {
                        if (fr.f.radius > 0.0f) {
                            const Vec3 d = fr.f.centre - b.position;
                            if (dot(d, d) > fr.f.radius * fr.f.radius) break;
                        }
                        b.velocity *= std::max(0.0f,
                                               1.0f - fr.f.strength * dt);
                        break;
                    }
                    case FieldKind::Vortex: {
                        const Vec3 d = fr.f.centre - b.position;
                        if (fr.f.radius > 0.0f
                            && dot(d, d) > fr.f.radius * fr.f.radius) break;
                        // Tangent direction = axis × radial.
                        const Vec3 t = cross(fr.f.direction, d);
                        const f32 tl = length(t);
                        if (tl < 1e-4f) break;
                        b.velocity += t * (fr.f.strength * dt / tl);
                        break;
                    }
                }
            }
        }
    }

    struct FieldRecord {
        ForceField f;
        bool       live{false};
    };

    Vec3                                gravity_{0.0f, -9.81f, 0.0f};
    f32                                 fixed_dt_{1.0f / 120.0f};
    f32                                 accumulator_{0.0f};
    u32                                 solver_iters_{4};
    f32                                 broad_cell_{0.0f};       // 0 = N² fallback
    f32                                 time_scale_{1.0f};
    cardinal::core::SlotPool<BodyTag>   pool_;
    std::vector<Body>                   bodies_;
    std::vector<ContactEvent>           contacts_;
    std::vector<std::pair<u32, u32>>    broad_pairs_;
    cardinal::core::SlotPool<JointTag>  joint_pool_;
    std::vector<DistanceJoint>          joints_;
    cardinal::core::SlotPool<FieldTag>  field_pool_;
    std::vector<FieldRecord>            fields_;
    ContactCallback                     contact_cb_{};
};

std::unique_ptr<World> World::create() {
    cardinal::log::infof("physics",
        "World online — gravity y=-9.81, fixed dt=1/120, 4 iters, "
        "OBB+capsule, sleeping, joints");
    return std::make_unique<WorldImpl>();
}

// =============================================================================
// Standalone shape queries — no World required.
// =============================================================================
namespace {
inline Body make_probe(const Collider& c, const Vec3& p) noexcept {
    Body b{};
    b.position = p;
    b.collider = c;
    b.live     = true;
    return b;
}
}  // namespace

bool test_overlap(const Collider& a, const ShapeTransform& at,
                  const Collider& b, const ShapeTransform& bt,
                  ContactEvent* out_contact) noexcept
{
    const Body A = make_probe(a, at.position);
    const Body B = make_probe(b, bt.position);
    if (!aabb_overlap(body_aabb(A), body_aabb(B))) return false;
    ContactEvent ev{};
    const bool hit = narrow_test(A, B, ev);
    if (hit && out_contact) *out_contact = ev;
    return hit;
}

Vec3 closest_point_on_collider(const Collider& c, const ShapeTransform& t,
                               const Vec3& q) noexcept
{
    switch (c.kind) {
        case ShapeKind::Sphere: {
            const Vec3 d   = q - t.position;
            const f32  d2  = dot(d, d);
            const f32  r   = c.sphere.radius;
            if (d2 <= r * r) return q;
            const f32 inv = 1.0f / std::sqrt(d2);
            return t.position + d * (r * inv);
        }
        case ShapeKind::Box: {
            const Vec3& he = c.box.half_extents;
            const Vec3 d   = q - t.position;
            const Vec3 cl  = { clamp(d.x, -he.x, he.x),
                               clamp(d.y, -he.y, he.y),
                               clamp(d.z, -he.z, he.z) };
            return t.position + cl;
        }
        case ShapeKind::Plane: {
            const Vec3 n   = c.plane.normal;
            const f32  s   = dot(n, q) + c.plane.d;
            return q - n * s;
        }
        case ShapeKind::Capsule: {
            // Segment endpoint wrt local Y; ShapeTransform doesn't carry rotation
            // so the segment is along world Y.
            const f32 hh = c.capsule.half_height;
            const Vec3 a0 = t.position + Vec3{0,  hh, 0};
            const Vec3 a1 = t.position - Vec3{0,  hh, 0};
            const Vec3 cp = closest_point_on_segment(a0, a1, q);
            const Vec3 d  = q - cp;
            const f32  l2 = dot(d, d);
            const f32  r  = c.capsule.radius;
            if (l2 <= r * r) return q;
            return cp + d * (r / std::sqrt(l2));
        }
    }
    return q;
}

f32 distance_to_collider(const Collider& c, const ShapeTransform& t,
                         const Vec3& q) noexcept
{
    const Vec3 cp = closest_point_on_collider(c, t, q);
    const Vec3 d  = q - cp;
    const f32  l2 = dot(d, d);
    return l2 > 0.0f ? std::sqrt(l2) : 0.0f;
}

AABB collider_world_aabb(const Collider& c, const ShapeTransform& t) noexcept {
    return body_aabb(make_probe(c, t.position));
}

}  // namespace cardinal::physics
