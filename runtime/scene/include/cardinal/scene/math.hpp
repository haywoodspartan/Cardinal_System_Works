#pragma once

// =============================================================================
// Cardinal Scene — math.
//
// As of the core-types unification, scene::Vec3 / Vec4 / Mat4 / Quat are
// `using` aliases of the canonical types in <cardinal/core/math.hpp>. This
// header exists for back-compat (a lot of code says scn::Vec3 / scn::Mat4)
// and to add scene-specific helpers (Ray, ray_*_intersect, unproject).
//
// Memory layout is identical to the core types — no conversions needed
// at module boundaries (physics::Vec3, world::Vec3, render uniform-buffer
// uploads all see the same { f32 x, y, z } layout).
// =============================================================================

#include <cardinal/core/math.hpp>
#include <cardinal/core/spatial.hpp>
#include <cardinal/core/types.hpp>

#include <cardinal/core/cmath.hpp>

namespace cardinal::scene {

// Re-export canonical math constants. `scn::kPi` etc. keep working.
constexpr float kPi       = 3.14159265358979323846f;
constexpr float kDegToRad = kPi / 180.0f;
constexpr float kRadToDeg = 180.0f / kPi;

// Canonical types — single definition lives in cardinal::core.
using Vec3 = cardinal::core::Vec3;
using Vec4 = cardinal::core::Vec4;
using Mat4 = cardinal::core::Mat4;
using Ray  = cardinal::core::Ray;

// Re-export the free vector functions so unqualified `dot(a,b)` still
// works inside namespace cardinal::scene (and ADL still finds them via
// the underlying type's namespace, but explicit re-export is friendlier).
using cardinal::core::dot;
using cardinal::core::cross;
using cardinal::core::length;
using cardinal::core::normalize;
using cardinal::core::lerp;

// =============================================================================
// Scene-specific spatial helpers — viewport picking + Y-plane intersect.
// =============================================================================

// Construct a world-space ray from a normalised-device coordinate
// (ndc in [-1, +1] on each axis; +Y is up).
inline Ray unproject_ndc_ray(float ndc_x, float ndc_y,
                             const Mat4& view, const Mat4& proj)
{
    const Mat4 inv_vp = (proj * view).inverse();
    const Vec4 near_clip{ ndc_x, ndc_y, 0.0f, 1.0f };
    const Vec4 far_clip { ndc_x, ndc_y, 1.0f, 1.0f };
    const Vec4 nw       = inv_vp * near_clip;
    const Vec4 fw       = inv_vp * far_clip;
    Vec3 n{ nw.x / nw.w, nw.y / nw.w, nw.z / nw.w };
    Vec3 f{ fw.x / fw.w, fw.y / fw.w, fw.z / fw.w };
    return Ray{ n, normalize(f - n) };
}

// Intersect ray with a sphere at `center` with radius `r`. Returns true
// and writes the smallest positive `t` when there's an intersection in
// front of the ray's origin.
inline bool ray_sphere_intersect(const Ray& r, const Vec3& center, float radius,
                                 float* out_t = nullptr) noexcept
{
    const Vec3 oc = r.origin - center;
    const float b = dot(oc, r.direction);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = b * b - c;
    if (discriminant < 0.0f) return false;
    const float s = cardinal::sqrt(discriminant);
    const float t0 = -b - s;
    const float t1 = -b + s;
    const float t = (t0 > 0.0f) ? t0 : ((t1 > 0.0f) ? t1 : -1.0f);
    if (t < 0.0f) return false;
    if (out_t) *out_t = t;
    return true;
}

// Intersect ray with the world Y = `plane_y` plane. Returns true and writes
// the world-space hit point when the ray is heading toward the plane.
inline bool ray_plane_y_intersect(const Ray& r, float plane_y,
                                  Vec3* out_hit = nullptr) noexcept
{
    if (cardinal::abs(r.direction.y) < 1e-6f) return false;
    const float t = (plane_y - r.origin.y) / r.direction.y;
    if (t <= 0.0f) return false;
    if (out_hit) *out_hit = r.at(t);
    return true;
}

}  // namespace cardinal::scene
