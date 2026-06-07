#pragma once

// =============================================================================
// cardinal::core::curve — interpolation, easing, and spline evaluation.
//
// Animation, camera rigs, tweening, and the sequencer/curve editor all need
// the same scalar math that core only had piecemeal (a Vec3 lerp, a clamp):
// remap/smoothstep, the Penner easing family (quad…bounce, in/out/in-out),
// and through-point spline evaluation (Catmull-Rom) + control-point curves
// (cubic Bezier, Hermite). All pure, constexpr-where-possible, header-only.
//
// Easing functions take a normalised t in [0,1] and return the eased value,
// with f(0)=0 and f(1)=1 (so they compose with lerp: a + (b-a)*ease(t)).
//
// FOUNDATION RULE: lives in cardinal::core. Reachable as cardinal::curve::*.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/cmath.hpp>     // cardinal::sin/cos/pow/sqrt/abs
#include <cardinal/core/math/math.hpp> // Vec3 (spline overloads)

namespace cardinal::core::curve {

inline constexpr cardinal::f32 kPi = 3.14159265358979323846f;

// ---- interpolation helpers ----------------------------------------------
inline constexpr cardinal::f32 lerp(cardinal::f32 a, cardinal::f32 b, cardinal::f32 t) noexcept {
    return a + (b - a) * t;
}
inline constexpr cardinal::f32 inverse_lerp(cardinal::f32 a, cardinal::f32 b, cardinal::f32 v) noexcept {
    return (b != a) ? (v - a) / (b - a) : 0.0f;
}
inline constexpr cardinal::f32 remap(cardinal::f32 v, cardinal::f32 in_lo, cardinal::f32 in_hi,
                                     cardinal::f32 out_lo, cardinal::f32 out_hi) noexcept {
    return lerp(out_lo, out_hi, inverse_lerp(in_lo, in_hi, v));
}
inline constexpr cardinal::f32 saturate(cardinal::f32 v) noexcept {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}
inline constexpr cardinal::f32 smoothstep(cardinal::f32 t) noexcept {
    t = saturate(t);
    return t * t * (3.0f - 2.0f * t);
}
inline constexpr cardinal::f32 smootherstep(cardinal::f32 t) noexcept {
    t = saturate(t);
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

// ---- Penner easing family (t in [0,1]; f(0)=0, f(1)=1) ------------------
inline constexpr cardinal::f32 in_quad(cardinal::f32 t)  noexcept { return t * t; }
inline constexpr cardinal::f32 out_quad(cardinal::f32 t) noexcept { return t * (2.0f - t); }
inline constexpr cardinal::f32 in_out_quad(cardinal::f32 t) noexcept {
    return t < 0.5f ? 2.0f * t * t : 1.0f - (-2.0f * t + 2.0f) * (-2.0f * t + 2.0f) * 0.5f;
}
inline constexpr cardinal::f32 in_cubic(cardinal::f32 t)  noexcept { return t * t * t; }
inline constexpr cardinal::f32 out_cubic(cardinal::f32 t) noexcept { const cardinal::f32 u = 1.0f - t; return 1.0f - u * u * u; }
inline constexpr cardinal::f32 in_out_cubic(cardinal::f32 t) noexcept {
    if (t < 0.5f) return 4.0f * t * t * t;
    const cardinal::f32 u = -2.0f * t + 2.0f;
    return 1.0f - u * u * u * 0.5f;
}
inline cardinal::f32 in_sine(cardinal::f32 t)  noexcept { return 1.0f - cardinal::cos(t * kPi * 0.5f); }
inline cardinal::f32 out_sine(cardinal::f32 t) noexcept { return cardinal::sin(t * kPi * 0.5f); }
inline cardinal::f32 in_out_sine(cardinal::f32 t) noexcept { return -(cardinal::cos(kPi * t) - 1.0f) * 0.5f; }
inline cardinal::f32 in_expo(cardinal::f32 t)  noexcept { return t <= 0.0f ? 0.0f : cardinal::pow(2.0f, 10.0f * t - 10.0f); }
inline cardinal::f32 out_expo(cardinal::f32 t) noexcept { return t >= 1.0f ? 1.0f : 1.0f - cardinal::pow(2.0f, -10.0f * t); }
inline cardinal::f32 in_out_expo(cardinal::f32 t) noexcept {
    if (t <= 0.0f) return 0.0f;
    if (t >= 1.0f) return 1.0f;
    return t < 0.5f ? cardinal::pow(2.0f, 20.0f * t - 10.0f) * 0.5f
                    : (2.0f - cardinal::pow(2.0f, -20.0f * t + 10.0f)) * 0.5f;
}
inline constexpr cardinal::f32 in_back(cardinal::f32 t) noexcept {
    constexpr cardinal::f32 c1 = 1.70158f, c3 = c1 + 1.0f;
    return c3 * t * t * t - c1 * t * t;
}
inline constexpr cardinal::f32 out_back(cardinal::f32 t) noexcept {
    constexpr cardinal::f32 c1 = 1.70158f, c3 = c1 + 1.0f;
    const cardinal::f32 u = t - 1.0f;
    return 1.0f + c3 * u * u * u + c1 * u * u;
}
inline cardinal::f32 out_bounce(cardinal::f32 t) noexcept {
    constexpr cardinal::f32 n1 = 7.5625f, d1 = 2.75f;
    if (t < 1.0f / d1)        return n1 * t * t;
    else if (t < 2.0f / d1) { t -= 1.5f / d1;  return n1 * t * t + 0.75f; }
    else if (t < 2.5f / d1) { t -= 2.25f / d1; return n1 * t * t + 0.9375f; }
    else                    { t -= 2.625f / d1; return n1 * t * t + 0.984375f; }
}
inline cardinal::f32 in_bounce(cardinal::f32 t) noexcept { return 1.0f - out_bounce(1.0f - t); }
inline cardinal::f32 in_out_bounce(cardinal::f32 t) noexcept {
    return t < 0.5f ? (1.0f - out_bounce(1.0f - 2.0f * t)) * 0.5f
                    : (1.0f + out_bounce(2.0f * t - 1.0f)) * 0.5f;
}

// ---- spline evaluation --------------------------------------------------
// Catmull-Rom: a C1 curve that passes THROUGH p1 (t=0) and p2 (t=1); p0/p3 are
// the neighbouring control points that set the tangents.
inline constexpr cardinal::f32 catmull_rom(cardinal::f32 p0, cardinal::f32 p1,
                                           cardinal::f32 p2, cardinal::f32 p3, cardinal::f32 t) noexcept {
    const cardinal::f32 t2 = t * t, t3 = t2 * t;
    return 0.5f * ((2.0f * p1) +
                   (-p0 + p2) * t +
                   (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
                   (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
}
inline Vec3 catmull_rom(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3,
                        cardinal::f32 t) noexcept {
    return { catmull_rom(p0.x, p1.x, p2.x, p3.x, t),
             catmull_rom(p0.y, p1.y, p2.y, p3.y, t),
             catmull_rom(p0.z, p1.z, p2.z, p3.z, t) };
}
// Cubic Bezier through p0 (t=0) and p3 (t=1); p1/p2 are the control handles.
inline constexpr cardinal::f32 bezier(cardinal::f32 p0, cardinal::f32 p1,
                                      cardinal::f32 p2, cardinal::f32 p3, cardinal::f32 t) noexcept {
    const cardinal::f32 u = 1.0f - t, u2 = u * u, u3 = u2 * u, t2 = t * t, t3 = t2 * t;
    return u3 * p0 + 3.0f * u2 * t * p1 + 3.0f * u * t2 * p2 + t3 * p3;
}
inline Vec3 bezier(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3,
                   cardinal::f32 t) noexcept {
    return { bezier(p0.x, p1.x, p2.x, p3.x, t),
             bezier(p0.y, p1.y, p2.y, p3.y, t),
             bezier(p0.z, p1.z, p2.z, p3.z, t) };
}
// Cubic Hermite: endpoints p0 (t=0), p1 (t=1) with tangents m0, m1.
inline constexpr cardinal::f32 hermite(cardinal::f32 p0, cardinal::f32 m0,
                                       cardinal::f32 p1, cardinal::f32 m1, cardinal::f32 t) noexcept {
    const cardinal::f32 t2 = t * t, t3 = t2 * t;
    return (2.0f * t3 - 3.0f * t2 + 1.0f) * p0 + (t3 - 2.0f * t2 + t) * m0 +
           (-2.0f * t3 + 3.0f * t2) * p1 + (t3 - t2) * m1;
}

}  // namespace cardinal::core::curve

namespace cardinal { namespace curve = core::curve; }
