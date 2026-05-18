// =============================================================================
// Cardinal Physics — exotic helpers implementation.
// =============================================================================
#include <cardinal/physics/exotic.hpp>

#include <cardinal/core/algorithm.hpp>   // cardinal::min/max/clamp
#include <cardinal/core/cmath.hpp>       // cardinal scalar math
#include <cardinal/core/limits.hpp>      // cardinal::numeric_limits

namespace cardinal::physics {

// =============================================================================
// physics::photon
// =============================================================================
namespace photon {

// Approximate visible-wavelength → linear sRGB conversion. Standard
// piecewise-linear fit popular in graphics demos (sufficient for VFX
// colour ramps; not photometrically rigorous).
Vec3 wavelength_to_rgb(f64 wl_nm) noexcept {
    if (wl_nm < 380.0 || wl_nm > 750.0) return Vec3{0, 0, 0};

    f64 r = 0, g = 0, b = 0;
    if      (wl_nm < 440.0) { r = -(wl_nm - 440.0) / (440.0 - 380.0); b = 1.0; }
    else if (wl_nm < 490.0) { g =  (wl_nm - 440.0) / (490.0 - 440.0); b = 1.0; }
    else if (wl_nm < 510.0) { g =  1.0; b = -(wl_nm - 510.0) / (510.0 - 490.0); }
    else if (wl_nm < 580.0) { r =  (wl_nm - 510.0) / (580.0 - 510.0); g = 1.0; }
    else if (wl_nm < 645.0) { r =  1.0; g = -(wl_nm - 645.0) / (645.0 - 580.0); }
    else                    { r =  1.0; }

    // Falloff at the edges of the visible spectrum.
    f64 fall = 1.0;
    if      (wl_nm < 420.0) fall = 0.3 + 0.7 * (wl_nm - 380.0) / (420.0 - 380.0);
    else if (wl_nm > 700.0) fall = 0.3 + 0.7 * (750.0 - wl_nm) / (750.0 - 700.0);

    return Vec3{ static_cast<f32>(r * fall),
                 static_cast<f32>(g * fall),
                 static_cast<f32>(b * fall) };
}

}  // namespace photon

// =============================================================================
// physics::spatial
// =============================================================================
namespace spatial {

Vec3 slerp(const Vec3& a, const Vec3& b, f32 t) noexcept {
    f32 d = dot(a, b);
    d = cardinal::max(-1.0f, cardinal::min(1.0f, d));
    const f32 theta = cardinal::acos(d);
    // Qualify to disambiguate from cardinal::core::lerp (visible via ADL).
    if (theta < 1e-6f) return cardinal::physics::spatial::lerp(a, b, t);
    const f32 inv_sin = 1.0f / cardinal::sin(theta);
    const f32 s_a = cardinal::sin((1.0f - t) * theta) * inv_sin;
    const f32 s_b = cardinal::sin(t * theta) * inv_sin;
    return Vec3{ a.x * s_a + b.x * s_b,
                 a.y * s_a + b.y * s_b,
                 a.z * s_a + b.z * s_b };
}

Vec3 reflect(const Vec3& v, const Vec3& n) noexcept {
    // r = v - 2 (v·n) n
    const f32 d = dot(v, n);
    return Vec3{ v.x - 2.0f * d * n.x,
                 v.y - 2.0f * d * n.y,
                 v.z - 2.0f * d * n.z };
}

Vec3 refract(const Vec3& v, const Vec3& n, f32 eta) noexcept {
    const f32 d  = dot(v, n);
    const f32 k  = 1.0f - eta * eta * (1.0f - d * d);
    if (k < 0.0f) return Vec3{0, 0, 0};   // total internal reflection
    const f32 c  = eta * d + cardinal::sqrt(k);
    return Vec3{ eta * v.x - c * n.x,
                 eta * v.y - c * n.y,
                 eta * v.z - c * n.z };
}

Vec3 golden_spiral_point(u32 i, u32 N) noexcept {
    if (N == 0) return Vec3{0, 1, 0};
    const f64 ga    = cardinal::constants::golden_angle_rad;
    const f64 phi   = cardinal::acos(1.0 - 2.0 * (static_cast<f64>(i) + 0.5)
                                          / static_cast<f64>(N));
    const f64 theta = ga * static_cast<f64>(i);
    const f64 sp    = cardinal::sin(phi);
    return Vec3{ static_cast<f32>(cardinal::cos(theta) * sp),
                 static_cast<f32>(cardinal::cos(phi)),
                 static_cast<f32>(cardinal::sin(theta) * sp) };
}

Vec3 golden_disc_point(u32 i, u32 N) noexcept {
    if (N == 0) return Vec3{0, 0, 0};
    const f64 ga    = cardinal::constants::golden_angle_rad;
    const f64 r     = cardinal::sqrt((static_cast<f64>(i) + 0.5)
                                / static_cast<f64>(N));
    const f64 theta = ga * static_cast<f64>(i);
    return Vec3{ static_cast<f32>(r * cardinal::cos(theta)),
                 0.0f,
                 static_cast<f32>(r * cardinal::sin(theta)) };
}

}  // namespace spatial

// =============================================================================
// physics::time + relativity
// =============================================================================
namespace time {

f64 lorentz_factor(f64 v_mps) noexcept {
    const f64 c   = cardinal::constants::light_speed_mps;
    const f64 b2  = (v_mps * v_mps) / (c * c);
    if (b2 >= 1.0) return cardinal::numeric_limits<f64>::infinity();
    return 1.0 / cardinal::sqrt(1.0 - b2);
}

f64 add_velocities(f64 u, f64 v) noexcept {
    const f64 c2 = cardinal::constants::light_speed_mps
                 * cardinal::constants::light_speed_mps;
    return (u + v) / (1.0 + (u * v) / c2);
}

f64 relativistic_doppler(f64 src_hz, f64 v_recede) noexcept {
    const f64 beta = v_recede / cardinal::constants::light_speed_mps;
    if (beta <= -1.0 || beta >= 1.0) return 0.0;
    return src_hz * cardinal::sqrt((1.0 - beta) / (1.0 + beta));
}

}  // namespace time

}  // namespace cardinal::physics
