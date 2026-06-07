#pragma once

// =============================================================================
// Cardinal Physics — exotic helpers.
//
// "Bend physics inside the engine" toolkit. Three sub-namespaces cover the
// common shapes:
//
//   physics::photon  — electromagnetic / photon math (E↔ν↔λ↔p)
//   physics::spatial — pure geometric helpers (volumes, areas, slerp,
//                      golden-spiral sphere sampling, reflect / refract)
//   physics::time    — time + relativity helpers (Lorentz γ, dilation,
//                      length contraction, relativistic velocity addition,
//                      and a per-world `TimeScale` knob for slow-mo /
//                      fast-forward without touching wall-clock dt)
//
// All physical formulas use SI base units (metres, seconds, kg, joules).
// Game code working in centimetres / arbitrary units rescales before
// consuming. Units are stamped in every function name + comment.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/cmath.hpp>   // cardinal scalar math
#include <cardinal/physics/physics.hpp>

namespace cardinal::physics {

// =============================================================================
// physics::photon — electromagnetic / photon energy + momentum
// =============================================================================
namespace photon {

// E = h·ν  (joules from hertz).
inline f64 energy_from_frequency(f64 frequency_hz) noexcept {
    return cardinal::constants::planck_const * frequency_hz;
}
// E = h·c / λ  (joules from metres).
inline f64 energy_from_wavelength(f64 wavelength_m) noexcept {
    return cardinal::constants::planck_const *
           cardinal::constants::light_speed_mps / wavelength_m;
}
// p = h / λ  (kg·m/s from metres).
inline f64 momentum_from_wavelength(f64 wavelength_m) noexcept {
    return cardinal::constants::planck_const / wavelength_m;
}
// λ = c / ν.
inline f64 wavelength_from_frequency(f64 frequency_hz) noexcept {
    return cardinal::constants::light_speed_mps / frequency_hz;
}
// ν = c / λ.
inline f64 frequency_from_wavelength(f64 wavelength_m) noexcept {
    return cardinal::constants::light_speed_mps / wavelength_m;
}
// Convert energy in electron-volts to / from joules. Useful when reading
// values from physics datasheets (which usually report eV).
inline f64 ev_to_joules(f64 ev) noexcept { return ev * cardinal::constants::elementary_charge_c; }
inline f64 joules_to_ev(f64 j ) noexcept { return j  / cardinal::constants::elementary_charge_c; }

// Visible-spectrum constants (nm) — useful for shader / VFX colour ramps.
inline constexpr f64 visible_min_nm = 380.0;
inline constexpr f64 visible_max_nm = 750.0;

// Approximate sRGB colour for a wavelength (nm). Returns linear-space
// RGB in [0, 1]. Implementation uses the standard CIE 1931 piecewise
// approximation popular in graphics demos.
Vec3 wavelength_to_rgb(f64 wavelength_nm) noexcept;

}  // namespace photon

// =============================================================================
// physics::spatial — pure geometric helpers
// =============================================================================
namespace spatial {

// ---- Closed-form volumes + surface areas ---------------------------------
inline f64 sphere_volume(f64 r) noexcept {
    return (4.0 / 3.0) * cardinal::constants::pi * r * r * r;
}
inline f64 sphere_area(f64 r) noexcept {
    return 4.0 * cardinal::constants::pi * r * r;
}
inline f64 box_volume(f64 x, f64 y, f64 z) noexcept { return x * y * z; }
inline f64 box_area  (f64 x, f64 y, f64 z) noexcept { return 2.0 * (x*y + y*z + z*x); }
inline f64 cylinder_volume(f64 r, f64 height) noexcept {
    return cardinal::constants::pi * r * r * height;
}
inline f64 cylinder_area(f64 r, f64 height) noexcept {
    return 2.0 * cardinal::constants::pi * r * (r + height);
}
inline f64 capsule_volume(f64 r, f64 cylinder_height) noexcept {
    // Cylinder middle + 2 hemispheres = cylinder + sphere.
    return cylinder_volume(r, cylinder_height) + sphere_volume(r);
}
inline f64 capsule_area(f64 r, f64 cylinder_height) noexcept {
    return 2.0 * cardinal::constants::pi * r * cylinder_height + sphere_area(r);
}
inline f64 cone_volume(f64 r, f64 height) noexcept {
    return cardinal::constants::pi * r * r * height / 3.0;
}
inline f64 torus_volume(f64 major_r, f64 minor_r) noexcept {
    return 2.0 * cardinal::constants::pi * cardinal::constants::pi *
           major_r * minor_r * minor_r;
}

// ---- Vec3 helpers (mirror physics::Vec3) ---------------------------------
inline f32 distance(const Vec3& a, const Vec3& b) noexcept {
    return length(b - a);
}
inline f32 distance_sq(const Vec3& a, const Vec3& b) noexcept {
    const Vec3 d = b - a; return dot(d, d);
}
inline Vec3 midpoint(const Vec3& a, const Vec3& b) noexcept {
    return Vec3{ (a.x+b.x) * 0.5f, (a.y+b.y) * 0.5f, (a.z+b.z) * 0.5f };
}
inline Vec3 lerp(const Vec3& a, const Vec3& b, f32 t) noexcept {
    return a + (b - a) * t;
}

// Spherical lerp between two unit vectors (returns unit vector).
Vec3 slerp(const Vec3& a, const Vec3& b, f32 t) noexcept;

// Reflect / refract — same semantics as GLSL `reflect`, `refract`. Both
// `n` and the result are world-space directions; `eta` is η₁/η₂ for
// refraction (1.0 / 1.33 for air → water etc).
Vec3 reflect(const Vec3& v, const Vec3& n) noexcept;
Vec3 refract(const Vec3& v, const Vec3& n, f32 eta) noexcept;

// Phyllotaxis sphere sampling — returns the i-th of N points evenly
// distributed on a unit sphere via the golden-angle spiral. Used for
// uniform sampling without explicit Monte Carlo (debris distribution,
// AI directional sampling, environment probes).
Vec3 golden_spiral_point(u32 i, u32 N) noexcept;

// Same idea on a 2-D disc — i-th of N points on a unit disc.
Vec3 golden_disc_point(u32 i, u32 N) noexcept;   // returns (x, 0, z)

}  // namespace spatial

// =============================================================================
// physics::time — time + relativistic helpers + per-world time scale
// =============================================================================
namespace time {

// Lorentz factor γ = 1 / √(1 − v²/c²). Returns +∞ at v = c.
f64 lorentz_factor(f64 velocity_mps) noexcept;
// Dilated time as observed in the rest frame for a moving clock with
// proper time `proper_dt` and velocity `v`.   dt_observed = γ · dt_proper
inline f64 dilated_time(f64 proper_dt, f64 velocity_mps) noexcept {
    return proper_dt * lorentz_factor(velocity_mps);
}
// Length contraction: a rod with rest length L appears L / γ when moving.
inline f64 contracted_length(f64 proper_length, f64 velocity_mps) noexcept {
    return proper_length / lorentz_factor(velocity_mps);
}
// Relativistic velocity addition (1-D, same direction):
//   u' = (u + v) / (1 + u·v/c²)
f64 add_velocities(f64 u_mps, f64 v_mps) noexcept;
// Mass-energy equivalence:  E₀ = m·c²  (rest energy in joules from kg).
inline f64 rest_energy(f64 mass_kg) noexcept {
    return mass_kg * cardinal::constants::light_speed_mps
                   * cardinal::constants::light_speed_mps;
}
// Total relativistic energy:  E = γ·m·c².
inline f64 total_energy(f64 mass_kg, f64 velocity_mps) noexcept {
    return lorentz_factor(velocity_mps) * rest_energy(mass_kg);
}

// Frequency ↔ period.
inline f64 period_from_frequency(f64 hz)  noexcept { return 1.0 / hz; }
inline f64 frequency_from_period(f64 sec) noexcept { return 1.0 / sec; }

// Doppler shift for a light source approaching/receding observer:
//   ν_observed = ν_source · √((1 - β) / (1 + β))   (recession positive β)
f64 relativistic_doppler(f64 source_frequency_hz, f64 receding_velocity_mps) noexcept;

// =============================================================================
// TimeScale — per-world simulation-time multiplier.
//
// The physics::World::step takes a wall-clock dt; gameplay wraps dt with
// `TimeScale::effective_dt(wall_dt)` to apply slow-mo / fast-forward / pause
// without changing the renderer or input timing.
// =============================================================================
struct TimeScale {
    f64  scale{1.0};      // 0.5 = half-speed sim, 2.0 = double, etc.
    bool paused{false};
    f64 effective_dt(f64 wall_dt) const noexcept {
        return paused ? 0.0 : (wall_dt * scale);
    }
};

}  // namespace time
}  // namespace cardinal::physics
