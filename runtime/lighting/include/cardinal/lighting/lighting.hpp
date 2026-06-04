#pragma once

// =============================================================================
// Cardinal — Lighting module.
//
// Two complementary lighting paths share this module's primitives:
//
//   1. raytracer.hpp  — RTX-style CPU path tracer (primary + direct + N-bounce
//                       indirect with PBR Cook-Torrance BRDF). Reference for
//                       the eventual RT-RHI shader path; ships with deterministic
//                       seeded RNG so output is regression-pinnable.
//
//   2. bake.hpp       — Offline lightmap + irradiance-probe-volume baker, both
//                       integrated by the path tracer above. Output buffers are
//                       sampled at runtime (bilinear for lightmaps, trilinear
//                       for probes) with no further raycasting.
//
// This header just declares the shared building blocks: PBR Material, the
// CPU-reference scene primitives (Sphere / Plane / TriBVHGeometryStub), and
// the aggregating Scene that the tracers read.
//
// Design rules (consistent with postfx + render::algo + particles):
//   * Knob-driven editor introspection via render::Knob.
//   * Every public surface NaN-defended at every read site (fz helper).
//   * CPU reference first; HLSL counterpart lands when RHI raytracing is wired.
//   * No filesystem / no allocation in tight inner loops (bakers reserve up
//     front; runtime samplers are pure functions of the cached buffers).
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/containers.hpp>
#include <cardinal/scene/math.hpp>
#include <cardinal/scene/light.hpp>

namespace cardinal::lighting {

using scene::Vec3;
using scene::Ray;

// ---------------------------------------------------------------------------
// Material — minimal PBR description. base_color is sRGB-linear (renderer
// pipelines tonemap on the way out; bake output is also linear). Metallic ∈
// [0,1] picks between dielectric F0=0.04 and base_color-tinted F0. Roughness
// is perceptual; squared internally for GGX (so the editor slider feels
// linear). Emissive is added pre-BRDF — it lights other surfaces via the
// indirect path.
// ---------------------------------------------------------------------------
struct Material {
    Vec3  base_color{0.8f, 0.8f, 0.8f};
    Vec3  emissive  {0.0f, 0.0f, 0.0f};
    float metallic  {0.0f};
    float roughness {0.5f};
};

// ---------------------------------------------------------------------------
// CPU-reference scene primitives. The RT-RHI path will trace into the real
// BVH built from scene::Mesh; until then the tracer reads these explicit
// primitives so the reference output is fully deterministic and the bake
// pipeline has something to integrate.
// ---------------------------------------------------------------------------
struct Sphere {
    Vec3  center {0, 0, 0};
    float radius {1.0f};
    u32   material_idx{0};
};

// Infinite plane (point + normal). For the floor / walls in the bake test
// scenes. n must be unit; the tracer renormalizes defensively on read.
struct Plane {
    Vec3 point  {0, 0, 0};
    Vec3 normal {0, 1, 0};
    u32  material_idx{0};
};

// ---------------------------------------------------------------------------
// Scene — aggregated geometry + material palette + reference to the
// host's scene::LightSet (this module never owns lights; it borrows the
// scene's set). Hosts hand a Scene to the path tracer or to the bakers.
// ---------------------------------------------------------------------------
struct Scene {
    cardinal::vector<Sphere>   spheres;
    cardinal::vector<Plane>    planes;
    cardinal::vector<Material> materials;
    const scene::LightSet*     lights{nullptr};   // borrowed (must outlive)
};

// ---------------------------------------------------------------------------
// RayHit — populated by Scene intersection. material_idx is an index into
// Scene::materials, or kInvalidMaterial when the ray misses (the tracer's
// shade_radiance treats that as "evaluate the sky / ambient").
// ---------------------------------------------------------------------------
constexpr u32 kInvalidMaterial = 0xFFFFFFFFu;

struct RayHit {
    Vec3  position     {0, 0, 0};
    Vec3  normal       {0, 1, 0};   // outward-facing, unit
    float t            {0.0f};      // distance along ray (-1 = miss)
    u32   material_idx {kInvalidMaterial};
};

// ---------------------------------------------------------------------------
// Free helpers — intersection + visibility. Used by both raytracer.hpp
// (primary + indirect) and bake.hpp (occlusion of probe / lightmap rays).
// ---------------------------------------------------------------------------

// Closest-hit query against every Sphere + Plane in `scene`. Returns true
// + populates `out_hit` if any positive-t intersection is found; false on
// miss (sky / no geometry). NaN-defensive: rays with non-finite origin /
// direction are treated as misses (silent — the tracer must never crash).
bool trace_closest(const Ray& ray, const Scene& scene, RayHit& out_hit) noexcept;

// Any-hit visibility test along `direction` for length `max_dist`. Used
// by direct lighting to test occlusion between a surface point and a
// light source. Returns true if NOTHING blocks the path (free line of
// sight); false if an occluder is found. Honours a small `eps` epsilon
// on the start to dodge self-intersection.
bool visible(const Vec3& from, const Vec3& direction, float max_dist,
             const Scene& scene) noexcept;

// ---------------------------------------------------------------------------
// BRDF evaluation — Cook-Torrance microfacet (GGX-D + Schlick-Fresnel +
// Smith-G) for the specular lobe + Lambertian diffuse, mixed by metallic.
// Both inputs and outputs are NaN-defended; a NaN normal collapses to
// a degenerate-but-bounded zero shading contribution (no poison).
//
// Inputs:
//   n  — surface normal (unit)
//   v  — view direction toward camera (unit)
//   l  — light direction toward light (unit)
//   m  — material
//
// Returns the BRDF value (W/m²/sr per unit irradiance) such that the
// outgoing radiance is brdf(n,v,l,m) * light_irradiance * max(dot(n,l), 0).
// ---------------------------------------------------------------------------
Vec3 brdf(const Vec3& n, const Vec3& v, const Vec3& l, const Material& m) noexcept;

// ---------------------------------------------------------------------------
// Deterministic RNG — splitmix32. Same impl as particles for output
// consistency, exposed publicly so bakers + tests can seed identically.
// ---------------------------------------------------------------------------
class Rng {
public:
    explicit Rng(u32 seed) noexcept : s_(seed) {}
    u32   next_u32() noexcept;
    float next_f01() noexcept;                  // uniform in [0,1)
    Vec3  next_cosine_hemisphere(const Vec3& n) noexcept;  // cosine-weighted on n
    Vec3  next_unit_sphere() noexcept;          // uniform on S²
private:
    u32 s_;
};

// ---------------------------------------------------------------------------
// Color helpers — Vec3 doesn't ship a hadamard, but lighting reads/writes
// color * color all over (base * incoming, F0 * fresnel etc). These live
// here so every lighting source uses the same one.
// ---------------------------------------------------------------------------
inline Vec3 mul(const Vec3& a, const Vec3& b) noexcept {
    return { a.x * b.x, a.y * b.y, a.z * b.z };
}
inline Vec3 add(const Vec3& a, const Vec3& b) noexcept {
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

// Sanitize a scalar — every public-knob read goes through this so a NaN
// or +Inf authoring value can't poison persistent state. Returns
// `fallback` when v is non-finite.
inline float fz(float v, float fallback) noexcept {
    return cardinal::isfinite(v) ? v : fallback;
}
inline Vec3 fz(const Vec3& v, const Vec3& fallback) noexcept {
    return {
        cardinal::isfinite(v.x) ? v.x : fallback.x,
        cardinal::isfinite(v.y) ? v.y : fallback.y,
        cardinal::isfinite(v.z) ? v.z : fallback.z,
    };
}

}  // namespace cardinal::lighting
