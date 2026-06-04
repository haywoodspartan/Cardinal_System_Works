#include <cardinal/lighting/lighting.hpp>

#include <cardinal/core/cmath.hpp>
#include <cardinal/core/algorithm.hpp>     // cardinal::clamp

namespace cardinal::lighting {

namespace {

// Self-intersection epsilon. Rays cast from a surface point are nudged
// along the normal by this much before the intersection test, so a
// freshly-shaded fragment doesn't immediately re-hit itself.
constexpr float kSurfaceEps = 1.0e-4f;

// Floor used in the GGX denominator + dot-product clamps. Avoids
// divide-by-zero without changing values that are already > kFloor.
constexpr float kFloor = 1.0e-6f;

inline float saturate(float v) noexcept {
    if (!cardinal::isfinite(v)) return 0.0f;
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

inline bool is_finite_vec3(const Vec3& v) noexcept {
    return cardinal::isfinite(v.x) && cardinal::isfinite(v.y) && cardinal::isfinite(v.z);
}

// Ray-plane intersection, returns true + writes t on hit in front of origin.
// n is renormalized defensively (a host plane with a non-unit normal would
// otherwise scale the BRDF NdotL terms).
bool ray_plane(const Ray& r, const Vec3& point, const Vec3& normal_in,
               float* out_t, Vec3* out_n) noexcept
{
    const Vec3 n = scene::normalize(normal_in);
    const float denom = scene::dot(r.direction, n);
    if (denom > -kFloor && denom < kFloor) return false;  // parallel
    const float t = scene::dot(point - r.origin, n) / denom;
    if (t <= 0.0f) return false;
    if (out_t) *out_t = t;
    // Flip toward the ray so hit.normal is the side the ray hit (matters
    // for two-sided floors / walls in the bake scenes).
    if (out_n) *out_n = (denom < 0.0f) ? n : -n;
    return true;
}

// Ray-sphere — outward-facing normal at hit. Re-implemented (rather
// than calling scene::ray_sphere_intersect) so we also produce the
// outward normal and reject degenerate radii.
bool ray_sphere(const Ray& r, const Sphere& s, float* out_t, Vec3* out_n) noexcept
{
    const float radius = fz(s.radius, 0.0f);
    if (radius <= kFloor) return false;
    const Vec3  oc = r.origin - s.center;
    const float b  = scene::dot(oc, r.direction);
    const float c  = scene::dot(oc, oc) - radius * radius;
    const float disc = b * b - c;
    if (disc < 0.0f) return false;
    const float sd = cardinal::sqrt(disc);
    const float t0 = -b - sd;
    const float t1 = -b + sd;
    const float t  = (t0 > kSurfaceEps) ? t0 : ((t1 > kSurfaceEps) ? t1 : -1.0f);
    if (t < 0.0f) return false;
    if (out_t) *out_t = t;
    if (out_n) {
        const Vec3 p = r.origin + r.direction * t;
        *out_n = scene::normalize(p - s.center);
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// trace_closest
// ---------------------------------------------------------------------------
bool trace_closest(const Ray& ray, const Scene& scene, RayHit& out_hit) noexcept
{
    // NaN-defensive entry. A non-finite origin / direction would produce
    // NaN-poisoned t-values and crash the t-comparison below (NaN < NaN
    // is unordered-false → "no hit found" but also "no miss" — the loop
    // would silently skip every primitive). Treat as miss explicitly.
    if (!is_finite_vec3(ray.origin) || !is_finite_vec3(ray.direction)) return false;

    float closest_t = 1.0e30f;
    bool  found     = false;
    Vec3  closest_n {0, 1, 0};
    u32   closest_m {kInvalidMaterial};
    Vec3  closest_p {0, 0, 0};

    for (const Sphere& s : scene.spheres) {
        float t; Vec3 n;
        if (!ray_sphere(ray, s, &t, &n)) continue;
        if (t < closest_t) {
            closest_t = t; closest_n = n; closest_m = s.material_idx;
            closest_p = ray.origin + ray.direction * t;
            found = true;
        }
    }
    for (const Plane& pl : scene.planes) {
        float t; Vec3 n;
        if (!ray_plane(ray, pl.point, pl.normal, &t, &n)) continue;
        if (t < closest_t) {
            closest_t = t; closest_n = n; closest_m = pl.material_idx;
            closest_p = ray.origin + ray.direction * t;
            found = true;
        }
    }

    if (!found) return false;
    out_hit.position     = closest_p;
    out_hit.normal       = closest_n;
    out_hit.t            = closest_t;
    out_hit.material_idx = closest_m;
    return true;
}

// ---------------------------------------------------------------------------
// visible — any-hit along (direction * max_dist). Used by next-event
// estimation to occlude a sample against a light.
// ---------------------------------------------------------------------------
bool visible(const Vec3& from, const Vec3& direction, float max_dist,
             const Scene& scene_) noexcept
{
    if (!is_finite_vec3(from) || !is_finite_vec3(direction)) return false;
    if (!cardinal::isfinite(max_dist) || max_dist <= 0.0f) return true;

    Ray r{from + direction * kSurfaceEps, direction};
    for (const Sphere& s : scene_.spheres) {
        float t;
        if (ray_sphere(r, s, &t, nullptr) && t < max_dist - kSurfaceEps) return false;
    }
    for (const Plane& pl : scene_.planes) {
        float t;
        if (ray_plane(r, pl.point, pl.normal, &t, nullptr) && t < max_dist - kSurfaceEps) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// brdf — Cook-Torrance microfacet (GGX-D + Schlick-Fresnel + Smith-G)
// for specular + Lambertian diffuse, energy-mixed by metallic.
//
// All public-input NaN handled at the start (a NaN normal / view dir would
// otherwise produce NaN dot products → NaN BRDF → poisoned every pixel).
// ---------------------------------------------------------------------------
Vec3 brdf(const Vec3& n_in, const Vec3& v_in, const Vec3& l_in,
          const Material& m_in) noexcept
{
    if (!is_finite_vec3(n_in) || !is_finite_vec3(v_in) || !is_finite_vec3(l_in)) {
        return {0, 0, 0};
    }
    const Vec3 n = scene::normalize(n_in);
    const Vec3 v = scene::normalize(v_in);
    const Vec3 l = scene::normalize(l_in);

    const float NdotL = scene::dot(n, l);
    const float NdotV = scene::dot(n, v);
    if (NdotL <= 0.0f || NdotV <= 0.0f) return {0, 0, 0};

    const Vec3  h     = scene::normalize(v + l);
    const float NdotH = saturate(scene::dot(n, h));
    const float VdotH = saturate(scene::dot(v, h));

    // Perceptual roughness → α via squaring (Burley / Disney convention).
    const float roughness = saturate(fz(m_in.roughness, 0.5f));
    const float alpha     = cardinal::clamp(roughness * roughness, 0.002f, 1.0f);
    const float metallic  = saturate(fz(m_in.metallic, 0.0f));

    const Vec3  base = fz(m_in.base_color, Vec3{0.8f, 0.8f, 0.8f});

    // Schlick F0 — 0.04 for dielectrics, base_color for metals.
    const Vec3 F0 = {
        0.04f + (base.x - 0.04f) * metallic,
        0.04f + (base.y - 0.04f) * metallic,
        0.04f + (base.z - 0.04f) * metallic,
    };
    const float fresnel_pow = (1.0f - VdotH);
    const float fresnel_5   = fresnel_pow * fresnel_pow * fresnel_pow * fresnel_pow * fresnel_pow;
    const Vec3  F = {
        F0.x + (1.0f - F0.x) * fresnel_5,
        F0.y + (1.0f - F0.y) * fresnel_5,
        F0.z + (1.0f - F0.z) * fresnel_5,
    };

    // GGX D.
    const float alpha2 = alpha * alpha;
    const float denom_d = (NdotH * NdotH) * (alpha2 - 1.0f) + 1.0f;
    const float D = alpha2 / (scene::kPi * denom_d * denom_d + kFloor);

    // Smith G (separable GGX, Schlick approximation).
    const float k     = (roughness + 1.0f) * (roughness + 1.0f) * 0.125f;
    const float G_v   = NdotV / (NdotV * (1.0f - k) + k + kFloor);
    const float G_l   = NdotL / (NdotL * (1.0f - k) + k + kFloor);
    const float G     = G_v * G_l;

    const float spec_denom = 4.0f * NdotV * NdotL + kFloor;
    const Vec3 specular = {
        D * G * F.x / spec_denom,
        D * G * F.y / spec_denom,
        D * G * F.z / spec_denom,
    };

    // Energy conservation: kS = F, kD = (1-F) * (1-metallic).
    const Vec3 kD = {
        (1.0f - F.x) * (1.0f - metallic),
        (1.0f - F.y) * (1.0f - metallic),
        (1.0f - F.z) * (1.0f - metallic),
    };
    const Vec3 diffuse = {
        kD.x * base.x / scene::kPi,
        kD.y * base.y / scene::kPi,
        kD.z * base.z / scene::kPi,
    };

    return add(diffuse, specular);
}

// ---------------------------------------------------------------------------
// Rng — splitmix32. Returns NaN-free outputs by construction.
// ---------------------------------------------------------------------------
u32 Rng::next_u32() noexcept {
    s_ += 0x9E3779B9u;
    u32 z = s_;
    z = (z ^ (z >> 16)) * 0x21f0aaadu;
    z = (z ^ (z >> 15)) * 0x735a2d97u;
    z =  z ^ (z >> 15);
    return z;
}

float Rng::next_f01() noexcept {
    return static_cast<float>(next_u32()) * (1.0f / 4294967296.0f);
}

Vec3 Rng::next_cosine_hemisphere(const Vec3& n_in) noexcept {
    // Malley's method: uniform-in-disk + lift onto z = sqrt(1-r²).
    const Vec3 n = is_finite_vec3(n_in) ? scene::normalize(n_in) : Vec3{0, 1, 0};
    const float r1 = next_f01();
    const float r2 = next_f01();
    const float r       = cardinal::sqrt(r1);
    const float phi     = 2.0f * scene::kPi * r2;
    const float x_local = r * cardinal::cos(phi);
    const float y_local = r * cardinal::sin(phi);
    const float z_local = cardinal::sqrt(1.0f - r1);

    // Build an orthonormal basis around n.
    Vec3 a = (cardinal::abs(n.x) > 0.9f) ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
    Vec3 t = scene::normalize(scene::cross(a, n));
    Vec3 b = scene::cross(n, t);

    return {
        t.x * x_local + b.x * y_local + n.x * z_local,
        t.y * x_local + b.y * y_local + n.y * z_local,
        t.z * x_local + b.z * y_local + n.z * z_local,
    };
}

Vec3 Rng::next_unit_sphere() noexcept {
    // Standard inverse-CDF on (z, phi).
    const float z   = 2.0f * next_f01() - 1.0f;
    const float phi = 2.0f * scene::kPi * next_f01();
    const float r   = cardinal::sqrt(cardinal::clamp(1.0f - z * z, 0.0f, 1.0f));
    return { r * cardinal::cos(phi), r * cardinal::sin(phi), z };
}

}  // namespace cardinal::lighting
