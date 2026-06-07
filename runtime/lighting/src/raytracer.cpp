#include <cardinal/lighting/raytracer.hpp>

#include <cardinal/core/std/cmath.hpp>
#include <cardinal/core/std/algorithm.hpp>

#include <chrono>

namespace cardinal::lighting {

namespace {

constexpr u32   kHardBounceCap = 16;
constexpr u32   kSppMax        = 4096;
constexpr float kSurfaceEps    = 1.0e-4f;

inline float saturate(float v) noexcept {
    if (!cardinal::isfinite(v)) return 0.0f;
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

inline u8 to_srgb_u8(float linear) noexcept {
    // Match the engine's render::algo tonemap exit: gamma 2.2 approx
    // (production tonemap is in the pipeline; this is the bake-time /
    // debug-image exit so the test can compare bytes against expected).
    const float v = saturate(linear);
    const float s = (v <= 0.0031308f)
        ? 12.92f * v
        : 1.055f * cardinal::pow(v, 1.0f / 2.4f) - 0.055f;
    const float clamped = saturate(s) * 255.0f + 0.5f;
    return static_cast<u8>(clamped);
}

// Look up a knob value by id with a NaN-defensive fallback. Knob.id is
// cardinal::string and supports operator==(const char*).
float knob_f(const cardinal::vector<Knob>& knobs, const char* id, float fallback) noexcept {
    for (const Knob& k : knobs) {
        if (k.id == id) return fz(k.f, fallback);
    }
    return fallback;
}
int knob_i(const cardinal::vector<Knob>& knobs, const char* id, int fallback) noexcept {
    for (const Knob& k : knobs) {
        if (k.id == id) return k.i;
    }
    return fallback;
}
bool knob_b(const cardinal::vector<Knob>& knobs, const char* id, bool fallback) noexcept {
    for (const Knob& k : knobs) {
        if (k.id == id) return k.b;
    }
    return fallback;
}

Knob make_f_knob(const char* id, const char* label, float def, float lo, float hi,
                 const char* tip)
{
    Knob k;
    k.kind = KnobKind::Float;
    k.id = id; k.label = label; k.group = "RTX"; k.tooltip = tip;
    k.f = def; k.f_min = lo; k.f_max = hi;
    return k;
}
Knob make_i_knob(const char* id, const char* label, int def, int lo, int hi,
                 const char* tip)
{
    Knob k;
    k.kind = KnobKind::Int;
    k.id = id; k.label = label; k.group = "RTX"; k.tooltip = tip;
    k.i = def; k.i_min = lo; k.i_max = hi;
    return k;
}
Knob make_b_knob(const char* id, const char* label, bool def, const char* tip)
{
    Knob k;
    k.kind = KnobKind::Bool;
    k.id = id; k.label = label; k.group = "RTX"; k.tooltip = tip;
    k.b = def;
    return k;
}

}  // namespace

cardinal::shared_ptr<PathTracer> PathTracer::create(PathTracerConfig cfg)
{
    auto pt = cardinal::shared_ptr<PathTracer>(new PathTracer());
    pt->build_knobs_(cfg);
    return pt;
}

void PathTracer::build_knobs_(const PathTracerConfig& cfg) {
    knobs_.clear();
    knobs_.push_back(make_i_knob("rtx.max_bounces",
        "Max Bounces",
        static_cast<int>(cardinal::clamp(cfg.max_bounces, 0u, kHardBounceCap)),
        0, static_cast<int>(kHardBounceCap),
        "Cap on indirect bounces. 0 = direct only; pure noise gain above ~5."));
    knobs_.push_back(make_i_knob("rtx.spp",
        "Samples Per Pixel",
        static_cast<int>(cardinal::clamp(cfg.samples_per_pixel, 1u, kSppMax)),
        1, static_cast<int>(kSppMax),
        "Primary rays per pixel. Variance is O(1/sqrt(spp))."));
    knobs_.push_back(make_i_knob("rtx.rr_depth",
        "Russian-Roulette Depth",
        static_cast<int>(cardinal::clamp(cfg.russian_roulette_depth, 0u, kHardBounceCap)),
        0, static_cast<int>(kHardBounceCap),
        "Bounce at which RR termination kicks in (unbiased throughput weight)."));
    knobs_.push_back(make_b_knob("rtx.indirect",
        "Enable Indirect",
        cfg.enable_indirect,
        "Disable to flatten the image to direct-only (debug)."));
    knobs_.push_back(make_b_knob("rtx.nee",
        "Next-Event Estimation",
        cfg.enable_next_event,
        "Explicit shadow-ray sampling of every light at each hit."));
    knobs_.push_back(make_f_knob("rtx.sky.r", "Sky R", saturate(cfg.sky_color.x), 0.0f, 4.0f,
        "Miss-radiance red channel (HDR)."));
    knobs_.push_back(make_f_knob("rtx.sky.g", "Sky G", saturate(cfg.sky_color.y), 0.0f, 4.0f,
        "Miss-radiance green channel (HDR)."));
    knobs_.push_back(make_f_knob("rtx.sky.b", "Sky B", saturate(cfg.sky_color.z), 0.0f, 4.0f,
        "Miss-radiance blue channel (HDR)."));
}

PathTracerConfig PathTracer::config() const noexcept {
    PathTracerConfig c;
    c.max_bounces        = static_cast<u32>(cardinal::clamp(
        knob_i(knobs_, "rtx.max_bounces", 3), 0, static_cast<int>(kHardBounceCap)));
    c.samples_per_pixel  = static_cast<u32>(cardinal::clamp(
        knob_i(knobs_, "rtx.spp", 1), 1, static_cast<int>(kSppMax)));
    c.russian_roulette_depth = static_cast<u32>(cardinal::clamp(
        knob_i(knobs_, "rtx.rr_depth", 2), 0, static_cast<int>(kHardBounceCap)));
    c.enable_indirect    = knob_b(knobs_, "rtx.indirect", true);
    c.enable_next_event  = knob_b(knobs_, "rtx.nee", true);
    c.sky_color = {
        saturate(knob_f(knobs_, "rtx.sky.r", 0.20f)),
        saturate(knob_f(knobs_, "rtx.sky.g", 0.32f)),
        saturate(knob_f(knobs_, "rtx.sky.b", 0.55f)),
    };
    return c;
}

// ---------------------------------------------------------------------------
// shade_direct — sum over LightSet, occlusion-tested.
// ---------------------------------------------------------------------------
Vec3 PathTracer::shade_direct(const RayHit& hit, const Vec3& view_dir,
                              const Scene& sc) noexcept
{
    if (hit.material_idx >= sc.materials.size()) return {0, 0, 0};
    if (!sc.lights) return {0, 0, 0};

    const Material& mat = sc.materials[hit.material_idx];
    Vec3 result = fz(mat.emissive, Vec3{0, 0, 0});

    // Ambient — flat term from LightSet (treat as already-cosine-weighted
    // hemisphere irradiance and route through the diffuse term).
    const Vec3 amb = fz(sc.lights->ambient(), Vec3{0, 0, 0});
    const Vec3 base = fz(mat.base_color, Vec3{0.8f, 0.8f, 0.8f});
    result = add(result, mul(amb, base));

    for (const scene::Light& L : sc.lights->lights()) {
        Vec3  light_dir  {0, 1, 0};
        float light_dist {1.0e6f};
        Vec3  light_radiance{0, 0, 0};

        switch (L.kind) {
            case scene::LightKind::Directional: {
                light_dir = scene::normalize(-fz(L.direction, Vec3{0, -1, 0}));
                light_dist = 1.0e6f;
                light_radiance = fz(L.color, Vec3{1, 1, 1}) * fz(L.intensity, 1.0f);
                break;
            }
            case scene::LightKind::Point: {
                Vec3 to_light = fz(L.position, Vec3{0, 0, 0}) - hit.position;
                light_dist = scene::length(to_light);
                if (light_dist < 1.0e-4f) continue;
                light_dir = to_light * (1.0f / light_dist);
                const float range = fz(L.range, 30.0f);
                const float att = (range > 0.0f && light_dist < range)
                    ? (1.0f - light_dist / range) * (1.0f - light_dist / range)
                    : 0.0f;
                if (att <= 0.0f) continue;
                light_radiance = fz(L.color, Vec3{1, 1, 1}) * (fz(L.intensity, 1.0f) * att);
                break;
            }
            case scene::LightKind::Spot: {
                Vec3 to_light = fz(L.position, Vec3{0, 0, 0}) - hit.position;
                light_dist = scene::length(to_light);
                if (light_dist < 1.0e-4f) continue;
                light_dir = to_light * (1.0f / light_dist);
                const float range = fz(L.range, 30.0f);
                const float dist_att = (range > 0.0f && light_dist < range)
                    ? (1.0f - light_dist / range) * (1.0f - light_dist / range)
                    : 0.0f;
                const Vec3  spot_axis = scene::normalize(fz(L.direction, Vec3{0, -1, 0}));
                const float cos_at_p  = scene::dot(spot_axis, -light_dir);
                const float inner = fz(L.spot_inner_cos, 0.95f);
                const float outer = fz(L.spot_outer_cos, 0.85f);
                const float t = (inner > outer)
                    ? saturate((cos_at_p - outer) / (inner - outer)) : 1.0f;
                const float att = dist_att * t * t;
                if (att <= 0.0f) continue;
                light_radiance = fz(L.color, Vec3{1, 1, 1}) * (fz(L.intensity, 1.0f) * att);
                break;
            }
        }

        const float NdotL = saturate(scene::dot(hit.normal, light_dir));
        if (NdotL <= 0.0f) continue;

        // Occlusion (next-event shadow ray).
        if (!visible(hit.position + hit.normal * kSurfaceEps,
                     light_dir, light_dist, sc)) {
            ++stats_.rays_shadow;
            continue;
        }
        ++stats_.rays_shadow;

        const Vec3 f = brdf(hit.normal, view_dir, light_dir, mat);
        const Vec3 contrib = mul(f, light_radiance) * NdotL;
        result = add(result, contrib);
    }
    return result;
}

// ---------------------------------------------------------------------------
// trace_radiance — recursive path trace with RR termination + NEE.
// ---------------------------------------------------------------------------
Vec3 PathTracer::trace_radiance(const Ray& ray_in, const Scene& sc, Rng& rng) noexcept
{
    const PathTracerConfig cfg = config();
    const Vec3 sky = fz(cfg.sky_color, Vec3{0.20f, 0.32f, 0.55f});

    Vec3 throughput{1, 1, 1};
    Vec3 radiance  {0, 0, 0};
    Ray  ray = ray_in;

    for (u32 depth = 0; depth <= cfg.max_bounces; ++depth) {
        stats_.max_depth_observed =
            (depth > stats_.max_depth_observed) ? depth : stats_.max_depth_observed;

        RayHit hit;
        if (depth == 0) ++stats_.rays_primary; else ++stats_.rays_indirect;
        if (!trace_closest(ray, sc, hit)) {
            ++stats_.misses;
            radiance = add(radiance, mul(throughput, sky));
            break;
        }
        ++stats_.hits;

        // Direct lighting at this hit (next-event estimation). Always run
        // at the first hit so a 0-bounce config still has lit geometry.
        if (cfg.enable_next_event || depth == 0) {
            const Vec3 v = -ray.direction;
            const Vec3 dir = shade_direct(hit, v, sc);
            radiance = add(radiance, mul(throughput, dir));
        }

        if (!cfg.enable_indirect || depth == cfg.max_bounces) break;

        // BRDF-importance-sample the next direction — cosine-weighted on
        // the surface hemisphere (good for diffuse-dominant surfaces; a
        // GGX-importance-sampling refinement is a follow-up).
        const Material& mat = (hit.material_idx < sc.materials.size())
            ? sc.materials[hit.material_idx] : Material{};
        const Vec3 next_dir = rng.next_cosine_hemisphere(hit.normal);
        const float NdotNext = saturate(scene::dot(hit.normal, next_dir));
        if (NdotNext <= 0.0f) break;

        // f * cos / pdf, pdf = cos/π for cosine-weighted hemisphere → π * f.
        const Vec3 f = brdf(hit.normal, -ray.direction, next_dir, mat);
        const Vec3 throughput_step = {
            f.x * scene::kPi,
            f.y * scene::kPi,
            f.z * scene::kPi,
        };
        throughput = mul(throughput, throughput_step);

        // Russian-roulette termination — unbiased.
        if (depth >= cfg.russian_roulette_depth) {
            const float p_keep = saturate(cardinal::clamp(
                cardinal::abs(throughput.x) > cardinal::abs(throughput.y)
                    ? (cardinal::abs(throughput.x) > cardinal::abs(throughput.z)
                        ? cardinal::abs(throughput.x) : cardinal::abs(throughput.z))
                    : (cardinal::abs(throughput.y) > cardinal::abs(throughput.z)
                        ? cardinal::abs(throughput.y) : cardinal::abs(throughput.z)),
                0.05f, 0.95f));
            if (rng.next_f01() > p_keep) break;
            throughput = throughput * (1.0f / p_keep);
        }

        ray = Ray{hit.position + hit.normal * kSurfaceEps, next_dir};
    }

    // NaN-defensive exit: a NaN slipped through anywhere → zero out so
    // the framebuffer can't be poisoned.
    if (!cardinal::isfinite(radiance.x) || !cardinal::isfinite(radiance.y) ||
        !cardinal::isfinite(radiance.z)) {
        return {0, 0, 0};
    }
    return radiance;
}

// ---------------------------------------------------------------------------
// trace_image — pinhole camera, jittered SPP, dual output buffers.
// ---------------------------------------------------------------------------
void PathTracer::trace_image(u32 width, u32 height,
                             const Vec3& camera_pos,
                             const Vec3& forward_in, const Vec3& right_in,
                             const Vec3& up_in,
                             float vertical_fov_rad,
                             const Scene& sc,
                             float* out_radiance,
                             u8*    out_rgba) noexcept
{
    if (width == 0 || height == 0 || !out_radiance || !out_rgba) return;
    const auto t0 = std::chrono::high_resolution_clock::now();

    const PathTracerConfig cfg = config();
    Rng rng(cfg.rng_seed);

    const Vec3 forward = scene::normalize(forward_in);
    const Vec3 right   = scene::normalize(right_in);
    const Vec3 up      = scene::normalize(up_in);

    const float aspect    = static_cast<float>(width) / static_cast<float>(height);
    const float fov_safe  = fz(vertical_fov_rad, 60.0f * scene::kDegToRad);
    const float half_h    = cardinal::tan(0.5f * fov_safe);
    const float half_w    = half_h * aspect;
    const float inv_w     = 1.0f / static_cast<float>(width);
    const float inv_h     = 1.0f / static_cast<float>(height);

    for (u32 y = 0; y < height; ++y) {
        for (u32 x = 0; x < width; ++x) {
            Vec3 accum{0, 0, 0};
            for (u32 s = 0; s < cfg.samples_per_pixel; ++s) {
                const float jx = (cfg.samples_per_pixel > 1) ? rng.next_f01() : 0.5f;
                const float jy = (cfg.samples_per_pixel > 1) ? rng.next_f01() : 0.5f;
                const float u  = ((static_cast<float>(x) + jx) * inv_w) * 2.0f - 1.0f;
                const float v  = 1.0f - ((static_cast<float>(y) + jy) * inv_h) * 2.0f;
                Vec3 dir = forward + right * (u * half_w) + up * (v * half_h);
                dir = scene::normalize(dir);
                Ray r{camera_pos, dir};
                accum = add(accum, trace_radiance(r, sc, rng));
            }
            const float inv_spp = 1.0f / static_cast<float>(cfg.samples_per_pixel);
            const Vec3 c{accum.x * inv_spp, accum.y * inv_spp, accum.z * inv_spp};
            const usize px = (static_cast<usize>(y) * width + x);
            out_radiance[px * 3 + 0] = c.x;
            out_radiance[px * 3 + 1] = c.y;
            out_radiance[px * 3 + 2] = c.z;
            out_rgba[px * 4 + 0] = to_srgb_u8(c.x);
            out_rgba[px * 4 + 1] = to_srgb_u8(c.y);
            out_rgba[px * 4 + 2] = to_srgb_u8(c.z);
            out_rgba[px * 4 + 3] = 255;
        }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    stats_.total_us = static_cast<float>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
}

}  // namespace cardinal::lighting
