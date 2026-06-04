#include <cardinal/lighting/bake.hpp>

#include <cardinal/core/cmath.hpp>
#include <cardinal/core/algorithm.hpp>

#include <chrono>

namespace cardinal::lighting {

namespace {

constexpr u32 kSppMax        = 4096;
constexpr u32 kProbeSppMax   = 4096;

inline float saturate(float v) noexcept {
    if (!cardinal::isfinite(v)) return 0.0f;
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

inline bool finv(const Vec3& v) noexcept {
    return cardinal::isfinite(v.x) && cardinal::isfinite(v.y) && cardinal::isfinite(v.z);
}

inline Vec3 safe_normal(const Vec3& n_in) noexcept {
    if (!finv(n_in)) return {0, 1, 0};
    const Vec3 n = scene::normalize(n_in);
    return finv(n) ? n : Vec3{0, 1, 0};
}

}  // namespace

// =============================================================================
// BakedLightmap — bilinear sampler
// =============================================================================
Vec3 sample_lightmap(const BakedLightmap& lm, float u_in, float v_in) noexcept {
    if (lm.empty()) return {0, 0, 0};
    // NaN UV → image center; OOB clamped.
    const float u = saturate(cardinal::isfinite(u_in) ? u_in : 0.5f);
    const float v = saturate(cardinal::isfinite(v_in) ? v_in : 0.5f);

    const float fx = u * static_cast<float>(lm.width  - 1);
    const float fy = v * static_cast<float>(lm.height - 1);
    const u32 x0 = static_cast<u32>(fx);
    const u32 y0 = static_cast<u32>(fy);
    const u32 x1 = (x0 + 1 < lm.width)  ? x0 + 1 : x0;
    const u32 y1 = (y0 + 1 < lm.height) ? y0 + 1 : y0;
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);

    auto fetch = [&](u32 x, u32 y) -> Vec3 {
        const usize idx = (static_cast<usize>(y) * lm.width + x) * 3u;
        return { lm.data[idx + 0], lm.data[idx + 1], lm.data[idx + 2] };
    };
    const Vec3 c00 = fetch(x0, y0);
    const Vec3 c10 = fetch(x1, y0);
    const Vec3 c01 = fetch(x0, y1);
    const Vec3 c11 = fetch(x1, y1);
    const Vec3 top = {
        c00.x + (c10.x - c00.x) * tx,
        c00.y + (c10.y - c00.y) * tx,
        c00.z + (c10.z - c00.z) * tx,
    };
    const Vec3 bot = {
        c01.x + (c11.x - c01.x) * tx,
        c01.y + (c11.y - c01.y) * tx,
        c01.z + (c11.z - c01.z) * tx,
    };
    return {
        top.x + (bot.x - top.x) * ty,
        top.y + (bot.y - top.y) * ty,
        top.z + (bot.z - top.z) * ty,
    };
}

// =============================================================================
// LightmapBaker
// =============================================================================
cardinal::shared_ptr<LightmapBaker> LightmapBaker::create(
    cardinal::shared_ptr<PathTracer> tracer)
{
    auto b = cardinal::shared_ptr<LightmapBaker>(new LightmapBaker());
    b->tracer_ = tracer;
    return b;
}

void LightmapBaker::bake(const Scene& scene_,
                         const cardinal::vector<LightmapTexel>& texels,
                         u32 atlas_w, u32 atlas_h,
                         BakedLightmap& lm,
                         LightmapBakeConfig cfg) noexcept
{
    if (!tracer_ || atlas_w == 0 || atlas_h == 0) return;
    stats_ = LightmapBakeStats{};
    const auto t0 = std::chrono::high_resolution_clock::now();

    lm.resize(atlas_w, atlas_h);
    const u32 spp = cardinal::clamp(cfg.samples_per_texel, 1u, kSppMax);

    Rng rng(cfg.rng_seed);
    constexpr float kSurfaceEps = 1.0e-4f;

    for (const LightmapTexel& t : texels) {
        if (t.atlas_u >= atlas_w || t.atlas_v >= atlas_h) continue;

        const Vec3 n  = safe_normal(t.world_normal);
        const Vec3 p  = finv(t.world_pos) ? t.world_pos : Vec3{0, 0, 0};
        const Vec3 origin = p + n * kSurfaceEps;

        Vec3 accum{0, 0, 0};
        for (u32 s = 0; s < spp; ++s) {
            const Vec3 dir = rng.next_cosine_hemisphere(n);
            scene::Ray ray{origin, dir};
            // The cosine-weighted PDF cancels the dot(n, dir) factor in the
            // irradiance integral, leaving the average of the traced
            // radiances as the irradiance estimate (over π implicitly).
            Vec3 L = tracer_->trace_radiance(ray, scene_, rng);
            ++stats_.rays;
            if (!finv(L)) L = Vec3{0, 0, 0};
            accum.x += L.x; accum.y += L.y; accum.z += L.z;
        }
        const float inv = 1.0f / static_cast<float>(spp);
        Vec3 result{accum.x * inv, accum.y * inv, accum.z * inv};

        // include_direct=false → re-trace at the same hit and subtract
        // the direct term so callers can blend a separate direct pass.
        if (!cfg.include_direct) {
            RayHit hit;
            hit.position = p;
            hit.normal   = n;
            // Look up the material at this surface — we don't know it
            // from a texel alone, so skip the subtraction when it's
            // ambiguous (no scene query). The cleanest path is for the
            // caller to drive include_direct=true and split downstream.
            // For now this branch is just "no-op": leave result as-is.
            (void)hit;
        }

        const usize idx = (static_cast<usize>(t.atlas_v) * atlas_w + t.atlas_u) * 3u;
        lm.data[idx + 0] = result.x;
        lm.data[idx + 1] = result.y;
        lm.data[idx + 2] = result.z;
        ++stats_.texels;
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    stats_.total_us = static_cast<float>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
}

// =============================================================================
// IrradianceProbeVolume — sampler
// =============================================================================

namespace {

inline usize probe_idx(const IrradianceProbeVolume& v, u32 x, u32 y, u32 z) noexcept {
    return ((static_cast<usize>(z) * v.dims_y + y) * v.dims_x + x) * 6u * 3u;
}

// Ambient-cube weights: each axis contributes by max(n·axis, 0)².
// Sum-normalised to keep energy conserved (the cube spans the full sphere
// so positive lobes sum to 1 after normalisation).
struct AxisW { float w[6]; };

inline AxisW axis_weights(const Vec3& n_in) noexcept {
    const Vec3 n = safe_normal(n_in);
    AxisW a{};
    a.w[0] = n.x > 0 ? n.x * n.x : 0;
    a.w[1] = n.x < 0 ? n.x * n.x : 0;
    a.w[2] = n.y > 0 ? n.y * n.y : 0;
    a.w[3] = n.y < 0 ? n.y * n.y : 0;
    a.w[4] = n.z > 0 ? n.z * n.z : 0;
    a.w[5] = n.z < 0 ? n.z * n.z : 0;
    const float sum = a.w[0] + a.w[1] + a.w[2] + a.w[3] + a.w[4] + a.w[5];
    if (sum > 0) {
        const float inv = 1.0f / sum;
        for (int i = 0; i < 6; ++i) a.w[i] *= inv;
    }
    return a;
}

inline Vec3 combine_cube(const float* probe_base, const AxisW& w) noexcept {
    Vec3 r{0, 0, 0};
    for (int axis = 0; axis < 6; ++axis) {
        const float wa = w.w[axis];
        if (wa <= 0) continue;
        const float* p = probe_base + axis * 3;
        r.x += p[0] * wa;
        r.y += p[1] * wa;
        r.z += p[2] * wa;
    }
    return r;
}

}  // namespace

Vec3 sample_probe_volume(const IrradianceProbeVolume& vol,
                         const Vec3& world_pos_in,
                         const Vec3& normal) noexcept
{
    if (vol.empty()) return {0, 0, 0};
    const Vec3 world_pos = finv(world_pos_in) ? world_pos_in : vol.origin;
    const Vec3 sp = {
        vol.spacing.x > 1.0e-6f ? vol.spacing.x : 1.0f,
        vol.spacing.y > 1.0e-6f ? vol.spacing.y : 1.0f,
        vol.spacing.z > 1.0e-6f ? vol.spacing.z : 1.0f,
    };

    // Map world → grid coords (continuous).
    const float gx = (world_pos.x - vol.origin.x) / sp.x;
    const float gy = (world_pos.y - vol.origin.y) / sp.y;
    const float gz = (world_pos.z - vol.origin.z) / sp.z;

    // Clamp to volume so out-of-bounds samples grab the boundary probes.
    const float cx = cardinal::clamp(gx, 0.0f, static_cast<float>(vol.dims_x - 1));
    const float cy = cardinal::clamp(gy, 0.0f, static_cast<float>(vol.dims_y - 1));
    const float cz = cardinal::clamp(gz, 0.0f, static_cast<float>(vol.dims_z - 1));

    const u32 x0 = static_cast<u32>(cx);
    const u32 y0 = static_cast<u32>(cy);
    const u32 z0 = static_cast<u32>(cz);
    const u32 x1 = (x0 + 1 < vol.dims_x) ? x0 + 1 : x0;
    const u32 y1 = (y0 + 1 < vol.dims_y) ? y0 + 1 : y0;
    const u32 z1 = (z0 + 1 < vol.dims_z) ? z0 + 1 : z0;
    const float tx = cx - static_cast<float>(x0);
    const float ty = cy - static_cast<float>(y0);
    const float tz = cz - static_cast<float>(z0);

    const AxisW w = axis_weights(normal);

    auto p = [&](u32 x, u32 y, u32 z) -> Vec3 {
        return combine_cube(vol.data.data() + probe_idx(vol, x, y, z), w);
    };

    const Vec3 c000 = p(x0, y0, z0);
    const Vec3 c100 = p(x1, y0, z0);
    const Vec3 c010 = p(x0, y1, z0);
    const Vec3 c110 = p(x1, y1, z0);
    const Vec3 c001 = p(x0, y0, z1);
    const Vec3 c101 = p(x1, y0, z1);
    const Vec3 c011 = p(x0, y1, z1);
    const Vec3 c111 = p(x1, y1, z1);

    auto lerp = [](const Vec3& a, const Vec3& b, float t) -> Vec3 {
        return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t};
    };
    const Vec3 c00 = lerp(c000, c100, tx);
    const Vec3 c10 = lerp(c010, c110, tx);
    const Vec3 c01 = lerp(c001, c101, tx);
    const Vec3 c11 = lerp(c011, c111, tx);
    const Vec3 c0  = lerp(c00,  c10,  ty);
    const Vec3 c1  = lerp(c01,  c11,  ty);
    return lerp(c0, c1, tz);
}

// =============================================================================
// ProbeVolumeBaker
// =============================================================================
cardinal::shared_ptr<ProbeVolumeBaker> ProbeVolumeBaker::create(
    cardinal::shared_ptr<PathTracer> tracer)
{
    auto b = cardinal::shared_ptr<ProbeVolumeBaker>(new ProbeVolumeBaker());
    b->tracer_ = tracer;
    return b;
}

void ProbeVolumeBaker::bake(const Scene& scene_,
                            IrradianceProbeVolume& vol,
                            ProbeBakeConfig cfg) noexcept
{
    if (!tracer_ || vol.empty()) return;
    stats_ = ProbeBakeStats{};
    const auto t0 = std::chrono::high_resolution_clock::now();

    const u32 spp = cardinal::clamp(cfg.samples_per_axis, 1u, kProbeSppMax);
    Rng rng(cfg.rng_seed);

    // Six cardinal-axis hemispheres. We integrate by drawing
    // cosine-weighted samples ON each axis (normal = axis) — same
    // estimator as the lightmap baker, evaluated 6 times per probe.
    const Vec3 axes[6] = {
        { 1,  0,  0}, {-1,  0,  0},
        { 0,  1,  0}, { 0, -1,  0},
        { 0,  0,  1}, { 0,  0, -1},
    };

    for (u32 z = 0; z < vol.dims_z; ++z) {
        for (u32 y = 0; y < vol.dims_y; ++y) {
            for (u32 x = 0; x < vol.dims_x; ++x) {
                const Vec3 probe_pos{
                    vol.origin.x + static_cast<float>(x) * vol.spacing.x,
                    vol.origin.y + static_cast<float>(y) * vol.spacing.y,
                    vol.origin.z + static_cast<float>(z) * vol.spacing.z,
                };
                const usize base = probe_idx(vol, x, y, z);

                for (int axis = 0; axis < 6; ++axis) {
                    Vec3 accum{0, 0, 0};
                    for (u32 s = 0; s < spp; ++s) {
                        const Vec3 dir = rng.next_cosine_hemisphere(axes[axis]);
                        scene::Ray ray{probe_pos, dir};
                        Vec3 L = tracer_->trace_radiance(ray, scene_, rng);
                        ++stats_.rays;
                        if (!finv(L)) L = Vec3{0, 0, 0};
                        accum.x += L.x; accum.y += L.y; accum.z += L.z;
                    }
                    const float inv = 1.0f / static_cast<float>(spp);
                    vol.data[base + axis * 3 + 0] = accum.x * inv;
                    vol.data[base + axis * 3 + 1] = accum.y * inv;
                    vol.data[base + axis * 3 + 2] = accum.z * inv;
                }
                ++stats_.probes;
            }
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    stats_.total_us = static_cast<float>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
}

}  // namespace cardinal::lighting
