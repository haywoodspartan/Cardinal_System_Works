#pragma once

// =============================================================================
// Cardinal — Baked lighting.
//
// Two complementary offline integrators consume the path tracer from
// raytracer.hpp:
//
//   1. BakedLightmap + LightmapBaker — a 2D RGB-float atlas. Authors mark
//      surfaces as static; the baker runs N cosine-weighted rays per texel,
//      integrates the path-traced radiance, and writes the result into the
//      atlas. Runtime sampling is a bilinear UV lookup — no rays.
//
//   2. IrradianceProbeVolume + ProbeVolumeBaker — a 3D grid of probes
//      covering a level AABB. Each probe stores a 6-axis "ambient cube"
//      (Valve's HL2 encoding: one RGB per ±X / ±Y / ±Z hemisphere). The
//      6-axis form is the sweet spot between L0 SH (no directionality)
//      and L2 SH (9× the memory + integration cost) — enough variation
//      to shade a normal-mapped surface plausibly, cheap to store and
//      to sample at runtime via trilinear-between-probes + axis weights.
//
// Both are pure CPU + deterministic given a seed. The GPU runtime samplers
// land alongside the RT-RHI port; until then hosts can use the CPU
// `sample()` calls below for editor preview / debug overlays.
//
// Design rules (carried from raytracer.hpp):
//   * NaN-defensive at every public surface (fz / finite-check everywhere)
//   * No allocation in the runtime samplers (the baked buffers are read-only)
//   * Knob-introspectable bake settings (samples-per-texel, samples-per-probe,
//     max bounces) reuse render::Knob so the editor panel works untouched
// =============================================================================

#include <cardinal/lighting/raytracer.hpp>

namespace cardinal::lighting {

// ---------------------------------------------------------------------------
// BakedLightmap — width × height RGB-float atlas. (u, v) in [0, 1) wraps
// to clamp at the boundary (no tiling for now — tiling lightmaps are
// content-pipeline territory). Pure data; sampler is a free function.
// ---------------------------------------------------------------------------
struct BakedLightmap {
    u32                       width  {0};
    u32                       height {0};
    cardinal::vector<float>   data;          // width * height * 3, RGB linear

    void resize(u32 w, u32 h) {
        width = w; height = h;
        data.assign(static_cast<usize>(w) * h * 3u, 0.0f);
    }
    bool empty() const noexcept { return width == 0 || height == 0; }
};

// Bilinear sample of a baked lightmap. (u, v) clamped to [0, 1]. NaN UV
// → centre sample (defined, finite). Out-of-bounds resolved by edge clamp.
Vec3 sample_lightmap(const BakedLightmap& lm, float u, float v) noexcept;

// ---------------------------------------------------------------------------
// LightmapTexel — one shading point the baker integrates. Authoring tools
// (or the host's UV unwrap) populate `world_pos / world_normal / uv` and
// hand the resulting array to the baker. The baker doesn't need any mesh
// connectivity — every texel is independent (fully parallelisable, even
// though the reference baker walks them serially).
// ---------------------------------------------------------------------------
struct LightmapTexel {
    Vec3  world_pos    {0, 0, 0};
    Vec3  world_normal {0, 1, 0};
    u32   atlas_u      {0};   // pixel coordinate; the baker writes data[(v*w+u)*3..+3]
    u32   atlas_v      {0};
};

// ---------------------------------------------------------------------------
// LightmapBaker — integrates per-texel irradiance through the host's
// PathTracer. The baker borrows the tracer (must outlive); seed defaults
// to PathTracerConfig::rng_seed but can be overridden per-bake. Stats
// pin total ray count + wall-clock for editor inspection.
// ---------------------------------------------------------------------------
struct LightmapBakeConfig {
    u32  samples_per_texel {16};   // cosine-weighted hemisphere samples; clamped to [1, 4096]
    u32  rng_seed          {0xBA64u};
    bool include_direct    {true}; // false → bake INDIRECT only (e.g. mixed-mode lighting)
};

struct LightmapBakeStats {
    u64   rays      {0};
    u32   texels    {0};
    float total_us  {0.0f};
};

class LightmapBaker {
public:
    static cardinal::shared_ptr<LightmapBaker> create(
        cardinal::shared_ptr<PathTracer> tracer);

    // Run a bake. `lm` is resized to (atlas_w × atlas_h) and overwritten.
    // Every texel in `texels` is integrated; texels with out-of-range
    // atlas coords are silently skipped (the baker never reads/writes
    // outside `lm.data`). NaN positions / normals are coerced to (0, +Y)
    // — the texel still bakes (to a defined finite value) rather than
    // crashing or being skipped.
    void bake(const Scene& scene,
              const cardinal::vector<LightmapTexel>& texels,
              u32 atlas_w, u32 atlas_h,
              BakedLightmap& lm,
              LightmapBakeConfig cfg = {}) noexcept;

    LightmapBakeStats stats() const noexcept { return stats_; }

private:
    LightmapBaker() = default;
    cardinal::shared_ptr<PathTracer> tracer_;
    LightmapBakeStats stats_;
};

// ---------------------------------------------------------------------------
// IrradianceProbeVolume — 3D grid of probes inside an AABB. Each probe
// stores six RGB values (the ambient cube: +X/-X/+Y/-Y/+Z/-Z hemispheres).
// Memory: dims.x*dims.y*dims.z * 6 * 3 floats. A 16³ volume = 196,608
// floats = 768 KB; reasonable for a level chunk.
// ---------------------------------------------------------------------------
struct IrradianceProbeVolume {
    Vec3 origin {0, 0, 0};              // world position of probe (0, 0, 0)
    Vec3 spacing{1, 1, 1};              // distance between adjacent probes
    u32  dims_x {0}, dims_y{0}, dims_z{0};

    // Layout: data[(((z * dims_y) + y) * dims_x + x) * 6 + axis] * 3 floats
    //         axis = 0..5 → +X, -X, +Y, -Y, +Z, -Z
    cardinal::vector<float> data;

    void resize(u32 dx, u32 dy, u32 dz) {
        dims_x = dx; dims_y = dy; dims_z = dz;
        data.assign(static_cast<usize>(dx) * dy * dz * 6u * 3u, 0.0f);
    }
    bool   empty() const noexcept { return dims_x == 0 || dims_y == 0 || dims_z == 0; }
    usize  probe_count() const noexcept {
        return static_cast<usize>(dims_x) * dims_y * dims_z;
    }
};

// Sample the volume at `world_pos` with surface normal `normal`. Returns
// the irradiance arriving from the hemisphere facing `normal`. Uses
// trilinear interpolation between the 8 nearest probes + ambient-cube
// weights from `normal`. Points outside the volume are clamped to the
// boundary probe (no false darkness past the volume edge).
Vec3 sample_probe_volume(const IrradianceProbeVolume& vol,
                         const Vec3& world_pos,
                         const Vec3& normal) noexcept;

// ---------------------------------------------------------------------------
// ProbeVolumeBaker — integrates each probe's 6 ambient-cube faces.
// ---------------------------------------------------------------------------
struct ProbeBakeConfig {
    u32  samples_per_axis {32};    // rays per hemisphere; total per probe = 6 × this
    u32  rng_seed         {0xB0BBu};
    bool include_direct   {true};
};

struct ProbeBakeStats {
    u64   rays   {0};
    u32   probes {0};
    float total_us {0.0f};
};

class ProbeVolumeBaker {
public:
    static cardinal::shared_ptr<ProbeVolumeBaker> create(
        cardinal::shared_ptr<PathTracer> tracer);

    void bake(const Scene& scene,
              IrradianceProbeVolume& vol,
              ProbeBakeConfig cfg = {}) noexcept;

    ProbeBakeStats stats() const noexcept { return stats_; }

private:
    ProbeVolumeBaker() = default;
    cardinal::shared_ptr<PathTracer> tracer_;
    ProbeBakeStats stats_;
};

}  // namespace cardinal::lighting
