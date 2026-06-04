#pragma once

// =============================================================================
// Cardinal — RTX-style path tracer (CPU reference).
//
// Recursive Monte Carlo: primary ray → BRDF importance sample → next ray,
// accumulating direct lighting at each bounce via next-event estimation
// against the host's scene::LightSet. Russian-roulette termination keeps
// the estimator unbiased past max_bounces; explicit hard cap stops runaway
// recursion at 16 bounces no matter what the RR draws.
//
// Knob-driven for editor introspection (max_bounces, spp, RR depth,
// disable-indirect for debug). Every public surface NaN-defended.
//
// The HLSL counterpart will land alongside the RT-RHI port; until then
// this is the byte-stable reference the GPU output validates against.
// =============================================================================

#include <cardinal/lighting/lighting.hpp>
#include <cardinal/render/pipeline.hpp>     // for render::Knob editor introspection

namespace cardinal::lighting {

// Aliased so PathTracerConfig doesn't have to qualify the names.
using Knob     = cardinal::render::Knob;
using KnobKind = cardinal::render::KnobKind;

// ---------------------------------------------------------------------------
// PathTracerConfig — owns the Knob table the editor reads, plus the
// authoring defaults. Hosts construct one, hand it to PathTracer::create,
// and can later mutate via Knob::set_*.
// ---------------------------------------------------------------------------
struct PathTracerConfig {
    u32   max_bounces       {3};    // 0 = direct only; clamped to [0, 16]
    u32   samples_per_pixel {1};    // primary rays per pixel; clamped to [1, 4096]
    u32   russian_roulette_depth{2};// kick in RR at this bounce depth
    bool  enable_indirect   {true}; // false = direct-only (debug)
    bool  enable_next_event {true}; // sample lights explicitly at each hit
    u32   rng_seed          {0xC0FFEEu};
    Vec3  sky_color         {0.20f, 0.32f, 0.55f};   // hit-miss return
};

// ---------------------------------------------------------------------------
// PathTracerStats — per-frame counters. Read after trace_image() or any
// of the lower-level trace_* calls. Useful for editor inspection.
// ---------------------------------------------------------------------------
struct PathTracerStats {
    u64 rays_primary        {0};
    u64 rays_shadow         {0};
    u64 rays_indirect       {0};
    u64 hits                {0};
    u64 misses              {0};
    u32 max_depth_observed  {0};   // useful for tuning max_bounces
    float total_us          {0.0f};
};

// ---------------------------------------------------------------------------
// PathTracer — pure-function tracer (no per-frame allocation after the
// initial Knob table is built). One instance per renderer / bake job;
// thread-safety is the caller's responsibility (each worker thread should
// own its own PathTracer to avoid Rng/stats contention).
// ---------------------------------------------------------------------------
class PathTracer {
public:
    static cardinal::shared_ptr<PathTracer> create(PathTracerConfig cfg = {});

    // -- Single-ray entrypoint (deterministic given the seed + bounce).
    //    Used by trace_image() and by the lightmap baker. Returns the
    //    radiance along `ray` (sky on miss; BRDF + indirect at hits).
    Vec3 trace_radiance(const Ray& ray, const Scene& scene, Rng& rng) noexcept;

    // -- Direct-only shading at a known hit (no bounce). Sums NdotL × BRDF
    //    over every light in the scene, occlusion-tested. Public so the
    //    irradiance baker can call it without re-tracing primary rays.
    Vec3 shade_direct(const RayHit& hit, const Vec3& view_dir,
                      const Scene& scene) noexcept;

    // -- Full-image trace: writes `out_rgba` (HDR linear * 255 clamped, alpha
    //    255) and a separate `out_radiance` (float RGB per pixel) so callers
    //    can post-process / re-tonemap without re-tracing. View ray =
    //    pinhole from camera_pos through pixel center mapped to a plane at
    //    `focal_dist` from camera, oriented by (forward, right, up). spp
    //    pulled from cfg, jittered uniformly per pixel.
    void trace_image(u32 width, u32 height,
                     const Vec3& camera_pos,
                     const Vec3& forward, const Vec3& right, const Vec3& up,
                     float vertical_fov_rad,
                     const Scene& scene,
                     float* out_radiance,    // width*height*3
                     u8*    out_rgba) noexcept;

    // -- Editor surface.
    const cardinal::vector<Knob>& knobs() const noexcept { return knobs_; }
    cardinal::vector<Knob>&       knobs()       noexcept { return knobs_; }
    PathTracerConfig               config() const noexcept;  // re-read knobs

    PathTracerStats stats() const noexcept { return stats_; }
    void            reset_stats() noexcept { stats_ = PathTracerStats{}; }

private:
    PathTracer() = default;
    void build_knobs_(const PathTracerConfig& cfg);

    cardinal::vector<Knob> knobs_;
    PathTracerStats        stats_;
};

}  // namespace cardinal::lighting
