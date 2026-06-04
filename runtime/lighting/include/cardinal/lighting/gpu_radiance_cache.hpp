#pragma once

// =============================================================================
// Cardinal — Radiance Cache (AEGIS Block 7 → Radiance Cache).
//
// Wraps the existing lighting::IrradianceProbeVolume + ProbeVolumeBaker as
// a render::graph::Pass so the AEGIS orchestrator can drive irradiance
// probe updates per frame. Output is the ambient-cube data packed into a
// graph-managed buffer; hosts sample at shading time via the existing
// lighting::sample_probe_volume() helper.
//
// Probe layout (matches IrradianceProbeVolume::data layout exactly):
//   dims_x × dims_y × dims_z × 6 axes × 3 floats = 18 × probe_count floats
//
// Pairs with ReSTIR DI (sibling subsystem in the same AEGIS block) — DI
// handles direct lighting; the Radiance Cache provides cached indirect
// irradiance the host can sample to shortcut multi-bounce path tracing.
//
// Convention matches the rest of the gpu_* family: shared_ptr<State>,
// add_to_graph factory, hlsl_source stub, NaN-defended record.
// =============================================================================

#include <cardinal/lighting/bake.hpp>
#include <cardinal/lighting/raytracer.hpp>
#include <cardinal/render/graph.hpp>

namespace cardinal::lighting::gpu {

class RadianceCachePass {
public:
    struct State {
        // Inputs (borrowed; must outlive execute).
        cardinal::shared_ptr<PathTracer> tracer;
        const Scene*                     scene{nullptr};

        // Volume layout (set by the host before each build).
        cardinal::scene::Vec3 origin   {0, 0, 0};
        cardinal::scene::Vec3 spacing  {1, 1, 1};
        cardinal::u32         dims_x   {0};
        cardinal::u32         dims_y   {0};
        cardinal::u32         dims_z   {0};
        cardinal::u32         samples_per_axis {8};
        cardinal::u32         rng_seed {0xCACE0001u};

        // Output: probe volume data buffer (matches IrradianceProbeVolume::data
        // layout: dims_x * dims_y * dims_z * 6 * 3 floats, axis-major in
        // order +X, -X, +Y, -Y, +Z, -Z per probe).
        cardinal::render::graph::ResourceHandle out_probe_data;

        // Stats published after each execute.
        cardinal::u32 probes_baked {0};
        cardinal::u64 rays_cast    {0};
        float         total_us     {0.0f};
    };

    static cardinal::shared_ptr<State> add_to_graph(
        cardinal::render::graph::Graph& g,
        cardinal::shared_ptr<PathTracer> tracer,
        const Scene& scene,
        cardinal::scene::Vec3 origin,
        cardinal::scene::Vec3 spacing,
        cardinal::u32 dims_x, cardinal::u32 dims_y, cardinal::u32 dims_z,
        cardinal::u32 samples_per_axis = 8);

    static const char* hlsl_source() noexcept;
};

}  // namespace cardinal::lighting::gpu
