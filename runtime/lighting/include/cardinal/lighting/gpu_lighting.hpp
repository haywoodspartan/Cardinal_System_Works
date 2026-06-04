#pragma once

// =============================================================================
// Cardinal — GPU lighting passes (forward shading + RTX path trace).
//
// Two graph-hosted passes that round out the engine's "every per-frame
// operation is a Pass on render::graph::Graph" goal for the lighting
// half of the pipeline:
//
//   * ForwardShadePass — per-fragment PBR direct lighting. Takes the
//                        rasterizer's output (world pos / normal / base
//                        color / metallic / roughness as parallel SoA
//                        buffers) + a packed light list, evaluates the
//                        Cook-Torrance BRDF for every fragment, writes
//                        shaded RGB. CPU reference calls into
//                        cardinal::lighting::brdf so this pass is
//                        bit-for-bit identical to the shipping CPU
//                        path under graph::CpuBackend.
//
//   * PathTracePass    — full-image path trace through the host's
//                        CPU reference PathTracer. Output is HDR
//                        linear radiance (one float-RGB per pixel) +
//                        an sRGB-encoded u8 RGBA buffer for direct
//                        display. The HLSL source documents the
//                        DXR-style raygen + miss + closest-hit shader
//                        that the RhiBackend will dispatch when the
//                        RT-RHI surface lands.
//
// Both follow the established graph pass convention:
//   * State struct held by a shared_ptr at the host (multiple instances
//     coexist without globals).
//   * add_to_graph(...) declares output buffers + wires the record fn.
//   * hlsl_source() returns the shader source-string for the editor
//     inspector + the eventual DXC compile.
//   * NaN-defended on every public read site.
// =============================================================================

#include <cardinal/lighting/raytracer.hpp>
#include <cardinal/render/graph.hpp>

namespace cardinal::lighting::gpu {

// ---------------------------------------------------------------------------
// PackedLight — light list entry. Same union of kinds as scene::Light,
// packed into a fixed-size record so the host can upload N lights as a
// single contiguous buffer.
// Layout: { kind(u32), _pad, position(3f), direction(3f), color(3f),
//           intensity(f), range(f), inner_cos(f), outer_cos(f) }
// = 16 floats per light.
// ---------------------------------------------------------------------------
constexpr cardinal::u32 kPackedLightFloats = 16;

struct PackedLight {
    cardinal::u32 kind;    // 0 = Directional, 1 = Point, 2 = Spot
    cardinal::u32 _pad0;
    float position[3];
    float direction[3];
    float color[3];
    float intensity;
    float range;
    float inner_cos;
    float outer_cos;
    float _pad1;           // 16-float / 64-byte alignment (cache-line half)
};
static_assert(sizeof(PackedLight) == kPackedLightFloats * sizeof(float),
              "PackedLight must be 16 floats");

// ---------------------------------------------------------------------------
// ForwardShadePass
//   * in_world_pos    : 3 * N floats SoA (x, y, z)
//   * in_world_normal : 3 * N floats SoA (nx, ny, nz)
//   * in_material     : 5 * N floats SoA (r, g, b, metallic, roughness)
//   * in_lights       : kPackedLightFloats * light_count floats
//   * in_ambient      : 3 floats (RGB)
//   * out_shaded      : 3 * N floats SoA (r, g, b linear)
// ---------------------------------------------------------------------------
class ForwardShadePass {
public:
    struct State {
        cardinal::render::graph::ResourceHandle in_world_pos;
        cardinal::render::graph::ResourceHandle in_world_normal;
        cardinal::render::graph::ResourceHandle in_material;
        cardinal::render::graph::ResourceHandle in_lights;
        cardinal::render::graph::ResourceHandle in_ambient;
        cardinal::render::graph::ResourceHandle out_shaded;
        cardinal::u32 fragment_count {0};
        cardinal::u32 light_count    {0};
        // Stats:
        cardinal::u32 fragments_lit  {0};   // NdotL > 0 on at least one light
    };

    static cardinal::shared_ptr<State> add_to_graph(
        cardinal::render::graph::Graph& g,
        cardinal::render::graph::ResourceHandle in_world_pos,
        cardinal::render::graph::ResourceHandle in_world_normal,
        cardinal::render::graph::ResourceHandle in_material,
        cardinal::render::graph::ResourceHandle in_lights,
        cardinal::render::graph::ResourceHandle in_ambient,
        cardinal::u32 fragment_count,
        cardinal::u32 light_count);

    static const char* hlsl_source() noexcept;
};

// ---------------------------------------------------------------------------
// PathTracePass — wraps the host's CPU PathTracer into a graph node.
// The pass's record() borrows the host-owned PathTracer + Scene (no copy)
// and writes the trace results into the declared output buffers.
// ---------------------------------------------------------------------------
class PathTracePass {
public:
    struct State {
        // Inputs (borrowed; must outlive the graph's execute).
        cardinal::shared_ptr<PathTracer> tracer;
        const Scene*                     scene{nullptr};

        // Camera (pinhole) parameters — set by the host before each execute.
        cardinal::scene::Vec3 camera_pos    {0, 0, 0};
        cardinal::scene::Vec3 forward       {0, 0, 1};
        cardinal::scene::Vec3 right         {1, 0, 0};
        cardinal::scene::Vec3 up            {0, 1, 0};
        float                 vfov_rad      {0.785f};   // 45°

        // Output buffers.
        cardinal::render::graph::ResourceHandle out_radiance;   // 3 * W * H floats
        cardinal::render::graph::ResourceHandle out_rgba;       // 4 * W * H bytes

        cardinal::u32 width  {0};
        cardinal::u32 height {0};

        // Stats published after each execute().
        float trace_us {0.0f};
        cardinal::u64 rays_total {0};
    };

    static cardinal::shared_ptr<State> add_to_graph(
        cardinal::render::graph::Graph& g,
        cardinal::shared_ptr<PathTracer> tracer,
        const Scene& scene,
        cardinal::u32 width, cardinal::u32 height);

    static const char* hlsl_source() noexcept;
};

}  // namespace cardinal::lighting::gpu
