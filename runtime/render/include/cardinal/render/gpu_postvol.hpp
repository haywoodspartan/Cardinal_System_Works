#pragma once

// =============================================================================
// Cardinal — TAA + Volumetric Fog (AEGIS Blocks 9 & 12).
//
// Two passes that close major gaps in the AEGIS spec:
//
//   * TAAPass — Temporal Anti-Aliasing. Pairs with MotionVectorPass:
//     reads current-frame HDR radiance, reprojects previous-frame
//     history through the motion-vector buffer, applies neighbourhood
//     colour clamping to suppress ghosting, and blends history with
//     current at a fixed alpha. Output is the temporally-accumulated
//     HDR radiance + a new history buffer for next frame. AEGIS
//     Block 12 → "TAA / History".
//
//   * VolumetricFogPass — froxel-based volumetric fog with single
//     scattering. Builds a 3D scattering volume (tiles_x × tiles_y ×
//     slice_count froxels) by evaluating the light list at each
//     froxel's world position, applying Henyey-Greenstein phase
//     function for the view direction, integrating front-to-back.
//     Per-pixel: sample the froxel at the surface's depth slice and
//     composite over the scene colour. AEGIS Block 9 → "Volumetrics /
//     Fog / Participating Media".
//
// Both follow the established graph::Pass convention: shared_ptr<State>,
// add_to_graph factory, NaN-defended record, HLSL source-string
// matching the CPU reference's algorithm.
// =============================================================================

#include <cardinal/render/graph.hpp>
#include <cardinal/core/types.hpp>

namespace cardinal::render::gpu {

// ---------------------------------------------------------------------------
// TAAPass — temporal accumulation
//
//   in_current : W * H * 3 floats (HDR linear — typically VBufResolve output)
//   in_history : W * H * 3 floats (previous frame's TAA output)
//   in_motion  : W * H × u32 (snorm16 ×2 packed motion vectors)
//   out_radiance : W * H * 3 floats (blended HDR — what downstream
//                  Tonemap consumes)
//   out_history  : W * H * 3 floats (copy of out_radiance for next frame's
//                  in_history — declared by the pass)
//
// Algorithm:
//   1. Decode motion vector → previous-pixel coordinate.
//   2. If previous pixel is offscreen → no history, use current.
//   3. Sample 3×3 neighbourhood of current frame, compute min/max RGB
//      "colour box". This is the classic Karis-Lottes neighbourhood
//      clamp — suppresses ghosting from re-emerging geometry.
//   4. Clamp the history sample to the colour box.
//   5. Blend: out = lerp(clamped_history, current, alpha).
//
// alpha = 0.0 → pure history (smoothest, most ghosting)
// alpha = 1.0 → pure current (sharpest, no accumulation)
// 0.1 (default) gives a 10-frame effective sample budget.
// ---------------------------------------------------------------------------
class TAAPass {
public:
    struct State {
        graph::ResourceHandle in_current;
        graph::ResourceHandle in_history;
        graph::ResourceHandle in_motion;
        graph::ResourceHandle out_radiance;
        graph::ResourceHandle out_history;
        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        float         blend_alpha {0.1f};
        // Stats:
        cardinal::u32 pixels_blended      {0};
        cardinal::u32 pixels_disoccluded  {0};   // history sample fell offscreen
        cardinal::u32 pixels_clamped      {0};   // history was outside neighbourhood box
    };

    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_current,
        graph::ResourceHandle in_history,
        graph::ResourceHandle in_motion,
        cardinal::u32 width, cardinal::u32 height,
        float blend_alpha = 0.1f);

    static const char* hlsl_source() noexcept;
};

// ---------------------------------------------------------------------------
// VolumetricFogPass — froxel grid single-scattering volumetric fog
//
// Standard "Volumetric Fog" technique (Frostbite SIGGRAPH 2014):
// build a sparse 3D volume in view-space (Z slices distributed
// exponentially to put more resolution near the camera), shade each
// froxel from the scene light list using the Henyey-Greenstein phase
// function, and integrate front-to-back along view rays.
//
// Output froxel layout: tiles_x * tiles_y * slice_count records of
// 4 floats each: (scattered_r, scattered_g, scattered_b, transmittance).
//
//   in_lights    : light_count * 16 floats (PackedLight)
//   in_depth     : W * H f32 (V-Buf depth — used as fog max distance)
//   in_view_proj : 16 floats (informational — froxel positions come from
//                  cfg + slice exponential mapping)
//   out_scattering : tiles_x*tiles_y*slice_count * 4 floats
//   out_fog_rgba   : W*H*4 bytes — per-pixel applied fog (RGBA, alpha =
//                    255 - transmittance). Host composites this over
//                    the scene colour via CompositePresentPass.
//
// Default config: 240×135 tiles × 64 slices for a 1920×1080 viewport
// (one froxel per 8×8 pixels). Density 0.05 / m, Henyey-Greenstein g = 0.7
// (forward-scattering — typical "morning fog" look).
// ---------------------------------------------------------------------------
constexpr cardinal::u32 kFogTileSize    = 8;     // pixels per froxel column
constexpr cardinal::u32 kFogSliceCount  = 64;    // depth slices
constexpr cardinal::u32 kFogFroxelFloats = 4;    // (r, g, b, transmittance)

class VolumetricFogPass {
public:
    struct State {
        graph::ResourceHandle in_lights;
        graph::ResourceHandle in_depth;
        graph::ResourceHandle in_view_proj;
        graph::ResourceHandle in_ambient;
        graph::ResourceHandle out_scattering;
        graph::ResourceHandle out_fog_rgba;
        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        cardinal::u32 tiles_x {0}, tiles_y {0};
        cardinal::u32 slice_count {kFogSliceCount};
        cardinal::u32 light_count {0};
        float         density    {0.05f};   // extinction per world-space unit
        float         anisotropy {0.7f};    // HG g, in [-1, 1]
        float         max_distance {200.0f};// fog volume far plane
        // Stats:
        cardinal::u32 froxels_lit       {0};
        cardinal::u32 pixels_in_fog     {0};
    };

    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_lights,
        graph::ResourceHandle in_depth,
        graph::ResourceHandle in_view_proj,
        graph::ResourceHandle in_ambient,
        cardinal::u32 width, cardinal::u32 height,
        cardinal::u32 light_count,
        float density = 0.05f,
        float anisotropy = 0.7f,
        float max_distance = 200.0f);

    static const char* hlsl_source() noexcept;
};

}  // namespace cardinal::render::gpu
