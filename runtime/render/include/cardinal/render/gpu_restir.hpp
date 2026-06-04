#pragma once

// =============================================================================
// Cardinal — ReSTIR Direct Illumination (AEGIS Block 7 → "ReSTIR DI + GI
// Unified" Key Innovation).
//
// ReSTIR = Reservoir-based Spatial-Temporal Importance Resampling. The
// classic GI-2020 SIGGRAPH paper algorithm: each pixel maintains a small
// "reservoir" that contains ONE chosen sample plus the running statistics
// needed to keep its selection weighted-uniform over an arbitrarily-large
// candidate set. By blending reservoirs across spatial neighbours +
// temporal history, the cost-per-pixel is constant while the effective
// number of light samples reaches the thousands.
//
// This module ships the three passes that make up the DI pipeline:
//
//   1. ReSTIRSamplePass     — per pixel, draw M candidates from the
//                              light list. Target function = BRDF × NdotL ×
//                              radiance (no visibility yet — that's the
//                              cheap part of "cheap target + expensive
//                              validate"). Output: initial reservoir.
//
//   2. ReSTIRSpatialPass    — per pixel, sample K reservoirs from spatial
//                              neighbours within a radius. Combine with
//                              geometric similarity test (depth + normal
//                              must agree) so reservoirs don't bleed
//                              across silhouettes.
//
//   3. ReSTIRTemporalPass   — per pixel, reproject through the motion
//                              vector to find last-frame's reservoir at
//                              this surface. Combine with the temporally-
//                              reprojected version, clamping the history
//                              M-count to avoid temporal lag on disocclusion.
//
// The final pixel radiance is obtained by reading reservoir.chosen_light,
// evaluating the full Cook-Torrance BRDF × visibility × light radiance
// once, and dividing by the reservoir's target_pdf. The shading happens
// inside VBufResolvePass (which now optionally reads the reservoir
// buffer instead of looping every light).
//
// Reservoir layout: 16 bytes / pixel = 4 dwords:
//   chosen_light : u32   — index into the light list (kInvalidLight = none)
//   w_sum        : f32   — sum of weights seen so far (Σ w_i)
//   m_count      : f32   — number of candidates seen (M; float so RIS
//                          combining can fractionally weight history)
//   target_pdf   : f32   — target function value at the chosen sample
//
// The CPU references are bit-stable; HLSL source-strings sketch the GPU
// implementation matching the established convention. NaN-defended at
// every public read site.
// =============================================================================

#include <cardinal/render/graph.hpp>
#include <cardinal/core/types.hpp>

namespace cardinal::render::gpu {

constexpr cardinal::u32 kReservoirBytes  = 16;   // 4 dwords / pixel
constexpr cardinal::u32 kReservoirDwords = 4;
constexpr cardinal::u32 kInvalidLight    = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// ReSTIRSamplePass — initial-candidate pass.
//
//   in_world_pos   : 3 * W * H floats (reconstructed world position;
//                    host wires from V-Buf depth + inv-VP)
//   in_world_normal: 3 * W * H floats (from V-Buf normal_packed,
//                    oct-unpacked by the host or by an upstream pass)
//   in_lights      : light_count × 16 floats (PackedLight)
//   in_seeds       : W * H × u32 — per-pixel PRNG seed (host typically
//                    seeds from frame_index * pixel_hash)
//   out_reservoirs : W * H × kReservoirBytes
// ---------------------------------------------------------------------------
class ReSTIRSamplePass {
public:
    struct State {
        graph::ResourceHandle in_world_pos;
        graph::ResourceHandle in_world_normal;
        graph::ResourceHandle in_lights;
        graph::ResourceHandle in_seeds;
        graph::ResourceHandle out_reservoirs;
        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        cardinal::u32 light_count {0};
        cardinal::u32 candidates_per_pixel {32};   // M
        // Stats:
        cardinal::u32 pixels_sampled  {0};
        cardinal::u32 pixels_with_valid_pick {0};
    };
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_world_pos,
        graph::ResourceHandle in_world_normal,
        graph::ResourceHandle in_lights,
        graph::ResourceHandle in_seeds,
        cardinal::u32 width, cardinal::u32 height,
        cardinal::u32 light_count,
        cardinal::u32 candidates_per_pixel = 32);
    static const char* hlsl_source() noexcept;
};

// ---------------------------------------------------------------------------
// ReSTIRSpatialPass — spatial neighbour reuse.
//
//   in_reservoirs : current frame's reservoirs (output of Sample)
//   in_world_pos / in_world_normal : geometric-similarity inputs
//   in_seeds : per-pixel PRNG seed (re-used or re-seeded)
//   out_reservoirs : combined reservoirs
//
// Each thread reads its own reservoir + K_NEIGHBORS spatial samples
// drawn from a (radius_px × radius_px) disk centred on the pixel.
// Neighbours that fail the geometric similarity check (|Δdepth| > eps
// or n·n' < 0.9) are skipped. The reservoir-combine math is the
// standard ReSTIR RIS weight-sum-then-renormalize.
// ---------------------------------------------------------------------------
class ReSTIRSpatialPass {
public:
    struct State {
        graph::ResourceHandle in_reservoirs;
        graph::ResourceHandle in_world_pos;
        graph::ResourceHandle in_world_normal;
        graph::ResourceHandle in_seeds;
        graph::ResourceHandle out_reservoirs;
        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        cardinal::u32 neighbour_count {5};
        float         radius_px       {16.0f};
        // Stats:
        cardinal::u32 reservoirs_combined {0};
        cardinal::u32 neighbours_rejected {0};
    };
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_reservoirs,
        graph::ResourceHandle in_world_pos,
        graph::ResourceHandle in_world_normal,
        graph::ResourceHandle in_seeds,
        cardinal::u32 width, cardinal::u32 height,
        cardinal::u32 neighbour_count = 5,
        float radius_px = 16.0f);
    static const char* hlsl_source() noexcept;
};

// ---------------------------------------------------------------------------
// ReSTIRTemporalPass — temporal-history reuse via motion vectors.
//
//   in_reservoirs       : current-frame reservoirs (output of Spatial)
//   in_prev_reservoirs  : previous-frame reservoirs
//   in_motion           : V-Buf motion vectors (snorm16 ×2 packed u32)
//   in_world_normal     : current-frame normals (for disocclusion test)
//   in_prev_world_normal: previous-frame normals (same test)
//   out_reservoirs      : temporally-combined reservoirs
//
// History M-count is clamped at kMaxHistoryM so a single bad sample
// can't dominate forever. Disocclusion (normal flipped or depth mis-
// match) → history reservoir discarded for that pixel.
// ---------------------------------------------------------------------------
constexpr float kMaxHistoryM = 20.0f;

class ReSTIRTemporalPass {
public:
    struct State {
        graph::ResourceHandle in_reservoirs;
        graph::ResourceHandle in_prev_reservoirs;
        graph::ResourceHandle in_motion;
        graph::ResourceHandle in_world_normal;
        graph::ResourceHandle in_prev_world_normal;
        graph::ResourceHandle out_reservoirs;
        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        float         max_history_m {kMaxHistoryM};
        // Stats:
        cardinal::u32 pixels_with_history  {0};
        cardinal::u32 pixels_disoccluded   {0};
    };
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_reservoirs,
        graph::ResourceHandle in_prev_reservoirs,
        graph::ResourceHandle in_motion,
        graph::ResourceHandle in_world_normal,
        graph::ResourceHandle in_prev_world_normal,
        cardinal::u32 width, cardinal::u32 height,
        float max_history_m = kMaxHistoryM);
    static const char* hlsl_source() noexcept;
};

}  // namespace cardinal::render::gpu
