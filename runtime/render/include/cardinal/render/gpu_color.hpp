#pragma once

// =============================================================================
// Cardinal — Color Grading + Depth of Field (AEGIS Block 12).
//
// Two more post-processing passes filling the remaining Block 12 gaps:
//
//   * ColorGradingPass — 3D LUT lookup with trilinear interpolation.
//     Industry-standard color grading: the artist authors a 16³ /
//     32³ / 64³ texture mapping input RGB → output RGB. Per pixel the
//     pass quantizes input color into the LUT grid, fetches 8 neighbours,
//     trilinear-interpolates. Block 12 → "Color Grading".
//
//   * DepthOfFieldPass — circle-of-confusion (CoC) weighted Gaussian
//     blur using the V-Buf depth channel. Per pixel, |depth - focal_z|
//     drives the blur radius; pixels at the focal plane stay sharp,
//     pixels at far / near extremes blur out. Gather-style sample
//     pattern (no need for separable two-pass filtering on the CPU
//     reference). Block 12 → "Depth of Field".
// =============================================================================

#include <cardinal/render/graph.hpp>
#include <cardinal/core/types.hpp>

namespace cardinal::render::gpu {

// ---------------------------------------------------------------------------
// ColorGradingPass
//
//   in_color  : W * H * 4 bytes RGBA8 (post-tonemap LDR is typical)
//   in_lut    : lut_size^3 * 3 floats (RGB triples, row-major in
//               (b * lut_size + g) * lut_size + r order)
//   out_color : W * H * 4 bytes RGBA8 (graded)
//
// LUT sizes: 16, 32, 64 typical. Larger = more memory, finer gradient.
// Identity LUT (input → input) is the no-op; the editor's "grading off"
// switch wires this.
// ---------------------------------------------------------------------------
class ColorGradingPass {
public:
    struct State {
        graph::ResourceHandle in_color;
        graph::ResourceHandle in_lut;
        graph::ResourceHandle out_color;
        cardinal::u32 width    {0};
        cardinal::u32 height   {0};
        cardinal::u32 lut_size {32};
        // Stats:
        cardinal::u32 pixels_graded {0};
    };

    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_color,
        graph::ResourceHandle in_lut,
        cardinal::u32 width, cardinal::u32 height,
        cardinal::u32 lut_size = 32);

    static const char* hlsl_source() noexcept;
};

// ---------------------------------------------------------------------------
// DepthOfFieldPass
//
//   in_color : W * H * 4 bytes RGBA8
//   in_depth : W * H f32 (V-Buf depth — NDC [0, 1])
//   out_color : W * H * 4 bytes RGBA8 (CoC-blurred)
//
// CoC mapping:
//   focal_z       — depth value at which pixels stay sharp [0, 1]
//   focal_range   — depth band around focal_z that's also sharp
//   max_blur_px   — pixel radius at the maximum CoC
//
// CoC(depth) = clamp((|depth - focal_z| - focal_range) / coc_falloff, 0, 1)
//   * max_blur_px
//
// Per pixel: 16-sample gather in a disk of radius CoC. Samples are
// uniformly distributed over an angular ring + centre. Output is the
// weighted-average colour.
// ---------------------------------------------------------------------------
class DepthOfFieldPass {
public:
    struct State {
        graph::ResourceHandle in_color;
        graph::ResourceHandle in_depth;
        graph::ResourceHandle out_color;
        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        float focal_z       {0.5f};
        float focal_range   {0.1f};
        float coc_falloff   {0.4f};
        float max_blur_px   {8.0f};
        cardinal::u32 sample_count {16};
        // Stats:
        cardinal::u32 pixels_blurred       {0};
        cardinal::u32 pixels_in_focus      {0};
    };

    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_color,
        graph::ResourceHandle in_depth,
        cardinal::u32 width, cardinal::u32 height,
        float focal_z = 0.5f,
        float focal_range = 0.1f,
        float coc_falloff = 0.4f,
        float max_blur_px = 8.0f);

    static const char* hlsl_source() noexcept;
};

}  // namespace cardinal::render::gpu
