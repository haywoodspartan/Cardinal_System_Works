#pragma once

// =============================================================================
// Cardinal — tessellation math (Phase 6 baseline).
//
// Pure functions that compute hull-shader tessellation factors. Used both
// CPU-side (for LOD selection / instance culling) and copied verbatim into
// HLSL via shaders/include/cardinal_tessellation.hlsli — keeping one canonical
// implementation across CPU and GPU avoids the classic "looks different on
// the tess shader vs. the picker" bug.
//
// Three policies:
//   - distance       : factor = clamp(scale / camera_distance, lo, hi)
//                      Cheap, robust, doesn't need triangle data.
//   - edge_screen    : factor = max(2, edge_pixels / target_pixels_per_seg)
//                      Uniform on-screen triangle size — best quality
//                      for a fixed perf budget, but needs each edge length
//                      in pixels.
//   - phong_blend    : tess factor blended by surface curvature, computed
//                      from per-vertex normal divergence. Used for PN-tris.
//
// Hardware tessellation factors are clamped to [1, 64] by every modern GPU;
// we clamp to that range here so callers can't generate impossible factors.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <algorithm>
#include <cmath>

namespace cardinal::render::tess {

// Hardware ceiling — DX12 / Vulkan both cap tess factors at 64.
inline constexpr float kMaxFactor = 64.0f;
inline constexpr float kMinFactor = 1.0f;

inline float clamp_factor(float f) noexcept {
    if (f < kMinFactor) return kMinFactor;
    if (f > kMaxFactor) return kMaxFactor;
    return f;
}

// ---------------------------------------------------------------------------
// Distance policy — factor = scale / max(distance, 0.001f), clamped.
// `scale` is the world-space distance at which factor = 1. Pick it so that
// triangles at ~1 unit produce ~scale subdivisions.
// ---------------------------------------------------------------------------
inline float factor_distance(float camera_to_patch_distance,
                             float scale,
                             float min_factor = kMinFactor,
                             float max_factor = kMaxFactor) noexcept
{
    const float d = (camera_to_patch_distance > 0.001f)
                  ? camera_to_patch_distance : 0.001f;
    const float f = scale / d;
    return std::clamp(f, min_factor, max_factor);
}

// ---------------------------------------------------------------------------
// Edge-length / screen-space policy — produces uniform pixel-sized
// post-tessellation triangles. Pass each edge's length in pixels (which the
// hull shader can compute from clip-space-projected positions); we return
// the per-edge factor that yields target_pixels_per_segment-sized output.
// ---------------------------------------------------------------------------
inline float factor_edge(float edge_length_pixels,
                         float target_pixels_per_segment = 16.0f,
                         float min_factor = kMinFactor,
                         float max_factor = kMaxFactor) noexcept
{
    if (target_pixels_per_segment < 1.0f) target_pixels_per_segment = 1.0f;
    const float f = edge_length_pixels / target_pixels_per_segment;
    return std::clamp(f, min_factor, max_factor);
}

// ---------------------------------------------------------------------------
// PN-triangle blend factor for Phong tessellation. Smooths the surface by
// blending control points against vertex normals; the blend weight follows
// vertex-normal curvature.
// ---------------------------------------------------------------------------
inline float phong_blend_weight(float n_dot_n_neighbour) noexcept {
    // 0 when normals coincide (flat patch — no extra detail needed),
    // 1 when normals are perpendicular (sharp curve — full PN smoothing).
    const float k = 1.0f - std::abs(n_dot_n_neighbour);
    return std::clamp(k, 0.0f, 1.0f);
}

// ---------------------------------------------------------------------------
// LOD selection — pick a "tess quality" rung 0..3 from camera distance.
// Used by the editor knob ("Tessellation Quality: Off / Low / Medium / High")
// and by the hull shader to gate work.
// ---------------------------------------------------------------------------
enum class Quality : u32 { Off = 0, Low, Medium, High };

inline float quality_scale(Quality q) noexcept {
    switch (q) {
        case Quality::Off:    return 1.0f;     // no extra subdivision
        case Quality::Low:    return 8.0f;     // up to 8x at 1 unit
        case Quality::Medium: return 16.0f;
        case Quality::High:   return 32.0f;
    }
    return 1.0f;
}

}  // namespace cardinal::render::tess
