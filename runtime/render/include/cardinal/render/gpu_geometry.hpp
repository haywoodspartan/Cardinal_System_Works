#pragma once

// =============================================================================
// Cardinal — Adaptive-precision geometry engine.
//
// The render module's existing precision.hpp gives us the format toolbox
// (FP32 / FP16 / FP8 E4M3+E5M2 / FP4 E2M1+E3M0 round-trip helpers). This
// header layers a *selection* and *math-division* engine on top:
//
//   1. GpuCapabilities — what precision tiers the device exposes. Hosts
//      populate this from rhi::Device caps when the RHI compute surface
//      lands; the CpuBackend tests synthesize the cap set directly.
//
//   2. GeometryTier — the engine-facing tier enum (Fp32 / Fp16 / Fp8 /
//      Fp4). Each tier carries TWO related properties at once:
//        * the per-component quantization precision (from precision.hpp), and
//        * a micro-triangle subdivision factor: 1 / 2 / 4 / 8.
//      Lower precision → more micro-triangles per source triangle in the
//      same shader-invocation budget. That's the "math-division" of the
//      polygon the user asked for: a single source triangle is split
//      into N quantized micro-triangles so the renderer expresses detail
//      at finer-than-source-triangle granularity, without paying for the
//      full FP32 vertex cost on every fragment.
//
//   3. select_tier(caps, max_tier) — picks the HIGHEST tier the device
//      supports, capped by `max_tier`. FP4 is the most aggressive
//      (8 micro-tris per input); falling back to FP32 always works
//      because every GPU runs FP32.
//
//   4. AdaptiveGeometryPass — a graph-hosted Pass that consumes an input
//      triangle buffer + a GpuCapabilities, picks the tier, subdivides
//      each triangle into N micro-triangles, and writes the (quantized
//      → dequantized) micro-triangle buffer. The Pass also publishes
//      the selected tier + per-tier quantization-error stats back to
//      the host through its State struct so the editor can show what
//      precision the GPU is actually running at.
//
// Subdivision pattern per tier:
//   * Fp32  — N = 1: emit (a, b, c) as-is, FP32 verts pass through.
//   * Fp16  — N = 2: longest-edge split. Pick the longest of (ab, bc, ca);
//             midpoint M; emit two children sharing M as their new vertex.
//   * Fp8   — N = 4: 4-1 split. Midpoints of all three edges, four
//             children: (a, Mab, Mca), (Mab, b, Mbc), (Mca, Mbc, c),
//             (Mab, Mbc, Mca). This is the classic Loop / Catmull-Clark
//             refinement step.
//   * Fp4   — N = 8: 4-1 split, then longest-edge split on each child.
//             8 micro-triangles per input, expressed in 4-bit components.
//
// All subdivision is deterministic + bit-stable: same input + same
// selected tier → same micro-triangle buffer. NaN inputs collapse to
// zero before subdivision so the persistent buffer can't be poisoned.
// =============================================================================

#include <cardinal/render/graph.hpp>
#include <cardinal/render/precision.hpp>
#include <cardinal/core/types.hpp>

namespace cardinal::render::gpu {

// ---------------------------------------------------------------------------
// PrecisionCaps — precision-tier capability descriptor for the
// math-division engine. Populated from rhi::Device caps (when the RHI
// compute surface lands) OR synthesized by the host for testing. FP32
// is always available; the booleans escalate from there.
//
// Named PrecisionCaps (not GpuCapabilities) so it doesn't collide with
// rhi::GpuCapabilities — different concept (rhi caps describe device
// features; PrecisionCaps describes which precision tiers the math-
// division engine can dispatch at). The two compose: a host queries
// rhi::Device::capabilities(), maps the relevant bits into a
// PrecisionCaps, and hands the latter to AdaptiveGeometryPass.
// ---------------------------------------------------------------------------
struct PrecisionCaps {
    bool fp16_supported {true};   // Shader 6.2 / Vulkan 1.2 — essentially universal
    bool fp8_supported  {false};  // Hopper, Ada Lovelace, RDNA4
    bool fp4_supported  {false};  // Blackwell, RDNA5

    // Per-input micro-triangle budget cap. Even with FP4 support, hosts
    // can cap the math-division when the dispatch budget is tight (e.g.
    // mobile thermal throttle, low-latency frames). Defaults to 8 (the
    // FP4 native count); set to 1 to force no subdivision regardless of
    // tier escalation.
    cardinal::u32 max_micro_triangles_per_input {8};
};

// Back-compat alias for code that pre-dates the rename. Will be removed
// once the AegisRenderPipeline integration that pairs PrecisionCaps with
// rhi::GpuCapabilities lands and every call site is updated.
using GpuCapabilities [[deprecated("Use PrecisionCaps")]] = PrecisionCaps;

// ---------------------------------------------------------------------------
// GeometryTier — paired (precision, subdivision factor).
// ---------------------------------------------------------------------------
enum class GeometryTier : cardinal::u32 {
    Fp32 = 0,
    Fp16,
    Fp8,
    Fp4,
};

struct TierProperties {
    precision::Format format;
    cardinal::u32     bits_per_component;       // 32, 16, 8, 4
    cardinal::u32     micro_triangles_per_input;// 1, 2, 4, 8
    float             quantization_error_bound; // worst-case |q(x) - x| for x ∈ [-1, 1]
    const char*       name;
};
TierProperties properties_of(GeometryTier tier) noexcept;
const char*    tier_name    (GeometryTier tier) noexcept;

// Pick the highest tier the device supports, capped by `max_tier` AND by
// caps.max_micro_triangles_per_input. Returns GeometryTier::Fp32 if
// nothing else fits — FP32 is always available.
GeometryTier select_tier(const PrecisionCaps& caps,
                         GeometryTier max_tier = GeometryTier::Fp4) noexcept;

// ---------------------------------------------------------------------------
// AdaptiveGeometryPass
//   Input  : `in_triangles`            M triangles, 9 floats each (3 verts xyz)
//   Output : `out_micro_tris`          M * N * 9 floats (N from selected tier)
//            `out_count`               1 × u32 (total micro-triangles emitted)
//
// The output is stored as dequantized FP32 floats for downstream
// compatibility (the rasterizer + cull passes consume FP32). Vertices
// are first quantized to the tier's format then immediately dequantized,
// so the output IS what the GPU would see after the lower-precision
// math step. The full-precision micro-triangle count + cumulative
// quantization error are published in State for editor inspection.
// ---------------------------------------------------------------------------
class AdaptiveGeometryPass {
public:
    struct State {
        graph::ResourceHandle in_triangles;
        graph::ResourceHandle out_micro_tris;
        graph::ResourceHandle out_count;
        GeometryTier  selected_tier              {GeometryTier::Fp32};
        cardinal::u32 input_triangle_count       {0};
        cardinal::u32 micro_triangles_per_input  {1};
        // Stats:
        cardinal::u32 micro_triangles_emitted    {0};
        float         total_quantization_error   {0.0f};  // L1-sum across all output verts
    };

    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_triangles,
        cardinal::u32 input_triangle_count,
        const PrecisionCaps& caps,
        GeometryTier max_tier = GeometryTier::Fp4);

    static const char* hlsl_source() noexcept;
};

}  // namespace cardinal::render::gpu
