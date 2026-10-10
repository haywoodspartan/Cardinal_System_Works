#pragma once

// =============================================================================
// Cardinal — Screen-Space Global Illumination + Ambient Occlusion
// (AEGIS Block 8 → "Indirect / Final Gather", screen rung).
//
// The first rung of a Lumen-style final gather: before any world-space probe
// cache or distance-field cone trace, gather what the SCREEN already knows.
// For every shaded pixel the pass marches a small set of screen-space
// directions through the V-Buffer depth, reconstructs each sample's world
// position with the same inverse view-proj the resolve used, and treats every
// sample that rises above the pixel's tangent plane (within a world-space
// radius) as
//   (a) an OCCLUDER of the flat ambient term the resolve added, and
//   (b) an EMITTER of one bounce of the direct radiance the resolve already
//       computed for it (colour bleed).
// Both effects are weighted by ssgi_sample_weight(...) — the single function
// that decides the look (falloff with distance, cosine lobe, thickness).
//
// Why this slice first: it needs no acceleration structure, no history and no
// extra scene upload; it is ONE dispatch over buffers the resolve already
// produced; and it is the half of Lumen's final gather (screen traces) that
// gives contact darkening + bounce colour — the two cues most responsible for
// the grounded "Luminous" look. The world-space fallback (probe cache / SDF
// cones) for samples that leave the screen slots in behind it — see
// docs/rendering_roadmap.md.
//
// Determinism contract (same as every gpu_* pass): the CPU record and the
// HLSL kernel are operation-for-operation mirrors. Sample positions are
// INTEGER pixel offsets (compass-direction table × integer step length +
// integer per-pixel jitter) so CPU and GPU march the exact same pixels; the
// only float work is reconstruction + dot products (harness EpsPbr compare).
// Tangent-plane rejection carries a bias so co-planar samples (cos θ ≈ 0, a
// 1-ulp knife-edge on every flat facet) reject identically on both sides.
//
// Output = in_radiance + albedo·ambient·(ao − 1) + albedo·(1 − metal)·bounce,
// i.e. the resolve's flat ambient is re-weighted by AO and the gathered
// bounce is added as diffuse. Sky pixels pass through untouched.
// =============================================================================

#include <cardinal/render/graph.hpp>
#include <cardinal/core/types.hpp>

namespace cardinal::render::gpu {

constexpr cardinal::u32 kSsgiDirections  = 8;      // compass directions (integer deltas)
constexpr cardinal::u32 kSsgiMaxSteps    = 8;      // per direction
constexpr float         kSsgiTangentBias = 0.05f;  // min cos θ above the tangent plane

struct SsgiParams {
    float         radius       {1.0f};   // world-space gather radius
    float         intensity    {1.0f};   // bounce (indirect diffuse) scale
    float         ao_strength  {1.0f};   // 0 = no darkening, 1 = full
    cardinal::u32 steps        {4};      // 1..kSsgiMaxSteps samples per direction
    cardinal::u32 base_step_px {2};      // pixel stride between successive steps
};

// Per-sample weight: how strongly a sample at world distance `dist`
// (0 < dist <= radius) whose direction makes `cos_theta` (> kSsgiTangentBias)
// with the surface normal occludes the ambient term and contributes its
// radiance as bounce. Returns >= 0; 1 means "a full tap". The CPU reference
// calls this directly; the HLSL kernel carries a verbatim port, so the two
// must be kept in step (see ScreenSpaceGIPass::hlsl_source).
float ssgi_sample_weight(float dist, float cos_theta, float radius) noexcept;

class ScreenSpaceGIPass {
public:
    struct State {
        graph::ResourceHandle in_depth;          // W*H f32 (V-Buf depth, [0,1])
        graph::ResourceHandle in_prim_id;        // W*H u32 (kInvalidPrimId = sky)
        graph::ResourceHandle in_normal;         // W*H u32 oct snorm16x2
        graph::ResourceHandle in_mat_id;         // W*H u32
        graph::ResourceHandle in_materials;      // material_count * 8 f32
        graph::ResourceHandle in_radiance;       // W*H*3 f32 (VBufResolve output)
        graph::ResourceHandle in_inv_view_proj;  // 16 f32, row-major (same as resolve)
        graph::ResourceHandle in_ambient;        // 3 f32 — the flat term the resolve added
        graph::ResourceHandle out_radiance;      // W*H*3 f32 (HDR linear)
        graph::ResourceHandle out_ao;            // W*H f32, 1 = unoccluded (debug / stats)

        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        cardinal::u32 material_count {0};
        SsgiParams    params;

        // Stats (published after each execute):
        cardinal::u32 pixels_shaded {0};
        cardinal::u32 pixels_sky    {0};
        cardinal::u64 taps_accepted {0};   // samples in range AND above the tangent plane
        float         mean_ao       {1.0f};
    };

    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_depth,
        graph::ResourceHandle in_prim_id,
        graph::ResourceHandle in_normal,
        graph::ResourceHandle in_mat_id,
        graph::ResourceHandle in_materials,
        graph::ResourceHandle in_radiance,
        graph::ResourceHandle in_inv_view_proj,
        graph::ResourceHandle in_ambient,
        cardinal::u32 width, cardinal::u32 height,
        cardinal::u32 material_count,
        const SsgiParams& params = {});

    static const char* hlsl_source() noexcept;
};

}  // namespace cardinal::render::gpu
