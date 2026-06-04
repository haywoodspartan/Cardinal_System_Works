#pragma once

// =============================================================================
// Cardinal — Visibility Buffer + Hi-Z occlusion (AEGIS pipeline blocks 4 & 6).
//
// The AEGIS spec's "Visibility Buffer (Compact)" + "Hi-Z Occlusion" + the
// downstream "ReSTIR DI / GI" all rotate around two passes:
//
//   1. VisibilityBufferPass  — replaces the classic G-Buffer write. Per
//      pixel the rasterizer emits 6 compact channels — the minimum info
//      needed to RECONSTRUCT every shading attribute later via a compute
//      pass that resolves (PrimID, MatID, barycentrics) → world pos /
//      normal / material params on demand. Memory bandwidth drops ~5×
//      vs a fat G-Buffer; the trade is one extra fetch in the resolver.
//
//        layout per pixel (24 bytes = 6 dwords, all SoA-paged buffers):
//           depth          : f32   — closest-z (NDC [0, 1])
//           primitive_id   : u32   — index into the triangle stream
//           material_id    : u32   — index into the material palette
//           normal_packed  : u32   — oct-encoded world normal (2× 16-bit)
//           motion_vector  : u32   — (Δx, Δy) packed as 2× snorm16
//           aux            : u32   — user-defined (cluster id, layer, etc.)
//
//   2. HiZBuildPass — builds a depth pyramid from the visibility buffer's
//      depth channel. Each successively coarser mip stores the MAXIMUM
//      depth in its 2×2 footprint (conservative-far reduction) so the
//      occlusion-test pass can sample one tap and reject any AABB whose
//      nearest-z is behind the mip's max-z. This is the well-known
//      Pyramid-Z / Hi-Z construction used by every GPU-driven renderer
//      since Frostbite 1.0; the AEGIS spec lists it under Visibility &
//      Culling Pipeline → Hi-Z Occlusion.
//
// Both passes are graph::Pass nodes, NaN-defended at every public read
// site, deterministic given the same input, with HLSL source-strings
// shipped alongside the CPU reference for the eventual RhiBackend
// compute dispatch.
// =============================================================================

#include <cardinal/render/graph.hpp>
#include <cardinal/core/types.hpp>

namespace cardinal::render::gpu {

// ---------------------------------------------------------------------------
// VisibilityBufferPass
//   in_tris       : M triangles, world-space (9 floats / tri)
//   in_mat_ids    : M material indices (one per triangle, u32)
//   in_matrix     : 16 floats, row-major view-proj
//
//   out_depth     : W * H × f32 — closest-z per pixel (cleared to 1.0)
//   out_prim_id   : W * H × u32 — winning triangle ID per pixel (kInvalidPrim = miss)
//   out_mat_id    : W * H × u32 — winning material ID per pixel (kInvalidMat  = miss)
//   out_normal    : W * H × u32 — oct-packed world-space normal of winning tri
//   out_motion    : W * H × u32 — per-pixel motion vector (snorm16 ×2)
//   out_aux       : W * H × u32 — caller-defined auxiliary u32 per winning tri
//
// The pass is invariant under triangle reorderings as long as IDs remain
// stable: same scene + same camera + same prev-frame state → identical
// V-Buf bytes (used as the regression-pin invariant in the tests).
// ---------------------------------------------------------------------------
constexpr cardinal::u32 kInvalidPrimId = 0xFFFFFFFFu;
constexpr cardinal::u32 kInvalidMatId  = 0xFFFFFFFFu;

class VisibilityBufferPass {
public:
    struct State {
        graph::ResourceHandle in_tris;
        graph::ResourceHandle in_mat_ids;
        graph::ResourceHandle in_matrix;
        graph::ResourceHandle in_aux_per_tri;   // M × u32 (auxiliary channel; nullable handle)

        graph::ResourceHandle out_depth;
        graph::ResourceHandle out_prim_id;
        graph::ResourceHandle out_mat_id;
        graph::ResourceHandle out_normal;
        graph::ResourceHandle out_motion;       // currently always zero — host wires
                                                // prev-MVP to fill in a follow-up commit
        graph::ResourceHandle out_aux;

        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        cardinal::u32 triangle_count {0};
        // Stats:
        cardinal::u32 fragments_drawn        {0};
        cardinal::u32 triangles_offscreen    {0};
    };

    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_tris,
        graph::ResourceHandle in_mat_ids,
        graph::ResourceHandle in_matrix,
        graph::ResourceHandle in_aux_per_tri,    // pass {} for no aux channel
        cardinal::u32 width, cardinal::u32 height,
        cardinal::u32 triangle_count);

    static const char* hlsl_source() noexcept;
};

// ---------------------------------------------------------------------------
// HiZBuildPass — builds a power-of-two-step max-depth pyramid from a
// W × H input depth buffer. The output is a single contiguous buffer
// holding mip 0 (W × H), mip 1 (W/2 × H/2), … down to a 1 × 1 root mip.
//
// Layout (SoA per mip, contiguous):
//   [mip 0: W*H f32] [mip 1: (W/2)*(H/2) f32] … [mip N: 1 f32]
//
// State::mip_offsets gives the byte offset of each mip in the buffer so
// downstream consumers (HiZ-occlusion-test) can sample any level.
// ---------------------------------------------------------------------------
constexpr cardinal::u32 kMaxHiZMips = 16;   // log2(65536) — covers any sane viewport

class HiZBuildPass {
public:
    struct State {
        graph::ResourceHandle in_depth;        // W * H f32
        graph::ResourceHandle out_pyramid;     // sum-of-mips f32
        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        cardinal::u32 mip_count {0};
        // Per-mip layout (filled in at add_to_graph time).
        cardinal::u32 mip_widths   [kMaxHiZMips] {};
        cardinal::u32 mip_heights  [kMaxHiZMips] {};
        cardinal::u32 mip_offsets_f[kMaxHiZMips] {};   // in floats, NOT bytes
        // Stats:
        cardinal::u32 mips_built {0};
        float         max_depth  {0.0f};               // root mip's value (entire scene's farthest)
    };

    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_depth,
        cardinal::u32 width, cardinal::u32 height);

    static const char* hlsl_source() noexcept;
};

// ---------------------------------------------------------------------------
// HiZOcclusionTestPass — tests N world-space AABBs against the depth
// pyramid built by HiZBuildPass.
//
// Per AABB:
//   1. Project all 8 corners through view-proj → take screen-space bounds
//      (min_x, min_y, max_x, max_y) + nearest depth.
//   2. Pick the mip level whose footprint encloses the AABB's screen rect
//      (roughly log2 of the rect's max dimension).
//   3. Sample max-depth at that mip; if AABB's nearest-depth is greater
//      (= farther), the AABB is fully occluded → visible = 0.
//
// Output: N u8s, 1 = visible, 0 = occluded. Same encoding as the existing
// FrustumCullPass output for drop-in chaining (frustum → Hi-Z → cluster).
// ---------------------------------------------------------------------------
class HiZOcclusionTestPass {
public:
    struct State {
        graph::ResourceHandle in_aabbs;        // 6*N floats SoA (min_x, min_y, min_z, max_x, max_y, max_z)
        graph::ResourceHandle in_matrix;       // 16 floats view-proj
        graph::ResourceHandle in_pyramid;      // HiZBuildPass::out_pyramid
        graph::ResourceHandle out_visibility;  // N × u8

        // Pyramid descriptor — populated by add_to_graph from the
        // upstream HiZBuildPass::State.
        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        cardinal::u32 mip_count {0};
        cardinal::u32 mip_widths   [kMaxHiZMips] {};
        cardinal::u32 mip_heights  [kMaxHiZMips] {};
        cardinal::u32 mip_offsets_f[kMaxHiZMips] {};

        cardinal::u32 aabb_count {0};
        // Stats:
        cardinal::u32 visible {0};
        cardinal::u32 occluded {0};
    };

    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_aabbs,
        graph::ResourceHandle in_matrix,
        cardinal::u32 aabb_count,
        const HiZBuildPass::State& hiz);

    static const char* hlsl_source() noexcept;
};

}  // namespace cardinal::render::gpu
