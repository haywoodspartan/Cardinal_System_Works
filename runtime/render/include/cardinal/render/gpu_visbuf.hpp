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

// ---------------------------------------------------------------------------
// TriangleBinPass — screen-tile binning + triangle setup for the V-Buffer.
//
// The reference rasterizer is O(pixels × triangles): every pixel thread
// walks the whole triangle stream. This pass turns that into
// O(pixels × triangles-in-my-tile):
//
//   clear  : one thread per tile  → bin_counts[tile] = 0
//   bin    : one thread per tri   → project + set up ONCE (screen-space
//            verts, 1/area, bbox, material / oct-normal / aux) into a 64-byte
//            setup record, then append the triangle id to every kVisTileSize²
//            tile its bbox touches (InterlockedAdd on the tile's count).
//
// Fixed-capacity bins keep the buffer size static for the persistent graph:
// a tile whose count exceeds `bin_capacity` is marked OVERFLOWED and the
// raster kernel falls back to walking every setup record for that tile —
// slower, never wrong. Bin order is non-deterministic on the GPU (atomics),
// so the binned raster resolves winners by (depth, then LOWER triangle id),
// which is exactly what the reference's ascending-order strict-< scan
// produces — the V-Buffer stays byte-identical whichever path runs.
//
// Setup record (16 dwords):
//   [0..5] px0 py0 px1 py1 px2 py2   [6..8] pz0 pz1 pz2   [9] inv_area
//   [10] flags (bit0 valid, bit1 area>0)  [11] mat_id  [12] oct normal
//   [13] aux  [14] bbox min (x0 | y0<<16)  [15] bbox max, exclusive (x1 | y1<<16)
// ---------------------------------------------------------------------------
constexpr cardinal::u32 kVisTileSize        = 16;   // px per bin tile (square)
constexpr cardinal::u32 kDefaultBinCapacity = 256;  // tri ids per tile before overflow
constexpr cardinal::u32 kSetupRecordDwords  = 16;
// Setup records are stored SoAoV — Structure of Arrays of Vectors: four
// uint4/float4 PLANES, each npad × 16 B, npad = triangle count rounded up to
// EVEN so every plane starts 32-byte aligned. A cooperative load where thread
// t fetches record t then issues four fully-contiguous 16-byte-per-lane
// loads (512 B per 32-lane wave, one cache-line-aligned transaction each)
// instead of striding 64 B between lanes. Plane k of record t sits at byte
// (k * npad + t) * 16.
constexpr cardinal::u32 kSetupPlanes        = 4;
inline cardinal::usize setup_npad(cardinal::u32 tri_count) noexcept {
    return (static_cast<cardinal::usize>(tri_count) + 1u) & ~static_cast<cardinal::usize>(1u);
}
// Perfect block dimensions + grid-stride loops for the one-thread-per-element
// kernels: 256-thread blocks (8 waves of 32 / 4 of 64 — full occupancy on
// every desktop GPU) and a capped grid where each thread strides
// `groups × 256` over the element range, so a 10-million-element dispatch
// costs 4096 resident blocks, not 40 000 tiny ones.
constexpr cardinal::u32 kBlockThreads       = 256;
constexpr cardinal::u32 kGridStrideMaxGroups = 4096;
inline cardinal::u32 grid_stride_groups(cardinal::u32 n) noexcept {
    const cardinal::u32 g = (n + kBlockThreads - 1) / kBlockThreads;
    return g == 0 ? 1u : (g > kGridStrideMaxGroups ? kGridStrideMaxGroups : g);
}

class TriangleBinPass {
public:
    struct State {
        graph::ResourceHandle in_tris;          // M × 9 f32 world-space
        graph::ResourceHandle in_mat_ids;       // M × u32
        graph::ResourceHandle in_matrix;        // 16 f32 row-major VP
        graph::ResourceHandle in_aux_per_tri;   // optional M × u32
        graph::ResourceHandle in_vis_per_group; // optional u8 per group of vis_group_size tris: 0 = skip
        graph::ResourceHandle out_setup;        // M × 64 B setup records
        graph::ResourceHandle out_bin_counts;   // tiles × u32 (raw append count)
        graph::ResourceHandle out_bins;         // tiles × bin_capacity × u32 tri ids
        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        cardinal::u32 triangle_count {0};
        cardinal::u32 tiles_x {0}, tiles_y {0};
        cardinal::u32 bin_capacity {kDefaultBinCapacity};
        cardinal::u32 vis_group_size {64};  // tris per visibility byte (meshlet size)
        // Stats (CPU record):
        cardinal::u32 tris_culled    {0};   // skipped by the visibility gate (never set up)
        cardinal::u32 tris_binned    {0};   // passed setup, appended to ≥1 tile
        cardinal::u32 tris_rejected  {0};   // behind / offscreen / degenerate
        cardinal::u32 bin_entries    {0};   // total (tri, tile) pairs
        cardinal::u32 overflow_tiles {0};   // tiles whose count exceeded capacity
        cardinal::u32 tiles_rejected {0};   // bbox tiles skipped by the conservative edge test
    };
    // Adds TWO passes (clear, bin). `bin_capacity` is clamped to ≥ 1.
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_tris,
        graph::ResourceHandle in_mat_ids,
        graph::ResourceHandle in_matrix,
        graph::ResourceHandle in_aux_per_tri,    // pass {} for no aux channel
        cardinal::u32 width, cardinal::u32 height,
        cardinal::u32 triangle_count,
        cardinal::u32 bin_capacity = kDefaultBinCapacity,
        // Visibility gate: u8 per group of `vis_group_size` consecutive
        // triangles (MeshletCullPass / HiZOcclusionTestPass output over
        // MeshletBuildPass's fixed-size meshlets). A 0 skips the group's
        // triangles entirely — no setup, no binning (all-zero records keep
        // the overflow walk honest). Pass {} to bin everything.
        graph::ResourceHandle in_vis_per_group = {},
        cardinal::u32 vis_group_size = 64);
    static const char* hlsl_clear_source() noexcept;   // TriangleBinClearPass kernel
    static const char* hlsl_source() noexcept;         // TriangleBinPass kernel
};

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
        // Binned inputs (all valid ⇔ `binned`): from TriangleBinPass.
        graph::ResourceHandle in_setup;
        graph::ResourceHandle in_bin_counts;
        graph::ResourceHandle in_bins;

        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        cardinal::u32 triangle_count {0};
        cardinal::u32 bin_capacity {0};
        cardinal::u32 tiles_x {0};
        bool          binned {false};
        // Stats:
        cardinal::u32 fragments_drawn        {0};   // brute: accepted fragments (incl. overwrites); binned: covered pixels
        cardinal::u32 triangles_offscreen    {0};   // brute: behind/offscreen; binned: all setup-rejected tris
        cardinal::u32 overflow_tiles         {0};   // binned path: tiles that fell back to every record
    };

    // Reference (brute) raster: per pixel, walk every triangle.
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_tris,
        graph::ResourceHandle in_mat_ids,
        graph::ResourceHandle in_matrix,
        graph::ResourceHandle in_aux_per_tri,    // pass {} for no aux channel
        cardinal::u32 width, cardinal::u32 height,
        cardinal::u32 triangle_count);
    // Binned raster: per pixel, walk only the triangles binned into its
    // kVisTileSize² tile (falls back to every setup record on overflow).
    // `bins` must have been added to the same graph for the same W/H/M.
    // Output bytes are identical to the reference form.
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        const TriangleBinPass::State& bins,
        graph::ResourceHandle in_aux_per_tri,    // same handle given to the bin pass
        cardinal::u32 width, cardinal::u32 height,
        cardinal::u32 triangle_count);

    static const char* hlsl_source() noexcept;          // reference kernel
    static const char* hlsl_source_binned() noexcept;   // binned kernel
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
    struct State;
    // One graph pass per mip level; each pass's user context names its mip.
    struct MipCtx { State* st{nullptr}; cardinal::u32 mip{0}; };
    struct State {
        graph::ResourceHandle in_depth;        // W * H f32
        graph::ResourceHandle out_pyramid;     // sum-of-mips f32
        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        cardinal::u32 mip_count {0};
        // Per-mip layout (filled in at declare time).
        cardinal::u32 mip_widths   [kMaxHiZMips] {};
        cardinal::u32 mip_heights  [kMaxHiZMips] {};
        cardinal::u32 mip_offsets_f[kMaxHiZMips] {};   // in floats, NOT bytes
        bool          persistent {false};              // pyramid survives across executes
        cardinal::vector<MipCtx> mip_ctx;              // stable per-pass contexts
        // Stats:
        cardinal::u32 mips_built {0};
        float         max_depth  {0.0f};               // root mip's value (entire scene's farthest)
    };

    // Two-step form. declare() sizes the pyramid + mip table so a consumer
    // that must run BEFORE this frame's build (the occlusion test against
    // LAST frame's pyramid) can be wired first; add_passes() appends the
    // per-mip reduction passes once the depth source exists. `persistent`
    // keeps the pyramid across executes (graph::BufferDesc::persistent).
    static cardinal::shared_ptr<State> declare(graph::Graph& g,
                                               cardinal::u32 width, cardinal::u32 height,
                                               bool persistent = false);
    static void add_passes(graph::Graph& g, State& st, graph::ResourceHandle in_depth);

    // One-step form: declare + add_passes.
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_depth,
        cardinal::u32 width, cardinal::u32 height,
        bool persistent = false);

    static const char* hlsl_source() noexcept;
};

// ---------------------------------------------------------------------------
// HiZOcclusionTestPass — per bounding box: screen-rect frustum reject with the
// CURRENT view-proj, then (when enabled and the host says the pyramid is
// valid) an occlusion test against the pyramid using the view-proj that BUILT
// it — last frame's. Bounds come either as SoA AABBs or straight from
// MeshletBuildPass records (bounds at floats [2..7] of each 12-float record).
//
// Occlusion per box: project the 8 corners, take the screen rect + nearest
// depth, pick the mip whose texel footprint covers the rect, read the max
// depth there; nearer-than-max = visible. A corner behind the camera keeps
// the box visible (conservative). Optional `prior` visibility (u8 per box,
// e.g. from MeshletCullPass) short-circuits boxes already culled.
//
// Output: N u8 (buffer rounded up to a u32), 1 = visible, 0 = culled — the
// MeshletCullPass encoding, so it can gate TriangleBinPass directly.
// ---------------------------------------------------------------------------
class HiZOcclusionTestPass {
public:
    enum class Layout : cardinal::u32 { AabbSoA = 0, MeshletRecords = 1 };
    struct State {
        graph::ResourceHandle in_aabbs;         // SoA 6*N floats, or N x 12-float meshlet records
        graph::ResourceHandle in_matrix;        // 16 floats view-proj — CURRENT frame (frustum rect)
        graph::ResourceHandle in_prev_matrix;   // 16 floats view-proj that built the pyramid
        graph::ResourceHandle in_pyramid;       // HiZBuildPass::out_pyramid
        graph::ResourceHandle in_prior_vis;     // optional N u8: 0 = already culled
        graph::ResourceHandle in_history_valid; // optional 1 u32: 0 = pyramid not valid yet
        graph::ResourceHandle in_layout;        // imported mip table (widths, heights, offsets)
        graph::ResourceHandle out_visibility;   // N x u8 (rounded up to 4 bytes)
        Layout        layout {Layout::AabbSoA};
        bool          occlusion {true};         // false = frustum reject only
        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        cardinal::u32 mip_count {0};
        cardinal::u32 mip_widths   [kMaxHiZMips] {};
        cardinal::u32 mip_heights  [kMaxHiZMips] {};
        cardinal::u32 mip_offsets_f[kMaxHiZMips] {};
        cardinal::u32 layout_words [3 * kMaxHiZMips] {};   // import source for in_layout
        cardinal::u32 aabb_count {0};
        // Stats:
        cardinal::u32 visible {0};
        cardinal::u32 occluded {0};
        cardinal::u32 frustum_culled {0};
        cardinal::u32 prior_culled {0};
        bool          history_used {false};     // the pyramid was sampled this execute
    };

    // SoA AABBs, no prior visibility, pyramid always trusted (the same graph
    // builds it before the test reads it).
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_aabbs,
        graph::ResourceHandle in_matrix,
        cardinal::u32 aabb_count,
        const HiZBuildPass::State& hiz);

    // Full form. An invalid `in_history_valid` means the pyramid is trusted;
    // `occlusion` false means frustum reject only (the pyramid is never read).
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        Layout layout,
        graph::ResourceHandle in_bounds,
        cardinal::u32 count,
        graph::ResourceHandle in_matrix,
        graph::ResourceHandle in_prev_matrix,
        graph::ResourceHandle in_prior_vis,
        graph::ResourceHandle in_history_valid,
        const HiZBuildPass::State& hiz,
        bool occlusion = true);

    static const char* hlsl_source() noexcept;
};

}  // namespace cardinal::render::gpu
