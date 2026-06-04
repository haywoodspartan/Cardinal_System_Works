#pragma once

// =============================================================================
// Cardinal — GPU pass library.
//
// Concrete passes that consume the render::graph framework. Each pass
// ships in two halves:
//   * a CPU reference impl that runs under graph::CpuBackend (matches
//     the engine's existing in-tree CPU path bit-for-bit so tests can
//     validate equivalence headlessly), and
//   * an HLSL shader source-string that compiles via DXC when the RHI
//     compute surface lands (the same convention render::algo uses).
//
// The pass + the graph are the structural answer to "make the rendering
// pipeline entirely GPU backed": instead of issuing per-frame work as
// ad-hoc CPU code that happens to call into vkCmdDraw at the end, every
// per-frame operation is a Pass node with explicit Read / Write resource
// declarations and a record() that lives behind the graph::Backend
// abstraction. CpuBackend runs it on the host today; RhiBackend will
// dispatch the HLSL kernel tomorrow — no change to the pass declaration.
//
// First two passes (this header):
//   * FrustumCullPass     — AABB-vs-6-plane reject; CPU ref calls into
//                           the engine's SIMD frustum_cull_aabbs.
//   * VertexTransformPass — per-vertex world position + normal under a
//                           per-instance model matrix. CPU ref mirrors
//                           the ForwardRenderer's Phase 2 transform.
//
// Both expose an `add_to_graph(...)` factory that wires the pass into a
// host's graph at declaration time. The pass owns a small user-context
// struct (kept inside an opaque cardinal::shared_ptr held by the host)
// so multiple instances coexist without globals.
// =============================================================================

#include <cardinal/render/graph.hpp>
#include <cardinal/core/types.hpp>

namespace cardinal::render::gpu {

// ---------------------------------------------------------------------------
// FrustumCullPass — input: SoA AABBs (one buffer with 6 contiguous arrays of
// `count` floats: min_x, min_y, min_z, max_x, max_y, max_z) + a planes
// buffer (24 floats, six { nx, ny, nz, d } records — matches the engine's
// cardinal::core::geom::Frustum::planes layout exactly).
// Output: a visibility-bits buffer of `count` bytes, 1 = visible, 0 = culled.
//
// The CPU reference under graph::CpuBackend calls into
// cardinal::core::simd::frustum_cull_aabbs (the engine's AVX-512 cull),
// so this pass is bit-for-bit identical to the existing ForwardRenderer
// cull path under the simulator.
// ---------------------------------------------------------------------------
class FrustumCullPass {
public:
    struct State {
        graph::ResourceHandle in_aabbs;     // 6 * count floats SoA
        graph::ResourceHandle in_planes;    // 24 floats
        graph::ResourceHandle out_bits;     // count bytes
        cardinal::u32         count {0};
        // Last-frame stats for editor inspection. Populated by record().
        cardinal::u32         visible{0};
        cardinal::u32         culled {0};
    };

    // Add this pass to graph `g`. Caller imports / declares `in_aabbs`
    // and `in_planes`; the pass declares + returns the `out_bits` handle.
    // Returns a shared State the host keeps so it can read stats later.
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_aabbs,
        graph::ResourceHandle in_planes,
        cardinal::u32 count);

    // HLSL source for the GPU compute kernel. Editor displays it in the
    // shader inspector; the RHI compute backend compiles it via DXC.
    // One thread = one AABB; group size 64.
    static const char* hlsl_source() noexcept;
};

// ---------------------------------------------------------------------------
// VertexTransformPass — input: local-space vertices SoA (6 contiguous
// arrays of `vertex_count` floats: x, y, z, nx, ny, nz) + a row-major
// model matrix (16 floats, 4×4 — same convention as cardinal::core::Mat4).
// Output: world-space vertices SoA (same layout as input). Normals are
// transformed by the matrix's upper-left 3×3 (uniform scale assumed; for
// non-uniform scale a follow-up pass produces the inverse-transpose).
//
// The CPU reference matches scene::ForwardRenderer's per-triangle
// transform inside Phase 2: world_pos = M * (x, y, z, 1); world_normal =
// normalize(rotate(M_3x3, (nx, ny, nz))).
// ---------------------------------------------------------------------------
class VertexTransformPass {
public:
    struct State {
        graph::ResourceHandle in_local;     // 6 * vertex_count floats SoA
        graph::ResourceHandle in_matrix;    // 16 floats, row-major
        graph::ResourceHandle out_world;    // 6 * vertex_count floats SoA
        cardinal::u32         vertex_count {0};
        cardinal::u32         vertices_written{0};   // stats
    };

    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_local,
        graph::ResourceHandle in_matrix,
        cardinal::u32 vertex_count);

    static const char* hlsl_source() noexcept;
};

}  // namespace cardinal::render::gpu
