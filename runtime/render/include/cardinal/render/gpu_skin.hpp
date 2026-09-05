#pragma once
// =============================================================================
// Cardinal — SkinPass: GPU linear-blend skinning as a render-graph compute
// pass.
//
// One thread per vertex. Reads the mesh's bind-pose vertex stream, the
// per-vertex joint influences and the entity's joint palette (global *
// inverse_bind, one Mat4 per joint) and writes the deformed vertex stream
// in the SAME 44-byte scene::Vertex layout, so anything downstream that
// consumes a rigid vertex buffer (the vis-buffer VertexTransformPass, the
// forward raster path, the AEGIS meshlet builder) can consume a skinned one
// without knowing.
//
// The CPU record() is scene::skin::skin_vertices verbatim — CpuBackend runs
// the exact reference, RhiBackend runs hlsl_source(), and tests/skin pins
// them against each other (and, on a GPU box, against the real dispatch).
//
// Buffer layouts (byte-addressed; little-endian):
//   in_verts      : vertex_count × 44  {pos f3, nrm f3, col f3, uv f2}
//   in_influences : vertex_count × 24  {u16 joint[4], f32 weight[4]}
//   in_palette    : joint_count  × 64  column-major Mat4 (m[col][row])
//   out_verts     : vertex_count × 44  (same layout as in_verts)
// =============================================================================
#include <cardinal/render/graph.hpp>
#include <cardinal/core/types.hpp>

namespace cardinal::render::gpu {

class SkinPass {
public:
    struct State {
        graph::ResourceHandle in_verts;
        graph::ResourceHandle in_influences;
        graph::ResourceHandle in_palette;
        graph::ResourceHandle out_verts;
        cardinal::u32 vertex_count{0};
        cardinal::u32 joint_count {0};
    };
    // Declares out_verts (transient, vertex_count × 44 B) and adds the pass.
    // Returns null when any input handle is invalid or vertex_count == 0.
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_verts,
        graph::ResourceHandle in_influences,
        graph::ResourceHandle in_palette,
        cardinal::u32 vertex_count,
        cardinal::u32 joint_count);
    static const char* hlsl_source() noexcept;

    static constexpr cardinal::u32 kVertexBytes    = 44;
    static constexpr cardinal::u32 kInfluenceBytes = 24;
    static constexpr cardinal::u32 kMat4Bytes      = 64;
    static constexpr cardinal::u32 kThreadsPerGroup = 64;
};

}  // namespace cardinal::render::gpu
