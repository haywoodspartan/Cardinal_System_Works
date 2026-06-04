#pragma once

// =============================================================================
// Cardinal — GPU rasterization + polygon ops.
//
// Three passes that round out the engine's "express every per-frame op as
// a graph node" goal for 2D / 2.5D primitives:
//
//   * TriangleRasterPass    — software-style scan-conversion of post-
//                              transform triangles into a (color + depth)
//                              framebuffer. Useful for occlusion-buffer
//                              prepass + low-resolution debug overlays
//                              where the rasterizer needs to run on the
//                              same compute queue as the rest of the
//                              pipeline (no roundtrip through the raster
//                              pipeline's render-pass machinery).
//
//   * PolygonClipPass       — Sutherland-Hodgman convex polygon clip
//                              against one plane. UI / decals / clipped
//                              geometry all need this; running it on the
//                              GPU lets the host avoid the
//                              CPU-roundtrip per clip.
//
//   * PolygonTriangulatePass — Fan-triangulation of a convex polygon
//                              into triangles ready for raster. Pairs
//                              with PolygonClipPass: clip → triangulate
//                              → raster as a single graph chain.
//
// CPU references for all three are exact reproductions of the standard
// textbook algorithms (deterministic). HLSL source for each is embedded
// in hlsl_source() and matches the CPU reference output byte-for-byte
// — same fixed-point tie-break for raster edge tests, same first-vertex
// anchor for fan triangulation, same inside-test for SH clipping.
// =============================================================================

#include <cardinal/render/graph.hpp>
#include <cardinal/core/types.hpp>

namespace cardinal::render::gpu {

// ---------------------------------------------------------------------------
// TriangleRasterPass — input:
//   * `in_tris`  : N triangles, layout (struct-of-arrays per-vertex):
//                  [px_v0[0..N), py_v0[0..N), depth_v0[0..N),
//                   r_v0,        g_v0,        b_v0,
//                   px_v1, py_v1, depth_v1, r_v1, g_v1, b_v1,
//                   px_v2, py_v2, depth_v2, r_v2, g_v2, b_v2]
//                  → 18 × N floats. (px, py) are pixel-space coordinates
//                  in [0, width) × [0, height). depth ∈ [0, 1].
// Outputs (declared by the pass; handles returned in State):
//   * out_color  : width × height bytes × 4 (RGBA8, sRGB-encoded)
//   * out_depth  : width × height × 1 float (closest-z wins)
// ---------------------------------------------------------------------------
class TriangleRasterPass {
public:
    struct State {
        graph::ResourceHandle in_tris;
        graph::ResourceHandle out_color;     // width * height * 4 bytes
        graph::ResourceHandle out_depth;     // width * height * 4 bytes
        cardinal::u32         width    {0};
        cardinal::u32         height   {0};
        cardinal::u32         triangle_count {0};
        // Last-frame stats for editor inspection.
        cardinal::u32         fragments_drawn  {0};
        cardinal::u32         fragments_depth_killed {0};
        cardinal::u32         triangles_offscreen {0};
    };

    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_tris,
        cardinal::u32 width, cardinal::u32 height,
        cardinal::u32 triangle_count);

    static const char* hlsl_source() noexcept;
};

// ---------------------------------------------------------------------------
// PolygonClipPass — Sutherland-Hodgman convex polygon clipping against
// one plane. Input + output buffers are SoA: 3 contiguous arrays of
// floats (x, y, z) — one per vertex.
//
// `max_output_count` caps how much output the pass can produce; for a
// convex polygon SH can add at most one vertex per clip plane (so for
// a single-plane clip, max_output_count = input_count + 1 is sufficient).
// ---------------------------------------------------------------------------
class PolygonClipPass {
public:
    struct State {
        graph::ResourceHandle in_verts;        // 3 * input_count floats SoA
        graph::ResourceHandle in_plane;        // 4 floats (nx, ny, nz, d)
        graph::ResourceHandle out_verts;       // 3 * max_output_count floats SoA
        graph::ResourceHandle out_count;       // 1 × u32
        cardinal::u32         input_count       {0};
        cardinal::u32         max_output_count  {0};
        // Last clip's actual output count (host-readable via out_count buffer).
        cardinal::u32         produced          {0};
    };

    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_verts,
        graph::ResourceHandle in_plane,
        cardinal::u32 input_count,
        cardinal::u32 max_output_count);

    static const char* hlsl_source() noexcept;
};

// ---------------------------------------------------------------------------
// PolygonTriangulatePass — fan triangulation of a convex polygon.
// Input: N vertices (SoA x, y, z). Output: (N-2) triangles, AoS
// (each triangle = 9 floats = 3 verts × {x, y, z}).
// ---------------------------------------------------------------------------
class PolygonTriangulatePass {
public:
    struct State {
        graph::ResourceHandle in_verts;        // 3 * input_count floats SoA
        graph::ResourceHandle out_tris;        // 9 * (input_count - 2) floats AoS
        cardinal::u32         input_count {0};
        cardinal::u32         triangles_produced {0};
    };

    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_verts,
        cardinal::u32 input_count);

    static const char* hlsl_source() noexcept;
};

}  // namespace cardinal::render::gpu
