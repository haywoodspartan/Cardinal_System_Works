#pragma once

// =============================================================================
// Cardinal — GPU primitive builders + wireframe viewport.
//
// Closes the loop between AdaptiveGeometryPass and what the editor actually
// renders: every primitive the engine exposes (Sphere, Cube — and the
// triangle-buffer entrypoint for imported meshes) now ships as a
// graph::Pass that emits an FP32 world-space triangle stream, ready to
// hand straight to AdaptiveGeometryPass. The pipeline becomes:
//
//   SpherePrimitivePass / CubePrimitivePass / [host-imported triangles]
//                       ↓
//             AdaptiveGeometryPass   (tier-selected, micro-divides each tri)
//                       ↓
//                 WireframePass      (per-edge Bresenham line rasterizer)
//                       ↓
//            color + depth framebuffer in the editor viewport
//
// WireframePass IS the wireframe / polygon viewport. The editor toggles
// between "wireframe" (just the edge pass output) and "polygon" (raster
// fill + wireframe overlay on top); under either view the user can see
// exactly how the math-division engine subdivided the primitive at the
// active GeometryTier — FP32 sphere shows the source triangulation, FP4
// shows 8× as many micro-triangle edges. Same source primitive, same
// camera, different tier → visibly more triangle resolution.
//
// All passes follow the graph-pass convention: NaN-defended at every read
// site, deterministic given the same input, HLSL source-string shipped
// alongside the CPU reference for the eventual RhiBackend compute path.
// =============================================================================

#include <cardinal/render/graph.hpp>
#include <cardinal/core/types.hpp>

namespace cardinal::render::gpu {

// ---------------------------------------------------------------------------
// SphereSpec — UV-sphere parameters. Triangle count = 2 * lat * lon.
// ---------------------------------------------------------------------------
struct SphereSpec {
    float         radius            {1.0f};
    cardinal::u32 latitude_segments {16};   // horizontal rings (excl. poles)
    cardinal::u32 longitude_segments{32};   // vertical wedges
    float         center_x{0}, center_y{0}, center_z{0};
};

class SpherePrimitivePass {
public:
    struct State {
        graph::ResourceHandle out_tris;
        cardinal::u32 triangle_count {0};
        SphereSpec    spec;
    };
    static cardinal::shared_ptr<State> add_to_graph(graph::Graph& g, SphereSpec spec);
    static const char* hlsl_source() noexcept;
};

// ---------------------------------------------------------------------------
// CubeSpec — axis-aligned cube. 6 faces × 2 triangles each = 12 triangles.
// ---------------------------------------------------------------------------
struct CubeSpec {
    float half_extent {1.0f};
    float center_x{0}, center_y{0}, center_z{0};
};

class CubePrimitivePass {
public:
    struct State {
        graph::ResourceHandle out_tris;
        cardinal::u32 triangle_count {12};
        CubeSpec      spec;
    };
    static cardinal::shared_ptr<State> add_to_graph(graph::Graph& g, CubeSpec spec);
    static const char* hlsl_source() noexcept;
};

// ---------------------------------------------------------------------------
// WireframePass — projects 3D world-space triangles through a 4×4 view-proj
// matrix, then rasterizes each edge as a Bresenham line into a color +
// depth framebuffer. Lines per triangle = 3 (ab, bc, ca).
//
//   in_tris    : M * 9 floats (3 verts xyz, world-space)
//   in_matrix  : 16 floats, row-major 4×4 view-proj
//   out_color  : W * H * 4 bytes (RGBA8). Pixels not touched by any edge
//                remain at their cleared value (0,0,0,255).
//   out_depth  : W * H floats (closest-edge depth wins per pixel).
//
// The Bresenham step matches the CPU reference in HLSL byte-for-byte so
// the wireframe overlay looks identical on the GPU when the RhiBackend
// compute surface lands.
// ---------------------------------------------------------------------------
class WireframePass {
public:
    struct State {
        graph::ResourceHandle in_tris;
        graph::ResourceHandle in_matrix;
        graph::ResourceHandle out_color;
        graph::ResourceHandle out_depth;
        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        cardinal::u32 triangle_count {0};
        float line_r {1.0f}, line_g {1.0f}, line_b {1.0f};
        // Stats:
        cardinal::u32 pixels_drawn         {0};
        cardinal::u32 edges_drawn          {0};
        cardinal::u32 triangles_offscreen  {0};
    };

    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_tris,
        graph::ResourceHandle in_matrix,
        cardinal::u32 width, cardinal::u32 height,
        cardinal::u32 triangle_count,
        float line_r = 1.0f, float line_g = 1.0f, float line_b = 1.0f);

    static const char* hlsl_source() noexcept;
};

}  // namespace cardinal::render::gpu
