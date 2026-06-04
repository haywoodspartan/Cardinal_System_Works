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
// PlaneSpec — subdivided XZ-plane grid centred at world origin (or offset
// via center_*). Produces sx * sy * 2 triangles (two per grid cell), CCW
// when viewed from +Y. The high-subdivision case (sx=sy=16+) is what the
// polygon viewport's "rainbow floor" is built from.
// ---------------------------------------------------------------------------
struct PlaneSpec {
    float         half_size_x {1.0f};
    float         half_size_z {1.0f};
    cardinal::u32 subdivisions_x {1};
    cardinal::u32 subdivisions_z {1};
    float         center_x{0}, center_y{0}, center_z{0};
};

class PlanePrimitivePass {
public:
    struct State {
        graph::ResourceHandle out_tris;
        cardinal::u32 triangle_count {0};
        PlaneSpec     spec;
    };
    static cardinal::shared_ptr<State> add_to_graph(graph::Graph& g, PlaneSpec spec);
    static const char* hlsl_source() noexcept;
};

// ---------------------------------------------------------------------------
// PyramidSpec — 4-triangle tetrahedron (apex + 3 sides) OR 6-triangle
// square-base pyramid (apex + 4 sides + 2-tri base) depending on `square_base`.
// ---------------------------------------------------------------------------
struct PyramidSpec {
    float         base_half_extent {1.0f};
    float         height           {1.5f};
    bool          square_base      {true};   // false = tetrahedron, true = square pyramid
    float         center_x{0}, center_y{0}, center_z{0};
};

class PyramidPrimitivePass {
public:
    struct State {
        graph::ResourceHandle out_tris;
        cardinal::u32 triangle_count {0};
        PyramidSpec   spec;
    };
    static cardinal::shared_ptr<State> add_to_graph(graph::Graph& g, PyramidSpec spec);
    static const char* hlsl_source() noexcept;
};

// ---------------------------------------------------------------------------
// PolygonViewportPass — the "polygon viewport" the editor uses to inspect
// triangle topology. For each input triangle:
//   1. Project all 3 verts through the 4×4 view-proj.
//   2. Assign a distinct colour from a deterministic hash of (triangle_id
//      + color_seed) — splitmix32 → HSV(hue, 1, 1) → RGB. Every triangle
//      gets a visually distinct flat colour, so the user sees exactly
//      what the math-division engine produced at the active tier.
//   3. Edge-function inside-test + barycentric interpolation, depth-test
//      with closest-z wins, sRGB-clamped u8 output.
//
// The host pairs this with WireframePass when both fill + edges are wanted
// (the typical editor "polygon" view-mode draws PolygonViewportPass first,
// then composites WireframePass on top with line_r/g/b = (0, 0, 0)).
//
// In: M triangles (9 floats each, world-space) + 4×4 view-proj.
// Out: width × height colour + depth.
// ---------------------------------------------------------------------------
class PolygonViewportPass {
public:
    struct State {
        graph::ResourceHandle in_tris;
        graph::ResourceHandle in_matrix;
        graph::ResourceHandle out_color;
        graph::ResourceHandle out_depth;
        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        cardinal::u32 triangle_count {0};
        cardinal::u32 color_seed {0};
        // Stats:
        cardinal::u32 fragments_drawn        {0};
        cardinal::u32 triangles_drawn        {0};
        cardinal::u32 triangles_offscreen    {0};
    };

    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_tris,
        graph::ResourceHandle in_matrix,
        cardinal::u32 width, cardinal::u32 height,
        cardinal::u32 triangle_count,
        cardinal::u32 color_seed = 0);

    static const char* hlsl_source() noexcept;
};

// ---------------------------------------------------------------------------
// WorldLabelPass — projects world-space anchor points through a view-proj
// matrix and emits per-anchor screen-space records the host UI (ImGui /
// editor panel) consumes to render a floating "(x, y, z)" text tag at that
// pixel position. This is the engine-side half of the editor's vertex-
// coordinate tooltip: the projection + visibility + occlusion checks live
// in the graph; the actual glyph rasterisation is the host's job.
//
// For each input anchor (world.xyz + label_id) the pass writes one
// LabelOut record:
//   { screen_x_px : f32, screen_y_px : f32, depth_ndc01 : f32,
//     visible    : u32  (1 = on-screen + in front of camera, 0 = cull),
//     label_id   : u32 }  → 5 dwords / 20 bytes / record
//
// Optionally, when draw_anchors is on, the pass declares its own RGBA8
// out_color framebuffer and stamps a small 3×3 anchor crosshair into it
// at each visible projected anchor — the editor uses this to see where
// the text will land before the UI layer composes it on top (matches
// the "show me where the label anchors" debug mode in the reference
// image). out_color is declared by the pass, NOT supplied by the host.
// ---------------------------------------------------------------------------
constexpr cardinal::u32 kWorldLabelRecordFloats = 5;   // sx, sy, depth, visible, label_id

class WorldLabelPass {
public:
    struct State {
        graph::ResourceHandle in_world_points;   // 3 * N floats SoA (x, y, z)
        graph::ResourceHandle in_label_ids;      // N × u32 (caller-defined)
        graph::ResourceHandle in_matrix;         // 16 floats row-major
        graph::ResourceHandle out_records;       // N × kWorldLabelRecordFloats * 4 bytes
        graph::ResourceHandle out_color;         // optional — width*height*4 (anchor markers)
        cardinal::u32         label_count        {0};
        cardinal::u32         width              {0};
        cardinal::u32         height             {0};
        bool                  draw_anchors       {true};
        float                 anchor_r{1.0f}, anchor_g{1.0f}, anchor_b{1.0f};
        // Stats:
        cardinal::u32         visible_count      {0};
        cardinal::u32         culled_count       {0};
    };

    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_world_points,
        graph::ResourceHandle in_label_ids,
        graph::ResourceHandle in_matrix,
        cardinal::u32 label_count,
        cardinal::u32 width, cardinal::u32 height,
        bool draw_anchors = true);

    static const char* hlsl_source() noexcept;
};

// ---------------------------------------------------------------------------
// WorldAxisGizmoPass — draws the world-space X / Y / Z axes from a chosen
// origin as colored line segments (X = red, Y = green, Z = blue) into a
// color + depth framebuffer. The classic editor axis tripod. Length is
// per-axis; depth-tested so the gizmo correctly composites behind nearer
// geometry. This is the cyan-axis-edge sort of marker visible in the
// reference image — the engine-side debug overlay every polygon viewport
// ships with.
// ---------------------------------------------------------------------------
class WorldAxisGizmoPass {
public:
    struct State {
        graph::ResourceHandle in_matrix;        // 16 floats view-proj
        graph::ResourceHandle out_color;
        graph::ResourceHandle out_depth;
        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        // Origin + per-axis length in world units.
        float origin_x{0}, origin_y{0}, origin_z{0};
        float length_x{1.0f}, length_y{1.0f}, length_z{1.0f};
        // Stats:
        cardinal::u32 axis_pixels_drawn {0};
    };

    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_matrix,
        cardinal::u32 width, cardinal::u32 height,
        float origin_x = 0, float origin_y = 0, float origin_z = 0,
        float length_x = 1.0f, float length_y = 1.0f, float length_z = 1.0f);

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
