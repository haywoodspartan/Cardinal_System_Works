#pragma once

// =============================================================================
// Cardinal — AEGIS Render Pipeline 2.0 passes + top-level orchestrator.
//
// This header completes the AEGIS spec coverage: the passes here fill in
// the boxes the engine doesn't have yet, and AegisPipeline at the bottom
// is the orchestrator that wires every existing + new pass into a single
// graph::Graph in the order the AEGIS diagram specifies — top to bottom:
//
//   1. CPU / Game Layer       (host-driven, not a graph node)
//   2. Frame Graph            (render::graph)
//   3. Virtual Geometry 2.0   (GeometryClassifyPass + MeshletBuildPass +
//                              MeshletCullPass + ScreenSpaceErrorPass)
//   4. Visibility & Culling   (FrustumCullPass + HiZBuild/OcclusionTest
//                              + DrawIndirectGenPass)
//   5. Geometry & Raster      (VertexTransformPass + TriangleRasterPass +
//                              AdaptiveGeometryPass for math-division)
//   6. Visibility Buffer      (VisibilityBufferPass)
//   7. Lighting & Ray Tracing (ForwardShadePass + PathTracePass +
//                              TiledLightCullPass + VBufResolvePass)
//   8. Material & Shading     (handled inside ForwardShadePass /
//                              VBufResolvePass via the material palette)
//   9. Volumetrics & Transparency (postfx::Chain Bloom + future fog)
//  10. Async Compute Scheduler (graph topology + the existing physics
//                               + particles GPU passes)
//  11. Streaming & IO         (host-driven, not a graph node)
//  12. Post Processing        (MotionVectorPass + TonemapPass + postfx)
//  13. Output & Presentation  (CompositePresentPass)
//  14. Data & Precision       (gpu_geometry GeometryTier engine)
//  15. RTX & GPU HW Blocks    (handled via the RhiBackend when compute lands)
//
// Each new pass follows the established convention: shared_ptr<State>,
// add_to_graph() factory, hlsl_source() stub, NaN-defended record(). The
// orchestrator is a thin builder — hosts construct an AegisPipeline,
// hand it scene inputs + a Graph, and AegisPipeline::build() declares
// every necessary pass in order.
// =============================================================================

#include <cardinal/render/graph.hpp>
#include <cardinal/render/gpu_passes.hpp>
#include <cardinal/render/gpu_raster.hpp>
#include <cardinal/render/gpu_geometry.hpp>
#include <cardinal/render/gpu_primitives.hpp>
#include <cardinal/render/gpu_visbuf.hpp>
#include <cardinal/core/types.hpp>

namespace cardinal::render::gpu {

// ===========================================================================
// Block 3 — Virtual Geometry 2.0
// ===========================================================================

enum class GeometryClass : cardinal::u32 {
    Planar          = 0,   // flat surfaces — analytic patches + FP8 residuals
    HardSurface     = 1,   // curved hard-surface — FP16 meshlet verts
    Organic         = 2,   // organic — FP16 meshlet verts, normal compression
    Displaced       = 3,   // displaced — FP32 control points + FP8 displacement
    Foliage         = 4,   // foliage/hair — alpha-tested, special LOD
};

// GeometryClassifyPass — assigns each input triangle one GeometryClass.
// CPU reference uses a simple heuristic on the triangle's planarity +
// surface curvature; production uses authored metadata + ML classifier.
class GeometryClassifyPass {
public:
    struct State {
        graph::ResourceHandle in_tris;        // M * 9 floats
        graph::ResourceHandle in_curvature;   // M floats (optional; nullable)
        graph::ResourceHandle out_class;      // M u32
        cardinal::u32 triangle_count {0};
        // Stats: tris per class.
        cardinal::u32 class_counts[5] {};
    };
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_tris,
        graph::ResourceHandle in_curvature,
        cardinal::u32 triangle_count);
    static const char* hlsl_source() noexcept;
};

// MeshletBuildPass — partitions a triangle stream into meshlets of
// kMaxTrisPerMeshlet triangles. Each meshlet record is 12 floats:
//   first_tri (u32), tri_count (u32), bounds_min (3 floats),
//   bounds_max (3 floats), normal_cone_axis (3 floats), normal_cone_cutoff (f32)
constexpr cardinal::u32 kMaxTrisPerMeshlet = 64;
constexpr cardinal::u32 kMeshletRecordFloats = 12;

class MeshletBuildPass {
public:
    struct State {
        graph::ResourceHandle in_tris;
        graph::ResourceHandle out_meshlets;
        graph::ResourceHandle out_meshlet_count;   // 1 u32
        cardinal::u32 triangle_count {0};
        cardinal::u32 meshlet_count {0};
    };
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_tris,
        cardinal::u32 triangle_count);
    static const char* hlsl_source() noexcept;
};

// MeshletCullPass — per meshlet: frustum reject (bounds vs view-proj) +
// backface reject (normal cone vs view direction). Output u8/meshlet
// visibility flag.
class MeshletCullPass {
public:
    struct State {
        graph::ResourceHandle in_meshlets;
        graph::ResourceHandle in_matrix;         // view-proj
        graph::ResourceHandle in_camera_dir;     // 3 floats
        graph::ResourceHandle out_visibility;    // N u8
        cardinal::u32 meshlet_count {0};
        cardinal::u32 visible_count {0};
        cardinal::u32 frustum_culled {0};
        cardinal::u32 backface_culled {0};
    };
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_meshlets,
        graph::ResourceHandle in_matrix,
        graph::ResourceHandle in_camera_dir,
        cardinal::u32 meshlet_count);
    static const char* hlsl_source() noexcept;
};

// ScreenSpaceErrorPass — computes screen-space size of each meshlet's
// bounding sphere and selects LOD level (0 = highest detail). Output:
// u32 per meshlet = chosen LOD.
class ScreenSpaceErrorPass {
public:
    struct State {
        graph::ResourceHandle in_meshlets;
        graph::ResourceHandle in_matrix;        // view-proj
        graph::ResourceHandle out_lod;          // N u32
        cardinal::u32 meshlet_count {0};
        cardinal::u32 viewport_height {1080};
        float error_threshold_px {2.0f};
        // Stats — count per LOD level.
        cardinal::u32 lod_counts[8] {};
    };
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_meshlets,
        graph::ResourceHandle in_matrix,
        cardinal::u32 meshlet_count,
        cardinal::u32 viewport_height = 1080);
    static const char* hlsl_source() noexcept;
};

// ===========================================================================
// Block 4 — Draw Indirect Generation (compacts visibility list into a
// dense indirect-draw command buffer the rasterizer's indirect dispatch
// path consumes).
// ===========================================================================

struct IndirectDrawCmd {
    cardinal::u32 first_triangle;
    cardinal::u32 triangle_count;
    cardinal::u32 meshlet_id;
    cardinal::u32 _pad;
};
static_assert(sizeof(IndirectDrawCmd) == 16, "IndirectDrawCmd must be 16 bytes");

class DrawIndirectGenPass {
public:
    struct State {
        graph::ResourceHandle in_visibility;     // N u8
        graph::ResourceHandle in_meshlets;       // N * kMeshletRecordFloats f32
        graph::ResourceHandle out_commands;      // up to N IndirectDrawCmd
        graph::ResourceHandle out_count;         // 1 u32
        cardinal::u32 meshlet_count {0};
        cardinal::u32 commands_emitted {0};
    };
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_visibility,
        graph::ResourceHandle in_meshlets,
        cardinal::u32 meshlet_count);
    static const char* hlsl_source() noexcept;
};

// ===========================================================================
// Block 7 — Tiled Light Culling
// ===========================================================================

// 2D tiled light culling: divides the screen into 16×16 tiles, each tile
// gets a list of light indices that affect it. Output: per-tile counts +
// concatenated index arrays.
constexpr cardinal::u32 kLightTileSize = 16;
constexpr cardinal::u32 kMaxLightsPerTile = 64;

class TiledLightCullPass {
public:
    struct State {
        graph::ResourceHandle in_lights;         // N * 16 floats (PackedLight)
        graph::ResourceHandle in_depth;          // W * H f32 (from V-Buf)
        graph::ResourceHandle in_matrix;         // inv view-proj
        graph::ResourceHandle out_tile_lights;   // tiles * kMaxLightsPerTile u32
        graph::ResourceHandle out_tile_counts;   // tiles u32
        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        cardinal::u32 light_count {0};
        cardinal::u32 tiles_x {0}, tiles_y {0};
        // Stats:
        cardinal::u32 total_light_assignments {0};
        cardinal::u32 truncated_tiles {0};
    };
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_lights,
        graph::ResourceHandle in_depth,
        graph::ResourceHandle in_matrix,
        cardinal::u32 width, cardinal::u32 height,
        cardinal::u32 light_count);
    static const char* hlsl_source() noexcept;
};

// ===========================================================================
// Block 7 — V-Buffer Resolve (samples the V-Buf, reconstructs world pos /
// normal / material params, computes shading)
// ===========================================================================

class VBufResolvePass {
public:
    struct State {
        graph::ResourceHandle in_depth;          // V-Buf
        graph::ResourceHandle in_prim_id;
        graph::ResourceHandle in_mat_id;
        graph::ResourceHandle in_normal;
        graph::ResourceHandle in_materials;      // 8 floats per material (base + emissive + metallic + roughness)
        graph::ResourceHandle in_tile_lights;    // tile light index lists
        graph::ResourceHandle in_tile_counts;
        graph::ResourceHandle in_lights;         // PackedLight array
        graph::ResourceHandle in_ambient;        // 3 floats
        graph::ResourceHandle out_radiance;      // W * H * 3 floats (HDR linear)
        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        cardinal::u32 material_count {0};
        cardinal::u32 light_count {0};
        // Stats:
        cardinal::u32 pixels_shaded {0};
        cardinal::u32 pixels_sky    {0};
    };
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_depth,
        graph::ResourceHandle in_prim_id,
        graph::ResourceHandle in_mat_id,
        graph::ResourceHandle in_normal,
        graph::ResourceHandle in_materials,
        graph::ResourceHandle in_tile_lights,
        graph::ResourceHandle in_tile_counts,
        graph::ResourceHandle in_lights,
        graph::ResourceHandle in_ambient,
        cardinal::u32 width, cardinal::u32 height,
        cardinal::u32 material_count,
        cardinal::u32 light_count);
    static const char* hlsl_source() noexcept;
};

// ===========================================================================
// Block 12 — Post Processing
// ===========================================================================

// MotionVectorPass — packs (sx, sy) delta into V-Buf's motion channel
// from a (current, previous) view-proj pair. Per pixel: project the
// reconstructed world position through both matrices, take the screen-
// space delta, encode as snorm16 pair into u32.
class MotionVectorPass {
public:
    struct State {
        graph::ResourceHandle in_depth;
        graph::ResourceHandle in_view_proj;          // current 4x4
        graph::ResourceHandle in_view_proj_prev;     // previous 4x4
        graph::ResourceHandle out_motion;            // W * H u32 (snorm16 ×2)
        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        cardinal::u32 pixels_with_motion {0};
    };
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_depth,
        graph::ResourceHandle in_view_proj,
        graph::ResourceHandle in_view_proj_prev,
        cardinal::u32 width, cardinal::u32 height);
    static const char* hlsl_source() noexcept;
};

// TonemapPass — HDR linear → SDR sRGB-encoded byte buffer. ACES filmic
// approximation; configurable exposure.
class TonemapPass {
public:
    struct State {
        graph::ResourceHandle in_radiance;        // W * H * 3 floats (HDR linear)
        graph::ResourceHandle out_rgba;           // W * H * 4 bytes (sRGB)
        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        float exposure {1.0f};
        // Stats:
        cardinal::u32 pixels_clipped {0};      // pixels that saturate at 1.0 after tonemap
    };
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_radiance,
        cardinal::u32 width, cardinal::u32 height,
        float exposure = 1.0f);
    static const char* hlsl_source() noexcept;
};

// ===========================================================================
// Block 13 — Composite + Present
// ===========================================================================

// Final composite: scene color + UI overlay + gizmo overlay → presentation
// buffer. Composite uses alpha-test over each overlay buffer.
class CompositePresentPass {
public:
    struct State {
        graph::ResourceHandle in_scene;           // W * H * 4 bytes (post-tonemap)
        graph::ResourceHandle in_ui;              // optional W * H * 4 bytes
        graph::ResourceHandle in_gizmo;           // optional W * H * 4 bytes
        graph::ResourceHandle out_presentation;   // W * H * 4 bytes (final)
        cardinal::u32 width  {0};
        cardinal::u32 height {0};
        cardinal::u32 pixels_overdrawn {0};       // pixels touched by overlay layers
    };
    static cardinal::shared_ptr<State> add_to_graph(
        graph::Graph& g,
        graph::ResourceHandle in_scene,
        graph::ResourceHandle in_ui,         // pass {} for none
        graph::ResourceHandle in_gizmo,      // pass {} for none
        cardinal::u32 width, cardinal::u32 height);
    static const char* hlsl_source() noexcept;
};

// ===========================================================================
// AegisPipeline — top-level orchestrator that wires the AEGIS diagram
// end-to-end as a single graph::Graph. Hosts construct one, hand it the
// scene resource handles + viewport, and the pipeline declares every
// pass in the correct dependency order.
// ===========================================================================

struct AegisConfig {
    cardinal::u32 width  {1920};
    cardinal::u32 height {1080};
    cardinal::u32 material_count {0};
    cardinal::u32 light_count {0};
    float         exposure {1.0f};
    GpuCapabilities caps;                  // FP32 / FP16 / FP8 / FP4 support
    GeometryTier    max_tier {GeometryTier::Fp4};
};

struct AegisSceneInputs {
    graph::ResourceHandle tris;              // M * 9 floats (world-space)
    graph::ResourceHandle curvature;         // optional M floats — pass {} if absent
    graph::ResourceHandle material_ids;      // M u32
    graph::ResourceHandle materials;         // material_count * 8 floats
    graph::ResourceHandle lights;            // light_count * 16 floats (PackedLight)
    graph::ResourceHandle ambient;           // 3 floats
    graph::ResourceHandle view_proj;         // 16 floats
    graph::ResourceHandle view_proj_prev;    // optional 16 floats — pass {} for no motion vectors
    graph::ResourceHandle camera_dir;        // 3 floats
    graph::ResourceHandle ui_overlay;        // optional w*h*4 bytes
    graph::ResourceHandle gizmo_overlay;     // optional w*h*4 bytes
    cardinal::u32 triangle_count {0};
};

struct AegisOutputs {
    // Block 6
    graph::ResourceHandle vbuf_depth;
    graph::ResourceHandle vbuf_prim;
    graph::ResourceHandle vbuf_mat;
    graph::ResourceHandle vbuf_normal;
    graph::ResourceHandle vbuf_motion;
    // Block 12
    graph::ResourceHandle radiance_hdr;
    graph::ResourceHandle tonemapped;
    // Block 13
    graph::ResourceHandle presentation;
};

struct AegisStageRefs {
    cardinal::shared_ptr<GeometryClassifyPass::State>  classify;
    cardinal::shared_ptr<MeshletBuildPass::State>       meshlets;
    cardinal::shared_ptr<MeshletCullPass::State>        meshlet_cull;
    cardinal::shared_ptr<ScreenSpaceErrorPass::State>   sse;
    cardinal::shared_ptr<FrustumCullPass::State>        frustum_cull;
    cardinal::shared_ptr<HiZBuildPass::State>           hiz;
    cardinal::shared_ptr<DrawIndirectGenPass::State>    indirect_gen;
    cardinal::shared_ptr<AdaptiveGeometryPass::State>   adaptive;
    cardinal::shared_ptr<VisibilityBufferPass::State>   vbuf;
    cardinal::shared_ptr<TiledLightCullPass::State>     light_cull;
    cardinal::shared_ptr<VBufResolvePass::State>        resolve;
    cardinal::shared_ptr<MotionVectorPass::State>       motion;
    cardinal::shared_ptr<TonemapPass::State>            tonemap;
    cardinal::shared_ptr<CompositePresentPass::State>   composite;
};

class AegisPipeline {
public:
    static cardinal::shared_ptr<AegisPipeline> create(AegisConfig cfg);

    // Wires every required pass into `g` in the order the AEGIS spec
    // requires. Returns the output handles + per-stage State pointers
    // so the host can inspect stats per frame.
    void build(graph::Graph& g,
               const AegisSceneInputs& inputs,
               AegisOutputs& out,
               AegisStageRefs& stages);

    const AegisConfig& config() const noexcept { return cfg_; }

    // Editor-facing tier readback after build() (selected by
    // AdaptiveGeometryPass).
    GeometryTier selected_tier() const noexcept { return selected_tier_; }

private:
    AegisPipeline() = default;
    AegisConfig  cfg_;
    GeometryTier selected_tier_ {GeometryTier::Fp32};
};

}  // namespace cardinal::render::gpu
