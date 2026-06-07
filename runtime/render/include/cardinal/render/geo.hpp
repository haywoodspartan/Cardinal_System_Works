#pragma once

// =============================================================================
// Cardinal — meshlet / cluster geometry (Phase 6 baseline).
//
// First step toward Nanite-style virtualised geometry: slice each mesh into
// fixed-size *meshlets* (groups of <= 64 triangles), tag each one with a
// bounding sphere + a backface-cone, and cull them collectively before
// drawing. Meshlets are also the natural unit for hardware mesh shaders —
// when those land in the RHI, the same data feeds them directly.
//
// What this module ships today:
//   ✓ build_meshlets — partition an indexed mesh into clusters, compute
//                      bounds + backface cone per cluster
//   ✓ Cluster culling — frustum + backface cone tests in CPU code, ready
//                      to drive a CPU-driven indirect draw or a GPU compute
//                      pass once we wire one
//   ✓ Distance LOD   — pick a coarse / fine cluster set based on camera
//
// Nanite-style virtualised geometry (CPU reference impls — the same math
// moves to a compute pass when GPU dispatch lands, exactly like the
// culling above):
//   ✓ Cluster-LOD DAG — build_cluster_dag groups near clusters and
//                      replaces each group with a vertex-clustering-
//                      simplified merge, recording a monotone geometric
//                      error per node (the hierarchical structure
//                      lod_parent/lod_level always reserved space for)
//   ✓ Perfect LOD cut — select_lod_cut walks the DAG and emits the
//                      coarsest cluster set whose projected error is
//                      under a pixel budget: continuous + pop-free
//                      because error is monotone along the DAG
//   ✓ Polygon subdivision — subdivide() does watertight 1→4 micro-poly
//                      tessellation with optional PN-triangle (curved
//                      point-normal) placement, the polygon analogue of
//                      the algo::TessFactor "phong" policy
//
// What's intentionally NOT here yet:
//   ✗ Software rasteriser for sub-pixel triangles — needs a compute pass;
//     hardware raster is fine until the editor has dense scenes
//   ✗ GPU-driven culling / DAG traversal — when compute dispatch lands
//     the same algorithm moves to a shader; the math here is the
//     reference impl
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/scene/math.hpp>

#include <cardinal/core/std/containers.hpp>

namespace cardinal::render::geo {

// Hardware-mesh-shader-compatible limits. Both NV and AMD support up to 64
// vertices and 124 prims per meshlet — we clamp to 64 prims to leave room
// for index packing.
inline constexpr u32 kMaxVertsPerMeshlet = 64;
inline constexpr u32 kMaxPrimsPerMeshlet = 64;

// Per-meshlet bounds. Sphere is the conservative culling primitive; the
// cone is a tight backface culler — when the angle between the camera-to-
// cluster vector and the cone axis exceeds (cone_angle_cos), every triangle
// in the cluster is back-facing, so we drop the entire cluster cheaply.
struct ClusterBounds {
    scene::Vec3 sphere_center{};
    float       sphere_radius{0.0f};

    scene::Vec3 cone_apex{};         // origin (cluster centroid)
    scene::Vec3 cone_axis{};         // average normal, unit-length
    float       cone_angle_cos{1.0f};// 1.0 = no cone (don't reject)
};

// Vertex/index ranges in a packed buffer. The whole mesh's verts and inds
// stay in their original buffers; meshlets reference contiguous spans.
struct Meshlet {
    u32 vertex_offset{0};
    u32 vertex_count{0};
    u32 index_offset{0};
    u32 index_count{0};       // multiple of 3
    ClusterBounds bounds{};

    // Reserved for the LOD DAG that lands when the streaming work begins.
    u32 lod_parent{u32(-1)};
    u32 lod_level{0};
};

struct Mesh {
    cardinal::vector<Meshlet>     meshlets;
    // Source data — each meshlet's index range refers into `indices`,
    // which references vertices in the consumer's own vertex buffer.
    cardinal::vector<u32>         indices;
};

// ---------------------------------------------------------------------------
// Build meshlets from an indexed triangle list.
//
// The consumer owns the vertex buffer (we only re-arrange indices). Pass
// the position pointer + stride + count so we can compute per-cluster
// bounds without copying. Normals are optional — if non-null, we compute
// a backface cone; otherwise the cone is left wide-open (cos = -1).
// ---------------------------------------------------------------------------
struct BuildOptions {
    u32  max_verts{kMaxVertsPerMeshlet};
    u32  max_prims{kMaxPrimsPerMeshlet};
};

Mesh build_meshlets(const u32* triangle_indices, u32 index_count,
                    const float* positions_xyz, u32 vertex_count,
                    u32 vertex_stride_bytes,
                    const float* normals_xyz   = nullptr,
                    u32 normal_stride_bytes    = 0,
                    BuildOptions opts = {});

// ---------------------------------------------------------------------------
// Cluster culling — CPU reference impl.
//
// Frustum: pass 6 planes (left/right/bottom/top/near/far), each as a Vec4
// {nx, ny, nz, d} where the visible half-space is dot(plane, point) >= 0.
// Cone: pass the camera world position; we form the cluster→camera vector
// and test cone_angle_cos.
// ---------------------------------------------------------------------------
struct Frustum {
    scene::Vec4 planes[6]{};   // outward-pointing
};

// Build a world-space frustum from a view-projection matrix (Gribb-Hartmann).
Frustum frustum_from_vp(const scene::Mat4& vp);

bool sphere_inside_frustum(const Frustum& f,
                           const scene::Vec3& center, float radius) noexcept;

bool cluster_passes_backface(const ClusterBounds& b,
                             const scene::Vec3& camera_pos) noexcept;

// One-shot test combining both. Returns true when the cluster is potentially
// visible and should be drawn.
inline bool cluster_visible(const ClusterBounds& b, const Frustum& f,
                            const scene::Vec3& camera_pos) noexcept {
    return sphere_inside_frustum(f, b.sphere_center, b.sphere_radius)
        && cluster_passes_backface(b, camera_pos);
}

// ---------------------------------------------------------------------------
// Distance LOD — when the cluster is small in screen space, swap to a
// coarser variant (or skip outright). Returns lod 0 (full detail), 1 (half),
// or 2 (quarter) using a simple distance ramp. Future iterations stream
// from a Nanite-style DAG.
// ---------------------------------------------------------------------------
struct LodConfig {
    float distance_full{20.0f};      // below this → lod 0
    float distance_half{60.0f};      // 20..60     → lod 1
    float distance_quarter{200.0f};  // 60..200    → lod 2; beyond → cull
};

u32 cluster_lod(const ClusterBounds& b, const scene::Vec3& camera_pos,
                const LodConfig& cfg = {}) noexcept;

// ---------------------------------------------------------------------------
// Polygon subdivision — adaptive micro-poly tessellation. Nanite adds
// geometric detail where the screen needs it; this is the polygon side of
// that. 1→4 midpoint split applied `levels` times. A shared-edge midpoint
// cache keys on the ordered (min,max) vertex pair so a coincident edge
// splits to the *same* new vertex from either triangle — the result is
// watertight (no cracks). When per-vertex normals are supplied and
// curve_blend>0 the new midpoints are pulled onto the PN-triangle (curved
// point-normal) limit surface, so a coarse cage gains smooth silhouettes.
// ---------------------------------------------------------------------------
struct TriMesh {
    cardinal::vector<scene::Vec3> positions;
    cardinal::vector<scene::Vec3> normals;     // empty, or == positions.size()
    cardinal::vector<u32>         indices;     // 3 per triangle
};

struct SubdivOptions {
    u32   levels{1};          // uniform 1→4 passes (0 = passthrough copy)
    float curve_blend{0.0f};  // 0 = planar midpoints, 1 = full PN-triangle
};

TriMesh subdivide(const TriMesh& in, SubdivOptions opts = {});

// Map a tess factor (algo::TessFactor output, ~1..64) to a uniform subdiv
// level. Cluster-granular like Nanite: every triangle in a cluster takes
// the same level, so independent clusters never crack against each other.
u32 subdiv_level_for_factor(float tess_factor, u32 max_levels = 4) noexcept;

// ---------------------------------------------------------------------------
// Hierarchical cluster-LOD DAG — the Nanite core. Level 0 is the input
// meshlets. Each higher level groups spatially-near clusters and replaces
// each group with a vertex-clustering-simplified merge (~2:1 reduction),
// recording the geometric error introduced. `error` is forced
// monotonically non-decreasing up the DAG so an error-bounded cut is
// unambiguous and pop-free.
// ---------------------------------------------------------------------------
struct LodNode {
    ClusterBounds bounds{};
    float         error{0.0f};      // world-space simplification error here
    u32           lod_level{0};
    u32           first_child{u32(-1)};  // offset into ClusterDag::child_ids
    u32           child_count{0};        // 0 == leaf (a level-0 meshlet)
    u32           index_offset{0};  // into ClusterDag::indices
    u32           index_count{0};   // multiple of 3
};

struct ClusterDag {
    cardinal::vector<LodNode>     nodes;      // level 0 first, ascending
    cardinal::vector<scene::Vec3> positions;  // welded position pool (all levels)
    cardinal::vector<u32>         indices;    // geometry for every level
    cardinal::vector<u32>         child_ids;  // CSR child pool (see LodNode)
    u32                           root{u32(-1)};
    u32                           level_count{0};
};

struct DagOptions {
    u32 group_size{4};        // child clusters merged per parent (>=2)
    u32 max_levels{8};        // safety cap
};

ClusterDag build_cluster_dag(const Mesh& base,
                             const float* positions_xyz, u32 vertex_count,
                             u32 vertex_stride_bytes,
                             DagOptions opts = {});

// Error-bounded LOD cut — Nanite's "perfect LOD". Descends from the root
// while a node's projected error exceeds the pixel budget; emits the
// coarsest cluster whose screen error is acceptable. proj_scale is pixels
// per world-unit at unit depth ( = viewport_height / (2 tan(fovy/2)) ).
struct LodCutParams {
    scene::Vec3 camera_pos{};
    float       proj_scale{1000.0f};
    float       error_threshold_px{1.0f};
};

cardinal::vector<u32> select_lod_cut(const ClusterDag& dag,
                                     const LodCutParams& p);

}  // namespace cardinal::render::geo
