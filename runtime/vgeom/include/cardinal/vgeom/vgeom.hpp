#pragma once

// =============================================================================
// Cardinal — Virtualized Geometry (vgeom).
//
// Goal: film-quality polygon counts at real-time rates by paying CPU/GPU
// cost proportional to what the camera ACTUALLY SEES, not to the source
// mesh's master triangle count. The user keeps "everything authored at
// max" and the runtime picks a per-frame "cut" through a precomputed LOD
// hierarchy — so a 10M-tri statue renders 5k tris when it's a speck on
// the horizon, 200k tris when it fills the screen, and the cut moves
// continuously as the camera approaches.
//
// Toggleable per-mesh: a mesh that's set up for vgeom renders through
// the cluster path; otherwise it falls through to the existing direct
// vertex-buffer path. No global on/off — the renderer transparently
// dispatches based on the mesh.
//
// =============================================================================
// Algorithm overview (the "concrete" — what we actually do, in order)
// =============================================================================
//
// Cook time (offline / mesh-import; slow + amortised):
//
//   1. Decompose source mesh into CLUSTERS of ~kClusterTargetTris
//      triangles each. Greedy BFS: pick a seed triangle, expand to
//      neighbours via the edge-adjacency graph until the cluster is
//      "full" or no neighbour remains. Each cluster gets:
//        - its own packed Vertex array (no shared buffers)
//        - a bounding sphere
//        - parent / children indices into the hierarchy
//        - a "geometric error" — the visual error if you were to
//          replace this cluster's children with itself (root has the
//          biggest error — it's just a coarse blob; leaves have ~0
//          error — they ARE the source mesh).
//
//   2. Build the hierarchy by SIMPLIFICATION. Group adjacent clusters
//      (kClusterGroupSize per group), simplify each group with quadric-
//      error edge collapse to roughly half its triangle count, re-
//      cluster the simplified mesh — those become parent clusters.
//      Repeat until one cluster covers the whole mesh. The result is a
//      tree (the canonical "DAG" version of this lets adjacent groups
//      share boundary edges; documented as a future upgrade — the tree
//      version may show seams between adjacent clusters at different
//      LODs, accepted limitation for the first cut).
//
//   3. Pack everything into a streaming-friendly layout:
//        - Cluster header table: bounds, error, parent/child indices,
//          vertex offsets — small, always resident.
//        - Vertex blob: per-cluster packed verts, can be paged in /
//          out per cluster.
//
// Runtime (per frame; cheap):
//
//   4. PICK THE CUT through the hierarchy:
//        - Start at the root cluster.
//        - For each visited cluster, project its bounding sphere into
//          screen space and compute its pixel coverage.
//        - If pixel coverage × cluster's geometric_error > pixel
//          tolerance (e.g. 1 pixel of allowed error), DESCEND to
//          children. Otherwise USE this cluster.
//        - The result is the "cut" — a set of clusters that together
//          cover the whole mesh at appropriate detail.
//
//   5. CULL the cut:
//        - Frustum-cull each cluster against the camera frustum.
//        - (Future) hi-Z occlusion-cull against last-frame depth.
//
//   6. RENDER the cut:
//        - For each surviving cluster, append its vertices into the
//          forward renderer's per-frame transform queue. Each cluster
//          becomes its own EntityWork in the renderer's existing
//          parallel transform pipeline — same fan-out as one big
//          entity, but split at cluster granularity so the work
//          load-balances across workers.
//
// Streaming (future — first cut keeps everything resident):
//
//   7. Cluster vertex blobs are paged in/out of a fixed CPU+GPU
//      residency budget, LRU-evicted. The cut from step 4 emits load
//      requests for clusters that aren't resident; the requests fan
//      out via cardinal::core::io. While a cluster is loading, the
//      renderer falls back to its parent (one level up the hierarchy).
//
// =============================================================================
// Tunables
// =============================================================================
//
//   kClusterTargetTris   : ~target triangles per cluster. 128 matches
//                          mesh-shader-friendly grouping on most GPUs.
//                          Smaller = finer LOD granularity but more
//                          per-cluster overhead. Larger = less overhead,
//                          coarser LOD, more wasted cluster-corner work.
//
//   kClusterGroupSize    : how many clusters share a parent. 4 → at
//                          each LOD level the triangle count roughly
//                          halves and the cluster count roughly
//                          quarters. Smaller groups give finer DAG
//                          control; bigger groups simplify faster.
//
//   kPixelErrorTolerance : screen-space error budget per cluster, in
//                          pixels. 1.0 = "you can't see the difference
//                          between this cluster and its parent because
//                          the difference is < 1 pixel on screen".
//                          Lower = more triangles, sharper edges;
//                          higher = fewer triangles, more visible LOD
//                          transitions.
//
// =============================================================================
// Public API
// =============================================================================

#include <cardinal/core/math.hpp>      // core::Vec3 / Vec4 / Mat4 (canonical types)
#include <cardinal/core/types.hpp>

#include <memory>
#include <vector>

namespace cardinal::vgeom {

// Geometry types — aliased to the canonical core math so we don't pull
// scene/scene.hpp here (would create a layering cycle: scene's renderer
// uses vgeom, vgeom would need to use scene). The Mesh-attachment glue
// lives in scene where Mesh is defined.
using Vec3 = cardinal::core::Vec3;
using Vec4 = cardinal::core::Vec4;
using Mat4 = cardinal::core::Mat4;

// Vertex layout used by the cook + select pipeline. Identical layout
// to scene::Vertex so callers can reinterpret_cast at the boundary
// without copying. Asserts at the boundary (in cluster.cpp) verify
// the size + alignment match.
struct Vertex {
    Vec3 position;
    Vec3 normal;
    Vec3 color;
};

// Tunables — see comments above.
inline constexpr u32   kClusterTargetTris   = 128;
inline constexpr u32   kClusterGroupSize    = 4;
inline constexpr float kPixelErrorTolerance = 1.0f;

// Forward decl for the hierarchy + a Cluster — full layout is in
// cluster.hpp, kept opaque to most callers.
struct Cluster;
struct Hierarchy;

// Per-frame stats — what fraction of the master polygon count we
// actually drew. Tank this onto the Studio profiler panel to prove
// the system is doing its job.
struct FrameStats {
    u32 master_tri_count{0};       // source mesh triangle total (all clusters at leaves)
    u32 drawn_tri_count{0};        // tris in the cut, after cull
    u32 cut_cluster_count{0};      // clusters in the cut, after cull
    u32 culled_cluster_count{0};   // clusters cut visited but frustum/back-faced out
};

// =============================================================================
// Cooking — turn a source mesh into a vgeom hierarchy.
//
// Slow (seconds for a 1M-tri mesh — clustering + iterative simplification
// is the expensive part). Run this offline at asset-import time and
// serialize the result; runtime just deserializes + binds.
//
// For the first cut we cook directly from a scene::Mesh's CPU shadow at
// runtime (so users can experiment without an asset pipeline change).
// Once the cook stabilises this moves into cardinal::cook with a proper
// disk format.
// =============================================================================
struct CookDesc {
    // Source vertices — triangle-list topology, no index buffer. Layout-
    // identical to scene::Vertex (callers can reinterpret_cast).
    const Vertex*  vertices{nullptr};
    u32            vertex_count{0};

    // Optional name — propagated into log messages and the Studio panel.
    const char*    name{nullptr};

    // Override the default cluster size if you have a reason. 0 = default.
    u32            cluster_target_tris{0};

    // Override the simplification ratio per LOD level (0..1). 0 = default
    // (0.5 — each parent has half the tris of its children combined).
    float          simplify_ratio{0.0f};
};

// Run the cook. Returns nullptr on bad input (no verts, non-triangle-list).
// The returned Hierarchy is reference-counted — share it across multiple
// scene entities that reuse the same source mesh.
std::shared_ptr<Hierarchy> cook(const CookDesc& desc);

// =============================================================================
// Selection — pick the per-frame cut through the hierarchy.
//
// Inputs: the hierarchy, the camera (view + projection), the viewport
// pixel size, and the entity's local-to-world matrix. Output: a list
// of cluster indices in the hierarchy + summary stats.
//
// Cheap (O(visited clusters) — typically a few hundred at most for a
// 10M-tri mesh because only the spatial path the camera looks at gets
// descended).
// =============================================================================
struct SelectInput {
    const Hierarchy*  hierarchy{nullptr};
    Mat4              model;          // entity's local→world
    Mat4              view;
    Mat4              proj;
    f32               viewport_pixel_height{1080.0f};
    f32               pixel_error_tolerance{kPixelErrorTolerance};
};

struct SelectOutput {
    // Cluster ids selected, in front-to-back order. Renderer iterates
    // these and emits each cluster's vertices into the transform queue.
    std::vector<u32>    cluster_ids;
    FrameStats          stats;
};

// Run selection. Allocates into out.cluster_ids (clears + reserves
// internally — call repeatedly without per-frame heap churn).
void select(const SelectInput& in, SelectOutput& out);

// Per-mesh attachment lives in the scene layer (where Mesh is defined)
// to avoid a layering cycle (vgeom-here -> scene-Mesh -> vgeom). Look
// for cardinal::scene::vgeom_attach / vgeom_enable / vgeom_hierarchy_of
// in <cardinal/scene/scene.hpp>.

}  // namespace cardinal::vgeom
