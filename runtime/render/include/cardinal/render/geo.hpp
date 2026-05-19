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
// What's intentionally NOT here yet:
//   ✗ DAG of cluster LODs (Nanite's hierarchical streaming) — shape is
//     reserved (Cluster::lod_parent), generator is single-level today
//   ✗ Software rasteriser for sub-pixel triangles — needs a compute pass;
//     hardware raster is fine until the editor has dense scenes
//   ✗ GPU-driven culling — when compute dispatch lands the same algorithm
//     moves to a shader; the math here is the reference impl
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/scene/math.hpp>

#include <cardinal/core/containers.hpp>

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

}  // namespace cardinal::render::geo
