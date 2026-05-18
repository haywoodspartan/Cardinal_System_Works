#pragma once

// =============================================================================
// Cardinal — vgeom cluster + hierarchy data structures.
//
// Layout chosen for streaming-friendliness: cluster headers (small,
// always resident) live separately from cluster vertex blobs (paged).
// Today we keep everything resident — see vgeom.hpp for the streaming
// roadmap. The split is in place so the page-in/out logic can be added
// without changing the hot-path layout.
// =============================================================================

#include <cardinal/vgeom/vgeom.hpp>     // Vertex, Vec3, Mat4 type aliases

#include <cardinal/core/containers.hpp>

namespace cardinal::vgeom {

// One cluster = a contiguous run of triangles in the hierarchy. The
// vertex data is a triangle list, no index buffer (matches the existing
// forward renderer's contract). Each cluster is independent — its
// vertex run starts at vertex_offset and is vertex_count verts long
// (= 3 × triangle count, since triangle list).
struct Cluster {
    // ---- Header (always resident) --------------------------------------
    // Bounding sphere — local space (mesh space). LOD selection projects
    // this into screen space, no per-vertex work.
    Vec3 center{0, 0, 0};
    f32  radius{0.0f};

    // Geometric error if you replace this cluster's children with this
    // cluster. The runtime cuts above the level where pixel-projected
    // geometric_error exceeds the tolerance. Leaves get 0 (they ARE
    // the source data); the root is the largest.
    f32         geometric_error{0.0f};

    // Hierarchy links. parent == kInvalid for the root; child_count == 0
    // for leaves. Children live in the same vector at indices
    // [first_child_id, first_child_id + child_count).
    static constexpr u32 kInvalid = static_cast<u32>(-1);
    u32         parent_id{kInvalid};
    u32         first_child_id{kInvalid};
    u32         child_count{0};

    // What level of the hierarchy this cluster lives at. 0 = leaves
    // (highest detail), increases towards the root.
    u32         level{0};

    // ---- Vertex blob slice (potentially paged) -------------------------
    // Offset into Hierarchy::vertices and length in Vertex units.
    u32         vertex_offset{0};
    u32         vertex_count{0};
};

// The whole hierarchy for one source mesh. Owned via shared_ptr by
// vgeom::attach so multiple scene::Entity instances can reuse it.
struct Hierarchy {
    // SoA-friendly: clusters[] is the header table, vertices[] is the
    // shared blob clusters slice into. clusters[0] == root by
    // convention; leaves are scattered at the deepest levels.
    cardinal::vector<Cluster>  clusters;
    cardinal::vector<Vertex>   vertices;

    // Master triangle count = the source mesh's triangle count, the
    // floor of "what we'd draw without vgeom". Stats divide drawn /
    // master to report the savings ratio in the Studio panel.
    u32                   master_tri_count{0};

    // Bounding sphere of the WHOLE mesh (== root cluster's bounds).
    Vec3                  root_center{0, 0, 0};
    f32                   root_radius{0.0f};

    // Diagnostic — set by cook(), copied to log lines so the user can
    // see the structure of what got built.
    u32                         level_count{0};
    u32                         leaf_count{0};
    u32                         total_cluster_count{0};
};

// Helpers used by cook + select. Inlined small ones here; bigger ones
// live in cluster.cpp.

// Project a local-space sphere into pixel coverage. Returns the
// projected radius in pixels (a useful proxy for "screen-space size of
// this cluster's bounds" — selection uses it to decide DESCEND vs USE).
//
// Math: world-space radius = local radius × max-axis scale of the
// model matrix. Distance to camera in world space → projected radius
// via fovy / viewport_height (perspective). Orthographic projection
// gives a constant projection ratio — handled inline by inspecting
// proj.m[2][3] / proj.m[3][3].
f32 project_sphere_pixel_radius(const Vec3& center_local,
                                f32 radius_local,
                                const Mat4& model,
                                const Mat4& view,
                                const Mat4& proj,
                                f32 viewport_pixel_height) noexcept;

// World-space center distance squared from camera. Used for front-to-
// back ordering of the cut (so the GPU's early-z gets best culling).
f32 distance2_from_camera(const Vec3& center_local,
                          const Mat4& model,
                          const Mat4& view) noexcept;

}  // namespace cardinal::vgeom
