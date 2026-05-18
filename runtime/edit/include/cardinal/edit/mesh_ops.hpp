#pragma once

// =============================================================================
// Cardinal — CPU mesh authoring operations.
//
// Operates on a flat std::vector<Vertex>-style buffer (3 verts per triangle).
// Output is a fresh vector — none of these mutate input. Pure CPU; the host
// uploads the result to the GPU via Mesh::from_vertices().
//
// The point isn't to replace Blender — these are quick-edit ops you'd do
// in a level editor:
//   - Generate primitive (cube / sphere / cylinder / cone / torus / disk)
//   - Subdivide (each tri -> 4 tris by midpoint split)
//   - Mirror across a plane (X=0, Y=0, Z=0)
//   - Decimate (clustering simplifier — quick + lossy)
//   - Smooth (Laplacian relaxation)
//   - Recompute normals (face or smooth)
//   - Translate / scale / rotate (bake the transform into vertices)
//   - Tint (multiply vertex color by a scalar tint)
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/scene/math.hpp>

#include <vector>

namespace cardinal::scene { struct Vertex; }

namespace cardinal::edit::mesh_ops {

using Vertex = cardinal::scene::Vertex;
using Vec3   = cardinal::scene::Vec3;

// ---------------------------------------------------------------------------
// Primitive generators — same shape family Mesh::make_* exposes, but the
// raw vertex buffer is returned for further editing (subdivide, mirror, …).
// All are uniform color — caller can re-tint.
// ---------------------------------------------------------------------------
std::vector<Vertex> make_box     (float size = 1.0f);
std::vector<Vertex> make_plane   (float size = 1.0f, u32 subdivisions = 1);
std::vector<Vertex> make_sphere  (float radius = 1.0f, u32 segments = 24);
std::vector<Vertex> make_cylinder(float radius = 1.0f, float height = 2.0f,
                                  u32 segments = 24);
std::vector<Vertex> make_cone    (float radius = 1.0f, float height = 2.0f,
                                  u32 segments = 24);
std::vector<Vertex> make_torus   (float major = 1.0f, float minor = 0.3f,
                                  u32 major_segs = 32, u32 minor_segs = 16);
std::vector<Vertex> make_disk    (float radius = 1.0f, u32 segments = 32);

// ---------------------------------------------------------------------------
// Editing ops — every op returns a NEW vector.
// ---------------------------------------------------------------------------

// Each triangle becomes 4 (midpoint subdivision). Loop n times for 4^n.
std::vector<Vertex> subdivide(const std::vector<Vertex>& in, u32 levels = 1);

// Mirror across an axis-aligned plane through the origin.
//   axis: 0=X, 1=Y, 2=Z. Triangle winding is reversed so face normals
//   continue to point outward.
std::vector<Vertex> mirror(const std::vector<Vertex>& in, u32 axis);

// Cluster-decimate — merges vertices that fall into the same world-space
// grid cell of `cell_size`. Triangles whose three verts collapse into ≤2
// distinct cells are dropped. Quick + lossy; doesn't preserve UVs.
std::vector<Vertex> decimate_cluster(const std::vector<Vertex>& in, float cell_size);

// Laplacian smoothing — each vertex moves toward the average of its 1-ring.
// `iterations` controls how many sweeps; >5 produces a noticeably puffy
// result. `lambda` (0..1) scales the move per iteration.
std::vector<Vertex> smooth_laplacian(const std::vector<Vertex>& in,
                                     u32 iterations = 1, float lambda = 0.5f);

// Recompute per-vertex normals from the triangle topology. `smooth=true`
// averages adjacent face normals; `false` makes hard edges (one normal per
// triangle, all 3 verts of a triangle share the face normal).
std::vector<Vertex> recompute_normals(const std::vector<Vertex>& in, bool smooth = true);

// Bake a transform into vertex positions.
std::vector<Vertex> bake_transform(const std::vector<Vertex>& in,
                                   const cardinal::scene::Mat4& xform);

// Apply a flat color tint (multiplied by existing vertex color).
std::vector<Vertex> tint(const std::vector<Vertex>& in, const Vec3& tint);

// Append `b` after `a`, returning the concatenation.
std::vector<Vertex> append(std::vector<Vertex> a, const std::vector<Vertex>& b);

// Returns triangle count + axis-aligned bounding box.
struct MeshStats {
    u32   tri_count{0};
    u32   vert_count{0};
    Vec3  aabb_min{0.0f, 0.0f, 0.0f};
    Vec3  aabb_max{0.0f, 0.0f, 0.0f};
    float surface_area{0.0f};
};
MeshStats stats(const std::vector<Vertex>& in);

}  // namespace cardinal::edit::mesh_ops
