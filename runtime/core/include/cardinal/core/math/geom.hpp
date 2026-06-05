#pragma once

// =============================================================================
// Cardinal — Geometry Core + Framework.
//
// Pure-math geometry primitives + spatial structures. Higher-level than the
// existing render::geo (which builds GPU meshlets) — this module is the
// CPU-side toolbox gameplay code, AI, physics, partition all share.
//
// Primitives:
//   AABB      — axis-aligned bounding box
//   OBB       — oriented bounding box
//   Sphere    — center + radius
//   Capsule   — point0 + point1 + radius (swept sphere)
//   Plane     — normal + d
//   Frustum   — 6 planes (Gribb-Hartmann)
//   Triangle  — 3 vertices, helpers for area / normal / centroid
//   Polygon2D — convex 2D polygon (point-in-poly, expand, intersect)
//
// Spatial structures:
//   AabbBvh   — top-down AABB BVH with stackless traversal
//
// Tests:
//   contains, intersects (every primitive pair we care about)
//   raycast against AABB / sphere / capsule / triangle
// =============================================================================

#include <cardinal/core/math.hpp>
#include <cardinal/core/types.hpp>

#include <functional>
#include <vector>

namespace cardinal::core::geom {

using Vec3 = cardinal::core::Vec3;
using Vec4 = cardinal::core::Vec4;
using Mat4 = cardinal::core::Mat4;

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------
struct AABB {
    Vec3 min{0,0,0};
    Vec3 max{0,0,0};

    static AABB make_empty() noexcept;
    static AABB from_center_extent(const Vec3& c, const Vec3& he) noexcept;

    Vec3  center()  const noexcept;
    Vec3  extent()  const noexcept;
    Vec3  size()    const noexcept;
    f32   volume()  const noexcept;
    f32   surface_area() const noexcept;
    bool  empty()   const noexcept;

    void  expand(const Vec3& p) noexcept;
    void  expand(const AABB& o) noexcept;
    void  inflate(f32 by) noexcept;
    bool  contains(const Vec3& p) const noexcept;
    bool  contains(const AABB& o) const noexcept;
    bool  intersects(const AABB& o) const noexcept;
};

struct OBB {
    Vec3 center{0,0,0};
    Vec3 half_extents{1,1,1};
    // Local-axis triple — column 0 = x, column 1 = y, column 2 = z.
    // Must be orthonormal.
    Vec3 axis_x{1,0,0}, axis_y{0,1,0}, axis_z{0,0,1};
    AABB world_aabb() const noexcept;
    bool contains(const Vec3& p) const noexcept;
};

struct Sphere {
    Vec3 center{0,0,0};
    f32  radius{1.0f};
    bool contains(const Vec3& p) const noexcept;
    bool intersects(const Sphere& o) const noexcept;
    bool intersects(const AABB& a)   const noexcept;
};

struct Capsule {
    Vec3 a{0,0,0}, b{0,1,0};
    f32  radius{0.5f};
    AABB world_aabb() const noexcept;
};

struct Plane {
    Vec3 normal{0,1,0};
    f32  d{0.0f};
    f32  distance_to(const Vec3& p) const noexcept;
};

struct Frustum {
    Plane planes[6]{};        // outward-pointing
    static Frustum from_view_proj(const Mat4& view_proj) noexcept;

    bool contains(const Vec3& p)    const noexcept;
    bool intersects(const Sphere& s) const noexcept;
    bool intersects(const AABB& a)   const noexcept;
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 normal()   const noexcept;
    Vec3 centroid() const noexcept;
    f32  area()     const noexcept;
};

// 2D convex polygon. Vertices in CCW order. Used by navmesh + AI cones.
struct Polygon2D {
    std::vector<f32> xs;
    std::vector<f32> ys;
    bool contains(f32 x, f32 y) const noexcept;
    f32  area() const noexcept;
};

// ---------------------------------------------------------------------------
// Ray + intersections
// ---------------------------------------------------------------------------
struct Ray {
    Vec3 origin{0,0,0};
    Vec3 direction{0,0,1};       // assumed normalised

    Vec3 at(f32 t) const noexcept {
        return Vec3{ origin.x + direction.x * t,
                     origin.y + direction.y * t,
                     origin.z + direction.z * t };
    }
};

struct Hit {
    bool hit{false};
    f32  t{0.0f};
    Vec3 point{0,0,0};
    Vec3 normal{0,0,1};
    explicit operator bool() const noexcept { return hit; }
};

Hit raycast_aabb    (const Ray& r, const AABB& a, f32 max_t = 1e6f) noexcept;
Hit raycast_sphere  (const Ray& r, const Sphere& s, f32 max_t = 1e6f) noexcept;
Hit raycast_triangle(const Ray& r, const Triangle& tri, f32 max_t = 1e6f) noexcept;
Hit raycast_capsule (const Ray& r, const Capsule& c, f32 max_t = 1e6f) noexcept;

// ---------------------------------------------------------------------------
// AABB BVH — top-down median-split builder + iterative traversal.
//
// One BVH node per leaf; leaves can hold up to `kLeafBucket` primitive ids.
// Traverse with a hit-callback; the callback returns false to abort early.
// ---------------------------------------------------------------------------
inline constexpr u32 kLeafBucket = 8;

class AabbBvh {
public:
    void build(const std::vector<AABB>& boxes,
               const std::vector<u32>&  ids);

    void clear() noexcept;
    usize node_count() const noexcept;
    usize primitive_count() const noexcept;
    AABB  root_aabb() const noexcept;

    using AabbHitCallback = std::function<bool(u32 prim_id, const AABB& box)>;
    void traverse_aabb(const AABB& query, const AabbHitCallback& cb) const;

    using SphereHitCallback = std::function<bool(u32 prim_id, const AABB& box)>;
    void traverse_sphere(const Sphere& query, const SphereHitCallback& cb) const;

    using FrustumHitCallback = std::function<bool(u32 prim_id, const AABB& box)>;
    void traverse_frustum(const Frustum& f, const FrustumHitCallback& cb) const;

    using RayHitCallback = std::function<bool(u32 prim_id, const AABB& box, f32 t)>;
    // Returns total leaves visited (instrumentation).
    u32 traverse_ray(const Ray& r, f32 max_t, const RayHitCallback& cb) const;

    // Public so anonymous-namespace helpers in geom.cpp can name it. Don't
    // depend on field order across versions — treat as opaque.
    struct Node {
        AABB box{};
        i32  left_child{-1};
        i32  right_child{-1};
        u32  prim_offset{0};
        u32  prim_count{0};
        bool is_leaf() const noexcept { return prim_count > 0; }
    };

private:
    std::vector<Node> nodes_;
    std::vector<u32>  prim_indices_;     // referenced by leaves
    std::vector<AABB> prim_boxes_;
    std::vector<u32>  prim_ids_;

    i32 build_recursive_(u32 begin, u32 end);
};

// ---------------------------------------------------------------------------
// Dynamic mesh — like edit::mesh_ops but with topology operations that
// mutate in place + maintain adjacency. Useful for runtime sculpting,
// mesh booleans (future), procedural splitting.
// ---------------------------------------------------------------------------
struct DynamicMesh {
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec3> colors;
    std::vector<u32>  indices;          // triangle list

    // Bookkeeping helpers.
    void  reserve(u32 verts, u32 tris);
    void  clear() noexcept;
    u32   vertex_count() const noexcept;
    u32   triangle_count() const noexcept;
    AABB  bounds() const noexcept;

    // Topology ops.
    u32   add_vertex(const Vec3& p, const Vec3& n = {0,1,0}, const Vec3& c = {1,1,1});
    void  add_triangle(u32 a, u32 b, u32 c);
    void  recompute_normals_smooth();

    // Subdivide every triangle into 4 (midpoint).
    void  subdivide_once();

    // Vertex weld — collapses verts within `epsilon` of each other.
    void  weld_vertices(f32 epsilon);
};

}  // namespace cardinal::geom
