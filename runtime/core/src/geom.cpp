#include <cardinal/core/geom.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace cardinal::core::geom {

// ---------------------------------------------------------------------------
// AABB
// ---------------------------------------------------------------------------
AABB AABB::make_empty() noexcept {
    AABB a;
    a.min = { std::numeric_limits<f32>::max(),
              std::numeric_limits<f32>::max(),
              std::numeric_limits<f32>::max() };
    a.max = {-std::numeric_limits<f32>::max(),
             -std::numeric_limits<f32>::max(),
             -std::numeric_limits<f32>::max() };
    return a;
}
AABB AABB::from_center_extent(const Vec3& c, const Vec3& he) noexcept {
    return AABB{ {c.x - he.x, c.y - he.y, c.z - he.z},
                 {c.x + he.x, c.y + he.y, c.z + he.z} };
}
Vec3 AABB::center() const noexcept {
    return Vec3{ (min.x + max.x) * 0.5f,
                 (min.y + max.y) * 0.5f,
                 (min.z + max.z) * 0.5f };
}
Vec3 AABB::extent() const noexcept {
    return Vec3{ (max.x - min.x) * 0.5f,
                 (max.y - min.y) * 0.5f,
                 (max.z - min.z) * 0.5f };
}
Vec3 AABB::size() const noexcept {
    return Vec3{ max.x - min.x, max.y - min.y, max.z - min.z };
}
f32 AABB::volume() const noexcept {
    const Vec3 s = size();
    return std::max(0.0f, s.x) * std::max(0.0f, s.y) * std::max(0.0f, s.z);
}
f32 AABB::surface_area() const noexcept {
    const Vec3 s = size();
    return 2.0f * (s.x * s.y + s.y * s.z + s.z * s.x);
}
bool AABB::empty() const noexcept {
    return min.x > max.x || min.y > max.y || min.z > max.z;
}
void AABB::expand(const Vec3& p) noexcept {
    if (p.x < min.x) min.x = p.x;
    if (p.y < min.y) min.y = p.y;
    if (p.z < min.z) min.z = p.z;
    if (p.x > max.x) max.x = p.x;
    if (p.y > max.y) max.y = p.y;
    if (p.z > max.z) max.z = p.z;
}
void AABB::expand(const AABB& o) noexcept {
    if (o.empty()) return;
    expand(o.min); expand(o.max);
}
void AABB::inflate(f32 by) noexcept {
    min.x -= by; min.y -= by; min.z -= by;
    max.x += by; max.y += by; max.z += by;
}
bool AABB::contains(const Vec3& p) const noexcept {
    return p.x >= min.x && p.x <= max.x &&
           p.y >= min.y && p.y <= max.y &&
           p.z >= min.z && p.z <= max.z;
}
bool AABB::contains(const AABB& o) const noexcept {
    return contains(o.min) && contains(o.max);
}
bool AABB::intersects(const AABB& o) const noexcept {
    return !(min.x > o.max.x || max.x < o.min.x ||
             min.y > o.max.y || max.y < o.min.y ||
             min.z > o.max.z || max.z < o.min.z);
}

// ---------------------------------------------------------------------------
// OBB
// ---------------------------------------------------------------------------
AABB OBB::world_aabb() const noexcept {
    AABB a = AABB::make_empty();
    for (i32 i = 0; i < 8; ++i) {
        const f32 sx = (i & 1) ? +half_extents.x : -half_extents.x;
        const f32 sy = (i & 2) ? +half_extents.y : -half_extents.y;
        const f32 sz = (i & 4) ? +half_extents.z : -half_extents.z;
        const Vec3 corner{
            center.x + axis_x.x*sx + axis_y.x*sy + axis_z.x*sz,
            center.y + axis_x.y*sx + axis_y.y*sy + axis_z.y*sz,
            center.z + axis_x.z*sx + axis_y.z*sy + axis_z.z*sz,
        };
        a.expand(corner);
    }
    return a;
}
bool OBB::contains(const Vec3& p) const noexcept {
    const Vec3 d{ p.x - center.x, p.y - center.y, p.z - center.z };
    auto proj = [&](const Vec3& a) { return a.x*d.x + a.y*d.y + a.z*d.z; };
    return std::abs(proj(axis_x)) <= half_extents.x
        && std::abs(proj(axis_y)) <= half_extents.y
        && std::abs(proj(axis_z)) <= half_extents.z;
}

// ---------------------------------------------------------------------------
// Sphere
// ---------------------------------------------------------------------------
bool Sphere::contains(const Vec3& p) const noexcept {
    const f32 dx = p.x - center.x, dy = p.y - center.y, dz = p.z - center.z;
    return dx*dx + dy*dy + dz*dz <= radius*radius;
}
bool Sphere::intersects(const Sphere& o) const noexcept {
    const f32 dx = o.center.x - center.x;
    const f32 dy = o.center.y - center.y;
    const f32 dz = o.center.z - center.z;
    const f32 r  = radius + o.radius;
    return dx*dx + dy*dy + dz*dz <= r*r;
}
bool Sphere::intersects(const AABB& a) const noexcept {
    f32 d2 = 0.0f;
    for (i32 i = 0; i < 3; ++i) {
        const f32 c = (&center.x)[i];
        const f32 mn = (&a.min.x)[i];
        const f32 mx = (&a.max.x)[i];
        if (c < mn) { const f32 d = mn - c; d2 += d * d; }
        else if (c > mx) { const f32 d = c - mx; d2 += d * d; }
    }
    return d2 <= radius * radius;
}

// ---------------------------------------------------------------------------
// Capsule
// ---------------------------------------------------------------------------
AABB Capsule::world_aabb() const noexcept {
    AABB box = AABB::make_empty();
    box.expand(a); box.expand(b); box.inflate(radius);
    return box;
}

// ---------------------------------------------------------------------------
// Plane
// ---------------------------------------------------------------------------
f32 Plane::distance_to(const Vec3& p) const noexcept {
    return normal.x * p.x + normal.y * p.y + normal.z * p.z + d;
}

// ---------------------------------------------------------------------------
// Frustum
// ---------------------------------------------------------------------------
namespace {
inline Plane normalise(Plane p) {
    const f32 l = std::sqrt(p.normal.x*p.normal.x + p.normal.y*p.normal.y + p.normal.z*p.normal.z);
    if (l > 1e-6f) { p.normal.x /= l; p.normal.y /= l; p.normal.z /= l; p.d /= l; }
    return p;
}
}

Frustum Frustum::from_view_proj(const Mat4& m) noexcept {
    Frustum f{};
    // Gribb-Hartmann: plane equations directly from M rows (column-major
    // here means M.m[col][row]).
    auto mk = [&](i32 sgn, i32 row) {
        Plane p{};
        p.normal.x = m.m[0][3] + sgn * m.m[0][row];
        p.normal.y = m.m[1][3] + sgn * m.m[1][row];
        p.normal.z = m.m[2][3] + sgn * m.m[2][row];
        p.d        = m.m[3][3] + sgn * m.m[3][row];
        return normalise(p);
    };
    f.planes[0] = mk(+1, 0);  // left
    f.planes[1] = mk(-1, 0);  // right
    f.planes[2] = mk(+1, 1);  // bottom
    f.planes[3] = mk(-1, 1);  // top
    f.planes[4] = mk(+1, 2);  // near
    f.planes[5] = mk(-1, 2);  // far
    return f;
}

bool Frustum::contains(const Vec3& p) const noexcept {
    for (const auto& pl : planes) if (pl.distance_to(p) < 0.0f) return false;
    return true;
}
bool Frustum::intersects(const Sphere& s) const noexcept {
    for (const auto& p : planes)
        if (p.distance_to(s.center) < -s.radius) return false;
    return true;
}
bool Frustum::intersects(const AABB& a) const noexcept {
    for (const auto& p : planes) {
        // Pick the most-positive corner of the AABB w.r.t. the plane normal.
        const Vec3 c{
            p.normal.x >= 0.0f ? a.max.x : a.min.x,
            p.normal.y >= 0.0f ? a.max.y : a.min.y,
            p.normal.z >= 0.0f ? a.max.z : a.min.z,
        };
        if (p.distance_to(c) < 0.0f) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Triangle
// ---------------------------------------------------------------------------
Vec3 Triangle::normal() const noexcept {
    const Vec3 e1{b.x - a.x, b.y - a.y, b.z - a.z};
    const Vec3 e2{c.x - a.x, c.y - a.y, c.z - a.z};
    Vec3 n{ e1.y * e2.z - e1.z * e2.y,
            e1.z * e2.x - e1.x * e2.z,
            e1.x * e2.y - e1.y * e2.x };
    const f32 l = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
    if (l > 1e-6f) { n.x /= l; n.y /= l; n.z /= l; }
    return n;
}
Vec3 Triangle::centroid() const noexcept {
    return Vec3{ (a.x + b.x + c.x) / 3.0f,
                 (a.y + b.y + c.y) / 3.0f,
                 (a.z + b.z + c.z) / 3.0f };
}
f32 Triangle::area() const noexcept {
    const Vec3 e1{b.x - a.x, b.y - a.y, b.z - a.z};
    const Vec3 e2{c.x - a.x, c.y - a.y, c.z - a.z};
    const Vec3 cr{ e1.y * e2.z - e1.z * e2.y,
                   e1.z * e2.x - e1.x * e2.z,
                   e1.x * e2.y - e1.y * e2.x };
    return 0.5f * std::sqrt(cr.x*cr.x + cr.y*cr.y + cr.z*cr.z);
}

// ---------------------------------------------------------------------------
// Polygon2D
// ---------------------------------------------------------------------------
bool Polygon2D::contains(f32 x, f32 y) const noexcept {
    if (xs.size() < 3) return false;
    bool inside = false;
    const usize n = xs.size();
    for (usize i = 0, j = n - 1; i < n; j = i++) {
        const f32 xi = xs[i], yi = ys[i];
        const f32 xj = xs[j], yj = ys[j];
        const bool cross = ((yi > y) != (yj > y)) &&
            (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10f) + xi);
        if (cross) inside = !inside;
    }
    return inside;
}
f32 Polygon2D::area() const noexcept {
    if (xs.size() < 3) return 0.0f;
    f32 a = 0.0f;
    const usize n = xs.size();
    for (usize i = 0, j = n - 1; i < n; j = i++) {
        a += (xs[j] + xs[i]) * (ys[j] - ys[i]);
    }
    return std::abs(a) * 0.5f;
}

// ---------------------------------------------------------------------------
// Raycasts
// ---------------------------------------------------------------------------
Hit raycast_aabb(const Ray& r, const AABB& a, f32 max_t) noexcept {
    Hit h{};
    f32 t_min = 0.0f, t_max = max_t;
    for (i32 i = 0; i < 3; ++i) {
        const f32 o = (&r.origin.x)[i];
        const f32 d = (&r.direction.x)[i];
        const f32 mn = (&a.min.x)[i];
        const f32 mx = (&a.max.x)[i];
        if (std::abs(d) < 1e-6f) {
            if (o < mn || o > mx) return h;
            continue;
        }
        const f32 inv = 1.0f / d;
        f32 t1 = (mn - o) * inv;
        f32 t2 = (mx - o) * inv;
        if (t1 > t2) std::swap(t1, t2);
        if (t1 > t_min) t_min = t1;
        if (t2 < t_max) t_max = t2;
        if (t_min > t_max) return h;
    }
    h.hit = true;
    h.t   = t_min;
    h.point = Vec3{ r.origin.x + r.direction.x * t_min,
                    r.origin.y + r.direction.y * t_min,
                    r.origin.z + r.direction.z * t_min };
    return h;
}

Hit raycast_sphere(const Ray& r, const Sphere& s, f32 max_t) noexcept {
    Hit h{};
    const Vec3 oc{ r.origin.x - s.center.x, r.origin.y - s.center.y, r.origin.z - s.center.z };
    const f32 b = oc.x*r.direction.x + oc.y*r.direction.y + oc.z*r.direction.z;
    const f32 c = oc.x*oc.x + oc.y*oc.y + oc.z*oc.z - s.radius*s.radius;
    const f32 disc = b*b - c;
    if (disc < 0.0f) return h;
    const f32 sq = std::sqrt(disc);
    const f32 t = -b - sq;
    if (t < 0.0f || t > max_t) return h;
    h.hit = true; h.t = t;
    h.point = Vec3{ r.origin.x + r.direction.x * t,
                    r.origin.y + r.direction.y * t,
                    r.origin.z + r.direction.z * t };
    h.normal = Vec3{ (h.point.x - s.center.x) / s.radius,
                     (h.point.y - s.center.y) / s.radius,
                     (h.point.z - s.center.z) / s.radius };
    return h;
}

Hit raycast_triangle(const Ray& r, const Triangle& tri, f32 max_t) noexcept {
    // Möller-Trumbore.
    Hit h{};
    const Vec3 e1{tri.b.x - tri.a.x, tri.b.y - tri.a.y, tri.b.z - tri.a.z};
    const Vec3 e2{tri.c.x - tri.a.x, tri.c.y - tri.a.y, tri.c.z - tri.a.z};
    const Vec3 p{ r.direction.y * e2.z - r.direction.z * e2.y,
                  r.direction.z * e2.x - r.direction.x * e2.z,
                  r.direction.x * e2.y - r.direction.y * e2.x };
    const f32 det = e1.x * p.x + e1.y * p.y + e1.z * p.z;
    if (std::abs(det) < 1e-6f) return h;
    const f32 inv = 1.0f / det;
    const Vec3 t{ r.origin.x - tri.a.x, r.origin.y - tri.a.y, r.origin.z - tri.a.z };
    const f32 u = (t.x * p.x + t.y * p.y + t.z * p.z) * inv;
    if (u < 0.0f || u > 1.0f) return h;
    const Vec3 q{ t.y * e1.z - t.z * e1.y,
                  t.z * e1.x - t.x * e1.z,
                  t.x * e1.y - t.y * e1.x };
    const f32 v = (r.direction.x * q.x + r.direction.y * q.y + r.direction.z * q.z) * inv;
    if (v < 0.0f || u + v > 1.0f) return h;
    const f32 tt = (e2.x * q.x + e2.y * q.y + e2.z * q.z) * inv;
    if (tt < 0.0f || tt > max_t) return h;
    h.hit = true; h.t = tt;
    h.point = Vec3{ r.origin.x + r.direction.x * tt,
                    r.origin.y + r.direction.y * tt,
                    r.origin.z + r.direction.z * tt };
    h.normal = tri.normal();
    return h;
}

Hit raycast_capsule(const Ray& r, const Capsule& cap, f32 max_t) noexcept {
    // Coarse: test inflated AABB. Accurate capsule raycast is involved;
    // this is the gameplay-grade approximation that's plenty for AI cones.
    Hit h{};
    const AABB box = cap.world_aabb();
    h = raycast_aabb(r, box, max_t);
    if (!h.hit) return h;
    // Refine: distance from hit point to the line segment.
    const Vec3 ab{ cap.b.x - cap.a.x, cap.b.y - cap.a.y, cap.b.z - cap.a.z };
    const f32 ab_len2 = ab.x*ab.x + ab.y*ab.y + ab.z*ab.z;
    if (ab_len2 < 1e-6f) {
        Sphere s{ cap.a, cap.radius };
        return raycast_sphere(r, s, max_t);
    }
    return h;
}

// ---------------------------------------------------------------------------
// AabbBvh
// ---------------------------------------------------------------------------
void AabbBvh::clear() noexcept {
    nodes_.clear();
    prim_indices_.clear();
    prim_boxes_.clear();
    prim_ids_.clear();
}

usize AabbBvh::node_count() const noexcept { return nodes_.size(); }
usize AabbBvh::primitive_count() const noexcept { return prim_boxes_.size(); }

AABB AabbBvh::root_aabb() const noexcept {
    return nodes_.empty() ? AABB::make_empty() : nodes_.front().box;
}

i32 AabbBvh::build_recursive_(u32 begin, u32 end) {
    Node n{};
    AABB box = AABB::make_empty();
    AABB centroid_box = AABB::make_empty();
    for (u32 i = begin; i < end; ++i) {
        const u32 pid = prim_indices_[i];
        box.expand(prim_boxes_[pid]);
        centroid_box.expand(prim_boxes_[pid].center());
    }
    n.box = box;
    const u32 count = end - begin;
    if (count <= kLeafBucket) {
        n.prim_offset = begin;
        n.prim_count  = count;
        nodes_.push_back(n);
        return static_cast<i32>(nodes_.size() - 1);
    }
    // Choose split axis by largest centroid-box extent.
    const Vec3 ce = centroid_box.size();
    i32 axis = 0;
    if (ce.y > ce.x && ce.y >= ce.z) axis = 1;
    else if (ce.z > ce.x)            axis = 2;

    auto first_axis = [this, axis](u32 a) {
        const Vec3 c = prim_boxes_[prim_indices_[a]].center();
        return (axis == 0) ? c.x : (axis == 1 ? c.y : c.z);
    };
    const u32 mid = begin + count / 2;
    std::nth_element(prim_indices_.begin() + begin,
                     prim_indices_.begin() + mid,
                     prim_indices_.begin() + end,
                     [&](u32 a, u32 b) {
                         const Vec3 ca = prim_boxes_[a].center();
                         const Vec3 cb = prim_boxes_[b].center();
                         const f32 va = (axis == 0) ? ca.x : (axis == 1 ? ca.y : ca.z);
                         const f32 vb = (axis == 0) ? cb.x : (axis == 1 ? cb.y : cb.z);
                         return va < vb;
                     });
    (void)first_axis;
    nodes_.push_back(n);
    const i32 my = static_cast<i32>(nodes_.size() - 1);
    const i32 lc = build_recursive_(begin, mid);
    const i32 rc = build_recursive_(mid, end);
    nodes_[my].left_child  = lc;
    nodes_[my].right_child = rc;
    return my;
}

void AabbBvh::build(const std::vector<AABB>& boxes,
                    const std::vector<u32>&  ids)
{
    clear();
    if (boxes.empty()) return;
    prim_boxes_ = boxes;
    prim_ids_   = ids.empty() ? std::vector<u32>(boxes.size(), 0) : ids;
    if (ids.empty()) for (u32 i = 0; i < boxes.size(); ++i) prim_ids_[i] = i;
    prim_indices_.resize(boxes.size());
    for (u32 i = 0; i < boxes.size(); ++i) prim_indices_[i] = i;
    nodes_.reserve(boxes.size() * 2);
    build_recursive_(0, static_cast<u32>(boxes.size()));
}

namespace {
template <class Predicate, class LeafCb>
void traverse_iter(const std::vector<AabbBvh::Node>& nodes,
                   const std::vector<u32>& prim_indices,
                   const std::vector<u32>& prim_ids,
                   const std::vector<AABB>& prim_boxes,
                   const Predicate& test_box,
                   const LeafCb& leaf_cb)
{
    if (nodes.empty()) return;
    std::vector<i32> stack;
    stack.reserve(64);
    stack.push_back(0);
    while (!stack.empty()) {
        const i32 idx = stack.back();
        stack.pop_back();
        const auto& n = nodes[idx];
        if (!test_box(n.box)) continue;
        if (n.is_leaf()) {
            for (u32 i = 0; i < n.prim_count; ++i) {
                const u32 pid = prim_indices[n.prim_offset + i];
                if (!leaf_cb(prim_ids[pid], prim_boxes[pid])) return;
            }
        } else {
            if (n.right_child >= 0) stack.push_back(n.right_child);
            if (n.left_child  >= 0) stack.push_back(n.left_child);
        }
    }
}
}  // namespace

void AabbBvh::traverse_aabb(const AABB& q, const AabbHitCallback& cb) const {
    traverse_iter(nodes_, prim_indices_, prim_ids_, prim_boxes_,
        [&](const AABB& b){ return b.intersects(q); }, cb);
}
void AabbBvh::traverse_sphere(const Sphere& q, const SphereHitCallback& cb) const {
    traverse_iter(nodes_, prim_indices_, prim_ids_, prim_boxes_,
        [&](const AABB& b){ return q.intersects(b); }, cb);
}
void AabbBvh::traverse_frustum(const Frustum& f, const FrustumHitCallback& cb) const {
    traverse_iter(nodes_, prim_indices_, prim_ids_, prim_boxes_,
        [&](const AABB& b){ return f.intersects(b); }, cb);
}
u32 AabbBvh::traverse_ray(const Ray& r, f32 max_t,
                          const RayHitCallback& cb) const
{
    if (nodes_.empty()) return 0;
    u32 leaves_visited = 0;
    std::vector<i32> stack;
    stack.reserve(64);
    stack.push_back(0);
    while (!stack.empty()) {
        const i32 idx = stack.back();
        stack.pop_back();
        const auto& n = nodes_[idx];
        const Hit h = raycast_aabb(r, n.box, max_t);
        if (!h.hit) continue;
        if (n.is_leaf()) {
            ++leaves_visited;
            for (u32 i = 0; i < n.prim_count; ++i) {
                const u32 pid = prim_indices_[n.prim_offset + i];
                const Hit ph = raycast_aabb(r, prim_boxes_[pid], max_t);
                if (ph.hit && !cb(prim_ids_[pid], prim_boxes_[pid], ph.t)) return leaves_visited;
            }
        } else {
            if (n.right_child >= 0) stack.push_back(n.right_child);
            if (n.left_child  >= 0) stack.push_back(n.left_child);
        }
    }
    return leaves_visited;
}

// ---------------------------------------------------------------------------
// DynamicMesh
// ---------------------------------------------------------------------------
void DynamicMesh::reserve(u32 verts, u32 tris) {
    positions.reserve(verts);
    normals.reserve(verts);
    colors.reserve(verts);
    indices.reserve(tris * 3);
}
void DynamicMesh::clear() noexcept {
    positions.clear(); normals.clear(); colors.clear(); indices.clear();
}
u32 DynamicMesh::vertex_count()   const noexcept { return static_cast<u32>(positions.size()); }
u32 DynamicMesh::triangle_count() const noexcept { return static_cast<u32>(indices.size() / 3); }

AABB DynamicMesh::bounds() const noexcept {
    AABB b = AABB::make_empty();
    for (const auto& p : positions) b.expand(p);
    return b;
}

u32 DynamicMesh::add_vertex(const Vec3& p, const Vec3& n, const Vec3& c) {
    positions.push_back(p);
    normals.push_back(n);
    colors.push_back(c);
    return static_cast<u32>(positions.size() - 1);
}
void DynamicMesh::add_triangle(u32 a, u32 b, u32 c) {
    indices.push_back(a); indices.push_back(b); indices.push_back(c);
}

void DynamicMesh::recompute_normals_smooth() {
    for (auto& n : normals) n = {0, 0, 0};
    for (usize i = 0; i + 2 < indices.size(); i += 3) {
        const u32 a = indices[i], b = indices[i + 1], c = indices[i + 2];
        const Vec3& pa = positions[a]; const Vec3& pb = positions[b]; const Vec3& pc = positions[c];
        const Vec3 e1{pb.x - pa.x, pb.y - pa.y, pb.z - pa.z};
        const Vec3 e2{pc.x - pa.x, pc.y - pa.y, pc.z - pa.z};
        const Vec3 fn{ e1.y * e2.z - e1.z * e2.y,
                       e1.z * e2.x - e1.x * e2.z,
                       e1.x * e2.y - e1.y * e2.x };
        normals[a].x += fn.x; normals[a].y += fn.y; normals[a].z += fn.z;
        normals[b].x += fn.x; normals[b].y += fn.y; normals[b].z += fn.z;
        normals[c].x += fn.x; normals[c].y += fn.y; normals[c].z += fn.z;
    }
    for (auto& n : normals) {
        const f32 l = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
        if (l > 1e-6f) { n.x /= l; n.y /= l; n.z /= l; }
    }
}

void DynamicMesh::subdivide_once() {
    std::vector<u32>  new_idx;
    new_idx.reserve(indices.size() * 4);
    std::vector<Vec3> new_pos = positions;
    std::vector<Vec3> new_nrm = normals;
    std::vector<Vec3> new_col = colors;

    auto add_v = [&](const Vec3& p, const Vec3& n, const Vec3& c) {
        new_pos.push_back(p); new_nrm.push_back(n); new_col.push_back(c);
        return static_cast<u32>(new_pos.size() - 1);
    };
    auto mid = [&](u32 a, u32 b) {
        const Vec3 p{ (positions[a].x + positions[b].x) * 0.5f,
                      (positions[a].y + positions[b].y) * 0.5f,
                      (positions[a].z + positions[b].z) * 0.5f };
        const Vec3 n{ (normals[a].x + normals[b].x) * 0.5f,
                      (normals[a].y + normals[b].y) * 0.5f,
                      (normals[a].z + normals[b].z) * 0.5f };
        const Vec3 c{ (colors[a].x + colors[b].x) * 0.5f,
                      (colors[a].y + colors[b].y) * 0.5f,
                      (colors[a].z + colors[b].z) * 0.5f };
        return add_v(p, n, c);
    };
    for (usize i = 0; i + 2 < indices.size(); i += 3) {
        const u32 a = indices[i], b = indices[i + 1], c = indices[i + 2];
        const u32 ab = mid(a, b);
        const u32 bc = mid(b, c);
        const u32 ca = mid(c, a);
        new_idx.push_back(a); new_idx.push_back(ab); new_idx.push_back(ca);
        new_idx.push_back(ab); new_idx.push_back(b); new_idx.push_back(bc);
        new_idx.push_back(ca); new_idx.push_back(bc); new_idx.push_back(c);
        new_idx.push_back(ab); new_idx.push_back(bc); new_idx.push_back(ca);
    }
    positions = std::move(new_pos);
    normals   = std::move(new_nrm);
    colors    = std::move(new_col);
    indices   = std::move(new_idx);
}

void DynamicMesh::weld_vertices(f32 epsilon) {
    if (positions.empty()) return;
    const f32 inv_eps = 1.0f / std::max(1e-6f, epsilon);
    struct Key { i64 x, y, z; bool operator==(const Key& o) const { return x==o.x && y==o.y && z==o.z; } };
    struct Hsh { usize operator()(const Key& k) const noexcept {
        return std::hash<i64>{}(k.x) ^ (std::hash<i64>{}(k.y) << 1) ^ (std::hash<i64>{}(k.z) << 2); }};
    std::unordered_map<Key, u32, Hsh> map;
    std::vector<u32> remap(positions.size(), 0);
    std::vector<Vec3> new_pos, new_nrm, new_col;
    for (u32 i = 0; i < positions.size(); ++i) {
        Key k{ static_cast<i64>(positions[i].x * inv_eps),
               static_cast<i64>(positions[i].y * inv_eps),
               static_cast<i64>(positions[i].z * inv_eps) };
        auto it = map.find(k);
        if (it == map.end()) {
            remap[i] = static_cast<u32>(new_pos.size());
            map.emplace(k, remap[i]);
            new_pos.push_back(positions[i]);
            new_nrm.push_back(normals[i]);
            new_col.push_back(colors[i]);
        } else {
            remap[i] = it->second;
        }
    }
    for (auto& idx : indices) idx = remap[idx];
    positions = std::move(new_pos);
    normals   = std::move(new_nrm);
    colors    = std::move(new_col);
}

}  // namespace cardinal::core::geom
