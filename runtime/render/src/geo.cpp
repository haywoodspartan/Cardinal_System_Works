// =============================================================================
// Cardinal — meshlet generation + cluster culling.
// =============================================================================
#include <cardinal/render/geo.hpp>

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/cmath.hpp>
#include <cardinal/core/cstring.hpp>

namespace cardinal::render::geo {

namespace {

inline scene::Vec3 read_vec3(const float* base, u32 stride_bytes, u32 index) {
    const auto* p = reinterpret_cast<const float*>(
        reinterpret_cast<const u8*>(base) + index * stride_bytes);
    return { p[0], p[1], p[2] };
}

// Welzl-ish bounding sphere from points. Cheap O(N²) impl is fine — meshlet
// has at most 64 vertices.
void compute_sphere(const scene::Vec3* p, u32 count,
                    scene::Vec3& out_center, float& out_radius) {
    if (count == 0) { out_center = {}; out_radius = 0.0f; return; }
    // 1) centroid as initial center
    scene::Vec3 c{};
    for (u32 i = 0; i < count; ++i) c += p[i];
    c = c * (1.0f / static_cast<float>(count));
    // 2) farthest point sets the radius
    float r2 = 0.0f;
    for (u32 i = 0; i < count; ++i) {
        const scene::Vec3 d = p[i] - c;
        const float dd = scene::dot(d, d);
        if (dd > r2) r2 = dd;
    }
    out_center = c;
    out_radius = cardinal::sqrt(r2);
}

// Backface cone — average the per-triangle normals (weighted by triangle
// area), then find the maximum half-angle that still encloses every
// triangle. Returns axis (unit) + cos(half_angle).
void compute_cone(const scene::Vec3* tri_normals, u32 tri_count,
                  scene::Vec3& out_axis, float& out_cos) {
    if (tri_count == 0) {
        out_axis = { 0, 0, 1 };
        out_cos  = -1.0f;     // wide-open cone (don't reject)
        return;
    }
    scene::Vec3 axis{};
    for (u32 i = 0; i < tri_count; ++i) axis += tri_normals[i];
    axis = scene::normalize(axis);
    if (scene::dot(axis, axis) < 0.0001f) {
        out_axis = { 0, 0, 1 }; out_cos = -1.0f; return;
    }
    // Find the smallest dot — that's the widest deviation from axis.
    float min_dot = 1.0f;
    for (u32 i = 0; i < tri_count; ++i) {
        const float d = scene::dot(axis, tri_normals[i]);
        if (d < min_dot) min_dot = d;
    }
    out_axis = axis;
    out_cos  = min_dot;
}

scene::Vec3 face_normal(const scene::Vec3& a,
                        const scene::Vec3& b,
                        const scene::Vec3& c)
{
    return scene::normalize(scene::cross(b - a, c - a));
}

}  // namespace

Mesh build_meshlets(const u32* triangle_indices, u32 index_count,
                    const float* positions_xyz, u32 vertex_count,
                    u32 vertex_stride_bytes,
                    const float* normals_xyz, u32 normal_stride_bytes,
                    BuildOptions opts)
{
    Mesh out;
    if (triangle_indices == nullptr || positions_xyz == nullptr ||
        index_count < 3 || (index_count % 3) != 0) {
        return out;
    }
    if (opts.max_verts == 0 || opts.max_prims == 0) opts = {};
    if (opts.max_verts > kMaxVertsPerMeshlet) opts.max_verts = kMaxVertsPerMeshlet;
    if (opts.max_prims > kMaxPrimsPerMeshlet) opts.max_prims = kMaxPrimsPerMeshlet;

    out.indices.reserve(index_count);

    // Greedy partition: scan the triangle list in order, accumulating
    // triangles into the current meshlet until either the unique-vertex
    // count reaches max_verts or the prim count reaches max_prims.
    Meshlet cur;
    cur.index_offset = 0;
    cur.vertex_offset = 0;

    // Map global vertex index → local index inside the current meshlet.
    // 64 entries → small linear search beats cardinal::unordered_map handily.
    u32 local_to_global[kMaxVertsPerMeshlet]{};
    u32 local_count = 0;

    auto find_or_insert = [&](u32 g) -> int {
        for (u32 i = 0; i < local_count; ++i)
            if (local_to_global[i] == g) return static_cast<int>(i);
        if (local_count >= opts.max_verts) return -1;
        local_to_global[local_count] = g;
        return static_cast<int>(local_count++);
    };

    auto flush = [&]() {
        if (cur.index_count == 0) return;

        // Compute bounds on the meshlet's unique vertices.
        scene::Vec3 verts[kMaxVertsPerMeshlet];
        for (u32 i = 0; i < local_count; ++i) {
            verts[i] = read_vec3(positions_xyz, vertex_stride_bytes,
                                 local_to_global[i]);
        }
        compute_sphere(verts, local_count,
                       cur.bounds.sphere_center, cur.bounds.sphere_radius);
        cur.bounds.cone_apex = cur.bounds.sphere_center;

        // Compute a per-triangle normal cone if the caller provided
        // vertex normals; otherwise derive normals from positions so we
        // still get a usable backface cone.
        const u32 tri_count = cur.index_count / 3;
        scene::Vec3 normals[kMaxPrimsPerMeshlet];
        const u32* idx_base = out.indices.data() + cur.index_offset;
        for (u32 t = 0; t < tri_count; ++t) {
            const u32 a = idx_base[t*3 + 0];
            const u32 b = idx_base[t*3 + 1];
            const u32 c = idx_base[t*3 + 2];
            const scene::Vec3 pa = read_vec3(positions_xyz, vertex_stride_bytes, a);
            const scene::Vec3 pb = read_vec3(positions_xyz, vertex_stride_bytes, b);
            const scene::Vec3 pc = read_vec3(positions_xyz, vertex_stride_bytes, c);
            normals[t] = face_normal(pa, pb, pc);
        }
        (void)normals_xyz; (void)normal_stride_bytes;   // reserved for future per-vertex blending
        compute_cone(normals, tri_count,
                     cur.bounds.cone_axis, cur.bounds.cone_angle_cos);

        cur.vertex_count = local_count;
        out.meshlets.push_back(cur);

        // Reset for the next meshlet.
        cur = Meshlet{};
        cur.index_offset  = static_cast<u32>(out.indices.size());
        cur.vertex_offset = 0;
        local_count = 0;
    };

    for (u32 i = 0; i + 2 < index_count; i += 3) {
        const u32 a = triangle_indices[i + 0];
        const u32 b = triangle_indices[i + 1];
        const u32 c = triangle_indices[i + 2];
        if (a >= vertex_count || b >= vertex_count || c >= vertex_count) continue;

        // If we'd exceed the prim cap, flush before adding this tri.
        if (cur.index_count / 3 >= opts.max_prims) flush();

        // Check if we can fit the triangle's three verts (some may already be
        // present). We probe but only commit if all three fit.
        u32 saved_local = local_count;
        const int la = find_or_insert(a);
        const int lb = (la >= 0) ? find_or_insert(b) : -1;
        const int lc = (la >= 0 && lb >= 0) ? find_or_insert(c) : -1;
        if (la < 0 || lb < 0 || lc < 0) {
            // Roll back any inserts then flush + retry on a fresh meshlet.
            local_count = saved_local;
            flush();
            const int la2 = find_or_insert(a);
            const int lb2 = find_or_insert(b);
            const int lc2 = find_or_insert(c);
            (void)la2; (void)lb2; (void)lc2;     // empty meshlet → always fits
            // Use the GLOBAL indices; the consumer applies the meshlet's
            // local→global table when it wants compact indexing.
            out.indices.push_back(a);
            out.indices.push_back(b);
            out.indices.push_back(c);
            cur.index_count += 3;
            continue;
        }
        out.indices.push_back(a);
        out.indices.push_back(b);
        out.indices.push_back(c);
        cur.index_count += 3;
    }
    flush();

    return out;
}

// ---------------------------------------------------------------------------
// Frustum + culling
// ---------------------------------------------------------------------------
Frustum frustum_from_vp(const scene::Mat4& vp) {
    // Gribb-Hartmann — extract frustum planes from a column-major VP.
    // The matrix layout matches our scene::Mat4 (m[col][row]).
    // Each plane has the form (a*x + b*y + c*z + d >= 0) for points inside.
    auto row = [&](int r) -> scene::Vec4 {
        return { vp.m[0][r], vp.m[1][r], vp.m[2][r], vp.m[3][r] };
    };
    const scene::Vec4 r0 = row(0);
    const scene::Vec4 r1 = row(1);
    const scene::Vec4 r2 = row(2);
    const scene::Vec4 r3 = row(3);

    auto add = [](const scene::Vec4& a, const scene::Vec4& b) -> scene::Vec4 {
        return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
    };
    auto sub = [](const scene::Vec4& a, const scene::Vec4& b) -> scene::Vec4 {
        return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
    };
    auto normalize_plane = [](scene::Vec4& p) {
        const float l = cardinal::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
        if (l > 1e-6f) { const float inv = 1.0f / l; p.x*=inv; p.y*=inv; p.z*=inv; p.w*=inv; }
    };

    Frustum f;
    f.planes[0] = add(r3, r0);   // left
    f.planes[1] = sub(r3, r0);   // right
    f.planes[2] = add(r3, r1);   // bottom
    f.planes[3] = sub(r3, r1);   // top
    f.planes[4] = r2;            // near (z >= 0 in clip space)
    f.planes[5] = sub(r3, r2);   // far
    for (auto& p : f.planes) normalize_plane(p);
    return f;
}

bool sphere_inside_frustum(const Frustum& f,
                           const scene::Vec3& center, float radius) noexcept
{
    for (const auto& p : f.planes) {
        const float dist = p.x * center.x + p.y * center.y + p.z * center.z + p.w;
        if (dist < -radius) return false;       // fully outside this plane
    }
    return true;
}

bool cluster_passes_backface(const ClusterBounds& b,
                             const scene::Vec3& camera_pos) noexcept
{
    // If cone_angle_cos == -1 the cone is wide open — never reject.
    if (b.cone_angle_cos <= -1.0f + 1e-4f) return true;

    scene::Vec3 to_cluster = b.sphere_center - camera_pos;
    const float l2 = scene::dot(to_cluster, to_cluster);
    if (l2 < 1e-6f) return true;
    to_cluster = to_cluster * (1.0f / cardinal::sqrt(l2));

    // The cluster is back-facing when EVERY triangle's normal points away
    // from the camera, which is the case when:
    //   dot(view_dir, cone_axis) >= cone_angle_cos
    // (we use the cluster→camera direction; cone_axis points outward).
    const float d = scene::dot(to_cluster, b.cone_axis);
    return d < b.cone_angle_cos;
}

u32 cluster_lod(const ClusterBounds& b, const scene::Vec3& camera_pos,
                const LodConfig& cfg) noexcept
{
    const scene::Vec3 v = b.sphere_center - camera_pos;
    const float d = cardinal::sqrt(scene::dot(v, v));
    if (d < cfg.distance_full)    return 0;
    if (d < cfg.distance_half)    return 1;
    if (d < cfg.distance_quarter) return 2;
    return 3;   // beyond max — caller decides whether to render or cull
}

}  // namespace cardinal::render::geo
