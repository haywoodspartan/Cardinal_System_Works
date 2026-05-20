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

// =============================================================================
// Nanite-style virtualised geometry — CPU reference impls. The same math
// moves to a compute pass when GPU dispatch lands (see header).
// =============================================================================
namespace {

inline u64 edge_key(u32 a, u32 b) noexcept {
    const u32 lo = (a < b) ? a : b;
    const u32 hi = (a < b) ? b : a;
    return (static_cast<u64>(hi) << 32) | static_cast<u64>(lo);
}

// PN-triangle edge point at the midpoint of edge (P1,N1)->(P2,N2). The PN
// patch restricted to an edge is the cubic through b1=P1, b2, b3, b4=P2;
// at u=0.5 that is (b1 + 3 b2 + 3 b3 + b4)/8. Depends only on the two
// endpoints + their normals, so both triangles that share the edge get
// the identical point — the subdivided mesh stays watertight.
scene::Vec3 pn_edge_mid(const scene::Vec3& P1, const scene::Vec3& N1,
                        const scene::Vec3& P2, const scene::Vec3& N2) noexcept {
    const scene::Vec3 d12 = P2 - P1;
    const scene::Vec3 d21 = P1 - P2;
    const float w12 = scene::dot(d12, N1);
    const float w21 = scene::dot(d21, N2);
    const scene::Vec3 b2 = (P1 * 2.0f + P2 - N1 * w12) * (1.0f / 3.0f);
    const scene::Vec3 b3 = (P2 * 2.0f + P1 - N2 * w21) * (1.0f / 3.0f);
    return (P1 + b2 * 3.0f + b3 * 3.0f + P2) * (1.0f / 8.0f);
}

// Rossignac-Borrel vertex clustering on a triangle soup. Reads `gpos`
// (never mutated here), welds verts onto a coarsening grid sized to ~halve
// the triangle count, and writes a *local* (0-based) simplified mesh into
// out_pos/out_idx. Returns the max vertex displacement — a conservative,
// deterministic geometric-error bound. Caller rebases onto its pools.
float simplify_cluster(const cardinal::vector<scene::Vec3>& gpos,
                       const cardinal::vector<u32>& tri,
                       cardinal::vector<scene::Vec3>& out_pos,
                       cardinal::vector<u32>& out_idx) {
    out_pos.clear();
    out_idx.clear();
    const u32 corners = static_cast<u32>(tri.size());
    if (corners < 3u) return 0.0f;

    scene::Vec3 mn = gpos[tri[0]];
    scene::Vec3 mx = mn;
    for (u32 i = 0; i < corners; ++i) {
        const scene::Vec3& p = gpos[tri[i]];
        mn.x = cardinal::min(mn.x, p.x); mn.y = cardinal::min(mn.y, p.y);
        mn.z = cardinal::min(mn.z, p.z);
        mx.x = cardinal::max(mx.x, p.x); mx.y = cardinal::max(mx.y, p.y);
        mx.z = cardinal::max(mx.z, p.z);
    }
    const scene::Vec3 ext{ mx.x - mn.x, mx.y - mn.y, mx.z - mn.z };

    const u32 tri_count = corners / 3u;
    u32 gridN = static_cast<u32>(cardinal::cbrt(
        cardinal::max(1.0f, static_cast<float>(tri_count) * 0.5f)));
    if (gridN < 2u)  gridN = 2u;
    if (gridN > 32u) gridN = 32u;

    auto axis_cell = [&](float v, float lo, float e) -> u32 {
        // `e <= 1e-12f` is NaN-blind (NaN unordered to 1e-12 → false).
        // A NaN ext (gpos with any NaN component → mn/mx min/max
        // propagates NaN → ext = mx - mn = NaN) reaches the cast as
        // NaN, and `static_cast<int>(NaN)` is UB per [conv.fpint]p1.
        // Same float→int UB cast class as world dd37414, scene
        // ad55e96, brush 79e437d, vgeom 1c537cf, render
        // subdiv_level_for_factor c420695, studio 3b7dd77. Treat
        // any non-finite input as "cell 0" (the same defined
        // collapse used by other axis-bucketing functions in this
        // family). Preserves the meshlet build path for hostile
        // input rather than UB on cast.
        if (!cardinal::isfinite(e) || e <= 1e-12f) return 0u;
        const float scaled = (v - lo) / e * static_cast<float>(gridN);
        if (!cardinal::isfinite(scaled)) return 0u;
        int c = static_cast<int>(scaled);
        if (c < 0) c = 0;
        if (c >= static_cast<int>(gridN)) c = static_cast<int>(gridN) - 1;
        return static_cast<u32>(c);
    };
    auto cell_of = [&](const scene::Vec3& p) -> u32 {
        const u32 cx = axis_cell(p.x, mn.x, ext.x);
        const u32 cy = axis_cell(p.y, mn.y, ext.y);
        const u32 cz = axis_cell(p.z, mn.z, ext.z);
        return (cx * gridN + cy) * gridN + cz;
    };

    cardinal::unordered_map<u32, u32> cell_to_local;
    cardinal::vector<scene::Vec3> sum;
    cardinal::vector<u32>         cnt;
    cardinal::vector<u32>         corner_local;
    corner_local.reserve(corners);
    for (u32 i = 0; i < corners; ++i) {
        const scene::Vec3& p = gpos[tri[i]];
        const u32 c = cell_of(p);
        auto it = cell_to_local.find(c);
        u32 li;
        if (it == cell_to_local.end()) {
            li = static_cast<u32>(sum.size());
            cell_to_local.emplace(c, li);
            sum.push_back(scene::Vec3{0, 0, 0});
            cnt.push_back(0u);
        } else {
            li = it->second;
        }
        sum[li]  = sum[li] + p;
        cnt[li] += 1u;
        corner_local.push_back(li);
    }
    out_pos.resize(sum.size());
    for (usize i = 0; i < sum.size(); ++i)
        out_pos[i] = sum[i] * (1.0f / static_cast<float>(cnt[i]));

    float err = 0.0f;
    for (u32 i = 0; i < corners; ++i) {
        const scene::Vec3 d = gpos[tri[i]] - out_pos[corner_local[i]];
        const float dl = cardinal::sqrt(scene::dot(d, d));
        if (dl > err) err = dl;
    }
    for (u32 t = 0; t < tri_count; ++t) {
        const u32 a = corner_local[t * 3 + 0];
        const u32 b = corner_local[t * 3 + 1];
        const u32 c = corner_local[t * 3 + 2];
        if (a == b || b == c || a == c) continue;     // collapsed → drop
        out_idx.push_back(a);
        out_idx.push_back(b);
        out_idx.push_back(c);
    }
    return err;
}

// 30-bit Morton of a centroid quantised in the level's bounds — gives a
// deterministic, locality-preserving grouping order.
u32 morton3(const scene::Vec3& c, const scene::Vec3& mn,
            const scene::Vec3& ext) noexcept {
    auto q = [](float v, float lo, float e) -> u32 {
        // Same NaN-cast UB class as axis_cell (this file) and the
        // world/scene/brush/vgeom/studio fixes — `e <= 1e-12f` is
        // NaN-blind, and `static_cast<int>(NaN)` is UB per
        // [conv.fpint]p1. Realistic path: build_cluster_dag's sort
        // comparator (this file, line 597+) calls morton3 on
        // dag.nodes[A].bounds.sphere_center. A NaN sphere_center
        // (from a NaN mesh vertex propagated through bounds
        // computation) makes both ma and mb produce UB values; the
        // u32 sort then operates on UB-tainted keys. Coerce
        // non-finite axis to "bucket 0" — defined and deterministic.
        if (!cardinal::isfinite(e) || e <= 1e-12f) return 0u;
        const float scaled = (v - lo) / e * 1023.0f;
        if (!cardinal::isfinite(scaled)) return 0u;
        int i = static_cast<int>(scaled);
        if (i < 0)    i = 0;
        if (i > 1023) i = 1023;
        return static_cast<u32>(i);
    };
    auto split = [](u32 a) -> u32 {
        a &= 0x3FFu;
        a = (a | (a << 16)) & 0x030000FFu;
        a = (a | (a <<  8)) & 0x0300F00Fu;
        a = (a | (a <<  4)) & 0x030C30C3u;
        a = (a | (a <<  2)) & 0x09249249u;
        return a;
    };
    const u32 x = q(c.x, mn.x, ext.x);
    const u32 y = q(c.y, mn.y, ext.y);
    const u32 z = q(c.z, mn.z, ext.z);
    return split(x) | (split(y) << 1) | (split(z) << 2);
}

}  // namespace

u32 subdiv_level_for_factor(float tess_factor, u32 max_levels) noexcept {
    // `tess_factor <= 1.0f` is NaN-blind (NaN<=1 unordered-false). A
    // NaN tess_factor flowed to `log2(NaN) = NaN`, `ceil(NaN) = NaN`,
    // `static_cast<u32>(NaN)` is UB per [conv.fpint]p1 — same float→
    // int cast class fixed across world / mesh_ops / tex_ops / scene /
    // physics / brush / vgeom (1c537cf et al.). +Inf is also a problem
    // independently: log2(+Inf) = +Inf, ceil(+Inf) = +Inf, cast to u32
    // is UB on out-of-range. Treat any non-finite or <= 1 as "no
    // subdivision needed" — the documented semantic for factor 1.0.
    if (!cardinal::isfinite(tess_factor) || tess_factor <= 1.0f) return 0u;
    // factor 2→1, 4→2, 8→3 split passes = ceil(log2(factor)).
    u32 lv = static_cast<u32>(cardinal::ceil(cardinal::log2(tess_factor) - 1e-4f));
    if (lv > max_levels) lv = max_levels;
    return lv;
}

TriMesh subdivide(const TriMesh& in, SubdivOptions opts) {
    TriMesh m = in;
    if ((m.indices.size() % 3u) != 0u) { m.indices.clear(); return m; }
    bool have_n = (!m.normals.empty()) &&
                  (m.normals.size() == m.positions.size());
    if (!have_n) m.normals.clear();

    for (u32 pass = 0; pass < opts.levels; ++pass) {
        if (m.indices.empty()) break;
        cardinal::vector<scene::Vec3> pos = m.positions;
        cardinal::vector<scene::Vec3> nrm = m.normals;
        cardinal::vector<u32>         idx;
        idx.reserve(m.indices.size() * 4u);
        cardinal::unordered_map<u64, u32> mid;
        mid.reserve(m.indices.size());

        auto get_mid = [&](u32 a, u32 b) -> u32 {
            const u64 k = edge_key(a, b);
            auto it = mid.find(k);
            if (it != mid.end()) return it->second;
            const scene::Vec3& Pa = m.positions[a];
            const scene::Vec3& Pb = m.positions[b];
            scene::Vec3 v = (Pa + Pb) * 0.5f;
            if (have_n && opts.curve_blend > 0.0f) {
                const scene::Vec3 cv = pn_edge_mid(Pa, m.normals[a],
                                                   Pb, m.normals[b]);
                v = v + (cv - v) * opts.curve_blend;
            }
            const u32 id = static_cast<u32>(pos.size());
            pos.push_back(v);
            if (have_n) nrm.push_back(scene::normalize(m.normals[a] + m.normals[b]));
            mid.emplace(k, id);
            return id;
        };

        for (usize t = 0; t + 2 < m.indices.size(); t += 3) {
            const u32 i0 = m.indices[t + 0];
            const u32 i1 = m.indices[t + 1];
            const u32 i2 = m.indices[t + 2];
            const u32 a  = get_mid(i0, i1);
            const u32 b  = get_mid(i1, i2);
            const u32 c  = get_mid(i2, i0);
            const u32 sub[12] = { i0, a, c,  i1, b, a,  i2, c, b,  a, b, c };
            for (u32 e : sub) idx.push_back(e);
        }
        m.positions.swap(pos);
        m.normals.swap(nrm);
        m.indices.swap(idx);
    }
    return m;
}

ClusterDag build_cluster_dag(const Mesh& base,
                             const float* positions_xyz, u32 vertex_count,
                             u32 vertex_stride_bytes, DagOptions opts) {
    ClusterDag dag;
    if (positions_xyz == nullptr || vertex_count == 0 ||
        base.meshlets.empty()) return dag;
    if (opts.group_size < 2u) opts.group_size = 2u;
    if (opts.max_levels < 1u) opts.max_levels = 1u;

    dag.positions.reserve(vertex_count);
    for (u32 i = 0; i < vertex_count; ++i)
        dag.positions.push_back(read_vec3(positions_xyz, vertex_stride_bytes, i));

    // Level 0 — the input meshlets, geometry copied verbatim.
    dag.indices = base.indices;
    cardinal::vector<u32> cur;
    cur.reserve(base.meshlets.size());
    for (const auto& ml : base.meshlets) {
        LodNode n;
        n.bounds       = ml.bounds;
        n.lod_level    = 0u;
        n.index_offset = ml.index_offset;
        n.index_count  = ml.index_count;
        cur.push_back(static_cast<u32>(dag.nodes.size()));
        dag.nodes.push_back(n);
    }
    dag.level_count = 1u;

    // Build one parent over the node ids in `ids[g0,g1)`; appends geometry
    // + child links and returns the new node id.
    auto make_parent = [&](const cardinal::vector<u32>& ids,
                           usize g0, usize g1, u32 lvl) -> u32 {
        cardinal::vector<u32> tri;
        float child_err = 0.0f;
        for (usize k = g0; k < g1; ++k) {
            const LodNode& ch = dag.nodes[ids[k]];
            for (u32 i = 0; i < ch.index_count; ++i)
                tri.push_back(dag.indices[ch.index_offset + i]);
            child_err = cardinal::max(child_err, ch.error);
        }
        cardinal::vector<scene::Vec3> lp;
        cardinal::vector<u32>         li;
        const float serr = simplify_cluster(dag.positions, tri, lp, li);

        const u32 io = static_cast<u32>(dag.indices.size());
        if (li.empty()) {
            // Simplifier collapsed everything — keep child geometry so the
            // level still makes progress purely by node-count grouping.
            for (u32 v : tri) dag.indices.push_back(v);
        } else {
            const u32 vbase = static_cast<u32>(dag.positions.size());
            for (const auto& p : lp) dag.positions.push_back(p);
            for (u32 v : li) dag.indices.push_back(vbase + v);
        }
        const u32 icount = static_cast<u32>(dag.indices.size()) - io;

        cardinal::vector<scene::Vec3> vp;
        vp.reserve(icount);
        for (u32 i = 0; i < icount; ++i)
            vp.push_back(dag.positions[dag.indices[io + i]]);

        LodNode p;
        compute_sphere(vp.data(), static_cast<u32>(vp.size()),
                       p.bounds.sphere_center, p.bounds.sphere_radius);
        p.bounds.cone_apex      = p.bounds.sphere_center;
        p.bounds.cone_angle_cos = -1.0f;      // coarse proxy: never backface-cull
        p.error        = cardinal::max(child_err, serr);
        p.lod_level    = lvl;
        p.first_child  = static_cast<u32>(dag.child_ids.size());
        p.child_count  = static_cast<u32>(g1 - g0);
        p.index_offset = io;
        p.index_count  = icount;
        for (usize k = g0; k < g1; ++k) dag.child_ids.push_back(ids[k]);
        const u32 pid = static_cast<u32>(dag.nodes.size());
        dag.nodes.push_back(p);
        return pid;
    };

    u32 level = 0u;
    while (cur.size() > 1u) {
        ++level;
        const bool finalize = (level + 1u >= opts.max_levels);

        scene::Vec3 mn = dag.nodes[cur[0]].bounds.sphere_center;
        scene::Vec3 mx = mn;
        for (u32 id : cur) {
            const scene::Vec3 c = dag.nodes[id].bounds.sphere_center;
            mn.x = cardinal::min(mn.x, c.x); mn.y = cardinal::min(mn.y, c.y);
            mn.z = cardinal::min(mn.z, c.z);
            mx.x = cardinal::max(mx.x, c.x); mx.y = cardinal::max(mx.y, c.y);
            mx.z = cardinal::max(mx.z, c.z);
        }
        const scene::Vec3 ext{ mx.x - mn.x, mx.y - mn.y, mx.z - mn.z };

        cardinal::vector<u32> order = cur;
        cardinal::sort(order.begin(), order.end(), [&](u32 A, u32 B) {
            const u32 ma = morton3(dag.nodes[A].bounds.sphere_center, mn, ext);
            const u32 mb = morton3(dag.nodes[B].bounds.sphere_center, mn, ext);
            if (ma != mb) return ma < mb;
            return A < B;                       // deterministic tiebreak
        });

        cardinal::vector<u32> next;
        const usize gs = finalize ? order.size() : opts.group_size;
        for (usize g = 0; g < order.size(); g += gs) {
            const usize ge = cardinal::min<usize>(g + gs, order.size());
            next.push_back(make_parent(order, g, ge, level));
        }
        cur.swap(next);
        ++dag.level_count;
        if (finalize) break;
    }

    dag.root = cur.empty() ? u32(-1) : cur[0];
    return dag;
}

cardinal::vector<u32> select_lod_cut(const ClusterDag& dag,
                                     const LodCutParams& p) {
    cardinal::vector<u32> out;
    if (dag.nodes.empty() || dag.root == u32(-1)) return out;
    cardinal::vector<u32> stack;
    stack.push_back(dag.root);
    while (!stack.empty()) {
        const u32 id = stack.back();
        stack.pop_back();
        const LodNode& n = dag.nodes[id];
        bool refine = false;
        if (n.child_count > 0u) {
            const scene::Vec3 d = n.bounds.sphere_center - p.camera_pos;
            float dist = cardinal::sqrt(scene::dot(d, d)) - n.bounds.sphere_radius;
            if (dist < 1e-3f) dist = 1e-3f;
            const float screen_err = n.error * p.proj_scale / dist;
            refine = (screen_err > p.error_threshold_px);
        }
        if (refine) {
            for (u32 k = 0; k < n.child_count; ++k)
                stack.push_back(dag.child_ids[n.first_child + k]);
        } else {
            out.push_back(id);
        }
    }
    return out;
}

}  // namespace cardinal::render::geo
