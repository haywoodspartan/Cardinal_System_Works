// =============================================================================
// Cardinal — vgeom runtime cut selection.
//
// Algorithm (from the public-API comment in vgeom.hpp, "step 4"):
//
//   def select(node):
//       if leaf(node) or projected_pixel_radius(node) * geometric_error(node)
//                       <= pixel_tolerance:
//           emit(node)        # use this cluster
//       else:
//           for c in children(node): select(c)
//
// Iteratively (no recursion — keeps the call stack bounded for deep
// hierarchies) via a small stack vector.
//
// Frustum culling lives here too: a cluster whose bounding sphere is
// entirely outside the camera frustum is dropped without being descended
// into. Saves a lot of work for off-screen geometry.
// =============================================================================
#include <cardinal/vgeom/vgeom.hpp>
#include <cardinal/vgeom/cluster.hpp>

#include <cardinal/core/simd_math.hpp>
#include <cardinal/core/types.hpp>

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/cmath.hpp>
#include <cardinal/core/containers.hpp>
#include <cardinal/core/utility.hpp>

namespace cardinal::vgeom {

namespace {

// Build the 6 frustum planes from a view-projection matrix
// (Gribb–Hartmann method). Plane format: (nx, ny, nz, d) with the
// convention "point on the inside has nx*x + ny*y + nz*z + d >= 0".
struct Plane { f32 nx, ny, nz, d; };
struct Frustum6 { cardinal::array<Plane, 6> p; };

Frustum6 frustum_from_vp(const Mat4& vp) noexcept {
    Frustum6 f{};
    auto plane = [&](Plane& p, f32 a, f32 b, f32 c, f32 d) noexcept {
        const f32 len = cardinal::sqrt(a*a + b*b + c*c);
        const f32 inv = (len > 1e-8f) ? (1.0f / len) : 1.0f;
        p.nx = a * inv; p.ny = b * inv; p.nz = c * inv; p.d = d * inv;
    };
    // Row-major access (m[col][row] in column-major Mat4 → row r col c
    // = m[c][r]).
    auto v = [&](int r, int c) noexcept { return vp.m[c][r]; };
    plane(f.p[0],  v(3,0)+v(0,0),  v(3,1)+v(0,1),  v(3,2)+v(0,2),  v(3,3)+v(0,3));   // left
    plane(f.p[1],  v(3,0)-v(0,0),  v(3,1)-v(0,1),  v(3,2)-v(0,2),  v(3,3)-v(0,3));   // right
    plane(f.p[2],  v(3,0)+v(1,0),  v(3,1)+v(1,1),  v(3,2)+v(1,2),  v(3,3)+v(1,3));   // bottom
    plane(f.p[3],  v(3,0)-v(1,0),  v(3,1)-v(1,1),  v(3,2)-v(1,2),  v(3,3)-v(1,3));   // top
    plane(f.p[4],  v(3,0)+v(2,0),  v(3,1)+v(2,1),  v(3,2)+v(2,2),  v(3,3)+v(2,3));   // near
    plane(f.p[5],  v(3,0)-v(2,0),  v(3,1)-v(2,1),  v(3,2)-v(2,2),  v(3,3)-v(2,3));   // far
    return f;
}

// Max-column-length of model — the uniform-scale approximation used to
// scale a local-space radius into world space. Constant per select()
// call (one matrix per call), so we hoist it outside the BFS loop and
// apply it via a single SIMD vec_scale_f32 over the radii array.
//
// Same approximation as project_sphere_pixel_radius in cluster.cpp;
// non-uniform scale would require an OBB or per-axis bound.
f32 model_radius_scale(const Mat4& m) noexcept {
    auto col_len = [&](int c) noexcept {
        return cardinal::sqrt(m.m[c][0]*m.m[c][0]
                       + m.m[c][1]*m.m[c][1]
                       + m.m[c][2]*m.m[c][2]);
    };
    return cardinal::max(col_len(0), cardinal::max(col_len(1), col_len(2)));
}

}  // namespace

void select(const SelectInput& in, SelectOutput& out) {
    out.cluster_ids.clear();
    out.stats = FrameStats{};
    if (in.hierarchy == nullptr || in.hierarchy->clusters.empty()) return;

    out.stats.master_tri_count = in.hierarchy->master_tri_count;

    // Build vp once for frustum + screen projection.
    auto matmul = [](const Mat4& a, const Mat4& b) noexcept {
        Mat4 r{};
        for (int c = 0; c < 4; ++c)
        for (int rr = 0; rr < 4; ++rr) {
            f32 s = 0;
            for (int k = 0; k < 4; ++k) s += a.m[k][rr] * b.m[c][k];
            r.m[c][rr] = s;
        }
        return r;
    };
    const Mat4 vp = matmul(in.proj, in.view);
    const Frustum6 frustum = frustum_from_vp(vp);

    // Flatten the 6 planes into the 24-float layout simd::frustum_cull_spheres
    // wants: { nx, ny, nz, d } × 6.
    f32 planes_flat[24];
    for (int p = 0; p < 6; ++p) {
        planes_flat[p*4 + 0] = frustum.p[p].nx;
        planes_flat[p*4 + 1] = frustum.p[p].ny;
        planes_flat[p*4 + 2] = frustum.p[p].nz;
        planes_flat[p*4 + 3] = frustum.p[p].d;
    }

    // BFS by level: pack the current frontier into SoA, batch-cull via
    // SIMD (16 spheres / iter on AVX-512), then per-cluster apply the
    // screen-space LOD test and produce the next frontier from
    // surviving inner nodes' children. This preserves the original
    // DFS's subtree-pruning behaviour (a culled cluster's children
    // never enter the next frontier) while letting the cull run
    // batched instead of one scalar 6-plane test per cluster.
    //
    // Thread-local storage for the per-call scratch — multiple threads
    // calling select() concurrently each get their own buffers.
    static thread_local cardinal::vector<u32> frontier;
    static thread_local cardinal::vector<u32> next_frontier;
    static thread_local cardinal::vector<f32> local_xyz;   // AoS: 3*N floats
    static thread_local cardinal::vector<f32> world_xyz;   // AoS: 3*N floats
    static thread_local cardinal::vector<f32> local_r;     // N floats
    static thread_local cardinal::vector<f32> sx, sy, sz, sr;
    static thread_local cardinal::vector<u8>  bits;

    frontier.clear();
    frontier.reserve(256);
    frontier.push_back(0u);
    out.cluster_ids.reserve(64);

    // Hoist model → SIMD-friendly Mat4Compact + the world radius scale
    // OUTSIDE the BFS loop. Both are constants for this select() call.
    cardinal::core::simd::Mat4Compact M_compact{};
    // Mat4 is column-major flat 16 floats; Mat4Compact is 16 floats. Same layout.
    {
        const f32* src = &in.model.m[0][0];
        for (int k = 0; k < 16; ++k) M_compact.m[k] = src[k];
    }
    const f32 radius_scale = model_radius_scale(in.model);

    while (!frontier.empty()) {
        const usize n = frontier.size();
        local_xyz.resize(n * 3);
        local_r  .resize(n);

        // Pack local-space sphere centres into AoS xyz triples + radii
        // into a parallel array. This is the only per-cluster scalar
        // step left in the cull pipeline.
        for (usize i = 0; i < n; ++i) {
            const Cluster& c = in.hierarchy->clusters[frontier[i]];
            local_xyz[i*3 + 0] = c.center.x;
            local_xyz[i*3 + 1] = c.center.y;
            local_xyz[i*3 + 2] = c.center.z;
            local_r[i]         = c.radius;
        }

        // SIMD-batched M·local_xyz → world_xyz. transform_points_mat4
        // walks the AoS triples in one cache-friendly pass; on AVX-512
        // it processes 16 vertices per iteration.
        world_xyz.resize(n * 3);
        cardinal::core::simd::transform_points_mat4(
            world_xyz.data(), local_xyz.data(), M_compact, n);

        // SIMD-batched radius scale — single pass over local_r → sr.
        sr.resize(n);
        cardinal::core::simd::vec_scale_f32(
            sr.data(), local_r.data(), radius_scale, n);

        // Scatter AoS world centres into the SoA cx/cy/cz arrays the
        // cull kernel wants. (frustum_cull_spheres takes 4 SoA arrays;
        // a future "transform_points_mat4_to_soa" primitive would let
        // us skip this scatter, but for typical frontier sizes the
        // memory bandwidth cost is negligible.)
        sx.resize(n); sy.resize(n); sz.resize(n);
        for (usize i = 0; i < n; ++i) {
            sx[i] = world_xyz[i*3 + 0];
            sy[i] = world_xyz[i*3 + 1];
            sz[i] = world_xyz[i*3 + 2];
        }

        // Batched 6-plane reject — bit i of bits[i/8] is set iff
        // cluster `frontier[i]` is at least partially inside the frustum.
        bits.assign((n + 7) / 8, 0u);
        cardinal::core::simd::frustum_cull_spheres(
            bits.data(), planes_flat,
            sx.data(), sy.data(), sz.data(), sr.data(), n);

        next_frontier.clear();
        for (usize i = 0; i < n; ++i) {
            const u32 cid = frontier[i];
            const bool visible = (bits[i >> 3] & (1u << (i & 7))) != 0;
            if (!visible) {
                ++out.stats.culled_cluster_count;
                continue;
            }
            const Cluster& c = in.hierarchy->clusters[cid];
            const bool is_leaf = (c.child_count == 0 ||
                                  c.first_child_id == Cluster::kInvalid);
            bool emit_this = is_leaf;
            if (!emit_this) {
                const f32 pixel_r = project_sphere_pixel_radius(
                    c.center, c.radius,
                    in.model, in.view, in.proj,
                    in.viewport_pixel_height);
                // Error projected to pixels = pixel_r/local_radius * local_error.
                // Equivalently: (pixel_r * geometric_error) / local_radius.
                const f32 pixel_error = (c.radius > 1e-6f)
                    ? (pixel_r * c.geometric_error / c.radius)
                    : 0.0f;
                emit_this = (pixel_error <= in.pixel_error_tolerance);
            }
            if (emit_this) {
                out.cluster_ids.push_back(cid);
                ++out.stats.cut_cluster_count;
                out.stats.drawn_tri_count += c.vertex_count / 3;
            } else {
                for (u32 k = 0; k < c.child_count; ++k) {
                    next_frontier.push_back(c.first_child_id + k);
                }
            }
        }
        cardinal::swap(frontier, next_frontier);
    }

    // ----- Morton-order the cut for spatial draw locality ----------------
    //
    // BFS-by-level emits cluster_ids in level-order, which has no spatial
    // coherence — adjacent draws can land on opposite sides of the mesh
    // and thrash the GPU's vertex / texture cache. Sorting by 3D Morton
    // (Z-curve) of the cluster centre groups spatially-near clusters
    // adjacently in the draw stream.
    //
    // Quantisation: project each centre into the root cluster's bounding
    // sphere and map to the 21-bit grid simd::morton_encode_3d_array
    // expects. The root sphere covers the whole mesh, so this maps the
    // mesh's local bounds to [0, 2^21-1] in each axis.
    if (out.cluster_ids.size() > 1 && !in.hierarchy->clusters.empty()) {
        static thread_local cardinal::vector<u32>                 mqx, mqy, mqz;
        static thread_local cardinal::vector<u64>                 mort_codes;
        static thread_local cardinal::vector<cardinal::pair<u64, u32>> mort_pairs;

        const usize n = out.cluster_ids.size();
        mqx.resize(n); mqy.resize(n); mqz.resize(n);
        mort_codes.resize(n);
        mort_pairs.resize(n);

        const Cluster& root = in.hierarchy->clusters[0];
        const f32 ext = root.radius * 2.0f;
        const f32 inv = (ext > 1e-6f) ? (2097151.0f / ext) : 0.0f;
        const f32 base_x = root.center.x - root.radius;
        const f32 base_y = root.center.y - root.radius;
        const f32 base_z = root.center.z - root.radius;

        for (usize i = 0; i < n; ++i) {
            const Vec3& c = in.hierarchy->clusters[out.cluster_ids[i]].center;
            mqx[i] = static_cast<u32>(cardinal::clamp((c.x - base_x) * inv, 0.0f, 2097151.0f));
            mqy[i] = static_cast<u32>(cardinal::clamp((c.y - base_y) * inv, 0.0f, 2097151.0f));
            mqz[i] = static_cast<u32>(cardinal::clamp((c.z - base_z) * inv, 0.0f, 2097151.0f));
        }
        cardinal::core::simd::morton_encode_3d_array(
            mort_codes.data(), mqx.data(), mqy.data(), mqz.data(), n);
        for (usize i = 0; i < n; ++i) {
            mort_pairs[i] = { mort_codes[i], out.cluster_ids[i] };
        }
        cardinal::sort(mort_pairs.begin(), mort_pairs.end(),
                  [](const auto& a, const auto& b){ return a.first < b.first; });
        for (usize i = 0; i < n; ++i) {
            out.cluster_ids[i] = mort_pairs[i].second;
        }
    }
}

}  // namespace cardinal::vgeom
