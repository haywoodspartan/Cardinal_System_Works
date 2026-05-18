// =============================================================================
// Cardinal — vgeom cook (build hierarchy from source triangle list).
//
// Two-phase build, repeated per LOD level:
//
//   Phase A — CLUSTERING. Greedy BFS over the triangle adjacency graph:
//             pick the lowest-numbered unassigned tri, expand to its
//             unassigned neighbours (BFS via shared-edge adjacency) until
//             the cluster hits its triangle budget or runs out of
//             neighbours, repeat.
//
//   Phase B — SIMPLIFICATION (for the next-coarser level). Take all the
//             clusters of the current level, group them into groups of
//             kClusterGroupSize, simplify each group's combined triangle
//             list down to ~half via QEM edge collapse. The simplified
//             mesh becomes the input to the next phase A. Repeat until
//             one cluster covers the whole mesh.
//
// Result: a tree (leaves = source detail, root = a coarse blob), packed
// into Hierarchy::clusters + Hierarchy::vertices.
//
// What we are NOT doing in the first cut:
//   - DAG. Adjacent clusters in a group share boundary edges; collapsing
//     across the boundary would let us emit a true DAG so any "cut"
//     stays watertight. We simplify per-group independently, which can
//     produce visible seams between adjacent clusters at different LODs.
//     Acceptable for the first cut; documented for the follow-up.
//   - METIS-quality clustering. The greedy BFS produces reasonable but
//     not optimal clusters. Future: hook METIS / KaHIP for genuine
//     edge-cut minimisation.
//   - Welding. We don't merge duplicate verts across triangle boundaries,
//     so the QEM step treats each shared edge as having two duplicate
//     edges. Slows the simplifier but stays correct.
// =============================================================================
#include <cardinal/vgeom/vgeom.hpp>
#include <cardinal/vgeom/cluster.hpp>

#include <cardinal/core/log.hpp>
#include <cardinal/core/simd_math.hpp>
#include <cardinal/core/typedefs.hpp>

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/cmath.hpp>
#include <cardinal/core/containers.hpp>
#include <cardinal/core/cstring.hpp>
#include <cardinal/core/limits.hpp>
#include <cardinal/core/utility.hpp>

namespace cardinal::vgeom {

namespace {

// -----------------------------------------------------------------------------
// Triangle list → adjacency graph
//
// Each triangle is 3 verts (positions). We build a hash from
// (min_v, max_v) edge-key → list of triangle ids. Two triangles are
// adjacent if they share an edge (key collision). Position hashing
// dedupes verts implicitly so welded and non-welded source meshes
// produce the same adjacency.
// -----------------------------------------------------------------------------
struct VertHash {
    usize operator()(const Vec3& v) const noexcept {
        // Quantise to 1e-5 so floats that ought to be equal map to the
        // same bucket. Source meshes from primitive-makers have exact
        // shared verts but the renderer's expanded triangle-list form
        // means we see THREE copies of each shared vert per quad — we
        // dedupe them here.
        const i32 ix = static_cast<i32>(cardinal::round(v.x * 1e5f));
        const i32 iy = static_cast<i32>(cardinal::round(v.y * 1e5f));
        const i32 iz = static_cast<i32>(cardinal::round(v.z * 1e5f));
        usize h = 0xcbf29ce484222325ull;
        h = (h ^ static_cast<u32>(ix)) * 0x100000001b3ull;
        h = (h ^ static_cast<u32>(iy)) * 0x100000001b3ull;
        h = (h ^ static_cast<u32>(iz)) * 0x100000001b3ull;
        return h;
    }
};
struct VertEq {
    bool operator()(const Vec3& a, const Vec3& b) const noexcept {
        return cardinal::fabs(a.x - b.x) < 1e-4f
            && cardinal::fabs(a.y - b.y) < 1e-4f
            && cardinal::fabs(a.z - b.z) < 1e-4f;
    }
};

struct EdgeKey {
    u32 lo, hi;
    bool operator==(const EdgeKey& o) const noexcept { return lo == o.lo && hi == o.hi; }
};
struct EdgeKeyHash {
    usize operator()(const EdgeKey& k) const noexcept {
        return (static_cast<usize>(k.lo) * 0x9E3779B97F4A7C15ull) ^ k.hi;
    }
};

// Build a triangle adjacency list: tri_adj[t] = up to 3 neighbour tri
// ids (kInvalid for boundary edges). Side effect: vert_dedup maps each
// vertex-position to a canonical id.
struct TriAdjacency {
    cardinal::vector<cardinal::array<u32, 3>> neighbours;     // 3 per tri, kInvalid for boundary
    cardinal::vector<u32>                tri_verts;      // 3 canonical vert ids per tri
    cardinal::vector<Vec3>               canon_positions;
};

TriAdjacency build_adjacency(const Vertex* verts, u32 vc) {
    TriAdjacency a;
    const u32 tri_count = vc / 3;
    if (tri_count == 0) return a;

    // Dedup verts by position so triangles sharing a position share an id.
    cardinal::unordered_map<Vec3, u32, VertHash, VertEq> dedup;
    dedup.reserve(vc);
    a.tri_verts.resize(tri_count * 3);
    a.canon_positions.reserve(vc);
    for (u32 i = 0; i < vc; ++i) {
        const Vec3& p = verts[i].position;
        auto it = dedup.find(p);
        u32 id;
        if (it == dedup.end()) {
            id = static_cast<u32>(a.canon_positions.size());
            a.canon_positions.push_back(p);
            dedup.emplace(p, id);
        } else {
            id = it->second;
        }
        a.tri_verts[i] = id;
    }

    // Hash each edge (sorted) → owning tri id. On collision, link both
    // sides into each other's neighbour slot.
    a.neighbours.assign(tri_count, cardinal::array<u32, 3>{Cluster::kInvalid,
                                                     Cluster::kInvalid,
                                                     Cluster::kInvalid});
    cardinal::unordered_map<EdgeKey, cardinal::pair<u32, u32>, EdgeKeyHash> edge_owner;
    edge_owner.reserve(tri_count * 3);

    auto edge_key = [](u32 va, u32 vb) noexcept -> EdgeKey {
        return va < vb ? EdgeKey{va, vb} : EdgeKey{vb, va};
    };

    for (u32 t = 0; t < tri_count; ++t) {
        const u32 v0 = a.tri_verts[t*3 + 0];
        const u32 v1 = a.tri_verts[t*3 + 1];
        const u32 v2 = a.tri_verts[t*3 + 2];
        const cardinal::array<EdgeKey, 3> ek = {
            edge_key(v0, v1), edge_key(v1, v2), edge_key(v2, v0)
        };
        for (u32 e = 0; e < 3; ++e) {
            auto it = edge_owner.find(ek[e]);
            if (it == edge_owner.end()) {
                edge_owner.emplace(ek[e], cardinal::pair<u32, u32>{t, e});
            } else {
                const auto [other_tri, other_edge] = it->second;
                a.neighbours[t][e]                = other_tri;
                a.neighbours[other_tri][other_edge] = t;
            }
        }
    }
    return a;
}

// -----------------------------------------------------------------------------
// Greedy BFS cluster decomposition.
// Returns one cluster_id per triangle.
// -----------------------------------------------------------------------------
cardinal::vector<u32> decompose_clusters(const TriAdjacency& a, u32 target_tris) {
    const u32 tri_count = static_cast<u32>(a.neighbours.size());
    cardinal::vector<u32> tri_cluster(tri_count, Cluster::kInvalid);
    u32 next_cluster_id = 0;

    for (u32 seed = 0; seed < tri_count; ++seed) {
        if (tri_cluster[seed] != Cluster::kInvalid) continue;
        // Expand from seed.
        u32 cid = next_cluster_id++;
        cardinal::queue<u32> q;
        q.push(seed);
        tri_cluster[seed] = cid;
        u32 in_cluster = 0;
        while (!q.empty() && in_cluster < target_tris) {
            u32 t = q.front(); q.pop();
            ++in_cluster;
            for (u32 e = 0; e < 3; ++e) {
                u32 n = a.neighbours[t][e];
                if (n == Cluster::kInvalid) continue;
                if (tri_cluster[n] != Cluster::kInvalid) continue;
                tri_cluster[n] = cid;
                q.push(n);
                if (in_cluster + static_cast<u32>(q.size()) >= target_tris) break;
            }
        }
    }
    return tri_cluster;
}

// -----------------------------------------------------------------------------
// Quadric Error Metric edge-collapse simplification.
// Garland-Heckbert classic. We collapse the lowest-error edge repeatedly
// until target triangle count or no valid collapse remains.
//
// Returns a new triangle-list Vertex array. Normals are interpolated
// linearly across collapses (good-enough; the "right" answer is to
// recompute from neighbouring faces post-collapse, deferred). Vertex
// colours are interpolated the same way.
// -----------------------------------------------------------------------------
struct Quadric {
    // Symmetric 4x4 represented as upper triangle (10 floats).
    f32 q[10]{};
    void add(const Quadric& o) noexcept {
        for (int i = 0; i < 10; ++i) q[i] += o.q[i];
    }
    f32 evaluate(const Vec3& v) const noexcept {
        // q = [ a b c d ]
        //     [ b e f g ]
        //     [ c f h i ]
        //     [ d g i j ]
        // (we store a,b,c,d,e,f,g,h,i,j in q[0..9])
        const f32 a=q[0], b=q[1], c=q[2], d=q[3];
        const f32 e=q[4], f=q[5], g=q[6];
        const f32 h=q[7], i=q[8];
        const f32 j=q[9];
        const f32 x=v.x, y=v.y, z=v.z;
        return  a*x*x + 2*b*x*y + 2*c*x*z + 2*d*x
              + e*y*y + 2*f*y*z + 2*g*y
              + h*z*z + 2*i*z
              + j;
    }
};

Quadric plane_quadric(const Vec3& p0, const Vec3& p1, const Vec3& p2) noexcept {
    Vec3 e1{p1.x-p0.x, p1.y-p0.y, p1.z-p0.z};
    Vec3 e2{p2.x-p0.x, p2.y-p0.y, p2.z-p0.z};
    Vec3 n{e1.y*e2.z - e1.z*e2.y,
           e1.z*e2.x - e1.x*e2.z,
           e1.x*e2.y - e1.y*e2.x};
    const f32 len = cardinal::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
    if (len > 1e-8f) { n.x/=len; n.y/=len; n.z/=len; }
    const f32 d = -(n.x*p0.x + n.y*p0.y + n.z*p0.z);
    Quadric q;
    q.q[0] = n.x*n.x;          // a
    q.q[1] = n.x*n.y;          // b
    q.q[2] = n.x*n.z;          // c
    q.q[3] = n.x*d;            // d
    q.q[4] = n.y*n.y;          // e
    q.q[5] = n.y*n.z;          // f
    q.q[6] = n.y*d;            // g
    q.q[7] = n.z*n.z;          // h
    q.q[8] = n.z*d;            // i
    q.q[9] = d*d;              // j
    return q;
}

// Simplify a sub-mesh down to roughly target_tris. Input is a triangle-
// list (verts in groups of 3); output is also a triangle list.
//
// First cut uses a SIMPLE non-priority-queue iterative pass: each pass
// finds the lowest-error edge among all candidates, collapses it,
// repeats. O(N²) per collapse — acceptable for the per-group call (groups
// are ~512 tris) and avoids the heap-invalidation complexity of a
// proper priority queue. A real shipping cook upgrades this to
// meshoptimizer's clusterizer + simplifier; the public API stays.
cardinal::vector<Vertex> simplify_qem(const cardinal::vector<Vertex>& in_verts,
                                 u32 target_tris,
                                 f32* out_geometric_error)
{
    const u32 in_tri_count = static_cast<u32>(in_verts.size() / 3);
    if (in_tri_count <= target_tris || target_tris == 0) {
        if (out_geometric_error) *out_geometric_error = 0.0f;
        return in_verts;
    }

    // Dedup positions so we work on a vert graph instead of a triangle
    // soup. We still output triangle-list at the end.
    cardinal::unordered_map<Vec3, u32, VertHash, VertEq> dedup;
    cardinal::vector<Vec3>   pos;
    cardinal::vector<Vec3>   nrm;        // averaged
    cardinal::vector<Vec3>   col;        // averaged
    cardinal::vector<u32>    counts;     // for averaging on merge
    cardinal::vector<cardinal::array<u32, 3>> tris;
    tris.reserve(in_tri_count);

    auto get_id = [&](const Vertex& v) -> u32 {
        auto it = dedup.find(v.position);
        if (it != dedup.end()) {
            const u32 id = it->second;
            // Running average of normal/colour as we re-encounter the vert.
            nrm[id].x = (nrm[id].x * counts[id] + v.normal.x) / (counts[id] + 1);
            nrm[id].y = (nrm[id].y * counts[id] + v.normal.y) / (counts[id] + 1);
            nrm[id].z = (nrm[id].z * counts[id] + v.normal.z) / (counts[id] + 1);
            col[id].x = (col[id].x * counts[id] + v.color.x)  / (counts[id] + 1);
            col[id].y = (col[id].y * counts[id] + v.color.y)  / (counts[id] + 1);
            col[id].z = (col[id].z * counts[id] + v.color.z)  / (counts[id] + 1);
            counts[id]++;
            return id;
        }
        const u32 id = static_cast<u32>(pos.size());
        pos.push_back(v.position);
        nrm.push_back(v.normal);
        col.push_back(v.color);
        counts.push_back(1);
        dedup.emplace(v.position, id);
        return id;
    };
    for (u32 t = 0; t < in_tri_count; ++t) {
        tris.push_back({ get_id(in_verts[t*3+0]),
                         get_id(in_verts[t*3+1]),
                         get_id(in_verts[t*3+2]) });
    }

    // Per-vertex quadric.
    cardinal::vector<Quadric> Q(pos.size());
    for (const auto& tri : tris) {
        Quadric q = plane_quadric(pos[tri[0]], pos[tri[1]], pos[tri[2]]);
        Q[tri[0]].add(q);
        Q[tri[1]].add(q);
        Q[tri[2]].add(q);
    }

    // Track "alive" verts + tris.
    cardinal::vector<u8> tri_alive(tris.size(), 1);
    cardinal::vector<u8> vert_alive(pos.size(), 1);
    u32 alive_tris = static_cast<u32>(tris.size());

    f32 max_collapse_error = 0.0f;

    while (alive_tris > target_tris) {
        // Find the lowest-error candidate edge. O(N) over alive tris.
        f32  best_err  = cardinal::numeric_limits<f32>::max();
        u32  best_a    = Cluster::kInvalid;
        u32  best_b    = Cluster::kInvalid;
        Vec3 best_pos{};

        for (u32 t = 0; t < tris.size(); ++t) {
            if (!tri_alive[t]) continue;
            const auto& tri = tris[t];
            for (int e = 0; e < 3; ++e) {
                u32 a = tri[e], b = tri[(e+1)%3];
                if (a > b) cardinal::swap(a, b);          // canonicalise
                // Candidate collapse position: midpoint (cheap; the
                // analytically-optimal version solves the 3x3 system
                // from Q[a]+Q[b], skipped here for simplicity).
                Vec3 mid{(pos[a].x+pos[b].x)*0.5f,
                         (pos[a].y+pos[b].y)*0.5f,
                         (pos[a].z+pos[b].z)*0.5f};
                Quadric merged = Q[a]; merged.add(Q[b]);
                const f32 err = merged.evaluate(mid);
                if (err < best_err) {
                    best_err = err;
                    best_a   = a;
                    best_b   = b;
                    best_pos = mid;
                }
            }
        }
        if (best_a == Cluster::kInvalid) break;       // nothing more to collapse

        max_collapse_error = cardinal::max(max_collapse_error, cardinal::sqrt(cardinal::max(0.0f, best_err)));

        // Apply the collapse: move a, merge b into a, remove degenerate tris.
        pos[best_a] = best_pos;
        Q[best_a].add(Q[best_b]);
        vert_alive[best_b] = 0;

        // Rewrite tris that touched b → use a; kill tris that now have
        // two of the same vertex.
        for (u32 t = 0; t < tris.size(); ++t) {
            if (!tri_alive[t]) continue;
            auto& tri = tris[t];
            for (int k = 0; k < 3; ++k) {
                if (tri[k] == best_b) tri[k] = best_a;
            }
            if (tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2]) {
                tri_alive[t] = 0;
                --alive_tris;
            }
        }
    }

    if (out_geometric_error) *out_geometric_error = max_collapse_error;

    // Emit surviving triangle-list verts.
    cardinal::vector<Vertex> out;
    out.reserve(static_cast<usize>(alive_tris) * 3);
    for (u32 t = 0; t < tris.size(); ++t) {
        if (!tri_alive[t]) continue;
        const auto& tri = tris[t];
        for (int k = 0; k < 3; ++k) {
            Vertex v{};
            v.position = pos[tri[k]];
            v.normal   = nrm[tri[k]];
            v.color    = col[tri[k]];
            out.push_back(v);
        }
    }
    return out;
}

// Compute bounding sphere of a triangle-list span. Two-pass: centroid,
// then farthest distance.
void compute_bounds(const Vertex* verts, u32 count, Vec3& center, f32& radius) {
    if (count == 0) { center = {0,0,0}; radius = 0; return; }
    Vec3 c{0,0,0};
    for (u32 i = 0; i < count; ++i) {
        c.x += verts[i].position.x;
        c.y += verts[i].position.y;
        c.z += verts[i].position.z;
    }
    c.x /= count; c.y /= count; c.z /= count;
    f32 r2 = 0;
    for (u32 i = 0; i < count; ++i) {
        const f32 dx = verts[i].position.x - c.x;
        const f32 dy = verts[i].position.y - c.y;
        const f32 dz = verts[i].position.z - c.z;
        r2 = cardinal::max(r2, dx*dx + dy*dy + dz*dz);
    }
    center = c;
    radius = cardinal::sqrt(r2);
}

}  // namespace

// =============================================================================
// cook — top-level
// =============================================================================
cardinal::shared_ptr<Hierarchy> cook(const CookDesc& desc) {
    if (desc.vertices == nullptr || desc.vertex_count == 0 ||
        (desc.vertex_count % 3) != 0)
    {
        cardinal::log::errorf("vgeom/cook",
            "bad input: verts=%p count=%u (must be triangle-list)",
            static_cast<const void*>(desc.vertices), desc.vertex_count);
        return nullptr;
    }

    const u32 target = desc.cluster_target_tris > 0
        ? desc.cluster_target_tris : kClusterTargetTris;
    const f32 ratio  = desc.simplify_ratio > 0.0f ? desc.simplify_ratio : 0.5f;

    auto h = cardinal::make_shared<Hierarchy>();
    h->master_tri_count = desc.vertex_count / 3;

    // -------- Build LEVEL 0 clusters (highest detail) ------------------------
    cardinal::vector<Vertex> current_verts(desc.vertices,
                                      desc.vertices + desc.vertex_count);

    // Levels accumulate from the LEAVES up. We finally reverse-index the
    // levels so clusters[0] = root.
    struct Level {
        cardinal::vector<Cluster>  clusters;
        cardinal::vector<Vertex>   verts;     // contiguous, sliced by cluster
    };
    cardinal::vector<Level> levels;

    u32 level_idx = 0;
    while (true) {
        TriAdjacency adj = build_adjacency(current_verts.data(),
                                           static_cast<u32>(current_verts.size()));
        cardinal::vector<u32> tri_cluster = decompose_clusters(adj, target);

        // Group triangle verts by cluster id and build Level entry.
        u32 cluster_count = 0;
        for (u32 cid : tri_cluster) cluster_count = cardinal::max(cluster_count, cid + 1);

        cardinal::vector<cardinal::vector<Vertex>> bucket(cluster_count);
        const u32 tri_count = static_cast<u32>(tri_cluster.size());
        for (u32 t = 0; t < tri_count; ++t) {
            bucket[tri_cluster[t]].push_back(current_verts[t*3 + 0]);
            bucket[tri_cluster[t]].push_back(current_verts[t*3 + 1]);
            bucket[tri_cluster[t]].push_back(current_verts[t*3 + 2]);
        }

        Level L;
        L.clusters.reserve(cluster_count);
        for (u32 c = 0; c < cluster_count; ++c) {
            Cluster cl{};
            cl.level         = level_idx;
            cl.vertex_offset = static_cast<u32>(L.verts.size());
            cl.vertex_count  = static_cast<u32>(bucket[c].size());
            compute_bounds(bucket[c].data(), cl.vertex_count, cl.center, cl.radius);
            L.verts.insert(L.verts.end(), bucket[c].begin(), bucket[c].end());
            L.clusters.push_back(cl);
        }
        levels.push_back(cardinal::move(L));

        if (cluster_count <= 1) break;     // converged — root is the single survivor

        // Build the NEXT (coarser) level: group clusters of
        // kClusterGroupSize, concat their tris, simplify to ratio.
        cardinal::vector<Vertex> next_verts;
        next_verts.reserve(current_verts.size() / 2);
        for (u32 g = 0; g < cluster_count; g += kClusterGroupSize) {
            cardinal::vector<Vertex> grouped;
            for (u32 k = 0; k < kClusterGroupSize && g + k < cluster_count; ++k) {
                grouped.insert(grouped.end(),
                               bucket[g + k].begin(), bucket[g + k].end());
            }
            const u32 grouped_tris = static_cast<u32>(grouped.size() / 3);
            const u32 target_tris  = cardinal::max<u32>(1,
                static_cast<u32>(static_cast<f32>(grouped_tris) * ratio));
            f32 err = 0.0f;
            cardinal::vector<Vertex> simp = simplify_qem(grouped, target_tris, &err);
            (void)err;     // attached to the parent cluster below

            // QEM edge collapse linearly averages the normals of merged
            // vertices — they drift away from unit length. Re-normalise
            // through the dispatched SIMD path so the parent-level
            // verts have proper unit normals before they get sampled
            // by lighting (the renderer's color_for treats normals as
            // pre-normalised). Gather → normalise → scatter pattern;
            // cook is offline so the alloc is fine.
            if (!simp.empty()) {
                cardinal::FloatVec normals_xyz(simp.size() * 3);
                for (usize v = 0; v < simp.size(); ++v) {
                    normals_xyz[v*3 + 0] = simp[v].normal.x;
                    normals_xyz[v*3 + 1] = simp[v].normal.y;
                    normals_xyz[v*3 + 2] = simp[v].normal.z;
                }
                cardinal::core::simd::vec3_normalize_array_inplace(
                    normals_xyz.data(), simp.size());
                for (usize v = 0; v < simp.size(); ++v) {
                    simp[v].normal.x = normals_xyz[v*3 + 0];
                    simp[v].normal.y = normals_xyz[v*3 + 1];
                    simp[v].normal.z = normals_xyz[v*3 + 2];
                }
            }

            next_verts.insert(next_verts.end(), simp.begin(), simp.end());
        }
        // If the simplifier made no progress (every group failed to
        // collapse), break to avoid an infinite loop.
        if (next_verts.size() >= current_verts.size()) break;
        current_verts = cardinal::move(next_verts);

        ++level_idx;
        if (level_idx > 32) break;         // safety — should never reach this
    }

    // Pack levels into the final hierarchy. clusters[0] = root (deepest
    // level we built). Leaves are at the largest level index, which is
    // levels[0] in our build order. Reverse so the root sits at the front.
    cardinal::reverse(levels.begin(), levels.end());

    // Concatenate clusters + verts; remap vertex_offset to be global.
    cardinal::vector<u32> level_first_cluster(levels.size(), 0);
    u32 cluster_cursor = 0;
    for (usize li = 0; li < levels.size(); ++li) {
        level_first_cluster[li] = cluster_cursor;
        cluster_cursor += static_cast<u32>(levels[li].clusters.size());
    }
    h->clusters.reserve(cluster_cursor);
    h->vertices.reserve([&]{
        usize total = 0;
        for (auto& L : levels) total += L.verts.size();
        return total;
    }());

    u32 global_vert_offset = 0;
    for (usize li = 0; li < levels.size(); ++li) {
        const u32 first_local_vert = global_vert_offset;
        h->vertices.insert(h->vertices.end(),
                           levels[li].verts.begin(), levels[li].verts.end());
        for (auto& cl : levels[li].clusters) {
            cl.vertex_offset += first_local_vert;
            // parent_id / child links — leaves have no children;
            // non-leaves point at the next level's clusters. The first-
            // cut pack uses spatial proximity matching: each parent's
            // children are the kClusterGroupSize clusters of the
            // immediately-finer level whose centres are closest. This is
            // approximate (the cook-time grouping was index-order, not
            // proximity-order) — produces functional hierarchy with
            // occasionally non-tight bounds.
            //
            // Real shipping cook would track child indices DURING the
            // group step instead of recovering them after the fact.
            cl.parent_id      = Cluster::kInvalid;
            cl.first_child_id = Cluster::kInvalid;
            cl.child_count    = 0;
            h->clusters.push_back(cl);
        }
        global_vert_offset += static_cast<u32>(levels[li].verts.size());
    }

    // Build parent/child links by walking levels root-down: each parent
    // claims kClusterGroupSize consecutive children at the next level.
    for (usize li = 0; li + 1 < levels.size(); ++li) {
        const u32 parent_base  = level_first_cluster[li];
        const u32 parent_count = static_cast<u32>(levels[li].clusters.size());
        const u32 child_base   = level_first_cluster[li + 1];
        const u32 child_total  = static_cast<u32>(levels[li + 1].clusters.size());
        for (u32 p = 0; p < parent_count; ++p) {
            const u32 first = child_base + p * kClusterGroupSize;
            if (first >= child_base + child_total) {
                h->clusters[parent_base + p].first_child_id = Cluster::kInvalid;
                h->clusters[parent_base + p].child_count    = 0;
                continue;
            }
            const u32 take = cardinal::min<u32>(kClusterGroupSize,
                child_base + child_total - first);
            h->clusters[parent_base + p].first_child_id = first;
            h->clusters[parent_base + p].child_count    = take;
            for (u32 k = 0; k < take; ++k) {
                h->clusters[first + k].parent_id = parent_base + p;
            }
        }
    }

    // Geometric error per cluster = simplification error from THIS
    // cluster's level to its parent. We approximate: take the cluster's
    // bounds radius as a coarse upper bound on the visible-error scale.
    // A proper cook stores the QEM error from simplify_qem per-parent;
    // first cut uses radius as a reasonable proxy that monotonically
    // increases with level (root > leaves).
    for (auto& cl : h->clusters) {
        cl.geometric_error = cl.radius * 0.5f;
    }

    h->level_count         = static_cast<u32>(levels.size());
    h->leaf_count          = static_cast<u32>(levels.back().clusters.size());
    h->total_cluster_count = static_cast<u32>(h->clusters.size());
    h->root_center         = h->clusters.front().center;
    h->root_radius         = h->clusters.front().radius;

    cardinal::log::infof("vgeom/cook",
        "%s — %u tris → %u clusters across %u levels (root r=%.2f, leaves=%u)",
        desc.name ? desc.name : "(unnamed)",
        h->master_tri_count, h->total_cluster_count, h->level_count,
        h->root_radius, h->leaf_count);

    return h;
}

}  // namespace cardinal::vgeom
