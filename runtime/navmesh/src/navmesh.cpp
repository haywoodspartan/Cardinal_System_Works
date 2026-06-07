#include <cardinal/navmesh/navmesh.hpp>

#include <cardinal/core/std/algorithm.hpp>    // cardinal::max/reverse/*_heap
#include <cardinal/core/std/cmath.hpp>        // cardinal::sqrt
#include <cardinal/core/std/containers.hpp>   // cardinal::unordered_map
#include <cardinal/core/std/limits.hpp>       // cardinal::numeric_limits
// cardinal::pair/make_pair/hash arrive via navmesh.hpp + core/types.hpp

namespace cardinal::navmesh {

namespace {

inline f32 vdist(const cardinal::scene::Vec3& a, const cardinal::scene::Vec3& b) {
    const f32 dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return cardinal::sqrt(dx*dx + dy*dy + dz*dz);
}

inline f32 cross_xz(const cardinal::scene::Vec3& a,
                    const cardinal::scene::Vec3& b,
                    const cardinal::scene::Vec3& c) {
    // Signed 2D cross (XZ plane).
    return (b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x);
}

// Edge key — undirected pair of vertex indices.
struct EdgeKey {
    u32 a, b;
    bool operator==(const EdgeKey& o) const noexcept { return a == o.a && b == o.b; }
};
struct EdgeHash { usize operator()(const EdgeKey& k) const noexcept {
    return cardinal::hash<u64>{}((static_cast<u64>(k.a) << 32) | k.b);
}};
EdgeKey make_edge(u32 a, u32 b) {
    return (a < b) ? EdgeKey{a, b} : EdgeKey{b, a};
}

}  // namespace

// ---------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------
void Mesh::recompute_centroids() noexcept {
    for (auto& p : polys) {
        const auto& a = vertices[p.verts[0]];
        const auto& b = vertices[p.verts[1]];
        const auto& c = vertices[p.verts[2]];
        p.centroid = { (a.x + b.x + c.x) / 3.0f,
                       (a.y + b.y + c.y) / 3.0f,
                       (a.z + b.z + c.z) / 3.0f };
    }
}

void Mesh::build_from_triangles(const cardinal::vector<cardinal::scene::Vec3>& verts,
                                const cardinal::vector<u32>& tri_indices)
{
    clear();
    vertices = verts;
    polys.reserve(tri_indices.size() / 3);
    cardinal::unordered_map<EdgeKey, cardinal::pair<u32, u32>, EdgeHash> edge_to_poly_edge;
    for (usize i = 0; i + 2 < tri_indices.size(); i += 3) {
        Poly p{};
        p.verts = { tri_indices[i], tri_indices[i + 1], tri_indices[i + 2] };
        polys.push_back(p);
    }
    // Build adjacency by looking up shared edges.
    for (u32 pi = 0; pi < polys.size(); ++pi) {
        Poly& p = polys[pi];
        for (u32 ei = 0; ei < 3; ++ei) {
            const u32 a = p.verts[ei];
            const u32 b = p.verts[(ei + 1) % 3];
            const EdgeKey k = make_edge(a, b);
            auto it = edge_to_poly_edge.find(k);
            if (it == edge_to_poly_edge.end()) {
                edge_to_poly_edge.emplace(k, cardinal::make_pair(pi, ei));
            } else {
                const auto [other_pi, other_ei] = it->second;
                p.neighbours[ei]                    = other_pi;
                polys[other_pi].neighbours[other_ei]= pi;
            }
        }
    }
    recompute_centroids();
}

PolyId Mesh::nearest_poly(const cardinal::scene::Vec3& p) const noexcept {
    if (polys.empty()) return kInvalidPoly;
    f32 best = cardinal::numeric_limits<f32>::max();
    PolyId best_id = 0;
    for (u32 i = 0; i < polys.size(); ++i) {
        const f32 d = vdist(p, polys[i].centroid);
        if (d < best) { best = d; best_id = i; }
    }
    return best_id;
}

// ---------------------------------------------------------------------------
// PathQuery
// ---------------------------------------------------------------------------
void PathQuery::open_push_(PolyId id, f32 fv) {
    open_.emplace_back(fv, id);
    cardinal::push_heap(open_.begin(), open_.end(),
        [](const auto& a, const auto& b){ return a.first > b.first; });
}
bool PathQuery::open_pop_(PolyId& out, f32& f) {
    if (open_.empty()) return false;
    cardinal::pop_heap(open_.begin(), open_.end(),
        [](const auto& a, const auto& b){ return a.first > b.first; });
    f   = open_.back().first;
    out = open_.back().second;
    open_.pop_back();
    return true;
}

namespace {

// Funnel string-pulling — UE5-style. Walks the polygon path and shortens
// it by looking at portal-edge endpoints (left + right), only adding a
// waypoint when the funnel would invert.
void funnel_path(const Mesh& mesh,
                 const cardinal::scene::Vec3& start,
                 const cardinal::scene::Vec3& goal,
                 const cardinal::vector<PolyId>& poly_path,
                 cardinal::vector<cardinal::scene::Vec3>& out)
{
    out.clear();
    out.push_back(start);
    if (poly_path.size() <= 1) {
        out.push_back(goal);
        return;
    }
    // For each transition (poly_path[i] -> poly_path[i+1]), find the
    // shared edge and put its left/right endpoints into a portals list.
    struct Portal { cardinal::scene::Vec3 left, right; };
    cardinal::vector<Portal> portals;
    portals.reserve(poly_path.size());
    for (usize i = 0; i + 1 < poly_path.size(); ++i) {
        const Poly& a = mesh.polys[poly_path[i]];
        const PolyId next = poly_path[i + 1];
        for (u32 ei = 0; ei < 3; ++ei) {
            if (a.neighbours[ei] == next) {
                const u32 va = a.verts[ei];
                const u32 vb = a.verts[(ei + 1) % 3];
                Portal pt{};
                pt.left  = mesh.vertices[va];
                pt.right = mesh.vertices[vb];
                portals.push_back(pt);
                break;
            }
        }
    }
    // Final portal = the goal point as a degenerate edge.
    Portal goal_p{ goal, goal };
    portals.push_back(goal_p);

    cardinal::scene::Vec3 apex = start;
    cardinal::scene::Vec3 left_p  = portals[0].left;
    cardinal::scene::Vec3 right_p = portals[0].right;
    usize apex_index = 0, left_index = 0, right_index = 0;

    for (usize i = 1; i < portals.size(); ++i) {
        const auto new_left  = portals[i].left;
        const auto new_right = portals[i].right;
        // Update right.
        if (cross_xz(apex, right_p, new_right) <= 0.0f) {
            if (apex.x == right_p.x && apex.z == right_p.z ||
                cross_xz(apex, left_p, new_right) > 0.0f)
            {
                right_p     = new_right;
                right_index = i;
            } else {
                out.push_back(left_p);
                apex        = left_p;
                apex_index  = left_index;
                left_p      = apex;
                right_p     = apex;
                left_index  = apex_index;
                right_index = apex_index;
                i = apex_index;
                continue;
            }
        }
        // Update left.
        if (cross_xz(apex, left_p, new_left) >= 0.0f) {
            if (apex.x == left_p.x && apex.z == left_p.z ||
                cross_xz(apex, right_p, new_left) < 0.0f)
            {
                left_p     = new_left;
                left_index = i;
            } else {
                out.push_back(right_p);
                apex        = right_p;
                apex_index  = right_index;
                left_p      = apex;
                right_p     = apex;
                left_index  = apex_index;
                right_index = apex_index;
                i = apex_index;
                continue;
            }
        }
    }
    out.push_back(goal);
}

}  // namespace

PathStats PathQuery::find_path(const Mesh& mesh,
                                const cardinal::scene::Vec3& start,
                                const cardinal::scene::Vec3& goal,
                                cardinal::vector<cardinal::scene::Vec3>& out_waypoints)
{
    PathStats st{};
    out_waypoints.clear();
    if (mesh.empty()) return st;

    const PolyId start_p = mesh.nearest_poly(start);
    const PolyId goal_p  = mesh.nearest_poly(goal);
    if (start_p == kInvalidPoly || goal_p == kInvalidPoly) return st;

    const u32 N = static_cast<u32>(mesh.polys.size());
    g_.assign(N, cardinal::numeric_limits<f32>::max());
    f_.assign(N, cardinal::numeric_limits<f32>::max());
    parent_.assign(N, kInvalidPoly);
    closed_.assign(N, false);
    open_.clear();

    g_[start_p] = 0.0f;
    f_[start_p] = vdist(mesh.polys[start_p].centroid, mesh.polys[goal_p].centroid);
    open_push_(start_p, f_[start_p]);

    while (true) {
        PolyId cur; f32 fc;
        if (!open_pop_(cur, fc)) break;
        if (closed_[cur]) continue;
        closed_[cur] = true;
        ++st.poly_visited;
        st.open_set_max = cardinal::max(st.open_set_max, static_cast<u32>(open_.size()));
        if (cur == goal_p) {
            // Reconstruct polygon path.
            cardinal::vector<PolyId> rev;
            rev.push_back(cur);
            while (parent_[rev.back()] != kInvalidPoly) rev.push_back(parent_[rev.back()]);
            cardinal::reverse(rev.begin(), rev.end());
            funnel_path(mesh, start, goal, rev, out_waypoints);
            st.found          = true;
            st.poly_path_cost = g_[cur];
            st.waypoint_count = static_cast<u32>(out_waypoints.size());
            return st;
        }
        for (u32 ei = 0; ei < 3; ++ei) {
            const PolyId nb = mesh.polys[cur].neighbours[ei];
            if (nb == kInvalidPoly || closed_[nb]) continue;
            const f32 step = vdist(mesh.polys[cur].centroid, mesh.polys[nb].centroid);
            const f32 tentative = g_[cur] + step;
            if (tentative < g_[nb]) {
                g_[nb] = tentative;
                parent_[nb] = cur;
                f_[nb] = tentative + vdist(mesh.polys[nb].centroid, mesh.polys[goal_p].centroid);
                open_push_(nb, f_[nb]);
            }
        }
    }
    return st;
}

}  // namespace cardinal::navmesh
