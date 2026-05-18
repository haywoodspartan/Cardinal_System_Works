#pragma once

// =============================================================================
// Cardinal — Polygon Navmesh + Funnel pathfinding.
//
// Richer than nav::Grid: navigable surface as a triangulation of arbitrary
// convex polygons (typically triangles). Edges can be marked "portal" if
// they connect adjacent polygons; A* over polygon centroids gives the
// "polygon path", then the **funnel algorithm** straightens it through
// the portals into a sequence of waypoint positions.
//
// Build path: start with a list of triangles + adjacency. We expose:
//   - Mesh container (Polys + Verts)
//   - PathQuery (node A*, then funnel to straight waypoints)
//   - Nearest-poly query for "snap a world point to the navmesh"
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/containers.hpp>   // cardinal::array, cardinal::vector
#include <cardinal/core/utility.hpp>      // cardinal::pair
#include <cardinal/scene/math.hpp>

namespace cardinal::navmesh {

using PolyId = u32;
inline constexpr PolyId kInvalidPoly = 0xFFFFFFFFu;

struct Poly {
    cardinal::array<u32,    3> verts {0, 0, 0};
    cardinal::array<PolyId, 3> neighbours{kInvalidPoly, kInvalidPoly, kInvalidPoly};
    cardinal::scene::Vec3 centroid{0,0,0};
};

struct Mesh {
    cardinal::vector<cardinal::scene::Vec3> vertices;
    cardinal::vector<Poly>                  polys;

    void clear() noexcept { vertices.clear(); polys.clear(); }
    bool empty() const noexcept { return polys.empty(); }
    void recompute_centroids() noexcept;

    // Build a mesh from a triangle soup. Auto-detects edge sharing to
    // populate `neighbours`. O(N log N) — fine for the editor.
    void build_from_triangles(const cardinal::vector<cardinal::scene::Vec3>& verts,
                              const cardinal::vector<u32>& tri_indices);

    // World-space point → polygon containing it (or nearest, if not exact).
    // Returns kInvalidPoly when the mesh is empty.
    PolyId nearest_poly(const cardinal::scene::Vec3& p) const noexcept;
};

// ---------------------------------------------------------------------------
// PathQuery
// ---------------------------------------------------------------------------
struct PathStats {
    u32  poly_visited{0};
    u32  open_set_max{0};
    f32  poly_path_cost{0.0f};
    bool found{false};
    u32  waypoint_count{0};
};

class PathQuery {
public:
    // Polygon-A* + funnel-string-pulling = waypoints. `out_waypoints`
    // includes start and goal positions at the endpoints.
    PathStats find_path(const Mesh& mesh,
                        const cardinal::scene::Vec3& start,
                        const cardinal::scene::Vec3& goal,
                        cardinal::vector<cardinal::scene::Vec3>& out_waypoints);

private:
    // Scratch.
    cardinal::vector<f32>    g_;
    cardinal::vector<f32>    f_;
    cardinal::vector<PolyId> parent_;
    cardinal::vector<bool>   closed_;
    cardinal::vector<cardinal::pair<f32, PolyId>> open_;

    void open_push_(PolyId id, f32 fv);
    bool open_pop_(PolyId& out, f32& f);
};

}  // namespace cardinal::navmesh
