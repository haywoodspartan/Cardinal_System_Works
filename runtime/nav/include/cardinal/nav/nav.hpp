#pragma once

// =============================================================================
// Cardinal — Navigation / Pathfinding.
//
// Grid-based for now (the most useful baseline + simplest to debug). Each
// cell has a `cost` float (1 = default, INF = blocked). A* with octile
// heuristic finds shortest paths between two cells. Optional flow-field
// generator computes a "every cell points toward a goal" map for
// many-agent scenarios (cheaper than per-agent A*).
//
// Coordinates are integer cell indices. The Grid doesn't know about world
// units — the integrator wraps it (e.g. 1 cell = 0.5 world units, then
// world_xz → cell_xz via floor()). Keeps the pathfinder small.
//
// All allocators reuse storage between calls so steady-state queries are
// allocation-free.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <limits>
#include <memory>
#include <vector>

namespace cardinal::nav {

inline constexpr float kBlocked = std::numeric_limits<float>::infinity();

// ---------------------------------------------------------------------------
// Grid — 2D cost field. Indexed by (x, y), origin top-left, x grows right,
// y grows down (matching most map editors).
// ---------------------------------------------------------------------------
class Grid {
public:
    Grid(u32 width, u32 height, float default_cost = 1.0f);

    u32 width()  const noexcept { return w_; }
    u32 height() const noexcept { return h_; }

    bool   in_bounds(i32 x, i32 y) const noexcept;
    float  cost(i32 x, i32 y) const noexcept;
    void   set_cost (i32 x, i32 y, float c) noexcept;
    void   set_blocked(i32 x, i32 y) noexcept;
    void   fill(float c) noexcept;
    bool   is_blocked(i32 x, i32 y) const noexcept;

    // Total open cells (cost < kBlocked).
    u32    open_cell_count() const noexcept;

    // Direct buffer access — for the editor heat-map renderer.
    const std::vector<float>& cells() const noexcept { return cells_; }

private:
    u32 w_{0};
    u32 h_{0};
    std::vector<float> cells_;
};

// ---------------------------------------------------------------------------
// A* pathfinder.
//
// PathQuery is a reusable scratch object — construct once per agent (or
// per editor session) and call find_path() repeatedly. Internal vectors
// resize with the grid; subsequent same-grid calls reuse storage.
// ---------------------------------------------------------------------------
struct CellCoord { i32 x{0}, y{0}; bool operator==(const CellCoord& o) const noexcept { return x==o.x && y==o.y; } };

struct PathStats {
    u32  nodes_visited{0};
    u32  open_set_max {0};
    f32  path_cost    {0.0f};
    bool found        {false};
};

class PathQuery {
public:
    explicit PathQuery();

    // Find a path from `start` to `goal`. On success, `out_cells` is filled
    // with the sequence of cells from start to goal inclusive. `stats` is
    // populated either way (even when no path is found).
    PathStats find_path(const Grid& grid,
                        CellCoord start, CellCoord goal,
                        std::vector<CellCoord>& out_cells,
                        bool allow_diagonal = true);

private:
    struct Node {
        i32   parent_x{-1}, parent_y{-1};
        float g{kBlocked};            // cost from start
        float f{kBlocked};            // g + heuristic
        bool  closed{false};
        u32   open_index{static_cast<u32>(-1)};
    };

    std::vector<Node>                          nodes_;
    u32                                        grid_w_{0};
    u32                                        grid_h_{0};
    // Open set as a min-heap of (f, packed_xy).
    std::vector<std::pair<float, u64>>         open_;

    void ensure_(u32 w, u32 h);
    void open_push_(u64 key, float f);
    bool open_pop_(u64& out_key, float& out_f);
};

// ---------------------------------------------------------------------------
// Flow field — for many-agent pathing toward a single goal cell, generate
// a "best direction" per cell once, then every agent samples it. Much
// cheaper than per-agent A* when N agents share a goal.
//
// Output is a vector of CellCoord deltas (-1, 0, +1) per cell, indexed
// flat: flow[y * grid.width() + x]. Cells unreachable from the goal get
// (0, 0).
// ---------------------------------------------------------------------------
struct FlowDir { i8 dx{0}, dy{0}; };

void compute_flow_field(const Grid& grid, CellCoord goal,
                        std::vector<FlowDir>& out_flow,
                        std::vector<float>& out_distance);

}  // namespace cardinal::nav
