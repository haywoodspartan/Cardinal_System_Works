#include <cardinal/nav/nav.hpp>

#include <cardinal/core/algorithm.hpp>    // cardinal::fill/min/max/reverse/*_heap
#include <cardinal/core/cmath.hpp>        // cardinal::isfinite/abs
#include <cardinal/core/containers.hpp>   // cardinal::priority_queue
// cardinal::pair / cardinal::greater arrive via nav.hpp / core/types.hpp

namespace cardinal::nav {

namespace {

inline u64 pack_xy(i32 x, i32 y) noexcept {
    return (static_cast<u64>(static_cast<u32>(x)) << 32) |
            static_cast<u64>(static_cast<u32>(y));
}
inline void unpack_xy(u64 k, i32& x, i32& y) noexcept {
    x = static_cast<i32>(k >> 32);
    y = static_cast<i32>(k & 0xFFFFFFFFu);
}

}  // namespace

// ---------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------
Grid::Grid(u32 width, u32 height, float default_cost)
    : w_(width), h_(height), cells_(static_cast<usize>(width) * height, default_cost) {}

bool Grid::in_bounds(i32 x, i32 y) const noexcept {
    return x >= 0 && y >= 0 && static_cast<u32>(x) < w_ && static_cast<u32>(y) < h_;
}

float Grid::cost(i32 x, i32 y) const noexcept {
    if (!in_bounds(x, y)) return kBlocked;
    return cells_[static_cast<usize>(y) * w_ + static_cast<usize>(x)];
}

void Grid::set_cost(i32 x, i32 y, float c) noexcept {
    if (!in_bounds(x, y)) return;
    cells_[static_cast<usize>(y) * w_ + static_cast<usize>(x)] = c;
}

void Grid::set_blocked(i32 x, i32 y) noexcept { set_cost(x, y, kBlocked); }

void Grid::fill(float c) noexcept { cardinal::fill(cells_.begin(), cells_.end(), c); }

bool Grid::is_blocked(i32 x, i32 y) const noexcept {
    return !cardinal::isfinite(cost(x, y));
}

u32 Grid::open_cell_count() const noexcept {
    u32 n = 0;
    for (float c : cells_) if (cardinal::isfinite(c)) ++n;
    return n;
}

// ---------------------------------------------------------------------------
// PathQuery
// ---------------------------------------------------------------------------
PathQuery::PathQuery() = default;

void PathQuery::ensure_(u32 w, u32 h) {
    if (w == grid_w_ && h == grid_h_ && nodes_.size() == static_cast<usize>(w) * h) {
        // Re-init existing scratch.
        for (auto& n : nodes_) {
            n.parent_x = -1; n.parent_y = -1;
            n.g = kBlocked; n.f = kBlocked;
            n.closed = false;
            n.open_index = static_cast<u32>(-1);
        }
    } else {
        grid_w_ = w; grid_h_ = h;
        nodes_.assign(static_cast<usize>(w) * h, Node{});
    }
    open_.clear();
}

void PathQuery::open_push_(u64 key, float f) {
    open_.emplace_back(f, key);
    cardinal::push_heap(open_.begin(), open_.end(),
        [](const auto& a, const auto& b){ return a.first > b.first; });
}
bool PathQuery::open_pop_(u64& out_key, float& out_f) {
    if (open_.empty()) return false;
    cardinal::pop_heap(open_.begin(), open_.end(),
        [](const auto& a, const auto& b){ return a.first > b.first; });
    out_f   = open_.back().first;
    out_key = open_.back().second;
    open_.pop_back();
    return true;
}

PathStats PathQuery::find_path(const Grid& grid,
                               CellCoord start, CellCoord goal,
                               cardinal::vector<CellCoord>& out_cells,
                               bool allow_diagonal)
{
    PathStats st{};
    out_cells.clear();
    if (!grid.in_bounds(start.x, start.y) || !grid.in_bounds(goal.x, goal.y)) return st;
    if (grid.is_blocked(start.x, start.y) || grid.is_blocked(goal.x, goal.y)) return st;

    const u32 W = grid.width();
    const u32 H = grid.height();
    ensure_(W, H);

    auto idx = [W](i32 x, i32 y) { return static_cast<usize>(y) * W + static_cast<usize>(x); };
    auto h_oct = [&](i32 x, i32 y) {
        const i32 dx = cardinal::abs(x - goal.x);
        const i32 dy = cardinal::abs(y - goal.y);
        const float D = 1.0f, D2 = 1.41421356f;
        return D * (dx + dy) + (D2 - 2.0f * D) * cardinal::min(dx, dy);
    };

    Node& s = nodes_[idx(start.x, start.y)];
    s.g = 0.0f;
    s.f = h_oct(start.x, start.y);
    open_push_(pack_xy(start.x, start.y), s.f);

    static constexpr i32 d8x[8] = { 1,-1, 0, 0, 1, 1,-1,-1};
    static constexpr i32 d8y[8] = { 0, 0, 1,-1, 1,-1, 1,-1};
    const i32 dirs = allow_diagonal ? 8 : 4;

    while (true) {
        u64 key; float fcur;
        if (!open_pop_(key, fcur)) break;
        i32 cx, cy; unpack_xy(key, cx, cy);
        Node& cn = nodes_[idx(cx, cy)];
        if (cn.closed) continue;
        cn.closed = true;
        ++st.nodes_visited;
        st.open_set_max = cardinal::max(st.open_set_max, static_cast<u32>(open_.size()));

        if (cx == goal.x && cy == goal.y) {
            // Reconstruct
            i32 rx = cx, ry = cy;
            while (rx >= 0) {
                out_cells.push_back({rx, ry});
                Node& rn = nodes_[idx(rx, ry)];
                rx = rn.parent_x; ry = rn.parent_y;
            }
            cardinal::reverse(out_cells.begin(), out_cells.end());
            st.found     = true;
            st.path_cost = cn.g;
            return st;
        }

        for (i32 i = 0; i < dirs; ++i) {
            const i32 nx = cx + d8x[i];
            const i32 ny = cy + d8y[i];
            if (!grid.in_bounds(nx, ny)) continue;
            const float ncost = grid.cost(nx, ny);
            if (!cardinal::isfinite(ncost)) continue;
            const float step = (i < 4 ? 1.0f : 1.41421356f) * ncost;
            const float tentative = cn.g + step;
            Node& nn = nodes_[idx(nx, ny)];
            if (tentative < nn.g) {
                nn.g = tentative;
                nn.parent_x = cx; nn.parent_y = cy;
                nn.f = tentative + h_oct(nx, ny);
                open_push_(pack_xy(nx, ny), nn.f);
            }
        }
    }
    return st;
}

// ---------------------------------------------------------------------------
// Flow field — Dijkstra from goal, then per-cell pick the lowest-distance
// neighbour as the flow direction.
// ---------------------------------------------------------------------------
void compute_flow_field(const Grid& grid, CellCoord goal,
                        cardinal::vector<FlowDir>& out_flow,
                        cardinal::vector<float>& out_distance)
{
    const u32 W = grid.width();
    const u32 H = grid.height();
    out_flow.assign(static_cast<usize>(W) * H, FlowDir{0, 0});
    out_distance.assign(static_cast<usize>(W) * H, kBlocked);
    if (!grid.in_bounds(goal.x, goal.y) || grid.is_blocked(goal.x, goal.y)) return;

    auto idx = [W](i32 x, i32 y) { return static_cast<usize>(y) * W + static_cast<usize>(x); };

    using Q = cardinal::pair<float, u64>;
    cardinal::priority_queue<Q, cardinal::vector<Q>, cardinal::greater<>> pq;
    out_distance[idx(goal.x, goal.y)] = 0.0f;
    pq.push({0.0f, pack_xy(goal.x, goal.y)});

    static constexpr i32 d8x[8] = { 1,-1, 0, 0, 1, 1,-1,-1};
    static constexpr i32 d8y[8] = { 0, 0, 1,-1, 1,-1, 1,-1};
    static constexpr float d8cost[8] = {1,1,1,1, 1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f};

    while (!pq.empty()) {
        const auto [d, key] = pq.top(); pq.pop();
        i32 cx, cy; unpack_xy(key, cx, cy);
        if (d > out_distance[idx(cx, cy)]) continue;
        for (i32 i = 0; i < 8; ++i) {
            const i32 nx = cx + d8x[i];
            const i32 ny = cy + d8y[i];
            if (!grid.in_bounds(nx, ny)) continue;
            const float c = grid.cost(nx, ny);
            if (!cardinal::isfinite(c)) continue;
            const float nd = d + d8cost[i] * c;
            if (nd < out_distance[idx(nx, ny)]) {
                out_distance[idx(nx, ny)] = nd;
                pq.push({nd, pack_xy(nx, ny)});
            }
        }
    }

    // Per-cell best neighbour direction.
    for (u32 y = 0; y < H; ++y) {
        for (u32 x = 0; x < W; ++x) {
            const float d_here = out_distance[idx(x, y)];
            if (!cardinal::isfinite(d_here)) continue;
            float best = d_here;
            i8 best_dx = 0, best_dy = 0;
            for (i32 i = 0; i < 8; ++i) {
                const i32 nx = static_cast<i32>(x) + d8x[i];
                const i32 ny = static_cast<i32>(y) + d8y[i];
                if (!grid.in_bounds(nx, ny)) continue;
                const float dn = out_distance[idx(nx, ny)];
                if (dn < best) {
                    best = dn;
                    best_dx = static_cast<i8>(d8x[i]);
                    best_dy = static_cast<i8>(d8y[i]);
                }
            }
            out_flow[idx(x, y)] = FlowDir{best_dx, best_dy};
        }
    }
}

}  // namespace cardinal::nav
