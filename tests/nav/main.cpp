// =============================================================================
// Cardinal — deterministic navigation / pathfinding regression suite.
//
// nav::Grid + PathQuery (A* octile) + compute_flow_field are the AI-
// movement backbone. A regression silently breaks navigation — agents
// can't path, take longer/wrong routes, or the flow field stops
// pointing at the goal. Pure, headless, fully deterministic given grid
// + start + goal + diagonal flag (the Studio nav panel drives the same
// code). Exact small-grid paths/costs are pinned, plus determinism +
// reusable-scratch correctness. NOTE: the grid pather deliberately
// allows diagonal corner-cutting (it only checks the diagonal's
// destination cell) — that behaviour is locked as-is, not assumed
// away. Exit 0 = all pass.
// =============================================================================

#include <cardinal/nav/nav.hpp>
#include <cardinal/core/log.hpp>

#include <vector>

namespace {

namespace nv = cardinal::nav;
using cardinal::i32;
using cardinal::u32;
using nv::CellCoord;

constexpr float kD2 = 1.41421356f;          // diagonal step

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("navtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool approxf(float a, float b, float e = 1e-4f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

bool has_cell(const std::vector<CellCoord>& v, i32 x, i32 y) {
    for (const CellCoord& c : v) if (c.x == x && c.y == y) return true;
    return false;
}
// Brace-free cell compare — a CellCoord{a,b} literal inside CHECK(...)
// would split the macro arg on the brace comma (the preprocessor only
// balances parentheses, not braces).
bool cell_eq(const CellCoord& c, i32 x, i32 y) { return c.x == x && c.y == y; }

// ---- Grid basics ----------------------------------------------------
void test_grid() {
    nv::Grid g(5, 4);
    CHECK(g.width() == 5u && g.height() == 4u);
    CHECK(g.in_bounds(0, 0) && g.in_bounds(4, 3));
    CHECK(!g.in_bounds(5, 0) && !g.in_bounds(0, 4));
    CHECK(!g.in_bounds(-1, 0) && !g.in_bounds(0, -1));
    CHECK(approxf(g.cost(0, 0), 1.0f));
    CHECK(g.cost(5, 0) == nv::kBlocked);          // OOB reads as blocked
    CHECK(g.open_cell_count() == 20u);

    CHECK(!g.is_blocked(2, 2));
    g.set_blocked(2, 2);
    CHECK(g.is_blocked(2, 2));
    CHECK(g.cost(2, 2) == nv::kBlocked);
    CHECK(g.open_cell_count() == 19u);

    g.set_cost(1, 1, 3.0f);
    CHECK(approxf(g.cost(1, 1), 3.0f) && !g.is_blocked(1, 1));
    CHECK(g.open_cell_count() == 19u);            // weighted still "open"

    g.fill(2.0f);
    CHECK(approxf(g.cost(0, 0), 2.0f) && g.open_cell_count() == 20u);
    g.fill(nv::kBlocked);
    CHECK(g.open_cell_count() == 0u && g.is_blocked(0, 0));
}

// ---- A*: exact paths + costs ---------------------------------------
void test_astar() {
    nv::PathQuery q;
    std::vector<CellCoord> out;

    // Straight line, orthogonal only.
    {
        nv::Grid g(5, 1);
        const nv::PathStats st = q.find_path(g, {0,0}, {4,0}, out, false);
        CHECK(st.found);
        CHECK(approxf(st.path_cost, 4.0f));
        CHECK(out.size() == sz(5));
        CHECK(cell_eq(out.front(),0,0) && cell_eq(out.back(),4,0));
    }
    // Diagonal optimum vs Manhattan when diagonals disabled.
    {
        nv::Grid g(5, 5);
        const nv::PathStats d = q.find_path(g, {0,0}, {4,4}, out, true);
        CHECK(d.found);
        CHECK(approxf(d.path_cost, 4.0f * kD2));      // 4 diagonal steps
        CHECK(out.size() == sz(5));
        CHECK(cell_eq(out.front(),0,0) && cell_eq(out.back(),4,4));

        const nv::PathStats m = q.find_path(g, {0,0}, {4,4}, out, false);
        CHECK(m.found);
        CHECK(approxf(m.path_cost, 8.0f));            // 4 + 4 orthogonal
        CHECK(out.size() == sz(9));
    }
    // Blocked centre forces a detour; path never enters the blocked cell.
    {
        nv::Grid g(3, 3);
        g.set_blocked(1, 1);
        const nv::PathStats o = q.find_path(g, {0,1}, {2,1}, out, false);
        CHECK(o.found && approxf(o.path_cost, 4.0f) && out.size() == sz(5));
        CHECK(!has_cell(out, 1, 1));
        // Diagonal: corner-cut around the single blocker (documented).
        const nv::PathStats dd = q.find_path(g, {0,1}, {2,1}, out, true);
        CHECK(dd.found && approxf(dd.path_cost, 2.0f * kD2));
        CHECK(out.size() == sz(3) && !has_cell(out, 1, 1));
    }
    // A full wall → no path; stats sane, out empty.
    {
        nv::Grid g(3, 3);
        g.set_blocked(1, 0); g.set_blocked(1, 1); g.set_blocked(1, 2);
        const nv::PathStats st = q.find_path(g, {0,1}, {2,1}, out, true);
        CHECK(!st.found);
        CHECK(out.empty());
        CHECK(st.path_cost == 0.0f);
        CHECK(st.nodes_visited > 0u);              // it did search
    }
    // start == goal → trivial 1-cell path, zero cost.
    {
        nv::Grid g(4, 4);
        const nv::PathStats st = q.find_path(g, {2,2}, {2,2}, out, true);
        CHECK(st.found);
        CHECK(out.size() == sz(1) && cell_eq(out.front(),2,2));
        CHECK(st.path_cost == 0.0f);
    }
    // Invalid endpoints → not found, no output.
    {
        nv::Grid g(3, 3);
        CHECK(!q.find_path(g, {-1,0}, {1,1}, out, true).found);
        CHECK(out.empty());
        CHECK(!q.find_path(g, {0,0}, {5,5}, out, true).found);
        g.set_blocked(0, 0);
        CHECK(!q.find_path(g, {0,0}, {2,2}, out, true).found);
        nv::Grid g2(3, 3);
        g2.set_blocked(2, 2);
        CHECK(!q.find_path(g2, {0,0}, {2,2}, out, true).found);
    }
    // Cost model: step cost = base * destination-cell cost.
    {
        nv::Grid g(3, 1);
        g.set_cost(1, 0, 5.0f);                    // only route passes here
        const nv::PathStats st = q.find_path(g, {0,0}, {2,0}, out, false);
        CHECK(st.found);
        CHECK(approxf(st.path_cost, 6.0f));           // 1*5 (into 1,0) + 1*1
        CHECK(out.size() == sz(3));
    }
}

// ---- determinism + reusable-scratch correctness --------------------
void test_determinism_reuse() {
    nv::Grid g(6, 6);
    g.set_blocked(2, 2); g.set_blocked(2, 3); g.set_blocked(3, 2);

    nv::PathQuery q;                              // reused across calls
    std::vector<CellCoord> a, b, other;
    const nv::PathStats sa = q.find_path(g, {0,0}, {5,5}, a, true);
    // A different query on the SAME scratch object…
    q.find_path(g, {5,0}, {0,5}, other, true);
    // …then the original again must reproduce bit-identically.
    const nv::PathStats sb = q.find_path(g, {0,0}, {5,5}, b, true);

    CHECK(sa.found && sb.found);
    CHECK(a == b);                                // same path sequence
    CHECK(approxf(sa.path_cost, sb.path_cost, 0.0f));// exactly equal
    CHECK(sa.nodes_visited == sb.nodes_visited);
    CHECK(sa.open_set_max  == sb.open_set_max);
    CHECK(!a.empty() && cell_eq(a.front(),0,0) && cell_eq(a.back(),5,5));
}

// ---- flow field -----------------------------------------------------
void test_flow_field() {
    nv::Grid g(4, 4);
    const CellCoord goal{0, 0};
    std::vector<nv::FlowDir> flow;
    std::vector<float>       dist;
    compute_flow_field(g, goal, flow, dist);

    const u32 W = g.width();
    auto at = [W](i32 x, i32 y) {
        return static_cast<cardinal::usize>(y) *
               static_cast<cardinal::usize>(W) +
               static_cast<cardinal::usize>(x);
    };
    CHECK(flow.size() == sz(16) && dist.size() == sz(16));
    CHECK(dist[at(0,0)] == 0.0f);                 // goal distance 0
    CHECK(dist[at(3,3)] < nv::kBlocked);          // reachable (finite)
    CHECK(dist[at(3,3)] > 0.0f);
    CHECK(flow[at(0,0)].dx == 0 && flow[at(0,0)].dy == 0);  // goal: no dir

    // Following the flow from any open cell strictly descends distance
    // and reaches the goal in a bounded number of steps.
    i32 x = 3, y = 3;
    int steps = 0;
    bool reached = false;
    for (; steps <= 64; ++steps) {
        if (x == 0 && y == 0) { reached = true; break; }
        const nv::FlowDir fd = flow[at(x, y)];
        if (fd.dx == 0 && fd.dy == 0) break;       // stuck (shouldn't happen)
        x += fd.dx; y += fd.dy;
    }
    CHECK(reached);

    // A blocked cell: unreachable distance + zero flow.
    nv::Grid gb(4, 4);
    gb.set_blocked(1, 1);
    compute_flow_field(gb, goal, flow, dist);
    CHECK(dist[at(1,1)] == nv::kBlocked);
    CHECK(flow[at(1,1)].dx == 0 && flow[at(1,1)].dy == 0);
    CHECK(dist[at(3,3)] < nv::kBlocked);          // other cells still route

    // Invalid goal → early return: all cleared, no crash.
    nv::Grid gc(3, 3);
    compute_flow_field(gc, CellCoord{-1, 0}, flow, dist);
    CHECK(flow.size() == sz(9));
    bool all_clear = true;
    for (const auto& f : flow) if (f.dx != 0 || f.dy != 0) all_clear = false;
    for (float d : dist) if (d != nv::kBlocked) all_clear = false;
    CHECK(all_clear);
}

}  // namespace

int main() {
    test_grid();
    test_astar();
    test_determinism_reuse();
    test_flow_field();

    if (g_fail == 0) {
        cardinal::log::infof("navtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("navtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
