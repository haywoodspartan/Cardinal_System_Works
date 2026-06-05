// =============================================================================
// Studio — Navigation panel implementation.
// =============================================================================
#include "nav_panel.hpp"

#include <cardinal/nav/nav.hpp>

#include <cardinal/ui/imgui.hpp>

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/cmath.hpp>
#include <cardinal/core/cstdio.hpp>

namespace cardinal::ui::panels::nav_panel {

void draw(cardinal::nav::Grid* grid, State& s,
          const char* title, bool* p_open)
{
    if (!ImGui::Begin(title ? title : "Navigation", p_open,
                      0)) { ImGui::End(); return; }
    if (grid == nullptr) {
        ImGui::TextDisabled("(no nav::Grid bound)");
        ImGui::End();
        return;
    }

    ImGui::Text("Grid: %u x %u  (%u open)",
        grid->width(), grid->height(), grid->open_cell_count());
    ImGui::Checkbox("Allow diagonal", &s.allow_diagonal);
    ImGui::SameLine();
    ImGui::Checkbox("Paint blocked", &s.paint_blocked);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    ImGui::SliderFloat("Cell px", &s.cell_pixels, 4.0f, 48.0f, "%.0f");
    if (ImGui::Button("Clear all")) grid->fill(1.0f);

    // Run A* live every frame so painting / start / goal updates redraw
    // the path as you move them. Cheap on the Studio's grid sizes.
    static cardinal::nav::PathQuery q;
    static cardinal::vector<cardinal::nav::CellCoord> path;
    const auto stats = q.find_path(*grid,
        {s.start_x, s.start_y}, {s.goal_x, s.goal_y},
        path, s.allow_diagonal);

    ImGui::TextDisabled(
        "Path: %s | nodes=%u open-max=%u cost=%.2f",
        stats.found ? "FOUND" : "no path",
        stats.nodes_visited, stats.open_set_max, stats.path_cost);

    ImGui::Separator();

    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float cs = s.cell_pixels;
    const float W = static_cast<float>(grid->width());
    const float H = static_cast<float>(grid->height());

    ImDrawList* dl = ImGui::GetWindowDrawList();
    // Background.
    dl->AddRectFilled(origin,
        {origin.x + W * cs, origin.y + H * cs}, IM_COL32(15, 18, 22, 255));

    // Cells.
    for (cardinal::u32 y = 0; y < grid->height(); ++y) {
        for (cardinal::u32 x = 0; x < grid->width(); ++x) {
            const float c = grid->cost(static_cast<cardinal::i32>(x),
                                        static_cast<cardinal::i32>(y));
            ImU32 col = IM_COL32_BLACK_TRANS;
            if (!cardinal::isfinite(c)) {
                col = IM_COL32(60, 30, 30, 240);
            } else if (c > 1.5f) {
                const int r = cardinal::clamp<int>(80 + static_cast<int>(c * 20.0f), 80, 200);
                col = IM_COL32(r, r / 2, 30, 220);
            }
            if (col != IM_COL32_BLACK_TRANS) {
                const ImVec2 p0(origin.x + x * cs, origin.y + y * cs);
                const ImVec2 p1(p0.x + cs, p0.y + cs);
                dl->AddRectFilled(p0, p1, col);
            }
        }
    }

    // Path.
    for (usize i = 1; i < path.size(); ++i) {
        const auto& a = path[i - 1];
        const auto& b = path[i];
        const ImVec2 pa(origin.x + (a.x + 0.5f) * cs, origin.y + (a.y + 0.5f) * cs);
        const ImVec2 pb(origin.x + (b.x + 0.5f) * cs, origin.y + (b.y + 0.5f) * cs);
        dl->AddLine(pa, pb, IM_COL32(120, 220, 255, 230), 2.0f);
    }

    // Start / Goal markers.
    auto cell_circle = [&](cardinal::i32 x, cardinal::i32 y, ImU32 col) {
        const ImVec2 c(origin.x + (x + 0.5f) * cs,
                       origin.y + (y + 0.5f) * cs);
        dl->AddCircleFilled(c, cs * 0.30f, col);
    };
    cell_circle(s.start_x, s.start_y, IM_COL32( 90, 220,  90, 255));
    cell_circle(s.goal_x,  s.goal_y,  IM_COL32(230, 110, 110, 255));

    // Border.
    dl->AddRect(origin,
        {origin.x + W * cs, origin.y + H * cs},
        IM_COL32(80, 80, 80, 255));

    // Reserve the area so subsequent items lay out below.
    ImGui::Dummy(ImVec2(W * cs, H * cs));

    // Mouse interaction.
    if (ImGui::IsItemHovered()) {
        const ImVec2 mp = ImGui::GetIO().MousePos;
        const cardinal::i32 cx = static_cast<cardinal::i32>((mp.x - origin.x) / cs);
        const cardinal::i32 cy = static_cast<cardinal::i32>((mp.y - origin.y) / cs);
        ImGui::SetTooltip("(%d, %d)", cx, cy);
        const ImGuiIO& io = ImGui::GetIO();
        if (grid->in_bounds(cx, cy)) {
            if (io.KeyShift && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                s.start_x = cx; s.start_y = cy;
            } else if (io.KeyCtrl && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                s.goal_x = cx; s.goal_y = cy;
            } else if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                if (s.paint_blocked) grid->set_blocked(cx, cy);
                else                  grid->set_cost(cx, cy, 1.0f);
            }
        }
    }
    ImGui::TextDisabled("LMB paint, Shift+LMB = start, Ctrl+LMB = goal");

    ImGui::End();
}

}  // namespace cardinal::ui::panels::nav_panel
