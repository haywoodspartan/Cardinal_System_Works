#pragma once

// =============================================================================
// Studio — Navigation panel.
//
// Visualises a cardinal::nav::Grid as a 2D heat map, with click-to-toggle
// blocked cells, click-and-drag start/goal placement, and a live A* path
// trace overlay. Useful for verifying the pathfinder behaves correctly
// before wiring it into game code.
// =============================================================================

#include <cardinal/core/types.hpp>

namespace cardinal::nav { class Grid; }

namespace cardinal::ui::panels::nav_panel {

struct State {
    cardinal::i32 start_x{1}, start_y{1};
    cardinal::i32 goal_x{20}, goal_y{20};
    bool          allow_diagonal{true};
    bool          paint_blocked{true};   // LMB mode: paint or pick cell
    cardinal::f32 cell_pixels{16.0f};
};

void draw(cardinal::nav::Grid* grid, State& state,
          const char* title = "Navigation",
          bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::nav_panel
