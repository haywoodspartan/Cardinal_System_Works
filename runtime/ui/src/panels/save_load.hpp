#pragma once

// =============================================================================
// Studio — Save / Load panel.
//
// Lets the user save the current world to a chosen path or load one back.
// Drives cardinal::serial::save_world / load_world. Works against the
// active cardinal::game::Game so the load path knows how to spawn the
// right classes.
// =============================================================================

namespace cardinal::game { class Game; }
namespace cardinal::sky  { class Sky;  }

namespace cardinal::ui::panels::save_load_panel {

void draw(cardinal::game::Game* game,
          cardinal::sky::Sky*  sky,
          const char* title = "Save / Load",
          bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::save_load_panel
