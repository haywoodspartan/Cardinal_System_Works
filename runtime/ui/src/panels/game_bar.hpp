#pragma once

// =============================================================================
// Studio — Game lifecycle bar.
//
// Drives a cardinal::game::Game: Play / Pause / Resume / Stop with state
// readout. Keyboard hotkeys (when no input field has focus):
//
//   F5 — Play (or Resume from Paused)
//   F6 — Pause / Resume toggle (only while game_active)
//   F8 — Stop
//
// Visually distinct from the Sim bar (which controls sub-stepping +
// time-scale across both editor + play modes).
// =============================================================================

namespace cardinal::game { class Game; }

namespace cardinal::ui::panels::game_bar_panel {

void draw(cardinal::game::Game* game,
          const char* title = "Game",
          bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::game_bar_panel
