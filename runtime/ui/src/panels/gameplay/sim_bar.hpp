#pragma once

// =============================================================================
// Studio — Simulation control bar.
//
// Tiny toolbar that drives a cardinal::sim::SimWorld:
//   [Play] [Pause] [Step] | time scale slider 0..4x | sim time / real time
//
// Hotkeys (no text input focused):
//   Space — play/pause
//   .     — step one frame
//   1..5  — jump time scale to 0.25 / 0.5 / 1.0 / 2.0 / 4.0
// =============================================================================

namespace cardinal::sim { class SimWorld; }

namespace cardinal::ui::panels::sim_bar_panel {

void draw(cardinal::sim::SimWorld* sim,
          const char* title = "Simulation",
          bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::sim_bar_panel
