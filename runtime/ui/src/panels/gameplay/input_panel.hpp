#pragma once

// =============================================================================
// Studio — Input panel.
//
// Shows live key + mouse + action state from a cardinal::input::Manager.
// Right pane lists all bindings; user can clear bindings or rebind by
// "press a key" capture mode.
// =============================================================================

namespace cardinal::input { class Manager; }

namespace cardinal::ui::panels::input_panel {

void draw(cardinal::input::Manager* mgr,
          const char* title = "Input",
          bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::input_panel
