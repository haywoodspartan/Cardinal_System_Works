#pragma once

// =============================================================================
// Studio — Editor mode toolbar.
//
// Horizontal strip of mode buttons (Select / Place / Sculpt / Paint /
// Foliage / Mesh / Landscape / Measure). Highlights the active mode;
// clicking switches the EditorState's mode (which fires its on_change
// callback). Hotkey hints in the tooltip.
// =============================================================================

namespace cardinal::edit { class EditorState; }

namespace cardinal::ui::panels::editor_modes_panel {

void draw(cardinal::edit::EditorState* state,
          const char* title = "Modes", bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::editor_modes_panel
