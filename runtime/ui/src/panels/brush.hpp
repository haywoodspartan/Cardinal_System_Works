#pragma once

// =============================================================================
// Studio — Brush panel.
//
// Edits a cardinal::edit::brush::Brush in place. Sliders for radius /
// strength / spacing, dropdowns for falloff / mode. The viewport
// renderer reads the brush off the same struct; the panel is just UI.
// =============================================================================

namespace cardinal::edit::brush { struct Brush; }

namespace cardinal::ui::panels::brush_panel {

void draw(cardinal::edit::brush::Brush* brush,
          const char* title = "Brush", bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::brush_panel
