#pragma once

// =============================================================================
// Studio — Animation curve editor.
//
// Edits a cardinal::anim::Curve<float> in place. Drag keys to move them in
// (time, value); double-click empty area to add a key; right-click a key
// to remove. Live overlay of the curve (linear / cubic) sampled across the
// visible range.
//
// Vec3/Vec4 curves should be edited as three / four float curves; the
// curve editor only deals with floats — the panel host plumbs the per-axis
// extraction.
// =============================================================================

namespace cardinal::anim {
template <class T> struct Curve;
}

namespace cardinal::ui::panels::curve_editor_panel {

void draw(cardinal::anim::Curve<float>* curve,
          const char* title = "Curve Editor",
          bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::curve_editor_panel
