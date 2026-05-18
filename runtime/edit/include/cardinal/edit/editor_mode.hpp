#pragma once

// =============================================================================
// Cardinal — Editor mode state machine (UE5-style).
//
// The Studio's behaviour swings depending on what the user is currently
// trying to do:
//
//   Select   — default. Click to pick, gizmo to move, marquee, etc.
//   Place    — drag from the asset palette, click in viewport to instantiate.
//   Sculpt   — terrain brush. LMB raises, Shift+LMB lowers, Ctrl+LMB smooths.
//   Paint    — vertex paint with the brush color.
//   Foliage  — scatter brush, fills an area with an asset.
//   Mesh     — show the mesh-editor toolbox (extrude, subdivide, …).
//   Landscape— grid-edit terrain heights via per-cell handles.
//   Measure  — pick two points, show distance / angle.
//
// The mode is a single enum kept in EditorState; panels and the viewport
// query it to know whether to consume an input event vs. forward it. Mode
// transitions fire a callback so subsystems (gizmo, brush HUD, etc.) can
// re-arm themselves.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <functional>
#include <string>

namespace cardinal::edit {

enum class EditorMode : u32 {
    Select    = 0,
    Place     = 1,
    Sculpt    = 2,
    Paint     = 3,
    Foliage   = 4,
    Mesh      = 5,
    Landscape = 6,
    Measure   = 7,
};

const char* editor_mode_name(EditorMode m) noexcept;
const char* editor_mode_glyph(EditorMode m) noexcept;
const char* editor_mode_tooltip(EditorMode m) noexcept;

class EditorState {
public:
    EditorState() = default;

    EditorMode mode() const noexcept { return mode_; }
    void       set_mode(EditorMode m);

    // Subscribe to mode-change events. Last setter wins (single slot).
    using OnModeChange = std::function<void(EditorMode old, EditorMode now)>;
    void on_mode_change(OnModeChange cb) { on_change_ = std::move(cb); }

    // Per-mode flags (all panels share one struct so adding a flag doesn't
    // touch every call site).
    bool brush_active() const noexcept {
        return mode_ == EditorMode::Sculpt ||
               mode_ == EditorMode::Paint  ||
               mode_ == EditorMode::Foliage;
    }
    bool gizmo_active() const noexcept {
        return mode_ == EditorMode::Select || mode_ == EditorMode::Place;
    }

    // Optional human-readable status — shown in the bottom status bar.
    const std::string& status_text() const noexcept { return status_; }
    void set_status_text(std::string s) { status_ = std::move(s); }

private:
    EditorMode    mode_{EditorMode::Select};
    OnModeChange  on_change_{};
    std::string   status_{};
};

}  // namespace cardinal::edit
