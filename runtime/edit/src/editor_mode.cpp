#include <cardinal/edit/editor_mode.hpp>

namespace cardinal::edit {

const char* editor_mode_name(EditorMode m) noexcept {
    switch (m) {
        case EditorMode::Select:    return "Select";
        case EditorMode::Place:     return "Place";
        case EditorMode::Sculpt:    return "Sculpt";
        case EditorMode::Paint:     return "Paint";
        case EditorMode::Foliage:   return "Foliage";
        case EditorMode::Mesh:      return "Mesh";
        case EditorMode::Landscape: return "Landscape";
        case EditorMode::Measure:   return "Measure";
    }
    return "?";
}

const char* editor_mode_glyph(EditorMode m) noexcept {
    switch (m) {
        case EditorMode::Select:    return "[+]";
        case EditorMode::Place:     return "[P]";
        case EditorMode::Sculpt:    return "[S]";
        case EditorMode::Paint:     return "[B]";
        case EditorMode::Foliage:   return "[F]";
        case EditorMode::Mesh:      return "[M]";
        case EditorMode::Landscape: return "[L]";
        case EditorMode::Measure:   return "[=]";
    }
    return "[?]";
}

const char* editor_mode_tooltip(EditorMode m) noexcept {
    switch (m) {
        case EditorMode::Select:
            return "Select Mode (Q)\nClick entities to pick. Drag the gizmo to move.";
        case EditorMode::Place:
            return "Place Mode (W)\nClick the viewport to spawn the active asset.";
        case EditorMode::Sculpt:
            return "Sculpt Mode (E)\nLMB raises terrain; Shift lowers; Ctrl smooths.";
        case EditorMode::Paint:
            return "Paint Mode (R)\nVertex / texture paint with the active brush color.";
        case EditorMode::Foliage:
            return "Foliage Mode (T)\nScatter the active asset in a brush region.";
        case EditorMode::Mesh:
            return "Mesh Mode (Y)\nExtrude / subdivide / mirror the selected mesh.";
        case EditorMode::Landscape:
            return "Landscape Mode (U)\nGrid-edit terrain heights via per-cell handles.";
        case EditorMode::Measure:
            return "Measure Mode (I)\nClick two points to read distance / angle.";
    }
    return "";
}

void EditorState::set_mode(EditorMode m) {
    if (m == mode_) return;
    const EditorMode prev = mode_;
    mode_ = m;
    if (on_change_) on_change_(prev, m);
}

}  // namespace cardinal::edit
