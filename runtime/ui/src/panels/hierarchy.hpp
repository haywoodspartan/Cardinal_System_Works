#pragma once

// =============================================================================
// Studio — scene hierarchy panel (free-function helper).
//
// Lifted out of studio.cpp so the hierarchy logic edits in isolation. The
// StudioImpl class still owns one HierarchyPanelState (filter buffer +
// expand pulse) and forwards its `draw_scene_hierarchy` virtual call here.
// Self-contained: drag-to-reparent uses scene::Scene::set_parent which has
// its own cycle guard, so this helper has no dependency on the rest of
// Studio's panel state.
// =============================================================================

#include <cardinal/core/types.hpp>

namespace cardinal::scene { class Scene; }

namespace cardinal::ui::panels {

struct HierarchyPanelState {
    // ImGui::InputTextWithHint backing buffer; bytes outside the trailing
    // \0 are arbitrary. 64 chars fits any reasonable entity-name substring.
    char filter[64]{};
    // One-frame pulse: +1 forces every TreeNode open this frame, -1 closes
    // them. Reset to 0 at the bottom of draw() so subsequent frames honour
    // per-row arrow clicks again.
    int  expand_pulse{0};
};

// Render the hierarchy panel. `selected_id_inout` is the editor's currently
// selected entity (0 = none); the helper writes to it on click / context-
// menu Select / Delete.
void draw(cardinal::scene::Scene& scene,
          cardinal::u32* selected_id_inout,
          const char*    title,
          bool*          p_open,
          HierarchyPanelState& state);

}  // namespace cardinal::ui::panels
