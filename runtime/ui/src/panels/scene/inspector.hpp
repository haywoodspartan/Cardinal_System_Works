#pragma once

// =============================================================================
// Studio — inspector panel (free-function helper).
//
// Pure stateless: every input comes through the call. The host hands us
// the scene + selected entity id; we render editable controls for the
// selection's name, visibility, transform, tint, and mesh metadata.
// =============================================================================

#include <cardinal/core/types.hpp>

namespace cardinal::scene { class Scene; }

namespace cardinal::ui::panels::inspector_panel {

// out_edit_active / out_edit_committed (optional): edit-lifecycle report for
// host-side undo — `active` is true while any inspector widget is mid-edit
// (drag/typing in progress); `committed` is true on the frame an edit finishes
// (drag released / text confirmed / checkbox toggled / reset clicked). The host
// snapshots the entity before the edit begins and pushes one undo entry on
// commit (same pattern as the gizmo's drag-start/drag-end undo).
void draw(cardinal::scene::Scene& scene,
          cardinal::u32 selected_id,
          const char*   title,
          bool*         p_open,
          bool*         out_edit_active    = nullptr,
          bool*         out_edit_committed = nullptr);

}  // namespace cardinal::ui::panels::inspector_panel
