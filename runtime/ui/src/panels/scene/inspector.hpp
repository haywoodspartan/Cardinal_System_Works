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

void draw(cardinal::scene::Scene& scene,
          cardinal::u32 selected_id,
          const char*   title,
          bool*         p_open);

}  // namespace cardinal::ui::panels::inspector_panel
