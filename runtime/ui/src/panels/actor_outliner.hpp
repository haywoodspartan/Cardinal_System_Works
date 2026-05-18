#pragma once

// =============================================================================
// Studio — Actor Outliner + Inspector.
//
// Two-pane panel: tree of actors on the left, component inspector on the
// right for the selected actor. The component inspector handles every
// built-in component type (Transform, Mesh, Camera, Light, AudioEmitter,
// RigidBody, Tag, Script). Custom components fall through to a generic
// "(type) (no inspector)" line.
//
// Selection is host-owned (the host passes a u32* selected_actor_id_inout).
// =============================================================================

#include <cardinal/core/types.hpp>

namespace cardinal::actor { class World; }

namespace cardinal::ui::panels::actor_outliner_panel {

void draw(cardinal::actor::World* world,
          cardinal::u32* selected_actor_id_inout,
          const char* title = "Actors",
          bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::actor_outliner_panel
