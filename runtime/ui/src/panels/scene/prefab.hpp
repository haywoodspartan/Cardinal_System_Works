#pragma once

// =============================================================================
// Studio — Prefab Library panel.
//
// The runtime prefab workflow: capture the currently-selected actor's
// component configuration into a named, reusable template, then stamp out
// independent instances into the live world. Backed by actor::World's
// create_prefab / spawn_prefab / prefab_names API.
//
//   * "Capture Selected" — snapshots the selected actor into a prefab named
//     by the text field (defaults to the actor's name).
//   * Per-prefab row — Spawn (stamp an instance, auto-selected) + Delete.
//
// Selection is host-owned (u32* selected_actor_id_inout), shared with the
// Actor Outliner / Inspector / Class Picker so capture + spawn agree on
// "the selected actor".
// =============================================================================

#include <cardinal/core/types.hpp>

namespace cardinal::actor { class World; }

namespace cardinal::ui::panels::prefab_panel {

// Persistent panel state the host owns (the in-progress prefab-name text
// buffer). One instance per Studio; pass the same object every frame.
struct State {
    char name_buf[128]{};
    // Disk path for save/load of the whole prefab library. Default lives
    // next to world/sky saves under save/.
    char library_path[512]{ "save/prefabs.cardinalprefab" };
    // Last-operation feedback line shown under the Save/Load buttons.
    char status[160]{};
    // Spawn placement — where the next stamped instance lands (instead of
    // piling at the origin). With auto_step on, X advances by spawn_step
    // after each spawn so repeated stamps lay out in a row.
    float spawn_pos[3]{ 0.0f, 0.0f, 0.0f };
    float spawn_step{ 2.0f };
    bool  auto_step{ true };
    // Rename popup state: which prefab is being renamed + the edit buffer.
    char  rename_target[128]{};
    char  rename_buf[128]{};
};

void draw(State& state,
          cardinal::actor::World* world,
          cardinal::u32* selected_actor_id_inout,
          const char* title = "Prefabs",
          bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::prefab_panel
