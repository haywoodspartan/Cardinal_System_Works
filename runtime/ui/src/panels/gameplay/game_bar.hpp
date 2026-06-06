#pragma once

// =============================================================================
// Studio — Game lifecycle bar.
//
// Drives a cardinal::game::Game: Play / Pause / Resume / Stop with state
// readout, the play-in-editor snapshot/restore, and the actor-world undo
// history (WorldHistory) with a revision-debounced auto-checkpoint.
//
// Split into update() + draw() so the per-frame LOGIC runs regardless of
// panel visibility:
//   * update(state, game) — call EVERY frame, unconditionally. Advances the
//     auto-checkpoint debounce + handles the Play/Pause/Stop hotkeys
//     (F5/F6/F8/ESC). Lives outside draw() so authored edits made while the
//     Game panel is hidden / a viewport is maximized still get undo
//     checkpoints (the per-edit-undo contract no longer silently breaks).
//   * draw(state, game, ...) — call only when the panel is visible. Renders
//     the Play/Stop buttons, Restore-on-Stop toggle, Checkpoint/Undo/Redo,
//     and the state readout.
//
// Keyboard hotkeys (when no input field has focus):
//   F5 — Play (or Resume from Paused)   F6 — Pause / Resume toggle
//   F8 — Stop                            ESC — pause/resume while playing
//
// NOTE on undo scope: WorldHistory snapshots the ACTOR world only. The Studio
// sample's edit::UndoStack is a SEPARATE undo for the scene render graph
// (scene::Entity gizmo drags / placements / deletes). They cover two distinct
// representations, so they remain separate; a single unified undo would need
// a combined actor+scene snapshot history.
// =============================================================================

#include <cardinal/core/types.hpp>            // cardinal::string / u64
#include <cardinal/serial/world_history.hpp>  // cardinal::serial::WorldHistory

namespace cardinal::game { class Game; }

namespace cardinal::ui::panels::game_bar_panel {

// Host-owned persistent state (one per Studio). The host calls update() each
// frame and draw() when the panel is visible, passing the same object.
struct State {
    // explicit-ctor type — needs an explicit initializer so State{} compiles.
    cardinal::serial::WorldHistory history{64};   // actor-world undo snapshots
    cardinal::string snapshot;                // in-memory PIE snapshot
    bool          restore_on_stop = true;
    cardinal::u64 seen_rev        = 0;
    cardinal::u64 captured_rev    = 0;
    int           stable_frames   = 0;
    bool          initialized     = false;    // lazy baseline checkpoint guard
};

// Per-frame logic — call unconditionally, every frame (within the ImGui
// frame, since it reads input for the hotkeys).
void update(State& state, cardinal::game::Game* game);

// Render the bar — call only when the panel is visible.
void draw(State& state, cardinal::game::Game* game,
          const char* title = "Game", bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::game_bar_panel
