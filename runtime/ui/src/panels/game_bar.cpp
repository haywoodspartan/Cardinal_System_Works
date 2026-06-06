// =============================================================================
// Studio — Game bar implementation.
// =============================================================================
#include "game_bar.hpp"

#include <cardinal/game/game.hpp>
#include <cardinal/serial/serial.hpp>
#include <cardinal/actor/world.hpp>

#include <cardinal/ui/imgui.hpp>

namespace cardinal::ui::panels::game_bar_panel {

namespace {

// Mark the current revision as already-captured so undo/redo/restore (which
// bump the revision while rebuilding the world) don't spuriously re-capture.
void sync_rev(State& s, cardinal::game::Game* game) {
    s.seen_rev = s.captured_rev = game->world().revision();
    s.stable_frames = 0;
}

// Play: snapshot the authored world (in-memory PIE snapshot + a natural undo
// checkpoint), then enter play.
void do_play(State& s, cardinal::game::Game* game) {
    s.snapshot = cardinal::serial::serialize_world(game->world());
    s.history.capture(game->world());
    sync_rev(s, game);
    game->start_play();
}

// Stop: leave play, then restore the pre-Play snapshot (unless toggled off),
// so playtest mutations don't corrupt the authored scene.
void do_stop(State& s, cardinal::game::Game* game) {
    game->stop_play();
    if (s.restore_on_stop && !s.snapshot.empty()) {
        cardinal::serial::deserialize_world(*game, s.snapshot, /*replace_existing=*/true);
        sync_rev(s, game);
    }
}

}  // namespace

void update(State& s, cardinal::game::Game* game) {
    if (game == nullptr) return;
    const cardinal::game::GameState state = game->state();

    // Lazy baseline checkpoint on first update.
    if (!s.initialized) {
        s.history.capture(game->world());
        sync_rev(s, game);
        s.initialized = true;
    }

    // Debounced auto-checkpoint (editor mode only — Play's mutations are the
    // live sim, not authored edits). Runs every frame regardless of whether
    // the Game panel is visible, so the per-edit-undo contract holds even
    // when the panel is closed or a viewport is maximized.
    if (state == cardinal::game::GameState::Stopped) {
        const cardinal::u64 rev = game->world().revision();
        if (rev == s.seen_rev) {
            if (s.stable_frames < 100000) ++s.stable_frames;
        } else {
            s.seen_rev = rev;
            s.stable_frames = 0;
        }
        // >= (not ==) so a skipped frame near the threshold still captures.
        if (s.stable_frames >= 30 && rev != s.captured_rev) {   // ~0.5s settle
            s.history.capture(game->world());
            s.captured_rev = rev;
        }
    }

    // Play/Pause/Stop hotkeys — global, so they live here (always called) not
    // in draw(). Guarded against typing into a text field.
    //   F5 Play/Resume · F6 Pause-toggle · F8 Stop · ESC pause/resume
    // (Ctrl+Z/Y are intentionally NOT bound — the host owns the scene-graph
    //  undo chord; WorldHistory undo is button-driven. See header.)
    const ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard) {
        if (ImGui::IsKeyPressed(ImGuiKey_F5, false)) {
            if (state == cardinal::game::GameState::Stopped) do_play(s, game);
            else if (state == cardinal::game::GameState::Paused) game->resume_play();
        }
        if (ImGui::IsKeyPressed(ImGuiKey_F6, false)) {
            if (state == cardinal::game::GameState::Playing)      game->pause_play();
            else if (state == cardinal::game::GameState::Paused)  game->resume_play();
        }
        if (ImGui::IsKeyPressed(ImGuiKey_F8, false)) {
            if (state != cardinal::game::GameState::Stopped) do_stop(s, game);
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
            if      (state == cardinal::game::GameState::Playing) game->pause_play();
            else if (state == cardinal::game::GameState::Paused)  game->resume_play();
        }
    }
}

void draw(State& s, cardinal::game::Game* game, const char* title, bool* p_open) {
    if (!ImGui::Begin(title ? title : "Game", p_open,
                      ImGuiWindowFlags_NoScrollbar))
    { ImGui::End(); return; }
    if (game == nullptr) {
        ImGui::TextDisabled("(no game::Game bound)");
        ImGui::End();
        return;
    }

    const cardinal::game::GameState state = game->state();

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(12, 6));
    if (state == cardinal::game::GameState::Stopped) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.65f, 0.30f, 1.0f));
        if (ImGui::Button("Play")) do_play(s, game);
        ImGui::PopStyleColor();
    } else if (state == cardinal::game::GameState::Paused) {
        if (ImGui::Button("Resume")) game->resume_play();
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.55f, 0.20f, 1.0f));
        if (ImGui::Button("Pause")) game->pause_play();
        ImGui::PopStyleColor();
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(state == cardinal::game::GameState::Stopped);
    if (ImGui::Button("Stop")) do_stop(s, game);
    ImGui::EndDisabled();
    ImGui::PopStyleVar();

    ImGui::SameLine();
    ImGui::Checkbox("Restore on Stop", &s.restore_on_stop);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Snapshot the scene on Play and restore it on Stop, "
                          "so playtesting doesn't alter your authored scene.");

    // ---- Undo / redo history (actor world) ---------------------------
    // Checkpoint snapshots the world; Undo/Redo step through them. Only
    // meaningful while editing (Stopped). This undoes ACTOR edits (inspector
    // / outliner / layout / placement); the scene-graph gizmo undo is the
    // host's separate Ctrl+Z stack (see header note).
    ImGui::BeginDisabled(state != cardinal::game::GameState::Stopped);
    if (ImGui::Button("Checkpoint")) { s.history.capture(game->world()); sync_rev(s, game); }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Snapshot the current scene as an undo checkpoint.");
    ImGui::SameLine();
    ImGui::BeginDisabled(!s.history.can_undo());
    if (ImGui::Button("Undo")) { if (s.history.undo(*game)) sync_rev(s, game); }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!s.history.can_redo());
    if (ImGui::Button("Redo")) { if (s.history.redo(*game)) sync_rev(s, game); }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("(%zu/%zu)", s.history.cursor() + 1, s.history.depth());
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::Text("|  state: ");
    ImGui::SameLine();
    ImVec4 col(0.7f, 0.7f, 0.7f, 1.0f);
    switch (state) {
        case cardinal::game::GameState::Stopped: col = {0.7f,  0.7f,  0.7f,  1.0f}; break;
        case cardinal::game::GameState::Playing: col = {0.30f, 0.85f, 0.45f, 1.0f}; break;
        case cardinal::game::GameState::Paused:  col = {0.95f, 0.75f, 0.30f, 1.0f}; break;
    }
    ImGui::TextColored(col, "%s", cardinal::game::game_state_name(state));

    ImGui::Separator();
    ImGui::Text("Game actors: %u (%u pending begin_play)",
                game->game_actor_count(), game->begin_play_pending());

    ImGui::End();
}

}  // namespace cardinal::ui::panels::game_bar_panel
