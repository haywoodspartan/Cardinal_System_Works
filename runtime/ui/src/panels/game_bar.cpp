// =============================================================================
// Studio — Game bar implementation.
// =============================================================================
#include "game_bar.hpp"

#include <cardinal/game/game.hpp>
#include <cardinal/serial/serial.hpp>
#include <cardinal/serial/world_history.hpp>
#include <cardinal/actor/world.hpp>

#include <cardinal/ui/imgui.hpp>

namespace cardinal::ui::panels::game_bar_panel {

void draw(cardinal::game::Game* game, const char* title, bool* p_open) {
    if (!ImGui::Begin(title ? title : "Game", p_open,
                      ImGuiWindowFlags_NoScrollbar))
    { ImGui::End(); return; }
    if (game == nullptr) {
        ImGui::TextDisabled("(no game::Game bound)");
        ImGui::End();
        return;
    }

    const cardinal::game::GameState state = game->state();

    // ---- Play-mode scene snapshot / restore --------------------------
    //
    // Playtesting mutates the live world (physics moves bodies, scripts
    // spawn/destroy actors, the player walks around). Without a snapshot
    // those changes would permanently corrupt the authored scene. On Play
    // we serialize the world to an in-memory snapshot; on Stop we rebuild
    // it, restoring the scene exactly as it was — the standard UE/Unity PIE
    // contract. Toggle off to keep play-mode changes. In-memory (no temp
    // file): nothing touches disk for a transient playtest.
    static bool             s_restore_on_stop = true;
    static cardinal::string s_snapshot;

    // Undo / redo history — snapshot checkpoints of the authored world.
    // Auto-captures: a checkpoint is taken when the scene SETTLES after an
    // edit (debounced via the World revision counter, so a continuous drag
    // yields ONE checkpoint, not one per frame). sync_rev() marks the
    // current revision as already-captured, so undo/redo/restore — which
    // bump the revision while rebuilding the world — don't spuriously
    // re-capture the result as a new checkpoint.
    static cardinal::serial::WorldHistory s_history;
    static cardinal::u64 s_seen_rev = 0, s_captured_rev = 0;
    static int s_stable_frames = 0;
    auto sync_rev = [&]() {
        s_seen_rev = s_captured_rev = game->world().revision();
        s_stable_frames = 0;
    };
    if (s_history.empty()) { s_history.capture(game->world()); sync_rev(); }

    // Debounced auto-checkpoint (editor mode only — never during Play, whose
    // mutations are the live sim, not authored edits).
    if (state == cardinal::game::GameState::Stopped) {
        const cardinal::u64 rev = game->world().revision();
        if (rev == s_seen_rev) {
            if (s_stable_frames < 100000) ++s_stable_frames;
        } else {
            s_seen_rev = rev;
            s_stable_frames = 0;
        }
        if (s_stable_frames == 30 && rev != s_captured_rev) {   // ~0.5s settle
            s_history.capture(game->world());
            s_captured_rev = rev;
        }
    }

    auto play_with_snapshot = [&]() {
        s_snapshot = cardinal::serial::serialize_world(game->world());
        s_history.capture(game->world());   // also a natural undo checkpoint
        sync_rev();
        game->start_play();
    };
    auto stop_with_restore = [&]() {
        game->stop_play();
        if (s_restore_on_stop && !s_snapshot.empty()) {
            cardinal::serial::deserialize_world(*game, s_snapshot, /*replace_existing=*/true);
            sync_rev();                     // restored state isn't a new edit
        }
    };

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(12, 6));
    if (state == cardinal::game::GameState::Stopped) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.65f, 0.30f, 1.0f));
        if (ImGui::Button("Play")) play_with_snapshot();
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
    if (ImGui::Button("Stop")) stop_with_restore();
    ImGui::EndDisabled();
    ImGui::PopStyleVar();

    ImGui::SameLine();
    ImGui::Checkbox("Restore on Stop", &s_restore_on_stop);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Snapshot the scene on Play and restore it on Stop, "
                          "so playtesting doesn't alter your authored scene.");

    // ---- Undo / redo history -----------------------------------------
    // Checkpoint snapshots the world; Undo/Redo step through them. Only
    // meaningful while editing (Stopped) — while playing, the world is the
    // live sim, not the authored scene.
    ImGui::BeginDisabled(state != cardinal::game::GameState::Stopped);
    if (ImGui::Button("Checkpoint")) { s_history.capture(game->world()); sync_rev(); }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Snapshot the current scene as an undo checkpoint.");
    ImGui::SameLine();
    ImGui::BeginDisabled(!s_history.can_undo());
    if (ImGui::Button("Undo")) { if (s_history.undo(*game)) sync_rev(); }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!s_history.can_redo());
    if (ImGui::Button("Redo")) { if (s_history.redo(*game)) sync_rev(); }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("(%zu/%zu)", s_history.cursor() + 1, s_history.depth());
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

    // Hotkeys — read at the top of the frame so any panel that consumes
    // them sees a consistent state. WantCaptureKeyboard guards against
    // typing into a text field.
    //
    // F5  : Play / Resume
    // F6  : Pause toggle
    // F8  : Stop
    // ESC : viewport "pause sim" — pauses while Playing, resumes from
    //       Paused. Studio thread (panels, gizmos, asset browser) keeps
    //       running because it reads UI input directly through ImGui;
    //       only the sim tick group + gameplay action map are gated.
    //       Mirrors UE5's PIE escape behaviour.
    const ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard) {
        if (ImGui::IsKeyPressed(ImGuiKey_F5, false)) {
            if (state == cardinal::game::GameState::Stopped) play_with_snapshot();
            else if (state == cardinal::game::GameState::Paused) game->resume_play();
        }
        if (ImGui::IsKeyPressed(ImGuiKey_F6, false)) {
            if (state == cardinal::game::GameState::Playing)      game->pause_play();
            else if (state == cardinal::game::GameState::Paused)  game->resume_play();
        }
        if (ImGui::IsKeyPressed(ImGuiKey_F8, false)) {
            if (state != cardinal::game::GameState::Stopped) stop_with_restore();
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
            if      (state == cardinal::game::GameState::Playing) game->pause_play();
            else if (state == cardinal::game::GameState::Paused)  game->resume_play();
            // In Stopped state ESC is a no-op (Studio is already idle).
        }
        // Ctrl+Z undo / Ctrl+Y (or Ctrl+Shift+Z) redo — editor mode only.
        if (state == cardinal::game::GameState::Stopped && io.KeyCtrl) {
            if (ImGui::IsKeyPressed(ImGuiKey_Z, false)) {
                if (io.KeyShift) { if (s_history.redo(*game)) sync_rev(); }
                else             { if (s_history.undo(*game)) sync_rev(); }
            }
            if (ImGui::IsKeyPressed(ImGuiKey_Y, false)) { if (s_history.redo(*game)) sync_rev(); }
        }
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::game_bar_panel
