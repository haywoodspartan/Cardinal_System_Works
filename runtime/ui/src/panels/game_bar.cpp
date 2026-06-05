// =============================================================================
// Studio — Game bar implementation.
// =============================================================================
#include "game_bar.hpp"

#include <cardinal/game/game.hpp>
#include <cardinal/serial/serial.hpp>
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
    // we serialize the world to a snapshot file; on Stop we reload it,
    // restoring the scene exactly as it was — the standard UE/Unity PIE
    // contract. Toggle off to keep play-mode changes.
    static bool s_restore_on_stop = true;
    static bool s_has_snapshot    = false;
    constexpr const char* kSnapPath = "save/.pie_snapshot.cardinalworld";

    auto play_with_snapshot = [&]() {
        cardinal::string err;
        cardinal::serial::save_world(game->world(), kSnapPath, &err);
        s_has_snapshot = err.empty();
        game->start_play();
    };
    auto stop_with_restore = [&]() {
        game->stop_play();
        if (s_restore_on_stop && s_has_snapshot) {
            cardinal::serial::load_world(*game, kSnapPath, /*replace_existing=*/true);
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
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::game_bar_panel
