// =============================================================================
// Studio — Save / Load panel implementation.
// =============================================================================
#include "save_load.hpp"

#include <cardinal/game/game.hpp>
#include <cardinal/serial/serial.hpp>
#include <cardinal/sky/sky.hpp>

#include <imgui.h>

#include <cstring>

namespace cardinal::ui::panels::save_load_panel {

void draw(cardinal::game::Game* game, cardinal::sky::Sky* sky,
          const char* title, bool* p_open)
{
    if (!ImGui::Begin(title ? title : "Save / Load", p_open)) { ImGui::End(); return; }
    if (game == nullptr) {
        ImGui::TextDisabled("(no game::Game bound)");
        ImGui::End();
        return;
    }

    static char world_path[512] = "save/world.cardinalworld";
    static char sky_path[512]   = "save/sky.cardinalsky";
    static bool replace_on_load = true;
    static cardinal::serial::SaveStats last_save;
    static cardinal::serial::LoadStats last_load;
    static std::string last_error;

    if (ImGui::CollapsingHeader("World", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::InputText("Path##world", world_path, sizeof(world_path));
        if (ImGui::Button("Save world")) {
            last_error.clear();
            last_save = cardinal::serial::save_world(game->world(),
                                                    world_path, &last_error);
        }
        ImGui::SameLine();
        ImGui::Checkbox("Replace existing on load", &replace_on_load);

        if (ImGui::Button("Load world")) {
            last_error.clear();
            last_load = cardinal::serial::load_world(*game,
                                                    world_path,
                                                    replace_on_load,
                                                    &last_error);
        }

        ImGui::TextDisabled("Last save: %u actors, %u props (%llu bytes)",
            last_save.actors_written, last_save.properties_written,
            static_cast<unsigned long long>(last_save.bytes_written));
        ImGui::TextDisabled("Last load: %u spawned, %u props, %u skipped, %u errors",
            last_load.actors_spawned, last_load.properties_applied,
            last_load.actors_skipped, last_load.errors);
        if (!last_error.empty()) {
            ImGui::TextColored(ImVec4(0.95f, 0.4f, 0.4f, 1.0f),
                               "Error: %s", last_error.c_str());
        }
    }

    if (sky && ImGui::CollapsingHeader("Sky", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::InputText("Path##sky", sky_path, sizeof(sky_path));
        if (ImGui::Button("Save sky")) {
            std::string err;
            cardinal::serial::save_sky(*sky, sky_path, &err);
            if (!err.empty()) last_error = err;
        }
        ImGui::SameLine();
        if (ImGui::Button("Load sky")) {
            std::string err;
            cardinal::serial::load_sky(*sky, sky_path, &err);
            if (!err.empty()) last_error = err;
        }
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::save_load_panel
