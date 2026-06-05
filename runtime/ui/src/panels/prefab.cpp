// =============================================================================
// Studio — Prefab Library panel implementation.
// =============================================================================
#include "prefab.hpp"

#include <cardinal/actor/world.hpp>
#include <cardinal/serial/serial.hpp>
#include <cardinal/ui/imgui.hpp>

#include <cardinal/core/cstdio.hpp>
#include <cardinal/core/cstring.hpp>

namespace cardinal::ui::panels::prefab_panel {

void draw(State& state,
          cardinal::actor::World* world,
          cardinal::u32* selected_actor_id_inout,
          const char* title, bool* p_open)
{
    if (!ImGui::Begin(title ? title : "Prefabs", p_open, 0)) {
        ImGui::End();
        return;
    }
    if (world == nullptr) {
        ImGui::TextDisabled("(no actor::World bound)");
        ImGui::End();
        return;
    }

    const cardinal::u32 sel = selected_actor_id_inout ? *selected_actor_id_inout : 0u;
    cardinal::actor::Actor* sel_actor = world->find(sel);

    // ---- Capture section ----------------------------------------------
    ImGui::SeparatorText("Capture");
    if (sel_actor != nullptr) {
        ImGui::Text("Selected: %s  (%zu component%s)",
                    sel_actor->name().c_str(),
                    sel_actor->components().size(),
                    sel_actor->components().size() == 1 ? "" : "s");

        // Default the name field to the actor's name the first time a new
        // actor is selected + the buffer is empty.
        if (state.name_buf[0] == '\0') {
            cardinal::snprintf(state.name_buf, sizeof(state.name_buf),
                               "%s", sel_actor->name().c_str());
        }
        ImGui::SetNextItemWidth(-110.0f);
        ImGui::InputText("##prefab_name", state.name_buf, sizeof(state.name_buf));
        ImGui::SameLine();
        const bool has_name = state.name_buf[0] != '\0';
        if (!has_name) ImGui::BeginDisabled();
        if (ImGui::Button("Capture", ImVec2(100.0f, 0.0f))) {
            if (world->create_prefab(state.name_buf, sel)) {
                // Clear the buffer so the next selection re-seeds it.
                state.name_buf[0] = '\0';
            }
        }
        if (!has_name) ImGui::EndDisabled();
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Snapshot the selected actor's components into "
                              "a reusable prefab template.");
        }

        // ---- Prefab instance linkage (Revert / Apply) -----------------
        // When the selected actor is an INSTANCE of a prefab, offer the
        // Unity-style edit loop: revert local edits, or push them up.
        const cardinal::string linked = world->prefab_of(sel);
        if (!linked.empty()) {
            ImGui::Spacing();
            ImGui::Text("Instance of prefab: %s", linked.c_str());
            if (ImGui::Button("Revert to Prefab")) {
                if (world->revert_to_prefab(sel)) {
                    cardinal::snprintf(state.status, sizeof(state.status),
                                       "Reverted to prefab '%s'.", linked.c_str());
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Discard this instance's local edits and "
                                  "restore the prefab's component values.");
            }
            ImGui::SameLine();
            if (ImGui::Button("Apply to Prefab")) {
                if (world->apply_to_prefab(sel)) {
                    cardinal::snprintf(state.status, sizeof(state.status),
                                       "Applied edits to prefab '%s'.", linked.c_str());
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Push this instance's current component "
                                  "values back into the prefab — every future "
                                  "spawn + revert inherits them.");
            }
        }
    } else {
        ImGui::TextDisabled("(select an actor in the Outliner to capture it)");
        // Reset the staged name when nothing is selected.
        state.name_buf[0] = '\0';
    }

    // ---- Disk (save / load the whole library) -------------------------
    ImGui::SeparatorText("Library File");
    ImGui::SetNextItemWidth(-160.0f);
    ImGui::InputText("##prefab_lib_path", state.library_path, sizeof(state.library_path));
    ImGui::SameLine();
    if (ImGui::Button("Save##lib", ImVec2(72.0f, 0.0f))) {
        cardinal::string err;
        const auto ss = cardinal::serial::save_prefabs(*world, state.library_path, &err);
        if (!err.empty()) {
            cardinal::snprintf(state.status, sizeof(state.status),
                               "Save failed: %s", err.c_str());
        } else {
            cardinal::snprintf(state.status, sizeof(state.status),
                               "Saved %u prefab(s), %u component(s).",
                               ss.prefabs_written, ss.components_written);
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Load##lib", ImVec2(72.0f, 0.0f))) {
        cardinal::string err;
        const auto ls = cardinal::serial::load_prefabs(*world, state.library_path,
                                                       /*replace_existing=*/true, &err);
        if (!err.empty()) {
            cardinal::snprintf(state.status, sizeof(state.status),
                               "Load failed: %s", err.c_str());
        } else {
            cardinal::snprintf(state.status, sizeof(state.status),
                               "Loaded %u prefab(s), %u component(s)%s.",
                               ls.prefabs_loaded, ls.components_loaded,
                               ls.components_skipped ? " (some skipped)" : "");
        }
    }
    if (state.status[0] != '\0') {
        ImGui::TextDisabled("%s", state.status);
    }

    // ---- Library section ----------------------------------------------
    ImGui::SeparatorText("Library");
    const auto names = world->prefab_names();
    if (names.empty()) {
        ImGui::TextDisabled("(no prefabs captured yet)");
        ImGui::End();
        return;
    }
    ImGui::Text("%zu prefab%s", names.size(), names.size() == 1 ? "" : "s");

    if (ImGui::BeginTable("##prefab_table", 3,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH |
                          ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Name",       ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Components",  ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableSetupColumn("Actions",     ImGuiTableColumnFlags_WidthFixed, 150.0f);
        ImGui::TableHeadersRow();

        // We may delete during iteration; collect a pending delete and apply
        // after the loop so the names vector stays valid.
        cardinal::string to_delete;

        for (const auto& n : names) {
            ImGui::TableNextRow();
            ImGui::PushID(n.c_str());

            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(n.c_str());

            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%u", world->prefab_component_count(n));

            ImGui::TableSetColumnIndex(2);
            if (ImGui::SmallButton("Spawn")) {
                if (auto* inst = world->spawn_prefab(n)) {
                    if (selected_actor_id_inout) *selected_actor_id_inout = inst->id();
                }
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Delete")) {
                to_delete = n;
            }

            ImGui::PopID();
        }
        ImGui::EndTable();

        if (!to_delete.empty()) world->remove_prefab(to_delete);
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::prefab_panel
