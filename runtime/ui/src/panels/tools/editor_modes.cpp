// =============================================================================
// Studio — Editor modes toolbar implementation.
// =============================================================================
#include "editor_modes.hpp"

#include <cardinal/edit/editor_mode.hpp>
#include <cardinal/core/std/cstdio.hpp>

#include <cardinal/ui/imgui.hpp>

namespace cardinal::ui::panels::editor_modes_panel {

void draw(cardinal::edit::EditorState* state, const char* title, bool* p_open) {
    if (!ImGui::Begin(title ? title : "Modes", p_open,
                      ImGuiWindowFlags_NoScrollbar)) { ImGui::End(); return; }
    if (state == nullptr) {
        ImGui::TextDisabled("(no EditorState bound)");
        ImGui::End();
        return;
    }

    const cardinal::edit::EditorMode all[] = {
        cardinal::edit::EditorMode::Select,
        cardinal::edit::EditorMode::Place,
        cardinal::edit::EditorMode::Sculpt,
        cardinal::edit::EditorMode::Paint,
        cardinal::edit::EditorMode::Foliage,
        cardinal::edit::EditorMode::Mesh,
        cardinal::edit::EditorMode::Landscape,
        cardinal::edit::EditorMode::Measure,
    };

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 6));
    for (cardinal::edit::EditorMode m : all) {
        const bool active = (state->mode() == m);
        if (active) {
            ImGui::PushStyleColor(ImGuiCol_Button,
                                  ImVec4(0.30f, 0.55f, 0.85f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                  ImVec4(0.35f, 0.62f, 0.92f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                  ImVec4(0.25f, 0.50f, 0.80f, 1.0f));
        }
        char label[48];
        cardinal::snprintf(label, sizeof(label), "%s %s##mode%u",
                      cardinal::edit::editor_mode_glyph(m),
                      cardinal::edit::editor_mode_name(m),
                      static_cast<unsigned>(m));
        if (ImGui::Button(label)) state->set_mode(m);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", cardinal::edit::editor_mode_tooltip(m));
        }
        if (active) ImGui::PopStyleColor(3);
        ImGui::SameLine();
    }
    ImGui::PopStyleVar();
    ImGui::NewLine();

    if (!state->status_text().empty()) {
        ImGui::Separator();
        ImGui::TextDisabled("%s", state->status_text().c_str());
    }

    // Hotkeys: Q W E R T Y U I — only when no text input is focused.
    const ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard) {
        struct KH { ImGuiKey k; cardinal::edit::EditorMode m; };
        const KH keys[] = {
            { ImGuiKey_Q, cardinal::edit::EditorMode::Select    },
            { ImGuiKey_W, cardinal::edit::EditorMode::Place     },
            { ImGuiKey_E, cardinal::edit::EditorMode::Sculpt    },
            { ImGuiKey_R, cardinal::edit::EditorMode::Paint     },
            { ImGuiKey_T, cardinal::edit::EditorMode::Foliage   },
            { ImGuiKey_Y, cardinal::edit::EditorMode::Mesh      },
            { ImGuiKey_U, cardinal::edit::EditorMode::Landscape },
            { ImGuiKey_I, cardinal::edit::EditorMode::Measure   },
        };
        for (const auto& kh : keys) {
            if (ImGui::IsKeyPressed(kh.k, /*repeat=*/false)) state->set_mode(kh.m);
        }
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::editor_modes_panel
