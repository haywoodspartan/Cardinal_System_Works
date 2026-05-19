// =============================================================================
// Studio — Input panel implementation.
// =============================================================================
#include "input_panel.hpp"

#include <cardinal/input/input.hpp>

#include <cardinal/ui/imgui.hpp>

#include <cardinal/core/cstdio.hpp>

namespace cardinal::ui::panels::input_panel {

namespace {

const char* bind_kind_name(cardinal::input::BindKind k) {
    switch (k) {
        case cardinal::input::BindKind::Key:           return "Key";
        case cardinal::input::BindKind::MouseButton:   return "Mouse";
        case cardinal::input::BindKind::GamepadButton: return "PadBtn";
        case cardinal::input::BindKind::GamepadAxis:   return "PadAxis";
    }
    return "?";
}

const char* binding_code_name(const cardinal::input::Binding& b) {
    using namespace cardinal::input;
    switch (b.kind) {
        case BindKind::Key:         return key_code_name(static_cast<KeyCode>(b.code));
        case BindKind::MouseButton: {
            switch (static_cast<MouseButton>(b.code)) {
                case MouseButton::Left:   return "Left";
                case MouseButton::Right:  return "Right";
                case MouseButton::Middle: return "Middle";
                case MouseButton::X1:     return "X1";
                case MouseButton::X2:     return "X2";
                default: return "?";
            }
        }
        default: return "?";
    }
}

}  // namespace

void draw(cardinal::input::Manager* mgr, const char* title, bool* p_open) {
    if (!ImGui::Begin(title ? title : "Input", p_open,
                      ImGuiWindowFlags_NoMove)) { ImGui::End(); return; }
    if (mgr == nullptr) {
        ImGui::TextDisabled("(no input::Manager bound)");
        ImGui::End();
        return;
    }

    if (ImGui::CollapsingHeader("Live state", ImGuiTreeNodeFlags_DefaultOpen)) {
        const auto& m = mgr->mouse_state();
        ImGui::Text("Mouse: (%d, %d)  delta (%+d, %+d)  wheel %d",
                    m.x, m.y, m.dx, m.dy, m.wheel);
        ImGui::Text("Buttons: L=%s R=%s M=%s",
                    mgr->mouse_down(cardinal::input::MouseButton::Left)   ? "down" : ".",
                    mgr->mouse_down(cardinal::input::MouseButton::Right)  ? "down" : ".",
                    mgr->mouse_down(cardinal::input::MouseButton::Middle) ? "down" : ".");

        // Inline strip of currently-down keys.
        cardinal::string down_list;
        for (cardinal::u32 i = 1; i < static_cast<cardinal::u32>(cardinal::input::KeyCode::Count); ++i) {
            const auto k = static_cast<cardinal::input::KeyCode>(i);
            if (mgr->key_down(k)) {
                if (!down_list.empty()) down_list += ", ";
                down_list += cardinal::input::key_code_name(k);
            }
        }
        ImGui::Text("Keys down: %s",
                    down_list.empty() ? "(none)" : down_list.c_str());

        const auto stats = mgr->stats();
        ImGui::TextDisabled("events processed: %llu",
            static_cast<unsigned long long>(stats.events_processed));
    }

    if (ImGui::CollapsingHeader("Actions", ImGuiTreeNodeFlags_DefaultOpen)) {
        const auto names = mgr->action_names();
        if (names.empty()) {
            ImGui::TextDisabled("(no actions bound)");
        } else if (ImGui::BeginTable("##actions", 4,
                       ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                       ImGuiTableFlags_SizingStretchProp))
        {
            ImGui::TableSetupColumn("name",     ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("state",    ImGuiTableColumnFlags_WidthFixed,  90.0f);
            ImGui::TableSetupColumn("bindings", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("",         ImGuiTableColumnFlags_WidthFixed,  60.0f);
            ImGui::TableHeadersRow();
            for (const auto& n : names) {
                ImGui::PushID(n.c_str());
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(n.c_str());
                ImGui::TableSetColumnIndex(1);
                const bool down = mgr->action_down(n);
                ImGui::TextColored(down ? ImVec4(0.4f, 0.95f, 0.4f, 1.0f)
                                        : ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                                   "%s", down ? "DOWN" : "up");
                ImGui::TableSetColumnIndex(2);
                cardinal::string blist;
                for (const auto& b : mgr->action_bindings(n)) {
                    if (!blist.empty()) blist += ", ";
                    blist += bind_kind_name(b.kind);
                    blist += ":";
                    blist += binding_code_name(b);
                }
                ImGui::TextUnformatted(blist.c_str());
                ImGui::TableSetColumnIndex(3);
                if (ImGui::SmallButton("Clear")) mgr->clear_action(n);
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
    }

    if (ImGui::CollapsingHeader("Axes", ImGuiTreeNodeFlags_DefaultOpen)) {
        const auto names = mgr->axis_names();
        if (names.empty()) {
            ImGui::TextDisabled("(no axes bound)");
        } else for (const auto& n : names) {
            ImGui::PushID(n.c_str());
            const float v = mgr->axis(n);
            ImGui::Text("%s", n.c_str());
            ImGui::SameLine(160.0f);
            char ov[16];
            cardinal::snprintf(ov, sizeof(ov), "%+.2f", v);
            ImGui::ProgressBar((v + 1.0f) * 0.5f, ImVec2(-FLT_MIN, 0), ov);
            ImGui::PopID();
        }
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::input_panel
