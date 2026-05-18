// =============================================================================
// Studio — console panel implementation.
// =============================================================================
#include "console.hpp"

#include <cardinal/console/console.hpp>

#include <imgui.h>

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/cstring.hpp>
#include <cardinal/core/utility.hpp>

namespace cardinal::ui::panels {

void draw(const ConsoleEvalFn& eval,
          const char* title,
          bool* p_open,
          ConsolePanelState& state)
{
    if (!ImGui::Begin(title ? title : "Console", p_open)) { ImGui::End(); return; }

    // Autocomplete suggestion strip — sits between the scrollback and
    // the input. Driven by the engine console registry's prefix search;
    // empty input ⇒ no suggestions, so the strip costs no vertical
    // pixels until the user starts typing. Click a suggestion to fill
    // the input; Tab also accepts the first match.
    cardinal::vector<cardinal::string> suggestions;
    if (state.input[0] != '\0') {
        suggestions = cardinal::console::Registry::instance().complete(
            state.input, 8);
    }
    const float suggest_h = suggestions.empty()
        ? 0.0f
        : ImGui::GetFrameHeightWithSpacing() * cardinal::min<float>(
              static_cast<float>(suggestions.size()), 4.0f);

    const float input_h   = ImGui::GetFrameHeightWithSpacing();
    const float scroll_h  = ImGui::GetContentRegionAvail().y - input_h - suggest_h
                            - (suggestions.empty() ? 0.0f : ImGui::GetStyle().ItemSpacing.y);

    if (ImGui::BeginChild("##console_log",
                          ImVec2(0, scroll_h),
                          ImGuiChildFlags_None,
                          ImGuiWindowFlags_HorizontalScrollbar)) {
        for (const auto& line : state.log) {
            ImGui::TextUnformatted(line.c_str());
        }
        if (state.scroll_to_bottom) {
            ImGui::SetScrollHereY(1.0f);
            state.scroll_to_bottom = false;
        }
    }
    ImGui::EndChild();

    // Suggestions strip — scrollable list of matched names. Each row is
    // clickable: click → fill the input, then the user hits Enter.
    if (!suggestions.empty()) {
        if (ImGui::BeginChild("##console_suggest",
                              ImVec2(0, suggest_h),
                              ImGuiChildFlags_FrameStyle |
                              ImGuiChildFlags_AutoResizeY)) {
            for (const auto& s : suggestions) {
                if (ImGui::Selectable(s.c_str(), false,
                                      ImGuiSelectableFlags_AllowDoubleClick)) {
                    // Single click — autocomplete; the user still presses
                    // Enter to execute. Double click = execute immediately.
                    cardinal::strncpy(state.input, s.c_str(), sizeof(state.input) - 1);
                    state.input[sizeof(state.input) - 1] = '\0';
                    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                        state.pending_submit = true;
                    } else {
                        // Re-focus the input so the user can keep typing.
                        state.request_focus = true;
                    }
                }
                if (ImGui::IsItemHovered()) {
                    // Tooltip with the cvar / command's help string.
                    const auto& reg = cardinal::console::Registry::instance();
                    if (const auto* cv = reg.find_cvar(s)) {
                        if (!cv->help.empty()) ImGui::SetTooltip("%s", cv->help.c_str());
                    } else if (const auto* cc = reg.find_command(s)) {
                        if (!cc->help.empty()) ImGui::SetTooltip("%s", cc->help.c_str());
                    }
                }
            }
        }
        ImGui::EndChild();
    }

    ImGui::Separator();
    ImGui::SetNextItemWidth(-1);
    if (state.request_focus) {
        ImGui::SetKeyboardFocusHere();
        state.request_focus = false;
    }
    const bool entered = ImGui::InputText("##console_input",
                                          state.input, sizeof(state.input),
                                          ImGuiInputTextFlags_EnterReturnsTrue);
    // Tab → accept first suggestion (only when input is focused & has text).
    if (ImGui::IsItemFocused() && !suggestions.empty() &&
        ImGui::IsKeyPressed(ImGuiKey_Tab)) {
        cardinal::strncpy(state.input, suggestions.front().c_str(),
                     sizeof(state.input) - 1);
        state.input[sizeof(state.input) - 1] = '\0';
        state.request_focus = true;
    }
    if (entered || state.pending_submit) {
        state.pending_submit = false;
        cardinal::string in = state.input;
        if (!in.empty()) {
            state.log.push_back("> " + in);
            if (eval) {
                cardinal::string out = eval(in.c_str());
                if (!out.empty()) state.log.push_back(cardinal::move(out));
            }
            state.input[0] = '\0';
            state.scroll_to_bottom = true;
            ImGui::SetKeyboardFocusHere(-1);   // keep focus on the input
        }
    }
    ImGui::End();
}

}  // namespace cardinal::ui::panels
