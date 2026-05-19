// =============================================================================
// Studio — C++ scripting & sandbox panel implementation.
// =============================================================================
#include "cppscript.hpp"

#include <cardinal/cppscript/cppscript.hpp>
#include <cardinal/sandbox/sandbox.hpp>

#include <cardinal/ui/imgui.hpp>

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/cstdio.hpp>
#include <cardinal/core/cstring.hpp>

namespace cardinal::ui::panels::cppscript_panel {

namespace {

const char* status_name(cardinal::cppscript::JobStatus s) {
    using S = cardinal::cppscript::JobStatus;
    switch (s) {
        case S::Pending:        return "Pending";
        case S::Compiling:      return "Compiling";
        case S::LoadPending:    return "LoadPending";
        case S::Loaded:         return "Loaded";
        case S::CompileFailed:  return "CompileFailed";
        case S::LoadFailed:     return "LoadFailed";
    }
    return "?";
}

ImU32 status_color(cardinal::cppscript::JobStatus s) {
    using S = cardinal::cppscript::JobStatus;
    switch (s) {
        case S::Pending:        return IM_COL32(150, 150, 150, 255);
        case S::Compiling:      return IM_COL32(180, 180, 100, 255);
        case S::LoadPending:    return IM_COL32(160, 200, 100, 255);
        case S::Loaded:         return IM_COL32(100, 220, 100, 255);
        case S::CompileFailed:  return IM_COL32(230, 100, 100, 255);
        case S::LoadFailed:     return IM_COL32(230, 130,  60, 255);
    }
    return IM_COL32_WHITE;
}

}  // namespace

void draw(cardinal::cppscript::Engine* engine,
          const char* title,
          bool* p_open,
          State& state,
          SandboxLookup lookup_sandbox)
{
    if (!ImGui::Begin(title ? title : "Code Sandbox", p_open,
                      ImGuiWindowFlags_NoMove)) { ImGui::End(); return; }

    if (engine == nullptr) {
        ImGui::TextDisabled("(no cppscript::Engine bound)");
        ImGui::End();
        return;
    }

    // ---- Compiler banner --------------------------------------------------
    const cardinal::string compiler = engine->compiler_path();
    if (compiler.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(220, 80, 80, 255));
        ImGui::TextWrapped("No C++ compiler discovered. Install Visual Studio 2022 / "
                           "clang-cl, or set CARDINAL_CPPSCRIPT_COMPILER.");
        ImGui::PopStyleColor();
    } else {
        ImGui::TextDisabled("compiler:");
        ImGui::SameLine();
        ImGui::TextWrapped("%s", compiler.c_str());
    }
    ImGui::Separator();

    // ---- Compile input bar ------------------------------------------------
    ImGui::SetNextItemWidth(-220.0f);
    bool submit = ImGui::InputTextWithHint("##cs_path",
        "path to .cpp (drop a file here or paste)",
        state.input_path, sizeof(state.input_path),
        ImGuiInputTextFlags_EnterReturnsTrue);
    // Drag-drop target on the input field — drop a file from any source
    // that supplies an external payload (Studio's asset browser, OS file
    // explorer via Win32 message hook).
    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload =
                ImGui::AcceptDragDropPayload("CARDINAL_FILE_PATH")) {
            const usize n = cardinal::min<usize>(static_cast<usize>(payload->DataSize),
                                            sizeof(state.input_path) - 1);
            cardinal::memcpy(state.input_path, payload->Data, n);
            state.input_path[n] = '\0';
        }
        ImGui::EndDragDropTarget();
    }
    ImGui::SameLine();
    if (ImGui::Button("Compile + Load") || submit) {
        if (state.input_path[0] != '\0') {
            engine->compile_and_load(state.input_path);
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Watch") && state.input_path[0] != '\0') {
        engine->watch(state.input_path);
    }
    ImGui::SameLine();
    ImGui::Checkbox("subprocess", &state.prefer_subprocess);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "When checked, the host should construct the resulting plugin "
            "inside a cardinal::sandbox subprocess so a script crash can't "
            "take the editor down. (Wired by the host; the panel only sets "
            "the preference flag.)");
    }
    ImGui::Separator();

    // ---- Job table --------------------------------------------------------
    auto jobs = engine->jobs();
    // Newest first — read the top of the panel as "what just happened?".
    cardinal::sort(jobs.begin(), jobs.end(),
        [](const auto& a, const auto& b) { return a.handle.id > b.handle.id; });

    const float bottom_h = 180.0f;   // diagnostics pane below
    const float table_h  = cardinal::max(80.0f, ImGui::GetContentRegionAvail().y - bottom_h);
    if (ImGui::BeginTable("##cs_jobs", 6,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_ScrollY  | ImGuiTableFlags_SizingStretchProp,
                          ImVec2(0, table_h)))
    {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("#",       ImGuiTableColumnFlags_WidthFixed, 36.0f);
        ImGui::TableSetupColumn("status",  ImGuiTableColumnFlags_WidthFixed, 110.0f);
        ImGui::TableSetupColumn("source",  ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("time",    ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("sandbox", ImGuiTableColumnFlags_WidthFixed, 140.0f);
        ImGui::TableSetupColumn("",        ImGuiTableColumnFlags_WidthFixed, 130.0f);
        ImGui::TableHeadersRow();

        for (const auto& j : jobs) {
            ImGui::TableNextRow();
            ImGui::PushID(static_cast<int>(j.handle.id));

            ImGui::TableSetColumnIndex(0);
            const bool selected = (state.selected_job_id == j.handle.id);
            char idbuf[16]; cardinal::snprintf(idbuf, sizeof(idbuf), "%llu",
                static_cast<unsigned long long>(j.handle.id));
            if (ImGui::Selectable(idbuf, selected,
                ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap))
            {
                state.selected_job_id = j.handle.id;
            }

            ImGui::TableSetColumnIndex(1);
            ImGui::PushStyleColor(ImGuiCol_Text, status_color(j.status));
            ImGui::TextUnformatted(status_name(j.status));
            ImGui::PopStyleColor();
            if (j.status == cardinal::cppscript::JobStatus::CompileFailed ||
                j.status == cardinal::cppscript::JobStatus::LoadFailed) {
                ImGui::SameLine();
                ImGui::TextDisabled("rc=%lld", static_cast<long long>(j.exit_code));
            }

            ImGui::TableSetColumnIndex(2);
            ImGui::TextUnformatted(j.source_path.c_str());
            if (ImGui::IsItemHovered() && !j.dll_path.empty()) {
                ImGui::SetTooltip("DLL: %s", j.dll_path.c_str());
            }

            ImGui::TableSetColumnIndex(3);
            if (j.elapsed_seconds > 0.0) {
                ImGui::Text("%.2fs", j.elapsed_seconds);
            } else {
                ImGui::TextDisabled("-");
            }

            ImGui::TableSetColumnIndex(4);
            cardinal::sandbox::Status sb_status{};
            const bool has_sb = lookup_sandbox && lookup_sandbox(j.source_path, &sb_status);
            if (has_sb) {
                ImGui::Text("%s pid=%llu", sb_status.alive ? "alive" : "DEAD",
                            static_cast<unsigned long long>(sb_status.pid));
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Mode: %s\nTicks: %llu\nLast tick: %.3f ms\nLast error: %s",
                        sb_status.mode == cardinal::sandbox::Mode::Subprocess
                            ? "Subprocess" : "InProcess",
                        static_cast<unsigned long long>(sb_status.ticks),
                        sb_status.last_tick_ms,
                        sb_status.last_error.empty() ? "(none)" : sb_status.last_error.c_str());
                }
            } else {
                ImGui::TextDisabled("(in-process)");
            }

            ImGui::TableSetColumnIndex(5);
            if (j.status == cardinal::cppscript::JobStatus::Loaded) {
                if (ImGui::SmallButton("Reload")) {
                    engine->reload(j.source_path.c_str());
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("Unload")) {
                    engine->unload(j.source_path.c_str());
                }
            } else if (j.status == cardinal::cppscript::JobStatus::CompileFailed ||
                       j.status == cardinal::cppscript::JobStatus::LoadFailed) {
                if (ImGui::SmallButton("Retry")) {
                    engine->reload(j.source_path.c_str());
                }
            } else {
                ImGui::TextDisabled("(working...)");
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    // ---- Diagnostics pane -------------------------------------------------
    ImGui::Separator();
    ImGui::Text("Diagnostics");
    if (ImGui::BeginChild("##cs_diag", ImVec2(0, 0),
                          ImGuiChildFlags_FrameStyle,
                          ImGuiWindowFlags_HorizontalScrollbar))
    {
        if (state.selected_job_id != 0) {
            // Find selected job — could cache, but jobs list is short.
            cardinal::cppscript::JobInfo sel{};
            for (const auto& j : jobs) {
                if (j.handle.id == state.selected_job_id) { sel = j; break; }
            }
            if (sel.handle.id == 0) {
                ImGui::TextDisabled("(job %llu evicted)",
                    static_cast<unsigned long long>(state.selected_job_id));
            } else {
                ImGui::TextDisabled("source: %s", sel.source_path.c_str());
                if (!sel.dll_path.empty()) {
                    ImGui::TextDisabled("dll:    %s", sel.dll_path.c_str());
                }
                ImGui::Separator();
                if (sel.diagnostics.empty()) {
                    ImGui::TextDisabled("(no compiler output)");
                } else {
                    ImGui::TextUnformatted(sel.diagnostics.c_str());
                }
            }
        } else {
            ImGui::TextDisabled("Select a job above to see compiler output / diagnostics.");
        }
    }
    ImGui::EndChild();

    ImGui::End();
}

}  // namespace cardinal::ui::panels::cppscript_panel
