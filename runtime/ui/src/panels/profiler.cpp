// =============================================================================
// Studio — Profiler panel implementation.
// =============================================================================
#include "profiler.hpp"

#include <cardinal/core/async.hpp>

#include <imgui.h>

#include <algorithm>

namespace cardinal::ui::panels::profiler_panel {

void draw(State& s, cardinal::f32 frame_ms_now,
          const char* title, bool* p_open)
{
    // Push into rolling history before any early return so the chart fills
    // even when the user collapses the panel.
    s.frame_history[s.history_pos] = frame_ms_now;
    s.history_pos = (s.history_pos + 1u) % IM_ARRAYSIZE(s.frame_history);
    if (s.history_count < IM_ARRAYSIZE(s.frame_history)) ++s.history_count;
    s.last_frame_ms = frame_ms_now;

    if (!ImGui::Begin(title ? title : "Profiler", p_open)) { ImGui::End(); return; }

    if (ImGui::CollapsingHeader("Frame", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Compute a max for the y-axis from the visible history.
        float ymax = 1.0f;
        for (cardinal::u32 i = 0; i < s.history_count; ++i) {
            ymax = std::max(ymax, s.frame_history[i]);
        }
        char overlay[64];
        std::snprintf(overlay, sizeof(overlay),
            "%.2f ms (%.0f FPS)",
            s.last_frame_ms,
            s.last_frame_ms > 0.001f ? 1000.0f / s.last_frame_ms : 0.0f);
        ImGui::PlotLines("##frame_ms", s.frame_history,
            IM_ARRAYSIZE(s.frame_history),
            static_cast<int>(s.history_pos),
            overlay, 0.0f, ymax * 1.1f, ImVec2(-FLT_MIN, 80.0f));
    }

    if (ImGui::CollapsingHeader("Phases (per frame, EMA-smoothed)",
                                ImGuiTreeNodeFlags_DefaultOpen))
    {
        const auto phases = cardinal::async::phase_samples();
        if (phases.empty()) {
            ImGui::TextDisabled("(no FrameScope() blocks reporting yet)");
        } else if (ImGui::BeginTable("##phases", 5,
                       ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                       ImGuiTableFlags_SizingStretchProp))
        {
            ImGui::TableSetupColumn("phase",   ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("last ms", ImGuiTableColumnFlags_WidthFixed,  90.0f);
            ImGui::TableSetupColumn("avg ms",  ImGuiTableColumnFlags_WidthFixed,  90.0f);
            ImGui::TableSetupColumn("peak ms", ImGuiTableColumnFlags_WidthFixed,  90.0f);
            ImGui::TableSetupColumn("calls",   ImGuiTableColumnFlags_WidthFixed,  60.0f);
            ImGui::TableHeadersRow();
            for (const auto& p : phases) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(p.name.c_str());
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.3f", p.ms_last);
                ImGui::TableSetColumnIndex(2); ImGui::Text("%.3f", p.ms_avg);
                ImGui::TableSetColumnIndex(3); ImGui::Text("%.3f", p.ms_max);
                ImGui::TableSetColumnIndex(4); ImGui::Text("%u",   p.calls_last);
            }
            ImGui::EndTable();
        }
    }

    if (ImGui::CollapsingHeader("Workers", ImGuiTreeNodeFlags_DefaultOpen)) {
        const auto workers = cardinal::async::worker_stats();
        ImGui::Text("Active workers: %zu", workers.size());
        if (workers.empty()) {
            ImGui::TextDisabled("(no worker registered yet — JobSystem not started?)");
        } else if (ImGui::BeginTable("##workers", 6,
                       ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                       ImGuiTableFlags_SizingStretchProp))
        {
            ImGui::TableSetupColumn("worker", ImGuiTableColumnFlags_WidthFixed,  60.0f);
            ImGui::TableSetupColumn("core",   ImGuiTableColumnFlags_WidthFixed,  60.0f);
            ImGui::TableSetupColumn("numa",   ImGuiTableColumnFlags_WidthFixed,  56.0f);
            ImGui::TableSetupColumn("tier",   ImGuiTableColumnFlags_WidthFixed,  84.0f);
            ImGui::TableSetupColumn("busy",   ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("jobs",   ImGuiTableColumnFlags_WidthFixed, 100.0f);
            ImGui::TableHeadersRow();
            for (const auto& w : workers) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("%u", w.worker_id);
                ImGui::TableSetColumnIndex(1); ImGui::Text("%u", w.logical_core);
                ImGui::TableSetColumnIndex(2); ImGui::Text("%u", w.numa_node);
                ImGui::TableSetColumnIndex(3);
                // Color-coded tier label so imbalances stand out at a glance.
                ImVec4   col;
                const char* label;
                switch (w.tier) {
                    case cardinal::async::WorkerTier::Performance:
                        col = ImVec4(1.0f, 0.85f, 0.40f, 1.0f); label = "perf";       break;
                    case cardinal::async::WorkerTier::General:
                        col = ImVec4(0.7f, 0.85f, 1.00f, 1.0f); label = "general";    break;
                    case cardinal::async::WorkerTier::Background:
                    default:
                        col = ImVec4(0.6f, 0.6f,  0.65f, 1.0f); label = "background"; break;
                }
                ImGui::TextColored(col, "%s", label);
                ImGui::TableSetColumnIndex(4);
                char ov[32];
                std::snprintf(ov, sizeof(ov), "%.1f%%", w.busy_fraction * 100.0);
                ImGui::ProgressBar(static_cast<float>(w.busy_fraction),
                                   ImVec2(-FLT_MIN, 0), ov);
                ImGui::TableSetColumnIndex(5);
                ImGui::Text("%llu (steals %llu)",
                    static_cast<unsigned long long>(w.jobs_executed),
                    static_cast<unsigned long long>(w.bytes_stolen));
            }
            ImGui::EndTable();
        }
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::profiler_panel
