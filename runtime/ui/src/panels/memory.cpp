// =============================================================================
// Studio — Memory & Budget panel implementation.
// =============================================================================
#include "memory.hpp"

#include <cardinal/core/budget.hpp>
#include <cardinal/core/memory.hpp>

#include <imgui.h>

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/cstdio.hpp>

namespace cardinal::ui::panels::memory_panel {

namespace {

ImU32 tier_color(cardinal::memory::Pressure p) {
    using P = cardinal::memory::Pressure;
    switch (p) {
        case P::Low:      return IM_COL32(110, 220, 110, 255);
        case P::Medium:   return IM_COL32(220, 200,  90, 255);
        case P::High:     return IM_COL32(230, 140,  60, 255);
        case P::Critical: return IM_COL32(230,  80,  80, 255);
    }
    return IM_COL32_WHITE;
}

void tier_swatch(cardinal::memory::Pressure p) {
    ImGui::PushStyleColor(ImGuiCol_Text, tier_color(p));
    ImGui::Text("[%s]", cardinal::memory::pressure_name(p));
    ImGui::PopStyleColor();
}

void mb_text(const char* label, cardinal::u64 bytes) {
    ImGui::Text("%-18s %.1f MB", label,
        static_cast<double>(bytes) / (1024.0 * 1024.0));
}

void mb_progress(double frac, const char* overlay) {
    frac = cardinal::clamp(frac, 0.0, 1.0);
    ImVec4 col(0.40f, 0.85f, 0.40f, 1.0f);
    if (frac >= 0.92) col = ImVec4(0.90f, 0.30f, 0.30f, 1.0f);
    else if (frac >= 0.80) col = ImVec4(0.95f, 0.60f, 0.25f, 1.0f);
    else if (frac >= 0.60) col = ImVec4(0.90f, 0.85f, 0.30f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, col);
    ImGui::ProgressBar(static_cast<float>(frac), ImVec2(-FLT_MIN, 0), overlay);
    ImGui::PopStyleColor();
}

}  // namespace

void draw(cardinal::budget::Broker* broker, const char* title, bool* p_open) {
    if (!ImGui::Begin(title ? title : "Memory & Budgets", p_open)) {
        ImGui::End();
        return;
    }

    if (broker == nullptr) {
        ImGui::TextDisabled("(no budget broker bound)");
        ImGui::End();
        return;
    }

    const auto& snap = broker->last_snapshot();

    if (ImGui::CollapsingHeader("System RAM", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Pressure: ");
        ImGui::SameLine();
        tier_swatch(snap.system_pressure);

        const cardinal::u64 used =
            (snap.system.total_bytes > snap.system.available_bytes)
                ? (snap.system.total_bytes - snap.system.available_bytes)
                : 0;
        char overlay[64];
        cardinal::snprintf(overlay, sizeof(overlay),
            "%.1f / %.1f GiB (%.1f%%)",
            static_cast<double>(used)                  / (1024.0 * 1024.0 * 1024.0),
            static_cast<double>(snap.system.total_bytes)/ (1024.0 * 1024.0 * 1024.0),
            snap.system.load_percent);
        mb_progress(snap.system.load_percent / 100.0, overlay);

        mb_text("Total:",     snap.system.total_bytes);
        mb_text("Available:", snap.system.available_bytes);
    }

    if (ImGui::CollapsingHeader("Process", ImGuiTreeNodeFlags_DefaultOpen)) {
        mb_text("Working set:", snap.process.working_set_bytes);
        mb_text("Peak WS:",     snap.process.peak_working_set_bytes);
        mb_text("Private:",     snap.process.private_bytes);
    }

    if (ImGui::CollapsingHeader("GPU VRAM", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Pressure: ");
        ImGui::SameLine();
        tier_swatch(snap.gpu_pressure);

        if (snap.gpu_budget_bytes == 0) {
            ImGui::TextDisabled("(no GPU bound, or driver doesn't report a budget)");
        } else {
            const double frac = static_cast<double>(snap.gpu_current_usage_bytes) /
                                static_cast<double>(snap.gpu_budget_bytes);
            char overlay[64];
            cardinal::snprintf(overlay, sizeof(overlay),
                "%.1f / %.1f GiB (%.1f%%)",
                static_cast<double>(snap.gpu_current_usage_bytes)
                    / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(snap.gpu_budget_bytes)
                    / (1024.0 * 1024.0 * 1024.0),
                frac * 100.0);
            mb_progress(frac, overlay);

            mb_text("Budget:",  snap.gpu_budget_bytes);
            mb_text("Used:",    snap.gpu_current_usage_bytes);
        }
    }

    if (ImGui::CollapsingHeader("Subsystems", ImGuiTreeNodeFlags_DefaultOpen)) {
        const auto reports = broker->subsystem_reports();
        if (reports.empty()) {
            ImGui::TextDisabled("(no subsystems registered)");
        } else {
            if (ImGui::BeginTable("##mem_subs", 5,
                    ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                    ImGuiTableFlags_SizingStretchProp))
            {
                ImGui::TableSetupColumn("name",   ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("domain", ImGuiTableColumnFlags_WidthFixed,  60.0f);
                ImGui::TableSetupColumn("tier",   ImGuiTableColumnFlags_WidthFixed,  90.0f);
                ImGui::TableSetupColumn("used",   ImGuiTableColumnFlags_WidthFixed, 100.0f);
                ImGui::TableSetupColumn("range",  ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableHeadersRow();

                for (const auto& r : reports) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(r.name.c_str());
                    ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted(
                        cardinal::budget::domain_name(r.domain));
                    ImGui::TableSetColumnIndex(2); tier_swatch(r.last_seen_tier);
                    ImGui::TableSetColumnIndex(3); ImGui::Text("%.1f MB",
                        static_cast<double>(r.used_bytes) / (1024.0 * 1024.0));
                    ImGui::TableSetColumnIndex(4);
                    if (r.advisory_max_bytes > 0) {
                        const double frac = cardinal::clamp(
                            static_cast<double>(r.used_bytes) /
                            static_cast<double>(r.advisory_max_bytes), 0.0, 1.0);
                        char ov[64];
                        cardinal::snprintf(ov, sizeof(ov), "%.1f / %.1f MB",
                            static_cast<double>(r.used_bytes) / (1024.0 * 1024.0),
                            static_cast<double>(r.advisory_max_bytes) / (1024.0 * 1024.0));
                        mb_progress(frac, ov);
                    } else {
                        ImGui::TextDisabled("(no advisory range)");
                    }
                }
                ImGui::EndTable();
            }
        }
    }

    if (ImGui::CollapsingHeader("Debug — force tier", 0)) {
        const char* names[] = { "Low", "Medium", "High", "Critical" };
        static int sys_force = -1;
        static int gpu_force = -1;
        ImGui::TextDisabled("Pin a pressure tier to test subsystem callbacks.");

        ImGui::PushID("sys");
        ImGui::Text("System: ");
        ImGui::SameLine();
        const int prev_sys = sys_force;
        if (ImGui::Combo("##sysforce", &sys_force,
                         "(real)\0Low\0Medium\0High\0Critical\0")) {
            if (sys_force <= 0) {
                broker->debug_clear_force(cardinal::budget::Domain::System);
            } else {
                broker->debug_force_pressure(cardinal::budget::Domain::System,
                    static_cast<cardinal::memory::Pressure>(sys_force - 1));
            }
            (void)prev_sys;
            (void)names;
        }
        ImGui::PopID();

        ImGui::PushID("gpu");
        ImGui::Text("GPU:    ");
        ImGui::SameLine();
        if (ImGui::Combo("##gpuforce", &gpu_force,
                         "(real)\0Low\0Medium\0High\0Critical\0")) {
            if (gpu_force <= 0) {
                broker->debug_clear_force(cardinal::budget::Domain::Gpu);
            } else {
                broker->debug_force_pressure(cardinal::budget::Domain::Gpu,
                    static_cast<cardinal::memory::Pressure>(gpu_force - 1));
            }
        }
        ImGui::PopID();
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::memory_panel
