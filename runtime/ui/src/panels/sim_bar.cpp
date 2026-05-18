// =============================================================================
// Studio — Simulation control bar implementation.
// =============================================================================
#include "sim_bar.hpp"

#include <cardinal/sim/sim.hpp>

#include <imgui.h>

#include <cardinal/core/cstdio.hpp>

namespace cardinal::ui::panels::sim_bar_panel {

void draw(cardinal::sim::SimWorld* sim, const char* title, bool* p_open) {
    if (!ImGui::Begin(title ? title : "Simulation", p_open,
                      ImGuiWindowFlags_NoScrollbar))
    { ImGui::End(); return; }
    if (sim == nullptr) {
        ImGui::TextDisabled("(no SimWorld bound)");
        ImGui::End();
        return;
    }

    const auto stats = sim->stats();
    const bool paused = sim->paused();

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 6));
    if (paused) {
        if (ImGui::Button("Play")) sim->resume();
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.45f, 0.20f, 1.0f));
        if (ImGui::Button("Pause")) sim->pause();
        ImGui::PopStyleColor();
    }
    ImGui::SameLine();
    if (ImGui::Button("Step")) sim->step_one_frame();
    ImGui::SameLine();
    ImGui::Text("|");
    ImGui::SameLine();

    float ts = sim->time_scale();
    ImGui::SetNextItemWidth(180.0f);
    if (ImGui::SliderFloat("##timescale", &ts, 0.0f, 4.0f, "%.2fx")) {
        sim->set_time_scale(ts);
    }
    ImGui::SameLine();
    if (ImGui::Button("1x")) sim->set_time_scale(1.0f);
    ImGui::PopStyleVar();

    ImGui::Separator();
    ImGui::Text("sim time : %.2f s", stats.sim_time_seconds);
    ImGui::Text("real time: %.2f s", stats.real_time_seconds);
    ImGui::Text("ticks    : %llu (last physics substeps: %llu)",
        static_cast<unsigned long long>(stats.total_ticks),
        static_cast<unsigned long long>(stats.physics_substeps_last));

    // Hotkeys.
    const ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard) {
        if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
            paused ? sim->resume() : sim->pause();
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Period, false)) sim->step_one_frame();
        if (ImGui::IsKeyPressed(ImGuiKey_1, false)) sim->set_time_scale(0.25f);
        if (ImGui::IsKeyPressed(ImGuiKey_2, false)) sim->set_time_scale(0.50f);
        if (ImGui::IsKeyPressed(ImGuiKey_3, false)) sim->set_time_scale(1.00f);
        if (ImGui::IsKeyPressed(ImGuiKey_4, false)) sim->set_time_scale(2.00f);
        if (ImGui::IsKeyPressed(ImGuiKey_5, false)) sim->set_time_scale(4.00f);
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::sim_bar_panel
