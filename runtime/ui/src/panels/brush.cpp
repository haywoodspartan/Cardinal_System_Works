// =============================================================================
// Studio — Brush panel implementation.
// =============================================================================
#include "brush.hpp"

#include <cardinal/edit/brush.hpp>

#include <cardinal/ui/imgui.hpp>

namespace cardinal::ui::panels::brush_panel {

void draw(cardinal::edit::brush::Brush* b, const char* title, bool* p_open) {
    if (!ImGui::Begin(title ? title : "Brush", p_open,
                      0)) { ImGui::End(); return; }
    if (b == nullptr) {
        ImGui::TextDisabled("(no brush bound)");
        ImGui::End();
        return;
    }

    const char* falloff_names[] = { "None", "Linear", "Smooth", "Gaussian" };
    int f = static_cast<int>(b->falloff);
    if (ImGui::Combo("Falloff", &f, falloff_names, IM_ARRAYSIZE(falloff_names))) {
        b->falloff = static_cast<cardinal::edit::brush::Falloff>(f);
    }

    const char* mode_names[] = { "Add", "Subtract", "Set", "Smooth", "Erase" };
    int m = static_cast<int>(b->mode);
    if (ImGui::Combo("Mode", &m, mode_names, IM_ARRAYSIZE(mode_names))) {
        b->mode = static_cast<cardinal::edit::brush::Mode>(m);
    }

    ImGui::SliderFloat("Radius",   &b->radius_world, 0.05f, 50.0f, "%.2f");
    ImGui::SliderFloat("Strength", &b->strength,     0.0f,  20.0f, "%.2f");
    ImGui::SliderFloat("Spacing",  &b->spacing,      0.05f,  2.0f, "%.2f");

    if (b->mode == cardinal::edit::brush::Mode::Set ||
        b->mode == cardinal::edit::brush::Mode::Erase)
    {
        ImGui::SliderFloat("Target value", &b->target_value, -100.0f, 100.0f, "%.2f");
    }

    // Falloff preview — sample 64 points along the radius and plot.
    static float curve[64];
    for (int i = 0; i < 64; ++i) {
        const float r = (static_cast<float>(i) / 63.0f) * b->radius_world;
        curve[i] = cardinal::edit::brush::weight_at(r, *b);
    }
    ImGui::Separator();
    ImGui::TextDisabled("Falloff curve (center -> edge)");
    ImGui::PlotLines("##falloff_curve", curve, IM_ARRAYSIZE(curve),
                     0, nullptr, 0.0f, 1.0f, ImVec2(-FLT_MIN, 80.0f));

    ImGui::End();
}

}  // namespace cardinal::ui::panels::brush_panel
