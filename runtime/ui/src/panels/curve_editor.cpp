// =============================================================================
// Studio — Curve editor implementation.
// =============================================================================
#include "curve_editor.hpp"

#include <cardinal/anim/anim.hpp>

#include <imgui.h>

#include <algorithm>
#include <cstdio>

namespace cardinal::ui::panels::curve_editor_panel {

void draw(cardinal::anim::Curve<float>* curve, const char* title, bool* p_open) {
    if (!ImGui::Begin(title ? title : "Curve Editor", p_open)) {
        ImGui::End();
        return;
    }
    if (curve == nullptr) {
        ImGui::TextDisabled("(no curve bound)");
        ImGui::End();
        return;
    }

    // ---- Toolbar -----
    if (ImGui::Button("+ Key")) curve->add_key(curve->duration() + 0.5f, 0.0f);
    ImGui::SameLine();
    if (ImGui::Button("Clear")) curve->clear();
    ImGui::SameLine();
    const char* wraps[] = { "Clamp", "Loop", "PingPong" };
    int w = static_cast<int>(curve->wrap);
    ImGui::SetNextItemWidth(110.0f);
    if (ImGui::Combo("Wrap", &w, wraps, IM_ARRAYSIZE(wraps))) {
        curve->wrap = static_cast<cardinal::anim::WrapMode>(w);
    }

    // ---- Compute display range ----
    float t_min = 0.0f, t_max = std::max(1.0f, curve->duration() + 1.0f);
    float v_min = -1.0f, v_max = 1.0f;
    for (const auto& k : curve->keys) {
        v_min = std::min(v_min, k.value);
        v_max = std::max(v_max, k.value);
    }
    if (v_max - v_min < 0.5f) { v_min -= 0.5f; v_max += 0.5f; }

    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const ImVec2 graph_min = ImGui::GetCursorScreenPos();
    const float  width  = std::max(120.0f, avail.x);
    const float  height = std::max(120.0f, avail.y - 40.0f);
    const ImVec2 graph_max(graph_min.x + width, graph_min.y + height);

    auto x_of = [&](float t) {
        return graph_min.x + (t - t_min) / (t_max - t_min) * width;
    };
    auto y_of = [&](float v) {
        return graph_max.y - (v - v_min) / (v_max - v_min) * height;
    };
    auto t_of_x = [&](float x) {
        return t_min + (x - graph_min.x) / width * (t_max - t_min);
    };
    auto v_of_y = [&](float y) {
        return v_min + (graph_max.y - y) / height * (v_max - v_min);
    };

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(graph_min, graph_max, IM_COL32(20, 22, 26, 255));
    dl->AddRect(graph_min, graph_max, IM_COL32(80, 80, 80, 255));

    // Grid.
    for (int i = 0; i <= 8; ++i) {
        const float fx = graph_min.x + i / 8.0f * width;
        dl->AddLine({fx, graph_min.y}, {fx, graph_max.y}, IM_COL32(50, 50, 50, 200));
    }
    for (int j = 0; j <= 4; ++j) {
        const float fy = graph_min.y + j / 4.0f * height;
        dl->AddLine({graph_min.x, fy}, {graph_max.x, fy}, IM_COL32(50, 50, 50, 200));
    }

    // Curve sampled at every pixel.
    if (!curve->empty()) {
        ImVec2 prev{ x_of(t_min), y_of(curve->sample(t_min)) };
        const int steps = std::max(64, static_cast<int>(width));
        for (int i = 1; i <= steps; ++i) {
            const float t = t_min + (t_max - t_min) * i / steps;
            const ImVec2 cur{ x_of(t), y_of(curve->sample(t)) };
            dl->AddLine(prev, cur, IM_COL32(110, 200, 255, 220), 1.5f);
            prev = cur;
        }
    }

    // Keys.
    int hovered_key = -1;
    for (int i = 0; i < static_cast<int>(curve->keys.size()); ++i) {
        const auto& k = curve->keys[i];
        const ImVec2 p{ x_of(k.time), y_of(k.value) };
        const float r = 6.0f;
        const ImVec2 hit_min{ p.x - r, p.y - r };
        const ImVec2 hit_max{ p.x + r, p.y + r };
        const bool hovered = ImGui::IsMouseHoveringRect(hit_min, hit_max);
        const ImU32 col = hovered ? IM_COL32(255, 220, 100, 255)
                                  : IM_COL32(220, 180,  80, 255);
        dl->AddCircleFilled(p, r, col);
        dl->AddCircle(p, r + 1.0f, IM_COL32(20, 20, 20, 255), 0, 1.5f);
        if (hovered) hovered_key = i;
    }
    ImGui::Dummy(ImVec2(width, height));

    // ---- Interaction -----
    static int  drag_key = -1;
    static bool drag_active = false;
    const ImGuiIO& io = ImGui::GetIO();
    const bool over_graph = ImGui::IsMouseHoveringRect(graph_min, graph_max);

    if (over_graph) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            if (hovered_key >= 0) { drag_key = hovered_key; drag_active = true; }
        }
        if (drag_active && drag_key >= 0 && drag_key < static_cast<int>(curve->keys.size())) {
            auto& k = curve->keys[drag_key];
            k.time  = std::max(0.0f, t_of_x(io.MousePos.x));
            k.value = v_of_y(io.MousePos.y);
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            // Re-sort by time after a drag.
            if (drag_active) {
                std::sort(curve->keys.begin(), curve->keys.end(),
                    [](const auto& a, const auto& b){ return a.time < b.time; });
            }
            drag_key = -1; drag_active = false;
        }
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && hovered_key < 0) {
            curve->add_key(t_of_x(io.MousePos.x), v_of_y(io.MousePos.y));
        }
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && hovered_key >= 0) {
            curve->keys.erase(curve->keys.begin() + hovered_key);
        }
    }

    ImGui::Text("keys: %zu, duration: %.2f s", curve->keys.size(), curve->duration());
    ImGui::TextDisabled("LMB drag = move • dbl-click = add • RMB on key = delete");

    // Per-key inspector for the hovered key.
    if (hovered_key >= 0 && hovered_key < static_cast<int>(curve->keys.size())) {
        auto& k = curve->keys[hovered_key];
        ImGui::Separator();
        ImGui::Text("Key #%d", hovered_key);
        ImGui::DragFloat("t",  &k.time,  0.05f, 0.0f, 1000.0f, "%.3f");
        ImGui::DragFloat("v",  &k.value, 0.05f);
        const char* modes[] = { "Step", "Linear", "CubicHermite" };
        int m = static_cast<int>(k.mode);
        if (ImGui::Combo("Interp", &m, modes, IM_ARRAYSIZE(modes))) {
            k.mode = static_cast<cardinal::anim::InterpMode>(m);
        }
        if (k.mode == cardinal::anim::InterpMode::CubicHermite) {
            ImGui::DragFloat("In tangent",  &k.in_tangent,  0.05f);
            ImGui::DragFloat("Out tangent", &k.out_tangent, 0.05f);
        }
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::curve_editor_panel
