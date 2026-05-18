// =============================================================================
// Studio — Sky / Time-of-Day panel implementation.
// =============================================================================
#include "sky_panel.hpp"

#include <cardinal/sky/sky.hpp>

#include <imgui.h>

#include <cstdio>

namespace cardinal::ui::panels::sky_panel {

namespace {

const char* hour_label(float h) {
    static char buf[16];
    const int hh = static_cast<int>(h);
    const int mm = static_cast<int>((h - hh) * 60.0f);
    std::snprintf(buf, sizeof(buf), "%02d:%02d", hh, mm);
    return buf;
}

void color_swatch(const char* label, const cardinal::scene::Vec3& c) {
    const ImU32 col = IM_COL32(
        static_cast<int>(c.x * 255.0f),
        static_cast<int>(c.y * 255.0f),
        static_cast<int>(c.z * 255.0f), 255);
    const ImVec2 sz(20.0f, 20.0f);
    ImGui::TextDisabled("%s", label);
    ImGui::SameLine();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::GetWindowDrawList()->AddRectFilled(
        p, ImVec2(p.x + sz.x, p.y + sz.y), col);
    ImGui::Dummy(sz);
}

}  // namespace

void draw(cardinal::sky::Sky* sky, const char* title, bool* p_open) {
    if (!ImGui::Begin(title ? title : "Sky / Time of Day", p_open)) {
        ImGui::End();
        return;
    }
    if (sky == nullptr) {
        ImGui::TextDisabled("(no sky::Sky bound)");
        ImGui::End();
        return;
    }

    if (ImGui::CollapsingHeader("Clock", ImGuiTreeNodeFlags_DefaultOpen)) {
        float h = sky->hour();
        if (ImGui::SliderFloat("Hour", &h, 0.0f, 23.999f, hour_label(h))) {
            sky->set_hour(h);
        }
        bool frozen = sky->frozen();
        if (ImGui::Checkbox("Freeze time", &frozen)) sky->set_frozen(frozen);

        float ts = sky->time_scale();
        if (ImGui::SliderFloat("Hours per real second", &ts, 0.0f, 1440.0f, "%.3f")) {
            sky->set_time_scale(ts);
        }
        float day_len = sky->day_length_seconds();
        if (ImGui::SliderFloat("Day length (sec)", &day_len, 1.0f, 86400.0f,
                               "%.0f", ImGuiSliderFlags_Logarithmic)) {
            sky->set_day_length_seconds(day_len);
        }
        // Quick presets.
        if (ImGui::SmallButton("Real-time"))     sky->set_day_length_seconds(86400.0f);
        ImGui::SameLine();
        if (ImGui::SmallButton("1 hour day"))    sky->set_day_length_seconds(3600.0f);
        ImGui::SameLine();
        if (ImGui::SmallButton("10 minute day")) sky->set_day_length_seconds(600.0f);
        ImGui::SameLine();
        if (ImGui::SmallButton("60 second day")) sky->set_day_length_seconds(60.0f);
    }

    if (ImGui::CollapsingHeader("Live state", ImGuiTreeNodeFlags_DefaultOpen)) {
        const auto& s = sky->state();
        ImGui::Text("Hour:           %s", hour_label(s.hour));
        ImGui::Text("Sun direction: (%+.2f, %+.2f, %+.2f)",
                    s.sun_dir.x, s.sun_dir.y, s.sun_dir.z);
        ImGui::Text("Sun intensity: %.2f", s.sun_intensity);
        color_swatch("zenith ",  s.zenith);
        color_swatch("horizon",  s.horizon);
        color_swatch("sun    ",  s.sun_color);
        color_swatch("ambient",  s.ambient);
    }

    if (ImGui::CollapsingHeader("Phase keys", 0)) {
        if (ImGui::Button("Reset to defaults")) sky->reset_to_defaults();
        ImGui::SameLine();
        if (ImGui::Button("Sort by hour")) sky->sort_keys();
        if (ImGui::BeginTable("##sky_keys", 6,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                ImGuiTableFlags_SizingStretchProp))
        {
            ImGui::TableSetupColumn("hour",    ImGuiTableColumnFlags_WidthFixed,  60.0f);
            ImGui::TableSetupColumn("zenith",  ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("horizon", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("sun col", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("sun int", ImGuiTableColumnFlags_WidthFixed,  90.0f);
            ImGui::TableSetupColumn("ambient", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();
            for (auto& k : sky->keys()) {
                ImGui::PushID(&k);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::SetNextItemWidth(-FLT_MIN);
                    ImGui::DragFloat("##h", &k.hour, 0.05f, 0.0f, 23.999f, "%.2f");
                ImGui::TableSetColumnIndex(1); ImGui::SetNextItemWidth(-FLT_MIN);
                    ImGui::ColorEdit3("##ze", &k.zenith.x);
                ImGui::TableSetColumnIndex(2); ImGui::SetNextItemWidth(-FLT_MIN);
                    ImGui::ColorEdit3("##ho", &k.horizon.x);
                ImGui::TableSetColumnIndex(3); ImGui::SetNextItemWidth(-FLT_MIN);
                    ImGui::ColorEdit3("##sc", &k.sun_color.x);
                ImGui::TableSetColumnIndex(4); ImGui::SetNextItemWidth(-FLT_MIN);
                    ImGui::DragFloat("##si", &k.sun_intensity, 0.01f, 0.0f, 4.0f, "%.2f");
                ImGui::TableSetColumnIndex(5); ImGui::SetNextItemWidth(-FLT_MIN);
                    ImGui::ColorEdit3("##am", &k.ambient.x);
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::sky_panel
