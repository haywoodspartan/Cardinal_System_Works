// =============================================================================
// Studio — Particles panel implementation.
// =============================================================================
#include "particles_panel.hpp"

#include <cardinal/particles/particles.hpp>

#include <cardinal/ui/imgui.hpp>

#include <cardinal/core/cstdio.hpp>

namespace cardinal::ui::panels::particles_panel {

namespace {

void edit_emitter(cardinal::particles::Emitter& e) {
    auto& d = e.desc();
    ImGui::Checkbox("emitting", &d.emitting);
    ImGui::SameLine();
    ImGui::Text("(live %zu / total %llu)",
        e.live_count(), static_cast<unsigned long long>(e.total_spawned()));

    ImGui::DragFloat3("origin",       &d.origin.x, 0.1f);
    ImGui::SliderFloat("rate (/sec)", &d.rate_per_second, 0.0f, 5000.0f, "%.0f",
                       ImGuiSliderFlags_Logarithmic);
    {
        int mp = static_cast<int>(d.max_particles);
        if (ImGui::SliderInt("max particles", &mp, 16, 65536, "%d",
                             ImGuiSliderFlags_Logarithmic)) {
            d.max_particles = static_cast<cardinal::u32>(mp);
        }
    }

    ImGui::DragFloatRange2("lifetime",
        &d.lifetime_min, &d.lifetime_max, 0.05f, 0.05f, 30.0f, "%.2fs");
    ImGui::DragFloatRange2("size",
        &d.size_min, &d.size_max, 0.005f, 0.001f, 5.0f, "%.3f");
    ImGui::DragFloat3("velocity min", &d.velocity_min.x, 0.05f);
    ImGui::DragFloat3("velocity max", &d.velocity_max.x, 0.05f);
    ImGui::DragFloat3("gravity",       &d.gravity.x, 0.05f);
    ImGui::SliderFloat("drag (/sec)",  &d.drag, 0.0f, 5.0f, "%.2f");

    auto color_picker = [](const char* lbl, cardinal::u32& rgba) {
        float v[4] = {
            ((rgba >>  0) & 0xFF) / 255.0f,
            ((rgba >>  8) & 0xFF) / 255.0f,
            ((rgba >> 16) & 0xFF) / 255.0f,
            ((rgba >> 24) & 0xFF) / 255.0f,
        };
        if (ImGui::ColorEdit4(lbl, v)) {
            rgba = (static_cast<cardinal::u32>(v[0] * 255.0f)) |
                   (static_cast<cardinal::u32>(v[1] * 255.0f) << 8) |
                   (static_cast<cardinal::u32>(v[2] * 255.0f) << 16) |
                   (static_cast<cardinal::u32>(v[3] * 255.0f) << 24);
        }
    };
    color_picker("start color", d.start_rgba);
    color_picker("end   color", d.end_rgba);

    if (ImGui::Button("Clear particles")) e.clear();
}

}  // namespace

void draw(cardinal::particles::System* sys, const char* title, bool* p_open) {
    if (!ImGui::Begin(title ? title : "Particles", p_open)) { ImGui::End(); return; }
    if (sys == nullptr) {
        ImGui::TextDisabled("(no particles::System bound)");
        ImGui::End();
        return;
    }

    const auto stats = sys->stats();
    ImGui::Text("Emitters: %u | Live: %u | Total spawned: %llu",
        stats.emitters_active, stats.particles_alive,
        static_cast<unsigned long long>(stats.particles_total_spawned));

    if (ImGui::Button("+ Add emitter")) {
        cardinal::particles::EmitterDesc d{};
        d.name = "Emitter";
        d.origin = {0.0f, 1.0f, 0.0f};
        d.rate_per_second = 60.0f;
        sys->add(d);
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear all")) sys->clear();

    ImGui::Separator();

    cardinal::particles::Emitter* to_remove = nullptr;
    int idx = 0;
    for (const auto& e : sys->emitters()) {
        ImGui::PushID(e.get());
        char hdr[80];
        cardinal::snprintf(hdr, sizeof(hdr), "[%d] %s (%zu live)",
            idx, e->desc().name.empty() ? "Emitter" : e->desc().name.c_str(),
            e->live_count());
        if (ImGui::CollapsingHeader(hdr, ImGuiTreeNodeFlags_DefaultOpen)) {
            edit_emitter(*e);
            if (ImGui::Button("Remove")) to_remove = e.get();
        }
        ImGui::PopID();
        ++idx;
    }
    if (to_remove != nullptr) sys->remove(to_remove);

    ImGui::End();
}

}  // namespace cardinal::ui::panels::particles_panel
