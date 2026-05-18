// =============================================================================
// Studio — inspector panel implementation.
// =============================================================================
#include "inspector.hpp"

#include <cardinal/scene/scene.hpp>

#include <imgui.h>

#include <cstring>

namespace cardinal::ui::panels::inspector_panel {

void draw(cardinal::scene::Scene& scene,
          cardinal::u32 selected_id,
          const char* title,
          bool* p_open)
{
    if (!ImGui::Begin(title ? title : "Inspector", p_open)) { ImGui::End(); return; }

    if (selected_id == 0) {
        ImGui::TextDisabled("(no selection)");
        ImGui::End();
        return;
    }
    auto* e = scene.find_by_id(selected_id);
    if (e == nullptr) {
        ImGui::TextDisabled("(entity %u not found)", selected_id);
        ImGui::End();
        return;
    }

    char namebuf[128];
    std::strncpy(namebuf, e->name.c_str(), sizeof(namebuf) - 1);
    namebuf[sizeof(namebuf) - 1] = '\0';
    if (ImGui::InputText("Name", namebuf, sizeof(namebuf))) {
        e->name = namebuf;
    }
    ImGui::Checkbox("Visible", &e->visible);

    ImGui::SeparatorText("Transform");
    ImGui::DragFloat3("Position", &e->transform.translation.x, 0.05f);
    // Display rotation in degrees but store radians.
    float rot_deg[3] = {
        e->transform.rotation_euler.x * (180.0f / 3.14159265f),
        e->transform.rotation_euler.y * (180.0f / 3.14159265f),
        e->transform.rotation_euler.z * (180.0f / 3.14159265f),
    };
    if (ImGui::DragFloat3("Rotation", rot_deg, 0.5f)) {
        e->transform.rotation_euler.x = rot_deg[0] * (3.14159265f / 180.0f);
        e->transform.rotation_euler.y = rot_deg[1] * (3.14159265f / 180.0f);
        e->transform.rotation_euler.z = rot_deg[2] * (3.14159265f / 180.0f);
    }
    ImGui::DragFloat3("Scale", &e->transform.scale.x, 0.01f, 0.001f, 100.0f);

    ImGui::SeparatorText("Material");
    ImGui::ColorEdit3("Tint", &e->tint.x);

    ImGui::SeparatorText("Mesh");
    if (e->mesh) {
        ImGui::Text("Name:  %s", e->mesh->name().c_str());
        ImGui::Text("Verts: %u", e->mesh->vertex_count());
    } else {
        ImGui::TextDisabled("(none)");
    }

    ImGui::Separator();
    if (ImGui::Button("Delete entity")) {
        scene.remove_entity(selected_id);
    }
    ImGui::End();
}

}  // namespace cardinal::ui::panels::inspector_panel
