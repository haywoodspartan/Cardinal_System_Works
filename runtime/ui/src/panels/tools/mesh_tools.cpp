// =============================================================================
// Studio — Mesh tools panel implementation.
// =============================================================================
#include "mesh_tools.hpp"

#include <cardinal/ui/imgui.hpp>

namespace cardinal::ui::panels::mesh_tools_panel {

void draw(State* st, bool selection_has_mesh,
          const char* title, bool* p_open)
{
    if (!ImGui::Begin(title ? title : "Mesh Tools", p_open,
                      0)) {
        ImGui::End();
        return;
    }
    if (st == nullptr) {
        ImGui::TextDisabled("(no state)");
        ImGui::End();
        return;
    }

    // Reset clicked flags — host reads them in the same frame after the
    // panel call.
    st->generate_clicked         = false;
    st->subdivide_clicked        = false;
    st->mirror_clicked           = false;
    st->decimate_clicked         = false;
    st->smooth_clicked           = false;
    st->recompute_normals_clicked= false;

    if (ImGui::CollapsingHeader("Generate primitive",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        const char* names[] = { "Box", "Sphere", "Cylinder", "Cone",
                                "Torus", "Disk", "Plane" };
        int k = static_cast<int>(st->prim_kind);
        if (ImGui::Combo("Shape", &k, names, IM_ARRAYSIZE(names))) {
            st->prim_kind = static_cast<PrimitiveKind>(k);
        }
        switch (st->prim_kind) {
            case PrimitiveKind::Box:
                ImGui::SliderFloat("Size",        &st->prim_size,    0.05f, 50.0f, "%.2f");
                break;
            case PrimitiveKind::Plane:
                ImGui::SliderFloat("Size",        &st->prim_size,    0.1f, 100.0f, "%.2f");
                {
                    int seg = static_cast<int>(st->prim_segments);
                    if (ImGui::SliderInt("Subdivisions", &seg, 1, 64))
                        st->prim_segments = static_cast<cardinal::u32>(seg);
                }
                break;
            case PrimitiveKind::Sphere:
            case PrimitiveKind::Disk:
                ImGui::SliderFloat("Radius",      &st->prim_radius,  0.05f, 25.0f, "%.2f");
                {
                    int seg = static_cast<int>(st->prim_segments);
                    if (ImGui::SliderInt("Segments", &seg, 4, 128))
                        st->prim_segments = static_cast<cardinal::u32>(seg);
                }
                break;
            case PrimitiveKind::Cylinder:
            case PrimitiveKind::Cone:
                ImGui::SliderFloat("Radius",      &st->prim_radius,  0.05f, 25.0f, "%.2f");
                ImGui::SliderFloat("Height",      &st->prim_height,  0.05f, 50.0f, "%.2f");
                {
                    int seg = static_cast<int>(st->prim_segments);
                    if (ImGui::SliderInt("Segments", &seg, 3, 128))
                        st->prim_segments = static_cast<cardinal::u32>(seg);
                }
                break;
            case PrimitiveKind::Torus:
                ImGui::SliderFloat("Major radius",&st->prim_radius,  0.1f, 25.0f, "%.2f");
                ImGui::SliderFloat("Minor radius",&st->prim_minor,   0.02f, 5.0f, "%.2f");
                {
                    int seg = static_cast<int>(st->prim_segments);
                    if (ImGui::SliderInt("Major segs", &seg, 4, 256))
                        st->prim_segments = static_cast<cardinal::u32>(seg);
                }
                break;
        }
        if (ImGui::Button("Generate", ImVec2(-FLT_MIN, 0))) {
            st->generate_clicked = true;
        }
    }

    if (ImGui::CollapsingHeader("Edit selected", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!selection_has_mesh) {
            ImGui::TextDisabled("(select an entity with a mesh to enable)");
        }
        ImGui::BeginDisabled(!selection_has_mesh);

        // Subdivide
        {
            int n = static_cast<int>(st->subdivide_levels);
            ImGui::SliderInt("Levels##sub", &n, 1, 4);
            st->subdivide_levels = static_cast<cardinal::u32>(n);
            ImGui::SameLine();
            if (ImGui::Button("Subdivide")) st->subdivide_clicked = true;
        }
        // Mirror
        {
            const char* axes[] = { "X", "Y", "Z" };
            int a = static_cast<int>(st->mirror_axis);
            ImGui::Combo("Axis##mirror", &a, axes, IM_ARRAYSIZE(axes));
            st->mirror_axis = static_cast<cardinal::u32>(a);
            ImGui::SameLine();
            if (ImGui::Button("Mirror")) st->mirror_clicked = true;
        }
        // Decimate
        {
            ImGui::SliderFloat("Cell##dec", &st->decimate_cell, 0.01f, 5.0f, "%.3f");
            ImGui::SameLine();
            if (ImGui::Button("Decimate")) st->decimate_clicked = true;
        }
        // Smooth
        {
            int it = static_cast<int>(st->smooth_iterations);
            ImGui::SliderInt("Iters##sm", &it, 1, 10);
            st->smooth_iterations = static_cast<cardinal::u32>(it);
            ImGui::SliderFloat("Lambda##sm", &st->smooth_lambda, 0.0f, 1.0f, "%.2f");
            if (ImGui::Button("Smooth")) st->smooth_clicked = true;
        }
        // Normals
        {
            ImGui::Checkbox("Smooth normals", &st->recompute_normals_smooth);
            ImGui::SameLine();
            if (ImGui::Button("Recompute")) st->recompute_normals_clicked = true;
        }

        ImGui::EndDisabled();
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::mesh_tools_panel
