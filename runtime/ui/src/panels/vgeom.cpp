// =============================================================================
// Cardinal Studio — Virtualized Geometry diagnostic panel impl.
//
// Walks the scene each frame, looking for entities whose mesh has an
// attached vgeom hierarchy. For each match: render a row showing the
// drawn / master triangle ratio published by the renderer + an enable
// toggle. For meshes that aren't attached yet, render an "Attach"
// button so the user can cook them with one click.
// =============================================================================
#include <cardinal/ui/vgeom_panel.hpp>

#include <cardinal/scene/scene.hpp>
#include <cardinal/vgeom/vgeom.hpp>
#include <cardinal/vgeom/cluster.hpp>

#include <imgui.h>

#include <cstdio>
#include <unordered_set>

namespace cardinal::ui {

namespace {

// Format a savings percentage with a colour ramp:
//   < 25 %   = red    (vgeom isn't winning much — camera is right on top of it)
//   < 75 %   = amber  (decent savings)
//   ≥ 75 %   = green  (large savings — what vgeom is for)
ImVec4 savings_colour(float pct) {
    if (pct < 25.0f) return ImVec4(0.85f, 0.30f, 0.30f, 1.0f);
    if (pct < 75.0f) return ImVec4(0.95f, 0.75f, 0.30f, 1.0f);
    return                 ImVec4(0.40f, 0.85f, 0.45f, 1.0f);
}

}  // namespace

void draw_vgeom_panel(cardinal::scene::Scene* scene, bool* p_open) {
    if (!ImGui::Begin("Virtualized Geometry", p_open,
                      ImGuiWindowFlags_AlwaysAutoResize))
    { ImGui::End(); return; }
    if (scene == nullptr) {
        ImGui::TextDisabled("(no Scene bound)");
        ImGui::End();
        return;
    }

    // ----- Walk the scene; classify meshes by attached/visible/etc. ------
    //
    // A given Mesh can be referenced by multiple Entities (instancing
    // in spirit). We dedupe by Mesh* so the table shows one row per
    // unique mesh, not one row per entity.
    struct Row {
        cardinal::scene::Mesh*       mesh{nullptr};
        const cardinal::scene::Entity* sample_entity{nullptr};
        bool                         attached{false};
        bool                         enabled{false};
        cardinal::vgeom::FrameStats  stats{};
        u32                          ref_count{0};
    };

    std::vector<Row> rows;
    std::unordered_set<cardinal::scene::Mesh*> seen;
    rows.reserve(scene->entities().size());
    for (const auto& e : scene->entities()) {
        if (!e.mesh) continue;
        auto* mp = e.mesh.get();
        auto it = seen.find(mp);
        if (it != seen.end()) {
            // Bump the ref count for the existing row.
            for (auto& r : rows) {
                if (r.mesh == mp) { r.ref_count++; break; }
            }
            continue;
        }
        seen.insert(mp);
        Row r{};
        r.mesh          = mp;
        r.sample_entity = &e;
        r.ref_count     = 1;
        r.attached      = cardinal::scene::vgeom_attached(*mp);
        r.enabled       = r.attached && cardinal::scene::vgeom_enabled(*mp);
        if (r.attached) r.stats = cardinal::scene::vgeom_last_stats(*mp);
        rows.push_back(r);
    }

    // ----- Aggregate banner ----------------------------------------------
    u32 total_master = 0;
    u32 total_drawn  = 0;
    u32 attached_n   = 0;
    for (const auto& r : rows) {
        if (!r.attached || !r.enabled) continue;
        total_master += r.stats.master_tri_count;
        total_drawn  += r.stats.drawn_tri_count;
        ++attached_n;
    }
    ImGui::Text("Meshes in scene: %zu     Attached & enabled: %u",
                rows.size(), attached_n);
    if (total_master > 0) {
        const float pct = 100.0f * (1.0f - static_cast<float>(total_drawn) /
                                            static_cast<float>(total_master));
        ImGui::TextColored(savings_colour(pct),
            "Aggregate: drew %u / %u tris  (%.1f%% saved)",
            total_drawn, total_master, pct);
    } else {
        ImGui::TextDisabled("Aggregate: no enabled vgeom meshes — attach one to see savings.");
    }

    ImGui::Separator();

    // ----- Per-mesh table -------------------------------------------------
    if (ImGui::BeginTable("##vgeom_meshes", 7,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
            ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("Mesh",     ImGuiTableColumnFlags_WidthFixed, 220.0f);
        ImGui::TableSetupColumn("Refs",     ImGuiTableColumnFlags_WidthFixed,  44.0f);
        ImGui::TableSetupColumn("Status",   ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("Master",   ImGuiTableColumnFlags_WidthFixed,  80.0f);
        ImGui::TableSetupColumn("Drawn",    ImGuiTableColumnFlags_WidthFixed,  80.0f);
        ImGui::TableSetupColumn("Saved",    ImGuiTableColumnFlags_WidthFixed,  80.0f);
        ImGui::TableSetupColumn("Cut / Culled", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        for (auto& r : rows) {
            ImGui::TableNextRow();
            // Mesh name.
            ImGui::TableSetColumnIndex(0);
            const std::string& mname = r.mesh->name();
            ImGui::Text("%s", mname.empty() ? "(unnamed)" : mname.c_str());
            // Ref count.
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%u", r.ref_count);
            // Status / toggle.
            ImGui::TableSetColumnIndex(2);
            if (!r.attached) {
                // Tag a stable id from the Mesh pointer so multiple meshes
                // don't collide on "Attach".
                ImGui::PushID(r.mesh);
                if (ImGui::Button("Attach")) {
                    cardinal::scene::vgeom_attach(*r.mesh);
                }
                ImGui::PopID();
            } else {
                ImGui::PushID(r.mesh);
                bool en = r.enabled;
                if (ImGui::Checkbox("on", &en)) {
                    cardinal::scene::vgeom_set_enabled(*r.mesh, en);
                }
                ImGui::PopID();
            }
            // Master + drawn + saved.
            if (r.attached && r.enabled && r.stats.master_tri_count > 0) {
                const float pct = 100.0f *
                    (1.0f - static_cast<float>(r.stats.drawn_tri_count) /
                            static_cast<float>(r.stats.master_tri_count));
                ImGui::TableSetColumnIndex(3); ImGui::Text("%u", r.stats.master_tri_count);
                ImGui::TableSetColumnIndex(4); ImGui::Text("%u", r.stats.drawn_tri_count);
                ImGui::TableSetColumnIndex(5);
                ImGui::TextColored(savings_colour(pct), "%.1f%%", pct);
                ImGui::TableSetColumnIndex(6);
                ImGui::Text("%u clusters / %u culled",
                    r.stats.cut_cluster_count, r.stats.culled_cluster_count);
            } else if (r.attached) {
                ImGui::TableSetColumnIndex(3); ImGui::TextDisabled("—");
                ImGui::TableSetColumnIndex(4); ImGui::TextDisabled("—");
                ImGui::TableSetColumnIndex(5); ImGui::TextDisabled("disabled");
                ImGui::TableSetColumnIndex(6); ImGui::TextDisabled("—");
            } else {
                // Not attached — show how big it WOULD be to give the
                // user a hint of whether attaching is worthwhile.
                ImGui::TableSetColumnIndex(3);
                ImGui::TextDisabled("%u", r.mesh->vertex_count() / 3);
                ImGui::TableSetColumnIndex(4); ImGui::TextDisabled("—");
                ImGui::TableSetColumnIndex(5); ImGui::TextDisabled("—");
                ImGui::TableSetColumnIndex(6); ImGui::TextDisabled("not attached");
            }
        }
        ImGui::EndTable();
    }

    ImGui::Separator();
    ImGui::TextDisabled(
        "Master = source mesh tri count. Drawn = tris emitted by the cut\n"
        "this frame (selected clusters after frustum cull). Saved = the\n"
        "fraction of the master count vgeom skipped — bigger is better.");

    ImGui::End();
}

}  // namespace cardinal::ui
