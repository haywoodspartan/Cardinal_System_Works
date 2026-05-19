// =============================================================================
// Studio — World Systems panel implementation.
// =============================================================================
#include "world_systems.hpp"

#include <cardinal/ai/ai.hpp>
#include <cardinal/core/hal.hpp>
#include <cardinal/core/io.hpp>
#include <cardinal/level/level.hpp>
#include <cardinal/mass/mass.hpp>
#include <cardinal/navmesh/navmesh.hpp>
#include <cardinal/partition/partition.hpp>

#include <cardinal/ui/imgui.hpp>

#include <cardinal/core/cstdio.hpp>

namespace cardinal::ui::panels::world_systems_panel {

namespace {

void tab_hal() {
    const auto& os = cardinal::hal::os_info();
    ImGui::Text("OS:         %s %s", os.name.c_str(), os.kernel.c_str());
    ImGui::Text("Host:       %s (user %s)", os.host_name.c_str(), os.user_name.c_str());
    ImGui::Text("CPUs:       %u logical / %u physical",
                os.logical_cpu_count, os.physical_cpu_count);
    ImGui::Text("RAM:        %.1f GiB",
                static_cast<double>(os.total_ram_bytes) / (1024.0*1024.0*1024.0));
    ImGui::Text("Page size:  %u bytes", os.page_size_bytes);
    const auto& cf = cardinal::hal::cpu_features();
    ImGui::Text("Vendor:     %s", cf.vendor[0] ? cf.vendor : "(unknown)");
    ImGui::Text("Brand:      %s", cf.brand[0]  ? cf.brand  : "(unknown)");
    ImGui::Text("Features:   %s%s%s%s%s%s%s%s%s",
        cf.sse2  ? "SSE2 "    : "",
        cf.sse42 ? "SSE4.2 "  : "",
        cf.avx   ? "AVX "     : "",
        cf.avx2  ? "AVX2 "    : "",
        cf.avx512f ? "AVX512 ": "",
        cf.fma3  ? "FMA3 "    : "",
        cf.bmi1  ? "BMI1 "    : "",
        cf.bmi2  ? "BMI2 "    : "",
        cf.aes   ? "AES "     : "");
    const auto mem = cardinal::hal::process_memory();
    ImGui::Separator();
    ImGui::Text("Process working set: %.1f MiB",
        static_cast<double>(mem.working_set_bytes) / (1024.0 * 1024.0));
    ImGui::Text("Peak working set:    %.1f MiB",
        static_cast<double>(mem.peak_working_set_bytes) / (1024.0 * 1024.0));
    ImGui::Text("Private commit:      %.1f MiB",
        static_cast<double>(mem.private_bytes) / (1024.0 * 1024.0));
}

void tab_io(cardinal::io::Dispatcher* d) {
    if (d == nullptr) { ImGui::TextDisabled("(no dispatcher bound)"); return; }
    const auto s = d->stats();
    ImGui::Text("In flight:    %u | Queued: %u", s.in_flight_total, s.queued_total);
    ImGui::Text("Last tick:    %.3f ms",         s.last_tick_ms);
    ImGui::Text("Bytes in flight: %llu  done: %llu",
        static_cast<unsigned long long>(s.bytes_in_flight),
        static_cast<unsigned long long>(s.bytes_completed));
    ImGui::Text("Requests: %llu seen / %llu done / %llu failed",
        static_cast<unsigned long long>(s.requests_seen),
        static_cast<unsigned long long>(s.requests_completed),
        static_cast<unsigned long long>(s.requests_failed));
    if (ImGui::BeginTable("##io_per_pri", 3,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("priority"); ImGui::TableSetupColumn("queued");
        ImGui::TableSetupColumn("in flight"); ImGui::TableHeadersRow();
        for (cardinal::u32 i = 0; i < 5; ++i) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(cardinal::io::priority_name(
                static_cast<cardinal::io::Priority>(i)));
            ImGui::TableSetColumnIndex(1); ImGui::Text("%u", s.queued[i]);
            ImGui::TableSetColumnIndex(2); ImGui::Text("%u", s.in_flight[i]);
        }
        ImGui::EndTable();
    }
}

void tab_partition(cardinal::partition::WorldPartition* p) {
    if (p == nullptr) { ImGui::TextDisabled("(no partition bound)"); return; }
    const auto s = p->stats();
    ImGui::Text("Cells: %u total | %u loaded | %u loading | %u unloading",
                s.cell_count, s.loaded, s.loading, s.unloading);
    ImGui::Text("Lifetime: %llu loaded / %llu unloaded",
        static_cast<unsigned long long>(s.cells_loaded_total),
        static_cast<unsigned long long>(s.cells_unloaded_total));
    if (ImGui::BeginTable("##cells", 5,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
            ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("id",     ImGuiTableColumnFlags_WidthFixed, 50.0f);
        ImGui::TableSetupColumn("name",   ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("mode",   ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableSetupColumn("state",  ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableSetupColumn("dist",   ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableHeadersRow();
        for (const auto& r : p->describe_cells()) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("%u", r.id);
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(r.desc ? r.desc->name.c_str() : "?");
            ImGui::TableSetColumnIndex(2);
            ImGui::TextUnformatted(cardinal::partition::stream_mode_name(
                r.desc ? r.desc->mode : cardinal::partition::StreamMode::Always));
            ImGui::TableSetColumnIndex(3);
            ImVec4 col(0.7f, 0.7f, 0.7f, 1.0f);
            if (r.state == cardinal::partition::CellState::Loaded)
                col = {0.40f, 0.95f, 0.50f, 1.0f};
            else if (r.state == cardinal::partition::CellState::Loading)
                col = {0.95f, 0.85f, 0.30f, 1.0f};
            ImGui::TextColored(col, "%s",
                cardinal::partition::cell_state_name(r.state));
            ImGui::TableSetColumnIndex(4);
            ImGui::Text("%.1fm", r.closest_viewer_distance);
        }
        ImGui::EndTable();
    }
}

void tab_level(cardinal::level::LevelManager* lm,
                const cardinal::level::HlodTree* tree) {
    if (lm == nullptr) { ImGui::TextDisabled("(no level manager)"); }
    else {
        ImGui::Text("Placements: %zu", lm->placement_count());
        for (const auto* p : lm->placements()) {
            char hdr[80];
            cardinal::snprintf(hdr, sizeof(hdr), "[%u] %s (%zu actors)",
                p->id,
                p->instance ? p->instance->desc().name.c_str() : "?",
                p->spawned_actor_ids.size());
            if (ImGui::TreeNode(hdr)) {
                ImGui::Text("Translation: (%.2f, %.2f, %.2f)",
                    p->translation.x, p->translation.y, p->translation.z);
                ImGui::Text("Scale:       (%.2f, %.2f, %.2f)",
                    p->scale.x, p->scale.y, p->scale.z);
                ImGui::TreePop();
            }
        }
    }
    ImGui::Separator();
    ImGui::Text("HLOD tree");
    if (tree == nullptr || tree->nodes.empty()) {
        ImGui::TextDisabled("(no tree)");
    } else {
        u32 leaves = 0, internals = 0;
        for (const auto& n : tree->nodes) (n.is_leaf ? ++leaves : ++internals);
        ImGui::Text("Nodes: %zu | leaves: %u | internals: %u | root: %u",
            tree->nodes.size(), leaves, internals, tree->root);
    }
}

void tab_mass(cardinal::mass::World* w) {
    if (w == nullptr) { ImGui::TextDisabled("(no mass world bound)"); return; }
    const auto s = w->stats();
    ImGui::Text("Entities:    %u (lifetime created %llu / destroyed %llu)",
        s.entities,
        static_cast<unsigned long long>(s.entities_created_total),
        static_cast<unsigned long long>(s.entities_destroyed_total));
    ImGui::Text("Archetypes:  %u | Chunks: %u | Components: %u",
        s.archetype_count, s.chunk_count, s.component_count);
    if (ImGui::BeginTable("##components", 3,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
            ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("bit",   ImGuiTableColumnFlags_WidthFixed, 40.0f);
        ImGui::TableSetupColumn("name",  ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("size",  ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableHeadersRow();
        for (cardinal::u32 b = 0; b < s.component_count; ++b) {
            const auto* d = w->describe_component(b);
            if (d == nullptr) continue;
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("%u", d->bit_index);
            ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted(d->name.c_str());
            ImGui::TableSetColumnIndex(2); ImGui::Text("%u B", d->size);
        }
        ImGui::EndTable();
    }
}

void tab_ai(cardinal::ai::PerceptionWorld* p) {
    if (p == nullptr) { ImGui::TextDisabled("(no perception world)"); return; }
    ImGui::Text("Sensors:   %zu", p->sensor_count());
    ImGui::Text("Stimuli:   %zu", p->stimulus_count());
    ImGui::Text("Last events: %zu", p->last_events().size());
    if (ImGui::BeginTable("##events", 4,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
            ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("sensor"); ImGui::TableSetupColumn("kind");
        ImGui::TableSetupColumn("dist");   ImGui::TableSetupColumn("pos");
        ImGui::TableHeadersRow();
        for (const auto& e : p->last_events()) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("%u", e.sensor);
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(e.kind == cardinal::ai::StimulusKind::Sight ? "Sight" : "Sound");
            ImGui::TableSetColumnIndex(2); ImGui::Text("%.2fm", e.distance);
            ImGui::TableSetColumnIndex(3); ImGui::Text("(%.1f, %.1f, %.1f)",
                e.position.x, e.position.y, e.position.z);
        }
        ImGui::EndTable();
    }
}

void tab_navmesh(const cardinal::navmesh::Mesh* m) {
    if (m == nullptr || m->empty()) { ImGui::TextDisabled("(no navmesh)"); return; }
    ImGui::Text("Vertices:  %zu", m->vertices.size());
    ImGui::Text("Polygons:  %zu", m->polys.size());
    u32 walls = 0;
    for (const auto& p : m->polys) {
        for (auto n : p.neighbours) if (n == cardinal::navmesh::kInvalidPoly) ++walls;
    }
    ImGui::Text("Boundary edges: %u", walls);
}

}  // namespace

void draw(const Inputs& in, const char* title, bool* p_open) {
    if (!ImGui::Begin(title ? title : "World Systems", p_open,
                      ImGuiWindowFlags_NoMove)) { ImGui::End(); return; }
    if (ImGui::BeginTabBar("##ws_tabs")) {
        if (ImGui::BeginTabItem("HAL"))        { tab_hal();                            ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("IO"))         { tab_io(in.io_dispatcher);             ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Partition"))  { tab_partition(in.partition);          ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Level/HLOD")) { tab_level(in.level_manager, in.hlod_tree); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Mass ECS"))   { tab_mass(in.mass_world);              ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("AI"))         { tab_ai(in.ai_perception);             ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Navmesh"))    { tab_navmesh(in.navmesh);              ImGui::EndTabItem(); }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

}  // namespace cardinal::ui::panels::world_systems_panel
