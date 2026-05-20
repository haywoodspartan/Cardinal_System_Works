// =============================================================================
// Studio — Cook + Pack panel implementation.
// =============================================================================
#include "cook_pack.hpp"

#include <cardinal/cook/cook.hpp>
#include <cardinal/pack/pack.hpp>
#include <cardinal/project/project.hpp>
#include <cardinal/shader/shader.hpp>

#include <cardinal/ui/imgui.hpp>

#include <cardinal/core/cstdio.hpp>
#include <cardinal/core/fstream.hpp>

namespace cardinal::ui::panels::cook_pack_panel {

namespace {

ImU32 status_color(cardinal::cook::CookResult::Status s) {
    using S = cardinal::cook::CookResult::Status;
    switch (s) {
        case S::Cooked:  return IM_COL32(110, 220, 110, 255);
        case S::Skipped: return IM_COL32(180, 180, 180, 220);
        case S::Failed:  return IM_COL32(220, 110, 110, 255);
    }
    return IM_COL32_WHITE;
}

const char* status_name(cardinal::cook::CookResult::Status s) {
    using S = cardinal::cook::CookResult::Status;
    switch (s) {
        case S::Cooked:  return "Cooked";
        case S::Skipped: return "Skipped";
        case S::Failed:  return "Failed";
    }
    return "?";
}

bool do_pack(cardinal::project::Project& proj,
             const cardinal::vector<cardinal::cook::CookResult>& results,
             cardinal::string& out_path)
{
    const cardinal::string pack_path = proj.dirs().pack + "/" + proj.info().default_pack_name + ".cpk";
    cardinal::pack::Builder b(pack_path);
    // Walk the project's cooked tree (source-of-truth) and add each.
    for (const auto& r : results) {
        if (r.cooked_relpath.empty()) continue;
        const cardinal::string abs = proj.dirs().cooked + "/" + r.cooked_relpath;
        cardinal::ifstream f(abs, cardinal::ios::binary);
        if (!f) continue;
        f.seekg(0, cardinal::ios::end);
        const auto n = f.tellg();
        // tellg → -1 on non-seekable stream → cast wraps to SIZE_MAX →
        // bad_alloc. Skip the file and continue packing the rest. Same
        // 6a1640d guard pattern as cook/shader read_all.
        if (n < 0) continue;
        f.seekg(0, cardinal::ios::beg);
        cardinal::vector<cardinal::u8> bytes(static_cast<cardinal::usize>(n));
        f.read(reinterpret_cast<char*>(bytes.data()), n);
        // Strip ".cooked" off for the logical key.
        cardinal::string key = r.source_relpath;
        b.add(key, r.type, bytes);
    }
    // Also add any cooked files not in the results (e.g. from a previous run).
    for (const auto& cf : proj.list_cooked_assets()) {
        bool already = false;
        for (const auto& r : results) if (r.cooked_relpath == cf) { already = true; break; }
        if (already) continue;
        const cardinal::string abs = proj.dirs().cooked + "/" + cf;
        cardinal::ifstream f(abs, cardinal::ios::binary);
        if (!f) continue;
        f.seekg(0, cardinal::ios::end);
        const auto n = f.tellg();
        if (n < 0) continue;        // see first loop above
        f.seekg(0, cardinal::ios::beg);
        cardinal::vector<cardinal::u8> bytes(static_cast<cardinal::usize>(n));
        f.read(reinterpret_cast<char*>(bytes.data()), n);
        cardinal::cook::CookedAsset cooked;
        if (!cardinal::cook::CookedAsset::deserialize(bytes, cooked)) continue;
        cardinal::string key = cf;
        if (key.size() > 7 && key.substr(key.size() - 7) == ".cooked") {
            key = key.substr(0, key.size() - 7);
        }
        b.add(key, cooked.type, bytes);
    }
    cardinal::string err;
    if (!b.finalize(&err)) return false;
    out_path = pack_path;
    return true;
}

}  // namespace

void draw(cardinal::project::Project* project,
          cardinal::cook::CookerRegistry* registry,
          cardinal::shader::Compiler* shader_compiler,
          State* state,
          const char* title, bool* p_open)
{
    if (!ImGui::Begin(title ? title : "Cook & Pack", p_open,
                      ImGuiWindowFlags_NoMove)) { ImGui::End(); return; }
    if (project == nullptr || registry == nullptr || state == nullptr) {
        ImGui::TextDisabled("(no project, cooker registry, or state bound)");
        ImGui::End();
        return;
    }

    ImGui::Checkbox("Force re-cook everything (ignore manifest)", &state->force_cook);
    ImGui::Checkbox("Open pack archive after build", &state->open_pack_after_build);

    ImGui::Separator();
    if (ImGui::Button("Cook all", ImVec2(140.0f, 0.0f))) {
        cardinal::cook::CookContext ctx{};
        ctx.shader_compiler = shader_compiler;
        cardinal::cook::cook_all(*project, *registry,
            state->last_results, state->force_cook, ctx);
    }
    ImGui::SameLine();
    const bool can_pack = !state->last_results.empty()
        || !project->list_cooked_assets().empty();
    ImGui::BeginDisabled(!can_pack);
    if (ImGui::Button("Pack", ImVec2(120.0f, 0.0f))) {
        cardinal::string p;
        if (do_pack(*project, state->last_results, p)) {
            state->last_pack_path = p;
            if (state->open_pack_after_build) {
                state->opened_archive = cardinal::pack::Archive::open(p);
            }
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Cook & Pack", ImVec2(140.0f, 0.0f))) {
        cardinal::cook::CookContext ctx{};
        ctx.shader_compiler = shader_compiler;
        cardinal::cook::cook_all(*project, *registry,
            state->last_results, state->force_cook, ctx);
        cardinal::string p;
        if (do_pack(*project, state->last_results, p)) {
            state->last_pack_path = p;
            if (state->open_pack_after_build) {
                state->opened_archive = cardinal::pack::Archive::open(p);
            }
        }
    }
    ImGui::EndDisabled();

    ImGui::Separator();

    if (ImGui::CollapsingHeader("Last cook results", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (state->last_results.empty()) {
            ImGui::TextDisabled("(none — hit Cook all)");
        } else if (ImGui::BeginTable("##cook_results", 5,
                       ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                       ImGuiTableFlags_SizingStretchProp))
        {
            ImGui::TableSetupColumn("status", ImGuiTableColumnFlags_WidthFixed, 70.0f);
            ImGui::TableSetupColumn("source", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("type",   ImGuiTableColumnFlags_WidthFixed, 80.0f);
            ImGui::TableSetupColumn("bytes",  ImGuiTableColumnFlags_WidthFixed, 80.0f);
            ImGui::TableSetupColumn("notes",  ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();
            for (const auto& r : state->last_results) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::PushStyleColor(ImGuiCol_Text, status_color(r.status));
                ImGui::TextUnformatted(status_name(r.status));
                ImGui::PopStyleColor();
                ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted(r.source_relpath.c_str());
                ImGui::TableSetColumnIndex(2);
                ImGui::TextUnformatted(cardinal::cook::asset_type_name(r.type));
                ImGui::TableSetColumnIndex(3); ImGui::Text("%u", r.payload_bytes);
                ImGui::TableSetColumnIndex(4);
                if (r.status == cardinal::cook::CookResult::Status::Failed)
                    ImGui::TextColored(ImVec4(0.95f, 0.4f, 0.4f, 1.0f),
                                       "%s", r.error_message.c_str());
                else
                    ImGui::TextDisabled("%s", r.diagnostics.c_str());
            }
            ImGui::EndTable();
        }
    }

    if (ImGui::CollapsingHeader("Open pack archive", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (state->opened_archive == nullptr) {
            ImGui::TextDisabled("(no pack open)");
        } else {
            const auto stats = state->opened_archive->stats();
            ImGui::Text("Path:        %s", state->opened_archive->path().c_str());
            ImGui::Text("Entries:     %u", stats.entry_count);
            ImGui::Text("File size:   %.2f KiB",
                static_cast<double>(stats.file_size) / 1024.0);
            ImGui::Text("Streamed:    %llu bytes (%llu in flight)",
                static_cast<unsigned long long>(stats.bytes_streamed),
                static_cast<unsigned long long>(stats.streams_in_flight));
            ImGui::Separator();
            if (ImGui::BeginTable("##pack_entries", 4,
                    ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                    ImGuiTableFlags_SizingStretchProp))
            {
                ImGui::TableSetupColumn("key",    ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("type",   ImGuiTableColumnFlags_WidthFixed,  80.0f);
                ImGui::TableSetupColumn("offset", ImGuiTableColumnFlags_WidthFixed, 110.0f);
                ImGui::TableSetupColumn("size",   ImGuiTableColumnFlags_WidthFixed,  90.0f);
                ImGui::TableHeadersRow();
                for (const auto& e : state->opened_archive->entries()) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(e.key.c_str());
                    ImGui::TableSetColumnIndex(1);
                    ImGui::TextUnformatted(cardinal::cook::asset_type_name(e.type));
                    ImGui::TableSetColumnIndex(2); ImGui::Text("%llu",
                        static_cast<unsigned long long>(e.offset));
                    ImGui::TableSetColumnIndex(3); ImGui::Text("%llu",
                        static_cast<unsigned long long>(e.size));
                }
                ImGui::EndTable();
            }
        }
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::cook_pack_panel
