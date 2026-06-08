// =============================================================================
// Studio — Project panel implementation.
// =============================================================================
#include "project_panel.hpp"

#include <cardinal/project/project.hpp>
#include <cardinal/buildtool/pipeline.hpp>   // BuildCookRun pipeline (Package)

#include <cardinal/ui/imgui.hpp>

#include <cardinal/core/std/cstdio.hpp>
#include <cardinal/core/std/cstring.hpp>

namespace cardinal::ui::panels::project_panel {

Action draw(cardinal::shared_ptr<cardinal::project::Project>* current_project,
            cardinal::project::RecentProjects* recents,
            const char* title, bool* p_open)
{
    Action act{};
    if (!ImGui::Begin(title ? title : "Project", p_open,
                      0)) { ImGui::End(); return act; }

    static char         new_root[1024]   = "G:/CardinalProjects/MyGame";
    static char         new_name[256]    = "MyGame";
    static char         new_author[256]  = "";
    static char         new_engine[1024] = "G:/Cardinal_System_Works";
    static int          template_idx     = 0;
    static char         open_root[1024]  = "G:/CardinalProjects/MyGame";
    static cardinal::string  last_error;

    if (ImGui::CollapsingHeader("New project", ImGuiTreeNodeFlags_DefaultOpen)) {
        const char* templates[] = {
            "Blank", "First Person", "Top-Down", "Cinematic"
        };
        ImGui::Combo("Template", &template_idx, templates, IM_ARRAYSIZE(templates));
        ImGui::TextDisabled("%s", cardinal::project::template_kind_description(
            static_cast<cardinal::project::TemplateKind>(template_idx)));
        ImGui::InputText("Path",   new_root,   sizeof(new_root));
        ImGui::InputText("Name",   new_name,   sizeof(new_name));
        ImGui::InputText("Author", new_author, sizeof(new_author));
        ImGui::InputText("Engine root", new_engine, sizeof(new_engine));
        ImGui::TextDisabled("Path to your Cardinal checkout — baked into the project's build files.");
        if (ImGui::Button("Create project", ImVec2(160.0f, 0.0f))) {
            cardinal::project::InstantiateOptions opts{};
            opts.root = new_root;
            opts.kind = static_cast<cardinal::project::TemplateKind>(template_idx);
            opts.info.name           = new_name;
            opts.info.author         = new_author;
            opts.info.engine_version = "0.1.0";
            opts.info.engine_root    = new_engine;
            cardinal::string err;
            auto p = cardinal::project::instantiate_template(opts, &err);
            if (p) {
                if (current_project) *current_project = p;
                if (recents) { recents->add(opts.root); recents->save(); }
                act.opened_project = p;
                last_error.clear();
            } else {
                last_error = err;
            }
        }
    }

    if (ImGui::CollapsingHeader("Open existing", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::InputText("Path##open", open_root, sizeof(open_root));
        if (ImGui::Button("Open", ImVec2(160.0f, 0.0f))) {
            cardinal::string err;
            auto p = cardinal::project::Project::open(open_root, &err);
            if (p) {
                if (current_project) *current_project = p;
                if (recents) { recents->add(open_root); recents->save(); }
                act.opened_project = p;
                last_error.clear();
            } else {
                last_error = err;
            }
        }
    }

    if (recents && ImGui::CollapsingHeader("Recent projects", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (recents->entries().empty()) {
            ImGui::TextDisabled("(no recent projects)");
        } else {
            cardinal::string to_remove;
            for (const auto& r : recents->entries()) {
                ImGui::PushID(r.c_str());
                if (ImGui::SmallButton("Open")) {
                    cardinal::string err;
                    auto p = cardinal::project::Project::open(r, &err);
                    if (p) {
                        if (current_project) *current_project = p;
                        recents->add(r); recents->save();
                        act.opened_project = p;
                        last_error.clear();
                    } else last_error = err;
                }
                ImGui::SameLine();
                ImGui::Selectable(r.c_str());
                // Right-click a recent → copy its path or drop it from the list.
                if (ImGui::BeginPopupContextItem("##recent_ctx")) {
                    if (ImGui::MenuItem("Copy path")) ImGui::SetClipboardText(r.c_str());
                    if (ImGui::MenuItem("Remove from list")) to_remove = r;
                    ImGui::EndPopup();
                }
                ImGui::PopID();
            }
            if (!to_remove.empty()) { recents->remove(to_remove); recents->save(); }
        }
    }

    if (current_project && *current_project &&
        ImGui::CollapsingHeader("Current project", ImGuiTreeNodeFlags_DefaultOpen))
    {
        auto& info = (*current_project)->info();
        const auto& dirs = (*current_project)->dirs();
        ImGui::Text("Root:    %s", dirs.root.c_str());
        char buf[256];
        cardinal::snprintf(buf, sizeof(buf), "%s", info.name.c_str());
        if (ImGui::InputText("Name##c", buf, sizeof(buf))) info.name = buf;
        cardinal::snprintf(buf, sizeof(buf), "%s", info.author.c_str());
        if (ImGui::InputText("Author##c", buf, sizeof(buf))) info.author = buf;
        cardinal::snprintf(buf, sizeof(buf), "%s", info.description.c_str());
        if (ImGui::InputText("Desc##c", buf, sizeof(buf))) info.description = buf;
        cardinal::snprintf(buf, sizeof(buf), "%s", info.default_pack_name.c_str());
        if (ImGui::InputText("Pack name##c", buf, sizeof(buf))) info.default_pack_name = buf;
        ImGui::Checkbox("Cook on save", &info.cook_on_save);
        ImGui::SameLine();
        ImGui::Checkbox("Pack on cook", &info.pack_on_cook);

        ImGui::Separator();
        if (ImGui::Button("Save manifest", ImVec2(150.0f, 0.0f))) {
            (*current_project)->save();
            act.save_clicked = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Cook all"))      act.cook_clicked = true;
        ImGui::SameLine();
        if (ImGui::Button("Pack"))          act.pack_clicked = true;
        ImGui::SameLine();
        if (ImGui::Button("Cook & Pack"))   act.cook_and_pack_clicked = true;

        // ---- Build system ------------------------------------------------
        ImGui::Separator();
        ImGui::TextDisabled("Build system");
        char ebuf[1024];
        cardinal::snprintf(ebuf, sizeof(ebuf), "%s", info.engine_root.c_str());
        if (ImGui::InputText("Engine root##c", ebuf, sizeof(ebuf))) info.engine_root = ebuf;
        if (ImGui::Button("Generate build files", ImVec2(170.0f, 0.0f))) {
            (*current_project)->save();   // persist engine_root first
            cardinal::string gerr;
            if (cardinal::project::generate_build_files(**current_project, &gerr))
                last_error.clear();
            else
                last_error = gerr;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("Writes CMakeLists.txt + build.bat/.sh + src/main.cpp");

        // ---- Package (BuildCookRun pipeline) -----------------------------
        ImGui::Separator();
        ImGui::TextDisabled("Package (Build \xE2\x86\x92 Cook \xE2\x86\x92 Pack \xE2\x86\x92 Archive)");
        static int  pkg_config  = 1;   // 0 Debug / 1 Development / 2 Shipping
        static bool pkg_archive = false;
        static cardinal::string pkg_status;
        const char* configs[] = { "Debug", "Development", "Shipping" };
        ImGui::SetNextItemWidth(160.0f);
        ImGui::Combo("Configuration", &pkg_config, configs, IM_ARRAYSIZE(configs));
        ImGui::SameLine();
        ImGui::Checkbox("Archive copy", &pkg_archive);
        if (ImGui::Button("Build, Cook, Run", ImVec2(170.0f, 0.0f))) {
            (*current_project)->save();
            cardinal::buildtool::PipelineOptions opt;
            opt.config      = static_cast<cardinal::buildtool::BuildConfig>(pkg_config);
            opt.target      = cardinal::buildtool::BuildTarget::Game;
            opt.do_build    = true;   opt.run_compile = false;   // generate; compile via build.bat
            opt.do_cook     = true;   opt.do_pack     = true;
            opt.do_archive  = pkg_archive;
            auto res = cardinal::buildtool::run_pipeline(**current_project, opt);
            if (res.ok)
                pkg_status = "Packaged \xE2\x86\x92 " + res.artifact_dir;
            else {
                pkg_status = "Package FAILED";
                for (const auto& s : res.stages)
                    if (s.status == cardinal::buildtool::StageStatus::Failed)
                        pkg_status = "Package FAILED at " + s.name + ": " + s.detail;
            }
        }
        ImGui::SameLine();
        ImGui::TextDisabled("Cooks assets + builds a shippable .cpk bundle");
        if (!pkg_status.empty()) {
            const bool failed = pkg_status.find("FAILED") != cardinal::string::npos;
            ImGui::TextColored(failed ? ImVec4(0.95f, 0.4f, 0.4f, 1.0f)
                                      : ImVec4(0.4f, 0.85f, 0.4f, 1.0f),
                               "%s", pkg_status.c_str());
        }

        ImGui::Separator();
        static ImGuiTextFilter asset_filter;
        asset_filter.Draw("Filter assets", 180.0f);
        const auto srcs = (*current_project)->list_source_assets();
        ImGui::TextDisabled("Source assets (%d):", static_cast<int>(srcs.size()));
        for (const auto& s : srcs) {
            if (asset_filter.IsActive() && !asset_filter.PassFilter(s.c_str())) continue;
            ImGui::PushID(s.c_str());
            ImGui::Selectable(s.c_str());
            if (ImGui::BeginPopupContextItem("##src_ctx")) {
                if (ImGui::MenuItem("Copy path")) ImGui::SetClipboardText(s.c_str());
                ImGui::EndPopup();
            }
            ImGui::PopID();
        }
        const auto cooks = (*current_project)->list_cooked_assets();
        ImGui::TextDisabled("Cooked assets (%d):", static_cast<int>(cooks.size()));
        for (const auto& s : cooks) {
            if (asset_filter.IsActive() && !asset_filter.PassFilter(s.c_str())) continue;
            ImGui::PushID(s.c_str());
            ImGui::Selectable(s.c_str());
            if (ImGui::BeginPopupContextItem("##cook_ctx")) {
                if (ImGui::MenuItem("Copy path")) ImGui::SetClipboardText(s.c_str());
                ImGui::EndPopup();
            }
            ImGui::PopID();
        }
    }

    if (!last_error.empty()) {
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.95f, 0.4f, 0.4f, 1.0f),
                           "Error: %s", last_error.c_str());
    }

    ImGui::End();
    return act;
}

}  // namespace cardinal::ui::panels::project_panel
