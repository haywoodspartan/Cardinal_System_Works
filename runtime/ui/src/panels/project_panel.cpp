// =============================================================================
// Studio — Project panel implementation.
// =============================================================================
#include "project_panel.hpp"

#include <cardinal/project/project.hpp>

#include <cardinal/ui/imgui.hpp>

#include <cardinal/core/cstdio.hpp>
#include <cardinal/core/cstring.hpp>

namespace cardinal::ui::panels::project_panel {

Action draw(cardinal::shared_ptr<cardinal::project::Project>* current_project,
            cardinal::project::RecentProjects* recents,
            const char* title, bool* p_open)
{
    Action act{};
    if (!ImGui::Begin(title ? title : "Project", p_open)) { ImGui::End(); return act; }

    static char         new_root[1024]   = "G:/CardinalProjects/MyGame";
    static char         new_name[256]    = "MyGame";
    static char         new_author[256]  = "";
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
        if (ImGui::Button("Create project", ImVec2(160.0f, 0.0f))) {
            cardinal::project::InstantiateOptions opts{};
            opts.root = new_root;
            opts.kind = static_cast<cardinal::project::TemplateKind>(template_idx);
            opts.info.name           = new_name;
            opts.info.author         = new_author;
            opts.info.engine_version = "0.1.0";
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
        } else for (const auto& r : recents->entries()) {
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
            ImGui::TextUnformatted(r.c_str());
            ImGui::PopID();
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

        ImGui::Separator();
        ImGui::TextDisabled("Source assets:");
        const auto srcs = (*current_project)->list_source_assets();
        for (const auto& s : srcs) ImGui::Text("  %s", s.c_str());
        ImGui::TextDisabled("Cooked assets:");
        const auto cooks = (*current_project)->list_cooked_assets();
        for (const auto& s : cooks) ImGui::Text("  %s", s.c_str());
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
