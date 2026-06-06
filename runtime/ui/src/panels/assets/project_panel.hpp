#pragma once

// =============================================================================
// Studio — Project panel.
//
// Create / open / save Cardinal projects. Pick a template, browse recent
// projects, edit the open project's metadata.
// =============================================================================

#include <cardinal/core/types.hpp>

namespace cardinal::project {
    class Project;
    class RecentProjects;
}

namespace cardinal::ui::panels::project_panel {

// Returned per draw — host reads these to know what the user clicked.
struct Action {
    cardinal::shared_ptr<cardinal::project::Project> opened_project;   // non-null when OPEN/CREATE happened
    bool        save_clicked{false};
    bool        cook_clicked{false};
    bool        pack_clicked{false};
    bool        cook_and_pack_clicked{false};
};

Action draw(cardinal::shared_ptr<cardinal::project::Project>* current_project,
            cardinal::project::RecentProjects* recents,
            const char* title = "Project",
            bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::project_panel
