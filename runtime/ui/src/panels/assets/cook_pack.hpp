#pragma once

// =============================================================================
// Studio — Cook + Pack panel.
//
// Drives the cook + pack pipeline against the active Project. Shows the
// last cook's per-asset results, the pack file's contents, and one-click
// "Cook" / "Pack" / "Cook & Pack" buttons.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/cook/cook.hpp>      // CookResult is held by value in State
#include <cardinal/core/containers.hpp>

namespace cardinal::project { class Project; }
namespace cardinal::pack    { class Archive; }
namespace cardinal::shader  { class Compiler; }

namespace cardinal::ui::panels::cook_pack_panel {

struct State {
    bool                  force_cook{false};
    bool                  open_pack_after_build{true};
    cardinal::string           last_pack_path;
    cardinal::vector<cardinal::cook::CookResult> last_results;
    cardinal::shared_ptr<cardinal::pack::Archive> opened_archive;
};

void draw(cardinal::project::Project* project,
          cardinal::cook::CookerRegistry* registry,
          cardinal::shader::Compiler* shader_compiler,
          State* state,
          const char* title = "Cook & Pack",
          bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::cook_pack_panel
