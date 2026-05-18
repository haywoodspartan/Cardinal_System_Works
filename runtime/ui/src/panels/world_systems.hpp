#pragma once

// =============================================================================
// Studio — combined "World Systems" panel.
//
// Folds the new world-side modules (HAL info, IoDispatcher stats, World
// Partition cells, Level placements, HLOD tree, Mass entities, AI sensors,
// Navmesh stats) into one tabbed panel so the View menu doesn't grow
// linearly with every new system.
// =============================================================================

namespace cardinal::io        { class Dispatcher; }
namespace cardinal::partition { class WorldPartition; }
namespace cardinal::level     { class LevelManager; struct HlodTree; }
namespace cardinal::mass      { class World; }
namespace cardinal::ai        { class PerceptionWorld; }
namespace cardinal::navmesh   { struct Mesh; }

namespace cardinal::ui::panels::world_systems_panel {

struct Inputs {
    cardinal::io::Dispatcher*         io_dispatcher{nullptr};
    cardinal::partition::WorldPartition* partition{nullptr};
    cardinal::level::LevelManager*    level_manager{nullptr};
    const cardinal::level::HlodTree*  hlod_tree{nullptr};
    cardinal::mass::World*            mass_world{nullptr};
    cardinal::ai::PerceptionWorld*    ai_perception{nullptr};
    const cardinal::navmesh::Mesh*    navmesh{nullptr};
};

void draw(const Inputs& in,
          const char* title = "World Systems",
          bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::world_systems_panel
