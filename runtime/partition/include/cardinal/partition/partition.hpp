#pragma once

// =============================================================================
// Cardinal — World Partition System.
//
// Cell-based world streaming smarter than world::WorldStreamer's pure radius:
//
//   * Cells are AABB rectangles in world space (any size, any layout)
//   * Each cell tags itself with a Streaming Strategy:
//       - Always       : never unloads (e.g. skybox, persistent geometry)
//       - Distance     : load when camera within `load_radius`
//       - Vision       : load when ANY active viewer's frustum intersects it
//       - Distance + Vision : either condition triggers load
//   * Multiple Viewers — a Viewer is camera + frustum + position; the
//     partition unions every viewer's needs
//   * Hysteresis — `load_radius` < `unload_radius` so cells don't churn at
//     the boundary
//   * Soft cap — total resident count limited; LRU eviction when over
//
// On per-cell load / unload the system fires callbacks; the integrator
// spawns / despawns actors, allocates / frees streamed assets, etc.
// =============================================================================

#include <cardinal/core/types.hpp>        // function/memory/string
#include <cardinal/core/containers.hpp>   // unordered_map/unordered_set/vector
#include <cardinal/core/geom.hpp>
#include <cardinal/scene/math.hpp>

namespace cardinal::partition {

using CellId = u32;
inline constexpr CellId kInvalidCellId = 0;

enum class StreamMode : u32 {
    Always           = 0,
    Distance         = 1,
    Vision           = 2,
    DistanceOrVision = 3,
};
const char* stream_mode_name(StreamMode m) noexcept;

struct CellDesc {
    cardinal::string         name;             // human-readable
    cardinal::core::geom::AABB bounds{};         // world-space AABB
    StreamMode          mode{StreamMode::Distance};
    f32                 load_radius  {64.0f};   // metres; mode = Distance / DV
    f32                 unload_radius{96.0f};   // hysteresis margin
    u32                 priority{0};      // higher = loads first under cap
};

enum class CellState : u32 {
    Unloaded = 0,
    Loading  = 1,
    Loaded   = 2,
    Unloading= 3,
};
const char* cell_state_name(CellState s) noexcept;

struct Viewer {
    cardinal::scene::Vec3 position{0, 0, 0};
    cardinal::scene::Vec3 forward {0, 0, -1};
    cardinal::scene::Vec3 up      {0, 1, 0};
    cardinal::core::geom::Frustum frustum{};
    f32                   fov_y_rad{60.0f * 0.0174532925f};
    bool                  active{true};
};

struct WorldPartitionDesc {
    u32 max_resident_cells{32};
    f32 default_load_radius  {64.0f};
    f32 default_unload_radius{96.0f};
};

struct WorldPartitionStats {
    u32 cell_count{0};
    u32 loaded{0};
    u32 loading{0};
    u32 unloading{0};
    u64 cells_loaded_total{0};
    u64 cells_unloaded_total{0};
};

class WorldPartition {
public:
    static cardinal::shared_ptr<WorldPartition> create(const WorldPartitionDesc& desc = {});
    ~WorldPartition();

    // ----- Cells -----------------------------------------------------
    CellId   add_cell(const CellDesc& desc);
    bool     remove_cell(CellId id);
    void     clear_cells() noexcept;
    usize    cell_count() const noexcept;
    const CellDesc*  describe(CellId id) const noexcept;
    CellState        state    (CellId id) const noexcept;

    // ----- Viewers ---------------------------------------------------
    u32      add_viewer(const Viewer& v);
    void     remove_viewer(u32 viewer_id);
    void     update_viewer(u32 viewer_id, const Viewer& v);
    usize    viewer_count() const noexcept;

    // ----- Per-frame -------------------------------------------------
    using OnLoad   = cardinal::function<void(CellId, const CellDesc&)>;
    using OnUnload = cardinal::function<void(CellId, const CellDesc&)>;
    void set_on_load  (OnLoad   cb);
    void set_on_unload(OnUnload cb);

    // Recomputes desired set; fires load/unload events as needed.
    void tick();

    // Forces a cell into a state (test / scripting).
    void force_load  (CellId id);
    void force_unload(CellId id);

    // ----- Inspection -------------------------------------------------
    WorldPartitionStats stats() const noexcept;
    cardinal::vector<CellId> loaded_cells() const;

    // For panel rendering.
    struct CellRow {
        CellId          id;
        const CellDesc* desc;
        CellState       state;
        f32             closest_viewer_distance;
        bool            in_any_view;
    };
    cardinal::vector<CellRow> describe_cells() const;

private:
    WorldPartition() = default;
    bool initialize_(const WorldPartitionDesc& desc);

    struct Impl;
    Impl* impl_{nullptr};
};

}  // namespace cardinal::partition
