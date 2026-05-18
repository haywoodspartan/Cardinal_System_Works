#pragma once

// =============================================================================
// Cardinal — World grid + chunk streamer (3D).
//
// Splits the world into a uniform XYZ grid of chunks. Each chunk has integer
// (i32 x, i32 y, i32 z) coordinates and covers `chunk_size` world units per
// side. Three distinct extents per axis let games be very wide horizontally
// AND very deep vertically — useful for voxel terrain, sky islands, deep
// dungeons, etc. Each axis ranges over the full signed-int32 domain (i.e.
// roughly ±2.1 × 10⁹ chunks, which at the default 32u chunk size is about
// ±68 × 10⁹ world units per side — far past anything practical).
//
// 0,0,0 is the world centre, NOT an edge or limit. Chunks left of 0 are
// negative; chunks below ground are negative on Y.
//
// Three primary types:
//
//   ChunkCoord    — (x, y, z) integer chunk index. ChunkCoordHash makes
//                   it a usable key for unordered_set / unordered_map.
//
//   WorldGrid     — config + math: world ↔ chunk coord conversion, AABB
//                   helpers, "is in bounds", visible-set computation.
//
//   WorldStreamer — runtime updater. tick(camera_x/y/z) diffs the previous
//                   visible set against the new one and fires `on_load`
//                   for chunks that just entered radius and `on_unload`
//                   for ones that just left.
//
// The visible set has a cylindrical shape by default — XZ disc of radius
// `render_distance_xz` plus Y slab of half-height `render_distance_y`.
// That matches typical game worlds (large horizontal play area, fewer
// chunks above/below) and keeps the active count manageable. For pure
// voxel scenes set both radii equal to get a sphere.
// =============================================================================

#include <cardinal/core/math.hpp>
#include <cardinal/core/spatial.hpp>
#include <cardinal/core/types.hpp>        // function/memory
#include <cardinal/core/utility.hpp>      // cardinal::move
#include <cardinal/core/containers.hpp>   // unordered_set/vector

namespace cardinal::world {

// Canonical types — single definitions live in cardinal::core. The
// world module previously rolled its own ChunkCoord + ChunkCoordHash +
// Vec3 — aliasing here means scene::Scene + cardinal::world::WorldStreamer
// + physics::raycast all speak the exact same struct.
using Vec3           = cardinal::core::Vec3;
using ChunkCoord     = cardinal::core::ChunkCoord;
using ChunkCoordHash = cardinal::core::ChunkCoordHash;
using ChunkSet       = cardinal::core::ChunkSet;

// Configuration for a WorldGrid.
//
//   chunk_size    : world units per chunk side (uniform XYZ).
//   extent_x/y/z  : half-extent per axis. The world spans
//                   [-extent_a, +extent_a) chunks on each axis. INT32_MAX/2
//                   gives effectively unbounded worlds (the streamer never
//                   trips the bounds check). Defaults: x = z = 1024 (64 km
//                   side at 32 u chunks), y = 64 (4 km vertical) — plenty
//                   for most worlds without being silly.
//   render_distance_xz : horizontal load radius, in chunks. The visible
//                        region forms a XZ disc of this radius around the
//                        camera's chunk.
//   render_distance_y  : vertical load half-height, in chunks. Combined
//                        with render_distance_xz this gives a cylinder.
//                        Set equal to render_distance_xz for a true sphere.
struct WorldGridDesc {
    f32 chunk_size{32.0f};
    i32 extent_x{1024};
    i32 extent_y{64};
    i32 extent_z{1024};
    i32 render_distance_xz{8};
    i32 render_distance_y{4};
};

// =============================================================================
// WorldGrid — pure math + config. No state besides the desc fields.
// =============================================================================
class WorldGrid {
public:
    explicit WorldGrid(const WorldGridDesc& d = {}) noexcept;

    // ----- Config (mutable at runtime). -----
    void set_chunk_size       (f32 s) noexcept;
    void set_render_distance_xz(i32 d) noexcept;
    void set_render_distance_y (i32 d) noexcept;
    void set_extent_x         (i32 half_chunks) noexcept;
    void set_extent_y         (i32 half_chunks) noexcept;
    void set_extent_z         (i32 half_chunks) noexcept;
    // Convenience: set all three extents at once.
    void set_extent_uniform   (i32 half_chunks) noexcept;

    f32  chunk_size       () const noexcept { return desc_.chunk_size; }
    i32  render_distance_xz() const noexcept { return desc_.render_distance_xz; }
    i32  render_distance_y () const noexcept { return desc_.render_distance_y; }
    i32  extent_x         () const noexcept { return desc_.extent_x; }
    i32  extent_y         () const noexcept { return desc_.extent_y; }
    i32  extent_z         () const noexcept { return desc_.extent_z; }
    const WorldGridDesc& desc() const noexcept { return desc_; }

    // ----- Coordinate helpers. -----
    ChunkCoord chunk_of(f32 wx, f32 wy, f32 wz) const noexcept;
    Vec3       chunk_world_min   (ChunkCoord c) const noexcept;
    Vec3       chunk_world_max   (ChunkCoord c) const noexcept;
    Vec3       chunk_world_center(ChunkCoord c) const noexcept;
    bool       in_world_bounds   (ChunkCoord c) const noexcept;

    // World footprint reporting. Total side per axis = 2 × extent × chunk_size.
    f32 world_size_x_units() const noexcept { return desc_.chunk_size * static_cast<f32>(desc_.extent_x * 2); }
    f32 world_size_y_units() const noexcept { return desc_.chunk_size * static_cast<f32>(desc_.extent_y * 2); }
    f32 world_size_z_units() const noexcept { return desc_.chunk_size * static_cast<f32>(desc_.extent_z * 2); }
    // Total chunk count (i64 because the product blows past u32 quickly).
    i64 total_chunk_count() const noexcept {
        const i64 sx = static_cast<i64>(desc_.extent_x) * 2;
        const i64 sy = static_cast<i64>(desc_.extent_y) * 2;
        const i64 sz = static_cast<i64>(desc_.extent_z) * 2;
        return sx * sy * sz;
    }

    // Build the visible chunk set around `camera_chunk` at the current
    // render distances. Cylindrical region: dx²+dz² ≤ r_xz² AND |dy| ≤ r_y.
    // Chunks outside world bounds are filtered out. `out_visible` is cleared
    // before the call.
    void compute_visible_set(ChunkCoord camera_chunk,
                             cardinal::vector<ChunkCoord>& out_visible) const;

private:
    WorldGridDesc desc_{};
};

// =============================================================================
// WorldStreamer — diffs the visible set frame-over-frame and fires load /
// unload callbacks. Single-threaded.
// =============================================================================
class WorldStreamer {
public:
    using OnChunkLoad   = cardinal::function<void(ChunkCoord)>;
    using OnChunkUnload = cardinal::function<void(ChunkCoord)>;

    explicit WorldStreamer(WorldGrid& grid) noexcept;

    void set_on_load  (OnChunkLoad cb)   noexcept { on_load_   = cardinal::move(cb); }
    void set_on_unload(OnChunkUnload cb) noexcept { on_unload_ = cardinal::move(cb); }

    // Call once per frame with the camera's full 3D world position.
    // Recomputes the active chunk set; fires `on_load` for new chunks
    // + `on_unload` for chunks that just left. Returns true iff the
    // active set changed this tick.
    bool tick(f32 camera_x, f32 camera_y, f32 camera_z);

    // Read-only views of the current state.
    ChunkCoord       camera_chunk () const noexcept { return cam_chunk_; }
    const ChunkSet&  active       () const noexcept { return active_;    }
    bool             is_loaded    (ChunkCoord c) const noexcept;
    usize            active_count () const noexcept { return active_.size(); }

    // Force-evict every active chunk (fires on_unload for each, then clears).
    void evict_all();

    // Force the next tick to re-evaluate even if camera_chunk didn't move.
    // Use after WorldGrid::set_render_distance_* / chunk_size / extent changes.
    void invalidate() noexcept { dirty_ = true; }

private:
    WorldGrid&             grid_;
    OnChunkLoad            on_load_{};
    OnChunkUnload          on_unload_{};
    ChunkCoord             cam_chunk_{};
    bool                   first_tick_{true};
    bool                   dirty_{false};
    ChunkSet               active_;
    cardinal::vector<ChunkCoord> visible_scratch_;
};

}  // namespace cardinal::world
