// =============================================================================
// Cardinal — World grid + streamer (3D).
// =============================================================================
#include <cardinal/world/world.hpp>

#include <cardinal/core/log.hpp>

#include <algorithm>
#include <cmath>

namespace cardinal::world {

// =============================================================================
// WorldGrid
// =============================================================================
WorldGrid::WorldGrid(const WorldGridDesc& d) noexcept : desc_(d) {
    if (desc_.chunk_size         <= 0.0f) desc_.chunk_size         = 1.0f;
    if (desc_.extent_x           <= 0)    desc_.extent_x           = 1;
    if (desc_.extent_y           <= 0)    desc_.extent_y           = 1;
    if (desc_.extent_z           <= 0)    desc_.extent_z           = 1;
    if (desc_.render_distance_xz < 0)     desc_.render_distance_xz = 0;
    if (desc_.render_distance_y  < 0)     desc_.render_distance_y  = 0;
}

void WorldGrid::set_chunk_size(f32 s) noexcept           { desc_.chunk_size = std::max(1e-3f, s); }
void WorldGrid::set_render_distance_xz(i32 d) noexcept   { desc_.render_distance_xz = std::max(0, d); }
void WorldGrid::set_render_distance_y (i32 d) noexcept   { desc_.render_distance_y  = std::max(0, d); }
void WorldGrid::set_extent_x(i32 h) noexcept             { desc_.extent_x = std::max(1, h); }
void WorldGrid::set_extent_y(i32 h) noexcept             { desc_.extent_y = std::max(1, h); }
void WorldGrid::set_extent_z(i32 h) noexcept             { desc_.extent_z = std::max(1, h); }
void WorldGrid::set_extent_uniform(i32 h) noexcept {
    set_extent_x(h); set_extent_y(h); set_extent_z(h);
}

ChunkCoord WorldGrid::chunk_of(f32 wx, f32 wy, f32 wz) const noexcept {
    const f32 inv = 1.0f / desc_.chunk_size;
    return {
        static_cast<i32>(std::floor(wx * inv)),
        static_cast<i32>(std::floor(wy * inv)),
        static_cast<i32>(std::floor(wz * inv)),
    };
}

Vec3 WorldGrid::chunk_world_min(ChunkCoord c) const noexcept {
    return { static_cast<f32>(c.x) * desc_.chunk_size,
             static_cast<f32>(c.y) * desc_.chunk_size,
             static_cast<f32>(c.z) * desc_.chunk_size };
}
Vec3 WorldGrid::chunk_world_max(ChunkCoord c) const noexcept {
    const Vec3 mn = chunk_world_min(c);
    return { mn.x + desc_.chunk_size, mn.y + desc_.chunk_size, mn.z + desc_.chunk_size };
}
Vec3 WorldGrid::chunk_world_center(ChunkCoord c) const noexcept {
    const Vec3 mn = chunk_world_min(c);
    const f32 h = desc_.chunk_size * 0.5f;
    return { mn.x + h, mn.y + h, mn.z + h };
}

bool WorldGrid::in_world_bounds(ChunkCoord c) const noexcept {
    return c.x >= -desc_.extent_x && c.x < desc_.extent_x
        && c.y >= -desc_.extent_y && c.y < desc_.extent_y
        && c.z >= -desc_.extent_z && c.z < desc_.extent_z;
}

void WorldGrid::compute_visible_set(ChunkCoord cam,
                                    std::vector<ChunkCoord>& out) const
{
    out.clear();
    const i32 rxz = desc_.render_distance_xz;
    const i32 ry  = desc_.render_distance_y;
    out.reserve(static_cast<usize>((2*rxz + 1) * (2*rxz + 1) * (2*ry + 1)));

    // Cylindrical region: dx²+dz² ≤ rxz² AND |dy| ≤ ry. For a sphere,
    // set the two radii equal and add the dy² term to the disc check.
    const i32 rxz2 = rxz * rxz;
    for (i32 dy = -ry; dy <= ry; ++dy) {
        for (i32 dz = -rxz; dz <= rxz; ++dz) {
            for (i32 dx = -rxz; dx <= rxz; ++dx) {
                if (dx * dx + dz * dz > rxz2) continue;
                ChunkCoord c{ cam.x + dx, cam.y + dy, cam.z + dz };
                if (!in_world_bounds(c)) continue;
                out.push_back(c);
            }
        }
    }
}

// =============================================================================
// WorldStreamer
// =============================================================================
WorldStreamer::WorldStreamer(WorldGrid& g) noexcept : grid_(g) {
    active_.reserve(256);
}

bool WorldStreamer::is_loaded(ChunkCoord c) const noexcept {
    return active_.find(c) != active_.end();
}

bool WorldStreamer::tick(f32 cx, f32 cy, f32 cz) {
    const ChunkCoord cc = grid_.chunk_of(cx, cy, cz);
    if (!first_tick_ && cc == cam_chunk_ && !dirty_) return false;
    cam_chunk_  = cc;
    first_tick_ = false;
    dirty_      = false;

    grid_.compute_visible_set(cc, visible_scratch_);

    // Build the new active set; diff against existing active_.
    ChunkSet next;
    next.reserve(visible_scratch_.size());
    for (const auto& vc : visible_scratch_) next.insert(vc);

    // Loads first (apps want load-before-unload so neighbour queries
    // during unload still see old chunks if needed).
    for (const auto& c : next) {
        if (active_.find(c) == active_.end()) {
            if (on_load_) on_load_(c);
        }
    }
    // Unloads.
    for (const auto& c : active_) {
        if (next.find(c) == next.end()) {
            if (on_unload_) on_unload_(c);
        }
    }
    active_ = std::move(next);
    return true;
}

void WorldStreamer::evict_all() {
    if (on_unload_) {
        for (const auto& c : active_) on_unload_(c);
    }
    active_.clear();
    first_tick_ = true;
}

}  // namespace cardinal::world
