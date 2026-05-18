#include <cardinal/partition/partition.hpp>

#include <cardinal/core/log.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace cardinal::partition {

const char* stream_mode_name(StreamMode m) noexcept {
    switch (m) {
        case StreamMode::Always:           return "Always";
        case StreamMode::Distance:         return "Distance";
        case StreamMode::Vision:           return "Vision";
        case StreamMode::DistanceOrVision: return "Dist+Vision";
    }
    return "?";
}
const char* cell_state_name(CellState s) noexcept {
    switch (s) {
        case CellState::Unloaded:  return "Unloaded";
        case CellState::Loading:   return "Loading";
        case CellState::Loaded:    return "Loaded";
        case CellState::Unloading: return "Unloading";
    }
    return "?";
}

struct WorldPartition::Impl {
    WorldPartitionDesc desc{};
    struct Entry {
        CellId    id;
        CellDesc  desc;
        CellState state{CellState::Unloaded};
        u64       last_loaded_ms{0};
    };
    std::unordered_map<CellId, Entry> cells;
    std::unordered_map<u32, Viewer>   viewers;
    OnLoad   on_load{};
    OnUnload on_unload{};
    CellId   next_cell_id{1};
    u32      next_viewer_id{1};
    u64      cells_loaded_total{0};
    u64      cells_unloaded_total{0};
};

std::shared_ptr<WorldPartition> WorldPartition::create(const WorldPartitionDesc& desc) {
    auto p = std::shared_ptr<WorldPartition>(new WorldPartition());
    if (!p->initialize_(desc)) return nullptr;
    return p;
}
WorldPartition::~WorldPartition() { delete impl_; }

bool WorldPartition::initialize_(const WorldPartitionDesc& desc) {
    impl_ = new Impl{};
    impl_->desc = desc;
    cardinal::log::infof("partition",
        "WorldPartition online — cap %u resident, default load=%.1fm unload=%.1fm",
        desc.max_resident_cells, desc.default_load_radius, desc.default_unload_radius);
    return true;
}

CellId WorldPartition::add_cell(const CellDesc& desc) {
    Impl::Entry e{};
    e.id   = impl_->next_cell_id++;
    e.desc = desc;
    if (e.desc.load_radius   <= 0.0f) e.desc.load_radius   = impl_->desc.default_load_radius;
    if (e.desc.unload_radius <= e.desc.load_radius)
        e.desc.unload_radius = e.desc.load_radius * 1.5f;
    impl_->cells[e.id] = e;
    return e.id;
}

bool WorldPartition::remove_cell(CellId id) {
    auto it = impl_->cells.find(id);
    if (it == impl_->cells.end()) return false;
    if (it->second.state == CellState::Loaded && impl_->on_unload) {
        impl_->on_unload(id, it->second.desc);
    }
    impl_->cells.erase(it);
    return true;
}

void WorldPartition::clear_cells() noexcept {
    if (impl_->on_unload) {
        for (const auto& [id, e] : impl_->cells) {
            if (e.state == CellState::Loaded) impl_->on_unload(id, e.desc);
        }
    }
    impl_->cells.clear();
}

usize WorldPartition::cell_count() const noexcept { return impl_->cells.size(); }
const CellDesc* WorldPartition::describe(CellId id) const noexcept {
    auto it = impl_->cells.find(id);
    return it == impl_->cells.end() ? nullptr : &it->second.desc;
}
CellState WorldPartition::state(CellId id) const noexcept {
    auto it = impl_->cells.find(id);
    return it == impl_->cells.end() ? CellState::Unloaded : it->second.state;
}

u32 WorldPartition::add_viewer(const Viewer& v) {
    const u32 id = impl_->next_viewer_id++;
    impl_->viewers[id] = v;
    return id;
}
void WorldPartition::remove_viewer(u32 id) { impl_->viewers.erase(id); }
void WorldPartition::update_viewer(u32 id, const Viewer& v) {
    auto it = impl_->viewers.find(id);
    if (it != impl_->viewers.end()) it->second = v;
}
usize WorldPartition::viewer_count() const noexcept { return impl_->viewers.size(); }

void WorldPartition::set_on_load  (OnLoad   cb) { impl_->on_load   = std::move(cb); }
void WorldPartition::set_on_unload(OnUnload cb) { impl_->on_unload = std::move(cb); }

namespace {

f32 closest_dist_to_aabb(const cardinal::scene::Vec3& p, const cardinal::core::geom::AABB& a) {
    const f32 dx = std::max({a.min.x - p.x, 0.0f, p.x - a.max.x});
    const f32 dy = std::max({a.min.y - p.y, 0.0f, p.y - a.max.y});
    const f32 dz = std::max({a.min.z - p.z, 0.0f, p.z - a.max.z});
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

}  // namespace

void WorldPartition::tick() {
    // For each cell, decide WANT_LOADED based on every viewer.
    std::vector<CellId> want_load, want_unload;
    for (auto& [id, e] : impl_->cells) {
        bool wants = false;
        if (e.desc.mode == StreamMode::Always) wants = true;
        for (const auto& [_, v] : impl_->viewers) {
            if (!v.active) continue;
            const f32 d = closest_dist_to_aabb(v.position, e.desc.bounds);
            const bool dist_ok = (e.desc.mode == StreamMode::Distance ||
                                  e.desc.mode == StreamMode::DistanceOrVision) &&
                                 d <= (e.state == CellState::Loaded
                                       ? e.desc.unload_radius : e.desc.load_radius);
            const bool vis_ok  = (e.desc.mode == StreamMode::Vision ||
                                  e.desc.mode == StreamMode::DistanceOrVision) &&
                                 v.frustum.intersects(e.desc.bounds);
            if (dist_ok || vis_ok) { wants = true; break; }
        }
        if (wants && e.state != CellState::Loaded)   want_load  .push_back(id);
        if (!wants && e.state == CellState::Loaded)  want_unload.push_back(id);
    }

    // Cap residency.
    u32 currently_loaded = 0;
    for (const auto& [_, e] : impl_->cells) if (e.state == CellState::Loaded) ++currently_loaded;
    const u32 cap = impl_->desc.max_resident_cells;

    // Sort want_load by priority desc so the most important come first.
    std::sort(want_load.begin(), want_load.end(),
        [this](CellId a, CellId b) {
            return impl_->cells[a].desc.priority > impl_->cells[b].desc.priority;
        });

    // If we'll exceed cap, also evict lowest-priority current cells.
    if (cap > 0 &&
        currently_loaded + want_load.size() > cap + want_unload.size())
    {
        std::vector<CellId> evictable;
        for (const auto& [id, e] : impl_->cells) {
            if (e.state == CellState::Loaded &&
                e.desc.mode != StreamMode::Always &&
                std::find(want_unload.begin(), want_unload.end(), id) == want_unload.end()) {
                evictable.push_back(id);
            }
        }
        std::sort(evictable.begin(), evictable.end(),
            [this](CellId a, CellId b) {
                return impl_->cells[a].desc.priority < impl_->cells[b].desc.priority;
            });
        const i64 over = static_cast<i64>(currently_loaded) +
                         static_cast<i64>(want_load.size()) -
                         static_cast<i64>(want_unload.size()) -
                         static_cast<i64>(cap);
        for (i64 i = 0; i < over && i < static_cast<i64>(evictable.size()); ++i) {
            want_unload.push_back(evictable[static_cast<usize>(i)]);
        }
    }

    // Apply: unload first, then load.
    for (CellId id : want_unload) {
        auto& e = impl_->cells[id];
        if (e.state != CellState::Loaded) continue;
        if (impl_->on_unload) impl_->on_unload(id, e.desc);
        e.state = CellState::Unloaded;
        ++impl_->cells_unloaded_total;
    }
    for (CellId id : want_load) {
        auto& e = impl_->cells[id];
        if (e.state == CellState::Loaded) continue;
        if (impl_->on_load) impl_->on_load(id, e.desc);
        e.state = CellState::Loaded;
        ++impl_->cells_loaded_total;
    }
}

void WorldPartition::force_load(CellId id) {
    auto it = impl_->cells.find(id);
    if (it == impl_->cells.end() || it->second.state == CellState::Loaded) return;
    if (impl_->on_load) impl_->on_load(id, it->second.desc);
    it->second.state = CellState::Loaded;
    ++impl_->cells_loaded_total;
}
void WorldPartition::force_unload(CellId id) {
    auto it = impl_->cells.find(id);
    if (it == impl_->cells.end() || it->second.state != CellState::Loaded) return;
    if (impl_->on_unload) impl_->on_unload(id, it->second.desc);
    it->second.state = CellState::Unloaded;
    ++impl_->cells_unloaded_total;
}

WorldPartitionStats WorldPartition::stats() const noexcept {
    WorldPartitionStats s{};
    s.cell_count = static_cast<u32>(impl_->cells.size());
    for (const auto& [_, e] : impl_->cells) {
        if (e.state == CellState::Loaded)    ++s.loaded;
        if (e.state == CellState::Loading)   ++s.loading;
        if (e.state == CellState::Unloading) ++s.unloading;
    }
    s.cells_loaded_total   = impl_->cells_loaded_total;
    s.cells_unloaded_total = impl_->cells_unloaded_total;
    return s;
}

std::vector<CellId> WorldPartition::loaded_cells() const {
    std::vector<CellId> r;
    for (const auto& [id, e] : impl_->cells) if (e.state == CellState::Loaded) r.push_back(id);
    return r;
}

std::vector<WorldPartition::CellRow> WorldPartition::describe_cells() const {
    std::vector<CellRow> rows;
    rows.reserve(impl_->cells.size());
    for (const auto& [id, e] : impl_->cells) {
        CellRow r{};
        r.id    = id;
        r.desc  = &e.desc;
        r.state = e.state;
        r.closest_viewer_distance = std::numeric_limits<f32>::max();
        r.in_any_view = false;
        for (const auto& [_, v] : impl_->viewers) {
            if (!v.active) continue;
            const f32 d = closest_dist_to_aabb(v.position, e.desc.bounds);
            if (d < r.closest_viewer_distance) r.closest_viewer_distance = d;
            if (v.frustum.intersects(e.desc.bounds)) r.in_any_view = true;
        }
        rows.push_back(r);
    }
    std::sort(rows.begin(), rows.end(),
        [](const CellRow& a, const CellRow& b) { return a.id < b.id; });
    return rows;
}

}  // namespace cardinal::partition
