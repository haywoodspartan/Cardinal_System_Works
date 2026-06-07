// =============================================================================
// Cardinal — scene statistics implementation.
// =============================================================================
#include <cardinal/actor/scene_stats.hpp>

#include <cardinal/actor/world.hpp>
#include <cardinal/core/std/algorithm.hpp>   // cardinal::sort

namespace cardinal::actor {

namespace {
// Increment the count for `name` in a (name -> count) tally vector.
void tally(cardinal::vector<NameCount>& v, const cardinal::string& name) {
    for (auto& nc : v) {
        if (nc.name == name) { ++nc.count; return; }
    }
    v.push_back(NameCount{ name, 1 });
}
void sort_by_name(cardinal::vector<NameCount>& v) {
    cardinal::sort(v.begin(), v.end(),
        [](const NameCount& a, const NameCount& b) { return a.name < b.name; });
}
}  // namespace

u32 WorldStats::component_count(const cardinal::string& type) const {
    for (const auto& nc : by_component) if (nc.name == type) return nc.count;
    return 0;
}
u32 WorldStats::tag_count(const cardinal::string& tag) const {
    for (const auto& nc : by_tag) if (nc.name == tag) return nc.count;
    return 0;
}

cardinal::core::Vec3 WorldStats::bounds_center() const {
    return { (bounds_min.x + bounds_max.x) * 0.5f,
             (bounds_min.y + bounds_max.y) * 0.5f,
             (bounds_min.z + bounds_max.z) * 0.5f };
}
cardinal::core::Vec3 WorldStats::bounds_extent() const {
    return { bounds_max.x - bounds_min.x,
             bounds_max.y - bounds_min.y,
             bounds_max.z - bounds_min.z };
}

WorldStats compute_world_stats(const World& world) {
    WorldStats s;
    for (const auto& aptr : world.actors()) {
        const Actor& a = *aptr;
        if (!a.alive()) continue;

        ++s.actors;
        if (a.enabled()) ++s.enabled; else ++s.disabled;

        // Expand the world AABB by this actor's Transform translation.
        if (const auto* tc = a.get_component<TransformComponent>()) {
            const auto& p = tc->translation;
            if (!s.has_bounds) {
                s.has_bounds = true;
                s.bounds_min = { p.x, p.y, p.z };
                s.bounds_max = { p.x, p.y, p.z };
            } else {
                if (p.x < s.bounds_min.x) s.bounds_min.x = p.x;
                if (p.y < s.bounds_min.y) s.bounds_min.y = p.y;
                if (p.z < s.bounds_min.z) s.bounds_min.z = p.z;
                if (p.x > s.bounds_max.x) s.bounds_max.x = p.x;
                if (p.y > s.bounds_max.y) s.bounds_max.y = p.y;
                if (p.z > s.bounds_max.z) s.bounds_max.z = p.z;
            }
        }

        bool is_instance = false;
        for (const auto& c : a.components()) {
            const cardinal::string type = c->type_name();
            tally(s.by_component, type);
            if (type == "PrefabLink") is_instance = true;
            // Tag breakdown — count each tag string across the world. Tally
            // from THIS component (c), not get_component<TagComponent>()
            // which returns only the first: an actor may legally hold more
            // than one TagComponent, and the first-match query would count
            // its tags once per Tag component while dropping the rest.
            if (type == "Tag") {
                if (const auto* tc = static_cast<const TagComponent*>(c.get())) {
                    for (const auto& t : tc->tags) tally(s.by_tag, t);
                }
            }
        }
        if (is_instance) ++s.prefab_instances;
    }

    sort_by_name(s.by_component);
    sort_by_name(s.by_tag);
    return s;
}

}  // namespace cardinal::actor
