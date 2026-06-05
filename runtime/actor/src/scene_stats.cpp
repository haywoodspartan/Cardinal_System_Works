// =============================================================================
// Cardinal — scene statistics implementation.
// =============================================================================
#include <cardinal/actor/scene_stats.hpp>

#include <cardinal/actor/world.hpp>
#include <cardinal/core/algorithm.hpp>   // cardinal::sort

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

WorldStats compute_world_stats(const World& world) {
    WorldStats s;
    for (const auto& aptr : world.actors()) {
        const Actor& a = *aptr;
        if (!a.alive()) continue;

        ++s.actors;
        if (a.enabled()) ++s.enabled; else ++s.disabled;

        bool is_instance = false;
        for (const auto& c : a.components()) {
            const cardinal::string type = c->type_name();
            tally(s.by_component, type);
            if (type == "PrefabLink") is_instance = true;
            // Tag breakdown — count each tag string across the world.
            if (type == "Tag") {
                if (const auto* tc = a.get_component<TagComponent>()) {
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
