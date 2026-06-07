#pragma once

// =============================================================================
// Cardinal — scene statistics.
//
// compute_world_stats(world) summarises the authored world's composition: how
// many actors, how many enabled vs disabled, how many are prefab instances,
// and the breakdown by component type + by tag. The "scene overview" readout
// every editor has — pairs with validation (problems) + search (find) to
// answer "what's actually in this level".
//
// Pure + deterministic over the World, so it's unit-testable and cheap to run
// while the panel is open. Breakdowns are sorted by name for stable display.
// =============================================================================

#include <cardinal/core/types.hpp>        // cardinal::string
#include <cardinal/core/std/containers.hpp>   // cardinal::vector
#include <cardinal/core/math/math.hpp>         // cardinal::core::Vec3

namespace cardinal::actor {

class World;

struct NameCount {
    cardinal::string name;
    u32              count{0};
};

struct WorldStats {
    u32 actors{0};            // alive actors
    u32 enabled{0};
    u32 disabled{0};
    u32 prefab_instances{0};  // carry a PrefabLink
    cardinal::vector<NameCount> by_component;   // component type_name -> count, sorted
    cardinal::vector<NameCount> by_tag;         // tag -> count, sorted

    // World-space AABB of all alive actors' Transform translations (point
    // bounds, not mesh extents). has_bounds is false for an empty world or
    // one with no Transform-bearing actors (min/max stay zero). The building
    // block for "frame all" + a level-extent readout.
    bool                 has_bounds{false};
    cardinal::core::Vec3 bounds_min{0.0f, 0.0f, 0.0f};
    cardinal::core::Vec3 bounds_max{0.0f, 0.0f, 0.0f};
    cardinal::core::Vec3 bounds_center() const;   // (min + max) * 0.5
    cardinal::core::Vec3 bounds_extent() const;   // max - min (full size)

    // Convenience lookups (0 if absent).
    u32 component_count(const cardinal::string& type) const;
    u32 tag_count(const cardinal::string& tag) const;
};

WorldStats compute_world_stats(const World& world);

}  // namespace cardinal::actor
