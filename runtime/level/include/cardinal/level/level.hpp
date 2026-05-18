#pragma once

// =============================================================================
// Cardinal — Level Instance + HLOD.
//
// LevelInstance:
//   A reusable bundle of actors authored once + placed multiple times in
//   parent levels. Each placement is a transformed instance — moving the
//   placement transforms every contained actor as a unit. Spawn / despawn
//   is atomic: instantiating a LevelInstance creates all its actors,
//   destroying it removes all of them.
//
// HLOD (Hierarchical LOD):
//   At authoring time, a build step takes a cluster of meshes and bakes
//   a single "proxy mesh" representing them at distance. At runtime, when
//   the camera is far from the cluster, the proxy renders instead of the
//   N child meshes. Saves draw calls + vertex throughput dramatically.
//
//   Tree shape: each HlodNode owns child node ids (smaller clusters, or
//   leaves = actor ids). Distance bands choose which level of the tree
//   to render. Build is offline (Studio panel) or scripted via the
//   build_hlod_for_actors() helper.
// =============================================================================

#include <cardinal/actor/world.hpp>
#include <cardinal/core/types.hpp>
#include <cardinal/core/geom.hpp>
#include <cardinal/scene/math.hpp>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cardinal::level {

// ---------------------------------------------------------------------------
// LevelInstance — a reusable actor bundle.
//
// Build via LevelInstance::author_from(world, root_actor_ids), serialise
// via cardinal::serial (future), instantiate via spawn_in().
// ---------------------------------------------------------------------------
struct ActorTemplate {
    std::string             name;
    std::string             game_class;          // "" for non-GameActor entries
    cardinal::scene::Vec3   translation{0,0,0};
    cardinal::scene::Vec3   rotation_euler{0,0,0};
    cardinal::scene::Vec3   scale{1,1,1};
    std::string             mesh_asset;          // optional MeshComponent.asset_id
    cardinal::scene::Vec3   tint{1,1,1};
};

struct LevelInstanceDesc {
    std::string                  name;
    std::vector<ActorTemplate>   actors;
};

using InstanceId = u32;

class LevelInstance {
public:
    static std::shared_ptr<LevelInstance> create(LevelInstanceDesc desc);
    const LevelInstanceDesc& desc() const noexcept { return desc_; }

private:
    explicit LevelInstance(LevelInstanceDesc d) : desc_(std::move(d)) {}
    LevelInstanceDesc desc_;
};

// Spawned placement of a LevelInstance — transformed into the parent world.
struct Placement {
    InstanceId                                id{0};
    std::shared_ptr<LevelInstance>            instance;
    cardinal::scene::Vec3                     translation{0,0,0};
    cardinal::scene::Vec3                     rotation_euler{0,0,0};
    cardinal::scene::Vec3                     scale{1,1,1};
    std::vector<cardinal::actor::ActorId>     spawned_actor_ids;
};

class LevelManager {
public:
    static std::shared_ptr<LevelManager> create(cardinal::actor::World& world);
    ~LevelManager();

    InstanceId          spawn(std::shared_ptr<LevelInstance> inst,
                              const cardinal::scene::Vec3& translation = {0,0,0},
                              const cardinal::scene::Vec3& rotation    = {0,0,0},
                              const cardinal::scene::Vec3& scale       = {1,1,1});
    bool                despawn(InstanceId id);
    void                clear();
    usize               placement_count() const noexcept;

    const Placement*    find(InstanceId id) const;
    std::vector<const Placement*> placements() const;

private:
    explicit LevelManager(cardinal::actor::World& w) : world_(w) {}
    cardinal::actor::World&                       world_;
    std::vector<std::unique_ptr<Placement>>       placements_;
    InstanceId                                    next_id_{1};
};

// ---------------------------------------------------------------------------
// HLOD — Hierarchical LOD.
//
// Build a tree from a set of (id, world position, world AABB). Internal
// nodes group children spatially; leaves reference the input ids. At run
// time, sample the tree against the camera and produce the active set
// of "render either this internal proxy OR its descendants".
// ---------------------------------------------------------------------------
using HlodId      = u32;
inline constexpr HlodId kInvalidHlodId = 0;

struct HlodInput {
    HlodId               id{0};
    cardinal::scene::Vec3 position{0,0,0};
    cardinal::core::geom::AABB bounds{};
};

struct HlodNode {
    HlodId                   id{0};
    cardinal::core::geom::AABB     bounds{};
    cardinal::scene::Vec3    centroid{0,0,0};
    f32                      proxy_radius{0.0f};   // bounding-sphere radius
    bool                     is_leaf{true};
    HlodId                   parent{kInvalidHlodId};
    std::vector<HlodId>      children;             // empty when is_leaf
    std::vector<HlodId>      leaf_ids;             // when is_leaf, the input ids
};

struct HlodTree {
    std::vector<HlodNode>                  nodes;
    std::unordered_map<HlodId, u32>        index_of;     // node id -> nodes[]
    HlodId                                 root{kInvalidHlodId};

    const HlodNode* find(HlodId id) const noexcept;
};

struct HlodBuildOptions {
    u32 cluster_size{8};     // children per internal node
    u32 max_depth{8};
};

HlodTree build_hlod(const std::vector<HlodInput>& inputs,
                    const HlodBuildOptions& opts = {});

// ---------------------------------------------------------------------------
// HLOD selection — given camera + distance bands, walk the tree and emit
// either an internal-proxy id OR a list of leaf ids. Per-band threshold:
// distance from camera below which we recurse into children, above which
// we render the proxy.
// ---------------------------------------------------------------------------
struct HlodSelection {
    std::vector<HlodId> render_proxies;   // internal nodes rendered as proxy
    std::vector<HlodId> render_leaves;    // leaf ids drawn directly
};

void select_hlod(const HlodTree& tree,
                 const cardinal::scene::Vec3& camera_pos,
                 f32 leaf_distance, f32 proxy_distance,
                 HlodSelection& out);

}  // namespace cardinal::level
