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
#include <cardinal/core/types.hpp>        // function/memory/string
#include <cardinal/core/std/utility.hpp>      // cardinal::move
#include <cardinal/core/std/containers.hpp>   // unordered_map/vector
#include <cardinal/core/math/geom.hpp>
#include <cardinal/scene/math.hpp>

namespace cardinal::level {

// ---------------------------------------------------------------------------
// LevelInstance — a reusable actor bundle.
//
// Build via LevelInstance::author_from(world, root_actor_ids), serialise
// via cardinal::serial (future), instantiate via spawn_in().
// ---------------------------------------------------------------------------
struct ActorTemplate {
    cardinal::string             name;
    cardinal::string             game_class;          // "" for non-GameActor entries
    cardinal::scene::Vec3   translation{0,0,0};
    cardinal::scene::Vec3   rotation_euler{0,0,0};
    cardinal::scene::Vec3   scale{1,1,1};
    cardinal::string             mesh_asset;          // optional MeshComponent.asset_id
    cardinal::scene::Vec3   tint{1,1,1};
};

struct LevelInstanceDesc {
    cardinal::string                  name;
    cardinal::vector<ActorTemplate>   actors;
};

using InstanceId = u32;

class LevelInstance {
public:
    static cardinal::shared_ptr<LevelInstance> create(LevelInstanceDesc desc);
    const LevelInstanceDesc& desc() const noexcept { return desc_; }

private:
    explicit LevelInstance(LevelInstanceDesc d) : desc_(cardinal::move(d)) {}
    LevelInstanceDesc desc_;
};

// Spawned placement of a LevelInstance — transformed into the parent world.
struct Placement {
    InstanceId                                id{0};
    cardinal::shared_ptr<LevelInstance>            instance;
    cardinal::scene::Vec3                     translation{0,0,0};
    cardinal::scene::Vec3                     rotation_euler{0,0,0};
    cardinal::scene::Vec3                     scale{1,1,1};
    cardinal::vector<cardinal::actor::ActorId>     spawned_actor_ids;
};

class LevelManager {
public:
    static cardinal::shared_ptr<LevelManager> create(cardinal::actor::World& world);
    ~LevelManager();

    InstanceId          spawn(cardinal::shared_ptr<LevelInstance> inst,
                              const cardinal::scene::Vec3& translation = {0,0,0},
                              const cardinal::scene::Vec3& rotation    = {0,0,0},
                              const cardinal::scene::Vec3& scale       = {1,1,1});
    bool                despawn(InstanceId id);
    void                clear();
    usize               placement_count() const noexcept;

    const Placement*    find(InstanceId id) const;
    cardinal::vector<const Placement*> placements() const;

private:
    explicit LevelManager(cardinal::actor::World& w) : world_(w) {}
    cardinal::actor::World&                       world_;
    cardinal::vector<cardinal::unique_ptr<Placement>>       placements_;
    InstanceId                                    next_id_{1};
};

// ---------------------------------------------------------------------------
// AssetPlacement — drop scene::AssetCatalog assets into the level as real
// actors.
//
// The fix for "place objects/actors/meshes with the studio tools in full":
// scene::AssetCatalog factories only build scene::Scene render entities, so
// placed assets were invisible to actor::World, the Actor Outliner, Game,
// serial save/load, and the level. AssetPlacement makes the placed asset a
// first-class actor::Actor (Transform + Mesh + "placed" Tag) — so every
// actor-based Studio tool sees it — while a linked scene::Entity stays the
// render mirror (the renderer already walks scene entities).
//
// Authority model (per the chosen design): the ACTOR is authoritative.
//   - place()           : spawn render entity(ies) + the backing actor.
//   - sync_to_scene()   : actor Transform/tint -> scene entity (call each
//                         frame before render; pushes Outliner / serial-
//                         load / programmatic actor edits to the view).
//   - sync_from_scene() : scene entity -> actor (call right after the
//                         Studio gizmo / inspector, which edit the scene
//                         entity, so the authoritative actor + save-load
//                         pick the move up).
// Single-mesh assets are a 1:1 actor<->entity bind (full gizmo fidelity).
// Composite assets (e.g. tree = trunk+leaves) translate as a rigid group
// off the actor; sub-part authored rotation/scale is preserved.
// ---------------------------------------------------------------------------
}  // namespace cardinal::level

namespace cardinal::scene { class Scene; }
namespace cardinal::rhi   { class Device; }

namespace cardinal::level {

struct PlacedAsset {
    cardinal::string                          asset_id;
    cardinal::actor::ActorId                  actor{0};
    u32                                       primary_entity{0};
    cardinal::vector<u32>                     entities;          // primary first
    cardinal::vector<cardinal::scene::Vec3>   local_offset;      // vs actor origin
};

struct PlaceResult {
    cardinal::actor::ActorId actor{0};          // 0 == failed
    u32                      primary_entity{0}; // feed Studio selection
};

class AssetPlacement {
public:
    static cardinal::shared_ptr<AssetPlacement>
        create(cardinal::actor::World& world, cardinal::scene::Scene& scene);

    PlaceResult place(const char* asset_id,
                      cardinal::rhi::Device* device,
                      const cardinal::scene::Vec3& position);

    void sync_to_scene();     // actor (authoritative) -> scene render mirror
    void sync_from_scene();   // scene edit (Studio gizmo/inspector) -> actor

    bool   remove(cardinal::actor::ActorId actor);
    void   clear();
    usize  count() const noexcept { return placed_.size(); }
    const cardinal::vector<PlacedAsset>& placements() const noexcept {
        return placed_;
    }

private:
    AssetPlacement(cardinal::actor::World& w, cardinal::scene::Scene& s)
        : world_(w), scene_(s) {}
    cardinal::actor::World&       world_;
    cardinal::scene::Scene&       scene_;
    cardinal::vector<PlacedAsset> placed_;
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
    cardinal::vector<HlodId>      children;             // empty when is_leaf
    cardinal::vector<HlodId>      leaf_ids;             // when is_leaf, the input ids
};

struct HlodTree {
    cardinal::vector<HlodNode>                  nodes;
    cardinal::unordered_map<HlodId, u32>        index_of;     // node id -> nodes[]
    HlodId                                 root{kInvalidHlodId};

    const HlodNode* find(HlodId id) const noexcept;
};

struct HlodBuildOptions {
    u32 cluster_size{8};     // children per internal node
    u32 max_depth{8};
};

HlodTree build_hlod(const cardinal::vector<HlodInput>& inputs,
                    const HlodBuildOptions& opts = {});

// ---------------------------------------------------------------------------
// HLOD selection — given camera + distance bands, walk the tree and emit
// either an internal-proxy id OR a list of leaf ids. Per-band threshold:
// distance from camera below which we recurse into children, above which
// we render the proxy.
// ---------------------------------------------------------------------------
struct HlodSelection {
    cardinal::vector<HlodId> render_proxies;   // internal nodes rendered as proxy
    cardinal::vector<HlodId> render_leaves;    // leaf ids drawn directly
};

void select_hlod(const HlodTree& tree,
                 const cardinal::scene::Vec3& camera_pos,
                 f32 leaf_distance, f32 proxy_distance,
                 HlodSelection& out);

}  // namespace cardinal::level
