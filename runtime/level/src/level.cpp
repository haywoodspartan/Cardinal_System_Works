#include <cardinal/level/level.hpp>

#include <cardinal/actor/component.hpp>
#include <cardinal/scene/scene.hpp>
#include <cardinal/scene/assets.hpp>
#include <cardinal/core/diag/log.hpp>

#include <cardinal/core/std/algorithm.hpp>    // cardinal::sort/max
#include <cardinal/core/std/cmath.hpp>        // cardinal::sqrt
#include <cardinal/core/std/containers.hpp>   // cardinal::queue
#include <cardinal/core/std/utility.hpp>      // cardinal::move/pair

namespace cardinal::level {

// ---------------------------------------------------------------------------
// LevelInstance
// ---------------------------------------------------------------------------
cardinal::shared_ptr<LevelInstance> LevelInstance::create(LevelInstanceDesc desc) {
    return cardinal::shared_ptr<LevelInstance>(new LevelInstance(cardinal::move(desc)));
}

// ---------------------------------------------------------------------------
// LevelManager
// ---------------------------------------------------------------------------
cardinal::shared_ptr<LevelManager> LevelManager::create(cardinal::actor::World& w) {
    return cardinal::shared_ptr<LevelManager>(new LevelManager(w));
}
LevelManager::~LevelManager() { clear(); }

InstanceId LevelManager::spawn(cardinal::shared_ptr<LevelInstance> inst,
                                const cardinal::scene::Vec3& trans,
                                const cardinal::scene::Vec3& rot,
                                const cardinal::scene::Vec3& scl)
{
    if (inst == nullptr) return 0;
    auto p = cardinal::make_unique<Placement>();
    p->id              = next_id_++;
    p->instance        = inst;
    p->translation     = trans;
    p->rotation_euler  = rot;
    p->scale           = scl;

    for (const auto& at : inst->desc().actors) {
        cardinal::actor::Actor* a = world_.spawn(at.name);
        if (a == nullptr) continue;
        // Apply placement transform on top of the template's local transform.
        if (auto* tr = a->get_component<cardinal::actor::TransformComponent>()) {
            tr->translation = {
                trans.x + at.translation.x * scl.x,
                trans.y + at.translation.y * scl.y,
                trans.z + at.translation.z * scl.z,
            };
            tr->rotation_euler = {
                rot.x + at.rotation_euler.x,
                rot.y + at.rotation_euler.y,
                rot.z + at.rotation_euler.z,
            };
            tr->scale = {
                scl.x * at.scale.x,
                scl.y * at.scale.y,
                scl.z * at.scale.z,
            };
        }
        if (!at.mesh_asset.empty()) {
            auto* mc = a->add_component<cardinal::actor::MeshComponent>();
            mc->asset_id = at.mesh_asset;
            mc->tint     = at.tint;
        }
        p->spawned_actor_ids.push_back(a->id());
    }
    cardinal::log::infof("level",
        "spawned LevelInstance '%s' (%zu actors) -> placement %u",
        inst->desc().name.c_str(), p->spawned_actor_ids.size(), p->id);
    placements_.push_back(cardinal::move(p));
    return placements_.back()->id;
}

bool LevelManager::despawn(InstanceId id) {
    for (auto it = placements_.begin(); it != placements_.end(); ++it) {
        if ((*it)->id != id) continue;
        for (auto aid : (*it)->spawned_actor_ids) world_.destroy(aid);
        placements_.erase(it);
        return true;
    }
    return false;
}

void LevelManager::clear() {
    for (auto& p : placements_) {
        for (auto aid : p->spawned_actor_ids) world_.destroy(aid);
    }
    placements_.clear();
}

usize LevelManager::placement_count() const noexcept { return placements_.size(); }

const Placement* LevelManager::find(InstanceId id) const {
    for (const auto& p : placements_) if (p->id == id) return p.get();
    return nullptr;
}

cardinal::vector<const Placement*> LevelManager::placements() const {
    cardinal::vector<const Placement*> r;
    r.reserve(placements_.size());
    for (const auto& p : placements_) r.push_back(p.get());
    return r;
}

// ---------------------------------------------------------------------------
// AssetPlacement
// ---------------------------------------------------------------------------
cardinal::shared_ptr<AssetPlacement>
AssetPlacement::create(cardinal::actor::World& world,
                       cardinal::scene::Scene& scene) {
    return cardinal::shared_ptr<AssetPlacement>(
        new AssetPlacement(world, scene));
}

PlaceResult AssetPlacement::place(const char* asset_id,
                                  cardinal::rhi::Device* device,
                                  const cardinal::scene::Vec3& position,
                                  f32 yaw_rad) {
    PlaceResult out{};
    if (asset_id == nullptr) return out;

    // 1. Render representation via the existing catalog factory.
    cardinal::scene::AssetSpawnContext ctx{};
    ctx.scene    = &scene_;
    ctx.device   = device;
    ctx.position = position;
    const auto res = cardinal::scene::AssetCatalog::instance()
                         .spawn(asset_id, ctx);
    if (res.entity_ids.empty()) {
        cardinal::log::warnf("level/place",
            "asset '%s' produced no entities", asset_id);
        return out;
    }
    const u32 primary = res.primary_id ? res.primary_id
                                       : res.entity_ids.front();
    cardinal::scene::Entity* pe = scene_.find_by_id(primary);

    // 1b. Bake the requested placement yaw (about world up). Sub-entities of
    // a composite rotate as a rigid group about the placement point, using
    // the same convention as Mat4::rotation_xyz (x' = x*c + z*s,
    // z' = -x*s + z*c), so a rotated composite renders exactly as if the
    // whole asset were one mesh. Done BEFORE the actor copies the primary
    // transform and BEFORE offsets are captured — both pick up the rotated
    // form for free, and the per-frame syncs need no rotation math.
    if (yaw_rad != 0.0f) {
        const f32 cy = cardinal::cos(yaw_rad), sy = cardinal::sin(yaw_rad);
        for (u32 eid : res.entity_ids) {
            cardinal::scene::Entity* e = scene_.find_by_id(eid);
            if (e == nullptr) continue;
            if (eid != primary) {
                const f32 dx = e->transform.translation.x - position.x;
                const f32 dz = e->transform.translation.z - position.z;
                e->transform.translation.x = position.x + dx * cy + dz * sy;
                e->transform.translation.z = position.z - dx * sy + dz * cy;
            }
            e->transform.rotation_euler.y += yaw_rad;
        }
    }

    // 2. Authoritative backing actor.
    cardinal::actor::Actor* a =
        world_.spawn(pe ? pe->name : cardinal::string("PlacedAsset"));
    if (a == nullptr) return out;
    // World::spawn already attaches a default TransformComponent — fetch
    // it (a second add_component would leave get_component returning the
    // origin default).
    auto* tc = a->get_component<cardinal::actor::TransformComponent>();
    if (tc == nullptr)
        tc = a->add_component<cardinal::actor::TransformComponent>();
    if (pe) {
        tc->translation    = pe->transform.translation;
        tc->rotation_euler = pe->transform.rotation_euler;
        tc->scale          = pe->transform.scale;
    } else {
        tc->translation = position;
    }
    auto* mc = a->get_component<cardinal::actor::MeshComponent>();
    if (mc == nullptr)
        mc = a->add_component<cardinal::actor::MeshComponent>();
    mc->asset_id = asset_id;
    if (pe) mc->tint = pe->tint;
    auto* tg = a->get_component<cardinal::actor::TagComponent>();
    if (tg == nullptr)
        tg = a->add_component<cardinal::actor::TagComponent>();
    tg->add("placed");

    // 3. Track + capture each render entity's offset from the actor
    //    origin so composites translate as a rigid group.
    PlacedAsset rec{};
    rec.asset_id       = asset_id;
    rec.actor          = a->id();
    rec.primary_entity = primary;
    rec.entities.push_back(primary);
    rec.local_offset.push_back({0.0f, 0.0f, 0.0f});
    for (u32 eid : res.entity_ids) {
        if (eid == primary) continue;
        rec.entities.push_back(eid);
        cardinal::scene::Vec3 off{0, 0, 0};
        if (auto* e = scene_.find_by_id(eid)) {
            off = { e->transform.translation.x - tc->translation.x,
                    e->transform.translation.y - tc->translation.y,
                    e->transform.translation.z - tc->translation.z };
        }
        rec.local_offset.push_back(off);
    }
    placed_.push_back(cardinal::move(rec));

    out.actor          = a->id();
    out.primary_entity = primary;
    return out;
}

void AssetPlacement::sync_to_scene() {
    for (const auto& p : placed_) {
        cardinal::actor::Actor* a = world_.find(p.actor);
        if (a == nullptr || !a->alive()) continue;
        auto* tc = a->get_component<cardinal::actor::TransformComponent>();
        if (tc == nullptr) continue;
        auto* mc = a->get_component<cardinal::actor::MeshComponent>();
        for (usize i = 0; i < p.entities.size(); ++i) {
            cardinal::scene::Entity* e = scene_.find_by_id(p.entities[i]);
            if (e == nullptr) continue;
            const cardinal::scene::Vec3& off = p.local_offset[i];
            e->transform.translation = { tc->translation.x + off.x,
                                         tc->translation.y + off.y,
                                         tc->translation.z + off.z };
            if (i == 0) {
                // Primary is a full 1:1 bind; sub-parts keep authored
                // rotation/scale and just follow the group translation.
                e->transform.rotation_euler = tc->rotation_euler;
                e->transform.scale          = tc->scale;
                if (mc != nullptr) {
                    e->tint    = mc->tint;
                    e->visible = mc->visible;
                }
            }
        }
    }
}

void AssetPlacement::sync_from_scene() {
    for (auto& p : placed_) {
        cardinal::actor::Actor* a = world_.find(p.actor);
        if (a == nullptr || !a->alive()) continue;
        cardinal::scene::Entity* e = scene_.find_by_id(p.primary_entity);
        if (e == nullptr) continue;
        if (auto* tc =
                a->get_component<cardinal::actor::TransformComponent>()) {
            tc->translation    = e->transform.translation;
            tc->rotation_euler = e->transform.rotation_euler;
            tc->scale          = e->transform.scale;
        }
        if (auto* mc = a->get_component<cardinal::actor::MeshComponent>())
            mc->tint = e->tint;
    }
}

bool AssetPlacement::remove(cardinal::actor::ActorId actor) {
    for (usize i = 0; i < placed_.size(); ++i) {
        if (placed_[i].actor != actor) continue;
        for (u32 eid : placed_[i].entities) scene_.remove_entity(eid);
        world_.destroy(actor);
        placed_.erase(placed_.begin() + static_cast<cardinal::isize>(i));
        return true;
    }
    return false;
}

void AssetPlacement::clear() {
    for (auto& p : placed_) {
        for (u32 eid : p.entities) scene_.remove_entity(eid);
        world_.destroy(p.actor);
    }
    placed_.clear();
}

// ---------------------------------------------------------------------------
// HLOD
// ---------------------------------------------------------------------------
const HlodNode* HlodTree::find(HlodId id) const noexcept {
    auto it = index_of.find(id);
    return it == index_of.end() ? nullptr : &nodes[it->second];
}

namespace {

cardinal::scene::Vec3 vmean(const cardinal::vector<cardinal::scene::Vec3>& v) {
    cardinal::scene::Vec3 c{0,0,0};
    if (v.empty()) return c;
    for (const auto& p : v) { c.x += p.x; c.y += p.y; c.z += p.z; }
    const f32 inv = 1.0f / static_cast<f32>(v.size());
    return { c.x * inv, c.y * inv, c.z * inv };
}

cardinal::core::geom::AABB union_aabbs(const cardinal::vector<cardinal::core::geom::AABB>& v) {
    auto r = cardinal::core::geom::AABB::make_empty();
    for (const auto& a : v) r.expand(a);
    return r;
}

f32 vdist(const cardinal::scene::Vec3& a, const cardinal::scene::Vec3& b) {
    const f32 dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return cardinal::sqrt(dx*dx + dy*dy + dz*dz);
}

// Simple greedy spatial clustering — pick a seed, group its k-1 nearest
// unassigned, repeat. Cheap, fine for LOD baking.
void cluster_greedy(const cardinal::vector<HlodInput>& inputs,
                    u32 cluster_size,
                    cardinal::vector<cardinal::vector<u32>>& out_clusters)
{
    const u32 N = static_cast<u32>(inputs.size());
    cardinal::vector<bool> used(N, false);
    out_clusters.clear();
    for (u32 i = 0; i < N; ++i) {
        if (used[i]) continue;
        cardinal::vector<u32> c{ i };
        used[i] = true;
        // Collect cluster_size-1 nearest unused.
        cardinal::vector<cardinal::pair<f32, u32>> ranked;
        ranked.reserve(N);
        for (u32 j = 0; j < N; ++j) {
            if (used[j]) continue;
            ranked.push_back({ vdist(inputs[i].position, inputs[j].position), j });
        }
        // NaN-safe SWO. vdist(NaN, x) = sqrt(NaN^2 + ...) = NaN. With
        // any NaN distance in ranked, `a.first < b.first` violates
        // strict-weak-ordering (NaN compares unordered to everything),
        // and std::sort's introsort can fail to terminate or scribble
        // OOB — same UB class as sky::sort_keys (4ff85a8). Realistic
        // path: a corrupt scene file with `position = (nan, 0, 0)`
        // (sscanf("%f", ...) accepts "nan" verbatim) → HlodInput
        // with NaN position → vdist → NaN ranked.first. Also: the
        // bottom-up recursion (build_hlod:355 `hi.position =
        // cn->centroid`) propagates a NaN leaf upward via NaN-bearing
        // averages, so every subsequent layer's cluster_greedy hits
        // the same sort. Promote NaN to "greater than all finites" —
        // proper SWO; NaN-distance inputs cluster LAST (semantically
        // "farthest," which is the right bucket).
        cardinal::sort(ranked.begin(), ranked.end(),
            [](const auto& a, const auto& b){
                const bool af = cardinal::isfinite(a.first);
                const bool bf = cardinal::isfinite(b.first);
                if (af && bf) return a.first < b.first;
                return af && !bf;   // finite < NaN; NaN ~ NaN
            });
        for (const auto& [_, j] : ranked) {
            if (c.size() >= cluster_size) break;
            c.push_back(j);
            used[j] = true;
        }
        out_clusters.push_back(cardinal::move(c));
    }
}

}  // namespace

HlodTree build_hlod(const cardinal::vector<HlodInput>& inputs,
                    const HlodBuildOptions& opts)
{
    HlodTree t{};
    if (inputs.empty()) return t;

    // Leaf nodes — one per input.
    HlodId next_id = 1;
    cardinal::vector<HlodId> current_layer;
    current_layer.reserve(inputs.size());
    for (const auto& in : inputs) {
        HlodNode n{};
        n.id        = next_id++;
        n.bounds    = in.bounds.empty() ? cardinal::core::geom::AABB::from_center_extent(
                                              in.position, {0.5f, 0.5f, 0.5f})
                                        : in.bounds;
        n.centroid  = in.position;
        n.proxy_radius = cardinal::max({n.bounds.size().x, n.bounds.size().y, n.bounds.size().z}) * 0.5f;
        n.is_leaf   = true;
        n.leaf_ids  = { in.id };
        t.index_of[n.id] = static_cast<u32>(t.nodes.size());
        current_layer.push_back(n.id);
        t.nodes.push_back(cardinal::move(n));
    }

    // Bottom-up: cluster the current layer; each cluster becomes a parent.
    u32 depth = 0;
    while (current_layer.size() > 1 && depth < opts.max_depth) {
        cardinal::vector<HlodInput> cluster_inputs;
        cluster_inputs.reserve(current_layer.size());
        for (HlodId child_id : current_layer) {
            const auto* cn = t.find(child_id);
            if (cn == nullptr) continue;
            HlodInput hi{};
            hi.id       = child_id;
            hi.position = cn->centroid;
            hi.bounds   = cn->bounds;
            cluster_inputs.push_back(hi);
        }
        cardinal::vector<cardinal::vector<u32>> clusters;
        cluster_greedy(cluster_inputs, opts.cluster_size, clusters);
        if (clusters.size() >= current_layer.size()) break; // no progress

        cardinal::vector<HlodId> next_layer;
        next_layer.reserve(clusters.size());
        for (const auto& cluster : clusters) {
            HlodNode parent{};
            parent.id      = next_id++;
            parent.is_leaf = false;
            cardinal::vector<cardinal::scene::Vec3> centroids;
            cardinal::vector<cardinal::core::geom::AABB>  boxes;
            centroids.reserve(cluster.size());
            boxes.reserve(cluster.size());
            for (u32 idx : cluster) {
                const HlodId child_id = cluster_inputs[idx].id;
                parent.children.push_back(child_id);
                const auto* cn = t.find(child_id);
                if (cn) {
                    centroids.push_back(cn->centroid);
                    boxes.push_back(cn->bounds);
                }
                // Re-parent. We hold an index into nodes_; reach in.
                if (auto it = t.index_of.find(child_id); it != t.index_of.end()) {
                    t.nodes[it->second].parent = parent.id;
                }
            }
            parent.bounds       = union_aabbs(boxes);
            parent.centroid     = vmean(centroids);
            parent.proxy_radius = cardinal::max({parent.bounds.size().x,
                                             parent.bounds.size().y,
                                             parent.bounds.size().z}) * 0.5f;
            t.index_of[parent.id] = static_cast<u32>(t.nodes.size());
            next_layer.push_back(parent.id);
            t.nodes.push_back(cardinal::move(parent));
        }
        current_layer.swap(next_layer);
        ++depth;
    }
    if (!current_layer.empty()) t.root = current_layer.front();
    return t;
}

void select_hlod(const HlodTree& tree,
                 const cardinal::scene::Vec3& camera_pos,
                 f32 leaf_distance, f32 proxy_distance,
                 HlodSelection& out)
{
    out.render_proxies.clear();
    out.render_leaves.clear();
    if (tree.root == kInvalidHlodId) return;

    cardinal::queue<HlodId> q;
    q.push(tree.root);
    while (!q.empty()) {
        const HlodId id = q.front(); q.pop();
        const auto* n = tree.find(id);
        if (n == nullptr) continue;
        const f32 d = vdist(camera_pos, n->centroid) - n->proxy_radius;

        if (n->is_leaf) {
            // Leaves render only when within leaf_distance.
            if (d <= leaf_distance) {
                for (HlodId leaf : n->leaf_ids) out.render_leaves.push_back(leaf);
            }
            continue;
        }
        if (d > proxy_distance) {
            // Far enough that we render the proxy and don't recurse.
            out.render_proxies.push_back(id);
        } else {
            for (HlodId c : n->children) q.push(c);
        }
    }
}

}  // namespace cardinal::level
