#include <cardinal/level/level.hpp>

#include <cardinal/actor/component.hpp>
#include <cardinal/core/log.hpp>

#include <algorithm>
#include <cmath>
#include <queue>

namespace cardinal::level {

// ---------------------------------------------------------------------------
// LevelInstance
// ---------------------------------------------------------------------------
std::shared_ptr<LevelInstance> LevelInstance::create(LevelInstanceDesc desc) {
    return std::shared_ptr<LevelInstance>(new LevelInstance(std::move(desc)));
}

// ---------------------------------------------------------------------------
// LevelManager
// ---------------------------------------------------------------------------
std::shared_ptr<LevelManager> LevelManager::create(cardinal::actor::World& w) {
    return std::shared_ptr<LevelManager>(new LevelManager(w));
}
LevelManager::~LevelManager() { clear(); }

InstanceId LevelManager::spawn(std::shared_ptr<LevelInstance> inst,
                                const cardinal::scene::Vec3& trans,
                                const cardinal::scene::Vec3& rot,
                                const cardinal::scene::Vec3& scl)
{
    if (inst == nullptr) return 0;
    auto p = std::make_unique<Placement>();
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
    placements_.push_back(std::move(p));
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

std::vector<const Placement*> LevelManager::placements() const {
    std::vector<const Placement*> r;
    r.reserve(placements_.size());
    for (const auto& p : placements_) r.push_back(p.get());
    return r;
}

// ---------------------------------------------------------------------------
// HLOD
// ---------------------------------------------------------------------------
const HlodNode* HlodTree::find(HlodId id) const noexcept {
    auto it = index_of.find(id);
    return it == index_of.end() ? nullptr : &nodes[it->second];
}

namespace {

cardinal::scene::Vec3 vmean(const std::vector<cardinal::scene::Vec3>& v) {
    cardinal::scene::Vec3 c{0,0,0};
    if (v.empty()) return c;
    for (const auto& p : v) { c.x += p.x; c.y += p.y; c.z += p.z; }
    const f32 inv = 1.0f / static_cast<f32>(v.size());
    return { c.x * inv, c.y * inv, c.z * inv };
}

cardinal::core::geom::AABB union_aabbs(const std::vector<cardinal::core::geom::AABB>& v) {
    auto r = cardinal::core::geom::AABB::make_empty();
    for (const auto& a : v) r.expand(a);
    return r;
}

f32 vdist(const cardinal::scene::Vec3& a, const cardinal::scene::Vec3& b) {
    const f32 dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// Simple greedy spatial clustering — pick a seed, group its k-1 nearest
// unassigned, repeat. Cheap, fine for LOD baking.
void cluster_greedy(const std::vector<HlodInput>& inputs,
                    u32 cluster_size,
                    std::vector<std::vector<u32>>& out_clusters)
{
    const u32 N = static_cast<u32>(inputs.size());
    std::vector<bool> used(N, false);
    out_clusters.clear();
    for (u32 i = 0; i < N; ++i) {
        if (used[i]) continue;
        std::vector<u32> c{ i };
        used[i] = true;
        // Collect cluster_size-1 nearest unused.
        std::vector<std::pair<f32, u32>> ranked;
        ranked.reserve(N);
        for (u32 j = 0; j < N; ++j) {
            if (used[j]) continue;
            ranked.push_back({ vdist(inputs[i].position, inputs[j].position), j });
        }
        std::sort(ranked.begin(), ranked.end(),
            [](const auto& a, const auto& b){ return a.first < b.first; });
        for (const auto& [_, j] : ranked) {
            if (c.size() >= cluster_size) break;
            c.push_back(j);
            used[j] = true;
        }
        out_clusters.push_back(std::move(c));
    }
}

}  // namespace

HlodTree build_hlod(const std::vector<HlodInput>& inputs,
                    const HlodBuildOptions& opts)
{
    HlodTree t{};
    if (inputs.empty()) return t;

    // Leaf nodes — one per input.
    HlodId next_id = 1;
    std::vector<HlodId> current_layer;
    current_layer.reserve(inputs.size());
    for (const auto& in : inputs) {
        HlodNode n{};
        n.id        = next_id++;
        n.bounds    = in.bounds.empty() ? cardinal::core::geom::AABB::from_center_extent(
                                              in.position, {0.5f, 0.5f, 0.5f})
                                        : in.bounds;
        n.centroid  = in.position;
        n.proxy_radius = std::max({n.bounds.size().x, n.bounds.size().y, n.bounds.size().z}) * 0.5f;
        n.is_leaf   = true;
        n.leaf_ids  = { in.id };
        t.index_of[n.id] = static_cast<u32>(t.nodes.size());
        current_layer.push_back(n.id);
        t.nodes.push_back(std::move(n));
    }

    // Bottom-up: cluster the current layer; each cluster becomes a parent.
    u32 depth = 0;
    while (current_layer.size() > 1 && depth < opts.max_depth) {
        std::vector<HlodInput> cluster_inputs;
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
        std::vector<std::vector<u32>> clusters;
        cluster_greedy(cluster_inputs, opts.cluster_size, clusters);
        if (clusters.size() >= current_layer.size()) break; // no progress

        std::vector<HlodId> next_layer;
        next_layer.reserve(clusters.size());
        for (const auto& cluster : clusters) {
            HlodNode parent{};
            parent.id      = next_id++;
            parent.is_leaf = false;
            std::vector<cardinal::scene::Vec3> centroids;
            std::vector<cardinal::core::geom::AABB>  boxes;
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
            parent.proxy_radius = std::max({parent.bounds.size().x,
                                             parent.bounds.size().y,
                                             parent.bounds.size().z}) * 0.5f;
            t.index_of[parent.id] = static_cast<u32>(t.nodes.size());
            next_layer.push_back(parent.id);
            t.nodes.push_back(std::move(parent));
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

    std::queue<HlodId> q;
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
