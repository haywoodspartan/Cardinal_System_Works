// =============================================================================
// Cardinal — deterministic level / HLOD regression suite.
//
// LevelManager spawns a reusable LevelInstance into an actor::World,
// COMPOSING the placement transform onto each template:
//   translation = place.t + template.t * place.scale (componentwise)
//   rotation    = place.r + template.r
//   scale       = place.scale * template.scale
// despawn is deferred: it World::destroy()s the actors (alive=false, not
// yet swept) and drops the placement immediately. build_hlod is a
// bottom-up greedy spatial clustering — collinear inputs make the tree
// hand-exact (node ids, parent links, centroids, proxy_radius, bounds).
// select_hlod is a distance-band BFS emitting proxy vs leaf vs culled.
//
// Note: AABB::empty() is min>max, so a DEFAULT HlodInput.bounds (a zero
// box) is NOT empty — only AABB::make_empty() triggers the
// from_center_extent leaf path (size 1, proxy_radius 0.5). Pure CPU,
// headless, fully deterministic. Exit 0 = all pass.
// =============================================================================

#include <cardinal/level/level.hpp>
#include <cardinal/scene/scene.hpp>
#include <cardinal/scene/assets.hpp>
#include <cardinal/actor/component.hpp>
#include <cardinal/core/diag/log.hpp>

#include <string>
#include <vector>

namespace {

namespace lv   = cardinal::level;
namespace ac   = cardinal::actor;
namespace geom = cardinal::core::geom;
using Vec3     = cardinal::scene::Vec3;
using cardinal::u32;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("lvltest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-4f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
// double coords so plain integer literals widen without C4244.
bool apv(const Vec3& v, double x, double y, double z, double e = 1e-4) {
    return ap(v.x, static_cast<float>(x), static_cast<float>(e)) &&
           ap(v.y, static_cast<float>(y), static_cast<float>(e)) &&
           ap(v.z, static_cast<float>(z), static_cast<float>(e));
}
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }
bool has(const std::vector<lv::HlodId>& v, lv::HlodId id) {
    for (auto x : v) if (x == id) return true;
    return false;
}

Vec3 v3(double x, double y, double z) {
    return Vec3{ static_cast<float>(x),
                 static_cast<float>(y),
                 static_cast<float>(z) };
}
lv::HlodInput hin(u32 id, double x, double y, double z) {
    lv::HlodInput h;
    h.id       = id;
    h.position = v3(x, y, z);
    h.bounds   = geom::AABB::make_empty();        // → from_center_extent path
    return h;
}

// ---- LevelManager: transform compose + deferred despawn -----------
void test_level_manager() {
    ac::World w;
    auto lm = lv::LevelManager::create(w);

    lv::LevelInstanceDesc desc;
    desc.name = "props";
    {
        lv::ActorTemplate t;
        t.name = "a1";
        t.translation = v3(1, 2, 3);
        t.scale       = v3(2, 2, 2);
        desc.actors.push_back(t);
    }
    {
        lv::ActorTemplate t;
        t.name = "a2";
        t.translation = v3(4, 0, 0);
        t.mesh_asset  = "mesh.box";
        t.tint        = v3(0.5, 0.5, 0.5);
        desc.actors.push_back(t);
    }
    auto inst = lv::LevelInstance::create(desc);
    CHECK(inst != nullptr);
    CHECK(inst->desc().name == "props");
    CHECK(inst->desc().actors.size() == sz(2));

    // spawn with a placement transform: t=(10,0,0), r=0, s=(3,1,1).
    const lv::InstanceId pid =
        lm->spawn(inst, v3(10, 0, 0), v3(0, 0, 0), v3(3, 1, 1));
    CHECK(pid == 1u);
    CHECK(lm->placement_count() == sz(1));

    const lv::Placement* p = lm->find(pid);
    CHECK(p != nullptr);
    CHECK(p->id == 1u);
    CHECK(p->instance == inst);
    CHECK(apv(p->translation, 10, 0, 0));
    CHECK(p->spawned_actor_ids.size() == sz(2));

    ac::Actor* a1 = w.find(p->spawned_actor_ids[0]);
    ac::Actor* a2 = w.find(p->spawned_actor_ids[1]);
    CHECK(a1 != nullptr && a1->name() == "a1");
    CHECK(a2 != nullptr && a2->name() == "a2");
    // a1: translation = (10+1*3, 0+2*1, 0+3*1); scale = (3*2,1*2,1*2).
    auto* tr1 = a1->get_component<ac::TransformComponent>();
    CHECK(tr1 != nullptr);
    CHECK(apv(tr1->translation, 13, 2, 3));
    CHECK(apv(tr1->scale,        6, 2, 2));
    CHECK(a1->get_component<ac::MeshComponent>() == nullptr);  // no mesh
    // a2: translation = (10+4*3, 0, 0); scale = (3,1,1); has Mesh.
    auto* tr2 = a2->get_component<ac::TransformComponent>();
    CHECK(apv(tr2->translation, 22, 0, 0));
    CHECK(apv(tr2->scale,        3, 1, 1));
    auto* mc = a2->get_component<ac::MeshComponent>();
    CHECK(mc != nullptr && mc->asset_id == "mesh.box");
    CHECK(apv(mc->tint, 0.5f, 0.5f, 0.5f));

    // null instance → 0, no placement added.
    CHECK(lm->spawn(nullptr) == 0u);
    CHECK(lm->placement_count() == sz(1));

    // Second placement is independent (new ids, own transform).
    const lv::InstanceId pid2 = lm->spawn(inst, v3(100, 0, 0));
    CHECK(pid2 == 2u);
    CHECK(lm->placement_count() == sz(2));
    CHECK(lm->placements().size() == sz(2));
    CHECK(lm->placements()[0]->id == 1u && lm->placements()[1]->id == 2u);

    // despawn(1): deferred — actors World::destroy()'d (alive=false, not
    // yet swept) and the placement dropped immediately.
    const u32 a1id = p->spawned_actor_ids[0];
    CHECK(lm->despawn(1u));
    CHECK(lm->find(1u) == nullptr);
    CHECK(lm->placement_count() == sz(1));
    ac::Actor* a1_still = w.find(a1id);
    CHECK(a1_still != nullptr && !a1_still->alive());     // dead, not swept
    w.sweep();
    CHECK(w.find(a1id) == nullptr);                       // now gone

    CHECK(!lm->despawn(999u));                            // unknown → false

    // clear() destroys remaining placement's actors + empties.
    lm->clear();
    CHECK(lm->placement_count() == sz(0));
    CHECK(lm->find(2u) == nullptr);
}

// ---- build_hlod: empty / single / hand-exact clustered tree -------
void test_build_hlod() {
    {   // Empty input → empty tree, no root.
        lv::HlodTree t = lv::build_hlod({});
        CHECK(t.nodes.empty());
        CHECK(t.root == lv::kInvalidHlodId);
        CHECK(lv::kInvalidHlodId == 0u);
        CHECK(t.find(1u) == nullptr);
    }
    {   // Single input → the leaf IS the root.
        std::vector<lv::HlodInput> in{ hin(500u, 1, 2, 3) };
        lv::HlodTree t = lv::build_hlod(in);
        CHECK(t.nodes.size() == sz(1));
        CHECK(t.root == 1u);
        const lv::HlodNode* n = t.find(1u);
        CHECK(n != nullptr);
        CHECK(n->is_leaf);
        CHECK(n->leaf_ids.size() == sz(1) && n->leaf_ids[0] == 500u);
        CHECK(n->parent == lv::kInvalidHlodId);
        CHECK(apv(n->centroid, 1, 2, 3));
        CHECK(ap(n->proxy_radius, 0.5f));                 // make_empty → 1³ box
        CHECK(t.find(2u) == nullptr && t.find(0u) == nullptr);
    }
    {   // 4 collinear inputs, cluster_size 2 → fully predictable tree:
        //   leaves 1..4 ; mids 5={1,2} 6={3,4} ; root 7={5,6}.
        std::vector<lv::HlodInput> in{
            hin(100u, 0,0,0), hin(200u, 1,0,0),
            hin(300u, 10,0,0), hin(400u, 11,0,0) };
        lv::HlodBuildOptions o; o.cluster_size = 2u; o.max_depth = 8u;
        lv::HlodTree t = lv::build_hlod(in, o);
        CHECK(t.nodes.size() == sz(7));
        CHECK(t.root == 7u);

        const lv::HlodNode* r = t.find(7u);
        CHECK(r && !r->is_leaf);
        CHECK(r->leaf_ids.empty());
        CHECK(r->parent == lv::kInvalidHlodId);
        CHECK(r->children.size() == sz(2));
        CHECK(has(r->children, 5u) && has(r->children, 6u));
        CHECK(apv(r->centroid, 5.5f, 0, 0));
        CHECK(ap(r->proxy_radius, 6.0f));                 // size.x 12 * 0.5
        CHECK(apv(r->bounds.size(), 12, 1, 1));

        const lv::HlodNode* m5 = t.find(5u);
        CHECK(m5 && !m5->is_leaf && m5->parent == 7u);
        CHECK(m5->children.size() == sz(2));
        CHECK(has(m5->children, 1u) && has(m5->children, 2u));
        CHECK(apv(m5->centroid, 0.5f, 0, 0));
        CHECK(ap(m5->proxy_radius, 1.0f));

        const lv::HlodNode* m6 = t.find(6u);
        CHECK(m6 && m6->parent == 7u);
        CHECK(has(m6->children, 3u) && has(m6->children, 4u));
        CHECK(apv(m6->centroid, 10.5f, 0, 0));

        const lv::HlodNode* l1 = t.find(1u);
        CHECK(l1 && l1->is_leaf && l1->parent == 5u);
        CHECK(l1->leaf_ids.size() == sz(1) && l1->leaf_ids[0] == 100u);
        CHECK(apv(l1->centroid, 0, 0, 0));
        CHECK(ap(l1->proxy_radius, 0.5f));
        const lv::HlodNode* l4 = t.find(4u);
        CHECK(l4 && l4->is_leaf && l4->parent == 6u);
        CHECK(l4->leaf_ids[0] == 400u);
        CHECK(t.find(99u) == nullptr);
    }
    {   // 8 inputs, default cluster_size 8 → one parent over all leaves.
        std::vector<lv::HlodInput> in;
        for (u32 i = 0; i < 8u; ++i)
            in.push_back(hin(10u + i, static_cast<double>(i), 0, 0));
        lv::HlodTree t = lv::build_hlod(in);              // default opts
        CHECK(t.nodes.size() == sz(9));
        CHECK(t.root == 9u);
        const lv::HlodNode* r = t.find(9u);
        CHECK(r && !r->is_leaf && r->children.size() == sz(8));
        CHECK(t.find(1u) && t.find(1u)->parent == 9u && t.find(1u)->is_leaf);
    }
    {   // NaN position must NOT defeat cluster_greedy's sort. vdist(NaN,
        // x) = NaN — the original `a.first < b.first` comparator is
        // NaN-blind both ways, so (NaN, x) is treated as equivalent
        // while (x, y) with x<y orders strictly. That breaks transitivity
        // of equivalence → std::sort SWO violation → UB on MSVC's
        // introsort (infinite loop or OOB scribble). Bad-data path:
        // corrupt scene file with `position = (nan, 0, 0)` (sscanf
        // "%f" accepts "nan" verbatim) → HlodInput → build_hlod →
        // cluster_greedy. The fix promotes NaN to "greater than all
        // finites" so NaN inputs cluster LAST (semantically "farthest")
        // and the sort completes cleanly. Build must terminate within
        // the test timeout and emit a coherent tree (one root, all
        // leaves present including the NaN one).
        volatile float z = 0.0f;
        const float nanv = z / z;   // qNaN without <cmath>
        std::vector<lv::HlodInput> in;
        in.reserve(8);
        // 7 well-spaced finite inputs + 1 NaN position smack in the
        // middle of the vector — the position most likely to trigger
        // partition-pathology on a SWO-violating comparator.
        in.push_back(hin(100u, 0,  0, 0));
        in.push_back(hin(200u, 2,  0, 0));
        in.push_back(hin(300u, 4,  0, 0));
        in.push_back(hin(400u, 6,  0, 0));
        lv::HlodInput nan_input;
        nan_input.id       = 999u;
        nan_input.position = Vec3{ nanv, 0.0f, 0.0f };
        nan_input.bounds   = geom::AABB::make_empty();
        in.push_back(nan_input);
        in.push_back(hin(500u, 8,  0, 0));
        in.push_back(hin(600u, 10, 0, 0));
        in.push_back(hin(700u, 12, 0, 0));
        lv::HlodBuildOptions o; o.cluster_size = 4u; o.max_depth = 8u;
        // Without the fix this can hang or heap-corrupt. With the fix,
        // returns in O(n log n) and the tree is well-formed.
        lv::HlodTree t = lv::build_hlod(in, o);
        CHECK(!t.nodes.empty());
        CHECK(t.root != lv::kInvalidHlodId);
        // All 8 input ids must appear as leaves in the tree (the NaN
        // one is grouped, not dropped — it's still data the engine
        // needs to track even if its position is degenerate).
        u32 leaves_found = 0;
        for (const auto& n : t.nodes) {
            if (!n.is_leaf) continue;
            for (lv::HlodId lid : n.leaf_ids) {
                if (lid == 100u || lid == 200u || lid == 300u ||
                    lid == 400u || lid == 500u || lid == 600u ||
                    lid == 700u || lid == 999u) ++leaves_found;
            }
        }
        CHECK(leaves_found == 8u);
    }
}

// ---- select_hlod: proxy / recurse-to-leaf / cull / empty ----------
void test_select_hlod() {
    std::vector<lv::HlodInput> in{
        hin(100u, 0,0,0), hin(200u, 1,0,0),
        hin(300u, 10,0,0), hin(400u, 11,0,0) };
    lv::HlodBuildOptions o; o.cluster_size = 2u; o.max_depth = 8u;
    lv::HlodTree t = lv::build_hlod(in, o);                // root=7 (as above)
    lv::HlodSelection out;

    // Camera very far → render the root proxy, no recursion.
    lv::select_hlod(t, v3(1000, 0, 0), /*leaf*/10.0f, /*proxy*/50.0f, out);
    CHECK(out.render_proxies.size() == sz(1) && out.render_proxies[0] == 7u);
    CHECK(out.render_leaves.empty());

    // Camera at origin, proxy_distance 0 → recurse near subtree to leaves,
    // far subtree (node 6) stays a proxy.
    lv::select_hlod(t, v3(0, 0, 0), /*leaf*/1000.0f, /*proxy*/0.0f, out);
    CHECK(out.render_proxies.size() == sz(1) && out.render_proxies[0] == 6u);
    CHECK(out.render_leaves.size() == sz(2));
    CHECK(out.render_leaves[0] == 100u && out.render_leaves[1] == 200u);

    // Tight leaf_distance culls the farther near-leaf (node 2, d=0.5).
    lv::select_hlod(t, v3(0, 0, 0), /*leaf*/0.0f, /*proxy*/0.0f, out);
    CHECK(out.render_leaves.size() == sz(1) && out.render_leaves[0] == 100u);
    CHECK(out.render_proxies.size() == sz(1) && out.render_proxies[0] == 6u);

    // Empty tree → nothing, and the out vectors are cleared.
    lv::HlodTree empty = lv::build_hlod({});
    out.render_proxies.push_back(123u);                    // pre-dirty
    lv::select_hlod(empty, v3(0,0,0), 1.0f, 1.0f, out);
    CHECK(out.render_proxies.empty() && out.render_leaves.empty());

    // Single-leaf-root: rendered iff within leaf_distance.
    std::vector<lv::HlodInput> one{ hin(500u, 1, 2, 3) };
    lv::HlodTree st = lv::build_hlod(one);
    lv::select_hlod(st, v3(1, 2, 3), /*leaf*/10.0f, /*proxy*/0.0f, out);
    CHECK(out.render_leaves.size() == sz(1) && out.render_leaves[0] == 500u);
    CHECK(out.render_proxies.empty());
    lv::select_hlod(st, v3(1000, 0, 0), /*leaf*/1.0f, /*proxy*/0.0f, out);
    CHECK(out.render_leaves.empty() && out.render_proxies.empty());
}

}  // namespace

// ---- AssetPlacement: actor-authoritative placement + scene mirror ----
void test_asset_placement() {
    // Headless test asset (no rhi::Device): factory just adds a scene
    // entity at ctx.position. register_asset is id-idempotent.
    {
        cardinal::scene::AssetDesc d{};
        d.id      = "test.box";
        d.label   = "Test Box";
        d.category= "Test";
        d.factory = [](const cardinal::scene::AssetSpawnContext& ctx)
                        -> cardinal::scene::AssetSpawnResult {
            if (ctx.scene == nullptr) return {};
            auto& e = ctx.scene->add_entity("TestBox");
            e.transform.translation = ctx.position;
            e.tint = { 1.0f, 0.0f, 0.0f };
            return cardinal::scene::AssetSpawnResult{ { e.id }, e.id };
        };
        cardinal::scene::AssetCatalog::instance().register_asset(
            cardinal::move(d));
    }

    ac::World          w;
    cardinal::scene::Scene scene;
    auto pl = lv::AssetPlacement::create(w, scene);
    CHECK(pl != nullptr);

    const auto r = pl->place("test.box", nullptr, v3(5, 0, -3));
    CHECK(r.actor != 0u);
    CHECK(r.primary_entity != 0u);
    CHECK(pl->count() == sz(1));

    // The placed asset is a first-class actor (Outliner/Game/serial see
    // it) carrying Transform + Mesh + "placed" Tag.
    ac::Actor* a = w.find(r.actor);
    CHECK(a != nullptr);
    auto* tc = a ? a->get_component<ac::TransformComponent>() : nullptr;
    auto* mc = a ? a->get_component<ac::MeshComponent>()      : nullptr;
    auto* tg = a ? a->get_component<ac::TagComponent>()       : nullptr;
    CHECK(tc != nullptr && apv(tc->translation, 5, 0, -3));
    CHECK(mc != nullptr && mc->asset_id == "test.box");
    CHECK(tg != nullptr && tg->has("placed"));
    CHECK(scene.find_by_id(r.primary_entity) != nullptr);

    // Studio gizmo edits the scene entity -> sync_from_scene pulls it
    // onto the authoritative actor (so save-load / Outliner track it).
    scene.find_by_id(r.primary_entity)->transform.translation = v3(9, 1, 2);
    pl->sync_from_scene();
    CHECK(tc != nullptr && apv(tc->translation, 9, 1, 2));

    // Programmatic actor edit (Outliner / serial-load) -> sync_to_scene
    // pushes it to the render mirror.
    tc->translation = v3(-4, 6, 8);
    pl->sync_to_scene();
    CHECK(apv(scene.find_by_id(r.primary_entity)->transform.translation,
              -4, 6, 8));

    // Remove unwinds both the actor and the render entity.
    CHECK(pl->remove(r.actor));
    CHECK(pl->count() == sz(0));
    CHECK(scene.find_by_id(r.primary_entity) == nullptr);
    ac::Actor* dead = w.find(r.actor);
    CHECK(dead == nullptr || !dead->alive());
}

int main() {
    test_level_manager();
    test_build_hlod();
    test_select_hlod();
    test_asset_placement();

    if (g_fail == 0) {
        cardinal::log::infof("lvltest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("lvltest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
