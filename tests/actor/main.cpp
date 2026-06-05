// =============================================================================
// Cardinal — deterministic actor-system regression suite.
//
// Actor + Component + World are the entity backbone SimWorld iterates.
// Pinned:
//   * Actor id/name/parent/alive, spawn auto-adds a TransformComponent;
//   * add_component fires on_attach immediately and is NOT de-duped
//     (doc says "one per type" but the code pushes; get_component returns
//     the FIRST match by type_name) — actual behaviour locked;
//   * the component LIFECYCLE via external counters: on_attach at add,
//     on_tick per Actor::tick AND World::tick (live only), on_detach at
//     destruction; a dead Actor never ticks;
//   * the deferred destroy → sweep contract: kill() marks dead but
//     find()/actor_count() still see it and tick() skips it; sweep()
//     physically removes it and fires on_detach; ids never reused;
//   * blueprints (factory + build, missing→null, sorted names, replace,
//     unregister); the synchronous ordered event bus (payload,
//     unsubscribe, unknown-event no-op, monotonic ids across events);
//   * TagComponent add-dedupe / has / remove; built-in defaults;
//   * TransformComponent::matrix() determinism + default == identity.
//
// Pure CPU, headless, fully deterministic. Exit 0 = all pass.
// =============================================================================

#include <cardinal/actor/world.hpp>
#include <cardinal/actor/builtin_prefabs.hpp>
#include <cardinal/actor/validation.hpp>
#include <cardinal/actor/scene_stats.hpp>
#include <cardinal/core/log.hpp>

#include <any>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace ac = cardinal::actor;
using Mat4   = cardinal::scene::Mat4;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("actortest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-5f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }
bool streq(const char* a, const char* b) { return std::string(a) == b; }
bool mat_eq(const Mat4& A, const Mat4& B) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (A.m[i][j] != B.m[i][j]) return false;
    return true;
}
// volatile-launder a qNaN without dragging <cmath>/<limits> in.
float nan_f() { volatile float z = 0.0f; return z / z; }
// NaN never compares equal to itself — handy for "is finite" without
// pulling in <cmath>.
bool finite_eq(float v, float want) {
    if (v != v) return false;
    return ap(v, want);
}

// External-counter component so the lifecycle is observable AFTER the
// component (and its Actor) are destroyed. Default ctor exists so
// Actor::get_component<CounterComp>() (which does `T{}.type_name()`) works.
struct CounterComp : ac::Component {
    int*   a{nullptr};
    int*   t{nullptr};
    int*   d{nullptr};
    float* ld{nullptr};
    CounterComp() = default;
    CounterComp(int* a_, int* t_, int* d_, float* ld_)
        : a(a_), t(t_), d(d_), ld(ld_) {}
    const char* type_name() const noexcept override { return "Counter"; }
    void on_attach(ac::Actor&) override            { if (a) ++*a; }
    void on_tick  (ac::Actor&, float dt) override  { if (t) ++*t; if (ld) *ld = dt; }
    void on_detach(ac::Actor&) override            { if (d) ++*d; }
};

// A component whose on_tick spawns into the World — the canonical
// "spawn from a tick" gameplay pattern that reallocates World::actors_
// mid-iteration.
struct SpawnerComp : ac::Component {
    ac::World* w{nullptr};
    int*       spawns{nullptr};
    SpawnerComp() = default;
    SpawnerComp(ac::World* w_, int* s_) : w(w_), spawns(s_) {}
    const char* type_name() const noexcept override { return "Spawner"; }
    void on_tick(ac::Actor&, float) override {
        if (!w || !spawns) return;
        if (*spawns == 0)
            for (int k = 0; k < 512; ++k) w->spawn("spawned-in-tick");
        ++*spawns;
    }
};

// ---- spawn-during-tick must not invalidate the tick iterator ------
// Regression: World::tick / Actor::tick used range-for over a vector
// whose element-add (world.spawn / actor.add_component — an extremely
// common pattern from a tick) reallocates it mid-iteration. The freed
// old buffer holds moved-from (null) unique_ptrs, so the next
// `a->alive()` is a DETERMINISTIC null-deref crash. Index-snapshot
// iteration is realloc-safe and defers freshly-spawned actors to the
// next frame (the intended begin_play-before-first-tick semantics).
void test_spawn_during_tick() {
    ac::World w;
    const int kOrig = 8;
    cardinal::vector<int> ticks(static_cast<cardinal::usize>(kOrig), 0);
    int spawns = 0;

    ac::Actor* first = w.spawn("orig0");                 // index 0 = spawner
    first->add_component<SpawnerComp>(&w, &spawns);
    first->add_component<CounterComp>(nullptr, &ticks[0], nullptr, nullptr);
    for (int i = 1; i < kOrig; ++i) {
        ac::Actor* a = w.spawn("orig");
        a->add_component<CounterComp>(nullptr,
            &ticks[static_cast<cardinal::usize>(i)], nullptr, nullptr);
    }
    const cardinal::usize before = w.actor_count();
    CHECK(before == sz(kOrig));

    w.tick(0.016f);   // PRE-FIX: actors_ reallocs mid-iter → null-deref crash

    CHECK(spawns == 1);
    CHECK(w.actor_count() == before + sz(512));          // whole batch added
    bool all_ticked_once = true;
    for (int i = 0; i < kOrig; ++i)
        if (ticks[static_cast<cardinal::usize>(i)] != 1) all_ticked_once = false;
    CHECK(all_ticked_once);                              // every original ×1
    CHECK(w.find_by_name("spawned-in-tick") != nullptr); // it exists...
    // ...but a just-spawned actor must NOT have ticked this frame: no
    // CounterComp on them, and the count proves they were deferred.

    // Second tick: the 512 new actors now iterate too — still no crash,
    // no further batch (spawns>0), originals tick again exactly once.
    w.tick(0.016f);
    CHECK(spawns == 2);
    CHECK(w.actor_count() == before + sz(512));          // unchanged
    bool all_ticked_twice = true;
    for (int i = 0; i < kOrig; ++i)
        if (ticks[static_cast<cardinal::usize>(i)] != 2) all_ticked_twice = false;
    CHECK(all_ticked_twice);
}

// ---- Actor + component management ---------------------------------
void test_actor_components() {
    ac::World w;
    ac::Actor* a = w.spawn("hero");
    CHECK(a != nullptr);
    CHECK(a->id() == 1u);
    CHECK(a->name() == "hero");
    CHECK(a->alive());
    CHECK(a->parent() == ac::kInvalidActor);
    CHECK(ac::kInvalidActor == 0u);

    // spawn() auto-attaches exactly one TransformComponent.
    CHECK(a->components().size() == sz(1));
    auto* tr = a->get_component<ac::TransformComponent>();
    CHECK(tr != nullptr && streq(tr->type_name(), "Transform"));
    CHECK(a->get_component<ac::CameraComponent>() == nullptr);   // absent

    a->set_name("renamed");  CHECK(a->name() == "renamed");
    a->set_parent(7u);       CHECK(a->parent() == 7u);
    a->kill();               CHECK(!a->alive());

    ac::Actor* b = w.spawn("b");
    CHECK(b->id() == 2u);                                        // monotonic

    auto* rb = b->add_component<ac::RigidBodyComponent>();
    CHECK(rb != nullptr);
    CHECK(ap(rb->mass, 1.0f) && ap(rb->linear_damping, 0.05f));
    CHECK(rb->use_gravity && !rb->kinematic);
    CHECK(b->get_component<ac::RigidBodyComponent>() == rb);     // same object
    CHECK(b->components().size() == sz(2));                      // Transform+RB

    rb->mass = 5.0f;                                             // mutate via ptr
    CHECK(ap(b->get_component<ac::RigidBodyComponent>()->mass, 5.0f));

    // NOT de-duped: a second RigidBody is appended; get_component returns
    // the FIRST one (still mass 5), size grows.
    b->add_component<ac::RigidBodyComponent>();
    CHECK(b->components().size() == sz(3));
    CHECK(ap(b->get_component<ac::RigidBodyComponent>()->mass, 5.0f));

    // Built-in type names + defaults.
    ac::MeshComponent mc;   CHECK(streq(mc.type_name(), "Mesh"));
    CHECK(mc.visible && ap(mc.tint.x, 1.0f));
    ac::CameraComponent cc; CHECK(streq(cc.type_name(), "Camera") && !cc.active);
    ac::LightComponent lc;  CHECK(streq(lc.type_name(), "Light"));
    CHECK(lc.kind == ac::LightKind::Directional);
    ac::AudioEmitterComponent au;
    CHECK(streq(au.type_name(), "AudioEmitter"));
    CHECK(ap(au.volume,1.0f) && ap(au.pitch,1.0f) && !au.loop && au.is_3d);
    ac::ScriptComponent sc; CHECK(streq(sc.type_name(), "Script") && sc.enabled);
    ac::TagComponent tg;    CHECK(streq(tg.type_name(), "Tag"));
    CHECK(static_cast<cardinal::u32>(ac::LightKind::Directional) == 0u);
    CHECK(static_cast<cardinal::u32>(ac::LightKind::Point)       == 1u);
    CHECK(static_cast<cardinal::u32>(ac::LightKind::Spot)        == 2u);
}

// ---- PlayerController must not poison translation on non-finite dt ---
// `if (dt < 0.0f) dt = 0.0f;` was NaN-blind (NaN < 0 is false). Without
// the `!(dt > 0.0f)` fix, `disp = wish * (inv * spd * NaN)` is a NaN
// vector, `tr->translation += disp` teleports the player to NaN-land,
// `vy_ += gravity * NaN` and `tr->translation.y += vy_ * NaN` make
// every translation component permanently NaN. The fix clamps NaN dt
// to 0 same as the existing negative-dt clamp.
void test_player_controller_nonfinite_dt() {
    ac::World w;
    ac::Actor* a = w.spawn("player");
    auto* pc = a->add_component<ac::PlayerControllerComponent>();
    auto* tr = a->get_component<ac::TransformComponent>();
    CHECK(pc != nullptr && tr != nullptr);

    // Finite tick with W held — player moves. Capture the post-tick
    // translation so we can pin "NaN tick changes nothing" precisely.
    ac::PlayerInput in{};
    in.accept_input = true;
    in.move_z       = 1.0f;     // hold W (forward)
    pc->tick(0.5f, in);
    const float px = tr->translation.x;
    const float py = tr->translation.y;
    const float pz = tr->translation.z;
    // Finite tick produced finite translation (sanity).
    CHECK(finite_eq(px, px));   // NaN-trap: NaN != NaN
    CHECK(finite_eq(py, py));
    CHECK(finite_eq(pz, pz));

    // NaN dt — translation MUST be unchanged (dt clamped to 0).
    pc->tick(nan_f(), in);
    CHECK(ap(tr->translation.x, px));
    CHECK(ap(tr->translation.y, py));
    CHECK(ap(tr->translation.z, pz));

    // ±Inf dt — same: clamped to 0 by the isfinite guard. Construct via
    // volatile-launder to defeat constant folding.
    volatile float big = 1.0f; for (int i = 0; i < 16; ++i) big *= 1e30f;  // +Inf
    pc->tick(big, in);                                // +Inf
    CHECK(ap(tr->translation.x, px));
    CHECK(ap(tr->translation.y, py));
    CHECK(ap(tr->translation.z, pz));
    pc->tick(-big, in);                               // -Inf
    CHECK(ap(tr->translation.x, px));
    CHECK(ap(tr->translation.y, py));
    CHECK(ap(tr->translation.z, pz));
}

// ---- PlayerController direct-field tunables must not poison state --
// PlayerControllerComponent exposes move_speed / sprint_multiplier /
// look_sensitivity / gravity / jump_speed as public direct fields with
// no setters. Same desc-direct-field NaN sink class as scene::FlyCamera
// (d8f7ce1) and particles::EmitterDesc (3704b48). Each tunable feeds
// the integrator's persistent state and poisons it permanently on NaN:
//   * look_sensitivity NaN → yaw_/pitch_ NaN forever (cos/sin of NaN
//     stays NaN; rotation never recovers).
//   * gravity NaN → `vy_ += NaN * dt` poisons vy_ persistently; every
//     later tick reads NaN vy_ and writes NaN translation.y.
//   * move_speed / sprint_multiplier NaN → translation.x/z poisoned.
//   * jump_speed NaN → vy_ = NaN on jump frame; persistent poison.
void test_player_controller_nonfinite_tunables() {
    ac::World w;

    {   // look_sensitivity NaN with a mouse look: yaw_ must NOT become
        // NaN. With the sanitized default (0.0035) yaw advances by
        // mouse_dx * 0.0035.
        ac::Actor* a = w.spawn("p1");
        auto* pc = a->add_component<ac::PlayerControllerComponent>();
        pc->look_sensitivity = nan_f();
        ac::PlayerInput in{};
        in.accept_input = true; in.look = true; in.mouse_dx = 100.0f;
        pc->tick(0.0f, in);                                // dt=0, look only
        CHECK(finite_eq(pc->yaw_rad(), pc->yaw_rad()));    // not NaN
        // User's stored value preserved (still NaN) — editor visibility.
        CHECK(pc->look_sensitivity != pc->look_sensitivity);
    }
    {   // move_speed NaN with W held: translation.z must NOT become NaN.
        ac::Actor* a = w.spawn("p2");
        auto* pc = a->add_component<ac::PlayerControllerComponent>();
        auto* tr = a->get_component<ac::TransformComponent>();
        pc->move_speed = nan_f();
        ac::PlayerInput in{};
        in.accept_input = true; in.move_z = 1.0f;
        pc->tick(0.5f, in);
        CHECK(finite_eq(tr->translation.x, tr->translation.x));
        CHECK(finite_eq(tr->translation.y, tr->translation.y));
        CHECK(finite_eq(tr->translation.z, tr->translation.z));
        CHECK(pc->move_speed != pc->move_speed);            // preserved
    }
    {   // gravity NaN: vy_ MUST NOT become NaN (it's persistent across
        // ticks). After many finite ticks under fake-gravity, the
        // player should still have finite translation.y.
        ac::Actor* a = w.spawn("p3");
        auto* pc = a->add_component<ac::PlayerControllerComponent>();
        auto* tr = a->get_component<ac::TransformComponent>();
        pc->gravity = nan_f();
        ac::PlayerInput in{};
        in.accept_input = true;
        // 10 ticks of gravity integration with NaN gravity field —
        // safe default -19.62 used; player falls; translation.y stays
        // finite throughout (would go NaN on the very first tick
        // without the fix).
        for (int i = 0; i < 10; ++i) pc->tick(0.016f, in);
        CHECK(finite_eq(tr->translation.y, tr->translation.y));
        CHECK(pc->gravity != pc->gravity);                  // preserved
    }
    {   // sprint_multiplier NaN with sprint=true held — same protection.
        ac::Actor* a = w.spawn("p4");
        auto* pc = a->add_component<ac::PlayerControllerComponent>();
        auto* tr = a->get_component<ac::TransformComponent>();
        pc->move_speed        = 1.0f;
        pc->sprint_multiplier = nan_f();
        ac::PlayerInput in{};
        in.accept_input = true; in.move_z = 1.0f; in.sprint = true;
        pc->tick(0.5f, in);
        CHECK(finite_eq(tr->translation.x, tr->translation.x));
        CHECK(finite_eq(tr->translation.z, tr->translation.z));
        CHECK(pc->sprint_multiplier != pc->sprint_multiplier);
    }
}

// ---- TagComponent: add-dedupe / has / remove ----------------------
void test_tag_component() {
    ac::TagComponent tc;
    CHECK(tc.flags == 0u && tc.tags.empty());
    tc.add("a"); tc.add("b"); tc.add("a");              // dup ignored
    CHECK(tc.tags.size() == sz(2));
    CHECK(tc.has("a") && tc.has("b") && !tc.has("c"));
    tc.remove("a");
    CHECK(!tc.has("a") && tc.tags.size() == sz(1));
    tc.remove("zzz");                                    // no-op
    CHECK(tc.tags.size() == sz(1));
    tc.add("a");
    CHECK(tc.has("a") && tc.tags.size() == sz(2));
}

// ---- component lifecycle (attach / tick / detach) -----------------
void test_lifecycle() {
    int att = 0, tk = 0, det = 0;
    float ld = -1.0f;
    ac::World w;
    ac::Actor* a = w.spawn("x");
    a->add_component<CounterComp>(&att, &tk, &det, &ld);
    CHECK(att == 1 && tk == 0 && det == 0);             // on_attach @ add
    CHECK(a->get_component<CounterComp>() != nullptr);  // default-ctor T{} ok

    a->tick(0.5f);
    CHECK(tk == 1 && ap(ld, 0.5f));
    w.tick(0.25f);                                       // live → ticks
    CHECK(tk == 2 && ap(ld, 0.25f));

    a->kill();
    w.tick(0.1f);                                        // World skips dead
    CHECK(tk == 2);
    a->tick(0.1f);                                       // Actor::tick guards
    CHECK(tk == 2);

    w.sweep();                                           // ~Actor → on_detach
    CHECK(det == 1);
    CHECK(att == 1 && tk == 2);                          // counters survive
    CHECK(w.actor_count() == sz(0));
}

// ---- World spawn / destroy / deferred sweep -----------------------
void test_world_lifecycle() {
    ac::World w;
    ac::Actor* a1 = w.spawn("a");
    ac::Actor* a2 = w.spawn("b");
    ac::Actor* a3 = w.spawn("c");
    CHECK(a1->id()==1u && a2->id()==2u && a3->id()==3u);
    CHECK(w.actor_count() == sz(3));

    CHECK(w.find(2u) == a2);
    CHECK(w.find(99u) == nullptr);
    CHECK(w.find_by_name("c") == a3);
    CHECK(w.find_by_name("nope") == nullptr);

    // destroy() marks dead but does NOT remove until sweep().
    w.destroy(2u);
    CHECK(!a2->alive());
    CHECK(w.find(2u) == a2);                             // still present
    CHECK(w.actor_count() == sz(3));                     // not yet swept
    w.destroy(999u);                                     // unknown → no-op
    w.destroy(2u);                                       // already dead → no-op

    w.sweep();
    CHECK(w.actor_count() == sz(2));
    CHECK(w.find(2u) == nullptr);                        // physically gone
    CHECK(w.find(1u) != nullptr && w.find(3u) != nullptr);

    // ids are never reused after a sweep.
    ac::Actor* a4 = w.spawn("d");
    CHECK(a4->id() == 4u);
}

// ---- blueprints ---------------------------------------------------
void test_blueprints() {
    ac::World w;
    ac::Blueprint bp;
    bp.name  = "bp.box";
    bp.build = [](ac::Actor& a) {
        a.add_component<ac::RigidBodyComponent>();
        a.set_name("Box");
    };
    w.register_blueprint(bp);
    CHECK(w.find_blueprint("bp.box") != nullptr);
    CHECK(w.find_blueprint("bp.box")->name == "bp.box");
    CHECK(w.find_blueprint("nope") == nullptr);

    ac::Actor* a = w.spawn_blueprint("bp.box");
    CHECK(a != nullptr);
    CHECK(a->id() == 1u);
    CHECK(a->name() == "Box");                           // build's set_name won
    CHECK(a->get_component<ac::RigidBodyComponent>() != nullptr);
    CHECK(a->get_component<ac::TransformComponent>() != nullptr); // spawn auto

    CHECK(w.spawn_blueprint("missing") == nullptr);

    ac::Blueprint z; z.name = "bp.alpha"; z.build = [](ac::Actor&){};
    ac::Blueprint y; y.name = "bp.zeta";  y.build = [](ac::Actor&){};
    w.register_blueprint(z);
    w.register_blueprint(y);
    auto names = w.blueprint_names();
    CHECK(names.size() == sz(3));
    CHECK(names[0]=="bp.alpha" && names[1]=="bp.box" && names[2]=="bp.zeta");

    // Re-register same name replaces (no growth); unregister removes.
    ac::Blueprint bp2; bp2.name = "bp.box"; bp2.build = [](ac::Actor&){};
    w.register_blueprint(bp2);
    CHECK(w.blueprint_names().size() == sz(3));
    w.unregister_blueprint("bp.box");
    CHECK(w.find_blueprint("bp.box") == nullptr);
    CHECK(w.blueprint_names().size() == sz(2));
}

// ---- prefabs (capture live actor → template → stamp instances) ----
void test_prefabs() {
    ac::World w;

    // Configure a source actor with several components + authored values.
    ac::Actor* src = w.spawn("Crate");                     // auto-Transform
    auto* st = src->get_component<ac::TransformComponent>();
    st->translation = { 5.0f, 1.0f, -3.0f };
    st->scale       = { 2.0f, 2.0f, 2.0f };
    auto* sm = src->add_component<ac::MeshComponent>();
    sm->asset_id = "crate.mesh";
    sm->tint     = { 0.8f, 0.6f, 0.2f };
    auto* sr = src->add_component<ac::RigidBodyComponent>();
    sr->mass = 12.0f; sr->use_gravity = true;
    auto* stag = src->add_component<ac::TagComponent>();
    stag->add("pickup"); stag->add("breakable");

    // Capture → prefab. 4 components (Transform + Mesh + RigidBody + Tag).
    CHECK(w.create_prefab("Crate", src->id()));
    CHECK(w.has_prefab("Crate"));
    CHECK(w.prefab_component_count("Crate") == 4u);
    CHECK(!w.has_prefab("nope"));

    // Capturing a non-existent source fails cleanly.
    CHECK(!w.create_prefab("Bad", 9999u));

    // Stamp an instance — independent actor, distinct id, cloned values.
    ac::Actor* inst = w.spawn_prefab("Crate");
    CHECK(inst != nullptr);
    CHECK(inst->id() != src->id());
    CHECK(inst->name() == "Crate (instance)");
    // The 4 captured components + 1 PrefabLink (spawn_prefab tags every
    // instance with its lineage) — and no duplicate auto-Transform.
    CHECK(inst->components().size() == sz(5));
    CHECK(inst->get_component<ac::PrefabLinkComponent>() != nullptr);

    auto* it = inst->get_component<ac::TransformComponent>();
    CHECK(it != nullptr);
    CHECK(it->translation.x == 5.0f && it->translation.z == -3.0f);
    CHECK(it->scale.x == 2.0f);
    auto* im = inst->get_component<ac::MeshComponent>();
    CHECK(im != nullptr && im->asset_id == "crate.mesh");
    CHECK(im->tint.y == 0.6f);
    auto* ir = inst->get_component<ac::RigidBodyComponent>();
    CHECK(ir != nullptr && ir->mass == 12.0f);
    auto* itag = inst->get_component<ac::TagComponent>();
    CHECK(itag != nullptr && itag->has("pickup") && itag->has("breakable"));

    // Clone independence — mutating the instance must NOT touch the source
    // or a second instance.
    it->translation = { 99.0f, 0.0f, 0.0f };
    im->asset_id    = "other.mesh";
    CHECK(src->get_component<ac::TransformComponent>()->translation.x == 5.0f);
    CHECK(src->get_component<ac::MeshComponent>()->asset_id == "crate.mesh");

    ac::Actor* inst2 = w.spawn_prefab("Crate", "Crate #2");
    CHECK(inst2 != nullptr);
    CHECK(inst2->name() == "Crate #2");
    CHECK(inst2->get_component<ac::TransformComponent>()->translation.x == 5.0f);
    CHECK(inst2->get_component<ac::MeshComponent>()->asset_id == "crate.mesh");

    // Instances are real, live, tickable actors in the world.
    CHECK(w.find(inst->id())  == inst);
    CHECK(w.find(inst2->id()) == inst2);

    // Prefab listing + removal.
    CHECK(w.create_prefab("Barrel", src->id()));
    auto names = w.prefab_names();
    CHECK(names.size() == sz(2));
    CHECK(names[0] == "Barrel" && names[1] == "Crate");   // sorted
    w.remove_prefab("Crate");
    CHECK(!w.has_prefab("Crate"));
    CHECK(w.spawn_prefab("Crate") == nullptr);            // gone → null
    CHECK(w.prefab_names().size() == sz(1));

    // RigidBody runtime state (velocity) is NOT carried by the clone —
    // a stamped instance starts at rest even if the source was moving.
    sr->velocity = { 10.0f, 0.0f, 0.0f };
    CHECK(w.create_prefab("MovingCrate", src->id()));
    ac::Actor* rest = w.spawn_prefab("MovingCrate");
    CHECK(rest->get_component<ac::RigidBodyComponent>()->velocity.x == 0.0f);
}

// ---- prefab library management (rename / duplicate) ---------------
void test_prefab_library_mgmt() {
    ac::World w;
    ac::Actor* src = w.spawn("Barrel");
    src->add_component<ac::MeshComponent>()->asset_id = "barrel";
    src->add_component<ac::RigidBodyComponent>();
    CHECK(w.create_prefab("Barrel", src->id()));
    CHECK(w.prefab_component_count("Barrel") == sz(3));   // Transform+Mesh+RigidBody

    // Rename: key moves, component set preserved, old name gone.
    CHECK(w.rename_prefab("Barrel", "Crate"));
    CHECK(!w.has_prefab("Barrel"));
    CHECK(w.has_prefab("Crate"));
    CHECK(w.prefab_component_count("Crate") == sz(3));
    // Rename failure modes: missing source, empty target, taken target.
    CHECK(!w.rename_prefab("Barrel", "X"));               // source gone
    CHECK(!w.rename_prefab("Crate", ""));                 // empty target
    CHECK(w.create_prefab("Other", src->id()));
    CHECK(!w.rename_prefab("Crate", "Other"));            // target taken
    CHECK(w.has_prefab("Crate") && w.has_prefab("Other")); // both intact

    // Duplicate with auto-unique name.
    CHECK(w.duplicate_prefab("Crate"));                   // -> "Crate (copy)"
    CHECK(w.has_prefab("Crate (copy)"));
    CHECK(w.prefab_component_count("Crate (copy)") == sz(3));
    CHECK(w.duplicate_prefab("Crate"));                   // -> "Crate (copy 2)"
    CHECK(w.has_prefab("Crate (copy 2)"));

    // Duplicate with an explicit name; clone is independent of the source.
    CHECK(w.duplicate_prefab("Crate", "CrateV2"));
    CHECK(w.has_prefab("CrateV2"));
    // Editing an instance of the dup must not affect the original prefab.
    ac::Actor* v2 = w.spawn_prefab("CrateV2");
    v2->get_component<ac::MeshComponent>()->asset_id = "edited";
    ac::Actor* orig = w.spawn_prefab("Crate");
    CHECK(orig->get_component<ac::MeshComponent>()->asset_id == "barrel");

    // Duplicate failure modes.
    CHECK(!w.duplicate_prefab("Nope"));                   // no source
    CHECK(!w.duplicate_prefab("Crate", "Other"));         // explicit name taken
}

// ---- group prefabs (multi-actor templates) ------------------------
void test_group_prefab() {
    ac::World w;
    // A 3-actor cluster: a base at (10,0,0), a barrel +0,1,0 above it, a
    // sensor +0,2,1. Anchor is the FIRST captured actor (base).
    ac::Actor* base = w.spawn("Base");
    base->get_component<ac::TransformComponent>()->translation = { 10.0f, 0.0f, 0.0f };
    base->add_component<ac::MeshComponent>()->asset_id = "base";
    ac::Actor* barrel = w.spawn("Barrel");
    barrel->get_component<ac::TransformComponent>()->translation = { 10.0f, 1.0f, 0.0f };
    barrel->add_component<ac::MeshComponent>()->asset_id = "barrel";
    ac::Actor* sensor = w.spawn("Sensor");
    sensor->get_component<ac::TransformComponent>()->translation = { 10.0f, 2.0f, 1.0f };
    sensor->add_component<ac::LightComponent>();

    cardinal::vector<cardinal::u32> ids = { base->id(), barrel->id(), sensor->id() };
    CHECK(w.create_group_prefab("Turret", ids));
    CHECK(w.has_group_prefab("Turret"));
    CHECK(w.group_prefab_member_count("Turret") == 3u);

    // Spawn the group at a new location; relative layout preserved + offset.
    const cardinal::usize before = w.actor_count();
    auto inst = w.spawn_group_prefab("Turret", { 100.0f, 0.0f, 0.0f });
    CHECK(inst.size() == sz(3));
    CHECK(w.actor_count() == before + sz(3));

    // Member 0 (base) at the anchor -> exactly `at`.
    auto p0 = inst[0]->get_component<ac::TransformComponent>()->translation;
    CHECK(ap(p0.x, 100.0f) && ap(p0.y, 0.0f) && ap(p0.z, 0.0f));
    // Member 1 (barrel) kept its +0,1,0 relative offset.
    auto p1 = inst[1]->get_component<ac::TransformComponent>()->translation;
    CHECK(ap(p1.x, 100.0f) && ap(p1.y, 1.0f) && ap(p1.z, 0.0f));
    // Member 2 (sensor) kept +0,2,1.
    auto p2 = inst[2]->get_component<ac::TransformComponent>()->translation;
    CHECK(ap(p2.x, 100.0f) && ap(p2.y, 2.0f) && ap(p2.z, 1.0f));
    // Components are full clones.
    CHECK(inst[0]->get_component<ac::MeshComponent>()->asset_id == "base");
    CHECK(inst[1]->get_component<ac::MeshComponent>()->asset_id == "barrel");
    CHECK(inst[2]->get_component<ac::LightComponent>() != nullptr);

    // Independence — editing a spawned member doesn't change the source.
    inst[0]->get_component<ac::MeshComponent>()->asset_id = "edited";
    CHECK(base->get_component<ac::MeshComponent>()->asset_id == "base");

    // A 2nd spawn is independent of the 1st.
    auto inst2 = w.spawn_group_prefab("Turret", { 0.0f, 50.0f, 0.0f });
    CHECK(inst2.size() == sz(3));
    CHECK(ap(inst2[1]->get_component<ac::TransformComponent>()->translation.y, 51.0f));

    // PrefabLink is NOT captured into a group template (config, not lineage).
    ac::Actor* linkedActor = nullptr;
    for (auto* x : inst) {
        if (x->get_component<ac::PrefabLinkComponent>()) linkedActor = x;
    }
    CHECK(linkedActor == nullptr);

    // Listing + removal + count.
    CHECK(w.create_group_prefab("Pair", { base->id(), barrel->id() }));
    auto names = w.group_prefab_names();
    CHECK(names.size() == sz(2));
    CHECK(names[0] == "Pair" && names[1] == "Turret");   // sorted
    w.remove_group_prefab("Turret");
    CHECK(!w.has_group_prefab("Turret"));
    CHECK(w.spawn_group_prefab("Turret", { 0, 0, 0 }).empty());   // gone

    // Edge cases: empty id set + unknown name.
    CHECK(!w.create_group_prefab("Bad", {}));
    CHECK(!w.create_group_prefab("Bad", { 99999u }));    // no valid actors
    CHECK(w.spawn_group_prefab("Nope", { 0, 0, 0 }).empty());
}

// ---- starter prefab library ---------------------------------------
void test_builtin_prefabs() {
    ac::World w;
    const cardinal::u32 added = ac::register_builtin_prefabs(w);
    CHECK(added == 7u);                              // 7 starter prefabs

    // All canonical names present.
    CHECK(w.has_prefab("Point Light"));
    CHECK(w.has_prefab("Directional Light"));
    CHECK(w.has_prefab("Spot Light"));
    CHECK(w.has_prefab("Physics Cube"));
    CHECK(w.has_prefab("Camera"));
    CHECK(w.has_prefab("Trigger Volume"));
    CHECK(w.has_prefab("Player Start"));

    // Component composition spot-checks.
    CHECK(w.prefab_component_count("Point Light") == sz(2));   // Transform + Light
    CHECK(w.prefab_component_count("Physics Cube") == sz(3));  // Transform + Mesh + RigidBody

    // Stamping a starter prefab yields a working instance.
    ac::Actor* lamp = w.spawn_prefab("Point Light");
    CHECK(lamp != nullptr);
    auto* l = lamp->get_component<ac::LightComponent>();
    CHECK(l != nullptr && l->kind == ac::LightKind::Point);
    ac::Actor* cube = w.spawn_prefab("Physics Cube");
    CHECK(cube != nullptr);
    CHECK(cube->get_component<ac::RigidBodyComponent>() != nullptr);
    CHECK(cube->get_component<ac::MeshComponent>() != nullptr);
    ac::Actor* trig = w.spawn_prefab("Trigger Volume");
    CHECK(trig != nullptr);
    CHECK(trig->get_component<ac::TagComponent>()->has("trigger"));

    // Idempotent + non-clobbering: a second call adds nothing (all present),
    // and a designer's same-name prefab is preserved.
    CHECK(ac::register_builtin_prefabs(w) == 0u);

    ac::World w2;
    ac::Actor* custom = w2.spawn("MyLamp");
    custom->add_component<ac::MeshComponent>();      // 1 non-default component
    CHECK(w2.create_prefab("Point Light", custom->id()));   // user defines it first
    const cardinal::u32 added2 = ac::register_builtin_prefabs(w2);
    CHECK(added2 == 6u);                             // Point Light skipped
    // The user's version (Transform + Mesh) survived, not the builtin Light one.
    CHECK(w2.prefab_component_count("Point Light") == sz(2));
    ac::Actor* mine = w2.spawn_prefab("Point Light");
    CHECK(mine->get_component<ac::MeshComponent>() != nullptr);
    CHECK(mine->get_component<ac::LightComponent>() == nullptr);   // not the builtin
}

// ---- actor search queries (find_all_by_name / _by_tag) ------------
void test_find_all_queries() {
    ac::World w;
    ac::Actor* crate1 = w.spawn("Crate");
    ac::Actor* crate2 = w.spawn("crate_big");      // lowercase + substring
    ac::Actor* lamp   = w.spawn("Lamp");
    w.spawn("Wall");
    crate1->add_component<ac::TagComponent>()->add("pickup");
    lamp->add_component<ac::TagComponent>()->add("light");

    // Case-insensitive substring: "crate" matches "Crate" + "crate_big".
    auto by_name = w.find_all_by_name("crate");
    CHECK(by_name.size() == sz(2));
    // Case-sensitive variant: "Crate" matches only "Crate".
    auto cs = w.find_all_by_name("Crate", /*case_insensitive=*/false);
    CHECK(cs.size() == sz(1) && cs[0] == crate1);

    // Empty substring -> all alive actors (the "no filter" case).
    CHECK(w.find_all_by_name("").size() == sz(4));

    // No match -> empty.
    CHECK(w.find_all_by_name("zzz").empty());

    // Tag query, alive-filtered.
    auto pick = w.find_all_by_tag("pickup");
    CHECK(pick.size() == sz(1) && pick[0] == crate1);
    CHECK(w.find_all_by_tag("light").size() == sz(1));
    CHECK(w.find_all_by_tag("none").empty());

    // Dead actors drop out of both queries (alive-filtered).
    w.destroy(crate2->id());                         // mark dead (not yet swept)
    CHECK(w.find_all_by_name("crate").size() == sz(1));   // only "Crate" now
    CHECK(w.find_all_by_name("").size() == sz(3));        // 4 - 1 dead
}

// ---- bulk operations (on a set of ids, e.g. a filter result) ------
void test_bulk_ops() {
    ac::World w;
    ac::Actor* a = w.spawn("A");
    ac::Actor* b = w.spawn("B");
    ac::Actor* c = w.spawn("C");
    cardinal::vector<cardinal::u32> ids = { a->id(), b->id() };   // not c

    // Bulk disable -> only the listed actors flip; unknown ids skipped.
    cardinal::vector<cardinal::u32> with_bogus = { a->id(), b->id(), 9999u };
    CHECK(w.bulk_set_enabled(with_bogus, false) == 2u);          // 9999 skipped
    CHECK(!a->enabled() && !b->enabled());
    CHECK(c->enabled());                                          // untouched
    CHECK(w.bulk_set_enabled(ids, true) == 2u);
    CHECK(a->enabled() && b->enabled());

    // Bulk add tag -> adds a TagComponent where missing, dedupes.
    CHECK(w.bulk_add_tag(ids, "enemy") == 2u);
    CHECK(a->get_component<ac::TagComponent>() != nullptr);
    CHECK(a->get_component<ac::TagComponent>()->has("enemy"));
    CHECK(b->get_component<ac::TagComponent>()->has("enemy"));
    CHECK(c->get_component<ac::TagComponent>() == nullptr);       // not in set
    // The tagged set is now queryable via find_all_by_tag.
    CHECK(w.find_all_by_tag("enemy").size() == sz(2));
    // Re-add dedupes (still one "enemy" tag each).
    w.bulk_add_tag(ids, "enemy");
    CHECK(a->get_component<ac::TagComponent>()->tags.size() == sz(1));
    // Empty tag is a no-op.
    CHECK(w.bulk_add_tag(ids, "") == 0u);

    // Bulk remove tag -> only affects actors that have it.
    CHECK(w.bulk_remove_tag(ids, "enemy") == 2u);
    CHECK(!a->get_component<ac::TagComponent>()->has("enemy"));
    CHECK(w.find_all_by_tag("enemy").empty());
    CHECK(w.bulk_remove_tag(ids, "enemy") == 0u);                // already gone

    // Bulk destroy -> kills listed actors (deferred), counts live kills.
    CHECK(w.bulk_destroy(ids) == 2u);
    CHECK(!a->alive() && !b->alive());
    CHECK(c->alive());
    CHECK(w.bulk_destroy(ids) == 0u);                            // already dead
    w.sweep();
    CHECK(w.actor_count() == sz(1));                             // only C remains
}

// ---- has_component / remove_component -----------------------------
void test_has_remove_component() {
    ac::World w;
    ac::Actor* a = w.spawn("E");                  // auto-Transform
    CHECK(a->has_component("Transform"));
    CHECK(a->has_component<ac::TransformComponent>());
    CHECK(!a->has_component("Mesh"));
    CHECK(!a->has_component<ac::MeshComponent>());

    a->add_component<ac::MeshComponent>();
    a->add_component<ac::LightComponent>();
    CHECK(a->has_component("Mesh") && a->has_component("Light"));
    CHECK(a->components().size() == sz(3));        // Transform + Mesh + Light

    // Remove the Mesh — fires on_detach, drops the count, leaves the rest.
    CHECK(a->remove_component("Mesh"));
    CHECK(!a->has_component("Mesh"));
    CHECK(a->has_component("Light"));               // unaffected
    CHECK(a->has_component("Transform"));
    CHECK(a->components().size() == sz(2));

    // Removing an absent type returns false, no-op.
    CHECK(!a->remove_component("Mesh"));
    CHECK(a->components().size() == sz(2));

    // Template remove form.
    CHECK(a->remove_component<ac::LightComponent>());
    CHECK(!a->has_component("Light"));
    CHECK(a->components().size() == sz(1));         // just Transform

    // on_detach actually fires on remove — use a counting component.
    static int s_live = 0;
    struct Counted : ac::Component {
        const char* type_name() const noexcept override { return "Counted"; }
        void on_attach(ac::Actor&) override { ++s_live; }
        void on_detach(ac::Actor&) override { --s_live; }
    };
    s_live = 0;
    a->add_component<Counted>();
    CHECK(s_live == 1);
    CHECK(a->remove_component("Counted"));
    CHECK(s_live == 0);                             // on_detach ran
}

// ---- actor enable / disable (sim gating) --------------------------
void test_enable_disable() {
    ac::World w;
    ac::Actor* a = w.spawn("Mover");
    CHECK(a->enabled());                          // default on

    // A component that counts its ticks.
    static int s_ticks = 0;
    struct Ticker : ac::Component {
        const char* type_name() const noexcept override { return "Ticker"; }
        void on_tick(ac::Actor&, float) override { ++s_ticks; }
    };
    a->add_component<Ticker>();

    // Enabled: World::tick drives the component.
    s_ticks = 0;
    w.tick(0.016f);
    CHECK(s_ticks == 1);

    // Disabled: World::tick SKIPS the actor entirely.
    a->set_enabled(false);
    CHECK(!a->enabled());
    s_ticks = 0;
    w.tick(0.016f);
    CHECK(s_ticks == 0);                          // gated out

    // Re-enable resumes ticking.
    a->set_enabled(true);
    s_ticks = 0;
    w.tick(0.016f);
    CHECK(s_ticks == 1);

    // A disabled actor stays alive + findable (not destroyed).
    a->set_enabled(false);
    CHECK(a->alive());
    CHECK(w.find(a->id()) == a);
    CHECK(w.actor_count() == sz(1));

    // duplicate carries the disabled state.
    ac::Actor* dup = w.duplicate(a->id());
    CHECK(dup != nullptr && !dup->enabled());
    // ...and an enabled source yields an enabled copy.
    a->set_enabled(true);
    ac::Actor* dup2 = w.duplicate(a->id());
    CHECK(dup2 != nullptr && dup2->enabled());
}

// ---- World revision counter (auto-checkpoint signal) --------------
void test_revision() {
    ac::World w;
    const cardinal::u64 r0 = w.revision();

    // Spawn bumps (creation).
    ac::Actor* a = w.spawn("A");
    CHECK(w.revision() > r0);
    const cardinal::u64 r1 = w.revision();
    ac::Actor* b = w.spawn("B");
    CHECK(w.revision() > r1);

    // Read-only queries do NOT bump.
    const cardinal::u64 r2 = w.revision();
    (void)w.find(a->id());
    (void)w.find_by_name("B");
    (void)w.actor_count();
    (void)w.find_all_by_name("A");
    CHECK(w.revision() == r2);

    // Manual bump (what the Inspector calls after a field edit).
    w.bump_revision();
    CHECK(w.revision() == r2 + 1);

    // Duplicate bumps.
    const cardinal::u64 r3 = w.revision();
    w.duplicate(a->id());
    CHECK(w.revision() > r3);

    // destroy bumps; sweep bumps only when it actually removes.
    const cardinal::u64 r4 = w.revision();
    w.destroy(b->id());
    CHECK(w.revision() > r4);
    // Destroying an ALREADY-dead (unswept) actor must NOT bump again — no
    // spurious undo checkpoint for a no-op.
    const cardinal::u64 r4b = w.revision();
    w.destroy(b->id());
    CHECK(w.revision() == r4b);
    const cardinal::u64 r5 = w.revision();
    w.sweep();
    CHECK(w.revision() > r5);                 // removed b
    const cardinal::u64 r6 = w.revision();
    w.sweep();
    CHECK(w.revision() == r6);                // nothing to remove -> no bump

    // Bulk ops bump once when they affect ≥1 actor, not at all otherwise.
    cardinal::vector<cardinal::u32> ids = { a->id() };
    const cardinal::u64 r7 = w.revision();
    CHECK(w.bulk_add_tag(ids, "x") == 1u);
    CHECK(w.revision() == r7 + 1);
    const cardinal::u64 r8 = w.revision();
    CHECK(w.bulk_add_tag({}, "x") == 0u);    // empty set
    CHECK(w.revision() == r8);               // no bump
}

// ---- O(1) id index: find() correctness across spawn/destroy/sweep -
void test_world_index() {
    ac::World w;

    // find on an empty world / id 0 (kInvalidActor) -> null.
    CHECK(w.find(0u) == nullptr);
    CHECK(w.find(123u) == nullptr);

    // Spawn a batch; every id resolves to the right actor.
    cardinal::vector<cardinal::u32> ids;
    for (int i = 0; i < 500; ++i) ids.push_back(w.spawn("A")->id());
    for (cardinal::u32 id : ids) {
        ac::Actor* a = w.find(id);
        CHECK(a != nullptr && a->id() == id);
    }
    CHECK(w.find(99999u) == nullptr);        // unknown id

    // destroy marks dead but find STILL returns it (unswept contract).
    ac::Actor* victim = w.find(ids[100]);
    w.destroy(ids[100]);
    CHECK(w.find(ids[100]) == victim);       // present until sweep
    CHECK(!victim->alive());

    // sweep removes the dead actor from the index.
    w.sweep();
    CHECK(w.find(ids[100]) == nullptr);
    // ...and leaves the rest intact.
    CHECK(w.find(ids[0]) != nullptr);
    CHECK(w.find(ids[499]) != nullptr);

    // Interleaved churn keeps the index consistent + ids are never reused.
    cardinal::u32 last = ids.back();
    for (int round = 0; round < 50; ++round) {
        ac::Actor* fresh = w.spawn("B");
        CHECK(fresh->id() > last);            // monotonic, never reused
        last = fresh->id();
        CHECK(w.find(fresh->id()) == fresh);
        w.destroy(fresh->id());
        w.sweep();
        CHECK(w.find(fresh->id()) == nullptr);   // gone after sweep
    }

    // Duplicate / array (which spawn via spawn_bare_) are indexed too.
    ac::Actor* src = w.spawn("Src");
    ac::Actor* dup = w.duplicate(src->id());
    CHECK(dup != nullptr && w.find(dup->id()) == dup);
    auto arr = w.array_actor(src->id(), 3, { 1.0f, 0.0f, 0.0f });
    for (auto* x : arr) CHECK(w.find(x->id()) == x);
}

// ---- component copy / paste (clipboard) ---------------------------
void test_copy_paste_component() {
    ac::World w;

    // Source actor with a configured Light.
    ac::Actor* src = w.spawn("Lamp");
    auto* sl = src->add_component<ac::LightComponent>();
    sl->kind = ac::LightKind::Spot;
    sl->intensity = 8.0f; sl->range = 30.0f;
    sl->color = { 0.2f, 0.4f, 0.8f };

    // Copy -> a self-describing blob tagged with the type.
    cardinal::string blob = ac::copy_component(*sl);
    CHECK(!blob.empty());
    CHECK(blob.compare(0, 5, "Light") == 0);    // type tag on line 1

    // Paste onto a DIFFERENT actor that has no Light yet -> adds one.
    ac::Actor* dst = w.spawn("Wall");
    CHECK(!dst->has_component("Light"));
    ac::Component* pasted = ac::paste_component(*dst, blob);
    CHECK(pasted != nullptr);
    CHECK(dst->has_component("Light"));
    auto* dl = dst->get_component<ac::LightComponent>();
    CHECK(dl != nullptr);
    CHECK(dl->kind == ac::LightKind::Spot);
    CHECK(ap(dl->intensity, 8.0f) && ap(dl->range, 30.0f));
    CHECK(ap(dl->color.z, 0.8f));
    // dst gained exactly one component (Transform + Light = 2).
    CHECK(dst->components().size() == sz(2));

    // Edit the source, re-copy, paste again -> OVERWRITES dst's Light in
    // place (paste-values), does NOT stack a second Light.
    sl->intensity = 1.0f;
    blob = ac::copy_component(*sl);
    ac::paste_component(*dst, blob);
    CHECK(dst->components().size() == sz(2));    // still one Light
    CHECK(ap(dst->get_component<ac::LightComponent>()->intensity, 1.0f));

    // Paste independence — editing dst's Light doesn't touch src's.
    dst->get_component<ac::LightComponent>()->intensity = 99.0f;
    CHECK(ap(src->get_component<ac::LightComponent>()->intensity, 1.0f));

    // Empty + unknown-type blobs paste cleanly to nullptr.
    CHECK(ac::paste_component(*dst, "") == nullptr);
    CHECK(ac::paste_component(*dst, "GameActor\n") == nullptr);  // not a factory type
    CHECK(ac::paste_component(*dst, "Bogus\n  x = 1\n") == nullptr);
}

// ---- alignment / distribution (level layout) ----------------------
void test_align_distribute() {
    using Ax = ac::World::Axis;
    using Mode = ac::World::AlignMode;
    ac::World w;

    auto at = [&](const char* n, float x, float y, float z) {
        ac::Actor* a = w.spawn(n);
        a->get_component<ac::TransformComponent>()->translation = { x, y, z };
        return a;
    };
    auto tx = [](ac::Actor* a) { return a->get_component<ac::TransformComponent>()->translation; };

    // Three actors at different Y.
    ac::Actor* a = at("A", 0.0f, 2.0f, 0.0f);
    ac::Actor* b = at("B", 0.0f, 6.0f, 0.0f);
    ac::Actor* c = at("C", 0.0f, 10.0f, 0.0f);
    cardinal::vector<cardinal::u32> ids = { a->id(), b->id(), c->id() };

    // Align Y center -> all share (2+10)/2 = 6.
    CHECK(w.align_actors(ids, Ax::Y, Mode::Center) == 3u);
    CHECK(ap(tx(a).y, 6.0f) && ap(tx(b).y, 6.0f) && ap(tx(c).y, 6.0f));
    // Align Y min -> all 6 now, so min == 6.
    at("reset", 0, 0, 0);                              // unrelated actor, not in set
    a->get_component<ac::TransformComponent>()->translation.y = 1.0f;
    c->get_component<ac::TransformComponent>()->translation.y = 9.0f;
    CHECK(w.align_actors(ids, Ax::Y, Mode::Min) == 3u);
    CHECK(ap(tx(a).y, 1.0f) && ap(tx(b).y, 1.0f) && ap(tx(c).y, 1.0f));   // min was 1
    // Align Y max.
    a->get_component<ac::TransformComponent>()->translation.y = 1.0f;
    c->get_component<ac::TransformComponent>()->translation.y = 9.0f;
    CHECK(w.align_actors(ids, Ax::Y, Mode::Max) == 3u);
    CHECK(ap(tx(a).y, 9.0f) && ap(tx(c).y, 9.0f));    // max was 9

    // Distribute along X: 0, 5, 7, 30 (unsorted) -> 0, 10, 20, 30 (even).
    ac::World w2;
    auto at2 = [&](const char* n, float x) {
        ac::Actor* z = w2.spawn(n);
        z->get_component<ac::TransformComponent>()->translation.x = x;
        return z;
    };
    ac::Actor* p0 = at2("P0", 0.0f);
    ac::Actor* p1 = at2("P1", 5.0f);
    ac::Actor* p2 = at2("P2", 7.0f);
    ac::Actor* p3 = at2("P3", 30.0f);
    cardinal::vector<cardinal::u32> ids2 = { p0->id(), p1->id(), p2->id(), p3->id() };
    // Only the 2 interior actors move (endpoints stay).
    CHECK(w2.distribute_actors(ids2, Ax::X) == 2u);
    auto x = [](ac::Actor* z) { return z->get_component<ac::TransformComponent>()->translation.x; };
    CHECK(ap(x(p0), 0.0f) && ap(x(p3), 30.0f));       // extremes unchanged
    CHECK(ap(x(p1), 10.0f) && ap(x(p2), 20.0f));      // evenly spaced

    // Distribute needs >=3.
    cardinal::vector<cardinal::u32> two = { p0->id(), p3->id() };
    CHECK(w2.distribute_actors(two, Ax::X) == 0u);

    // Empty set / no-transform safety.
    CHECK(w.align_actors({}, Ax::X, Mode::Center) == 0u);

    // NaN-safety: a NaN axis value must NOT reach the sort comparator
    // (strict-weak-ordering violation = std::sort UB / heap corruption).
    // The NaN actor is skipped; the finite ones still distribute.
    ac::World w3;
    auto mk = [&](float x) {
        ac::Actor* z = w3.spawn("n");
        z->get_component<ac::TransformComponent>()->translation.x = x;
        return z;
    };
    ac::Actor* q0 = mk(0.0f);
    ac::Actor* qn = mk(std::numeric_limits<float>::quiet_NaN());   // poison
    ac::Actor* q1 = mk(5.0f);
    ac::Actor* q2 = mk(10.0f);
    cardinal::vector<cardinal::u32> qids = { q0->id(), qn->id(), q1->id(), q2->id() };
    // 3 finite actors (0,5,10) distribute; the NaN one is excluded. Endpoints
    // 0 and 10 stay; the single interior (5) lands at the midpoint 5. No crash.
    const cardinal::u32 moved = w3.distribute_actors(qids, Ax::X);
    CHECK(moved == 1u);
    CHECK(ap(q0->get_component<ac::TransformComponent>()->translation.x, 0.0f));
    CHECK(ap(q2->get_component<ac::TransformComponent>()->translation.x, 10.0f));
    CHECK(ap(q1->get_component<ac::TransformComponent>()->translation.x, 5.0f));
}

// ---- scene validation ---------------------------------------------
void test_validation() {
    ac::World w;

    auto has_issue = [](const cardinal::vector<ac::ValidationIssue>& v,
                        ac::Severity sev, const char* substr) {
        for (const auto& i : v) {
            if (i.severity == sev && i.message.find(substr) != cardinal::string::npos)
                return true;
        }
        return false;
    };

    // A clean actor with a real mesh + a lit light: no issues from it.
    ac::Actor* ok = w.spawn("Ground");
    ok->add_component<ac::MeshComponent>()->asset_id = "ground.mesh";

    // Empty actor (only Transform) -> Info.
    w.spawn("EmptyOne");

    // Mesh with no asset -> Warning.
    ac::Actor* badmesh = w.spawn("Crate");
    badmesh->add_component<ac::MeshComponent>();          // asset_id empty

    // Zero-intensity light -> Warning.
    ac::Actor* dark = w.spawn("DeadLight");
    dark->add_component<ac::LightComponent>()->intensity = 0.0f;

    // Out-of-bounds position -> Warning.
    ac::Actor* lost = w.spawn("Lost");
    lost->get_component<ac::TransformComponent>()->translation.x = 1.0e7f;

    // Zero-scale axis -> Warning.
    ac::Actor* flat = w.spawn("Flat");
    flat->get_component<ac::TransformComponent>()->scale.y = 0.0f;

    // Two active cameras -> scene-level Warning.
    w.spawn("CamA")->add_component<ac::CameraComponent>()->active = true;
    w.spawn("CamB")->add_component<ac::CameraComponent>()->active = true;

    // Duplicate names -> scene-level Info.
    w.spawn("Twin");
    w.spawn("Twin");

    const auto issues = ac::validate_world(w);
    CHECK(!issues.empty());
    CHECK(has_issue(issues, ac::Severity::Info,    "no components beyond Transform"));
    CHECK(has_issue(issues, ac::Severity::Warning, "no asset_id"));
    CHECK(has_issue(issues, ac::Severity::Warning, "zero or negative intensity"));
    CHECK(has_issue(issues, ac::Severity::Warning, "far outside the world bounds"));
    CHECK(has_issue(issues, ac::Severity::Warning, "zero scale axis"));
    CHECK(has_issue(issues, ac::Severity::Warning, "cameras are marked active"));
    CHECK(has_issue(issues, ac::Severity::Info,    "share the name 'Twin'"));

    // count_issues by severity threshold.
    const cardinal::u32 warns = ac::count_issues(issues, ac::Severity::Warning);
    CHECK(warns >= 5u);                                   // the warnings above
    CHECK(ac::count_issues(issues, ac::Severity::Error) == 0u);

    // The "Ground" actor (valid mesh) produced no per-actor issue.
    for (const auto& i : issues) {
        if (i.actor == ok->id())
            CHECK(false && "clean actor should have no issues");
    }

    // A pristine world validates clean (one spawned actor with a mesh).
    ac::World clean;
    clean.spawn("Hero")->add_component<ac::MeshComponent>()->asset_id = "hero";
    CHECK(ac::validate_world(clean).empty());
}

// ---- validation auto-fix ------------------------------------------
void test_validation_autofix() {
    ac::World w;

    // Zero-scale axis.
    ac::Actor* flat = w.spawn("Flat");
    flat->get_component<ac::TransformComponent>()->scale.y = 0.0f;
    // Out-of-bounds position.
    ac::Actor* lost = w.spawn("Lost");
    lost->get_component<ac::TransformComponent>()->translation = { 1e9f, 0.0f, 0.0f };
    // Duplicate names.
    w.spawn("Twin");
    ac::Actor* twin2 = w.spawn("Twin");
    // A clean actor that must stay untouched.
    ac::Actor* keep = w.spawn("Keep");
    keep->get_component<ac::TransformComponent>()->translation = { 3.0f, 4.0f, 5.0f };

    const cardinal::u32 fixed = ac::auto_fix_world(w);
    CHECK(fixed == 3u);   // 1 scale + 1 position + 1 rename

    // Zero scale recovered to 1.
    CHECK(ap(flat->get_component<ac::TransformComponent>()->scale.y, 1.0f));
    // Out-of-bounds reset to origin.
    CHECK(ap(lost->get_component<ac::TransformComponent>()->translation.x, 0.0f));
    // The 2nd Twin got a unique name.
    CHECK(twin2->name() != "Twin");
    CHECK(twin2->name() == "Twin (2)");
    // Exactly one actor is now named "Twin" (exact, not substring).
    int exact_twins = 0;
    for (const auto& ap2 : w.actors())
        if (ap2->alive() && ap2->name() == "Twin") ++exact_twins;
    CHECK(exact_twins == 1);
    // The clean actor is untouched.
    CHECK(ap(keep->get_component<ac::TransformComponent>()->translation.x, 3.0f));

    // Re-validate: scale / bounds / dup-name issues are gone.
    auto after = ac::validate_world(w);
    for (const auto& i : after) {
        CHECK(i.message.find("zero scale axis") == cardinal::string::npos);
        CHECK(i.message.find("far outside")     == cardinal::string::npos);
        CHECK(i.message.find("share the name")  == cardinal::string::npos);
    }

    // A clean world -> 0 fixes.
    ac::World clean;
    clean.spawn("A"); clean.spawn("B");
    CHECK(ac::auto_fix_world(clean) == 0u);
}

// ---- scene statistics ---------------------------------------------
void test_world_stats() {
    ac::World w;

    // 2 mesh actors (one disabled), 1 light, 2 tagged.
    ac::Actor* m1 = w.spawn("M1");
    m1->add_component<ac::MeshComponent>();
    ac::Actor* m2 = w.spawn("M2");
    m2->add_component<ac::MeshComponent>();
    m2->set_enabled(false);
    ac::Actor* lt = w.spawn("L");
    lt->add_component<ac::LightComponent>();
    ac::Actor* t1 = w.spawn("T1");
    t1->add_component<ac::TagComponent>()->add("enemy");
    ac::Actor* t2 = w.spawn("T2");
    auto* tc2 = t2->add_component<ac::TagComponent>();
    tc2->add("enemy"); tc2->add("boss");

    // A prefab instance (carries a PrefabLink).
    CHECK(w.create_prefab("M1", m1->id()));
    ac::Actor* inst = w.spawn_prefab("M1");
    CHECK(inst != nullptr);

    auto s = ac::compute_world_stats(w);

    // 6 alive actors: M1, M2, L, T1, T2, instance.
    CHECK(s.actors == 6u);
    CHECK(s.enabled == 5u && s.disabled == 1u);     // M2 disabled
    CHECK(s.prefab_instances == 1u);                // the spawned instance

    // Component breakdown: every actor has a Transform (6).
    CHECK(s.component_count("Transform") == 6u);
    // Mesh: M1, M2, and the instance (cloned from M1) = 3.
    CHECK(s.component_count("Mesh") == 3u);
    CHECK(s.component_count("Light") == 1u);
    CHECK(s.component_count("Tag") == 2u);
    CHECK(s.component_count("PrefabLink") == 1u);    // only the instance
    CHECK(s.component_count("Nope") == 0u);

    // Tag breakdown: "enemy" on T1 + T2 = 2; "boss" on T2 = 1.
    CHECK(s.tag_count("enemy") == 2u);
    CHECK(s.tag_count("boss") == 1u);
    CHECK(s.tag_count("none") == 0u);

    // Breakdowns are sorted by name.
    for (cardinal::usize i = 1; i < s.by_component.size(); ++i)
        CHECK(s.by_component[i - 1].name < s.by_component[i].name);

    // Empty world -> zeros.
    ac::World empty;
    auto es = ac::compute_world_stats(empty);
    CHECK(es.actors == 0u && es.by_component.empty() && es.by_tag.empty());

    // Multi-Tag actor: an actor legally holding TWO TagComponents must have
    // BOTH components' tags counted (the tally reads the iterated component,
    // not get_component which returns only the first).
    ac::World mt;
    ac::Actor* m = mt.spawn("Multi");
    m->add_component<ac::TagComponent>()->add("alpha");
    m->add_component<ac::TagComponent>()->add("beta");   // second Tag component
    auto ms = ac::compute_world_stats(mt);
    CHECK(ms.component_count("Tag") == 2u);
    CHECK(ms.tag_count("alpha") == 1u);
    CHECK(ms.tag_count("beta") == 1u);                   // not dropped
}

// ---- grid snapping ------------------------------------------------
void test_snap_grid() {
    // The pure helper rounds to the nearest multiple.
    CHECK(ap(ac::snap_to_grid(0.4f, 1.0f), 0.0f));
    CHECK(ap(ac::snap_to_grid(0.6f, 1.0f), 1.0f));
    CHECK(ap(ac::snap_to_grid(2.3f, 0.5f), 2.5f));
    CHECK(ap(ac::snap_to_grid(-1.2f, 0.5f), -1.0f));
    CHECK(ap(ac::snap_to_grid(7.0f, 0.0f), 7.0f));     // step<=0 -> unchanged
    CHECK(ap(ac::snap_to_grid(7.0f, -1.0f), 7.0f));

    ac::World w;
    auto at = [&](const char* n, float x, float y, float z) {
        ac::Actor* a = w.spawn(n);
        a->get_component<ac::TransformComponent>()->translation = { x, y, z };
        return a;
    };
    ac::Actor* a = at("A", 0.4f, 1.6f, -0.9f);
    ac::Actor* b = at("B", 2.2f, 0.1f, 4.7f);
    cardinal::vector<cardinal::u32> ids = { a->id(), b->id() };

    CHECK(w.snap_actors_to_grid(ids, 1.0f) == 2u);
    auto ta = a->get_component<ac::TransformComponent>()->translation;
    auto tb = b->get_component<ac::TransformComponent>()->translation;
    CHECK(ap(ta.x, 0.0f) && ap(ta.y, 2.0f) && ap(ta.z, -1.0f));
    CHECK(ap(tb.x, 2.0f) && ap(tb.y, 0.0f) && ap(tb.z, 5.0f));

    // step<=0 + empty set are clean no-ops.
    CHECK(w.snap_actors_to_grid(ids, 0.0f) == 0u);
    CHECK(w.snap_actors_to_grid({}, 1.0f) == 0u);
}

// ---- prefab instance linkage (revert / apply edit loop) -----------
void test_prefab_link_revert_apply() {
    ac::World w;

    // Build + capture a "Box" prefab (Transform + Mesh).
    ac::Actor* src = w.spawn("Box");
    src->get_component<ac::TransformComponent>()->translation = { 1.0f, 0.0f, 0.0f };
    auto* sm = src->add_component<ac::MeshComponent>();
    sm->asset_id = "box.mesh";
    CHECK(w.create_prefab("Box", src->id()));
    // The source actor is NOT a prefab instance (no link).
    CHECK(w.prefab_of(src->id()).empty());
    // The prototype does NOT carry a PrefabLink (filtered on capture).
    const ac::Actor* proto = w.prefab_prototype("Box");
    CHECK(proto != nullptr);
    if (proto) {
        bool proto_has_link = false;
        for (const auto& c : proto->components())
            if (cardinal::strcmp(c->type_name(), "PrefabLink") == 0) proto_has_link = true;
        CHECK(!proto_has_link);
        CHECK(proto->components().size() == sz(2));   // Transform + Mesh only
    }

    // Stamp an instance — it IS linked to "Box".
    ac::Actor* inst = w.spawn_prefab("Box");
    CHECK(inst != nullptr);
    CHECK(w.prefab_of(inst->id()) == "Box");
    // Components: Transform + Mesh + PrefabLink = 3.
    CHECK(inst->components().size() == sz(3));
    CHECK(inst->get_component<ac::PrefabLinkComponent>() != nullptr);
    CHECK(inst->get_component<ac::PrefabLinkComponent>()->prefab_name == "Box");

    // Edit the instance locally.
    inst->get_component<ac::TransformComponent>()->translation = { 99.0f, 9.0f, 9.0f };
    inst->get_component<ac::MeshComponent>()->asset_id = "edited.mesh";

    // Revert — local edits discarded, prefab values restored, link kept.
    CHECK(w.revert_to_prefab(inst->id()));
    CHECK(ap(inst->get_component<ac::TransformComponent>()->translation.x, 1.0f));
    CHECK(inst->get_component<ac::MeshComponent>()->asset_id == "box.mesh");
    CHECK(w.prefab_of(inst->id()) == "Box");                 // link survived
    CHECK(inst->components().size() == sz(3));                // no dup link

    // Apply — edit the instance, push up into the prefab; new spawns inherit.
    inst->get_component<ac::TransformComponent>()->translation = { 5.0f, 5.0f, 5.0f };
    inst->get_component<ac::MeshComponent>()->asset_id = "v2.mesh";
    CHECK(w.apply_to_prefab(inst->id()));
    // Prototype updated (still 2 components, link excluded on capture).
    CHECK(w.prefab_component_count("Box") == sz(2));
    ac::Actor* inst2 = w.spawn_prefab("Box");
    CHECK(ap(inst2->get_component<ac::TransformComponent>()->translation.x, 5.0f));
    CHECK(inst2->get_component<ac::MeshComponent>()->asset_id == "v2.mesh");

    // Edge cases: non-instance actor + missing-prefab revert both fail clean.
    CHECK(!w.revert_to_prefab(src->id()));        // src has no link
    CHECK(!w.apply_to_prefab(src->id()));
    w.remove_prefab("Box");
    CHECK(!w.revert_to_prefab(inst->id()));       // prefab gone
    CHECK(w.prefab_of(inst->id()) == "Box");      // link string still readable
}

// ---- actor duplication (Ctrl-D primitive) -------------------------
void test_duplicate() {
    ac::World w;

    // Source with a couple of components + authored values.
    ac::Actor* src = w.spawn("Crate");
    src->get_component<ac::TransformComponent>()->translation = { 7.0f, 0.0f, 0.0f };
    auto* sm = src->add_component<ac::MeshComponent>();
    sm->asset_id = "crate.mesh";

    // Duplicate: distinct id, unique "(copy)" name, cloned values, alive.
    ac::Actor* d1 = w.duplicate(src->id());
    CHECK(d1 != nullptr);
    CHECK(d1->id() != src->id());
    CHECK(d1->name() == "Crate (copy)");
    CHECK(d1->alive());
    CHECK(w.find(d1->id()) == d1);
    auto* dt = d1->get_component<ac::TransformComponent>();
    CHECK(dt != nullptr && ap(dt->translation.x, 7.0f));
    CHECK(d1->get_component<ac::MeshComponent>() != nullptr);
    CHECK(d1->get_component<ac::MeshComponent>()->asset_id == "crate.mesh");
    // No duplicate auto-Transform: Transform + Mesh = 2.
    CHECK(d1->components().size() == sz(2));

    // Independence — edit the dup, source unchanged.
    dt->translation = { -1.0f, 0.0f, 0.0f };
    CHECK(ap(src->get_component<ac::TransformComponent>()->translation.x, 7.0f));

    // Second duplicate of the SAME source -> "(copy 2)" (copy is taken).
    ac::Actor* d2 = w.duplicate(src->id());
    CHECK(d2 != nullptr && d2->name() == "Crate (copy 2)");

    // Duplicating a COPY strips the suffix -> base "(copy)" path, which is
    // taken, so it lands on the next free "(copy N)".
    ac::Actor* d3 = w.duplicate(d1->id());
    CHECK(d3 != nullptr);
    CHECK(d3->name() == "Crate (copy 3)");   // copy + copy 2 taken

    // Duplicating a prefab instance keeps the PrefabLink (same prefab).
    CHECK(w.create_prefab("Crate", src->id()));
    ac::Actor* inst = w.spawn_prefab("Crate");
    CHECK(w.prefab_of(inst->id()) == "Crate");
    ac::Actor* inst_dup = w.duplicate(inst->id());
    CHECK(inst_dup != nullptr);
    CHECK(w.prefab_of(inst_dup->id()) == "Crate");   // link cloned through

    // Unknown id -> nullptr.
    CHECK(w.duplicate(99999u) == nullptr);
}

// ---- spawn placement (spawn_at / spawn_prefab_at) -----------------
void test_spawn_placement() {
    ac::World w;

    // spawn_at places a plain actor at a world position.
    ac::Actor* a = w.spawn_at("Marker", { 5.0f, 1.0f, -3.0f });
    CHECK(a != nullptr);
    auto* t = a->get_component<ac::TransformComponent>();
    CHECK(t != nullptr);
    CHECK(ap(t->translation.x, 5.0f) && ap(t->translation.y, 1.0f) && ap(t->translation.z, -3.0f));

    // spawn_prefab_at places a stamped instance (vs the prototype's origin).
    ac::Actor* src = w.spawn("Box");
    src->add_component<ac::MeshComponent>()->asset_id = "box";
    CHECK(w.create_prefab("Box", src->id()));
    // Prototype's transform is at origin.
    ac::Actor* i1 = w.spawn_prefab_at("Box", { 10.0f, 0.0f, 0.0f });
    CHECK(i1 != nullptr);
    CHECK(ap(i1->get_component<ac::TransformComponent>()->translation.x, 10.0f));
    CHECK(i1->get_component<ac::MeshComponent>() != nullptr);   // still a full clone
    CHECK(w.prefab_of(i1->id()) == "Box");                      // still linked

    // Two stamps at different positions don't pile up.
    ac::Actor* i2 = w.spawn_prefab_at("Box", { 12.0f, 0.0f, 0.0f });
    CHECK(ap(i2->get_component<ac::TransformComponent>()->translation.x, 12.0f));
    CHECK(i1->get_component<ac::TransformComponent>()->translation.x !=
          i2->get_component<ac::TransformComponent>()->translation.x);

    // Unknown prefab -> nullptr (placement no-op).
    CHECK(w.spawn_prefab_at("Nope", { 0, 0, 0 }) == nullptr);
}

// ---- array tool (spaced line of clones) ---------------------------
void test_array_actor() {
    ac::World w;
    ac::Actor* src = w.spawn("Pillar");
    src->get_component<ac::TransformComponent>()->translation = { 1.0f, 0.0f, 0.0f };
    src->add_component<ac::MeshComponent>()->asset_id = "pillar";

    // Array of 3, stepping +2 on X: copies at 3, 5, 7 (source stays at 1).
    auto made = w.array_actor(src->id(), 3, { 2.0f, 0.0f, 0.0f });
    CHECK(made.size() == sz(3));
    CHECK(ap(made[0]->get_component<ac::TransformComponent>()->translation.x, 3.0f));
    CHECK(ap(made[1]->get_component<ac::TransformComponent>()->translation.x, 5.0f));
    CHECK(ap(made[2]->get_component<ac::TransformComponent>()->translation.x, 7.0f));
    // Each copy is a full clone (carries the Mesh) with a distinct id.
    CHECK(made[0]->get_component<ac::MeshComponent>() != nullptr);
    CHECK(made[0]->id() != src->id() && made[1]->id() != made[0]->id());
    // Source unchanged.
    CHECK(ap(src->get_component<ac::TransformComponent>()->translation.x, 1.0f));
    // 1 source + 3 copies live in the world.
    CHECK(w.actor_count() == sz(4));

    // Edge cases: count 0 + unknown id -> empty, no spawns.
    const cardinal::usize before = w.actor_count();
    CHECK(w.array_actor(src->id(), 0, { 1, 0, 0 }).empty());
    CHECK(w.array_actor(99999u, 5, { 1, 0, 0 }).empty());
    CHECK(w.actor_count() == before);

    // A pathological count is CLAMPED (not an infinite loop / u32 wrap):
    // 0xFFFFFFFF would make `i <= count` always true. Clamp keeps it finite.
    ac::World big;
    ac::Actor* one = big.spawn("U");
    auto huge = big.array_actor(one->id(), 0xFFFFFFFFu, { 1.0f, 0.0f, 0.0f });
    CHECK(huge.size() == sz(4096));       // clamped to kMaxArray
    CHECK(big.actor_count() == sz(4097)); // source + 4096 copies
    // The far end is placed correctly (efficient stamping still positions).
    CHECK(ap(huge.back()->get_component<ac::TransformComponent>()->translation.x,
             4096.0f));
}

// ---- grid array (tile lattice) ------------------------------------
void test_array_grid() {
    ac::World w;
    ac::Actor* src = w.spawn("Tile");
    src->get_component<ac::TransformComponent>()->translation = { 0.0f, 0.0f, 0.0f };
    src->add_component<ac::MeshComponent>()->asset_id = "tile";

    // 3x1x2 grid, spacing (2,0,3): 6 cells, minus the (0,0,0) source = 5 copies.
    auto made = w.array_grid(src->id(), 3, 1, 2, { 2.0f, 0.0f, 3.0f });
    CHECK(made.size() == sz(5));
    CHECK(w.actor_count() == sz(6));         // source + 5 copies

    // Every copy is a full clone (carries the Mesh).
    for (auto* m : made) CHECK(m->get_component<ac::MeshComponent>() != nullptr);

    // Collect the copy positions + the source, verify they tile the lattice
    // exactly: x in {0,2,4}, z in {0,3} (6 cells), each once.
    auto cell_present = [&](float x, float z) {
        // source covers (0,0)
        if (ap(x, 0.0f) && ap(z, 0.0f)) return true;
        for (auto* m : made) {
            auto t = m->get_component<ac::TransformComponent>()->translation;
            if (ap(t.x, x) && ap(t.z, z)) return true;
        }
        return false;
    };
    CHECK(cell_present(0, 0) && cell_present(2, 0) && cell_present(4, 0));
    CHECK(cell_present(0, 3) && cell_present(2, 3) && cell_present(4, 3));

    // 1x1x1 grid -> only the origin cell (the source) -> 0 copies.
    ac::Actor* solo = w.spawn("Solo");
    CHECK(w.array_grid(solo->id(), 1, 1, 1, { 1, 1, 1 }).empty());

    // Zero dims are treated as 1 (no empty/degenerate grid).
    CHECK(w.array_grid(solo->id(), 0, 0, 0, { 1, 1, 1 }).empty());   // -> 1x1x1

    // Unknown source -> empty.
    CHECK(w.array_grid(99999u, 3, 3, 3, { 1, 1, 1 }).empty());

    // A huge grid is bounded (no runaway spawn). 100x100x100 = 1e6 cells.
    ac::World big;
    ac::Actor* b = big.spawn("B");
    auto huge = big.array_grid(b->id(), 100, 100, 100, { 1, 1, 1 });
    CHECK(huge.size() <= sz(4096));          // clamped
    CHECK(big.actor_count() <= sz(4097));
}

// ---- component serialization (round-trip via factory) -------------
void test_component_serialization() {
    // Local near-zero check (delta already subtracted at the call sites).
    auto fz = [](float d) { return (d < 0 ? -d : d) <= 1e-4f; };
    // Helper: feed every "key = value" line of a serialized block into a
    // freshly-made component of the same type, then compare.
    auto round_trip = [](const ac::Component& src) -> cardinal::unique_ptr<ac::Component> {
        cardinal::string blob;
        src.serialize_fields(blob);
        auto dst = ac::make_component_by_name(src.type_name());
        if (!dst) return dst;
        // Parse "  key = value\n" lines (leading 2 spaces, " = " sep).
        cardinal::usize i = 0;
        while (i < blob.size()) {
            cardinal::usize eol = blob.find('\n', i);
            if (eol == cardinal::string::npos) eol = blob.size();
            cardinal::string line = blob.substr(i, eol - i);
            i = eol + 1;
            // strip leading spaces
            cardinal::usize s = 0; while (s < line.size() && line[s] == ' ') ++s;
            const cardinal::usize eq = line.find(" = ", s);
            if (eq == cardinal::string::npos) continue;
            cardinal::string key = line.substr(s, eq - s);
            cardinal::string val = line.substr(eq + 3);
            dst->deserialize_field(key, val);
        }
        return dst;
    };

    // Factory returns the right type_name for each builtin; unknown → null.
    CHECK(ac::make_component_by_name("Transform") != nullptr);
    CHECK(ac::make_component_by_name("Mesh")      != nullptr);
    CHECK(ac::make_component_by_name("Bogus")     == nullptr);
    CHECK(ac::make_component_by_name("GameActor") == nullptr);  // game-module job

    // Transform.
    {
        ac::TransformComponent t;
        t.translation = { 1.5f, -2.0f, 3.25f };
        t.rotation_euler = { 0.1f, 0.2f, 0.3f };
        t.scale = { 2.0f, 4.0f, 8.0f };
        auto r = round_trip(t);
        auto* rt = static_cast<ac::TransformComponent*>(r.get());
        CHECK(rt != nullptr);
        CHECK(fz(rt->translation.x - 1.5f) && fz(rt->translation.z - 3.25f));
        CHECK(fz(rt->rotation_euler.y - 0.2f));
        CHECK(fz(rt->scale.z - 8.0f));
    }
    // Mesh.
    {
        ac::MeshComponent m;
        m.asset_id = "rocks/granite";
        m.tint = { 0.2f, 0.4f, 0.6f };
        m.visible = false;
        auto r = round_trip(m);
        auto* rm = static_cast<ac::MeshComponent*>(r.get());
        CHECK(rm->asset_id == "rocks/granite");
        CHECK(fz(rm->tint.y - 0.4f));
        CHECK(rm->visible == false);
    }
    // Light (enum kind + scalars).
    {
        ac::LightComponent l;
        l.kind = ac::LightKind::Spot;
        l.color = { 1.0f, 0.5f, 0.25f };
        l.intensity = 7.5f;
        l.range = 42.0f;
        auto r = round_trip(l);
        auto* rl = static_cast<ac::LightComponent*>(r.get());
        CHECK(rl->kind == ac::LightKind::Spot);
        CHECK(fz(rl->intensity - 7.5f) && fz(rl->range - 42.0f));
        CHECK(fz(rl->color.z - 0.25f));
    }
    // RigidBody (authored params only).
    {
        ac::RigidBodyComponent rb;
        rb.mass = 9.0f; rb.linear_damping = 0.2f;
        rb.use_gravity = false; rb.kinematic = true;
        auto r = round_trip(rb);
        auto* rr = static_cast<ac::RigidBodyComponent*>(r.get());
        CHECK(fz(rr->mass - 9.0f) && fz(rr->linear_damping - 0.2f));
        CHECK(rr->use_gravity == false && rr->kinematic == true);
    }
    // Tag (count + tag_N lines, spaces allowed in tag text).
    {
        ac::TagComponent tg;
        tg.flags = 0x5u;
        tg.add("pickup");
        tg.add("high value");      // contains a space
        auto r = round_trip(tg);
        auto* rtg = static_cast<ac::TagComponent*>(r.get());
        CHECK(rtg->flags == 0x5u);
        CHECK(rtg->has("pickup"));
        CHECK(rtg->has("high value"));
        CHECK(rtg->tags.size() == sz(2));
    }
    // Tag flags with the HIGH BIT set must round-trip (u32, not signed
    // strtol which would saturate 0x80000000 at INT_MAX).
    {
        ac::TagComponent tg;
        tg.flags = 0x80000001u;     // bit 31 + bit 0
        auto r = round_trip(tg);
        auto* rtg = static_cast<ac::TagComponent*>(r.get());
        CHECK(rtg->flags == 0x80000001u);
    }
    // AudioEmitter channel high-bit round-trip (also u32).
    {
        ac::AudioEmitterComponent ae;
        ae.cue_id = "c"; ae.channel = 0xFFFFFFF0u;
        auto r = round_trip(ae);
        auto* rae = static_cast<ac::AudioEmitterComponent*>(r.get());
        CHECK(rae->channel == 0xFFFFFFF0u);
    }
    // Camera.
    {
        ac::CameraComponent c;
        c.fov_y_rad = 1.2f; c.z_near = 0.1f; c.z_far = 250.0f; c.active = true;
        auto r = round_trip(c);
        auto* rc = static_cast<ac::CameraComponent*>(r.get());
        CHECK(fz(rc->fov_y_rad - 1.2f) && fz(rc->z_far - 250.0f) && rc->active);
    }
    // Script.
    {
        ac::ScriptComponent s;
        s.entry_name = "onSpawn"; s.source_path = "ai/turret.lua"; s.enabled = false;
        auto r = round_trip(s);
        auto* rs = static_cast<ac::ScriptComponent*>(r.get());
        CHECK(rs->entry_name == "onSpawn" && rs->source_path == "ai/turret.lua");
        CHECK(rs->enabled == false);
    }
    // PlayerController (authored tunables; runtime state stays default).
    {
        ac::PlayerControllerComponent p;
        p.move_speed = 12.0f; p.jump_speed = 9.5f; p.fly_mode = true;
        auto r = round_trip(p);
        auto* rp = static_cast<ac::PlayerControllerComponent*>(r.get());
        CHECK(fz(rp->move_speed - 12.0f) && fz(rp->jump_speed - 9.5f));
        CHECK(rp->fly_mode == true);
        CHECK(rp->grounded() == true);   // runtime default, not serialized
    }
}

// ---- event bus ----------------------------------------------------
void test_event_bus() {
    ac::World w;
    std::vector<int> got;
    const auto id1 = w.subscribe("hit",
        [&](const std::any& p){ got.push_back(std::any_cast<int>(p)); });
    const auto id2 = w.subscribe("hit",
        [&](const std::any&){ got.push_back(-1); });
    CHECK(id1 == 1u && id2 == 2u);                       // monotonic

    w.broadcast("hit", std::any(42));
    CHECK(got.size() == sz(2) && got[0] == 42 && got[1] == -1);  // in order

    w.broadcast("unknown", std::any(7));                 // no subs → no-op
    CHECK(got.size() == sz(2));

    w.unsubscribe(id1);
    w.broadcast("hit", std::any(99));
    CHECK(got.size() == sz(3) && got[2] == -1);          // only id2 fired

    // Handler ids are monotonic ACROSS events.
    const auto id3 = w.subscribe("die", [&](const std::any&){ got.push_back(7); });
    CHECK(id3 == 3u);
    w.broadcast("die");                                  // empty payload ok
    CHECK(got.size() == sz(4) && got[3] == 7);

    w.unsubscribe(9999u);                                // unknown → no-op
    w.broadcast("die");
    CHECK(got.size() == sz(5));
}

// ---- broadcast must not UAF when handler mutates subscribers_ -----
// Same range-for-over-mutating-vector UAF class as sim 1f10242 /
// actor::World::tick f3ed9c1 / game 5057580 / partition 309abdf.
// The dispatch was `for (auto& s : it->second) if (s.fn) s.fn(...)`
// over the LIVE subscriber list, with `it` held across the call.
// Snapshot semantics (this commit): handlers added/removed during
// a broadcast take effect on the NEXT broadcast.
void test_event_bus_reentrant() {
    ac::World w;
    int orig_count   = 0;
    int added_count  = 0;
    bool first       = true;

    // First handler spawns 64 new subscribers into the SAME event on
    // its first invocation. 64 push_backs force at least one realloc
    // of subscribers_["hit"]'s vector; pre-fix the dispatch's range-
    // for iterator would dangle into the freed old buffer.
    w.subscribe("hit", [&](const std::any&) {
        ++orig_count;
        if (first) {
            first = false;
            for (int i = 0; i < 64; ++i) {
                w.subscribe("hit", [&](const std::any&) { ++added_count; });
            }
        }
    });

    // Broadcast 1: orig fires, registers 64 new — none fire this
    // broadcast (snapshot semantics — same contract as actor/game/sim
    // spawn-during-tick).
    w.broadcast("hit", std::any(1));
    CHECK(orig_count  == 1);
    CHECK(added_count == 0);

    // Broadcast 2: orig fires again (first=false, no more adds), AND
    // all 64 deferred handlers fire.
    w.broadcast("hit", std::any(2));
    CHECK(orig_count  == 2);
    CHECK(added_count == 64);

    // Cross-event re-entry: a handler on "hit" calls subscribe on a
    // DIFFERENT event. That could rehash subscribers_ as an
    // unordered_map → pre-fix the `it` reference into subscribers_
    // would dangle → UAF on the next loop iteration's `it->second`
    // deref. With the snapshot, the inner vector is already copied
    // before any callback runs, decoupling from the outer map.
    ac::World w2;
    int hit_count = 0;
    int new_event_count = 0;
    bool first2 = true;
    w2.subscribe("hit", [&](const std::any&) {
        ++hit_count;
        if (first2) {
            first2 = false;
            // Many new events to force the unordered_map to rehash.
            for (int i = 0; i < 32; ++i) {
                std::string name = "evt" + std::to_string(i);
                w2.subscribe(name, [&](const std::any&) { ++new_event_count; });
            }
        }
    });
    // Pre-fix this would UAF on the next iter; post-fix completes.
    // We have only one "hit" handler so the next-iter step is end,
    // but the SNAPSHOT removes any dependence on `it` anyway.
    w2.broadcast("hit", std::any(3));
    CHECK(hit_count == 1);
    CHECK(new_event_count == 0);   // new evts not broadcast this call
}

// ---- TransformComponent::matrix() ---------------------------------
void test_transform_matrix() {
    ac::TransformComponent t;                            // identity defaults
    const Mat4 m1 = t.matrix();
    const Mat4 m2 = t.matrix();
    CHECK(mat_eq(m1, m2));                                // deterministic
    CHECK(mat_eq(m1, Mat4::identity()));                  // T0·R0·S1 = I

    t.translation = { 3.0f, -2.0f, 0.0f };
    const Mat4 m3 = t.matrix();
    CHECK(!mat_eq(m3, Mat4::identity()));                 // now differs
    CHECK(mat_eq(m3, t.matrix()));                        // still deterministic
}

}  // namespace

int main() {
    test_actor_components();
    test_spawn_during_tick();
    test_player_controller_nonfinite_dt();
    test_player_controller_nonfinite_tunables();
    test_tag_component();
    test_lifecycle();
    test_world_lifecycle();
    test_blueprints();
    test_prefabs();
    test_prefab_library_mgmt();
    test_group_prefab();
    test_builtin_prefabs();
    test_find_all_queries();
    test_bulk_ops();
    test_align_distribute();
    test_snap_grid();
    test_validation();
    test_validation_autofix();
    test_world_stats();
    test_prefab_link_revert_apply();
    test_has_remove_component();
    test_enable_disable();
    test_revision();
    test_world_index();
    test_copy_paste_component();
    test_duplicate();
    test_spawn_placement();
    test_array_actor();
    test_array_grid();
    test_component_serialization();
    test_event_bus();
    test_event_bus_reentrant();
    test_transform_matrix();

    if (g_fail == 0) {
        cardinal::log::infof("actortest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("actortest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
