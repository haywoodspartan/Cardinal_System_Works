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
#include <cardinal/core/log.hpp>

#include <any>
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
    test_tag_component();
    test_lifecycle();
    test_world_lifecycle();
    test_blueprints();
    test_event_bus();
    test_transform_matrix();

    if (g_fail == 0) {
        cardinal::log::infof("actortest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("actortest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
