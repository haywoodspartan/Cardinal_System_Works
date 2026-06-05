// =============================================================================
// Cardinal — deterministic game-framework regression suite.
//
// Two pure, deterministic surfaces the editor + Play button depend on:
//
//   * ClassRegistry / reflection — CARDINAL_REGISTER_GAME_CLASS thunks +
//     PROP_* descriptors. find/all_names(sorted)/all_in_category(prefix,
//     sorted)/size/last-wins, the create() factory, and that each
//     PropertyDef.ptr aliases the live instance describe_properties was
//     called with (read/write round-trip per Kind).
//   * Game lifecycle — the Stopped/Playing/Paused state machine, its
//     transition guards + on_state_change (old,new) args, game_active,
//     and spawn_class begin_play/end_play broadcast vs the
//     spawn-while-Stopped begin_play_pending path.
//
// The test exe links only cardinal::game (+ its deps); nothing
// auto-registers classes, so the process-singleton registry contains
// exactly what this file registers — fully deterministic.
//
// Exit 0 = all pass.
// =============================================================================

#include <cardinal/game/game.hpp>
#include <cardinal/game/game_actor.hpp>
#include <cardinal/game/reflection.hpp>
#include <cardinal/sim/sim.hpp>
#include <cardinal/actor/world.hpp>
#include <cardinal/core/log.hpp>

namespace cg = cardinal::game;
using Vec3 = cardinal::scene::Vec3;

// ---- Reflected test classes (registered at static-init by the macro) --
class TestActor : public cg::GameActor {
public:
    float            hp        = 10.0f;
    int              ammo      = 7;
    bool             invisible = false;
    Vec3             muzzle{1.0f, 2.0f, 3.0f};
    Vec3             tint{0.5f, 0.25f, 0.125f};
    cardinal::string label = "init";

    static inline int s_begin = 0;
    static inline int s_end   = 0;
    static inline int s_tick  = 0;
    void begin_play() override { ++s_begin; }
    void end_play()   override { ++s_end;   }
    void on_tick(float) override { ++s_tick; }
};
CARDINAL_REGISTER_GAME_CLASS(TestActor, "AI/Test",
    PROP_FLOAT (hp,   0.0f, 100.0f, "Hit points")
    PROP_INT   (ammo, 0,    99,     "Ammo count")
    PROP_BOOL  (invisible,          "Cloak")
    PROP_VEC3  (muzzle,             "Muzzle offset")
    PROP_COLOR (tint,               "Tint")
    PROP_STRING(label,              "Display label"));

class OtherActor : public cg::GameActor {
public:
    float x = 1.0f;
};
CARDINAL_REGISTER_GAME_CLASS(OtherActor, "Gameplay/Other",
    PROP_FLOAT(x, -1.0f, 1.0f, "X"));

// A GameActor whose begin_play/end_play spawn a batch of actors — the
// common "spawn from a lifecycle hook" pattern. NOT registered with the
// macro (keeps the ClassRegistry exactly {TestActor, OtherActor} for
// test_reflection); added to an Actor directly via add_component.
class SpawnerGA : public cg::GameActor {
public:
    static inline cardinal::actor::World* s_world = nullptr;
    static inline int s_begin = 0, s_end = 0;
    static inline int s_did_begin = 0, s_did_end = 0;
    void begin_play() override {
        ++s_begin;
        if (s_world && s_did_begin == 0) {
            s_did_begin = 1;
            for (int k = 0; k < 512; ++k) s_world->spawn("bspawn");
        }
    }
    void end_play() override {
        ++s_end;
        if (s_world && s_did_end == 0) {
            s_did_end = 1;
            for (int k = 0; k < 512; ++k) s_world->spawn("espawn");
        }
    }
};

namespace {

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("gametest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool feq(float a, float b) {
    const float d = a > b ? a - b : b - a;
    return d <= 1e-6f;
}

const cg::PropertyDef* find_prop(const cardinal::vector<cg::PropertyDef>& v,
                                 const char* n) {
    for (const auto& p : v) if (p.name == n) return &p;
    return nullptr;
}

// ---- reflection registry + property descriptors -------------------
void test_reflection() {
    auto& reg = cg::ClassRegistry::instance();
    CHECK(reg.size() == 2u);                         // exactly our two
    CHECK(reg.find("TestActor")  != nullptr);
    CHECK(reg.find("OtherActor") != nullptr);
    CHECK(reg.find("Nope")       == nullptr);

    auto names = reg.all_names();                    // sorted
    CHECK(names.size() == 2u);
    CHECK(names[0] == "OtherActor" && names[1] == "TestActor");

    // Prefix match, sorted by (category, name).
    auto ai = reg.all_in_category("AI");
    CHECK(ai.size() == 1u && ai[0]->name == "TestActor");
    CHECK(reg.all_in_category("AI/Test").size() == 1u);
    CHECK(reg.all_in_category("Gameplay").size() == 1u);
    CHECK(reg.all_in_category("Zzz").empty());
    auto all = reg.all_in_category("");              // "" prefix ⇒ all
    CHECK(all.size() == 2u);
    CHECK(all[0]->name == "TestActor");              // "AI/Test" < "Gameplay/Other"
    CHECK(all[1]->name == "OtherActor");

    // Factory + descriptors bind the SAME instance.
    const cg::ClassDef* def = reg.find("TestActor");
    CHECK(def != nullptr && static_cast<bool>(def->create) &&
          static_cast<bool>(def->describe_properties));
    if (!def) return;
    auto inst = def->create();
    CHECK(inst != nullptr);
    auto* ta = static_cast<TestActor*>(inst.get());

    auto props = def->describe_properties(inst.get());
    CHECK(props.size() == 6u);
    CHECK(props[0].name == "hp" && props[1].name == "ammo" &&
          props[5].name == "label");                 // macro-arg order

    const cg::PropertyDef* php = find_prop(props, "hp");
    CHECK(php && php->kind == cg::PropertyKind::Float);
    CHECK(php && feq(php->fmin, 0.0f) && feq(php->fmax, 100.0f) &&
          php->tooltip == "Hit points");
    if (php) {
        CHECK(*static_cast<float*>(php->ptr) == ta->hp);    // aliases instance
        *static_cast<float*>(php->ptr) = 42.5f;             // write-through
        CHECK(feq(ta->hp, 42.5f));
    }
    const cg::PropertyDef* pa = find_prop(props, "ammo");
    CHECK(pa && pa->kind == cg::PropertyKind::Int &&
          pa->imin == 0 && pa->imax == 99);
    if (pa) { *static_cast<int*>(pa->ptr) = 13; CHECK(ta->ammo == 13); }
    const cg::PropertyDef* pb = find_prop(props, "invisible");
    CHECK(pb && pb->kind == cg::PropertyKind::Bool);
    if (pb) { *static_cast<bool*>(pb->ptr) = true; CHECK(ta->invisible); }
    const cg::PropertyDef* pv = find_prop(props, "muzzle");
    CHECK(pv && pv->kind == cg::PropertyKind::Vec3);
    if (pv) {
        auto* m = static_cast<Vec3*>(pv->ptr);
        CHECK(feq(m->x, 1.0f) && feq(m->y, 2.0f) && feq(m->z, 3.0f));
    }
    const cg::PropertyDef* pc = find_prop(props, "tint");
    CHECK(pc && pc->kind == cg::PropertyKind::Color);
    const cg::PropertyDef* ps = find_prop(props, "label");
    CHECK(ps && ps->kind == cg::PropertyKind::String);
    if (ps) {
        CHECK(*static_cast<cardinal::string*>(ps->ptr) == "init");
        *static_cast<cardinal::string*>(ps->ptr) = "renamed";
        CHECK(ta->label == "renamed");
    }
}

// ---- registry last-wins on duplicate name -------------------------
void test_reflection_lastwins() {
    auto& reg = cg::ClassRegistry::instance();
    cg::ClassDef d1; d1.name = "Dup"; d1.category = "X/One";
    reg.register_class(cardinal::move(d1));
    CHECK(reg.size() == 3u);
    CHECK(reg.find("Dup") != nullptr &&
          reg.find("Dup")->category == "X/One");
    cg::ClassDef d2; d2.name = "Dup"; d2.category = "Y/Two";
    reg.register_class(cardinal::move(d2));
    CHECK(reg.size() == 3u);                          // replaced, not added
    CHECK(reg.find("Dup")->category == "Y/Two");      // last wins
}

// ---- GameActor::clone (prefab capture of game-class actors) -------
void test_gameactor_clone() {
    // A source TestActor with non-default reflected values.
    TestActor src;
    src.set_class_name("TestActor");
    src.hp        = 73.5f;
    src.ammo      = 42;
    src.invisible = true;
    src.muzzle    = { 9.0f, 8.0f, 7.0f };
    src.tint      = { 0.1f, 0.2f, 0.3f };
    src.label     = "boss";
    src._set_playing(true);   // runtime flag — must NOT survive the clone

    // clone() returns the base Component type; it must be a GameActor.
    auto comp = src.clone();
    CHECK(comp != nullptr);
    auto* clone = dynamic_cast<cg::GameActor*>(comp.get());
    CHECK(clone != nullptr);
    if (clone == nullptr) return;

    // Concrete subclass + class name preserved.
    auto* tc = dynamic_cast<TestActor*>(clone);
    CHECK(tc != nullptr);
    CHECK(clone->class_name() == "TestActor");

    // Every reflected property value copied across.
    if (tc != nullptr) {
        CHECK(feq(tc->hp, 73.5f));
        CHECK(tc->ammo == 42);
        CHECK(tc->invisible == true);
        CHECK(feq(tc->muzzle.x, 9.0f) && feq(tc->muzzle.y, 8.0f) && feq(tc->muzzle.z, 7.0f));
        CHECK(feq(tc->tint.x, 0.1f) && feq(tc->tint.z, 0.3f));
        CHECK(tc->label == "boss");
    }

    // Runtime state is reset — fresh instance is unattached + not playing.
    CHECK(clone->playing() == false);
    CHECK(clone->owner() == nullptr);

    // Clone independence — mutating the clone leaves the source untouched.
    if (tc != nullptr) {
        tc->hp = 1.0f; tc->label = "minion";
        CHECK(feq(src.hp, 73.5f));
        CHECK(src.label == "boss");
    }

    // Unregistered class -> clone returns nullptr (logged, prefab omits it).
    TestActor orphan;
    orphan.set_class_name("NotRegistered");
    CHECK(orphan.clone() == nullptr);

    // End-to-end: a prefab captured from an actor carrying a GameActor
    // stamps an instance that still carries the cloned GameActor.
    cardinal::actor::World w;
    cardinal::actor::Actor* a = w.spawn("Hero");      // auto-Transform
    auto* ga = a->add_component<TestActor>();
    ga->set_class_name("TestActor");
    ga->hp = 55.0f;
    CHECK(w.create_prefab("HeroPrefab", a->id()));
    // Transform + GameActor both cloneable now -> 2 components.
    CHECK(w.prefab_component_count("HeroPrefab") == 2u);
    cardinal::actor::Actor* inst = w.spawn_prefab("HeroPrefab");
    CHECK(inst != nullptr);
    auto* inst_ga = inst->get_component<cg::GameActor>();
    CHECK(inst_ga != nullptr);
    if (inst_ga != nullptr) {
        auto* inst_tc = dynamic_cast<TestActor*>(inst_ga);
        CHECK(inst_tc != nullptr && feq(inst_tc->hp, 55.0f));
        CHECK(inst_ga->class_name() == "TestActor");
    }
}

// ---- game_state_name + GameActor accessors ------------------------
void test_state_name_and_actor() {
    CHECK(cardinal::string(cg::game_state_name(cg::GameState::Stopped)) == "Stopped");
    CHECK(cardinal::string(cg::game_state_name(cg::GameState::Playing)) == "Playing");
    CHECK(cardinal::string(cg::game_state_name(cg::GameState::Paused))  == "Paused");
    CHECK(static_cast<cardinal::u32>(cg::GameState::Stopped) == 0u);
    CHECK(static_cast<cardinal::u32>(cg::GameState::Paused)  == 2u);

    TestActor a;
    CHECK(cardinal::string(a.type_name()) == "GameActor");   // uniform type tag
    CHECK(a.class_name().empty());
    a.set_class_name("Foo");
    CHECK(a.class_name() == "Foo");
    CHECK(a.playing() == false);
    a._set_playing(true);
    CHECK(a.playing() == true);
}

// ---- Game state machine + spawn_class lifecycle -------------------
void test_game_lifecycle() {
    TestActor::s_begin = 0;
    TestActor::s_end   = 0;

    cardinal::sim::SimWorld sw{cardinal::sim::SimDesc{}};
    cg::Game g(sw);
    CHECK(g.state() == cg::GameState::Stopped);
    CHECK(!g.game_active());
    CHECK(g.begin_play_pending() == 0u);
    CHECK(g.game_actor_count() == 0u);

    int changes = 0;
    cg::GameState lo = cg::GameState::Stopped, ln = cg::GameState::Stopped;
    g.set_on_state_change([&](cg::GameState o, cg::GameState n) {
        lo = o; ln = n; ++changes;
    });

    // spawn while Stopped → begin_play deferred (pending++), not begun.
    CHECK(g.spawn_class("DoesNotExist") == nullptr);
    CHECK(g.spawn_class("TestActor") != nullptr);
    CHECK(g.game_actor_count() == 1u);
    CHECK(g.begin_play_pending() == 1u);
    CHECK(TestActor::s_begin == 0);

    g.start_play();                                   // Stopped → Playing
    CHECK(g.state() == cg::GameState::Playing && g.game_active());
    CHECK(changes == 1 && lo == cg::GameState::Stopped &&
          ln == cg::GameState::Playing);
    CHECK(TestActor::s_begin == 1);                   // broadcast begin_play
    CHECK(g.begin_play_pending() == 0u);
    g.start_play();                                   // no-op (not Stopped)
    CHECK(g.state() == cg::GameState::Playing && changes == 1);

    // spawn while Playing → begin_play immediately.
    CHECK(g.spawn_class("TestActor", "second") != nullptr);
    CHECK(g.game_actor_count() == 2u);
    CHECK(TestActor::s_begin == 2);

    g.pause_play();                                   // Playing → Paused
    CHECK(g.state() == cg::GameState::Paused && g.game_active());
    CHECK(changes == 2 && lo == cg::GameState::Playing &&
          ln == cg::GameState::Paused);
    g.pause_play();                                   // no-op (not Playing)
    CHECK(g.state() == cg::GameState::Paused && changes == 2);

    g.resume_play();                                  // Paused → Playing
    CHECK(g.state() == cg::GameState::Playing);
    CHECK(changes == 3 && lo == cg::GameState::Paused &&
          ln == cg::GameState::Playing);
    g.resume_play();                                  // no-op (not Paused)
    CHECK(g.state() == cg::GameState::Playing && changes == 3);

    g.stop_play();                                    // Playing → Stopped
    CHECK(g.state() == cg::GameState::Stopped && !g.game_active());
    CHECK(changes == 4 && ln == cg::GameState::Stopped);
    CHECK(TestActor::s_end == 2);                     // end_play on both
    g.stop_play();                                    // no-op (Stopped)
    CHECK(g.state() == cg::GameState::Stopped && changes == 4);

    // pause/resume are no-ops from Stopped.
    g.pause_play();  CHECK(g.state() == cg::GameState::Stopped && changes == 4);
    g.resume_play(); CHECK(g.state() == cg::GameState::Stopped && changes == 4);
}

// ---- Game::tick → apply_lifecycle_ catch-up + on_tick gating -------
// The deferred-begin_play path: resume_play() does NOT broadcast (only
// start_play does), so a spawn-while-Paused stays pending until a
// Playing tick's PreUpdate handler runs apply_lifecycle_.
void test_game_tick_lifecycle() {
    TestActor::s_begin = 0;
    TestActor::s_end   = 0;
    TestActor::s_tick  = 0;

    cardinal::sim::SimDesc sd{};
    sd.fixed_dt      = 0.1f;                          // exact substep math
    sd.max_substeps  = 8u;
    cardinal::sim::SimWorld sw{sd};
    cg::Game g(sw);

    g.start_play();                                   // Stopped → Playing
    g.pause_play();                                   // → Paused

    cardinal::actor::Actor* a = g.spawn_class("TestActor", "deferred");
    CHECK(a != nullptr);
    CHECK(g.begin_play_pending() == 1u);              // Paused ⇒ deferred
    CHECK(TestActor::s_begin == 0);

    g.resume_play();                                  // Paused → Playing
    CHECK(g.state() == cg::GameState::Playing);
    CHECK(g.begin_play_pending() == 1u);              // resume does NOT broadcast
    CHECK(TestActor::s_begin == 0);

    g.tick(0.25f);                                    // ≥2 fixed substeps
    CHECK(g.begin_play_pending() == 0u);              // apply_lifecycle_ caught it
    CHECK(TestActor::s_begin == 1);                   // begun exactly once
    CHECK(TestActor::s_tick  >= 1);                   // Update handler on_ticked it

    const int tick_before = TestActor::s_tick;
    g.tick(0.25f);                                    // no re-begin; keeps ticking
    CHECK(TestActor::s_begin == 1);
    CHECK(TestActor::s_tick  > tick_before);

    g.stop_play();                                    // Playing → Stopped
    CHECK(g.state() == cg::GameState::Stopped);
    CHECK(TestActor::s_end == 1);                     // end_play broadcast

    const int tick_stopped = TestActor::s_tick;
    g.tick(0.5f);                                     // Stopped ⇒ Update gated off
    CHECK(TestActor::s_tick == tick_stopped);
}

// ---- begin_play_pending_ must not stick when pending actor dies ---
// Parked-supervised regression: a GameActor spawned while !Playing
// (Paused/Stopped) increments begin_play_pending_; broadcast_begin_
// play_ from start_play() unconditionally resets it, but resume_play()
// does NOT call broadcast and instead relies on apply_lifecycle_ to
// fire deferred begin_plays under the running tick. If the spawned
// actor is destroyed between spawn and the next Playing PreUpdate,
// apply_lifecycle_'s sweep sees !alive() and skips it; the pre-fix
// `pending = (fired>=pending) ? 0 : (pending-fired)` left
// pending=(spawn_count - 0) FOREVER, and apply_lifecycle_ then ran
// a no-op O(N) actors sweep every PreUpdate tick from then on.
void test_lifecycle_pending_stale_destroy() {
    TestActor::s_begin = 0;
    TestActor::s_end   = 0;
    TestActor::s_tick  = 0;

    cardinal::sim::SimDesc sd{};
    sd.fixed_dt     = 0.1f;
    sd.max_substeps = 8u;
    cardinal::sim::SimWorld sw{sd};
    cg::Game g(sw);

    g.start_play();                                   // → Playing
    g.pause_play();                                   // → Paused (Sim halted)

    // Spawn while Paused — deferred begin_play (++pending).
    cardinal::actor::Actor* a = g.spawn_class("TestActor", "doomed");
    CHECK(a != nullptr);
    CHECK(g.begin_play_pending() == 1u);
    CHECK(TestActor::s_begin == 0);                   // not yet begun

    // Destroy the pending actor BEFORE its begin_play. world.destroy
    // marks dead; sweep happens inside the next non-paused sim tick.
    sw.world().destroy(a->id());

    g.resume_play();                                  // → Playing
    CHECK(g.begin_play_pending() == 1u);              // resume doesn't broadcast

    // First Playing tick: apply_lifecycle_ runs the sweep. The doomed
    // actor is !alive() so it's skipped (fired=0). Pre-fix: pending
    // stayed at 1 forever. Post-fix: full sweep resets pending = 0.
    g.tick(0.25f);                                    // ≥1 PreUpdate
    CHECK(g.begin_play_pending() == 0u);              // RESET (was 1 pre-fix)
    CHECK(TestActor::s_begin == 0);                   // doomed never began

    // A subsequent spawn while Playing fires begin_play IMMEDIATELY
    // (state==Playing branch in spawn_class) and does NOT re-arm
    // pending — proves the counter is genuinely cleared and works
    // normally going forward.
    cardinal::actor::Actor* b = g.spawn_class("TestActor", "live");
    CHECK(b != nullptr);
    CHECK(g.begin_play_pending() == 0u);              // immediate, not deferred
    CHECK(TestActor::s_begin == 1);

    g.stop_play();
    CHECK(TestActor::s_end == 1);                     // only live actor ended
}

// ---- end_play fires on stop FROM Paused + spawn_class identity ----
void test_stop_from_paused_and_identity() {
    TestActor::s_begin = 0;
    TestActor::s_end   = 0;
    cardinal::sim::SimWorld sw{cardinal::sim::SimDesc{}};
    cg::Game g(sw);
    g.start_play();
    cardinal::actor::Actor* a = g.spawn_class("TestActor");   // Playing ⇒ immediate
    CHECK(a != nullptr);
    CHECK(TestActor::s_begin == 1);

    // The spawned GameActor is retrievable + carries the class identity.
    auto* ga = a ? a->get_component<cg::GameActor>() : nullptr;
    CHECK(ga != nullptr);
    if (ga) {
        CHECK(ga->class_name() == "TestActor");
        CHECK(cardinal::string(ga->type_name()) == "GameActor");
        CHECK(ga->playing() == true);
    }

    g.pause_play();                                   // pause does NOT end_play
    CHECK(g.state() == cg::GameState::Paused);
    CHECK(TestActor::s_end == 0);
    if (ga) CHECK(ga->playing() == true);             // still playing while paused

    g.stop_play();                                    // Paused → Stopped
    CHECK(g.state() == cg::GameState::Stopped);
    CHECK(TestActor::s_end == 1);                     // end_play fired from Paused
    if (ga) CHECK(ga->playing() == false);
}

// ---- game.cpp lifecycle loops must survive spawn-during-iteration --
// Regression: broadcast_begin_play_ / broadcast_end_play_ /
// apply_lifecycle_ / update_h_ iterated `for (auto& aptr :
// sim_.world().actors())`. A GameActor whose begin_play()/end_play()/
// on_tick() spawns (→ world.spawn() → actors_.push_back realloc) — a
// common pattern — reallocated the vector mid-iteration; the range-for
// then dangled into the freed old buffer whose unique_ptrs were
// moved-from to null → a deterministic null-deref crash on the next
// aptr->alive(). (Needs ≥2 original actors so the outer loop actually
// dereferences a freed slot after the realloc.)
void test_lifecycle_spawn_during_iteration() {
    SpawnerGA::s_world = nullptr;
    SpawnerGA::s_begin = SpawnerGA::s_end = 0;
    SpawnerGA::s_did_begin = SpawnerGA::s_did_end = 0;

    cardinal::sim::SimWorld sw{cardinal::sim::SimDesc{}};
    cg::Game g(sw);
    cardinal::actor::Actor* a = sw.world().spawn("spawner");   // index 0
    a->add_component<SpawnerGA>();
    for (int k = 0; k < 7; ++k) sw.world().spawn("plain");      // ≥2 total
    SpawnerGA::s_world = &sw.world();
    const cardinal::usize before = sw.world().actor_count();    // 8
    CHECK(before == 8u);

    g.start_play();   // broadcast_begin_play_ → begin_play() spawns 512
                      // mid-iteration. PRE-FIX: null-deref crash.
    CHECK(g.state() == cg::GameState::Playing);
    CHECK(SpawnerGA::s_begin == 1);
    CHECK(sw.world().actor_count() == before + 512u);
    CHECK(g.begin_play_pending() == 0u);

    g.stop_play();    // broadcast_end_play_ → end_play() spawns 512 more.
    CHECK(g.state() == cg::GameState::Stopped);
    CHECK(SpawnerGA::s_end == 1);
    CHECK(sw.world().actor_count() == before + 1024u);

    // A Playing tick now drives update_h_ + apply_lifecycle_ (same
    // fixed loops) over 1032 actors — must not crash.
    g.start_play();
    g.tick(0.016f);
    CHECK(g.state() == cg::GameState::Playing);
    CHECK(sw.world().actor_count() == before + 1024u);          // no new spawn
}

}  // namespace

int main() {
    test_reflection();
    test_reflection_lastwins();
    test_gameactor_clone();
    test_state_name_and_actor();
    test_game_lifecycle();
    test_game_tick_lifecycle();
    test_stop_from_paused_and_identity();
    test_lifecycle_spawn_during_iteration();
    test_lifecycle_pending_stale_destroy();

    if (g_fail == 0) {
        cardinal::log::infof("gametest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("gametest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
