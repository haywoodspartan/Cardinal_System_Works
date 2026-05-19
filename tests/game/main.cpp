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
    void begin_play() override { ++s_begin; }
    void end_play()   override { ++s_end;   }
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

}  // namespace

int main() {
    test_reflection();
    test_reflection_lastwins();
    test_state_name_and_actor();
    test_game_lifecycle();

    if (g_fail == 0) {
        cardinal::log::infof("gametest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("gametest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
