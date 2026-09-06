// =============================================================================
// Cardinal — Gameplay kit regression suite.
//
// Pins the quaternion transform ops (rotate local/world, look-at, turn-
// towards, orbit, slerp/look_rotation in core math), the TimerSet scheduler
// (one-shot, repeating, cancel, add-during-tick deferral), and every Kit
// GameActor driven through a real Game (Rotator, Orbiter, LookAt,
// WaypointMover, Lifetime, Spawner) on hand-computed expectations. Pure
// CPU, deterministic, headless. Exit 0 = all pass.
// =============================================================================

#include <cardinal/game/kit.hpp>
#include <cardinal/game/timer.hpp>
#include <cardinal/game/game.hpp>
#include <cardinal/game/reflection.hpp>
#include <cardinal/actor/transform_ops.hpp>
#include <cardinal/actor/world.hpp>
#include <cardinal/sim/sim.hpp>
#include <cardinal/core/diag/log.hpp>

#include <cmath>

namespace {

namespace cg = cardinal::game;
namespace xf = cardinal::actor::xf;
using cardinal::actor::TransformComponent;
using cardinal::actor::TagComponent;
using cardinal::scene::Vec3;
using cardinal::core::Quat;
using cardinal::u32;

int g_checks = 0;
int g_fail   = 0;
void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) { ++g_fail; cardinal::log::errorf("kittest", "FAIL  L%d  %s", line, expr); }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

constexpr float kPi = 3.14159265358979f;
bool ap(float a, float b, float e = 1e-3f) { return std::fabs(a - b) <= e; }
bool ap3(const Vec3& a, const Vec3& b, float e = 1e-3f) {
    return ap(a.x, b.x, e) && ap(a.y, b.y, e) && ap(a.z, b.z, e);
}
float len3(const Vec3& v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

// Set the GameActor's reflected property by name (what the inspector does).
template <class T>
bool set_prop(cg::GameActor* ga, const char* name, const T& value) {
    const cg::ClassDef* def = cg::ClassRegistry::instance().find(ga->class_name());
    if (!def || !def->describe_properties) return false;
    for (auto& p : def->describe_properties(ga))
        if (p.name == name) { *static_cast<T*>(p.ptr) = value; return true; }
    return false;
}
template <class T>
T get_prop(cg::GameActor* ga, const char* name) {
    const cg::ClassDef* def = cg::ClassRegistry::instance().find(ga->class_name());
    for (auto& p : def->describe_properties(ga))
        if (p.name == name) return *static_cast<T*>(p.ptr);
    return T{};
}
cg::GameActor* game_actor(cardinal::actor::Actor* a) {
    return a ? a->get_component<cg::GameActor>() : nullptr;
}
TransformComponent* xform(cardinal::actor::Actor* a) {
    return a ? a->get_component<TransformComponent>() : nullptr;
}
// Advance a Playing game by `seconds` in 0.1 s real ticks (SimDesc clamps
// real dt to 0.1, so one call per 0.1 s keeps sim time == wall time).
void run(cg::Game& g, float seconds) {
    const int n = static_cast<int>(seconds / 0.1f + 0.5f);
    for (int i = 0; i < n; ++i) g.tick(0.1f);
}

// ---- core quaternion helpers ------------------------------------------------
void test_quat_helpers() {
    using namespace cardinal::core;
    const Quat id = Quat::identity();
    const Quat y90 = axis_angle(Vec3{0, 1, 0}, kPi * 0.5f);
    CHECK(ap(quat_angle(id, y90), kPi * 0.5f));
    CHECK(ap(quat_angle(id, id), 0.0f));
    // slerp halfway = 45° about Y; forward (-Z) lands at (-sin45, 0, -cos45).
    const Quat h = slerp(id, y90, 0.5f);
    const Vec3 f = rotate(h, Vec3{0, 0, -1});
    CHECK(ap3(f, Vec3{-0.70711f, 0.0f, -0.70711f}));
    CHECK(ap(quat_angle(id, h), kPi * 0.25f));
    // slerp picks the short arc even when b is the negated quaternion.
    const Quat neg{ -y90.x, -y90.y, -y90.z, -y90.w };
    CHECK(ap(quat_angle(id, slerp(id, neg, 0.5f)), kPi * 0.25f));
    // look_rotation: forward maps to the requested direction, up stays up.
    const Vec3 dirs[] = { {0,0,-1}, {1,0,0}, {0,0,1}, {-1,0,0}, {0.6f,0.3f,-0.7f}, {0,1,0}, {0,-1,0} };
    for (const Vec3& d : dirs) {
        const Quat q = look_rotation(d, Vec3{0, 1, 0});
        const Vec3 got = rotate(q, Vec3{0, 0, -1});
        CHECK(ap3(got, normalize(d), 2e-3f));
        // orthonormal: right ⟂ forward
        const Vec3 r = rotate(q, Vec3{1, 0, 0});
        CHECK(ap(dot(r, got), 0.0f, 2e-3f));
    }
    // Facing +X with the default up: right should be -Z (right-handed frame).
    const Quat qx = look_rotation(Vec3{1, 0, 0}, Vec3{0, 1, 0});
    CHECK(ap3(rotate(qx, Vec3{1, 0, 0}), Vec3{0, 0, 1}));
    CHECK(ap3(rotate(qx, Vec3{0, 1, 0}), Vec3{0, 1, 0}));
}

// ---- transform ops ----------------------------------------------------------
void test_transform_ops() {
    TransformComponent t;
    CHECK(ap3(xf::forward(t), Vec3{0, 0, -1}));
    CHECK(ap3(xf::right(t),   Vec3{1, 0, 0}));
    CHECK(ap3(xf::up(t),      Vec3{0, 1, 0}));

    // 90° about +Y: forward -Z → -X.
    xf::rotate_local(t, Vec3{0, 1, 0}, kPi * 0.5f);
    CHECK(ap3(xf::forward(t), Vec3{-1, 0, 0}));
    CHECK(ap(t.rotation_euler.y, kPi * 0.5f, 1e-3f));
    // Another 90° local about +Y: -X → +Z (composition, not Euler add).
    xf::rotate_local(t, Vec3{0, 1, 0}, kPi * 0.5f);
    CHECK(ap3(xf::forward(t), Vec3{0, 0, 1}));
    // World vs local differ once the frame is rotated: pitch 90° about local
    // X vs world X give different forwards.
    TransformComponent a, b;
    a.rotation_euler = t.rotation_euler;
    b.rotation_euler = t.rotation_euler;
    xf::rotate_local(a, Vec3{1, 0, 0}, kPi * 0.5f);
    xf::rotate_world(b, Vec3{1, 0, 0}, kPi * 0.5f);
    CHECK(!ap3(xf::forward(a), xf::forward(b), 1e-2f));
    // Degenerate inputs are no-ops.
    TransformComponent z;
    xf::rotate_local(z, Vec3{0, 0, 0}, 1.0f);
    xf::rotate_world(z, Vec3{0, 1, 0}, std::nanf(""));
    CHECK(ap3(xf::forward(z), Vec3{0, 0, -1}));

    // look_at: from origin towards +X → forward +X; towards -Z → identity.
    TransformComponent l;
    xf::look_at(l, Vec3{5, 0, 0});
    CHECK(ap3(xf::forward(l), Vec3{1, 0, 0}));
    xf::look_at(l, Vec3{0, 0, -3});
    CHECK(ap3(xf::forward(l), Vec3{0, 0, -1}));
    CHECK(ap3(l.rotation_euler, Vec3{0, 0, 0}, 1e-3f));
    xf::look_at(l, l.translation);            // degenerate: unchanged
    CHECK(ap3(xf::forward(l), Vec3{0, 0, -1}));
    // Yaw-only ignores the target's height.
    TransformComponent ly;
    xf::look_at_yaw(ly, Vec3{3, 10, 0});
    CHECK(ap3(xf::forward(ly), Vec3{1, 0, 0}));

    // turn_towards: limited to max_rad per call, then reaches the goal.
    TransformComponent tt;                    // facing -Z; goal +X = 90° away
    float rem = xf::turn_towards(tt, Vec3{5, 0, 0}, kPi * 0.25f);
    CHECK(ap(rem, kPi * 0.25f));
    CHECK(ap(xf::angle_to(tt, xf::rotation(TransformComponent{})), kPi * 0.25f));
    rem = xf::turn_towards(tt, Vec3{5, 0, 0}, kPi * 0.25f);
    CHECK(ap(rem, 0.0f) && ap3(xf::forward(tt), Vec3{1, 0, 0}));
    rem = xf::turn_towards(tt, Vec3{5, 0, 0}, kPi);   // already there
    CHECK(ap(rem, 0.0f));

    // orbit: (1,0,0) about +Y by 90° around the origin → (0,0,-1), and the
    // orientation turns with it.
    TransformComponent o;
    o.translation = Vec3{1, 0, 0};
    xf::orbit(o, Vec3{0, 0, 0}, Vec3{0, 1, 0}, kPi * 0.5f);
    CHECK(ap3(o.translation, Vec3{0, 0, -1}));
    CHECK(ap3(xf::forward(o), Vec3{-1, 0, 0}));
    // Four quarter orbits return home (radius preserved).
    for (int i = 0; i < 3; ++i) xf::orbit(o, Vec3{0, 0, 0}, Vec3{0, 1, 0}, kPi * 0.5f);
    CHECK(ap3(o.translation, Vec3{1, 0, 0}));
    // set_rotation / rotation round-trip.
    TransformComponent rr;
    const Quat q = cardinal::core::axis_angle(cardinal::core::normalize(Vec3{1, 2, 3}), 0.7f);
    xf::set_rotation(rr, q);
    CHECK(ap(cardinal::core::quat_angle(xf::rotation(rr), q), 0.0f, 2e-3f));
}

// ---- TimerSet ---------------------------------------------------------------
void test_timers() {
    cg::TimerSet ts;
    int once = 0, rep = 0, nested = 0;
    auto h1 = ts.after(1.0f, [&] { ++once; });
    auto h2 = ts.every(0.5f, [&] { ++rep; });
    CHECK(h1.valid() && h2.valid() && ts.active_count() == 2u);
    CHECK(ap(ts.remaining(h1), 1.0f));
    CHECK(ts.tick(0.4f) == 0u);
    CHECK(ts.tick(0.2f) == 1u && rep == 1 && once == 0);   // 0.6 s: repeat fired
    CHECK(ts.tick(0.4f) == 2u && once == 1 && rep == 2);   // 1.0 s: one-shot + 2nd repeat
    CHECK(ts.active_count() == 1u && !ts.active(h1));
    CHECK(ts.tick(0.1f) == 0u && rep == 2);                // 1.1 s: next repeat is at 1.5
    CHECK(ts.tick(0.4f) == 1u && rep == 3);                // 1.5 s
    // Cancel stops it for good.
    CHECK(ts.cancel(h2));
    CHECK(!ts.cancel(h2));
    CHECK(ts.tick(5.0f) == 0u && rep == 3 && ts.active_count() == 0u);
    // A timer added inside a callback fires from the NEXT tick, not this one.
    cg::TimerSet ts2;
    ts2.after(0.0f, [&] { ts2.after(0.0f, [&] { ++nested; }); });
    CHECK(ts2.tick(0.1f) == 1u && nested == 0);
    CHECK(ts2.tick(0.1f) == 1u && nested == 1);
    // A repeating timer fires at most once per tick (no catch-up burst).
    cg::TimerSet ts3;
    int burst = 0;
    ts3.every(0.1f, [&] { ++burst; });
    CHECK(ts3.tick(10.0f) == 1u && burst == 1);
    // Non-finite dt is ignored.
    CHECK(ts3.tick(std::nanf("")) == 0u);
    // Cancel from inside the callback is honoured.
    cg::TimerSet ts4;
    int self_cancel = 0;
    cg::TimerSet::Handle hs;
    hs = ts4.every(0.1f, [&] { ++self_cancel; ts4.cancel(hs); });
    ts4.tick(0.2f); ts4.tick(0.2f);
    CHECK(self_cancel == 1 && ts4.active_count() == 0u);
}

// ---- Kit actors through a real Game ----------------------------------------
struct GameFixture {
    cardinal::sim::SimWorld sw;
    cg::Game game;
    GameFixture() : sw(make_desc()), game(sw) {}
    static cardinal::sim::SimDesc make_desc() {
        cardinal::sim::SimDesc d{};
        d.fixed_dt = 0.1f; d.max_real_dt = 0.1f; d.max_substeps = 4u;
        return d;
    }
};

void test_kit_registered() {
    CHECK(cg::kit::register_gameplay_kit() == 6u);
    for (const auto& n : cg::kit::kit_class_names())
        CHECK(cg::ClassRegistry::instance().find(n) != nullptr);
    const auto motion = cg::ClassRegistry::instance().all_in_category("Kit/Motion");
    CHECK(motion.size() >= 4u);
}

void test_rotator() {
    GameFixture f;
    auto* a = f.game.spawn_class("RotatorActor", "rot");
    CHECK(a != nullptr);
    if (!a) return;
    auto* ga = game_actor(a);
    CHECK(set_prop<float>(ga, "speed_deg_s", 90.0f));
    CHECK(set_prop<Vec3>(ga, "axis", Vec3{0, 1, 0}));
    f.game.start_play();
    run(f.game, 1.0f);                                  // 90°
    CHECK(ap3(xf::forward(*xform(a)), Vec3{-1, 0, 0}, 5e-3f));
    CHECK(ap(get_prop<float>(ga, "turned_deg"), 90.0f, 1e-2f));
    run(f.game, 1.0f);                                  // 180°
    CHECK(ap3(xf::forward(*xform(a)), Vec3{0, 0, 1}, 5e-3f));
    // Stopped → no more turning.
    f.game.stop_play();
    run(f.game, 1.0f);
    CHECK(ap3(xf::forward(*xform(a)), Vec3{0, 0, 1}, 5e-3f));
}

void test_orbiter() {
    GameFixture f;
    auto* a = f.game.spawn_class("OrbiterActor", "orb");
    CHECK(a != nullptr);
    if (!a) return;
    auto* ga = game_actor(a);
    set_prop<Vec3>(ga, "center", Vec3{0, 0, 0});
    set_prop<float>(ga, "radius", 3.0f);
    set_prop<float>(ga, "speed_deg_s", 90.0f);
    set_prop<bool>(ga, "face_center", true);
    f.game.start_play();                                // begin_play places it
    const Vec3 p0 = xform(a)->translation;
    CHECK(ap(len3(p0), 3.0f));
    run(f.game, 1.0f);                                  // quarter turn
    const Vec3 p1 = xform(a)->translation;
    CHECK(ap(len3(p1), 3.0f));
    CHECK(ap(cardinal::core::dot(p0, p1), 0.0f, 2e-2f));   // 90° apart
    CHECK(ap(p1.y, 0.0f));                              // stays in the XZ plane
    // Facing the centre: forward points from the actor to the origin.
    CHECK(ap3(xf::forward(*xform(a)), cardinal::core::normalize(Vec3{0, 0, 0} - p1), 5e-3f));
}

void test_look_at() {
    GameFixture f;
    auto* target = f.game.world().spawn_at("target", Vec3{5, 0, 0});
    target->add_component<TagComponent>()->add("player");
    auto* far = f.game.world().spawn_at("far", Vec3{-50, 0, 0});
    far->add_component<TagComponent>()->add("player");
    auto* a = f.game.spawn_class("LookAtActor", "looker");
    CHECK(a != nullptr);
    if (!a) return;
    auto* ga = game_actor(a);
    set_prop<cardinal::string>(ga, "target_tag", cardinal::string("player"));
    set_prop<float>(ga, "turn_rate_deg_s", 0.0f);       // snap
    f.game.start_play();
    run(f.game, 0.1f);
    CHECK(ap3(xf::forward(*xform(a)), Vec3{1, 0, 0}, 5e-3f));   // nearest wins
    // Limited turn rate: 45°/s towards a target at -X (180° away) → after
    // 1 s it is 45° along, after 4 s it has arrived.
    xform(target)->translation = Vec3{-5, 0, 0};
    xform(far)->translation    = Vec3{-500, 0, 0};
    set_prop<float>(ga, "turn_rate_deg_s", 45.0f);
    run(f.game, 1.0f);
    CHECK(ap(cardinal::core::quat_angle(xf::rotation(*xform(a)),
             cardinal::core::look_rotation(Vec3{-1, 0, 0}, Vec3{0, 1, 0})), kPi * 0.75f, 2e-2f));
    run(f.game, 3.5f);
    CHECK(ap3(xf::forward(*xform(a)), Vec3{-1, 0, 0}, 1e-2f));
    // yaw_only: target high above → forward stays level.
    xform(target)->translation = Vec3{3, 40, 0};
    set_prop<bool>(ga, "yaw_only", true);
    set_prop<float>(ga, "turn_rate_deg_s", 0.0f);
    run(f.game, 0.1f);
    CHECK(ap(xf::forward(*xform(a)).y, 0.0f, 1e-3f));
    CHECK(ap3(xf::forward(*xform(a)), Vec3{1, 0, 0}, 5e-3f));
}

void test_waypoint_mover() {
    GameFixture f;
    auto* a = f.game.spawn_class("WaypointMoverActor", "mover");
    CHECK(a != nullptr);
    if (!a) return;
    auto* ga = game_actor(a);
    set_prop<Vec3>(ga, "waypoint_a", Vec3{0, 0, 0});
    set_prop<Vec3>(ga, "waypoint_b", Vec3{10, 0, 0});
    set_prop<int>(ga, "waypoint_count", 2);
    set_prop<float>(ga, "speed", 5.0f);
    set_prop<int>(ga, "loop_mode", 0);                  // loop
    xform(a)->translation = Vec3{0, 0, 0};
    f.game.start_play();
    run(f.game, 1.0f);                                  // index 0 = A (already there) → heads to B
    CHECK(ap(xform(a)->translation.x, 5.0f, 1e-2f));
    CHECK(ap3(xf::forward(*xform(a)), Vec3{1, 0, 0}, 5e-3f));   // faces travel
    run(f.game, 1.5f);                                  // reached B at t=2, back towards A
    CHECK(xform(a)->translation.x < 10.0f && xform(a)->translation.x > 5.0f);
    CHECK(ap3(xf::forward(*xform(a)), Vec3{-1, 0, 0}, 5e-3f));
    // Once mode stops at the last point.
    GameFixture f2;
    auto* b = f2.game.spawn_class("WaypointMoverActor", "once");
    auto* gb = game_actor(b);
    set_prop<Vec3>(gb, "waypoint_a", Vec3{0, 0, 0});
    set_prop<Vec3>(gb, "waypoint_b", Vec3{2, 0, 0});
    set_prop<float>(gb, "speed", 5.0f);
    set_prop<int>(gb, "loop_mode", 2);
    f2.game.start_play();
    run(f2.game, 3.0f);
    CHECK(ap3(xform(b)->translation, Vec3{2, 0, 0}, 1e-3f));
    // Ping-pong with 3 points bounces between the ends.
    GameFixture f3;
    auto* c = f3.game.spawn_class("WaypointMoverActor", "pp");
    auto* gc = game_actor(c);
    set_prop<Vec3>(gc, "waypoint_a", Vec3{0, 0, 0});
    set_prop<Vec3>(gc, "waypoint_b", Vec3{1, 0, 0});
    set_prop<Vec3>(gc, "waypoint_c", Vec3{2, 0, 0});
    set_prop<int>(gc, "waypoint_count", 3);
    set_prop<float>(gc, "speed", 1.0f);
    set_prop<int>(gc, "loop_mode", 1);
    f3.game.start_play();
    run(f3.game, 3.0f);                                 // A→B→C (t=2) then back towards B
    CHECK(xform(c)->translation.x < 2.0f && xform(c)->translation.x > 0.5f);
}

void test_lifetime() {
    GameFixture f;
    auto* a = f.game.spawn_class("LifetimeActor", "short");
    CHECK(a != nullptr);
    if (!a) return;
    const auto id = a->id();
    set_prop<float>(game_actor(a), "seconds", 0.45f);
    f.game.start_play();
    run(f.game, 0.3f);
    CHECK(f.game.world().find(id) != nullptr && f.game.world().find(id)->alive());
    run(f.game, 0.5f);
    auto* after = f.game.world().find(id);
    CHECK(after == nullptr || !after->alive());
}

void test_spawner() {
    GameFixture f;
    auto& w = f.game.world();
    // Prefab template: a tagged actor captured into the prefab library.
    auto* tpl = w.spawn_at("CrateTemplate", Vec3{100, 0, 0});
    tpl->add_component<TagComponent>()->add("crate");
    CHECK(w.create_prefab("Crate", tpl->id()));
    w.destroy(tpl->id());
    auto* s = f.game.spawn_class("SpawnerActor", "spawner");
    CHECK(s != nullptr);
    if (!s) return;
    auto* gs = game_actor(s);
    set_prop<cardinal::string>(gs, "prefab_name", cardinal::string("Crate"));
    set_prop<float>(gs, "interval_s", 0.5f);
    set_prop<int>(gs, "max_alive", 2);
    set_prop<Vec3>(gs, "offset", Vec3{0, 1, 0});
    set_prop<bool>(gs, "spawn_on_begin", true);
    xform(s)->translation = Vec3{3, 0, 3};
    f.game.start_play();
    run(f.game, 0.1f);                                  // spawn_on_begin
    auto crates = [&] { u32 n = 0; for (auto* a : w.find_all_by_tag("crate")) if (a->alive()) ++n; return n; };
    CHECK(crates() == 1u);
    run(f.game, 2.0f);                                  // capped at max_alive
    CHECK(crates() == 2u);
    CHECK(get_prop<int>(gs, "spawned_total") == 2);
    // Spawned at the owner + offset.
    for (auto* a : w.find_all_by_tag("crate"))
        if (a->alive()) CHECK(ap3(xform(a)->translation, Vec3{3, 1, 3}));
    // Killing one frees a slot: the spawner refills it.
    for (auto* a : w.find_all_by_tag("crate")) { if (a->alive()) { w.destroy(a->id()); break; } }
    run(f.game, 1.0f);
    CHECK(crates() == 2u);
    CHECK(get_prop<int>(gs, "spawned_total") == 3);
    // Unknown prefab: nothing spawns, no crash.
    set_prop<cardinal::string>(gs, "prefab_name", cardinal::string("Nope"));
    w.bulk_destroy(w.find_all_by_tag("crate").empty() ? cardinal::vector<cardinal::actor::ActorId>{} :
                   [&] { cardinal::vector<cardinal::actor::ActorId> ids; for (auto* a : w.find_all_by_tag("crate")) ids.push_back(a->id()); return ids; }());
    run(f.game, 2.0f);
    CHECK(crates() == 0u);
}

}  // namespace

int main() {
    test_quat_helpers();
    test_transform_ops();
    test_timers();
    test_kit_registered();
    test_rotator();
    test_orbiter();
    test_look_at();
    test_waypoint_mover();
    test_lifetime();
    test_spawner();
    cardinal::log::infof("kittest", "%d checks, %d failed", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
