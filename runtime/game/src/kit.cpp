// =============================================================================
// Cardinal — Gameplay Kit. See kit.hpp.
// =============================================================================
#include <cardinal/game/kit.hpp>
#include <cardinal/game/game.hpp>
#include <cardinal/actor/actor.hpp>
#include <cardinal/actor/world.hpp>
#include <cardinal/actor/transform_ops.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/std/cmath.hpp>

namespace cardinal::game::kit {

namespace {

namespace xf = cardinal::actor::xf;
using cardinal::actor::TransformComponent;
using cardinal::scene::Vec3;
constexpr float kDeg2Rad = 0.017453292519943295f;

inline TransformComponent* xform(GameActor& ga) noexcept {
    return ga.owner() ? ga.owner()->get_component<TransformComponent>() : nullptr;
}
inline cardinal::actor::World* world_of(GameActor&) noexcept {
    Game* g = Game::active();
    return g ? &g->world() : nullptr;
}
inline float len(const Vec3& v) noexcept {
    return cardinal::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// ---------------------------------------------------------------------------
// RotatorActor
// ---------------------------------------------------------------------------
class RotatorActor final : public GameActor {
public:
    Vec3  axis{0.0f, 1.0f, 0.0f};
    float speed_deg_s{90.0f};
    bool  local_space{true};
    float turned_deg{0.0f};      // runtime: total angle turned since begin_play

    void begin_play() override { turned_deg = 0.0f; }
    void on_tick(float dt) override {
        auto* t = xform(*this);
        if (!t) return;
        const float rad = speed_deg_s * kDeg2Rad * dt;
        if (local_space) xf::rotate_local(*t, axis, rad);
        else             xf::rotate_world(*t, axis, rad);
        turned_deg += speed_deg_s * dt;
    }
};

// ---------------------------------------------------------------------------
// OrbiterActor
// ---------------------------------------------------------------------------
class OrbiterActor final : public GameActor {
public:
    Vec3  center{0.0f, 0.0f, 0.0f};
    float radius{3.0f};
    float speed_deg_s{45.0f};
    Vec3  axis{0.0f, 1.0f, 0.0f};
    float start_deg{0.0f};
    bool  face_center{true};

    void begin_play() override { angle_deg_ = start_deg; place_(); }
    void on_tick(float dt) override {
        angle_deg_ += speed_deg_s * dt;
        place_();
    }
    float angle_deg() const noexcept { return angle_deg_; }

private:
    void place_() {
        auto* t = xform(*this);
        if (!t) return;
        Vec3 ax = axis;
        if (len(ax) < 1.0e-8f) ax = Vec3{0.0f, 1.0f, 0.0f};
        ax = cardinal::core::normalize(ax);
        // A radial direction perpendicular to the axis (stable choice).
        const Vec3 helper = (cardinal::abs(ax.y) < 0.99f) ? Vec3{0.0f, 1.0f, 0.0f}
                                                          : Vec3{1.0f, 0.0f, 0.0f};
        const Vec3 r0 = cardinal::core::normalize(cardinal::core::cross(ax, helper));
        const auto q  = cardinal::core::axis_angle(ax, angle_deg_ * kDeg2Rad);
        t->translation = center + cardinal::core::rotate(q, r0) * radius;
        if (face_center) xf::look_at(*t, center, ax);
    }
    float angle_deg_{0.0f};
};

// ---------------------------------------------------------------------------
// LookAtActor
// ---------------------------------------------------------------------------
class LookAtActor final : public GameActor {
public:
    cardinal::string target_tag{"player"};
    float turn_rate_deg_s{0.0f};     // 0 = snap
    bool  yaw_only{false};

    void on_tick(float dt) override {
        auto* t = xform(*this);
        auto* w = world_of(*this);
        if (!t || !w) return;
        const Vec3 self = t->translation;
        // Nearest alive actor with the tag (excluding ourselves).
        cardinal::actor::Actor* best = nullptr;
        float best_d = 0.0f;
        for (auto* a : w->find_all_by_tag(target_tag)) {
            if (!a || !a->alive() || a == owner()) continue;
            auto* at = a->get_component<TransformComponent>();
            if (!at) continue;
            const float d = len(at->translation - self);
            if (!best || d < best_d) { best = a; best_d = d; }
        }
        if (!best) return;
        const Vec3 target = best->get_component<TransformComponent>()->translation;
        if (turn_rate_deg_s <= 0.0f) {
            if (yaw_only) xf::look_at_yaw(*t, target); else xf::look_at(*t, target);
        } else {
            xf::turn_towards(*t, target, turn_rate_deg_s * kDeg2Rad * dt, yaw_only);
        }
    }
};

// ---------------------------------------------------------------------------
// WaypointMoverActor
// ---------------------------------------------------------------------------
class WaypointMoverActor final : public GameActor {
public:
    Vec3  waypoint_a{0.0f, 0.0f, 0.0f};
    Vec3  waypoint_b{5.0f, 0.0f, 0.0f};
    Vec3  waypoint_c{5.0f, 0.0f, 5.0f};
    Vec3  waypoint_d{0.0f, 0.0f, 5.0f};
    int   waypoint_count{2};      // 2..4
    float speed{2.0f};            // units per second
    int   loop_mode{0};           // 0 loop, 1 ping-pong, 2 once
    bool  face_direction{true};

    void begin_play() override { index_ = 0; dir_ = 1; done_ = false; }
    void on_tick(float dt) override {
        auto* t = xform(*this);
        if (!t || done_) return;
        const int n = waypoint_count < 2 ? 2 : (waypoint_count > 4 ? 4 : waypoint_count);
        float budget = speed * dt;
        int guard = 0;
        while (budget > 0.0f && !done_ && guard++ < 8) {
            const Vec3 target = point_(index_);
            const Vec3 to = target - t->translation;
            const float d = len(to);
            if (d <= budget || d < 1.0e-5f) {
                t->translation = target;
                budget -= d;
                advance_(n);
            } else {
                const Vec3 step = to * (budget / d);
                if (face_direction) xf::look_at(*t, t->translation + to);
                t->translation += step;
                budget = 0.0f;
            }
        }
    }
    int  index() const noexcept { return index_; }
    bool done()  const noexcept { return done_; }

private:
    Vec3 point_(int i) const noexcept {
        switch (i) { case 0: return waypoint_a; case 1: return waypoint_b;
                     case 2: return waypoint_c; default: return waypoint_d; }
    }
    void advance_(int n) noexcept {
        switch (loop_mode) {
            case 1:   // ping-pong
                if (index_ + dir_ >= n || index_ + dir_ < 0) dir_ = -dir_;
                index_ += dir_;
                break;
            case 2:   // once
                if (index_ + 1 >= n) done_ = true; else ++index_;
                break;
            default:  // loop
                index_ = (index_ + 1) % n;
                break;
        }
    }
    int  index_{0};
    int  dir_{1};
    bool done_{false};
};

// ---------------------------------------------------------------------------
// LifetimeActor
// ---------------------------------------------------------------------------
class LifetimeActor final : public GameActor {
public:
    float seconds{5.0f};

    void begin_play() override { elapsed_ = 0.0f; }
    void on_tick(float dt) override {
        elapsed_ += dt;
        if (elapsed_ < seconds) return;
        auto* w = world_of(*this);
        if (w && owner() && owner()->alive()) w->destroy(owner()->id());
    }
    float elapsed() const noexcept { return elapsed_; }

private:
    float elapsed_{0.0f};
};

// ---------------------------------------------------------------------------
// SpawnerActor
// ---------------------------------------------------------------------------
class SpawnerActor final : public GameActor {
public:
    cardinal::string prefab_name;      // World prefab (Prefabs panel captures one)
    float interval_s{1.0f};
    int   max_alive{5};
    Vec3  offset{0.0f, 0.0f, 0.0f};
    bool  spawn_on_begin{true};
    int   spawned_total{0};            // runtime stat

    void begin_play() override {
        spawned_.clear();
        spawned_total = 0;
        timer_ = spawn_on_begin ? 0.0f : interval_s;
    }
    void on_tick(float dt) override {
        auto* w = world_of(*this);
        auto* t = xform(*this);
        if (!w || !t) return;
        // Prune dead / swept instances.
        usize keep = 0;
        for (usize i = 0; i < spawned_.size(); ++i) {
            auto* a = w->find(spawned_[i]);
            if (a && a->alive()) spawned_[keep++] = spawned_[i];
        }
        spawned_.resize(keep);
        timer_ -= dt;
        if (timer_ > 0.0f) return;
        timer_ = (interval_s > 1.0e-3f) ? interval_s : 1.0e-3f;
        if (max_alive > 0 && static_cast<int>(spawned_.size()) >= max_alive) return;
        if (prefab_name.empty() || !w->has_prefab(prefab_name)) return;
        auto* a = w->spawn_prefab_at(prefab_name, t->translation + offset);
        if (!a) return;
        spawned_.push_back(a->id());
        ++spawned_total;
    }
    usize alive_spawned() const noexcept { return spawned_.size(); }

private:
    cardinal::vector<cardinal::actor::ActorId> spawned_;
    float timer_{0.0f};
};

const char* const kKitClasses[] = {
    "RotatorActor", "OrbiterActor", "LookAtActor",
    "WaypointMoverActor", "LifetimeActor", "SpawnerActor",
};

}  // namespace

// ---- registrations ---------------------------------------------------------
CARDINAL_REGISTER_GAME_CLASS(RotatorActor, "Kit/Motion",
    PROP_VEC3 (axis,                       "Rotation axis")
    PROP_FLOAT(speed_deg_s, -720.0f, 720.0f, "Degrees per second (negative = reverse)")
    PROP_BOOL (local_space,                "Axis in local space (else world)")
    PROP_FLOAT(turned_deg, -1.0e9f, 1.0e9f, "Runtime: degrees turned since Play"))

CARDINAL_REGISTER_GAME_CLASS(OrbiterActor, "Kit/Motion",
    PROP_VEC3 (center,                     "World point to revolve around")
    PROP_FLOAT(radius, 0.0f, 1000.0f,      "Orbit radius")
    PROP_FLOAT(speed_deg_s, -720.0f, 720.0f, "Angular speed, degrees per second")
    PROP_VEC3 (axis,                       "Orbit plane normal")
    PROP_FLOAT(start_deg, -360.0f, 360.0f, "Starting angle")
    PROP_BOOL (face_center,                "Keep facing the centre"))

CARDINAL_REGISTER_GAME_CLASS(LookAtActor, "Kit/Motion",
    PROP_STRING(target_tag,                "Tag of the actor to face (nearest wins)")
    PROP_FLOAT (turn_rate_deg_s, 0.0f, 3600.0f, "Max turn rate; 0 = snap")
    PROP_BOOL  (yaw_only,                  "Turn about the vertical axis only"))

CARDINAL_REGISTER_GAME_CLASS(WaypointMoverActor, "Kit/Motion",
    PROP_VEC3 (waypoint_a,                 "Waypoint 1")
    PROP_VEC3 (waypoint_b,                 "Waypoint 2")
    PROP_VEC3 (waypoint_c,                 "Waypoint 3")
    PROP_VEC3 (waypoint_d,                 "Waypoint 4")
    PROP_INT  (waypoint_count, 2, 4,       "How many waypoints are used")
    PROP_FLOAT(speed, 0.0f, 1000.0f,       "Units per second")
    PROP_INT  (loop_mode, 0, 2,            "0 loop, 1 ping-pong, 2 once")
    PROP_BOOL (face_direction,             "Face the direction of travel"))

CARDINAL_REGISTER_GAME_CLASS(LifetimeActor, "Kit/Lifecycle",
    PROP_FLOAT(seconds, 0.0f, 3600.0f,     "Destroy the owner after this many seconds of play"))

CARDINAL_REGISTER_GAME_CLASS(SpawnerActor, "Kit/Lifecycle",
    PROP_STRING(prefab_name,               "World prefab to stamp (see the Prefabs panel)")
    PROP_FLOAT (interval_s, 0.01f, 3600.0f, "Seconds between spawns")
    PROP_INT   (max_alive, 0, 10000,       "Cap on live instances (0 = unlimited)")
    PROP_VEC3  (offset,                    "Spawn offset from the owner")
    PROP_BOOL  (spawn_on_begin,            "Spawn immediately on Play")
    PROP_INT   (spawned_total, 0, 1000000, "Runtime: instances spawned since Play"))

u32 register_gameplay_kit() {
    u32 n = 0;
    auto& reg = ClassRegistry::instance();
    for (const char* name : kKitClasses)
        if (reg.find(name) != nullptr) ++n;
    static bool logged = false;
    if (!logged) {
        logged = true;
        cardinal::log::infof("game/kit", "gameplay kit online — %u classes (Kit/Motion, Kit/Lifecycle)", n);
    }
    return n;
}

cardinal::vector<cardinal::string> kit_class_names() {
    cardinal::vector<cardinal::string> out;
    for (const char* name : kKitClasses) out.push_back(name);
    return out;
}

}  // namespace cardinal::game::kit
