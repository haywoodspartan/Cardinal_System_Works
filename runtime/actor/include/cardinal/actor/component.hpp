#pragma once

// =============================================================================
// Cardinal — Component primitives for the Actor system.
//
// Components are POD-ish state wrappers that an Actor owns by composition.
// We deliberately keep them simple (no virtual dispatch on hot paths) — the
// SimWorld iterates known component vectors per Actor.
//
// Component lifecycle:
//   - on_attach()  : Actor was just constructed and component added
//   - on_tick(dt)  : called each tick by SimWorld
//   - on_detach()  : Actor is being destroyed
//
// Defaults are no-ops; override what you need.
// =============================================================================

#include <cardinal/core/types.hpp>        // memory/string
#include <cardinal/core/containers.hpp>   // cardinal::vector
#include <cardinal/scene/math.hpp>

namespace cardinal::actor {

class Actor;

class Component {
public:
    Component() = default;
    virtual ~Component() = default;
    Component(const Component&) = delete;
    Component& operator=(const Component&) = delete;

    virtual const char* type_name() const noexcept = 0;
    virtual void on_attach(Actor& /*owner*/) {}
    virtual void on_tick  (Actor& /*owner*/, float /*dt*/) {}
    virtual void on_detach(Actor& /*owner*/) {}

    // Deep-copy this component for the Prefab system. The default returns
    // nullptr: a component that doesn't override clone() can't be captured
    // into a prefab (World::create_prefab logs + skips it). Every built-in
    // component below overrides this. Runtime-only state (playback handles,
    // cached look angles, grounded flags) is intentionally NOT copied — a
    // freshly stamped instance starts from a clean runtime baseline.
    virtual cardinal::unique_ptr<Component> clone() const { return nullptr; }
};

// ---------------------------------------------------------------------------
// Built-in components.
// ---------------------------------------------------------------------------

// Lives on every Actor by default — wraps cardinal::scene::Transform.
struct TransformComponent : Component {
    cardinal::scene::Vec3 translation{0, 0, 0};
    cardinal::scene::Vec3 rotation_euler{0, 0, 0};   // radians
    cardinal::scene::Vec3 scale{1, 1, 1};

    const char* type_name() const noexcept override { return "Transform"; }
    cardinal::scene::Mat4 matrix() const;

    cardinal::unique_ptr<Component> clone() const override {
        auto c = cardinal::make_unique<TransformComponent>();
        c->translation = translation; c->rotation_euler = rotation_euler;
        c->scale = scale;
        return c;
    }
};

// Optional — points at a scene::Mesh asset that the renderer should draw.
// We don't own the mesh; the scene::AssetCatalog does.
struct MeshComponent : Component {
    cardinal::string asset_id;             // catalog id, resolved at draw time
    cardinal::scene::Vec3 tint{1, 1, 1};
    bool visible{true};
    const char* type_name() const noexcept override { return "Mesh"; }

    cardinal::unique_ptr<Component> clone() const override {
        auto c = cardinal::make_unique<MeshComponent>();
        c->asset_id = asset_id; c->tint = tint; c->visible = visible;
        return c;
    }
};

// Camera component — when attached, the actor can be selected as the
// active simulation camera.
struct CameraComponent : Component {
    float fov_y_rad{60.0f * 0.0174532925f};
    float z_near{0.05f};
    float z_far{500.0f};
    bool  active{false};
    const char* type_name() const noexcept override { return "Camera"; }

    cardinal::unique_ptr<Component> clone() const override {
        auto c = cardinal::make_unique<CameraComponent>();
        c->fov_y_rad = fov_y_rad; c->z_near = z_near; c->z_far = z_far;
        c->active = active;
        return c;
    }
};

enum class LightKind : u32 { Directional = 0, Point = 1, Spot = 2 };
struct LightComponent : Component {
    LightKind kind{LightKind::Directional};
    cardinal::scene::Vec3 color{1, 1, 1};
    float intensity{1.0f};
    float range{20.0f};               // point/spot
    float spot_inner_cos{0.95f};
    float spot_outer_cos{0.85f};
    const char* type_name() const noexcept override { return "Light"; }

    cardinal::unique_ptr<Component> clone() const override {
        auto c = cardinal::make_unique<LightComponent>();
        c->kind = kind; c->color = color; c->intensity = intensity;
        c->range = range; c->spot_inner_cos = spot_inner_cos;
        c->spot_outer_cos = spot_outer_cos;
        return c;
    }
};

// Audio emitter — speaks to cardinal::audio. Loop, volume, pitch, channel.
struct AudioEmitterComponent : Component {
    cardinal::string cue_id;               // looked up in audio::Engine
    float       volume{1.0f};
    float       pitch{1.0f};
    bool        loop{false};
    bool        is_3d{true};
    bool        play_on_spawn{false};
    u32         channel{0};           // 0=Master, 1=Music, 2=SFX, 3=Voice, 4=UI
    bool        playing{false};       // runtime
    u64         instance_id{0};       // assigned by audio::Engine
    const char* type_name() const noexcept override { return "AudioEmitter"; }

    cardinal::unique_ptr<Component> clone() const override {
        auto c = cardinal::make_unique<AudioEmitterComponent>();
        c->cue_id = cue_id; c->volume = volume; c->pitch = pitch;
        c->loop = loop; c->is_3d = is_3d; c->play_on_spawn = play_on_spawn;
        c->channel = channel;
        // playing + instance_id are runtime handles — a fresh instance is
        // silent until something triggers it. Don't copy them.
        return c;
    }
};

// Rigid body — velocity / mass / drag. SimWorld's PrePhysics/Physics groups
// integrate these. We don't yet plug into a real solver — the basic euler
// integrator here covers projectiles + simple drops, which is enough for
// the sim sample to feel real.
struct RigidBodyComponent : Component {
    cardinal::scene::Vec3 velocity{0, 0, 0};
    cardinal::scene::Vec3 acceleration{0, 0, 0};
    float                 mass{1.0f};
    float                 linear_damping{0.05f};
    bool                  use_gravity{true};
    bool                  kinematic{false};   // when true, sim doesn't integrate
    const char* type_name() const noexcept override { return "RigidBody"; }

    cardinal::unique_ptr<Component> clone() const override {
        auto c = cardinal::make_unique<RigidBodyComponent>();
        // velocity + acceleration are runtime; a stamped instance starts
        // at rest. Copy only the authored body parameters.
        c->mass = mass; c->linear_damping = linear_damping;
        c->use_gravity = use_gravity; c->kinematic = kinematic;
        return c;
    }
};

// Tag set — string tags + bit flags. Used for queries (find_by_tag) and
// filtering in the inspector / sequencer.
struct TagComponent : Component {
    cardinal::vector<cardinal::string> tags;
    u32                      flags{0};
    bool has(const cardinal::string& t) const noexcept;
    void add(cardinal::string t);
    void remove(const cardinal::string& t);
    const char* type_name() const noexcept override { return "Tag"; }

    cardinal::unique_ptr<Component> clone() const override {
        auto c = cardinal::make_unique<TagComponent>();
        c->tags = tags; c->flags = flags;
        return c;
    }
};

// ---------------------------------------------------------------------------
// PlayerControllerComponent — first-person possession of the owning Actor.
//
// Reusable, engine-level player control. Deliberately decoupled from the
// input system AND the scene camera (mirrors scene::FlyCamera / FlyInput):
// the host fills a PlayerInput each frame from whatever source it has
// (cardinal::input::Manager, a net-replicated command, an AI brain) and
// calls tick(dt, in). The controller integrates WASD-relative motion +
// gravity/jump into the owner's TransformComponent and caches yaw/pitch.
// camera_eye()/camera_target() expose a first-person pose the host can
// copy onto its scene::Camera while a player is possessed; nothing here
// links actor -> input or actor -> camera ownership.
// ---------------------------------------------------------------------------
struct PlayerInput {
    bool  accept_input{true};   // gate (false = ignore, e.g. typing in UI)
    float move_x{0.0f};         // strafe  [-1,+1]  (A/D, left stick X)
    float move_z{0.0f};         // forward [-1,+1]  (W/S, left stick Y)
    bool  jump{false};
    bool  sprint{false};
    bool  look{false};          // apply mouse delta to yaw/pitch this frame
    float mouse_dx{0.0f};       // px since last frame
    float mouse_dy{0.0f};
};

struct PlayerControllerComponent : Component {
    // Tunables — editor-friendly ranges.
    float move_speed       {6.0f};    // world units / second
    float sprint_multiplier{2.0f};
    float look_sensitivity {0.0035f}; // radians / pixel
    float eye_height       {1.7f};    // camera offset above the actor origin
    float gravity          {-19.62f}; // 2g — snappy platformer feel
    float jump_speed       {7.0f};    // initial up velocity on jump
    float ground_y         {0.0f};    // simple floor plane (no physics dep)
    float min_pitch_rad    {-1.553f}; // ~-89°
    float max_pitch_rad    { 1.553f}; // ~+89°
    bool  fly_mode         {false};   // true = no gravity, free vertical

    const char* type_name() const noexcept override { return "PlayerController"; }
    void on_attach(Actor& a) override { owner_ = &a; }
    void on_detach(Actor&)   override { owner_ = nullptr; }

    cardinal::unique_ptr<Component> clone() const override {
        auto c = cardinal::make_unique<PlayerControllerComponent>();
        // Authored tunables only. yaw_/pitch_/vy_/grounded_/cam_* are
        // runtime — a fresh instance re-derives them from its own pose.
        c->move_speed = move_speed; c->sprint_multiplier = sprint_multiplier;
        c->look_sensitivity = look_sensitivity; c->eye_height = eye_height;
        c->gravity = gravity; c->jump_speed = jump_speed;
        c->ground_y = ground_y; c->min_pitch_rad = min_pitch_rad;
        c->max_pitch_rad = max_pitch_rad; c->fly_mode = fly_mode;
        return c;
    }

    // Seed yaw/pitch from a forward vector so possession is seamless from
    // wherever the editor camera was looking.
    void sync_look_from_forward(const cardinal::scene::Vec3& fwd) noexcept;

    // Advance one frame. Writes the owner's TransformComponent
    // (translation + rotation_euler.y = yaw) and refreshes the cached
    // first-person camera pose.
    void tick(float dt, const PlayerInput& in) noexcept;

    float yaw_rad()   const noexcept { return yaw_; }
    float pitch_rad() const noexcept { return pitch_; }
    bool  grounded()  const noexcept { return grounded_; }
    const cardinal::scene::Vec3& camera_eye()    const noexcept { return cam_eye_; }
    const cardinal::scene::Vec3& camera_target() const noexcept { return cam_target_; }

private:
    Actor*                owner_{nullptr};
    float                 yaw_{0.0f};
    float                 pitch_{0.0f};
    float                 vy_{0.0f};        // vertical velocity (gravity/jump)
    bool                  grounded_{true};
    cardinal::scene::Vec3 cam_eye_{0, 0, 0};
    cardinal::scene::Vec3 cam_target_{0, 0, -1};
};

// Script component — registers a name to call into cardinal::script /
// cardinal::cppscript. We don't run scripts here; we just store the binding
// so a higher-level system can dispatch.
struct ScriptComponent : Component {
    cardinal::string entry_name;           // e.g. "onSpawn" / "onTick"
    cardinal::string source_path;          // .lua or .cpp
    bool        enabled{true};
    const char* type_name() const noexcept override { return "Script"; }

    cardinal::unique_ptr<Component> clone() const override {
        auto c = cardinal::make_unique<ScriptComponent>();
        c->entry_name = entry_name; c->source_path = source_path;
        c->enabled = enabled;
        return c;
    }
};

}  // namespace cardinal::actor
