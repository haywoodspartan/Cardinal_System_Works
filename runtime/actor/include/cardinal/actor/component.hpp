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

#include <cardinal/core/types.hpp>
#include <cardinal/scene/math.hpp>

#include <memory>
#include <string>

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
};

// Optional — points at a scene::Mesh asset that the renderer should draw.
// We don't own the mesh; the scene::AssetCatalog does.
struct MeshComponent : Component {
    std::string asset_id;             // catalog id, resolved at draw time
    cardinal::scene::Vec3 tint{1, 1, 1};
    bool visible{true};
    const char* type_name() const noexcept override { return "Mesh"; }
};

// Camera component — when attached, the actor can be selected as the
// active simulation camera.
struct CameraComponent : Component {
    float fov_y_rad{60.0f * 0.0174532925f};
    float z_near{0.05f};
    float z_far{500.0f};
    bool  active{false};
    const char* type_name() const noexcept override { return "Camera"; }
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
};

// Audio emitter — speaks to cardinal::audio. Loop, volume, pitch, channel.
struct AudioEmitterComponent : Component {
    std::string cue_id;               // looked up in audio::Engine
    float       volume{1.0f};
    float       pitch{1.0f};
    bool        loop{false};
    bool        is_3d{true};
    bool        play_on_spawn{false};
    u32         channel{0};           // 0=Master, 1=Music, 2=SFX, 3=Voice, 4=UI
    bool        playing{false};       // runtime
    u64         instance_id{0};       // assigned by audio::Engine
    const char* type_name() const noexcept override { return "AudioEmitter"; }
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
};

// Tag set — string tags + bit flags. Used for queries (find_by_tag) and
// filtering in the inspector / sequencer.
struct TagComponent : Component {
    std::vector<std::string> tags;
    u32                      flags{0};
    bool has(const std::string& t) const noexcept;
    void add(std::string t);
    void remove(const std::string& t);
    const char* type_name() const noexcept override { return "Tag"; }
};

// Script component — registers a name to call into cardinal::script /
// cardinal::cppscript. We don't run scripts here; we just store the binding
// so a higher-level system can dispatch.
struct ScriptComponent : Component {
    std::string entry_name;           // e.g. "onSpawn" / "onTick"
    std::string source_path;          // .lua or .cpp
    bool        enabled{true};
    const char* type_name() const noexcept override { return "Script"; }
};

}  // namespace cardinal::actor
