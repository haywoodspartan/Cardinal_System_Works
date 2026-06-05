#include <cardinal/actor/component.hpp>
#include <cardinal/actor/actor.hpp>      // Actor::get_component<> (PlayerController)

#include <cardinal/core/algorithm.hpp>   // cardinal::find/remove/clamp/min/max
#include <cardinal/core/cmath.hpp>       // cardinal::sin/cos/atan2/asin/sqrt
#include <cardinal/core/utility.hpp>     // cardinal::move
#include <cardinal/core/cstdio.hpp>      // cardinal::snprintf
#include <cardinal/core/cstdlib.hpp>     // cardinal::strtof / strtol

namespace cardinal::actor {

// ---------------------------------------------------------------------------
// Serialization helpers — emit one "  key = value\n" line per field, parse
// values back. Floats are space-separated for vec3 (simpler + locale-robust
// to tokenize than "(x, y, z)"). Kept file-local; the line shape matches the
// rest of the cardinal text save format.
// ---------------------------------------------------------------------------
namespace {

void emit_f(cardinal::string& o, const char* k, float v) {
    char b[80]; cardinal::snprintf(b, sizeof(b), "  %s = %.6g\n", k, static_cast<double>(v));
    o += b;
}
void emit_i(cardinal::string& o, const char* k, int v) {
    char b[80]; cardinal::snprintf(b, sizeof(b), "  %s = %d\n", k, v);
    o += b;
}
void emit_u(cardinal::string& o, const char* k, u32 v) {
    char b[80]; cardinal::snprintf(b, sizeof(b), "  %s = %u\n", k, v);
    o += b;
}
void emit_b(cardinal::string& o, const char* k, bool v) {
    o += "  "; o += k; o += (v ? " = 1\n" : " = 0\n");
}
void emit_s(cardinal::string& o, const char* k, const cardinal::string& v) {
    o += "  "; o += k; o += " = "; o += v; o += "\n";
}
void emit_v3(cardinal::string& o, const char* k, const cardinal::scene::Vec3& v) {
    char b[112];
    cardinal::snprintf(b, sizeof(b), "  %s = %.6g %.6g %.6g\n", k,
                       static_cast<double>(v.x), static_cast<double>(v.y),
                       static_cast<double>(v.z));
    o += b;
}

float parse_f(const cardinal::string& s) {
    return cardinal::strtof(s.c_str(), nullptr);
}
int parse_i(const cardinal::string& s) {
    return static_cast<int>(cardinal::strtol(s.c_str(), nullptr, 10));
}
bool parse_b(const cardinal::string& s) {
    return !s.empty() && (s[0] == '1' || s[0] == 't' || s[0] == 'T');
}
void parse_v3(const cardinal::string& s, cardinal::scene::Vec3& out) {
    const char* p = s.c_str();
    char* end = nullptr;
    out.x = cardinal::strtof(p, &end); p = end;
    out.y = cardinal::strtof(p, &end); p = end;
    out.z = cardinal::strtof(p, &end);
}

}  // namespace

cardinal::scene::Mat4 TransformComponent::matrix() const {
    using namespace cardinal::scene;
    return Mat4::translation(translation)
         * Mat4::rotation_xyz(rotation_euler)
         * Mat4::scaling(scale);
}

bool TagComponent::has(const cardinal::string& t) const noexcept {
    return cardinal::find(tags.begin(), tags.end(), t) != tags.end();
}

void TagComponent::add(cardinal::string t) {
    if (!has(t)) tags.push_back(cardinal::move(t));
}

void TagComponent::remove(const cardinal::string& t) {
    tags.erase(cardinal::remove(tags.begin(), tags.end(), t), tags.end());
}

// ---------------------------------------------------------------------------
// PlayerControllerComponent
// ---------------------------------------------------------------------------
void PlayerControllerComponent::sync_look_from_forward(
    const cardinal::scene::Vec3& fwd) noexcept
{
    // Same convention as scene::FlyCamera::sync_from so possessing the
    // player from the editor camera doesn't snap the view.
    const float fy = cardinal::clamp(fwd.y, -1.0f, 1.0f);
    pitch_ = cardinal::asin(fy);
    yaw_   = cardinal::atan2(fwd.x, -fwd.z);   // +Z-back convention
}

void PlayerControllerComponent::tick(float dt, const PlayerInput& in) noexcept {
    if (owner_ == nullptr) return;
    auto* tr = owner_->get_component<TransformComponent>();
    if (tr == nullptr) return;
    // Non-finite dt (NaN / ±Inf): the original `if (dt < 0.0f)` was
    // NaN-blind (NaN < 0 is false) AND would let +Inf through. Result:
    // `disp = wish * (inv * spd * NaN)` is a NaN/±Inf vector and
    // `tr->translation += disp` permanently teleports the player out
    // of the world; `vy_ += gravity * NaN` and `tr->translation.y +=
    // vy_ * NaN` poison Y the same way. cardinal::isfinite covers all
    // three non-finite values; preserve the existing negative-clamp
    // intent alongside.
    if (!cardinal::isfinite(dt) || dt < 0.0f) dt = 0.0f;

    using cardinal::scene::Vec3;

    // Direct-field tunables are public on PlayerControllerComponent
    // with no setters (same desc-direct-field pattern as scene::
    // FlyCamera tunables d8f7ce1). Sanitize each at use site with the
    // struct's compile-time default as the safe fallback; preserve the
    // user's stored value so the editor inspector sees the bad input
    // rather than silent correction.
    //
    // look_sensitivity is the NASTIEST: yaw_/pitch_ are PERSISTENT
    // state, once poisoned NaN they stay NaN forever even if the
    // tunable is later corrected (cos/sin of NaN stays NaN).
    const float ls = cardinal::isfinite(look_sensitivity)
                         ? look_sensitivity : 0.0035f;
    // ---- Look: mouse delta -> yaw/pitch (FlyCamera convention) ----------
    if (in.accept_input && in.look) {
        yaw_   += in.mouse_dx * ls;
        pitch_ -= in.mouse_dy * ls;
        pitch_  = cardinal::clamp(pitch_, min_pitch_rad, max_pitch_rad);
    }

    const float cy = cardinal::cos(yaw_), sy = cardinal::sin(yaw_);
    const float cp = cardinal::cos(pitch_), sp = cardinal::sin(pitch_);

    // Full look vector (used for the camera) and the ground-projected
    // forward/right used for FPS-style movement (pitch ignored so looking
    // down doesn't slow you).
    const Vec3 look_fwd { sy * cp, sp, -cy * cp };
    const Vec3 fwd_flat { sy,      0.0f, -cy };
    const Vec3 right    { cy,      0.0f,  sy };

    // Movement tunables — sanitize-at-use with each field's compile-
    // time default as fallback (matches component.hpp:148/149/152/153).
    // NaN move_speed / sprint_multiplier → `disp = wish * (inv * NaN
    // * dt)` poisons tr->translation.x/z. NaN gravity → `vy_ += NaN *
    // dt` poisons vy_ PERSISTENTLY (vy_ stays NaN every later frame
    // until set_velocity / land), then `translation.y += vy_ * dt`
    // teleports Y forever. NaN jump_speed → `vy_ = NaN` on the jump
    // frame; same vy_ persistent poison.
    const float ms_safe = cardinal::isfinite(move_speed)
                              ? move_speed : 6.0f;
    const float sm_safe = cardinal::isfinite(sprint_multiplier)
                              ? sprint_multiplier : 2.0f;
    const float gr_safe = cardinal::isfinite(gravity)
                              ? gravity : -19.62f;
    const float js_safe = cardinal::isfinite(jump_speed)
                              ? jump_speed : 7.0f;

    // ---- Horizontal move: WASD-relative, normalised, sprint-scaled ------
    Vec3 disp{0, 0, 0};
    if (in.accept_input) {
        Vec3 wish = fwd_flat * in.move_z + right * in.move_x;
        const float l2 = wish.x*wish.x + wish.y*wish.y + wish.z*wish.z;
        if (l2 > 1e-6f) {
            const float inv = 1.0f / cardinal::sqrt(l2);
            const float spd = ms_safe * (in.sprint ? sm_safe : 1.0f);
            disp = wish * (inv * spd * dt);
        }
    }
    tr->translation.x += disp.x;
    tr->translation.z += disp.z;

    // ---- Vertical: fly-mode is free; otherwise gravity + jump + floor ---
    if (fly_mode) {
        vy_ = 0.0f;
        if (in.accept_input && in.jump)   tr->translation.y += ms_safe * dt;
        if (in.accept_input && in.sprint) tr->translation.y -= ms_safe * dt;
        grounded_ = false;
    } else {
        if (in.accept_input && in.jump && grounded_) {
            vy_ = js_safe;
            grounded_ = false;
        }
        vy_ += gr_safe * dt;
        tr->translation.y += vy_ * dt;
        if (tr->translation.y <= ground_y) {
            tr->translation.y = ground_y;
            vy_ = 0.0f;
            grounded_ = true;
        }
    }

    // Face the way we look (yaw only — body doesn't pitch).
    tr->rotation_euler.y = yaw_;

    // ---- Publish the first-person camera pose ---------------------------
    cam_eye_ = { tr->translation.x,
                 tr->translation.y + eye_height,
                 tr->translation.z };
    cam_target_ = { cam_eye_.x + look_fwd.x,
                    cam_eye_.y + look_fwd.y,
                    cam_eye_.z + look_fwd.z };
}

// ===========================================================================
// Component serialize_fields / deserialize_field — authored fields only
// (runtime state resets, matching clone()).
// ===========================================================================

// ---- Transform ----
void TransformComponent::serialize_fields(cardinal::string& out) const {
    emit_v3(out, "translation", translation);
    emit_v3(out, "rotation",    rotation_euler);
    emit_v3(out, "scale",       scale);
}
void TransformComponent::deserialize_field(const cardinal::string& k, const cardinal::string& v) {
    if      (k == "translation") parse_v3(v, translation);
    else if (k == "rotation")    parse_v3(v, rotation_euler);
    else if (k == "scale")       parse_v3(v, scale);
}

// ---- Mesh ----
void MeshComponent::serialize_fields(cardinal::string& out) const {
    emit_s (out, "asset_id", asset_id);
    emit_v3(out, "tint",     tint);
    emit_b (out, "visible",  visible);
}
void MeshComponent::deserialize_field(const cardinal::string& k, const cardinal::string& v) {
    if      (k == "asset_id") asset_id = v;
    else if (k == "tint")     parse_v3(v, tint);
    else if (k == "visible")  visible = parse_b(v);
}

// ---- Camera ----
void CameraComponent::serialize_fields(cardinal::string& out) const {
    emit_f(out, "fov_y_rad", fov_y_rad);
    emit_f(out, "z_near",    z_near);
    emit_f(out, "z_far",     z_far);
    emit_b(out, "active",    active);
}
void CameraComponent::deserialize_field(const cardinal::string& k, const cardinal::string& v) {
    if      (k == "fov_y_rad") fov_y_rad = parse_f(v);
    else if (k == "z_near")    z_near    = parse_f(v);
    else if (k == "z_far")     z_far     = parse_f(v);
    else if (k == "active")    active    = parse_b(v);
}

// ---- Light ----
void LightComponent::serialize_fields(cardinal::string& out) const {
    emit_u (out, "kind",           static_cast<u32>(kind));
    emit_v3(out, "color",          color);
    emit_f (out, "intensity",      intensity);
    emit_f (out, "range",          range);
    emit_f (out, "spot_inner_cos", spot_inner_cos);
    emit_f (out, "spot_outer_cos", spot_outer_cos);
}
void LightComponent::deserialize_field(const cardinal::string& k, const cardinal::string& v) {
    if      (k == "kind")           kind = static_cast<LightKind>(parse_i(v));
    else if (k == "color")          parse_v3(v, color);
    else if (k == "intensity")      intensity = parse_f(v);
    else if (k == "range")          range = parse_f(v);
    else if (k == "spot_inner_cos") spot_inner_cos = parse_f(v);
    else if (k == "spot_outer_cos") spot_outer_cos = parse_f(v);
}

// ---- AudioEmitter ----
void AudioEmitterComponent::serialize_fields(cardinal::string& out) const {
    emit_s(out, "cue_id",        cue_id);
    emit_f(out, "volume",        volume);
    emit_f(out, "pitch",         pitch);
    emit_b(out, "loop",          loop);
    emit_b(out, "is_3d",         is_3d);
    emit_b(out, "play_on_spawn", play_on_spawn);
    emit_u(out, "channel",       channel);
}
void AudioEmitterComponent::deserialize_field(const cardinal::string& k, const cardinal::string& v) {
    if      (k == "cue_id")        cue_id = v;
    else if (k == "volume")        volume = parse_f(v);
    else if (k == "pitch")         pitch = parse_f(v);
    else if (k == "loop")          loop = parse_b(v);
    else if (k == "is_3d")         is_3d = parse_b(v);
    else if (k == "play_on_spawn") play_on_spawn = parse_b(v);
    else if (k == "channel")       channel = static_cast<u32>(parse_i(v));
}

// ---- RigidBody ----
void RigidBodyComponent::serialize_fields(cardinal::string& out) const {
    emit_f(out, "mass",           mass);
    emit_f(out, "linear_damping", linear_damping);
    emit_b(out, "use_gravity",    use_gravity);
    emit_b(out, "kinematic",      kinematic);
}
void RigidBodyComponent::deserialize_field(const cardinal::string& k, const cardinal::string& v) {
    if      (k == "mass")           mass = parse_f(v);
    else if (k == "linear_damping") linear_damping = parse_f(v);
    else if (k == "use_gravity")    use_gravity = parse_b(v);
    else if (k == "kinematic")      kinematic = parse_b(v);
}

// ---- Tag ----
// Tags serialize as a count + one "tag_N" line each (tags may contain
// spaces, so we can't pack them on one line with the space-separated rule).
void TagComponent::serialize_fields(cardinal::string& out) const {
    emit_u(out, "flags", flags);
    emit_u(out, "tag_count", static_cast<u32>(tags.size()));
    for (usize i = 0; i < tags.size(); ++i) {
        char key[32];
        cardinal::snprintf(key, sizeof(key), "tag_%zu", i);
        emit_s(out, key, tags[i]);
    }
}
void TagComponent::deserialize_field(const cardinal::string& k, const cardinal::string& v) {
    if (k == "flags") { flags = static_cast<u32>(parse_i(v)); return; }
    if (k == "tag_count") return;   // size is implicit from the tag_N lines
    if (k.size() > 4 && k.compare(0, 4, "tag_") == 0) {
        // tag_N → append (dedup via add()).
        add(v);
    }
}

// ---- Script ----
void ScriptComponent::serialize_fields(cardinal::string& out) const {
    emit_s(out, "entry_name",  entry_name);
    emit_s(out, "source_path", source_path);
    emit_b(out, "enabled",     enabled);
}
void ScriptComponent::deserialize_field(const cardinal::string& k, const cardinal::string& v) {
    if      (k == "entry_name")  entry_name = v;
    else if (k == "source_path") source_path = v;
    else if (k == "enabled")     enabled = parse_b(v);
}

// ---- PrefabLink ----
void PrefabLinkComponent::serialize_fields(cardinal::string& out) const {
    emit_s(out, "prefab_name", prefab_name);
}
void PrefabLinkComponent::deserialize_field(const cardinal::string& k, const cardinal::string& v) {
    if (k == "prefab_name") prefab_name = v;
}

// ---- PlayerController (authored tunables only) ----
void PlayerControllerComponent::serialize_fields(cardinal::string& out) const {
    emit_f(out, "move_speed",        move_speed);
    emit_f(out, "sprint_multiplier", sprint_multiplier);
    emit_f(out, "look_sensitivity",  look_sensitivity);
    emit_f(out, "eye_height",        eye_height);
    emit_f(out, "gravity",           gravity);
    emit_f(out, "jump_speed",        jump_speed);
    emit_f(out, "ground_y",          ground_y);
    emit_f(out, "min_pitch_rad",     min_pitch_rad);
    emit_f(out, "max_pitch_rad",     max_pitch_rad);
    emit_b(out, "fly_mode",          fly_mode);
}
void PlayerControllerComponent::deserialize_field(const cardinal::string& k, const cardinal::string& v) {
    if      (k == "move_speed")        move_speed = parse_f(v);
    else if (k == "sprint_multiplier") sprint_multiplier = parse_f(v);
    else if (k == "look_sensitivity")  look_sensitivity = parse_f(v);
    else if (k == "eye_height")        eye_height = parse_f(v);
    else if (k == "gravity")           gravity = parse_f(v);
    else if (k == "jump_speed")        jump_speed = parse_f(v);
    else if (k == "ground_y")          ground_y = parse_f(v);
    else if (k == "min_pitch_rad")     min_pitch_rad = parse_f(v);
    else if (k == "max_pitch_rad")     max_pitch_rad = parse_f(v);
    else if (k == "fly_mode")          fly_mode = parse_b(v);
}

// ===========================================================================
// Component factory — type_name() string -> fresh built-in instance.
// ===========================================================================
cardinal::unique_ptr<Component> make_component_by_name(const cardinal::string& type_name) {
    if (type_name == "Transform")        return cardinal::make_unique<TransformComponent>();
    if (type_name == "Mesh")             return cardinal::make_unique<MeshComponent>();
    if (type_name == "Camera")           return cardinal::make_unique<CameraComponent>();
    if (type_name == "Light")            return cardinal::make_unique<LightComponent>();
    if (type_name == "AudioEmitter")     return cardinal::make_unique<AudioEmitterComponent>();
    if (type_name == "RigidBody")        return cardinal::make_unique<RigidBodyComponent>();
    if (type_name == "Tag")              return cardinal::make_unique<TagComponent>();
    if (type_name == "Script")           return cardinal::make_unique<ScriptComponent>();
    if (type_name == "PlayerController") return cardinal::make_unique<PlayerControllerComponent>();
    if (type_name == "PrefabLink")       return cardinal::make_unique<PrefabLinkComponent>();
    // "GameActor" is intentionally NOT handled here — the game module
    // reconstructs game classes via its ClassRegistry.
    return nullptr;
}

}  // namespace cardinal::actor
