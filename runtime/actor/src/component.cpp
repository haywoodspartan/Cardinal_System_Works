#include <cardinal/actor/component.hpp>
#include <cardinal/actor/actor.hpp>      // Actor::get_component<> (PlayerController)

#include <cardinal/core/algorithm.hpp>   // cardinal::find/remove/clamp/min/max
#include <cardinal/core/cmath.hpp>       // cardinal::sin/cos/atan2/asin/sqrt
#include <cardinal/core/utility.hpp>     // cardinal::move

namespace cardinal::actor {

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

}  // namespace cardinal::actor
