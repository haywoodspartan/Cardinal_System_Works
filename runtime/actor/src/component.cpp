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
    if (dt < 0.0f) dt = 0.0f;

    using cardinal::scene::Vec3;

    // ---- Look: mouse delta -> yaw/pitch (FlyCamera convention) ----------
    if (in.accept_input && in.look) {
        yaw_   += in.mouse_dx * look_sensitivity;
        pitch_ -= in.mouse_dy * look_sensitivity;
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

    // ---- Horizontal move: WASD-relative, normalised, sprint-scaled ------
    Vec3 disp{0, 0, 0};
    if (in.accept_input) {
        Vec3 wish = fwd_flat * in.move_z + right * in.move_x;
        const float l2 = wish.x*wish.x + wish.y*wish.y + wish.z*wish.z;
        if (l2 > 1e-6f) {
            const float inv = 1.0f / cardinal::sqrt(l2);
            const float spd = move_speed *
                              (in.sprint ? sprint_multiplier : 1.0f);
            disp = wish * (inv * spd * dt);
        }
    }
    tr->translation.x += disp.x;
    tr->translation.z += disp.z;

    // ---- Vertical: fly-mode is free; otherwise gravity + jump + floor ---
    if (fly_mode) {
        vy_ = 0.0f;
        if (in.accept_input && in.jump)   tr->translation.y += move_speed * dt;
        if (in.accept_input && in.sprint) tr->translation.y -= move_speed * dt;
        grounded_ = false;
    } else {
        if (in.accept_input && in.jump && grounded_) {
            vy_ = jump_speed;
            grounded_ = false;
        }
        vy_ += gravity * dt;
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
