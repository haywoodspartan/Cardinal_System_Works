// =============================================================================
// Cardinal — fly camera implementation.
// =============================================================================
#include <cardinal/scene/fly_camera.hpp>

#include <cardinal/core/std/algorithm.hpp>
#include <cardinal/core/std/cmath.hpp>

namespace cardinal::scene {

void FlyCamera::sync_from(const Camera& cam) noexcept {
    const Vec3 fwd = normalize(cam.target - cam.position);
    pitch_ = cardinal::asin(cardinal::max(-1.0f, cardinal::min(1.0f, fwd.y)));
    yaw_   = cardinal::atan2(fwd.x, -fwd.z);   // matches +Z back convention
    initialised_ = true;
}

void FlyCamera::tick(Camera& cam, const FlyInput& in, float dt) noexcept {
    if (!initialised_) sync_from(cam);
    if (!in.accept_input) return;
    // Non-finite dt (NaN / ±Inf): same intent as the original `dt <= 0`
    // clamp, but ordered compares don't catch NaN (NaN <= 0 is false)
    // and +Inf passes the > 0 gate. Unguarded, NaN dt flowed into
    // `cam.position += move * v` (v = speed * dt) and the camera
    // teleported permanently to NaN-land; +Inf would do the same to
    // ±Inf. cardinal::isfinite handles all three; preserve the
    // existing negative-clamp alongside.
    if (!cardinal::isfinite(dt) || dt < 0.0f) dt = 0.0f;

    // ---- Speed: scroll-wheel adjusts a multiplicative factor; clamp.
    if (cardinal::abs(in.scroll) > 0.001f) {
        const float scale = cardinal::pow(scroll_speed_scale, in.scroll);
        speed = cardinal::clamp(speed * scale, speed_min, speed_max);
    }

    // ---- Look: RMB drag updates yaw + pitch.
    //
    // Mouse-right  (dx > 0) → camera turns right  → yaw INCREASES, because
    //                         our forward vector is { sin(yaw)*cos(pitch),
    //                         sin(pitch), -cos(yaw)*cos(pitch) } and a
    //                         positive yaw step moves the X component up.
    // Mouse-down   (dy > 0) → camera looks down  → pitch DECREASES.
    //
    // Subtracting on yaw was the bug behind "RMB rotates the level map" —
    // it inverted the X axis so dragging right made the camera turn left,
    // which the user perceives as the world rotating with the cursor.
    if (in.look) {
        // look_sensitivity is a direct public field (FlyCamera tunable
        // with no setter — same desc-direct-field NaN sink as
        // cine::Sequence::speed 1ec131e). NaN look_sensitivity → `yaw_
        // += in.mouse_dx * NaN` poisons yaw_ permanently → every
        // subsequent cos/sin produces NaN forward/right vectors →
        // `cam.position += move * v` teleports the camera every tick.
        // Same for pitch_. Sanitize at use; preserve the user's stored
        // value so the editor inspector sees the bad input.
        const float ls = cardinal::isfinite(look_sensitivity)
                             ? look_sensitivity : 0.0035f;
        yaw_   += in.mouse_dx * ls;
        pitch_ -= in.mouse_dy * ls;
        pitch_  = cardinal::clamp(pitch_, min_pitch_rad, max_pitch_rad);
    }

    // Forward / right vectors derived from yaw + pitch.
    const float cy = cardinal::cos(yaw_),   sy = cardinal::sin(yaw_);
    const float cp = cardinal::cos(pitch_), sp = cardinal::sin(pitch_);
    const Vec3 forward { sy * cp, sp,  -cy * cp };
    const Vec3 right   { cy,      0.0f, sy };
    const Vec3 world_up{ 0.0f,    1.0f, 0.0f };

    // ---- Move: WASD/QE drives the position. Sprint multiplies.
    Vec3 move{};
    if (in.forward)  move += forward;
    if (in.backward) move -= forward;
    if (in.right)    move += right;
    if (in.left)     move -= right;
    if (in.up)       move += world_up;
    if (in.down)     move -= world_up;
    const float lensq = dot(move, move);
    if (lensq > 1e-6f) {
        move = move * (1.0f / cardinal::sqrt(lensq));
        // speed / sprint_multiplier are direct public fields (same
        // desc-direct-field NaN sink class as look_sensitivity above).
        // NaN in either → `v = NaN * dt` → `cam.position += move * NaN`
        // teleports the camera to NaN-land permanently. Sanitize at use.
        // Defaults match the struct's compile-time initializers
        // (4.0 / 4.0 — see fly_camera.hpp:49,52).
        const float spd = cardinal::isfinite(speed)
                              ? speed : 4.0f;
        const float sm  = cardinal::isfinite(sprint_multiplier)
                              ? sprint_multiplier : 4.0f;
        const float v = spd * (in.sprint ? sm : 1.0f) * dt;
        cam.position += move * v;
    }

    // Update target so look_at() sees the right forward vector.
    cam.target = cam.position + forward;
    cam.up     = world_up;
}

}  // namespace cardinal::scene
