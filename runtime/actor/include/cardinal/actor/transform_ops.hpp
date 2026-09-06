#pragma once
// =============================================================================
// Cardinal — Transform rotation ops (quaternion-backed).
//
// TransformComponent stores Euler XYZ radians (the editor-friendly form).
// Gameplay code wants to ROTATE things — spin about an axis, face a target,
// orbit a point, turn towards something at a limited rate — and doing that
// by adding to Euler angles is wrong for anything but a single fixed axis
// (gimbal coupling, no composition, no shortest-arc). These ops go through
// a quaternion every time and write the result back as Euler, so the
// inspector keeps showing plain degrees while gameplay composes correctly.
//
// Conventions (match the camera + PlayerController):
//   local forward = -Z, right = +X, up = +Y; angles in RADIANS.
//   rotation(t) = quat_from_euler_xyz(t.rotation_euler).
// =============================================================================

#include <cardinal/actor/component.hpp>
#include <cardinal/core/math/math.hpp>

namespace cardinal::actor::xf {

using Quat = cardinal::core::Quat;
using Vec3 = cardinal::scene::Vec3;

// Orientation as a quaternion / write one back (normalised).
Quat rotation(const TransformComponent& t) noexcept;
void set_rotation(TransformComponent& t, const Quat& q) noexcept;

// Local axes in world space.
Vec3 forward(const TransformComponent& t) noexcept;   // -Z rotated
Vec3 right  (const TransformComponent& t) noexcept;   // +X rotated
Vec3 up     (const TransformComponent& t) noexcept;   // +Y rotated

// Rotate about an axis expressed in LOCAL space (q * r) or WORLD space
// (r * q). A zero-length axis or non-finite angle is a no-op.
void rotate_local(TransformComponent& t, const Vec3& axis, float rad) noexcept;
void rotate_world(TransformComponent& t, const Vec3& axis, float rad) noexcept;

// Face `target` (forward = towards it). Degenerate direction (target at the
// position) leaves the rotation unchanged. `up_hint` resolves roll.
void look_at(TransformComponent& t, const Vec3& target,
             const Vec3& up_hint = Vec3{0.0f, 1.0f, 0.0f}) noexcept;
// Yaw-only variant: turns about +Y only (the pitch to the target is ignored).
void look_at_yaw(TransformComponent& t, const Vec3& target) noexcept;

// Turn towards `target` by at most `max_rad` this call (shortest arc,
// slerp-limited). Returns the remaining angle to the goal after the turn.
float turn_towards(TransformComponent& t, const Vec3& target, float max_rad,
                   bool yaw_only = false,
                   const Vec3& up_hint = Vec3{0.0f, 1.0f, 0.0f}) noexcept;

// Revolve the transform about `center` on `axis` by `rad`: the position
// moves on the circle AND the orientation turns with it (rigid orbit).
void orbit(TransformComponent& t, const Vec3& center, const Vec3& axis, float rad) noexcept;

// Angle (radians, [0, π]) between the transform's orientation and `q`.
float angle_to(const TransformComponent& t, const Quat& q) noexcept;

}  // namespace cardinal::actor::xf
