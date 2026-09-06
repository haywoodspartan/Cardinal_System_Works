// =============================================================================
// Cardinal — Transform rotation ops. See transform_ops.hpp.
// =============================================================================
#include <cardinal/actor/transform_ops.hpp>
#include <cardinal/core/std/cmath.hpp>

namespace cardinal::actor::xf {

namespace {
using cardinal::core::axis_angle;
using cardinal::core::rotate;
using cardinal::core::normalize;
using cardinal::core::slerp;
using cardinal::core::look_rotation;
using cardinal::core::quat_angle;

inline bool finite3(const Vec3& v) noexcept {
    return cardinal::isfinite(v.x) && cardinal::isfinite(v.y) && cardinal::isfinite(v.z);
}
inline float len(const Vec3& v) noexcept {
    return cardinal::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}
}  // namespace

Quat rotation(const TransformComponent& t) noexcept {
    return normalize(cardinal::core::quat_from_euler_xyz(t.rotation_euler));
}

void set_rotation(TransformComponent& t, const Quat& q) noexcept {
    t.rotation_euler = cardinal::core::quat_to_euler_xyz(normalize(q));
}

Vec3 forward(const TransformComponent& t) noexcept { return rotate(rotation(t), Vec3{0.0f, 0.0f, -1.0f}); }
Vec3 right  (const TransformComponent& t) noexcept { return rotate(rotation(t), Vec3{1.0f, 0.0f,  0.0f}); }
Vec3 up     (const TransformComponent& t) noexcept { return rotate(rotation(t), Vec3{0.0f, 1.0f,  0.0f}); }

void rotate_local(TransformComponent& t, const Vec3& axis, float rad) noexcept {
    if (!finite3(axis) || !cardinal::isfinite(rad) || len(axis) < 1.0e-8f) return;
    set_rotation(t, rotation(t) * axis_angle(normalize(axis), rad));
}

void rotate_world(TransformComponent& t, const Vec3& axis, float rad) noexcept {
    if (!finite3(axis) || !cardinal::isfinite(rad) || len(axis) < 1.0e-8f) return;
    set_rotation(t, axis_angle(normalize(axis), rad) * rotation(t));
}

void look_at(TransformComponent& t, const Vec3& target, const Vec3& up_hint) noexcept {
    const Vec3 dir = target - t.translation;
    if (!finite3(dir) || len(dir) < 1.0e-6f) return;
    set_rotation(t, look_rotation(dir, up_hint));
}

void look_at_yaw(TransformComponent& t, const Vec3& target) noexcept {
    Vec3 dir = target - t.translation;
    dir.y = 0.0f;
    if (!finite3(dir) || len(dir) < 1.0e-6f) return;
    set_rotation(t, look_rotation(dir, Vec3{0.0f, 1.0f, 0.0f}));
}

float turn_towards(TransformComponent& t, const Vec3& target, float max_rad,
                   bool yaw_only, const Vec3& up_hint) noexcept {
    Vec3 dir = target - t.translation;
    if (yaw_only) dir.y = 0.0f;
    if (!finite3(dir) || len(dir) < 1.0e-6f || !cardinal::isfinite(max_rad)) return 0.0f;
    const Quat cur  = rotation(t);
    const Quat goal = look_rotation(dir, yaw_only ? Vec3{0.0f, 1.0f, 0.0f} : up_hint);
    const float angle = quat_angle(cur, goal);
    if (max_rad <= 0.0f) return angle;
    if (angle <= max_rad) { set_rotation(t, goal); return 0.0f; }
    set_rotation(t, slerp(cur, goal, max_rad / angle));
    return angle - max_rad;
}

void orbit(TransformComponent& t, const Vec3& center, const Vec3& axis, float rad) noexcept {
    if (!finite3(axis) || !finite3(center) || !cardinal::isfinite(rad) || len(axis) < 1.0e-8f) return;
    const Quat r = axis_angle(normalize(axis), rad);
    t.translation = center + rotate(r, t.translation - center);
    set_rotation(t, r * rotation(t));
}

float angle_to(const TransformComponent& t, const Quat& q) noexcept {
    return quat_angle(rotation(t), q);
}

}  // namespace cardinal::actor::xf
