#include <cardinal/scene/light.hpp>

#include <algorithm>
#include <cmath>

namespace cardinal::scene {

namespace {

inline float clamp01(float x) noexcept {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

inline float vlen(const Vec3& v) noexcept {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline Vec3 vnorm(const Vec3& v) noexcept {
    const float l = vlen(v);
    if (l < 1e-6f) return {0, 0, 1};
    return { v.x / l, v.y / l, v.z / l };
}

}  // namespace

Vec3 LightSet::shade(const Vec3& world_pos, const Vec3& world_normal,
                     const Vec3& base_color) const noexcept
{
    Vec3 acc{
        base_color.x * ambient_.x,
        base_color.y * ambient_.y,
        base_color.z * ambient_.z
    };
    const Vec3 n = vnorm(world_normal);

    for (const auto& l : lights_) {
        Vec3  L{};
        float atten = 1.0f;
        switch (l.kind) {
            case LightKind::Directional: {
                // L points from surface toward the sun.
                L = vnorm({ -l.direction.x, -l.direction.y, -l.direction.z });
                break;
            }
            case LightKind::Point: {
                Vec3 to{ l.position.x - world_pos.x,
                         l.position.y - world_pos.y,
                         l.position.z - world_pos.z };
                const float d = vlen(to);
                if (d < 1e-4f || d > l.range) continue;
                L = { to.x / d, to.y / d, to.z / d };
                // Inverse-square with linear taper at the range edge.
                const float k = 1.0f - (d / l.range);
                atten = k * k;
                break;
            }
            case LightKind::Spot: {
                Vec3 to{ l.position.x - world_pos.x,
                         l.position.y - world_pos.y,
                         l.position.z - world_pos.z };
                const float d = vlen(to);
                if (d < 1e-4f || d > l.range) continue;
                L = { to.x / d, to.y / d, to.z / d };
                const Vec3 axis = vnorm(l.direction);
                // Cone test — light direction is the cone axis (forward).
                const float cosAngle = -(L.x * axis.x + L.y * axis.y + L.z * axis.z);
                if (cosAngle < l.spot_outer_cos) continue;
                const float k = 1.0f - (d / l.range);
                atten = k * k;
                if (cosAngle < l.spot_inner_cos) {
                    const float fall = (cosAngle - l.spot_outer_cos) /
                        std::max(1e-4f, (l.spot_inner_cos - l.spot_outer_cos));
                    atten *= clamp01(fall);
                }
                break;
            }
        }
        const float NdotL = std::max(0.0f, n.x * L.x + n.y * L.y + n.z * L.z);
        const float k = NdotL * atten * l.intensity;
        acc.x += base_color.x * l.color.x * k;
        acc.y += base_color.y * l.color.y * k;
        acc.z += base_color.z * l.color.z * k;
    }
    return acc;
}

}  // namespace cardinal::scene
