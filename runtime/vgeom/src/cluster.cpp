// =============================================================================
// Cardinal — vgeom math helpers (sphere projection + frustum distance).
//
// The Mesh-attachment registry lives in scene/ — this layer is pure
// algorithm + data, no scene::Mesh awareness.
// =============================================================================
#include <cardinal/vgeom/vgeom.hpp>
#include <cardinal/vgeom/cluster.hpp>

#include <cmath>

namespace cardinal::vgeom {

f32 project_sphere_pixel_radius(const Vec3& center_local,
                                f32 radius_local,
                                const Mat4& model,
                                const Mat4& view,
                                const Mat4& proj,
                                f32 viewport_pixel_height) noexcept
{
    if (radius_local <= 0.0f) return 0.0f;

    auto mul_pos = [](const Mat4& m, const Vec3& v) noexcept {
        Vec3 r{};
        r.x = m.m[0][0]*v.x + m.m[1][0]*v.y + m.m[2][0]*v.z + m.m[3][0];
        r.y = m.m[0][1]*v.x + m.m[1][1]*v.y + m.m[2][1]*v.z + m.m[3][1];
        r.z = m.m[0][2]*v.x + m.m[1][2]*v.y + m.m[2][2]*v.z + m.m[3][2];
        return r;
    };
    const Vec3 world_center = mul_pos(model, center_local);

    auto col_len = [&](int c) noexcept {
        return std::sqrt(model.m[c][0]*model.m[c][0]
                       + model.m[c][1]*model.m[c][1]
                       + model.m[c][2]*model.m[c][2]);
    };
    const f32 s   = std::max(col_len(0), std::max(col_len(1), col_len(2)));
    const f32 wr  = radius_local * s;

    const Vec3 view_center = mul_pos(view, world_center);
    const f32  z = -view_center.z;

    const f32 m11      = proj.m[1][1];
    const f32 is_ortho = (std::fabs(proj.m[3][3]) > 0.5f) ? 1.0f : 0.0f;

    if (z <= wr && is_ortho == 0.0f) return 1e9f;
    if (m11 == 0.0f) return 1e9f;

    const f32 perspective_pixel_r = (wr * m11) / std::max(z, 1e-3f)
                                  * (viewport_pixel_height * 0.5f);
    const f32 ortho_pixel_r       = wr * std::fabs(m11)
                                  * (viewport_pixel_height * 0.5f);
    return is_ortho > 0.0f ? ortho_pixel_r : perspective_pixel_r;
}

f32 distance2_from_camera(const Vec3& center_local,
                          const Mat4& model,
                          const Mat4& view) noexcept
{
    Vec3 w{};
    w.x = model.m[0][0]*center_local.x + model.m[1][0]*center_local.y + model.m[2][0]*center_local.z + model.m[3][0];
    w.y = model.m[0][1]*center_local.x + model.m[1][1]*center_local.y + model.m[2][1]*center_local.z + model.m[3][1];
    w.z = model.m[0][2]*center_local.x + model.m[1][2]*center_local.y + model.m[2][2]*center_local.z + model.m[3][2];
    Vec3 v{};
    v.x = view.m[0][0]*w.x + view.m[1][0]*w.y + view.m[2][0]*w.z + view.m[3][0];
    v.y = view.m[0][1]*w.x + view.m[1][1]*w.y + view.m[2][1]*w.z + view.m[3][1];
    v.z = view.m[0][2]*w.x + view.m[1][2]*w.y + view.m[2][2]*w.z + view.m[3][2];
    return v.x*v.x + v.y*v.y + v.z*v.z;
}

}  // namespace cardinal::vgeom
