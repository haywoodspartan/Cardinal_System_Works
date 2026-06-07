// =============================================================================
// Cardinal UI — gizmo math primitives (Studio decomposition, step 2).
//
// Verbatim move of project_ / mouse_world_ray_ / ray_plane_ out of
// studio.cpp. No logic change — the same code in its own TU so the
// gizmo math (a regression-prone area: detached-window placement /
// aspect) is independently editable without opening the ~4000-line
// StudioImpl monolith. StudioImpl forwards to these.
// =============================================================================
#include "studio_gizmo_math.hpp"

#include <cardinal/core/std/cmath.hpp>   // cardinal::fabs / isfinite

namespace cardinal::ui::detail {

bool project_(const cardinal::scene::Mat4& vp, const ImVec2& ip,
              const ImVec2& av, const cardinal::scene::Vec3& wp,
              ImVec2& out) {
    const cardinal::scene::Vec4 c =
        vp * cardinal::scene::Vec4{ wp.x, wp.y, wp.z, 1.0f };
    if (c.w <= 0.001f) return false;
    out.x = ip.x + (c.x / c.w * 0.5f + 0.5f) * av.x;
    out.y = ip.y + (1.0f - (c.y / c.w * 0.5f + 0.5f)) * av.y;
    return true;
}

bool mouse_world_ray_(const ImVec2& ip, const ImVec2& av,
                      const cardinal::scene::Mat4& inv_vp,
                      cardinal::scene::Vec3& ro,
                      cardinal::scene::Vec3& rd) {
    const ImVec2 mp = ImGui::GetIO().MousePos;
    const float nx =        ((mp.x - ip.x) / av.x) * 2.0f - 1.0f;
    const float ny = 1.0f - ((mp.y - ip.y) / av.y) * 2.0f;
    const cardinal::scene::Vec4 nc{ nx, ny, 0.0f, 1.0f };
    const cardinal::scene::Vec4 fc{ nx, ny, 1.0f, 1.0f };
    const cardinal::scene::Vec4 nw = inv_vp * nc;
    const cardinal::scene::Vec4 fw = inv_vp * fc;
    if (cardinal::fabs(nw.w) < 1e-6f || cardinal::fabs(fw.w) < 1e-6f) return false;
    ro = { nw.x/nw.w, nw.y/nw.w, nw.z/nw.w };
    const cardinal::scene::Vec3 fp{ fw.x/fw.w, fw.y/fw.w, fw.z/fw.w };
    rd = cardinal::scene::normalize(
        cardinal::scene::Vec3{ fp.x-ro.x, fp.y-ro.y, fp.z-ro.z });
    return true;
}

bool ray_plane_(const cardinal::scene::Vec3& ro,
                const cardinal::scene::Vec3& rd,
                const cardinal::scene::Vec3& p0,
                const cardinal::scene::Vec3& n,
                cardinal::scene::Vec3& hit) {
    const float dn = cardinal::scene::dot(rd, n);
    if (cardinal::fabs(dn) < 1e-6f) return false;
    const cardinal::scene::Vec3 w{ p0.x-ro.x, p0.y-ro.y, p0.z-ro.z };
    const float t = cardinal::scene::dot(w, n) / dn;
    if (!cardinal::isfinite(t)) return false;
    hit = { ro.x+rd.x*t, ro.y+rd.y*t, ro.z+rd.z*t };
    return true;
}

}  // namespace cardinal::ui::detail
