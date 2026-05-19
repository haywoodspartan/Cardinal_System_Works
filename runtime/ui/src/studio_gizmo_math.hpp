#pragma once

// =============================================================================
// Cardinal UI — gizmo math primitives (Studio decomposition, step 2).
//
// project_ / mouse_world_ray_ / ray_plane_ carved VERBATIM out of the
// StudioImpl monolith. All three are pure / member-independent (they
// only touch their parameters + the ImGui IO global + free scene-math
// helpers — never `this`), so relocating them is behaviour-identical
// and cannot perturb the fragile gizmo/overlay state. StudioImpl keeps
// thin one-line forwarders so every existing call site is unchanged.
//
// Private UI-internal header (runtime/ui/src, NOT exported).
// =============================================================================

#include <cardinal/ui/imgui.hpp>     // ImVec2, ImGui::GetIO
#include <cardinal/scene/math.hpp>   // cardinal::scene::Mat4 / Vec3 / Vec4

namespace cardinal::ui::detail {

// World point -> screen pixel inside the panel image rect [ip, ip+av].
// Returns false when the point is at/behind the near plane.
bool project_(const cardinal::scene::Mat4& vp, const ImVec2& ip,
              const ImVec2& av, const cardinal::scene::Vec3& wp,
              ImVec2& out);

// Current mouse position -> world ray (ro = origin, rd = normalised
// direction) via the inverse view-proj and the panel image rect.
// Returns false on a degenerate (w≈0) unprojection.
bool mouse_world_ray_(const ImVec2& ip, const ImVec2& av,
                      const cardinal::scene::Mat4& inv_vp,
                      cardinal::scene::Vec3& ro,
                      cardinal::scene::Vec3& rd);

// Ray (ro, rd) vs plane (point p0, normal n) -> hit point.
// Returns false when the ray is parallel to the plane.
bool ray_plane_(const cardinal::scene::Vec3& ro,
                const cardinal::scene::Vec3& rd,
                const cardinal::scene::Vec3& p0,
                const cardinal::scene::Vec3& n,
                cardinal::scene::Vec3& hit);

}  // namespace cardinal::ui::detail
