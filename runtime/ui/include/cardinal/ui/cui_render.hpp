#pragma once

// =============================================================================
// Cardinal UI — Cardinal Slate (cui) -> ImGui draw-list bridge.
//
// Renders a cui::DrawList through an ImGui ImDrawList so a Cardinal Slate widget
// tree can draw INSIDE the existing ImGui-hosted Studio during the migration —
// the same role ImGui::GetForegroundDrawList() plays for the in-game HUD. Once
// the phase-2 RHI-backed cui renderer + font system land, this bridge becomes
// optional (it stays useful for embedding cui inside ImGui tools).
// =============================================================================

#include <cardinal/cui/draw.hpp>

namespace cardinal::ui {

// `imgui_draw_list` is a (void*)ImDrawList* — typically
// (void*)ImGui::GetWindowDrawList(). `origin` is added to every cui coordinate,
// so pass the target window's content-region screen position to place the tree.
void cui_render(const cardinal::cui::DrawList& dl,
                void* imgui_draw_list,
                cardinal::cui::Vec2 origin = {}) noexcept;

}  // namespace cardinal::ui
