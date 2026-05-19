#pragma once

// =============================================================================
// Cardinal UI — ImGui master reference template.
//
// SAME rule as the cardinal::core FOUNDATION RULE, applied to the editor's
// third-party UI dependency: <imgui.h> is referenced HERE and nowhere else
// in the Studio function surface. Every Studio panel / facade TU that draws
// UI includes <cardinal/ui/imgui.hpp> instead of <imgui.h> directly, so the
// Dear ImGui version / config / include policy stays restructurable from one
// place (and a future ImGui bump, a wrapper shim, or an IMGUI_USER_CONFIG
// only has to change this header).
//
// SCOPE — deliberately "just the Studio functions currently":
//   * In  : runtime/ui Studio code — studio.cpp, studio_engine.cpp, and the
//            src/panels/*.{cpp,hpp} draw functions. These only need the
//            public ImGui API surface (this header).
//   * Out : imgui_internal.h and the imgui_impl_*.h backend headers stay as
//            explicit includes in the two studio-internal / backend-wiring
//            TUs that actually need them (studio.cpp does layout-sanity +
//            backend init; studio_engine.cpp does dock-builder layout) —
//            pulling those heavy headers into 38 panels would bloat build
//            time for no API benefit. They are a separate, narrower
//            concern than the panel "functions".
//   * Out : runtime/ui/external/imgui_backends/* are vendored upstream
//            sources — they #include "imgui.h" by design and are never
//            routed through this master reference.
//
// imgui::imgui is linked PRIVATE on cardinal_ui, so this header is only
// resolvable from within the UI library's own TUs — exactly the intended
// blast radius (non-UI code never sees ImGui).
// =============================================================================

#include <imgui.h>

// ---------------------------------------------------------------------------
// Reusable ImGui scope helpers — the SAME "template reusable structs for
// reused functions" discipline applied to Dear ImGui's Push/Pop and
// Begin/End pairs. A dropped Pop*/EndDisabled is a classic, recurring
// ImGui defect (it silently corrupts every subsequent widget's style /
// id stack). These RAII guards make the pair impossible to unbalance and
// live in ONE place so all ~25 Studio panel TUs share the exact same
// vetted implementation instead of re-hand-rolling Push/Pop 100+ times.
//
// Templated where the underlying ImGui call is type-overloaded (PushID
// over int/const char*/void*, PushStyleVar over float/ImVec2) so one
// guard serves every overload. Usage:
//
//   { cardinal::ui::StyleColorScope c(ImGuiCol_Text, col);
//     ImGui::Text("..."); }                       // PopStyleColor at }
//   { cardinal::ui::IdScope id(row);  draw_row(); }   // PopID at }
//   { cardinal::ui::DisabledScope d(!enabled); ... }  // EndDisabled at }
// ---------------------------------------------------------------------------
namespace cardinal::ui {

// ImGui::PushID / PopID. Id ∈ { int, const char*, const void* / T* }.
template <class Id>
struct IdScope {
    explicit IdScope(Id id) noexcept { ImGui::PushID(id); }
    ~IdScope() { ImGui::PopID(); }
    IdScope(const IdScope&)            = delete;
    IdScope& operator=(const IdScope&) = delete;
};
template <class Id> IdScope(Id) -> IdScope<Id>;

// ImGui::PushStyleVar / PopStyleVar. V ∈ { float, ImVec2 }.
template <class V>
struct StyleVarScope {
    StyleVarScope(ImGuiStyleVar var, V value) noexcept {
        ImGui::PushStyleVar(var, value);
    }
    ~StyleVarScope() { ImGui::PopStyleVar(); }
    StyleVarScope(const StyleVarScope&)            = delete;
    StyleVarScope& operator=(const StyleVarScope&) = delete;
};
template <class V> StyleVarScope(ImGuiStyleVar, V) -> StyleVarScope<V>;

// ImGui::PushStyleColor / PopStyleColor (ImU32 or ImVec4 colour).
struct StyleColorScope {
    StyleColorScope(ImGuiCol idx, ImU32 col) noexcept {
        ImGui::PushStyleColor(idx, col);
    }
    StyleColorScope(ImGuiCol idx, const ImVec4& col) noexcept {
        ImGui::PushStyleColor(idx, col);
    }
    ~StyleColorScope() { ImGui::PopStyleColor(); }
    StyleColorScope(const StyleColorScope&)            = delete;
    StyleColorScope& operator=(const StyleColorScope&) = delete;
};

// ImGui::BeginDisabled / EndDisabled.
struct DisabledScope {
    explicit DisabledScope(bool disabled = true) noexcept {
        ImGui::BeginDisabled(disabled);
    }
    ~DisabledScope() { ImGui::EndDisabled(); }
    DisabledScope(const DisabledScope&)            = delete;
    DisabledScope& operator=(const DisabledScope&) = delete;
};

// Dear-ImGui-demo-style "(?)" hover help marker — the duplicated
// IsItemHovered()+tooltip idiom, once.
inline void HelpMarker(const char* desc) {
    ImGui::TextDisabled("(?)");
    if (desc && ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

}  // namespace cardinal::ui
