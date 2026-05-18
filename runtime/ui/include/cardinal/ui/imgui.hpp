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
