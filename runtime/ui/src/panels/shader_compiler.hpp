#pragma once

// =============================================================================
// Studio — Shader compiler panel.
//
// Free-form HLSL editor + compile button + cache stats. Compile picks the
// stage from a dropdown, writes the result + diagnostics back into the
// panel.
// =============================================================================

namespace cardinal::shader { class Compiler; }

namespace cardinal::ui::panels::shader_compiler_panel {

void draw(cardinal::shader::Compiler* compiler,
          const char* title = "Shader Compiler",
          bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::shader_compiler_panel
