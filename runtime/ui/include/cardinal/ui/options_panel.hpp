#pragma once

// =============================================================================
// Studio Options / Settings panel — a categorised editor over every console
// CVar (the same runtime knobs the console `set` command + config files
// drive: r.*, world.*, time.*, shadow.*, …). Bool → checkbox, Int/Float →
// slider (or input when unbounded), String → read-only (edit via the
// console). A Save/Load bar persists to a settings file via the console
// registry's save_cvars / exec_file.
//
// Free-function panel (reads console::Registry::instance()), so any host can
// drop it in with no Studio-class dependency — same shape as simd_panel /
// vgeom_panel.
// =============================================================================

namespace cardinal::ui::panels::options_panel {

void draw(const char* title = "Options", bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::options_panel
