#pragma once

// =============================================================================
// Studio — Memory & Budget panel.
//
// Reads-only of the live cardinal::budget::Broker:
//   - System RAM  : total / available / load %  (+ pressure tier swatch)
//   - Process     : working set / peak / private bytes
//   - GPU VRAM    : budget / current usage      (+ pressure tier swatch)
//   - Subsystems  : every registered consumer with used/min/max bar
//
// Plus debug dropdowns to pin pressure tiers (so a developer can verify
// their subsystem callbacks fire without stressing the OS).
// =============================================================================

#include <cardinal/core/types.hpp>

namespace cardinal::budget { class Broker; }

namespace cardinal::ui::panels::memory_panel {

void draw(cardinal::budget::Broker* broker, const char* title, bool* p_open);

}  // namespace cardinal::ui::panels::memory_panel
