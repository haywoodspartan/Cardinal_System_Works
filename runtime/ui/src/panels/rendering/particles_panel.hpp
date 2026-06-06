#pragma once

// =============================================================================
// Studio — Particle system panel.
//
// Lists every emitter in a cardinal::particles::System with controls for
// rate, lifetime, velocity range, gravity, drag, color start/end, max
// particles, emit toggle. Spawns a fresh emitter via "+ Add Emitter".
// =============================================================================

namespace cardinal::particles { class System; }

namespace cardinal::ui::panels::particles_panel {

void draw(cardinal::particles::System* sys,
          const char* title = "Particles",
          bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::particles_panel
