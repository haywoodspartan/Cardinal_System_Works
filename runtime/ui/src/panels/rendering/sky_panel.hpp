#pragma once

// =============================================================================
// Studio — Sky / Time-of-Day panel.
//
// Drives a cardinal::sky::Sky:
//   - Hour slider (0..24, scrub interactively)
//   - Time scale (hours/sec) + day-length helper (1 minute = 24h day, etc.)
//   - Freeze toggle
//   - Live preview of sun direction + colors
//   - Editable phase keys (zenith / horizon / sun colors)
// =============================================================================

namespace cardinal::sky { class Sky; }

namespace cardinal::ui::panels::sky_panel {

void draw(cardinal::sky::Sky* sky,
          const char* title = "Sky / Time of Day",
          bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::sky_panel
