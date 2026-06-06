#pragma once

// =============================================================================
// Studio — Audio Mixer panel.
//
// Per-channel: vertical fader, mute, solo, peak meter (currently driven
// by the sum of active-instance attenuated volumes per channel — close
// enough for a development overlay).
//
// Listener position read-only display + a "preview tone" button that fires
// a 440 Hz / 0.5 s sine on the selected channel (using a built-in cue
// registered by the panel on first use).
// =============================================================================

namespace cardinal::audio { class Engine; }

namespace cardinal::ui::panels::mixer_panel {

void draw(cardinal::audio::Engine* engine,
          const char* title = "Mixer",
          bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::mixer_panel
