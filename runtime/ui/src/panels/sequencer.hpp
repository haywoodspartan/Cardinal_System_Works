#pragma once

// =============================================================================
// Studio — Sequencer (cinematic timeline).
//
// Renders a cardinal::cine::Sequence as a horizontal timeline with one row
// per track + a transport bar (play/pause/stop/seek). Track contents
// (animation clips, camera shots, audio cues, event markers) appear as
// colored regions / diamonds the user can drag to reposition.
//
// State is mutated in place — the host doesn't need to re-set anything.
// =============================================================================

namespace cardinal::cine { class Player; struct Sequence; }

namespace cardinal::ui::panels::sequencer_panel {

void draw(cardinal::cine::Player* player,
          const char* title = "Sequencer",
          bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::sequencer_panel
