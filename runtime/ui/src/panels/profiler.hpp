#pragma once

// =============================================================================
// Studio — Profiler panel.
//
// Reads from cardinal::async (frame phases + worker stats) and renders:
//   - per-frame timing chart (last ~120 frames)
//   - per-phase ms/avg/max table
//   - per-worker busy bar + jobs-executed counter
//   - active worker count + perf/general split
//
// No state besides a tiny circular history of overall frame time the panel
// builds itself (cardinal::async doesn't track total frame time — only
// per-phase).
// =============================================================================

#include <cardinal/core/types.hpp>

namespace cardinal::ui::panels::profiler_panel {

struct State {
    cardinal::f32 frame_history[256]{};   // ms
    cardinal::u32 history_pos{0};
    cardinal::u32 history_count{0};
    cardinal::f32 last_frame_ms{0.0f};
};

// `frame_ms_now` is the host's measurement of the most recent end-of-frame
// duration. The panel pushes it into its rolling chart.
void draw(State& state, cardinal::f32 frame_ms_now,
          const char* title = "Profiler", bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::profiler_panel
