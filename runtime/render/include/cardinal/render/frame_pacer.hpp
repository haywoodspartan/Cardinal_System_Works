#pragma once

// =============================================================================
// Cardinal — FrameTelemetry (AEGIS Block 13 → Frame Pacing telemetry).
//
// Per-pipeline frame-timing analyzer + sleep budget. DISTINCT from
// cardinal::core::FramePacer, which is the engine's multi-context FPS
// quota manager (main loop + viewport panels each registered with a
// PaceContextId, foreground/background-aware throttling). This class
// is the per-pipeline jitter analyzer the AEGIS pipeline owns:
//
//   * begin_frame() / end_frame() at the pipeline's frame boundary
//   * rolling 60-frame window → avg / min / max / jitter (stddev)
//   * Locked / Capped / Uncapped sleep modes
//   * inject_frame_ms() for deterministic tests (no wall-clock flakes)
//
// Hosts that just want "throttle the main loop to 60 FPS" continue to
// use core::FramePacer via Engine::pacer(). Hosts that want pipeline-
// level jitter telemetry for the AEGIS debug overlay use this.
//
// Class was previously named FramePacer; renamed to FrameTelemetry to
// remove the collision with core::FramePacer per the engine-wide
// "uniform to cardinal::*" cleanup directive.
//
// Frame-pacing controller for the AEGIS pipeline. The host calls
// begin_frame() at the top of each render frame and end_frame() at the
// bottom; the pacer:
//   * measures wall-clock per frame via std::chrono::high_resolution_clock
//   * tracks a rolling window of recent frame times for jitter analysis
//   * applies a CPU-side sleep budget at end_frame() to hit a target FPS
//     (important for VRR / G-Sync / FreeSync stability — uneven frame
//     times look worse than slower-but-stable frames)
//   * publishes stats: avg/min/max frame ms, jitter (stddev), dropped
//     frame count (frames exceeding 2× target ms), achieved FPS
//
// The pacer is non-graph (no Pass / ExecutionContext). It lives at the
// host loop level and pairs with the AegisPipelineRunner: the typical
// usage is:
//
//   auto pacer  = FramePacer::create(60);     // target 60 FPS
//   auto runner = AegisPipelineRunner::create(cfg, mode);
//   while (running) {
//       pacer->begin_frame();
//       runner->build(scene_inputs);
//       runner->execute();
//       pacer->end_frame();                    // sleeps if budget remaining
//       update_editor_telemetry(pacer->stats());
//   }
//
// Three pacing modes:
//   * Locked   — sleep to hit target_fps; classic locked refresh
//   * Capped   — never exceed target_fps but allow undershoot
//   * Uncapped — measure but never sleep (uncapped + VRR display)
//
// All three publish the same telemetry; only the sleep behaviour differs.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/containers.hpp>

namespace cardinal::render {

enum class FrameTelemetryMode : cardinal::u32 {
    Locked   = 0,   // sleep to exactly hit target_fps
    Capped   = 1,   // sleep only if running over budget
    Uncapped = 2,   // measure only; never sleep
};

// Back-compat alias for the previous mode-enum name.
using FramePacerMode [[deprecated("Use FrameTelemetryMode")]] = FrameTelemetryMode;

class FrameTelemetry {
public:
    static cardinal::shared_ptr<FrameTelemetry> create(cardinal::u32 target_fps = 60,
                                                       FrameTelemetryMode mode = FrameTelemetryMode::Capped);

    void begin_frame() noexcept;
    void end_frame()   noexcept;

    struct Stats {
        cardinal::u32 target_fps        {60};
        cardinal::u32 frames_observed   {0};
        cardinal::u32 frames_dropped    {0};   // frame ms > 2 × target ms
        float         avg_frame_ms      {0.0f};
        float         min_frame_ms      {0.0f};
        float         max_frame_ms      {0.0f};
        float         jitter_ms         {0.0f}; // stddev across window
        float         achieved_fps      {0.0f};
        float         last_frame_ms     {0.0f};
        float         last_sleep_ms     {0.0f};
        FrameTelemetryMode mode         {FrameTelemetryMode::Capped};
    };
    Stats stats() const noexcept;

    void set_target_fps(cardinal::u32 fps) noexcept;
    void set_mode(FrameTelemetryMode mode) noexcept;
    void reset() noexcept;

    // Editor / test surface — inject a synthetic "frame just took ms ms"
    // measurement without actually sleeping. Used by the tests to verify
    // pacing math without timing flakes.
    void inject_frame_ms(float ms) noexcept;

private:
    FrameTelemetry() = default;

    cardinal::u32       target_fps_  {60};
    FrameTelemetryMode  mode_        {FrameTelemetryMode::Capped};
    cardinal::u64     begin_time_ns_ {0};  // high_resolution_clock counts since epoch

    // Rolling window (last 60 frames) — used for avg/min/max/jitter.
    static constexpr cardinal::u32 kWindowFrames = 60;
    cardinal::vector<float> recent_ms_;
    cardinal::u32 window_head_     {0};

    cardinal::u32 frames_observed_ {0};
    cardinal::u32 frames_dropped_  {0};
    float         last_frame_ms_   {0.0f};
    float         last_sleep_ms_   {0.0f};
};

// Back-compat alias for the previous class name.
using FramePacer [[deprecated("Use render::FrameTelemetry to disambiguate from core::FramePacer")]] = FrameTelemetry;

}  // namespace cardinal::render
