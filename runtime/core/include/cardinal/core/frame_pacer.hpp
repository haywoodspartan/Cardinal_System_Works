#pragma once

// =============================================================================
// Cardinal — frame pacer.
//
// Handles three things in one place so neither the renderer nor the editor
// has to scatter timing logic:
//
//   1. **Foreground / background detection.** Win32 GetForegroundWindow()
//      compared against the application's main HWND. The pacer auto-flips
//      between two FrameLimit profiles based on which is active.
//
//   2. **Multi-context FPS quotas.** Each "render context" (the main loop
//      itself + each editor viewport panel) registers an integer id with
//      its own desired FPS cap. The pacer takes the *minimum non-zero*
//      cap across active contexts as the effective target — i.e. if any
//      viewport asks for 240 Hz we run the loop at 240, but if every
//      viewport says 30 we run at 30. A cap of 0 means "uncapped" for
//      that context (no contribution to the min).
//
//   3. **Sleep budget.** wait_for_next_frame() stalls just long enough
//      that the requested cap is honoured. Uses the OS timer via
//      std::this_thread::sleep_for, with a small spin-yield on the last
//      sub-millisecond for jitter reduction.
//
// The pacer is platform-aware: on non-Windows the foreground check
// degrades gracefully (always reports foreground = true) so the timing
// behaviour stays correct.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <chrono>
#include <memory>

namespace cardinal::core {

// One FPS cap profile. 0.0f means "uncapped" — the pacer will not sleep
// on behalf of that profile (effectively no cap from this context).
struct FrameLimit {
    float fps_foreground{0.0f};   // active when our HWND is the foreground window
    float fps_background{30.0f};  // active when another window has focus
};

// Stable id for a render context registered with the pacer. The main loop
// is implicitly id 0; viewport panels typically register ids 1..N.
using PaceContextId = u32;
constexpr PaceContextId kPaceMainLoop = 0;

class FramePacer {
public:
    static std::unique_ptr<FramePacer> create();
    virtual ~FramePacer() = default;
    FramePacer(const FramePacer&)            = delete;
    FramePacer& operator=(const FramePacer&) = delete;

    // Tell the pacer which OS window to track for foreground/background
    // detection. Cast a HWND to void* on Windows; pass nullptr to disable
    // (the pacer will assume foreground at all times).
    virtual void set_window(void* native_handle) noexcept = 0;

    // Set or update the FPS quota for a context. Calling with both fields
    // 0 effectively un-throttles that context.
    virtual void set_limit(PaceContextId ctx, const FrameLimit& limit) = 0;
    virtual FrameLimit limit(PaceContextId ctx) const = 0;
    virtual void clear_limit(PaceContextId ctx) = 0;

    // Sleep until the next frame deadline implied by the active limits.
    // No-op when no limit is set or when the previous frame already
    // overran the budget. Returns the time slept (microseconds).
    virtual u64 wait_for_next_frame() = 0;

    // Cached foreground state (refreshed each wait_for_next_frame call).
    virtual bool is_foreground() const noexcept = 0;

    // The effective FPS cap actually applied this frame (min of active
    // contexts), or 0.0f if uncapped. Useful for stats overlays.
    virtual float effective_fps_cap() const noexcept = 0;

protected:
    FramePacer() = default;
};

}  // namespace cardinal::core
