#pragma once

// =============================================================================
// Cardinal — time + frame timing primitives.
//
// `Clock`       — wraps std::chrono::steady_clock with engine-friendly types
// `now_*()`     — monotonic time-since-process-start in s / ms / us / ns
// `FrameTimer`  — running frame-time + EMA fps; tick() each frame
// `ScopedTimer` — RAII duration timer; logs at scope exit
//
// All times are seconds (f64) unless suffixed _ms / _us / _ns. The clock is
// monotonic (never goes backward, immune to wall-clock changes), which is
// what every game-loop / profiler / pacer wants.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <chrono>

namespace cardinal::core {

class Clock {
public:
    using Steady = std::chrono::steady_clock;
    using Tp     = Steady::time_point;
    using Dur    = Steady::duration;

    static Tp   now() noexcept { return Steady::now(); }

    static f64  to_seconds(Dur d) noexcept {
        return std::chrono::duration<f64>(d).count();
    }
    static f64  to_ms(Dur d) noexcept {
        return std::chrono::duration<f64, std::milli>(d).count();
    }
    static u64  to_ns(Dur d) noexcept {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
    }

private:
    Clock() = default;
};

// Monotonic time since the process start (specifically since the first call
// to one of these). Cheap to call; intended for per-frame timestamps.
f64 now_seconds() noexcept;
f64 now_ms()      noexcept;
u64 now_us()      noexcept;
u64 now_ns()      noexcept;

// Frame timer — owns the previous frame's timestamp + a running EMA of fps.
// Call tick() at a consistent point each frame (top of loop is conventional).
class FrameTimer {
public:
    FrameTimer() noexcept;

    // Returns the delta-time (seconds) since the previous tick(). The first
    // call returns 0. Caps the dt at `max_dt_seconds` so a debugger break
    // doesn't cause a 30-second physics step on resume.
    f64 tick() noexcept;

    f64 dt_seconds()      const noexcept { return last_dt_; }
    f64 fps_instant()     const noexcept { return inst_fps_; }
    f64 fps_ema()         const noexcept { return ema_fps_;  }
    u64 frame_count()     const noexcept { return count_;    }

    void set_max_dt_seconds(f64 s) noexcept { max_dt_ = s;  }
    void set_ema_alpha    (f64 a) noexcept { ema_alpha_ = a; }

private:
    Clock::Tp prev_{};
    f64       last_dt_{0.0};
    f64       inst_fps_{0.0};
    f64       ema_fps_{0.0};
    f64       ema_alpha_{0.10};   // 10% per frame; ~10-frame window
    f64       max_dt_{0.25};      // cap dt at 250ms (= ≥4 FPS minimum)
    u64       count_{0};
};

// RAII duration timer. Captures a label + start tp on construction; on
// destruction calls `f(label, ms)` so the caller decides whether to log,
// accumulate to a profiler, etc.
template <typename Sink>
class ScopedTimer {
public:
    ScopedTimer(const char* label, Sink sink) noexcept
        : label_(label), sink_(static_cast<Sink&&>(sink)), start_(Clock::now()) {}
    ~ScopedTimer() noexcept {
        const f64 ms = Clock::to_ms(Clock::now() - start_);
        sink_(label_, ms);
    }
    ScopedTimer(const ScopedTimer&)            = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
private:
    const char* label_;
    Sink        sink_;
    Clock::Tp   start_;
};

}  // namespace cardinal::core
