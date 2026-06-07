// =============================================================================
// Cardinal — time + frame timing implementation.
// =============================================================================
#include <cardinal/core/clock/time.hpp>

namespace cardinal::core {

namespace {
const Clock::Tp& g_origin() noexcept {
    // Captured on first call. Function-local statics are thread-safe init
    // under C++11+.
    static const Clock::Tp t0 = Clock::now();
    return t0;
}
}  // namespace

f64 now_seconds() noexcept { return Clock::to_seconds(Clock::now() - g_origin()); }
f64 now_ms     () noexcept { return Clock::to_ms     (Clock::now() - g_origin()); }
u64 now_ns     () noexcept { return Clock::to_ns     (Clock::now() - g_origin()); }
u64 now_us     () noexcept { return now_ns() / 1000ull; }

FrameTimer::FrameTimer() noexcept : prev_(Clock::now()) {}

f64 FrameTimer::tick() noexcept {
    const auto now = Clock::now();
    f64 dt = Clock::to_seconds(now - prev_);
    if (dt < 0.0) dt = 0.0;
    if (dt > max_dt_) dt = max_dt_;
    prev_     = now;
    last_dt_  = dt;
    inst_fps_ = (dt > 0.0) ? (1.0 / dt) : 0.0;
    if (count_ == 0) {
        ema_fps_ = inst_fps_;
    } else {
        ema_fps_ = ema_fps_ * (1.0 - ema_alpha_) + inst_fps_ * ema_alpha_;
    }
    ++count_;
    return dt;
}

}  // namespace cardinal::core
