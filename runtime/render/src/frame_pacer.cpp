#include <cardinal/render/frame_pacer.hpp>

#include <cardinal/core/cmath.hpp>
#include <cardinal/core/algorithm.hpp>

#include <chrono>
#include <thread>

namespace cardinal::render {

namespace {

inline cardinal::u64 now_ns() noexcept {
    return static_cast<cardinal::u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count());
}

inline float target_ms_for(cardinal::u32 fps) noexcept {
    if (fps == 0) return 0.0f;
    return 1000.0f / static_cast<float>(fps);
}

}  // namespace

cardinal::shared_ptr<FrameTelemetry> FrameTelemetry::create(cardinal::u32 fps, FrameTelemetryMode mode) {
    auto p = cardinal::shared_ptr<FrameTelemetry>(new FrameTelemetry());
    p->target_fps_ = cardinal::clamp(fps, 1u, 1000u);
    p->mode_       = mode;
    p->recent_ms_.assign(kWindowFrames, 0.0f);
    return p;
}

void FrameTelemetry::set_target_fps(cardinal::u32 fps) noexcept {
    target_fps_ = cardinal::clamp(fps, 1u, 1000u);
}
void FrameTelemetry::set_mode(FrameTelemetryMode mode) noexcept { mode_ = mode; }

void FrameTelemetry::reset() noexcept {
    frames_observed_ = 0;
    frames_dropped_  = 0;
    last_frame_ms_   = 0.0f;
    last_sleep_ms_   = 0.0f;
    window_head_     = 0;
    recent_ms_.assign(kWindowFrames, 0.0f);
}

void FrameTelemetry::begin_frame() noexcept {
    begin_time_ns_ = now_ns();
}

void FrameTelemetry::end_frame() noexcept {
    const cardinal::u64 t1 = now_ns();
    const float work_ms = static_cast<float>(t1 - begin_time_ns_) / 1.0e6f;
    last_frame_ms_ = work_ms;

    // Sleep for the remaining budget (if any).
    float sleep_ms = 0.0f;
    if (mode_ != FrameTelemetryMode::Uncapped) {
        const float target = target_ms_for(target_fps_);
        if (work_ms < target) {
            sleep_ms = target - work_ms;
            // Locked + Capped both sleep when undershooting. Locked uses a
            // small safety margin so the OS schedule slop doesn't make
            // every frame overshoot.
            const float margin = (mode_ == FrameTelemetryMode::Locked) ? 0.1f : 0.0f;
            const float to_sleep = sleep_ms - margin;
            if (to_sleep > 0.0f) {
                std::this_thread::sleep_for(std::chrono::microseconds(
                    static_cast<cardinal::i64>(to_sleep * 1000.0f)));
            }
        }
    }
    last_sleep_ms_ = sleep_ms;

    inject_frame_ms(work_ms + sleep_ms);
}

void FrameTelemetry::inject_frame_ms(float ms) noexcept {
    if (!cardinal::isfinite(ms) || ms < 0.0f) ms = 0.0f;
    recent_ms_[window_head_ % kWindowFrames] = ms;
    window_head_++;
    frames_observed_++;
    const float target = target_ms_for(target_fps_);
    if (target > 0.0f && ms > 2.0f * target) frames_dropped_++;
}

FrameTelemetry::Stats FrameTelemetry::stats() const noexcept {
    Stats s;
    s.target_fps      = target_fps_;
    s.frames_observed = frames_observed_;
    s.frames_dropped  = frames_dropped_;
    s.last_frame_ms   = last_frame_ms_;
    s.last_sleep_ms   = last_sleep_ms_;
    s.mode            = mode_;

    const cardinal::u32 n = (frames_observed_ < kWindowFrames)
        ? frames_observed_ : kWindowFrames;
    if (n == 0) return s;

    float sum = 0, mn = 1.0e30f, mx = 0.0f;
    for (cardinal::u32 i = 0; i < n; ++i) {
        const float v = recent_ms_[i];
        sum += v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    const float avg = sum / static_cast<float>(n);
    float var = 0.0f;
    for (cardinal::u32 i = 0; i < n; ++i) {
        const float d = recent_ms_[i] - avg;
        var += d * d;
    }
    var /= static_cast<float>(n);
    s.avg_frame_ms = avg;
    s.min_frame_ms = mn;
    s.max_frame_ms = mx;
    s.jitter_ms    = cardinal::sqrt(var);
    s.achieved_fps = (avg > 0.0f) ? (1000.0f / avg) : 0.0f;
    return s;
}

}  // namespace cardinal::render
