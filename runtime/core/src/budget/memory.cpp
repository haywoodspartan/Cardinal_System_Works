#include <cardinal/core/budget/memory.hpp>

namespace cardinal::memory {

const char* pressure_name(Pressure p) noexcept {
    switch (p) {
        case Pressure::Low:      return "Low";
        case Pressure::Medium:   return "Medium";
        case Pressure::High:     return "High";
        case Pressure::Critical: return "Critical";
    }
    return "?";
}

Pressure classify(double used_pct, const PressureThresholds& th) noexcept {
    if (used_pct >= th.critical_at_used_pct) return Pressure::Critical;
    if (used_pct >= th.high_at_used_pct)     return Pressure::High;
    if (used_pct >= th.medium_at_used_pct)   return Pressure::Medium;
    return Pressure::Low;
}

// ---------------------------------------------------------------------------
// Monitor — caches the last sample to avoid per-frame syscalls.
// ---------------------------------------------------------------------------
Monitor::Monitor() noexcept = default;

void Monitor::tick(u32 interval_ms) noexcept {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now();
    if (sampled_once_) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 now - last_sample_).count();
        if (static_cast<u64>(elapsed) < interval_ms) return;
    }
    refresh();
}

void Monitor::refresh() noexcept {
    last_sys_     = query_system();
    last_proc_    = query_process();
    last_sample_  = std::chrono::steady_clock::now();
    sampled_once_ = true;
}

Pressure Monitor::system_pressure(const PressureThresholds& th) const noexcept {
    return classify(last_sys_.load_percent, th);
}

}  // namespace cardinal::memory
