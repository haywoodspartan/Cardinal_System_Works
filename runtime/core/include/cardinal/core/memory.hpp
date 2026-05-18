#pragma once

// =============================================================================
// Cardinal — system + process memory monitoring.
//
// Cross-platform (Win32 + Linux) snapshots that the BudgetBroker (budget.hpp)
// uses to derive a pressure tier and notify subsystems. Pure introspection —
// no allocation hooks, no overrides; we read the OS counters and react.
//
//   SystemSnapshot   — total / available physical RAM the OS is reporting.
//                      On Windows: GlobalMemoryStatusEx.
//                      On Linux:   /proc/meminfo (MemTotal + MemAvailable).
//
//   ProcessSnapshot  — what THIS process currently owns.
//                      On Windows: GetProcessMemoryInfo (working set + private).
//                      On Linux:   /proc/self/status (VmRSS + VmData).
//
// Costs: each snapshot is a single syscall (Win) or a small file read (Linux).
// Cheap enough to call once per frame; the Monitor caches at a configurable
// interval so subsystems can poll without re-syscalling.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <chrono>

namespace cardinal::memory {

// ---------------------------------------------------------------------------
// Snapshots — pure data, no work happens here.
// ---------------------------------------------------------------------------
struct SystemSnapshot {
    u64    total_bytes{0};       // physical RAM the OS sees
    u64    available_bytes{0};   // free + reclaimable, what the OS hands out
    double load_percent{0.0};    // 100.0 - (available / total * 100)
};

struct ProcessSnapshot {
    u64 working_set_bytes{0};       // resident pages this process owns
    u64 peak_working_set_bytes{0};  // high-water mark since process start
    u64 private_bytes{0};           // committed (private) virtual memory
};

// ---------------------------------------------------------------------------
// Pressure tiers. Subsystems get notified on transitions and react:
//
//   Low      — plenty of headroom. Subsystems may grow caches.
//   Medium   — comfortable. Hold steady; new growth is OK if cheap.
//   High     — tight. Stop growing; voluntary trim of soft caches.
//   Critical — emergency. Forced eviction; refuse new allocs where possible.
//
// The thresholds are configurable per Domain (system vs. GPU). Defaults
// target the typical ratio: free / total. Higher = less free = worse.
// ---------------------------------------------------------------------------
enum class Pressure : u32 {
    Low      = 0,
    Medium   = 1,
    High     = 2,
    Critical = 3,
};

const char* pressure_name(Pressure p) noexcept;

struct PressureThresholds {
    // Used percentages that promote into each tier (monotonically increasing).
    //   used_pct = 100 * (1 - available/total)  on system memory.
    //   used_pct = 100 * (current/budget)       on GPU memory.
    double medium_at_used_pct  {60.0};
    double high_at_used_pct    {80.0};
    double critical_at_used_pct{92.0};
};

Pressure classify(double used_pct, const PressureThresholds& thresholds) noexcept;

// ---------------------------------------------------------------------------
// Snapshot APIs — no caching. Call when you need a fresh number.
// ---------------------------------------------------------------------------
SystemSnapshot   query_system () noexcept;
ProcessSnapshot  query_process() noexcept;

// ---------------------------------------------------------------------------
// Monitor — light cache + tick driver. Stores the last snapshot, refreshes
// at most once every `interval_ms`. Construction is free; tick() does the
// work. Thread-safe to read after a tick has completed (see budget broker).
//
// Typical use: one Monitor lives in the engine, the BudgetBroker calls
// tick() each frame and inspects last_*().
// ---------------------------------------------------------------------------
class Monitor {
public:
    Monitor() noexcept;

    // No-op if it's not yet been `interval_ms` since the last sample.
    void tick(u32 interval_ms = 250) noexcept;

    // Force a sample regardless of interval.
    void refresh() noexcept;

    const SystemSnapshot&  last_system () const noexcept { return last_sys_;  }
    const ProcessSnapshot& last_process() const noexcept { return last_proc_; }

    // Convenience: classify the current system snapshot under the given
    // thresholds. Reads cached snapshot — does not refresh.
    Pressure system_pressure(const PressureThresholds& th = {}) const noexcept;

private:
    SystemSnapshot                                 last_sys_{};
    ProcessSnapshot                                last_proc_{};
    std::chrono::steady_clock::time_point          last_sample_{};
    bool                                           sampled_once_{false};
};

}  // namespace cardinal::memory
