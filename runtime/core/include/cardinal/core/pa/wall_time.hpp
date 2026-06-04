#pragma once

// =============================================================================
// Cardinal core — Time / CpuTime / CpuUsage — modern C++20 port
// of Pearl Abyss PaTime.h's wall-clock + CPU-accounting surface.
//
// Mapping vs. original:
//   * Time wraps std::chrono::system_clock + the calendar fields the Pa
//     code base treats as the canonical local-time struct (Year/Month/Day/
//     Hour/Minute/Second/MillSecond/DayOfWeek). Internally we keep two
//     representations side-by-side: a system_clock::time_point for math and
//     a SYSTEMTIME-equivalent broken-down struct for accessors. Conversion
//     uses std::chrono::time_point + std::chrono::current_zone() (C++20).
//   * CpuTime + CpuUsage wrap Win32 GetSystemTimes (Idle/Kernel/User
//     FILETIME triple) into a portable struct. Non-Windows: all zeros (the
//     CPU-busy meter is a Win-only feature for now — Linux equivalent is
//     /proc/stat parsing, deferred until needed).
//
// Coexists with cardinal::core::time (mono clock for engine timing).
// Time is wall-clock for date-stamped logs, cron-like schedules, mail
// timestamps — everything that wants real calendar fields.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/platform.hpp>

#if CARDINAL_PLATFORM_WINDOWS
#include <Windows.h>   // SYSTEMTIME, FILETIME, GetSystemTimes
#endif

namespace cardinal::core {

// Day-of-week constants matching SYSTEMTIME::wDayOfWeek (Sunday = 0).
enum class DayOfWeek : u16 {
    Sunday = 0, Monday = 1, Tuesday = 2, Wednesday = 3,
    Thursday = 4, Friday = 5, Saturday = 6, Unknown = 0xFFFFu,
};

// ---------------------------------------------------------------------------
// Free clock helpers — Pa-style globals folded under cardinal::core.
// GetUtc32/64 return seconds since the Unix epoch (PA convention).
// ---------------------------------------------------------------------------
[[nodiscard]] u32 get_utc_32() noexcept;   // truncates to u32 (good until 2106)
[[nodiscard]] u64 get_utc_64() noexcept;

// ---------------------------------------------------------------------------
// Time — calendar-broken-down wall clock.
// ---------------------------------------------------------------------------
class Time {
public:
    Time() noexcept;       // initialised to current local time
    Time(u16 year, u16 month, u16 day, u16 hour, u16 minute, u16 second) noexcept;
    explicit Time(u64 utc_seconds) noexcept;

    [[nodiscard]] u16 year()         const noexcept { return year_; }
    [[nodiscard]] u16 month()        const noexcept { return month_; }
    [[nodiscard]] u16 day()          const noexcept { return day_; }
    [[nodiscard]] u16 hour()         const noexcept { return hour_; }
    [[nodiscard]] u16 minute()       const noexcept { return minute_; }
    [[nodiscard]] u16 second()       const noexcept { return second_; }
    [[nodiscard]] u16 millisecond()  const noexcept { return millisecond_; }
    [[nodiscard]] DayOfWeek day_of_week() const noexcept { return day_of_week_; }
    [[nodiscard]] u64 utc_seconds()  const noexcept;   // recomputed from fields

    void reset() noexcept;     // set to current local time
    void set(u16 year, u16 month, u16 day, u16 hour, u16 minute, u16 second) noexcept;
    void set_utc(u64 utc_seconds) noexcept;

    // Add N seconds to the current value.
    void add_seconds(u64 value) noexcept;

    // Add N days then overwrite the H:M:S to the given values. Useful for
    // "next 04:00 utc" cron-like schedules.
    void add_day_and_set_time(u16 add_days, u16 hour, u16 minute, u16 second) noexcept;

    // Difference in seconds (this - other).
    [[nodiscard]] i64 operator-(const Time& other) const noexcept;

    // Ordering — defined in terms of utc_seconds().
    [[nodiscard]] bool operator<=(const Time& other) const noexcept {
        return static_cast<i64>(utc_seconds()) <= static_cast<i64>(other.utc_seconds());
    }
    [[nodiscard]] bool operator<(const Time& other) const noexcept {
        return static_cast<i64>(utc_seconds()) <  static_cast<i64>(other.utc_seconds());
    }
    [[nodiscard]] bool operator==(const Time& other) const noexcept {
        return utc_seconds() == other.utc_seconds();
    }

    // Compute milliseconds until the given target time-of-day (day-of-week
    // wildcard variant). Returns 0 if the target is already past today.
    [[nodiscard]] u32 wait_milliseconds(DayOfWeek dow, u16 hour, u16 minute, u16 second) noexcept;
    [[nodiscard]] u32 wait_milliseconds(u16 hour, u16 minute, u16 second) noexcept;
    [[nodiscard]] u32 wait_milliseconds(u16 minute, u16 second) noexcept;

private:
    u16 year_;
    u16 month_;
    u16 day_;
    u16 hour_;
    u16 minute_;
    u16 second_;
    u16 millisecond_;
    DayOfWeek day_of_week_;
};

// ---------------------------------------------------------------------------
// CpuTime — snapshot of Idle/Kernel/User CPU time (FILETIME-equivalent).
// All values are in 100-ns ticks (Windows convention).
// ---------------------------------------------------------------------------
class CpuTime {
public:
    CpuTime() noexcept;

    [[nodiscard]] i32 reset() noexcept;   // refresh from OS; returns errno-style code (0 = ok)

    [[nodiscard]] u64 idle()   const noexcept { return idle_; }
    [[nodiscard]] u64 kernel() const noexcept { return kernel_; }
    [[nodiscard]] u64 user()   const noexcept { return user_; }

private:
    u64 idle_;
    u64 kernel_;
    u64 user_;
};

// ---------------------------------------------------------------------------
// CpuUsage — rolling busy-rate (%) based on CpuTime deltas. Keeps the last
// two samples in a small circular buffer and averages them — matches PA's
// 0..100 % output range.
// ---------------------------------------------------------------------------
class CpuUsage {
public:
    CpuUsage() noexcept;

    // Returns the current busy rate (0..100). Refreshes the sample buffer.
    [[nodiscard]] i32 reset_and_calculate_busy_rate() noexcept;

private:
    CpuTime current_;
    u64     last_idle_;
    u64     last_kernel_;
    u64     last_user_;
    i32     samples_[2];
    u32     sample_cursor_;
    u32     sample_count_;
};

}  // namespace cardinal::core
