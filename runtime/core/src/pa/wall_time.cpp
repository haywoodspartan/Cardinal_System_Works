#include <cardinal/core/pa/wall_time.hpp>

#include <ctime>
#include <chrono>

#if CARDINAL_PLATFORM_WINDOWS
#include <Windows.h>
#endif

namespace cardinal::core::pa {

namespace {

#if CARDINAL_PLATFORM_WINDOWS
inline u64 filetime_to_u64(const FILETIME& ft) noexcept {
    return (static_cast<u64>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
}
#endif

// Days-in-month, accounting for leap years. Used by add_day_and_set_time.
inline bool is_leap(u16 y) noexcept { return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0); }
inline u16  days_in_month(u16 y, u16 m) noexcept {
    constexpr u16 d[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (m < 1 || m > 12) return 0;
    if (m == 2 && is_leap(y)) return 29;
    return d[m - 1];
}

// Zeller's-style 0..6 (Sun..Sat) day-of-week from y/m/d.
inline u16 day_of_week_of(u16 y, u16 m, u16 d) noexcept {
    if (m < 3) { m += 12; --y; }
    const u32 k = y % 100;
    const u32 j = y / 100;
    const u32 h = (d + (13 * (m + 1)) / 5 + k + k / 4 + j / 4 + 5 * j) % 7;
    // Zeller h: 0=Sat, 1=Sun, ... shift to 0=Sun convention
    return static_cast<u16>((h + 6) % 7);
}

}  // namespace

u32 get_utc_32() noexcept { return static_cast<u32>(get_utc_64()); }

u64 get_utc_64() noexcept {
    using namespace std::chrono;
    return static_cast<u64>(duration_cast<seconds>(system_clock::now().time_since_epoch()).count());
}

// ---------------------------------------------------------------------------

Time::Time() noexcept { reset(); }

Time::Time(u16 year, u16 month, u16 day, u16 hour, u16 minute, u16 second) noexcept {
    set(year, month, day, hour, minute, second);
}

Time::Time(u64 utc_seconds) noexcept { set_utc(utc_seconds); }

void Time::reset() noexcept {
#if CARDINAL_PLATFORM_WINDOWS
    SYSTEMTIME st{};
    ::GetLocalTime(&st);
    year_         = st.wYear;
    month_        = st.wMonth;
    day_          = st.wDay;
    hour_         = st.wHour;
    minute_       = st.wMinute;
    second_       = st.wSecond;
    millisecond_  = st.wMilliseconds;
    day_of_week_  = static_cast<DayOfWeek>(st.wDayOfWeek);
#else
    using namespace std::chrono;
    const auto now = system_clock::now();
    const std::time_t t = system_clock::to_time_t(now);
    std::tm lt{};
    #if defined(_MSC_VER)
        ::localtime_s(&lt, &t);
    #else
        ::localtime_r(&t, &lt);
    #endif
    year_         = static_cast<u16>(lt.tm_year + 1900);
    month_        = static_cast<u16>(lt.tm_mon + 1);
    day_          = static_cast<u16>(lt.tm_mday);
    hour_         = static_cast<u16>(lt.tm_hour);
    minute_       = static_cast<u16>(lt.tm_min);
    second_       = static_cast<u16>(lt.tm_sec);
    millisecond_  = static_cast<u16>(duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000);
    day_of_week_  = static_cast<DayOfWeek>(lt.tm_wday);
#endif
}

void Time::set(u16 y, u16 m, u16 d, u16 h, u16 mi, u16 s) noexcept {
    year_        = y;
    month_       = m;
    day_         = d;
    hour_        = h;
    minute_      = mi;
    second_      = s;
    millisecond_ = 0;
    day_of_week_ = static_cast<DayOfWeek>(day_of_week_of(y, m, d));
}

void Time::set_utc(u64 utc_seconds) noexcept {
    std::time_t t = static_cast<std::time_t>(utc_seconds);
    std::tm lt{};
#if defined(_MSC_VER)
    ::localtime_s(&lt, &t);
#else
    ::localtime_r(&t, &lt);
#endif
    year_         = static_cast<u16>(lt.tm_year + 1900);
    month_        = static_cast<u16>(lt.tm_mon + 1);
    day_          = static_cast<u16>(lt.tm_mday);
    hour_         = static_cast<u16>(lt.tm_hour);
    minute_       = static_cast<u16>(lt.tm_min);
    second_       = static_cast<u16>(lt.tm_sec);
    millisecond_  = 0;
    day_of_week_  = static_cast<DayOfWeek>(lt.tm_wday);
}

u64 Time::utc_seconds() const noexcept {
    std::tm lt{};
    lt.tm_year = year_ - 1900;
    lt.tm_mon  = month_ - 1;
    lt.tm_mday = day_;
    lt.tm_hour = hour_;
    lt.tm_min  = minute_;
    lt.tm_sec  = second_;
    return static_cast<u64>(std::mktime(&lt));
}

void Time::add_seconds(u64 value) noexcept {
    set_utc(utc_seconds() + value);
}

void Time::add_day_and_set_time(u16 add_days, u16 h, u16 mi, u16 s) noexcept {
    u16 d = day_;
    u16 m = month_;
    u16 y = year_;
    u32 remaining = add_days;
    while (remaining > 0) {
        const u16 mdays = days_in_month(y, m);
        if (d + remaining <= mdays) { d = static_cast<u16>(d + remaining); remaining = 0; }
        else {
            remaining -= (mdays - d + 1u);
            d = 1;
            if (++m > 12) { m = 1; ++y; }
        }
    }
    set(y, m, d, h, mi, s);
}

i64 Time::operator-(const Time& other) const noexcept {
    return static_cast<i64>(utc_seconds()) - static_cast<i64>(other.utc_seconds());
}

u32 Time::wait_milliseconds(DayOfWeek dow, u16 h, u16 mi, u16 s) noexcept {
    Time now;
    Time target = now;
    target.set(now.year_, now.month_, now.day_, h, mi, s);
    // Advance days until the target's day-of-week matches.
    u16 dadd = 0;
    while (target.day_of_week() != dow || target <= now) {
        ++dadd;
        target = now;
        target.add_day_and_set_time(dadd, h, mi, s);
        if (dadd > 7) break;
    }
    const i64 diff = target - now;
    if (diff <= 0) return 0;
    return static_cast<u32>(diff * 1000);
}

u32 Time::wait_milliseconds(u16 h, u16 mi, u16 s) noexcept {
    Time now;
    Time target = now;
    target.set(now.year_, now.month_, now.day_, h, mi, s);
    if (target <= now) target.add_day_and_set_time(1, h, mi, s);
    const i64 diff = target - now;
    if (diff <= 0) return 0;
    return static_cast<u32>(diff * 1000);
}

u32 Time::wait_milliseconds(u16 mi, u16 s) noexcept {
    Time now;
    return wait_milliseconds(now.hour(), mi, s);
}

// ---------------------------------------------------------------------------

CpuTime::CpuTime() noexcept : idle_(0), kernel_(0), user_(0) {}

i32 CpuTime::reset() noexcept {
#if CARDINAL_PLATFORM_WINDOWS
    FILETIME idle{}, kernel{}, user{};
    if (!::GetSystemTimes(&idle, &kernel, &user)) {
        return static_cast<i32>(::GetLastError());
    }
    idle_   = filetime_to_u64(idle);
    kernel_ = filetime_to_u64(kernel);
    user_   = filetime_to_u64(user);
    return 0;
#else
    idle_ = kernel_ = user_ = 0;
    return 0;
#endif
}

// ---------------------------------------------------------------------------

CpuUsage::CpuUsage() noexcept
    : last_idle_(0), last_kernel_(0), last_user_(0)
    , sample_cursor_(0), sample_count_(0) {
    samples_[0] = samples_[1] = 0;
}

i32 CpuUsage::reset_and_calculate_busy_rate() noexcept {
    (void)current_.reset();
    const u64 idle   = current_.idle();
    const u64 kernel = current_.kernel();
    const u64 user   = current_.user();

    i32 busy_rate = 0;
    if (last_kernel_ != 0 || last_user_ != 0) {
        const u64 d_idle   = idle   - last_idle_;
        const u64 d_kernel = kernel - last_kernel_;
        const u64 d_user   = user   - last_user_;
        const u64 d_total  = d_kernel + d_user;     // d_kernel already includes idle on Win
        if (d_total > 0) {
            const u64 busy = (d_total > d_idle) ? (d_total - d_idle) : 0;
            busy_rate = static_cast<i32>((busy * 100u) / d_total);
            if (busy_rate < 0)   busy_rate = 0;
            if (busy_rate > 100) busy_rate = 100;
        }
    }
    last_idle_   = idle;
    last_kernel_ = kernel;
    last_user_   = user;

    samples_[sample_cursor_] = busy_rate;
    sample_cursor_ = (sample_cursor_ + 1u) % 2u;
    if (sample_count_ < 2u) ++sample_count_;

    i32 sum = 0;
    for (u32 i = 0; i < sample_count_; ++i) sum += samples_[i];
    return (sample_count_ > 0) ? (sum / static_cast<i32>(sample_count_)) : 0;
}

}  // namespace cardinal::core::pa
