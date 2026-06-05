#pragma once

// =============================================================================
// Cardinal core — Stopwatch + RepeatableTimer — modern C++20 port of
// Pearl Abyss PaTimer.h.
//
// Stopwatch — std::chrono::steady_clock-backed elapsed-time meter; the
// minimal "QueryPerformanceCounter + diff" pattern most Pa call sites want.
// Coexists with cardinal::core::time (the mono-clock surface backing
// FrameScope / FramePacer).
//
// RepeatableTimer<TimerId, Tick> — priority-queue interval timer; same
// semantics as the original: register N entries with (delay, interval),
// end_register() seals start times, wait_milliseconds() returns the time
// until the next event and pops any already-elapsed events into the caller's
// timeout list. Tick type is parameterised — defaults to u64 milliseconds
// pulled from steady_clock.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/platform.hpp>
#include <cardinal/core/containers.hpp>   // cardinal::vector

#include <algorithm>   // std::push_heap / pop_heap

#include <chrono>      // std::chrono::steady_clock

namespace cardinal::core {

// ---------------------------------------------------------------------------
// Stopwatch — elapsed-time counter (steady_clock).
// ---------------------------------------------------------------------------
class Stopwatch {
public:
    Stopwatch() noexcept : start_(std::chrono::steady_clock::now()) {}

    void   restart() noexcept { start_ = std::chrono::steady_clock::now(); }

    [[nodiscard]] u64 elapsed_ms() const noexcept {
        using namespace std::chrono;
        return static_cast<u64>(duration_cast<milliseconds>(steady_clock::now() - start_).count());
    }
    [[nodiscard]] u64 elapsed_us() const noexcept {
        using namespace std::chrono;
        return static_cast<u64>(duration_cast<microseconds>(steady_clock::now() - start_).count());
    }
    [[nodiscard]] u64 elapsed_ns() const noexcept {
        using namespace std::chrono;
        return static_cast<u64>(duration_cast<nanoseconds>(steady_clock::now() - start_).count());
    }

private:
    std::chrono::steady_clock::time_point start_;
};

// ---------------------------------------------------------------------------
// Default tick functor — milliseconds since process start (steady_clock).
// Templated so RepeatableTimer can be re-targeted onto a virtual clock
// (server simulation tick, replay-driven tick) by injecting a different
// TickCountFunctor — matches the original Pa template knob.
// ---------------------------------------------------------------------------
struct DefaultTickCountFunctor {
    [[nodiscard]] u64 operator()() const noexcept {
        using namespace std::chrono;
        return static_cast<u64>(
            duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count());
    }
};

// ---------------------------------------------------------------------------
// RepeatableTimer — N-entry interval timer over a configurable clock.
// Maintains a min-heap keyed by absolute timeout tick; wait_milliseconds()
// is O(log N) amortised.
// ---------------------------------------------------------------------------
template <class TimerId = u32, class Tick = u64,
          class TickCountFunctor = DefaultTickCountFunctor>
class RepeatableTimer {
public:
    struct Entry {
        TimerId id;
        Tick    delay;          // initial delay in ticks
        Tick    interval;       // repeat interval in ticks (0 = one-shot)
        Tick    timeout_tick;   // absolute tick when this entry fires next

        // Min-heap predicate — earliest-firing first (std::*_heap is a
        // max-heap by default, so we invert the comparison).
        [[nodiscard]] bool operator<(const Entry& other) const noexcept {
            return timeout_tick > other.timeout_tick;
        }
    };

    explicit RepeatableTimer(TickCountFunctor tcf = TickCountFunctor()) noexcept
        : tick_(tcf), is_register_end_(false) {}

    RepeatableTimer(const RepeatableTimer&)            = delete;
    RepeatableTimer& operator=(const RepeatableTimer&) = delete;

    // Register an entry. ERROR_ALREADY_EXISTS = 183 (Win) if a duplicate id
    // is rejected by `prevent_duplicate`.
    [[nodiscard]] i32 register_entry(TimerId id, u32 delay_ms, u32 interval_ms,
                                     bool prevent_duplicate = true) noexcept {
        if (prevent_duplicate) {
            for (const auto& e : list_) if (e.id == id) return 183;  // ERROR_ALREADY_EXISTS
        }
        const Tick cur = tick_();
        list_.push_back(Entry{id,
                              static_cast<Tick>(delay_ms),
                              static_cast<Tick>(interval_ms),
                              static_cast<Tick>(cur + delay_ms)});
        std::push_heap(list_.begin(), list_.end());
        return 0;
    }

    // Re-stamp every entry's timeout tick from "now + delay" — call once
    // after the last register_entry to align the start times.
    void end_register() noexcept {
        const Tick cur = tick_();
        for (auto& e : list_) e.timeout_tick = cur + e.delay;
        std::make_heap(list_.begin(), list_.end());
        is_register_end_ = true;
    }

    void clear() noexcept { list_.clear(); is_register_end_ = false; }
    [[nodiscard]] usize size()  const noexcept { return list_.size(); }
    [[nodiscard]] bool  empty() const noexcept { return list_.empty(); }

    // Returns the time (in milliseconds) until the next pending event. Any
    // already-elapsed events are popped into `out_timeout_list` and (for
    // repeating entries) re-pushed with their next absolute tick.
    [[nodiscard]] u32 wait_milliseconds(cardinal::vector<TimerId>& out_timeout_list) noexcept {
        if (list_.empty()) return 0;
        const Tick cur = tick_();
        while (!list_.empty() && list_.front().timeout_tick <= cur) {
            Entry e = list_.front();
            std::pop_heap(list_.begin(), list_.end());
            list_.pop_back();

            out_timeout_list.push_back(e.id);

            if (e.interval > 0) {
                e.timeout_tick += e.interval;
                list_.push_back(e);
                std::push_heap(list_.begin(), list_.end());
            }
        }
        if (list_.empty()) return 0;
        const Tick remaining = list_.front().timeout_tick - cur;
        return static_cast<u32>(remaining);
    }

    [[nodiscard]] const cardinal::vector<Entry>& entries() const noexcept { return list_; }

private:
    cardinal::vector<Entry> list_;          // heap-ordered by timeout_tick (min)
    TickCountFunctor        tick_;
    bool                    is_register_end_;
};

}  // namespace cardinal::core
