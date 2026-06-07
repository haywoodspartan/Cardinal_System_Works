// =============================================================================
// Cardinal — function-event timeline ring.
// =============================================================================
#include <cardinal/trace/timeline.hpp>

#include <cardinal/core/std/algorithm.hpp>   // cardinal::min
#include <cardinal/core/std/cstring.hpp>     // cardinal::strncpy
// cardinal::chrono / cardinal::atomic arrive via timeline.hpp

#if defined(_WIN32)
#include <Windows.h>
#endif

namespace cardinal::trace {

Timeline::Timeline() : ring_(kCap), t0_(cardinal::chrono::steady_clock::now()) {}

Timeline& Timeline::instance() {
    static Timeline t;
    return t;
}

void Timeline::push(const char* name, const char* category, EventKind kind, u32 depth) {
    Event e{};
    e.t_ns = static_cast<u64>(cardinal::chrono::duration_cast<cardinal::chrono::nanoseconds>(
                cardinal::chrono::steady_clock::now() - t0_).count());
#if defined(_WIN32)
    e.thread_id = static_cast<u32>(GetCurrentThreadId());
#else
    e.thread_id = 0;
#endif
    e.depth = depth;
    e.kind  = kind;
    if (name) {
        cardinal::strncpy(e.name, name, sizeof(e.name) - 1);
    }
    if (category) {
        cardinal::strncpy(e.category, category, sizeof(e.category) - 1);
    }

    const u64 idx = write_index_.fetch_add(1, cardinal::memory_order_relaxed);
    ring_[idx % kCap] = e;
}

void Timeline::snapshot(cardinal::vector<Event>& out) const {
    const u64 written = write_index_.load(cardinal::memory_order_relaxed);
    const u64 count   = cardinal::min<u64>(written, kCap);
    out.resize(static_cast<size_t>(count));
    if (count == 0) return;
    const u64 start = (written - count) % kCap;
    for (u64 i = 0; i < count; ++i) {
        out[static_cast<size_t>(i)] = ring_[(start + i) % kCap];
    }
}

void Timeline::clear() {
    write_index_.store(0, cardinal::memory_order_relaxed);
}

}  // namespace cardinal::trace
