// =============================================================================
// Cardinal — chrome://tracing JSON exporter implementation.
//
// Sample storage is a SoA-friendly fixed-cap ring. We don't grow on
// overflow — once full, we drop and log once. JSON serialisation is
// inline (no std::format / nlohmann::json dependency).
// =============================================================================
#include <cardinal/core/diag/trace_export.hpp>

#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/platform.hpp>

#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#if CARDINAL_PLATFORM_WINDOWS
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <Windows.h>
#endif

namespace cardinal::trace {

namespace {

struct Event {
    const char* name;     // FrameScope names are static string literals
    i64         start_ns;
    i64         end_ns;
    u32         tid;
};

// Tunables. ~64K events * 40 bytes ≈ 2.5 MiB worst-case in-memory.
constexpr usize kMaxEvents = 64u * 1024u;

std::mutex            g_mtx;
std::vector<Event>    g_events;
std::atomic<bool>     g_capturing{false};
u32                   g_frames_remaining{0};
// g_frames_total is read OUTSIDE g_mtx by note_frame_boundary's early-
// out at line 151 (per-frame hot path — locking each frame would dwarf
// the work in the not-capturing case). begin_capture writes it under
// the lock at line 54. Non-atomic concurrent access of those two
// (read unlocked / write locked) is a data race per the C++ memory
// model regardless of x86's atomic-aligned-u32 behaviour, so promote
// to std::atomic with relaxed ordering — the locked writers stay
// inside the lock, the unlocked reader gets a defined load. Same
// shape as the data-race-on-shared-state vein (io 82604f9, budget
// 033c642+c7c856d, audio a317126).
std::atomic<u32>      g_frames_total{0};
bool                  g_warned_full{false};

}  // namespace

void begin_capture(u32 frame_budget) {
    std::lock_guard<std::mutex> lg(g_mtx);
    g_events.clear();
    g_events.reserve(kMaxEvents);
    g_frames_remaining = frame_budget;
    g_frames_total.store(frame_budget, std::memory_order_relaxed);
    g_warned_full      = false;
    g_capturing.store(true, std::memory_order_release);
    cardinal::log::infof("trace",
        "capture started (frame_budget=%s)",
        frame_budget == 0 ? "until end_capture" :
            (std::to_string(frame_budget) + " frames").c_str());
}

bool end_capture(const std::string& output_path) {
    if (!g_capturing.exchange(false, std::memory_order_acq_rel)) return false;

    std::lock_guard<std::mutex> lg(g_mtx);

    FILE* fp = std::fopen(output_path.c_str(), "wb");
    if (fp == nullptr) {
        cardinal::log::errorf("trace",
            "end_capture: cannot open %s for writing", output_path.c_str());
        g_events.clear();
        return false;
    }

    // Chrome trace format: top-level array of event objects, each with
    // ph, name, ts (microseconds), dur, pid, tid. Pretty-printed for
    // human inspection; perfetto / chrome accept either.
#if CARDINAL_PLATFORM_WINDOWS
    const u32 pid = static_cast<u32>(GetCurrentProcessId());
#else
    const u32 pid = 0;
#endif

    std::fputc('[', fp);
    bool first = true;
    for (const auto& e : g_events) {
        if (!first) std::fputc(',', fp);
        first = false;
        const i64 ts_us  = e.start_ns / 1000;
        const i64 dur_us = (e.end_ns - e.start_ns) / 1000;
        std::fprintf(fp,
            "\n  {\"ph\":\"X\",\"name\":\"%s\",\"ts\":%lld,\"dur\":%lld,"
            "\"pid\":%u,\"tid\":%u}",
            e.name ? e.name : "(null)",
            static_cast<long long>(ts_us),
            static_cast<long long>(dur_us > 0 ? dur_us : 1),
            pid, e.tid);
    }
    std::fputc('\n', fp);
    std::fputc(']', fp);
    std::fclose(fp);

    cardinal::log::infof("trace",
        "wrote %zu events to %s (%u frames, %s)",
        g_events.size(), output_path.c_str(),
        g_frames_total.load(std::memory_order_relaxed),
        g_warned_full ? "BUFFER WAS FULL — output truncated" : "complete");
    g_events.clear();
    return true;
}

bool capturing() noexcept { return g_capturing.load(std::memory_order_acquire); }

FrameCapture::FrameCapture(std::string dump_path) noexcept
    : path_(std::move(dump_path)), was_capturing_(capturing())
{
    if (!was_capturing_) begin_capture(/*frame_budget*/ 1);
}

FrameCapture::~FrameCapture() {
    if (was_capturing_) return;        // someone else owns the capture
    if (path_.empty()) {
        // Drop the data — caller wanted just to scope-isolate but not save.
        g_capturing.store(false, std::memory_order_release);
        std::lock_guard<std::mutex> lg(g_mtx);
        g_events.clear();
    } else {
        end_capture(path_);
    }
}

namespace detail {

void record_scope(const char* name, i64 start_ns, i64 end_ns, u32 tid) noexcept {
    if (!g_capturing.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lg(g_mtx);
    if (g_events.size() >= kMaxEvents) {
        if (!g_warned_full) {
            cardinal::log::warnf("trace",
                "event buffer full (%zu) — further samples dropped this capture",
                kMaxEvents);
            g_warned_full = true;
        }
        return;
    }
    g_events.push_back(Event{name, start_ns, end_ns, tid});
}

void note_frame_boundary() noexcept {
    if (!g_capturing.load(std::memory_order_acquire)) return;
    // Unlocked relaxed load — paired with the locked store in
    // begin_capture (line 54). Sufficient: the only sequence we need
    // to preserve is "if g_capturing is true and g_frames_total has
    // been observed > 0, then the locked block sees a coherent
    // g_frames_remaining" — which is guaranteed because g_capturing
    // is acquire/release-ordered and the lock then serialises every
    // remaining access.
    if (g_frames_total.load(std::memory_order_relaxed) == 0) return;
    std::lock_guard<std::mutex> lg(g_mtx);
    if (g_frames_remaining == 0) return;
    if (--g_frames_remaining == 0) {
        // Auto-stop. Caller still has to call end_capture(path) to write.
        // We don't write here because we don't know where to put it.
        cardinal::log::infof("trace",
            "frame budget reached (%u frames, %zu events) — call end_capture(\"path.json\")",
            g_frames_total.load(std::memory_order_relaxed), g_events.size());
    }
}

}  // namespace detail

}  // namespace cardinal::trace
