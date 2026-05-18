#pragma once

// =============================================================================
// Cardinal — IoDispatcher.
//
// One process-wide dispatcher coalesces async file I/O requests + executes
// them on the worker pool (cardinal::async). Patterns inspired by UE5's
// IoDispatcher / Bungie's tile streamer:
//
//   - Priority queue (Critical / High / Normal / Low / Background)
//   - Per-priority concurrency caps (no more than N Critical in flight)
//   - Batching: group requests for the same source file into one open+read
//   - Per-request callback fires on the worker; integrator marshals to main
//   - Live stats (bytes in flight, queue depths, throughput)
//
// Use this for streamed assets (textures, audio, sequences) — the cooked
// .pack file's load_async() goes through here too.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace cardinal::io {

// ---------------------------------------------------------------------------
// Priority — strictly ordered.
// ---------------------------------------------------------------------------
enum class Priority : u32 {
    Background = 0,    // background prefetch, gameplay can wait
    Low        = 1,
    Normal     = 2,
    High       = 3,
    Critical   = 4,    // blocking gameplay; cut to head of queue
};

const char* priority_name(Priority p) noexcept;

// ---------------------------------------------------------------------------
// Request — what the caller hands the dispatcher.
// ---------------------------------------------------------------------------
using RequestHandle = u64;

struct Request {
    std::string  path;        // absolute file path
    u64          offset{0};   // file offset in bytes (0 = start)
    u64          size  {0};   // bytes to read; 0 = entire file from offset
    Priority     priority{Priority::Normal};
    // Tag for grouping / cancellation. Multiple requests sharing a tag
    // can be cancelled together (cancel(tag)).
    u64          tag{0};
    // Callback fires on a worker thread when the read completes (or fails).
    // bytes is empty on failure.
    std::function<void(const std::vector<u8>& bytes)> on_done;
};

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------
struct DispatcherDesc {
    u32 max_concurrent_critical {2};
    u32 max_concurrent_high     {4};
    u32 max_concurrent_normal   {8};
    u32 max_concurrent_low      {4};
    u32 max_concurrent_background{2};
    // Total in-flight cap (sum across priorities). 0 = unbounded.
    u32 max_concurrent_total    {16};
};

struct DispatcherStats {
    u32 in_flight_total {0};
    u32 in_flight[5]    {};      // by priority
    u32 queued_total    {0};
    u32 queued[5]       {};
    u64 requests_seen   {0};
    u64 requests_completed{0};
    u64 requests_failed {0};
    u64 bytes_in_flight {0};
    u64 bytes_completed {0};
    f64 last_tick_ms    {0.0};
};

class Dispatcher {
public:
    static std::shared_ptr<Dispatcher> create(const DispatcherDesc& desc = {});
    ~Dispatcher();
    Dispatcher(const Dispatcher&) = delete;
    Dispatcher& operator=(const Dispatcher&) = delete;

    // Submit a request — returns a handle the caller can cancel.
    RequestHandle submit(Request req);

    // Convenience that returns a future for the bytes (in addition to the
    // optional on_done callback).
    std::shared_future<std::vector<u8>> submit_and_future(Request req);

    // Cancel by handle or by tag. Returns count cancelled. In-flight reads
    // can't be aborted mid-fread; we mark them so the on_done isn't fired.
    u32 cancel(RequestHandle h);
    u32 cancel_tag(u64 tag);

    // Drive scheduling. Cheap; called once per frame from the main loop.
    // Pumps queued requests onto the async pool up to per-priority caps,
    // collects completions, fires on_done callbacks.
    void tick();

    DispatcherStats stats() const noexcept;

    // Forward-declared opaque impl, kept public so anonymous-namespace
    // helpers in io.cpp can take Impl* in their signatures.
    struct Impl;

private:
    Dispatcher() = default;
    bool initialize_(const DispatcherDesc& desc);

    Impl* impl_{nullptr};
};

}  // namespace cardinal::io
