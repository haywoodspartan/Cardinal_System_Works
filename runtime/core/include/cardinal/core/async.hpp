#pragma once

// =============================================================================
// Cardinal — high-level concurrency primitives on top of cardinal::JobSystem.
//
// JobSystem (jobs.hpp) is the low-level fiber scheduler. The patterns most
// engine code actually wants are:
//
//   parallel_for(N, [&](u32 i){ work(i); });          — split a range
//   parallel_for_each(vec, [&](T& x){ work(x); });    — split a container
//   auto f = async([]{ return load_thing(); });       — fire-and-forget
//   f.then([](auto v){ ... });                        — continuation
//
// Plus a `FramePhase` scheduler that tags work into named buckets
// (Animation, Physics, Render Prep, Async I/O) and prints per-phase
// timing into the profiler. The profiler panel reads its data from here.
//
// Everything below is THIN — the heavy lifting is in JobSystem. This header
// is just ergonomics + bookkeeping.
// =============================================================================

#include <cardinal/core/jobs.hpp>
#include <cardinal/core/types.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cardinal::async {

// ---------------------------------------------------------------------------
// Process-wide pool — set once at boot, queried by parallel_for / async.
//
// Keeping the pool a global-ish singleton lets every subsystem submit work
// without plumbing a JobSystem* through every call site. The host owns the
// JobSystem instance; we just borrow it.
// ---------------------------------------------------------------------------
void          set_pool(JobSystem* js) noexcept;
JobSystem*    pool() noexcept;

// ---------------------------------------------------------------------------
// parallel_for(begin, end, fn[, grain])
//
//   Splits [begin, end) into chunks of `grain` and submits one job per
//   chunk. Blocks the caller until all chunks are done. `fn(i)` is called
//   for each i in [begin, end).
//
//   `grain` defaults to a value derived from worker_count + range size:
//   we don't want millions of tiny jobs, but we do want every worker to
//   get something. The default keeps total job count near ~4x worker count.
//
// Safe to call from a worker fiber (will yield on wait) OR from an
// external thread (busy-wait via JobSystem::wait).
// ---------------------------------------------------------------------------
void parallel_for(u32 begin, u32 end,
                  const std::function<void(u32)>& fn,
                  u32 grain = 0);

// ---------------------------------------------------------------------------
// parallel_invoke({task1, task2, task3, ...}) — fork-join over a small set
// of independent callables.
//
// Submits each task as its own job, waits for all of them to finish, then
// returns. Use for per-frame phases where a handful of subsystems can run
// concurrently (particles + sky + ai_perception + audio listener update +
// mass entity drift, etc.) — overhead is one heap-allocated closure per
// task, dwarfed by the work.
//
// Two overloads:
//   - initializer_list  : compile-time list, ergonomic for fixed sets
//   - span / vector     : runtime-built list, useful when you accumulate
//                         tasks in a loop (SimWorld handlers, e.g.).
//
// Falls back to a sequential for-loop when no pool is bound. Calls in
// declaration order in that case so single-threaded behaviour is
// deterministic.
//
// Safe to call from any thread; from a worker fiber it yields on wait,
// from an external thread it busy-waits.
// ---------------------------------------------------------------------------
void parallel_invoke(std::initializer_list<std::function<void()>> tasks);
void parallel_invoke(const std::vector<std::function<void()>>& tasks);

// ---------------------------------------------------------------------------
// parallel_for_each(container, fn) — convenience over parallel_for.
// Falls back to a serial loop if no pool is bound (cheap for tests).
// ---------------------------------------------------------------------------
template <class Container, class Fn>
void parallel_for_each(Container& c, Fn&& fn) {
    const u32 n = static_cast<u32>(c.size());
    if (n == 0) return;
    if (pool() == nullptr) {
        for (auto& x : c) fn(x);
        return;
    }
    std::vector<typename Container::value_type*> ptrs;
    ptrs.reserve(n);
    for (auto& x : c) ptrs.push_back(&x);
    parallel_for(0u, n, [&](u32 i) { fn(*ptrs[i]); });
}

// ---------------------------------------------------------------------------
// Future<T> — minimal continuation-style future.
//
// Built on a heap-allocated control block. Construction is via async() —
// the caller never builds one directly. Destructor is trivial (shared_ptr
// cleans up).
//
// Only one continuation per future. wait() returns the value (or default
// if cancelled / not ready and called when no value will ever arrive).
//
// Attach (then) vs. resolve (the async worker) serialise on State::mtx so
// the {done, on_done} hand-off is atomic: whichever side wins the lock,
// the continuation fires exactly once with the resolved value — no lost
// wakeup if the worker resolves in the window between then()'s done-check
// and its on_done store. `done` stays atomic so ready()/wait() keep their
// cheap lock-free poll; the mutex only guards the one-shot transition and
// is uncontended in the common case (this is asset I/O, not a hot loop).
// ---------------------------------------------------------------------------
template <class T>
class Future {
public:
    struct State {
        std::mutex                    mtx;
        std::atomic<bool>             done{false};
        T                             value{};
        std::function<void(const T&)> on_done{};
    };

    Future() = default;
    explicit Future(std::shared_ptr<State> s) : state_(std::move(s)) {}

    bool ready() const noexcept {
        return state_ && state_->done.load(std::memory_order_acquire);
    }

    // Spin-wait — uses std::this_thread::yield. For long-running work
    // prefer .then() so the calling thread can keep doing other things.
    T wait() const {
        while (state_ && !state_->done.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        return state_ ? state_->value : T{};
    }

    // Run `cb(value)` once the future resolves. If already resolved, fires
    // immediately on the calling thread; otherwise it is handed to the
    // resolver. The lock makes the check-then-store atomic against the
    // worker's resolve so the callback can't be dropped in the gap.
    void then(std::function<void(const T&)> cb) {
        if (!state_) return;
        {
            std::lock_guard<std::mutex> lk(state_->mtx);
            if (!state_->done.load(std::memory_order_acquire)) {
                state_->on_done = std::move(cb);
                return;                       // resolver will fire it
            }
        }
        // Already resolved: value is immutable now (set under the same
        // lock before done flipped). Fire outside the lock so user code
        // never runs with State::mtx held.
        cb(state_->value);
    }

private:
    std::shared_ptr<State> state_;
};

// async(fn) — submit fn() to the pool, return a Future for its result.
// fn must be invocable with no args and return T.
template <class Fn>
auto async(Fn&& fn) -> Future<decltype(fn())> {
    using T = decltype(fn());
    auto state = std::make_shared<typename Future<T>::State>();
    auto* js   = pool();
    auto runnable = new std::function<void()>(
        [state, fn = std::forward<Fn>(fn)]() mutable {
            // Run the work OUTSIDE the lock — it's the whole point of
            // async(); only the publish + continuation hand-off is guarded.
            T result = fn();
            std::function<void(const T&)> cb;
            {
                std::lock_guard<std::mutex> lk(state->mtx);
                state->value = std::move(result);
                state->done.store(true, std::memory_order_release);
                cb = std::move(state->on_done);   // take ownership; ...
                state->on_done = nullptr;         // ... never fire it twice
            }
            // then() that raced us either stored its cb (we run it here)
            // or lost and will see done==true and fire itself. Exactly one
            // path runs the continuation. Run it with the lock released.
            if (cb) cb(state->value);
        });
    if (js != nullptr) {
        js->submit(
            [](void* user) {
                auto* f = static_cast<std::function<void()>*>(user);
                (*f)();
                delete f;
            },
            runnable);
    } else {
        // No pool — fall back to running inline so tests still work.
        (*runnable)();
        delete runnable;
    }
    return Future<T>(state);
}

// ---------------------------------------------------------------------------
// FramePhase — named timing buckets. The profiler panel surfaces these.
//
// Usage:
//   { FrameScope s("Physics"); update_physics(); }
//   { FrameScope s("Render Prep"); ... }
//
// At end-of-frame, frame_advance() rolls the per-phase running totals into
// a circular history (last N frames) so the panel can plot them.
// ---------------------------------------------------------------------------
class FrameScope {
public:
    explicit FrameScope(const char* name) noexcept;
    ~FrameScope() noexcept;
private:
    const char*                                            name_;
    std::chrono::high_resolution_clock::time_point         t0_;
};

void  frame_advance() noexcept;          // call once per frame
u32   frame_history_size() noexcept;     // ~120 frames

struct PhaseSample {
    std::string name;
    double      ms_last{0.0};
    double      ms_avg {0.0};   // EMA over last ~60 frames
    double      ms_max {0.0};   // peak over the same window
    u32         calls_last{0};  // how many FrameScope() events fired
};
std::vector<PhaseSample> phase_samples();

// ---------------------------------------------------------------------------
// Per-thread CPU sampler. Each registered worker reports its busy fraction
// (work time / wall time over the last sample window). The profiler panel
// renders these as a bar chart so the user can see imbalanced workers.
// ---------------------------------------------------------------------------
// Tier mirrors cardinal::WorkerTier 1:1 (kept as a separate enum so async.hpp
// doesn't have to depend on jobs.hpp; the cast happens at the registration
// site). Performance = strong physical core (P-core / V-Cache CCD), General
// = other physical core, Background = SMT sibling or Intel E-core.
enum class WorkerTier : u8 {
    Performance = 0,
    General     = 1,
    Background  = 2,
};

struct WorkerStats {
    u32         worker_id{0};
    u32         logical_core{0};
    u32         numa_node{0};
    WorkerTier  tier{WorkerTier::Performance};
    double      busy_fraction{0.0};   // 0..1
    u64         jobs_executed{0};
    u64         bytes_stolen{0};      // count of jobs taken from another worker
    // Convenience — true iff tier == Performance. Kept for back-compat with
    // panels written against the prior 2-tier API.
    bool        is_perf_core{false};
};
std::vector<WorkerStats> worker_stats();

// Internal — workers call these on enter / exit / steal.
namespace detail {
    void note_worker_run_begin (u32 worker_id) noexcept;
    void note_worker_run_end   (u32 worker_id) noexcept;
    void note_worker_steal     (u32 worker_id) noexcept;
    void register_worker_topology(u32 worker_id, u32 logical_core,
                                  WorkerTier tier, u32 numa_node) noexcept;
}

}  // namespace cardinal::async
