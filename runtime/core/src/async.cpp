#include <cardinal/core/async.hpp>
#include <cardinal/core/trace_export.hpp>

#include <algorithm>
#include <atomic>
#include <iterator>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace cardinal::async {

namespace {

// Set once at boot via set_pool(); read by every parallel_for and async()
// call thereafter. Pin to its own cache line + mark read-mostly so the
// hot per-frame queries don't get false-shared with neighbouring writes.
CARDINAL_READ_MOSTLY JobSystem* g_pool = nullptr;

// FramePhase storage — single mutex guards the running map + history.
struct PhaseAccum {
    std::string name;
    double      sum_ms{0.0};
    u32         calls{0};
    double      ema_ms{0.0};
    double      max_window_ms{0.0};
};
std::mutex                                  g_phase_mtx;
std::unordered_map<std::string, PhaseAccum> g_phase_running;     // current frame
std::unordered_map<std::string, PhaseAccum> g_phase_published;   // last frame's snapshot

// Worker stats — atomic-only access (no lock needed for the hot path).
struct WorkerAccum {
    std::atomic<u64> jobs{0};
    std::atomic<u64> steals{0};
    std::atomic<i64> busy_ns{0};
    std::atomic<i64> last_run_begin{0};
    u32              logical_core{0};
    u32              numa_node{0};
    WorkerTier       tier{WorkerTier::Performance};
};
constexpr u32         kMaxWorkers = 64;
WorkerAccum           g_workers[kMaxWorkers]{};
std::atomic<u32>      g_worker_max_id{0};
std::atomic<i64>      g_window_start_ns{0};

i64 now_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

void set_pool(JobSystem* js) noexcept { g_pool = js; }
JobSystem* pool() noexcept            { return g_pool; }

// ---------------------------------------------------------------------------
// parallel_for
// ---------------------------------------------------------------------------
namespace {

struct ChunkArg {
    u32                              begin;
    u32                              end;
    const std::function<void(u32)>*  fn;
};

void chunk_runner(void* user) {
    auto* a = static_cast<ChunkArg*>(user);
    // Mark this worker as busy for the duration of the chunk so the
    // Profiler panel's per-worker bars reflect real parallel_for activity.
    const u32 wid = JobSystem::current_worker_id();
    if (wid != static_cast<u32>(-1)) detail::note_worker_run_begin(wid);
    for (u32 i = a->begin; i < a->end; ++i) (*a->fn)(i);
    if (wid != static_cast<u32>(-1)) detail::note_worker_run_end(wid);
}

}  // namespace

// Below this many items, parallel_for inlines on the calling thread.
// Empirically tuned: dispatch + cross-thread wakeup + main-thread wait
// adds ~50-200us, so unless we're handing each worker > ~1 µs × kInlineThreshold
// of work the parallelism is net-negative. Callers that know their work is
// heavy per-item can pass a custom grain to override.
constexpr u32 kParallelForInlineThreshold = 32;

void parallel_for(u32 begin, u32 end,
                  const std::function<void(u32)>& fn,
                  u32 grain)
{
    if (begin >= end) return;
    const u32 n = end - begin;

    // Inline path: no pool, single item, or small-work fast path. Saves
    // a counter/inbox/wakeup roundtrip + main-thread wait spin. Callers
    // that genuinely want forced dispatch on small N can pass grain=1
    // to skip the inline threshold (we still skip when n <= 1).
    JobSystem* js = pool();
    const bool small_work = (grain == 0) && (n < kParallelForInlineThreshold);
    if (js == nullptr || n <= 1 || small_work) {
        for (u32 i = begin; i < end; ++i) fn(i);
        return;
    }

    // Pick a sensible grain if the caller didn't.
    if (grain == 0) {
        const u32 wc = std::max(1u, js->worker_count());
        // ~4 chunks per worker — minimum 1. Verified well-suited to the
        // renderer's per-EntityWork Phase 2 shape (typical N <= 128 on
        // 32-worker hosts → grain == 1, every work item is its own
        // chunk, work-stealing balances variable per-item cost).
        grain = std::max(1u, n / (wc * 4u));
    }

    const u32 chunk_count = (n + grain - 1) / grain;
    std::vector<ChunkArg> args(chunk_count);
    for (u32 c = 0; c < chunk_count; ++c) {
        const u32 cb = begin + c * grain;
        const u32 ce = std::min(end, cb + grain);
        args[c] = ChunkArg{ cb, ce, &fn };
    }

    Counter* ctr = js->acquire_counter();
    for (u32 c = 0; c < chunk_count; ++c) {
        js->submit(&chunk_runner, &args[c], ctr);
    }
    js->wait(ctr);
    js->release(ctr);
}

// ---------------------------------------------------------------------------
// parallel_invoke — fork-join over a small set of independent callables.
//
// Heap-allocates one std::function per task because JobFunc is a raw
// function pointer (no captures) and the callables come from the caller
// with their own state. The closures live until wait() returns, then
// we delete them. For per-frame N-task phases this is ~ a few hundred
// nanoseconds of overhead, dwarfed by the actual work in each task.
// ---------------------------------------------------------------------------
// Below this many tasks, parallel_invoke inlines on the calling thread.
// Same reasoning as kParallelForInlineThreshold: dispatch overhead exceeds
// the work for short fan-outs. Two- or three-task fan-outs of µs-scale
// callables (sky.tick + particles.tick + ai.tick, for instance) are
// almost always faster sequentially. Callers with genuinely heavy
// per-task work can build a longer task list to clear the threshold.
constexpr usize kParallelInvokeInlineThreshold = 4;

namespace {
// Shared kernel for both parallel_invoke overloads. Iterates the input
// once, heap-allocates one closure per task (caller's callable copied
// in), submits, waits, deletes. Returns immediately on empty input or
// when the task count is below the inline threshold.
template <class Iter>
void parallel_invoke_impl(Iter first, Iter last) {
    auto* js = pool();
    const usize n = static_cast<usize>(std::distance(first, last));
    if (js == nullptr || n < kParallelInvokeInlineThreshold) {
        for (auto it = first; it != last; ++it) (*it)();
        return;
    }

    struct Closure { std::function<void()> fn; };
    std::vector<Closure*> heap_closures;
    heap_closures.reserve(n);
    for (auto it = first; it != last; ++it) heap_closures.push_back(new Closure{*it});

    auto run = +[](void* user) {
        auto* c = static_cast<Closure*>(user);
        if (c->fn) c->fn();
    };

    Counter* ctr = js->acquire_counter();
    for (auto* c : heap_closures) js->submit(run, c, ctr);
    js->wait(ctr);
    js->release(ctr);

    for (auto* c : heap_closures) delete c;
}
}  // namespace

void parallel_invoke(std::initializer_list<std::function<void()>> tasks) {
    parallel_invoke_impl(tasks.begin(), tasks.end());
}
void parallel_invoke(const std::vector<std::function<void()>>& tasks) {
    parallel_invoke_impl(tasks.begin(), tasks.end());
}

// ---------------------------------------------------------------------------
// FrameScope
// ---------------------------------------------------------------------------
FrameScope::FrameScope(const char* name) noexcept
    : name_(name), t0_(std::chrono::high_resolution_clock::now()) {}

FrameScope::~FrameScope() noexcept {
    using clk = std::chrono::high_resolution_clock;
    const auto t1 = clk::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0_).count();

    // Mirror the sample into the chrome-tracing buffer when a capture
    // is active. The check is a single atomic load on the fast path
    // (no capture → returns immediately), so always-on profiling cost
    // is negligible.
    if (cardinal::trace::capturing()) {
        const i64 t0_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              t0_.time_since_epoch()).count();
        const i64 t1_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              t1.time_since_epoch()).count();
        const u32 wid   = JobSystem::current_worker_id();
        // Workers' tids start at 1 so the main thread (wid == u32(-1))
        // gets tid 0 in the trace, which renders as the leftmost lane.
        const u32 tid   = (wid == static_cast<u32>(-1)) ? 0u : (wid + 1u);
        cardinal::trace::detail::record_scope(name_, t0_ns, t1_ns, tid);
    }

    std::lock_guard<std::mutex> lg(g_phase_mtx);
    auto& acc = g_phase_running[name_];
    if (acc.name.empty()) acc.name = name_;
    acc.sum_ms += ms;
    acc.calls  += 1;
}

void frame_advance() noexcept {
    cardinal::trace::detail::note_frame_boundary();
    const double alpha = 0.10;     // ~10-frame EMA
    std::lock_guard<std::mutex> lg(g_phase_mtx);
    // Roll running -> published, applying EMA + windowed max.
    for (auto& [name, run] : g_phase_running) {
        auto& pub = g_phase_published[name];
        if (pub.name.empty()) pub.name = name;
        pub.sum_ms        = run.sum_ms;
        pub.calls         = run.calls;
        pub.ema_ms        = pub.ema_ms == 0.0
            ? run.sum_ms
            : (alpha * run.sum_ms + (1.0 - alpha) * pub.ema_ms);
        pub.max_window_ms = std::max(pub.max_window_ms * 0.99,   // slow decay
                                     run.sum_ms);
    }
    g_phase_running.clear();
}

u32 frame_history_size() noexcept { return 120; }

std::vector<PhaseSample> phase_samples() {
    std::vector<PhaseSample> out;
    std::lock_guard<std::mutex> lg(g_phase_mtx);
    out.reserve(g_phase_published.size());
    for (const auto& [name, pub] : g_phase_published) {
        PhaseSample s{};
        s.name        = name;
        s.ms_last     = pub.sum_ms;
        s.ms_avg      = pub.ema_ms;
        s.ms_max      = pub.max_window_ms;
        s.calls_last  = pub.calls;
        out.push_back(std::move(s));
    }
    std::sort(out.begin(), out.end(),
        [](const PhaseSample& a, const PhaseSample& b) {
            return a.ms_avg > b.ms_avg;
        });
    return out;
}

// ---------------------------------------------------------------------------
// Worker stats
// ---------------------------------------------------------------------------
namespace detail {

void register_worker_topology(u32 wid, u32 lcore, WorkerTier tier, u32 numa_node) noexcept {
    if (wid >= kMaxWorkers) return;
    g_workers[wid].logical_core = lcore;
    g_workers[wid].tier         = tier;
    g_workers[wid].numa_node    = numa_node;
    u32 prev = g_worker_max_id.load(std::memory_order_relaxed);
    while (wid > prev &&
           !g_worker_max_id.compare_exchange_weak(prev, wid)) { /* spin */ }
    if (g_window_start_ns.load(std::memory_order_relaxed) == 0) {
        g_window_start_ns.store(now_ns(), std::memory_order_relaxed);
    }
}

void note_worker_run_begin(u32 wid) noexcept {
    if (wid >= kMaxWorkers) return;
    g_workers[wid].last_run_begin.store(now_ns(), std::memory_order_relaxed);
}

void note_worker_run_end(u32 wid) noexcept {
    if (wid >= kMaxWorkers) return;
    const i64 t = g_workers[wid].last_run_begin.load(std::memory_order_relaxed);
    if (t > 0) {
        const i64 dt = now_ns() - t;
        if (dt > 0) g_workers[wid].busy_ns.fetch_add(dt, std::memory_order_relaxed);
        g_workers[wid].last_run_begin.store(0, std::memory_order_relaxed);
    }
    g_workers[wid].jobs.fetch_add(1, std::memory_order_relaxed);
}

void note_worker_steal(u32 wid) noexcept {
    if (wid >= kMaxWorkers) return;
    g_workers[wid].steals.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace detail

std::vector<WorkerStats> worker_stats() {
    const i64 now = now_ns();
    i64 win_start = g_window_start_ns.load(std::memory_order_relaxed);
    if (win_start == 0) win_start = now;

    const double window_s =
        std::max<double>(1e-6, static_cast<double>(now - win_start) * 1e-9);

    const u32 max_id = g_worker_max_id.load(std::memory_order_relaxed);
    std::vector<WorkerStats> out;
    out.reserve(max_id + 1u);
    for (u32 i = 0; i <= max_id && i < kMaxWorkers; ++i) {
        WorkerStats s{};
        s.worker_id      = i;
        s.logical_core   = g_workers[i].logical_core;
        s.numa_node      = g_workers[i].numa_node;
        s.tier           = g_workers[i].tier;
        s.is_perf_core   = (g_workers[i].tier == WorkerTier::Performance);
        s.jobs_executed  = g_workers[i].jobs.load(std::memory_order_relaxed);
        s.bytes_stolen   = g_workers[i].steals.load(std::memory_order_relaxed);
        const double busy_s =
            static_cast<double>(g_workers[i].busy_ns.load(std::memory_order_relaxed)) * 1e-9;
        s.busy_fraction  = std::min(1.0, busy_s / window_s);
        out.push_back(s);
    }
    // Reset the busy_ns accumulator if the window is over ~1s old, so the
    // chart stays responsive instead of approaching a fixed steady state.
    if (window_s > 1.0) {
        for (u32 i = 0; i <= max_id && i < kMaxWorkers; ++i) {
            g_workers[i].busy_ns.store(0, std::memory_order_relaxed);
        }
        g_window_start_ns.store(now, std::memory_order_relaxed);
    }
    return out;
}

}  // namespace cardinal::async
