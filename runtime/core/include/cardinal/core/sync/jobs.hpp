#pragma once

#include <cardinal/core/topology.hpp>
#include <cardinal/core/types.hpp>

#include <vector>

namespace cardinal {

struct Counter;  // opaque
using JobFunc = void (*)(void* user_data);

// Worker tier — gives schedulers a hint about how strong each worker is.
//
// Performance: full physical core on the strongest cluster.
//              - AMD: V-Cache CCD if present, else cluster 0.
//              - Intel hybrid: P-cores.
//              - Other: all physical cores in cluster 0.
//   General  : full physical core in another cluster (NUMA-remote on
//              multi-CCD AMD desktop, or extra cluster on EPYC/Threadripper).
//   Background: weak workers — SMT siblings (the *other* logical thread on
//              an already-busy physical core) and Intel E-cores. SMT siblings
//              don't add a full core's throughput; they steal L1/L2 from the
//              primary thread but recover ~15-30% extra throughput when the
//              primary thread is memory-stalled. Use these for IO, async
//              decompression, telemetry — anything where latency matters less
//              than not pulling cycles away from the perf tier.
enum class WorkerTier : u8 {
    Performance = 0,
    General     = 1,
    Background  = 2,
};

// Worker placement plan derived from topology.
//
// Layout: workers come out grouped by tier, then by NUMA node within tier
// (so workers 0..N are NUMA-local Performance, N..M are local General, etc.).
// All three vectors are parallel — index `i` describes worker `i`.
//
//   worker_cores         : OS logical id (pin target)
//   worker_numa_nodes    : NUMA node the core lives on (0 on UMA hosts)
//   worker_tiers         : tier label
//   perf / general /
//   background_worker_count : prefix counts; sum == worker_cores.size()
struct WorkerPlan {
    std::vector<u32>        worker_cores;
    std::vector<u32>        worker_numa_nodes;
    std::vector<WorkerTier> worker_tiers;
    u32                     perf_worker_count{0};
    u32                     general_worker_count{0};
    u32                     background_worker_count{0};
};

// Build a WorkerPlan that uses every logical thread the host exposes:
// strong threads in the perf/general tiers, SMT siblings + E-cores in
// background. NUMA-aware ordering keeps the perf tier on the lowest-id
// NUMA node (or V-Cache CCD on AMD).
WorkerPlan derive_worker_plan(const topology::CpuInfo& info);

// Conservative variant — one worker per *physical* core, no SMT siblings,
// no E-cores. Useful for benchmarks or for hosts that prefer to leave SMT
// capacity for other processes (servers, dedicated game-state machines).
WorkerPlan derive_worker_plan_physical_only(const topology::CpuInfo& info);

// Topology-aware work-stealing job system.
//
// Phase 1.0: thread-pool with Chase-Lev deque per worker.
//   - submit()  : push job to the calling worker's deque (or worker 0 if external)
//   - wait()    : spin-yield until counter reaches zero
//                 TODO(phase-1.5): from a worker, park current fiber instead
//   - release() : return counter to the pool
class JobSystem {
public:
    JobSystem();
    ~JobSystem();

    JobSystem(const JobSystem&)            = delete;
    JobSystem& operator=(const JobSystem&) = delete;

    void start(const WorkerPlan& plan);
    void shutdown();

    // counter == nullptr → allocate a fresh counter (caller must release()).
    // Else                 → re-use caller's counter (must already be acquired).
    Counter* submit(JobFunc fn, void* user_data, Counter* counter = nullptr);

    void wait(Counter* counter);
    void release(Counter* counter);

    // Acquire a counter from the pool without submitting yet (for batches).
    Counter* acquire_counter(i32 initial_value = 0);

    u32 worker_count() const noexcept;
    u32 perf_worker_count() const noexcept;
    u32 general_worker_count()    const noexcept;
    u32 background_worker_count() const noexcept;

    // Per-worker introspection. Returns Performance / 0 for any out-of-range id
    // so callers can be lazy on edge cases.
    WorkerTier worker_tier      (u32 worker_id) const noexcept;
    u32        worker_numa_node (u32 worker_id) const noexcept;
    u32        worker_logical_core(u32 worker_id) const noexcept;

    // Returns the worker id of the calling thread, or u32(-1) if external.
    static u32 current_worker_id() noexcept;

    // ---- Diagnostics ---------------------------------------------------
    //
    // Per-worker monotonic counters, snapshot-once. All counters are
    // best-effort relaxed-atomic increments on the worker hot path (no
    // synchronisation overhead beyond a per-worker cache-line hit).
    //
    //   jobs_executed  : count of jobs whose fn() returned on this worker
    //   jobs_stolen    : count of jobs taken from ANOTHER worker's deque
    //                    (subset of jobs_executed; high ratio = poor
    //                    scheduling locality, low ratio = good)
    //   steal_attempts : total steal calls (success + miss). steals/attempts
    //                    is the steal hit rate; low rate = workers idle
    //                    while others are loaded.
    //   sleep_events   : times this worker fell into the condvar wait — a
    //                    proxy for "starvation events"; high counts mean
    //                    the queue ran dry repeatedly.
    //   queue_depth    : approximate own-deque depth at the moment of
    //                    snapshot (best-effort under concurrent steals).
    struct WorkerStats {
        u64 jobs_executed{0};
        u64 jobs_stolen{0};
        u64 steal_attempts{0};
        u64 sleep_events{0};
        u32 queue_depth{0};
    };

    struct JobSystemStats {
        std::vector<WorkerStats> per_worker;
        u64 total_jobs_executed{0};
        u64 total_jobs_stolen{0};
        u64 total_steal_attempts{0};
        u32 max_queue_depth{0};
    };

    // Snapshot all worker stats in one call. Allocates per_worker
    // (worker_count() entries) — call at frame boundaries, not per-job.
    JobSystemStats stats() const noexcept;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace cardinal
