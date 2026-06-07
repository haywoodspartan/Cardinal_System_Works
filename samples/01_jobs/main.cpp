#include <cardinal/core/sync/jobs.hpp>
#include <cardinal/core/sync/topology.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <vector>

namespace ct = cardinal::topology;

using cardinal::i32;
using cardinal::u32;
using cardinal::u64;

namespace {

void busy_work(u64 iters, std::atomic<u64>& sink) {
    u64 acc = 1;
    for (u64 i = 1; i <= iters; ++i) {
        acc = acc * 1664525u + 1013904223u + i;
    }
    sink.fetch_add(acc, std::memory_order_relaxed);
}

struct WorkArgs {
    u64 iters;
    std::atomic<u64>* sink;
};

void leaf_job(void* user) {
    auto* a = static_cast<WorkArgs*>(user);
    busy_work(a->iters, *a->sink);
}

// ---------------------------------------------------------------------------
// Bench 1 — flat fan-out / fan-in.
// ---------------------------------------------------------------------------
double bench_flat(cardinal::JobSystem& js, u32 jobs, u64 iters_per_job) {
    std::atomic<u64> sink{0};
    std::vector<WorkArgs> args(jobs);
    for (auto& a : args) {
        a.iters = iters_per_job;
        a.sink  = &sink;
    }

    auto* counter = js.acquire_counter(0);
    auto t0 = std::chrono::steady_clock::now();

    for (u32 i = 0; i < jobs; ++i) {
        js.submit(leaf_job, &args[i], counter);
    }
    js.wait(counter);
    js.release(counter);

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (sink.load() == 0) std::fprintf(stderr, "(unused)\n");
    return ms;
}

// ---------------------------------------------------------------------------
// Bench 2 — recursive fan-out: each parent submits N children and WAITS on
//           the counter before returning. Exercises the fiber park path.
// ---------------------------------------------------------------------------
struct ParentArgs {
    cardinal::JobSystem* js;
    u32 fanout;
    u32 depth;
    u64 leaf_iters;
    std::atomic<u64>* sink;
};

void parent_job(void* user);  // forward

void run_parent(cardinal::JobSystem& js, u32 fanout, u32 depth,
                u64 leaf_iters, std::atomic<u64>& sink) {
    if (depth == 0) {
        busy_work(leaf_iters, sink);
        return;
    }

    auto* counter = js.acquire_counter(0);
    std::vector<ParentArgs> args(fanout);
    for (auto& a : args) {
        a.js         = &js;
        a.fanout     = fanout;
        a.depth      = depth - 1;
        a.leaf_iters = leaf_iters;
        a.sink       = &sink;
    }
    for (u32 i = 0; i < fanout; ++i) {
        js.submit(parent_job, &args[i], counter);
    }
    js.wait(counter);  // <-- fiber park here when called from a worker fiber
    js.release(counter);
}

void parent_job(void* user) {
    auto* a = static_cast<ParentArgs*>(user);
    run_parent(*a->js, a->fanout, a->depth, a->leaf_iters, *a->sink);
}

double bench_tree(cardinal::JobSystem& js, u32 fanout, u32 depth, u64 leaf_iters) {
    std::atomic<u64> sink{0};
    auto t0 = std::chrono::steady_clock::now();
    run_parent(js, fanout, depth, leaf_iters, sink);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (sink.load() == 0) std::fprintf(stderr, "(unused)\n");
    return ms;
}

}  // namespace

int main() {
    auto info = ct::detect();
    auto plan = cardinal::derive_worker_plan(info);

    std::printf("Workers: %zu  (perf=%u, general=%u, background=%u)\n",
                plan.worker_cores.size(),
                plan.perf_worker_count,
                plan.general_worker_count,
                plan.background_worker_count);
    std::printf("Pinned to OS-ids:");
    for (size_t i = 0; i < plan.worker_cores.size(); ++i) {
        const char* tag = "?";
        switch (plan.worker_tiers[i]) {
            case cardinal::WorkerTier::Performance: tag = "P"; break;
            case cardinal::WorkerTier::General:     tag = "G"; break;
            case cardinal::WorkerTier::Background:  tag = "b"; break;
        }
        std::printf(" %u(%s/n%u)", plan.worker_cores[i], tag,
                    plan.worker_numa_nodes[i]);
    }
    std::printf("\n\n");

    cardinal::JobSystem js;
    js.start(plan);

    for (int w = 0; w < 2; ++w) (void)bench_flat(js, 256, 5'000);

    // ---- Bench 1: flat fan-out ----
    std::printf("--- Flat fan-out (external submit, fiber-park wait) ---\n");
    std::printf("%-12s %-12s %-14s %-14s\n", "JOBS", "ITERS/JOB", "TOTAL_MS", "JOBS/SEC");
    std::printf("------------------------------------------------------\n");
    constexpr u64 iters = 50'000;
    constexpr u32 job_counts[] = { 64, 256, 1024, 4096, 16384 };
    for (u32 n : job_counts) {
        double best = 1e30;
        for (int r = 0; r < 3; ++r) {
            double ms = bench_flat(js, n, iters);
            if (ms < best) best = ms;
        }
        std::printf("%-12u %-12llu %-14.3f %-14.0f\n",
                    n, static_cast<unsigned long long>(iters), best,
                    static_cast<double>(n) / (best / 1000.0));
    }

    // ---- Bench 2: recursive tree (parents wait on children inside fibers) ----
    std::printf("\n--- Tree fan-out (parents fiber-park on children) ---\n");
    std::printf("%-10s %-10s %-12s %-14s %-14s\n",
                "FANOUT", "DEPTH", "LEAVES", "TOTAL_MS", "LEAVES/SEC");
    std::printf("--------------------------------------------------------------\n");
    struct TreeCfg { u32 fanout; u32 depth; };
    constexpr TreeCfg cfgs[] = { {4, 4}, {8, 3}, {4, 6}, {8, 4} };
    for (const auto& cfg : cfgs) {
        double best = 1e30;
        for (int r = 0; r < 3; ++r) {
            double ms = bench_tree(js, cfg.fanout, cfg.depth, iters);
            if (ms < best) best = ms;
        }
        u64 leaves = 1;
        for (u32 d = 0; d < cfg.depth; ++d) leaves *= cfg.fanout;
        std::printf("%-10u %-10u %-12llu %-14.3f %-14.0f\n",
                    cfg.fanout, cfg.depth,
                    static_cast<unsigned long long>(leaves),
                    best,
                    static_cast<double>(leaves) / (best / 1000.0));
    }

    js.shutdown();
    return 0;
}
