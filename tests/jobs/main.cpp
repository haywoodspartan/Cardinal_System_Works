// =============================================================================
// Cardinal — JobSystem regression suite (schedule-independent).
//
// The work-stealing job system is the parallel substrate under net
// async loads, pack streaming, and the sim's parallel tick-group
// dispatch. A regression that DROPS or DOUBLE-RUNS a task corrupts
// those silently. Thread order is nondeterministic, so every assertion
// is a schedule-INDEPENDENT invariant: every submitted job runs exactly
// once, the aggregate is identical across runs, wait() truly blocks
// until completion, and the diagnostics ledger reconciles exactly with
// the work submitted. Exit 0 = all pass.
// =============================================================================

#include <cardinal/core/jobs.hpp>
#include <cardinal/core/topology.hpp>
#include <cardinal/core/log.hpp>

#include <atomic>
#include <vector>

namespace {

namespace cc = cardinal;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("jobstest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

// Per-job context. `slot` is unique per job (no sharing → no data race);
// `total` is the shared atomic tally. A dropped job leaves slot==0; a
// double-run leaves slot==2 — both caught.
struct Ctx {
    int*              slot{nullptr};
    std::atomic<int>* total{nullptr};
    std::atomic<cc::u32>* worker{nullptr};   // optional: record worker id
    int               spin{0};
};

void job_count(void* p) {
    Ctx* c = static_cast<Ctx*>(p);
    // Deterministic busy work so wait() genuinely has to wait.
    volatile int s = 0;
    for (int k = 0; k < c->spin; ++k) s += k;
    (void)s;
    *c->slot += 1;
    c->total->fetch_add(1, std::memory_order_relaxed);
    if (c->worker)
        c->worker->store(cc::JobSystem::current_worker_id(),
                         std::memory_order_relaxed);
}

// Run `n` jobs on `js` against one shared counter; verify exactly-once.
void run_batch(cc::JobSystem& js, int n, int spin, bool& ok_out,
               int* total_out) {
    std::vector<int> slots(static_cast<cardinal::usize>(n), 0);
    std::atomic<int> total{0};
    std::vector<Ctx> ctx(static_cast<cardinal::usize>(n));
    for (int i = 0; i < n; ++i) {
        ctx[static_cast<cardinal::usize>(i)].slot =
            &slots[static_cast<cardinal::usize>(i)];
        ctx[static_cast<cardinal::usize>(i)].total = &total;
        ctx[static_cast<cardinal::usize>(i)].spin  = spin;
    }
    cc::Counter* cnt = js.acquire_counter();
    for (int i = 0; i < n; ++i)
        js.submit(&job_count, &ctx[static_cast<cardinal::usize>(i)], cnt);
    js.wait(cnt);
    js.release(cnt);

    bool all_one = true;
    int  ones = 0;
    for (int v : slots) {
        if (v == 1) ++ones;
        else        all_one = false;          // 0 = dropped, ≥2 = double-run
    }
    ok_out    = all_one && (total.load() == n) && (ones == n);
    *total_out = total.load();
}

}  // namespace

int main() {
    // ---- WorkerPlan invariants -------------------------------------
    const cc::topology::CpuInfo info = cc::topology::detect();
    const cc::WorkerPlan full = cc::derive_worker_plan(info);
    const cc::WorkerPlan phys = cc::derive_worker_plan_physical_only(info);

    auto plan_ok = [](const cc::WorkerPlan& p) {
        return p.worker_cores.size() == p.worker_numa_nodes.size()
            && p.worker_cores.size() == p.worker_tiers.size()
            && (static_cast<cardinal::usize>(p.perf_worker_count) +
                p.general_worker_count + p.background_worker_count)
                   == p.worker_cores.size();
    };
    CHECK(plan_ok(full));
    CHECK(plan_ok(phys));
    CHECK(!phys.worker_cores.empty());                 // host has ≥1 core
    CHECK(phys.worker_cores.size() <= full.worker_cores.size());

    cc::JobSystem js;
    js.start(phys);
    CHECK(js.worker_count() == static_cast<cc::u32>(phys.worker_cores.size()));
    CHECK(js.worker_count() >= 1u);
    CHECK(js.perf_worker_count()       == phys.perf_worker_count);
    CHECK(js.general_worker_count()    == phys.general_worker_count);
    CHECK(js.background_worker_count() == phys.background_worker_count);
    // External (main) thread is not a worker.
    CHECK(cc::JobSystem::current_worker_id() == static_cast<cc::u32>(-1));
    // Lazy out-of-range introspection (documented: Performance / 0).
    CHECK(js.worker_tier(99999u) == cc::WorkerTier::Performance);
    CHECK(js.worker_numa_node(99999u) == 0u);
    for (cc::u32 w = 0; w < js.worker_count(); ++w) {
        const cc::u32 t = static_cast<cc::u32>(js.worker_tier(w));
        CHECK(t <= 2u);                                 // valid tier label
        (void)js.worker_numa_node(w);
        (void)js.worker_logical_core(w);
    }

    // ---- T1: every submitted job runs exactly once -----------------
    {
        bool ok = false; int tot = 0;
        run_batch(js, 5000, 4000, ok, &tot);
        CHECK(ok);
        CHECK(tot == 5000);
    }

    // ---- T2: aggregate is identical across independent runs --------
    {
        bool ok1 = false, ok2 = false; int t1 = 0, t2 = 0;
        run_batch(js, 4096, 1500, ok1, &t1);
        run_batch(js, 4096, 1500, ok2, &t2);
        CHECK(ok1 && ok2);
        CHECK(t1 == 4096 && t2 == 4096);
        CHECK(t1 == t2);                                // deterministic agg
    }

    // ---- T3: wait() truly blocks until ALL jobs finished -----------
    // run_batch already asserts every slot == 1 immediately AFTER
    // wait(); if wait() returned early, some slot would still be 0.
    {
        bool ok = false; int tot = 0;
        run_batch(js, 3000, 8000, ok, &tot);            // heavier spin
        CHECK(ok);                                      // all done post-wait
        CHECK(tot == 3000);
    }

    // ---- T4: a running job sees a valid worker id ------------------
    {
        int slot = 0;
        std::atomic<int> total{0};
        std::atomic<cc::u32> wid{ static_cast<cc::u32>(-2) };
        Ctx c; c.slot = &slot; c.total = &total; c.worker = &wid; c.spin = 0;
        cc::Counter* cnt = js.acquire_counter();
        js.submit(&job_count, &c, cnt);
        js.wait(cnt);
        js.release(cnt);
        CHECK(slot == 1 && total.load() == 1);
        const cc::u32 w = wid.load();
        CHECK(w != static_cast<cc::u32>(-1));           // ran on a worker
        CHECK(w != static_cast<cc::u32>(-2));           // callback fired
        CHECK(w < js.worker_count());
    }

    // ---- T5: diagnostics ledger reconciles EXACTLY -----------------
    // Between two stats() snapshots, total_jobs_executed must rise by
    // exactly the number of jobs submitted in between — proof of no
    // drop and no double-run, cross-checked through the independent
    // diagnostics path. Also: per-worker sum == reported total.
    {
        const cc::u64 before = js.stats().total_jobs_executed;
        bool ok = false; int tot = 0;
        run_batch(js, 2500, 500, ok, &tot);
        CHECK(ok && tot == 2500);
        const cc::JobSystem::JobSystemStats s = js.stats();
        CHECK(s.per_worker.size() == sz(static_cast<int>(js.worker_count())));
        CHECK(s.total_jobs_executed - before == 2500u);
        cc::u64 sum = 0;
        for (const auto& w : s.per_worker) sum += w.jobs_executed;
        CHECK(sum == s.total_jobs_executed);
    }

    js.shutdown();

    if (g_fail == 0) {
        cardinal::log::infof("jobstest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("jobstest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
