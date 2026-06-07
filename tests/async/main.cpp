// =============================================================================
// Cardinal — cardinal::async regression suite.
//
// The ergonomic future/continuation + parallel_for/invoke layer over
// JobSystem that pack::Archive::load_async and the per-frame parallel
// phases build on. A regression in value propagation, exactly-once
// range coverage, or continuation firing silently breaks async asset
// loads + parallel work. Thread order is nondeterministic, so pooled
// assertions are schedule-INDEPENDENT (exactly-once, all-ran, value
// correctness); the no-pool fallback path is fully deterministic and
// its documented declaration-order contract is pinned. Exit 0 = pass.
//
// NOTE: Future::then() attached BEFORE resolution is now a GUARANTEED
// contract — attach (then) and resolve (the worker) serialise on
// State::mtx, so the continuation fires exactly once with the resolved
// value no matter which side wins the race. The old minimal impl could
// silently drop that callback (the load_async lost-wakeup); Phase C
// slams the then()-vs-resolve window 2000x and pins exactly-once +
// value correctness regardless of who wins.
// =============================================================================

#include <cardinal/core/sync/async.hpp>
#include <cardinal/core/sync/jobs.hpp>
#include <cardinal/core/sync/topology.hpp>
#include <cardinal/core/diag/log.hpp>

#include <atomic>
#include <functional>
#include <thread>
#include <vector>

namespace {

namespace ca = cardinal::async;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("asynctest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

// ---- Phase A: no pool bound → deterministic inline fallbacks --------
void test_no_pool() {
    CHECK(ca::pool() == nullptr);                  // nothing bound yet

    // async() runs inline → Future already resolved on return.
    auto f = ca::async([] { return 7; });
    CHECK(f.ready());
    CHECK(f.wait() == 7);
    int got = -1, hits = 0;
    f.then([&](const int& v) { got = v; ++hits; });
    CHECK(got == 7 && hits == 1);                  // fires immediately, once

    // parallel_invoke with no pool: sequential in DECLARATION ORDER.
    std::vector<int> order;
    ca::parallel_invoke({
        [&] { order.push_back(0); },
        [&] { order.push_back(1); },
        [&] { order.push_back(2); },
    });
    CHECK(order.size() == sz(3));
    CHECK(order.size() == sz(3) && order[0] == 0 &&
          order[1] == 1 && order[2] == 2);

    // parallel_for_each falls back to a serial loop (each once).
    std::vector<int> v{ 1, 2, 3, 4, 5 };
    ca::parallel_for_each(v, [](int& x) { x *= 2; });
    CHECK(v.size() == sz(5));
    CHECK(v[0]==2 && v[1]==4 && v[2]==6 && v[3]==8 && v[4]==10);

    // Empty container → no-op, no crash.
    std::vector<int> empty;
    ca::parallel_for_each(empty, [](int& x) { x = 99; });
    CHECK(empty.empty());

    // Default / empty Future is inert + safe (null state guard).
    ca::Future<int> d;
    CHECK(!d.ready());
    CHECK(d.wait() == 0);                          // returns T{}, no hang
    int touched = 0;
    d.then([&](const int&) { ++touched; });
    CHECK(touched == 0);                           // no state → no-op
}

// ---- Phase B: pooled — schedule-independent invariants --------------
void test_pooled(cardinal::JobSystem& js) {
    ca::set_pool(&js);
    CHECK(ca::pool() == &js);

    // Many independent asyncs — each Future carries its OWN value (no
    // cross-talk through the shared pool).
    std::vector<ca::Future<int>> futs;
    for (int i = 0; i < 256; ++i)
        futs.push_back(ca::async([i] { return i * i; }));
    bool all_vals = true;
    for (int i = 0; i < 256; ++i)
        if (futs[static_cast<cardinal::usize>(i)].wait() != i * i)
            all_vals = false;
    CHECK(all_vals);

    // then() attached AFTER the future resolved → fires immediately on
    // the calling thread, exactly once, with the resolved value.
    auto f = ca::async([] { return 42; });
    CHECK(f.wait() == 42);
    int got = -1; int hits = 0;
    f.then([&](const int& v) { got = v; ++hits; });
    CHECK(got == 42 && hits == 1);

    // parallel_for: exactly-once coverage of [0,N) (drop→0, dup→≥2),
    // and it BLOCKS until every chunk finished. Default + explicit
    // grains both cover the full range.
    auto cover = [&](cardinal::u32 grain) {
        const int N = 4000;
        std::vector<int> slot(sz(N), 0);
        ca::parallel_for(0u, static_cast<cardinal::u32>(N),
            [&](cardinal::u32 i) { ++slot[static_cast<cardinal::usize>(i)]; },
            grain);
        int ones = 0; bool allone = true;
        for (int x : slot) { if (x == 1) ++ones; else allone = false; }
        return allone && ones == N;
    };
    CHECK(cover(0u));       // default grain
    CHECK(cover(1u));       // one job per element
    CHECK(cover(64u));      // chunked
    CHECK(cover(0u));       // determinism: same result a second time

    // Empty range → fn never called.
    {
        std::atomic<int> calls{0};
        ca::parallel_for(5u, 5u,
            [&](cardinal::u32) { calls.fetch_add(1); });
        CHECK(calls.load() == 0);
    }

    // parallel_invoke (pooled): every task runs exactly once; the call
    // blocks until all are done.
    {
        std::atomic<int> a{0}, b{0}, c{0}, d{0};
        ca::parallel_invoke({
            [&] { a.fetch_add(1); },
            [&] { b.fetch_add(1); },
            [&] { c.fetch_add(1); },
            [&] { d.fetch_add(1); },
        });
        CHECK(a.load()==1 && b.load()==1 && c.load()==1 && d.load()==1);

        // vector overload, runtime-built task list.
        std::vector<std::function<void()>> tasks;
        std::atomic<int> sum{0};
        for (int i = 0; i < 32; ++i)
            tasks.push_back([&sum] { sum.fetch_add(1); });
        ca::parallel_invoke(tasks);
        CHECK(sum.load() == 32);                    // all ran, blocked
    }

    // parallel_for_each (pooled): each element visited exactly once.
    {
        std::vector<int> v(1000, 1);
        ca::parallel_for_each(v, [](int& x) { x += 41; });
        long total = 0;
        for (int x : v) total += x;
        CHECK(total == 1000L * 42L);                // every element +41 once
    }

    ca::set_pool(nullptr);
    CHECK(ca::pool() == nullptr);
}

// ---- Phase C: then()-vs-resolve attach race (lost-wakeup) -----------
// The bug this suite previously could NOT assert: then() reads
// done==false and is about to store on_done while the worker resolves,
// reads on_done (still empty), skips it — the continuation vanishes
// (exactly the pack::Archive::load_async drop). State::mtx now makes
// attach+resolve atomic, so whichever side wins the lock the callback
// fires EXACTLY ONCE with this iteration's own value. then() right
// before f.wait() lands the attach squarely in the danger window;
// 2000 iterations make a single lost/dup wakeup overwhelmingly likely
// to be caught. Schedule-independent: we assert the invariant, not an
// ordering. hits uses release/acquire so the worker-side `seen` write
// is visibly ordered before we read it.
void test_then_attach_race(cardinal::JobSystem& js) {
    ca::set_pool(&js);
    CHECK(ca::pool() == &js);

    bool all_once = true;     // fired exactly once every iteration
    bool all_vals = true;     // ...and always with its own value
    for (int i = 0; i < 2000; ++i) {
        auto f = ca::async([i] { return i; });
        std::atomic<int> hits{0};
        int seen = -1;
        f.then([&](const int& v) {
            seen = v;
            hits.fetch_add(1, std::memory_order_release);
        });
        f.wait();
        // The continuation runs just after done flips (resolver side) or
        // inline on this thread (then() side, post-resolve). wait() can
        // return before the resolver-side cb lands, so spin for it. A
        // healthy worker fires in microseconds; exhausting this bound
        // means the wakeup was lost — which is the failure we want.
        for (int spin = 0;
             spin < 2'000'000 && hits.load(std::memory_order_acquire) == 0;
             ++spin) {
            std::this_thread::yield();
        }
        if (hits.load(std::memory_order_acquire) != 1) all_once = false;
        if (seen != i)                                  all_vals = false;
    }
    CHECK(all_once);   // no lost wakeup, no double fire — ever
    CHECK(all_vals);   // each future's continuation saw its OWN value

    ca::set_pool(nullptr);
    CHECK(ca::pool() == nullptr);
}

// ---- FrameScope / phase bookkeeping — structural (timing-agnostic) --
void test_frame_phase() {
    {
        ca::FrameScope s("UnitTestPhase");
        volatile int z = 0;
        for (int k = 0; k < 5000; ++k) z += k;
        (void)z;
    }
    ca::frame_advance();
    CHECK(ca::frame_history_size() > 0u);
    const std::vector<ca::PhaseSample> ps = ca::phase_samples();
    bool found = false;
    for (const auto& p : ps) {
        if (p.name == "UnitTestPhase") {
            found = true;
            CHECK(p.calls_last >= 1u);
            CHECK(p.ms_last >= 0.0 && p.ms_avg >= 0.0 && p.ms_max >= 0.0);
        }
    }
    CHECK(found);
}

}  // namespace

int main() {
    test_no_pool();                                // before any pool bound

    cardinal::JobSystem js;
    js.start(cardinal::derive_worker_plan_physical_only(
        cardinal::topology::detect()));
    test_pooled(js);
    test_then_attach_race(js);                      // lost-wakeup stress
    js.shutdown();

    test_frame_phase();

    if (g_fail == 0) {
        cardinal::log::infof("asynctest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("asynctest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
