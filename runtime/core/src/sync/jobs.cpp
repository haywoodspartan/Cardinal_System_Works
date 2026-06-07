#include <cardinal/core/sync/jobs.hpp>

#include <cardinal/core/sync/affinity.hpp>
#include <cardinal/core/platform.hpp>

// Compiler optimization is disabled for this TU because MSVC's /O2 (and
// likely clang-cl's equivalent) cache memory accesses across SwitchToFiber
// in ways that break fiber resumption. Atomic members + acquire/release
// pairing was insufficient. The proper fix is to replace Win32 fibers with
// a hand-rolled x86_64 fiber switch (in a .asm file) — at that point the
// compiler can see exactly what's clobbered and we can re-enable /O2.
// See feedback_msvc_fiber_optimizer.md in agent memory for context.
#if CARDINAL_COMPILER_MSVC
    #pragma optimize("", off)
#endif

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#if CARDINAL_PLATFORM_WINDOWS
    #include <Windows.h>
#elif CARDINAL_PLATFORM_LINUX
    #include <ucontext.h>
#endif

namespace cardinal {

namespace {

// ============================================================================
// Tunables
// ============================================================================
constexpr u32   deque_pow2        = 10;
constexpr u32   deque_capacity    = 1u << deque_pow2;
constexpr u32   deque_mask        = deque_capacity - 1;
constexpr usize fiber_stack_bytes = 1024 * 1024;  // 1 MiB — generous for now
constexpr u32   fiber_pool_size   = 256;
constexpr u32   job_pool_size     = 64 * 1024;
constexpr u32   counter_pool_size = 8192;
constexpr usize cl                = 64;

}  // namespace

// ============================================================================
// Chase-Lev work-stealing deque (Lê et al. 2013, weak-memory variant).
// ============================================================================
template <typename T>
class WorkStealingDeque {
public:
    WorkStealingDeque() = default;

    WorkStealingDeque(const WorkStealingDeque&)            = delete;
    WorkStealingDeque& operator=(const WorkStealingDeque&) = delete;

    // Returns false if the deque is full (size == capacity). This ring
    // has no growable backing array, so the producer MUST handle a full
    // deque rather than overwrite a live, unconsumed slot. Reading top_
    // with acquire is conservative: a stale (smaller) t can only make us
    // report "full" slightly early — it can never let us overflow.
    [[nodiscard]] bool push(T item) {
        const i64 b = bottom_.load(std::memory_order_relaxed);
        const i64 t = top_.load(std::memory_order_acquire);
        if (b - t >= static_cast<i64>(deque_capacity)) return false;
        ring_[static_cast<u32>(b) & deque_mask].store(item, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        bottom_.store(b + 1, std::memory_order_relaxed);
        return true;
    }

    bool pop(T& out) {
        i64 b = bottom_.load(std::memory_order_relaxed) - 1;
        bottom_.store(b, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        i64 t = top_.load(std::memory_order_relaxed);
        if (t > b) {
            bottom_.store(b + 1, std::memory_order_relaxed);
            return false;
        }
        T item = ring_[static_cast<u32>(b) & deque_mask].load(std::memory_order_relaxed);
        if (t == b) {
            if (!top_.compare_exchange_strong(
                    t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
                bottom_.store(b + 1, std::memory_order_relaxed);
                return false;
            }
            bottom_.store(b + 1, std::memory_order_relaxed);
        }
        out = item;
        return true;
    }

    bool steal(T& out) {
        i64 t = top_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        i64 b = bottom_.load(std::memory_order_acquire);
        if (t >= b) return false;
        T item = ring_[static_cast<u32>(t) & deque_mask].load(std::memory_order_relaxed);
        if (!top_.compare_exchange_strong(
                t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
            return false;
        }
        out = item;
        return true;
    }

    // Best-effort snapshot of the current occupancy. Under concurrent
    // steals the read can transiently show 0 even when there's work
    // queued; tolerated for diagnostic use (the worker thread itself
    // doesn't rely on this for scheduling decisions).
    u32 size_approx() const noexcept {
        const i64 b = bottom_.load(std::memory_order_relaxed);
        const i64 t = top_.load(std::memory_order_relaxed);
        return (b > t) ? static_cast<u32>(b - t) : 0u;
    }

private:
    alignas(cl) std::atomic<i64> top_{0};
    alignas(cl) std::atomic<i64> bottom_{0};
    alignas(cl) std::array<std::atomic<T>, deque_capacity> ring_{};
};

// ============================================================================
// Internal types
// ============================================================================
struct FiberSlot;

struct Counter {
    std::atomic<i32> value{0};
    std::atomic<FiberSlot*> waiters{nullptr};
    std::atomic<bool> in_use{false};
};

struct Job {
    JobFunc fn{nullptr};
    void* user_data{nullptr};
    Counter* counter{nullptr};
    std::atomic<bool> in_use{false};
};

struct FiberSlot {
    void* native{nullptr};                   // LPVOID on Win, ucontext_t* on Linux
#if CARDINAL_PLATFORM_LINUX
    std::vector<u8> stack;
#endif
    // Atomic because they cross SwitchToFiber boundaries. Acquire/release
    // pairing forces the compiler to emit real memory ops the optimizer
    // cannot cache or reorder across the (opaque-to-it) fiber switch.
    std::atomic<Job*> current_job{nullptr};
    u32 home_worker{0};
    FiberSlot* park_next{nullptr};
    Counter* parked_on{nullptr};
    std::atomic<bool> in_use{false};         // pool ownership
};

struct alignas(cl) Worker {
    std::thread thread;
    void* scheduler_fiber{nullptr};
    // Atomic for the same reason as FiberSlot::current_job above.
    std::atomic<FiberSlot*> current_fiber{nullptr};
    std::atomic<FiberSlot*> releasing_fiber{nullptr};

    WorkStealingDeque<Job*> jobs;
    WorkStealingDeque<FiberSlot*> ready_fibers;

    // Per-worker external inbox. External submits round-robin into here so
    // there's no contention with other workers' inboxes.
    std::mutex inbox_mtx;
    std::vector<Job*> inbox;

    // Worker sleep on a condvar; waker calls notify_one.
    std::mutex sleep_mtx;
    std::condition_variable sleep_cv;

    u32        worker_id{0};
    u32        logical_core{0};
    u32        numa_node{0};
    WorkerTier tier{WorkerTier::Performance};

    // Diagnostic counters — updated relaxed-atomic on the hot path.
    // Snapshot via JobSystem::stats() at frame boundaries.
    std::atomic<u64> jobs_executed_count{0};
    std::atomic<u64> jobs_stolen_count{0};
    std::atomic<u64> steal_attempts_count{0};
    std::atomic<u64> sleep_events_count{0};
};

// ============================================================================
// Impl
// ============================================================================
struct JobSystem::Impl {
    std::vector<Worker> workers;
    std::vector<Job> job_pool;
    std::vector<Counter> counter_pool;
    std::vector<FiberSlot> fiber_pool;

    std::atomic<u32> job_alloc_cursor{0};
    std::atomic<u32> counter_alloc_cursor{0};
    std::atomic<u32> fiber_alloc_cursor{0};
    std::atomic<u32> external_rr{0};

    std::atomic<bool> running{false};
    u32 perf_count{0};
    u32 general_count{0};
    u32 background_count{0};

    static thread_local u32 tls_worker_id;
    static thread_local Impl* tls_impl;

    // ------------------------------------------------------------------------
    // Generic CAS-scan pool acquire.
    // ------------------------------------------------------------------------
    template <typename Slot>
    static Slot* try_acquire(std::vector<Slot>& pool, std::atomic<u32>& cursor) {
        const u32 n = static_cast<u32>(pool.size());
        u32 start = cursor.fetch_add(1, std::memory_order_relaxed) % n;
        for (u32 i = 0; i < n; ++i) {
            u32 idx = (start + i) % n;
            bool expected = false;
            if (pool[idx].in_use.compare_exchange_strong(
                    expected, true, std::memory_order_acquire, std::memory_order_relaxed)) {
                return &pool[idx];
            }
        }
        return nullptr;
    }

    // Block until a slot frees up. Pools are sized for steady-state load,
    // but a burst can momentarily saturate them — yield instead of asserting
    // so the system stays alive (assert is a no-op under NDEBUG).
    template <typename Slot>
    Slot* acquire_blocking(std::vector<Slot>& pool, std::atomic<u32>& cursor) {
        for (int spin = 0;; ++spin) {
            Slot* s = try_acquire(pool, cursor);
            if (s != nullptr) return s;
            if (spin > 64) {
                std::this_thread::yield();
                spin = 0;
            }
        }
    }

    Job*       acquire_job()   { return acquire_blocking(job_pool,   job_alloc_cursor); }
    FiberSlot* acquire_fiber() { return acquire_blocking(fiber_pool, fiber_alloc_cursor); }
    Counter*   acquire_counter_internal() {
        Counter* c = acquire_blocking(counter_pool, counter_alloc_cursor);
        c->value.store(0, std::memory_order_relaxed);
        c->waiters.store(nullptr, std::memory_order_relaxed);
        return c;
    }

    void release_job(Job* j) {
        j->fn        = nullptr;
        j->user_data = nullptr;
        j->counter   = nullptr;
        j->in_use.store(false, std::memory_order_release);
    }
    void release_counter_internal(Counter* c) {
        c->value.store(0, std::memory_order_relaxed);
        c->waiters.store(nullptr, std::memory_order_relaxed);
        c->in_use.store(false, std::memory_order_release);
    }
    void release_fiber(FiberSlot* f) {
        f->current_job.store(nullptr, std::memory_order_relaxed);
        f->park_next   = nullptr;
        f->parked_on   = nullptr;
        f->in_use.store(false, std::memory_order_release);
    }

    // ------------------------------------------------------------------------
    // Fiber primitives — Win32 fibers / Linux ucontext.
    //
    // ucontext is the slow-but-correct path on Linux. It performs a syscall
    // (rt_sigprocmask) on every switch. TODO(perf): replace with hand-rolled
    // x86_64 SysV ABI assembly fiber switch (~30 lines, ~10x faster) once we
    // have a Linux dev machine to validate against.
    // ------------------------------------------------------------------------
#if CARDINAL_PLATFORM_WINDOWS
    static void CALLBACK fiber_proc_win(LPVOID arg) {
        static_cast<Impl*>(arg)->fiber_main_loop();
    }
    // Compiler barrier on both sides — SwitchToFiber is opaque to the
    // optimizer, which would otherwise cache loads/stores across it. The
    // signal_fence is free at runtime; it just constrains the compiler.
    static void switch_to(void* /*from*/, void* to) noexcept {
        std::atomic_signal_fence(std::memory_order_seq_cst);
        SwitchToFiber(to);
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }

    void create_fiber_native(FiberSlot* slot) {
        slot->native = CreateFiber(fiber_stack_bytes, fiber_proc_win, this);
        assert(slot->native != nullptr && "CreateFiber failed");
    }
    void destroy_fiber_native(FiberSlot* slot) {
        if (slot->native != nullptr) DeleteFiber(slot->native);
        slot->native = nullptr;
    }

#elif CARDINAL_PLATFORM_LINUX
    static void linux_fiber_entry(int hi, int lo) {
        std::uintptr_t p = (static_cast<std::uintptr_t>(static_cast<u32>(hi)) << 32)
                         |  static_cast<std::uintptr_t>(static_cast<u32>(lo));
        reinterpret_cast<Impl*>(p)->fiber_main_loop();
    }
    static void switch_to(void* from, void* to) noexcept {
        std::atomic_signal_fence(std::memory_order_seq_cst);
        swapcontext(static_cast<ucontext_t*>(from), static_cast<ucontext_t*>(to));
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }

    void create_fiber_native(FiberSlot* slot) {
        slot->stack.assign(fiber_stack_bytes, 0);
        auto* ctx = new ucontext_t{};
        getcontext(ctx);
        ctx->uc_stack.ss_sp   = slot->stack.data();
        ctx->uc_stack.ss_size = slot->stack.size();
        ctx->uc_link          = nullptr;
        std::uintptr_t p = reinterpret_cast<std::uintptr_t>(this);
        makecontext(ctx,
            reinterpret_cast<void (*)()>(linux_fiber_entry), 2,
            static_cast<int>(p >> 32),
            static_cast<int>(p & 0xFFFFFFFFu));
        slot->native = ctx;
    }
    void destroy_fiber_native(FiberSlot* slot) {
        delete static_cast<ucontext_t*>(slot->native);
        slot->native = nullptr;
    }
#endif

    // ------------------------------------------------------------------------
    // Counter park/wake protocol.
    //
    // Race-safe wait pattern:
    //   1) Caller (in fiber) checks counter; > 0 → continues.
    //   2) Push fiber onto counter->waiters via CAS.
    //   3) Re-check counter.  == 0 → may have been swept already; sweep again
    //      idempotently. (exchange returns nullptr if empty; harmless.)
    //   4) Switch back to scheduler.
    // ------------------------------------------------------------------------
    void park_fiber(Counter* c, FiberSlot* fiber) {
        fiber->parked_on = c;
        FiberSlot* head = c->waiters.load(std::memory_order_acquire);
        do {
            fiber->park_next = head;
        } while (!c->waiters.compare_exchange_weak(
                    head, fiber, std::memory_order_release, std::memory_order_acquire));
    }

    void wake_counter_waiters(Counter* c) {
        FiberSlot* head = c->waiters.exchange(nullptr, std::memory_order_acq_rel);
        while (head != nullptr) {
            FiberSlot* next = head->park_next;
            head->park_next = nullptr;
            head->parked_on = nullptr;
            Worker& home = workers[head->home_worker];
            // Provably never full: at most fiber_pool_size (256) fibers
            // can exist and deque_capacity is 1024, so a parked-fiber
            // re-queue always fits.
            (void)home.ready_fibers.push(head);
            wake_worker(head->home_worker);
            head = next;
        }
    }

    // ------------------------------------------------------------------------
    // Worker wakeup / sleep.
    // ------------------------------------------------------------------------
    void wake_worker(u32 worker_id) {
        Worker& w = workers[worker_id];
        std::lock_guard<std::mutex> lg(w.sleep_mtx);
        w.sleep_cv.notify_one();
    }

    void try_drain_own_inbox(Worker& self) {
        std::unique_lock<std::mutex> lock(self.inbox_mtx, std::try_to_lock);
        if (!lock.owns_lock() || self.inbox.empty()) return;
        constexpr int drain_limit = 128;
        for (int n = 0; n < drain_limit && !self.inbox.empty(); ++n) {
            Job* j = self.inbox.back();
            // Deque full → stop draining; leave the rest in the inbox for
            // later worker_main iterations to pick up as the deque empties.
            // This backpressure is what stops the fixed-capacity ring from
            // overflowing (was: unconditional push → ring overwrite →
            // stale/null Job* dispatched → segfault on many-producer /
            // few-consumer hosts, e.g. a low-core CI runner).
            if (!self.jobs.push(j)) break;
            self.inbox.pop_back();
        }
    }

    // ------------------------------------------------------------------------
    // Scheduling: scheduler fiber drives the worker.
    // ------------------------------------------------------------------------
    void schedule_job(Worker& self, Job* job) {
        FiberSlot* fiber = acquire_fiber();
        if (fiber->native == nullptr) {
            create_fiber_native(fiber);
        }
        // Order: write current_job and home_worker first; publish via release
        // store on current_fiber. Fiber's matching acquire load ensures it
        // sees both writes before reading current_job.
        fiber->current_job.store(job, std::memory_order_relaxed);
        fiber->home_worker = self.worker_id;
        self.current_fiber.store(fiber, std::memory_order_release);
        switch_to(self.scheduler_fiber, fiber->native);
    }

    void worker_main(u32 worker_id) {
        tls_worker_id = worker_id;
        tls_impl      = this;

        Worker& self = workers[worker_id];
        affinity::pin_current_thread(self.logical_core);

#if CARDINAL_PLATFORM_WINDOWS
        self.scheduler_fiber = ConvertThreadToFiber(nullptr);
#elif CARDINAL_PLATFORM_LINUX
        self.scheduler_fiber = new ucontext_t{};
        getcontext(static_cast<ucontext_t*>(self.scheduler_fiber));
#endif

        std::mt19937 rng{worker_id * 2654435761u};

        while (running.load(std::memory_order_acquire)) {
            // Recycle the previously-completed fiber, if any.
            FiberSlot* releasing = self.releasing_fiber.load(std::memory_order_acquire);
            if (releasing != nullptr) {
                release_fiber(releasing);
                self.releasing_fiber.store(nullptr, std::memory_order_relaxed);
            }

            // 1) Resumed parked fibers — best cache locality, take priority.
            FiberSlot* resumed = nullptr;
            if (self.ready_fibers.pop(resumed)) {
                // Release-publish current_fiber so the resumed fiber's
                // acquire load on resume sees us as its assignment.
                self.current_fiber.store(resumed, std::memory_order_release);
                switch_to(self.scheduler_fiber, resumed->native);
                continue;
            }

            // 2) Drain own external inbox into own deque.
            try_drain_own_inbox(self);

            // 3) Pop own deque.
            Job* job = nullptr;
            if (self.jobs.pop(job)) {
                schedule_job(self, job);
                continue;
            }

            // 4) Steal from a random victim.
            const u32 n = static_cast<u32>(workers.size());
            if (n > 1) {
                std::uniform_int_distribution<u32> pick(0, n - 1);
                u32 victim = pick(rng);
                if (victim == worker_id) victim = (victim + 1) % n;
                self.steal_attempts_count.fetch_add(1, std::memory_order_relaxed);
                if (workers[victim].jobs.steal(job)) {
                    self.jobs_stolen_count.fetch_add(1, std::memory_order_relaxed);
                    schedule_job(self, job);
                    continue;
                }
            }

            // 5) Sleep on condvar; submit() / wake_counter_waiters() notify us.
            //    Time out periodically as a belt-and-braces against missed wakes.
            self.sleep_events_count.fetch_add(1, std::memory_order_relaxed);
            {
                std::unique_lock<std::mutex> lock(self.sleep_mtx);
                self.sleep_cv.wait_for(lock, std::chrono::microseconds(200));
            }
        }

#if CARDINAL_PLATFORM_WINDOWS
        ConvertFiberToThread();
#elif CARDINAL_PLATFORM_LINUX
        delete static_cast<ucontext_t*>(self.scheduler_fiber);
        self.scheduler_fiber = nullptr;
#endif
    }

    // Each job fiber loops forever — runs one job, switches back, gets a new
    // job, runs it, etc. Fibers stay resident; only their assignment changes.
    void fiber_main_loop() {
        while (true) {
            u32 wid = tls_worker_id;
            Worker& self = workers[wid];
            // Acquire load pairs with the release store made by the scheduler
            // when it published us as the current fiber — guarantees we see
            // the current_job write that preceded it.
            FiberSlot* me = self.current_fiber.load(std::memory_order_acquire);
            Job* job = me->current_job.load(std::memory_order_relaxed);

            if (job != nullptr) {
                job->fn(job->user_data);
                self.jobs_executed_count.fetch_add(1, std::memory_order_relaxed);
                Counter* c = job->counter;
                release_job(job);
                if (c != nullptr) {
                    if (c->value.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        wake_counter_waiters(c);
                        // Wake any external thread sleeping in JobSystem::
                        // wait via the C++20 atomic wait/notify path. No-op
                        // if no one is waiting (cheap on the fast path).
                        c->value.notify_all();
                    }
                }
            }

            // Mark self for recycling and switch back to scheduler.
            me->current_job.store(nullptr, std::memory_order_relaxed);
            self.releasing_fiber.store(me,    std::memory_order_release);
            self.current_fiber.store(nullptr, std::memory_order_relaxed);
            switch_to(me->native, self.scheduler_fiber);
            // Resumed: scheduler set self.current_fiber = us and assigned a
            // new job (or we're being woken from a park; current_job may be
            // null for that path — see wait()).
        }
    }
};

thread_local u32 JobSystem::Impl::tls_worker_id   = static_cast<u32>(-1);
thread_local JobSystem::Impl* JobSystem::Impl::tls_impl = nullptr;

// ============================================================================
// JobSystem public API
// ============================================================================
JobSystem::JobSystem() : impl_(new Impl{}) {
    impl_->job_pool     = std::vector<Job>(job_pool_size);
    impl_->counter_pool = std::vector<Counter>(counter_pool_size);
    impl_->fiber_pool   = std::vector<FiberSlot>(fiber_pool_size);
}

JobSystem::~JobSystem() {
    if (impl_->running.load(std::memory_order_acquire)) shutdown();
    // Tear down any allocated fiber natives.
    for (auto& f : impl_->fiber_pool) {
        if (f.native != nullptr) impl_->destroy_fiber_native(&f);
    }
    delete impl_;
}

void JobSystem::start(const WorkerPlan& plan_in) {
    assert(!impl_->running.load() && "JobSystem already started");
    assert(!plan_in.worker_cores.empty() && "WorkerPlan must have at least one core");

    // Defensive: the assert above is compiled out in release. An empty
    // plan (topology detection degenerating to nothing on an exotic VM)
    // would make worker_count()==0 and turn the submit/steal
    // `% worker_count` into an integer divide-by-zero crash. Synthesize a
    // single core-0 Performance worker so the system always has >=1.
    WorkerPlan fallback;
    if (plan_in.worker_cores.empty()) {
        fallback.worker_cores      = { 0u };
        fallback.worker_numa_nodes = { 0u };
        fallback.worker_tiers      = { WorkerTier::Performance };
        fallback.perf_worker_count = 1u;
    }
    const WorkerPlan& plan = plan_in.worker_cores.empty() ? fallback : plan_in;

    impl_->workers          = std::vector<Worker>(plan.worker_cores.size());
    impl_->perf_count       = plan.perf_worker_count;
    impl_->general_count    = plan.general_worker_count;
    impl_->background_count = plan.background_worker_count;

    const bool has_tiers = plan.worker_tiers.size() == plan.worker_cores.size();
    const bool has_numa  = plan.worker_numa_nodes.size() == plan.worker_cores.size();
    for (u32 i = 0; i < plan.worker_cores.size(); ++i) {
        Worker& w      = impl_->workers[i];
        w.worker_id    = i;
        w.logical_core = plan.worker_cores[i];
        w.numa_node    = has_numa ? plan.worker_numa_nodes[i] : 0u;
        w.tier         = has_tiers ? plan.worker_tiers[i]
                                   : (i < plan.perf_worker_count
                                        ? WorkerTier::Performance
                                        : WorkerTier::General);
        w.inbox.reserve(4 * 1024);
    }

    impl_->running.store(true, std::memory_order_release);
    for (u32 i = 0; i < plan.worker_cores.size(); ++i) {
        impl_->workers[i].thread = std::thread([this, i]() { impl_->worker_main(i); });
    }
}

void JobSystem::shutdown() {
    impl_->running.store(false, std::memory_order_release);
    // Wake all sleepers so they observe running=false and exit.
    for (auto& w : impl_->workers) {
        std::lock_guard<std::mutex> lg(w.sleep_mtx);
        w.sleep_cv.notify_all();
    }
    for (auto& w : impl_->workers) {
        if (w.thread.joinable()) w.thread.join();
    }
    impl_->workers.clear();
}

Counter* JobSystem::acquire_counter(i32 initial_value) {
    Counter* c = impl_->acquire_counter_internal();
    if (c != nullptr) c->value.store(initial_value, std::memory_order_relaxed);
    return c;
}

Counter* JobSystem::submit(JobFunc fn, void* user_data, Counter* counter) {
    Counter* c = (counter != nullptr) ? counter : impl_->acquire_counter_internal();
    c->value.fetch_add(1, std::memory_order_relaxed);

    Job* j = impl_->acquire_job();
    j->fn        = fn;
    j->user_data = user_data;
    j->counter   = c;

    u32 wid = current_worker_id();
    if (wid != static_cast<u32>(-1)) {
        // Internal — push onto own deque (single-owner ✓). If the deque
        // is momentarily full, fall back to our own inbox instead of
        // overwriting a live slot; try_drain_own_inbox reclaims it once
        // the deque drains.
        if (!impl_->workers[wid].jobs.push(j)) {
            std::lock_guard<std::mutex> lg(impl_->workers[wid].inbox_mtx);
            impl_->workers[wid].inbox.push_back(j);
        }
        // Wake a random other worker in case it's idle, to encourage stealing.
        const u32 n = static_cast<u32>(impl_->workers.size());
        if (n > 1) {
            u32 wake = (wid + 1) % n;
            impl_->wake_worker(wake);
        }
    } else {
        // External — round-robin to a worker's inbox.
        const u32 n = static_cast<u32>(impl_->workers.size());
        u32 target = impl_->external_rr.fetch_add(1, std::memory_order_relaxed) % n;
        {
            std::lock_guard<std::mutex> lg(impl_->workers[target].inbox_mtx);
            impl_->workers[target].inbox.push_back(j);
        }
        impl_->wake_worker(target);
    }

    return c;
}

void JobSystem::wait(Counter* counter) {
    if (counter == nullptr) return;
    if (counter->value.load(std::memory_order_acquire) == 0) return;

    u32 wid = current_worker_id();
    if (wid != static_cast<u32>(-1)) {
        // From a job fiber — park self on the counter and yield to scheduler.
        Worker& self = impl_->workers[wid];
        FiberSlot* me = self.current_fiber.load(std::memory_order_acquire);
        assert(me != nullptr);

        impl_->park_fiber(counter, me);

        // Race-safe wake: if value already hit 0, sweep idempotently. If our
        // park is in the (perhaps already-empty) waiter list, this either
        // pushes us to ready_fibers ourselves, or is a no-op because the
        // earlier waker already pushed us.
        if (counter->value.load(std::memory_order_acquire) == 0) {
            impl_->wake_counter_waiters(counter);
        }

        impl_->switch_to(me->native, self.scheduler_fiber);
        // Resumed by scheduler from ready_fibers.
        return;
    }

    // External thread — sleep on the counter's atomic via C++20 futex
    // wait/notify. Maps to WaitOnAddress on Windows, FUTEX_WAIT on Linux.
    //
    // Previous implementation busy-spun std::this_thread::yield() which
    // burned a full core whenever the main thread waited on per-frame
    // jobs. With ~50 wait calls per frame and tasks completing in tens
    // of microseconds each, that loop ate ~10-20 ms / frame in pure spin
    // overhead — and starved the workers we were waiting on by competing
    // for CPU time.
    //
    // Pairing: workers in fiber_main_loop call counter->value.notify_all()
    // when the value transitions to 0 (see the fetch_sub site there).
    for (;;) {
        const i32 v = counter->value.load(std::memory_order_acquire);
        if (v <= 0) return;
        counter->value.wait(v, std::memory_order_acquire);
    }
}

void JobSystem::release(Counter* counter) {
    if (counter != nullptr) impl_->release_counter_internal(counter);
}

u32 JobSystem::worker_count() const noexcept            { return static_cast<u32>(impl_->workers.size()); }
u32 JobSystem::perf_worker_count() const noexcept       { return impl_->perf_count; }
u32 JobSystem::general_worker_count() const noexcept    { return impl_->general_count; }
u32 JobSystem::background_worker_count() const noexcept { return impl_->background_count; }
u32 JobSystem::current_worker_id() noexcept             { return Impl::tls_worker_id; }

WorkerTier JobSystem::worker_tier(u32 wid) const noexcept {
    if (wid >= impl_->workers.size()) return WorkerTier::Performance;
    return impl_->workers[wid].tier;
}
u32 JobSystem::worker_numa_node(u32 wid) const noexcept {
    if (wid >= impl_->workers.size()) return 0u;
    return impl_->workers[wid].numa_node;
}
u32 JobSystem::worker_logical_core(u32 wid) const noexcept {
    if (wid >= impl_->workers.size()) return 0u;
    return impl_->workers[wid].logical_core;
}

JobSystem::JobSystemStats JobSystem::stats() const noexcept {
    JobSystemStats out{};
    const usize n = impl_->workers.size();
    out.per_worker.resize(n);
    for (usize i = 0; i < n; ++i) {
        const Worker& w = impl_->workers[i];
        WorkerStats& s = out.per_worker[i];
        s.jobs_executed  = w.jobs_executed_count .load(std::memory_order_relaxed);
        s.jobs_stolen    = w.jobs_stolen_count   .load(std::memory_order_relaxed);
        s.steal_attempts = w.steal_attempts_count.load(std::memory_order_relaxed);
        s.sleep_events   = w.sleep_events_count  .load(std::memory_order_relaxed);
        s.queue_depth    = w.jobs.size_approx();
        out.total_jobs_executed   += s.jobs_executed;
        out.total_jobs_stolen     += s.jobs_stolen;
        out.total_steal_attempts  += s.steal_attempts;
        if (s.queue_depth > out.max_queue_depth) out.max_queue_depth = s.queue_depth;
    }
    return out;
}

// ============================================================================
// derive_worker_plan
//
// Goal: light up every logical thread the host exposes, but classify each
// worker by *strength* so schedulers can route heavy work to strong cores
// and IO / async / telemetry to the weak ones.
//
// Tier rules:
//   Performance: primary thread (lowest os_id) of every physical core that
//                lives on the strongest cluster.
//                  - AMD with V-Cache: V-Cache CCD wins.
//                  - AMD multi-CCD (e.g. 9950X 2-CCD): cluster 0.
//                  - Intel hybrid: every P-core.
//                  - Other: cluster 0.
//   General    : primary thread of every other physical core.
//                  - AMD second CCD (NUMA-remote on desktop NPS=2 / server).
//                  - Intel hybrid: nothing here (E-cores are Background).
//   Background : SMT siblings + Intel E-cores.
//                  - SMT sibling is the *second* logical thread on a core
//                    whose primary already runs a Perf or General worker.
//                    Realistic SMT uplift over the primary thread alone is
//                    ~15-30% on memory-stalled workloads, much less when
//                    the primary is compute-bound — hence "background".
//                  - Intel E-cores are full physical cores but slower
//                    per-clock; they share a cluster among themselves.
//
// Within each tier, workers are emitted ordered by (numa_node, cluster_id,
// os_id). That way pinning indices 0..N-1 keeps a contiguous run on the
// lowest-numbered NUMA node, which matches how most callers slice work.
// ============================================================================
namespace {

struct CoreSummary {
    u32 primary_os_id;          // os_id of the core's *first* logical thread
    std::vector<u32> sibling_os_ids;
    u32 cluster_id;
    u32 numa_node;
    topology::CoreClass core_class;
};

std::vector<CoreSummary> summarise_physical_cores(const topology::CpuInfo& info) {
    std::vector<CoreSummary> out(info.physical_core_count);
    for (auto& s : out) s.primary_os_id = static_cast<u32>(-1);

    for (const auto& lc : info.logical_cores) {
        if (lc.physical_core_id >= out.size()) continue;
        CoreSummary& cs = out[lc.physical_core_id];
        if (cs.primary_os_id == static_cast<u32>(-1) || lc.os_id < cs.primary_os_id) {
            // Demote the previous "primary" to sibling and adopt this one.
            if (cs.primary_os_id != static_cast<u32>(-1)) {
                cs.sibling_os_ids.push_back(cs.primary_os_id);
            }
            cs.primary_os_id = lc.os_id;
            cs.cluster_id   = lc.cluster_id;
            cs.numa_node    = lc.numa_node;
            cs.core_class   = lc.core_class;
        } else {
            cs.sibling_os_ids.push_back(lc.os_id);
        }
    }
    // Drop any unfilled entries (shouldn't happen but defensive).
    out.erase(std::remove_if(out.begin(), out.end(),
                  [](const CoreSummary& cs) { return cs.primary_os_id == static_cast<u32>(-1); }),
              out.end());
    return out;
}

void append_worker(WorkerPlan& plan, u32 os_id, u32 numa, WorkerTier tier) {
    plan.worker_cores.push_back(os_id);
    plan.worker_numa_nodes.push_back(numa);
    plan.worker_tiers.push_back(tier);
}

// Stable sort key: NUMA node, then cluster, then os_id. Keeps each NUMA
// node's workers contiguous in the plan vector.
auto core_order = [](const CoreSummary& a, const CoreSummary& b) noexcept {
    if (a.numa_node  != b.numa_node)  return a.numa_node  < b.numa_node;
    if (a.cluster_id != b.cluster_id) return a.cluster_id < b.cluster_id;
    return a.primary_os_id < b.primary_os_id;
};

}  // namespace

WorkerPlan derive_worker_plan(const topology::CpuInfo& info) {
    WorkerPlan plan{};
    if (info.logical_cores.empty()) return plan;

    auto cores = summarise_physical_cores(info);
    std::sort(cores.begin(), cores.end(), core_order);

    const i32 vcache_cluster = topology::v_cache_cluster_index(info);

    auto is_perf_core = [&](const CoreSummary& cs) {
        if (vcache_cluster >= 0) return cs.cluster_id == static_cast<u32>(vcache_cluster);
        if (info.is_hybrid)      return cs.core_class == topology::CoreClass::Performance;
        return cs.cluster_id == 0u;
    };
    auto is_efficiency_core = [&](const CoreSummary& cs) {
        return info.is_hybrid && cs.core_class == topology::CoreClass::Efficiency;
    };

    // 1) Performance tier: primary thread of every "strong" physical core.
    for (const auto& cs : cores) {
        if (is_perf_core(cs)) {
            append_worker(plan, cs.primary_os_id, cs.numa_node, WorkerTier::Performance);
        }
    }
    plan.perf_worker_count = static_cast<u32>(plan.worker_cores.size());

    // 2) General tier: primary thread of every other non-E-core physical core.
    //    On Intel hybrid this is empty (everything is either P-core perf
    //    or E-core background). On AMD multi-CCD this is the secondary CCD.
    const u32 before_general = static_cast<u32>(plan.worker_cores.size());
    for (const auto& cs : cores) {
        if (is_perf_core(cs)) continue;
        if (is_efficiency_core(cs)) continue;
        append_worker(plan, cs.primary_os_id, cs.numa_node, WorkerTier::General);
    }
    plan.general_worker_count = static_cast<u32>(plan.worker_cores.size()) - before_general;

    // 3) Background tier: SMT siblings of every primary core, plus E-cores
    //    (E-cores get their primary thread here, AND any siblings — Intel
    //    Atom currently doesn't ship SMT but the loop doesn't care).
    const u32 before_bg = static_cast<u32>(plan.worker_cores.size());
    for (const auto& cs : cores) {
        if (is_efficiency_core(cs)) {
            append_worker(plan, cs.primary_os_id, cs.numa_node, WorkerTier::Background);
        }
        // Sort siblings so multi-SMT (SMT4+) machines emit consistently.
        std::vector<u32> sibs = cs.sibling_os_ids;
        std::sort(sibs.begin(), sibs.end());
        for (u32 sib : sibs) {
            append_worker(plan, sib, cs.numa_node, WorkerTier::Background);
        }
    }
    plan.background_worker_count = static_cast<u32>(plan.worker_cores.size()) - before_bg;

    return plan;
}

WorkerPlan derive_worker_plan_physical_only(const topology::CpuInfo& info) {
    // Same structure but skip step 3 — leaves SMT siblings idle so the OS
    // can park other processes there. Useful for headless servers that share
    // the box, or for benchmarking against the no-SMT baseline.
    WorkerPlan plan{};
    if (info.logical_cores.empty()) return plan;

    auto cores = summarise_physical_cores(info);
    std::sort(cores.begin(), cores.end(), core_order);

    const i32 vcache_cluster = topology::v_cache_cluster_index(info);
    auto is_perf_core = [&](const CoreSummary& cs) {
        if (vcache_cluster >= 0) return cs.cluster_id == static_cast<u32>(vcache_cluster);
        if (info.is_hybrid)      return cs.core_class == topology::CoreClass::Performance;
        return cs.cluster_id == 0u;
    };

    for (const auto& cs : cores) {
        if (is_perf_core(cs)) {
            append_worker(plan, cs.primary_os_id, cs.numa_node, WorkerTier::Performance);
        }
    }
    plan.perf_worker_count = static_cast<u32>(plan.worker_cores.size());

    const u32 before_general = static_cast<u32>(plan.worker_cores.size());
    for (const auto& cs : cores) {
        if (!is_perf_core(cs)) {
            append_worker(plan, cs.primary_os_id, cs.numa_node, WorkerTier::General);
        }
    }
    plan.general_worker_count = static_cast<u32>(plan.worker_cores.size()) - before_general;
    plan.background_worker_count = 0;
    return plan;
}

}  // namespace cardinal
