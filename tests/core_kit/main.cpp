// =============================================================================
// Cardinal — core kit regression suite. Exercises every primitive in
// cardinal::core's recently-grouped subdirs: InterLock + ThreadLock + Access
// + lock guards, fixed-capacity stack strings, concurrent / waitable /
// circular / priority queues, Time / CpuTime / CpuUsage, Stopwatch +
// RepeatableTimer, Directory + File round-trip, worker Thread + cooperative
// stop, SehManager + minidump. Exit 0 = all pass.
// =============================================================================

#include <cardinal/core/sync/lock.hpp>
#include <cardinal/core/sync/access.hpp>
#include <cardinal/core/sync/worker_thread.hpp>
#include <cardinal/core/container/queue.hpp>
#include <cardinal/core/clock/wall_time.hpp>
#include <cardinal/core/clock/stopwatch.hpp>
#include <cardinal/core/os/file.hpp>
#include <cardinal/core/os/directory.hpp>
#include <cardinal/core/os/seh.hpp>
#include <cardinal/core/string/fixed_string.hpp>
#include <cardinal/core/string/string_id.hpp>
#include <cardinal/core/string/string_builder.hpp>
#include <cardinal/core/rng.hpp>
#include <cardinal/core/noise.hpp>
#include <cardinal/core/small_vector.hpp>
#include <cardinal/core/dense_map.hpp>
#include <cardinal/core/sparse_set.hpp>
#include <cardinal/core/slot_map.hpp>
#include <cardinal/core/spsc_ring.hpp>
#include <cardinal/core/inplace_function.hpp>
#include <cardinal/core/flags.hpp>
#include <cardinal/core/containers.hpp>
#include <cardinal/core/log.hpp>
#include <cardinal/core/utility.hpp>

#include <chrono>
#include <thread>

namespace {

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("core_kit", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

using namespace cardinal::core;

// ---------------------------------------------------------------------------
// InterLock / ThreadLock / NullLock / lock guards
// ---------------------------------------------------------------------------
void test_interlock() {
    cardinal::i32 v32 = 0;
    CHECK(InterLock::increment(v32) == 1);
    CHECK(InterLock::increment(v32) == 2);
    CHECK(InterLock::decrement(v32) == 1);
    CHECK(InterLock::exchange_add(v32, 5) == 1);   // returns previous
    CHECK(v32 == 6);
    CHECK(InterLock::exchange(v32, 42) == 6);
    CHECK(v32 == 42);
    cardinal::i32 prev = InterLock::exchange_compare(v32, 100, 42);
    CHECK(prev == 42 && v32 == 100);
    prev = InterLock::exchange_compare(v32, 999, 0);   // mismatch — no change
    CHECK(prev == 100 && v32 == 100);

    cardinal::i64 v64 = 0;
    CHECK(InterLock::increment(v64) == 1);
    CHECK(InterLock::exchange_add(v64, 1000) == 1);
    CHECK(v64 == 1001);
}

void test_thread_lock() {
    ThreadLock lk("test", "lk");
    CHECK(lk.is_opened());
    {
        ExclusiveLockGuard<ThreadLock> g(&lk);
    }
    {
        SharedLockGuard<ThreadLock> g(&lk);
    }
    {
        TryExclusiveLockGuard<ThreadLock> g(&lk);
        CHECK(g.is_locked());
    }
    NullLock nl;
    {
        ExclusiveLockGuard<NullLock> g(&nl);
    }
    CHECK(nl.is_opened());
}

// ---------------------------------------------------------------------------
// Access — attach/detach gate with embedded lock + AccessGuard RAII.
// ---------------------------------------------------------------------------
void test_access() {
    Access<ThreadLock> acc;
    CHECK(acc.is_opened());
    CHECK(acc.is_attachable());
    CHECK(acc.attach_count() == 0);

    // First attach — gate open, count goes 0 → 1, returns true.
    CHECK(acc.attach());
    CHECK(acc.attach_count() == 1);
    // Second attach — still open, count 1 → 2, returns true.
    CHECK(acc.attach());
    CHECK(acc.attach_count() == 2);

    // detach #1 — gate still open, so even though count drops the
    // "you may now destroy" signal is false (others may attach again).
    CHECK(acc.detach() == false);
    CHECK(acc.attach_count() == 1);

    // Producer begins shutdown: close the gate.
    acc.set_attachable(false);
    CHECK(!acc.is_attachable());

    // A new attach attempt now fails — count stays at 1.
    CHECK(acc.attach() == false);
    CHECK(acc.attach_count() == 1);

    // The remaining holder detaches. Gate is closed AND count reaches 0,
    // so detach returns true — caller is the last holder and may destroy.
    CHECK(acc.detach() == true);
    CHECK(acc.attach_count() == 0);

    // Recycle the access object back into use.
    acc.set_attachable_and_reset_attach_count();
    CHECK(acc.is_attachable());
    CHECK(acc.attach_count() == 0);

    // AccessGuard — RAII detach. Caller checks the attach() result before
    // constructing the guard .
    {
        const bool got = acc.attach();
        CHECK(got);
        AccessGuard<Access<ThreadLock>> g(acc);
        CHECK(acc.attach_count() == 1);
        // guard's dtor will fire detach
    }
    CHECK(acc.attach_count() == 0);

    // NullLock specialisation — single-thread path, lock ops are no-ops
    // but the gate + count semantics must still work.
    Access<NullLock> single_thread;
    CHECK(single_thread.attach());
    CHECK(single_thread.attach_count() == 1);
    single_thread.set_attachable(false);
    CHECK(single_thread.detach() == true);
}

// ---------------------------------------------------------------------------
// fixed-capacity stack strings
// ---------------------------------------------------------------------------
void test_string() {
    StringA<32> a;
    CHECK(a.is_null());
    a.set("Hello");
    CHECK(!a.is_null());
    CHECK(a.length() == 5u);
    a += " World";
    CHECK(a.length() == 11u);
    CHECK(::strcmp(a.c_str(), "Hello World") == 0);
    a.replace('o', '0');
    CHECK(::strcmp(a.c_str(), "Hell0 W0rld") == 0);
    a.reset();
    CHECK(a.is_null());
    a.format("v=%d.%d", 1, 2);
    CHECK(::strcmp(a.c_str(), "v=1.2") == 0);

    StringW<32> w;
    w.set(L"Hello");
    CHECK(w.length() == 5u);
    w += L" Wide";
    CHECK(w.length() == 10u);
    CHECK(::wcscmp(w.c_str(), L"Hello Wide") == 0);

    // Cross-conversion.
    StringA<32> a2;
    a2.set(L"WideToNarrow");
    CHECK(::strcmp(a2.c_str(), "WideToNarrow") == 0);

    StringW<32> w2;
    w2.set("NarrowToWide");
    CHECK(::wcscmp(w2.c_str(), L"NarrowToWide") == 0);
}

// ---------------------------------------------------------------------------
// queue suite — SyncQueue / WaitableQueue / StaticCircularQueue / PriorityQueue
// ---------------------------------------------------------------------------
void test_sync_queue() {
    SyncQueue<int> q;
    CHECK(q.open() == 0);
    q.push(10); q.push(20); q.push(30);
    CHECK(q.size() == 3u);
    int v = 0;
    CHECK(q.pop_front(v) == kQueueOk && v == 10);
    CHECK(q.pop_back(v)  == kQueueOk && v == 30);
    CHECK(q.size() == 1u);
    CHECK(q.is_exist(20));
    q.erase(20);
    CHECK(q.size() == 0u);
    CHECK(q.pop_front(v) == kQueueEmpty);
    q.close();
}

void test_dedup_queue() {
    NonDuplicableUnorderedSyncQueue<int> q;
    CHECK(q.open() == 0);
    q.push(1); q.push(2); q.push(1); q.push(2);
    CHECK(q.size() == 2u);
    int v = 0;
    CHECK(q.pop_front(v) == kQueueOk);
    CHECK(q.size() == 1u);
    q.close();
}

void test_waitable_queue() {
    WaitableQueue<int> q;
    CHECK(q.open() == 0);

    int v = 0;
    CHECK(q.pop_front(50, v) == kQueueEmpty);   // 50ms timeout, nothing to pop

    // Producer thread pushes after 50ms; consumer waits up to 500ms.
    std::thread producer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        q.push(42);
    });
    CHECK(q.pop_front(500, v) == kQueueOk && v == 42);
    producer.join();
    q.close();
}

void test_circular_queue() {
    StaticCircularQueue<int, 4> q;     // capacity 5 slots (4+1) — 4 live + 1 head
    CHECK(q.empty());
    q.push(1); q.push(2); q.push(3); q.push(4);
    CHECK(!q.empty());
    CHECK(q.front() == 1);
    // rbegin() returns the index of the most-recently-pushed element.
    // After pushing 1..4, push_index advanced to 4; rbegin should report
    // the slot one before push_index, i.e. slot 3 holding value 4.
    CHECK(q.get(q.rbegin()) == 4);
    CHECK(q.pop() == 1);
    CHECK(q.pop() == 2);
    // Queue now holds [3, 4]. Pushing 3 more drives the buffer to its
    // capacity (4 live) — the oldest (3) is overwritten on the 3rd push.
    // Final state: [4, 5, 6, 7].
    q.push(5); q.push(6); q.push(7);
    // After the wrap, rbegin should still report the latest push (7).
    CHECK(q.get(q.rbegin()) == 7);
    CHECK(q.pop() == 4);
    CHECK(q.pop() == 5);

    // rbegin behavior when push_index wraps to 0: push 4 more so push_index
    // lands at slot 0 (modulo 5), rbegin should be slot 4 (the last slot).
    StaticCircularQueue<int, 4> w;
    w.push(10); w.push(20); w.push(30); w.push(40); w.push(50);   // push_index = 0 after 5 pushes
    CHECK(w.get(w.rbegin()) == 50);
}

void test_priority_queue() {
    PriorityQueue<int> q;
    q.push(3); q.push(1); q.push(4); q.push(1); q.push(5);
    CHECK(q.size() == 5u);
    CHECK(q.is_exist(4));
    CHECK(q.top() == 5);     // max-heap default
    q.pop();
    CHECK(q.top() == 4);
    q.clear();
    CHECK(q.size() == 0u);
}

// ---------------------------------------------------------------------------
// CpuTime / CpuUsage
// ---------------------------------------------------------------------------
void test_time() {
    const cardinal::u64 utc = get_utc_64();
    CHECK(utc > 1700000000ull);   // after 2023-11
    const cardinal::u32 utc32 = get_utc_32();
    CHECK(utc32 != 0);

    Time t;          // current local time
    CHECK(t.year() >= 2025u);
    CHECK(t.month() >= 1u && t.month() <= 12u);
    CHECK(t.day() >= 1u && t.day() <= 31u);

    Time epoch;
    epoch.set(2025, 1, 1, 0, 0, 0);
    CHECK(epoch.year() == 2025u && epoch.month() == 1u && epoch.day() == 1u);
    CHECK(epoch.day_of_week() == DayOfWeek::Wednesday);   // 2025-01-01 = Wed

    Time later = epoch;
    later.add_seconds(3600);   // +1 hour
    // mktime(local-tm) -> UTC seconds, +3600 UTC seconds -> localtime(),
    // round-trip is timezone-invariant: any local 00:00:00 + 1 hour =
    // local 01:00:00 on the same day (no DST transition at Jan 1).
    CHECK(later.hour()  == 1u);
    CHECK(later.day()   == 1u);
    CHECK(later.month() == 1u);
    CHECK(later.year()  == 2025u);

    // 24 hours forward — same hour, next day.
    Time next_day = epoch;
    next_day.add_seconds(24u * 3600u);
    CHECK(next_day.hour() == 0u);
    CHECK(next_day.day()  == 2u);
    CHECK(next_day.day_of_week() == DayOfWeek::Thursday);   // 2025-01-02 = Thu

    Time future;
    future.set(2025, 1, 1, 0, 0, 0);
    future.add_day_and_set_time(5, 12, 0, 0);
    CHECK(future.day() == 6u && future.month() == 1u);
    CHECK(future.hour() == 12u);

    // Month-boundary rollover: Jan 30 + 5 days = Feb 4.
    Time month_roll;
    month_roll.set(2025, 1, 30, 0, 0, 0);
    month_roll.add_day_and_set_time(5, 12, 0, 0);
    CHECK(month_roll.month() == 2u && month_roll.day() == 4u);

    // Year-boundary rollover: Dec 30 2024 + 5 days = Jan 4 2025.
    Time year_roll;
    year_roll.set(2024, 12, 30, 0, 0, 0);
    year_roll.add_day_and_set_time(5, 0, 0, 0);
    CHECK(year_roll.year() == 2025u && year_roll.month() == 1u && year_roll.day() == 4u);

    // Leap-year boundary: 2024-02-28 + 1 day = 2024-02-29 (leap year).
    Time leap_into;
    leap_into.set(2024, 2, 28, 0, 0, 0);
    leap_into.add_day_and_set_time(1, 12, 0, 0);
    CHECK(leap_into.year() == 2024u && leap_into.month() == 2u && leap_into.day() == 29u);
    CHECK(leap_into.hour() == 12u);
    // And one more day rolls over to March.
    leap_into.add_day_and_set_time(1, 0, 0, 0);
    CHECK(leap_into.month() == 3u && leap_into.day() == 1u);

    // Non-leap year: 2025-02-28 + 1 day jumps straight to March 1.
    Time non_leap;
    non_leap.set(2025, 2, 28, 0, 0, 0);
    non_leap.add_day_and_set_time(1, 0, 0, 0);
    CHECK(non_leap.month() == 3u && non_leap.day() == 1u);

    // wait_milliseconds smoke test: target time 30 minutes from now should
    // be > 0 and well under 24h * 3600 * 1000 (24 hours in ms).
    Time now2;
    const cardinal::u16 minute_now = now2.minute();
    const cardinal::u16 minute_target = static_cast<cardinal::u16>((minute_now + 30u) % 60u);
    const cardinal::u32 wait_ms = now2.wait_milliseconds(minute_target, 0u);
    // 30 minutes is between 0 and ~24 hours regardless of which hour we're in.
    CHECK(wait_ms > 0u);
    CHECK(wait_ms <= 24u * 3600u * 1000u);

    CpuTime ct;
    CHECK(ct.reset() == 0);

    CpuUsage cu;
    (void)cu.reset_and_calculate_busy_rate();    // first call seeds; no return check
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const cardinal::i32 busy = cu.reset_and_calculate_busy_rate();
    CHECK(busy >= 0 && busy <= 100);
}

// ---------------------------------------------------------------------------
// stopwatch + timer — Stopwatch + RepeatableTimer
// ---------------------------------------------------------------------------
void test_stopwatch() {
    Stopwatch sw;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK(sw.elapsed_ms() >= 9u);    // tolerate one-tick jitter
    sw.restart();
    CHECK(sw.elapsed_ms() < 5u);
}

void test_repeatable_timer() {
    // ---- Phase 1: two one-shot timers, separate fire windows. -----------
    // Verifies the min-heap pops entries in chronological order and that a
    // one-shot entry is not re-armed.
    RepeatableTimer<cardinal::u32> rt;
    CHECK(rt.register_entry(/*id=*/1, /*delay_ms=*/30,  /*interval_ms=*/0) == 0);
    CHECK(rt.register_entry(/*id=*/2, /*delay_ms=*/120, /*interval_ms=*/0) == 0);
    CHECK(rt.register_entry(/*id=*/1, /*delay_ms=*/10,  /*interval_ms=*/0) == 183);   // duplicate
    rt.end_register();

    // Wait past id=1 but well before id=2 — only id=1 should fire.
    std::this_thread::sleep_for(std::chrono::milliseconds(70));
    cardinal::vector<cardinal::u32> fired;
    (void)rt.wait_milliseconds(fired);
    CHECK(fired.size() == 1u);
    if (fired.size() == 1u) CHECK(fired[0] == 1u);

    // Wait past id=2.
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    cardinal::vector<cardinal::u32> fired2;
    (void)rt.wait_milliseconds(fired2);
    CHECK(fired2.size() == 1u);
    if (fired2.size() == 1u) CHECK(fired2[0] == 2u);

    // ---- Phase 2: interval timer re-arms correctly. ---------------------
    // After ~120ms with delay=30 + interval=40, entry 9 should fire at
    // 30, 70, 110 — i.e. 3 times. Use size >= 2 to tolerate Windows
    // sleep_for jitter (15.6ms scheduler tick).
    RepeatableTimer<cardinal::u32> rt2;
    CHECK(rt2.register_entry(/*id=*/9, /*delay_ms=*/30, /*interval_ms=*/40) == 0);
    rt2.end_register();

    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    cardinal::vector<cardinal::u32> fired3;
    (void)rt2.wait_milliseconds(fired3);
    CHECK(fired3.size() >= 2u);
    for (auto id : fired3) CHECK(id == 9u);
}

// ---------------------------------------------------------------------------
// Directory — create / list / remove
// ---------------------------------------------------------------------------
void test_directory_make_and_remove() {
    // Use a unique tmp path inside cwd.
    const wchar_t* tmp = L"./.pa_test_dir";
    (void)Directory::remove(tmp);   // best-effort cleanup
    CHECK(Directory::make(tmp) == 0);

    // Make + list a sub-tree.
    CHECK(Directory::make(L"./.pa_test_dir/sub") == 0);

    Directory d;
    CHECK(d.begin(L"./.pa_test_dir") == 0);
    bool saw_sub = false;
    int  count   = 0;
    do {
        const auto& e = d.get();
        if (e.file_name == L"sub") saw_sub = true;
        ++count;
    } while (d.next() == 0);
    d.end();
    CHECK(count >= 1);
    CHECK(saw_sub);

    CHECK(Directory::remove(tmp) == 0);
}

// ---------------------------------------------------------------------------
// File — write + read round-trip
// ---------------------------------------------------------------------------
void test_file_round_trip() {
    const wchar_t* path = L"./.pa_test_file.bin";

    {
        SyncWriteFile w;
        CHECK(w.open(path) == 0);
        const cardinal::u32 data[] = { 0xDEADBEEFu, 0xCAFEBABEu, 0x12345678u };
        CHECK(w.write(data, sizeof(data)) == 0);
        CHECK(w.flush() == 0);
        w.close();
    }

    cardinal::u64 size = 0;
    CHECK(File::get_size(path, size) == 0);
    CHECK(size == 12u);

    {
        SyncReadFile r;
        CHECK(r.open(path, /*writable=*/false) == 0);
        cardinal::u32 buf[3] = {};
        cardinal::u32 sz = sizeof(buf);
        CHECK(r.read(buf, sz) == 0);
        CHECK(sz == 12u);
        CHECK(buf[0] == 0xDEADBEEFu);
        CHECK(buf[1] == 0xCAFEBABEu);
        CHECK(buf[2] == 0x12345678u);
        r.close();
    }
    // Clean up.
    (void)Directory::remove(path);
}

// ---------------------------------------------------------------------------
// worker Thread — start + cooperative stop
// ---------------------------------------------------------------------------
class TestThread : public Thread {
public:
    TestThread() : Thread(L"CoreKitTestThread", 0, false), counter_(0) {}
    cardinal::atomic<cardinal::i32> counter_;
protected:
    cardinal::i32 run() noexcept override {
        while (!stop_requested()) {
            counter_.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return 0;
    }
};

void test_thread() {
    TestThread t;
    CHECK(!t.is_started());
    CHECK(t.start(/*stack_size=*/0) == 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK(t.counter_.load() > 0);
    CHECK(t.stop() == 0);
    CHECK(t.wait(500) == 0);
    CHECK(!t.is_started());
}

// ---------------------------------------------------------------------------
// SehManager — set_handler / dump_mini (Windows only; non-Windows is no-op stub)
// ---------------------------------------------------------------------------
void test_seh() {
#if CARDINAL_PLATFORM_WINDOWS
    auto& mgr = SehManager::instance();
    CHECK(mgr.set_handler(L"./.pa_seh_", "test-user", false, nullptr) == 0);
    CHECK(mgr.is_set());
    mgr.set_dump_file_name(L"./.pa_seh_dump.dmp");
    const cardinal::i32 r = mgr.dump_mini(false);
    CHECK(r == 0);                                       // wrote successfully
    CHECK(::wcsstr(mgr.dump_file_name(), L".pa_seh_dump.dmp") != nullptr);
    mgr.dont_catch_exception();
    CHECK(!mgr.is_set());

    // Clean up the dump file.
    (void)Directory::remove(L"./.pa_seh_dump.dmp");
#endif
}

// Lifetime probe for SmallVector — counts live instances so the test can
// assert construct/destruct balance (no leaks, no double-frees) across spill,
// copy, move + scope exit.
struct Tracked {
    static inline int s_alive = 0;
    int v{0};
    Tracked()                 { ++s_alive; }
    Tracked(int x) : v(x)     { ++s_alive; }
    Tracked(const Tracked& o) : v(o.v) { ++s_alive; }
    Tracked(Tracked&& o) noexcept : v(o.v) { ++s_alive; }
    Tracked& operator=(const Tracked&) = default;
    Tracked& operator=(Tracked&&) noexcept = default;
    ~Tracked()                { --s_alive; }
};

void test_small_vector() {
    // Inline storage: no heap allocation until size exceeds N.
    cardinal::small_vector<int, 4> v;
    CHECK(v.empty());
    CHECK(v.is_inline());
    CHECK(v.capacity() == 4);
    v.push_back(1); v.push_back(2); v.push_back(3); v.push_back(4);
    CHECK(v.size() == 4);
    CHECK(v.is_inline());                       // still inline AT capacity N
    v.push_back(5);                             // spills to the heap
    CHECK(v.size() == 5);
    CHECK(!v.is_inline());
    CHECK(v.capacity() >= 5);
    CHECK(v[0] == 1 && v[4] == 5);
    CHECK(v.front() == 1 && v.back() == 5);

    bool threw = false;
    try { (void)v.at(99); } catch (const std::out_of_range&) { threw = true; }
    CHECK(threw);

    int sum = 0; for (int x : v) sum += x;
    CHECK(sum == 15);

    v.pop_back();
    CHECK(v.size() == 4 && v.back() == 4);

    // Copy is a deep, equal copy; mutating the copy doesn't touch the source.
    cardinal::small_vector<int, 4> c = v;
    CHECK(c == v);
    c.push_back(99);
    CHECK(c != v);

    // Move from a HEAP source steals the block; source left empty-valid.
    cardinal::small_vector<int, 2> h;
    for (int i = 0; i < 10; ++i) h.push_back(i);
    CHECK(!h.is_inline());
    cardinal::small_vector<int, 2> hm = cardinal::move(h);
    CHECK(hm.size() == 10 && hm[9] == 9);
    CHECK(h.empty());

    // Move from an INLINE source move-constructs each element.
    cardinal::small_vector<int, 4> s; s.push_back(7); s.push_back(8);
    CHECK(s.is_inline());
    cardinal::small_vector<int, 4> sm = cardinal::move(s);
    CHECK(sm.size() == 2 && sm[0] == 7 && sm[1] == 8);
    CHECK(s.empty());

    // resize grow/shrink + erase shift + clear.
    cardinal::small_vector<int, 4> r;
    r.resize(3, 5); CHECK(r.size() == 3 && r[2] == 5);
    r.resize(1);    CHECK(r.size() == 1);
    r.push_back(10); r.push_back(20); r.push_back(30);   // {5,10,20,30}
    r.erase(r.begin() + 1);                              // {5,20,30}
    CHECK(r.size() == 3 && r[1] == 20 && r[2] == 30);
    r.clear(); CHECK(r.empty());

    // Range erase [first,last): middle, empty no-op, to-end.
    cardinal::small_vector<int, 4> e;
    for (int i = 1; i <= 6; ++i) e.push_back(i);          // {1..6} (heap, cap 4)
    e.erase(e.begin() + 1, e.begin() + 4);               // remove {2,3,4} -> {1,5,6}
    CHECK(e.size() == 3 && e[0] == 1 && e[1] == 5 && e[2] == 6);
    e.erase(e.begin(), e.begin());                       // empty range = no-op
    CHECK(e.size() == 3);
    e.erase(e.begin() + 1, e.end());                     // remove {5,6} -> {1}
    CHECK(e.size() == 1 && e[0] == 1);

    // swap across inline (a) + heap (b) storage.
    cardinal::small_vector<int, 4> a{1, 2};
    cardinal::small_vector<int, 4> b{9, 8, 7, 6, 5};     // b is on the heap
    a.swap(b);
    CHECK(a.size() == 5 && a[0] == 9);
    CHECK(b.size() == 2 && b[0] == 1);

    // Lifetime balance: spill + copy + move + scope-exit leaves zero alive.
    Tracked::s_alive = 0;
    {
        cardinal::small_vector<Tracked, 2> t;
        t.emplace_back(); t.emplace_back(); t.emplace_back();   // spill to heap
        cardinal::small_vector<Tracked, 2> t2 = t;              // copy
        cardinal::small_vector<Tracked, 2> t3 = cardinal::move(t2);
        t.pop_back();
        CHECK(Tracked::s_alive > 0);
    }
    CHECK(Tracked::s_alive == 0);                              // no leaks / double-frees
}

void test_string_id() {
    using cardinal::StringId;

    // Literal ids are COMPILE-TIME constants.
    static_assert("world.undo"_sid == "world.undo"_sid, "literal id stable");
    static_assert("a"_sid != "b"_sid,                   "distinct literals differ");
    constexpr StringId k = "material.albedo"_sid;
    static_assert(k.valid(), "non-empty literal id is valid");

    // Literal, const char*, and cardinal::string of the SAME text all agree —
    // a runtime id matches a compile-time literal id.
    const StringId        lit  = "spawn.enemy"_sid;
    const StringId        cstr{"spawn.enemy"};
    const cardinal::string s   = "spawn.enemy";
    const StringId        run{s};
    CHECK(lit == cstr);
    CHECK(lit == run);
    CHECK(lit.value() == run.value());

    // Distinctness, default-invalid, empty-string-valid.
    CHECK(StringId("foo") != StringId("bar"));
    CHECK(!StringId().valid());
    CHECK(StringId("").valid());                 // "" hashes to the FNV basis
    CHECK(StringId() != StringId(""));

    // Usable as an unordered_map key; a runtime id finds a literal-keyed entry.
    cardinal::unordered_map<StringId, int> m;
    m[lit]        = 7;
    m["other"_sid] = 9;
    CHECK(m[run] == 7);
    CHECK(m.size() == 2);

    // Explicit u64 round-trip (e.g. (de)serialised ids).
    const StringId rebuilt{ static_cast<cardinal::u64>(lit) };
    CHECK(rebuilt == lit);
}

enum class TestPerm : cardinal::u32 { None = 0, Read = 1, Write = 2, Exec = 4 };
CARDINAL_ENABLE_FLAGS(TestPerm)

void test_flags() {
    using F = cardinal::Flags<TestPerm>;

    F f;
    CHECK(f.none() && !f.any() && !static_cast<bool>(f));
    CHECK(!f.has(TestPerm::None));               // "None = 0" never reads as present

    f.set(TestPerm::Read);
    CHECK(f.has(TestPerm::Read) && !f.has(TestPerm::Write));
    CHECK(f.any() && static_cast<bool>(f) && f.value() == 1u);

    // Enum-level operator| (via CARDINAL_ENABLE_FLAGS) builds a Flags.
    F rw = TestPerm::Read | TestPerm::Write;
    CHECK(rw.has(TestPerm::Read) && rw.has(TestPerm::Write) && !rw.has(TestPerm::Exec));
    CHECK(rw.has_all(TestPerm::Read | TestPerm::Write));
    CHECK(!rw.has_any(TestPerm::Exec));
    CHECK(rw.has_any(TestPerm::Read | TestPerm::Exec));      // Read present

    rw.clear(TestPerm::Read);
    CHECK(!rw.has(TestPerm::Read) && rw.has(TestPerm::Write));
    rw.toggle(TestPerm::Exec);
    CHECK(rw.has(TestPerm::Exec));
    rw.toggle(TestPerm::Exec);
    CHECK(!rw.has(TestPerm::Exec));

    // Operators: |, &, ^, ~, ==.
    const F a = TestPerm::Read, b = TestPerm::Write;
    CHECK((a | b).has_all(TestPerm::Read | TestPerm::Write));
    CHECK((a & b).none());                        // disjoint single bits
    CHECK(((a | b) & a) == a);
    CHECK((a ^ a).none());
    CHECK((~F()).any());                          // complement of empty has bits set
    CHECK(a == F(TestPerm::Read) && a != b);

    // Compile-time evaluation.
    static_assert((TestPerm::Read | TestPerm::Write).has(TestPerm::Read), "ct has");
    static_assert(F(TestPerm::Exec).value() == 4u, "ct value");
    static_assert(!F().any(), "ct empty");
}

void test_dense_map() {
    cardinal::dense_map<int, int> m;
    CHECK(m.empty() && m.size() == 0);

    // Insert new + duplicate (dup leaves existing value).
    auto [p1, ins1] = m.insert(1, 100);
    CHECK(ins1 && *p1 == 100);
    auto [p2, ins2] = m.insert(1, 999);
    CHECK(!ins2 && *p2 == 100);
    CHECK(m.size() == 1);

    // find / contains / operator[].
    CHECK(m.find(1) != nullptr && *m.find(1) == 100);
    CHECK(m.find(2) == nullptr);
    CHECK(m.contains(1) && !m.contains(2));
    m[2] = 200;
    CHECK(m.contains(2) && *m.find(2) == 200);
    CHECK(m[1] == 100);
    CHECK(m.size() == 2);

    // insert_or_assign overwrites.
    m.insert_or_assign(1, 111);
    CHECK(*m.find(1) == 111);

    // erase + re-find + tombstone reuse.
    CHECK(m.erase(2));
    CHECK(!m.contains(2));
    CHECK(!m.erase(2));
    CHECK(m.size() == 1);
    m[2] = 222;
    CHECK(*m.find(2) == 222 && m.size() == 2);

    // Stress: many inserts force multiple rehashes; all stay findable.
    cardinal::dense_map<int, int> big;
    const int N = 2000;
    for (int i = 0; i < N; ++i) big[i] = i * 3;
    CHECK(big.size() == static_cast<cardinal::usize>(N));
    bool all_found = true;
    for (int i = 0; i < N; ++i) {
        const int* v = big.find(i);
        if (v == nullptr || *v != i * 3) { all_found = false; break; }
    }
    CHECK(all_found);

    // Iteration visits each live entry exactly once.
    long long sum = 0; cardinal::usize count = 0;
    big.for_each([&](const int& k, int& v) { sum += v; ++count; (void)k; });
    long long expect = 0; for (int i = 0; i < N; ++i) expect += static_cast<long long>(i) * 3;
    CHECK(count == static_cast<cardinal::usize>(N));
    CHECK(sum == expect);

    cardinal::usize rc = 0;
    for (auto kv : big) { (void)kv; ++rc; }            // range-for over iterator
    CHECK(rc == static_cast<cardinal::usize>(N));

    // Erase the evens → tombstone-heavy → odds still present, evens gone.
    for (int i = 0; i < N; i += 2) big.erase(i);
    CHECK(big.size() == static_cast<cardinal::usize>(N / 2));
    bool parity_ok = true;
    for (int i = 1; i < N; i += 2) if (!big.contains(i)) parity_ok = false;
    for (int i = 0; i < N; i += 2) if (big.contains(i)) parity_ok = false;
    CHECK(parity_ok);

    // Re-insert erased keys (tombstone reuse) → back to N, all findable.
    for (int i = 0; i < N; i += 2) big[i] = i * 3;
    CHECK(big.size() == static_cast<cardinal::usize>(N));
    CHECK(big.find(0) != nullptr && *big.find(0) == 0);
    CHECK(big.find(N - 1) != nullptr && *big.find(N - 1) == (N - 1) * 3);

    // StringId keys (uses the std::hash<StringId> specialisation).
    cardinal::dense_map<cardinal::StringId, int> sm;
    sm["alpha"_sid] = 1;
    sm["beta"_sid]  = 2;
    CHECK(sm.size() == 2);
    CHECK(sm.find(cardinal::StringId("alpha")) != nullptr &&
          *sm.find(cardinal::StringId("alpha")) == 1);   // runtime id finds literal-keyed entry
    CHECK(sm.contains("beta"_sid));

    big.clear();
    CHECK(big.empty() && big.size() == 0);
}

void test_sparse_set() {
    cardinal::sparse_set<int> s;
    CHECK(s.empty() && s.size() == 0);
    CHECK(!s.contains(5));

    s.insert(5, 50);
    s.insert(1, 10);
    s.insert(100, 1000);
    CHECK(s.size() == 3);
    CHECK(s.contains(5) && s.contains(1) && s.contains(100));
    CHECK(s.get(5) != nullptr && *s.get(5) == 50);
    CHECK(s.get(7) == nullptr);

    // Overwrite existing key.
    s.insert(5, 55);
    CHECK(*s.get(5) == 55 && s.size() == 3);

    // Remove the middle key → swap-with-last keeps the others valid.
    CHECK(s.remove(1));
    CHECK(!s.contains(1) && s.size() == 2);
    CHECK(*s.get(5) == 55 && *s.get(100) == 1000);
    CHECK(!s.remove(1));

    // Dense iteration covers exactly the remaining entries.
    long long ks = 0, vs = 0; cardinal::usize n = 0;
    s.for_each([&](cardinal::u32 k, int& v) { ks += k; vs += v; ++n; });
    CHECK(n == 2 && ks == (5 + 100) && vs == (55 + 1000));
    long long vspan = 0; for (int v : s.values()) vspan += v;
    CHECK(vspan == 55 + 1000);

    // Stress: many sparse keys → all findable; remove-evens parity holds.
    cardinal::sparse_set<int> big;
    const cardinal::u32 N = 2000;
    for (cardinal::u32 i = 0; i < N; ++i) big.insert(i * 3, static_cast<int>(i));
    CHECK(big.size() == N);
    bool all_ok = true;
    for (cardinal::u32 i = 0; i < N; ++i) {
        const int* p = big.get(i * 3);
        if (p == nullptr || *p != static_cast<int>(i)) { all_ok = false; break; }
    }
    CHECK(all_ok);
    for (cardinal::u32 i = 0; i < N; i += 2) big.remove(i * 3);
    CHECK(big.size() == N / 2);
    bool parity = true;
    for (cardinal::u32 i = 1; i < N; i += 2) if (!big.contains(i * 3)) parity = false;
    for (cardinal::u32 i = 0; i < N; i += 2) if (big.contains(i * 3))  parity = false;
    CHECK(parity);

    // clear + reuse.
    big.clear();
    CHECK(big.empty());
    big.insert(7, 77);
    CHECK(big.size() == 1 && *big.get(7) == 77);

    // Lifetime balance across insert/remove/scope-exit.
    Tracked::s_alive = 0;
    {
        cardinal::sparse_set<Tracked> ts;
        ts.insert(0, Tracked(1));
        ts.insert(2, Tracked(2));
        ts.insert(4, Tracked(3));
        CHECK(ts.size() == 3 && Tracked::s_alive == 3);
        ts.remove(2);                                   // swap-pop
        CHECK(ts.size() == 2 && Tracked::s_alive == 2);
    }
    CHECK(Tracked::s_alive == 0);
}

void test_rng() {
    // Determinism: same seed → identical sequence.
    cardinal::Rng a(42), b(42);
    bool same = true;
    for (int i = 0; i < 16; ++i) if (a.next_u64() != b.next_u64()) same = false;
    CHECK(same);

    // Different seed → different sequence (overwhelmingly likely).
    cardinal::Rng c(43), d(42);
    bool diff = false;
    for (int i = 0; i < 8; ++i) if (c.next_u64() != d.next_u64()) { diff = true; break; }
    CHECK(diff);

    // reseed restarts the stream.
    cardinal::Rng e(7);
    const cardinal::u64 e0 = e.next_u64();
    e.reseed(7);
    CHECK(e.next_u64() == e0);

    // Unit-interval reals stay in [0,1).
    cardinal::Rng r(123);
    bool in01 = true, f01 = true;
    for (int i = 0; i < 2000; ++i) {
        const double x = r.next_double();
        if (x < 0.0 || x >= 1.0) in01 = false;
        const float y = r.next_float();
        if (y < 0.0f || y >= 1.0f) f01 = false;
    }
    CHECK(in01 && f01);

    // Inclusive integer range: in bounds + both ends reachable.
    int mn = 1000, mx = -1000; bool inb = true;
    for (int i = 0; i < 20000; ++i) {
        const int v = r.range(0, 9);
        if (v < 0 || v > 9) inb = false;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    CHECK(inb && mn == 0 && mx == 9);
    CHECK(r.range(5, 5) == 5);
    CHECK(r.range(3, 2) == 3);            // hi <= lo → lo

    // Float range in [lo, hi).
    bool frb = true;
    for (int i = 0; i < 2000; ++i) {
        const float v = r.range_f(2.0f, 5.0f);
        if (v < 2.0f || v >= 5.0f) frb = false;
    }
    CHECK(frb);

    // next_bool yields both; chance() edges.
    bool sawT = false, sawF = false;
    for (int i = 0; i < 200; ++i) { if (r.next_bool()) sawT = true; else sawF = true; }
    CHECK(sawT && sawF);
    CHECK(!r.chance(0.0));                // [0,1) draw is never < 0
    CHECK(r.chance(1.0));                 // and always < 1
}

void test_noise() {
    namespace nz = cardinal::noise;

    // hash white noise: deterministic + position/seed sensitive.
    CHECK(nz::hash2_u32(3, 7, 0) == nz::hash2_u32(3, 7, 0));     // same coord+seed
    CHECK(nz::hash2_u32(3, 7, 0) != nz::hash2_u32(7, 3, 0));     // x/y not symmetric
    CHECK(nz::hash2_u32(3, 7, 0) != nz::hash2_u32(3, 7, 1));     // seed matters
    CHECK(nz::hash3_u32(1, 2, 3, 0) != nz::hash3_u32(1, 2, 4, 0));

    // hash2_unit in [0,1) over a grid.
    bool unit_ok = true;
    for (int y = -8; y < 8; ++y)
        for (int x = -8; x < 8; ++x) {
            const float v = nz::hash2_unit(x, y, 5);
            if (v < 0.0f || v >= 1.0f) unit_ok = false;
        }
    CHECK(unit_ok);

    // value_noise_2d: in [0,1), equals the corner hash at integer coords,
    // and is continuous (a tiny step gives a tiny change).
    bool vn_ok = true;
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x) {
            const float v = nz::value_noise_2d(static_cast<float>(x) * 0.37f,
                                               static_cast<float>(y) * 0.37f, 9);
            if (v < 0.0f || v >= 1.0f) vn_ok = false;
        }
    CHECK(vn_ok);
    CHECK(cardinal::abs(nz::value_noise_2d(5.0f, 5.0f, 9) - nz::hash2_unit(5, 5, 9)) < 1e-5f);
    const float n0 = nz::value_noise_2d(5.0f,   5.0f, 9);
    const float n1 = nz::value_noise_2d(5.001f, 5.0f, 9);
    CHECK(cardinal::abs(n0 - n1) < 0.01f);        // continuity (no white-noise jumps)

    // value noise is NOT flat (different cells differ).
    CHECK(cardinal::abs(nz::value_noise_2d(2.5f, 2.5f, 9) -
                        nz::value_noise_2d(40.5f, 17.5f, 9)) > 1e-4f);

    // fBm: deterministic (same inputs → identical output), in [0,1).
    CHECK(nz::fbm_2d(1.5f, 2.5f, 3, 5) == nz::fbm_2d(1.5f, 2.5f, 3, 5));
    bool fbm_ok = true;
    for (int i = 0; i < 500; ++i) {
        const float fx = static_cast<float>(i) * 0.13f;
        const float v = nz::fbm_2d(fx, fx * 0.5f, 11, 4);
        if (v < 0.0f || v >= 1.0f) fbm_ok = false;
    }
    CHECK(fbm_ok);
}

void test_string_builder() {
    cardinal::StringBuilder sb;
    CHECK(sb.empty() && sb.size() == 0);

    sb.append("hello").append(' ').append("world");
    CHECK(sb.str() == "hello world" && sb.size() == 11);

    sb.clear();
    sb.append_int(-42).append(' ').append_uint(7u).append(' ').append(true);
    CHECK(sb.str() == "-42 7 true");

    sb.clear(); sb.append_hex(0xdeadu, true);
    CHECK(sb.str() == "0xdead");
    sb.clear(); sb.append_hex(255u, false);
    CHECK(sb.str() == "ff");

    sb.clear(); sb.append_float(3.14159, 3);
    CHECK(sb.str() == "3.14");                       // %.3g → 3 sig figs

    // Fluent operator<< across types.
    cardinal::StringBuilder s2;
    s2 << "x=" << 10 << ", y=" << 2.5 << ", ok=" << true << '\n';
    CHECK(s2.str() == "x=10, y=2.5, ok=true\n");

    // append_line.
    cardinal::StringBuilder s3;
    s3.append_line("a").append_line("b");
    CHECK(s3.str() == "a\nb\n");

    // append_repeat.
    cardinal::StringBuilder s4;
    s4.append_repeat('-', 5);
    CHECK(s4.str() == "-----");

    // take() moves out + resets.
    cardinal::StringBuilder s5; s5.append("moved");
    const cardinal::string out = s5.take();
    CHECK(out == "moved" && s5.empty());

    // c_str round-trip.
    cardinal::StringBuilder s6; s6 << "cstr" << 1;
    CHECK(cardinal::string(s6.c_str()) == "cstr1");
}

void test_slot_map() {
    using SM = cardinal::slot_map<int>;
    using H  = SM::handle_type;

    SM m;
    CHECK(m.empty() && m.size() == 0);
    const H a = m.insert(10);
    const H b = m.insert(20);
    const H c = m.insert(30);
    CHECK(m.size() == 3);
    CHECK(a.valid() && m.contains(a));
    CHECK(m.get(a) != nullptr && *m.get(a) == 10);
    CHECK(*m.get(b) == 20 && *m.get(c) == 30);

    // Erase b → its handle goes stale; others unaffected; no double-free.
    CHECK(m.erase(b));
    CHECK(!m.contains(b) && m.get(b) == nullptr);
    CHECK(!m.erase(b));
    CHECK(m.size() == 2 && *m.get(a) == 10 && *m.get(c) == 30);

    // Stale-handle safety: the next insert RECYCLES b's slot index with a
    // bumped generation. The old handle must NOT alias the new occupant.
    const H d = m.insert(40);
    CHECK(d.index == b.index);                  // slot index reused
    CHECK(d.generation != b.generation);        // generation bumped
    CHECK(m.get(d) != nullptr && *m.get(d) == 40);
    CHECK(m.get(b) == nullptr);                 // stale handle → null, not 40

    // for_each visits exactly the live entries.
    long long sum = 0; cardinal::usize n = 0;
    m.for_each([&](H, int& v) { sum += v; ++n; });
    CHECK(n == 3 && sum == (10 + 30 + 40));

    *m.get(a) = 11;
    CHECK(*m.get(a) == 11);

    // Stress: insert N, erase evens, survivors keep their handles, count exact.
    cardinal::slot_map<int> big;
    cardinal::vector<H> hs;
    const int N = 1000;
    for (int i = 0; i < N; ++i) hs.push_back(big.insert(i));
    for (int i = 0; i < N; i += 2) big.erase(hs[i]);
    CHECK(big.size() == static_cast<cardinal::usize>(N / 2));
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        const int* p = big.get(hs[i]);
        if (i % 2 == 0) { if (p != nullptr) ok = false; }       // erased → stale
        else            { if (p == nullptr || *p != i) ok = false; }
    }
    CHECK(ok);
    big.clear();
    CHECK(big.empty());

    // Lifetime: erase drops live_count; everything destroyed on scope exit.
    Tracked::s_alive = 0;
    {
        cardinal::slot_map<Tracked> tm;
        const auto h1 = tm.insert(Tracked(1));
        const auto h2 = tm.insert(Tracked(2));
        CHECK(tm.size() == 2);
        CHECK(tm.erase(h1));
        CHECK(tm.size() == 1 && !tm.contains(h1) && tm.contains(h2));
    }
    CHECK(Tracked::s_alive == 0);
}

void test_spsc_ring() {
    // Single-threaded: fill to capacity, full rejects, FIFO drain, empty.
    cardinal::core::SpscRing<int, 4> r;
    CHECK(r.empty() && r.capacity() == 4);
    CHECK(r.try_push(1) && r.try_push(2) && r.try_push(3) && r.try_push(4));
    CHECK(!r.try_push(5));                         // full (no overwrite, no block)
    CHECK(r.size_approx() == 4);
    int out = -1;
    CHECK(r.try_pop(out) && out == 1);
    CHECK(r.try_pop(out) && out == 2);
    CHECK(r.try_push(5));                          // room freed
    CHECK(r.try_pop(out) && out == 3);
    CHECK(r.try_pop(out) && out == 4);
    CHECK(r.try_pop(out) && out == 5);
    CHECK(!r.try_pop(out));                        // empty
    CHECK(r.empty());

    // Cross-thread SPSC: one producer pushes 0..M, one consumer pops M, must
    // see every value exactly once, in FIFO order, with the right sum. Only
    // the consumer thread writes the check locals → no data race on them.
    static constexpr int M = 100000;
    cardinal::core::SpscRing<int, 1024> q;
    std::thread producer([&] {
        for (int i = 0; i < M; ++i) while (!q.try_push(i)) { /* spin until room */ }
    });
    int       expect    = 0;
    int       got       = 0;
    long long sum       = 0;
    bool      order_ok  = true;
    std::thread consumer([&] {
        int v;
        while (got < M) {
            if (q.try_pop(v)) {
                if (v != expect) order_ok = false;     // strict FIFO
                ++expect; ++got; sum += v;
            }
        }
    });
    producer.join();
    consumer.join();
    CHECK(got == M);
    CHECK(order_ok);
    long long expect_sum = 0;
    for (int i = 0; i < M; ++i) expect_sum += i;
    CHECK(sum == expect_sum);                      // nothing lost or duplicated
    CHECK(q.empty());
}

void test_inplace_function() {
    using Fn = cardinal::inplace_function<int(int), 64>;

    Fn f = [](int x) { return x + 1; };
    CHECK(static_cast<bool>(f));
    CHECK(f(41) == 42);

    // Capture-by-value.
    const int base = 100;
    Fn g = [base](int x) { return base + x; };
    CHECK(g(5) == 105);

    // Copy is independent.
    Fn gc = g;
    CHECK(gc(5) == 105 && g(7) == 107);

    // Move leaves the source empty.
    Fn gm = cardinal::move(g);
    CHECK(static_cast<bool>(gm) && gm(1) == 101);
    CHECK(!static_cast<bool>(g));

    // Reassign + reset + nullptr.
    f = [](int x) { return x * 2; };
    CHECK(f(21) == 42);
    f = nullptr;
    CHECK(!static_cast<bool>(f));

    // Mutable lambda keeps state across calls (const operator() invokes it).
    cardinal::inplace_function<int()> counter = [n = 0]() mutable { return ++n; };
    CHECK(counter() == 1 && counter() == 2 && counter() == 3);

    // void return + captured side effect.
    int sink = 0;
    cardinal::inplace_function<void(int)> sinker = [&sink](int v) { sink += v; };
    sinker(10); sinker(5);
    CHECK(sink == 15);

    // Default-constructed is empty.
    cardinal::inplace_function<void()> empty;
    CHECK(!static_cast<bool>(empty));

    // The real use case: heap-free callback lists (no std::function nodes).
    cardinal::small_vector<cardinal::inplace_function<int(int)>, 4> fns;
    fns.push_back([](int x) { return x; });
    fns.push_back([](int x) { return x * x; });
    fns.push_back([](int x) { return x + 100; });
    CHECK(fns[0](7) == 7 && fns[1](7) == 49 && fns[2](7) == 107);

    // Lifetime: a captured Tracked is destroyed exactly once on reset/scope.
    Tracked::s_alive = 0;
    {
        cardinal::inplace_function<int()> holder = [t = Tracked(9)]() { return t.v; };
        CHECK(holder() == 9 && Tracked::s_alive >= 1);
        cardinal::inplace_function<int()> moved = cardinal::move(holder);
        CHECK(moved() == 9);
    }
    CHECK(Tracked::s_alive == 0);
}

}  // namespace

int main() {
    cardinal::log::infof("core_kit", "core kit regression suite");

    test_interlock();
    test_thread_lock();
    test_access();
    test_string();
    test_small_vector();
    test_string_id();
    test_flags();
    test_dense_map();
    test_sparse_set();
    test_rng();
    test_noise();
    test_string_builder();
    test_slot_map();
    test_spsc_ring();
    test_inplace_function();
    test_sync_queue();
    test_dedup_queue();
    test_waitable_queue();
    test_circular_queue();
    test_priority_queue();
    test_time();
    test_stopwatch();
    test_repeatable_timer();
    test_directory_make_and_remove();
    test_file_round_trip();
    test_thread();
    test_seh();

    cardinal::log::infof("core_kit", "checks=%d  failures=%d", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
