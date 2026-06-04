// =============================================================================
// Cardinal — pa::* regression suite. Exercises every modernised port of the
// Pearl Abyss PaLock / PaFile / PaDirectory / PaQueue / PaSeh / PaString /
// PaThread / PaTime / PaTimer surfaces. Exit 0 = all pass.
// =============================================================================

#include <cardinal/core/pa.hpp>
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
        cardinal::log::errorf("pa", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

using namespace cardinal::core::pa;

// ---------------------------------------------------------------------------
// PaLock / InterLock / ThreadLock / NullLock / lock guards
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
// PaAccess — attach/detach gate with embedded lock + AccessGuard RAII.
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
    // constructing the guard (PA convention).
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
// PaString
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
// PaQueue — SyncQueue / WaitableQueue / StaticCircularQueue / PriorityQueue
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
// PaTime / CpuTime / CpuUsage
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
// PaTimer — Stopwatch + RepeatableTimer
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
// PaDirectory — create / list / remove
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
// PaFile — write + read round-trip
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
// PaThread — start + cooperative stop
// ---------------------------------------------------------------------------
class TestThread : public Thread {
public:
    TestThread() : Thread(L"PaTestThread", 0, false), counter_(0) {}
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
// PaSeh — set_handler / dump_mini (Windows only; non-Windows is no-op stub)
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

}  // namespace

int main() {
    cardinal::log::infof("pa", "pa::* regression suite");

    test_interlock();
    test_thread_lock();
    test_access();
    test_string();
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

    cardinal::log::infof("pa", "checks=%d  failures=%d", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
