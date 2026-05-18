// =============================================================================
// Cardinal — deterministic Handle / SlotPool / IdGen regression suite.
//
// The stale-handle / use-after-free guard the whole engine leans on:
// physics BodyHandle/JointHandle/FieldHandle, every tag-typed handle,
// and the threadsafe monotonic id source. If SlotPool stops bumping the
// generation on slot reuse, an old handle silently resolves to the
// WRONG live object — memory-safety-adjacent, gameplay-corrupting, and
// invisible until it ships. Pure + header-only + deterministic; the
// concurrency check uses std::thread but asserts only schedule-
// independent set properties. Exit 0 = all pass.
// =============================================================================

#include <cardinal/core/id_gen.hpp>     // brings handle.hpp
#include <cardinal/core/log.hpp>

#include <thread>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace {

namespace co = cardinal::core;
using cardinal::u32;
using cardinal::u64;

struct TagA {};
struct TagB {};
using HA = co::Handle<TagA>;
using HB = co::Handle<TagB>;
using PoolA = co::SlotPool<TagA>;

// Compile-time tag safety: distinct tags → distinct, non-interchangeable
// handle types (a MeshHandle can't be passed where an EntityHandle is).
static_assert(!std::is_same_v<HA, HB>,
              "tag-typed handles must be distinct types");

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("idgentest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

// ---- Handle<Tag> ----------------------------------------------------
void test_handle() {
    HA def;
    CHECK(!def.valid());                       // gen 0 = invalid
    CHECK(!static_cast<bool>(def));
    CHECK(def.index == 0u && def.generation == 0u);

    HA a{ 5u, 2u };
    CHECK(a.valid() && static_cast<bool>(a));
    HA a2{ 5u, 2u };
    HA b{ 5u, 3u };                            // same index, diff gen
    HA c{ 6u, 2u };                            // diff index, same gen
    CHECK(a == a2);
    CHECK(a != b);                             // generation matters
    CHECK(a != c);                             // index matters
    CHECK(!(a == b) && !(a == c));

    // packed() ⇄ from_packed round-trip, incl. high bits.
    HA big{ 0xDEADBEEFu, 0x12345678u };
    const u64 p = big.packed();
    CHECK(p == ((static_cast<u64>(0x12345678u) << 32) | 0xDEADBEEFu));
    const HA un = HA::from_packed(p);
    CHECK(un == big && un.index == 0xDEADBEEFu &&
          un.generation == 0x12345678u);

    // std::hash specialisation: equal handles → equal hash; usable as a
    // key. (splitmix64 is deterministic — same input, same hash.)
    std::hash<HA> hsh;
    CHECK(hsh(a) == hsh(a2));
    CHECK(hsh(a) == hsh(HA{5u, 2u}));
    std::unordered_set<HA> set;
    set.insert(a);
    set.insert(b);
    set.insert(c);
    set.insert(a2);                            // dup of a — no-op
    CHECK(set.size() == sz(3));
    CHECK(set.count(a) == sz(1));
    CHECK(set.count(HA{99u, 99u}) == sz(0));
    set.erase(b);
    CHECK(set.count(b) == sz(0) && set.size() == sz(2));
}

// ---- SlotPool: basic alloc / free / accounting ----------------------
void test_pool_basic() {
    PoolA pool;
    CHECK(pool.capacity() == sz(0) && pool.live_count() == sz(0));

    const HA h0 = pool.allocate();
    CHECK(h0.index == 0u && h0.generation == 1u);   // gen 0 reserved
    CHECK(h0.valid() && pool.alive(h0));
    const HA h1 = pool.allocate();
    const HA h2 = pool.allocate();
    CHECK(h1.index == 1u && h2.index == 2u);
    CHECK(pool.capacity() == sz(3) && pool.live_count() == sz(3));
    CHECK(pool.alive(h0) && pool.alive(h1) && pool.alive(h2));

    // free a live handle → true; it's no longer alive; double-free → false.
    CHECK(pool.free(h1));
    CHECK(!pool.alive(h1));
    CHECK(pool.live_count() == sz(2));
    CHECK(pool.capacity() == sz(3));            // capacity doesn't shrink
    CHECK(!pool.free(h1));                      // double free rejected
    CHECK(!pool.free(HA{}));                    // invalid handle
    CHECK(!pool.free(HA{ 99u, 1u }));           // out-of-range index
    CHECK(!pool.alive(HA{}));                   // default never alive
    CHECK(!pool.alive(HA{ 0u, 999u }));         // gen mismatch
    CHECK(!pool.alive(HA{ 999u, 1u }));         // index OOB

    pool.clear();
    CHECK(pool.capacity() == sz(0) && pool.live_count() == sz(0));
    CHECK(!pool.alive(h0) && !pool.alive(h2));  // pre-clear handles dead
}

// ---- SlotPool: the use-after-free generation guard ------------------
void test_pool_reuse_guard() {
    PoolA pool;
    const HA a = pool.allocate();               // {0,1}
    CHECK(pool.free(a));
    const HA b = pool.allocate();               // reuses index 0
    CHECK(b.index == a.index);                  // slot recycled
    CHECK(b.generation == a.generation + 1u);   // generation bumped
    CHECK(a != b);
    CHECK(!pool.alive(a));                      // STALE handle is dead
    CHECK(pool.alive(b));                       // fresh handle is alive
    // Free/realloc again — generation keeps climbing, old stays dead.
    CHECK(pool.free(b));
    const HA cc = pool.allocate();
    CHECK(cc.index == 0u && cc.generation == 3u);
    CHECK(!pool.alive(a) && !pool.alive(b) && pool.alive(cc));
}

// ---- SlotPool: LIFO slot recycling is the documented contract -------
void test_pool_lifo() {
    PoolA pool;
    const HA a = pool.allocate();   // idx 0
    const HA b = pool.allocate();   // idx 1
    const HA c = pool.allocate();   // idx 2
    CHECK(pool.free(b));            // free_ = [1]
    CHECK(pool.free(a));            // free_ = [1,0]
    const HA r0 = pool.allocate();  // pop_back → idx 0 (last freed first)
    const HA r1 = pool.allocate();  // pop_back → idx 1
    CHECK(r0.index == 0u);
    CHECK(r1.index == 1u);
    CHECK(pool.capacity() == sz(3));           // no growth — both reused
    CHECK(pool.alive(c) && pool.alive(r0) && pool.alive(r1));
    CHECK(!pool.alive(a) && !pool.alive(b));   // originals stale
}

// ---- SlotPool: churn fuzz — invariants hold + deterministic ---------
u32 xs(u32& s) { s ^= s<<13; s ^= s>>17; s ^= s<<5; return s; }

void run_fuzz(u32 seed, std::vector<cardinal::usize>& trace) {
    PoolA pool;
    struct Held { HA h; bool alive; };
    std::vector<Held> held;
    u32 s = seed ? seed : 1u;
    for (int step = 0; step < 4000; ++step) {
        const bool do_alloc = held.empty() || (xs(s) & 1u);
        if (do_alloc) {
            const HA h = pool.allocate();
            held.push_back({ h, true });
        } else {
            const cardinal::usize k = xs(s) % held.size();
            if (held[k].alive) {
                const bool ok = pool.free(held[k].h);
                ::check_impl(ok, "fuzz: free(live) must succeed", __LINE__);
                held[k].alive = false;
            }
        }
        // Invariants after every op.
        cardinal::usize live = 0;
        for (const Held& e : held) {
            if (pool.alive(e.h) != e.alive)
                ::check_impl(false, "fuzz: alive() disagrees with model",
                             __LINE__);
            if (e.alive) ++live;
        }
        if (pool.live_count() != live)
            ::check_impl(false, "fuzz: live_count mismatch", __LINE__);
        trace.push_back(pool.capacity());
        trace.push_back(pool.live_count());
    }
    // No two simultaneously-alive handles compare equal.
    for (cardinal::usize i = 0; i < held.size(); ++i)
        for (cardinal::usize j = i + 1; j < held.size(); ++j)
            if (held[i].alive && held[j].alive && held[i].h == held[j].h)
                ::check_impl(false, "fuzz: two alive handles equal",
                             __LINE__);
}

void test_pool_fuzz() {
    std::vector<cardinal::usize> t1, t2;
    run_fuzz(0xC0FFEEu, t1);
    run_fuzz(0xC0FFEEu, t2);
    CHECK(t1 == t2);                            // fully deterministic
    CHECK(!t1.empty());
    // capacity is monotonically non-decreasing across the run.
    bool mono = true;
    for (cardinal::usize i = 2; i < t1.size(); i += 2)
        if (t1[i] < t1[i - 2]) mono = false;
    CHECK(mono);
}

// ---- IdGen ----------------------------------------------------------
void test_idgen() {
    co::IdGen<u64> g;                            // default first = 1
    CHECK(g.peek() == 1u);
    CHECK(g.next() == 1u);
    CHECK(g.next() == 2u);
    CHECK(g.next() == 3u);
    CHECK(g.peek() == 4u);                       // next value to vend

    co::IdGen<u32> g2{ 100u };
    CHECK(g2.next() == 100u);
    CHECK(g2.next() == 101u);

    // Concurrency: every value vended by next() is unique and the set
    // of N*M results is exactly [1, N*M] — schedule-independent, so the
    // assertion is deterministic even though thread order is not.
    constexpr int N = 8;
    constexpr int M = 20000;
    co::IdGen<u64> shared;                       // first = 1
    std::vector<u64> out(static_cast<cardinal::usize>(N * M), 0);
    std::vector<std::thread> ts;
    for (int t = 0; t < N; ++t) {
        ts.emplace_back([&, t]() {
            const int base = t * M;
            for (int i = 0; i < M; ++i)
                out[static_cast<cardinal::usize>(base + i)] = shared.next();
        });
    }
    for (auto& th : ts) th.join();

    std::vector<unsigned char> seen(
        static_cast<cardinal::usize>(N * M) + 1u, 0u);
    bool dup = false, oob = false;
    for (u64 v : out) {
        if (v < 1u || v > static_cast<u64>(N * M)) { oob = true; continue; }
        unsigned char& c = seen[static_cast<cardinal::usize>(v)];
        if (c) dup = true;
        c = 1u;
    }
    CHECK(!oob);                                 // all within [1, N*M]
    CHECK(!dup);                                 // no value vended twice
    bool all = true;
    for (int v = 1; v <= N * M; ++v)
        if (!seen[static_cast<cardinal::usize>(v)]) all = false;
    CHECK(all);                                  // full coverage
    CHECK(shared.peek() == static_cast<u64>(N * M) + 1u);
}

}  // namespace

int main() {
    test_handle();
    test_pool_basic();
    test_pool_reuse_guard();
    test_pool_lifo();
    test_pool_fuzz();
    test_idgen();

    if (g_fail == 0) {
        cardinal::log::infof("idgentest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("idgentest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
