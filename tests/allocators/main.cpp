// =============================================================================
// Cardinal — LinearAllocator / PoolAllocator regression suite.
// =============================================================================

#include <cardinal/core/linear_allocator.hpp>
#include <cardinal/core/pool_allocator.hpp>
#include <cardinal/core/log.hpp>

#include <atomic>
#include <thread>
#include <vector>

namespace {

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("alloc", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

using cardinal::core::LinearAllocator;
using cardinal::core::ScopedArenaMarker;
using cardinal::core::PoolAllocator;

// ---------------------------------------------------------------------------
// LinearAllocator
// ---------------------------------------------------------------------------
void test_linear_basics() {
    LinearAllocator a(1024);
    CHECK(a.capacity() == 1024u);
    CHECK(a.used() == 0u);
    CHECK(a.remaining() == 1024u);

    void* p1 = a.allocate(32);
    CHECK(p1 != nullptr);
    CHECK(a.used() >= 32u);

    void* p2 = a.allocate(64, 16);   // 16-byte aligned
    CHECK(p2 != nullptr);
    CHECK(reinterpret_cast<cardinal::usize>(p2) % 16u == 0u);

    // Zero-byte alloc returns nullptr (no-op).
    void* p3 = a.allocate(0);
    CHECK(p3 == nullptr);

    a.reset();
    CHECK(a.used() == 0u);
}

void test_linear_exhaustion() {
    LinearAllocator a(64);
    // Allocate just under capacity.
    void* p1 = a.allocate(48);
    CHECK(p1 != nullptr);
    // The next allocation that doesn't fit must return nullptr.
    void* p2 = a.allocate(32);
    CHECK(p2 == nullptr);
    // Smaller allocation in the remaining space should still succeed.
    void* p3 = a.allocate(8);
    CHECK(p3 != nullptr);
}

struct Probe {
    int  a;
    int  b;
    Probe() : a(0), b(0) {}
    Probe(int x, int y) : a(x), b(y) {}
};

void test_linear_make() {
    LinearAllocator arena(1024);
    Probe* p = arena.make<Probe>(7, 11);
    CHECK(p != nullptr);
    CHECK(p->a == 7);
    CHECK(p->b == 11);

    Probe* arr = arena.make_array<Probe>(4);
    CHECK(arr != nullptr);
    CHECK(arr[0].a == 0 && arr[0].b == 0);   // default-constructed
    CHECK(arr[3].a == 0 && arr[3].b == 0);
}

void test_linear_marker_rewind() {
    LinearAllocator arena(1024);
    void* a1 = arena.allocate(64);
    CHECK(a1 != nullptr);
    const auto used_before_scope = arena.used();

    {
        ScopedArenaMarker scope(arena);
        void* a2 = arena.allocate(128);
        CHECK(a2 != nullptr);
        void* a3 = arena.allocate(32);
        CHECK(a3 != nullptr);
        CHECK(arena.used() > used_before_scope);
        // Marker should reflect the arena's offset at scope entry.
        CHECK(scope.marker().offset == used_before_scope);
    }
    // After scope: arena has rewound to used_before_scope.
    CHECK(arena.used() == used_before_scope);

    // After rewind, the allocator can reuse the rewound region.
    void* a4 = arena.allocate(128);
    CHECK(a4 != nullptr);
}

void test_linear_non_owning() {
    alignas(16) cardinal::u8 fixed_buffer[256];
    LinearAllocator a(fixed_buffer, sizeof(fixed_buffer));
    CHECK(a.capacity() == 256u);
    void* p = a.allocate(128);
    CHECK(p != nullptr);
    CHECK(p == fixed_buffer);   // first alloc at offset 0
    // Destructor must NOT free the caller-owned buffer — no crash here.
}

void test_linear_concurrent_alloc() {
    LinearAllocator arena(1024 * 1024);   // 1 MB
    constexpr int kThreads = 8;
    constexpr int kAllocsPerThread = 1000;

    std::atomic<int> ok_count{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < kAllocsPerThread; ++i) {
                void* p = arena.allocate(64, 16);
                if (p != nullptr && reinterpret_cast<cardinal::usize>(p) % 16u == 0u) {
                    ok_count.fetch_add(1);
                }
            }
        });
    }
    for (auto& th : threads) th.join();
    // Every alloc should have succeeded — 8 * 1000 * 64 = 512 KB < 1 MB.
    CHECK(ok_count.load() == kThreads * kAllocsPerThread);
    // And the arena should have consumed at least that many aligned bytes.
    CHECK(arena.used() >= static_cast<cardinal::usize>(kThreads * kAllocsPerThread * 64));
}

// ---------------------------------------------------------------------------
// PoolAllocator
// ---------------------------------------------------------------------------
struct Node {
    int value;
    int padding[3];  // total 16 bytes — well over pointer-size threshold
    Node() : value(0) {}
    explicit Node(int v) : value(v) {}
};

void test_pool_basics() {
    PoolAllocator<Node> pool(8);
    CHECK(pool.capacity() == 8u);
    CHECK(pool.used() == 0u);
    CHECK(pool.available() == 8u);

    Node* n1 = pool.make(42);
    CHECK(n1 != nullptr && n1->value == 42);
    CHECK(pool.owns(n1));
    CHECK(pool.used() == 1u);

    Node* n2 = pool.make(99);
    CHECK(n2 != nullptr && n2->value == 99);
    CHECK(n1 != n2);                          // distinct slots
    CHECK(pool.used() == 2u);

    pool.destroy(n1);
    CHECK(pool.used() == 1u);
    pool.destroy(n2);
    CHECK(pool.used() == 0u);
}

void test_pool_exhaustion_and_reuse() {
    PoolAllocator<Node> pool(4);
    Node* ns[4]{};
    for (int i = 0; i < 4; ++i) {
        ns[i] = pool.make(i);
        CHECK(ns[i] != nullptr);
    }
    CHECK(pool.available() == 0u);
    // Pool exhausted — next make must return nullptr.
    Node* extra = pool.make(99);
    CHECK(extra == nullptr);

    // Free one, verify it reuses the slot.
    pool.destroy(ns[1]);
    CHECK(pool.available() == 1u);
    Node* reused = pool.make(77);
    CHECK(reused != nullptr);
    CHECK(reused->value == 77);
    // Pool reuses freed slot (LIFO free list — most recently freed comes back).
    CHECK(reused == ns[1]);
    pool.destroy(ns[0]);
    pool.destroy(ns[2]);
    pool.destroy(ns[3]);
    pool.destroy(reused);
    CHECK(pool.used() == 0u);
}

void test_pool_destructor_called() {
    static int s_live_count = 0;
    struct Live {
        Live()  { ++s_live_count; }
        ~Live() { --s_live_count; }
        cardinal::u8 padding[16 - sizeof(int)];   // make sure >= sizeof(void*)
    };
    s_live_count = 0;
    PoolAllocator<Live> pool(4);
    Live* a = pool.make();
    Live* b = pool.make();
    CHECK(s_live_count == 2);
    pool.destroy(a);
    CHECK(s_live_count == 1);
    pool.destroy(b);
    CHECK(s_live_count == 0);
}

void test_pool_owns() {
    PoolAllocator<Node> pool(4);
    Node* p = pool.make(1);
    CHECK(pool.owns(p));
    Node stack_node(2);
    CHECK(!pool.owns(&stack_node));
    pool.destroy(p);
}

void test_pool_concurrent() {
    PoolAllocator<Node> pool(1024);
    constexpr int kThreads = 4;
    constexpr int kIters   = 200;
    std::atomic<int> alloc_failures{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            std::vector<Node*> local;
            local.reserve(kIters);
            for (int i = 0; i < kIters; ++i) {
                Node* n = pool.make(t * 10000 + i);
                if (n == nullptr) { alloc_failures.fetch_add(1); break; }
                local.push_back(n);
            }
            for (Node* n : local) pool.destroy(n);
        });
    }
    for (auto& th : threads) th.join();
    // 4 * 200 = 800 < 1024 — no allocation should have failed.
    CHECK(alloc_failures.load() == 0);
    CHECK(pool.used() == 0u);    // all released back
}

}  // namespace

int main() {
    cardinal::log::infof("alloc", "allocator regression suite");

    test_linear_basics();
    test_linear_exhaustion();
    test_linear_make();
    test_linear_marker_rewind();
    test_linear_non_owning();
    test_linear_concurrent_alloc();

    test_pool_basics();
    test_pool_exhaustion_and_reuse();
    test_pool_destructor_called();
    test_pool_owns();
    test_pool_concurrent();

    cardinal::log::infof("alloc", "checks=%d  failures=%d", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
