// =============================================================================
// Cardinal — GlobalObjectManager + shared_object regression suite.
// =============================================================================

#include <cardinal/core/global_object_manager.hpp>
#include <cardinal/core/shared_object.hpp>
#include <cardinal/core/log.hpp>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

namespace {

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("gobj", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

using cardinal::core::GlobalObjectManager;
using cardinal::core::shared_object;

// ---------------------------------------------------------------------------
// GlobalObjectManager — type-keyed singleton registry
// ---------------------------------------------------------------------------
struct FakeDevice    { int id = 0; };
struct FakeJobSys    { int workers = 4; };
struct FakeAssetMgr  { int loaded = 0; };

void test_type_keyed_registration() {
    auto& mgr = GlobalObjectManager::instance();
    mgr.clear();   // isolate this test
    CHECK(mgr.size() == 0u);

    FakeDevice dev{42};
    FakeJobSys js;

    mgr.register_object<FakeDevice>(&dev);
    mgr.register_object<FakeJobSys>(&js);
    CHECK(mgr.size() == 2u);

    // get<T>() returns the registered pointer.
    FakeDevice*   got_dev = mgr.get<FakeDevice>();
    FakeJobSys*   got_js  = mgr.get<FakeJobSys>();
    CHECK(got_dev == &dev);
    CHECK(got_js  == &js);
    CHECK(got_dev->id == 42);

    // Unregistered type — nullptr.
    CHECK(mgr.get<FakeAssetMgr>() == nullptr);

    // Re-register overwrites.
    FakeDevice dev2{99};
    mgr.register_object<FakeDevice>(&dev2);
    CHECK(mgr.size() == 2u);     // count unchanged (replaced, not added)
    CHECK(mgr.get<FakeDevice>() == &dev2);

    // Unregister drops the entry.
    mgr.unregister_object<FakeDevice>();
    CHECK(mgr.size() == 1u);
    CHECK(mgr.get<FakeDevice>() == nullptr);

    mgr.clear();
}

void test_string_keyed_registration() {
    auto& mgr = GlobalObjectManager::instance();
    mgr.clear();

    FakeDevice gpu0{0};
    FakeDevice gpu1{1};

    mgr.register_object<FakeDevice>("gpu0", &gpu0);
    mgr.register_object<FakeDevice>("gpu1", &gpu1);
    // Type-only registration coexists with string-keyed.
    FakeDevice primary{99};
    mgr.register_object<FakeDevice>(&primary);

    CHECK(mgr.size() == 3u);
    CHECK(mgr.get<FakeDevice>("gpu0") == &gpu0);
    CHECK(mgr.get<FakeDevice>("gpu1") == &gpu1);
    CHECK(mgr.get<FakeDevice>()       == &primary);

    // Different key — nullptr.
    CHECK(mgr.get<FakeDevice>("gpu99") == nullptr);

    mgr.unregister_object<FakeDevice>("gpu0");
    CHECK(mgr.size() == 2u);
    CHECK(mgr.get<FakeDevice>("gpu0") == nullptr);
    // Other entries untouched.
    CHECK(mgr.get<FakeDevice>("gpu1") == &gpu1);
    CHECK(mgr.get<FakeDevice>()       == &primary);

    mgr.clear();
}

void test_free_functions() {
    cardinal::core::GlobalObjectManager::instance().clear();
    FakeJobSys js;
    cardinal::core::register_global<FakeJobSys>(&js);
    CHECK(cardinal::core::global<FakeJobSys>() == &js);
    cardinal::core::unregister_global<FakeJobSys>();
    CHECK(cardinal::core::global<FakeJobSys>() == nullptr);
}

void test_concurrent_register() {
    auto& mgr = GlobalObjectManager::instance();
    mgr.clear();

    // 8 threads each register a distinct string-keyed FakeDevice;
    // verify they all land + are retrievable.
    struct Entry { FakeDevice dev; std::string key; };
    constexpr int kThreads = 8;
    std::vector<Entry> entries(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        entries[i].dev.id = i + 1000;
        entries[i].key    = "dev_" + std::to_string(i);
    }

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i]() {
            mgr.register_object<FakeDevice>(entries[i].key, &entries[i].dev);
        });
    }
    for (auto& th : threads) th.join();

    CHECK(mgr.size() == static_cast<cardinal::usize>(kThreads));
    for (int i = 0; i < kThreads; ++i) {
        FakeDevice* got = mgr.get<FakeDevice>(entries[i].key);
        CHECK(got == &entries[i].dev);
        CHECK(got->id == i + 1000);
    }
    mgr.clear();
}

// ---------------------------------------------------------------------------
// shared_object<T> — find-or-create with weak_ptr cache + auto-eviction
// ---------------------------------------------------------------------------
struct Resource {
    static int s_constructed;
    static int s_destroyed;
    std::string name;
    Resource()                             { ++s_constructed; }
    explicit Resource(std::string n) : name(std::move(n)) { ++s_constructed; }
    ~Resource()                            { ++s_destroyed; }
};
int Resource::s_constructed = 0;
int Resource::s_destroyed   = 0;

void test_shared_object_intern() {
    Resource::s_constructed = 0;
    Resource::s_destroyed   = 0;
    cardinal::core::shared_object_cache_clear<Resource>();

    // First call constructs.
    auto r1 = shared_object<Resource>("rocks", []() -> Resource* {
        return new Resource("rocks");
    });
    CHECK(r1 != nullptr);
    CHECK(r1->name == "rocks");
    CHECK(Resource::s_constructed == 1);
    CHECK(cardinal::core::shared_object_cache_size<Resource>() == 1u);

    // Second call with same key returns the SAME instance — factory NOT
    // invoked again.
    auto r2 = shared_object<Resource>("rocks", []() -> Resource* {
        return new Resource("SHOULD_NOT_RUN");
    });
    CHECK(r2.get() == r1.get());
    CHECK(Resource::s_constructed == 1);   // factory didn't re-fire
    CHECK(r2->name == "rocks");

    // Different key creates a distinct instance.
    auto r3 = shared_object<Resource>("water", []() -> Resource* {
        return new Resource("water");
    });
    CHECK(r3 != nullptr);
    CHECK(r3.get() != r1.get());
    CHECK(Resource::s_constructed == 2);
    CHECK(cardinal::core::shared_object_cache_size<Resource>() == 2u);
    CHECK(Resource::s_destroyed == 0);

    // Drop all references to "rocks" — cache entry must auto-evict and
    // the Resource destructor must fire.
    r1.reset();
    r2.reset();
    CHECK(Resource::s_destroyed == 1);
    CHECK(cardinal::core::shared_object_cache_size<Resource>() == 1u);

    // "water" still alive — count stays.
    CHECK(r3 != nullptr);
    CHECK(cardinal::core::shared_object_cache_size<Resource>() == 1u);

    // Re-request "rocks" — cache was evicted, so factory fires fresh.
    auto r4 = shared_object<Resource>("rocks", []() -> Resource* {
        return new Resource("rocks_v2");
    });
    CHECK(Resource::s_constructed == 3);
    CHECK(r4->name == "rocks_v2");

    // Cleanup.
    r3.reset();
    r4.reset();
    CHECK(Resource::s_destroyed == 3);
    CHECK(cardinal::core::shared_object_cache_size<Resource>() == 0u);
}

void test_shared_object_default_construct() {
    cardinal::core::shared_object_cache_clear<Resource>();
    Resource::s_constructed = 0;
    Resource::s_destroyed   = 0;

    // Default-construct variant.
    auto r = shared_object<Resource>("default_one");
    CHECK(r != nullptr);
    CHECK(Resource::s_constructed == 1);

    auto r2 = shared_object<Resource>("default_one");
    CHECK(r2.get() == r.get());

    r.reset();
    r2.reset();
    CHECK(Resource::s_destroyed == 1);
    CHECK(cardinal::core::shared_object_cache_size<Resource>() == 0u);
}

void test_shared_object_concurrent() {
    cardinal::core::shared_object_cache_clear<Resource>();
    Resource::s_constructed = 0;
    Resource::s_destroyed   = 0;

    // 8 threads race to acquire the SAME key. All must get the same
    // instance — factory may run more than once (loser threads discard),
    // but only ONE survives in the cache + only one final destructor.
    constexpr int kThreads = 8;
    std::vector<std::shared_ptr<Resource>> per_thread(kThreads);
    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    std::atomic<int> factory_calls{0};
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i]() {
            per_thread[i] = shared_object<Resource>("shared", [&]() -> Resource* {
                factory_calls.fetch_add(1);
                return new Resource("shared");
            });
        });
    }
    for (auto& th : threads) th.join();

    // All threads got the same pointer.
    const Resource* canonical = per_thread[0].get();
    CHECK(canonical != nullptr);
    for (int i = 1; i < kThreads; ++i) {
        CHECK(per_thread[i].get() == canonical);
    }
    // Factory may have fired multiple times (race), but the cache has
    // exactly one survivor — and exactly one Resource is alive.
    CHECK(factory_calls.load() >= 1);
    CHECK(cardinal::core::shared_object_cache_size<Resource>() == 1u);
    // Loser threads' Resources got destroyed already.
    CHECK(Resource::s_constructed - Resource::s_destroyed == 1);

    // Drop all references → final eviction + final dtor.
    for (auto& p : per_thread) p.reset();
    CHECK(cardinal::core::shared_object_cache_size<Resource>() == 0u);
    CHECK(Resource::s_constructed == Resource::s_destroyed);
}

}  // namespace

int main() {
    cardinal::log::infof("gobj", "GlobalObjectManager + shared_object regression suite");

    test_type_keyed_registration();
    test_string_keyed_registration();
    test_free_functions();
    test_concurrent_register();

    test_shared_object_intern();
    test_shared_object_default_construct();
    test_shared_object_concurrent();

    cardinal::log::infof("gobj", "checks=%d  failures=%d", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
