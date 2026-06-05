#pragma once

// =============================================================================
// cardinal::core::shared_object<T>(key, factory) — find-or-create
// flyweight factory. Multiple call sites that pass the same key get the
// same shared_ptr<T> instance; when the last shared_ptr drops, the entry
// is auto-evicted from the cache via a custom deleter.
//
// Use cases:
//   * Asset cache — shared_object<Texture>("rocks/granite.dds", load_from_disk)
//     returns the same Texture instance to every caller asking for that
//     filename. When the last consumer releases, the texture unloads.
//   * Named singletons — shared_object<Logger>("net", make_logger) gives
//     every net subsystem the same Logger without having to plumb it
//     through every constructor.
//   * Material / shader interning — same content hash → same compiled
//     program, no duplicate GPU upload.
//
// Mechanism:
//   The cache stores std::weak_ptr<T> keyed by string. Lookup tries
//   weak.lock(); on hit, returns the shared_ptr. On miss (or expired
//   weak), calls factory(), wraps the result in a shared_ptr with a
//   custom deleter that removes the cache entry before deleting the
//   object. The cache thus self-cleans — no manual eviction needed.
//
// Thread safety: the cache is guarded by an internal mutex. Factory is
// called WITHOUT the lock held (so a slow factory — disk read, GPU
// upload — doesn't block other lookups). If two threads race on the
// same miss-and-create, both factories run; the loser's result is
// discarded and both callers get the winner's shared_ptr.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <memory>           // std::shared_ptr / weak_ptr
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace cardinal::core {

namespace detail {

// One cache per T (instantiated by the shared_object<T> template). Holds
// a weak_ptr per key so live shared_ptrs stay alive without the cache
// owning them; expired weak entries get pruned on lookup.
template <class T>
class SharedObjectCache {
public:
    [[nodiscard]] static SharedObjectCache& instance() noexcept {
        static SharedObjectCache c;
        return c;
    }

    // Try to find an existing live instance. Returns nullptr on miss.
    [[nodiscard]] std::shared_ptr<T> try_get(const std::string& key) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = entries_.find(key);
        if (it == entries_.end()) return {};
        if (auto live = it->second.lock()) return live;
        // Expired — clean up.
        entries_.erase(it);
        return {};
    }

    // Race-safe insert: if another thread won the race, return THAT
    // instance and drop the caller's. Otherwise register and return.
    [[nodiscard]] std::shared_ptr<T> insert_or_get(const std::string& key,
                                                   std::shared_ptr<T> candidate) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        // Re-check after taking the lock — another thread may have inserted
        // while we ran the factory.
        const auto it = entries_.find(key);
        if (it != entries_.end()) {
            if (auto live = it->second.lock()) return live;   // we lost the race
            entries_.erase(it);
        }
        entries_.emplace(key, candidate);
        return candidate;
    }

    // Called from the custom deleter when the last shared_ptr drops.
    void erase(const std::string& key) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = entries_.find(key);
        if (it != entries_.end() && it->second.expired()) {
            entries_.erase(it);
        }
    }

    [[nodiscard]] usize size() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.size();
    }

    // Forcefully drop every entry (test-suite convenience). Live
    // shared_ptrs continue to function — the cache just forgets them.
    void clear() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.clear();
    }

private:
    SharedObjectCache() = default;
    mutable std::mutex                              mutex_;
    std::unordered_map<std::string, std::weak_ptr<T>> entries_;
};

// Custom deleter — runs when the last shared_ptr drops. Erases the
// (now-expired) cache entry, then deletes the object. Holds the key by
// value so the cache lookup works after the user's string_view is gone.
template <class T>
struct CacheEvictingDeleter {
    std::string key;
    void operator()(T* p) const noexcept {
        SharedObjectCache<T>::instance().erase(key);
        delete p;
    }
};

}  // namespace detail

// ---------------------------------------------------------------------------
// shared_object<T>(key, factory) — find-or-create-and-cache.
//
// Factory must be callable as `T*()` (raw pointer) OR `std::shared_ptr<T>()`.
// The raw-pointer overload is the simpler path; the shared_ptr overload
// is useful when the factory has its own deleter / aliasing setup that
// needs to be preserved.
// ---------------------------------------------------------------------------

template <class T, class Factory>
[[nodiscard]] std::shared_ptr<T> shared_object(std::string_view key, Factory&& factory) {
    auto& cache = detail::SharedObjectCache<T>::instance();
    const std::string skey(key);
    if (auto live = cache.try_get(skey)) return live;

    // Miss — run the factory outside the lock so a slow factory doesn't
    // serialise other lookups.
    using ResultType = std::invoke_result_t<Factory&>;
    std::shared_ptr<T> candidate;
    if constexpr (std::is_same_v<ResultType, std::shared_ptr<T>>) {
        // Factory returns shared_ptr directly — wrap it so our deleter
        // chains to the factory's by aliasing.
        std::shared_ptr<T> from_factory = std::forward<Factory>(factory)();
        if (!from_factory) return {};
        // Re-own with our cache-evicting deleter (aliased to keep the
        // factory's underlying deleter alive).
        T* raw = from_factory.get();
        candidate = std::shared_ptr<T>(raw,
            [moved = std::move(from_factory), skey](T*) noexcept {
                detail::SharedObjectCache<T>::instance().erase(skey);
                // `moved` falls out of scope here, triggering the
                // factory's deleter on the underlying object.
            });
    } else {
        // Factory returns raw T* — wrap directly.
        T* raw = std::forward<Factory>(factory)();
        if (raw == nullptr) return {};
        candidate = std::shared_ptr<T>(raw, detail::CacheEvictingDeleter<T>{skey});
    }

    return cache.insert_or_get(skey, std::move(candidate));
}

// Variant that default-constructs T when not present. Convenient for
// the named-singleton pattern (shared_object<Logger>("net")).
template <class T>
[[nodiscard]] std::shared_ptr<T> shared_object(std::string_view key) {
    return shared_object<T>(key, []() -> T* { return new T(); });
}

// ---- introspection / test helpers ----------------------------------------
template <class T>
[[nodiscard]] inline usize shared_object_cache_size() noexcept {
    return detail::SharedObjectCache<T>::instance().size();
}

template <class T>
inline void shared_object_cache_clear() noexcept {
    detail::SharedObjectCache<T>::instance().clear();
}

}  // namespace cardinal::core
