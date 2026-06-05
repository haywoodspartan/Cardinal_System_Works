#pragma once

// =============================================================================
// cardinal::core::GlobalObjectManager — type-keyed service locator for the
// engine's process-lifetime singletons (Device, JobSystem, FrameTelemetry,
// AssetCache, etc.). Subsystems register themselves at startup; consumers
// look up by type — no static-init-order dance, no Meyers-singleton
// per subsystem, no extern globals.
//
// Two registration modes:
//   * register_object<T>(T*)            — type-keyed. One instance per T.
//     get<T>() returns it. Most subsystems use this.
//   * register_object<T>(key, T*)       — string-keyed. Multiple instances
//     of the same T discriminated by name (multi-device hosts, named
//     scenes, debug panels keyed by id, etc.).
//
// Ownership: the manager NEVER owns the registered instance. Subsystems
// register a pointer they own (member of a host object, static lifetime,
// etc.) and call unregister_object<T>() in their destructor. The
// type-erased pointer-only storage keeps the manager free of vtable /
// allocator dependencies on every registered type.
//
// Thread safety: register / unregister / get / clear are all guarded
// by an internal std::mutex. get() is a brief critical section (linear
// scan of a small vector — engine singletons are tens, not thousands).
// If the contention ever matters, switch to a flat_map<type_index,...>
// (the cardinal flat container — already shipped).
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/containers.hpp>    // cardinal::vector

#include <mutex>
#include <string>
#include <string_view>
#include <typeindex>

namespace cardinal::core {

class GlobalObjectManager {
public:
    // ---- singleton access ---------------------------------------------
    [[nodiscard]] static GlobalObjectManager& instance() noexcept;

    GlobalObjectManager(const GlobalObjectManager&)            = delete;
    GlobalObjectManager& operator=(const GlobalObjectManager&) = delete;

    // ---- type-keyed registration --------------------------------------
    // Registers `instance` under the type T. If T was already registered,
    // overwrites the prior pointer (caller is responsible for ensuring
    // there's no in-flight get<T>() consuming the old pointer).
    template <class T>
    void register_object(T* instance) noexcept {
        register_raw_(std::type_index(typeid(T)), {}, static_cast<void*>(instance));
    }

    template <class T>
    void unregister_object() noexcept {
        unregister_raw_(std::type_index(typeid(T)), {});
    }

    // Returns nullptr if no T is registered. Type-safe — never dereferences
    // a foreign pointer as T.
    template <class T>
    [[nodiscard]] T* get() noexcept {
        return static_cast<T*>(get_raw_(std::type_index(typeid(T)), {}));
    }

    // ---- string-keyed registration ------------------------------------
    // Same as the type-only path but lets multiple instances of the same
    // T coexist under different names (multi-device hosts, named scenes,
    // etc.). Type and key together identify an entry.
    template <class T>
    void register_object(std::string_view key, T* instance) noexcept {
        register_raw_(std::type_index(typeid(T)), std::string(key),
                      static_cast<void*>(instance));
    }

    template <class T>
    void unregister_object(std::string_view key) noexcept {
        unregister_raw_(std::type_index(typeid(T)), std::string(key));
    }

    template <class T>
    [[nodiscard]] T* get(std::string_view key) noexcept {
        return static_cast<T*>(get_raw_(std::type_index(typeid(T)),
                                        std::string(key)));
    }

    // ---- bulk reset ---------------------------------------------------
    // Drops every entry without invoking any destructor — manager doesn't
    // own the pointers. Mostly useful at shutdown for a final clean-slate.
    void clear() noexcept;

    [[nodiscard]] usize size() const noexcept;

private:
    GlobalObjectManager() noexcept = default;
    ~GlobalObjectManager() noexcept = default;

    void  register_raw_(std::type_index type, std::string key, void* instance) noexcept;
    void  unregister_raw_(std::type_index type, const std::string& key) noexcept;
    void* get_raw_       (std::type_index type, const std::string& key) noexcept;

    struct Entry {
        std::type_index type;
        std::string     key;       // empty = type-only registration
        void*           instance;
    };

    mutable std::mutex mutex_;
    // Small linear vector — engine singletons are tens, not thousands.
    // O(N) scan beats hash overhead at this size. Swap for flat_map if
    // contention ever shows up in a profile.
    cardinal::vector<Entry> entries_;
};

// ---------------------------------------------------------------------------
// Convenience free functions — for call sites that just want the locator
// without the singleton boilerplate.
// ---------------------------------------------------------------------------
template <class T>
inline void register_global(T* instance) noexcept {
    GlobalObjectManager::instance().register_object<T>(instance);
}

template <class T>
inline void register_global(std::string_view key, T* instance) noexcept {
    GlobalObjectManager::instance().register_object<T>(key, instance);
}

template <class T>
inline void unregister_global() noexcept {
    GlobalObjectManager::instance().unregister_object<T>();
}

template <class T>
inline void unregister_global(std::string_view key) noexcept {
    GlobalObjectManager::instance().unregister_object<T>(key);
}

template <class T>
[[nodiscard]] inline T* global() noexcept {
    return GlobalObjectManager::instance().get<T>();
}

template <class T>
[[nodiscard]] inline T* global(std::string_view key) noexcept {
    return GlobalObjectManager::instance().get<T>(key);
}

}  // namespace cardinal::core
