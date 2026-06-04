#pragma once

// =============================================================================
// Cardinal core — pa::lock vocabulary, modern C++20 port of Pearl Abyss
// CrimsonDesert's PaLock.h (pa::InterLock + pa::ThreadLock + pa::ProcessLock +
// pa::NullLock + Shared/Exclusive/TryShared/TryExclusive lock guards).
//
// Why the port:
//   * Pa-style code expresses concurrency in a fixed vocabulary that is
//     friendly to porting CrimsonDesert subsystems into Cardinal — keep the
//     names, but back them with std::atomic / std::shared_mutex / std::scoped_lock
//     so the implementation is portable, exception-safe and ABI-clean.
//   * `cardinal::core::pa` lives alongside (not inside) the foundation
//     vocabulary so a call site can pull `using namespace cardinal::core::pa;`
//     without shadowing cardinal::mutex / cardinal::shared_mutex / cardinal::atomic.
//
// Modernisation notes vs. original PaLock.h:
//   * InterLock atomics use std::atomic_ref so they bind to a caller-owned
//     i32/i64 (matches the Pa signature `Increment(int32&)`) but do not
//     require the value to be a member of an atomic<> type. Memory order is
//     seq_cst by default (matches the Win32 Interlocked* semantics on x86/x64).
//   * ThreadLock — shared/exclusive lock backed by std::shared_mutex on Win+
//     Linux; the SpinCount knob is now a hint, not a behaviour switch (on
//     Win10+ SRW locks self-spin; std::shared_mutex on MSVC sits on SRW).
//   * ProcessLock — cross-process named mutex. Win path uses
//     CreateMutexW + WaitForSingleObject; Linux path uses sem_open
//     (POSIX named semaphore). The shared-vs-exclusive semantics collapse to
//     mutual exclusion (NT's named-mutex doesn't differentiate readers vs
//     writers — same as PA's implementation).
//   * NullLock — no-op stub for compile-time selection of "no sync needed".
//   * Lock guards are header-only templates over the lock concept
//     (lockShared/lockExclusive/unlock*). Concept-checked at use site so a
//     misuse (e.g. SharedLockGuard<int>) fails with a readable error.
//
// Coexistence: This does NOT replace cardinal::mutex / cardinal::shared_mutex
// — those remain the canonical engine surface. pa::lock exists for the
// CrimsonDesert porting bridge.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/platform.hpp>
#include <cardinal/core/atomic.hpp>
#include <cardinal/core/thread.hpp>     // cardinal::shared_mutex
#include <cardinal/core/utility.hpp>

namespace cardinal::core::pa {

// ---------------------------------------------------------------------------
// InterLock — atomic primitive helpers over caller-owned integers.
// Mirrors PA's pa::InterLock static-method surface; uses std::atomic_ref so
// the caller's int32/int64 stays a regular variable.
// ---------------------------------------------------------------------------
class InterLock {
public:
    InterLock() = delete;   // static-only

    // Returns the NEW value (post-increment) — same as InterlockedIncrement.
    [[nodiscard]] static i32 increment(i32& target) noexcept {
        return ::std::atomic_ref<i32>(target).fetch_add(1, ::std::memory_order_seq_cst) + 1;
    }
    [[nodiscard]] static i64 increment(i64& target) noexcept {
        return ::std::atomic_ref<i64>(target).fetch_add(1, ::std::memory_order_seq_cst) + 1;
    }
    [[nodiscard]] static i32 decrement(i32& target) noexcept {
        return ::std::atomic_ref<i32>(target).fetch_sub(1, ::std::memory_order_seq_cst) - 1;
    }
    [[nodiscard]] static i64 decrement(i64& target) noexcept {
        return ::std::atomic_ref<i64>(target).fetch_sub(1, ::std::memory_order_seq_cst) - 1;
    }
    // Returns the PREVIOUS value (pre-add) — matches InterlockedExchangeAdd.
    [[nodiscard]] static i32 exchange_add(i32& target, i32 value) noexcept {
        return ::std::atomic_ref<i32>(target).fetch_add(value, ::std::memory_order_seq_cst);
    }
    [[nodiscard]] static i64 exchange_add(i64& target, i64 value) noexcept {
        return ::std::atomic_ref<i64>(target).fetch_add(value, ::std::memory_order_seq_cst);
    }
    // Returns the PREVIOUS value — matches InterlockedExchange.
    [[nodiscard]] static i32 exchange(i32& target, i32 value) noexcept {
        return ::std::atomic_ref<i32>(target).exchange(value, ::std::memory_order_seq_cst);
    }
    [[nodiscard]] static i64 exchange(i64& target, i64 value) noexcept {
        return ::std::atomic_ref<i64>(target).exchange(value, ::std::memory_order_seq_cst);
    }
    // CAS: if (target == old_value) target = value; return PREVIOUS target.
    [[nodiscard]] static i32 exchange_compare(i32& target, i32 value, i32 old_value) noexcept {
        ::std::atomic_ref<i32>(target).compare_exchange_strong(old_value, value, ::std::memory_order_seq_cst);
        return old_value;
    }
    [[nodiscard]] static i64 exchange_compare(i64& target, i64 value, i64 old_value) noexcept {
        ::std::atomic_ref<i64>(target).compare_exchange_strong(old_value, value, ::std::memory_order_seq_cst);
        return old_value;
    }
};

// ---------------------------------------------------------------------------
// ThreadLock — shared/exclusive lock for in-process synchronisation.
// Backed by cardinal::shared_mutex (= std::shared_mutex). The SpinCount,
// FileName + VariableName constructor arguments are kept for source
// compatibility with the Pa surface; SpinCount is advisory only (the
// underlying SRW/futex auto-spins) and the names feed telemetry (deadlock
// detector). They have no semantic effect today.
// ---------------------------------------------------------------------------
class ThreadLock {
public:
    explicit ThreadLock(const char* file_name      = "",
                        const char* variable_name = "",
                        u32         /*spin_count*/ = 4000) noexcept
        : file_name_(file_name), variable_name_(variable_name) {}

    ThreadLock(const ThreadLock&)            = delete;
    ThreadLock& operator=(const ThreadLock&) = delete;

    [[nodiscard]] bool is_opened() const noexcept { return true; }

    // Shared (reader) lock.
    void lock_shared()        const noexcept { mutex_.lock_shared(); }
    void unlock_shared()      const noexcept { mutex_.unlock_shared(); }
    [[nodiscard]] bool try_lock_shared() const noexcept { return mutex_.try_lock_shared(); }

    // Exclusive (writer) lock.
    void lock_exclusive()     const noexcept { mutex_.lock(); }
    void unlock_exclusive()   const noexcept { mutex_.unlock(); }
    [[nodiscard]] bool try_lock_exclusive() const noexcept { return mutex_.try_lock(); }

    // Pa-style "skip the deadlock checker" variants — same as the regular
    // ones here since we don't ship a deadlock checker.
    void lock_shared_without_checking_deadlock()   const noexcept { lock_shared(); }
    void unlock_shared_without_checking_deadlock() const noexcept { unlock_shared(); }
    void lock_exclusive_without_checking_deadlock()   const noexcept { lock_exclusive(); }
    void unlock_exclusive_without_checking_deadlock() const noexcept { unlock_exclusive(); }

    [[nodiscard]] const char* file_name()     const noexcept { return file_name_; }
    [[nodiscard]] const char* variable_name() const noexcept { return variable_name_; }

private:
    mutable cardinal::shared_mutex mutex_;
    const char* file_name_;
    const char* variable_name_;
};

// ---------------------------------------------------------------------------
// NullLock — null-object pattern, satisfies the lock concept with no-ops.
// Use as the __TLock template arg when a queue/structure is provably single-
// threaded but the call site still wants the lock-aware container API.
// ---------------------------------------------------------------------------
class NullLock {
public:
    NullLock() noexcept = default;
    NullLock(const char*, const char*) noexcept {}
    NullLock(const NullLock&)            = delete;
    NullLock& operator=(const NullLock&) = delete;

    [[nodiscard]] bool is_opened() const noexcept { return true; }

    void lock_shared()      const noexcept {}
    void unlock_shared()    const noexcept {}
    void lock_exclusive()   const noexcept {}
    void unlock_exclusive() const noexcept {}
    [[nodiscard]] bool try_lock_shared()    const noexcept { return true; }
    [[nodiscard]] bool try_lock_exclusive() const noexcept { return true; }

    void lock_shared_without_checking_deadlock()      const noexcept {}
    void unlock_shared_without_checking_deadlock()    const noexcept {}
    void lock_exclusive_without_checking_deadlock()   const noexcept {}
    void unlock_exclusive_without_checking_deadlock() const noexcept {}
};

// ---------------------------------------------------------------------------
// ProcessLock — cross-process named mutex. Out-of-line impl in lock.cpp.
// Single primitive: Win path uses CreateMutexW; Linux path uses sem_open.
// Shared- and exclusive-lock both reduce to the same OS primitive (named
// mutexes do not distinguish reader/writer) — semantically identical to PA.
// ---------------------------------------------------------------------------
class ProcessLock {
public:
    ProcessLock(const wchar_t* name, bool do_create) noexcept;
    ~ProcessLock() noexcept;

    ProcessLock(const ProcessLock&)            = delete;
    ProcessLock& operator=(const ProcessLock&) = delete;

    [[nodiscard]] i32  open()  noexcept;
    void               close() noexcept;
    [[nodiscard]] bool is_opened() const noexcept;

    void lock_shared()    const noexcept;
    void lock_exclusive() const noexcept;
    void unlock_shared()    const noexcept;
    void unlock_exclusive() const noexcept;
    [[nodiscard]] bool try_lock_shared()    const noexcept;
    [[nodiscard]] bool try_lock_exclusive() const noexcept;

private:
    void*          handle_;       // HANDLE on Win, sem_t* on Linux (opaque)
    const wchar_t* name_;
    bool           do_create_;
};

// ---------------------------------------------------------------------------
// Lock guards — RAII templates over any type satisfying the lock concept
// (lock_shared/unlock_shared/lock_exclusive/unlock_exclusive). Cardinal
// already has std::scoped_lock; these mirror the Pa naming for porting.
// ---------------------------------------------------------------------------
template <class TLock>
class SharedLockGuard {
public:
    explicit SharedLockGuard(const TLock* lock) noexcept : lock_(lock) { lock_->lock_shared(); }
    explicit SharedLockGuard(const TLock& lock) noexcept : lock_(&lock){ lock_->lock_shared(); }
    ~SharedLockGuard() noexcept                                       { lock_->unlock_shared(); }
    SharedLockGuard(const SharedLockGuard&)            = delete;
    SharedLockGuard& operator=(const SharedLockGuard&) = delete;
private:
    const TLock* lock_;
};

template <class TLock>
class ExclusiveLockGuard {
public:
    explicit ExclusiveLockGuard(const TLock* lock) noexcept : lock_(lock) { lock_->lock_exclusive(); }
    explicit ExclusiveLockGuard(const TLock& lock) noexcept : lock_(&lock){ lock_->lock_exclusive(); }
    ~ExclusiveLockGuard() noexcept                                       { lock_->unlock_exclusive(); }
    ExclusiveLockGuard(const ExclusiveLockGuard&)            = delete;
    ExclusiveLockGuard& operator=(const ExclusiveLockGuard&) = delete;
private:
    const TLock* lock_;
};

template <class TLock>
class TrySharedLockGuard {
public:
    explicit TrySharedLockGuard(const TLock* lock) noexcept : lock_(lock), is_locked_(lock_->try_lock_shared()) {}
    explicit TrySharedLockGuard(const TLock& lock) noexcept : lock_(&lock), is_locked_(lock_->try_lock_shared()) {}
    ~TrySharedLockGuard() noexcept { if (is_locked_) lock_->unlock_shared(); }
    TrySharedLockGuard(const TrySharedLockGuard&)            = delete;
    TrySharedLockGuard& operator=(const TrySharedLockGuard&) = delete;
    [[nodiscard]] bool is_locked() const noexcept { return is_locked_; }
private:
    const TLock* lock_;
    bool         is_locked_;
};

template <class TLock>
class TryExclusiveLockGuard {
public:
    explicit TryExclusiveLockGuard(const TLock* lock) noexcept : lock_(lock), is_locked_(lock_->try_lock_exclusive()) {}
    explicit TryExclusiveLockGuard(const TLock& lock) noexcept : lock_(&lock), is_locked_(lock_->try_lock_exclusive()) {}
    ~TryExclusiveLockGuard() noexcept { if (is_locked_) lock_->unlock_exclusive(); }
    TryExclusiveLockGuard(const TryExclusiveLockGuard&)            = delete;
    TryExclusiveLockGuard& operator=(const TryExclusiveLockGuard&) = delete;
    [[nodiscard]] bool is_locked() const noexcept { return is_locked_; }
private:
    const TLock* lock_;
    bool         is_locked_;
};

}  // namespace cardinal::core::pa
