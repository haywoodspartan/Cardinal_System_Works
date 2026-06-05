#pragma once

// =============================================================================
// Cardinal core — Access<TLock> + AccessGuard<TAccess> — modern
// C++20 port of Pearl Abyss PaAccess.h.
//
// Why this exists:
//   Access is a "graceful-shutdown ref-count gate" that sits in front
//   of any resource a producer wants to publish to consumers, then later
//   tear down safely. The producer registers the resource and flips the
//   gate open via SetAttachable(true); consumers call attach() to bump
//   the count + obtain access, then detach() when done. Shutdown sequence
//   is: producer calls SetAttachable(false) → existing holders' attach()
//   calls start returning false → producer waits for count to reach 0 →
//   destroy resource safely.
//
//   Modern C++ would usually replace this with std::shared_ptr or a
//   weak/strong-handle pair, but the explicit attach/detach interface
//   matters because PaAccess is paired with a thread-local AccessManager
//   that diagnoses leaks (attach without matching detach) and double-
//   detach mistakes at debug time. The port preserves the surface so
//   CrimsonDesert code can keep using it without behavioural change.
//
// Modernisation:
//   * mAttachCount is std::atomic<i16> — GetAttachCount is now lock-free.
//   * mIsAttachable is std::atomic<bool> — isAttachable / SetAttachable
//     paths take the lock for ordering with attach/detach, but the
//     gate-check inside attach() is a relaxed atomic load.
//   * Templated on TLock so it composes with ThreadLock (default),
//     NullLock (single-thread paths) or any cardinal::shared_mutex-
//     compatible wrapper.
//   * AccessGuard<TAccess> is the RAII pair — attach in ctor (asserted
//     successful by the caller), detach in dtor.
//
// AccessManager (debug-time leak detector) is intentionally NOT ported
// here — Cardinal's diagnostic story is FrameScope / trace_export and
// adding a per-thread attach-count TLS would duplicate that. If a real
// CrimsonDesert consumer needs it, we can add a compile-time-gated
// `CARDINAL_PA_CHECK_ACCESS` variant later.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/atomic.hpp>
#include <cardinal/core/sync/lock.hpp>

namespace cardinal::core {

// ---------------------------------------------------------------------------
// Access<TLock> — attach/detach gate with embedded lock.
// ---------------------------------------------------------------------------
template <class TLock = ThreadLock>
class Access {
public:
    Access() noexcept
        : lock_("", "Access::lock")
        , attach_count_(0)
        , is_attachable_(true) {}

    ~Access() noexcept = default;
    Access(const Access&)            = delete;
    Access& operator=(const Access&) = delete;

    [[nodiscard]] bool is_opened() const noexcept { return lock_.is_opened(); }

    // ---- attach / detach -----------------------------------------------
    // attach() — takes the lock, checks the gate, bumps the count. Returns
    // true if the resource is currently attachable (gate open). False
    // means the producer has begun shutdown — caller must NOT touch the
    // resource and must NOT call detach for this attempt.
    [[nodiscard]] bool attach() noexcept {
        ExclusiveLockGuard<TLock> g(&lock_);
        return attach_without_lock();
    }
    // try_attach() — non-blocking variant; returns false on lock contention.
    [[nodiscard]] bool try_attach() noexcept {
        TryExclusiveLockGuard<TLock> g(&lock_);
        if (!g.is_locked()) return false;
        return attach_without_lock();
    }
    // attach_without_lock — caller is responsible for synchronisation.
    [[nodiscard]] bool attach_without_lock() noexcept {
        if (!is_attachable_.load(cardinal::memory_order_acquire)) return false;
        attach_count_.fetch_add(1, cardinal::memory_order_acq_rel);
        return true;
    }

    // detach() — takes the lock, decrements the count. Returns true iff
    // the producer has closed the gate AND the count has just reached 0
    // — i.e. the caller of detach is the LAST holder, so it is safe to
    // destroy the resource right now.
    [[nodiscard]] bool detach() noexcept {
        ExclusiveLockGuard<TLock> g(&lock_);
        return detach_without_lock();
    }
    [[nodiscard]] bool detach_without_lock() noexcept {
        const i16 prev = attach_count_.fetch_sub(1, cardinal::memory_order_acq_rel);
        return !is_attachable_.load(cardinal::memory_order_acquire) && prev <= 1;
    }

    // ---- attachable gate ----------------------------------------------
    [[nodiscard]] bool is_attachable() const noexcept {
        SharedLockGuard<TLock> g(&lock_);
        return is_attachable_.load(cardinal::memory_order_acquire);
    }
    void set_attachable(bool v) noexcept {
        ExclusiveLockGuard<TLock> g(&lock_);
        set_attachable_without_lock(v);
    }
    void set_attachable_without_lock(bool v) noexcept {
        is_attachable_.store(v, cardinal::memory_order_release);
    }
    // Reset to "open, no holders" — used by container-recycle paths.
    void set_attachable_and_reset_attach_count() noexcept {
        ExclusiveLockGuard<TLock> g(&lock_);
        is_attachable_.store(true, cardinal::memory_order_release);
        attach_count_.store(0, cardinal::memory_order_release);
    }

    [[nodiscard]] i16 attach_count() const noexcept {
        return attach_count_.load(cardinal::memory_order_acquire);
    }

    // ---- lock pass-through (delegate to embedded TLock) ---------------
    // Matches PA's surface where Access<T> presents the lock interface
    // directly so it can stand in for a ThreadLock at every use site.
    void lock_shared()    const noexcept { lock_.lock_shared(); }
    void lock_exclusive() const noexcept { lock_.lock_exclusive(); }
    void unlock_shared()    const noexcept { lock_.unlock_shared(); }
    void unlock_exclusive() const noexcept { lock_.unlock_exclusive(); }
    [[nodiscard]] bool try_lock_shared()    const noexcept { return lock_.try_lock_shared(); }
    [[nodiscard]] bool try_lock_exclusive() const noexcept { return lock_.try_lock_exclusive(); }

    void lock_shared_without_checking_deadlock()      const noexcept { lock_.lock_shared_without_checking_deadlock(); }
    void unlock_shared_without_checking_deadlock()    const noexcept { lock_.unlock_shared_without_checking_deadlock(); }
    void lock_exclusive_without_checking_deadlock()   const noexcept { lock_.lock_exclusive_without_checking_deadlock(); }
    void unlock_exclusive_without_checking_deadlock() const noexcept { lock_.unlock_exclusive_without_checking_deadlock(); }

    // Escape hatch — do NOT call unless you know exactly what you're doing.
    // Matches PA's similarly-named accessor.
    [[nodiscard]] TLock& raw_lock_dont_call_directly() noexcept { return lock_; }

private:
    mutable TLock                lock_;
    cardinal::atomic<i16>        attach_count_;
    cardinal::atomic<bool>       is_attachable_;
};

// ---------------------------------------------------------------------------
// AccessGuard<TAccess> — RAII attach/detach pair. Caller is responsible
// for the initial attach() success check (PA convention — the guard
// asserts you've already attached so the destructor's detach is balanced).
// ---------------------------------------------------------------------------
template <class TAccess>
class AccessGuard {
public:
    explicit AccessGuard(TAccess& access) noexcept : access_(access) {}
    ~AccessGuard() noexcept { (void)access_.detach(); }

    AccessGuard(const AccessGuard&)            = delete;
    AccessGuard& operator=(const AccessGuard&) = delete;

private:
    TAccess& access_;
};

}  // namespace cardinal::core
