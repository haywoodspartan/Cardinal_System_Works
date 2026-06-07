#pragma once

// =============================================================================
// cardinal::core::SpscRing<T, Capacity> — lock-free single-producer /
// single-consumer bounded queue (Lamport ring).
//
// StaticCircularQueue is single-threaded + OVERWRITES the oldest entry on
// full. This is the cross-THREAD primitive: exactly one producer thread and
// exactly one consumer thread hand items across with no locks and no lost or
// duplicated entries — for game-thread→render-thread command handoff, job
// results, async log lines. try_push fails (returns false) when full and
// try_pop fails when empty; neither overwrites or blocks.
//
// Correctness (the textbook SPSC ordering):
//   producer: write slot, then publish tail with RELEASE; reads head ACQUIRE.
//   consumer: read tail ACQUIRE (sees the slot write), read slot, free it by
//             publishing head with RELEASE; producer reads that head ACQUIRE.
// Only the producer writes tail_, only the consumer writes head_, so each owns
// its index (relaxed self-load). head_/tail_ sit on separate cache lines to
// avoid false sharing. One spare slot distinguishes full from empty.
//
// T must be default-constructible (slots are pre-constructed) and move/copy-
// assignable. Use it for small trivially-movable payloads (pointers, ids,
// PODs) — the SPSC hot-path case.
//
// FOUNDATION RULE: lives in cardinal::core. Exposed as cardinal::spsc_ring.
// =============================================================================

#include <cardinal/core/types.hpp>     // usize
#include <cardinal/core/std/atomic.hpp>    // cardinal::atomic + memory_order_*
#include <cardinal/core/std/utility.hpp>   // cardinal::move

namespace cardinal::core {

template <class T, cardinal::usize Capacity>
class SpscRing {
    static_assert(Capacity >= 1, "SpscRing capacity must be >= 1");
    static constexpr cardinal::usize kSlots = Capacity + 1u;   // 1 spare: full != empty

public:
    using value_type = T;
    using size_type  = cardinal::usize;

    SpscRing() noexcept : head_(0), tail_(0) {}
    SpscRing(const SpscRing&)            = delete;   // a queue identity isn't copyable
    SpscRing& operator=(const SpscRing&) = delete;

    // Producer side. Returns false (does NOT overwrite/block) when full.
    bool try_push(const T& v) noexcept { return emplace_(v); }
    bool try_push(T&& v)      noexcept { return emplace_(cardinal::move(v)); }

    // Consumer side. Returns false when empty; on success moves into `out`.
    bool try_pop(T& out) noexcept {
        const size_type h = head_.load(cardinal::memory_order_relaxed);   // consumer owns head_
        if (h == tail_.load(cardinal::memory_order_acquire)) return false; // empty
        out = cardinal::move(slots_[h]);
        head_.store(next_(h), cardinal::memory_order_release);            // free the slot
        return true;
    }

    // Snapshots — exact only when the relevant side is quiescent; otherwise a
    // racy estimate (fine for telemetry / "is there work?").
    [[nodiscard]] bool empty() const noexcept {
        return head_.load(cardinal::memory_order_acquire) ==
               tail_.load(cardinal::memory_order_acquire);
    }
    [[nodiscard]] size_type size_approx() const noexcept {
        const size_type t = tail_.load(cardinal::memory_order_acquire);
        const size_type h = head_.load(cardinal::memory_order_acquire);
        return (t + kSlots - h) % kSlots;
    }
    [[nodiscard]] static constexpr size_type capacity() noexcept { return Capacity; }

private:
    template <class V>
    bool emplace_(V&& v) noexcept {
        const size_type t = tail_.load(cardinal::memory_order_relaxed);   // producer owns tail_
        const size_type n = next_(t);
        if (n == head_.load(cardinal::memory_order_acquire)) return false; // full
        slots_[t] = static_cast<V&&>(v);
        tail_.store(n, cardinal::memory_order_release);                   // publish slot
        return true;
    }
    static constexpr size_type next_(size_type i) noexcept { return (i + 1u) % kSlots; }

    alignas(64) cardinal::atomic<size_type> head_;   // consumer cursor
    alignas(64) cardinal::atomic<size_type> tail_;   // producer cursor
    alignas(64) T                           slots_[kSlots];
};

}  // namespace cardinal::core

namespace cardinal {
template <class T, cardinal::usize Capacity>
using spsc_ring = core::SpscRing<T, Capacity>;
}  // namespace cardinal
