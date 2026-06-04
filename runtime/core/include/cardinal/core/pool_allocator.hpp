#pragma once

// =============================================================================
// cardinal::core::PoolAllocator<T> — fixed-size slab pool.
//
// A single contiguous slab of N slots, each sized + aligned for T. Free
// slots are linked through an intrusive free list (each free slot's first
// sizeof(void*) bytes hold the next-pointer). Acquire / release are O(1).
// Mutex-guarded so it's safe to share across threads.
//
// Right fit:
//   * Many small, fixed-size objects with high churn (particles, scene
//     nodes, render commands, packet headers).
//   * Known maximum count — set capacity at construction, never reallocate.
//   * Want O(1) acquire/release without the malloc overhead per call.
//
// Wrong fit:
//   * Variable-sized objects — use LinearAllocator or a general allocator.
//   * Unknown max count — pool exhausts and acquire returns nullptr.
//   * Single-threaded sequential alloc + drop — LinearAllocator is faster.
//
// Modernisations vs. Pearl Abyss PACustomPoolAllocator:
//   * Single std::mutex instead of CRITICAL_SECTION (consistent with
//     cardinal's std-vocabulary policy; same behaviour on Win10+ SRW).
//   * Template on T — sizeof / alignment derived at compile time.
//     Caller doesn't pass element size at construction.
//   * make<T>(args...) / destroy(T*) — type-aware lifecycle, perfect
//     forwarded ctor args (PA was untyped, hand-managed dtor calls).
//   * Intrusive free list — zero per-slot bookkeeping bytes
//     (slot storage is reused as next-pointer when free).
//     PA used an external uint32 bitmap which costs O(N) scans on
//     exhaustion lookup and 4 bytes / slot.
//
// Concurrency note: the v1 implementation is mutex-protected. A
// thread-local-segment variant (each thread owns its own free list +
// mailbox for cross-thread frees) is the natural next step once a real
// consumer profiles hot enough to need it — adding that under the same
// allocate/deallocate surface is a one-file change.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <cstddef>     // std::byte
#include <mutex>       // std::mutex / std::lock_guard
#include <new>         // placement new, std::align_val_t
#include <utility>     // std::forward
#include <type_traits> // std::is_trivially_destructible_v

namespace cardinal::core {

template <class T>
class PoolAllocator {
public:
    static_assert(sizeof(T) >= sizeof(void*),
                  "PoolAllocator<T>: T must be at least pointer-sized — "
                  "intrusive free list stores next-ptr in the slot. Wrap "
                  "smaller types or use a different allocator.");

    explicit PoolAllocator(usize element_capacity) noexcept
        : storage_(static_cast<Slot*>(
              operator new(sizeof(Slot) * element_capacity,
                           std::align_val_t{alignof(Slot)},
                           std::nothrow)))
        , capacity_(element_capacity)
        , free_head_(nullptr)
        , used_count_(0) {
        // Build the initial free list: link every slot to the next, last
        // points to nullptr. Done in reverse so free_head starts at slot 0.
        if (storage_ != nullptr) {
            for (usize i = 0; i < capacity_; ++i) {
                storage_[i].next = (i + 1 < capacity_) ? &storage_[i + 1] : nullptr;
            }
            free_head_ = &storage_[0];
        }
    }

    ~PoolAllocator() noexcept {
        if (storage_ != nullptr) {
            operator delete(storage_, std::align_val_t{alignof(Slot)});
        }
    }

    PoolAllocator(const PoolAllocator&)            = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;

    // ---- raw acquire / release ----------------------------------------
    // Returns nullptr if the pool is exhausted.
    [[nodiscard]] T* allocate() noexcept {
        std::lock_guard<std::mutex> lk(mutex_);
        if (free_head_ == nullptr) return nullptr;
        Slot* slot = free_head_;
        free_head_ = slot->next;
        ++used_count_;
        return reinterpret_cast<T*>(slot);
    }

    void deallocate(T* p) noexcept {
        if (p == nullptr) return;
        std::lock_guard<std::mutex> lk(mutex_);
        Slot* slot = reinterpret_cast<Slot*>(p);
        slot->next = free_head_;
        free_head_ = slot;
        --used_count_;
    }

    // ---- typed factory -------------------------------------------------
    template <class... Args>
    [[nodiscard]] T* make(Args&&... args) noexcept {
        T* p = allocate();
        if (p == nullptr) return nullptr;
        // Direct-init via placement new — calls explicit ctors correctly.
        // The ternary form `p ? ::new(...) T(...) : nullptr` confuses MSVC's
        // template instantiator when T has an explicit single-arg ctor.
        return ::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
    }

    void destroy(T* p) noexcept {
        if (p == nullptr) return;
        if constexpr (!std::is_trivially_destructible_v<T>) {
            p->~T();
        }
        deallocate(p);
    }

    // ---- stats --------------------------------------------------------
    [[nodiscard]] usize capacity() const noexcept { return capacity_; }
    [[nodiscard]] usize used() const noexcept {
        std::lock_guard<std::mutex> lk(mutex_);
        return used_count_;
    }
    [[nodiscard]] usize available() const noexcept {
        std::lock_guard<std::mutex> lk(mutex_);
        return capacity_ - used_count_;
    }

    // True iff the pool owns this pointer (within its slab range). Useful
    // for debug checks before deallocate when a heterogeneous lifecycle
    // mixes pool + heap pointers.
    [[nodiscard]] bool owns(const T* p) const noexcept {
        const auto* raw = reinterpret_cast<const Slot*>(p);
        return storage_ != nullptr && raw >= storage_ && raw < storage_ + capacity_;
    }

private:
    union Slot {
        alignas(T) std::byte data[sizeof(T)];
        Slot*               next;
    };

    Slot*         storage_;
    usize         capacity_;
    Slot*         free_head_;
    usize         used_count_;
    mutable std::mutex mutex_;
};

}  // namespace cardinal::core
