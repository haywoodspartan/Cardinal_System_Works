#pragma once

// =============================================================================
// cardinal::core::LinearAllocator — bump / arena allocator.
//
// One contiguous buffer + an atomic offset. Every allocate() advances the
// offset by `(bytes + align padding)`; there is no free per allocation.
// Release happens en masse via reset() (drop everything) or rewind(Marker)
// (drop back to a checkpoint). This is the right allocator for:
//
//   * Per-frame transient work — graph nodes, pass scratch, command list
//     staging — where you allocate hundreds of small objects per frame and
//     free them all at once.
//   * Bulk-allocate-then-walk patterns: build a list of N items, iterate,
//     discard.
//   * Anywhere fragmentation kills a general-purpose allocator.
//
// Critical contract: LinearAllocator does NOT call destructors. make<T>()
// constructs in place but the buffer can only be freed via reset() / rewind()
// which abandon the object. Use it for trivially destructible types, or types
// whose destructor side effects can be skipped without harm (POD, std::byte
// scratch buffers, transient graph nodes).
//
// Thread safety: allocate() is lock-free (std::atomic fetch_add). reset() and
// rewind() are single-threaded — caller must externally synchronise (typical
// frame loop has a single point that resets the arena between frames).
// make<T>() is thus thread-safe, but Marker / rewind is a barrier point.
//
// Modernisations vs. the legacy CRITICAL_SECTION-locked arena:
//   * std::atomic<usize> bump pointer instead of CRITICAL_SECTION — single
//     fetch_add per allocation, no kernel transition on contention.
//   * make<T>(...) supports forwarded variadic args (earlier variants capped at 4
//     non-perfect-forwarded parameter packs).
//   * RAII ScopedMarker — scoped rewind via destructor, removes the
//     "did I remember to rewind?" footgun.
//   * Optional owning constructor (allocator new[]s + delete[]s) for ad-hoc
//     arenas; non-owning constructor for caller-provided buffers (typical
//     for thread-local arenas backed by a fixed reserved region).
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/atomic.hpp>

#include <cstddef>     // std::byte, std::max_align_t
#include <cstdint>     // std::uintptr_t
#include <cstdlib>     // std::aligned_alloc / std::free
#include <new>         // placement new
#include <utility>     // std::forward

namespace cardinal::core {

// ---------------------------------------------------------------------------
// LinearAllocator — atomic bump-pointer arena.
// ---------------------------------------------------------------------------
class LinearAllocator {
public:
    // Owning ctor — allocates `capacity_bytes` aligned to `alignment`.
    // Default alignment matches std::max_align_t, sufficient for any
    // fundamental type. Caller can pass higher (e.g. 64 for cache lines).
    explicit LinearAllocator(usize capacity_bytes,
                             usize alignment = alignof(std::max_align_t)) noexcept
        : buffer_(static_cast<u8*>(operator new(capacity_bytes,
                                                std::align_val_t{alignment}, std::nothrow)))
        , capacity_(capacity_bytes)
        , offset_(0)
        , alignment_(alignment)
        , owns_buffer_(true) {}

    // Non-owning ctor — caller owns the buffer (typical for thread-local
    // arenas with a process-lifetime reservation).
    LinearAllocator(void* buffer, usize capacity_bytes) noexcept
        : buffer_(static_cast<u8*>(buffer))
        , capacity_(capacity_bytes)
        , offset_(0)
        , alignment_(alignof(std::max_align_t))
        , owns_buffer_(false) {}

    ~LinearAllocator() noexcept {
        if (owns_buffer_ && buffer_ != nullptr) {
            operator delete(buffer_, std::align_val_t{alignment_});
        }
    }

    LinearAllocator(const LinearAllocator&)            = delete;
    LinearAllocator& operator=(const LinearAllocator&) = delete;

    // ---- raw allocation -----------------------------------------------
    // Returns nullptr if the arena is exhausted. Aligns the returned
    // ABSOLUTE pointer to `align` (not just the offset — the underlying
    // buffer may have weaker alignment than requested at the call site).
    [[nodiscard]] void* allocate(usize bytes, usize align = alignof(std::max_align_t)) noexcept {
        if (bytes == 0) return nullptr;
        const std::uintptr_t buf_addr = reinterpret_cast<std::uintptr_t>(buffer_);
        // Atomic CAS loop — fetch current offset, compute aligned start
        // from the ABSOLUTE address, try to commit the new end.
        usize old_off = offset_.load(memory_order_relaxed);
        while (true) {
            const std::uintptr_t cur_addr     = buf_addr + old_off;
            const std::uintptr_t aligned_addr = align_up_addr_(cur_addr, align);
            const usize          aligned_off  = static_cast<usize>(aligned_addr - buf_addr);
            const usize          new_off      = aligned_off + bytes;
            if (new_off > capacity_) return nullptr;
            if (offset_.compare_exchange_weak(old_off, new_off,
                                              memory_order_acq_rel,
                                              memory_order_relaxed)) {
                return buffer_ + aligned_off;
            }
            // CAS failed — another thread won the race; retry with the
            // refreshed `old_off` that the failed CAS wrote back.
        }
    }

    // ---- typed factory -------------------------------------------------
    template <class T, class... Args>
    [[nodiscard]] T* make(Args&&... args) noexcept {
        void* p = allocate(sizeof(T), alignof(T));
        if (p == nullptr) return nullptr;
        return ::new (p) T(std::forward<Args>(args)...);
    }

    template <class T>
    [[nodiscard]] T* make_array(usize n) noexcept {
        if (n == 0) return nullptr;
        void* p = allocate(sizeof(T) * n, alignof(T));
        if (!p) return nullptr;
        T* arr = static_cast<T*>(p);
        for (usize i = 0; i < n; ++i) ::new (arr + i) T();
        return arr;
    }

    // ---- marker / rewind -----------------------------------------------
    // A Marker is an opaque token recording the arena's offset at a point
    // in time. Rewinding restores the offset to that point — every object
    // allocated AFTER the mark is invalidated.
    struct Marker { usize offset; };

    [[nodiscard]] Marker mark() const noexcept {
        return Marker{offset_.load(memory_order_acquire)};
    }

    // Rewind is NOT thread-safe — caller must externally synchronise so no
    // concurrent allocations happen across this call.
    void rewind(Marker m) noexcept {
        offset_.store(m.offset, memory_order_release);
    }

    // Drop everything. Not thread-safe (same as rewind).
    void reset() noexcept {
        offset_.store(0, memory_order_release);
    }

    // ---- stats --------------------------------------------------------
    [[nodiscard]] usize capacity()  const noexcept { return capacity_; }
    [[nodiscard]] usize used()      const noexcept { return offset_.load(memory_order_acquire); }
    [[nodiscard]] usize remaining() const noexcept { return capacity_ - used(); }

    [[nodiscard]] void* buffer() noexcept { return buffer_; }

private:
    static constexpr std::uintptr_t align_up_addr_(std::uintptr_t addr, usize align) noexcept {
        // Power-of-2 alignment.
        return (addr + (align - 1)) & ~(static_cast<std::uintptr_t>(align) - 1);
    }

    u8*           buffer_;
    usize         capacity_;
    atomic<usize> offset_;
    usize         alignment_;
    bool          owns_buffer_;
};

// ---------------------------------------------------------------------------
// ScopedArenaMarker — RAII marker for scoped rewind.
//
//   {
//       ScopedArenaMarker scope(arena);
//       Foo* f = arena.make<Foo>(...);
//       bar(f);
//   }  // arena rewinds back to whatever its offset was before the scope
//
// Caller is responsible for not allocating across threads during the
// scope (rewind() is not thread-safe).
// ---------------------------------------------------------------------------
class ScopedArenaMarker {
public:
    explicit ScopedArenaMarker(LinearAllocator& arena) noexcept
        : arena_(arena), marker_(arena.mark()) {}

    ~ScopedArenaMarker() noexcept { arena_.rewind(marker_); }

    ScopedArenaMarker(const ScopedArenaMarker&)            = delete;
    ScopedArenaMarker& operator=(const ScopedArenaMarker&) = delete;

    [[nodiscard]] LinearAllocator::Marker marker() const noexcept { return marker_; }

private:
    LinearAllocator&        arena_;
    LinearAllocator::Marker marker_;
};

}  // namespace cardinal::core
