#pragma once

// =============================================================================
// cardinal::core::ArenaAllocator<T> — std::allocator-conforming adapter that
// forwards allocate() to a LinearAllocator. deallocate() is a no-op (the
// arena reclaims en masse via reset() / rewind()).
//
// Use this to back std::vector / std::deque / std::list / cardinal::vector
// with arena memory:
//
//   LinearAllocator arena(64 * 1024);
//   cardinal::vector<int, cardinal::core::ArenaAllocator<int>>
//       scratch(cardinal::core::ArenaAllocator<int>{arena});
//   scratch.reserve(1000);
//   scratch.push_back(...);   // arena absorbs the alloc
//   // ... use scratch ...
//   arena.reset();            // reclaims everything; scratch is dangling
//                             // (don't touch it again — its iterators /
//                             //  data() pointer are invalid).
//
// Properties:
//   * Stateful — equality compares the underlying arena pointer; rebound
//     copies share the same arena.
//   * propagate_on_container_*  = true_type for swap/copy/move-assign so
//     the contained allocator follows the container (matches std::vector
//     semantics with stateful allocators).
//   * Does not throw — allocate() returns nullptr on arena exhaustion;
//     std::vector treats that as bad_alloc internally. Don't push past
//     the arena's capacity unless you've sized it for the worst case.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/alloc/linear_allocator.hpp>

#include <new>            // std::bad_alloc
#include <type_traits>

namespace cardinal::core {

template <class T>
class ArenaAllocator {
public:
    using value_type = T;

    // Stateful — copy-propagation enabled so the allocator follows the
    // container through swap / copy-assign / move-assign.
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap            = std::true_type;
    using is_always_equal                         = std::false_type;

    explicit ArenaAllocator(LinearAllocator& arena) noexcept : arena_(&arena) {}

    template <class U>
    ArenaAllocator(const ArenaAllocator<U>& other) noexcept : arena_(other.arena()) {}

    [[nodiscard]] T* allocate(usize n) {
        void* p = arena_->allocate(n * sizeof(T), alignof(T));
        if (p == nullptr) {
            // Arena exhausted. std::vector catches this and surfaces
            // bad_alloc to the caller — same shape as std::allocator
            // running out of memory.
            throw std::bad_alloc();
        }
        return static_cast<T*>(p);
    }

    void deallocate(T* /*p*/, usize /*n*/) noexcept {
        // No-op — arena reclaims en masse via reset() / rewind(). The
        // memory remains valid until the underlying arena resets.
    }

    [[nodiscard]] LinearAllocator* arena() const noexcept { return arena_; }

    template <class U>
    [[nodiscard]] bool operator==(const ArenaAllocator<U>& other) const noexcept {
        return arena_ == other.arena();
    }
    template <class U>
    [[nodiscard]] bool operator!=(const ArenaAllocator<U>& other) const noexcept {
        return !(*this == other);
    }

private:
    LinearAllocator* arena_;
};

}  // namespace cardinal::core
