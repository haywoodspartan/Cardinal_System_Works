#pragma once

// =============================================================================
// Cardinal — typed opaque handle.
//
// `Handle<Tag>` is a 64-bit (index, generation) pair distinguishable at the
// type level by an empty `Tag` struct. The tag prevents accidentally passing
// e.g. a MeshHandle where an EntityHandle is expected — a class of bug that
// raw `u32 id` fields can't catch.
//
// Usage:
//
//     struct MeshTag {};
//     using MeshHandle = cardinal::core::Handle<MeshTag>;
//
//     MeshHandle h = pool.alloc(...);
//     if (h.valid()) { ... }
//
// The generation counter lets us detect use-after-free: when a slot is
// recycled the pool bumps its generation, and any old handle with the
// previous generation compares unequal even if the index matches.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <functional>     // std::hash specialisation

namespace cardinal::core {

template <typename Tag>
struct Handle {
    u32 index{0};
    u32 generation{0};

    constexpr Handle() = default;
    constexpr Handle(u32 idx, u32 gen) noexcept : index(idx), generation(gen) {}

    constexpr bool valid() const noexcept { return generation != 0; }
    constexpr explicit operator bool() const noexcept { return valid(); }

    constexpr bool operator==(const Handle& o) const noexcept {
        return index == o.index && generation == o.generation;
    }
    constexpr bool operator!=(const Handle& o) const noexcept { return !(*this == o); }

    constexpr u64 packed() const noexcept {
        return (static_cast<u64>(generation) << 32) | static_cast<u64>(index);
    }
    static constexpr Handle from_packed(u64 p) noexcept {
        return Handle{ static_cast<u32>(p & 0xffffffffu),
                       static_cast<u32>(p >> 32) };
    }
};

}  // namespace cardinal::core

namespace std {
template <typename Tag>
struct hash<cardinal::core::Handle<Tag>> {
    cardinal::usize operator()(const cardinal::core::Handle<Tag>& h) const noexcept {
        // splitmix64 of the packed value — fast, well-distributed, no hash flooding.
        cardinal::u64 x = h.packed();
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        x =  x ^ (x >> 31);
        return static_cast<cardinal::usize>(x);
    }
};
}  // namespace std
