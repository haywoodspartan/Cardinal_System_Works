#pragma once

// =============================================================================
// Cardinal Core — canonical spatial primitives.
//
// Single source of truth for AABB, Ray, ChunkCoord (3D) used by the
// scene + physics + world + render modules. Modules previously rolled
// their own near-identical structs; aliasing here keeps memory layouts
// (and hence cross-module type-punning) honest.
//
// AABB and Ray are now thin aliases into cardinal::core::geom (see
// core/geom.hpp) so legacy code referencing core::AABB picks up the
// comprehensive operator set (expand, inflate, intersects, contains, …)
// without any source change.
// =============================================================================

#include <cardinal/core/math/geom.hpp>
#include <cardinal/core/math/math.hpp>
#include <cardinal/core/types.hpp>

#include <cstdint>
#include <unordered_set>

namespace cardinal::core {

// Comprehensive AABB / Ray live in cardinal::core::geom — re-exported here
// so existing `cardinal::core::AABB` / `cardinal::core::Ray` users
// continue to compile.
using AABB = ::cardinal::core::geom::AABB;
using Ray  = ::cardinal::core::geom::Ray;

// 3D integer chunk coord (x, y, z). Default constructs to (0, 0, 0).
struct ChunkCoord {
    i32 x{0}, y{0}, z{0};
    constexpr ChunkCoord() = default;
    constexpr ChunkCoord(i32 cx, i32 cy, i32 cz) noexcept : x(cx), y(cy), z(cz) {}
    constexpr bool operator==(const ChunkCoord& o) const noexcept {
        return x == o.x && y == o.y && z == o.z;
    }
    constexpr bool operator!=(const ChunkCoord& o) const noexcept { return !(*this == o); }
};

// Hash for unordered_set / unordered_map keys. Combines all three axes
// via splitmix-style mixing — distinct chunks with same xz but different
// y are NOT hash-equal (would collide under the old u64 packing).
struct ChunkCoordHash {
    std::size_t operator()(const ChunkCoord& c) const noexcept {
        std::uint64_t h = static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.x))
                        + 0x9E3779B97F4A7C15ull;
        h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ull;
        h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.y));
        h = (h ^ (h >> 27)) * 0x94D049BB133111EBull;
        h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.z));
        h = (h ^ (h >> 31));
        return static_cast<std::size_t>(h);
    }
};

// Standard chunk-set typedef — every container that keys on ChunkCoord
// across the engine uses this.
using ChunkSet = std::unordered_set<ChunkCoord, ChunkCoordHash>;

}  // namespace cardinal::core
