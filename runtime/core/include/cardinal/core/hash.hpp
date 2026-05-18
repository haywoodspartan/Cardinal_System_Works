#pragma once

// =============================================================================
// Cardinal — fast deterministic hashing primitives.
//
// `splitmix64`  — the canonical mix; great as a building block, terrible as
//                 a streaming hash (no incremental state).
// `fnv1a64`     — FNV-1a, 64-bit. Streamable; reasonable distribution; tiny.
// `fxhash`      — Firefox-style multiplicative hash; fastest of the three
//                 for fixed-size keys (4–32 bytes); best for hot paths like
//                 hash-map keying.
//
// All three are deterministic across runs / platforms — call them with the
// same bytes and get the same u64. Useful for serialised content IDs.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <cstring>        // std::memcpy

namespace cardinal::core {

constexpr u64 kFnvPrime64  = 1099511628211ull;
constexpr u64 kFnvOffset64 = 14695981039346656037ull;

inline constexpr u64 splitmix64(u64 x) noexcept {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    x =  x ^ (x >> 31);
    return x;
}

// FNV-1a streaming hash. Pass `seed = kFnvOffset64` to start, then chain
// across multiple regions if needed.
inline u64 fnv1a64(const void* data, usize len, u64 seed = kFnvOffset64) noexcept {
    const auto* p = static_cast<const u8*>(data);
    u64 h = seed;
    for (usize i = 0; i < len; ++i) {
        h ^= static_cast<u64>(p[i]);
        h *= kFnvPrime64;
    }
    return h;
}
inline u64 fnv1a64(const char* s, u64 seed = kFnvOffset64) noexcept {
    u64 h = seed;
    while (*s) { h ^= static_cast<u64>(static_cast<u8>(*s++)); h *= kFnvPrime64; }
    return h;
}

// fxhash — fast for fixed-size keys, used by Firefox / rustc hash maps.
inline u64 fxhash64(u64 hash, u64 word) noexcept {
    constexpr u64 K = 0x517cc1b727220a95ull;
    return ((hash << 5) | (hash >> (64 - 5))) ^ word * K;
}
inline u64 fxhash64(const void* data, usize len, u64 seed = 0) noexcept {
    const auto* p = static_cast<const u8*>(data);
    u64 h = seed;
    while (len >= 8) {
        u64 w; std::memcpy(&w, p, 8);
        h = fxhash64(h, w);
        p += 8; len -= 8;
    }
    u64 tail = 0;
    std::memcpy(&tail, p, len);
    if (len) h = fxhash64(h, tail);
    return h;
}

}  // namespace cardinal::core
