#pragma once

// =============================================================================
// cardinal::core::noise — deterministic procedural noise (hash + value + fBm).
//
// Rng is a stateful STREAM; procedural generation needs the opposite — a pure,
// position-addressable field: sample(x, y) must always return the same value
// for the same coordinate + seed, in any order, with no shared state. That's
// what terrain heightmaps, biome/scatter masks, texture detail, and stable
// per-cell jitter need. Built on splitmix64 (same mixer Rng seeds from), so
// it's integer-deterministic + cross-platform reproducible.
//
//   hash2_u32 / hash3_u32  — per-integer-cell white noise (well-mixed u32).
//   hash2_unit             — same, mapped to [0,1).
//   value_noise_2d         — smooth value noise (quintic-interpolated lattice
//                            of hashed corners), continuous, in [0,1).
//   fbm_2d                 — fractal sum of octaves of value_noise_2d, [0,1).
//
// Free functions in cardinal::core::noise; reachable as cardinal::noise::*.
// FOUNDATION RULE: lives in cardinal::core (uses splitmix64 + core cmath).
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/hash.hpp>   // splitmix64
#include <cardinal/core/std/cmath.hpp>      // cardinal::floor

namespace cardinal::core::noise {

// ---- white noise: deterministic per-integer-cell hash --------------------
inline cardinal::u32 hash2_u32(cardinal::i32 x, cardinal::i32 y,
                               cardinal::u32 seed = 0) noexcept {
    cardinal::u64 h = static_cast<cardinal::u64>(static_cast<cardinal::u32>(x)) * 0x9E3779B97F4A7C15ull;
    h ^= static_cast<cardinal::u64>(static_cast<cardinal::u32>(y)) * 0xC2B2AE3D27D4EB4Full;
    h ^= static_cast<cardinal::u64>(seed) * 0x165667B19E3779F9ull;
    return static_cast<cardinal::u32>(splitmix64(h));
}
inline cardinal::u32 hash3_u32(cardinal::i32 x, cardinal::i32 y, cardinal::i32 z,
                               cardinal::u32 seed = 0) noexcept {
    cardinal::u64 h = static_cast<cardinal::u64>(static_cast<cardinal::u32>(x)) * 0x9E3779B97F4A7C15ull;
    h ^= static_cast<cardinal::u64>(static_cast<cardinal::u32>(y)) * 0xC2B2AE3D27D4EB4Full;
    h ^= static_cast<cardinal::u64>(static_cast<cardinal::u32>(z)) * 0x27D4EB2F165667C5ull;
    h ^= static_cast<cardinal::u64>(seed) * 0x165667B19E3779F9ull;
    return static_cast<cardinal::u32>(splitmix64(h));
}
// White noise in [0,1) — top 24 bits → float.
inline float hash2_unit(cardinal::i32 x, cardinal::i32 y, cardinal::u32 seed = 0) noexcept {
    return static_cast<float>(hash2_u32(x, y, seed) >> 8) * (1.0f / 16777216.0f);
}

// ---- smooth value noise --------------------------------------------------
namespace detail {
inline cardinal::i32 floor_i(float v) noexcept {
    return static_cast<cardinal::i32>(cardinal::floor(v));
}
// Quintic smootherstep 6t^5 - 15t^4 + 10t^3 (Perlin's fade; C2-continuous).
inline float fade(float t) noexcept { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }
}  // namespace detail

// Continuous value noise in [0,1): bilinear blend of the 4 hashed lattice
// corners with quintic interpolation. At integer coordinates it returns the
// corner's hash exactly.
inline float value_noise_2d(float x, float y, cardinal::u32 seed = 0) noexcept {
    const cardinal::i32 x0 = detail::floor_i(x);
    const cardinal::i32 y0 = detail::floor_i(y);
    const float ux = detail::fade(x - static_cast<float>(x0));
    const float uy = detail::fade(y - static_cast<float>(y0));
    const float c00 = hash2_unit(x0,     y0,     seed);
    const float c10 = hash2_unit(x0 + 1, y0,     seed);
    const float c01 = hash2_unit(x0,     y0 + 1, seed);
    const float c11 = hash2_unit(x0 + 1, y0 + 1, seed);
    const float a = c00 + (c10 - c00) * ux;
    const float b = c01 + (c11 - c01) * ux;
    return a + (b - a) * uy;
}

// Fractal Brownian motion — sum of `octaves` value-noise layers at rising
// frequency / falling amplitude, normalised to [0,1).
inline float fbm_2d(float x, float y, cardinal::u32 seed = 0, int octaves = 4,
                    float lacunarity = 2.0f, float gain = 0.5f) noexcept {
    float sum = 0.0f, amp = 0.5f, freq = 1.0f, norm = 0.0f;
    for (int o = 0; o < octaves; ++o) {
        sum  += amp * value_noise_2d(x * freq, y * freq,
                                     seed + static_cast<cardinal::u32>(o) * 1013904223u);
        norm += amp;
        freq *= lacunarity;
        amp  *= gain;
    }
    return norm > 0.0f ? sum / norm : 0.0f;
}

}  // namespace cardinal::core::noise

namespace cardinal { namespace noise = core::noise; }
