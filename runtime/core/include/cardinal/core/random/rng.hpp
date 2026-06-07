#pragma once

// =============================================================================
// cardinal::core::Rng — fast, seedable, deterministic pseudo-random generator.
//
// Games need randomness everywhere — procedural generation, gameplay rolls,
// particle jitter, AI — and the engine had only splitmix64 (a one-shot mixer,
// not a stateful stream). Rng is a xoshiro256** generator: ~sub-ns per draw,
// excellent statistical quality, and INTEGER-ONLY internals so the same seed
// yields the identical sequence on every platform/compiler — essential for
// deterministic replay + lockstep networking. Seeded via splitmix64 so even a
// low-entropy seed (0, 1, a frame index) expands to a well-distributed state.
//
// Not cryptographic. For per-system independence, give each system its own Rng
// with a distinct seed rather than sharing one global stream.
//
// FOUNDATION RULE: lives in cardinal::core. Exposed as cardinal::Rng.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/hash.hpp>   // splitmix64 (seed expansion)

namespace cardinal::core {

class Rng {
public:
    Rng() noexcept { reseed(0x9E3779B97F4A7C15ull); }
    explicit Rng(cardinal::u64 seed) noexcept { reseed(seed); }

    // Restart the stream from `seed` — same seed ⇒ identical sequence.
    void reseed(cardinal::u64 seed) noexcept {
        cardinal::u64 z = seed;
        for (cardinal::u64& v : s_) { z += 0x9E3779B97F4A7C15ull; v = splitmix64(z); }
        if ((s_[0] | s_[1] | s_[2] | s_[3]) == 0u) s_[0] = 0x9E3779B97F4A7C15ull;  // never all-zero
    }

    // ---- raw draws ----------------------------------------------------
    cardinal::u64 next_u64() noexcept {
        const cardinal::u64 result = rotl_(s_[1] * 5u, 7) * 9u;   // xoshiro256** scrambler
        const cardinal::u64 t = s_[1] << 17;
        s_[2] ^= s_[0]; s_[3] ^= s_[1]; s_[1] ^= s_[2]; s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl_(s_[3], 45);
        return result;
    }
    cardinal::u32 next_u32() noexcept { return static_cast<cardinal::u32>(next_u64() >> 32); }

    // ---- unit-interval reals (53/24-bit mantissa, [0,1)) --------------
    double next_double() noexcept { return static_cast<double>(next_u64() >> 11) * 0x1.0p-53; }
    float  next_float()  noexcept { return static_cast<float>(next_u64() >> 40) * 0x1.0p-24f; }

    // ---- ranges -------------------------------------------------------
    // Integer in [lo, hi] INCLUSIVE (returns lo if hi <= lo). Uses modulo;
    // the bias is negligible for typical game ranges.
    cardinal::i32 range(cardinal::i32 lo, cardinal::i32 hi) noexcept {
        if (hi <= lo) return lo;
        const cardinal::u64 span = static_cast<cardinal::u64>(static_cast<cardinal::i64>(hi) - lo) + 1u;
        return lo + static_cast<cardinal::i32>(next_u64() % span);
    }
    // Float in [lo, hi).
    float range_f(float lo, float hi) noexcept { return lo + next_float() * (hi - lo); }

    bool next_bool() noexcept { return (next_u64() & 1u) != 0u; }
    // True with probability p (clamped to [0,1] by the [0,1) draw: p<=0 → never,
    // p>=1 → always).
    bool chance(double p) noexcept { return next_double() < p; }

private:
    cardinal::u64 s_[4];

    static constexpr cardinal::u64 rotl_(cardinal::u64 x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }
};

}  // namespace cardinal::core

namespace cardinal {
using Rng = core::Rng;
}  // namespace cardinal
