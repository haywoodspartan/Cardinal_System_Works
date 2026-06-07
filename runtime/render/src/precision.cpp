// =============================================================================
// Cardinal — FP8 / FP4 packed-math implementation.
//
// Reference impls — careful + slow rather than the optimised SIMD variants
// real production code would ship. Used by:
//   - the algo registry's CPU "preview" path (so the editor can show you
//     what value a given precision will round your input to)
//   - unit tests that pin the bit layouts against external golden values
//
// The shader-side counterparts in cardinal_packed_math.hlsli MUST agree
// bit-for-bit (asserted by the tests + by visual inspection of any
// precision-sensitive shader output).
// =============================================================================
#include <cardinal/render/precision.hpp>

#include <cardinal/core/std/bit.hpp>
#include <cardinal/core/std/cmath.hpp>
#include <cardinal/core/std/limits.hpp>

namespace cardinal::render::precision {

// ---------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------
const char* format_name(Format f) noexcept {
    switch (f) {
        case Format::FP32:     return "FP32 (passthrough)";
        case Format::FP16:     return "FP16 (binary16)";
        case Format::FP8_E4M3: return "FP8 E4M3";
        case Format::FP8_E5M2: return "FP8 E5M2";
        case Format::FP4_E2M1: return "FP4 E2M1";
        case Format::FP4_E3M0: return "FP4 E3M0";
    }
    return "?";
}

u32 format_bits(Format f) noexcept {
    switch (f) {
        case Format::FP32:     return 32;
        case Format::FP16:     return 16;
        case Format::FP8_E4M3:
        case Format::FP8_E5M2: return 8;
        case Format::FP4_E2M1:
        case Format::FP4_E3M0: return 4;
    }
    return 32;
}
u32 format_lane_count_in_dword(Format f) noexcept {
    return 32u / format_bits(f);
}

float format_max_finite(Format f) noexcept {
    switch (f) {
        case Format::FP32:     return cardinal::numeric_limits<float>::max();
        case Format::FP16:     return 65504.0f;
        case Format::FP8_E4M3: return 448.0f;
        case Format::FP8_E5M2: return 57344.0f;
        case Format::FP4_E2M1: return 6.0f;
        case Format::FP4_E3M0: return 16.0f;       // 2^4
    }
    return 0.0f;
}

float format_smallest_subnormal(Format f) noexcept {
    switch (f) {
        case Format::FP32:     return cardinal::numeric_limits<float>::denorm_min();
        case Format::FP16:     return 5.96046448e-8f;        // 2^-24
        case Format::FP8_E4M3: return 1.0f / 512.0f;         // 2^-9
        case Format::FP8_E5M2: return 1.0f / 65536.0f;       // 2^-16
        case Format::FP4_E2M1: return 0.5f;                  // smallest non-zero
        case Format::FP4_E3M0: return 0.125f;                // 2^-3
    }
    return 0.0f;
}

// Bit-cast helpers — constexpr type-puns via the Foundation's bit_cast.
constexpr u32   fp32_bits(float f) noexcept { return cardinal::bit_cast<u32>(f); }
constexpr float bits_fp32(u32 u)   noexcept { return cardinal::bit_cast<float>(u); }

// ---------------------------------------------------------------------------
// FP16
// ---------------------------------------------------------------------------
u16 fp32_to_fp16(float x) noexcept {
    const u32 u  = fp32_bits(x);
    const u32 sign = (u >> 31) & 0x1u;
    const u32 exp  = (u >> 23) & 0xFFu;
    const u32 mant = u & 0x7FFFFFu;

    if (exp == 0xFF) {                      // inf / nan
        const u16 m = (mant != 0) ? (u16)((mant >> 13) | 0x200) : 0;
        return (u16)((sign << 15) | (0x1F << 10) | m);
    }
    int e = (int)exp - 127 + 15;            // rebias
    if (e >= 31) {                          // overflow → ±inf
        return (u16)((sign << 15) | (0x1F << 10));
    }
    if (e <= 0) {                           // subnormal / zero
        if (e < -10) return (u16)(sign << 15);     // underflow → ±0
        u32 m = mant | 0x800000u;                  // implicit 1
        const int shift = 14 - e;
        const u32 round = (m >> (shift - 1)) & 1u;
        return (u16)((sign << 15) | ((m >> shift) + round));
    }
    // Normal — round-to-nearest-even on the dropped bits.
    const u32 m_round_bit = (mant >> 12) & 1u;
    const u32 m_sticky    = mant & 0xFFFu;
    u32       m_short     = mant >> 13;
    if (m_round_bit && (m_sticky != 0 || (m_short & 1))) {
        ++m_short;
        if (m_short == 0x400) { m_short = 0; ++e; if (e >= 31) {
            return (u16)((sign << 15) | (0x1F << 10));
        }}
    }
    return (u16)((sign << 15) | ((u32)e << 10) | m_short);
}

float fp16_to_fp32(u16 h) noexcept {
    const u32 sign = (h >> 15) & 1u;
    const u32 exp  = (h >> 10) & 0x1Fu;
    const u32 mant = h & 0x3FFu;
    u32 u;
    if (exp == 0) {
        if (mant == 0) {                                     // ±0
            u = sign << 31;
        } else {                                             // subnormal
            int e = -14;
            u32 m = mant;
            while ((m & 0x400) == 0) { m <<= 1; --e; }
            m &= 0x3FF;
            u = (sign << 31) | ((u32)(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {                                // inf / nan
        u = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        u = (sign << 31) | ((u32)(exp - 15 + 127) << 23) | (mant << 13);
    }
    return bits_fp32(u);
}

// ---------------------------------------------------------------------------
// Common quantisation kernel — works for any (exp_bits, mant_bits, bias,
// has_inf) combination. Rounds-to-nearest-even on the dropped mantissa.
// Returns the format's packed bit pattern shifted into the low bits.
// ---------------------------------------------------------------------------
namespace {
struct Spec {
    int exp_bits;
    int mant_bits;
    int bias;
    bool has_inf;        // E5M2 has ±inf (mant==0, exp=full); E4M3 NaN at full
    int  total_bits() const { return 1 + exp_bits + mant_bits; }
};

inline u32 quantise_to_packed(Spec s, float x) noexcept {
    const u32 u    = fp32_bits(x);
    const u32 sign = (u >> 31) & 1u;
    const u32 e32  = (u >> 23) & 0xFFu;
    const u32 m32  = u & 0x7FFFFFu;

    const u32 max_exp_packed = (1u << s.exp_bits) - 1u;

    // NaN — propagate.
    if (e32 == 0xFF && m32 != 0) {
        // E4M3 has no inf; the canonical NaN encoding is 0x7F (positive)
        // / 0xFF (negative) per OFP8 spec. E5M2 follows IEEE: inf has m=0,
        // any non-zero mantissa is NaN.
        if (s.has_inf) {
            return (sign << (s.exp_bits + s.mant_bits))
                 | (max_exp_packed << s.mant_bits)
                 | 1u;
        } else {
            return (sign << (s.exp_bits + s.mant_bits))
                 | (max_exp_packed << s.mant_bits)
                 | ((1u << s.mant_bits) - 1u);
        }
    }
    // ±Inf.
    if (e32 == 0xFF) {
        if (s.has_inf) {
            return (sign << (s.exp_bits + s.mant_bits))
                 | (max_exp_packed << s.mant_bits);
        }
        // No inf — saturate to max-finite.
        return (sign << (s.exp_bits + s.mant_bits))
             | ((max_exp_packed - 1u) << s.mant_bits)
             | ((1u << s.mant_bits) - 1u);
    }

    // ±0
    if (e32 == 0 && m32 == 0) {
        return sign << (s.exp_bits + s.mant_bits);
    }

    // Compute target unbiased exponent.
    int unbiased = (int)e32 - 127;
    int e_target = unbiased + s.bias;

    // Reconstruct the mantissa with the implicit 1 bit prepended.
    u32 mant_full = (e32 == 0) ? m32 : (m32 | 0x800000u);
    int mant_full_msb = (e32 == 0) ? 22 : 23;   // position of leading bit

    // Shift down to s.mant_bits — round-to-nearest-even on dropped bits.
    int shift = mant_full_msb - s.mant_bits;
    if (shift < 0) shift = 0;

    auto round_rne = [&](u32 m, int sh) -> u32 {
        if (sh <= 0) return m;
        // The subnormal path feeds sh = shift + extra, which can reach
        // ~45. A shift count >= the operand width is undefined behaviour
        // on a u32. `m` here is a reconstructed fp32 mantissa (< 2^24),
        // so any sh >= 32 shifts every bit — including the round bit at
        // position sh-1 (>= 31, where m is 0) — out: the result is
        // exactly 0 with no round-up. Guard makes that explicit + UB-free
        // (sh in [25,31] is already <32, defined, and also yields 0).
        if (sh >= 32) return 0u;
        const u32 round_bit = (m >> (sh - 1)) & 1u;
        const u32 sticky    = (sh >= 2) ? (m & ((1u << (sh - 1)) - 1u)) : 0u;
        u32 result = m >> sh;
        if (round_bit && (sticky || (result & 1u))) ++result;
        return result;
    };

    if (e_target <= 0) {
        // Subnormal in the target format. Shift extra to align.
        const int extra = 1 - e_target;
        if (extra > 24) {
            return sign << (s.exp_bits + s.mant_bits);   // underflow → ±0
        }
        const u32 m = round_rne(mant_full, shift + extra);
        if (m == 0) return sign << (s.exp_bits + s.mant_bits);
        // m may have rounded up into the implicit-1 slot — promote to normal.
        if (m == (1u << s.mant_bits)) {
            return (sign << (s.exp_bits + s.mant_bits))
                 | ((u32)1 << s.mant_bits);   // normal exponent 1, mantissa 0
        }
        return (sign << (s.exp_bits + s.mant_bits)) | m;
    }

    u32 m_short = round_rne(mant_full, shift);
    // Mantissa overflow (e.g. 1.111 + 1 ulp = 10.000): bump exponent.
    if (m_short >= (2u << s.mant_bits)) {
        m_short >>= 1;
        ++e_target;
    }
    // Strip implicit 1 bit.
    m_short &= ((1u << s.mant_bits) - 1u);

    if (e_target >= (int)max_exp_packed) {
        if (s.has_inf) {
            return (sign << (s.exp_bits + s.mant_bits))
                 | (max_exp_packed << s.mant_bits);
        }
        // Saturate to max finite.
        return (sign << (s.exp_bits + s.mant_bits))
             | ((max_exp_packed - 1u) << s.mant_bits)
             | ((1u << s.mant_bits) - 1u);
    }
    return (sign << (s.exp_bits + s.mant_bits))
         | ((u32)e_target << s.mant_bits)
         | m_short;
}

inline float dequantise_packed(Spec s, u32 v) noexcept {
    const u32 sign = (v >> (s.exp_bits + s.mant_bits)) & 1u;
    const u32 e_pk = (v >> s.mant_bits) & ((1u << s.exp_bits) - 1u);
    const u32 m_pk = v & ((1u << s.mant_bits) - 1u);
    const u32 max_exp_packed = (1u << s.exp_bits) - 1u;

    if (e_pk == 0) {
        if (m_pk == 0) return sign ? -0.0f : 0.0f;
        // Subnormal: value = (-1)^s * 2^(1 - bias) * (m / 2^mant_bits)
        const float frac = (float)m_pk / (float)(1u << s.mant_bits);
        const float val  = cardinal::ldexp(frac, 1 - s.bias);
        return sign ? -val : val;
    }
    if (e_pk == max_exp_packed) {
        if (s.has_inf) {
            if (m_pk == 0) return sign
                ? -cardinal::numeric_limits<float>::infinity()
                :  cardinal::numeric_limits<float>::infinity();
            return cardinal::numeric_limits<float>::quiet_NaN();
        }
        // E4M3: full-exp + max-mant is NaN; otherwise normal.
        if (m_pk == ((1u << s.mant_bits) - 1u))
            return cardinal::numeric_limits<float>::quiet_NaN();
    }
    const float frac = 1.0f + (float)m_pk / (float)(1u << s.mant_bits);
    const float val  = cardinal::ldexp(frac, (int)e_pk - s.bias);
    return sign ? -val : val;
}
}  // namespace

// ---------------------------------------------------------------------------
// FP8 wrappers
// ---------------------------------------------------------------------------
u8 fp32_to_fp8_e4m3(float x) noexcept {
    return (u8)quantise_to_packed({4, 3, 7,  /*has_inf*/ false}, x);
}
u8 fp32_to_fp8_e5m2(float x) noexcept {
    return (u8)quantise_to_packed({5, 2, 15, /*has_inf*/ true},  x);
}
float fp8_e4m3_to_fp32(u8 v) noexcept {
    return dequantise_packed({4, 3, 7, false}, v);
}
float fp8_e5m2_to_fp32(u8 v) noexcept {
    return dequantise_packed({5, 2, 15, true}, v);
}

// ---------------------------------------------------------------------------
// FP4 wrappers — same kernel; nibble in the low 4 bits.
// ---------------------------------------------------------------------------
u8 fp32_to_fp4_e2m1(float x) noexcept {
    return (u8)(quantise_to_packed({2, 1, 1, /*has_inf*/ false}, x) & 0xFu);
}
u8 fp32_to_fp4_e3m0(float x) noexcept {
    return (u8)(quantise_to_packed({3, 0, 3, /*has_inf*/ false}, x) & 0xFu);
}
float fp4_e2m1_to_fp32(u8 nib) noexcept {
    return dequantise_packed({2, 1, 1, false}, nib & 0xFu);
}
float fp4_e3m0_to_fp32(u8 nib) noexcept {
    return dequantise_packed({3, 0, 3, false}, nib & 0xFu);
}

u32 pack8_fp4_e2m1(const float v[8]) noexcept {
    u32 r = 0;
    for (int i = 0; i < 8; ++i) r |= ((u32)fp32_to_fp4_e2m1(v[i]) & 0xFu) << (i * 4);
    return r;
}
u32 pack8_fp4_e3m0(const float v[8]) noexcept {
    u32 r = 0;
    for (int i = 0; i < 8; ++i) r |= ((u32)fp32_to_fp4_e3m0(v[i]) & 0xFu) << (i * 4);
    return r;
}
cardinal::array<float, 8> unpack8_fp4_e2m1(u32 p) noexcept {
    cardinal::array<float, 8> r{};
    for (int i = 0; i < 8; ++i) r[i] = fp4_e2m1_to_fp32((u8)((p >> (i * 4)) & 0xFu));
    return r;
}
cardinal::array<float, 8> unpack8_fp4_e3m0(u32 p) noexcept {
    cardinal::array<float, 8> r{};
    for (int i = 0; i < 8; ++i) r[i] = fp4_e3m0_to_fp32((u8)((p >> (i * 4)) & 0xFu));
    return r;
}

// ---------------------------------------------------------------------------
// One-shot quantise — used by the algo registry's preview path.
// ---------------------------------------------------------------------------
float quantise(Format f, float x) noexcept {
    switch (f) {
        case Format::FP32:     return x;
        case Format::FP16:     return fp16_to_fp32(fp32_to_fp16(x));
        case Format::FP8_E4M3: return fp8_e4m3_to_fp32(fp32_to_fp8_e4m3(x));
        case Format::FP8_E5M2: return fp8_e5m2_to_fp32(fp32_to_fp8_e5m2(x));
        case Format::FP4_E2M1: return fp4_e2m1_to_fp32(fp32_to_fp4_e2m1(x));
        case Format::FP4_E3M0: return fp4_e3m0_to_fp32(fp32_to_fp4_e3m0(x));
    }
    return x;
}

}  // namespace cardinal::render::precision
