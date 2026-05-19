#pragma once

// =============================================================================
// Cardinal — FP8 / FP4 packed-math primitives.
//
// Sub-byte floating-point formats used by modern AI accelerators (Hopper,
// Ada, Blackwell-class) and increasingly by graphics for bandwidth-bound
// intermediates: spherical-harmonic probes, screen-space data structures,
// neural materials, attention compute.
//
// Formats implemented (CPU reference + HLSL counterparts in
// shaders/include/cardinal_packed_math.hlsli):
//
//   FP8_E4M3   — sign + 4-bit exp + 3-bit mantissa, bias 7
//                Range ±448, no infinity, single NaN at 0xFE/0x7E.
//                Used for forward activations / weights.
//
//   FP8_E5M2   — sign + 5-bit exp + 2-bit mantissa, bias 15
//                IEEE-754 alike (smaller). Has ±inf and NaN. Wider range
//                (±57344) at the cost of mantissa precision. Gradient-friendly.
//
//   FP4_E2M1   — sign + 2-bit exp + 1-bit mantissa, bias 1
//                16 distinct values total: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6.
//                Used for MoE weight quantisation, very-low-precision MLPs.
//
//   FP4_E3M0   — sign + 3-bit exp + 0-bit mantissa, bias 3
//                Pure powers of two: 0, ±2^-3 .. ±2^4. Cheapest possible
//                signed float. Used in research papers, niche today.
//
// All conversions are exact + deterministic, with round-to-nearest-even
// (RNE) on FP32 → packed paths. Subnormals are honoured. NaN survives
// through encode → decode where the format permits it.
//
// Pack helpers stuff multiple values into a uint32 (4× FP8 or 8× FP4) so
// the layout matches what the shader-side counterparts expect — a single
// 32-bit load loads a full vector.
// =============================================================================

#include <cardinal/core/types.hpp>   // foundation: u8..u64 / usize / f32

#include <cardinal/core/containers.hpp>
#include <cardinal/core/cstring.hpp>

namespace cardinal::render::precision {

// ---------------------------------------------------------------------------
// Format identifiers — keep in sync with the algo-registry "PrecisionMath"
// category and the HLSL preprocessor switches in cardinal_packed_math.hlsli.
// ---------------------------------------------------------------------------
enum class Format : u32 {
    FP32 = 0,        // passthrough — reference
    FP16,            // IEEE-754 binary16, sign + 5e + 10m
    FP8_E4M3,        // 1+4+3, bias 7
    FP8_E5M2,        // 1+5+2, bias 15
    FP4_E2M1,        // 1+2+1, bias 1
    FP4_E3M0,        // 1+3+0, bias 3
};

const char* format_name (Format f) noexcept;
u32         format_bits (Format f) noexcept;        // 8, 16, 8, 8, 4, 4
u32         format_lane_count_in_dword(Format f) noexcept;
float       format_max_finite(Format f) noexcept;
float       format_smallest_subnormal(Format f) noexcept;

// ---------------------------------------------------------------------------
// FP16 — for completeness (the engine already uses min16float in shaders).
// ---------------------------------------------------------------------------
u16 fp32_to_fp16 (float x) noexcept;
float fp16_to_fp32(u16 h) noexcept;

// ---------------------------------------------------------------------------
// FP8 E4M3 / E5M2
// ---------------------------------------------------------------------------
// Encode a single FP32 to the packed FP8 byte. Round-to-nearest-even on
// the mantissa; saturate to ±max_finite on overflow (E4M3 has no inf,
// E5M2 saturates to ±inf via the standard-encoding 0x7C/0xFC).
u8    fp32_to_fp8_e4m3(float x) noexcept;
u8    fp32_to_fp8_e5m2(float x) noexcept;
float fp8_e4m3_to_fp32(u8 v)    noexcept;
float fp8_e5m2_to_fp32(u8 v)    noexcept;

// Pack 4× FP8 values into a single dword (LSB = lane 0, MSB = lane 3).
inline u32 pack4_fp8_e4m3(float a, float b, float c, float d) noexcept {
    return  (u32)fp32_to_fp8_e4m3(a)
         | ((u32)fp32_to_fp8_e4m3(b) << 8)
         | ((u32)fp32_to_fp8_e4m3(c) << 16)
         | ((u32)fp32_to_fp8_e4m3(d) << 24);
}
inline u32 pack4_fp8_e5m2(float a, float b, float c, float d) noexcept {
    return  (u32)fp32_to_fp8_e5m2(a)
         | ((u32)fp32_to_fp8_e5m2(b) << 8)
         | ((u32)fp32_to_fp8_e5m2(c) << 16)
         | ((u32)fp32_to_fp8_e5m2(d) << 24);
}
inline cardinal::array<float, 4> unpack4_fp8_e4m3(u32 p) noexcept {
    return { fp8_e4m3_to_fp32((u8)(p & 0xFF)),
             fp8_e4m3_to_fp32((u8)((p >> 8)  & 0xFF)),
             fp8_e4m3_to_fp32((u8)((p >> 16) & 0xFF)),
             fp8_e4m3_to_fp32((u8)((p >> 24) & 0xFF)) };
}
inline cardinal::array<float, 4> unpack4_fp8_e5m2(u32 p) noexcept {
    return { fp8_e5m2_to_fp32((u8)(p & 0xFF)),
             fp8_e5m2_to_fp32((u8)((p >> 8)  & 0xFF)),
             fp8_e5m2_to_fp32((u8)((p >> 16) & 0xFF)),
             fp8_e5m2_to_fp32((u8)((p >> 24) & 0xFF)) };
}

// ---------------------------------------------------------------------------
// FP4 E2M1 / E3M0 — packed two-per-byte, eight-per-dword.
// ---------------------------------------------------------------------------
u8    fp32_to_fp4_e2m1(float x) noexcept;     // 4-bit value in low nibble
u8    fp32_to_fp4_e3m0(float x) noexcept;
float fp4_e2m1_to_fp32(u8 nib)  noexcept;
float fp4_e3m0_to_fp32(u8 nib)  noexcept;

// Pack 8× FP4 into a single dword. Lane 0 = bits 0..3, Lane 7 = bits 28..31.
u32 pack8_fp4_e2m1(const float v[8]) noexcept;
u32 pack8_fp4_e3m0(const float v[8]) noexcept;
cardinal::array<float, 8> unpack8_fp4_e2m1(u32 p) noexcept;
cardinal::array<float, 8> unpack8_fp4_e3m0(u32 p) noexcept;

// ---------------------------------------------------------------------------
// Round-trip — quantise + dequantise an FP32 through the named format.
// Useful for unit tests + the algo registry's CPU reference path.
// ---------------------------------------------------------------------------
float quantise(Format f, float x) noexcept;

}  // namespace cardinal::render::precision
