// =============================================================================
// cardinal_packed_math.hlsli — FP8 / FP4 packed-math (GPU side).
//
// Bit-for-bit counterpart to runtime/render/include/cardinal/render/precision.hpp.
// Each `cardinal_q*` function takes an FP32 and returns its quantised FP32
// (round-tripped through the named format). Each `cardinal_pack*` packs
// multiple lanes into a uint dword for storage.
//
// Native intrinsic path (Hopper / Ada / Blackwell-class hardware):
//   #define CARDINAL_HAS_FP8_INTRINSICS 1   // before including this header
// when the target SM exposes the matching intrinsics. Today DXC's exposure
// of FP8 / FP4 ops varies by version + target; the default emulated path
// works on every SM 6.0+ adapter, so it ships first.
//
// Layout convention — same as the CPU side:
//   pack4_fp8(a,b,c,d)        a → bits 0..7,   d → bits 24..31
//   pack8_fp4(v0..v7)         v0 → bits 0..3,  v7 → bits 28..31
// =============================================================================

#ifndef CARDINAL_PACKED_MATH_HLSLI
#define CARDINAL_PACKED_MATH_HLSLI

#ifndef CARDINAL_HAS_FP8_INTRINSICS
    #define CARDINAL_HAS_FP8_INTRINSICS 0
#endif

// ---------------------------------------------------------------------------
// FP16 round-trip (already supported natively, exposed here for symmetry).
// ---------------------------------------------------------------------------
float cardinal_q_fp16(float x) {
    return (float)f16tof32(f32tof16(x));
}

// ---------------------------------------------------------------------------
// Helper: bit pattern of an FP32 (uint).
// ---------------------------------------------------------------------------
uint  cardinal_fp32_bits(float f) { return asuint(f); }
float cardinal_bits_fp32(uint  u) { return asfloat(u); }

// ---------------------------------------------------------------------------
// FP8 — emulated round-trip via integer ops. Mirrors quantise_to_packed +
// dequantise_packed in the CPU reference.
// ---------------------------------------------------------------------------
//   spec:  exp_bits, mant_bits, bias, has_inf
//   layout: [sign:1][exp:exp_bits][mant:mant_bits]
//
// We unroll the kernel for each format (E4M3 / E5M2) so the compiler can
// constant-fold the masks + shifts.
//
// Round-to-nearest-even on the dropped mantissa bits (round_bit + sticky).

uint _cardinal_q_packed(float x, int exp_bits, int mant_bits, int bias,
                        bool has_inf)
{
    uint u    = asuint(x);
    uint sign = (u >> 31) & 1u;
    uint e32  = (u >> 23) & 0xFFu;
    uint m32  = u & 0x7FFFFFu;
    uint maxe = (1u << exp_bits) - 1u;

    if (e32 == 0xFFu) {
        // NaN — propagate.
        if (m32 != 0u) {
            return has_inf
                ? (sign << (exp_bits + mant_bits)) | (maxe << mant_bits) | 1u
                : (sign << (exp_bits + mant_bits)) | (maxe << mant_bits) | ((1u << mant_bits) - 1u);
        }
        // Inf or saturate.
        return has_inf
            ? (sign << (exp_bits + mant_bits)) | (maxe << mant_bits)
            : (sign << (exp_bits + mant_bits)) | ((maxe - 1u) << mant_bits) | ((1u << mant_bits) - 1u);
    }
    if (e32 == 0u && m32 == 0u) return sign << (exp_bits + mant_bits);

    int unbiased = (int)e32 - 127;
    int e_target = unbiased + bias;
    uint mant_full   = (e32 == 0u) ? m32 : (m32 | 0x800000u);
    int  mant_msb    = (e32 == 0u) ? 22 : 23;
    int  shift       = max(mant_msb - mant_bits, 0);

    // Subnormal target.
    if (e_target <= 0) {
        int extra = 1 - e_target;
        if (extra > 24) return sign << (exp_bits + mant_bits);
        int sh = shift + extra;
        uint round_bit = (sh > 0) ? ((mant_full >> (sh - 1)) & 1u) : 0u;
        uint sticky    = (sh > 1) ? (mant_full & ((1u << (sh - 1)) - 1u)) : 0u;
        uint m         = (sh > 0) ? (mant_full >> sh) : mant_full;
        if (round_bit && (sticky != 0u || (m & 1u))) ++m;
        if (m == 0u) return sign << (exp_bits + mant_bits);
        if (m == (1u << mant_bits))
            return (sign << (exp_bits + mant_bits)) | ((uint)1 << mant_bits);
        return (sign << (exp_bits + mant_bits)) | m;
    }

    // Normal target.
    uint round_bit = (shift > 0) ? ((mant_full >> (shift - 1)) & 1u) : 0u;
    uint sticky    = (shift > 1) ? (mant_full & ((1u << (shift - 1)) - 1u)) : 0u;
    uint m_short   = (shift > 0) ? (mant_full >> shift) : mant_full;
    if (round_bit && (sticky != 0u || (m_short & 1u))) ++m_short;
    if (m_short >= (2u << mant_bits)) { m_short >>= 1; ++e_target; }
    m_short &= ((1u << mant_bits) - 1u);

    if (e_target >= (int)maxe) {
        return has_inf
            ? (sign << (exp_bits + mant_bits)) | (maxe << mant_bits)
            : (sign << (exp_bits + mant_bits)) | ((maxe - 1u) << mant_bits) | ((1u << mant_bits) - 1u);
    }
    return (sign << (exp_bits + mant_bits))
         | ((uint)e_target << mant_bits)
         | m_short;
}

float _cardinal_d_packed(uint v, int exp_bits, int mant_bits, int bias, bool has_inf) {
    uint sign = (v >> (exp_bits + mant_bits)) & 1u;
    uint e_pk = (v >> mant_bits) & ((1u << exp_bits) - 1u);
    uint m_pk = v & ((1u << mant_bits) - 1u);
    uint maxe = (1u << exp_bits) - 1u;

    if (e_pk == 0u) {
        if (m_pk == 0u) return sign ? -0.0 : 0.0;
        float frac = (float)m_pk / (float)(1u << mant_bits);
        float val  = ldexp(frac, 1 - bias);
        return sign ? -val : val;
    }
    if (e_pk == maxe) {
        if (has_inf) {
            if (m_pk == 0u) return sign ? -1.0 / 0.0 : 1.0 / 0.0;
            return 0.0 / 0.0;
        }
        if (m_pk == ((1u << mant_bits) - 1u)) return 0.0 / 0.0;
    }
    float frac = 1.0 + (float)m_pk / (float)(1u << mant_bits);
    float val  = ldexp(frac, (int)e_pk - bias);
    return sign ? -val : val;
}

// FP8 E4M3
uint  cardinal_qbits_fp8_e4m3(float x) { return _cardinal_q_packed(x, 4, 3, 7,  false); }
float cardinal_dq_fp8_e4m3   (uint v)  { return _cardinal_d_packed(v, 4, 3, 7,  false); }
float cardinal_q_fp8_e4m3    (float x) { return cardinal_dq_fp8_e4m3(cardinal_qbits_fp8_e4m3(x)); }

// FP8 E5M2
uint  cardinal_qbits_fp8_e5m2(float x) { return _cardinal_q_packed(x, 5, 2, 15, true); }
float cardinal_dq_fp8_e5m2   (uint v)  { return _cardinal_d_packed(v, 5, 2, 15, true); }
float cardinal_q_fp8_e5m2    (float x) { return cardinal_dq_fp8_e5m2(cardinal_qbits_fp8_e5m2(x)); }

// FP4 E2M1
uint  cardinal_qbits_fp4_e2m1(float x) { return _cardinal_q_packed(x, 2, 1, 1, false) & 0xFu; }
float cardinal_dq_fp4_e2m1   (uint v)  { return _cardinal_d_packed(v & 0xFu, 2, 1, 1, false); }
float cardinal_q_fp4_e2m1    (float x) { return cardinal_dq_fp4_e2m1(cardinal_qbits_fp4_e2m1(x)); }

// FP4 E3M0
uint  cardinal_qbits_fp4_e3m0(float x) { return _cardinal_q_packed(x, 3, 0, 3, false) & 0xFu; }
float cardinal_dq_fp4_e3m0   (uint v)  { return _cardinal_d_packed(v & 0xFu, 3, 0, 3, false); }
float cardinal_q_fp4_e3m0    (float x) { return cardinal_dq_fp4_e3m0(cardinal_qbits_fp4_e3m0(x)); }

// ---------------------------------------------------------------------------
// Packing helpers — match the CPU layout.
// ---------------------------------------------------------------------------
uint cardinal_pack4_fp8_e4m3(float4 v) {
    return  (cardinal_qbits_fp8_e4m3(v.x))
         | ((cardinal_qbits_fp8_e4m3(v.y) & 0xFFu) << 8)
         | ((cardinal_qbits_fp8_e4m3(v.z) & 0xFFu) << 16)
         | ((cardinal_qbits_fp8_e4m3(v.w) & 0xFFu) << 24);
}
uint cardinal_pack4_fp8_e5m2(float4 v) {
    return  (cardinal_qbits_fp8_e5m2(v.x))
         | ((cardinal_qbits_fp8_e5m2(v.y) & 0xFFu) << 8)
         | ((cardinal_qbits_fp8_e5m2(v.z) & 0xFFu) << 16)
         | ((cardinal_qbits_fp8_e5m2(v.w) & 0xFFu) << 24);
}
float4 cardinal_unpack4_fp8_e4m3(uint p) {
    return float4(
        cardinal_dq_fp8_e4m3((p)       & 0xFFu),
        cardinal_dq_fp8_e4m3((p >> 8)  & 0xFFu),
        cardinal_dq_fp8_e4m3((p >> 16) & 0xFFu),
        cardinal_dq_fp8_e4m3((p >> 24) & 0xFFu));
}
float4 cardinal_unpack4_fp8_e5m2(uint p) {
    return float4(
        cardinal_dq_fp8_e5m2((p)       & 0xFFu),
        cardinal_dq_fp8_e5m2((p >> 8)  & 0xFFu),
        cardinal_dq_fp8_e5m2((p >> 16) & 0xFFu),
        cardinal_dq_fp8_e5m2((p >> 24) & 0xFFu));
}

// FP4 — eight lanes per dword.
uint cardinal_pack8_fp4_e2m1(float v0, float v1, float v2, float v3,
                             float v4, float v5, float v6, float v7)
{
    uint r = 0;
    r |= cardinal_qbits_fp4_e2m1(v0)       ;
    r |= cardinal_qbits_fp4_e2m1(v1) << 4 ;
    r |= cardinal_qbits_fp4_e2m1(v2) << 8 ;
    r |= cardinal_qbits_fp4_e2m1(v3) << 12;
    r |= cardinal_qbits_fp4_e2m1(v4) << 16;
    r |= cardinal_qbits_fp4_e2m1(v5) << 20;
    r |= cardinal_qbits_fp4_e2m1(v6) << 24;
    r |= cardinal_qbits_fp4_e2m1(v7) << 28;
    return r;
}

#endif  // CARDINAL_PACKED_MATH_HLSLI
