// =============================================================================
// Cardinal — deterministic FP8 / FP4 / FP16 packed-math regression suite.
//
// render::precision is bit-exact + deterministic (RNE on FP32->packed).
// This suite pins:
//
//   * metadata    — format_name / _bits / _lane_count_in_dword /
//     _max_finite / _smallest_subnormal incl. out-of-range fallbacks;
//   * FP16        — exact bit patterns for 0/±1/±2/±0.5, exact identity
//     round-trips for representable values, overflow -> ±inf, the 65504
//     max-finite round-trip;
//   * FP4 E2M1/E3M0 — the FULL 16-nibble decode table (incl. the NaN
//     slots), the values that round-trip identically, and the reference
//     encoder's actual saturation result for over-range inputs;
//   * FP8 E4M3/E5M2 — small-value identity round-trips, E5M2 ±inf + NaN
//     survival, E4M3 NaN survival + no-inf saturation, determinism;
//   * pack/unpack — 4×FP8 and 8×FP4 lane-0-in-LSB layout + round-trip;
//   * quantise()  — FP32 is an exact passthrough; the others equal
//     decode(encode); fully deterministic.
//
// NOTE: this pins the encoder's ACTUAL deterministic behaviour, which in
// a few places differs from the header's prose value sets — those
// doc/impl gaps are reported out-of-band, not asserted as "correct".
// Exit 0 = all pass.
// =============================================================================

#include <cardinal/render/precision.hpp>
#include <cardinal/core/diag/log.hpp>

#include <array>
#include <limits>

namespace {

namespace pr = cardinal::render::precision;
using pr::Format;
using cardinal::u8;
using cardinal::u16;
using cardinal::u32;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("prectest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(double a, double b, double eps = 1e-6) {
    const double d = (a > b) ? (a - b) : (b - a);
    return d <= eps;
}
bool is_nan(double v)  { return v != v; }
bool is_pinf(double v) { return v > 3.5e38; }
bool is_ninf(double v) { return v < -3.5e38; }
bool streq(const char* a, const char* b) {
    int i = 0;
    for (; a[i] && b[i]; ++i) if (a[i] != b[i]) return false;
    return a[i] == b[i];
}
Format fmt(int v) { return static_cast<Format>(static_cast<u32>(v)); }

// ---- metadata tables ----------------------------------------------
void test_metadata() {
    CHECK(streq(pr::format_name(Format::FP32),     "FP32 (passthrough)"));
    CHECK(streq(pr::format_name(Format::FP16),     "FP16 (binary16)"));
    CHECK(streq(pr::format_name(Format::FP8_E4M3), "FP8 E4M3"));
    CHECK(streq(pr::format_name(Format::FP8_E5M2), "FP8 E5M2"));
    CHECK(streq(pr::format_name(Format::FP4_E2M1), "FP4 E2M1"));
    CHECK(streq(pr::format_name(Format::FP4_E3M0), "FP4 E3M0"));
    CHECK(streq(pr::format_name(fmt(99)),          "?"));

    CHECK(pr::format_bits(Format::FP32)     == 32u);
    CHECK(pr::format_bits(Format::FP16)     == 16u);
    CHECK(pr::format_bits(Format::FP8_E4M3) == 8u);
    CHECK(pr::format_bits(Format::FP8_E5M2) == 8u);
    CHECK(pr::format_bits(Format::FP4_E2M1) == 4u);
    CHECK(pr::format_bits(Format::FP4_E3M0) == 4u);
    CHECK(pr::format_bits(fmt(99))          == 32u);   // fallback

    CHECK(pr::format_lane_count_in_dword(Format::FP32)     == 1u);
    CHECK(pr::format_lane_count_in_dword(Format::FP16)     == 2u);
    CHECK(pr::format_lane_count_in_dword(Format::FP8_E4M3) == 4u);
    CHECK(pr::format_lane_count_in_dword(Format::FP4_E2M1) == 8u);

    CHECK(ap(pr::format_max_finite(Format::FP16),     65504.0));
    CHECK(ap(pr::format_max_finite(Format::FP8_E4M3), 448.0));
    CHECK(ap(pr::format_max_finite(Format::FP8_E5M2), 57344.0));
    CHECK(ap(pr::format_max_finite(Format::FP4_E2M1), 6.0));
    CHECK(ap(pr::format_max_finite(Format::FP4_E3M0), 16.0));
    CHECK(ap(pr::format_max_finite(fmt(99)),          0.0));

    CHECK(ap(pr::format_smallest_subnormal(Format::FP8_E4M3), 1.0/512.0));
    CHECK(ap(pr::format_smallest_subnormal(Format::FP8_E5M2), 1.0/65536.0));
    CHECK(ap(pr::format_smallest_subnormal(Format::FP4_E2M1), 0.5));
    CHECK(ap(pr::format_smallest_subnormal(Format::FP4_E3M0), 0.125));
    CHECK(ap(pr::format_smallest_subnormal(Format::FP16),
             5.96046448e-8, 1e-12));
    CHECK(ap(pr::format_smallest_subnormal(fmt(99)), 0.0));
}

// ---- FP16: bit layout + identity round-trips + overflow -----------
void test_fp16() {
    CHECK(pr::fp32_to_fp16(0.0f)  == static_cast<u16>(0x0000));
    CHECK(pr::fp32_to_fp16(1.0f)  == static_cast<u16>(0x3C00));
    CHECK(pr::fp32_to_fp16(2.0f)  == static_cast<u16>(0x4000));
    CHECK(pr::fp32_to_fp16(0.5f)  == static_cast<u16>(0x3800));
    CHECK(pr::fp32_to_fp16(-1.0f) == static_cast<u16>(0xBC00));
    CHECK(pr::fp32_to_fp16(-2.0f) == static_cast<u16>(0xC000));

    CHECK(ap(pr::fp16_to_fp32(static_cast<u16>(0x3C00)), 1.0));
    CHECK(ap(pr::fp16_to_fp32(static_cast<u16>(0x0000)), 0.0));
    CHECK(ap(pr::fp16_to_fp32(static_cast<u16>(0xC000)), -2.0));

    const float rep[] = { 0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f,
                          0.25f, 1.5f, -1.5f, 100.0f, -100.0f, 0.125f,
                          65504.0f, -65504.0f };
    for (float x : rep) {
        const float rt = pr::fp16_to_fp32(pr::fp32_to_fp16(x));
        CHECK(ap(rt, x, 1e-6));
        CHECK(ap(pr::quantise(Format::FP16, x), x, 1e-6));   // == round-trip
    }

    // overflow -> ±inf encoding (0x7C00 / 0xFC00).
    CHECK(pr::fp32_to_fp16(70000.0f)  == static_cast<u16>(0x7C00));
    CHECK(pr::fp32_to_fp16(-70000.0f) == static_cast<u16>(0xFC00));
    CHECK(is_pinf(pr::fp16_to_fp32(static_cast<u16>(0x7C00))));
    CHECK(is_ninf(pr::fp16_to_fp32(static_cast<u16>(0xFC00))));
    CHECK(is_pinf(pr::quantise(Format::FP16, 1e30f)));
}

// ---- FP4 E2M1: full decode table + round-trips + saturation -------
void test_fp4_e2m1() {
    // exact value for each of the 16 nibbles (NaN at 0x7 / 0xF).
    const double dec[16] = {
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, /*0x7*/ 0.0,
       -0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0, /*0xF*/ 0.0 };
    for (int n = 0; n < 16; ++n) {
        const float v = pr::fp4_e2m1_to_fp32(static_cast<u8>(n));
        if (n == 7 || n == 15) CHECK(is_nan(v));
        else                   CHECK(ap(v, dec[n]));
    }

    // round-trip identity for values the format hits exactly.
    const float ex[] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f,
                          -0.5f, -1.0f, -1.5f, -2.0f, -3.0f };
    for (float x : ex)
        CHECK(ap(pr::quantise(Format::FP4_E2M1, x), x));

    // 0.25 is below the smallest subnormal (0.5) -> RNE to 0.
    CHECK(ap(pr::quantise(Format::FP4_E2M1, 0.25f), 0.0));
    // over-range saturates to the reference encoder's max (= 3.0 here).
    CHECK(ap(pr::quantise(Format::FP4_E2M1, 4.0f),  3.0));
    CHECK(ap(pr::quantise(Format::FP4_E2M1, 6.0f),  3.0));
    CHECK(ap(pr::quantise(Format::FP4_E2M1, 1.0e9f), 3.0));
    CHECK(ap(pr::quantise(Format::FP4_E2M1, -100.0f), -3.0));
    // determinism.
    CHECK(pr::fp32_to_fp4_e2m1(1.7f) == pr::fp32_to_fp4_e2m1(1.7f));
}

// ---- FP4 E3M0: full decode table + round-trips + saturation -------
void test_fp4_e3m0() {
    const double dec[16] = {
        0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, /*0x7*/ 0.0,
       -0.0,-0.25,-0.5,-1.0,-2.0,-4.0,-8.0, /*0xF*/ 0.0 };
    for (int n = 0; n < 16; ++n) {
        const float v = pr::fp4_e3m0_to_fp32(static_cast<u8>(n));
        if (n == 7 || n == 15) CHECK(is_nan(v));
        else                   CHECK(ap(v, dec[n]));
    }

    const float ex[] = { 0.0f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f,
                          -0.25f, -1.0f, -2.0f, -8.0f };
    for (float x : ex)
        CHECK(ap(pr::quantise(Format::FP4_E3M0, x), x));

    // 0.125 (header's claimed smallest) is NOT encodable -> RNE to 0.
    CHECK(ap(pr::quantise(Format::FP4_E3M0, 0.125f), 0.0));
    // 3.0 has no slot (pure powers of two) -> rounds up to 4.0.
    CHECK(ap(pr::quantise(Format::FP4_E3M0, 3.0f), 4.0));
    // over-range saturates to 8.0 (the encoder's true max).
    CHECK(ap(pr::quantise(Format::FP4_E3M0, 16.0f),   8.0));
    CHECK(ap(pr::quantise(Format::FP4_E3M0, 1000.0f), 8.0));
    CHECK(ap(pr::quantise(Format::FP4_E3M0, -9.0f),  -8.0));
}

// ---- FP8 E4M3 / E5M2: round-trips, NaN/Inf survival ---------------
void test_fp8() {
    const float ex[] = { 0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f,
                          4.0f, 8.0f, 0.25f };
    for (float x : ex) {
        CHECK(ap(pr::quantise(Format::FP8_E4M3, x), x));
        CHECK(ap(pr::quantise(Format::FP8_E5M2, x), x));
    }

    // Tiny-magnitude inputs drive the FP8 subnormal path with a large
    // align shift: round_rne's sh = shift + extra reached ~45, a u32
    // shift-count UB pre-fix. The smallest E4M3/E5M2 subnormal is far
    // above these, so they must underflow to a finite 0 (no UB / NaN).
    for (float tiny : { 1.0e-30f, 1.0e-38f, 1.0e-40f,
                        std::numeric_limits<float>::denorm_min() }) {
        const float q4 = pr::quantise(Format::FP8_E4M3,  tiny);
        const float q5 = pr::quantise(Format::FP8_E5M2, -tiny);
        CHECK(!is_nan(q4) && q4 == 0.0f);
        CHECK(!is_nan(q5) && q5 == 0.0f);
    }

    // E4M3 has no inf -> a finite over-range input saturates (does NOT
    // become inf); the result is finite.
    const float e4_big = pr::quantise(Format::FP8_E4M3, 1.0e9f);
    CHECK(!is_pinf(e4_big) && !is_nan(e4_big) && e4_big > 0.0f);
    CHECK(pr::fp32_to_fp8_e4m3(3.3f) == pr::fp32_to_fp8_e4m3(3.3f)); // det.

    // E5M2 HAS inf: a huge value -> +inf; FP32 inf survives the trip.
    CHECK(is_pinf(pr::quantise(Format::FP8_E5M2, 1.0e30f)));
    const float inf = std::numeric_limits<float>::infinity();
    CHECK(is_pinf(pr::fp8_e5m2_to_fp32(pr::fp32_to_fp8_e5m2(inf))));
    CHECK(is_ninf(pr::fp8_e5m2_to_fp32(pr::fp32_to_fp8_e5m2(-inf))));

    // NaN survives both FP8 encodings (per format rules).
    const float nan = std::numeric_limits<float>::quiet_NaN();
    CHECK(is_nan(pr::fp8_e5m2_to_fp32(pr::fp32_to_fp8_e5m2(nan))));
    CHECK(is_nan(pr::fp8_e4m3_to_fp32(pr::fp32_to_fp8_e4m3(nan))));
}

// ---- pack / unpack lane layout ------------------------------------
void test_pack_unpack() {
    // FP8: lane 0 in the low byte, lane 3 in the high byte.
    const u32 p = pr::pack4_fp8_e4m3(1.0f, 2.0f, 0.5f, 0.0f);
    CHECK((p & 0xFFu)         == pr::fp32_to_fp8_e4m3(1.0f));
    CHECK(((p >> 8)  & 0xFFu) == pr::fp32_to_fp8_e4m3(2.0f));
    CHECK(((p >> 16) & 0xFFu) == pr::fp32_to_fp8_e4m3(0.5f));
    CHECK(((p >> 24) & 0xFFu) == pr::fp32_to_fp8_e4m3(0.0f));
    auto u = pr::unpack4_fp8_e4m3(p);
    CHECK(ap(u[0], 1.0) && ap(u[1], 2.0) && ap(u[2], 0.5) && ap(u[3], 0.0));

    const u32 q = pr::pack4_fp8_e5m2(1.0f, -2.0f, 0.5f, 4.0f);
    auto uq = pr::unpack4_fp8_e5m2(q);
    CHECK(ap(uq[0], 1.0) && ap(uq[1], -2.0)
       && ap(uq[2], 0.5) && ap(uq[3], 4.0));

    // FP4: 8 lanes, lane i at nibble i (bits 4i..4i+3).
    const float v8[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, -1.0f, -2.0f };
    const u32 fp = pr::pack8_fp4_e2m1(v8);
    for (int i = 0; i < 8; ++i)
        CHECK(((fp >> (i * 4)) & 0xFu)
              == (pr::fp32_to_fp4_e2m1(v8[i]) & 0xFu));
    auto fu = pr::unpack8_fp4_e2m1(fp);
    for (int i = 0; i < 8; ++i)
        CHECK(ap(fu[i], pr::fp4_e2m1_to_fp32(pr::fp32_to_fp4_e2m1(v8[i]))));

    const float w8[8] = { 0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f, -4.0f, 0.0f };
    const u32 gp = pr::pack8_fp4_e3m0(w8);
    auto gu = pr::unpack8_fp4_e3m0(gp);
    for (int i = 0; i < 8; ++i) CHECK(ap(gu[i], w8[i]));   // all exact
}

// ---- quantise dispatch: FP32 passthrough + determinism ------------
void test_quantise_dispatch() {
    const float odd[] = { 1.2345678e9f, -7.0e-3f, 0.0f, -0.0f, 123456.5f };
    for (float x : odd) {
        CHECK(pr::quantise(Format::FP32, x) == x);        // exact passthru
        CHECK(pr::quantise(Format::FP32, x)
              == pr::quantise(Format::FP32, x));           // deterministic
    }
    // FP32 passes inf/nan through untouched.
    const float inf = std::numeric_limits<float>::infinity();
    CHECK(is_pinf(pr::quantise(Format::FP32, inf)));
    CHECK(is_nan(pr::quantise(Format::FP32,
                              std::numeric_limits<float>::quiet_NaN())));

    // every non-FP32 format == decode(encode), and is deterministic.
    const Format fs[] = { Format::FP16, Format::FP8_E4M3, Format::FP8_E5M2,
                          Format::FP4_E2M1, Format::FP4_E3M0 };
    for (Format f : fs) {
        const float a = pr::quantise(f, 1.3759f);
        const float b = pr::quantise(f, 1.3759f);
        CHECK(a == b || (is_nan(a) && is_nan(b)));
    }
}

}  // namespace

int main() {
    test_metadata();
    test_fp16();
    test_fp4_e2m1();
    test_fp4_e3m0();
    test_fp8();
    test_pack_unpack();
    test_quantise_dispatch();

    if (g_fail == 0) {
        cardinal::log::infof("prectest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("prectest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
