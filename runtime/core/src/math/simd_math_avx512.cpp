// =============================================================================
// Cardinal — SIMD math: AVX-512 (512-bit, 16 float lanes).
//
// Compiled with /arch:AVX512 (MSVC) — see CMakeLists. The dispatcher
// only binds these kernels when CpuFeatures::avx512f is true at boot,
// so any CPU lacking the ISA never executes this code.
//
// We also use AVX-512 masked instructions for the trailing-tail case
// (no scalar epilogue needed) — that's the architectural payoff over
// just doing two AVX2 chunks.
// =============================================================================
#define CARDINAL_SIMD_TIER avx512
#include "simd_math_kernels.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cardinal::core::simd::avx512 {

// 16-lane store-mask for the trailing tail. n_remaining ∈ [0, 16].
static inline __mmask16 tail_mask(usize n_remaining) noexcept {
    return static_cast<__mmask16>((1u << n_remaining) - 1u);
}

void vec_add_f32(f32* out, const f32* a, const f32* b, usize n) {
    usize i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        _mm512_storeu_ps(out + i, _mm512_add_ps(va, vb));
    }
    if (i < n) {
        __mmask16 m = tail_mask(n - i);
        __m512 va = _mm512_maskz_loadu_ps(m, a + i);
        __m512 vb = _mm512_maskz_loadu_ps(m, b + i);
        _mm512_mask_storeu_ps(out + i, m, _mm512_add_ps(va, vb));
    }
}
void vec_sub_f32(f32* out, const f32* a, const f32* b, usize n) {
    usize i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i,
            _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    }
    if (i < n) {
        __mmask16 m = tail_mask(n - i);
        _mm512_mask_storeu_ps(out + i, m, _mm512_sub_ps(
            _mm512_maskz_loadu_ps(m, a + i),
            _mm512_maskz_loadu_ps(m, b + i)));
    }
}
void vec_mul_f32(f32* out, const f32* a, const f32* b, usize n) {
    usize i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i,
            _mm512_mul_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    }
    if (i < n) {
        __mmask16 m = tail_mask(n - i);
        _mm512_mask_storeu_ps(out + i, m, _mm512_mul_ps(
            _mm512_maskz_loadu_ps(m, a + i),
            _mm512_maskz_loadu_ps(m, b + i)));
    }
}
void vec_scale_f32(f32* out, const f32* a, f32 s, usize n) {
    const __m512 vs = _mm512_set1_ps(s);
    usize i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), vs));
    }
    if (i < n) {
        __mmask16 m = tail_mask(n - i);
        _mm512_mask_storeu_ps(out + i, m,
            _mm512_mul_ps(_mm512_maskz_loadu_ps(m, a + i), vs));
    }
}
void vec_axpy_f32(f32* y, f32 a, const f32* x, usize n) {
    const __m512 va = _mm512_set1_ps(a);
    usize i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vy = _mm512_loadu_ps(y + i);
        _mm512_storeu_ps(y + i, _mm512_fmadd_ps(va, vx, vy));
    }
    if (i < n) {
        __mmask16 m = tail_mask(n - i);
        __m512 vx = _mm512_maskz_loadu_ps(m, x + i);
        __m512 vy = _mm512_maskz_loadu_ps(m, y + i);
        _mm512_mask_storeu_ps(y + i, m, _mm512_fmadd_ps(va, vx, vy));
    }
}

f32 dot_f32(const f32* a, const f32* b, usize n) {
    __m512 acc = _mm512_setzero_ps();
    usize i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    if (i < n) {
        __mmask16 m = tail_mask(n - i);
        __m512 va = _mm512_maskz_loadu_ps(m, a + i);
        __m512 vb = _mm512_maskz_loadu_ps(m, b + i);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    return _mm512_reduce_add_ps(acc);
}
f32 sum_f32(const f32* a, usize n) {
    __m512 acc = _mm512_setzero_ps();
    usize i = 0;
    for (; i + 16 <= n; i += 16) {
        acc = _mm512_add_ps(acc, _mm512_loadu_ps(a + i));
    }
    if (i < n) {
        __mmask16 m = tail_mask(n - i);
        acc = _mm512_add_ps(acc, _mm512_maskz_loadu_ps(m, a + i));
    }
    return _mm512_reduce_add_ps(acc);
}
f32 min_f32(const f32* a, usize n) {
    if (n == 0) return std::numeric_limits<f32>::infinity();
    __m512 acc = _mm512_set1_ps(a[0]);
    usize i = 0;
    for (; i + 16 <= n; i += 16) {
        acc = _mm512_min_ps(acc, _mm512_loadu_ps(a + i));
    }
    if (i < n) {
        __mmask16 m = tail_mask(n - i);
        // Initialise masked-off lanes to +inf so they don't influence the min.
        __m512 v = _mm512_mask_loadu_ps(_mm512_set1_ps(
            std::numeric_limits<f32>::infinity()), m, a + i);
        acc = _mm512_min_ps(acc, v);
    }
    return _mm512_reduce_min_ps(acc);
}
f32 max_f32(const f32* a, usize n) {
    if (n == 0) return -std::numeric_limits<f32>::infinity();
    __m512 acc = _mm512_set1_ps(a[0]);
    usize i = 0;
    for (; i + 16 <= n; i += 16) {
        acc = _mm512_max_ps(acc, _mm512_loadu_ps(a + i));
    }
    if (i < n) {
        __mmask16 m = tail_mask(n - i);
        __m512 v = _mm512_mask_loadu_ps(_mm512_set1_ps(
            -std::numeric_limits<f32>::infinity()), m, a + i);
        acc = _mm512_max_ps(acc, v);
    }
    return _mm512_reduce_max_ps(acc);
}

void transform_points_mat4(f32* out_xyz, const f32* in_xyz,
                           const Mat4Compact& M, usize count)
{
    const f32* m = M.m;
    const __m512 m00 = _mm512_set1_ps(m[0]);  const __m512 m04 = _mm512_set1_ps(m[4]);
    const __m512 m08 = _mm512_set1_ps(m[8]);  const __m512 m12 = _mm512_set1_ps(m[12]);
    const __m512 m01 = _mm512_set1_ps(m[1]);  const __m512 m05 = _mm512_set1_ps(m[5]);
    const __m512 m09 = _mm512_set1_ps(m[9]);  const __m512 m13 = _mm512_set1_ps(m[13]);
    const __m512 m02 = _mm512_set1_ps(m[2]);  const __m512 m06 = _mm512_set1_ps(m[6]);
    const __m512 m10 = _mm512_set1_ps(m[10]); const __m512 m14 = _mm512_set1_ps(m[14]);

    usize i = 0;
    for (; i + 16 <= count; i += 16) {
        const f32* p = in_xyz + i*3;
        // Gather 16 verts' xs / ys / zs from 48 floats of AoS input.
        const __m512 xs = _mm512_set_ps(
            p[45], p[42], p[39], p[36], p[33], p[30], p[27], p[24],
            p[21], p[18], p[15], p[12], p[ 9], p[ 6], p[ 3], p[ 0]);
        const __m512 ys = _mm512_set_ps(
            p[46], p[43], p[40], p[37], p[34], p[31], p[28], p[25],
            p[22], p[19], p[16], p[13], p[10], p[ 7], p[ 4], p[ 1]);
        const __m512 zs = _mm512_set_ps(
            p[47], p[44], p[41], p[38], p[35], p[32], p[29], p[26],
            p[23], p[20], p[17], p[14], p[11], p[ 8], p[ 5], p[ 2]);

        __m512 ox = _mm512_fmadd_ps(m00, xs, m12);
        ox = _mm512_fmadd_ps(m04, ys, ox);
        ox = _mm512_fmadd_ps(m08, zs, ox);

        __m512 oy = _mm512_fmadd_ps(m01, xs, m13);
        oy = _mm512_fmadd_ps(m05, ys, oy);
        oy = _mm512_fmadd_ps(m09, zs, oy);

        __m512 oz = _mm512_fmadd_ps(m02, xs, m14);
        oz = _mm512_fmadd_ps(m06, ys, oz);
        oz = _mm512_fmadd_ps(m10, zs, oz);

        alignas(64) f32 ox_arr[16], oy_arr[16], oz_arr[16];
        _mm512_store_ps(ox_arr, ox);
        _mm512_store_ps(oy_arr, oy);
        _mm512_store_ps(oz_arr, oz);
        f32* o = out_xyz + i*3;
        for (int k = 0; k < 16; ++k) {
            o[k*3 + 0] = ox_arr[k];
            o[k*3 + 1] = oy_arr[k];
            o[k*3 + 2] = oz_arr[k];
        }
    }
    // Tail in scalar — splitting up an AoS triple into masked-load
    // 16-lane gather is more code than it's worth for the trailing
    // <16 verts.
    for (; i < count; ++i) {
        const f32 x = in_xyz[i*3 + 0];
        const f32 y = in_xyz[i*3 + 1];
        const f32 z = in_xyz[i*3 + 2];
        out_xyz[i*3 + 0] = m[0]*x + m[4]*y + m[8] *z + m[12];
        out_xyz[i*3 + 1] = m[1]*x + m[5]*y + m[9] *z + m[13];
        out_xyz[i*3 + 2] = m[2]*x + m[6]*y + m[10]*z + m[14];
    }
}

void vec3_cross_array(f32* out_xyz, const f32* a_xyz, const f32* b_xyz, usize count) {
    usize i = 0;
    for (; i + 16 <= count; i += 16) {
        const f32* a = a_xyz + i*3;
        const f32* b = b_xyz + i*3;
        const __m512 ax = _mm512_set_ps(
            a[45], a[42], a[39], a[36], a[33], a[30], a[27], a[24],
            a[21], a[18], a[15], a[12], a[ 9], a[ 6], a[ 3], a[ 0]);
        const __m512 ay = _mm512_set_ps(
            a[46], a[43], a[40], a[37], a[34], a[31], a[28], a[25],
            a[22], a[19], a[16], a[13], a[10], a[ 7], a[ 4], a[ 1]);
        const __m512 az = _mm512_set_ps(
            a[47], a[44], a[41], a[38], a[35], a[32], a[29], a[26],
            a[23], a[20], a[17], a[14], a[11], a[ 8], a[ 5], a[ 2]);
        const __m512 bx = _mm512_set_ps(
            b[45], b[42], b[39], b[36], b[33], b[30], b[27], b[24],
            b[21], b[18], b[15], b[12], b[ 9], b[ 6], b[ 3], b[ 0]);
        const __m512 by = _mm512_set_ps(
            b[46], b[43], b[40], b[37], b[34], b[31], b[28], b[25],
            b[22], b[19], b[16], b[13], b[10], b[ 7], b[ 4], b[ 1]);
        const __m512 bz = _mm512_set_ps(
            b[47], b[44], b[41], b[38], b[35], b[32], b[29], b[26],
            b[23], b[20], b[17], b[14], b[11], b[ 8], b[ 5], b[ 2]);

        const __m512 ox = _mm512_fmsub_ps(ay, bz, _mm512_mul_ps(az, by));
        const __m512 oy = _mm512_fmsub_ps(az, bx, _mm512_mul_ps(ax, bz));
        const __m512 oz = _mm512_fmsub_ps(ax, by, _mm512_mul_ps(ay, bx));

        alignas(64) f32 xa[16], ya[16], za[16];
        _mm512_store_ps(xa, ox); _mm512_store_ps(ya, oy); _mm512_store_ps(za, oz);
        f32* o = out_xyz + i*3;
        for (int k = 0; k < 16; ++k) {
            o[k*3+0] = xa[k]; o[k*3+1] = ya[k]; o[k*3+2] = za[k];
        }
    }
    for (; i < count; ++i) {
        const f32 ax = a_xyz[i*3+0], ay = a_xyz[i*3+1], az = a_xyz[i*3+2];
        const f32 bx = b_xyz[i*3+0], by = b_xyz[i*3+1], bz = b_xyz[i*3+2];
        out_xyz[i*3+0] = ay*bz - az*by;
        out_xyz[i*3+1] = az*bx - ax*bz;
        out_xyz[i*3+2] = ax*by - ay*bx;
    }
}

void vec3_length_array(f32* out_lens, const f32* in_xyz, usize count) {
    usize i = 0;
    for (; i + 16 <= count; i += 16) {
        const f32* p = in_xyz + i*3;
        const __m512 xs = _mm512_set_ps(
            p[45], p[42], p[39], p[36], p[33], p[30], p[27], p[24],
            p[21], p[18], p[15], p[12], p[ 9], p[ 6], p[ 3], p[ 0]);
        const __m512 ys = _mm512_set_ps(
            p[46], p[43], p[40], p[37], p[34], p[31], p[28], p[25],
            p[22], p[19], p[16], p[13], p[10], p[ 7], p[ 4], p[ 1]);
        const __m512 zs = _mm512_set_ps(
            p[47], p[44], p[41], p[38], p[35], p[32], p[29], p[26],
            p[23], p[20], p[17], p[14], p[11], p[ 8], p[ 5], p[ 2]);
        __m512 sq = _mm512_mul_ps(xs, xs);
        sq = _mm512_fmadd_ps(ys, ys, sq);
        sq = _mm512_fmadd_ps(zs, zs, sq);
        _mm512_storeu_ps(out_lens + i, _mm512_sqrt_ps(sq));
    }
    for (; i < count; ++i) {
        const f32 x = in_xyz[i*3+0], y = in_xyz[i*3+1], z = in_xyz[i*3+2];
        out_lens[i] = std::sqrt(x*x + y*y + z*z);
    }
}

void vec3_normalize_array_inplace(f32* io_xyz, usize count) {
    const __m512 eps_v = _mm512_set1_ps(1e-8f);
    usize i = 0;
    for (; i + 16 <= count; i += 16) {
        f32* p = io_xyz + i*3;
        const __m512 xs = _mm512_set_ps(
            p[45], p[42], p[39], p[36], p[33], p[30], p[27], p[24],
            p[21], p[18], p[15], p[12], p[ 9], p[ 6], p[ 3], p[ 0]);
        const __m512 ys = _mm512_set_ps(
            p[46], p[43], p[40], p[37], p[34], p[31], p[28], p[25],
            p[22], p[19], p[16], p[13], p[10], p[ 7], p[ 4], p[ 1]);
        const __m512 zs = _mm512_set_ps(
            p[47], p[44], p[41], p[38], p[35], p[32], p[29], p[26],
            p[23], p[20], p[17], p[14], p[11], p[ 8], p[ 5], p[ 2]);
        __m512 sq = _mm512_mul_ps(xs, xs);
        sq = _mm512_fmadd_ps(ys, ys, sq);
        sq = _mm512_fmadd_ps(zs, zs, sq);
        // AVX-512: native mask register from cmp; mask_blend selects
        // the original (zero-vector lanes) or normalised value.
        const __mmask16 keep = _mm512_cmp_ps_mask(sq, eps_v, _CMP_GE_OQ);
        const __m512 inv = _mm512_div_ps(_mm512_set1_ps(1.0f), _mm512_sqrt_ps(sq));
        const __m512 nxs = _mm512_mask_blend_ps(keep, xs, _mm512_mul_ps(xs, inv));
        const __m512 nys = _mm512_mask_blend_ps(keep, ys, _mm512_mul_ps(ys, inv));
        const __m512 nzs = _mm512_mask_blend_ps(keep, zs, _mm512_mul_ps(zs, inv));
        alignas(64) f32 xa[16], ya[16], za[16];
        _mm512_store_ps(xa, nxs); _mm512_store_ps(ya, nys); _mm512_store_ps(za, nzs);
        for (int k = 0; k < 16; ++k) {
            p[k*3+0] = xa[k]; p[k*3+1] = ya[k]; p[k*3+2] = za[k];
        }
    }
    for (; i < count; ++i) {
        const f32 x = io_xyz[i*3+0], y = io_xyz[i*3+1], z = io_xyz[i*3+2];
        const f32 len2 = x*x + y*y + z*z;
        if (len2 < 1e-8f) continue;
        const f32 inv = 1.0f / std::sqrt(len2);
        io_xyz[i*3+0] = x * inv;
        io_xyz[i*3+1] = y * inv;
        io_xyz[i*3+2] = z * inv;
    }
}

// AVX-512 mat4_mul_array — genuine 512-bit quad-processor.
//
// We process FOUR 4×4 matrices per iteration by packing one 4-wide
// column from each of the 4 matrices into the 4 lanes of a single
// __m512:
//
//   ac_k = [ A0_col_k | A1_col_k | A2_col_k | A3_col_k ]    (k = 0..3)
//
// For each output column c, we then need a broadcast vector
//
//   bv_k = [ B0[c,k]×4 | B1[c,k]×4 | B2[c,k]×4 | B3[c,k]×4 ]
//
// which we build by packing the 4 matrices' B-column-c into one __m512
// and using vpermilps (intra-lane permute, 1-cycle latency on Zen 5
// and Sapphire Rapids — no expensive cross-lane shuffles). The FMA
// chain runs across all 16 lanes simultaneously, so we get the full
// 512-bit throughput benefit instead of the AVX2-equivalent 128-bit
// path the previous "pair" version was emitting.
//
// Trailing 1-3 matrices fall through to the 128-bit kernel — a 4-mat
// quad is the minimum useful 512-bit cell here, so a small remainder
// is cheap enough to do scalar-style.
void mat4_mul_array(f32* out, const f32* A, const f32* B, usize count) {
    // 128-bit kernel for the trailing remainder (count % 4 != 0).
    auto mul_one = [](const f32* a, const f32* b, f32* o) {
        const __m128 ac0 = _mm_loadu_ps(a + 0*4);
        const __m128 ac1 = _mm_loadu_ps(a + 1*4);
        const __m128 ac2 = _mm_loadu_ps(a + 2*4);
        const __m128 ac3 = _mm_loadu_ps(a + 3*4);
        for (int c = 0; c < 4; ++c) {
            const __m128 b0 = _mm_set1_ps(b[c*4 + 0]);
            const __m128 b1 = _mm_set1_ps(b[c*4 + 1]);
            const __m128 b2 = _mm_set1_ps(b[c*4 + 2]);
            const __m128 b3 = _mm_set1_ps(b[c*4 + 3]);
            __m128 oc = _mm_mul_ps(ac0, b0);
            oc = _mm_fmadd_ps(ac1, b1, oc);
            oc = _mm_fmadd_ps(ac2, b2, oc);
            oc = _mm_fmadd_ps(ac3, b3, oc);
            _mm_storeu_ps(o + c*4, oc);
        }
    };

    usize i = 0;
    for (; i + 4 <= count; i += 4) {
        const f32* a0 = A + (i+0) * 16;
        const f32* a1 = A + (i+1) * 16;
        const f32* a2 = A + (i+2) * 16;
        const f32* a3 = A + (i+3) * 16;
        const f32* b0 = B + (i+0) * 16;
        const f32* b1 = B + (i+1) * 16;
        const f32* b2 = B + (i+2) * 16;
        const f32* b3 = B + (i+3) * 16;
        f32*       o0 = out + (i+0) * 16;
        f32*       o1 = out + (i+1) * 16;
        f32*       o2 = out + (i+2) * 16;
        f32*       o3 = out + (i+3) * 16;

        // Pack column k from 4 matrices into one __m512.
        // _mm512_castps128_ps512 is a no-op cast that places the loaded
        // __m128 into lane 0 with upper lanes undefined; the three
        // inserts then fill lanes 1..3, fully defining the register
        // before any read.
        auto pack4_a = [&](int k) {
            __m512 v = _mm512_castps128_ps512(_mm_loadu_ps(a0 + k*4));
            v = _mm512_insertf32x4(v, _mm_loadu_ps(a1 + k*4), 1);
            v = _mm512_insertf32x4(v, _mm_loadu_ps(a2 + k*4), 2);
            v = _mm512_insertf32x4(v, _mm_loadu_ps(a3 + k*4), 3);
            return v;
        };
        const __m512 ac0 = pack4_a(0);
        const __m512 ac1 = pack4_a(1);
        const __m512 ac2 = pack4_a(2);
        const __m512 ac3 = pack4_a(3);

        // For each output column c, pack the matching B column from
        // the 4 matrices and broadcast each of its 4 elements within
        // each 128-bit lane via vpermilps. _MM_SHUFFLE(k,k,k,k) selects
        // element k from the lane and replicates it 4×.
        for (int c = 0; c < 4; ++c) {
            __m512 bc = _mm512_castps128_ps512(_mm_loadu_ps(b0 + c*4));
            bc = _mm512_insertf32x4(bc, _mm_loadu_ps(b1 + c*4), 1);
            bc = _mm512_insertf32x4(bc, _mm_loadu_ps(b2 + c*4), 2);
            bc = _mm512_insertf32x4(bc, _mm_loadu_ps(b3 + c*4), 3);

            const __m512 bv0 = _mm512_permute_ps(bc, _MM_SHUFFLE(0, 0, 0, 0));
            const __m512 bv1 = _mm512_permute_ps(bc, _MM_SHUFFLE(1, 1, 1, 1));
            const __m512 bv2 = _mm512_permute_ps(bc, _MM_SHUFFLE(2, 2, 2, 2));
            const __m512 bv3 = _mm512_permute_ps(bc, _MM_SHUFFLE(3, 3, 3, 3));

            __m512 oc = _mm512_mul_ps(ac0, bv0);
            oc = _mm512_fmadd_ps(ac1, bv1, oc);
            oc = _mm512_fmadd_ps(ac2, bv2, oc);
            oc = _mm512_fmadd_ps(ac3, bv3, oc);

            // Scatter the 4 result columns back to the 4 destinations.
            _mm_storeu_ps(o0 + c*4, _mm512_extractf32x4_ps(oc, 0));
            _mm_storeu_ps(o1 + c*4, _mm512_extractf32x4_ps(oc, 1));
            _mm_storeu_ps(o2 + c*4, _mm512_extractf32x4_ps(oc, 2));
            _mm_storeu_ps(o3 + c*4, _mm512_extractf32x4_ps(oc, 3));
        }
    }
    // Trailing 1-3 matrices via 128-bit kernel.
    for (; i < count; ++i) {
        mul_one(A + i*16, B + i*16, out + i*16);
    }
}

// Frustum cull — 16 spheres / iter (2 bytes / iter). AVX-512 mask
// registers replace the movemask round-trip — _mm512_cmp_ps_mask
// returns __mmask16 directly, _kand_mask16 ANDs the per-plane masks
// without round-tripping through float. Tail uses a single masked
// 16-lane pass with the mask trimmed to count_remaining bits.
void frustum_cull_spheres(u8* out_bits, const f32* planes,
                          const f32* cx, const f32* cy,
                          const f32* cz, const f32* r,
                          usize count)
{
    auto test16 = [&](__m512 vcx, __m512 vcy, __m512 vcz, __m512 vr) -> __mmask16 {
        __mmask16 acc = static_cast<__mmask16>(0xFFFF);  // all-1
        for (int p = 0; p < 6; ++p) {
            const __m512 nx = _mm512_set1_ps(planes[p*4 + 0]);
            const __m512 ny = _mm512_set1_ps(planes[p*4 + 1]);
            const __m512 nz = _mm512_set1_ps(planes[p*4 + 2]);
            const __m512 d  = _mm512_set1_ps(planes[p*4 + 3]);
            const __m512 d_plus_r = _mm512_add_ps(d, vr);
            __m512 t = _mm512_fmadd_ps(nz, vcz, d_plus_r);
            t       = _mm512_fmadd_ps(ny, vcy, t);
            t       = _mm512_fmadd_ps(nx, vcx, t);
            const __mmask16 pass =
                _mm512_cmp_ps_mask(t, _mm512_setzero_ps(), _CMP_GE_OQ);
            acc = _kand_mask16(acc, pass);
        }
        return acc;
    };

    usize i = 0;
    for (; i + 16 <= count; i += 16) {
        const __m512 vcx = _mm512_loadu_ps(cx + i);
        const __m512 vcy = _mm512_loadu_ps(cy + i);
        const __m512 vcz = _mm512_loadu_ps(cz + i);
        const __m512 vr  = _mm512_loadu_ps(r  + i);
        const __mmask16 m = test16(vcx, vcy, vcz, vr);
        // Write 2 bytes (16 bits). Output is byte-addressable, byte-
        // aligned at i/8 (i is a multiple of 16 → i/8 is even).
        out_bits[(i >> 3)    ] = static_cast<u8>( m       & 0xFFu);
        out_bits[(i >> 3) + 1] = static_cast<u8>((m >> 8) & 0xFFu);
    }
    if (i < count) {
        const usize n = count - i;            // n in [1, 15]
        const __mmask16 lm = tail_mask(n);
        // Masked-zero loads: lanes past the tail get 0. We mask the
        // result back down to `lm` so out-of-range lanes don't pollute.
        const __m512 vcx = _mm512_maskz_loadu_ps(lm, cx + i);
        const __m512 vcy = _mm512_maskz_loadu_ps(lm, cy + i);
        const __m512 vcz = _mm512_maskz_loadu_ps(lm, cz + i);
        const __m512 vr  = _mm512_maskz_loadu_ps(lm, r  + i);
        const __mmask16 m = static_cast<__mmask16>(test16(vcx, vcy, vcz, vr) & lm);
        out_bits[(i >> 3)    ] = static_cast<u8>( m       & 0xFFu);
        if (n > 8) out_bits[(i >> 3) + 1] = static_cast<u8>((m >> 8) & 0xFFu);
    }
}

// AABB transform — 16 AABBs / iter via 512-bit SoA. Tail uses masked
// loads + masked stores; lanes past `count` are unaffected.
void transform_aabb_array(
    f32* out_min_x, f32* out_min_y, f32* out_min_z,
    f32* out_max_x, f32* out_max_y, f32* out_max_z,
    const f32* in_min_x, const f32* in_min_y, const f32* in_min_z,
    const f32* in_max_x, const f32* in_max_y, const f32* in_max_z,
    const Mat4Compact& M, usize count)
{
    const __m512 m00 = _mm512_set1_ps(M.m[ 0]);
    const __m512 m01 = _mm512_set1_ps(M.m[ 1]);
    const __m512 m02 = _mm512_set1_ps(M.m[ 2]);
    const __m512 m10 = _mm512_set1_ps(M.m[ 4]);
    const __m512 m11 = _mm512_set1_ps(M.m[ 5]);
    const __m512 m12 = _mm512_set1_ps(M.m[ 6]);
    const __m512 m20 = _mm512_set1_ps(M.m[ 8]);
    const __m512 m21 = _mm512_set1_ps(M.m[ 9]);
    const __m512 m22 = _mm512_set1_ps(M.m[10]);
    const __m512 t0  = _mm512_set1_ps(M.m[12]);
    const __m512 t1  = _mm512_set1_ps(M.m[13]);
    const __m512 t2  = _mm512_set1_ps(M.m[14]);

    auto compute = [&](__m512 nx, __m512 ny, __m512 nz,
                       __m512 px, __m512 py, __m512 pz,
                       __m512& oxn, __m512& oxp,
                       __m512& oyn, __m512& oyp,
                       __m512& ozn, __m512& ozp)
    {
        auto axis = [&](__m512 mo0, __m512 mo1, __m512 mo2, __m512 to,
                        __m512& on, __m512& op) {
            __m512 a, b, sn = to, sp = to;
            a = _mm512_mul_ps(mo0, nx); b = _mm512_mul_ps(mo0, px);
            sn = _mm512_add_ps(sn, _mm512_min_ps(a, b));
            sp = _mm512_add_ps(sp, _mm512_max_ps(a, b));
            a = _mm512_mul_ps(mo1, ny); b = _mm512_mul_ps(mo1, py);
            sn = _mm512_add_ps(sn, _mm512_min_ps(a, b));
            sp = _mm512_add_ps(sp, _mm512_max_ps(a, b));
            a = _mm512_mul_ps(mo2, nz); b = _mm512_mul_ps(mo2, pz);
            sn = _mm512_add_ps(sn, _mm512_min_ps(a, b));
            sp = _mm512_add_ps(sp, _mm512_max_ps(a, b));
            on = sn; op = sp;
        };
        axis(m00, m10, m20, t0, oxn, oxp);
        axis(m01, m11, m21, t1, oyn, oyp);
        axis(m02, m12, m22, t2, ozn, ozp);
    };

    usize i = 0;
    for (; i + 16 <= count; i += 16) {
        const __m512 nx = _mm512_loadu_ps(in_min_x + i);
        const __m512 ny = _mm512_loadu_ps(in_min_y + i);
        const __m512 nz = _mm512_loadu_ps(in_min_z + i);
        const __m512 px = _mm512_loadu_ps(in_max_x + i);
        const __m512 py = _mm512_loadu_ps(in_max_y + i);
        const __m512 pz = _mm512_loadu_ps(in_max_z + i);
        __m512 oxn, oxp, oyn, oyp, ozn, ozp;
        compute(nx, ny, nz, px, py, pz, oxn, oxp, oyn, oyp, ozn, ozp);
        _mm512_storeu_ps(out_min_x + i, oxn); _mm512_storeu_ps(out_max_x + i, oxp);
        _mm512_storeu_ps(out_min_y + i, oyn); _mm512_storeu_ps(out_max_y + i, oyp);
        _mm512_storeu_ps(out_min_z + i, ozn); _mm512_storeu_ps(out_max_z + i, ozp);
    }
    if (i < count) {
        const __mmask16 m = tail_mask(count - i);
        const __m512 nx = _mm512_maskz_loadu_ps(m, in_min_x + i);
        const __m512 ny = _mm512_maskz_loadu_ps(m, in_min_y + i);
        const __m512 nz = _mm512_maskz_loadu_ps(m, in_min_z + i);
        const __m512 px = _mm512_maskz_loadu_ps(m, in_max_x + i);
        const __m512 py = _mm512_maskz_loadu_ps(m, in_max_y + i);
        const __m512 pz = _mm512_maskz_loadu_ps(m, in_max_z + i);
        __m512 oxn, oxp, oyn, oyp, ozn, ozp;
        compute(nx, ny, nz, px, py, pz, oxn, oxp, oyn, oyp, ozn, ozp);
        _mm512_mask_storeu_ps(out_min_x + i, m, oxn); _mm512_mask_storeu_ps(out_max_x + i, m, oxp);
        _mm512_mask_storeu_ps(out_min_y + i, m, oyn); _mm512_mask_storeu_ps(out_max_y + i, m, oyp);
        _mm512_mask_storeu_ps(out_min_z + i, m, ozn); _mm512_mask_storeu_ps(out_max_z + i, m, ozp);
    }
}

// Frustum-vs-AABB cull — 16 AABBs / iter, mask-register accumulator.
void frustum_cull_aabbs(u8* out_bits, const f32* planes,
                        const f32* min_x, const f32* min_y, const f32* min_z,
                        const f32* max_x, const f32* max_y, const f32* max_z,
                        usize count)
{
    struct PlaneSel { f32 nx, ny, nz, d; const f32 *px, *py, *pz; };
    PlaneSel sels[6];
    for (int p = 0; p < 6; ++p) {
        const f32 nx = planes[p*4 + 0];
        const f32 ny = planes[p*4 + 1];
        const f32 nz = planes[p*4 + 2];
        sels[p] = {
            nx, ny, nz, planes[p*4 + 3],
            (nx >= 0.0f) ? max_x : min_x,
            (ny >= 0.0f) ? max_y : min_y,
            (nz >= 0.0f) ? max_z : min_z,
        };
    }

    auto test16 = [&](usize i, __mmask16 load_mask) -> __mmask16 {
        __mmask16 acc = static_cast<__mmask16>(0xFFFF);
        for (int p = 0; p < 6; ++p) {
            const auto& s = sels[p];
            const __m512 nx = _mm512_set1_ps(s.nx);
            const __m512 ny = _mm512_set1_ps(s.ny);
            const __m512 nz = _mm512_set1_ps(s.nz);
            const __m512 d  = _mm512_set1_ps(s.d);
            const __m512 vpx = _mm512_maskz_loadu_ps(load_mask, s.px + i);
            const __m512 vpy = _mm512_maskz_loadu_ps(load_mask, s.py + i);
            const __m512 vpz = _mm512_maskz_loadu_ps(load_mask, s.pz + i);
            __m512 t = _mm512_fmadd_ps(nz, vpz, d);
            t       = _mm512_fmadd_ps(ny, vpy, t);
            t       = _mm512_fmadd_ps(nx, vpx, t);
            const __mmask16 pass =
                _mm512_cmp_ps_mask(t, _mm512_setzero_ps(), _CMP_GE_OQ);
            acc = _kand_mask16(acc, pass);
        }
        return acc;
    };

    usize i = 0;
    for (; i + 16 <= count; i += 16) {
        const __mmask16 m = test16(i, static_cast<__mmask16>(0xFFFF));
        out_bits[(i >> 3)    ] = static_cast<u8>( m       & 0xFFu);
        out_bits[(i >> 3) + 1] = static_cast<u8>((m >> 8) & 0xFFu);
    }
    if (i < count) {
        const usize n = count - i;
        const __mmask16 lm = tail_mask(n);
        const __mmask16 m  = static_cast<__mmask16>(test16(i, lm) & lm);
        out_bits[(i >> 3)    ] = static_cast<u8>( m       & 0xFFu);
        if (n > 8) out_bits[(i >> 3) + 1] = static_cast<u8>((m >> 8) & 0xFFu);
    }
}

// argmax — AVX-512 path. 16 lanes / iter; mirror of min_index_f32 with
// _CMP_GT_OQ + _mm512_max_ps. Tail uses masked load with -INF-fill so
// masked-off lanes never look larger than valid ones.
void max_index_f32(f32* out_value, u32* out_index,
                   const f32* in, usize count)
{
    if (count == 0) {
        *out_value = -std::numeric_limits<f32>::infinity();
        *out_index = 0;
        return;
    }
    const __m512 vminf = _mm512_set1_ps(-std::numeric_limits<f32>::infinity());

    __m512  cur_max;
    __m512i cur_idx;
    usize i = 0;

    if (count >= 16) {
        cur_max = _mm512_loadu_ps(in);
        cur_idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                    8, 9, 10, 11, 12, 13, 14, 15);
        i = 16;
        for (; i + 16 <= count; i += 16) {
            const __m512 v = _mm512_loadu_ps(in + i);
            const __m512i idx = _mm512_setr_epi32(
                static_cast<int>(i + 0),  static_cast<int>(i + 1),
                static_cast<int>(i + 2),  static_cast<int>(i + 3),
                static_cast<int>(i + 4),  static_cast<int>(i + 5),
                static_cast<int>(i + 6),  static_cast<int>(i + 7),
                static_cast<int>(i + 8),  static_cast<int>(i + 9),
                static_cast<int>(i + 10), static_cast<int>(i + 11),
                static_cast<int>(i + 12), static_cast<int>(i + 13),
                static_cast<int>(i + 14), static_cast<int>(i + 15));
            const __mmask16 greater = _mm512_cmp_ps_mask(v, cur_max, _CMP_GT_OQ);
            cur_max = _mm512_max_ps(cur_max, v);
            cur_idx = _mm512_mask_blend_epi32(greater, cur_idx, idx);
        }
    } else {
        const __mmask16 m = tail_mask(count);
        cur_max = _mm512_mask_loadu_ps(vminf, m, in);
        cur_idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                    8, 9, 10, 11, 12, 13, 14, 15);
        i = count;
    }

    if (i < count) {
        const usize n = count - i;
        const __mmask16 m = tail_mask(n);
        const __m512 v = _mm512_mask_loadu_ps(vminf, m, in + i);
        const __m512i idx = _mm512_setr_epi32(
            static_cast<int>(i + 0),  static_cast<int>(i + 1),
            static_cast<int>(i + 2),  static_cast<int>(i + 3),
            static_cast<int>(i + 4),  static_cast<int>(i + 5),
            static_cast<int>(i + 6),  static_cast<int>(i + 7),
            static_cast<int>(i + 8),  static_cast<int>(i + 9),
            static_cast<int>(i + 10), static_cast<int>(i + 11),
            static_cast<int>(i + 12), static_cast<int>(i + 13),
            static_cast<int>(i + 14), static_cast<int>(i + 15));
        const __mmask16 greater = _mm512_cmp_ps_mask(v, cur_max, _CMP_GT_OQ);
        cur_max = _mm512_max_ps(cur_max, v);
        cur_idx = _mm512_mask_blend_epi32(greater, cur_idx, idx);
    }

    alignas(64) f32 lane_v[16];
    alignas(64) u32 lane_i[16];
    _mm512_store_ps(lane_v, cur_max);
    _mm512_store_si512(reinterpret_cast<__m512i*>(lane_i), cur_idx);
    f32 best   = lane_v[0];
    u32 best_i = lane_i[0];
    for (int k = 1; k < 16; ++k) {
        if (lane_v[k] > best) { best = lane_v[k]; best_i = lane_i[k]; }
    }
    *out_value = best;
    *out_index = best_i;
}

// 3D Morton encoding — 8 u64 lanes / iter via __m512i. Same bit-
// spread + OR shift chain as the lower tiers, just widened to 512-bit.
// _mm512_cvtepu32_epi64 zero-extends 8 u32s into 8 u64 lanes.
namespace {
inline __m512i mt_spread21_8x(__m512i v) noexcept {
    const __m512i m0 = _mm512_set1_epi64(0x1fffffLL);
    const __m512i m1 = _mm512_set1_epi64(0x001f00000000ffffLL);
    const __m512i m2 = _mm512_set1_epi64(0x001f0000ff0000ffLL);
    const __m512i m3 = _mm512_set1_epi64(0x100f00f00f00f00fLL);
    const __m512i m4 = _mm512_set1_epi64(0x10c30c30c30c30c3LL);
    const __m512i m5 = _mm512_set1_epi64(0x1249249249249249LL);
    v = _mm512_and_si512(v, m0);
    v = _mm512_and_si512(_mm512_or_si512(v, _mm512_slli_epi64(v, 32)), m1);
    v = _mm512_and_si512(_mm512_or_si512(v, _mm512_slli_epi64(v, 16)), m2);
    v = _mm512_and_si512(_mm512_or_si512(v, _mm512_slli_epi64(v,  8)), m3);
    v = _mm512_and_si512(_mm512_or_si512(v, _mm512_slli_epi64(v,  4)), m4);
    v = _mm512_and_si512(_mm512_or_si512(v, _mm512_slli_epi64(v,  2)), m5);
    return v;
}
inline u64 mt_spread21_scalar(u32 v) noexcept {
    u64 r = static_cast<u64>(v) & 0x1fffffull;
    r = (r | (r << 32)) & 0x001f00000000ffffULL;
    r = (r | (r << 16)) & 0x001f0000ff0000ffULL;
    r = (r | (r <<  8)) & 0x100f00f00f00f00fULL;
    r = (r | (r <<  4)) & 0x10c30c30c30c30c3ULL;
    r = (r | (r <<  2)) & 0x1249249249249249ULL;
    return r;
}
}  // namespace

void morton_encode_3d_array(u64* out_codes,
                            const u32* x, const u32* y, const u32* z,
                            usize count)
{
    usize i = 0;
    for (; i + 8 <= count; i += 8) {
        const __m512i vx = _mm512_cvtepu32_epi64(_mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(x + i)));
        const __m512i vy = _mm512_cvtepu32_epi64(_mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(y + i)));
        const __m512i vz = _mm512_cvtepu32_epi64(_mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(z + i)));
        const __m512i sx = mt_spread21_8x(vx);
        const __m512i sy = mt_spread21_8x(vy);
        const __m512i sz = mt_spread21_8x(vz);
        const __m512i m  = _mm512_or_si512(sx,
                          _mm512_or_si512(_mm512_slli_epi64(sy, 1),
                                          _mm512_slli_epi64(sz, 2)));
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(out_codes + i), m);
    }
    for (; i < count; ++i) {
        out_codes[i] = mt_spread21_scalar(x[i])
                    | (mt_spread21_scalar(y[i]) << 1)
                    | (mt_spread21_scalar(z[i]) << 2);
    }
}

// argmin reduction — AVX-512 path. 16 lanes / iter; native __mmask16
// + _mm512_mask_blend_epi32 updates indices without round-tripping
// through float. Tail uses masked load with INF-fill so masked-off
// lanes never look smaller than valid ones.
void min_index_f32(f32* out_value, u32* out_index,
                   const f32* in, usize count)
{
    if (count == 0) {
        *out_value = std::numeric_limits<f32>::infinity();
        *out_index = 0;
        return;
    }
    const __m512 vinf = _mm512_set1_ps(std::numeric_limits<f32>::infinity());

    __m512  cur_min;
    __m512i cur_idx;
    usize i = 0;

    if (count >= 16) {
        cur_min = _mm512_loadu_ps(in);
        cur_idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                    8, 9, 10, 11, 12, 13, 14, 15);
        i = 16;
        for (; i + 16 <= count; i += 16) {
            const __m512 v = _mm512_loadu_ps(in + i);
            const __m512i idx = _mm512_setr_epi32(
                static_cast<int>(i + 0),  static_cast<int>(i + 1),
                static_cast<int>(i + 2),  static_cast<int>(i + 3),
                static_cast<int>(i + 4),  static_cast<int>(i + 5),
                static_cast<int>(i + 6),  static_cast<int>(i + 7),
                static_cast<int>(i + 8),  static_cast<int>(i + 9),
                static_cast<int>(i + 10), static_cast<int>(i + 11),
                static_cast<int>(i + 12), static_cast<int>(i + 13),
                static_cast<int>(i + 14), static_cast<int>(i + 15));
            const __mmask16 less = _mm512_cmp_ps_mask(v, cur_min, _CMP_LT_OQ);
            cur_min = _mm512_min_ps(cur_min, v);
            cur_idx = _mm512_mask_blend_epi32(less, cur_idx, idx);
        }
    } else {
        // Initial fill from a masked load; treat absent lanes as INF
        // so the cur_min seed never wins against real data.
        const __mmask16 m = tail_mask(count);
        cur_min = _mm512_mask_loadu_ps(vinf, m, in);
        cur_idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                    8, 9, 10, 11, 12, 13, 14, 15);
        i = count;
    }

    // Bulk-pass tail (count % 16). One masked-load iteration handles it.
    if (i < count) {
        const usize n = count - i;
        const __mmask16 m = tail_mask(n);
        const __m512 v = _mm512_mask_loadu_ps(vinf, m, in + i);
        const __m512i idx = _mm512_setr_epi32(
            static_cast<int>(i + 0),  static_cast<int>(i + 1),
            static_cast<int>(i + 2),  static_cast<int>(i + 3),
            static_cast<int>(i + 4),  static_cast<int>(i + 5),
            static_cast<int>(i + 6),  static_cast<int>(i + 7),
            static_cast<int>(i + 8),  static_cast<int>(i + 9),
            static_cast<int>(i + 10), static_cast<int>(i + 11),
            static_cast<int>(i + 12), static_cast<int>(i + 13),
            static_cast<int>(i + 14), static_cast<int>(i + 15));
        const __mmask16 less = _mm512_cmp_ps_mask(v, cur_min, _CMP_LT_OQ);
        cur_min = _mm512_min_ps(cur_min, v);
        cur_idx = _mm512_mask_blend_epi32(less, cur_idx, idx);
    }

    // Horizontal reduce — spill the 16 lanes and scan scalar.
    alignas(64) f32 lane_v[16];
    alignas(64) u32 lane_i[16];
    _mm512_store_ps(lane_v, cur_min);
    _mm512_store_si512(reinterpret_cast<__m512i*>(lane_i), cur_idx);
    f32 best   = lane_v[0];
    u32 best_i = lane_i[0];
    for (int k = 1; k < 16; ++k) {
        if (lane_v[k] < best) { best = lane_v[k]; best_i = lane_i[k]; }
    }
    *out_value = best;
    *out_index = best_i;
}

// Ray-vs-AABB batch — 16 AABBs / iter via __m512 + FMA. Same algebraic
// rearrangement as the AVX2 path: precompute origin*inv_dir per axis,
// then vfmsub231ps(slab, inv_dir, origin*inv_dir) gives the per-slab
// t directly. Tail uses masked load + mask_storeu — lanes past `count`
// are unaffected so the caller can scan the whole output buffer.
void ray_aabb_intersect_array(
    f32* out_t,
    f32 ox, f32 oy, f32 oz,
    f32 inv_dx, f32 inv_dy, f32 inv_dz,
    const f32* min_x, const f32* min_y, const f32* min_z,
    const f32* max_x, const f32* max_y, const f32* max_z,
    usize count)
{
    const __m512 vix = _mm512_set1_ps(inv_dx);
    const __m512 viy = _mm512_set1_ps(inv_dy);
    const __m512 viz = _mm512_set1_ps(inv_dz);
    const __m512 oix = _mm512_set1_ps(ox * inv_dx);
    const __m512 oiy = _mm512_set1_ps(oy * inv_dy);
    const __m512 oiz = _mm512_set1_ps(oz * inv_dz);
    const __m512 vinf = _mm512_set1_ps(std::numeric_limits<f32>::infinity());
    const __m512 vzero = _mm512_setzero_ps();

    auto compute = [&](__m512 mnx, __m512 mxx,
                       __m512 mny, __m512 mxy,
                       __m512 mnz, __m512 mxz) -> __m512
    {
        const __m512 t0x = _mm512_fmsub_ps(mnx, vix, oix);
        const __m512 t1x = _mm512_fmsub_ps(mxx, vix, oix);
        const __m512 t0y = _mm512_fmsub_ps(mny, viy, oiy);
        const __m512 t1y = _mm512_fmsub_ps(mxy, viy, oiy);
        const __m512 t0z = _mm512_fmsub_ps(mnz, viz, oiz);
        const __m512 t1z = _mm512_fmsub_ps(mxz, viz, oiz);

        const __m512 nx = _mm512_min_ps(t0x, t1x);
        const __m512 fx = _mm512_max_ps(t0x, t1x);
        const __m512 ny = _mm512_min_ps(t0y, t1y);
        const __m512 fy = _mm512_max_ps(t0y, t1y);
        const __m512 nz = _mm512_min_ps(t0z, t1z);
        const __m512 fz = _mm512_max_ps(t0z, t1z);

        const __m512 t_near = _mm512_max_ps(_mm512_max_ps(nx, ny), nz);
        const __m512 t_far  = _mm512_min_ps(_mm512_min_ps(fx, fy), fz);
        const __m512 t_cl   = _mm512_max_ps(t_near, vzero);
        // Native 16-bit mask: 1 where hit. mask_blend selects t_cl,
        // INF for misses — single-instruction final blend.
        const __mmask16 hit = _mm512_cmp_ps_mask(t_far, t_cl, _CMP_GE_OQ);
        return _mm512_mask_blend_ps(hit, vinf, t_cl);
    };

    usize i = 0;
    for (; i + 16 <= count; i += 16) {
        _mm512_storeu_ps(out_t + i, compute(
            _mm512_loadu_ps(min_x + i), _mm512_loadu_ps(max_x + i),
            _mm512_loadu_ps(min_y + i), _mm512_loadu_ps(max_y + i),
            _mm512_loadu_ps(min_z + i), _mm512_loadu_ps(max_z + i)));
    }
    if (i < count) {
        const __mmask16 m = tail_mask(count - i);
        // Use INF for masked-off lanes so the slab math doesn't go
        // wild on garbage; the masked store below ignores them anyway.
        const __m512 mnx = _mm512_mask_loadu_ps(vinf, m, min_x + i);
        const __m512 mxx = _mm512_mask_loadu_ps(vinf, m, max_x + i);
        const __m512 mny = _mm512_mask_loadu_ps(vinf, m, min_y + i);
        const __m512 mxy = _mm512_mask_loadu_ps(vinf, m, max_y + i);
        const __m512 mnz = _mm512_mask_loadu_ps(vinf, m, min_z + i);
        const __m512 mxz = _mm512_mask_loadu_ps(vinf, m, max_z + i);
        _mm512_mask_storeu_ps(out_t + i, m,
            compute(mnx, mxx, mny, mxy, mnz, mxz));
    }
}

}  // namespace cardinal::core::simd::avx512
