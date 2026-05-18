// =============================================================================
// Cardinal — SIMD math: AVX2 + FMA3.
//
// Same kernels as AVX but using FMA3 (_mm256_fmadd_ps) — one rounding
// step per madd instead of mul-then-add, faster on every CPU since
// Haswell. AVX2 also enables 256-bit integer ops; the kernels here are
// float-only so the win comes purely from FMA.
//
// Compiled with /arch:AVX2 (MSVC) — see CMakeLists.
// =============================================================================
#define CARDINAL_SIMD_TIER avx2
#include "simd_math_kernels.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cardinal::core::simd::avx2 {

#define CARDINAL_AVX2_TAIL(stmts) \
    for (; i < n; ++i) { stmts; }

void vec_add_f32(f32* out, const f32* a, const f32* b, usize n) {
    usize i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i,
            _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    CARDINAL_AVX2_TAIL(out[i] = a[i] + b[i]);
}
void vec_sub_f32(f32* out, const f32* a, const f32* b, usize n) {
    usize i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i,
            _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    CARDINAL_AVX2_TAIL(out[i] = a[i] - b[i]);
}
void vec_mul_f32(f32* out, const f32* a, const f32* b, usize n) {
    usize i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i,
            _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    CARDINAL_AVX2_TAIL(out[i] = a[i] * b[i]);
}
void vec_scale_f32(f32* out, const f32* a, f32 s, usize n) {
    const __m256 vs = _mm256_set1_ps(s);
    usize i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), vs));
    }
    CARDINAL_AVX2_TAIL(out[i] = a[i] * s);
}

// FMA3: y = a * x + y in a single rounded op. The point of having a
// separate AVX2 tier from AVX is exactly this — about 2× the
// throughput of split-mul-add on every uarch since Haswell.
void vec_axpy_f32(f32* y, f32 a, const f32* x, usize n) {
    const __m256 va = _mm256_set1_ps(a);
    usize i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, vx, vy));
    }
    CARDINAL_AVX2_TAIL(y[i] += a * x[i]);
}

static inline f32 hsum256_ps(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehl_ps(s, s);
    s = _mm_add_ps(s, sh);
    sh = _mm_shuffle_ps(s, s, _MM_SHUFFLE(0, 0, 0, 1));
    s = _mm_add_ss(s, sh);
    return _mm_cvtss_f32(s);
}
static inline f32 hmin256_ps(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_min_ps(lo, hi);
    __m128 sh = _mm_movehl_ps(s, s);
    s = _mm_min_ps(s, sh);
    sh = _mm_shuffle_ps(s, s, _MM_SHUFFLE(0, 0, 0, 1));
    s = _mm_min_ss(s, sh);
    return _mm_cvtss_f32(s);
}
static inline f32 hmax256_ps(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_max_ps(lo, hi);
    __m128 sh = _mm_movehl_ps(s, s);
    s = _mm_max_ps(s, sh);
    sh = _mm_shuffle_ps(s, s, _MM_SHUFFLE(0, 0, 0, 1));
    s = _mm_max_ss(s, sh);
    return _mm_cvtss_f32(s);
}

f32 dot_f32(const f32* a, const f32* b, usize n) {
    __m256 acc = _mm256_setzero_ps();
    usize i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);   // FMA — one rounded op.
    }
    f32 tail = 0.0f;
    for (; i < n; ++i) tail += a[i] * b[i];
    return hsum256_ps(acc) + tail;
}
f32 sum_f32(const f32* a, usize n) {
    __m256 acc = _mm256_setzero_ps();
    usize i = 0;
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(a + i));
    }
    f32 tail = 0.0f;
    for (; i < n; ++i) tail += a[i];
    return hsum256_ps(acc) + tail;
}
f32 min_f32(const f32* a, usize n) {
    if (n == 0) return std::numeric_limits<f32>::infinity();
    __m256 acc = _mm256_set1_ps(a[0]);
    usize i = 0;
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_min_ps(acc, _mm256_loadu_ps(a + i));
    }
    f32 m = hmin256_ps(acc);
    for (; i < n; ++i) m = std::min(m, a[i]);
    return m;
}
f32 max_f32(const f32* a, usize n) {
    if (n == 0) return -std::numeric_limits<f32>::infinity();
    __m256 acc = _mm256_set1_ps(a[0]);
    usize i = 0;
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_max_ps(acc, _mm256_loadu_ps(a + i));
    }
    f32 m = hmax256_ps(acc);
    for (; i < n; ++i) m = std::max(m, a[i]);
    return m;
}

void transform_points_mat4(f32* out_xyz, const f32* in_xyz,
                           const Mat4Compact& M, usize count)
{
    const f32* m = M.m;
    const __m256 m00 = _mm256_set1_ps(m[0]);  const __m256 m04 = _mm256_set1_ps(m[4]);
    const __m256 m08 = _mm256_set1_ps(m[8]);  const __m256 m12 = _mm256_set1_ps(m[12]);
    const __m256 m01 = _mm256_set1_ps(m[1]);  const __m256 m05 = _mm256_set1_ps(m[5]);
    const __m256 m09 = _mm256_set1_ps(m[9]);  const __m256 m13 = _mm256_set1_ps(m[13]);
    const __m256 m02 = _mm256_set1_ps(m[2]);  const __m256 m06 = _mm256_set1_ps(m[6]);
    const __m256 m10 = _mm256_set1_ps(m[10]); const __m256 m14 = _mm256_set1_ps(m[14]);

    usize i = 0;
    for (; i + 8 <= count; i += 8) {
        const f32* p = in_xyz + i*3;
        const __m256 xs = _mm256_set_ps(p[21], p[18], p[15], p[12], p[ 9], p[6], p[3], p[0]);
        const __m256 ys = _mm256_set_ps(p[22], p[19], p[16], p[13], p[10], p[7], p[4], p[1]);
        const __m256 zs = _mm256_set_ps(p[23], p[20], p[17], p[14], p[11], p[8], p[5], p[2]);

        // FMA chain — 3 ops/channel instead of 5 split-mul-add.
        __m256 ox = _mm256_fmadd_ps(m00, xs, m12);
        ox = _mm256_fmadd_ps(m04, ys, ox);
        ox = _mm256_fmadd_ps(m08, zs, ox);

        __m256 oy = _mm256_fmadd_ps(m01, xs, m13);
        oy = _mm256_fmadd_ps(m05, ys, oy);
        oy = _mm256_fmadd_ps(m09, zs, oy);

        __m256 oz = _mm256_fmadd_ps(m02, xs, m14);
        oz = _mm256_fmadd_ps(m06, ys, oz);
        oz = _mm256_fmadd_ps(m10, zs, oz);

        alignas(32) f32 ox_arr[8], oy_arr[8], oz_arr[8];
        _mm256_store_ps(ox_arr, ox);
        _mm256_store_ps(oy_arr, oy);
        _mm256_store_ps(oz_arr, oz);
        f32* o = out_xyz + i*3;
        for (int k = 0; k < 8; ++k) {
            o[k*3 + 0] = ox_arr[k];
            o[k*3 + 1] = oy_arr[k];
            o[k*3 + 2] = oz_arr[k];
        }
    }
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
    for (; i + 8 <= count; i += 8) {
        const f32* a = a_xyz + i*3;
        const f32* b = b_xyz + i*3;
        const __m256 ax = _mm256_set_ps(a[21], a[18], a[15], a[12], a[ 9], a[6], a[3], a[0]);
        const __m256 ay = _mm256_set_ps(a[22], a[19], a[16], a[13], a[10], a[7], a[4], a[1]);
        const __m256 az = _mm256_set_ps(a[23], a[20], a[17], a[14], a[11], a[8], a[5], a[2]);
        const __m256 bx = _mm256_set_ps(b[21], b[18], b[15], b[12], b[ 9], b[6], b[3], b[0]);
        const __m256 by = _mm256_set_ps(b[22], b[19], b[16], b[13], b[10], b[7], b[4], b[1]);
        const __m256 bz = _mm256_set_ps(b[23], b[20], b[17], b[14], b[11], b[8], b[5], b[2]);
        // FMA: ay*bz - az*by  ==  fmsub(ay, bz, az*by)
        const __m256 ox = _mm256_fmsub_ps(ay, bz, _mm256_mul_ps(az, by));
        const __m256 oy = _mm256_fmsub_ps(az, bx, _mm256_mul_ps(ax, bz));
        const __m256 oz = _mm256_fmsub_ps(ax, by, _mm256_mul_ps(ay, bx));
        alignas(32) f32 xa[8], ya[8], za[8];
        _mm256_store_ps(xa, ox); _mm256_store_ps(ya, oy); _mm256_store_ps(za, oz);
        f32* o = out_xyz + i*3;
        for (int k = 0; k < 8; ++k) {
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
    for (; i + 8 <= count; i += 8) {
        const f32* p = in_xyz + i*3;
        const __m256 xs = _mm256_set_ps(p[21], p[18], p[15], p[12], p[ 9], p[6], p[3], p[0]);
        const __m256 ys = _mm256_set_ps(p[22], p[19], p[16], p[13], p[10], p[7], p[4], p[1]);
        const __m256 zs = _mm256_set_ps(p[23], p[20], p[17], p[14], p[11], p[8], p[5], p[2]);
        // FMA-chained sum-of-squares.
        __m256 sq = _mm256_mul_ps(xs, xs);
        sq = _mm256_fmadd_ps(ys, ys, sq);
        sq = _mm256_fmadd_ps(zs, zs, sq);
        _mm256_storeu_ps(out_lens + i, _mm256_sqrt_ps(sq));
    }
    for (; i < count; ++i) {
        const f32 x = in_xyz[i*3+0], y = in_xyz[i*3+1], z = in_xyz[i*3+2];
        out_lens[i] = std::sqrt(x*x + y*y + z*z);
    }
}

void vec3_normalize_array_inplace(f32* io_xyz, usize count) {
    const __m256 eps_v = _mm256_set1_ps(1e-8f);
    usize i = 0;
    for (; i + 8 <= count; i += 8) {
        f32* p = io_xyz + i*3;
        const __m256 xs = _mm256_set_ps(p[21], p[18], p[15], p[12], p[ 9], p[6], p[3], p[0]);
        const __m256 ys = _mm256_set_ps(p[22], p[19], p[16], p[13], p[10], p[7], p[4], p[1]);
        const __m256 zs = _mm256_set_ps(p[23], p[20], p[17], p[14], p[11], p[8], p[5], p[2]);
        __m256 sq = _mm256_mul_ps(xs, xs);
        sq = _mm256_fmadd_ps(ys, ys, sq);
        sq = _mm256_fmadd_ps(zs, zs, sq);
        const __m256 mask = _mm256_cmp_ps(sq, eps_v, _CMP_GE_OQ);
        const __m256 inv  = _mm256_and_ps(mask,
                              _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_sqrt_ps(sq)));
        const __m256 nxs = _mm256_blendv_ps(xs, _mm256_mul_ps(xs, inv), mask);
        const __m256 nys = _mm256_blendv_ps(ys, _mm256_mul_ps(ys, inv), mask);
        const __m256 nzs = _mm256_blendv_ps(zs, _mm256_mul_ps(zs, inv), mask);
        alignas(32) f32 xa[8], ya[8], za[8];
        _mm256_store_ps(xa, nxs); _mm256_store_ps(ya, nys); _mm256_store_ps(za, nzs);
        for (int k = 0; k < 8; ++k) {
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

// AVX2 + FMA3 — same 4-wide column structure as SSE4.2, but the four
// mul+add accumulator pairs collapse into FMA3 chains: 3 fmadds per
// column instead of 4 muls + 3 adds. Cuts the per-matrix instruction
// count meaningfully on Haswell+ / Zen 2+.
void mat4_mul_array(f32* out, const f32* A, const f32* B, usize count) {
    for (usize i = 0; i < count; ++i) {
        const f32* a = A + i * 16;
        const f32* b = B + i * 16;
        f32*       o = out + i * 16;
        const __m128 ac0 = _mm_loadu_ps(a + 0*4);
        const __m128 ac1 = _mm_loadu_ps(a + 1*4);
        const __m128 ac2 = _mm_loadu_ps(a + 2*4);
        const __m128 ac3 = _mm_loadu_ps(a + 3*4);
        for (int c = 0; c < 4; ++c) {
            const __m128 b0 = _mm_set1_ps(b[c*4 + 0]);
            const __m128 b1 = _mm_set1_ps(b[c*4 + 1]);
            const __m128 b2 = _mm_set1_ps(b[c*4 + 2]);
            const __m128 b3 = _mm_set1_ps(b[c*4 + 3]);
            // Start with ac0*b0, FMA the rest.
            __m128 oc = _mm_mul_ps(ac0, b0);
            oc = _mm_fmadd_ps(ac1, b1, oc);
            oc = _mm_fmadd_ps(ac2, b2, oc);
            oc = _mm_fmadd_ps(ac3, b3, oc);
            _mm_storeu_ps(o + c*4, oc);
        }
    }
}

// Frustum cull — 8 spheres / iter (one byte / iter), with FMA3.
// Per plane: dist + r = (nx*cx) + (ny*cy) + (nz*cz) + (d + r), folded
// into 1 add (d+r) + 3 FMAs + 1 cmp. AND across the 6 planes, then
// movemask the result to 8 bits.
void frustum_cull_spheres(u8* out_bits, const f32* planes,
                          const f32* cx, const f32* cy,
                          const f32* cz, const f32* r,
                          usize count)
{
    auto test8 = [&](usize i) -> int {
        const __m256 vcx = _mm256_loadu_ps(cx + i);
        const __m256 vcy = _mm256_loadu_ps(cy + i);
        const __m256 vcz = _mm256_loadu_ps(cz + i);
        const __m256 vr  = _mm256_loadu_ps(r  + i);
        __m256 acc = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
        for (int p = 0; p < 6; ++p) {
            const __m256 nx = _mm256_set1_ps(planes[p*4 + 0]);
            const __m256 ny = _mm256_set1_ps(planes[p*4 + 1]);
            const __m256 nz = _mm256_set1_ps(planes[p*4 + 2]);
            const __m256 d  = _mm256_set1_ps(planes[p*4 + 3]);
            // (d + r) absorbs the per-sphere radius into the plane
            // offset, so the final compare is against zero.
            __m256 d_plus_r = _mm256_add_ps(d, vr);
            __m256 t = _mm256_fmadd_ps(nz, vcz, d_plus_r);
            t       = _mm256_fmadd_ps(ny, vcy, t);
            t       = _mm256_fmadd_ps(nx, vcx, t);
            __m256 pass = _mm256_cmp_ps(t, _mm256_setzero_ps(), _CMP_GE_OQ);
            acc = _mm256_and_ps(acc, pass);
        }
        return _mm256_movemask_ps(acc);
    };
    auto test_one = [&](usize i) -> bool {
        const f32 x = cx[i], y = cy[i], z = cz[i], rr = r[i];
        for (int p = 0; p < 6; ++p) {
            const f32 dist = planes[p*4+0]*x + planes[p*4+1]*y +
                             planes[p*4+2]*z + planes[p*4+3];
            if (dist + rr < 0.0f) return false;
        }
        return true;
    };

    usize i = 0;
    for (; i + 8 <= count; i += 8) {
        out_bits[i >> 3] = static_cast<u8>(test8(i));
    }
    if (i < count) {
        u8 byte = 0;
        for (usize b = 0; i + b < count; ++b) {
            if (test_one(i + b)) byte |= static_cast<u8>(1u << b);
        }
        out_bits[i >> 3] = byte;
    }
}

// AABB transform — 8 AABBs / iter with FMA3. The min/max merge stays
// as add → can't fold into FMA without losing the per-axis min/max
// semantics; but the per-axis _add_ chain folds into _fmadd_ on the
// dest accumulators when min(a,b) is computed first and then added.
void transform_aabb_array(
    f32* out_min_x, f32* out_min_y, f32* out_min_z,
    f32* out_max_x, f32* out_max_y, f32* out_max_z,
    const f32* in_min_x, const f32* in_min_y, const f32* in_min_z,
    const f32* in_max_x, const f32* in_max_y, const f32* in_max_z,
    const Mat4Compact& M, usize count)
{
    const __m256 m00 = _mm256_set1_ps(M.m[ 0]);
    const __m256 m01 = _mm256_set1_ps(M.m[ 1]);
    const __m256 m02 = _mm256_set1_ps(M.m[ 2]);
    const __m256 m10 = _mm256_set1_ps(M.m[ 4]);
    const __m256 m11 = _mm256_set1_ps(M.m[ 5]);
    const __m256 m12 = _mm256_set1_ps(M.m[ 6]);
    const __m256 m20 = _mm256_set1_ps(M.m[ 8]);
    const __m256 m21 = _mm256_set1_ps(M.m[ 9]);
    const __m256 m22 = _mm256_set1_ps(M.m[10]);
    const __m256 t0  = _mm256_set1_ps(M.m[12]);
    const __m256 t1  = _mm256_set1_ps(M.m[13]);
    const __m256 t2  = _mm256_set1_ps(M.m[14]);

    usize i = 0;
    for (; i + 8 <= count; i += 8) {
        const __m256 nx = _mm256_loadu_ps(in_min_x + i);
        const __m256 ny = _mm256_loadu_ps(in_min_y + i);
        const __m256 nz = _mm256_loadu_ps(in_min_z + i);
        const __m256 px = _mm256_loadu_ps(in_max_x + i);
        const __m256 py = _mm256_loadu_ps(in_max_y + i);
        const __m256 pz = _mm256_loadu_ps(in_max_z + i);

        // Per output axis: 3 inputs × {min, max}. We use _mm256_min/max_ps
        // as the per-axis selector then _add — the inner mul+add COULD
        // fuse if M_ij and the corresponding (n,p) only contributed to
        // one side, but here BOTH sides need both products, so we keep
        // mul + min/max + add explicit.
        auto axis = [&](__m256 mo0, __m256 mo1, __m256 mo2, __m256 to,
                        __m256& on, __m256& op) {
            __m256 a, b, sn = to, sp = to;
            a = _mm256_mul_ps(mo0, nx); b = _mm256_mul_ps(mo0, px);
            sn = _mm256_add_ps(sn, _mm256_min_ps(a, b));
            sp = _mm256_add_ps(sp, _mm256_max_ps(a, b));
            a = _mm256_mul_ps(mo1, ny); b = _mm256_mul_ps(mo1, py);
            sn = _mm256_add_ps(sn, _mm256_min_ps(a, b));
            sp = _mm256_add_ps(sp, _mm256_max_ps(a, b));
            a = _mm256_mul_ps(mo2, nz); b = _mm256_mul_ps(mo2, pz);
            sn = _mm256_add_ps(sn, _mm256_min_ps(a, b));
            sp = _mm256_add_ps(sp, _mm256_max_ps(a, b));
            on = sn; op = sp;
        };

        __m256 oxn, oxp, oyn, oyp, ozn, ozp;
        axis(m00, m10, m20, t0, oxn, oxp);
        axis(m01, m11, m21, t1, oyn, oyp);
        axis(m02, m12, m22, t2, ozn, ozp);

        _mm256_storeu_ps(out_min_x + i, oxn); _mm256_storeu_ps(out_max_x + i, oxp);
        _mm256_storeu_ps(out_min_y + i, oyn); _mm256_storeu_ps(out_max_y + i, oyp);
        _mm256_storeu_ps(out_min_z + i, ozn); _mm256_storeu_ps(out_max_z + i, ozp);
    }
    for (; i < count; ++i) {
        const f32 mn[3] = { in_min_x[i], in_min_y[i], in_min_z[i] };
        const f32 mx[3] = { in_max_x[i], in_max_y[i], in_max_z[i] };
        const f32 ms[3] = { M.m[12], M.m[13], M.m[14] };
        const f32 cols[3][3] = {
            { M.m[ 0], M.m[ 1], M.m[ 2] },
            { M.m[ 4], M.m[ 5], M.m[ 6] },
            { M.m[ 8], M.m[ 9], M.m[10] },
        };
        f32 on[3] = { ms[0], ms[1], ms[2] };
        f32 op[3] = { ms[0], ms[1], ms[2] };
        for (int ai = 0; ai < 3; ++ai) {
            for (int ao = 0; ao < 3; ++ao) {
                const f32 a = cols[ai][ao] * mn[ai];
                const f32 b = cols[ai][ao] * mx[ai];
                on[ao] += std::min(a, b);
                op[ao] += std::max(a, b);
            }
        }
        out_min_x[i] = on[0]; out_max_x[i] = op[0];
        out_min_y[i] = on[1]; out_max_y[i] = op[1];
        out_min_z[i] = on[2]; out_max_z[i] = op[2];
    }
}

// Frustum-vs-AABB cull — 8 AABBs / iter with FMA3 chain.
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

    auto test8 = [&](usize i) -> int {
        __m256 acc = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
        for (int p = 0; p < 6; ++p) {
            const auto& s = sels[p];
            const __m256 nx = _mm256_set1_ps(s.nx);
            const __m256 ny = _mm256_set1_ps(s.ny);
            const __m256 nz = _mm256_set1_ps(s.nz);
            const __m256 d  = _mm256_set1_ps(s.d);
            const __m256 vpx = _mm256_loadu_ps(s.px + i);
            const __m256 vpy = _mm256_loadu_ps(s.py + i);
            const __m256 vpz = _mm256_loadu_ps(s.pz + i);
            __m256 t = _mm256_fmadd_ps(nz, vpz, d);
            t       = _mm256_fmadd_ps(ny, vpy, t);
            t       = _mm256_fmadd_ps(nx, vpx, t);
            __m256 pass = _mm256_cmp_ps(t, _mm256_setzero_ps(), _CMP_GE_OQ);
            acc = _mm256_and_ps(acc, pass);
        }
        return _mm256_movemask_ps(acc);
    };
    auto test_one = [&](usize i) -> bool {
        for (int p = 0; p < 6; ++p) {
            const auto& s = sels[p];
            const f32 dist = s.nx*s.px[i] + s.ny*s.py[i] + s.nz*s.pz[i] + s.d;
            if (dist < 0.0f) return false;
        }
        return true;
    };

    usize i = 0;
    for (; i + 8 <= count; i += 8) out_bits[i >> 3] = static_cast<u8>(test8(i));
    if (i < count) {
        u8 byte = 0;
        for (usize b = 0; i + b < count; ++b) {
            if (test_one(i + b)) byte |= static_cast<u8>(1u << b);
        }
        out_bits[i >> 3] = byte;
    }
}

// argmax — AVX2 path. Same algorithm as the AVX tier (argmax doesn't
// have an FMA-able shape). Lives in this tier so the dispatcher lands
// explicitly here on AVX2 CPUs.
void max_index_f32(f32* out_value, u32* out_index,
                   const f32* in, usize count)
{
    if (count == 0) {
        *out_value = -std::numeric_limits<f32>::infinity();
        *out_index = 0;
        return;
    }
    if (count < 8) {
        f32 best = in[0]; u32 best_i = 0;
        for (usize i = 1; i < count; ++i) {
            if (in[i] > best) { best = in[i]; best_i = static_cast<u32>(i); }
        }
        *out_value = best; *out_index = best_i;
        return;
    }

    __m256  cur_max = _mm256_loadu_ps(in);
    __m256i cur_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    usize i = 8;
    for (; i + 8 <= count; i += 8) {
        const __m256  v   = _mm256_loadu_ps(in + i);
        const __m256i idx = _mm256_setr_epi32(
            static_cast<int>(i),     static_cast<int>(i + 1),
            static_cast<int>(i + 2), static_cast<int>(i + 3),
            static_cast<int>(i + 4), static_cast<int>(i + 5),
            static_cast<int>(i + 6), static_cast<int>(i + 7));
        const __m256 greater_mask = _mm256_cmp_ps(v, cur_max, _CMP_GT_OQ);
        cur_max = _mm256_max_ps(cur_max, v);
        cur_idx = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(cur_idx),
            _mm256_castsi256_ps(idx),
            greater_mask));
    }

    alignas(32) f32 lane_v[8];
    alignas(32) u32 lane_i[8];
    _mm256_store_ps(lane_v, cur_max);
    _mm256_store_si256(reinterpret_cast<__m256i*>(lane_i), cur_idx);
    f32 best   = lane_v[0];
    u32 best_i = lane_i[0];
    for (int k = 1; k < 8; ++k) {
        if (lane_v[k] > best) { best = lane_v[k]; best_i = lane_i[k]; }
    }
    for (; i < count; ++i) {
        if (in[i] > best) { best = in[i]; best_i = static_cast<u32>(i); }
    }
    *out_value = best;
    *out_index = best_i;
}

// 3D Morton encoding — AVX2 brings 256-bit integer shifts (vpsllq on
// __m256i), so we widen to 4 u64 lanes per iteration. _mm256_cvtepu32_
// epi64 zero-extends 4 u32s into 4 u64 lanes for the spread chain.
namespace {
inline __m256i mt_spread21_4x(__m256i v) noexcept {
    const __m256i m0 = _mm256_set1_epi64x(0x1fffffLL);
    const __m256i m1 = _mm256_set1_epi64x(0x001f00000000ffffLL);
    const __m256i m2 = _mm256_set1_epi64x(0x001f0000ff0000ffLL);
    const __m256i m3 = _mm256_set1_epi64x(0x100f00f00f00f00fLL);
    const __m256i m4 = _mm256_set1_epi64x(0x10c30c30c30c30c3LL);
    const __m256i m5 = _mm256_set1_epi64x(0x1249249249249249LL);
    v = _mm256_and_si256(v, m0);
    v = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi64(v, 32)), m1);
    v = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi64(v, 16)), m2);
    v = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi64(v,  8)), m3);
    v = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi64(v,  4)), m4);
    v = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi64(v,  2)), m5);
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
    for (; i + 4 <= count; i += 4) {
        const __m256i vx = _mm256_cvtepu32_epi64(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(x + i)));
        const __m256i vy = _mm256_cvtepu32_epi64(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(y + i)));
        const __m256i vz = _mm256_cvtepu32_epi64(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(z + i)));
        const __m256i sx = mt_spread21_4x(vx);
        const __m256i sy = mt_spread21_4x(vy);
        const __m256i sz = mt_spread21_4x(vz);
        const __m256i m  = _mm256_or_si256(sx,
                          _mm256_or_si256(_mm256_slli_epi64(sy, 1),
                                          _mm256_slli_epi64(sz, 2)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_codes + i), m);
    }
    for (; i < count; ++i) {
        out_codes[i] = mt_spread21_scalar(x[i])
                    | (mt_spread21_scalar(y[i]) << 1)
                    | (mt_spread21_scalar(z[i]) << 2);
    }
}

// argmin reduction — AVX2 path. Same algorithm as the AVX tier
// (argmin doesn't have an FMA-able shape). Lives in this tier so the
// dispatcher can land on AVX2 when an AVX2 CPU is detected without
// silently falling through to AVX (which is fine, just clearer).
void min_index_f32(f32* out_value, u32* out_index,
                   const f32* in, usize count)
{
    if (count == 0) {
        *out_value = std::numeric_limits<f32>::infinity();
        *out_index = 0;
        return;
    }
    if (count < 8) {
        f32 best = in[0]; u32 best_i = 0;
        for (usize i = 1; i < count; ++i) {
            if (in[i] < best) { best = in[i]; best_i = static_cast<u32>(i); }
        }
        *out_value = best; *out_index = best_i;
        return;
    }

    __m256  cur_min = _mm256_loadu_ps(in);
    __m256i cur_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    usize i = 8;
    for (; i + 8 <= count; i += 8) {
        const __m256  v   = _mm256_loadu_ps(in + i);
        const __m256i idx = _mm256_setr_epi32(
            static_cast<int>(i),     static_cast<int>(i + 1),
            static_cast<int>(i + 2), static_cast<int>(i + 3),
            static_cast<int>(i + 4), static_cast<int>(i + 5),
            static_cast<int>(i + 6), static_cast<int>(i + 7));
        const __m256 less_mask = _mm256_cmp_ps(v, cur_min, _CMP_LT_OQ);
        cur_min = _mm256_min_ps(cur_min, v);
        cur_idx = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(cur_idx),
            _mm256_castsi256_ps(idx),
            less_mask));
    }

    alignas(32) f32 lane_v[8];
    alignas(32) u32 lane_i[8];
    _mm256_store_ps(lane_v, cur_min);
    _mm256_store_si256(reinterpret_cast<__m256i*>(lane_i), cur_idx);
    f32 best   = lane_v[0];
    u32 best_i = lane_i[0];
    for (int k = 1; k < 8; ++k) {
        if (lane_v[k] < best) { best = lane_v[k]; best_i = lane_i[k]; }
    }
    for (; i < count; ++i) {
        if (in[i] < best) { best = in[i]; best_i = static_cast<u32>(i); }
    }
    *out_value = best;
    *out_index = best_i;
}

// Ray-vs-AABB batch — 8 AABBs / iter via __m256 + FMA3.
//
// Algebraic re-arrangement: (slab - origin) * inv_dir
//                         = slab * inv_dir - origin * inv_dir
// Precompute origin * inv_dir as a per-axis constant ONCE; the inner
// loop becomes vfmsub231ps(slab, inv_dir, origin_inv_dir) — half the
// op count of mul + sub.
void ray_aabb_intersect_array(
    f32* out_t,
    f32 ox, f32 oy, f32 oz,
    f32 inv_dx, f32 inv_dy, f32 inv_dz,
    const f32* min_x, const f32* min_y, const f32* min_z,
    const f32* max_x, const f32* max_y, const f32* max_z,
    usize count)
{
    const __m256 vix = _mm256_set1_ps(inv_dx);
    const __m256 viy = _mm256_set1_ps(inv_dy);
    const __m256 viz = _mm256_set1_ps(inv_dz);
    // o*i is per-axis constant; FMA folds slab*inv_dir - o*i into one op.
    const __m256 oix = _mm256_set1_ps(ox * inv_dx);
    const __m256 oiy = _mm256_set1_ps(oy * inv_dy);
    const __m256 oiz = _mm256_set1_ps(oz * inv_dz);
    const __m256 vinf = _mm256_set1_ps(std::numeric_limits<f32>::infinity());
    const __m256 vzero = _mm256_setzero_ps();

    usize i = 0;
    for (; i + 8 <= count; i += 8) {
        const __m256 t0x = _mm256_fmsub_ps(_mm256_loadu_ps(min_x + i), vix, oix);
        const __m256 t1x = _mm256_fmsub_ps(_mm256_loadu_ps(max_x + i), vix, oix);
        const __m256 t0y = _mm256_fmsub_ps(_mm256_loadu_ps(min_y + i), viy, oiy);
        const __m256 t1y = _mm256_fmsub_ps(_mm256_loadu_ps(max_y + i), viy, oiy);
        const __m256 t0z = _mm256_fmsub_ps(_mm256_loadu_ps(min_z + i), viz, oiz);
        const __m256 t1z = _mm256_fmsub_ps(_mm256_loadu_ps(max_z + i), viz, oiz);

        const __m256 nx = _mm256_min_ps(t0x, t1x);
        const __m256 fx = _mm256_max_ps(t0x, t1x);
        const __m256 ny = _mm256_min_ps(t0y, t1y);
        const __m256 fy = _mm256_max_ps(t0y, t1y);
        const __m256 nz = _mm256_min_ps(t0z, t1z);
        const __m256 fz = _mm256_max_ps(t0z, t1z);

        const __m256 t_near = _mm256_max_ps(_mm256_max_ps(nx, ny), nz);
        const __m256 t_far  = _mm256_min_ps(_mm256_min_ps(fx, fy), fz);
        const __m256 t_cl   = _mm256_max_ps(t_near, vzero);
        const __m256 hit    = _mm256_cmp_ps(t_far, t_cl, _CMP_GE_OQ);
        const __m256 result = _mm256_blendv_ps(vinf, t_cl, hit);
        _mm256_storeu_ps(out_t + i, result);
    }
    const f32 INF = std::numeric_limits<f32>::infinity();
    for (; i < count; ++i) {
        const f32 tx0 = (min_x[i] - ox) * inv_dx;
        const f32 tx1 = (max_x[i] - ox) * inv_dx;
        const f32 ty0 = (min_y[i] - oy) * inv_dy;
        const f32 ty1 = (max_y[i] - oy) * inv_dy;
        const f32 tz0 = (min_z[i] - oz) * inv_dz;
        const f32 tz1 = (max_z[i] - oz) * inv_dz;
        const f32 t_near = std::max(std::max(std::min(tx0, tx1), std::min(ty0, ty1)),
                                                std::min(tz0, tz1));
        const f32 t_far  = std::min(std::min(std::max(tx0, tx1), std::max(ty0, ty1)),
                                                std::max(tz0, tz1));
        const f32 t_clamped = std::max(t_near, 0.0f);
        out_t[i] = (t_far >= t_clamped) ? t_clamped : INF;
    }
}

}  // namespace cardinal::core::simd::avx2
