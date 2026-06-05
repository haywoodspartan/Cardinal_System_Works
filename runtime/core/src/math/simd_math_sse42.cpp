// =============================================================================
// Cardinal — SIMD math: SSE2/SSE4.1/SSE4.2 (+ SSE4A on AMD).
//
// 128-bit lanes, 4 floats per vector. SSE2 is the x86_64 baseline so
// no /arch flag is strictly needed; including it explicitly here so the
// intent is documented.
//
// SSE4A (AMD-only) is leveraged opportunistically for the LZCNT-style
// helpers; the float-math kernels themselves don't need it (it doesn't
// add float instructions). We probe for it at dispatch time anyway so
// the engine knows it's available.
// =============================================================================
#define CARDINAL_SIMD_TIER sse42
#include "simd_math_kernels.hpp"

#include <emmintrin.h>     // SSE2  — _mm_*
#include <smmintrin.h>     // SSE4.1 — _mm_floor_ps, _mm_blend_ps
#include <nmmintrin.h>     // SSE4.2 — _mm_crc32_u32, _mm_cmpestri (text)

#include <algorithm>
#include <cmath>
#include <limits>

namespace cardinal::core::simd::sse42 {

// Walk i over n in 4-element blocks. The trailing 1..3 elements run
// scalar — branchless tail beats masked loads on this tier.
#define CARDINAL_SSE_TAIL(stmts) \
    for (; i < n; ++i) { stmts; }

void vec_add_f32(f32* out, const f32* a, const f32* b, usize n) {
    usize i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(out + i, _mm_add_ps(va, vb));
    }
    CARDINAL_SSE_TAIL(out[i] = a[i] + b[i]);
}
void vec_sub_f32(f32* out, const f32* a, const f32* b, usize n) {
    usize i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(out + i, _mm_sub_ps(va, vb));
    }
    CARDINAL_SSE_TAIL(out[i] = a[i] - b[i]);
}
void vec_mul_f32(f32* out, const f32* a, const f32* b, usize n) {
    usize i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(out + i, _mm_mul_ps(va, vb));
    }
    CARDINAL_SSE_TAIL(out[i] = a[i] * b[i]);
}
void vec_scale_f32(f32* out, const f32* a, f32 s, usize n) {
    const __m128 vs = _mm_set1_ps(s);
    usize i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        _mm_storeu_ps(out + i, _mm_mul_ps(va, vs));
    }
    CARDINAL_SSE_TAIL(out[i] = a[i] * s);
}
void vec_axpy_f32(f32* y, f32 a, const f32* x, usize n) {
    const __m128 va = _mm_set1_ps(a);
    usize i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 vx = _mm_loadu_ps(x + i);
        __m128 vy = _mm_loadu_ps(y + i);
        // No FMA on plain SSE — explicit mul + add.
        _mm_storeu_ps(y + i, _mm_add_ps(vy, _mm_mul_ps(va, vx)));
    }
    CARDINAL_SSE_TAIL(y[i] += a * x[i]);
}

// Horizontal reduce a 4-lane vector to a scalar via halves.
static inline f32 hsum_ps(__m128 v) noexcept {
    __m128 sh1 = _mm_movehl_ps(v, v);
    __m128 sum = _mm_add_ps(v, sh1);
    sh1        = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 0, 1));
    sum        = _mm_add_ss(sum, sh1);
    return _mm_cvtss_f32(sum);
}
static inline f32 hmin_ps(__m128 v) noexcept {
    __m128 sh = _mm_movehl_ps(v, v);
    v = _mm_min_ps(v, sh);
    sh = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 1));
    v = _mm_min_ss(v, sh);
    return _mm_cvtss_f32(v);
}
static inline f32 hmax_ps(__m128 v) noexcept {
    __m128 sh = _mm_movehl_ps(v, v);
    v = _mm_max_ps(v, sh);
    sh = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 1));
    v = _mm_max_ss(v, sh);
    return _mm_cvtss_f32(v);
}

f32 dot_f32(const f32* a, const f32* b, usize n) {
    __m128 acc = _mm_setzero_ps();
    usize i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
    }
    f32 tail = 0.0f;
    for (; i < n; ++i) tail += a[i] * b[i];
    return hsum_ps(acc) + tail;
}
f32 sum_f32(const f32* a, usize n) {
    __m128 acc = _mm_setzero_ps();
    usize i = 0;
    for (; i + 4 <= n; i += 4) {
        acc = _mm_add_ps(acc, _mm_loadu_ps(a + i));
    }
    f32 tail = 0.0f;
    for (; i < n; ++i) tail += a[i];
    return hsum_ps(acc) + tail;
}
f32 min_f32(const f32* a, usize n) {
    if (n == 0) return std::numeric_limits<f32>::infinity();
    __m128 acc = _mm_set1_ps(a[0]);
    usize i = 0;
    for (; i + 4 <= n; i += 4) {
        acc = _mm_min_ps(acc, _mm_loadu_ps(a + i));
    }
    f32 m = hmin_ps(acc);
    for (; i < n; ++i) m = std::min(m, a[i]);
    return m;
}
f32 max_f32(const f32* a, usize n) {
    if (n == 0) return -std::numeric_limits<f32>::infinity();
    __m128 acc = _mm_set1_ps(a[0]);
    usize i = 0;
    for (; i + 4 <= n; i += 4) {
        acc = _mm_max_ps(acc, _mm_loadu_ps(a + i));
    }
    f32 m = hmax_ps(acc);
    for (; i < n; ++i) m = std::max(m, a[i]);
    return m;
}

// Per-vertex transform — the inner work for one vert is one Vec3 dot
// per output channel. We process 4 verts at a time by gathering an SoA
// stripe (xs / ys / zs) so each output channel is one vectorised
// fmadd-equivalent. For non-multiple-of-4 trailing verts we fall back
// to scalar.
void transform_points_mat4(f32* out_xyz, const f32* in_xyz,
                           const Mat4Compact& M, usize count)
{
    const f32* m = M.m;
    const __m128 m00 = _mm_set1_ps(m[0]);  const __m128 m04 = _mm_set1_ps(m[4]);
    const __m128 m08 = _mm_set1_ps(m[8]);  const __m128 m12 = _mm_set1_ps(m[12]);
    const __m128 m01 = _mm_set1_ps(m[1]);  const __m128 m05 = _mm_set1_ps(m[5]);
    const __m128 m09 = _mm_set1_ps(m[9]);  const __m128 m13 = _mm_set1_ps(m[13]);
    const __m128 m02 = _mm_set1_ps(m[2]);  const __m128 m06 = _mm_set1_ps(m[6]);
    const __m128 m10 = _mm_set1_ps(m[10]); const __m128 m14 = _mm_set1_ps(m[14]);

    usize i = 0;
    for (; i + 4 <= count; i += 4) {
        // Gather 4 verts' xs/ys/zs from interleaved AoS.
        const f32* p = in_xyz + i*3;
        const __m128 xs = _mm_set_ps(p[ 9], p[6], p[3], p[0]);
        const __m128 ys = _mm_set_ps(p[10], p[7], p[4], p[1]);
        const __m128 zs = _mm_set_ps(p[11], p[8], p[5], p[2]);

        const __m128 ox = _mm_add_ps(_mm_add_ps(_mm_mul_ps(m00, xs), _mm_mul_ps(m04, ys)),
                                     _mm_add_ps(_mm_mul_ps(m08, zs), m12));
        const __m128 oy = _mm_add_ps(_mm_add_ps(_mm_mul_ps(m01, xs), _mm_mul_ps(m05, ys)),
                                     _mm_add_ps(_mm_mul_ps(m09, zs), m13));
        const __m128 oz = _mm_add_ps(_mm_add_ps(_mm_mul_ps(m02, xs), _mm_mul_ps(m06, ys)),
                                     _mm_add_ps(_mm_mul_ps(m10, zs), m14));

        // Scatter back to AoS via 4 _mm_store_ss equivalents.
        alignas(16) f32 ox_arr[4], oy_arr[4], oz_arr[4];
        _mm_store_ps(ox_arr, ox);
        _mm_store_ps(oy_arr, oy);
        _mm_store_ps(oz_arr, oz);
        f32* o = out_xyz + i*3;
        for (int k = 0; k < 4; ++k) {
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
    // 4 verts per iteration via SoA-gather of xs/ys/zs.
    usize i = 0;
    for (; i + 4 <= count; i += 4) {
        const f32* a = a_xyz + i*3;
        const f32* b = b_xyz + i*3;
        const __m128 ax = _mm_set_ps(a[ 9], a[6], a[3], a[0]);
        const __m128 ay = _mm_set_ps(a[10], a[7], a[4], a[1]);
        const __m128 az = _mm_set_ps(a[11], a[8], a[5], a[2]);
        const __m128 bx = _mm_set_ps(b[ 9], b[6], b[3], b[0]);
        const __m128 by = _mm_set_ps(b[10], b[7], b[4], b[1]);
        const __m128 bz = _mm_set_ps(b[11], b[8], b[5], b[2]);
        const __m128 ox = _mm_sub_ps(_mm_mul_ps(ay, bz), _mm_mul_ps(az, by));
        const __m128 oy = _mm_sub_ps(_mm_mul_ps(az, bx), _mm_mul_ps(ax, bz));
        const __m128 oz = _mm_sub_ps(_mm_mul_ps(ax, by), _mm_mul_ps(ay, bx));
        alignas(16) f32 xa[4], ya[4], za[4];
        _mm_store_ps(xa, ox); _mm_store_ps(ya, oy); _mm_store_ps(za, oz);
        f32* o = out_xyz + i*3;
        for (int k = 0; k < 4; ++k) {
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
    for (; i + 4 <= count; i += 4) {
        const f32* p = in_xyz + i*3;
        const __m128 xs = _mm_set_ps(p[ 9], p[6], p[3], p[0]);
        const __m128 ys = _mm_set_ps(p[10], p[7], p[4], p[1]);
        const __m128 zs = _mm_set_ps(p[11], p[8], p[5], p[2]);
        const __m128 sq = _mm_add_ps(_mm_add_ps(_mm_mul_ps(xs, xs), _mm_mul_ps(ys, ys)),
                                     _mm_mul_ps(zs, zs));
        _mm_storeu_ps(out_lens + i, _mm_sqrt_ps(sq));
    }
    for (; i < count; ++i) {
        const f32 x = in_xyz[i*3+0], y = in_xyz[i*3+1], z = in_xyz[i*3+2];
        out_lens[i] = std::sqrt(x*x + y*y + z*z);
    }
}

void vec3_normalize_array_inplace(f32* io_xyz, usize count) {
    const __m128 eps_v = _mm_set1_ps(1e-8f);
    usize i = 0;
    for (; i + 4 <= count; i += 4) {
        f32* p = io_xyz + i*3;
        __m128 xs = _mm_set_ps(p[ 9], p[6], p[3], p[0]);
        __m128 ys = _mm_set_ps(p[10], p[7], p[4], p[1]);
        __m128 zs = _mm_set_ps(p[11], p[8], p[5], p[2]);
        const __m128 sq = _mm_add_ps(_mm_add_ps(_mm_mul_ps(xs, xs), _mm_mul_ps(ys, ys)),
                                     _mm_mul_ps(zs, zs));
        // mask = (sq >= eps); zero vectors get inv=0 → unchanged below.
        const __m128 mask = _mm_cmpge_ps(sq, eps_v);
        const __m128 inv  = _mm_and_ps(mask,
                              _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(sq)));
        // For lanes where mask==0 (zero vector), inv==0 — multiplied
        // back gives 0, which would clobber the original. Blend back
        // the original on those lanes. SSE4.1's blendv reads the high
        // bit of mask as the selector — perfect.
        const __m128 nxs = _mm_blendv_ps(xs, _mm_mul_ps(xs, inv), mask);
        const __m128 nys = _mm_blendv_ps(ys, _mm_mul_ps(ys, inv), mask);
        const __m128 nzs = _mm_blendv_ps(zs, _mm_mul_ps(zs, inv), mask);
        alignas(16) f32 xa[4], ya[4], za[4];
        _mm_store_ps(xa, nxs); _mm_store_ps(ya, nys); _mm_store_ps(za, nzs);
        for (int k = 0; k < 4; ++k) {
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

// SSE 4-wide column. Each column of A is one __m128. Each column of
// the output = sum over k of (A_col[k] * B[col, k]). Plain mul + add
// — no FMA on this tier.
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
            const __m128 oc =
                _mm_add_ps(_mm_add_ps(_mm_mul_ps(ac0, b0), _mm_mul_ps(ac1, b1)),
                           _mm_add_ps(_mm_mul_ps(ac2, b2), _mm_mul_ps(ac3, b3)));
            _mm_storeu_ps(o + c*4, oc);
        }
    }
}

// Frustum cull — 4 spheres / inner SSE iter, paired into 8 spheres /
// outer iter so each output write is one full byte. The 4-sphere
// inner kernel runs a 4-lane FMA-style chain (mul + add since SSE4.2
// has no FMA), accumulating "passes-this-plane" mask via AND across
// all 6 planes, then movemasks the 4 lanes to a nibble.
void frustum_cull_spheres(u8* out_bits, const f32* planes,
                          const f32* cx, const f32* cy,
                          const f32* cz, const f32* r,
                          usize count)
{
    auto test4 = [&](usize i) -> int {
        const __m128 vcx = _mm_loadu_ps(cx + i);
        const __m128 vcy = _mm_loadu_ps(cy + i);
        const __m128 vcz = _mm_loadu_ps(cz + i);
        const __m128 vr  = _mm_loadu_ps(r  + i);
        __m128 acc = _mm_castsi128_ps(_mm_set1_epi32(-1));  // all-1 (visible)
        for (int p = 0; p < 6; ++p) {
            const __m128 nx = _mm_set1_ps(planes[p*4 + 0]);
            const __m128 ny = _mm_set1_ps(planes[p*4 + 1]);
            const __m128 nz = _mm_set1_ps(planes[p*4 + 2]);
            const __m128 d  = _mm_set1_ps(planes[p*4 + 3]);
            // dist = nx*cx + ny*cy + nz*cz + d ; pass if dist + r >= 0.
            // SSE4.2 has no FMA, so it's mul + add chains.
            __m128 dist = _mm_add_ps(_mm_mul_ps(nx, vcx),
                            _mm_add_ps(_mm_mul_ps(ny, vcy),
                              _mm_add_ps(_mm_mul_ps(nz, vcz), d)));
            __m128 pass = _mm_cmpge_ps(_mm_add_ps(dist, vr), _mm_setzero_ps());
            acc = _mm_and_ps(acc, pass);
        }
        return _mm_movemask_ps(acc);   // 4 bits, lane order
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
        const int lo = test4(i);
        const int hi = test4(i + 4);
        out_bits[i >> 3] = static_cast<u8>(lo | (hi << 4));
    }
    if (i < count) {
        u8 byte = 0;
        for (usize b = 0; i + b < count; ++b) {
            if (test_one(i + b)) byte |= static_cast<u8>(1u << b);
        }
        out_bits[i >> 3] = byte;
    }
}

// AABB transform — 4 AABBs / iter. SoA layout means each m_ij can be
// broadcast once into __m128 and FMA'd against in_min[i] and in_max[i]
// in parallel; the per-axis min/max merge uses _mm_min_ps/_mm_max_ps.
void transform_aabb_array(
    f32* out_min_x, f32* out_min_y, f32* out_min_z,
    f32* out_max_x, f32* out_max_y, f32* out_max_z,
    const f32* in_min_x, const f32* in_min_y, const f32* in_min_z,
    const f32* in_max_x, const f32* in_max_y, const f32* in_max_z,
    const Mat4Compact& M, usize count)
{
    const __m128 m00 = _mm_set1_ps(M.m[ 0]);
    const __m128 m01 = _mm_set1_ps(M.m[ 1]);
    const __m128 m02 = _mm_set1_ps(M.m[ 2]);
    const __m128 m10 = _mm_set1_ps(M.m[ 4]);
    const __m128 m11 = _mm_set1_ps(M.m[ 5]);
    const __m128 m12 = _mm_set1_ps(M.m[ 6]);
    const __m128 m20 = _mm_set1_ps(M.m[ 8]);
    const __m128 m21 = _mm_set1_ps(M.m[ 9]);
    const __m128 m22 = _mm_set1_ps(M.m[10]);
    const __m128 t0  = _mm_set1_ps(M.m[12]);
    const __m128 t1  = _mm_set1_ps(M.m[13]);
    const __m128 t2  = _mm_set1_ps(M.m[14]);

    auto kernel = [&](usize i, usize n_lanes_unused = 4) {
        (void)n_lanes_unused;
        const __m128 nx = _mm_loadu_ps(in_min_x + i);
        const __m128 ny = _mm_loadu_ps(in_min_y + i);
        const __m128 nz = _mm_loadu_ps(in_min_z + i);
        const __m128 px = _mm_loadu_ps(in_max_x + i);
        const __m128 py = _mm_loadu_ps(in_max_y + i);
        const __m128 pz = _mm_loadu_ps(in_max_z + i);

        // For each output axis o, accumulate min/max across the 3
        // input-axis contributions starting from translation t_o.
        auto axis = [&](__m128 mo0, __m128 mo1, __m128 mo2, __m128 to,
                        __m128& out_n, __m128& out_p) {
            __m128 a, b;
            // Input X contribution
            a = _mm_mul_ps(mo0, nx);
            b = _mm_mul_ps(mo0, px);
            __m128 sum_n = _mm_add_ps(to, _mm_min_ps(a, b));
            __m128 sum_p = _mm_add_ps(to, _mm_max_ps(a, b));
            // Input Y contribution
            a = _mm_mul_ps(mo1, ny);
            b = _mm_mul_ps(mo1, py);
            sum_n = _mm_add_ps(sum_n, _mm_min_ps(a, b));
            sum_p = _mm_add_ps(sum_p, _mm_max_ps(a, b));
            // Input Z contribution
            a = _mm_mul_ps(mo2, nz);
            b = _mm_mul_ps(mo2, pz);
            sum_n = _mm_add_ps(sum_n, _mm_min_ps(a, b));
            sum_p = _mm_add_ps(sum_p, _mm_max_ps(a, b));
            out_n = sum_n;
            out_p = sum_p;
        };

        __m128 oxn, oxp, oyn, oyp, ozn, ozp;
        axis(m00, m10, m20, t0, oxn, oxp);
        axis(m01, m11, m21, t1, oyn, oyp);
        axis(m02, m12, m22, t2, ozn, ozp);

        _mm_storeu_ps(out_min_x + i, oxn); _mm_storeu_ps(out_max_x + i, oxp);
        _mm_storeu_ps(out_min_y + i, oyn); _mm_storeu_ps(out_max_y + i, oyp);
        _mm_storeu_ps(out_min_z + i, ozn); _mm_storeu_ps(out_max_z + i, ozp);
    };

    usize i = 0;
    for (; i + 4 <= count; i += 4) kernel(i);
    // Tail: scalar — 4-lane masked stores aren't worth the code at this width.
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

// Frustum-vs-AABB cull — 4 AABBs / iter inner, paired into 8 / outer
// for byte-aligned writes. Per plane we pre-pick max-or-min array
// pointers; the inner loop is then 3 mul + 3 add (no FMA on SSE4.2)
// + 1 cmp per plane.
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

    auto test4 = [&](usize i) -> int {
        __m128 acc = _mm_castsi128_ps(_mm_set1_epi32(-1));
        for (int p = 0; p < 6; ++p) {
            const auto& s = sels[p];
            const __m128 nx = _mm_set1_ps(s.nx);
            const __m128 ny = _mm_set1_ps(s.ny);
            const __m128 nz = _mm_set1_ps(s.nz);
            const __m128 d  = _mm_set1_ps(s.d);
            const __m128 vpx = _mm_loadu_ps(s.px + i);
            const __m128 vpy = _mm_loadu_ps(s.py + i);
            const __m128 vpz = _mm_loadu_ps(s.pz + i);
            __m128 dist = _mm_add_ps(_mm_mul_ps(nx, vpx),
                            _mm_add_ps(_mm_mul_ps(ny, vpy),
                              _mm_add_ps(_mm_mul_ps(nz, vpz), d)));
            __m128 pass = _mm_cmpge_ps(dist, _mm_setzero_ps());
            acc = _mm_and_ps(acc, pass);
        }
        return _mm_movemask_ps(acc);
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
    for (; i + 8 <= count; i += 8) {
        const int lo = test4(i);
        const int hi = test4(i + 4);
        out_bits[i >> 3] = static_cast<u8>(lo | (hi << 4));
    }
    if (i < count) {
        u8 byte = 0;
        for (usize b = 0; i + b < count; ++b) {
            if (test_one(i + b)) byte |= static_cast<u8>(1u << b);
        }
        out_bits[i >> 3] = byte;
    }
}

// argmax companion — same parallel-track pattern as min_index_f32 but
// blend on greater-than. Initial cur_max = first 4 elements; final
// horizontal reduce picks the largest across lanes.
void max_index_f32(f32* out_value, u32* out_index,
                   const f32* in, usize count)
{
    if (count == 0) {
        *out_value = -std::numeric_limits<f32>::infinity();
        *out_index = 0;
        return;
    }
    if (count < 4) {
        f32 best = in[0]; u32 best_i = 0;
        for (usize i = 1; i < count; ++i) {
            if (in[i] > best) { best = in[i]; best_i = static_cast<u32>(i); }
        }
        *out_value = best; *out_index = best_i;
        return;
    }

    __m128  cur_max = _mm_loadu_ps(in);
    __m128i cur_idx = _mm_setr_epi32(0, 1, 2, 3);
    usize i = 4;
    for (; i + 4 <= count; i += 4) {
        const __m128  v   = _mm_loadu_ps(in + i);
        const __m128i idx = _mm_setr_epi32(
            static_cast<int>(i),     static_cast<int>(i + 1),
            static_cast<int>(i + 2), static_cast<int>(i + 3));
        const __m128 greater_mask = _mm_cmpgt_ps(v, cur_max);
        cur_max = _mm_max_ps(cur_max, v);
        cur_idx = _mm_castps_si128(_mm_blendv_ps(
            _mm_castsi128_ps(cur_idx),
            _mm_castsi128_ps(idx),
            greater_mask));
    }

    alignas(16) f32 lane_v[4];
    alignas(16) u32 lane_i[4];
    _mm_store_ps(lane_v, cur_max);
    _mm_store_si128(reinterpret_cast<__m128i*>(lane_i), cur_idx);
    f32 best   = lane_v[0];
    u32 best_i = lane_i[0];
    for (int k = 1; k < 4; ++k) {
        if (lane_v[k] > best) { best = lane_v[k]; best_i = lane_i[k]; }
    }
    for (; i < count; ++i) {
        if (in[i] > best) { best = in[i]; best_i = static_cast<u32>(i); }
    }
    *out_value = best;
    *out_index = best_i;
}

// 3D Morton encoding (Z-curve) — 2 u64 lanes / iter via __m128i.
// Bit-spread sequence applied per 64-bit lane via vpsllq + vpand chains.
namespace {
inline __m128i spread21_2x_sse(__m128i v) noexcept {
    const __m128i m0 = _mm_set1_epi64x(0x1fffffLL);
    const __m128i m1 = _mm_set1_epi64x(0x001f00000000ffffLL);
    const __m128i m2 = _mm_set1_epi64x(0x001f0000ff0000ffLL);
    const __m128i m3 = _mm_set1_epi64x(0x100f00f00f00f00fLL);
    const __m128i m4 = _mm_set1_epi64x(0x10c30c30c30c30c3LL);
    const __m128i m5 = _mm_set1_epi64x(0x1249249249249249LL);
    v = _mm_and_si128(v, m0);
    v = _mm_and_si128(_mm_or_si128(v, _mm_slli_epi64(v, 32)), m1);
    v = _mm_and_si128(_mm_or_si128(v, _mm_slli_epi64(v, 16)), m2);
    v = _mm_and_si128(_mm_or_si128(v, _mm_slli_epi64(v,  8)), m3);
    v = _mm_and_si128(_mm_or_si128(v, _mm_slli_epi64(v,  4)), m4);
    v = _mm_and_si128(_mm_or_si128(v, _mm_slli_epi64(v,  2)), m5);
    return v;
}
inline u64 spread21_scalar(u32 v) noexcept {
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
    for (; i + 2 <= count; i += 2) {
        // _mm_loadl_epi64 + _mm_cvtepu32_epi64 gives us 2 u32s zero-
        // extended into 2 u64 lanes — exactly what spread21 wants.
        const __m128i vx = _mm_cvtepu32_epi64(_mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(x + i)));
        const __m128i vy = _mm_cvtepu32_epi64(_mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(y + i)));
        const __m128i vz = _mm_cvtepu32_epi64(_mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(z + i)));
        const __m128i sx = spread21_2x_sse(vx);
        const __m128i sy = spread21_2x_sse(vy);
        const __m128i sz = spread21_2x_sse(vz);
        const __m128i m  = _mm_or_si128(sx,
                          _mm_or_si128(_mm_slli_epi64(sy, 1),
                                       _mm_slli_epi64(sz, 2)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out_codes + i), m);
    }
    for (; i < count; ++i) {
        out_codes[i] = spread21_scalar(x[i])
                    | (spread21_scalar(y[i]) << 1)
                    | (spread21_scalar(z[i]) << 2);
    }
}

// argmin reduction — SSE4.2 path. 4 lanes / iter; track per-lane min
// + per-lane source index in parallel SIMD registers, then horizontal
// scalar reduce after the bulk loop. _mm_blendv_ps (SSE4.1+) selects
// based on the per-lane "is smaller" mask.
void min_index_f32(f32* out_value, u32* out_index,
                   const f32* in, usize count)
{
    if (count == 0) {
        *out_value = std::numeric_limits<f32>::infinity();
        *out_index = 0;
        return;
    }
    if (count < 4) {
        // Pure scalar tail.
        f32 best = in[0]; u32 best_i = 0;
        for (usize i = 1; i < count; ++i) {
            if (in[i] < best) { best = in[i]; best_i = static_cast<u32>(i); }
        }
        *out_value = best; *out_index = best_i;
        return;
    }

    __m128  cur_min = _mm_loadu_ps(in);
    __m128i cur_idx = _mm_setr_epi32(0, 1, 2, 3);
    usize i = 4;
    for (; i + 4 <= count; i += 4) {
        const __m128  v   = _mm_loadu_ps(in + i);
        const __m128i idx = _mm_setr_epi32(
            static_cast<int>(i),     static_cast<int>(i + 1),
            static_cast<int>(i + 2), static_cast<int>(i + 3));
        const __m128 less_mask = _mm_cmplt_ps(v, cur_min);
        cur_min = _mm_min_ps(cur_min, v);
        cur_idx = _mm_castps_si128(_mm_blendv_ps(
            _mm_castsi128_ps(cur_idx),
            _mm_castsi128_ps(idx),
            less_mask));
    }

    // Horizontal reduce — spill the 4 lanes and scan scalar.
    alignas(16) f32 lane_v[4];
    alignas(16) u32 lane_i[4];
    _mm_store_ps(lane_v, cur_min);
    _mm_store_si128(reinterpret_cast<__m128i*>(lane_i), cur_idx);
    f32 best   = lane_v[0];
    u32 best_i = lane_i[0];
    for (int k = 1; k < 4; ++k) {
        if (lane_v[k] < best) { best = lane_v[k]; best_i = lane_i[k]; }
    }
    // Scalar tail (count % 4 elements remaining).
    for (; i < count; ++i) {
        if (in[i] < best) { best = in[i]; best_i = static_cast<u32>(i); }
    }
    *out_value = best;
    *out_index = best_i;
}

// Ray-vs-AABB batch — 4 AABBs / iter via __m128 SoA. The slab math
// is a tight chain of (load - origin) * inv_dir, then per-axis
// min/max + cross-axis max/min, then blend INF for misses.
void ray_aabb_intersect_array(
    f32* out_t,
    f32 ox, f32 oy, f32 oz,
    f32 inv_dx, f32 inv_dy, f32 inv_dz,
    const f32* min_x, const f32* min_y, const f32* min_z,
    const f32* max_x, const f32* max_y, const f32* max_z,
    usize count)
{
    const __m128 vox = _mm_set1_ps(ox);
    const __m128 voy = _mm_set1_ps(oy);
    const __m128 voz = _mm_set1_ps(oz);
    const __m128 vix = _mm_set1_ps(inv_dx);
    const __m128 viy = _mm_set1_ps(inv_dy);
    const __m128 viz = _mm_set1_ps(inv_dz);
    const __m128 vinf = _mm_set1_ps(std::numeric_limits<f32>::infinity());
    const __m128 vzero = _mm_setzero_ps();

    usize i = 0;
    for (; i + 4 <= count; i += 4) {
        const __m128 t0x = _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(min_x + i), vox), vix);
        const __m128 t1x = _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(max_x + i), vox), vix);
        const __m128 t0y = _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(min_y + i), voy), viy);
        const __m128 t1y = _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(max_y + i), voy), viy);
        const __m128 t0z = _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(min_z + i), voz), viz);
        const __m128 t1z = _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(max_z + i), voz), viz);

        const __m128 nx = _mm_min_ps(t0x, t1x);
        const __m128 fx = _mm_max_ps(t0x, t1x);
        const __m128 ny = _mm_min_ps(t0y, t1y);
        const __m128 fy = _mm_max_ps(t0y, t1y);
        const __m128 nz = _mm_min_ps(t0z, t1z);
        const __m128 fz = _mm_max_ps(t0z, t1z);

        const __m128 t_near = _mm_max_ps(_mm_max_ps(nx, ny), nz);
        const __m128 t_far  = _mm_min_ps(_mm_min_ps(fx, fy), fz);
        const __m128 t_cl   = _mm_max_ps(t_near, vzero);
        const __m128 hit    = _mm_cmpge_ps(t_far, t_cl);
        // Blend INF where !hit, t_cl where hit. SSE4.1+ has _mm_blendv_ps.
        const __m128 result = _mm_blendv_ps(vinf, t_cl, hit);
        _mm_storeu_ps(out_t + i, result);
    }
    // Scalar tail.
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

}  // namespace cardinal::core::simd::sse42
