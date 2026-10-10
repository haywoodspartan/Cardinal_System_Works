// Baseline: the AVX2+FMA dot / sum exactly as they shipped before the
// multi-chain rewrite (one ymm accumulator). Compiled with /arch:AVX2.
#include "naive.hpp"
#include <immintrin.h>

namespace naive_avx2 {

using cardinal::f32; using cardinal::usize;

static inline f32 hsum256_ps(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

f32 dot_f32(const f32* a, const f32* b, usize n) {
    __m256 acc = _mm256_setzero_ps();
    usize i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
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

}  // namespace naive_avx2
