// Baseline: the AVX-512 dot / sum exactly as they shipped before the
// multi-chain rewrite (one zmm accumulator — latency-bound at one FMA per
// 4 cycles on a dual-pipe core). Compiled with /arch:AVX512.
#include "naive.hpp"
#include <immintrin.h>

namespace naive_avx512 {

using cardinal::f32; using cardinal::usize;

static inline __mmask16 tail_mask(usize n_remaining) noexcept {
    return static_cast<__mmask16>((1u << n_remaining) - 1u);
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

}  // namespace naive_avx512
