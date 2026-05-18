#pragma once

// =============================================================================
// Per-ISA kernel declarations.
//
// Internal header — included by the per-tier .cpp files (which DEFINE
// these inside a tier-named namespace) AND by the dispatcher (which
// READS them by name to populate the function-pointer table).
//
// The macro CARDINAL_SIMD_TIER picks which tier's namespace each TU
// emits its kernels into. The CPP file defines the macro before
// including this header.
// =============================================================================

#include <cardinal/core/simd_math.hpp>
#include <cardinal/core/types.hpp>

#ifndef CARDINAL_SIMD_TIER
#define CARDINAL_SIMD_TIER scalar
#endif

namespace cardinal::core::simd::CARDINAL_SIMD_TIER {

void vec_add_f32  (f32* out, const f32* a, const f32* b, usize n);
void vec_sub_f32  (f32* out, const f32* a, const f32* b, usize n);
void vec_mul_f32  (f32* out, const f32* a, const f32* b, usize n);
void vec_scale_f32(f32* out, const f32* a, f32 s, usize n);
void vec_axpy_f32 (f32* y, f32 a, const f32* x, usize n);

f32  dot_f32(const f32* a, const f32* b, usize n);
f32  sum_f32(const f32* a, usize n);
f32  min_f32(const f32* a, usize n);
f32  max_f32(const f32* a, usize n);

void transform_points_mat4(f32* out_xyz, const f32* in_xyz,
                           const Mat4Compact& M, usize count);

void vec3_cross_array(f32* out_xyz, const f32* a_xyz, const f32* b_xyz, usize count);
void vec3_length_array(f32* out_lens, const f32* in_xyz, usize count);
void vec3_normalize_array_inplace(f32* io_xyz, usize count);

// mat4_mul_array — only declared in the tiers that ship it (scalar +
// sse42 + avx2 + avx512). The AVX tier intentionally omits this
// declaration AND its implementation; the dispatcher picks up the
// next-lower tier (SSE4.2) for that op when running on an AVX-only
// CPU. Wrapping the decl in an opt-in macro lets each TU choose to
// expose it or not.
#if !defined(CARDINAL_SIMD_OMIT_MAT4_MUL_ARRAY)
void mat4_mul_array(f32* out, const f32* A, const f32* B, usize count);
#endif

void frustum_cull_spheres(u8* out_bits, const f32* planes,
                          const f32* cx, const f32* cy,
                          const f32* cz, const f32* r,
                          usize count);

void transform_aabb_array(
    f32* out_min_x, f32* out_min_y, f32* out_min_z,
    f32* out_max_x, f32* out_max_y, f32* out_max_z,
    const f32* in_min_x, const f32* in_min_y, const f32* in_min_z,
    const f32* in_max_x, const f32* in_max_y, const f32* in_max_z,
    const Mat4Compact& M, usize count);

void frustum_cull_aabbs(u8* out_bits, const f32* planes,
                        const f32* min_x, const f32* min_y, const f32* min_z,
                        const f32* max_x, const f32* max_y, const f32* max_z,
                        usize count);

void ray_aabb_intersect_array(
    f32* out_t,
    f32 ox, f32 oy, f32 oz,
    f32 inv_dx, f32 inv_dy, f32 inv_dz,
    const f32* min_x, const f32* min_y, const f32* min_z,
    const f32* max_x, const f32* max_y, const f32* max_z,
    usize count);

void min_index_f32(f32* out_value, u32* out_index,
                   const f32* in, usize count);

void max_index_f32(f32* out_value, u32* out_index,
                   const f32* in, usize count);

void morton_encode_3d_array(u64* out_codes,
                            const u32* x, const u32* y, const u32* z,
                            usize count);

}  // namespace cardinal::core::simd::CARDINAL_SIMD_TIER
