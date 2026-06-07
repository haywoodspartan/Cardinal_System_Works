// =============================================================================
// Cardinal — SIMD math dispatcher.
//
// Reads cardinal::hal::cpu_features() ONCE at static init and binds
// each function-pointer in the public API to the highest-tier
// implementation the host CPU supports. Per-op fallback: if the
// AVX-512 TU exports an op but a future build only ships scalar +
// SSE4.2, the dispatcher walks the ladder and lands at SSE4.2.
//
// Static-init order matters: this TU's init runs before main() starts,
// driven by the std::atomic<bool> "first-use" guard in init() —
// every public-API call also nudges init() so a thread that beats
// static-init ordering still gets a valid binding.
// =============================================================================
#include <cardinal/core/math/simd_math.hpp>

#include <cardinal/core/os/hal.hpp>
#include <cardinal/core/diag/log.hpp>

#include <atomic>

#include "simd_math_kernels.hpp"

// Pull in each per-tier namespace's declarations. Each include
// re-defines CARDINAL_SIMD_TIER and emits a fresh forward-declaration
// set into the matching namespace. We then compose a per-op tier
// preference in pick_*() helpers below.
namespace cardinal::core::simd::scalar {
    void vec_add_f32(f32*, const f32*, const f32*, usize);
    void vec_sub_f32(f32*, const f32*, const f32*, usize);
    void vec_mul_f32(f32*, const f32*, const f32*, usize);
    void vec_scale_f32(f32*, const f32*, f32, usize);
    void vec_axpy_f32(f32*, f32, const f32*, usize);
    f32  dot_f32(const f32*, const f32*, usize);
    f32  sum_f32(const f32*, usize);
    f32  min_f32(const f32*, usize);
    f32  max_f32(const f32*, usize);
    void transform_points_mat4(f32*, const f32*, const Mat4Compact&, usize);
    void vec3_cross_array(f32*, const f32*, const f32*, usize);
    void vec3_length_array(f32*, const f32*, usize);
    void vec3_normalize_array_inplace(f32*, usize);
    void mat4_mul_array(f32*, const f32*, const f32*, usize);
    void frustum_cull_spheres(u8*, const f32*, const f32*, const f32*, const f32*, const f32*, usize);
    void transform_aabb_array(f32*, f32*, f32*, f32*, f32*, f32*,
                              const f32*, const f32*, const f32*,
                              const f32*, const f32*, const f32*,
                              const Mat4Compact&, usize);
    void frustum_cull_aabbs(u8*, const f32*,
                            const f32*, const f32*, const f32*,
                            const f32*, const f32*, const f32*,
                            usize);
    void ray_aabb_intersect_array(f32*, f32, f32, f32, f32, f32, f32,
                                  const f32*, const f32*, const f32*,
                                  const f32*, const f32*, const f32*,
                                  usize);
    void min_index_f32(f32*, u32*, const f32*, usize);
    void max_index_f32(f32*, u32*, const f32*, usize);
    void morton_encode_3d_array(u64*, const u32*, const u32*, const u32*, usize);
}
namespace cardinal::core::simd::sse42 {
    void vec_add_f32(f32*, const f32*, const f32*, usize);
    void vec_sub_f32(f32*, const f32*, const f32*, usize);
    void vec_mul_f32(f32*, const f32*, const f32*, usize);
    void vec_scale_f32(f32*, const f32*, f32, usize);
    void vec_axpy_f32(f32*, f32, const f32*, usize);
    f32  dot_f32(const f32*, const f32*, usize);
    f32  sum_f32(const f32*, usize);
    f32  min_f32(const f32*, usize);
    f32  max_f32(const f32*, usize);
    void transform_points_mat4(f32*, const f32*, const Mat4Compact&, usize);
    void vec3_cross_array(f32*, const f32*, const f32*, usize);
    void vec3_length_array(f32*, const f32*, usize);
    void vec3_normalize_array_inplace(f32*, usize);
    void mat4_mul_array(f32*, const f32*, const f32*, usize);
    void frustum_cull_spheres(u8*, const f32*, const f32*, const f32*, const f32*, const f32*, usize);
    void transform_aabb_array(f32*, f32*, f32*, f32*, f32*, f32*,
                              const f32*, const f32*, const f32*,
                              const f32*, const f32*, const f32*,
                              const Mat4Compact&, usize);
    void frustum_cull_aabbs(u8*, const f32*,
                            const f32*, const f32*, const f32*,
                            const f32*, const f32*, const f32*,
                            usize);
    void ray_aabb_intersect_array(f32*, f32, f32, f32, f32, f32, f32,
                                  const f32*, const f32*, const f32*,
                                  const f32*, const f32*, const f32*,
                                  usize);
    void min_index_f32(f32*, u32*, const f32*, usize);
    void max_index_f32(f32*, u32*, const f32*, usize);
    void morton_encode_3d_array(u64*, const u32*, const u32*, const u32*, usize);
}
namespace cardinal::core::simd::avx {
    void vec_add_f32(f32*, const f32*, const f32*, usize);
    void vec_sub_f32(f32*, const f32*, const f32*, usize);
    void vec_mul_f32(f32*, const f32*, const f32*, usize);
    void vec_scale_f32(f32*, const f32*, f32, usize);
    void vec_axpy_f32(f32*, f32, const f32*, usize);
    f32  dot_f32(const f32*, const f32*, usize);
    f32  sum_f32(const f32*, usize);
    f32  min_f32(const f32*, usize);
    f32  max_f32(const f32*, usize);
    void transform_points_mat4(f32*, const f32*, const Mat4Compact&, usize);
    void vec3_cross_array(f32*, const f32*, const f32*, usize);
    void vec3_length_array(f32*, const f32*, usize);
    void vec3_normalize_array_inplace(f32*, usize);
    // mat4_mul_array intentionally OMITTED here — avx tier doesn't ship
    // it (see comment in simd_math.hpp). Dispatcher's tier walk leaves
    // the binding pointing at the previous tier (sse42::mat4_mul_array),
    // which is the per-op fallback design exercised for the first time.
    void frustum_cull_spheres(u8*, const f32*, const f32*, const f32*, const f32*, const f32*, usize);
    void transform_aabb_array(f32*, f32*, f32*, f32*, f32*, f32*,
                              const f32*, const f32*, const f32*,
                              const f32*, const f32*, const f32*,
                              const Mat4Compact&, usize);
    void frustum_cull_aabbs(u8*, const f32*,
                            const f32*, const f32*, const f32*,
                            const f32*, const f32*, const f32*,
                            usize);
    void ray_aabb_intersect_array(f32*, f32, f32, f32, f32, f32, f32,
                                  const f32*, const f32*, const f32*,
                                  const f32*, const f32*, const f32*,
                                  usize);
    void min_index_f32(f32*, u32*, const f32*, usize);
    void max_index_f32(f32*, u32*, const f32*, usize);
    void morton_encode_3d_array(u64*, const u32*, const u32*, const u32*, usize);
}
namespace cardinal::core::simd::avx2 {
    void vec_add_f32(f32*, const f32*, const f32*, usize);
    void vec_sub_f32(f32*, const f32*, const f32*, usize);
    void vec_mul_f32(f32*, const f32*, const f32*, usize);
    void vec_scale_f32(f32*, const f32*, f32, usize);
    void vec_axpy_f32(f32*, f32, const f32*, usize);
    f32  dot_f32(const f32*, const f32*, usize);
    f32  sum_f32(const f32*, usize);
    f32  min_f32(const f32*, usize);
    f32  max_f32(const f32*, usize);
    void transform_points_mat4(f32*, const f32*, const Mat4Compact&, usize);
    void vec3_cross_array(f32*, const f32*, const f32*, usize);
    void vec3_length_array(f32*, const f32*, usize);
    void vec3_normalize_array_inplace(f32*, usize);
    void mat4_mul_array(f32*, const f32*, const f32*, usize);
    void frustum_cull_spheres(u8*, const f32*, const f32*, const f32*, const f32*, const f32*, usize);
    void transform_aabb_array(f32*, f32*, f32*, f32*, f32*, f32*,
                              const f32*, const f32*, const f32*,
                              const f32*, const f32*, const f32*,
                              const Mat4Compact&, usize);
    void frustum_cull_aabbs(u8*, const f32*,
                            const f32*, const f32*, const f32*,
                            const f32*, const f32*, const f32*,
                            usize);
    void ray_aabb_intersect_array(f32*, f32, f32, f32, f32, f32, f32,
                                  const f32*, const f32*, const f32*,
                                  const f32*, const f32*, const f32*,
                                  usize);
    void min_index_f32(f32*, u32*, const f32*, usize);
    void max_index_f32(f32*, u32*, const f32*, usize);
    void morton_encode_3d_array(u64*, const u32*, const u32*, const u32*, usize);
}
namespace cardinal::core::simd::avx512 {
    void vec_add_f32(f32*, const f32*, const f32*, usize);
    void vec_sub_f32(f32*, const f32*, const f32*, usize);
    void vec_mul_f32(f32*, const f32*, const f32*, usize);
    void vec_scale_f32(f32*, const f32*, f32, usize);
    void vec_axpy_f32(f32*, f32, const f32*, usize);
    f32  dot_f32(const f32*, const f32*, usize);
    f32  sum_f32(const f32*, usize);
    f32  min_f32(const f32*, usize);
    f32  max_f32(const f32*, usize);
    void transform_points_mat4(f32*, const f32*, const Mat4Compact&, usize);
    void vec3_cross_array(f32*, const f32*, const f32*, usize);
    void vec3_length_array(f32*, const f32*, usize);
    void vec3_normalize_array_inplace(f32*, usize);
    void mat4_mul_array(f32*, const f32*, const f32*, usize);
    void frustum_cull_spheres(u8*, const f32*, const f32*, const f32*, const f32*, const f32*, usize);
    void transform_aabb_array(f32*, f32*, f32*, f32*, f32*, f32*,
                              const f32*, const f32*, const f32*,
                              const f32*, const f32*, const f32*,
                              const Mat4Compact&, usize);
    void frustum_cull_aabbs(u8*, const f32*,
                            const f32*, const f32*, const f32*,
                            const f32*, const f32*, const f32*,
                            usize);
    void ray_aabb_intersect_array(f32*, f32, f32, f32, f32, f32, f32,
                                  const f32*, const f32*, const f32*,
                                  const f32*, const f32*, const f32*,
                                  usize);
    void min_index_f32(f32*, u32*, const f32*, usize);
    void max_index_f32(f32*, u32*, const f32*, usize);
    void morton_encode_3d_array(u64*, const u32*, const u32*, const u32*, usize);
}

namespace cardinal::core::simd {

// ----- Storage for the function-pointer table ------------------------------
//
// Bound to scalar at static-init order 0 so a tight static-init race that
// beats our init() also lands on a valid pointer (correct, just slow).
void  (*vec_add_f32)  (f32*, const f32*, const f32*, usize)        = &scalar::vec_add_f32;
void  (*vec_sub_f32)  (f32*, const f32*, const f32*, usize)        = &scalar::vec_sub_f32;
void  (*vec_mul_f32)  (f32*, const f32*, const f32*, usize)        = &scalar::vec_mul_f32;
void  (*vec_scale_f32)(f32*, const f32*, f32, usize)               = &scalar::vec_scale_f32;
void  (*vec_axpy_f32) (f32*, f32, const f32*, usize)               = &scalar::vec_axpy_f32;
f32   (*dot_f32)(const f32*, const f32*, usize)                    = &scalar::dot_f32;
f32   (*sum_f32)(const f32*, usize)                                = &scalar::sum_f32;
f32   (*min_f32)(const f32*, usize)                                = &scalar::min_f32;
f32   (*max_f32)(const f32*, usize)                                = &scalar::max_f32;
void  (*transform_points_mat4)(f32*, const f32*, const Mat4Compact&, usize)
                                                                    = &scalar::transform_points_mat4;
void  (*vec3_cross_array)(f32*, const f32*, const f32*, usize)     = &scalar::vec3_cross_array;
void  (*vec3_length_array)(f32*, const f32*, usize)                = &scalar::vec3_length_array;
void  (*vec3_normalize_array_inplace)(f32*, usize)                 = &scalar::vec3_normalize_array_inplace;
void  (*mat4_mul_array)(f32*, const f32*, const f32*, usize)       = &scalar::mat4_mul_array;
void  (*frustum_cull_spheres)(u8*, const f32*, const f32*, const f32*, const f32*, const f32*, usize)
                                                                    = &scalar::frustum_cull_spheres;
void  (*transform_aabb_array)(f32*, f32*, f32*, f32*, f32*, f32*,
                              const f32*, const f32*, const f32*,
                              const f32*, const f32*, const f32*,
                              const Mat4Compact&, usize)
                                                                    = &scalar::transform_aabb_array;
void  (*frustum_cull_aabbs)(u8*, const f32*,
                            const f32*, const f32*, const f32*,
                            const f32*, const f32*, const f32*,
                            usize)                                  = &scalar::frustum_cull_aabbs;
void  (*ray_aabb_intersect_array)(f32*, f32, f32, f32, f32, f32, f32,
                                  const f32*, const f32*, const f32*,
                                  const f32*, const f32*, const f32*,
                                  usize)                            = &scalar::ray_aabb_intersect_array;
void  (*min_index_f32)(f32*, u32*, const f32*, usize)               = &scalar::min_index_f32;
void  (*max_index_f32)(f32*, u32*, const f32*, usize)               = &scalar::max_index_f32;
void  (*morton_encode_3d_array)(u64*, const u32*, const u32*, const u32*, usize)
                                                                    = &scalar::morton_encode_3d_array;

namespace {

std::atomic<bool> g_initialised{false};
Backend           g_active_backend{Backend::Scalar};
Backend           g_op_backend[static_cast<usize>(OpId::Count)] = {};

// Rank a tier by preference. Higher = better.
int tier_rank(Backend b) noexcept {
    switch (b) {
        case Backend::AVX512: return 5;
        case Backend::AVX2:   return 4;
        case Backend::AVX:    return 3;
        case Backend::SSE42:  return 2;
        case Backend::Scalar: return 1;
    }
    return 0;
}

}  // namespace

const char* backend_name(Backend b) noexcept {
    switch (b) {
        case Backend::Scalar: return "scalar";
        case Backend::SSE42:  return "SSE4.2";
        case Backend::AVX:    return "AVX";
        case Backend::AVX2:   return "AVX2+FMA3";
        case Backend::AVX512: return "AVX-512";
    }
    return "(?)";
}

Backend active_backend() noexcept    { return g_active_backend; }
Backend backend_for(OpId op) noexcept {
    const usize idx = static_cast<usize>(op);
    if (idx >= static_cast<usize>(OpId::Count)) return Backend::Scalar;
    return g_op_backend[idx];
}

void init() {
    bool already = g_initialised.exchange(true, std::memory_order_acq_rel);
    if (already) return;

    const auto& cf = cardinal::hal::cpu_features();

    // Determine the best tier the CPU actually supports.
    Backend best = Backend::Scalar;
    if (cf.sse42)   best = Backend::SSE42;
    if (cf.avx)     best = Backend::AVX;
    if (cf.avx2 && cf.fma3) best = Backend::AVX2;
    if (cf.avx512f) best = Backend::AVX512;
    g_active_backend = best;

    // Bind each pointer to the corresponding tier's implementation.
    // Add tiers in ladder order so falls-back work (every tier we
    // assign overwrites the previous one when the CPU supports it).
    auto bind_tier = [&](Backend tier) {
        switch (tier) {
            case Backend::Scalar:
                vec_add_f32   = &scalar::vec_add_f32;
                vec_sub_f32   = &scalar::vec_sub_f32;
                vec_mul_f32   = &scalar::vec_mul_f32;
                vec_scale_f32 = &scalar::vec_scale_f32;
                vec_axpy_f32  = &scalar::vec_axpy_f32;
                dot_f32       = &scalar::dot_f32;
                sum_f32       = &scalar::sum_f32;
                min_f32       = &scalar::min_f32;
                max_f32       = &scalar::max_f32;
                transform_points_mat4        = &scalar::transform_points_mat4;
                vec3_cross_array             = &scalar::vec3_cross_array;
                vec3_length_array            = &scalar::vec3_length_array;
                vec3_normalize_array_inplace = &scalar::vec3_normalize_array_inplace;
                mat4_mul_array               = &scalar::mat4_mul_array;
                frustum_cull_spheres         = &scalar::frustum_cull_spheres;
                transform_aabb_array         = &scalar::transform_aabb_array;
                frustum_cull_aabbs           = &scalar::frustum_cull_aabbs;
                ray_aabb_intersect_array     = &scalar::ray_aabb_intersect_array;
                min_index_f32                = &scalar::min_index_f32;
                max_index_f32                = &scalar::max_index_f32;
                morton_encode_3d_array       = &scalar::morton_encode_3d_array;
                for (auto& b : g_op_backend) b = Backend::Scalar;
                return;
            case Backend::SSE42:
                vec_add_f32   = &sse42::vec_add_f32;
                vec_sub_f32   = &sse42::vec_sub_f32;
                vec_mul_f32   = &sse42::vec_mul_f32;
                vec_scale_f32 = &sse42::vec_scale_f32;
                vec_axpy_f32  = &sse42::vec_axpy_f32;
                dot_f32       = &sse42::dot_f32;
                sum_f32       = &sse42::sum_f32;
                min_f32       = &sse42::min_f32;
                max_f32       = &sse42::max_f32;
                transform_points_mat4        = &sse42::transform_points_mat4;
                vec3_cross_array             = &sse42::vec3_cross_array;
                vec3_length_array            = &sse42::vec3_length_array;
                vec3_normalize_array_inplace = &sse42::vec3_normalize_array_inplace;
                mat4_mul_array               = &sse42::mat4_mul_array;
                frustum_cull_spheres         = &sse42::frustum_cull_spheres;
                transform_aabb_array         = &sse42::transform_aabb_array;
                frustum_cull_aabbs           = &sse42::frustum_cull_aabbs;
                ray_aabb_intersect_array     = &sse42::ray_aabb_intersect_array;
                min_index_f32                = &sse42::min_index_f32;
                max_index_f32                = &sse42::max_index_f32;
                morton_encode_3d_array       = &sse42::morton_encode_3d_array;
                for (auto& b : g_op_backend) b = Backend::SSE42;
                return;
            case Backend::AVX:
                vec_add_f32   = &avx::vec_add_f32;
                vec_sub_f32   = &avx::vec_sub_f32;
                vec_mul_f32   = &avx::vec_mul_f32;
                vec_scale_f32 = &avx::vec_scale_f32;
                vec_axpy_f32  = &avx::vec_axpy_f32;
                dot_f32       = &avx::dot_f32;
                sum_f32       = &avx::sum_f32;
                min_f32       = &avx::min_f32;
                max_f32       = &avx::max_f32;
                transform_points_mat4        = &avx::transform_points_mat4;
                vec3_cross_array             = &avx::vec3_cross_array;
                vec3_length_array            = &avx::vec3_length_array;
                vec3_normalize_array_inplace = &avx::vec3_normalize_array_inplace;
                // mat4_mul_array INTENTIONALLY NOT REASSIGNED — falls
                // back to whatever the previous bind_tier(SSE42) set
                // (sse42::mat4_mul_array). Tracked separately in
                // g_op_backend so the diagnostic panel can show the
                // mixed-tier reality.
                frustum_cull_spheres         = &avx::frustum_cull_spheres;
                transform_aabb_array         = &avx::transform_aabb_array;
                frustum_cull_aabbs           = &avx::frustum_cull_aabbs;
                ray_aabb_intersect_array     = &avx::ray_aabb_intersect_array;
                min_index_f32                = &avx::min_index_f32;
                max_index_f32                = &avx::max_index_f32;
                morton_encode_3d_array       = &avx::morton_encode_3d_array;
                for (auto& b : g_op_backend) b = Backend::AVX;
                g_op_backend[static_cast<usize>(OpId::Mat4Mul_Array)] = Backend::SSE42;
                return;
            case Backend::AVX2:
                vec_add_f32   = &avx2::vec_add_f32;
                vec_sub_f32   = &avx2::vec_sub_f32;
                vec_mul_f32   = &avx2::vec_mul_f32;
                vec_scale_f32 = &avx2::vec_scale_f32;
                vec_axpy_f32  = &avx2::vec_axpy_f32;
                dot_f32       = &avx2::dot_f32;
                sum_f32       = &avx2::sum_f32;
                min_f32       = &avx2::min_f32;
                max_f32       = &avx2::max_f32;
                transform_points_mat4        = &avx2::transform_points_mat4;
                vec3_cross_array             = &avx2::vec3_cross_array;
                vec3_length_array            = &avx2::vec3_length_array;
                vec3_normalize_array_inplace = &avx2::vec3_normalize_array_inplace;
                mat4_mul_array               = &avx2::mat4_mul_array;
                frustum_cull_spheres         = &avx2::frustum_cull_spheres;
                transform_aabb_array         = &avx2::transform_aabb_array;
                frustum_cull_aabbs           = &avx2::frustum_cull_aabbs;
                ray_aabb_intersect_array     = &avx2::ray_aabb_intersect_array;
                min_index_f32                = &avx2::min_index_f32;
                max_index_f32                = &avx2::max_index_f32;
                morton_encode_3d_array       = &avx2::morton_encode_3d_array;
                for (auto& b : g_op_backend) b = Backend::AVX2;
                return;
            case Backend::AVX512:
                vec_add_f32   = &avx512::vec_add_f32;
                vec_sub_f32   = &avx512::vec_sub_f32;
                vec_mul_f32   = &avx512::vec_mul_f32;
                vec_scale_f32 = &avx512::vec_scale_f32;
                vec_axpy_f32  = &avx512::vec_axpy_f32;
                dot_f32       = &avx512::dot_f32;
                sum_f32       = &avx512::sum_f32;
                min_f32       = &avx512::min_f32;
                max_f32       = &avx512::max_f32;
                transform_points_mat4        = &avx512::transform_points_mat4;
                vec3_cross_array             = &avx512::vec3_cross_array;
                vec3_length_array            = &avx512::vec3_length_array;
                vec3_normalize_array_inplace = &avx512::vec3_normalize_array_inplace;
                mat4_mul_array               = &avx512::mat4_mul_array;
                frustum_cull_spheres         = &avx512::frustum_cull_spheres;
                transform_aabb_array         = &avx512::transform_aabb_array;
                frustum_cull_aabbs           = &avx512::frustum_cull_aabbs;
                ray_aabb_intersect_array     = &avx512::ray_aabb_intersect_array;
                min_index_f32                = &avx512::min_index_f32;
                max_index_f32                = &avx512::max_index_f32;
                morton_encode_3d_array       = &avx512::morton_encode_3d_array;
                for (auto& b : g_op_backend) b = Backend::AVX512;
                return;
        }
    };

    // Apply ladder bottom-up so unsupported tiers don't regress past
    // the best supported one. (Today every per-op binding moves
    // together — separate per-op binding lands when individual ISA TUs
    // start shipping a strict subset of ops.)
    bind_tier(Backend::Scalar);
    if (tier_rank(best) >= tier_rank(Backend::SSE42))  bind_tier(Backend::SSE42);
    if (tier_rank(best) >= tier_rank(Backend::AVX))    bind_tier(Backend::AVX);
    if (tier_rank(best) >= tier_rank(Backend::AVX2))   bind_tier(Backend::AVX2);
    if (tier_rank(best) >= tier_rank(Backend::AVX512)) bind_tier(Backend::AVX512);

    cardinal::log::infof("simd",
        "math dispatch active: %s  (CPU=%s vendor=%s SSE4.2=%s AVX=%s AVX2=%s "
        "FMA3=%s AVX-512F=%s SSE4A=%s)",
        backend_name(best),
        cf.brand[0]  ? cf.brand  : "(unknown)",
        cf.vendor[0] ? cf.vendor : "(unknown)",
        cf.sse42  ? "yes" : "no",
        cf.avx    ? "yes" : "no",
        cf.avx2   ? "yes" : "no",
        cf.fma3   ? "yes" : "no",
        cf.avx512f? "yes" : "no",
        cf.sse4a  ? "yes" : "no");
}

namespace {

// Auto-init at static-init time. Order matters less than the hard
// guard in init() — even if a tighter static-init race calls into a
// pointer before this fires, the pointers default to scalar.
struct AutoInit { AutoInit() noexcept { init(); } };
AutoInit g_auto_init{};

}  // namespace

}  // namespace cardinal::core::simd
