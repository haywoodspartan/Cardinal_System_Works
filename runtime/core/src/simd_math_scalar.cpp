// =============================================================================
// Cardinal — SIMD math: scalar baseline.
//
// Pure-C++ implementations. Always compiled, always present — provides
// the floor of the dispatch ladder so we ALWAYS have something to bind.
// Compiles for any architecture without ISA-specific flags.
// =============================================================================
#define CARDINAL_SIMD_TIER scalar
#include "simd_math_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cardinal::core::simd::scalar {

void vec_add_f32(f32* out, const f32* a, const f32* b, usize n) {
    for (usize i = 0; i < n; ++i) out[i] = a[i] + b[i];
}
void vec_sub_f32(f32* out, const f32* a, const f32* b, usize n) {
    for (usize i = 0; i < n; ++i) out[i] = a[i] - b[i];
}
void vec_mul_f32(f32* out, const f32* a, const f32* b, usize n) {
    for (usize i = 0; i < n; ++i) out[i] = a[i] * b[i];
}
void vec_scale_f32(f32* out, const f32* a, f32 s, usize n) {
    for (usize i = 0; i < n; ++i) out[i] = a[i] * s;
}
void vec_axpy_f32(f32* y, f32 a, const f32* x, usize n) {
    for (usize i = 0; i < n; ++i) y[i] += a * x[i];
}

f32 dot_f32(const f32* a, const f32* b, usize n) {
    f32 acc = 0.0f;
    for (usize i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
}
f32 sum_f32(const f32* a, usize n) {
    f32 acc = 0.0f;
    for (usize i = 0; i < n; ++i) acc += a[i];
    return acc;
}
f32 min_f32(const f32* a, usize n) {
    if (n == 0) return std::numeric_limits<f32>::infinity();
    f32 m = a[0];
    for (usize i = 1; i < n; ++i) m = std::min(m, a[i]);
    return m;
}
f32 max_f32(const f32* a, usize n) {
    if (n == 0) return -std::numeric_limits<f32>::infinity();
    f32 m = a[0];
    for (usize i = 1; i < n; ++i) m = std::max(m, a[i]);
    return m;
}

// 4×3 affine transform of N positions. Reads/writes contiguous float-
// triples; M is a column-major 4×4. We compute (x', y', z') = M * (x, y, z, 1).
void transform_points_mat4(f32* out_xyz, const f32* in_xyz,
                           const Mat4Compact& M, usize count)
{
    const f32* m = M.m;
    for (usize i = 0; i < count; ++i) {
        const f32 x = in_xyz[i*3 + 0];
        const f32 y = in_xyz[i*3 + 1];
        const f32 z = in_xyz[i*3 + 2];
        out_xyz[i*3 + 0] = m[0]*x + m[4]*y + m[8] *z + m[12];
        out_xyz[i*3 + 1] = m[1]*x + m[5]*y + m[9] *z + m[13];
        out_xyz[i*3 + 2] = m[2]*x + m[6]*y + m[10]*z + m[14];
    }
}

void vec3_cross_array(f32* out_xyz, const f32* a_xyz, const f32* b_xyz, usize count) {
    for (usize i = 0; i < count; ++i) {
        const f32 ax = a_xyz[i*3+0], ay = a_xyz[i*3+1], az = a_xyz[i*3+2];
        const f32 bx = b_xyz[i*3+0], by = b_xyz[i*3+1], bz = b_xyz[i*3+2];
        out_xyz[i*3+0] = ay*bz - az*by;
        out_xyz[i*3+1] = az*bx - ax*bz;
        out_xyz[i*3+2] = ax*by - ay*bx;
    }
}

void vec3_length_array(f32* out_lens, const f32* in_xyz, usize count) {
    for (usize i = 0; i < count; ++i) {
        const f32 x = in_xyz[i*3+0], y = in_xyz[i*3+1], z = in_xyz[i*3+2];
        out_lens[i] = std::sqrt(x*x + y*y + z*z);
    }
}

void vec3_normalize_array_inplace(f32* io_xyz, usize count) {
    constexpr f32 eps = 1e-8f;
    for (usize i = 0; i < count; ++i) {
        const f32 x = io_xyz[i*3+0], y = io_xyz[i*3+1], z = io_xyz[i*3+2];
        const f32 len2 = x*x + y*y + z*z;
        if (len2 < eps) continue;        // leave zero/tiny vectors alone
        const f32 inv = 1.0f / std::sqrt(len2);
        io_xyz[i*3+0] = x * inv;
        io_xyz[i*3+1] = y * inv;
        io_xyz[i*3+2] = z * inv;
    }
}

// out[i] = A[i] × B[i]   (column-major 4x4 matrix multiply)
// Element layout: m[col*4 + row].
void mat4_mul_array(f32* out, const f32* A, const f32* B, usize count) {
    for (usize i = 0; i < count; ++i) {
        const f32* a = A + i * 16;
        const f32* b = B + i * 16;
        f32*       o = out + i * 16;
        for (int c = 0; c < 4; ++c) {
            const f32 b0 = b[c*4 + 0];
            const f32 b1 = b[c*4 + 1];
            const f32 b2 = b[c*4 + 2];
            const f32 b3 = b[c*4 + 3];
            for (int r = 0; r < 4; ++r) {
                o[c*4 + r] = a[0*4 + r]*b0 + a[1*4 + r]*b1
                           + a[2*4 + r]*b2 + a[3*4 + r]*b3;
            }
        }
    }
}

// Frustum-vs-sphere bulk cull. For each sphere, reject if any of the 6
// outward-facing planes has signed-distance < -radius. Output is a
// packed bitmask: bit i of out_bits[i/8] = 1 iff sphere i is visible.
//
// Pre-zeros the output bytes it touches so the caller doesn't have
// to. Process 8 spheres at a time so each pass writes one whole byte
// (matches the SIMD tiers' byte-aligned output cadence).
void frustum_cull_spheres(u8* out_bits, const f32* planes,
                          const f32* cx, const f32* cy,
                          const f32* cz, const f32* r,
                          usize count)
{
    auto test_one = [&](usize i) -> bool {
        const f32 x = cx[i], y = cy[i], z = cz[i], rr = r[i];
        for (int p = 0; p < 6; ++p) {
            const f32 nx = planes[p*4 + 0];
            const f32 ny = planes[p*4 + 1];
            const f32 nz = planes[p*4 + 2];
            const f32 d  = planes[p*4 + 3];
            const f32 dist = nx*x + ny*y + nz*z + d;
            if (dist + rr < 0.0f) return false;
        }
        return true;
    };
    usize i = 0;
    for (; i + 8 <= count; i += 8) {
        u8 byte = 0;
        for (int b = 0; b < 8; ++b) {
            if (test_one(i + b)) byte |= static_cast<u8>(1u << b);
        }
        out_bits[i >> 3] = byte;
    }
    if (i < count) {
        u8 byte = 0;
        for (usize b = 0; i + b < count; ++b) {
            if (test_one(i + b)) byte |= static_cast<u8>(1u << b);
        }
        out_bits[i >> 3] = byte;
    }
}

// AABB transform — Arvo's incremental method. For each input axis i,
// take its column of M (mi0, mi1, mi2) and its min/max projection
// against in_min[i] / in_max[i]; the smaller pair contribution lands
// in new_min, the larger in new_max. Translation is M[3].
void transform_aabb_array(
    f32* out_min_x, f32* out_min_y, f32* out_min_z,
    f32* out_max_x, f32* out_max_y, f32* out_max_z,
    const f32* in_min_x, const f32* in_min_y, const f32* in_min_z,
    const f32* in_max_x, const f32* in_max_y, const f32* in_max_z,
    const Mat4Compact& M, usize count)
{
    // Column-major Mat4: M.m[col*4 + row]. Translation = M.m[12..14].
    const f32 m00 = M.m[ 0], m01 = M.m[ 1], m02 = M.m[ 2];
    const f32 m10 = M.m[ 4], m11 = M.m[ 5], m12 = M.m[ 6];
    const f32 m20 = M.m[ 8], m21 = M.m[ 9], m22 = M.m[10];
    const f32 t0  = M.m[12], t1  = M.m[13], t2  = M.m[14];
    for (usize i = 0; i < count; ++i) {
        const f32 nx = in_min_x[i], ny = in_min_y[i], nz = in_min_z[i];
        const f32 px = in_max_x[i], py = in_max_y[i], pz = in_max_z[i];
        // Per output axis: sum the min/max of (m·n, m·p) along each input.
        f32 oxn = t0, oxp = t0;
        { f32 a = m00*nx, b = m00*px; oxn += std::min(a,b); oxp += std::max(a,b); }
        { f32 a = m10*ny, b = m10*py; oxn += std::min(a,b); oxp += std::max(a,b); }
        { f32 a = m20*nz, b = m20*pz; oxn += std::min(a,b); oxp += std::max(a,b); }
        f32 oyn = t1, oyp = t1;
        { f32 a = m01*nx, b = m01*px; oyn += std::min(a,b); oyp += std::max(a,b); }
        { f32 a = m11*ny, b = m11*py; oyn += std::min(a,b); oyp += std::max(a,b); }
        { f32 a = m21*nz, b = m21*pz; oyn += std::min(a,b); oyp += std::max(a,b); }
        f32 ozn = t2, ozp = t2;
        { f32 a = m02*nx, b = m02*px; ozn += std::min(a,b); ozp += std::max(a,b); }
        { f32 a = m12*ny, b = m12*py; ozn += std::min(a,b); ozp += std::max(a,b); }
        { f32 a = m22*nz, b = m22*pz; ozn += std::min(a,b); ozp += std::max(a,b); }
        out_min_x[i] = oxn; out_max_x[i] = oxp;
        out_min_y[i] = oyn; out_max_y[i] = oyp;
        out_min_z[i] = ozn; out_max_z[i] = ozp;
    }
}

// Frustum-vs-AABB cull — "p-vertex" trick. Per plane, pick the SoA
// array (max or min) per axis based on the plane normal sign. Then
// the inner test is just nx*p_x + ny*p_y + nz*p_z + d >= 0.
void frustum_cull_aabbs(u8* out_bits, const f32* planes,
                        const f32* min_x, const f32* min_y, const f32* min_z,
                        const f32* max_x, const f32* max_y, const f32* max_z,
                        usize count)
{
    // Pre-compute per-plane "p-corner" array selectors. Done outside
    // the per-AABB loop because plane normals are constants.
    struct PlaneSel {
        f32 nx, ny, nz, d;
        const f32 *px, *py, *pz;
    };
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

    auto test_one = [&](usize i) -> bool {
        for (int p = 0; p < 6; ++p) {
            const auto& s = sels[p];
            const f32 dist = s.nx * s.px[i] + s.ny * s.py[i] + s.nz * s.pz[i] + s.d;
            if (dist < 0.0f) return false;
        }
        return true;
    };

    usize i = 0;
    for (; i + 8 <= count; i += 8) {
        u8 byte = 0;
        for (int b = 0; b < 8; ++b) if (test_one(i + b)) byte |= static_cast<u8>(1u << b);
        out_bits[i >> 3] = byte;
    }
    if (i < count) {
        u8 byte = 0;
        for (usize b = 0; i + b < count; ++b) {
            if (test_one(i + b)) byte |= static_cast<u8>(1u << b);
        }
        out_bits[i >> 3] = byte;
    }
}

// 3D Morton encoding (Z-curve) — bit-spread the lower 21 bits of each
// of (x, y, z) so output bit 3*i = source bit i. Standard 5-shift
// sequence; portable across all 5 tiers (just widens to SIMD lanes).
namespace {
inline u64 spread21_bits(u32 v) noexcept {
    u64 r = static_cast<u64>(v) & 0x1fffffull;   // mask to 21 bits
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
    for (usize i = 0; i < count; ++i) {
        out_codes[i] = spread21_bits(x[i])
                    | (spread21_bits(y[i]) << 1)
                    | (spread21_bits(z[i]) << 2);
    }
}

// Find the minimum value AND its index in a single pass. On ties,
// the FIRST occurrence wins (matches scalar `<` strict-less semantics).
// count == 0 returns (+INF, 0).
void min_index_f32(f32* out_value, u32* out_index,
                   const f32* in, usize count)
{
    if (count == 0) {
        *out_value = std::numeric_limits<f32>::infinity();
        *out_index = 0;
        return;
    }
    f32 best   = in[0];
    u32 best_i = 0;
    for (usize i = 1; i < count; ++i) {
        if (in[i] < best) {
            best   = in[i];
            best_i = static_cast<u32>(i);
        }
    }
    *out_value = best;
    *out_index = best_i;
}

// Find the maximum value AND its index in a single pass. Mirror of
// min_index_f32 — FIRST occurrence wins on ties; count == 0 returns
// (-INF, 0).
void max_index_f32(f32* out_value, u32* out_index,
                   const f32* in, usize count)
{
    if (count == 0) {
        *out_value = -std::numeric_limits<f32>::infinity();
        *out_index = 0;
        return;
    }
    f32 best   = in[0];
    u32 best_i = 0;
    for (usize i = 1; i < count; ++i) {
        if (in[i] > best) {
            best   = in[i];
            best_i = static_cast<u32>(i);
        }
    }
    *out_value = best;
    *out_index = best_i;
}

// Ray-vs-AABB batch (slab method) — per AABB compute t_enter as the
// max of per-axis near distances, t_exit as the min of per-axis far
// distances. Ray hits iff t_exit >= max(t_enter, 0); we return that
// clamped t_enter (so origin-inside-AABB returns 0), or +INF for misses.
void ray_aabb_intersect_array(
    f32* out_t,
    f32 ox, f32 oy, f32 oz,
    f32 inv_dx, f32 inv_dy, f32 inv_dz,
    const f32* min_x, const f32* min_y, const f32* min_z,
    const f32* max_x, const f32* max_y, const f32* max_z,
    usize count)
{
    const f32 INF = std::numeric_limits<f32>::infinity();
    for (usize i = 0; i < count; ++i) {
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

}  // namespace cardinal::core::simd::scalar
