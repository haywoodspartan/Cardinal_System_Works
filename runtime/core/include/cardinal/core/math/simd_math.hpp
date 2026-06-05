#pragma once

// =============================================================================
// Cardinal — SIMD-dispatched math.
//
// One header, one compile-time API, one runtime-picked backend. The
// engine calls e.g. cardinal::core::simd::dot_f32(a, b, n) and the
// best implementation the host CPU supports executes — AVX-512 on a
// Zen 5, AVX2 on a Skylake, SSE4.2 on a 2010-era box, and a portable
// scalar fallback when nothing better exists.
//
// Design — function-pointer table:
//
//   - Each operation is a function-pointer global, declared here and
//     defined in simd_math_dispatch.cpp.
//   - At static-init the dispatcher reads cardinal::hal::cpu_features()
//     and binds each pointer to the highest-tier implementation the CPU
//     actually supports. Detection runs once (CARDINAL_READ_MOSTLY).
//   - Per-ISA implementations live in dedicated TUs compiled with
//     /arch:AVX, /arch:AVX2, /arch:AVX512 (etc.) so the compiler can
//     emit those instructions safely without polluting the rest of
//     the codebase.
//
// Why function pointers (vs. virtual / vs. templates):
//   - One indirect call per op (~ns), amortised over batch sizes that
//     are typically 100s to millions of elements.
//   - No vtable per-instance overhead; the call site is just a load +
//     indirect call, which the branch predictor handles trivially.
//   - Keeps the public API as a flat free-function namespace (no
//     "first construct a backend object" ceremony at every site).
//
// Tier ladder (highest preferred):
//
//      AVX-512  →  AVX2  →  AVX  →  SSE4.2  →  scalar
//
// Tier picked per-OP — if we don't have an AVX-512 implementation of a
// specific function, the dispatcher falls down the ladder for that
// pointer until it lands somewhere implemented. So adding a new ISA
// version of one function doesn't require shipping all five.
//
// Conventions:
//   - All pointer arguments are dense f32 / Vec3 / Vec4 arrays —
//     no stride parameter (stride-aware variants land if needed).
//   - Output buffers may alias inputs unless documented otherwise.
//   - Length 0 is a valid no-op for every operation.
//   - Alignment: pointers do NOT need to be aligned. Implementations
//     handle the prologue scalar tail; the SIMD core runs only over
//     the aligned span.
// =============================================================================

#include <cardinal/core/types.hpp>

namespace cardinal::core::simd {

// ---------------------------------------------------------------------------
// Backend identification — exposed for diagnostics + the Studio panel.
// ---------------------------------------------------------------------------
enum class Backend : u32 {
    Scalar     = 0,
    SSE42      = 1,
    AVX        = 2,
    AVX2       = 3,
    AVX512     = 4,
};
const char* backend_name(Backend b) noexcept;

// The backend that was picked at static init for the BULK of operations
// (each function-pointer may have fallen back independently — see notes
// below).
Backend active_backend() noexcept;

// Per-operation backend (introspection for the Studio panel — "which
// version of dot_f32 is bound"). Returns the ladder tier the dispatcher
// landed on for THIS op.
enum class OpId : u32 {
    VecAdd_f32,
    VecSub_f32,
    VecMul_f32,
    VecScale_f32,
    VecAxpy_f32,
    Dot_f32,
    Sum_f32,
    Min_f32,
    Max_f32,
    TransformPointsMat4,
    Vec3Cross_Array,
    Vec3Length_Array,
    Vec3Normalize_ArrayInplace,
    Mat4Mul_Array,
    FrustumCullSpheres_f32,
    TransformAabbArray,
    FrustumCullAabbs_f32,
    RayAabbIntersect_f32,
    MinIndex_f32,
    MaxIndex_f32,
    MortonEncode3d_u32,
    Count,
};
Backend backend_for(OpId op) noexcept;

// ---------------------------------------------------------------------------
// BLAS-style 1-D vector ops
//
// Pattern: out[i] = f(a[i], b[i] / scalar). Length n in elements.
// ---------------------------------------------------------------------------
extern void  (*vec_add_f32)  (f32* out, const f32* a, const f32* b, usize n);
extern void  (*vec_sub_f32)  (f32* out, const f32* a, const f32* b, usize n);
extern void  (*vec_mul_f32)  (f32* out, const f32* a, const f32* b, usize n);
extern void  (*vec_scale_f32)(f32* out, const f32* a, f32 s, usize n);

// y[i] += a * x[i]  — BLAS daxpy/saxpy.
// Used heavily by physics (force += accel * dt) and renderer
// (clip += basis * weight).
extern void  (*vec_axpy_f32) (f32* y, f32 a, const f32* x, usize n);

// ---------------------------------------------------------------------------
// Reductions — return a scalar over a span.
// ---------------------------------------------------------------------------
extern f32   (*dot_f32)(const f32* a, const f32* b, usize n);
extern f32   (*sum_f32)(const f32* a, usize n);
extern f32   (*min_f32)(const f32* a, usize n);
extern f32   (*max_f32)(const f32* a, usize n);

// ---------------------------------------------------------------------------
// Geometry — 4×3 transform of a flat array of float-triples (positions).
//
// Reads `count` Vec3-positions from `in` (12 bytes each, contiguous)
// and writes `count` transformed Vec3s into `out`. M is row-major
// 4×4 stored column-major in cardinal::core::Mat4 (m[col][row]); the
// implementation handles the indexing internally.
// ---------------------------------------------------------------------------
struct Mat4Compact {     // forward-friendly POD wrapping the 16 floats
    f32 m[16];           // column-major
};
extern void (*transform_points_mat4)(f32*       out_xyz,
                                     const f32* in_xyz,
                                     const Mat4Compact& M,
                                     usize      count);

// ---------------------------------------------------------------------------
// 3D vector batch ops — operate on AoS arrays of {x, y, z} triples.
// Input layouts: in_xyz / a_xyz / b_xyz are float-arrays of length
// count*3 in the obvious interleaved order. Output layouts: out_xyz is
// the same shape; out_lens is one float per input vector.
// ---------------------------------------------------------------------------

// out[i] = a[i] × b[i]  (cross product). Used by physics for torque
// (r × F) and broad-phase frustum-plane derivation.
extern void (*vec3_cross_array)(f32*       out_xyz,
                                const f32* a_xyz,
                                const f32* b_xyz,
                                usize      count);

// out_lens[i] = sqrt(in_xyz[i].x² + .y² + .z²). Distance / radius
// checks across batches of points.
extern void (*vec3_length_array)(f32*       out_lens,
                                 const f32* in_xyz,
                                 usize      count);

// io_xyz[i] = io_xyz[i] / max(|io_xyz[i]|, eps).  In-place to avoid
// the allocation cost when the caller only wants normals cleaned up.
// Zero-length input vectors stay zero (we don't divide by eps and
// accidentally synthesise a fake direction).
extern void (*vec3_normalize_array_inplace)(f32*  io_xyz,
                                            usize count);

// ---------------------------------------------------------------------------
// Matrix batch ops — column-major 4×4, packed contiguous as 16 floats per
// matrix. out[i] = A[i] × B[i] for each pair in the input arrays.
//
// Use cases (bandwidth-amortising matters):
//   - Skinning palette × inverse-bind matrices for N bones
//   - Per-frame world-matrix update for N entities (vp × model)
//   - Animation blending (final = blend × delta)
//
// Per-op fallback note: the AVX tier doesn't ship a dedicated
// implementation here — without FMA3 the inner-loop savings over the
// SSE4.2 4-wide column path are tiny. The dispatcher's tier walk
// leaves AVX bound to the SSE4.2 implementation, AVX2 jumps to its
// FMA3-fused version. This is the per-op fallback design exercised
// for the first time — future ISA TUs can ship a strict subset of
// ops and unsupported ops gracefully bind to whatever the next-lower
// tier ships.
extern void (*mat4_mul_array)(f32*       out_matrices,
                              const f32* A_matrices,
                              const f32* B_matrices,
                              usize      count);

// ---------------------------------------------------------------------------
// Frustum culling — bulk plane-sphere test.
//
// Tests `count` bounding spheres against a 6-plane frustum, writing a
// packed visibility bitmask to `out_bits`. Bit i of byte i/8 is set
// iff sphere i is fully or partially inside the frustum (i.e., not
// rejected by any of the 6 outward-facing planes).
//
// Inputs are SoA — caller builds parallel arrays of sphere centers
// (cx, cy, cz) and radii (r). For each sphere s:
//
//   visible = AND over p in [0..5] of (planes[p].n · sc + planes[p].d) >= -sr
//
// Plane layout: 24 floats { p0.nx, p0.ny, p0.nz, p0.d,
//                           p1.nx, p1.ny, p1.nz, p1.d,
//                           ... }
// — matches cardinal::Frustum::planes' in-memory layout (six contiguous
// {Vec3 normal; f32 d;} structs), so a renderer with a Frustum can
// pass &frustum.planes[0].normal.x directly.
//
// Output: out_bits must be at least (count + 7) / 8 bytes. The kernel
// writes whole bytes for the SIMD-batch portion and a partial byte for
// any remainder; bits past `count` in the last byte are zero. Call
// site does not need to pre-zero.
//
// Used by: per-entity broad-phase culling (renderer, partition,
// nav-mesh visibility cone). 16 spheres / iter on AVX-512 — that's
// the architectural payoff over the scalar Frustum::intersects test
// in cardinal::core::geom.
extern void (*frustum_cull_spheres)(u8*        out_bits,
                                    const f32* planes,
                                    const f32* cx,
                                    const f32* cy,
                                    const f32* cz,
                                    const f32* r,
                                    usize      count);

// ---------------------------------------------------------------------------
// AABB transform — Mat4 × N axis-aligned bounding boxes → world-space AABBs.
//
// SoA inputs: parallel arrays of {min,max} components — six total. Output
// is the same SoA shape. In-place is safe ONLY if all 12 pointers are
// distinct (output AABB construction reads multiple components before
// writing each).
//
// Algorithm: Arvo's incremental method — 9 mul + 18 min/max + 9 add per
// box, vs the naive 8-corner brute force (96 muls + 96 maxes). The
// matrix M is read once; per box we walk 3 input axes × 3 output axes.
//
// Used by: per-entity world AABB compute that feeds frustum_cull_aabbs
// (tighter than the bounding-sphere reject this replaces / augments).
extern void (*transform_aabb_array)(
    f32*       out_min_x, f32* out_min_y, f32* out_min_z,
    f32*       out_max_x, f32* out_max_y, f32* out_max_z,
    const f32* in_min_x,  const f32* in_min_y, const f32* in_min_z,
    const f32* in_max_x,  const f32* in_max_y, const f32* in_max_z,
    const Mat4Compact& M,
    usize      count);

// ---------------------------------------------------------------------------
// Frustum-vs-AABB cull — tighter than the sphere variant.
//
// For each plane p in 6, the "p-corner" (most-positive-corner along the
// plane normal) of an AABB is:
//   p_i = (planes[p].normal[i] >= 0) ? max[i] : min[i]    for i in {x,y,z}
// AABB is OUTSIDE the plane iff signed_dist(p_corner, plane) < 0.
//
// Optimisation: plane normals are scalars per-plane, so the per-axis
// "select max or min array" is a SCALAR pointer choice OUTSIDE the
// per-AABB inner loop — no per-lane blends, just FMAs on the chosen
// SoA arrays. 16 AABBs / iter on AVX-512.
//
// Plane layout matches frustum_cull_spheres: 24 floats { nx, ny, nz, d } × 6
// (i.e. cardinal::core::geom::Frustum::planes' in-memory layout).
extern void (*frustum_cull_aabbs)(u8*        out_bits,
                                  const f32* planes,
                                  const f32* min_x, const f32* min_y, const f32* min_z,
                                  const f32* max_x, const f32* max_y, const f32* max_z,
                                  usize      count);

// ---------------------------------------------------------------------------
// Ray-vs-AABB batch intersection (slab method) — one ray, N AABBs.
//
// Computes the parametric distance `t` along the ray (hit_point =
// origin + t * direction) at which the ray enters each AABB. Misses
// produce +INF so the caller can do a vectorised reduce-min to find
// the closest hit without a special "is hit" mask.
//
// Caller passes inv_dir (1 / direction) explicitly — the per-ray
// reciprocal is constant across batches against the same ray, and
// skipping it inside the inner loop saves N divisions. For axis-
// parallel rays where some `dir` component is exactly 0, callers
// should pass +INF or -INF for that lane (the slab math then
// degenerates correctly: the ray won't hit a slab it's parallel to
// unless its origin lies between min and max for that axis, which
// produces t_near = -INF and t_far = +INF as expected).
//
// SoA inputs: 6 parallel arrays for the AABBs' min/max per axis.
//
// If the ray's origin is inside an AABB, the kernel returns 0 (we
// clamp t_enter to 0 before the hit test) — same convention as a
// continuous-collision query.
//
// Used by: scene::pick_entity (currently scalar sphere test), audio
// occlusion ray casts, AI line-of-sight, future CPU BVH traversal,
// particle collision broad phase. AVX-512 path: 16 AABBs / iter.
extern void (*ray_aabb_intersect_array)(
    f32*       out_t,
    f32        ox, f32 oy, f32 oz,                // ray origin
    f32        inv_dx, f32 inv_dy, f32 inv_dz,    // 1 / direction
    const f32* min_x, const f32* min_y, const f32* min_z,
    const f32* max_x, const f32* max_y, const f32* max_z,
    usize      count);

// ---------------------------------------------------------------------------
// argmin reduction — find the smallest value AND its source index in
// a single pass over the input array.
//
// On ties, returns the FIRST occurrence (matches the convention of
// the scalar `for (i; in[i] < best; ...)` loop).
//
// Output:
//   *out_value = min(in[0..count))
//   *out_index = the i for which in[i] == *out_value (smallest such i)
// For count == 0: *out_value = +INF, *out_index = 0 (sentinel; caller
// can detect "no input" by checking value == INF).
//
// SIMD strategy: maintain a parallel (cur_min, cur_idx) pair of
// vectors during the bulk scan, conditionally update both via the
// per-lane "is smaller" mask. Final horizontal reduce picks the
// smallest across lanes. AVX-512 uses native __mmask16 +
// _mm512_mask_blend_epi32 to update indices without a float round-trip.
//
// Used by: scene::pick_entity (closest-hit reduction after the SIMD
// ray_aabb_intersect_array pass), future "closest enemy" / "lowest LOD"
// queries.
extern void (*min_index_f32)(f32* out_value, u32* out_index,
                             const f32* in, usize count);

// argmax companion — same pattern as min_index_f32 but find the
// LARGEST value + its source index. On ties, returns the FIRST
// occurrence (matches scalar `>` strict-greater semantics). For
// count == 0: *out_value = -INF, *out_index = 0.
extern void (*max_index_f32)(f32* out_value, u32* out_index,
                             const f32* in, usize count);

// ---------------------------------------------------------------------------
// 3D Morton code (Z-curve) — interleave the bits of (x, y, z) integer
// coordinates so spatially-near cells land near each other in 1D index
// space. Each input coord uses up to 21 bits (input bits beyond bit 20
// are silently truncated by the bit-spread mask); output is u64 with
// bits packed as ... z2 y2 x2 z1 y1 x1 z0 y0 x0.
//
// Bit-spread sequence per coord (5 shifts + masks per element); same
// pattern across all tiers, just widened to vector lanes.
//
// Used by: spatial-hash broad-phase, BVH construction (sort primitives
// by Morton), GPU-streaming sort orders, voxel chunk addressing.
extern void (*morton_encode_3d_array)(u64*       out_codes,
                                      const u32* x,
                                      const u32* y,
                                      const u32* z,
                                      usize      count);

// ---------------------------------------------------------------------------
// One-time initialisation (called automatically at static init —
// callable explicitly for testing or after CPU governor changes).
// ---------------------------------------------------------------------------
void init();

}  // namespace cardinal::core::simd
