// =============================================================================
// cardinal_intrinsics.hlsli — Shader Model 6+ optimisation helpers.
//
// Compiled with DXC against `*_6_5` profiles (Vulkan SPIR-V or DX12 DXIL).
// Helpers gather:
//
//   * FP16 (Rapid Packed Math) — promote float→half→float around hot ops
//   * Wave intrinsics          — reductions, ballots, lane queries
//   * Packed-data helpers      — encode normals / tangents into 32 bits
//   * Quad / divergence helpers — branch-free min/max, fused multiply-add
//
// Engine-side gating: cardinal::render::FeatureGate decides whether the
// pipeline using FP16 / wave-ops should be compiled or the baseline form.
// On hardware that lacks support, these helpers degrade to the float path
// rather than emitting an unsupported instruction.
//
// Design note: every helper returns the same type it consumed so call
// sites can drop the wrapper in without rewriting their dataflow.
// =============================================================================

#ifndef CARDINAL_INTRINSICS_HLSLI
#define CARDINAL_INTRINSICS_HLSLI

// ---------------------------------------------------------------------------
// FP16 / Rapid Packed Math
// ---------------------------------------------------------------------------
// On GPUs with native FP16 (RDNA, Ada, Arc) min16float doubles math
// throughput on supporting ops (add/mul/fma/dot). Use these around
// performance-critical inner loops where 16-bit precision is enough
// (lighting accumulation, post-process kernels, shadow filtering).
//
// Compile with -enable-16bit-types in DXC for this to actually produce
// FP16 instructions; otherwise it falls through to float.

#ifndef CARDINAL_USE_FP16
    #define CARDINAL_USE_FP16 1
#endif

#if CARDINAL_USE_FP16
    typedef min16float       cf16;
    typedef min16float2      cf16x2;
    typedef min16float3      cf16x3;
    typedef min16float4      cf16x4;
#else
    typedef float            cf16;
    typedef float2           cf16x2;
    typedef float3           cf16x3;
    typedef float4           cf16x4;
#endif

cf16   to_h(float v)            { return (cf16)v;       }
cf16x3 to_h3(float3 v)          { return (cf16x3)v;     }
float  from_h(cf16 v)           { return (float)v;      }
float3 from_h3(cf16x3 v)        { return (float3)v;     }

// FP16-aware fused multiply-add. NB: many shader compilers will fuse
// (a*b + c) automatically; this is for places where you want to be
// explicit about staying in FP16 land.
cf16   cardinal_fma_h (cf16   a, cf16   b, cf16   c) { return mad(a, b, c); }
cf16x3 cardinal_fma_h3(cf16x3 a, cf16x3 b, cf16x3 c) { return mad(a, b, c); }

// ---------------------------------------------------------------------------
// Wave intrinsics — SM 6.0+. Available on every modern dGPU; the engine's
// FeatureGate exposes a "Wave intrinsics" flag.
// ---------------------------------------------------------------------------
// Wave-wide reductions of a single-channel value. Useful for compaction,
// reduction passes, deferred binning. The wave is a hardware concept (32
// or 64 lanes); these intrinsics return a value identical across all lanes.
float  cardinal_wave_sum_f (float x)  { return WaveActiveSum(x);  }
float3 cardinal_wave_sum_f3(float3 x) {
    return float3(WaveActiveSum(x.x), WaveActiveSum(x.y), WaveActiveSum(x.z));
}
float  cardinal_wave_max_f (float x)  { return WaveActiveMax(x);  }
float  cardinal_wave_min_f (float x)  { return WaveActiveMin(x);  }

// Branchless ballot: returns the lane index of the first lane in the wave
// for which `mask` is true. Useful for "first lane writes the result"
// patterns. Returns 0 if no lane is active (caller-supplied invariant).
uint cardinal_wave_first_active_lane(bool mask) {
    return firstbitlow(WaveActiveBallot(mask).x);
}

// "Scalarise" — when every lane in the wave reads the same value through
// a divergent index, replace the per-lane gather with a single broadcast.
// Use ONLY when the caller can prove every lane reads the same index.
uint cardinal_wave_broadcast_first(uint v) {
    return WaveReadLaneFirst(v);
}

// ---------------------------------------------------------------------------
// Packed-data helpers — pack 3-channel data into 32 bits to halve memory
// bandwidth for normal maps, tangent frames, low-precision colours.
// ---------------------------------------------------------------------------
// Octahedral encoding — best general-purpose normal compression. Two-channel
// output in [-1, 1]; pack into snorm16 or unorm16x2 for 32-bit storage.
float2 cardinal_pack_normal_octa(float3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0f) {
        float2 abs_xy = abs(n.xy);
        n.xy = (1.0f - abs_xy.yx) * sign(n.xy);
    }
    return n.xy * 0.5f + 0.5f;
}

float3 cardinal_unpack_normal_octa(float2 e) {
    e = e * 2.0f - 1.0f;
    float3 n = float3(e.xy, 1.0f - abs(e.x) - abs(e.y));
    float t = saturate(-n.z);
    n.xy += n.xy >= 0.0f ? -t : t;
    return normalize(n);
}

// Pack RGBA8 into a uint, suitable for a uint vertex attribute that costs
// 4 bytes instead of 16. Caller scales/offsets the channels first.
uint cardinal_pack_rgba8(float4 c) {
    uint4 q = uint4(saturate(c) * 255.0f);
    return (q.r) | (q.g << 8) | (q.b << 16) | (q.a << 24);
}
float4 cardinal_unpack_rgba8(uint p) {
    return float4(
        ((p)       & 0xFFu),
        ((p >> 8)  & 0xFFu),
        ((p >> 16) & 0xFFu),
        ((p >> 24) & 0xFFu)) * (1.0f / 255.0f);
}

// ---------------------------------------------------------------------------
// Branch-free min/max — keep the SIMD lanes coherent.
// ---------------------------------------------------------------------------
float  cardinal_select_f (bool c, float  a, float  b) { return c ? a : b; }
float3 cardinal_select_f3(bool c, float3 a, float3 b) { return c ? a : b; }

#endif  // CARDINAL_INTRINSICS_HLSLI
