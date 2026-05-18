// =============================================================================
// cardinal_filters.hlsli — mip / down-sample kernels.
//
// Each function takes 4 RGB samples (typically a 2x2 footprint) and
// returns the filtered colour. Same coefficients as the CPU reference
// implementations in runtime/render/src/algos.cpp.
//
// Usage in a compute shader generating mip N+1 from mip N:
//
//     #include "cardinal_filters.hlsli"
//     float3 a = source.Load(int3(2*tid+0,        ...));
//     float3 b = source.Load(int3(2*tid+int2(1,0), ...));
//     float3 c = source.Load(int3(2*tid+int2(0,1), ...));
//     float3 d = source.Load(int3(2*tid+int2(1,1), ...));
//     dest[tid] = float4(cardinal_mip_kaiser(a, b, c, d), 1);
// =============================================================================

#ifndef CARDINAL_FILTERS_HLSLI
#define CARDINAL_FILTERS_HLSLI

float3 cardinal_mip_box(float3 a, float3 b, float3 c, float3 d) {
    return 0.25 * (a + b + c + d);
}
float3 cardinal_mip_tent(float3 a, float3 b, float3 c, float3 d) {
    return 0.4 * a + 0.2 * b + 0.2 * c + 0.2 * d;
}
float3 cardinal_mip_kaiser(float3 a, float3 b, float3 c, float3 d) {
    return 0.45 * a + 0.20 * b + 0.20 * c + 0.15 * d;
}
float3 cardinal_mip_lanczos2(float3 a, float3 b, float3 c, float3 d) {
    return 0.55 * a + 0.20 * b + 0.20 * c + 0.05 * d;
}
float3 cardinal_mip_catmull_rom(float3 a, float3 b, float3 c, float3 d) {
    return 0.50 * a + 0.225 * b + 0.225 * c + 0.05 * d;
}

#endif  // CARDINAL_FILTERS_HLSLI
