// =============================================================================
// cardinal_tonemap.hlsli — HLSL tonemap operators.
//
// One function per CPU-side tonemap algo registered in cardinal::render::algo.
// Same numerics — agreement asserted by the algo unit tests at build time.
//
// Usage in a fragment shader:
//
//     #include "cardinal_tonemap.hlsli"
//     float3 ldr = cardinal_tonemap_aces_approx(hdr);
//
// Pipelines specialise the chosen operator at compile-time via the algo
// registry's `hlsl_function` field — there's no per-pixel branch.
// =============================================================================

#ifndef CARDINAL_TONEMAP_HLSLI
#define CARDINAL_TONEMAP_HLSLI

float3 cardinal_tonemap_linear(float3 c) {
    return saturate(c);
}

float3 cardinal_tonemap_reinhard(float3 c) {
    return saturate(c / (c + 1.0));
}

float3 cardinal_tonemap_reinhard_ext(float3 c) {
    static const float W = 4.0;       // white point
    return saturate((c * (1.0 + c / (W * W))) / (c + 1.0));
}

// Hable / Uncharted 2 filmic.
float3 cardinal_tonemap_filmic(float3 c) {
    static const float A = 0.15, B = 0.50, C = 0.10;
    static const float D = 0.20, E = 0.02, F = 0.30;
    static const float W = 11.2;
    float3 x      = c * 2.0;
    float3 mapped = ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E / F;
    float  white  = ((W*(A*W + C*B) + D*E) / (W*(A*W + B) + D*F)) - E / F;
    return saturate(mapped / white);
}

// Krzysztof Narkowicz approximation — single rational, very cheap.
float3 cardinal_tonemap_aces_approx(float3 c) {
    static const float a = 2.51, b = 0.03, c1 = 2.43, d = 0.59, e = 0.14;
    return saturate((c * (a*c + b)) / (c * (c1*c + d) + e));
}

// Stephen Hill's 3x3 RRT/ODT-fitted curve (per channel).
float3 cardinal_tonemap_aces_full(float3 c) {
    float3 a = c * (c + 0.0245786) - 0.000090537;
    float3 b = c * (0.983729 * c + 0.4329510) + 0.238081;
    return saturate(a / b);
}

// Lottes 2016.
float3 cardinal_tonemap_lottes(float3 c) {
    static const float a       = 1.6;
    static const float d       = 0.977;
    static const float hdr_max = 8.0;
    static const float mid_in  = 0.18;
    static const float mid_out = 0.267;
    float ad   = pow(hdr_max, a*d) - pow(mid_in, a*d);
    float bb   = -pow(mid_in, a) + (pow(hdr_max, a*d)*mid_out
                  - pow(hdr_max, a)*mid_out*pow(mid_in, a*d) /
                    (pow(mid_in, a)*mid_out)) /
                  (pow(hdr_max, a*d) - pow(mid_in, a*d));
    float c2   = (pow(hdr_max, a*d)*mid_out
                  - pow(hdr_max, a)*mid_out*pow(mid_in, a*d) /
                    (pow(mid_in, a)*mid_out)) / ad;
    float3 pa  = pow(max(c, 0), a);
    return saturate(pa / (pow(pa, d) * bb + c2));
}

#endif  // CARDINAL_TONEMAP_HLSLI
