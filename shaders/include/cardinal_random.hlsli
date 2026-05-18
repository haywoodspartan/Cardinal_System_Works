// =============================================================================
// cardinal_random.hlsli — hash + RNG primitives, GPU side.
//
// Each function returns a float3 in [0,1] (channel-correlated for the
// integer hashes; independent for IGN). Companion CPU impls live in
// runtime/render/src/algos.cpp under the same names with the
// `cardinal_hash_` prefix.
// =============================================================================

#ifndef CARDINAL_RANDOM_HLSLI
#define CARDINAL_RANDOM_HLSLI

float3 cardinal_hash_wang(uint seed) {
    uint v = seed;
    v = (v ^ 61u) ^ (v >> 16);
    v *= 9u;
    v ^= (v >> 4);
    v *= 0x27d4eb2du;
    v ^= (v >> 15);
    float f = (v & 0x00FFFFFFu) / 16777216.0;
    return float3(f, frac(f * 1.6180339887), frac(f * 2.5034518));
}

float3 cardinal_hash_pcg(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    uint v     = (word >> 22u) ^ word;
    float f    = (v & 0x00FFFFFFu) / 16777216.0;
    return float3(f, frac(f * 1.6180339887), frac(f * 2.5034518));
}

// Dave Hoskins' Hash13 (3 inputs → 1 output, here we expand to 3 outputs).
float3 cardinal_hash_hash13(uint seed) {
    float3 p = float3(
        float((seed >>  0) & 0xFFu),
        float((seed >>  8) & 0xFFu),
        float((seed >> 16) & 0xFFu));
    p = frac(p * float3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yzx + 33.33);
    return frac(float3(
        (p.x + p.y) * p.z,
        (p.y + p.z) * p.x,
        (p.z + p.x) * p.y));
}

// Jorge Jimenez's Interleaved-Gradient Noise.
//   seed.lo16  → x screen coord
//   seed.hi16  → y screen coord
float3 cardinal_hash_ign(uint seed) {
    float x = float(seed & 0xFFFFu);
    float y = float(seed >> 16);
    float v = frac(52.9829189 * frac(0.06711056 * x + 0.00583715 * y));
    return float3(v, v, v);
}

#endif  // CARDINAL_RANDOM_HLSLI
