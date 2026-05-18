// =============================================================================
// cardinal_tessellation.hlsli — hull-shader tessellation factor helpers.
//
// HLSL counterpart to cardinal::render::tess (math.cpp). Both compute the
// same factor for the same inputs so the editor's tess-quality preview
// matches what the GPU produces. Hardware tess factors are clamped to
// [1, 64] by every modern adapter.
//
// Three policies:
//   distance      — cheap, depends only on a per-patch camera distance
//   edge_screen   — uniform on-screen triangle size (best quality budget)
//   phong_blend   — PN-triangles with curvature-driven extra detail
//
// Wire from a hull shader's patch-constant function:
//
//   HsConstants HSPatchConstants(InputPatch<VsOut, 3> ip,
//                                uint patch_id : SV_PrimitiveID)
//   {
//       float3 p = (ip[0].pos.xyz + ip[1].pos.xyz + ip[2].pos.xyz) / 3.0;
//       float  d = length(camera_pos - p);
//
//       HsConstants o;
//       o.tess_factor[0] = cardinal_tess_factor_distance(d, 16.0);
//       o.tess_factor[1] = o.tess_factor[0];
//       o.tess_factor[2] = o.tess_factor[0];
//       o.inside        = o.tess_factor[0];
//       return o;
//   }
// =============================================================================

#ifndef CARDINAL_TESSELLATION_HLSLI
#define CARDINAL_TESSELLATION_HLSLI

static const float CARDINAL_TESS_MIN = 1.0f;
static const float CARDINAL_TESS_MAX = 64.0f;

float cardinal_tess_clamp(float f) {
    return clamp(f, CARDINAL_TESS_MIN, CARDINAL_TESS_MAX);
}

// ---------------------------------------------------------------------------
// Distance policy — exactly mirrors render::tess::factor_distance().
// `scale` is the world-space distance at which factor = 1.
// ---------------------------------------------------------------------------
float cardinal_tess_factor_distance(float camera_to_patch, float scale) {
    float d = max(camera_to_patch, 0.001f);
    return cardinal_tess_clamp(scale / d);
}

// ---------------------------------------------------------------------------
// Edge-length policy — uniform on-screen triangle size.
// `edge_pixels` is the projected edge length in screen pixels;
// `target_pixels_per_segment` is how big a tessellated segment should be.
//
// To compute edge_pixels, project both endpoints to clip space, divide by
// w, multiply by viewport size, and take the length of the 2D delta.
// ---------------------------------------------------------------------------
float cardinal_tess_factor_edge(float edge_pixels, float target_pixels_per_segment) {
    float t = max(target_pixels_per_segment, 1.0f);
    return cardinal_tess_clamp(edge_pixels / t);
}

// Compute the projected edge length of a world-space segment in pixels.
// Caller passes the clip-space matrix and viewport dimensions.
float cardinal_edge_length_pixels(float3 a_world, float3 b_world,
                                  float4x4 clip_from_world,
                                  float2   viewport_size_px)
{
    float4 ca = mul(clip_from_world, float4(a_world, 1.0f));
    float4 cb = mul(clip_from_world, float4(b_world, 1.0f));
    // Skip behind-near-plane segments (calling code should also cull).
    if (ca.w <= 0.0f || cb.w <= 0.0f) return 1.0f;
    float2 sa = (ca.xy / ca.w) * 0.5f + 0.5f;
    float2 sb = (cb.xy / cb.w) * 0.5f + 0.5f;
    return length((sb - sa) * viewport_size_px);
}

// ---------------------------------------------------------------------------
// Phong / PN-triangle blend weight from a pair of vertex normals.
// 0 when normals are parallel (flat — tessellation adds nothing visual);
// 1 when perpendicular (sharp curve — full smoothing benefit).
// ---------------------------------------------------------------------------
float cardinal_phong_blend_weight(float3 n_a, float3 n_b) {
    float d = saturate(dot(n_a, n_b));
    return saturate(1.0f - d);
}

// Quality preset → world-space factor scale. Match this with the
// render::tess::Quality enum on the CPU.
//   0 = Off (factor = 1)
//   1 = Low      (8x at 1 unit)
//   2 = Medium  (16x)
//   3 = High    (32x)
float cardinal_tess_scale_for_quality(uint q) {
    if (q == 0) return 1.0f;
    if (q == 1) return 8.0f;
    if (q == 2) return 16.0f;
    return 32.0f;
}

#endif  // CARDINAL_TESSELLATION_HLSLI
