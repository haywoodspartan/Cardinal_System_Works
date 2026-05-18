// =============================================================================
// Cardinal — forward renderer (CPU-transformed vertex pipeline).
//
// Per frame:
//   1) Walk the scene; for each visible entity compute MVP and transform its
//      mesh vertices into clip-space float4 + per-vertex colour (decided by
//      the active ViewMode).
//   2) Concatenate the result into a CPU-writable scratch vertex buffer.
//   3) Bind the appropriate pipeline (Solid / Wireframe topology), upload,
//      issue one draw covering all entities.
//
// Why CPU-side transform: the RHI doesn't yet expose push constants or
// index buffers (Phase 5.1 work). Doing the math host-side keeps the
// pipeline shader trivial and lets the renderer ship today. The cost is
// O(verts) per frame — fine for the editor; absolutely the wrong choice
// for runtime rendering, which is why this whole TU is "forward renderer
// (baseline)" rather than "forward renderer".
// =============================================================================

#include <cardinal/scene/renderer.hpp>

#include <cardinal/core/async.hpp>
#include <cardinal/core/geom.hpp>
#include <cardinal/core/log.hpp>
#include <cardinal/core/simd_math.hpp>
#include <cardinal/rhi/rhi.hpp>
#include <cardinal/vgeom/vgeom.hpp>
#include <cardinal/vgeom/cluster.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

namespace cardinal::scene {

namespace {

constexpr const char* kSceneShader = R"(
// Semantic packing: rhi::VertexAttrib carries only (location, format,
// offset) — the D3D12 input-layout mapper invents HLSL semantics from
// location index (0=POSITION, 1=COLOR, 2+=TEXCOORD). To stay backend-
// portable without expanding the RHI API, the normal travels as
// TEXCOORD0 instead of NORMAL0. The semantic name is cosmetic; the
// shader treats it as a 3-float per-vertex input either way.
struct VSIn {
    float3 pos    : POSITION0;
    float3 color  : COLOR0;
    float3 normal : TEXCOORD0;
};
struct VSOut {
    float4 pos          : SV_POSITION;
    float3 color        : COLOR0;
    float3 world_pos    : TEXCOORD1;   // for upcoming GPU lighting (PS doesn't use yet)
    float3 world_normal : TEXCOORD2;   // for upcoming GPU lighting (PS doesn't use yet)
};

// Lights live in a bound StructuredBuffer (RHI storage-buffer slot 0 —
// Vulkan descriptor set 0 / binding 0; D3D12 root SRV t0), NOT in the
// push block. This lifted the old hard 2-light cap (the push block was
// saturated at the 256B portability ceiling): the array is now an
// arbitrarily-long GPU buffer the renderer rewrites each frame.
//
// Per-Light layout (3 × float4 = 48B, 16B-aligned so HLSL/SPIR-V and
// the C++ GpuLight mirror agree without padding):
//   dir_intensity : xyz = FROM-scene-to-light dir (Directional) OR
//                         world position (Point / Spot); w = intensity
//   color_range   : xyz = color; w = range (0.0 => Directional,
//                                            >0  => Point / Spot)
//   spot          : xyz = cone direction (FROM light TO target, unit);
//                   w   = outer_cos (0.0 => not a Spot; >0 => Spot cone)
//
// Kind disambiguation:
//   color_range.w == 0                  → Directional
//   color_range.w > 0 && spot.w == 0    → Point
//   color_range.w > 0 && spot.w > 0     → Spot
struct Light {
    float4 dir_intensity;
    float4 color_range;
    float4 spot;
};
[[vk::binding(0, 0)]] StructuredBuffer<Light> g_lights : register(t0);

// Per-entity surface material — RHI storage-buffer slot 1 (Vulkan
// descriptor set 0 / binding 1; D3D12 root SRV t1). One entry per
// distinct entity drawn this frame; the push block carries the index.
// 1 × float4 = 16B (mirrors C++ GpuMaterial, no padding):
//   spec.x = specular_intensity   spec.z = roughness  (PBR, unused yet)
//   spec.y = specular_power       spec.w = metalness  (PBR, unused yet)
struct GpuMaterial {
    float4 spec;
};
[[vk::binding(1, 0)]] StructuredBuffer<GpuMaterial> g_materials : register(t1);

// Directional shadow map — RHI sampled-texture slot 0, which the
// binding convention places at descriptor binding storage_buffer_slots
// (=2) on Vulkan and SRV register t2 on D3D12. The static/default
// sampler is s0. Pass 1 rendered scene depth from the sun's POV; the
// PS reprojects each fragment into this map for the shadow test.
// [[vk::combinedImageSampler]] + matching vk::binding makes DXC emit a
// single VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER at set 0 / binding 2
// — matching the Vulkan pipeline's DSL + bind_sampled_texture write.
// The attributes are ignored for DXIL, where register(t2)/register(s0)
// + the D3D12 SRV table + static sampler s0 do the equivalent.
[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
Texture2D    g_shadow     : register(t2);
[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
SamplerState g_shadow_smp : register(s0);

// Push-constant block — per-entity transform + scene-wide ambient/mode
// + light count & camera position + material index. Light + material
// arrays live in storage buffers, so the block stays tiny.
//
// Layout (each float4 is one 16B HLSL cbuffer row):
//   mvp             : 4 rows (64B)
//   model           : 4 rows (64B)
//   ambient_mode    : 1 row  (16B) — xyz=ambient, w=mode strength
//   light_count_pad : 1 row  (16B) — x=active light count (g_lights[0..x)),
//                                     yzw=camera world position (specular)
//   material_pad    : 1 row  (16B) — x=material index (g_materials[x]),
//                                     y=exposure, z=shadow valid (1/0),
//                                     w reserved
//   light_vp        : 4 rows (64B) — directional light-space view-proj
//                                     (pass-1 matrix) for the shadow test
// Total: 240 bytes — still under the 256B ceiling.
[[vk::push_constant]] cbuffer Push : register(b0) {
    float4x4 mvp;
    float4x4 model;
    float4   ambient_mode;
    float4   light_count_pad;
    float4   material_pad;
    float4x4 light_vp;
};

// 3×3 PCF directional shadow factor. Reprojects the world position
// into the pass-1 light clip space, then averages 9 depth comparisons
// around the texel. Returns 1.0 = fully lit, 0.0 = fully shadowed.
// Fragments outside the shadow frustum read as lit (no contact-less
// darkening at the map edge). Slope-scaled bias kills shadow acne on
// surfaces grazing the sun.
float directional_shadow(float3 world_pos, float ndotl) {
    float4 lc = mul(light_vp, float4(world_pos, 1.0));
    if (lc.w <= 0.0) return 1.0;
    float3 ndc = lc.xyz / lc.w;
    // Engine clip is RH z∈[0,1]; XY → [0,1] UV with V flipped.
    float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 ||
        ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;                       // outside the sun's frustum
    }
    float bias  = max(0.0008, 0.004 * (1.0 - ndotl));
    float dref  = ndc.z - bias;
    float texel = 1.0 / 2048.0;           // matches kShadowDim
    float lit   = 0.0;
    [unroll] for (int oy = -1; oy <= 1; ++oy)
    [unroll] for (int ox = -1; ox <= 1; ++ox) {
        float sd = g_shadow.SampleLevel(
            g_shadow_smp, uv + float2(ox, oy) * texel, 0).r;
        lit += (dref <= sd) ? 1.0 : 0.0;
    }
    return lit * (1.0 / 9.0);
}

VSOut VSMain(VSIn i) {
    VSOut o;
    float4 pos4    = float4(i.pos, 1.0);
    o.pos          = mul(mvp,   pos4);
    o.world_pos    = mul(model, pos4).xyz;
    // World-space normal via the upper-left 3x3 of model. Uniform-scale
    // assumption — non-uniform scale would need the inverse-transpose;
    // documented limitation matches the CPU rot_xform() helper today.
    o.world_normal = normalize(mul((float3x3)model, i.normal));
    o.color        = i.color;
    return o;
}

// ACES filmic tonemap (Narkowicz 2015 fit). PBR + the uncapped light
// buffer routinely push radiance well past 1.0; without a curve those
// pixels hard-clip to flat white and the chrome highlight / light-ring
// detail is lost. This cheap rational approximation rolls highlights
// off smoothly while keeping mids/shadows close to linear. Applied to
// the lit HDR result only (NOT the debug view-mode colours).
float3 aces_tonemap(float3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

float4 PSMain(VSOut i) : SV_TARGET {
    // GPU lighting — iterate the bound g_lights buffer (count from the
    // push block). Per-light kind gates on (color_range.w, spot.w):
    //   color.w == 0                  → Directional (dir_intensity.xyz = FROM-scene-TO-light)
    //   color.w >  0 && spot.w == 0   → Point      (dir_intensity.xyz = world position,
    //                                                falloff = (1-(d/r)^2)^2)
    //   color.w >  0 && spot.w >  0   → Spot       (Point falloff × cone mask;
    //                                                spot.xyz = cone direction
    //                                                from light TO target,
    //                                                spot.w   = outer_cos)
    // ambient_mode.w lerps unlit (debug view modes) vs lit (Solid).
    float3 normal = normalize(i.world_normal);
    int    nlit   = int(light_count_pad.x);
    // Camera position (specular view vector) arrives via light_count_pad.yzw.
    float3 view_pos = light_count_pad.yzw;
    float3 V        = normalize(view_pos - i.world_pos);
    // Per-entity material (slot 1). Metallic-roughness PBR:
    //   spec.x = spec_scale  — overall specular strength (legacy
    //                          "intensity"; keeps the demo slider live)
    //   spec.y = (legacy specular_power — unused under Cook-Torrance,
    //            superseded by roughness; kept for struct stability)
    //   spec.z = roughness   — GGX α basis (clamped off 0 to keep the
    //                          NDF non-singular)
    //   spec.w = metalness   — lerps F0 (0.04 dielectric → albedo) and
    //                          kills diffuse for conductors
    GpuMaterial mat        = g_materials[int(material_pad.x)];
    float  spec_scale      = mat.spec.x;
    float  roughness       = clamp(mat.spec.z, 0.045, 1.0);
    float  metalness       = saturate(mat.spec.w);
    float3 albedo          = i.color;
    float3 F0              = lerp(float3(0.04, 0.04, 0.04), albedo, metalness);
    const float PI         = 3.14159265;
    // Sky-derived ambient (no cubemap / IBL yet). The flat sky colour
    // (ambient_mode.xyz, fed from the Sky time-of-day system) becomes a
    // crude hemisphere — brighter looking up, dimmer at the ground
    // bounce — split into two terms:
    //   • diffuse irradiance  — dielectric only, tinted by albedo
    //   • environment specular — a roughness-aware Fresnel reflection of
    //     the hemisphere. This is the term that stops conductors
    //     (metalness→1, zero diffuse) reading near-black between direct
    //     highlights, which the Blinn→PBR switch otherwise introduced.
    float3 sky_amb  = ambient_mode.xyz;
    float3 env_up   = sky_amb * 1.6;
    float3 env_dn   = sky_amb * 0.4;
    float  NdotV_a  = max(dot(normal, V), 1e-4);
    float3 R        = reflect(-V, normal);
    float3 env_refl = lerp(env_dn, env_up, saturate(R.y * 0.5 + 0.5));
    float3 env_irr  = lerp(env_dn, env_up, saturate(normal.y * 0.5 + 0.5));
    // Roughness-aware Fresnel (Lagarde): rough metals reflect less of
    // the environment at grazing angles than a mirror would. F0 already
    // encodes metalness (0.04 dielectric → albedo).
    float3 one_mr   = float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness);
    float3 Famb     = F0 + (max(one_mr, F0) - F0) * pow(1.0 - NdotV_a, 5.0);
    float3 amb_diff = env_irr * albedo * (1.0 - metalness) * (1.0 - Famb);
    float3 amb_spec = env_refl * Famb * spec_scale;
    float3 lit      = amb_diff + amb_spec;
    // Dynamic count now (no fixed upper bound) — the loop reads
    // g_lights[0 .. nlit). Bounded by the renderer's per-frame upload.
    for (int li = 0; li < nlit; ++li) {
        Light  lt      = g_lights[li];
        float  range   = lt.color_range.w;
        float  outer   = lt.spot.w;
        float3 L;
        float  atten;
        bool   is_dir  = (range <= 0.0);   // Directional → casts shadow
        if (range > 0.0) {
            // Point or Spot — dir_intensity.xyz is a world position.
            float3 to_light = lt.dir_intensity.xyz - i.world_pos;
            float  d2       = dot(to_light, to_light);
            float  rinv     = 1.0 / max(range, 1e-3);
            float  falloff  = saturate(1.0 - d2 * rinv * rinv);
            atten           = falloff * falloff;
            L               = normalize(to_light);
            if (outer > 0.0) {
                // Spot cone mask. spot.xyz points FROM light TO target,
                // so we compare against -L (FROM target TO light → negate
                // to get FROM light TO surface direction component).
                // inner_cos is derived as a soft 30% lerp toward 1.0 so
                // callers only need to supply outer_cos (matches the
                // common "outer cone defines fall-off start" intuition).
                float3 cone_dir = normalize(lt.spot.xyz);
                float  spot_cos = dot(-L, cone_dir);
                float  inner    = lerp(outer, 1.0, 0.3);
                float  cone     = smoothstep(outer, inner, spot_cos);
                atten          *= cone;
            }
        } else {
            // Directional — dir_intensity.xyz is a (FROM-scene-TO-light) dir.
            L     = normalize(lt.dir_intensity.xyz);
            atten = 1.0;
        }
        // Cook-Torrance metallic-roughness BRDF. radiance already folds
        // light colour × intensity × attenuation (Point falloff / Spot
        // cone). Only shade the lit hemisphere (NdotL > 0).
        float  NdotL    = max(dot(normal, L), 0.0);
        float3 radiance = lt.color_range.xyz * lt.dir_intensity.w * atten;
        // Directional sun shadow (pass-1 map). material_pad.z gates it
        // so wire/gizmo + no-directional + missing-resource frames skip
        // the reprojection entirely.
        if (is_dir && material_pad.z > 0.5 && NdotL > 0.0) {
            radiance *= directional_shadow(i.world_pos, NdotL);
        }
        if (NdotL > 0.0) {
            float3 H     = normalize(L + V);
            float  NdotV = max(dot(normal, V), 1e-4);
            float  NdotH = max(dot(normal, H), 0.0);
            float  HdotV = max(dot(H, V),      0.0);

            // GGX / Trowbridge-Reitz normal distribution.
            float  a   = roughness * roughness;
            float  a2  = a * a;
            float  dnm = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
            float  D   = a2 / max(PI * dnm * dnm, 1e-7);

            // Smith geometry, Schlick-GGX (direct-light k = (r+1)^2/8).
            float  k   = (roughness + 1.0);  k = (k * k) * 0.125;
            float  Gv  = NdotV / (NdotV * (1.0 - k) + k);
            float  Gl  = NdotL / (NdotL * (1.0 - k) + k);
            float  G   = Gv * Gl;

            // Fresnel-Schlick.
            float3 F   = F0 + (1.0 - F0) * pow(1.0 - HdotV, 5.0);

            // Specular = DGF / (4 NdotV NdotL); scaled by the legacy
            // per-material strength so the demo's intensity slider still
            // does something meaningful.
            float3 spec = (D * G) * F /
                          max(4.0 * NdotV * NdotL, 1e-4);
            spec *= spec_scale;

            // Energy-conserving diffuse: what isn't specularly reflected
            // (1-F) and isn't metal (1-metalness), Lambertian / PI.
            float3 kd      = (1.0 - F) * (1.0 - metalness);
            float3 diffuse = kd * albedo / PI;

            lit += (diffuse + spec) * radiance * NdotL;
        }
    }
    // Tonemap the HDR PBR result, then blend against the raw debug
    // colour by mode strength (ambient_mode.w: 1 = Solid/lit, 0 = a
    // debug view mode). Tonemapping `lit` BEFORE the lerp keeps the
    // Normals / Heightmap / Polygons visualisers pixel-exact — they
    // must show raw encoded values, never an S-curve. Exposure rides
    // material_pad.y (per-frame host knob via LightSet::set_exposure);
    // guard 0 → 1.0 so an unset value is neutral, not black.
    float exposure = material_pad.y;
    if (exposure <= 0.0) exposure = 1.0;
    float3 mapped = aces_tonemap(lit * exposure);
    return float4(lerp(i.color, mapped, ambient_mode.w), 1.0);
}
)";

// Shadow depth-only shader. Pass 1 of shadow mapping renders the scene
// from the directional light's POV into a D32 depth texture; only the
// depth matters, so the VS just transforms by the light-space MVP and
// the PS is empty (the pipeline is created with color_format=Unknown →
// zero colour attachments, matching begin_shadow_pass). Reuses the
// RenderVertex layout (only POSITION0 is consumed). Push block is a
// single light-space MVP (light_vp * model), 64 bytes.
constexpr const char* kShadowShader = R"(
struct VSIn {
    float3 pos    : POSITION0;
    float3 color  : COLOR0;
    float3 normal : TEXCOORD0;
};
[[vk::push_constant]] cbuffer Push : register(b0) {
    float4x4 light_mvp;
};
float4 VSMain(VSIn i) : SV_POSITION {
    return mul(light_mvp, float4(i.pos, 1.0));
}
void PSMain() {}
)";

struct RenderVertex {
    Vec3 model_pos;     // model-space position; VS multiplies by push-constant MVP
    Vec3 model_normal;  // model-space normal;  VS multiplies by push-constant model 3x3
    Vec3 color;         // CPU-computed base color (entity tint × per-vertex color × ViewMode shade)
};

// CPU mirror of the HLSL `Light` struct (kSceneShader). 3×Vec4 = 48B,
// 16B-aligned — same layout the StructuredBuffer<Light> expects, no
// padding. Uploaded into the per-viewport light buffer each frame and
// bound at storage slot 0. Field semantics match the shader comment:
//   dir_intensity : xyz dir/pos, w intensity
//   color_range   : xyz color,   w range (0 => Directional)
//   spot          : xyz cone dir, w outer_cos (0 => not Spot)
struct GpuLight {
    Vec4 dir_intensity;
    Vec4 color_range;
    Vec4 spot;
};
static_assert(sizeof(GpuLight) == 48,
              "GpuLight must match HLSL StructuredBuffer<Light> stride (48B)");

// CPU mirror of the HLSL `GpuMaterial` struct (kSceneShader). 1×Vec4 =
// 16B, bound at storage slot 1. One entry per distinct entity drawn
// this frame; the push block's material_pad.x indexes it.
//   spec.x = specular_intensity   spec.z = roughness  (PBR, unused yet)
//   spec.y = specular_power       spec.w = metalness  (PBR, unused yet)
struct GpuMaterial {
    Vec4 spec;
};
static_assert(sizeof(GpuMaterial) == 16,
              "GpuMaterial must match HLSL StructuredBuffer<GpuMaterial> stride (16B)");

// Append the 12 edges of an axis-aligned box (defined by min/max) to
// the line-list vertex buffer. Each edge contributes 2 vertices (24
// total per AABB) coloured uniformly. Used by the gizmo pass.
inline void add_aabb_edges(std::vector<RenderVertex>& out,
                           const Vec3& mn, const Vec3& mx,
                           const Vec3& color)
{
    const Vec3 c[8] = {
        {mn.x, mn.y, mn.z}, {mx.x, mn.y, mn.z},
        {mx.x, mx.y, mn.z}, {mn.x, mx.y, mn.z},
        {mn.x, mn.y, mx.z}, {mx.x, mn.y, mx.z},
        {mx.x, mx.y, mx.z}, {mn.x, mx.y, mx.z},
    };
    static constexpr int E[12][2] = {
        {0,1},{1,2},{2,3},{3,0},        // bottom face
        {4,5},{5,6},{6,7},{7,4},        // top face
        {0,4},{1,5},{2,6},{3,7},        // verticals
    };
    // Gizmo lines don't use the world_normal varying — but RenderVertex
    // has a normal slot, so we fill it with a neutral up-vector. The
    // wireframe PS doesn't read it anyway.
    const Vec3 nz{0, 1, 0};
    for (auto& e : E) {
        out.push_back({c[e[0]], nz, color});
        out.push_back({c[e[1]], nz, color});
    }
}

// Append the 12 edges of a perspective camera frustum to the line-list
// buffer. Reconstructs the 8 world-space corners by inverting view*proj
// and projecting NDC corners back into world space.
inline void add_frustum_edges(std::vector<RenderVertex>& out,
                              const Mat4& vp,
                              const Vec3& color)
{
    const Mat4 inv = vp.inverse();
    auto unp = [&](float x, float y, float z) -> Vec3 {
        const Vec4 in{x, y, z, 1.0f};
        const Vec4 w  = inv * in;
        const float iw = (std::abs(w.w) > 1e-6f) ? (1.0f / w.w) : 1.0f;
        return Vec3{w.x * iw, w.y * iw, w.z * iw};
    };
    // NDC corners — z=0 near, z=1 far (DX/Vulkan reverse-Z aware
    // callers should adjust; we treat z in [0,1] which is the common
    // convention from cardinal::Mat4::perspective).
    const Vec3 c[8] = {
        unp(-1, -1, 0), unp( 1, -1, 0), unp( 1,  1, 0), unp(-1,  1, 0), // near
        unp(-1, -1, 1), unp( 1, -1, 1), unp( 1,  1, 1), unp(-1,  1, 1), // far
    };
    static constexpr int E[12][2] = {
        {0,1},{1,2},{2,3},{3,0},        // near face
        {4,5},{5,6},{6,7},{7,4},        // far face
        {0,4},{1,5},{2,6},{3,7},        // depth lines
    };
    // Gizmo lines don't use the world_normal varying — but RenderVertex
    // has a normal slot, so we fill it with a neutral up-vector. The
    // wireframe PS doesn't read it anyway.
    const Vec3 nz{0, 1, 0};
    for (auto& e : E) {
        out.push_back({c[e[0]], nz, color});
        out.push_back({c[e[1]], nz, color});
    }
}

// Append a wireframe right circular cone to the line-list buffer.
// Apex at `apex`, axis along unit-`axis` for length `length` (so the
// base centre sits at apex + axis*length), base radius `radius`,
// `segments` divisions of the base circle. Produces N spokes from the
// apex plus an N-segment polyline ring at the base — 2*segments lines
// total, 4*segments vertices.
//
// Used by the Spot-light gizmo: half-angle = acos(outer_cos), then
// radius = length × tan(half-angle).
inline void add_cone_edges(std::vector<RenderVertex>& out,
                           const Vec3& apex, const Vec3& axis_in,
                           float length, float radius,
                           int segments, const Vec3& color)
{
    if (segments < 3) segments = 3;
    if (segments > 64) segments = 64;
    // Normalize and build an orthonormal basis (u, v, axis). Pick the
    // world axis least parallel to `axis` as the seed for u, then
    // cross-product to get a robust right-handed frame.
    const float al = std::sqrt(axis_in.x*axis_in.x + axis_in.y*axis_in.y + axis_in.z*axis_in.z);
    const Vec3  axis = (al > 1e-6f) ? Vec3{axis_in.x/al, axis_in.y/al, axis_in.z/al}
                                    : Vec3{0, 1, 0};
    const Vec3  seed = (std::abs(axis.y) < 0.9f) ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
    // u = normalize(seed × axis); v = axis × u
    Vec3 u{
        seed.y*axis.z - seed.z*axis.y,
        seed.z*axis.x - seed.x*axis.z,
        seed.x*axis.y - seed.y*axis.x,
    };
    const float ul = std::sqrt(u.x*u.x + u.y*u.y + u.z*u.z);
    u = (ul > 1e-6f) ? Vec3{u.x/ul, u.y/ul, u.z/ul} : Vec3{1, 0, 0};
    const Vec3 v{
        axis.y*u.z - axis.z*u.y,
        axis.z*u.x - axis.x*u.z,
        axis.x*u.y - axis.y*u.x,
    };
    const Vec3 base_centre{
        apex.x + axis.x * length,
        apex.y + axis.y * length,
        apex.z + axis.z * length,
    };
    const Vec3 nz{0, 1, 0};      // unused by wireframe PS
    // Compute base ring then emit spokes + segments.
    std::vector<Vec3> ring;
    ring.reserve(static_cast<usize>(segments));
    const float two_pi = 6.28318530717958647692f;
    for (int s = 0; s < segments; ++s) {
        const float t  = static_cast<float>(s) / static_cast<float>(segments);
        const float a  = t * two_pi;
        const float ca = std::cos(a), sa = std::sin(a);
        ring.push_back(Vec3{
            base_centre.x + radius * (u.x * ca + v.x * sa),
            base_centre.y + radius * (u.y * ca + v.y * sa),
            base_centre.z + radius * (u.z * ca + v.z * sa),
        });
    }
    // Spokes: apex → ring[i].
    for (const auto& p : ring) {
        out.push_back({apex, nz, color});
        out.push_back({p,    nz, color});
    }
    // Ring: ring[i] → ring[(i+1)%segments].
    for (int s = 0; s < segments; ++s) {
        const Vec3& a = ring[static_cast<usize>(s)];
        const Vec3& b = ring[static_cast<usize>((s + 1) % segments)];
        out.push_back({a, nz, color});
        out.push_back({b, nz, color});
    }
}

// Append three orthogonal great circles (a wireframe "gyroscope"
// sphere) to the line-list buffer. Centre `c`, radius `r`, `segments`
// divisions per circle. Cheaper than a real icosphere and reads as a
// sphere fine from a few metres back. 3*segments line segments,
// 6*segments vertices total.
//
// Used by the Point-light gizmo (range-radius sphere centred at the
// light position).
inline void add_sphere_edges(std::vector<RenderVertex>& out,
                             const Vec3& c, float r,
                             int segments, const Vec3& color)
{
    if (segments < 6) segments = 6;
    if (segments > 64) segments = 64;
    const Vec3 nz{0, 1, 0};
    const float two_pi = 6.28318530717958647692f;
    for (int plane = 0; plane < 3; ++plane) {
        // Plane 0: XY (axis Z), Plane 1: XZ (axis Y), Plane 2: YZ (axis X).
        for (int s = 0; s < segments; ++s) {
            const float a0 = (static_cast<float>(s)     / segments) * two_pi;
            const float a1 = (static_cast<float>(s + 1) / segments) * two_pi;
            Vec3 p0, p1;
            switch (plane) {
                case 0:
                    p0 = Vec3{c.x + r*std::cos(a0), c.y + r*std::sin(a0), c.z};
                    p1 = Vec3{c.x + r*std::cos(a1), c.y + r*std::sin(a1), c.z};
                    break;
                case 1:
                    p0 = Vec3{c.x + r*std::cos(a0), c.y, c.z + r*std::sin(a0)};
                    p1 = Vec3{c.x + r*std::cos(a1), c.y, c.z + r*std::sin(a1)};
                    break;
                default:
                    p0 = Vec3{c.x, c.y + r*std::cos(a0), c.z + r*std::sin(a0)};
                    p1 = Vec3{c.x, c.y + r*std::cos(a1), c.z + r*std::sin(a1)};
                    break;
            }
            out.push_back({p0, nz, color});
            out.push_back({p1, nz, color});
        }
    }
}

// Append a unit-length direction arrow at a fixed origin offset. Two
// lines: the shaft (origin → origin+dir*length) and a small 4-segment
// arrowhead at the tip. Used by the Directional-light gizmo since
// directionals don't have a world position.
inline void add_dir_arrow(std::vector<RenderVertex>& out,
                          const Vec3& origin, const Vec3& dir_in,
                          float length, const Vec3& color)
{
    const float dl = std::sqrt(dir_in.x*dir_in.x + dir_in.y*dir_in.y + dir_in.z*dir_in.z);
    const Vec3  dir = (dl > 1e-6f) ? Vec3{dir_in.x/dl, dir_in.y/dl, dir_in.z/dl}
                                   : Vec3{0, 1, 0};
    const Vec3  tip{
        origin.x + dir.x * length,
        origin.y + dir.y * length,
        origin.z + dir.z * length,
    };
    const Vec3 nz{0, 1, 0};
    out.push_back({origin, nz, color});
    out.push_back({tip,    nz, color});
    // Arrowhead: 4 small segments back from the tip, in the orthogonal
    // plane of `dir`.
    const Vec3 seed = (std::abs(dir.y) < 0.9f) ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
    Vec3 u{
        seed.y*dir.z - seed.z*dir.y,
        seed.z*dir.x - seed.x*dir.z,
        seed.x*dir.y - seed.y*dir.x,
    };
    const float ul = std::sqrt(u.x*u.x + u.y*u.y + u.z*u.z);
    u = (ul > 1e-6f) ? Vec3{u.x/ul, u.y/ul, u.z/ul} : Vec3{1, 0, 0};
    const Vec3 v{
        dir.y*u.z - dir.z*u.y,
        dir.z*u.x - dir.x*u.z,
        dir.x*u.y - dir.y*u.x,
    };
    const float head_back = length * 0.25f;
    const float head_rad  = length * 0.12f;
    const Vec3 base{
        tip.x - dir.x * head_back,
        tip.y - dir.y * head_back,
        tip.z - dir.z * head_back,
    };
    static const float ca[4] = { 1.0f,  0.0f, -1.0f,  0.0f };
    static const float sa[4] = { 0.0f,  1.0f,  0.0f, -1.0f };
    for (int s = 0; s < 4; ++s) {
        const Vec3 p{
            base.x + head_rad * (u.x * ca[s] + v.x * sa[s]),
            base.y + head_rad * (u.y * ca[s] + v.y * sa[s]),
            base.z + head_rad * (u.z * ca[s] + v.z * sa[s]),
        };
        out.push_back({p,   nz, color});
        out.push_back({tip, nz, color});
    }
}

inline Vec3 hsv_to_rgb(float h, float s, float v) {
    const float i = std::floor(h * 6.0f);
    const float f = h * 6.0f - i;
    const float p = v * (1.0f - s);
    const float q = v * (1.0f - f * s);
    const float t = v * (1.0f - (1.0f - f) * s);
    switch (static_cast<int>(i) % 6) {
        case 0: return { v, t, p };
        case 1: return { q, v, p };
        case 2: return { p, v, t };
        case 3: return { p, q, v };
        case 4: return { t, p, v };
        default:return { v, p, q };
    }
}

// Light-space view-projection for the directional shadow pass. An RH
// orthographic box of half-extent `half` centred on the world origin,
// looked at from along the (incoming) sun direction. ortho matches the
// engine's RH / depth-0..1 convention (see Mat4::perspective): a glm
// orthoRH_ZO. `sun_dir` is the direction the light travels (points AT
// the scene, i.e. scene::Light::direction for a Directional).
[[maybe_unused]] inline Mat4 directional_light_vp(
    const Vec3& sun_dir, float half, float near_z, float far_z)
{
    // Normalise; guard a degenerate zero direction.
    const float dl = std::sqrt(sun_dir.x*sun_dir.x + sun_dir.y*sun_dir.y
                             + sun_dir.z*sun_dir.z);
    const Vec3 d = (dl > 1e-6f)
        ? Vec3{sun_dir.x/dl, sun_dir.y/dl, sun_dir.z/dl}
        : Vec3{0.0f, -1.0f, 0.0f};
    // Place the light "camera" back along -travel so the whole box is
    // in front of it; distance = far_z*0.5 keeps the scene mid-frustum.
    const float dist = far_z * 0.5f;
    const Vec3 eye{ -d.x * dist, -d.y * dist, -d.z * dist };
    const Vec3 up = (std::abs(d.y) < 0.95f) ? Vec3{0,1,0} : Vec3{0,0,1};
    const Mat4 view = Mat4::look_at(eye, Vec3{0,0,0}, up);

    Mat4 proj{};                       // zero-init
    proj.m[0][0] =  1.0f / half;       // 2/(r-l) with r=-l=half
    proj.m[1][1] =  1.0f / half;
    proj.m[2][2] = -1.0f / (far_z - near_z);          // RH, z∈[0,1]
    proj.m[3][2] = -near_z / (far_z - near_z);
    proj.m[3][3] =  1.0f;
    return proj * view;
}

// Pick a per-vertex color based on the view mode. World-space pos / normal
// are passed in so heightmap / normals can use them. The optional LightSet
// drives Solid mode's shading — directional + point + spot all honoured.
Vec3 color_for(ViewMode m, const Vec3& world_pos, const Vec3& world_normal,
               const Vec3& base_color, u32 tri_index, const Vec3& tint,
               const LightSet* lights)
{
    switch (m) {
        case ViewMode::Solid: {
            // GPU lighting in PS now — return just the tinted base
            // colour (the PBR albedo); the PS applies the Cook-Torrance
            // BRDF. (Used to call lights->shade() here; that work moved
            // to PSMain. world_pos / world_normal / `lights` are now
            // unused by this branch — kept in the signature because
            // other view modes still consume world_pos / normal.)
            (void)world_pos; (void)world_normal; (void)lights;
            return Vec3{ base_color.x * tint.x,
                         base_color.y * tint.y,
                         base_color.z * tint.z };
        }
        case ViewMode::Wireframe: return Vec3{ 0.55f, 0.85f, 1.0f };
        case ViewMode::Polygons: {
            // Deterministic hash of triangle index → vivid HSV colour.
            const float h = std::fmod(static_cast<float>(tri_index) * 0.6180339887f, 1.0f);
            return hsv_to_rgb(h, 0.6f, 0.95f);
        }
        case ViewMode::Heightmap: {
            // Map world Y into [0,1] over a wider range than the original
            // ±2 bracket — most demo scenes have terrain at -32..+32.
            // Cool→green→warm ramp so users can read elevation by hue.
            const float t = std::clamp((world_pos.y + 32.0f) / 64.0f, 0.0f, 1.0f);
            return Vec3{ t, 1.0f - std::abs(t - 0.5f) * 2.0f, 1.0f - t };
        }
        case ViewMode::Normals: {
            return Vec3{ world_normal.x * 0.5f + 0.5f,
                         world_normal.y * 0.5f + 0.5f,
                         world_normal.z * 0.5f + 0.5f };
        }
        case ViewMode::RTXPreview:
            // Until the real RT pass lands, mark RTX-mode pixels with a
            // distinct teal so it's obvious in the UI which mode is active.
            return Vec3{ 0.2f, 0.85f, 0.85f };
    }
    return base_color;
}

// -----------------------------------------------------------------------------
// ForwardRendererImpl
// -----------------------------------------------------------------------------
class ForwardRendererImpl final : public ForwardRenderer {
public:
    bool initialize(rhi::Device& dev, rhi::Swapchain& sw) {
        device_    = &dev;
        swapchain_ = &sw;

        auto vs = dev.compile_shader(rhi::ShaderStage::Vertex,   kSceneShader, "VSMain");
        auto fs = dev.compile_shader(rhi::ShaderStage::Fragment, kSceneShader, "PSMain");
        if (!vs.ok() || !fs.ok()) {
            cardinal::log::errorf("scene/renderer", "scene shader compile failed");
            return false;
        }

        rhi::PipelineDesc pd{};
        pd.vertex_shader   = std::move(vs);
        pd.fragment_shader = std::move(fs);
        pd.vertex_stride   = sizeof(RenderVertex);
        pd.color_format    = sw.color_format();
        // Opt into depth-test on backends that expose a depth attachment.
        // The CPU painter's-algorithm fallback below is still applied so
        // visually correct rendering happens even on the Vulkan path
        // (where the depth attachment isn't wired up yet).
        pd.depth_format    = sw.depth_format();
        pd.depth_test      = (sw.depth_format() != rhi::Format::Unknown);
        pd.depth_write     = pd.depth_test;
        pd.vertex_attribs  = {
            { 0, rhi::Format::R32G32B32_Float, offsetof(RenderVertex, model_pos)    },
            { 1, rhi::Format::R32G32B32_Float, offsetof(RenderVertex, color)        },
            { 2, rhi::Format::R32G32B32_Float, offsetof(RenderVertex, model_normal) },
        };
        // Push-constant block (matches kSceneShader Push cbuffer):
        //   mvp(64)+model(64)+ambient_mode(16)+light_count_pad(16)
        //   +material_pad(16)+light_vp(64) = 240 bytes. Under the 256B
        //   every-modern-desktop ceiling; the light/material arrays that
        //   would have blown it live in storage buffers.
        // Two storage-buffer slots + one sampled texture:
        //   slot 0 = g_lights    (StructuredBuffer<Light>,      t0)
        //   slot 1 = g_materials (StructuredBuffer<GpuMaterial>, t1)
        //   tex  0 = g_shadow    (Texture2D,                     t2, s0)
        // Vulkan: set 0 bindings 0/1 (SSBO) + 2 (combined sampler);
        // D3D12: root SRVs t0/t1 + SRV table t2 + static sampler s0.
        pd.push_constant_size    = 240;
        pd.storage_buffer_slots  = 2;
        pd.sampled_texture_slots = 1;

        // Solid topology pipeline.
        pd.topology = rhi::PrimitiveTopology::TriangleList;
        pso_solid_  = dev.create_pipeline(pd);

        // Wireframe pipeline (line topology — we expand each triangle into
        // 3 line segments before upload).
        rhi::PipelineDesc pd_wire = pd;
        // PipelineDesc moves shaders; recompile a second pair for the wire pass.
        pd_wire.vertex_shader   = dev.compile_shader(rhi::ShaderStage::Vertex,   kSceneShader, "VSMain");
        pd_wire.fragment_shader = dev.compile_shader(rhi::ShaderStage::Fragment, kSceneShader, "PSMain");
        pd_wire.topology = rhi::PrimitiveTopology::LineList;
        pso_wire_ = dev.create_pipeline(pd_wire);

        if (pso_solid_ == nullptr || pso_wire_ == nullptr) {
            cardinal::log::errorf("scene/renderer", "scene PSO creation failed");
            return false;
        }

        // ---- Shadow mapping resources (pass 1) ----------------------
        // Depth-only pipeline + a D32 shadow map. Created here; the
        // two-pass wiring (render scene from the sun's POV, then sample
        // in PSMain) lands next. nullptr-tolerant: a backend that hasn't
        // implemented create_texture (or a GPU that fails the alloc)
        // simply leaves shadows off — the main pass is unaffected.
        rhi::PipelineDesc pd_shadow{};
        pd_shadow.vertex_shader   = dev.compile_shader(
            rhi::ShaderStage::Vertex, kShadowShader, "VSMain");
        pd_shadow.fragment_shader = dev.compile_shader(
            rhi::ShaderStage::Fragment, kShadowShader, "PSMain");
        pd_shadow.vertex_stride   = sizeof(RenderVertex);
        pd_shadow.color_format    = rhi::Format::Unknown;     // depth-only
        pd_shadow.depth_format    = rhi::Format::D32_Float;
        pd_shadow.depth_test      = true;
        pd_shadow.depth_write     = true;
        pd_shadow.topology        = rhi::PrimitiveTopology::TriangleList;
        pd_shadow.vertex_attribs  = {
            { 0, rhi::Format::R32G32B32_Float, offsetof(RenderVertex, model_pos)    },
            { 1, rhi::Format::R32G32B32_Float, offsetof(RenderVertex, color)        },
            { 2, rhi::Format::R32G32B32_Float, offsetof(RenderVertex, model_normal) },
        };
        pd_shadow.push_constant_size = sizeof(Mat4);          // light_mvp
        pso_shadow_ = dev.create_pipeline(pd_shadow);

        rhi::TextureDesc td{};
        td.width  = kShadowDim;
        td.height = kShadowDim;
        td.format = rhi::Format::D32_Float;
        td.usage  = static_cast<u32>(rhi::TextureUsage::DepthRenderTarget)
                  | static_cast<u32>(rhi::TextureUsage::Sampled);
        shadow_tex_ = dev.create_texture(td);
        if (pso_shadow_ == nullptr || shadow_tex_ == nullptr) {
            // Non-fatal: log once, run without shadows.
            cardinal::log::warnf("scene/renderer",
                "shadow resources unavailable (pso=%p tex=%p) — shadows off",
                static_cast<void*>(pso_shadow_.get()),
                static_cast<void*>(shadow_tex_.get()));
        }

        // Per-viewport scratch buffers — sized + grown lazily inside render().
        // We need ONE scratch per viewport per frame: the GPU may not have
        // executed viewport 0's draw yet by the time we record viewport 1's,
        // so overwriting a single buffer between calls produces "all viewports
        // show the LAST viewport's vertex data" (we hit this with
        // ViewMode::Heightmap winning every panel). Index by the swapchain's
        // active_viewport() at render time.
        scratch_default_capacity_ = 64 * 1024 * sizeof(RenderVertex);
        return true;
    }

    void set_light_set(const LightSet* lights) override { lights_ = lights; }

    FrameStats frame_stats() const override { return stats_; }

    void set_show_aabbs  (bool on) override { show_aabbs_   = on; }
    void set_show_frustum(bool on) override { show_frustum_ = on; }
    void set_show_axes   (bool on) override { show_axes_    = on; }
    void set_show_lights (bool on) override { show_lights_  = on; }

    void render(Scene& scene, ViewMode mode, float aspect) override {
        // Reset per-frame counters even on early-out paths so stale
        // numbers don't linger in the editor overlay.
        stats_ = FrameStats{};
        if (swapchain_ == nullptr) return;
        const Mat4 view = scene.camera().view();
        const Mat4 proj = scene.camera().proj(aspect > 0.001f ? aspect : 1.0f);
        const Mat4 vp   = proj * view;

        // Per-phase wall-clock timer. high_resolution_clock has ~ns
        // precision on Windows + Linux; we report µs so the panel stays
        // human-readable. Each phase computes its own elapsed_us into
        // stats_.<phase>_us at end of phase.
        using clk = std::chrono::high_resolution_clock;
        auto t_cull_begin = clk::now();
        auto elapsed_us = [](clk::time_point a, clk::time_point b) noexcept {
            return std::chrono::duration<f32, std::micro>(b - a).count();
        };

        // ---------------------------------------------------------------
        // Frustum cull pass — collect the visible-by-flag entities into
        // SoA arrays of world-space bounding-AABB min/max, then run
        // the SIMD-batched 6-plane reject test (16 AABBs / iter on
        // AVX-512). Phase 1 below skips entities whose visibility bit
        // is clear, so culled meshes never get an EntityWork emitted
        // and Phase 2 never transforms their verts.
        //
        // AABB-vs-frustum is significantly tighter than the older
        // sphere-vs-frustum reject — for a long-thin or rotated mesh
        // the bounding sphere has dead space the AABB doesn't, so
        // more off-screen entities get caught.
        // ---------------------------------------------------------------
        const cardinal::core::geom::Frustum frustum =
            cardinal::core::geom::Frustum::from_view_proj(vp);
        cull_candidates_.clear();
        cull_min_x_.clear(); cull_min_y_.clear(); cull_min_z_.clear();
        cull_max_x_.clear(); cull_max_y_.clear(); cull_max_z_.clear();
        cull_candidates_.reserve(scene.entities().size());
        cull_min_x_.reserve(scene.entities().size());
        cull_min_y_.reserve(scene.entities().size());
        cull_min_z_.reserve(scene.entities().size());
        cull_max_x_.reserve(scene.entities().size());
        cull_max_y_.reserve(scene.entities().size());
        cull_max_z_.reserve(scene.entities().size());

        for (const auto& e : scene.entities()) {
            if (!scene.is_entity_visible(e) || e.mesh == nullptr) continue;
            if (e.mesh->cpu_vertices() == nullptr) continue;
            Vec3 mn{}, mx{};
            entity_world_aabb(e, mn, mx);
            cull_candidates_.push_back(&e);
            cull_min_x_.push_back(mn.x); cull_max_x_.push_back(mx.x);
            cull_min_y_.push_back(mn.y); cull_max_y_.push_back(mx.y);
            cull_min_z_.push_back(mn.z); cull_max_z_.push_back(mx.z);
        }

        const usize cand_n = cull_candidates_.size();
        cull_bits_.assign((cand_n + 7) / 8, 0u);
        if (cand_n > 0) {
            // Frustum::planes is a flat array of 6 {Vec3 normal; f32 d;} —
            // exactly the (nx,ny,nz,d) layout the SIMD kernel wants.
            const f32* planes_flat =
                reinterpret_cast<const f32*>(&frustum.planes[0].normal.x);
            cardinal::core::simd::frustum_cull_aabbs(
                cull_bits_.data(), planes_flat,
                cull_min_x_.data(), cull_min_y_.data(), cull_min_z_.data(),
                cull_max_x_.data(), cull_max_y_.data(), cull_max_z_.data(),
                cand_n);
        }

        // Stats — `entities_total` includes meshless / hidden too via a
        // separate pass below, but `entities_drawn` and `entities_culled`
        // come from the cull-bit count.
        u32 drawn = 0;
        for (usize i = 0; i < cand_n; ++i) {
            if (cull_bits_[i >> 3] & (1u << (i & 7))) ++drawn;
        }
        stats_.entities_total   = static_cast<u32>(scene.entities().size());
        stats_.entities_drawn   = drawn;
        stats_.entities_culled  = static_cast<u32>(cand_n) - drawn;

        const auto t_phase1_begin = clk::now();
        stats_.cull_us = elapsed_us(t_cull_begin, t_phase1_begin);

        // ---------------------------------------------------------------
        // Phase 1 — visibility + sizing pass (sequential, fast).
        //
        // Walk the scene once: filter out invisible / mesh-less entities,
        // compute per-entity output offsets, total vertex count. Big
        // entities (terrain chunks at default 64×64 = ~7400 tris each)
        // get SPLIT into multiple work items so a single fat entity can't
        // peg one worker while 31 others idle. Each split keeps a
        // contiguous slice of cpu_verts_ — workers still write
        // synchronisation-free into their own slice.
        //
        // tri_index_global is derived from the entity's offset so the
        // Polygons view's per-triangle hue stays stable across runs.
        // ---------------------------------------------------------------
        const bool wire = (mode == ViewMode::Wireframe);
        const u32  verts_per_tri = wire ? 6u : 3u;

        // Triangle-count threshold to split an entity into sub-work-items.
        // 1024 tris × ~50 ns per tri (matrix-vec + lighting) ≈ ~50 µs of
        // worker time — well above the JobSystem dispatch overhead, well
        // below "hogs a whole frame on one core" territory. With 32
        // workers and a single 7400-tri terrain chunk we now produce 8
        // work items, fully utilising the perf tier.
        constexpr u32 kSplitTriThreshold = 1024;

        work_list_.clear();
        vgeom_cuts_.clear();
        usize total_out = 0;

        // We need a stable index for each vgeom cut so the EntityWork
        // can outlive the per-entity loop iteration without dangling
        // pointers into vgeom_cuts_'s internal buffer (we'd realloc).
        // Reserve up front to one per entity (upper bound).
        vgeom_cuts_.reserve(cand_n);

        // Walk the cull-candidates (the visible-by-flag set) instead
        // of every scene entity — this lets us index cull_bits_ by
        // candidate position. Entities that failed the cheap visibility
        // filter aren't in cull_candidates_ at all.
        for (usize ci = 0; ci < cand_n; ++ci) {
            // Skip frustum-culled entities entirely — they emit no
            // EntityWork and their verts never see the transform pass.
            if (!(cull_bits_[ci >> 3] & (1u << (ci & 7)))) continue;
            const Entity& e = *cull_candidates_[ci];

            // ----- VIRTUALIZED GEOMETRY PATH ---------------------------
            // When the mesh has an attached + enabled vgeom hierarchy
            // we run a per-frame selection that picks a subset of
            // clusters at the right LOD for the camera distance, and
            // emit ONE EntityWork per selected cluster. The "master"
            // CPU vertex buffer is bypassed entirely — we read from
            // hierarchy->vertices instead.
            //
            // Toggle is per-mesh (vgeom::is_enabled). When off, the
            // mesh falls through to the legacy path below.
            if (vgeom_enabled(*e.mesh)) {
                auto h = vgeom_hierarchy_of(*e.mesh);
                if (h != nullptr) {
                    vgeom::SelectInput si{};
                    si.hierarchy             = h.get();
                    si.model                 = e.transform.matrix();
                    si.view                  = view;
                    si.proj                  = proj;
                    // Pull viewport pixel height from the swapchain so
                    // far-camera meshes pick coarser LODs as the user
                    // resizes the panel down.
                    si.viewport_pixel_height = static_cast<f32>(
                        std::max<u32>(1u, swapchain_->viewport_height()));
                    si.pixel_error_tolerance = vgeom::kPixelErrorTolerance;

                    vgeom_cuts_.emplace_back();
                    vgeom::SelectOutput& so = vgeom_cuts_.back();
                    vgeom::select(si, so);

                    // Publish stats for the panel.
                    vgeom_publish_stats(*const_cast<Mesh*>(e.mesh.get()),
                                        so.stats);
                    // Roll up into the renderer's frame-stats aggregate so
                    // the RHI panel can show "vgeom: cut N tris vs M
                    // master tris" without reaching into per-mesh state.
                    stats_.vgeom_master_tris     += so.stats.master_tri_count;
                    stats_.vgeom_drawn_tris      += so.stats.drawn_tri_count;
                    stats_.vgeom_cut_clusters    += so.stats.cut_cluster_count;
                    stats_.vgeom_culled_clusters += so.stats.culled_cluster_count;

                    // Emit one EntityWork per selected cluster. Each
                    // cluster's vertices are vgeom::Vertex (layout-
                    // compatible with scene::Vertex — see static_assert
                    // in scene_vgeom.cpp). Cast at the boundary so the
                    // worker reads them as scene::Vertex without copy.
                    for (u32 cid : so.cluster_ids) {
                        const vgeom::Cluster& cl = h->clusters[cid];
                        const u32 tris = cl.vertex_count / 3;
                        if (tris == 0) continue;
                        EntityWork w{};
                        w.entity       = &e;
                        w.src_override = reinterpret_cast<const Vertex*>(
                                            &h->vertices[cl.vertex_offset]);
                        w.tri_first    = 0;
                        w.tri_count    = tris;
                        w.out_offset   = total_out;
                        w.tri_offset   = static_cast<u32>(total_out / verts_per_tri);
                        work_list_.push_back(w);
                        total_out += static_cast<usize>(tris) * verts_per_tri;
                    }
                    continue;   // skip legacy path for this entity
                }
            }

            // ----- LEGACY DIRECT PATH ----------------------------------
            const u32 vc         = e.mesh->vertex_count();
            const u32 tris_total = vc / 3;
            if (tris_total == 0) continue;

            // Split into sub-chunks of kSplitTriThreshold triangles each.
            // Single-chunk entities (cubes, spheres, gameplay actors)
            // fall through with one EntityWork like before — the
            // splitting only kicks in for terrain-scale meshes.
            for (u32 base = 0; base < tris_total; base += kSplitTriThreshold) {
                const u32 tris = std::min(kSplitTriThreshold, tris_total - base);
                EntityWork w{};
                w.entity     = &e;
                w.tri_first  = base;
                w.tri_count  = tris;
                w.out_offset = total_out;
                w.tri_offset = static_cast<u32>(total_out / verts_per_tri);
                work_list_.push_back(w);
                total_out += static_cast<usize>(tris) * verts_per_tri;
            }
        }

        stats_.work_items       = static_cast<u32>(work_list_.size());
        stats_.vertices_emitted = static_cast<u32>(total_out);

        const auto t_phase2_begin = clk::now();
        stats_.phase1_us = elapsed_us(t_phase1_begin, t_phase2_begin);

        if (total_out == 0) return;
        cpu_verts_.resize(total_out);

        // ---------------------------------------------------------------
        // Phase 2 — per-entity transform (parallel).
        //
        // Each entity owns a contiguous slice of cpu_verts_ — workers
        // never touch each other's slice, so no synchronisation. Falls
        // back to a serial loop when no JobSystem is bound (tests, early
        // boot, shutdown).
        //
        // The captured `vp` and `mode` are read-only across workers; the
        // per-entity model matrix is recomputed inside each task. Lights
        // are read by color_for() but never written.
        // ---------------------------------------------------------------
        auto transform_one = [&, this](u32 wi) {
            const EntityWork& w  = work_list_[wi];
            const Entity&     e  = *w.entity;
            // vgeom path supplies its own source slice (cluster verts);
            // legacy path reads from the mesh's CPU shadow.
            const Vertex*     src = (w.src_override != nullptr)
                ? w.src_override
                : e.mesh->cpu_vertices();
            const Mat4 model = e.transform.matrix();

            // World-space normal transform (uniform-scale heuristic). Used
            // for the COLOR computation only — the model-pos vertex output
            // gets its MVP applied in the vertex shader via push constant.
            auto rot_xform = [&](const Vec3& n) {
                Vec3 r{
                    model.m[0][0]*n.x + model.m[1][0]*n.y + model.m[2][0]*n.z,
                    model.m[0][1]*n.x + model.m[1][1]*n.y + model.m[2][1]*n.z,
                    model.m[0][2]*n.x + model.m[1][2]*n.y + model.m[2][2]*n.z,
                };
                const float l = std::sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
                if (l > 1e-6f) { r.x /= l; r.y /= l; r.z /= l; }
                return r;
            };

            usize out = w.out_offset;
            // Per-vertex work now: (1) world-space pos for color, (2) model
            // pos packed verbatim into the vertex buffer with w=1 so the
            // shader can multiply by push-constant MVP. The clip transform
            // moved to the GPU — that's the per-vertex MVP work the CPU
            // used to do. World pos still computed CPU-side because the
            // color path needs it for lighting.
            for (u32 t = 0; t < w.tri_count; ++t) {
                const u32 i = (w.tri_first + t) * 3;
                Vec3 wp[3]; Vec3 wn[3]; Vec3 col[3];
                for (int k = 0; k < 3; ++k) {
                    const Vertex& v = src[i + k];
                    // World-pos still needed for lighting + the Heightmap
                    // / Normals view modes that read positions. The shader
                    // reconstructs w=1 from the 12-byte vertex stream.
                    const Vec4 mpos4{ v.position.x, v.position.y, v.position.z, 1.0f };
                    const Vec4 wpos = model * mpos4;
                    wp[k]  = { wpos.x, wpos.y, wpos.z };
                    wn[k]  = rot_xform(v.normal);
                    col[k] = color_for(mode, wp[k], wn[k], v.color,
                                       w.tri_offset + t, e.tint, lights_);
                }
                // Pos + normal come straight from the source vertex
                // (model space). VS multiplies pos by MVP for clip,
                // normal by model 3x3 for world_normal varying. Color
                // is the CPU-computed base color (CPU lighting path
                // still active; GPU lighting comes next iteration).
                const Vertex& v0 = src[i + 0];
                const Vertex& v1 = src[i + 1];
                const Vertex& v2 = src[i + 2];
                if (wire) {
                    cpu_verts_[out++] = { v0.position, v0.normal, col[0] };
                    cpu_verts_[out++] = { v1.position, v1.normal, col[1] };
                    cpu_verts_[out++] = { v1.position, v1.normal, col[1] };
                    cpu_verts_[out++] = { v2.position, v2.normal, col[2] };
                    cpu_verts_[out++] = { v2.position, v2.normal, col[2] };
                    cpu_verts_[out++] = { v0.position, v0.normal, col[0] };
                } else {
                    cpu_verts_[out++] = { v0.position, v0.normal, col[0] };
                    cpu_verts_[out++] = { v1.position, v1.normal, col[1] };
                    cpu_verts_[out++] = { v2.position, v2.normal, col[2] };
                }
            }
        };

        const u32 work_count = static_cast<u32>(work_list_.size());
        if (cardinal::async::pool() != nullptr && work_count > 1) {
            // grain=0 lets parallel_for pick ~4 chunks per worker — that
            // amortises scheduling overhead while keeping work-stealing
            // available when one entity is much heavier than another.
            cardinal::async::parallel_for(0u, work_count, transform_one, /*grain*/ 0);
        } else {
            for (u32 i = 0; i < work_count; ++i) transform_one(i);
        }

        const auto t_sort_begin = clk::now();
        stats_.phase2_us = elapsed_us(t_phase2_begin, t_sort_begin);

        // Painter's-algorithm depth sort is gone: with model-space verts
        // there's no clip-space z to sort by, and both Vulkan + D3D12 now
        // ship a real depth attachment so the GPU handles ordering. If
        // a future swapchain drops the depth attachment we'd need to add
        // it back (sort per-entity by AABB centre, then within entity by
        // CPU-computed clip-z) — flagged for that day.

        const auto t_submit_begin = clk::now();
        stats_.sort_us = elapsed_us(t_sort_begin, t_submit_begin);

        // Pick the scratch slot for the viewport currently being drawn into.
        // The swapchain tracks "active viewport" — we mirror that index here
        // so each viewport's vertex stream lives in its own VRAM range and
        // doesn't get clobbered before the GPU has consumed it.
        const u32 vp_id = swapchain_->active_viewport();
        if (vp_id >= scratches_.size()) {
            scratches_.resize(static_cast<usize>(vp_id) + 1);
        }
        ScratchSlot& slot = scratches_[vp_id];

        const usize bytes = cpu_verts_.size() * sizeof(RenderVertex);
        if (bytes > slot.capacity) {
            // Grow geometrically; first alloc starts at the default size.
            slot.capacity = std::max(
                std::max(slot.capacity * 2, bytes),
                scratch_default_capacity_);
            rhi::BufferDesc bd{};
            bd.size = slot.capacity;
            bd.usage = static_cast<u32>(rhi::BufferUsage::Vertex);
            bd.cpu_writable = true;
            auto nb = device_->create_buffer(bd);
            if (nb == nullptr) return;
            slot.buffer = std::move(nb);
        }
        slot.buffer->upload(cpu_verts_.data(), bytes);

        // ---- Shadow pass 1: scene depth from the sun's POV ----------
        // Runs BEFORE any main-pass geometry so the swapchain can
        // suspend the viewport scope, render depth into shadow_tex_,
        // and resume the viewport cleanly. Triangle modes only (the
        // wire scratch is line-list); needs a Directional light + the
        // shadow resources (null on backends/GPUs that lack them).
        // PCF sampling in PSMain lands next — this iteration proves the
        // suspend → depth-render → resume scaffolding doesn't disturb
        // the main image. light_vp is also stashed for that next step.
        if (!wire && pso_shadow_ != nullptr && shadow_tex_ != nullptr
            && lights_ != nullptr && lights_->shadows_enabled()) {
            const scene::Light* sun = nullptr;
            for (const auto& L : lights_->lights()) {
                if (L.kind == LightKind::Directional) { sun = &L; break; }
            }
            if (sun != nullptr) {
                // Ortho box half-extent 16 covers the demo scene; the
                // helper places the light 0.5*far back along the sun
                // travel direction.
                light_vp_ = directional_light_vp(sun->direction,
                                                 16.0f, 0.05f, 64.0f);
                swapchain_->begin_shadow_pass(shadow_tex_.get());
                swapchain_->bind_pipeline(pso_shadow_.get());
                swapchain_->bind_vertex_buffer(slot.buffer.get(), 0);
                const Entity* last_e = nullptr;
                Mat4 lmvp{};
                for (const auto& w : work_list_) {
                    if (w.entity != last_e) {
                        lmvp = light_vp_ * w.entity->transform.matrix();
                        swapchain_->set_push_constants(0, &lmvp, sizeof(lmvp));
                        last_e = w.entity;
                    }
                    swapchain_->draw(w.tri_count * verts_per_tri,
                                     1, static_cast<u32>(w.out_offset), 0);
                }
                swapchain_->end_shadow_pass();
                shadow_ready_ = true;
            } else {
                shadow_ready_ = false;
            }
        } else {
            shadow_ready_ = false;
        }

        rhi::Pipeline* pso = (mode == ViewMode::Wireframe) ? pso_wire_.get() : pso_solid_.get();
        swapchain_->bind_pipeline(pso);
        swapchain_->bind_vertex_buffer(slot.buffer.get(), 0);

        // One draw per EntityWork — each gets its own MVP push. Splits
        // a few-hundred-entity scene into a few-hundred-draws stream;
        // modern GPUs eat that fine, and removing the per-vertex CPU
        // MVP multiply in transform_one() is the win.
        //
        // MVP changes only at ENTITY boundaries (split entities + the
        // morton-sorted vgeom clusters of a single entity all share
        // the same model matrix). Track the last entity and skip
        // push_constants + the matmul when the entity hasn't changed —
        // turns N redundant pushes into 1 per entity.
        // Push block — mirrors kSceneShader's cbuffer Push. Lighting
        // moved to the g_lights StructuredBuffer; the block now carries
        // only per-entity transform + scene ambient/mode + light count
        // & camera position. 160B — no longer at the 256B ceiling.
        struct PushBlock {
            Mat4 mvp;
            Mat4 model;
            Vec4 ambient_mode;                // xyz=ambient, w=mode strength
            Vec4 light_count_pad;             // x=count, yzw=camera world pos
            Vec4 material_pad;                // x=matidx, y=exposure, z=shadow
            Mat4 light_vp;                    // directional light-space VP
        };
        static_assert(sizeof(PushBlock) == 240,
                      "PushBlock must match kSceneShader Push cbuffer (240B)");

        // Build the per-frame light array. No fixed slot cap now — the
        // StructuredBuffer grows as needed, bounded only by kMaxLights
        // so a pathological LightSet can't blow the upload. Each kind
        // packs into GpuLight exactly as the shader's Light expects.
        constexpr u32 kMaxLights = 64;
        cpu_lights_.clear();
        Vec4 ambient_mode{0.05f, 0.05f, 0.06f, 0.0f};
        if (lights_ != nullptr) {
            const auto& amb = lights_->ambient();
            ambient_mode.x = amb.x; ambient_mode.y = amb.y; ambient_mode.z = amb.z;
            for (const auto& L : lights_->lights()) {
                if (cpu_lights_.size() >= kMaxLights) break;
                GpuLight g{};
                if (L.kind == LightKind::Directional) {
                    // direction points AT the scene; shader wants
                    // FROM-scene-TO-light → negate. range=0 = Directional.
                    g.dir_intensity = Vec4{-L.direction.x, -L.direction.y,
                                           -L.direction.z, L.intensity};
                    g.color_range   = Vec4{L.color.x, L.color.y, L.color.z, 0.0f};
                    g.spot          = Vec4{0, 0, 0, 0};
                } else if (L.kind == LightKind::Point) {
                    g.dir_intensity = Vec4{L.position.x, L.position.y,
                                           L.position.z, L.intensity};
                    g.color_range   = Vec4{L.color.x, L.color.y, L.color.z, L.range};
                    g.spot          = Vec4{0, 0, 0, 0};
                } else {  // Spot — Point packing + cone (dir + outer_cos).
                    // L.direction points FROM light TO target (engine
                    // convention) which is what the PS cone math expects.
                    // Floor outer_cos at 1e-3 so a fully-open cone still
                    // reads as a Spot rather than as a Point.
                    const float outer = std::max(L.spot_outer_cos, 1e-3f);
                    g.dir_intensity = Vec4{L.position.x, L.position.y,
                                           L.position.z, L.intensity};
                    g.color_range   = Vec4{L.color.x, L.color.y, L.color.z, L.range};
                    g.spot          = Vec4{L.direction.x, L.direction.y,
                                           L.direction.z, outer};
                }
                cpu_lights_.push_back(g);
            }
        }
        if (cpu_lights_.empty()) {
            // Fallback: one default directional so no-LightSet hosts
            // still see lit Solid mode.
            GpuLight g{};
            g.dir_intensity = Vec4{0.4f, 1.0f, -0.3f, 1.0f};
            g.color_range   = Vec4{1.0f, 1.0f, 1.0f, 0.0f};
            g.spot          = Vec4{0, 0, 0, 0};
            cpu_lights_.push_back(g);
        }
        ambient_mode.w = (mode == ViewMode::Solid) ? 1.0f : 0.0f;
        const u32 active_lights = static_cast<u32>(cpu_lights_.size());

        // Upload lights into this viewport's storage buffer + bind at
        // slot 0. Per-viewport ring (same multi-viewport hazard as the
        // vertex scratch: the GPU may still be reading viewport N's
        // light buffer when we record viewport N+1). bind_pipeline was
        // already issued above, so the descriptor/root binding lands on
        // the correct pipeline; it persists across the per-entity draw
        // loop (re-bound again for the gizmo pass after its own
        // bind_pipeline).
        if (vp_id >= light_scratches_.size()) {
            light_scratches_.resize(static_cast<usize>(vp_id) + 1);
        }
        ScratchSlot& lslot = light_scratches_[vp_id];
        const usize lbytes = cpu_lights_.size() * sizeof(GpuLight);
        if (lbytes > lslot.capacity) {
            lslot.capacity = std::max<usize>(
                std::max<usize>(lslot.capacity * 2, lbytes),
                static_cast<usize>(kMaxLights) * sizeof(GpuLight));
            rhi::BufferDesc bd{};
            bd.size         = lslot.capacity;
            bd.usage        = static_cast<u32>(rhi::BufferUsage::Storage);
            bd.cpu_writable = true;
            auto nb = device_->create_buffer(bd);
            if (nb != nullptr) lslot.buffer = std::move(nb);
        }
        if (lslot.buffer != nullptr) {
            lslot.buffer->upload(cpu_lights_.data(), lbytes);
            swapchain_->bind_storage_buffer(0, lslot.buffer.get());
        }

        // Per-entity material buffer (slot 1). One GpuMaterial per
        // distinct entity in work_list_ encounter order; the draw loop
        // below pushes the matching index. Defaults already equal the
        // shader's legacy specular constants, so entities that never
        // touch .material render exactly as before.
        cpu_materials_.clear();
        {
            const Entity* prev = nullptr;
            for (const auto& w : work_list_) {
                if (w.entity != prev) {
                    const auto& m = w.entity->material;
                    GpuMaterial gm{};
                    gm.spec = Vec4{m.specular_intensity, m.specular_power,
                                   m.roughness, m.metalness};
                    cpu_materials_.push_back(gm);
                    prev = w.entity;
                }
            }
        }
        if (cpu_materials_.empty()) {
            // Gizmo-only frame (no drawable entities) still needs a
            // valid slot-1 binding — the pipeline declares it.
            GpuMaterial def{};
            def.spec = Vec4{0.4f, 24.0f, 0.5f, 0.0f};
            cpu_materials_.push_back(def);
        }
        if (vp_id >= material_scratches_.size()) {
            material_scratches_.resize(static_cast<usize>(vp_id) + 1);
        }
        ScratchSlot& mslot = material_scratches_[vp_id];
        const usize mbytes = cpu_materials_.size() * sizeof(GpuMaterial);
        if (mbytes > mslot.capacity) {
            mslot.capacity = std::max<usize>(
                std::max<usize>(mslot.capacity * 2, mbytes),
                static_cast<usize>(64) * sizeof(GpuMaterial));
            rhi::BufferDesc bd{};
            bd.size         = mslot.capacity;
            bd.usage        = static_cast<u32>(rhi::BufferUsage::Storage);
            bd.cpu_writable = true;
            auto nb = device_->create_buffer(bd);
            if (nb != nullptr) mslot.buffer = std::move(nb);
        }
        if (mslot.buffer != nullptr) {
            mslot.buffer->upload(cpu_materials_.data(), mbytes);
            swapchain_->bind_storage_buffer(1, mslot.buffer.get());
        }
        // Sampled slot 0 = the pass-1 shadow map. Always bound when the
        // texture exists (the pipeline statically declares it); the
        // shadow TEST is gated separately by material_pad.z so wire /
        // no-directional frames just don't sample it.
        if (shadow_tex_ != nullptr) {
            swapchain_->bind_sampled_texture(0, shadow_tex_.get());
        }

        const Entity* last_entity = nullptr;
        int           mat_index   = -1;   // ++ per entity change → slot 1 idx
        PushBlock pc{};
        // Light-space VP from pass 1 + shadow-valid flag (material_pad.z).
        const float shadow_z = shadow_ready_ ? 1.0f : 0.0f;
        pc.light_vp = light_vp_;
        // Camera world position — PS needs it for the specular half-
        // vector. Pulled directly from scene::Camera (FlyCamera writes
        // to this each frame as the user navigates).
        const auto eye = scene.camera().position;
        // HDR exposure (material_pad.y) — scene-wide, from the bound
        // LightSet. Same value on every push; the PS guards 0 → 1.0.
        const float exposure =
            (lights_ != nullptr) ? lights_->exposure() : 1.0f;
        pc.ambient_mode    = ambient_mode;
        pc.light_count_pad = Vec4{static_cast<f32>(active_lights),
                                  eye.x, eye.y, eye.z};
        for (const auto& w : work_list_) {
            if (w.entity != last_entity) {
                ++mat_index;   // lockstep with the material pre-pass order
                pc.model        = w.entity->transform.matrix();
                pc.mvp          = vp * pc.model;
                pc.material_pad = Vec4{static_cast<f32>(mat_index),
                                       exposure, shadow_z, 0};
                swapchain_->set_push_constants(0, &pc, sizeof(pc));
                ++stats_.pc_pushes;
                last_entity = w.entity;
            }
            swapchain_->draw(w.tri_count * verts_per_tri,
                             1, static_cast<u32>(w.out_offset), 0);
            ++stats_.draw_calls;
        }

        // ---------------------------------------------------------------
        // Gizmo pass — wireframe AABBs + camera frustum overlay.
        //
        // Reuses pso_wire_ (LineList topology over RenderVertex). World-
        // space verts go straight into the buffer; we push `vp` as the
        // MVP so the existing shader does mul(vp, pos) which is correct
        // (model = identity for gizmos). Single extra draw + 1 push.
        // ---------------------------------------------------------------
        if (show_aabbs_ || show_frustum_ || show_axes_ || show_lights_) {
            gizmo_verts_.clear();

            if (show_axes_) {
                // 1-metre RGB axes at world origin. X=red, Y=green, Z=blue.
                // Standard editor convention; makes orientation obvious
                // when the scene is sparse + the frustum overlay is busy.
                const Vec3 nz{0, 1, 0};   // unused by wireframe PS
                gizmo_verts_.push_back({Vec3{0,0,0}, nz, Vec3{1.0f, 0.25f, 0.25f}});
                gizmo_verts_.push_back({Vec3{1,0,0}, nz, Vec3{1.0f, 0.25f, 0.25f}});
                gizmo_verts_.push_back({Vec3{0,0,0}, nz, Vec3{0.25f, 1.0f, 0.25f}});
                gizmo_verts_.push_back({Vec3{0,1,0}, nz, Vec3{0.25f, 1.0f, 0.25f}});
                gizmo_verts_.push_back({Vec3{0,0,0}, nz, Vec3{0.25f, 0.45f, 1.0f}});
                gizmo_verts_.push_back({Vec3{0,0,1}, nz, Vec3{0.25f, 0.45f, 1.0f}});
            }
            if (show_aabbs_) {
                // One AABB → 12 edges. Iterate cull_candidates_ which
                // holds per-entity min/max from the cull pre-pass plus
                // a parallel cull_bits_ visibility mask. Visible →
                // green; culled → red.
                const Vec3 col_visible{0.40f, 0.85f, 0.45f};
                const Vec3 col_culled {0.85f, 0.30f, 0.30f};
                for (usize i = 0; i < cull_candidates_.size(); ++i) {
                    const bool vis = (cull_bits_[i >> 3] & (1u << (i & 7))) != 0;
                    const Vec3& c = vis ? col_visible : col_culled;
                    const Vec3 lo{cull_min_x_[i], cull_min_y_[i], cull_min_z_[i]};
                    const Vec3 hi{cull_max_x_[i], cull_max_y_[i], cull_max_z_[i]};
                    add_aabb_edges(gizmo_verts_, lo, hi, c);
                }
            }
            if (show_frustum_) {
                add_frustum_edges(gizmo_verts_, vp,
                                  Vec3{0.95f, 0.95f, 0.95f});
            }
            if (show_lights_ && lights_ != nullptr) {
                // One wireframe per light, coloured to match the light's
                // emission tint so the gizmo and the actual lit pixels
                // share a colour cue:
                //   Directional → unit arrow at world origin, no position
                //   Point       → 3-axis "gyro" sphere, radius = range
                //   Spot        → cone, apex at position, length = range,
                //                 half-angle = acos(outer_cos)
                for (const auto& L : lights_->lights()) {
                    const Vec3 col{L.color.x, L.color.y, L.color.z};
                    switch (L.kind) {
                        case LightKind::Directional:
                            add_dir_arrow(gizmo_verts_,
                                          Vec3{0, 0, 0}, L.direction,
                                          1.5f, col);
                            break;
                        case LightKind::Point:
                            add_sphere_edges(gizmo_verts_,
                                             L.position, L.range,
                                             24, col);
                            break;
                        case LightKind::Spot: {
                            // outer_cos = cos(half-angle). cone base
                            // radius = length × tan(acos(outer_cos)) =
                            // length × sqrt(1-cos²)/cos. Clamp the
                            // cosine into (eps, 1) so a 0° or 180°
                            // degenerate cone doesn't divide by zero.
                            const float oc = std::clamp(L.spot_outer_cos, 1e-3f, 0.9999f);
                            const float sn = std::sqrt(1.0f - oc*oc);
                            const float br = L.range * (sn / oc);
                            add_cone_edges(gizmo_verts_,
                                           L.position, L.direction,
                                           L.range, br, 20, col);
                            break;
                        }
                    }
                }
            }

            if (!gizmo_verts_.empty()) {
                // Per-viewport scratch buffer for the gizmo line stream.
                if (vp_id >= gizmo_scratches_.size()) {
                    gizmo_scratches_.resize(static_cast<usize>(vp_id) + 1);
                }
                ScratchSlot& gslot = gizmo_scratches_[vp_id];
                const usize gbytes = gizmo_verts_.size() * sizeof(RenderVertex);
                if (gbytes > gslot.capacity) {
                    gslot.capacity = std::max(
                        std::max(gslot.capacity * 2, gbytes),
                        scratch_default_capacity_);
                    rhi::BufferDesc bd{};
                    bd.size = gslot.capacity;
                    bd.usage = static_cast<u32>(rhi::BufferUsage::Vertex);
                    bd.cpu_writable = true;
                    auto nb = device_->create_buffer(bd);
                    if (nb != nullptr) gslot.buffer = std::move(nb);
                }
                if (gslot.buffer != nullptr) {
                    gslot.buffer->upload(gizmo_verts_.data(), gbytes);
                    swapchain_->bind_pipeline(pso_wire_.get());
                    swapchain_->bind_vertex_buffer(gslot.buffer.get(), 0);
                    // Re-bind BOTH storage buffers: bind_pipeline above
                    // re-set the pipeline (on D3D12 that resets root
                    // args, so the SRVs must be set again; on Vulkan it's
                    // a harmless re-bind). The pipeline statically
                    // references g_lights AND g_materials, so both
                    // bindings must be valid even though gizmos render
                    // unlit (material_pad.x = 0 → a valid entry).
                    if (lslot.buffer != nullptr) {
                        swapchain_->bind_storage_buffer(0, lslot.buffer.get());
                    }
                    if (mslot.buffer != nullptr) {
                        swapchain_->bind_storage_buffer(1, mslot.buffer.get());
                    }
                    // pso_wire_ shares the scene shader, which statically
                    // references g_shadow — the sampled binding must be
                    // valid even though gizmos never sample it (material
                    // _pad.z = 0 below skips the shadow test entirely).
                    if (shadow_tex_ != nullptr) {
                        swapchain_->bind_sampled_texture(0, shadow_tex_.get());
                    }
                    // Gizmos are world-space + unlit → model = identity,
                    // mvp = vp, ambient_mode.w = 0 so the PS passes the
                    // input color straight through. light count = 0 so
                    // the PS light loop is skipped entirely; material
                    // index 0 is always valid (cpu_materials_ ≥ 1).
                    PushBlock gpc{};
                    gpc.mvp             = vp;
                    gpc.model           = Mat4::identity();
                    gpc.ambient_mode    = ambient_mode;
                    gpc.ambient_mode.w  = 0.0f;            // force unlit
                    gpc.light_count_pad = Vec4{0.0f, eye.x, eye.y, eye.z};
                    gpc.material_pad    = Vec4{0.0f, exposure, 0.0f, 0.0f};
                    gpc.light_vp        = Mat4::identity(); // unused (z=0)
                    swapchain_->set_push_constants(0, &gpc, sizeof(gpc));
                    ++stats_.pc_pushes;
                    swapchain_->draw(static_cast<u32>(gizmo_verts_.size()),
                                     1, 0, 0);
                    ++stats_.draw_calls;
                }
            }
        }

        stats_.submit_us = elapsed_us(t_submit_begin, clk::now());
    }

private:
    rhi::Device*    device_{nullptr};
    rhi::Swapchain* swapchain_{nullptr};

    std::unique_ptr<rhi::Pipeline> pso_solid_;
    std::unique_ptr<rhi::Pipeline> pso_wire_;
    // Shadow mapping (pass 1). pso_shadow_ is the depth-only pipeline
    // (kShadowShader, color_format=Unknown); shadow_tex_ is the D32
    // shadow map (DepthRenderTarget|Sampled). Both may be null if the
    // backend/GPU couldn't provide them — callers must null-check and
    // skip the shadow pass (main render is unaffected). The two-pass
    // render wiring + PCF sampling land in the next iteration.
    static constexpr cardinal::u32 kShadowDim = 2048;
    std::unique_ptr<rhi::Pipeline> pso_shadow_;
    std::unique_ptr<rhi::Texture>  shadow_tex_;
    // Light-space view-proj used for pass 1 this frame; consumed by the
    // PCF sample in PSMain (next iteration). shadow_ready_ = pass 1
    // actually ran (triangle mode + directional + resources present).
    Mat4                           light_vp_{};
    bool                           shadow_ready_{false};

    // One vertex scratch buffer per viewport — the GPU may not have consumed
    // viewport N's draw by the time we record viewport N+1, so they cannot
    // share storage. Indexed by Swapchain::active_viewport().
    struct ScratchSlot {
        std::unique_ptr<rhi::Buffer> buffer;
        usize                        capacity{0};
    };
    std::vector<ScratchSlot>       scratches_;
    usize                          scratch_default_capacity_{0};
    std::vector<RenderVertex>      cpu_verts_;

    // Per-viewport storage buffer holding the frame's GpuLight array
    // (bound at RHI storage slot 0). Same per-viewport ring rationale
    // as `scratches_` — viewport N's lights may still be in GPU flight
    // when viewport N+1 records. cpu_lights_ is the CPU staging vector,
    // reused across frames so the light upload doesn't churn the heap.
    std::vector<ScratchSlot>       light_scratches_;
    std::vector<GpuLight>          cpu_lights_;
    // Per-viewport material storage buffer (RHI slot 1) + CPU staging.
    // One GpuMaterial per distinct entity drawn this frame, in entity-
    // encounter order — the draw loop pushes the matching index. Same
    // per-viewport ring rationale as light_scratches_.
    std::vector<ScratchSlot>       material_scratches_;
    std::vector<GpuMaterial>       cpu_materials_;

    // Two-phase parallel transform — Phase 1 fills work_list_ sequentially
    // (cheap visibility cull + per-entity offset compute), Phase 2 fans out
    // to workers that each fill their own slice of cpu_verts_. Big
    // entities (terrain) get split into multiple EntityWorks so a single
    // fat mesh doesn't bottleneck on one worker.
    //
    // src_override: non-null when this work item draws from a vgeom
    // cluster's slice of the hierarchy vertex blob instead of from
    // entity.mesh->cpu_vertices(). When non-null, tri_first is 0 and
    // tri_count covers the whole cluster.
    struct EntityWork {
        const Entity* entity{nullptr};
        const Vertex* src_override{nullptr};
        u32           tri_first{0};    // first triangle in entity to process
        u32           tri_count{0};    // number of triangles in this work item
        usize         out_offset{0};   // index into cpu_verts_
        u32           tri_offset{0};   // global triangle id base (Polygons hue)
    };
    std::vector<EntityWork>        work_list_;
    // Per-frame scratch for vgeom selection — kept as a member so we
    // don't churn the heap each frame. One SelectOutput per vgeom-
    // enabled entity, reused across frames.
    std::vector<vgeom::SelectOutput> vgeom_cuts_;

    // SoA scratch for the per-frame frustum cull pass. Cleared and
    // refilled each render() call; capacity is retained so steady-state
    // entity counts don't churn the allocator. cull_bits_ is the packed
    // visibility mask the SIMD kernel writes — one bit per candidate.
    //
    // Six parallel min/max arrays per axis form the AABB SoA the
    // simd::frustum_cull_aabbs kernel wants. Caller fills these by
    // walking the candidate set once and calling entity_world_aabb.
    std::vector<const Entity*>  cull_candidates_;
    std::vector<f32>            cull_min_x_, cull_min_y_, cull_min_z_;
    std::vector<f32>            cull_max_x_, cull_max_y_, cull_max_z_;
    std::vector<u8>             cull_bits_;

    // Most recent frame_stats() snapshot. Reset at the top of render()
    // so even early-out frames produce a clean reading. Read by the
    // host via the public frame_stats() accessor.
    FrameStats                  stats_{};

    // Optional lighting + ambient. The host populates a LightSet each frame
    // (typically by mirroring actor::LightComponents) and points us at it
    // via set_light_set(). When null, color_for falls back to the cheap
    // built-in directional light.
    const LightSet*                lights_{nullptr};

    // Editor gizmo state. show_aabbs_ overlays a wireframe for every
    // entity world-AABB (green = visible, red = culled); show_frustum_
    // overlays the camera frustum. Cleared/built per render() into
    // gizmo_verts_; uploaded to per-viewport scratch slots so multi-
    // viewport rendering doesn't fight over the same vertex memory.
    bool                           show_aabbs_{false};
    bool                           show_frustum_{false};
    bool                           show_axes_{false};
    bool                           show_lights_{false};
    std::vector<RenderVertex>      gizmo_verts_;
    std::vector<ScratchSlot>       gizmo_scratches_;
};

}  // namespace

std::unique_ptr<ForwardRenderer> ForwardRenderer::create(rhi::Device& dev, rhi::Swapchain& sw) {
    auto r = std::make_unique<ForwardRendererImpl>();
    if (!r->initialize(dev, sw)) return nullptr;
    return r;
}

}  // namespace cardinal::scene
