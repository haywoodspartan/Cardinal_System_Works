#include <cardinal/render/gpu_ssgi.hpp>
#include <cardinal/render/gpu_visbuf.hpp>   // kInvalidPrimId

#include <cardinal/core/std/cmath.hpp>
#include <cardinal/core/std/cstring.hpp>   // cardinal::memcpy
#include <cardinal/core/std/algorithm.hpp>
#include <cardinal/core/std/utility.hpp>

namespace cardinal::render::gpu {

namespace rg = cardinal::render::graph;

// ============================================================================
// Sample weighting — the one function that decides the look.
// ============================================================================
float ssgi_sample_weight(float dist, float cos_theta, float radius) noexcept {
    // TODO(human): design the per-sample weight.
    //
    // Inputs are already filtered: 0 < dist <= radius and cos_theta is above
    // kSsgiTangentBias (the sample rises out of the surface's tangent plane).
    // The value you return scales BOTH the occlusion this sample contributes
    // to the ambient term and the bounce radiance it contributes. The HLSL
    // kernel in hlsl_source() must mirror whatever you write here, operation
    // for operation (it is ported after you decide).
    //
    // Placeholder so the pipeline validates end-to-end: a linear distance
    // falloff under a cosine lobe.
    const float t = dist / radius;          // 0..1 inside the gather radius
    return cos_theta * (1.0f - t);
}

namespace {

inline bool  isf(float v) noexcept { return cardinal::isfinite(v); }
inline float fzf(float v, float fb) noexcept { return isf(v) ? v : fb; }
inline float sat(float v) noexcept {
    if (!isf(v)) return 0.0f;
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}
inline void put_u32(cardinal::vector<cardinal::u8>& v, cardinal::u32 x) {
    v.push_back(static_cast<cardinal::u8>(x & 0xFFu));
    v.push_back(static_cast<cardinal::u8>((x >> 8) & 0xFFu));
    v.push_back(static_cast<cardinal::u8>((x >> 16) & 0xFFu));
    v.push_back(static_cast<cardinal::u8>((x >> 24) & 0xFFu));
}
inline void put_f32(cardinal::vector<cardinal::u8>& v, float f) {
    cardinal::u32 x;
    static_assert(sizeof(x) == sizeof(f), "f32 must be 32 bits");
    cardinal::memcpy(&x, &f, sizeof(x));
    put_u32(v, x);
}

// Inverse of gpu_visbuf's oct_pack — identical sequence to the resolve's copy
// so the normal this pass sees is bit-for-bit the normal the resolve shaded.
inline void oct_decode(cardinal::u32 packed, float& nx, float& ny, float& nz) noexcept {
    const cardinal::i16 ix = static_cast<cardinal::i16>(packed & 0xFFFFu);
    const cardinal::i16 iz = static_cast<cardinal::i16>(packed >> 16);
    float ox = static_cast<float>(ix) / 32767.0f;
    float oz = static_cast<float>(iz) / 32767.0f;
    const float oy = 1.0f - cardinal::abs(ox) - cardinal::abs(oz);
    if (oy < 0.0f) {
        const float tx = (1.0f - cardinal::abs(oz)) * (ox >= 0.0f ? 1.0f : -1.0f);
        const float tz = (1.0f - cardinal::abs(ox)) * (oz >= 0.0f ? 1.0f : -1.0f);
        ox = tx; oz = tz;
    }
    const float l2 = ox * ox + oy * oy + oz * oz;
    if (l2 > 1.0e-12f) {
        const float inv = 1.0f / cardinal::sqrt(l2);
        nx = ox * inv; ny = oy * inv; nz = oz * inv;
    } else {
        nx = 0.0f; ny = 1.0f; nz = 0.0f;
    }
}

// Compass directions, integer pixel deltas. Index order is the rotation
// order; a per-pixel hash picks the starting index so neighbouring pixels
// sample different direction sets (classic interleaved pattern).
constexpr int kDirX[kSsgiDirections] = { 1, 1, 0, -1, -1, -1,  0,  1 };
constexpr int kDirY[kSsgiDirections] = { 0, 1, 1,  1,  0, -1, -1, -1 };

void ssgi_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<ScreenSpaceGIPass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height, MC = st->material_count;
    if (W == 0 || H == 0) { ec.dispatch(0, 1, 1); return; }

    const float*         depth = static_cast<const float*>(ec.map_buffer_read(st->in_depth));
    const cardinal::u32* prim  = static_cast<const cardinal::u32*>(ec.map_buffer_read(st->in_prim_id));
    const cardinal::u32* norm  = static_cast<const cardinal::u32*>(ec.map_buffer_read(st->in_normal));
    const cardinal::u32* matid = static_cast<const cardinal::u32*>(ec.map_buffer_read(st->in_mat_id));
    const float*         mats  = static_cast<const float*>(ec.map_buffer_read(st->in_materials));
    const float*         rad   = static_cast<const float*>(ec.map_buffer_read(st->in_radiance));
    const float*         ivp_in= static_cast<const float*>(ec.map_buffer_read(st->in_inv_view_proj));
    const float*         amb   = static_cast<const float*>(ec.map_buffer_read(st->in_ambient));
    float*               out   = static_cast<float*>(ec.map_buffer_write(st->out_radiance));
    float*               out_ao= static_cast<float*>(ec.map_buffer_write(st->out_ao));
    if (!depth || !prim || !norm || !matid || !mats || !rad || !ivp_in || !amb || !out || !out_ao) {
        ec.dispatch(0, 1, 1); return;
    }

    // Parameters, NaN-defended + clamped exactly as the kernel's push block is.
    const SsgiParams& p = st->params;
    cardinal::u32 steps = p.steps < 1u ? 1u : (p.steps > kSsgiMaxSteps ? kSsgiMaxSteps : p.steps);
    cardinal::u32 bstep = p.base_step_px < 1u ? 1u : p.base_step_px;
    float radius = fzf(p.radius, 1.0f);
    if (radius < 1.0e-4f) radius = 1.0e-4f;
    const float r2        = radius * radius;
    const float intensity = fzf(p.intensity, 1.0f) < 0.0f ? 0.0f : fzf(p.intensity, 1.0f);
    const float ao_str    = sat(p.ao_strength);
    const float total     = static_cast<float>(kSsgiDirections * steps);

    float ivp[16];
    for (int k = 0; k < 16; ++k) ivp[k] = fzf(ivp_in[k], (k % 5 == 0) ? 1.0f : 0.0f);
    const float ar = fzf(amb[0], 0.0f), ag = fzf(amb[1], 0.0f), ab = fzf(amb[2], 0.0f);

    // Pixel centre -> NDC -> inverse view-proj (row-major). Same op order as
    // the resolve so P here equals the P it shaded.
    auto reconstruct = [&](cardinal::u32 px, cardinal::u32 py, cardinal::u32 i,
                           float& wx, float& wy, float& wz) {
        const float ndc_x = (2.0f * (static_cast<float>(px) + 0.5f) -
                             static_cast<float>(W)) / static_cast<float>(W);
        const float ndc_y = (static_cast<float>(H) -
                             2.0f * (static_cast<float>(py) + 0.5f)) / static_cast<float>(H);
        const float ndc_z = 2.0f * depth[i] - 1.0f;
        wx = ivp[0]  * ndc_x + ivp[1]  * ndc_y + ivp[2]  * ndc_z + ivp[3];
        wy = ivp[4]  * ndc_x + ivp[5]  * ndc_y + ivp[6]  * ndc_z + ivp[7];
        wz = ivp[8]  * ndc_x + ivp[9]  * ndc_y + ivp[10] * ndc_z + ivp[11];
        const float ww = ivp[12] * ndc_x + ivp[13] * ndc_y + ivp[14] * ndc_z + ivp[15];
        if (cardinal::abs(ww) > 1.0e-8f) { wx /= ww; wy /= ww; wz /= ww; }
    };

    cardinal::u32 shaded = 0, sky = 0;
    cardinal::u64 taps = 0;
    double ao_sum = 0.0;

    const cardinal::usize n = static_cast<cardinal::usize>(W) * H;
    for (cardinal::usize i = 0; i < n; ++i) {
        const cardinal::u32 px = static_cast<cardinal::u32>(i % W);
        const cardinal::u32 py = static_cast<cardinal::u32>(i / W);
        if (prim[i] == kInvalidPrimId) {
            out[i * 3 + 0] = rad[i * 3 + 0];
            out[i * 3 + 1] = rad[i * 3 + 1];
            out[i * 3 + 2] = rad[i * 3 + 2];
            out_ao[i] = 1.0f;
            ++sky;
            continue;
        }
        float Px, Py, Pz;
        reconstruct(px, py, static_cast<cardinal::u32>(i), Px, Py, Pz);
        float nx, ny, nz;
        oct_decode(norm[i], nx, ny, nz);

        // Integer per-pixel jitter: direction rotation + step phase.
        const cardinal::u32 rot   = ((px * 7u) + (py * 13u) + ((px ^ py) * 3u)) & 7u;
        const cardinal::u32 phase = ((px * 3u) + (py * 5u)) & 3u;

        float occ = 0.0f, ir = 0.0f, ig = 0.0f, ib = 0.0f;
        for (cardinal::u32 d = 0; d < kSsgiDirections; ++d) {
            const cardinal::u32 dir = (d + rot) & 7u;
            const int dx = kDirX[dir], dy = kDirY[dir];
            for (cardinal::u32 k = 0; k < steps; ++k) {
                const int off = static_cast<int>(bstep * (k + 1u) + phase);
                const int qx  = static_cast<int>(px) + dx * off;
                const int qy  = static_cast<int>(py) + dy * off;
                // Offsets grow monotonically, so leaving the viewport ends
                // this direction (no later step can come back inside).
                if (qx < 0 || qy < 0 || qx >= static_cast<int>(W) || qy >= static_cast<int>(H)) break;
                const cardinal::u32 q = static_cast<cardinal::u32>(qy) * W + static_cast<cardinal::u32>(qx);
                if (prim[q] == kInvalidPrimId) continue;          // sky: nothing to gather
                float Qx, Qy, Qz;
                reconstruct(static_cast<cardinal::u32>(qx), static_cast<cardinal::u32>(qy), q, Qx, Qy, Qz);
                const float vx = Qx - Px, vy = Qy - Py, vz = Qz - Pz;
                const float d2 = vx * vx + vy * vy + vz * vz;
                if (!(d2 > 1.0e-12f) || d2 > r2) continue;       // NaN-safe: NaN compares false
                const float dist  = cardinal::sqrt(d2);
                const float cos_t = (nx * vx + ny * vy + nz * vz) / dist;
                if (cos_t <= kSsgiTangentBias) continue;          // in / below the tangent plane
                const float w = ssgi_sample_weight(dist, cos_t, radius);
                if (!(w > 0.0f)) continue;                        // NaN / negative weights are dropped
                occ += w;
                ir  += rad[q * 3 + 0] * w;
                ig  += rad[q * 3 + 1] * w;
                ib  += rad[q * 3 + 2] * w;
                ++taps;
            }
        }
        float ao = 1.0f - ao_str * (occ / total);
        ao = sat(ao);

        const cardinal::u32 mid = matid[i];
        const float* m = (mid < MC) ? (mats + mid * 8) : nullptr;
        const float base_r = m ? fzf(m[0], 0.5f) : 0.5f;
        const float base_g = m ? fzf(m[1], 0.5f) : 0.5f;
        const float base_b = m ? fzf(m[2], 0.5f) : 0.5f;
        const float metal  = m ? sat(fzf(m[4], 0.0f)) : 0.0f;
        const float diffuse = (1.0f - metal) * intensity / total;

        out[i * 3 + 0] = rad[i * 3 + 0] + base_r * ar * (ao - 1.0f) + base_r * diffuse * ir;
        out[i * 3 + 1] = rad[i * 3 + 1] + base_g * ag * (ao - 1.0f) + base_g * diffuse * ig;
        out[i * 3 + 2] = rad[i * 3 + 2] + base_b * ab * (ao - 1.0f) + base_b * diffuse * ib;
        out_ao[i] = ao;
        ao_sum += ao;
        ++shaded;
    }
    st->pixels_shaded = shaded;
    st->pixels_sky    = sky;
    st->taps_accepted = taps;
    st->mean_ao       = shaded ? static_cast<float>(ao_sum / static_cast<double>(shaded)) : 1.0f;
    ec.dispatch((W + 7) / 8, (H + 7) / 8, 1);
}

}  // namespace

cardinal::shared_ptr<ScreenSpaceGIPass::State> ScreenSpaceGIPass::add_to_graph(
    rg::Graph& g,
    rg::ResourceHandle in_depth, rg::ResourceHandle in_prim, rg::ResourceHandle in_norm,
    rg::ResourceHandle in_mat, rg::ResourceHandle in_mats, rg::ResourceHandle in_rad,
    rg::ResourceHandle in_ivp, rg::ResourceHandle in_amb,
    cardinal::u32 W, cardinal::u32 H, cardinal::u32 MC, const SsgiParams& params)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_depth = in_depth; st->in_prim_id = in_prim; st->in_normal = in_norm;
    st->in_mat_id = in_mat;  st->in_materials = in_mats; st->in_radiance = in_rad;
    st->in_inv_view_proj = in_ivp; st->in_ambient = in_amb;
    st->width = W; st->height = H; st->material_count = MC;
    st->params = params;

    rg::BufferDesc od; od.name = "ssgi.rad";
    od.size_bytes = static_cast<cardinal::usize>(W) * H * 3 * sizeof(float);
    od.stride_bytes = sizeof(float);
    st->out_radiance = g.declare_buffer(od);
    rg::BufferDesc ad; ad.name = "ssgi.ao";
    ad.size_bytes = static_cast<cardinal::usize>(W) * H * sizeof(float);
    ad.stride_bytes = sizeof(float);
    st->out_ao = g.declare_buffer(ad);

    // Kernel-side clamps mirror ssgi_record's so the push block IS the CPU's view.
    const cardinal::u32 steps = params.steps < 1u ? 1u
                              : (params.steps > kSsgiMaxSteps ? kSsgiMaxSteps : params.steps);
    const cardinal::u32 bstep = params.base_step_px < 1u ? 1u : params.base_step_px;
    float radius = fzf(params.radius, 1.0f);
    if (radius < 1.0e-4f) radius = 1.0e-4f;
    const float intensity = fzf(params.intensity, 1.0f) < 0.0f ? 0.0f : fzf(params.intensity, 1.0f);
    const float ao_str    = sat(params.ao_strength);

    rg::PassDesc pd;
    pd.name = "ScreenSpaceGIPass"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_depth,         rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_prim,          rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{in_norm,          rg::AccessMode::Read,  2});
    pd.accesses.push_back(rg::ResourceAccess{in_mat,           rg::AccessMode::Read,  3});
    pd.accesses.push_back(rg::ResourceAccess{in_mats,          rg::AccessMode::Read,  4});
    pd.accesses.push_back(rg::ResourceAccess{in_rad,           rg::AccessMode::Read,  5});
    pd.accesses.push_back(rg::ResourceAccess{in_ivp,           rg::AccessMode::Read,  6});
    pd.accesses.push_back(rg::ResourceAccess{in_amb,           rg::AccessMode::Read,  7});
    pd.accesses.push_back(rg::ResourceAccess{st->out_radiance, rg::AccessMode::Write, 8});
    pd.accesses.push_back(rg::ResourceAccess{st->out_ao,       rg::AccessMode::Write, 9});
    pd.record = ssgi_record; pd.user_ctx = st.get();
    pd.dispatch_x = (W + 7) / 8; pd.dispatch_y = (H + 7) / 8; pd.dispatch_z = 1;
    pd.compute_hlsl = ScreenSpaceGIPass::hlsl_source();
    // b0 = { uint w, h, matCount, steps; float radius, intensity, aoStrength; uint baseStep }
    pd.push_constant_size = 32;
    pd.push_constants.clear();
    put_u32(pd.push_constants, W);
    put_u32(pd.push_constants, H);
    put_u32(pd.push_constants, MC);
    put_u32(pd.push_constants, steps);
    put_f32(pd.push_constants, radius);
    put_f32(pd.push_constants, intensity);
    put_f32(pd.push_constants, ao_str);
    put_u32(pd.push_constants, bstep);
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* ScreenSpaceGIPass::hlsl_source() noexcept {
    return
    "// Cardinal - ScreenSpaceGIPass.hlsl - screen-space AO + one-bounce diffuse.\n"
    "// One thread per pixel. Operation-for-operation mirror of ssgi_record:\n"
    "// integer compass marches (same pixels as the CPU), world positions via the\n"
    "// resolve's inverse view-proj, tangent-plane bias, ssgi_sample_weight port.\n"
    "ByteAddressBuffer   InDepth     : register(t0);  // f32 per pixel\n"
    "ByteAddressBuffer   InPrim      : register(t1);  // u32 per pixel\n"
    "ByteAddressBuffer   InNormal    : register(t2);  // oct snorm16x2 per pixel\n"
    "ByteAddressBuffer   InMatId     : register(t3);  // u32 per pixel\n"
    "ByteAddressBuffer   InMaterials : register(t4);  // 8 floats per material\n"
    "ByteAddressBuffer   InRadiance  : register(t5);  // float3 per pixel\n"
    "ByteAddressBuffer   InInvVP     : register(t6);  // 16 floats row-major\n"
    "ByteAddressBuffer   InAmbient   : register(t7);  // 3 floats\n"
    "RWByteAddressBuffer OutRadiance : register(u0);  // float3 per pixel\n"
    "RWByteAddressBuffer OutAO       : register(u1);  // f32 per pixel\n"
    "struct PushT { uint w; uint h; uint matCount; uint steps;\n"
    "               float radius; float intensity; float aoStrength; uint baseStep; };\n"
    "[[vk::push_constant]] ConstantBuffer<PushT> pc : register(b0);\n"
    "#define gW pc.w\n"
    "#define gH pc.h\n"
    "#define gMatCount pc.matCount\n"
    "#define gSteps pc.steps\n"
    "#define gRadius pc.radius\n"
    "#define gIntensity pc.intensity\n"
    "#define gAoStrength pc.aoStrength\n"
    "#define gBaseStep pc.baseStep\n"
    "#define INVALID_PRIM 0xFFFFFFFFu\n"
    "#define TANGENT_BIAS 0.05\n"
    "static const int kDirX[8] = { 1, 1, 0, -1, -1, -1,  0,  1 };\n"
    "static const int kDirY[8] = { 0, 1, 1,  1,  0, -1, -1, -1 };\n"
    "float fz(float v, float fb) { return isfinite(v) ? v : fb; }\n"
    "float sat1(float v) { return isfinite(v) ? clamp(v, 0.0, 1.0) : 0.0; }\n"
    "float3 oct_decode(uint p) {\n"
    "  int ix = (int)(p << 16) >> 16;\n"
    "  int iz = (int)p >> 16;\n"
    "  float ox = (float)ix / 32767.0;\n"
    "  float oz = (float)iz / 32767.0;\n"
    "  float oy = 1.0 - abs(ox) - abs(oz);\n"
    "  if (oy < 0.0) {\n"
    "    float tx = (1.0 - abs(oz)) * (ox >= 0.0 ? 1.0 : -1.0);\n"
    "    float tz = (1.0 - abs(ox)) * (oz >= 0.0 ? 1.0 : -1.0);\n"
    "    ox = tx; oz = tz;\n"
    "  }\n"
    "  float l2 = ox*ox + oy*oy + oz*oz;\n"
    "  if (l2 > 1.0e-12) { float inv = 1.0/sqrt(l2); return float3(ox,oy,oz)*inv; }\n"
    "  return float3(0,1,0);\n"
    "}\n"
    "// Port of ssgi_sample_weight (keep in step with gpu_ssgi.cpp).\n"
    "float sample_weight(float dist, float cos_t, float radius) {\n"
    "  float t = dist / radius;\n"
    "  return cos_t * (1.0 - t);\n"
    "}\n"
    "float3 reconstruct(uint px, uint py, uint i) {\n"
    "  float ndc_x = (2.0*((float)px + 0.5) - (float)gW) / (float)gW;\n"
    "  float ndc_y = ((float)gH - 2.0*((float)py + 0.5)) / (float)gH;\n"
    "  float ndc_z = 2.0*asfloat(InDepth.Load(i*4u)) - 1.0;\n"
    "  float4 r0 = asfloat(InInvVP.Load4(0));\n"
    "  float4 r1 = asfloat(InInvVP.Load4(16));\n"
    "  float4 r2 = asfloat(InInvVP.Load4(32));\n"
    "  float4 r3 = asfloat(InInvVP.Load4(48));\n"
    "  r0 = float4(fz(r0.x,1.0), fz(r0.y,0.0), fz(r0.z,0.0), fz(r0.w,0.0));\n"
    "  r1 = float4(fz(r1.x,0.0), fz(r1.y,1.0), fz(r1.z,0.0), fz(r1.w,0.0));\n"
    "  r2 = float4(fz(r2.x,0.0), fz(r2.y,0.0), fz(r2.z,1.0), fz(r2.w,0.0));\n"
    "  r3 = float4(fz(r3.x,0.0), fz(r3.y,0.0), fz(r3.z,0.0), fz(r3.w,1.0));\n"
    "  float3 nv = float3(ndc_x, ndc_y, ndc_z);\n"
    "  float wx = dot(r0.xyz, nv) + r0.w;\n"
    "  float wy = dot(r1.xyz, nv) + r1.w;\n"
    "  float wz = dot(r2.xyz, nv) + r2.w;\n"
    "  float ww = dot(r3.xyz, nv) + r3.w;\n"
    "  if (abs(ww) > 1.0e-8) { wx /= ww; wy /= ww; wz /= ww; }\n"
    "  return float3(wx, wy, wz);\n"
    "}\n"
    "[numthreads(8,8,1)]\n"
    "void CSMain(uint3 tid : SV_DispatchThreadID) {\n"
    "  uint px = tid.x, py = tid.y;\n"
    "  if (px >= gW || py >= gH) return;\n"
    "  uint i = py*gW + px;\n"
    "  float3 rad = asfloat(InRadiance.Load3(i*12u));\n"
    "  if (InPrim.Load(i*4u) == INVALID_PRIM) {\n"
    "    OutRadiance.Store3(i*12u, asuint(rad));\n"
    "    OutAO.Store(i*4u, asuint(1.0));\n"
    "    return;\n"
    "  }\n"
    "  uint  steps  = clamp(gSteps, 1u, 8u);\n"
    "  uint  bstep  = max(gBaseStep, 1u);\n"
    "  float radius = max(fz(gRadius, 1.0), 1.0e-4);\n"
    "  float r2     = radius * radius;\n"
    "  float inten  = max(fz(gIntensity, 1.0), 0.0);\n"
    "  float aoStr  = sat1(gAoStrength);\n"
    "  float total  = (float)(8u * steps);\n"
    "  float3 amb = asfloat(InAmbient.Load3(0));\n"
    "  amb = float3(fz(amb.x,0.0), fz(amb.y,0.0), fz(amb.z,0.0));\n"
    "  float3 P = reconstruct(px, py, i);\n"
    "  float3 N = oct_decode(InNormal.Load(i*4u));\n"
    "  uint rot   = ((px * 7u) + (py * 13u) + ((px ^ py) * 3u)) & 7u;\n"
    "  uint phase = ((px * 3u) + (py * 5u)) & 3u;\n"
    "  float occ = 0.0; float3 ind = float3(0,0,0);\n"
    "  [loop] for (uint d = 0u; d < 8u; ++d) {\n"
    "    uint dir = (d + rot) & 7u;\n"
    "    int dx = kDirX[dir], dy = kDirY[dir];\n"
    "    [loop] for (uint k = 0u; k < steps; ++k) {\n"
    "      int off = (int)(bstep * (k + 1u) + phase);\n"
    "      int qx = (int)px + dx * off;\n"
    "      int qy = (int)py + dy * off;\n"
    "      if (qx < 0 || qy < 0 || qx >= (int)gW || qy >= (int)gH) break;\n"
    "      uint q = (uint)qy * gW + (uint)qx;\n"
    "      if (InPrim.Load(q*4u) == INVALID_PRIM) continue;\n"
    "      float3 Q = reconstruct((uint)qx, (uint)qy, q);\n"
    "      float3 v = Q - P;\n"
    "      float d2 = v.x*v.x + v.y*v.y + v.z*v.z;\n"
    "      if (!(d2 > 1.0e-12) || d2 > r2) continue;\n"
    "      float dist  = sqrt(d2);\n"
    "      float cos_t = (N.x*v.x + N.y*v.y + N.z*v.z) / dist;\n"
    "      if (cos_t <= TANGENT_BIAS) continue;\n"
    "      float w = sample_weight(dist, cos_t, radius);\n"
    "      if (!(w > 0.0)) continue;\n"
    "      occ += w;\n"
    "      ind += asfloat(InRadiance.Load3(q*12u)) * w;\n"
    "    }\n"
    "  }\n"
    "  float ao = sat1(1.0 - aoStr * (occ / total));\n"
    "  uint mid = InMatId.Load(i*4u);\n"
    "  float3 base = float3(0.5, 0.5, 0.5); float metal = 0.0;\n"
    "  if (mid < gMatCount) {\n"
    "    float3 m = asfloat(InMaterials.Load3(mid*32u));\n"
    "    base  = float3(fz(m.x,0.5), fz(m.y,0.5), fz(m.z,0.5));\n"
    "    metal = sat1(fz(asfloat(InMaterials.Load(mid*32u+16u)), 0.0));\n"
    "  }\n"
    "  float diffuse = (1.0 - metal) * inten / total;\n"
    "  float3 o;\n"
    "  o.x = rad.x + base.x * amb.x * (ao - 1.0) + base.x * diffuse * ind.x;\n"
    "  o.y = rad.y + base.y * amb.y * (ao - 1.0) + base.y * diffuse * ind.y;\n"
    "  o.z = rad.z + base.z * amb.z * (ao - 1.0) + base.z * diffuse * ind.z;\n"
    "  OutRadiance.Store3(i*12u, asuint(o));\n"
    "  OutAO.Store(i*4u, asuint(ao));\n"
    "}\n";
}

}  // namespace cardinal::render::gpu
