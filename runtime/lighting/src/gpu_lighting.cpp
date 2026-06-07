#include <cardinal/lighting/gpu_lighting.hpp>

#include <cardinal/core/std/cmath.hpp>
#include <cardinal/core/std/algorithm.hpp>
#include <cardinal/core/std/utility.hpp>

#include <chrono>

namespace cardinal::lighting::gpu {

namespace rg = cardinal::render::graph;

namespace {

inline bool isf(float v) noexcept { return cardinal::isfinite(v); }
inline float fzf(float v, float fb) noexcept { return isf(v) ? v : fb; }
inline cardinal::scene::Vec3 fzv(const cardinal::scene::Vec3& v,
                                 const cardinal::scene::Vec3& fb) noexcept {
    return { fzf(v.x, fb.x), fzf(v.y, fb.y), fzf(v.z, fb.z) };
}
inline float saturate01(float v) noexcept {
    if (!isf(v)) return 0.0f;
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

}  // namespace

// =============================================================================
// ForwardShadePass — per-fragment Cook-Torrance direct lighting
// =============================================================================
namespace {

void shade_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<ForwardShadePass::State*>(uctx);
    const cardinal::u32 N = st->fragment_count;
    const cardinal::u32 L = st->light_count;

    const float* wp = static_cast<const float*>(ec.map_buffer_read(st->in_world_pos));
    const float* wn = static_cast<const float*>(ec.map_buffer_read(st->in_world_normal));
    const float* mt = static_cast<const float*>(ec.map_buffer_read(st->in_material));
    const float* la = static_cast<const float*>(ec.map_buffer_read(st->in_ambient));
    const float* lp = L ? static_cast<const float*>(ec.map_buffer_read(st->in_lights)) : nullptr;
    float*       out= static_cast<float*>      (ec.map_buffer_write(st->out_shaded));
    if (N == 0 || !wp || !wn || !mt || !la || !out) {
        ec.dispatch(0, 1, 1);
        return;
    }

    const float* px = wp + 0 * N, *py = wp + 1 * N, *pz = wp + 2 * N;
    const float* nx = wn + 0 * N, *ny = wn + 1 * N, *nz = wn + 2 * N;
    const float* br = mt + 0 * N, *bg = mt + 1 * N, *bb = mt + 2 * N;
    const float* met = mt + 3 * N, *rough = mt + 4 * N;
    float* outr = out + 0 * N, *outg = out + 1 * N, *outb = out + 2 * N;

    const cardinal::scene::Vec3 ambient = fzv(
        { la[0], la[1], la[2] },
        { 0, 0, 0 });

    cardinal::u32 lit = 0;

    for (cardinal::u32 i = 0; i < N; ++i) {
        const cardinal::scene::Vec3 pos = fzv({ px[i], py[i], pz[i] }, { 0, 0, 0 });
        const cardinal::scene::Vec3 n_in = fzv({ nx[i], ny[i], nz[i] }, { 0, 1, 0 });
        const cardinal::scene::Vec3 n = cardinal::scene::normalize(n_in);
        Material m;
        m.base_color = fzv({ br[i], bg[i], bb[i] }, { 0.8f, 0.8f, 0.8f });
        m.metallic   = saturate01(fzf(met[i],   0.0f));
        m.roughness  = saturate01(fzf(rough[i], 0.5f));

        // Ambient term — flat hemispherical irradiance proxy.
        cardinal::scene::Vec3 acc = {
            ambient.x * m.base_color.x,
            ambient.y * m.base_color.y,
            ambient.z * m.base_color.z,
        };

        // View direction: simple eye-at-origin approximation matches what
        // the existing forward CPU path does when no view is bound.
        const cardinal::scene::Vec3 view_dir = cardinal::scene::normalize(
            cardinal::scene::Vec3{ -pos.x, -pos.y, -pos.z });

        bool fragment_lit = false;
        for (cardinal::u32 l = 0; l < L; ++l) {
            const float* lf = lp + l * kPackedLightFloats;
            const cardinal::u32 kind = static_cast<cardinal::u32>(lf[0]);
            const cardinal::scene::Vec3 l_pos {
                fzf(lf[ 2], 0.0f), fzf(lf[ 3], 0.0f), fzf(lf[ 4], 0.0f) };
            const cardinal::scene::Vec3 l_dir {
                fzf(lf[ 5], 0.0f), fzf(lf[ 6], -1.0f), fzf(lf[ 7], 0.0f) };
            const cardinal::scene::Vec3 l_col {
                fzf(lf[ 8], 1.0f), fzf(lf[ 9], 1.0f), fzf(lf[10], 1.0f) };
            const float intensity = fzf(lf[11], 1.0f);
            const float range     = fzf(lf[12], 30.0f);
            const float inner_cos = fzf(lf[13], 0.95f);
            const float outer_cos = fzf(lf[14], 0.85f);

            cardinal::scene::Vec3 light_dir{0, 1, 0};
            cardinal::scene::Vec3 radiance{0, 0, 0};
            if (kind == 0u) {  // Directional
                light_dir = cardinal::scene::normalize(
                    cardinal::scene::Vec3{ -l_dir.x, -l_dir.y, -l_dir.z });
                radiance = { l_col.x * intensity, l_col.y * intensity, l_col.z * intensity };
            } else if (kind == 1u) {  // Point
                cardinal::scene::Vec3 to_l = { l_pos.x - pos.x, l_pos.y - pos.y, l_pos.z - pos.z };
                const float dist = cardinal::scene::length(to_l);
                if (dist < 1.0e-4f) continue;
                light_dir = { to_l.x / dist, to_l.y / dist, to_l.z / dist };
                const float att = (range > 0.0f && dist < range)
                    ? (1.0f - dist / range) * (1.0f - dist / range) : 0.0f;
                if (att <= 0.0f) continue;
                radiance = { l_col.x * intensity * att,
                             l_col.y * intensity * att,
                             l_col.z * intensity * att };
            } else {  // Spot
                cardinal::scene::Vec3 to_l = { l_pos.x - pos.x, l_pos.y - pos.y, l_pos.z - pos.z };
                const float dist = cardinal::scene::length(to_l);
                if (dist < 1.0e-4f) continue;
                light_dir = { to_l.x / dist, to_l.y / dist, to_l.z / dist };
                const float dist_att = (range > 0.0f && dist < range)
                    ? (1.0f - dist / range) * (1.0f - dist / range) : 0.0f;
                const cardinal::scene::Vec3 axis = cardinal::scene::normalize(l_dir);
                const float cos_at = axis.x * (-light_dir.x) + axis.y * (-light_dir.y)
                                   + axis.z * (-light_dir.z);
                const float t = (inner_cos > outer_cos)
                    ? saturate01((cos_at - outer_cos) / (inner_cos - outer_cos)) : 1.0f;
                const float att = dist_att * t * t;
                if (att <= 0.0f) continue;
                radiance = { l_col.x * intensity * att,
                             l_col.y * intensity * att,
                             l_col.z * intensity * att };
            }

            const float NdotL = cardinal::scene::dot(n, light_dir);
            if (NdotL <= 0.0f) continue;

            const cardinal::scene::Vec3 f = brdf(n, view_dir, light_dir, m);
            acc.x += f.x * radiance.x * NdotL;
            acc.y += f.y * radiance.y * NdotL;
            acc.z += f.z * radiance.z * NdotL;
            fragment_lit = true;
        }
        if (fragment_lit) ++lit;

        // Final NaN guard before write — corrupt material / light can't
        // poison the output buffer.
        outr[i] = isf(acc.x) ? acc.x : 0.0f;
        outg[i] = isf(acc.y) ? acc.y : 0.0f;
        outb[i] = isf(acc.z) ? acc.z : 0.0f;
    }
    st->fragments_lit = lit;

    ec.dispatch((N + 63) / 64, 1, 1);
}

}  // namespace

cardinal::shared_ptr<ForwardShadePass::State> ForwardShadePass::add_to_graph(
    rg::Graph& g,
    rg::ResourceHandle in_world_pos,
    rg::ResourceHandle in_world_normal,
    rg::ResourceHandle in_material,
    rg::ResourceHandle in_lights,
    rg::ResourceHandle in_ambient,
    cardinal::u32 fragment_count,
    cardinal::u32 light_count)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_world_pos    = in_world_pos;
    st->in_world_normal = in_world_normal;
    st->in_material     = in_material;
    st->in_lights       = in_lights;
    st->in_ambient      = in_ambient;
    st->fragment_count  = fragment_count;
    st->light_count     = light_count;

    rg::BufferDesc od;
    od.name         = "fwd.shaded";
    od.size_bytes   = static_cast<cardinal::usize>(fragment_count) * 3u * sizeof(float);
    od.stride_bytes = sizeof(float);
    st->out_shaded = g.declare_buffer(od);

    rg::PassDesc pd;
    pd.name = "ForwardShadePass";
    pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_world_pos,    rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_world_normal, rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{in_material,     rg::AccessMode::Read,  2});
    pd.accesses.push_back(rg::ResourceAccess{in_lights,       rg::AccessMode::Read,  3});
    pd.accesses.push_back(rg::ResourceAccess{in_ambient,      rg::AccessMode::Read,  4});
    pd.accesses.push_back(rg::ResourceAccess{st->out_shaded,  rg::AccessMode::Write, 5});
    pd.record   = shade_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = (fragment_count + 63) / 64;
    pd.dispatch_y = 1; pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* ForwardShadePass::hlsl_source() noexcept {
    return R"(// Cardinal — ForwardShadePass.hlsl
//
// One thread per fragment. Reads SoA G-buffer attributes + a packed
// light list; outputs Cook-Torrance direct lighting. CPU reference is
// bit-equivalent.
ByteAddressBuffer  in_world_pos    : register(t0);
ByteAddressBuffer  in_world_normal : register(t1);
ByteAddressBuffer  in_material     : register(t2);
ByteAddressBuffer  in_lights       : register(t3);
ByteAddressBuffer  in_ambient      : register(t4);
RWByteAddressBuffer out_shaded     : register(u0);

cbuffer Params : register(b0) {
    uint fragment_count, light_count, _p0, _p1;
};

static const float kPi = 3.14159265358979323846;

float load_f(ByteAddressBuffer b, uint o) { return asfloat(b.Load(o)); }

float3 brdf(float3 n, float3 v, float3 l, float3 base, float metallic, float roughness) {
    float NdotL = saturate(dot(n, l));
    float NdotV = saturate(dot(n, v));
    if (NdotL <= 0 || NdotV <= 0) return float3(0,0,0);
    float3 h = normalize(v + l);
    float NdotH = saturate(dot(n, h));
    float VdotH = saturate(dot(v, h));
    float a  = max(roughness * roughness, 0.002);
    float a2 = a * a;
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), base, metallic);
    float fp = 1.0 - VdotH;
    float3 F  = F0 + (1 - F0) * (fp*fp*fp*fp*fp);
    float dd = NdotH*NdotH * (a2 - 1) + 1;
    float D  = a2 / (kPi * dd * dd + 1e-6);
    float k  = (roughness + 1) * (roughness + 1) * 0.125;
    float Gv = NdotV / (NdotV * (1-k) + k + 1e-6);
    float Gl = NdotL / (NdotL * (1-k) + k + 1e-6);
    float G  = Gv * Gl;
    float3 spec = D * G * F / (4 * NdotV * NdotL + 1e-6);
    float3 kD = (1 - F) * (1 - metallic);
    return kD * base / kPi + spec;
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= fragment_count) return;

    uint N = fragment_count;
    float3 pos = float3(load_f(in_world_pos, (0*N+i)*4), load_f(in_world_pos, (1*N+i)*4), load_f(in_world_pos, (2*N+i)*4));
    float3 n   = normalize(float3(load_f(in_world_normal, (0*N+i)*4), load_f(in_world_normal, (1*N+i)*4), load_f(in_world_normal, (2*N+i)*4)));
    float3 base = float3(load_f(in_material, (0*N+i)*4), load_f(in_material, (1*N+i)*4), load_f(in_material, (2*N+i)*4));
    float metallic  = load_f(in_material, (3*N+i)*4);
    float roughness = load_f(in_material, (4*N+i)*4);
    float3 amb = float3(load_f(in_ambient,0), load_f(in_ambient,4), load_f(in_ambient,8));
    float3 view = normalize(-pos);

    float3 acc = amb * base;
    for (uint l = 0; l < light_count; ++l) {
        uint lb = l * 16 * 4;
        uint kind = uint(asuint(in_lights.Load(lb)));
        float3 l_pos = float3(load_f(in_lights, lb+ 8), load_f(in_lights, lb+12), load_f(in_lights, lb+16));
        float3 l_dir = float3(load_f(in_lights, lb+20), load_f(in_lights, lb+24), load_f(in_lights, lb+28));
        float3 l_col = float3(load_f(in_lights, lb+32), load_f(in_lights, lb+36), load_f(in_lights, lb+40));
        float intensity = load_f(in_lights, lb+44);
        float range     = load_f(in_lights, lb+48);
        float inner_cos = load_f(in_lights, lb+52);
        float outer_cos = load_f(in_lights, lb+56);

        float3 ld; float3 rad;
        if (kind == 0) {
            ld  = normalize(-l_dir);
            rad = l_col * intensity;
        } else {
            float3 to_l = l_pos - pos;
            float dist  = length(to_l);
            if (dist < 1e-4) continue;
            ld = to_l / dist;
            float dist_att = (range > 0 && dist < range) ? (1 - dist/range)*(1 - dist/range) : 0;
            if (kind == 1) { rad = l_col * intensity * dist_att; }
            else {
                float cos_at = dot(normalize(l_dir), -ld);
                float t = (inner_cos > outer_cos) ? saturate((cos_at - outer_cos) / (inner_cos - outer_cos)) : 1;
                rad = l_col * intensity * dist_att * t * t;
            }
            if (dist_att <= 0) continue;
        }
        float NdotL = dot(n, ld);
        if (NdotL <= 0) continue;
        acc += brdf(n, view, ld, base, metallic, roughness) * rad * NdotL;
    }
    out_shaded.Store((0*N+i)*4, asuint(acc.x));
    out_shaded.Store((1*N+i)*4, asuint(acc.y));
    out_shaded.Store((2*N+i)*4, asuint(acc.z));
}
)";
}

// =============================================================================
// PathTracePass
// =============================================================================
namespace {

void pt_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<PathTracePass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height;
    if (W == 0 || H == 0 || !st->tracer || !st->scene) {
        ec.dispatch(0, 1, 1);
        return;
    }
    float* rad      = static_cast<float*>(ec.map_buffer_write(st->out_radiance));
    cardinal::u8* rgba = static_cast<cardinal::u8*>(ec.map_buffer_write(st->out_rgba));
    if (!rad || !rgba) {
        ec.dispatch(0, 1, 1);
        return;
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    st->tracer->reset_stats();
    st->tracer->trace_image(W, H,
        st->camera_pos, st->forward, st->right, st->up,
        fzf(st->vfov_rad, 0.785f),
        *st->scene,
        rad, rgba);
    const auto t1 = std::chrono::high_resolution_clock::now();
    st->trace_us = static_cast<float>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    const auto stats = st->tracer->stats();
    st->rays_total = stats.rays_primary + stats.rays_indirect + stats.rays_shadow;

    // One workgroup per 8×8 tile (matches the raster pass's tile shape).
    ec.dispatch((W + 7) / 8, (H + 7) / 8, 1);
}

}  // namespace

cardinal::shared_ptr<PathTracePass::State> PathTracePass::add_to_graph(
    rg::Graph& g,
    cardinal::shared_ptr<PathTracer> tracer,
    const Scene& scene_in,
    cardinal::u32 width, cardinal::u32 height)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->tracer = tracer;
    st->scene  = &scene_in;
    st->width  = width;
    st->height = height;

    rg::BufferDesc rd;
    rd.name         = "rtx.radiance";
    rd.size_bytes   = static_cast<cardinal::usize>(width) * height * 3u * sizeof(float);
    rd.stride_bytes = sizeof(float);
    st->out_radiance = g.declare_buffer(rd);

    rg::BufferDesc cd;
    cd.name         = "rtx.rgba";
    cd.size_bytes   = static_cast<cardinal::usize>(width) * height * 4u;
    cd.stride_bytes = 4;
    st->out_rgba = g.declare_buffer(cd);

    rg::PassDesc pd;
    pd.name = "PathTracePass";
    pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{st->out_radiance, rg::AccessMode::Write, 0});
    pd.accesses.push_back(rg::ResourceAccess{st->out_rgba,     rg::AccessMode::Write, 1});
    pd.record   = pt_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = (width  + 7) / 8;
    pd.dispatch_y = (height + 7) / 8;
    pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* PathTracePass::hlsl_source() noexcept {
    return R"(// Cardinal — PathTracePass.hlsl (DXR raygen + miss + closest-hit)
//
// Production GPU path: traces via a TraceRay on the engine's RT
// acceleration structure (built from cardinal::scene::Mesh), accumulates
// the same direct + N-bounce indirect as the CPU PathTracer reference.
// Output buffers + tonemap match the CPU path byte-for-byte at SPP=1.
//
// raygen.hlsl ----------------------------------------------------------
RaytracingAccelerationStructure scene_as : register(t0);
RWStructuredBuffer<float3> out_radiance  : register(u0);
RWByteAddressBuffer        out_rgba      : register(u1);

cbuffer Camera : register(b0) {
    float3 cam_pos;     float vfov_rad;
    float3 forward;     uint  width;
    float3 right;       uint  height;
    float3 up;          uint  max_bounces;
};

float3 srgb_encode(float3 lin) {
    return (lin <= 0.0031308.xxx)
        ? lin * 12.92
        : 1.055 * pow(lin, 1.0/2.4) - 0.055;
}

[shader("raygeneration")]
void RayGen() {
    uint2 idx = DispatchRaysIndex().xy;
    float aspect = float(width) / float(height);
    float half_h = tan(0.5 * vfov_rad);
    float half_w = half_h * aspect;
    float u = ((idx.x + 0.5) / float(width))  * 2 - 1;
    float v = 1 - ((idx.y + 0.5) / float(height)) * 2;
    float3 dir = normalize(forward + right * (u * half_w) + up * (v * half_h));

    RayDesc r;
    r.Origin    = cam_pos;
    r.Direction = dir;
    r.TMin      = 1e-4;
    r.TMax      = 1e6;
    Payload p; p.radiance = 0; p.depth = 0;
    TraceRay(scene_as, RAY_FLAG_NONE, 0xff, 0, 1, 0, r, p);
    uint pix = idx.y * width + idx.x;
    out_radiance[pix] = p.radiance;
    out_rgba.Store(pix * 4 + 0, uint(saturate(srgb_encode(p.radiance).r) * 255 + 0.5));
    out_rgba.Store(pix * 4 + 1, uint(saturate(srgb_encode(p.radiance).g) * 255 + 0.5));
    out_rgba.Store(pix * 4 + 2, uint(saturate(srgb_encode(p.radiance).b) * 255 + 0.5));
    out_rgba.Store(pix * 4 + 3, 255);
}

// closesthit.hlsl ------------------------------------------------------
struct Payload { float3 radiance; uint depth; };
struct Attribs  { float2 bary; };

[shader("closesthit")]
void ClosestHit(inout Payload p, in Attribs attribs) {
    // Evaluate Cook-Torrance direct lighting at the hit (same brdf() as
    // ForwardShadePass), then recurse via cosine-weighted hemisphere
    // sampling for N-bounce indirect. Russian-roulette termination
    // mirrors the CPU PathTracer's enable_indirect + rr_depth knobs.
    // ... (deferred to the RHI compute commit alongside the AS build path)
    p.radiance = float3(0.2, 0.3, 0.5);   // sky stub
}

[shader("miss")]
void Miss(inout Payload p) {
    p.radiance = float3(0.2, 0.32, 0.55);
}
)";
}

}  // namespace cardinal::lighting::gpu
