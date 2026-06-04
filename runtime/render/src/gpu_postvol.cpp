#include <cardinal/render/gpu_postvol.hpp>

#include <cardinal/core/cmath.hpp>
#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/utility.hpp>

namespace cardinal::render::gpu {

namespace rg = cardinal::render::graph;

namespace {

inline bool isf(float v) noexcept { return cardinal::isfinite(v); }
inline float fzf(float v, float fb) noexcept { return isf(v) ? v : fb; }

inline float sat(float v) noexcept {
    if (!isf(v)) return 0.0f;
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}
inline cardinal::u8 to_u8(float v) noexcept {
    return static_cast<cardinal::u8>(sat(v) * 255.0f + 0.5f);
}

}  // namespace

// ============================================================================
// TAAPass
// ============================================================================
namespace {

void taa_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<TAAPass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height;
    if (W == 0 || H == 0) { ec.dispatch(0, 1, 1); return; }

    const float* cur  = static_cast<const float*>(ec.map_buffer_read(st->in_current));
    const float* hist = static_cast<const float*>(ec.map_buffer_read(st->in_history));
    const cardinal::u32* mv = static_cast<const cardinal::u32*>(
        ec.map_buffer_read(st->in_motion));
    float* out_rad = static_cast<float*>(ec.map_buffer_write(st->out_radiance));
    float* out_his = static_cast<float*>(ec.map_buffer_write(st->out_history));
    if (!cur || !out_rad || !out_his) { ec.dispatch(0, 1, 1); return; }

    const float alpha = sat(fzf(st->blend_alpha, 0.1f));
    cardinal::u32 blended = 0, disoccluded = 0, clamped_count = 0;

    auto load_rgb = [&](const float* buf, cardinal::u32 x, cardinal::u32 y,
                        float& r, float& g, float& b) noexcept {
        const cardinal::usize pix = static_cast<cardinal::usize>(y) * W + x;
        r = fzf(buf[pix * 3 + 0], 0.0f);
        g = fzf(buf[pix * 3 + 1], 0.0f);
        b = fzf(buf[pix * 3 + 2], 0.0f);
    };

    for (cardinal::u32 y = 0; y < H; ++y) {
        for (cardinal::u32 x = 0; x < W; ++x) {
            const cardinal::usize pix = static_cast<cardinal::usize>(y) * W + x;
            // Current pixel.
            const float cr = fzf(cur[pix * 3 + 0], 0.0f);
            const float cg = fzf(cur[pix * 3 + 1], 0.0f);
            const float cb = fzf(cur[pix * 3 + 2], 0.0f);

            // Decode motion vector (snorm16 ×2 in u32).
            int px_prev = static_cast<int>(x);
            int py_prev = static_cast<int>(y);
            if (mv) {
                const cardinal::u32 packed = mv[pix];
                const cardinal::i16 ix = static_cast<cardinal::i16>(packed & 0xFFFFu);
                const cardinal::i16 iy = static_cast<cardinal::i16>((packed >> 16) & 0xFFFFu);
                const float dx = static_cast<float>(ix) * (1.0f / 32767.0f) * static_cast<float>(W);
                const float dy = static_cast<float>(iy) * (1.0f / 32767.0f) * static_cast<float>(H);
                px_prev -= static_cast<int>(dx);
                py_prev -= static_cast<int>(dy);
            }

            bool has_history = hist != nullptr &&
                               px_prev >= 0 && py_prev >= 0 &&
                               static_cast<cardinal::u32>(px_prev) < W &&
                               static_cast<cardinal::u32>(py_prev) < H;

            float out_r, out_g, out_b;
            if (!has_history) {
                ++disoccluded;
                out_r = cr; out_g = cg; out_b = cb;
            } else {
                // Sample history at the reprojected pixel.
                float hr, hg, hb;
                load_rgb(hist, static_cast<cardinal::u32>(px_prev),
                         static_cast<cardinal::u32>(py_prev), hr, hg, hb);
                // 3×3 neighbourhood colour box around the current pixel.
                float mn_r = cr, mn_g = cg, mn_b = cb;
                float mx_r = cr, mx_g = cg, mx_b = cb;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        const int nx_i = static_cast<int>(x) + dx;
                        const int ny_i = static_cast<int>(y) + dy;
                        if (nx_i < 0 || ny_i < 0 ||
                            static_cast<cardinal::u32>(nx_i) >= W ||
                            static_cast<cardinal::u32>(ny_i) >= H) continue;
                        float nr, ng, nb;
                        load_rgb(cur, static_cast<cardinal::u32>(nx_i),
                                 static_cast<cardinal::u32>(ny_i), nr, ng, nb);
                        if (nr < mn_r) mn_r = nr; if (nr > mx_r) mx_r = nr;
                        if (ng < mn_g) mn_g = ng; if (ng > mx_g) mx_g = ng;
                        if (nb < mn_b) mn_b = nb; if (nb > mx_b) mx_b = nb;
                    }
                }
                // Clamp history to the colour box.
                const float clr = (hr < mn_r) ? mn_r : (hr > mx_r ? mx_r : hr);
                const float clg = (hg < mn_g) ? mn_g : (hg > mx_g ? mx_g : hg);
                const float clb = (hb < mn_b) ? mn_b : (hb > mx_b ? mx_b : hb);
                const bool was_clamped = (clr != hr) || (clg != hg) || (clb != hb);
                if (was_clamped) ++clamped_count;
                // Blend.
                out_r = clr + (cr - clr) * alpha;
                out_g = clg + (cg - clg) * alpha;
                out_b = clb + (cb - clb) * alpha;
                ++blended;
            }
            out_rad[pix * 3 + 0] = out_r;
            out_rad[pix * 3 + 1] = out_g;
            out_rad[pix * 3 + 2] = out_b;
            // History is just a copy of the output for next frame.
            out_his[pix * 3 + 0] = out_r;
            out_his[pix * 3 + 1] = out_g;
            out_his[pix * 3 + 2] = out_b;
        }
    }
    st->pixels_blended     = blended;
    st->pixels_disoccluded = disoccluded;
    st->pixels_clamped     = clamped_count;
    ec.dispatch((W + 7) / 8, (H + 7) / 8, 1);
}

}  // namespace

cardinal::shared_ptr<TAAPass::State> TAAPass::add_to_graph(
    rg::Graph& g,
    rg::ResourceHandle in_cur, rg::ResourceHandle in_hist, rg::ResourceHandle in_mv,
    cardinal::u32 W, cardinal::u32 H, float alpha)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_current = in_cur; st->in_history = in_hist; st->in_motion = in_mv;
    st->width = W; st->height = H; st->blend_alpha = alpha;
    rg::BufferDesc od; od.name = "taa.rad";
    od.size_bytes = static_cast<cardinal::usize>(W) * H * 3 * sizeof(float);
    od.stride_bytes = sizeof(float);
    st->out_radiance = g.declare_buffer(od);
    rg::BufferDesc hd; hd.name = "taa.history";
    hd.size_bytes = od.size_bytes; hd.stride_bytes = sizeof(float);
    st->out_history = g.declare_buffer(hd);
    rg::PassDesc pd;
    pd.name = "TAAPass"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_cur,            rg::AccessMode::Read,  0});
    if (in_hist.is_valid())
        pd.accesses.push_back(rg::ResourceAccess{in_hist,        rg::AccessMode::Read,  1});
    if (in_mv.is_valid())
        pd.accesses.push_back(rg::ResourceAccess{in_mv,          rg::AccessMode::Read,  2});
    pd.accesses.push_back(rg::ResourceAccess{st->out_radiance,  rg::AccessMode::Write, 3});
    pd.accesses.push_back(rg::ResourceAccess{st->out_history,   rg::AccessMode::Write, 4});
    pd.record = taa_record; pd.user_ctx = st.get();
    pd.dispatch_x = (W + 7) / 8; pd.dispatch_y = (H + 7) / 8;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* TAAPass::hlsl_source() noexcept {
    return R"(// Cardinal — TAAPass.hlsl
// One thread per pixel. Decode motion → previous-pixel → sample history,
// neighbourhood-clamp to suppress ghosting, blend with current at alpha.
ByteAddressBuffer   in_current : register(t0);
ByteAddressBuffer   in_history : register(t1);
ByteAddressBuffer   in_motion  : register(t2);
RWByteAddressBuffer out_rad    : register(u0);
RWByteAddressBuffer out_history: register(u1);
cbuffer Params : register(b0) { uint width, height, _p0, _p1; float blend_alpha, _p2, _p3, _p4; };
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) { /* RHI compute commit */ }
)";
}

// ============================================================================
// VolumetricFogPass
// ============================================================================
namespace {

// Henyey-Greenstein phase function for view-direction lighting in the fog.
inline float hg_phase(float cos_theta, float g) noexcept {
    const float g2 = g * g;
    const float denom = 1.0f + g2 - 2.0f * g * cos_theta;
    if (denom <= 1e-6f) return 0.0f;
    return (1.0f - g2) / (4.0f * 3.14159265f * cardinal::pow(denom, 1.5f));
}

// Exponential slice mapping → world-space distance at slice index.
inline float slice_to_distance(cardinal::u32 slice, cardinal::u32 slice_count,
                               float max_distance) noexcept {
    const float t = (static_cast<float>(slice) + 0.5f) /
                    static_cast<float>(slice_count);
    // Exponential: more resolution near the camera.
    const float e = cardinal::pow(2.0f, t * 7.0f) / 128.0f;   // 0..1 with bias toward 0
    return e * max_distance;
}

void fog_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<VolumetricFogPass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height;
    const cardinal::u32 TX = st->tiles_x, TY = st->tiles_y, SL = st->slice_count;
    if (W == 0 || H == 0 || TX == 0 || TY == 0 || SL == 0) {
        ec.dispatch(0, 1, 1); return;
    }
    const float* lights = static_cast<const float*>(ec.map_buffer_read(st->in_lights));
    const float* depth  = static_cast<const float*>(ec.map_buffer_read(st->in_depth));
    const float* amb    = static_cast<const float*>(ec.map_buffer_read(st->in_ambient));
    float*       scat   = static_cast<float*>      (ec.map_buffer_write(st->out_scattering));
    cardinal::u8* fog_rgba = static_cast<cardinal::u8*>(ec.map_buffer_write(st->out_fog_rgba));
    if (!depth || !amb || !scat || !fog_rgba) { ec.dispatch(0, 1, 1); return; }

    const float density = sat(fzf(st->density, 0.05f));
    const float g_phase = cardinal::clamp(fzf(st->anisotropy, 0.7f), -0.99f, 0.99f);
    const float max_d   = fzf(st->max_distance, 200.0f);
    const cardinal::u32 L = st->light_count;

    // Build the scattering volume. Each froxel stores (R, G, B, T) where
    // T = transmittance from the camera to that froxel's near face.
    cardinal::u32 lit = 0;
    const float amb_r = fzf(amb[0], 0.0f), amb_g = fzf(amb[1], 0.0f), amb_b = fzf(amb[2], 0.0f);
    for (cardinal::u32 ty = 0; ty < TY; ++ty) {
        for (cardinal::u32 tx = 0; tx < TX; ++tx) {
            float acc_r = 0, acc_g = 0, acc_b = 0;
            float transmittance = 1.0f;
            for (cardinal::u32 sl = 0; sl < SL; ++sl) {
                const float dist = slice_to_distance(sl, SL, max_d);
                const float slab_thickness = (sl + 1 < SL)
                    ? (slice_to_distance(sl + 1, SL, max_d) - dist) : 1.0f;
                const float local_extinction = density * slab_thickness;
                const float local_transmittance = cardinal::exp(-local_extinction);
                // Per-slice scattering: ambient + per-light HG-weighted radiance.
                float in_r = amb_r * density, in_g = amb_g * density, in_b = amb_b * density;
                if (lights && L > 0) {
                    for (cardinal::u32 l = 0; l < L; ++l) {
                        const float* lp = lights + l * 16;
                        const cardinal::u32 kind = static_cast<cardinal::u32>(lp[0]);
                        float ld_x = 0, ld_y = 1, ld_z = 0;
                        if (kind == 0u) {
                            ld_x = -fzf(lp[5], 0.0f);
                            ld_y = -fzf(lp[6], -1.0f);
                            ld_z = -fzf(lp[7], 0.0f);
                            const float nl = cardinal::sqrt(ld_x*ld_x + ld_y*ld_y + ld_z*ld_z);
                            if (nl > 1e-8f) { ld_x /= nl; ld_y /= nl; ld_z /= nl; }
                        }
                        const float intensity = fzf(lp[11], 1.0f);
                        const float cr = fzf(lp[8],  1.0f);
                        const float cg = fzf(lp[9],  1.0f);
                        const float cb = fzf(lp[10], 1.0f);
                        // View direction approx: +Z (camera looks down +Z).
                        const float view_z = 1.0f;
                        const float cos_t = ld_x * 0 + ld_y * 0 + ld_z * view_z;
                        const float phase = hg_phase(cos_t, g_phase);
                        in_r += cr * intensity * phase;
                        in_g += cg * intensity * phase;
                        in_b += cb * intensity * phase;
                    }
                }
                acc_r += in_r * transmittance * (1.0f - local_transmittance);
                acc_g += in_g * transmittance * (1.0f - local_transmittance);
                acc_b += in_b * transmittance * (1.0f - local_transmittance);
                transmittance *= local_transmittance;
                const cardinal::usize off = (static_cast<cardinal::usize>(sl) * TY * TX +
                                             static_cast<cardinal::usize>(ty) * TX + tx) * kFogFroxelFloats;
                scat[off + 0] = acc_r;
                scat[off + 1] = acc_g;
                scat[off + 2] = acc_b;
                scat[off + 3] = transmittance;
                if (acc_r + acc_g + acc_b > 0.0f) ++lit;
            }
        }
    }
    st->froxels_lit = lit;

    // Per-pixel composite: sample the froxel at the surface depth + write
    // an RGBA8 overlay the host (CompositePresentPass) blends over scene.
    cardinal::u32 px_in_fog = 0;
    for (cardinal::u32 y = 0; y < H; ++y) {
        for (cardinal::u32 x = 0; x < W; ++x) {
            const cardinal::usize pix = static_cast<cardinal::usize>(y) * W + x;
            const float d = sat(fzf(depth[pix], 1.0f));
            // Surface distance ≈ d * max_distance (approximation; real
            // depth → view-space-z needs inv-VP).
            const float surf_dist = d * max_d;
            cardinal::u32 sl = 0;
            for (cardinal::u32 i = 0; i < SL; ++i) {
                if (slice_to_distance(i, SL, max_d) >= surf_dist) { sl = i; break; }
                sl = i;
            }
            const cardinal::u32 tx = x / kFogTileSize;
            const cardinal::u32 ty = y / kFogTileSize;
            if (tx >= TX || ty >= TY) {
                fog_rgba[pix * 4 + 0] = 0;
                fog_rgba[pix * 4 + 1] = 0;
                fog_rgba[pix * 4 + 2] = 0;
                fog_rgba[pix * 4 + 3] = 0;
                continue;
            }
            const cardinal::usize off = (static_cast<cardinal::usize>(sl) * TY * TX +
                                         static_cast<cardinal::usize>(ty) * TX + tx) * kFogFroxelFloats;
            const float r = scat[off + 0];
            const float g = scat[off + 1];
            const float b = scat[off + 2];
            const float t = scat[off + 3];
            const float alpha_fog = 1.0f - t;
            fog_rgba[pix * 4 + 0] = to_u8(r);
            fog_rgba[pix * 4 + 1] = to_u8(g);
            fog_rgba[pix * 4 + 2] = to_u8(b);
            fog_rgba[pix * 4 + 3] = to_u8(alpha_fog);
            if (alpha_fog > 0.01f) ++px_in_fog;
        }
    }
    st->pixels_in_fog = px_in_fog;
    ec.dispatch(TX, TY, SL);
}

}  // namespace

cardinal::shared_ptr<VolumetricFogPass::State> VolumetricFogPass::add_to_graph(
    rg::Graph& g,
    rg::ResourceHandle in_lights, rg::ResourceHandle in_depth,
    rg::ResourceHandle in_vp, rg::ResourceHandle in_amb,
    cardinal::u32 W, cardinal::u32 H, cardinal::u32 L,
    float density, float g_phase, float max_d)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_lights = in_lights; st->in_depth = in_depth;
    st->in_view_proj = in_vp;  st->in_ambient = in_amb;
    st->width = W; st->height = H;
    st->tiles_x = (W + kFogTileSize - 1) / kFogTileSize;
    st->tiles_y = (H + kFogTileSize - 1) / kFogTileSize;
    st->light_count = L;
    st->density = density; st->anisotropy = g_phase; st->max_distance = max_d;

    rg::BufferDesc sd;
    sd.name = "fog.scatter";
    sd.size_bytes = static_cast<cardinal::usize>(st->tiles_x) *
                    st->tiles_y * st->slice_count * kFogFroxelFloats * sizeof(float);
    sd.stride_bytes = sizeof(float);
    st->out_scattering = g.declare_buffer(sd);

    rg::BufferDesc fd;
    fd.name = "fog.rgba"; fd.size_bytes = W * H * 4; fd.stride_bytes = 4;
    st->out_fog_rgba = g.declare_buffer(fd);

    rg::PassDesc pd;
    pd.name = "VolumetricFogPass"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_lights,           rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_depth,            rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{in_vp,               rg::AccessMode::Read,  2});
    pd.accesses.push_back(rg::ResourceAccess{in_amb,              rg::AccessMode::Read,  3});
    pd.accesses.push_back(rg::ResourceAccess{st->out_scattering,  rg::AccessMode::Write, 4});
    pd.accesses.push_back(rg::ResourceAccess{st->out_fog_rgba,    rg::AccessMode::Write, 5});
    pd.record = fog_record; pd.user_ctx = st.get();
    pd.dispatch_x = st->tiles_x; pd.dispatch_y = st->tiles_y; pd.dispatch_z = st->slice_count;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* VolumetricFogPass::hlsl_source() noexcept {
    return R"(// Cardinal — VolumetricFogPass.hlsl
// One thread per froxel. Evaluate light list with Henyey-Greenstein,
// integrate front-to-back along the slice column. Second dispatch
// composites per-pixel fog overlay using surface depth → slice mapping.
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) { /* RHI compute commit */ }
)";
}

}  // namespace cardinal::render::gpu
