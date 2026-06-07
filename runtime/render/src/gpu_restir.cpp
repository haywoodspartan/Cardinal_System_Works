#include <cardinal/render/gpu_restir.hpp>

#include <cardinal/core/std/cmath.hpp>
#include <cardinal/core/std/algorithm.hpp>
#include <cardinal/core/std/utility.hpp>

namespace cardinal::render::gpu {

namespace rg = cardinal::render::graph;

namespace {

inline bool isf(float v) noexcept { return cardinal::isfinite(v); }
inline float fzf(float v, float fb) noexcept { return isf(v) ? v : fb; }

// Splitmix32 — same RNG family the rest of the engine uses, so given a
// seed the ReSTIR draws are bit-stable across runs.
inline cardinal::u32 sm32(cardinal::u32& s) noexcept {
    s += 0x9E3779B9u;
    cardinal::u32 z = s;
    z = (z ^ (z >> 16)) * 0x21f0aaadu;
    z = (z ^ (z >> 15)) * 0x735a2d97u;
    return z ^ (z >> 15);
}
inline float rng01(cardinal::u32& s) noexcept {
    return static_cast<float>(sm32(s)) * (1.0f / 4294967296.0f);
}

struct Reservoir {
    cardinal::u32 chosen_light;
    float w_sum;
    float m_count;
    float target_pdf;
};

inline void reservoir_init(Reservoir& r) noexcept {
    r.chosen_light = kInvalidLight;
    r.w_sum       = 0.0f;
    r.m_count     = 0.0f;
    r.target_pdf  = 0.0f;
}

// Standard ReSTIR reservoir update: weighted-reservoir-sampling with
// the candidate's target_pdf as its weight.
inline void reservoir_update(Reservoir& r, cardinal::u32 light_idx,
                             float w_i, float tpdf, cardinal::u32& seed) noexcept {
    if (!isf(w_i) || w_i <= 0.0f) {
        r.m_count += 1.0f;
        return;
    }
    r.w_sum   += w_i;
    r.m_count += 1.0f;
    // Probability of selecting this candidate = w_i / r.w_sum.
    if (rng01(seed) < (w_i / r.w_sum)) {
        r.chosen_light = light_idx;
        r.target_pdf   = tpdf;
    }
}

// Combine two reservoirs (a := a ⊕ b), used by spatial + temporal reuse.
inline void reservoir_combine(Reservoir& a, const Reservoir& b,
                              float scale_m, cardinal::u32& seed) noexcept
{
    if (b.target_pdf <= 0.0f || !isf(b.w_sum)) {
        a.m_count += b.m_count * scale_m;
        return;
    }
    // Map b's "average weight" into a's frame.
    const float w_b = b.target_pdf * (b.w_sum / cardinal::clamp(b.m_count, 1.0f, 1.0e6f))
                     * b.m_count * scale_m;
    if (!isf(w_b) || w_b <= 0.0f) {
        a.m_count += b.m_count * scale_m;
        return;
    }
    a.w_sum   += w_b;
    a.m_count += b.m_count * scale_m;
    if (rng01(seed) < (w_b / a.w_sum)) {
        a.chosen_light = b.chosen_light;
        a.target_pdf   = b.target_pdf;
    }
}

// Target function: cheap proxy for the unshadowed BRDF × NdotL × light
// radiance. Used as the resampling weight inside the reservoir update.
// The full shaded value (including visibility) is evaluated downstream.
inline float target_function(const float* lights, cardinal::u32 light_idx,
                             cardinal::u32 light_count,
                             float px, float py, float pz,
                             float nx, float ny, float nz) noexcept
{
    if (light_idx >= light_count) return 0.0f;
    const float* L = lights + light_idx * 16;
    const cardinal::u32 kind = static_cast<cardinal::u32>(L[0]);
    float ld_x = 0, ld_y = 1, ld_z = 0;
    float intensity = 1.0f;
    if (kind == 0u) {  // Directional
        const float dx = -fzf(L[5], 0.0f);
        const float dy = -fzf(L[6], -1.0f);
        const float dz = -fzf(L[7], 0.0f);
        const float l = cardinal::sqrt(dx * dx + dy * dy + dz * dz);
        if (l > 1e-8f) { ld_x = dx / l; ld_y = dy / l; ld_z = dz / l; }
        intensity = fzf(L[11], 1.0f);
    } else {  // Point / Spot — approximate via to-light unit vector
        const float dx = fzf(L[2], 0.0f) - px;
        const float dy = fzf(L[3], 0.0f) - py;
        const float dz = fzf(L[4], 0.0f) - pz;
        const float d2 = dx * dx + dy * dy + dz * dz;
        const float d  = cardinal::sqrt(d2);
        if (d < 1e-4f) return 0.0f;
        ld_x = dx / d; ld_y = dy / d; ld_z = dz / d;
        const float range = fzf(L[12], 30.0f);
        if (d >= range) return 0.0f;
        const float att = (1.0f - d / range) * (1.0f - d / range);
        intensity = fzf(L[11], 1.0f) * att / d2;
    }
    const float NdotL = nx * ld_x + ny * ld_y + nz * ld_z;
    if (NdotL <= 0.0f) return 0.0f;
    const float color_avg = (fzf(L[8], 1.0f) + fzf(L[9], 1.0f) + fzf(L[10], 1.0f)) / 3.0f;
    return cardinal::clamp(NdotL * intensity * color_avg, 0.0f, 1.0e6f);
}

inline void store_reservoir(float* base, cardinal::usize pix, const Reservoir& r) noexcept {
    const cardinal::usize off = pix * kReservoirDwords;
    cardinal::u32 ci = r.chosen_light;
    base[off + 0] = *reinterpret_cast<float*>(&ci);
    base[off + 1] = r.w_sum;
    base[off + 2] = r.m_count;
    base[off + 3] = r.target_pdf;
}

inline Reservoir load_reservoir(const float* base, cardinal::usize pix) noexcept {
    const cardinal::usize off = pix * kReservoirDwords;
    Reservoir r;
    const float c = base[off + 0];
    r.chosen_light = *reinterpret_cast<const cardinal::u32*>(&c);
    r.w_sum       = base[off + 1];
    r.m_count     = base[off + 2];
    r.target_pdf  = base[off + 3];
    return r;
}

}  // namespace

// =============================================================================
// ReSTIRSamplePass
// =============================================================================
namespace {

void restir_sample_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<ReSTIRSamplePass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height, L = st->light_count;
    const float* wp = static_cast<const float*>(ec.map_buffer_read(st->in_world_pos));
    const float* wn = static_cast<const float*>(ec.map_buffer_read(st->in_world_normal));
    const float* lt = static_cast<const float*>(ec.map_buffer_read(st->in_lights));
    const cardinal::u32* seeds = static_cast<const cardinal::u32*>(
        ec.map_buffer_read(st->in_seeds));
    float* out = static_cast<float*>(ec.map_buffer_write(st->out_reservoirs));
    if (!wp || !wn || !out || !seeds) { ec.dispatch(0, 1, 1); return; }

    const float* px = wp + 0 * W * H;
    const float* py = wp + 1 * W * H;
    const float* pz = wp + 2 * W * H;
    const float* nx = wn + 0 * W * H;
    const float* ny = wn + 1 * W * H;
    const float* nz = wn + 2 * W * H;

    cardinal::u32 sampled = 0, valid = 0;
    const cardinal::u32 M = st->candidates_per_pixel;

    for (cardinal::u32 pix = 0; pix < W * H; ++pix) {
        cardinal::u32 seed = seeds[pix];
        Reservoir r;
        reservoir_init(r);
        if (L == 0 || !lt) {
            store_reservoir(out, pix, r);
            continue;
        }
        const float P_x = fzf(px[pix], 0.0f);
        const float P_y = fzf(py[pix], 0.0f);
        const float P_z = fzf(pz[pix], 0.0f);
        const float N_x = fzf(nx[pix], 0.0f);
        const float N_y = fzf(ny[pix], 1.0f);
        const float N_z = fzf(nz[pix], 0.0f);
        // Uniform sampling: candidate pdf = 1 / L.
        const float src_pdf = 1.0f / static_cast<float>(L);
        for (cardinal::u32 i = 0; i < M; ++i) {
            const cardinal::u32 idx = static_cast<cardinal::u32>(rng01(seed) * L) % L;
            const float tpdf = target_function(lt, idx, L, P_x, P_y, P_z, N_x, N_y, N_z);
            const float w_i  = (tpdf > 0.0f) ? (tpdf / src_pdf) : 0.0f;
            reservoir_update(r, idx, w_i, tpdf, seed);
        }
        store_reservoir(out, pix, r);
        ++sampled;
        if (r.chosen_light != kInvalidLight) ++valid;
    }
    st->pixels_sampled = sampled;
    st->pixels_with_valid_pick = valid;
    ec.dispatch((W + 7) / 8, (H + 7) / 8, 1);
}

}  // namespace

cardinal::shared_ptr<ReSTIRSamplePass::State> ReSTIRSamplePass::add_to_graph(
    rg::Graph& g,
    rg::ResourceHandle in_wp, rg::ResourceHandle in_wn,
    rg::ResourceHandle in_lt, rg::ResourceHandle in_seeds,
    cardinal::u32 W, cardinal::u32 H, cardinal::u32 L, cardinal::u32 M)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_world_pos    = in_wp;
    st->in_world_normal = in_wn;
    st->in_lights       = in_lt;
    st->in_seeds        = in_seeds;
    st->width = W; st->height = H; st->light_count = L;
    st->candidates_per_pixel = M;
    rg::BufferDesc od;
    od.name = "restir.sample"; od.size_bytes = static_cast<cardinal::usize>(W) * H * kReservoirBytes;
    od.stride_bytes = sizeof(float);
    st->out_reservoirs = g.declare_buffer(od);
    rg::PassDesc pd;
    pd.name = "ReSTIRSamplePass"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_wp,    rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_wn,    rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{in_lt,    rg::AccessMode::Read,  2});
    pd.accesses.push_back(rg::ResourceAccess{in_seeds, rg::AccessMode::Read,  3});
    pd.accesses.push_back(rg::ResourceAccess{st->out_reservoirs, rg::AccessMode::Write, 4});
    pd.record = restir_sample_record; pd.user_ctx = st.get();
    pd.dispatch_x = (W + 7) / 8; pd.dispatch_y = (H + 7) / 8;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* ReSTIRSamplePass::hlsl_source() noexcept {
    return R"(// Cardinal — ReSTIRSamplePass.hlsl
// One thread per pixel. M candidate draws, weighted reservoir sampling
// with target_pdf = unshadowed BRDF * NdotL * light_radiance.
[numthreads(8, 8, 1)] void main(uint3 tid : SV_DispatchThreadID) { /* RHI compute commit */ }
)";
}

// =============================================================================
// ReSTIRSpatialPass — neighbour reuse
// =============================================================================
namespace {

void restir_spatial_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<ReSTIRSpatialPass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height;
    const float* in_r = static_cast<const float*>(ec.map_buffer_read(st->in_reservoirs));
    const float* wp = static_cast<const float*>(ec.map_buffer_read(st->in_world_pos));
    const float* wn = static_cast<const float*>(ec.map_buffer_read(st->in_world_normal));
    const cardinal::u32* seeds = static_cast<const cardinal::u32*>(
        ec.map_buffer_read(st->in_seeds));
    float* out = static_cast<float*>(ec.map_buffer_write(st->out_reservoirs));
    if (!in_r || !wp || !wn || !out || !seeds) { ec.dispatch(0, 1, 1); return; }

    const float* nx = wn + 0 * W * H;
    const float* ny = wn + 1 * W * H;
    const float* nz = wn + 2 * W * H;
    const float* pz_buf = wp + 2 * W * H;

    const cardinal::u32 K = st->neighbour_count;
    const float radius = fzf(st->radius_px, 16.0f);

    cardinal::u32 combined = 0, rejected = 0;
    for (cardinal::u32 y = 0; y < H; ++y) {
        for (cardinal::u32 x = 0; x < W; ++x) {
            const cardinal::usize pix = static_cast<cardinal::usize>(y) * W + x;
            cardinal::u32 seed = seeds[pix];
            Reservoir r = load_reservoir(in_r, pix);
            const float center_n_x = fzf(nx[pix], 0.0f);
            const float center_n_y = fzf(ny[pix], 1.0f);
            const float center_n_z = fzf(nz[pix], 0.0f);
            const float center_pz  = fzf(pz_buf[pix], 0.0f);
            for (cardinal::u32 k = 0; k < K; ++k) {
                const float dx = (rng01(seed) * 2.0f - 1.0f) * radius;
                const float dy = (rng01(seed) * 2.0f - 1.0f) * radius;
                const int nx_i = static_cast<int>(x) + static_cast<int>(dx);
                const int ny_i = static_cast<int>(y) + static_cast<int>(dy);
                if (nx_i < 0 || ny_i < 0 ||
                    static_cast<cardinal::u32>(nx_i) >= W ||
                    static_cast<cardinal::u32>(ny_i) >= H) {
                    ++rejected; continue;
                }
                const cardinal::usize npix = static_cast<cardinal::usize>(ny_i) * W +
                                             static_cast<cardinal::usize>(nx_i);
                // Geometric similarity test: normals within ~25° and
                // depth within 0.1 (tunable).
                const float n_dot = center_n_x * fzf(nx[npix], 0.0f) +
                                    center_n_y * fzf(ny[npix], 1.0f) +
                                    center_n_z * fzf(nz[npix], 0.0f);
                const float depth_d = cardinal::abs(center_pz - fzf(pz_buf[npix], 0.0f));
                if (n_dot < 0.9f || depth_d > 0.1f) { ++rejected; continue; }
                const Reservoir nr = load_reservoir(in_r, npix);
                reservoir_combine(r, nr, 1.0f, seed);
                ++combined;
            }
            store_reservoir(out, pix, r);
        }
    }
    st->reservoirs_combined = combined;
    st->neighbours_rejected = rejected;
    ec.dispatch((W + 7) / 8, (H + 7) / 8, 1);
}

}  // namespace

cardinal::shared_ptr<ReSTIRSpatialPass::State> ReSTIRSpatialPass::add_to_graph(
    rg::Graph& g, rg::ResourceHandle in_res, rg::ResourceHandle in_wp,
    rg::ResourceHandle in_wn, rg::ResourceHandle in_seeds,
    cardinal::u32 W, cardinal::u32 H, cardinal::u32 K, float radius)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_reservoirs   = in_res;
    st->in_world_pos    = in_wp;
    st->in_world_normal = in_wn;
    st->in_seeds        = in_seeds;
    st->width = W; st->height = H;
    st->neighbour_count = K; st->radius_px = radius;
    rg::BufferDesc od; od.name = "restir.spatial";
    od.size_bytes = static_cast<cardinal::usize>(W) * H * kReservoirBytes;
    od.stride_bytes = sizeof(float);
    st->out_reservoirs = g.declare_buffer(od);
    rg::PassDesc pd;
    pd.name = "ReSTIRSpatialPass"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_res,            rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_wp,             rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{in_wn,             rg::AccessMode::Read,  2});
    pd.accesses.push_back(rg::ResourceAccess{in_seeds,          rg::AccessMode::Read,  3});
    pd.accesses.push_back(rg::ResourceAccess{st->out_reservoirs,rg::AccessMode::Write, 4});
    pd.record = restir_spatial_record; pd.user_ctx = st.get();
    pd.dispatch_x = (W + 7) / 8; pd.dispatch_y = (H + 7) / 8;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* ReSTIRSpatialPass::hlsl_source() noexcept {
    return R"(// Cardinal — ReSTIRSpatialPass.hlsl
// One thread per pixel. K random neighbours within radius_px, geometric
// similarity check, RIS reservoir combine.
[numthreads(8, 8, 1)] void main(uint3 tid : SV_DispatchThreadID) { /* RHI compute commit */ }
)";
}

// =============================================================================
// ReSTIRTemporalPass — motion-reprojected history reuse
// =============================================================================
namespace {

void restir_temporal_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<ReSTIRTemporalPass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height;
    const float* in_cur  = static_cast<const float*>(ec.map_buffer_read(st->in_reservoirs));
    const float* in_prev = static_cast<const float*>(ec.map_buffer_read(st->in_prev_reservoirs));
    const cardinal::u32* mv = static_cast<const cardinal::u32*>(
        ec.map_buffer_read(st->in_motion));
    const float* wn = static_cast<const float*>(ec.map_buffer_read(st->in_world_normal));
    const float* prev_wn = static_cast<const float*>(
        ec.map_buffer_read(st->in_prev_world_normal));
    float* out = static_cast<float*>(ec.map_buffer_write(st->out_reservoirs));
    if (!in_cur || !out) { ec.dispatch(0, 1, 1); return; }

    const cardinal::u32 N = W * H;
    const float* nx = wn ? wn + 0 * N : nullptr;
    const float* ny = wn ? wn + 1 * N : nullptr;
    const float* nz = wn ? wn + 2 * N : nullptr;
    const float* pnx = prev_wn ? prev_wn + 0 * N : nullptr;
    const float* pny = prev_wn ? prev_wn + 1 * N : nullptr;
    const float* pnz = prev_wn ? prev_wn + 2 * N : nullptr;

    cardinal::u32 history = 0, disoccluded = 0;
    const float max_m = fzf(st->max_history_m, kMaxHistoryM);

    cardinal::u32 seed_base = 0xA0A0A0A0u;
    for (cardinal::u32 y = 0; y < H; ++y) {
        for (cardinal::u32 x = 0; x < W; ++x) {
            const cardinal::usize pix = static_cast<cardinal::usize>(y) * W + x;
            cardinal::u32 seed = seed_base + static_cast<cardinal::u32>(pix);
            Reservoir r = load_reservoir(in_cur, pix);
            // Decode motion vector (snorm16 ×2). With the current
            // MotionVectorPass writing zeros until prev-MVP wiring lands,
            // history pixel = current pixel.
            int px_prev = static_cast<int>(x);
            int py_prev = static_cast<int>(y);
            if (mv) {
                const cardinal::u32 packed = mv[pix];
                const cardinal::i16 mvx = static_cast<cardinal::i16>(packed & 0xFFFFu);
                const cardinal::i16 mvy = static_cast<cardinal::i16>((packed >> 16) & 0xFFFFu);
                const float fx = static_cast<float>(mvx) * (1.0f / 32767.0f) * static_cast<float>(W);
                const float fy = static_cast<float>(mvy) * (1.0f / 32767.0f) * static_cast<float>(H);
                px_prev -= static_cast<int>(fx);
                py_prev -= static_cast<int>(fy);
            }
            if (in_prev && px_prev >= 0 && py_prev >= 0 &&
                static_cast<cardinal::u32>(px_prev) < W &&
                static_cast<cardinal::u32>(py_prev) < H) {
                const cardinal::usize ppix = static_cast<cardinal::usize>(py_prev) * W +
                                             static_cast<cardinal::usize>(px_prev);
                // Disocclusion test on normal.
                float ndot = 1.0f;
                if (nx && pnx) {
                    ndot = fzf(nx[pix], 0) * fzf(pnx[ppix], 0)
                         + fzf(ny[pix], 1) * fzf(pny[ppix], 1)
                         + fzf(nz[pix], 0) * fzf(pnz[ppix], 0);
                }
                if (ndot < 0.85f) {
                    ++disoccluded;
                } else {
                    Reservoir prev = load_reservoir(in_prev, ppix);
                    // Clamp history M.
                    if (prev.m_count > max_m) {
                        const float scale = max_m / prev.m_count;
                        prev.m_count = max_m;
                        prev.w_sum  *= scale;
                    }
                    reservoir_combine(r, prev, 1.0f, seed);
                    ++history;
                }
            }
            store_reservoir(out, pix, r);
        }
    }
    st->pixels_with_history = history;
    st->pixels_disoccluded  = disoccluded;
    ec.dispatch((W + 7) / 8, (H + 7) / 8, 1);
}

}  // namespace

cardinal::shared_ptr<ReSTIRTemporalPass::State> ReSTIRTemporalPass::add_to_graph(
    rg::Graph& g, rg::ResourceHandle in_cur, rg::ResourceHandle in_prev,
    rg::ResourceHandle in_mv, rg::ResourceHandle in_wn,
    rg::ResourceHandle in_prev_wn,
    cardinal::u32 W, cardinal::u32 H, float max_m)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_reservoirs        = in_cur;
    st->in_prev_reservoirs   = in_prev;
    st->in_motion            = in_mv;
    st->in_world_normal      = in_wn;
    st->in_prev_world_normal = in_prev_wn;
    st->width = W; st->height = H; st->max_history_m = max_m;
    rg::BufferDesc od; od.name = "restir.temporal";
    od.size_bytes = static_cast<cardinal::usize>(W) * H * kReservoirBytes;
    od.stride_bytes = sizeof(float);
    st->out_reservoirs = g.declare_buffer(od);
    rg::PassDesc pd;
    pd.name = "ReSTIRTemporalPass"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_cur,             rg::AccessMode::Read,  0});
    if (in_prev.is_valid())
        pd.accesses.push_back(rg::ResourceAccess{in_prev,         rg::AccessMode::Read,  1});
    if (in_mv.is_valid())
        pd.accesses.push_back(rg::ResourceAccess{in_mv,           rg::AccessMode::Read,  2});
    if (in_wn.is_valid())
        pd.accesses.push_back(rg::ResourceAccess{in_wn,           rg::AccessMode::Read,  3});
    if (in_prev_wn.is_valid())
        pd.accesses.push_back(rg::ResourceAccess{in_prev_wn,      rg::AccessMode::Read,  4});
    pd.accesses.push_back(rg::ResourceAccess{st->out_reservoirs, rg::AccessMode::Write, 5});
    pd.record = restir_temporal_record; pd.user_ctx = st.get();
    pd.dispatch_x = (W + 7) / 8; pd.dispatch_y = (H + 7) / 8;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* ReSTIRTemporalPass::hlsl_source() noexcept {
    return R"(// Cardinal — ReSTIRTemporalPass.hlsl
// One thread per pixel. Reproject via motion vector, disocclusion test
// on normal, clamp history M, RIS reservoir combine.
[numthreads(8, 8, 1)] void main(uint3 tid : SV_DispatchThreadID) { /* RHI compute commit */ }
)";
}

}  // namespace cardinal::render::gpu
