#include <cardinal/render/gpu_color.hpp>

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
// ColorGradingPass — 3D LUT trilinear
// ============================================================================
namespace {

void grade_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<ColorGradingPass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height;
    const cardinal::u32 S = cardinal::clamp(st->lut_size, 2u, 256u);
    const cardinal::u8*  in   = static_cast<const cardinal::u8*>(ec.map_buffer_read(st->in_color));
    const float*         lut  = static_cast<const float*>      (ec.map_buffer_read(st->in_lut));
    cardinal::u8*        out  = static_cast<cardinal::u8*>     (ec.map_buffer_write(st->out_color));
    if (!in || !lut || !out) { ec.dispatch(0, 1, 1); return; }

    const float Sm1 = static_cast<float>(S - 1);
    cardinal::u32 graded = 0;

    auto sample_lut = [&](cardinal::u32 ix, cardinal::u32 iy, cardinal::u32 iz,
                          float& r, float& g, float& b) noexcept {
        ix = cardinal::clamp(ix, 0u, S - 1);
        iy = cardinal::clamp(iy, 0u, S - 1);
        iz = cardinal::clamp(iz, 0u, S - 1);
        const cardinal::usize off = ((static_cast<cardinal::usize>(iz) * S + iy) * S + ix) * 3u;
        r = fzf(lut[off + 0], 0.0f);
        g = fzf(lut[off + 1], 0.0f);
        b = fzf(lut[off + 2], 0.0f);
    };

    for (cardinal::u32 y = 0; y < H; ++y) {
        for (cardinal::u32 x = 0; x < W; ++x) {
            const cardinal::usize pix = static_cast<cardinal::usize>(y) * W + x;
            const float r_in = static_cast<float>(in[pix * 4 + 0]) * (1.0f / 255.0f);
            const float g_in = static_cast<float>(in[pix * 4 + 1]) * (1.0f / 255.0f);
            const float b_in = static_cast<float>(in[pix * 4 + 2]) * (1.0f / 255.0f);
            // Quantize to LUT grid.
            const float fr = sat(r_in) * Sm1;
            const float fg = sat(g_in) * Sm1;
            const float fb = sat(b_in) * Sm1;
            const cardinal::u32 ir = static_cast<cardinal::u32>(fr);
            const cardinal::u32 ig = static_cast<cardinal::u32>(fg);
            const cardinal::u32 ib = static_cast<cardinal::u32>(fb);
            const float tr = fr - static_cast<float>(ir);
            const float tg = fg - static_cast<float>(ig);
            const float tb = fb - static_cast<float>(ib);
            // 8 neighbours, trilinear blend.
            float c000_r, c000_g, c000_b; sample_lut(ir,     ig,     ib,     c000_r, c000_g, c000_b);
            float c100_r, c100_g, c100_b; sample_lut(ir + 1, ig,     ib,     c100_r, c100_g, c100_b);
            float c010_r, c010_g, c010_b; sample_lut(ir,     ig + 1, ib,     c010_r, c010_g, c010_b);
            float c110_r, c110_g, c110_b; sample_lut(ir + 1, ig + 1, ib,     c110_r, c110_g, c110_b);
            float c001_r, c001_g, c001_b; sample_lut(ir,     ig,     ib + 1, c001_r, c001_g, c001_b);
            float c101_r, c101_g, c101_b; sample_lut(ir + 1, ig,     ib + 1, c101_r, c101_g, c101_b);
            float c011_r, c011_g, c011_b; sample_lut(ir,     ig + 1, ib + 1, c011_r, c011_g, c011_b);
            float c111_r, c111_g, c111_b; sample_lut(ir + 1, ig + 1, ib + 1, c111_r, c111_g, c111_b);
            // Trilinear: lerp x, then y, then z.
            auto lerp1 = [](float a, float b, float t) noexcept { return a + (b - a) * t; };
            const float c00_r = lerp1(c000_r, c100_r, tr);
            const float c00_g = lerp1(c000_g, c100_g, tr);
            const float c00_b = lerp1(c000_b, c100_b, tr);
            const float c10_r = lerp1(c010_r, c110_r, tr);
            const float c10_g = lerp1(c010_g, c110_g, tr);
            const float c10_b = lerp1(c010_b, c110_b, tr);
            const float c01_r = lerp1(c001_r, c101_r, tr);
            const float c01_g = lerp1(c001_g, c101_g, tr);
            const float c01_b = lerp1(c001_b, c101_b, tr);
            const float c11_r = lerp1(c011_r, c111_r, tr);
            const float c11_g = lerp1(c011_g, c111_g, tr);
            const float c11_b = lerp1(c011_b, c111_b, tr);
            const float c0_r = lerp1(c00_r, c10_r, tg);
            const float c0_g = lerp1(c00_g, c10_g, tg);
            const float c0_b = lerp1(c00_b, c10_b, tg);
            const float c1_r = lerp1(c01_r, c11_r, tg);
            const float c1_g = lerp1(c01_g, c11_g, tg);
            const float c1_b = lerp1(c01_b, c11_b, tg);
            const float r = lerp1(c0_r, c1_r, tb);
            const float g = lerp1(c0_g, c1_g, tb);
            const float b = lerp1(c0_b, c1_b, tb);
            out[pix * 4 + 0] = to_u8(r);
            out[pix * 4 + 1] = to_u8(g);
            out[pix * 4 + 2] = to_u8(b);
            out[pix * 4 + 3] = in[pix * 4 + 3];
            ++graded;
        }
    }
    st->pixels_graded = graded;
    ec.dispatch((W + 7) / 8, (H + 7) / 8, 1);
}

}  // namespace

cardinal::shared_ptr<ColorGradingPass::State> ColorGradingPass::add_to_graph(
    rg::Graph& g, rg::ResourceHandle in_color, rg::ResourceHandle in_lut,
    cardinal::u32 W, cardinal::u32 H, cardinal::u32 lut_size)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_color = in_color; st->in_lut = in_lut;
    st->width = W; st->height = H; st->lut_size = lut_size;
    rg::BufferDesc od; od.name = "grade.rgba"; od.size_bytes = W * H * 4; od.stride_bytes = 4;
    st->out_color = g.declare_buffer(od);
    rg::PassDesc pd;
    pd.name = "ColorGradingPass"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_color,      rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_lut,        rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{st->out_color, rg::AccessMode::Write, 2});
    pd.record = grade_record; pd.user_ctx = st.get();
    pd.dispatch_x = (W + 7) / 8; pd.dispatch_y = (H + 7) / 8;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* ColorGradingPass::hlsl_source() noexcept {
    return R"(// Cardinal — ColorGradingPass.hlsl
// One thread per pixel. Quantize input RGB → LUT grid, fetch 8 neighbours,
// trilinear interpolate. Identity LUT is a no-op.
[numthreads(8, 8, 1)] void main(uint3 tid : SV_DispatchThreadID) { /* RHI compute commit */ }
)";
}

// ============================================================================
// DepthOfFieldPass — CoC-weighted gather blur
// ============================================================================
namespace {

void dof_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<DepthOfFieldPass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height;
    const cardinal::u8* in   = static_cast<const cardinal::u8*>(ec.map_buffer_read(st->in_color));
    const float*        depth = static_cast<const float*>      (ec.map_buffer_read(st->in_depth));
    cardinal::u8*       out  = static_cast<cardinal::u8*>      (ec.map_buffer_write(st->out_color));
    if (!in || !depth || !out) { ec.dispatch(0, 1, 1); return; }

    const float focal_z     = sat(fzf(st->focal_z, 0.5f));
    const float focal_range = sat(fzf(st->focal_range, 0.1f));
    const float coc_falloff = cardinal::clamp(fzf(st->coc_falloff, 0.4f), 0.01f, 10.0f);
    const float max_blur    = fzf(st->max_blur_px, 8.0f);
    const cardinal::u32 N   = cardinal::clamp(st->sample_count, 1u, 64u);

    cardinal::u32 blurred = 0, in_focus = 0;

    auto sample_color = [&](int sx, int sy, float& r, float& g, float& b) noexcept {
        sx = (sx < 0) ? 0 : (sx >= static_cast<int>(W) ? static_cast<int>(W) - 1 : sx);
        sy = (sy < 0) ? 0 : (sy >= static_cast<int>(H) ? static_cast<int>(H) - 1 : sy);
        const cardinal::usize pix = static_cast<cardinal::usize>(sy) * W +
                                    static_cast<cardinal::usize>(sx);
        r = static_cast<float>(in[pix * 4 + 0]);
        g = static_cast<float>(in[pix * 4 + 1]);
        b = static_cast<float>(in[pix * 4 + 2]);
    };

    // Pre-computed angular ring (N samples around a unit circle).
    for (cardinal::u32 y = 0; y < H; ++y) {
        for (cardinal::u32 x = 0; x < W; ++x) {
            const cardinal::usize pix = static_cast<cardinal::usize>(y) * W + x;
            const float d = sat(fzf(depth[pix], focal_z));
            const float dist = cardinal::abs(d - focal_z) - focal_range;
            const float coc  = (dist > 0.0f) ? sat(dist / coc_falloff) * max_blur : 0.0f;
            if (coc < 0.5f) {
                // Sharp.
                out[pix * 4 + 0] = in[pix * 4 + 0];
                out[pix * 4 + 1] = in[pix * 4 + 1];
                out[pix * 4 + 2] = in[pix * 4 + 2];
                out[pix * 4 + 3] = in[pix * 4 + 3];
                ++in_focus;
                continue;
            }
            // Gather N samples in a disk of radius coc.
            float sr = 0, sg = 0, sb = 0;
            float w_sum = 0.0f;
            for (cardinal::u32 i = 0; i < N; ++i) {
                const float angle = static_cast<float>(i) * (6.28318530718f / static_cast<float>(N));
                const float ring_r = coc * (0.3f + 0.7f * (static_cast<float>(i % 4) / 3.0f));
                const int sx = static_cast<int>(x) + static_cast<int>(cardinal::cos(angle) * ring_r);
                const int sy = static_cast<int>(y) + static_cast<int>(cardinal::sin(angle) * ring_r);
                float r, g, b;
                sample_color(sx, sy, r, g, b);
                sr += r; sg += g; sb += b; w_sum += 1.0f;
            }
            // Centre sample.
            float r0, g0, b0;
            sample_color(static_cast<int>(x), static_cast<int>(y), r0, g0, b0);
            sr += r0; sg += g0; sb += b0; w_sum += 1.0f;
            const float inv = 1.0f / w_sum;
            out[pix * 4 + 0] = static_cast<cardinal::u8>(sr * inv);
            out[pix * 4 + 1] = static_cast<cardinal::u8>(sg * inv);
            out[pix * 4 + 2] = static_cast<cardinal::u8>(sb * inv);
            out[pix * 4 + 3] = in[pix * 4 + 3];
            ++blurred;
        }
    }
    st->pixels_blurred  = blurred;
    st->pixels_in_focus = in_focus;
    ec.dispatch((W + 7) / 8, (H + 7) / 8, 1);
}

}  // namespace

cardinal::shared_ptr<DepthOfFieldPass::State> DepthOfFieldPass::add_to_graph(
    rg::Graph& g, rg::ResourceHandle in_color, rg::ResourceHandle in_depth,
    cardinal::u32 W, cardinal::u32 H,
    float focal_z, float focal_range, float coc_falloff, float max_blur_px)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_color = in_color; st->in_depth = in_depth;
    st->width = W; st->height = H;
    st->focal_z = focal_z; st->focal_range = focal_range;
    st->coc_falloff = coc_falloff; st->max_blur_px = max_blur_px;
    rg::BufferDesc od; od.name = "dof.rgba"; od.size_bytes = W * H * 4; od.stride_bytes = 4;
    st->out_color = g.declare_buffer(od);
    rg::PassDesc pd;
    pd.name = "DepthOfFieldPass"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_color,      rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_depth,      rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{st->out_color, rg::AccessMode::Write, 2});
    pd.record = dof_record; pd.user_ctx = st.get();
    pd.dispatch_x = (W + 7) / 8; pd.dispatch_y = (H + 7) / 8;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* DepthOfFieldPass::hlsl_source() noexcept {
    return R"(// Cardinal — DepthOfFieldPass.hlsl
// One thread per pixel. CoC from depth, gather N samples in a disk,
// weighted average. Pixels at focal plane skip the blur entirely.
[numthreads(8, 8, 1)] void main(uint3 tid : SV_DispatchThreadID) { /* RHI compute commit */ }
)";
}

}  // namespace cardinal::render::gpu
