#include <cardinal/render/gpu_visbuf.hpp>

#include <cardinal/core/std/cmath.hpp>
#include <cardinal/core/std/algorithm.hpp>
#include <cardinal/core/std/utility.hpp>

namespace cardinal::render::gpu {

namespace rg = cardinal::render::graph;

namespace {

inline bool isf(float v) noexcept { return cardinal::isfinite(v); }
inline float fzf(float v, float fb) noexcept { return isf(v) ? v : fb; }

inline float edge_fn(float ax, float ay, float bx, float by,
                     float cx, float cy) noexcept {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

// Oct-encode a unit normal into a single u32 (16 bits per axis, snorm).
// Standard Cigolle et al. encoding — bit-stable + invertible to a normal
// within ~1° error across the sphere. Reference for the eventual HLSL.
inline cardinal::u32 oct_pack(float nx, float ny, float nz) noexcept {
    const float n_x = fzf(nx, 0.0f);
    const float n_y = fzf(ny, 1.0f);
    const float n_z = fzf(nz, 0.0f);
    const float inv_l1 = 1.0f / (cardinal::abs(n_x) + cardinal::abs(n_y) + cardinal::abs(n_z) + 1.0e-8f);
    float ox = n_x * inv_l1;
    float oz = n_z * inv_l1;
    if (n_y < 0.0f) {
        const float tx = (1.0f - cardinal::abs(oz)) * (ox >= 0.0f ? 1.0f : -1.0f);
        const float tz = (1.0f - cardinal::abs(ox)) * (oz >= 0.0f ? 1.0f : -1.0f);
        ox = tx; oz = tz;
    }
    // snorm16 quantise.
    const float qx = cardinal::clamp(ox, -1.0f, 1.0f);
    const float qz = cardinal::clamp(oz, -1.0f, 1.0f);
    const cardinal::u32 ix = static_cast<cardinal::u32>(
        static_cast<cardinal::i32>(qx * 32767.0f) & 0xFFFF);
    const cardinal::u32 iz = static_cast<cardinal::u32>(
        static_cast<cardinal::i32>(qz * 32767.0f) & 0xFFFF);
    return ix | (iz << 16);
}

}  // namespace

// =============================================================================
// VisibilityBufferPass
// =============================================================================
namespace {

void visbuf_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<VisibilityBufferPass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height;
    if (W == 0 || H == 0) { ec.dispatch(0, 1, 1); return; }

    const float* tris = static_cast<const float*>(ec.map_buffer_read(st->in_tris));
    const cardinal::u32* mat_ids = static_cast<const cardinal::u32*>(
        ec.map_buffer_read(st->in_mat_ids));
    const float* mat = static_cast<const float*>(ec.map_buffer_read(st->in_matrix));
    const cardinal::u32* aux_in = st->in_aux_per_tri.is_valid()
        ? static_cast<const cardinal::u32*>(ec.map_buffer_read(st->in_aux_per_tri))
        : nullptr;

    float*         depth    = static_cast<float*>        (ec.map_buffer_write(st->out_depth));
    cardinal::u32* prim_id  = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_prim_id));
    cardinal::u32* mat_id   = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_mat_id));
    cardinal::u32* normal   = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_normal));
    cardinal::u32* motion   = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_motion));
    cardinal::u32* aux_out  = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_aux));
    if (!depth || !prim_id || !mat_id || !normal || !motion || !aux_out) {
        ec.dispatch(0, 1, 1);
        return;
    }

    // Clear all 6 channels.
    for (cardinal::u32 i = 0; i < W * H; ++i) {
        depth  [i] = 1.0f;
        prim_id[i] = kInvalidPrimId;
        mat_id [i] = kInvalidMatId;
        normal [i] = 0u;
        motion [i] = 0u;
        aux_out[i] = 0u;
    }
    if (!tris || !mat || st->triangle_count == 0) {
        ec.dispatch(0, 1, 1);
        return;
    }

    float m[16];
    for (int i = 0; i < 16; ++i) m[i] = fzf(mat[i], (i % 5 == 0) ? 1.0f : 0.0f);

    const float hw = 0.5f * static_cast<float>(W);
    const float hh = 0.5f * static_cast<float>(H);

    cardinal::u32 frags = 0, off = 0;

    for (cardinal::u32 t = 0; t < st->triangle_count; ++t) {
        const float* v = tris + t * 9;
        float sx[3], sy[3], sz[3], sw[3];
        for (int k = 0; k < 3; ++k) {
            const float x = fzf(v[k * 3 + 0], 0.0f);
            const float y = fzf(v[k * 3 + 1], 0.0f);
            const float z = fzf(v[k * 3 + 2], 0.0f);
            const float cx = m[0]  * x + m[1]  * y + m[2]  * z + m[3];
            const float cy = m[4]  * x + m[5]  * y + m[6]  * z + m[7];
            const float cz = m[8]  * x + m[9]  * y + m[10] * z + m[11];
            const float cw = m[12] * x + m[13] * y + m[14] * z + m[15];
            sw[k] = cw;
            if (cardinal::abs(cw) < 1.0e-8f || !isf(cw) || cw <= 0.0f) {
                sx[k] = 0; sy[k] = 0; sz[k] = 1.0e30f;
            } else {
                sx[k] = cx / cw;
                sy[k] = cy / cw;
                sz[k] = cz / cw;
            }
        }
        const bool any_behind = (sw[0] <= 0.0f) || (sw[1] <= 0.0f) || (sw[2] <= 0.0f);
        const bool all_left   = sx[0] < -1.0f && sx[1] < -1.0f && sx[2] < -1.0f;
        const bool all_right  = sx[0] >  1.0f && sx[1] >  1.0f && sx[2] >  1.0f;
        const bool all_below  = sy[0] < -1.0f && sy[1] < -1.0f && sy[2] < -1.0f;
        const bool all_above  = sy[0] >  1.0f && sy[1] >  1.0f && sy[2] >  1.0f;
        if (any_behind || all_left || all_right || all_below || all_above) {
            ++off; continue;
        }
        float px[3], py[3], pz[3];
        for (int k = 0; k < 3; ++k) {
            px[k] = sx[k] * hw + hw;
            py[k] = hh - sy[k] * hh;
            pz[k] = 0.5f * (sz[k] + 1.0f);
        }
        float min_x = px[0], max_x = px[0];
        if (px[1] < min_x) min_x = px[1]; if (px[1] > max_x) max_x = px[1];
        if (px[2] < min_x) min_x = px[2]; if (px[2] > max_x) max_x = px[2];
        float min_y = py[0], max_y = py[0];
        if (py[1] < min_y) min_y = py[1]; if (py[1] > max_y) max_y = py[1];
        if (py[2] < min_y) min_y = py[2]; if (py[2] > max_y) max_y = py[2];
        int x0 = static_cast<int>(min_x);     if (x0 < 0) x0 = 0;
        int x1 = static_cast<int>(max_x) + 1; if (x1 > static_cast<int>(W)) x1 = static_cast<int>(W);
        int y0 = static_cast<int>(min_y);     if (y0 < 0) y0 = 0;
        int y1 = static_cast<int>(max_y) + 1; if (y1 > static_cast<int>(H)) y1 = static_cast<int>(H);
        if (x0 >= x1 || y0 >= y1) { ++off; continue; }

        const float area = edge_fn(px[0], py[0], px[1], py[1], px[2], py[2]);
        if (cardinal::abs(area) < 1.0e-6f) continue;
        const float inv_area = 1.0f / area;

        // Per-triangle attributes — material ID + winding-derived normal.
        const cardinal::u32 mid = mat_ids ? mat_ids[t] : t;
        const cardinal::u32 aux = aux_in  ? aux_in[t]  : t;
        // World-space normal from the triangle winding.
        const float ux = v[3] - v[0], uy = v[4] - v[1], uz = v[5] - v[2];
        const float vx = v[6] - v[0], vy = v[7] - v[1], vz = v[8] - v[2];
        float nx = uy * vz - uz * vy;
        float ny = uz * vx - ux * vz;
        float nz = ux * vy - uy * vx;
        const float nl = cardinal::sqrt(nx * nx + ny * ny + nz * nz);
        if (nl > 1.0e-8f) { nx /= nl; ny /= nl; nz /= nl; }
        else              { nx = 0; ny = 1; nz = 0; }
        const cardinal::u32 n_packed = oct_pack(nx, ny, nz);

        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                const float fx = static_cast<float>(x) + 0.5f;
                const float fy = static_cast<float>(y) + 0.5f;
                const float w0 = edge_fn(px[1], py[1], px[2], py[2], fx, fy);
                const float w1 = edge_fn(px[2], py[2], px[0], py[0], fx, fy);
                const float w2 = edge_fn(px[0], py[0], px[1], py[1], fx, fy);
                const bool inside = (area > 0.0f)
                    ? (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
                    : (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
                if (!inside) continue;
                const float b0 = w0 * inv_area;
                const float b1 = w1 * inv_area;
                const float b2 = w2 * inv_area;
                const float z  = b0 * pz[0] + b1 * pz[1] + b2 * pz[2];
                if (!isf(z) || z < 0.0f || z > 1.0f) continue;
                const cardinal::usize pix = static_cast<cardinal::usize>(y) * W +
                                            static_cast<cardinal::usize>(x);
                if (z >= depth[pix]) continue;
                depth  [pix] = z;
                prim_id[pix] = t;
                mat_id [pix] = mid;
                normal [pix] = n_packed;
                aux_out[pix] = aux;
                // motion vector = 0 for now (no prev-MVP wired yet); the
                // host can supply a prev-frame matrix in a follow-up
                // commit and the pass will fill the per-pixel
                // packed-snorm16 delta in the same loop.
                ++frags;
            }
        }
    }
    st->fragments_drawn     = frags;
    st->triangles_offscreen = off;

    ec.dispatch((W + 7) / 8, (H + 7) / 8, 1);
}

}  // namespace

cardinal::shared_ptr<VisibilityBufferPass::State> VisibilityBufferPass::add_to_graph(
    rg::Graph& g,
    rg::ResourceHandle in_tris,
    rg::ResourceHandle in_mat_ids,
    rg::ResourceHandle in_matrix,
    rg::ResourceHandle in_aux_per_tri,
    cardinal::u32 width, cardinal::u32 height,
    cardinal::u32 triangle_count)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_tris        = in_tris;
    st->in_mat_ids     = in_mat_ids;
    st->in_matrix      = in_matrix;
    st->in_aux_per_tri = in_aux_per_tri;
    st->width          = width;
    st->height         = height;
    st->triangle_count = triangle_count;

    auto decl = [&](const char* nm, cardinal::usize bytes, cardinal::u32 stride) {
        rg::BufferDesc d; d.name = nm; d.size_bytes = bytes; d.stride_bytes = stride;
        return g.declare_buffer(d);
    };
    const cardinal::usize px = static_cast<cardinal::usize>(width) * height;
    st->out_depth   = decl("vbuf.depth",   px * sizeof(float),         sizeof(float));
    st->out_prim_id = decl("vbuf.prim",    px * sizeof(cardinal::u32), sizeof(cardinal::u32));
    st->out_mat_id  = decl("vbuf.mat",     px * sizeof(cardinal::u32), sizeof(cardinal::u32));
    st->out_normal  = decl("vbuf.normal",  px * sizeof(cardinal::u32), sizeof(cardinal::u32));
    st->out_motion  = decl("vbuf.motion",  px * sizeof(cardinal::u32), sizeof(cardinal::u32));
    st->out_aux     = decl("vbuf.aux",     px * sizeof(cardinal::u32), sizeof(cardinal::u32));

    rg::PassDesc pd;
    pd.name = "VisibilityBufferPass";
    pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_tris,        rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_mat_ids,     rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{in_matrix,      rg::AccessMode::Read,  2});
    if (in_aux_per_tri.is_valid()) {
        pd.accesses.push_back(rg::ResourceAccess{in_aux_per_tri, rg::AccessMode::Read, 3});
    }
    pd.accesses.push_back(rg::ResourceAccess{st->out_depth,   rg::AccessMode::Write, 4});
    pd.accesses.push_back(rg::ResourceAccess{st->out_prim_id, rg::AccessMode::Write, 5});
    pd.accesses.push_back(rg::ResourceAccess{st->out_mat_id,  rg::AccessMode::Write, 6});
    pd.accesses.push_back(rg::ResourceAccess{st->out_normal,  rg::AccessMode::Write, 7});
    pd.accesses.push_back(rg::ResourceAccess{st->out_motion,  rg::AccessMode::Write, 8});
    pd.accesses.push_back(rg::ResourceAccess{st->out_aux,     rg::AccessMode::Write, 9});
    pd.record   = visbuf_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = (width  + 7) / 8;
    pd.dispatch_y = (height + 7) / 8;
    pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* VisibilityBufferPass::hlsl_source() noexcept {
    return R"(// Cardinal — VisibilityBufferPass.hlsl
//
// Tile-parallel. One thread per pixel. Same edge-function + barycentric
// inside-test as PolygonViewportPass; differs only in WHAT it writes:
// instead of an interpolated RGB, the V-Buf records prim_id, mat_id,
// oct-packed normal, motion vector, aux — the compact info needed to
// reconstruct attributes in a later compute resolver.
ByteAddressBuffer   in_tris    : register(t0);
ByteAddressBuffer   in_mat_ids : register(t1);
ByteAddressBuffer   in_matrix  : register(t2);
ByteAddressBuffer   in_aux     : register(t3);
RWByteAddressBuffer out_depth  : register(u0);
RWByteAddressBuffer out_prim   : register(u1);
RWByteAddressBuffer out_mat    : register(u2);
RWByteAddressBuffer out_normal : register(u3);
RWByteAddressBuffer out_motion : register(u4);
RWByteAddressBuffer out_aux    : register(u5);
cbuffer Params : register(b0) { uint width, height, tri_count, _pad; };
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // Per-pixel walk of every visible tri; identical algorithm to
    // PolygonViewportPass but writes the 6 V-Buf channels instead of
    // a single RGB. Body deferred to RHI compute commit.
}
)";
}

// =============================================================================
// HiZBuildPass
// =============================================================================
namespace {

cardinal::u32 compute_mip_layout(cardinal::u32 W, cardinal::u32 H,
                                 HiZBuildPass::State& st) noexcept
{
    cardinal::u32 off = 0;
    cardinal::u32 mw = W, mh = H;
    cardinal::u32 mips = 0;
    while (mw >= 1 && mh >= 1 && mips < kMaxHiZMips) {
        st.mip_widths[mips]    = mw;
        st.mip_heights[mips]   = mh;
        st.mip_offsets_f[mips] = off;
        off += mw * mh;
        ++mips;
        if (mw == 1 && mh == 1) break;
        mw = (mw > 1) ? (mw / 2) : 1;
        mh = (mh > 1) ? (mh / 2) : 1;
    }
    st.mip_count = mips;
    return off;   // total float count
}

void hiz_build_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<HiZBuildPass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height;
    if (W == 0 || H == 0 || st->mip_count == 0) {
        ec.dispatch(0, 1, 1);
        return;
    }
    const float* depth = static_cast<const float*>(ec.map_buffer_read(st->in_depth));
    float*       pyr   = static_cast<float*>      (ec.map_buffer_write(st->out_pyramid));
    if (!depth || !pyr) { ec.dispatch(0, 1, 1); return; }

    // Copy mip 0 from the input depth.
    for (cardinal::u32 i = 0; i < W * H; ++i) {
        pyr[st->mip_offsets_f[0] + i] = fzf(depth[i], 1.0f);
    }

    // Build each subsequent mip as max-of-2x2 footprint.
    for (cardinal::u32 m = 1; m < st->mip_count; ++m) {
        const cardinal::u32 prev_off = st->mip_offsets_f[m - 1];
        const cardinal::u32 prev_w   = st->mip_widths[m - 1];
        const cardinal::u32 prev_h   = st->mip_heights[m - 1];
        const cardinal::u32 cur_off  = st->mip_offsets_f[m];
        const cardinal::u32 cur_w    = st->mip_widths[m];
        const cardinal::u32 cur_h    = st->mip_heights[m];
        for (cardinal::u32 y = 0; y < cur_h; ++y) {
            for (cardinal::u32 x = 0; x < cur_w; ++x) {
                const cardinal::u32 px0 = x * 2;
                const cardinal::u32 py0 = y * 2;
                const cardinal::u32 px1 = (px0 + 1 < prev_w) ? (px0 + 1) : px0;
                const cardinal::u32 py1 = (py0 + 1 < prev_h) ? (py0 + 1) : py0;
                const float z00 = pyr[prev_off + py0 * prev_w + px0];
                const float z01 = pyr[prev_off + py0 * prev_w + px1];
                const float z10 = pyr[prev_off + py1 * prev_w + px0];
                const float z11 = pyr[prev_off + py1 * prev_w + px1];
                // Conservative-far reduction: pick the MAX of the 2x2.
                float zmax = z00;
                if (z01 > zmax) zmax = z01;
                if (z10 > zmax) zmax = z10;
                if (z11 > zmax) zmax = z11;
                pyr[cur_off + y * cur_w + x] = zmax;
            }
        }
    }
    st->mips_built = st->mip_count;
    st->max_depth  = pyr[st->mip_offsets_f[st->mip_count - 1]];

    ec.dispatch((W + 7) / 8, (H + 7) / 8, 1);
}

}  // namespace

cardinal::shared_ptr<HiZBuildPass::State> HiZBuildPass::add_to_graph(
    rg::Graph& g, rg::ResourceHandle in_depth,
    cardinal::u32 width, cardinal::u32 height)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_depth = in_depth;
    st->width    = width;
    st->height   = height;
    const cardinal::u32 total_floats = compute_mip_layout(width, height, *st);

    rg::BufferDesc od;
    od.name         = "hiz.pyramid";
    od.size_bytes   = static_cast<cardinal::usize>(total_floats) * sizeof(float);
    od.stride_bytes = sizeof(float);
    st->out_pyramid = g.declare_buffer(od);

    rg::PassDesc pd;
    pd.name = "HiZBuildPass";
    pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_depth,        rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{st->out_pyramid, rg::AccessMode::Write, 1});
    pd.record   = hiz_build_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = (width + 7) / 8;
    pd.dispatch_y = (height + 7) / 8;
    pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* HiZBuildPass::hlsl_source() noexcept {
    return R"(// Cardinal — HiZBuildPass.hlsl
//
// One dispatch per mip level. Each thread reads 4 texels from mip N-1
// and writes max(z) to mip N. Conservative-far reduction so the test
// pass's "is AABB occluded?" check is conservative (never falsely
// occludes a visible AABB).
ByteAddressBuffer   in_depth   : register(t0);
RWByteAddressBuffer out_pyramid : register(u0);
cbuffer Params : register(b0) {
    uint mip, mip_w, mip_h, _p0;
    uint prev_w, prev_h, prev_off_f, cur_off_f;
};
float load_f(ByteAddressBuffer b, uint o) { return asfloat(b.Load(o)); }
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint x = tid.x, y = tid.y;
    if (x >= mip_w || y >= mip_h) return;
    // Body deferred to RHI compute commit.
}
)";
}

// =============================================================================
// HiZOcclusionTestPass
// =============================================================================
namespace {

void hiz_test_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<HiZOcclusionTestPass::State*>(uctx);
    const cardinal::u32 N = st->aabb_count;
    if (N == 0 || st->mip_count == 0) {
        cardinal::u8* vis_short = static_cast<cardinal::u8*>(ec.map_buffer_write(st->out_visibility));
        if (vis_short) { for (cardinal::u32 i = 0; i < N; ++i) vis_short[i] = 1; }
        st->visible = N; st->occluded = 0;
        ec.dispatch(0, 1, 1);
        return;
    }
    const float* aabbs = static_cast<const float*>(ec.map_buffer_read(st->in_aabbs));
    const float* mat   = static_cast<const float*>(ec.map_buffer_read(st->in_matrix));
    const float* pyr   = static_cast<const float*>(ec.map_buffer_read(st->in_pyramid));
    cardinal::u8* vis  = static_cast<cardinal::u8*>(ec.map_buffer_write(st->out_visibility));
    if (!aabbs || !mat || !pyr || !vis) {
        ec.dispatch(0, 1, 1);
        return;
    }

    float m[16];
    for (int i = 0; i < 16; ++i) m[i] = fzf(mat[i], (i % 5 == 0) ? 1.0f : 0.0f);

    const float* mn_x = aabbs + 0 * N;
    const float* mn_y = aabbs + 1 * N;
    const float* mn_z = aabbs + 2 * N;
    const float* mx_x = aabbs + 3 * N;
    const float* mx_y = aabbs + 4 * N;
    const float* mx_z = aabbs + 5 * N;

    const cardinal::u32 W = st->width;
    const cardinal::u32 H = st->height;
    const float hw = 0.5f * static_cast<float>(W);
    const float hh = 0.5f * static_cast<float>(H);

    cardinal::u32 nv = 0, no = 0;

    for (cardinal::u32 i = 0; i < N; ++i) {
        const float ax = fzf(mn_x[i], 0.0f), ay = fzf(mn_y[i], 0.0f), az = fzf(mn_z[i], 0.0f);
        const float bx = fzf(mx_x[i], 0.0f), by = fzf(mx_y[i], 0.0f), bz = fzf(mx_z[i], 0.0f);
        const float corners[8][3] = {
            {ax, ay, az}, {bx, ay, az}, {ax, by, az}, {bx, by, az},
            {ax, ay, bz}, {bx, ay, bz}, {ax, by, bz}, {bx, by, bz},
        };
        float min_sx = 1.0e30f, max_sx = -1.0e30f;
        float min_sy = 1.0e30f, max_sy = -1.0e30f;
        float min_sz = 1.0e30f, max_sz = -1.0e30f;
        bool  any_behind = false;
        for (int c = 0; c < 8; ++c) {
            const float cx = m[0]  * corners[c][0] + m[1]  * corners[c][1] + m[2]  * corners[c][2] + m[3];
            const float cy = m[4]  * corners[c][0] + m[5]  * corners[c][1] + m[6]  * corners[c][2] + m[7];
            const float cz = m[8]  * corners[c][0] + m[9]  * corners[c][1] + m[10] * corners[c][2] + m[11];
            const float cw = m[12] * corners[c][0] + m[13] * corners[c][1] + m[14] * corners[c][2] + m[15];
            if (cw <= 0.0f || !isf(cw)) { any_behind = true; continue; }
            const float nx = cx / cw;
            const float ny = cy / cw;
            const float nz = 0.5f * (cz / cw + 1.0f);
            if (nx < min_sx) min_sx = nx; if (nx > max_sx) max_sx = nx;
            if (ny < min_sy) min_sy = ny; if (ny > max_sy) max_sy = ny;
            if (nz < min_sz) min_sz = nz; if (nz > max_sz) max_sz = nz;
        }
        // If anything is behind the camera, take the conservative path
        // and mark visible (the cluster cull will narrow this down later).
        if (any_behind || min_sx > max_sx) {
            vis[i] = 1; ++nv; continue;
        }
        // Clamp to NDC + map to pixel-rect.
        if (min_sx < -1.0f) min_sx = -1.0f;
        if (max_sx >  1.0f) max_sx =  1.0f;
        if (min_sy < -1.0f) min_sy = -1.0f;
        if (max_sy >  1.0f) max_sy =  1.0f;
        const float px_min = min_sx * hw + hw;
        const float px_max = max_sx * hw + hw;
        const float py_min = hh - max_sy * hh;
        const float py_max = hh - min_sy * hh;
        const float rect_w = px_max - px_min;
        const float rect_h = py_max - py_min;
        const float max_dim = (rect_w > rect_h) ? rect_w : rect_h;
        if (max_dim <= 0.0f) { vis[i] = 1; ++nv; continue; }
        // Pick mip such that 2^mip ≈ max_dim.
        cardinal::u32 mip = 0;
        {
            float d = max_dim;
            while (d > 1.0f && mip + 1 < st->mip_count) { d *= 0.5f; ++mip; }
        }
        const cardinal::u32 mw = st->mip_widths[mip];
        const cardinal::u32 mh = st->mip_heights[mip];
        const cardinal::u32 off = st->mip_offsets_f[mip];
        // Convert px rect → mip-coordinates by scaling W → mw.
        const float scale_x = static_cast<float>(mw) / static_cast<float>(W);
        const float scale_y = static_cast<float>(mh) / static_cast<float>(H);
        int mx0 = static_cast<int>(px_min * scale_x);
        int mx1 = static_cast<int>(px_max * scale_x) + 1;
        int my0 = static_cast<int>(py_min * scale_y);
        int my1 = static_cast<int>(py_max * scale_y) + 1;
        if (mx0 < 0) mx0 = 0;
        if (my0 < 0) my0 = 0;
        if (mx1 > static_cast<int>(mw)) mx1 = static_cast<int>(mw);
        if (my1 > static_cast<int>(mh)) my1 = static_cast<int>(mh);
        if (mx0 >= mx1 || my0 >= my1) { vis[i] = 1; ++nv; continue; }
        float max_pyramid = 0.0f;
        for (int y = my0; y < my1; ++y) {
            for (int x = mx0; x < mx1; ++x) {
                const float z = pyr[off + static_cast<cardinal::u32>(y) * mw +
                                          static_cast<cardinal::u32>(x)];
                if (z > max_pyramid) max_pyramid = z;
            }
        }
        // Occluded if AABB's nearest-z is BEHIND the pyramid's max-z.
        const bool occluded = (min_sz > max_pyramid);
        if (occluded) { vis[i] = 0; ++no; }
        else          { vis[i] = 1; ++nv; }
    }
    st->visible  = nv;
    st->occluded = no;
    ec.dispatch((N + 63) / 64, 1, 1);
}

}  // namespace

cardinal::shared_ptr<HiZOcclusionTestPass::State> HiZOcclusionTestPass::add_to_graph(
    rg::Graph& g,
    rg::ResourceHandle in_aabbs,
    rg::ResourceHandle in_matrix,
    cardinal::u32 aabb_count,
    const HiZBuildPass::State& hiz)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_aabbs   = in_aabbs;
    st->in_matrix  = in_matrix;
    st->in_pyramid = hiz.out_pyramid;
    st->width      = hiz.width;
    st->height     = hiz.height;
    st->mip_count  = hiz.mip_count;
    for (cardinal::u32 i = 0; i < hiz.mip_count; ++i) {
        st->mip_widths[i]    = hiz.mip_widths[i];
        st->mip_heights[i]   = hiz.mip_heights[i];
        st->mip_offsets_f[i] = hiz.mip_offsets_f[i];
    }
    st->aabb_count = aabb_count;

    rg::BufferDesc od;
    od.name         = "hiz.vis";
    od.size_bytes   = static_cast<cardinal::usize>(aabb_count);
    od.stride_bytes = 1;
    st->out_visibility = g.declare_buffer(od);

    rg::PassDesc pd;
    pd.name = "HiZOcclusionTestPass";
    pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_aabbs,           rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_matrix,          rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{hiz.out_pyramid,    rg::AccessMode::Read,  2});
    pd.accesses.push_back(rg::ResourceAccess{st->out_visibility, rg::AccessMode::Write, 3});
    pd.record   = hiz_test_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = (aabb_count + 63) / 64;
    pd.dispatch_y = 1; pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* HiZOcclusionTestPass::hlsl_source() noexcept {
    return R"(// Cardinal — HiZOcclusionTestPass.hlsl
//
// One thread per AABB. Project 8 corners, derive screen rect, sample
// the appropriate pyramid mip, compare AABB's nearest-z to mip's
// max-z. Output: 1 byte per AABB (1 = visible, 0 = occluded). Drop-in
// chain after FrustumCullPass.
ByteAddressBuffer   in_aabbs    : register(t0);
ByteAddressBuffer   in_matrix   : register(t1);
ByteAddressBuffer   in_pyramid  : register(t2);
RWByteAddressBuffer out_vis     : register(u0);
cbuffer Params : register(b0) {
    uint aabb_count, width, height, mip_count;
};
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // Body deferred to RHI compute commit.
}
)";
}

}  // namespace cardinal::render::gpu
