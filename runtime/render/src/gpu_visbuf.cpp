#include <cardinal/render/gpu_visbuf.hpp>

#include <cardinal/core/std/cmath.hpp>
#include <cardinal/core/std/algorithm.hpp>
#include <cardinal/core/std/cstring.hpp>
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
    // FIXED SRV LAYOUT: the aux access is ALWAYS declared (dummy when the
    // host has none) so the kernel's t3 never shifts; the push flag tells
    // the kernel whether t3 carries real per-tri aux. The CPU record keeps
    // gating on the ORIGINAL handle in State.
    rg::ResourceHandle aux_bind = in_aux_per_tri;
    if (!aux_bind.is_valid())
        aux_bind = g.declare_buffer(rg::BufferDesc{"vbuf.noaux", 4, 4, true});
    pd.accesses.push_back(rg::ResourceAccess{in_tris,        rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_mat_ids,     rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{in_matrix,      rg::AccessMode::Read,  2});
    pd.accesses.push_back(rg::ResourceAccess{aux_bind,       rg::AccessMode::Read,  3});
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
    pd.compute_hlsl = VisibilityBufferPass::hlsl_source();
    pd.push_constant_size = 16;
    pd.push_constants.clear();
    auto push_u32 = [&pd](cardinal::u32 x) {
        pd.push_constants.push_back(static_cast<cardinal::u8>( x        & 0xFF));
        pd.push_constants.push_back(static_cast<cardinal::u8>((x >>  8) & 0xFF));
        pd.push_constants.push_back(static_cast<cardinal::u8>((x >> 16) & 0xFF));
        pd.push_constants.push_back(static_cast<cardinal::u8>((x >> 24) & 0xFF));
    };
    push_u32(width);
    push_u32(height);
    push_u32(triangle_count);
    push_u32(in_aux_per_tri.is_valid() ? 1u : 0u);
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* VisibilityBufferPass::hlsl_source() noexcept {
    // One thread per PIXEL, looping every triangle in submission order —
    // deliberately mirrors the CPU rasterizer's winner selection: the CPU
    // iterates triangles outer and accepts on strict z < depth, so a per-pixel
    // inner loop over the SAME ascending order with the same strict compare
    // resolves to the identical winner (ties keep the earlier triangle).
    // O(pixels * tris) — validation-first; a binned rasterizer is the
    // optimization slice once numerics are pinned.
    return
    "// Cardinal - VisibilityBufferPass.hlsl - per-pixel triangle-loop raster.\n"
    "ByteAddressBuffer   InTris   : register(t0);   // 9 floats per tri\n"
    "ByteAddressBuffer   InMatIds : register(t1);   // u32 per tri\n"
    "ByteAddressBuffer   InMatrix : register(t2);   // 16 floats row-major VP\n"
    "ByteAddressBuffer   InAux    : register(t3);   // u32 per tri (or dummy)\n"
    "RWByteAddressBuffer OutDepth : register(u0);\n"
    "RWByteAddressBuffer OutPrim  : register(u1);\n"
    "RWByteAddressBuffer OutMat   : register(u2);\n"
    "RWByteAddressBuffer OutNrm   : register(u3);\n"
    "RWByteAddressBuffer OutMV    : register(u4);\n"
    "RWByteAddressBuffer OutAux   : register(u5);\n"
    "struct PushT { uint w; uint h; uint triCount; uint hasAux; };\n"
    "[[vk::push_constant]] ConstantBuffer<PushT> pc : register(b0);\n"
    "#define gW pc.w\n"
    "#define gH pc.h\n"
    "#define gTriCount pc.triCount\n"
    "#define gHasAux pc.hasAux\n"
    "float edge_fn(float ax, float ay, float bx, float by, float cx, float cy) {\n"
    "  return (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);\n"
    "}\n"
    "uint oct_pack(float3 n) {\n"
    "  float inv_l1 = 1.0 / (abs(n.x) + abs(n.y) + abs(n.z) + 1.0e-8);\n"
    "  float ox = n.x * inv_l1;\n"
    "  float oz = n.z * inv_l1;\n"
    "  if (n.y < 0.0) {\n"
    "    float tx = (1.0 - abs(oz)) * (ox >= 0.0 ? 1.0 : -1.0);\n"
    "    float tz = (1.0 - abs(ox)) * (oz >= 0.0 ? 1.0 : -1.0);\n"
    "    ox = tx; oz = tz;\n"
    "  }\n"
    "  uint ix = uint(int(clamp(ox, -1.0, 1.0) * 32767.0)) & 0xFFFFu;\n"
    "  uint iz = uint(int(clamp(oz, -1.0, 1.0) * 32767.0)) & 0xFFFFu;\n"
    "  return ix | (iz << 16);\n"
    "}\n"
    "[numthreads(8,8,1)]\n"
    "void CSMain(uint3 tid : SV_DispatchThreadID){\n"
    "  if (tid.x >= gW || tid.y >= gH) return;\n"
    "  uint pix = tid.y*gW + tid.x;\n"
    "  float bestZ = 1.0;\n"
    "  uint bestPrim = 0xFFFFFFFFu, bestMat = 0xFFFFFFFFu, bestNrm = 0u, bestAux = 0u;\n"
    "  float4 r0 = asfloat(InMatrix.Load4(0));\n"
    "  float4 r1 = asfloat(InMatrix.Load4(16));\n"
    "  float4 r2 = asfloat(InMatrix.Load4(32));\n"
    "  float4 r3 = asfloat(InMatrix.Load4(48));\n"
    "  float hw = 0.5*(float)gW, hh = 0.5*(float)gH;\n"
    "  float fx = (float)tid.x + 0.5, fy = (float)tid.y + 0.5;\n"
    "  [loop] for (uint t = 0u; t < gTriCount; ++t) {\n"
    "    uint b = t*36u;\n"
    "    float3 v0 = asfloat(InTris.Load3(b+0));\n"
    "    float3 v1 = asfloat(InTris.Load3(b+12));\n"
    "    float3 v2 = asfloat(InTris.Load3(b+24));\n"
    "    precise float sx[3], sy[3], sz[3], sw[3];\n"
    "    [unroll] for (uint k = 0u; k < 3u; ++k) {\n"
    "      float3 p = (k == 0u) ? v0 : ((k == 1u) ? v1 : v2);\n"
    "      precise float cw = dot(r3.xyz, p) + r3.w;\n"
    "      sw[k] = cw;\n"
    "      if (abs(cw) < 1.0e-8 || !isfinite(cw) || cw <= 0.0) {\n"
    "        sx[k] = 0.0; sy[k] = 0.0; sz[k] = 1.0e30;\n"
    "      } else {\n"
    "        sx[k] = (dot(r0.xyz, p) + r0.w) / cw;\n"
    "        sy[k] = (dot(r1.xyz, p) + r1.w) / cw;\n"
    "        sz[k] = (dot(r2.xyz, p) + r2.w) / cw;\n"
    "      }\n"
    "    }\n"
    "    if (sw[0] <= 0.0 || sw[1] <= 0.0 || sw[2] <= 0.0) continue;\n"
    "    if (sx[0] < -1.0 && sx[1] < -1.0 && sx[2] < -1.0) continue;\n"
    "    if (sx[0] >  1.0 && sx[1] >  1.0 && sx[2] >  1.0) continue;\n"
    "    if (sy[0] < -1.0 && sy[1] < -1.0 && sy[2] < -1.0) continue;\n"
    "    if (sy[0] >  1.0 && sy[1] >  1.0 && sy[2] >  1.0) continue;\n"
    "    precise float px[3], py[3], pz[3];\n"
    "    [unroll] for (uint k2 = 0u; k2 < 3u; ++k2) {\n"
    "      px[k2] = sx[k2]*hw + hw;\n"
    "      py[k2] = hh - sy[k2]*hh;\n"
    "      pz[k2] = 0.5*(sz[k2] + 1.0);\n"
    "    }\n"
    "    float min_x = min(px[0], min(px[1], px[2]));\n"
    "    float max_x = max(px[0], max(px[1], px[2]));\n"
    "    float min_y = min(py[0], min(py[1], py[2]));\n"
    "    float max_y = max(py[0], max(py[1], py[2]));\n"
    "    int x0 = (int)min_x;      if (x0 < 0) x0 = 0;\n"
    "    int x1 = (int)max_x + 1;  if (x1 > (int)gW) x1 = (int)gW;\n"
    "    int y0 = (int)min_y;      if (y0 < 0) y0 = 0;\n"
    "    int y1 = (int)max_y + 1;  if (y1 > (int)gH) y1 = (int)gH;\n"
    "    if ((int)tid.x < x0 || (int)tid.x >= x1) continue;\n"
    "    if ((int)tid.y < y0 || (int)tid.y >= y1) continue;\n"
    "    float area = edge_fn(px[0], py[0], px[1], py[1], px[2], py[2]);\n"
    "    if (abs(area) < 1.0e-6) continue;\n"
    "    float inv_area = 1.0 / area;\n"
    "    float w0 = edge_fn(px[1], py[1], px[2], py[2], fx, fy);\n"
    "    float w1 = edge_fn(px[2], py[2], px[0], py[0], fx, fy);\n"
    "    float w2 = edge_fn(px[0], py[0], px[1], py[1], fx, fy);\n"
    "    bool inside = (area > 0.0) ? (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0)\n"
    "                               : (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0);\n"
    "    if (!inside) continue;\n"
    "    precise float z = (w0*inv_area)*pz[0] + (w1*inv_area)*pz[1] + (w2*inv_area)*pz[2];\n"
    "    if (!isfinite(z) || z < 0.0 || z > 1.0) continue;\n"
    "    if (z >= bestZ) continue;\n"
    "    float3 nrm = cross(v1 - v0, v2 - v0);\n"
    "    float  nl  = sqrt(dot(nrm, nrm));\n"
    "    nrm = (nl > 1.0e-8) ? nrm / nl : float3(0, 1, 0);\n"
    "    bestZ    = z;\n"
    "    bestPrim = t;\n"
    "    bestMat  = InMatIds.Load(t*4u);\n"
    "    bestNrm  = oct_pack(nrm);\n"
    "    bestAux  = (gHasAux != 0u) ? InAux.Load(t*4u) : t;\n"
    "  }\n"
    "  OutDepth.Store(pix*4u, asuint(bestZ));\n"
    "  OutPrim.Store (pix*4u, bestPrim);\n"
    "  OutMat.Store  (pix*4u, bestMat);\n"
    "  OutNrm.Store  (pix*4u, bestNrm);\n"
    "  OutMV.Store   (pix*4u, 0u);\n"
    "  OutAux.Store  (pix*4u, bestAux);\n"
    "}\n";
}

// =============================================================================
// TriangleBinPass — tile binning + triangle setup (see gpu_visbuf.hpp).
// =============================================================================
namespace {

// One triangle's screen-space setup, mirroring visbuf_record's projection
// op-for-op so the binned raster reproduces the reference depth bit-exactly.
struct TriSetup {
    float px[3], py[3], pz[3];
    float inv_area;
    cardinal::u32 flags;     // bit0 valid, bit1 area > 0
    cardinal::u32 mid, nrm, aux;
    int x0, y0, x1, y1;      // bbox, x1/y1 exclusive
    bool offscreen;          // rejected by the behind/offscreen test (stats)
};

inline void setup_tri(const float* v, const float* m, cardinal::u32 W, cardinal::u32 H,
                      cardinal::u32 t, const cardinal::u32* mat_ids,
                      const cardinal::u32* aux_in, TriSetup& o) noexcept {
    o = TriSetup{};
    const float hw = 0.5f * static_cast<float>(W);
    const float hh = 0.5f * static_cast<float>(H);
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
            sx[k] = cx / cw; sy[k] = cy / cw; sz[k] = cz / cw;
        }
    }
    const bool any_behind = (sw[0] <= 0.0f) || (sw[1] <= 0.0f) || (sw[2] <= 0.0f);
    const bool all_left   = sx[0] < -1.0f && sx[1] < -1.0f && sx[2] < -1.0f;
    const bool all_right  = sx[0] >  1.0f && sx[1] >  1.0f && sx[2] >  1.0f;
    const bool all_below  = sy[0] < -1.0f && sy[1] < -1.0f && sy[2] < -1.0f;
    const bool all_above  = sy[0] >  1.0f && sy[1] >  1.0f && sy[2] >  1.0f;
    if (any_behind || all_left || all_right || all_below || all_above) {
        o.offscreen = true; return;
    }
    for (int k = 0; k < 3; ++k) {
        o.px[k] = sx[k] * hw + hw;
        o.py[k] = hh - sy[k] * hh;
        o.pz[k] = 0.5f * (sz[k] + 1.0f);
    }
    float min_x = o.px[0], max_x = o.px[0];
    if (o.px[1] < min_x) min_x = o.px[1]; if (o.px[1] > max_x) max_x = o.px[1];
    if (o.px[2] < min_x) min_x = o.px[2]; if (o.px[2] > max_x) max_x = o.px[2];
    float min_y = o.py[0], max_y = o.py[0];
    if (o.py[1] < min_y) min_y = o.py[1]; if (o.py[1] > max_y) max_y = o.py[1];
    if (o.py[2] < min_y) min_y = o.py[2]; if (o.py[2] > max_y) max_y = o.py[2];
    o.x0 = static_cast<int>(min_x);     if (o.x0 < 0) o.x0 = 0;
    o.x1 = static_cast<int>(max_x) + 1; if (o.x1 > static_cast<int>(W)) o.x1 = static_cast<int>(W);
    o.y0 = static_cast<int>(min_y);     if (o.y0 < 0) o.y0 = 0;
    o.y1 = static_cast<int>(max_y) + 1; if (o.y1 > static_cast<int>(H)) o.y1 = static_cast<int>(H);
    if (o.x0 >= o.x1 || o.y0 >= o.y1) { o.offscreen = true; return; }
    const float area = edge_fn(o.px[0], o.py[0], o.px[1], o.py[1], o.px[2], o.py[2]);
    if (cardinal::abs(area) < 1.0e-6f) return;          // degenerate: invalid, not "offscreen"
    o.inv_area = 1.0f / area;
    o.flags    = 1u | (area > 0.0f ? 2u : 0u);
    o.mid      = mat_ids ? mat_ids[t] : t;
    o.aux      = aux_in  ? aux_in[t]  : t;
    const float ux = v[3] - v[0], uy = v[4] - v[1], uz = v[5] - v[2];
    const float vx = v[6] - v[0], vy = v[7] - v[1], vz = v[8] - v[2];
    float nx = uy * vz - uz * vy;
    float ny = uz * vx - ux * vz;
    float nz = ux * vy - uy * vx;
    const float nl = cardinal::sqrt(nx * nx + ny * ny + nz * nz);
    if (nl > 1.0e-8f) { nx /= nl; ny /= nl; nz /= nl; }
    else              { nx = 0; ny = 1; nz = 0; }
    o.nrm = oct_pack(nx, ny, nz);
}

// SoAoV setup planes (see gpu_visbuf.hpp): dword q of record t lives at
// base[(q/4 * npad + t) * 4 + q%4].
inline void store_setup(cardinal::u32* base, cardinal::usize npad, cardinal::u32 t,
                        const TriSetup& s) noexcept {
    cardinal::u32 rec[16];
    float f[10] = { s.px[0], s.py[0], s.px[1], s.py[1], s.px[2], s.py[2],
                    s.pz[0], s.pz[1], s.pz[2], s.inv_area };
    for (int i = 0; i < 10; ++i) cardinal::memcpy(rec + i, &f[i], 4);
    rec[10] = s.flags; rec[11] = s.mid; rec[12] = s.nrm; rec[13] = s.aux;
    rec[14] = static_cast<cardinal::u32>(s.x0) | (static_cast<cardinal::u32>(s.y0) << 16);
    rec[15] = static_cast<cardinal::u32>(s.x1) | (static_cast<cardinal::u32>(s.y1) << 16);
    for (cardinal::u32 k = 0; k < kSetupPlanes; ++k)
        cardinal::memcpy(base + (k * npad + t) * 4, rec + k * 4, 16);
}

inline void load_setup(const cardinal::u32* base, cardinal::usize npad, cardinal::u32 t,
                       TriSetup& s) noexcept {
    cardinal::u32 rec[16];
    for (cardinal::u32 k = 0; k < kSetupPlanes; ++k)
        cardinal::memcpy(rec + k * 4, base + (k * npad + t) * 4, 16);
    float f[10];
    for (int i = 0; i < 10; ++i) cardinal::memcpy(&f[i], rec + i, 4);
    s.px[0] = f[0]; s.py[0] = f[1]; s.px[1] = f[2]; s.py[1] = f[3];
    s.px[2] = f[4]; s.py[2] = f[5]; s.pz[0] = f[6]; s.pz[1] = f[7]; s.pz[2] = f[8];
    s.inv_area = f[9];
    s.flags = rec[10]; s.mid = rec[11]; s.nrm = rec[12]; s.aux = rec[13];
    s.x0 = static_cast<int>(rec[14] & 0xFFFFu); s.y0 = static_cast<int>(rec[14] >> 16);
    s.x1 = static_cast<int>(rec[15] & 0xFFFFu); s.y1 = static_cast<int>(rec[15] >> 16);
    s.offscreen = false;
}

// Conservative tile-vs-triangle overlap over the tile's PIXEL-CENTRE
// rectangle (∩ the tri bbox). The edge function is affine, so its extreme
// over a rectangle sits at a corner; a tile whose four corners are all
// outside one edge (beyond a float-rounding margin) has no covered pixel and
// is not binned. The margin scales with the edge length × viewport extent
// (the magnitude of the terms whose rounding could flip a sign), so it is
// strictly conservative: the raster's own inside test stays the arbiter.
inline bool tile_may_overlap(const TriSetup& s, int rx0, int ry0, int rx1, int ry1,
                             float extent) noexcept {
    const float cx0 = static_cast<float>(rx0) + 0.5f, cy0 = static_cast<float>(ry0) + 0.5f;
    const float cx1 = static_cast<float>(rx1 - 1) + 0.5f, cy1 = static_cast<float>(ry1 - 1) + 0.5f;
    const bool pos = (s.flags & 2u) != 0u;
    for (int e = 0; e < 3; ++e) {
        const int a = (e + 1) % 3, b = (e + 2) % 3;     // edge (a→b) opposite vertex e
        const float ax = s.px[a], ay = s.py[a], bx = s.px[b], by = s.py[b];
        const float e00 = edge_fn(ax, ay, bx, by, cx0, cy0);
        const float e10 = edge_fn(ax, ay, bx, by, cx1, cy0);
        const float e01 = edge_fn(ax, ay, bx, by, cx0, cy1);
        const float e11 = edge_fn(ax, ay, bx, by, cx1, cy1);
        const float eps = 1.0e-5f * (cardinal::abs(bx - ax) + cardinal::abs(by - ay)) * extent;
        const float mx = cardinal::max(cardinal::max(e00, e10), cardinal::max(e01, e11));
        const float mn = cardinal::min(cardinal::min(e00, e10), cardinal::min(e01, e11));
        if (pos  && mx < -eps) return false;
        if (!pos && mn >  eps) return false;
    }
    return true;
}

void bin_clear_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<TriangleBinPass::State*>(uctx);
    auto* counts = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_bin_counts));
    const cardinal::u32 tiles = st->tiles_x * st->tiles_y;
    if (!counts) { ec.dispatch(0, 1, 1); return; }
    for (cardinal::u32 i = 0; i < tiles; ++i) counts[i] = 0u;
    ec.dispatch(grid_stride_groups(tiles), 1, 1);
}

void bin_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<TriangleBinPass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height, T = st->triangle_count;
    const float* tris = static_cast<const float*>(ec.map_buffer_read(st->in_tris));
    const cardinal::u32* mat_ids = static_cast<const cardinal::u32*>(ec.map_buffer_read(st->in_mat_ids));
    const float* mat = static_cast<const float*>(ec.map_buffer_read(st->in_matrix));
    const cardinal::u32* aux_in = st->in_aux_per_tri.is_valid()
        ? static_cast<const cardinal::u32*>(ec.map_buffer_read(st->in_aux_per_tri)) : nullptr;
    const cardinal::u8* vis_in = st->in_vis_per_group.is_valid()
        ? static_cast<const cardinal::u8*>(ec.map_buffer_read(st->in_vis_per_group)) : nullptr;
    const cardinal::u32 gs = st->vis_group_size == 0 ? 1u : st->vis_group_size;
    auto* setup  = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_setup));
    auto* counts = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_bin_counts));
    auto* bins   = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_bins));
    st->tris_binned = st->tris_rejected = st->bin_entries = st->overflow_tiles = 0;
    st->tiles_rejected = 0; st->tris_culled = 0;
    if (!setup || !counts || !bins || W == 0 || H == 0) { ec.dispatch(0, 1, 1); return; }
    const cardinal::u32 cap = st->bin_capacity;
    const cardinal::usize npad = setup_npad(T);
    float m[16];
    for (int i = 0; i < 16; ++i) m[i] = mat ? fzf(mat[i], (i % 5 == 0) ? 1.0f : 0.0f)
                                            : ((i % 5 == 0) ? 1.0f : 0.0f);
    for (cardinal::u32 t = 0; t < T; ++t) {
        if (vis_in && vis_in[t / gs] == 0u) {
            // Culled group (frustum / occlusion): an all-zero record keeps the
            // overflow walk honest, nothing is set up or binned.
            store_setup(setup, npad, t, TriSetup{});
            ++st->tris_culled;
            continue;
        }
        TriSetup s;
        if (tris) setup_tri(tris + t * 9, m, W, H, t, mat_ids, aux_in, s);
        else      s = TriSetup{};
        store_setup(setup, npad, t, s);
        if (!(s.flags & 1u)) { ++st->tris_rejected; continue; }
        ++st->tris_binned;
        const cardinal::u32 tx0 = static_cast<cardinal::u32>(s.x0) / kVisTileSize;
        const cardinal::u32 tx1 = static_cast<cardinal::u32>(s.x1 - 1) / kVisTileSize;
        const cardinal::u32 ty0 = static_cast<cardinal::u32>(s.y0) / kVisTileSize;
        const cardinal::u32 ty1 = static_cast<cardinal::u32>(s.y1 - 1) / kVisTileSize;
        const float extent = static_cast<float>(W + H);
        for (cardinal::u32 ty = ty0; ty <= ty1; ++ty)
            for (cardinal::u32 tx = tx0; tx <= tx1; ++tx) {
                // Tile ∩ bbox rectangle, then the conservative edge reject.
                const int rx0 = cardinal::max(static_cast<int>(tx * kVisTileSize), s.x0);
                const int ry0 = cardinal::max(static_cast<int>(ty * kVisTileSize), s.y0);
                const int rx1 = cardinal::min(static_cast<int>((tx + 1) * kVisTileSize), s.x1);
                const int ry1 = cardinal::min(static_cast<int>((ty + 1) * kVisTileSize), s.y1);
                if (rx0 >= rx1 || ry0 >= ry1) continue;
                if (!tile_may_overlap(s, rx0, ry0, rx1, ry1, extent)) { ++st->tiles_rejected; continue; }
                const cardinal::u32 tile = ty * st->tiles_x + tx;
                const cardinal::u32 idx  = counts[tile]++;
                ++st->bin_entries;
                if (idx < cap) bins[static_cast<cardinal::usize>(tile) * cap + idx] = t;
            }
    }
    const cardinal::u32 tiles = st->tiles_x * st->tiles_y;
    for (cardinal::u32 i = 0; i < tiles; ++i) if (counts[i] > cap) ++st->overflow_tiles;
    ec.dispatch(grid_stride_groups(T), 1, 1);
}

}  // namespace

cardinal::shared_ptr<TriangleBinPass::State> TriangleBinPass::add_to_graph(
    rg::Graph& g,
    rg::ResourceHandle in_tris,
    rg::ResourceHandle in_mat_ids,
    rg::ResourceHandle in_matrix,
    rg::ResourceHandle in_aux_per_tri,
    cardinal::u32 width, cardinal::u32 height,
    cardinal::u32 triangle_count,
    cardinal::u32 bin_capacity,
    rg::ResourceHandle in_vis_per_group,
    cardinal::u32 vis_group_size)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_tris        = in_tris;
    st->in_mat_ids     = in_mat_ids;
    st->in_matrix      = in_matrix;
    st->in_aux_per_tri = in_aux_per_tri;
    st->in_vis_per_group = in_vis_per_group;
    st->vis_group_size = vis_group_size == 0 ? 1u : vis_group_size;
    st->width          = width;
    st->height         = height;
    st->triangle_count = triangle_count;
    st->tiles_x        = (width  + kVisTileSize - 1) / kVisTileSize;
    st->tiles_y        = (height + kVisTileSize - 1) / kVisTileSize;
    st->bin_capacity   = bin_capacity == 0 ? 1u : bin_capacity;
    const cardinal::usize tiles = static_cast<cardinal::usize>(st->tiles_x) * st->tiles_y;
    const cardinal::usize T     = triangle_count == 0 ? 1u : triangle_count;   // never a 0-byte buffer
    auto decl = [&](const char* nm, cardinal::usize bytes, cardinal::u32 stride) {
        rg::BufferDesc d; d.name = nm; d.size_bytes = bytes; d.stride_bytes = stride;
        return g.declare_buffer(d);
    };
    st->out_setup      = decl("bin.setup",  kSetupPlanes * setup_npad(static_cast<cardinal::u32>(T)) * 16u, 16u);
    st->out_bin_counts = decl("bin.counts", (tiles == 0 ? 1u : tiles) * 4u, 4u);
    st->out_bins       = decl("bin.ids",    (tiles == 0 ? 1u : tiles) * st->bin_capacity * 4u, 4u);

    rg::ResourceHandle aux_bind = in_aux_per_tri;
    if (!aux_bind.is_valid())
        aux_bind = g.declare_buffer(rg::BufferDesc{"bin.noaux", 4, 4, true});
    rg::ResourceHandle vis_bind = in_vis_per_group;
    if (!vis_bind.is_valid())
        vis_bind = g.declare_buffer(rg::BufferDesc{"bin.novis", 4, 4, true});

    auto push_u32 = [](rg::PassDesc& pd, cardinal::u32 x) {
        for (int k = 0; k < 4; ++k)
            pd.push_constants.push_back(static_cast<cardinal::u8>((x >> (8 * k)) & 0xFF));
    };
    {
        rg::PassDesc pd;
        pd.name = "TriangleBinClearPass";
        pd.kind = rg::PassKind::Compute;
        pd.accesses.push_back(rg::ResourceAccess{st->out_bin_counts, rg::AccessMode::Write, 0});
        pd.record     = bin_clear_record;
        pd.user_ctx   = st.get();
        pd.dispatch_x = grid_stride_groups(static_cast<cardinal::u32>(tiles));
        pd.compute_hlsl = TriangleBinPass::hlsl_clear_source();
        pd.push_constant_size = 16;
        push_u32(pd, static_cast<cardinal::u32>(tiles)); push_u32(pd, pd.dispatch_x); push_u32(pd, 0u); push_u32(pd, 0u);
        g.add_pass(cardinal::move(pd));
    }
    {
        rg::PassDesc pd;
        pd.name = "TriangleBinPass";
        pd.kind = rg::PassKind::Compute;
        // t0 tris, t1 mat ids, t2 matrix, t3 aux (or dummy), t4 group visibility
        // (or dummy); u0 setup, u1 counts, u2 bins.
        pd.accesses.push_back(rg::ResourceAccess{in_tris,            rg::AccessMode::Read,      0});
        pd.accesses.push_back(rg::ResourceAccess{in_mat_ids,         rg::AccessMode::Read,      1});
        pd.accesses.push_back(rg::ResourceAccess{in_matrix,          rg::AccessMode::Read,      2});
        pd.accesses.push_back(rg::ResourceAccess{aux_bind,           rg::AccessMode::Read,      3});
        pd.accesses.push_back(rg::ResourceAccess{vis_bind,           rg::AccessMode::Read,      4});
        pd.accesses.push_back(rg::ResourceAccess{st->out_setup,      rg::AccessMode::Write,     5});
        pd.accesses.push_back(rg::ResourceAccess{st->out_bin_counts, rg::AccessMode::ReadWrite, 6});
        pd.accesses.push_back(rg::ResourceAccess{st->out_bins,       rg::AccessMode::Write,     7});
        pd.record     = bin_record;
        pd.user_ctx   = st.get();
        pd.dispatch_x = grid_stride_groups(triangle_count);
        pd.compute_hlsl = TriangleBinPass::hlsl_source();
        pd.push_constant_size = 48;
        push_u32(pd, width); push_u32(pd, height); push_u32(pd, triangle_count);
        push_u32(pd, (in_aux_per_tri.is_valid() ? 1u : 0u) | (in_vis_per_group.is_valid() ? 2u : 0u));
        push_u32(pd, st->tiles_x); push_u32(pd, st->bin_capacity);
        push_u32(pd, pd.dispatch_x);                                          // groups (grid stride)
        push_u32(pd, static_cast<cardinal::u32>(setup_npad(triangle_count))); // SoAoV plane stride
        push_u32(pd, st->vis_group_size); push_u32(pd, 0u); push_u32(pd, 0u); push_u32(pd, 0u);
        g.add_pass(cardinal::move(pd));
    }
    return st;
}

const char* TriangleBinPass::hlsl_clear_source() noexcept {
    return
    "// Cardinal - TriangleBinClearPass.hlsl - zero tile counts; 256-thread blocks,\n"
    "// grid-stride over the tile range.\n"
    "RWByteAddressBuffer OutCounts : register(u0);\n"
    "struct PushT { uint tiles; uint groups; uint p1; uint p2; };\n"
    "[[vk::push_constant]] ConstantBuffer<PushT> pc : register(b0);\n"
    "[numthreads(256,1,1)]\n"
    "void CSMain(uint3 tid : SV_DispatchThreadID){\n"
    "  uint stride = max(pc.groups, 1u) * 256u;\n"
    "  for (uint i = tid.x; i < pc.tiles; i += stride) OutCounts.Store(i*4u, 0u);\n"
    "}\n";
}

const char* TriangleBinPass::hlsl_source() noexcept {
    return
    "// Cardinal - TriangleBinPass.hlsl - one thread per triangle: project + set up\n"
    "// once (mirrors visbuf setup op-for-op), then append the id to every 16x16\n"
    "// tile the bbox touches. Bin order is atomic (unordered); the raster kernel\n"
    "// breaks depth ties toward the LOWER triangle id so results stay exact.\n"
    "ByteAddressBuffer   InTris    : register(t0);   // 9 floats per tri\n"
    "ByteAddressBuffer   InMatIds  : register(t1);   // u32 per tri\n"
    "ByteAddressBuffer   InMatrix  : register(t2);   // 16 floats row-major VP\n"
    "ByteAddressBuffer   InAux     : register(t3);   // u32 per tri (or dummy)\n"
    "ByteAddressBuffer   InVis     : register(t4);   // u8 per tri group (or dummy)\n"
    "RWByteAddressBuffer OutSetup  : register(u0);   // 64 B per tri\n"
    "RWByteAddressBuffer OutCounts : register(u1);   // u32 per tile\n"
    "RWByteAddressBuffer OutBins   : register(u2);   // cap u32 per tile\n"
    "struct PushT { uint w; uint h; uint triCount; uint flags; uint tilesX; uint cap; uint groups; uint npad;\n"
    "               uint groupSize; uint p0; uint p1; uint p2; };   // flags: bit0 aux, bit1 visibility gate\n"
    "[[vk::push_constant]] ConstantBuffer<PushT> pc : register(b0);\n"
    "float edge_fn(float ax, float ay, float bx, float by, float cx, float cy) {\n"
    "  return (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);\n"
    "}\n"
    "uint oct_pack(float3 n) {\n"
    "  float inv_l1 = 1.0 / (abs(n.x) + abs(n.y) + abs(n.z) + 1.0e-8);\n"
    "  float ox = n.x * inv_l1;\n"
    "  float oz = n.z * inv_l1;\n"
    "  if (n.y < 0.0) {\n"
    "    float tx = (1.0 - abs(oz)) * (ox >= 0.0 ? 1.0 : -1.0);\n"
    "    float tz = (1.0 - abs(ox)) * (oz >= 0.0 ? 1.0 : -1.0);\n"
    "    ox = tx; oz = tz;\n"
    "  }\n"
    "  uint ix = uint(int(clamp(ox, -1.0, 1.0) * 32767.0)) & 0xFFFFu;\n"
    "  uint iz = uint(int(clamp(oz, -1.0, 1.0) * 32767.0)) & 0xFFFFu;\n"
    "  return ix | (iz << 16);\n"
    "}\n"
    "// 256-thread blocks + grid-stride loop over the triangle range.\n"
    "[numthreads(256,1,1)]\n"
    "void CSMain(uint3 tid : SV_DispatchThreadID){\n"
    "  uint stride = max(pc.groups, 1u) * 256u;\n"
    "  for (uint t = tid.x; t < pc.triCount; t += stride) {\n"
    "  if ((pc.flags & 2u) != 0u) {\n"
    "    uint gi = t / max(pc.groupSize, 1u);\n"
    "    uint gv = (InVis.Load(gi & ~3u) >> ((gi & 3u) * 8u)) & 0xFFu;\n"
    "    if (gv == 0u) {   // culled group: zero record, nothing binned\n"
    "      OutSetup.Store4((0u*pc.npad + t)*16u, uint4(0u,0u,0u,0u));\n"
    "      OutSetup.Store4((1u*pc.npad + t)*16u, uint4(0u,0u,0u,0u));\n"
    "      OutSetup.Store4((2u*pc.npad + t)*16u, uint4(0u,0u,0u,0u));\n"
    "      OutSetup.Store4((3u*pc.npad + t)*16u, uint4(0u,0u,0u,0u));\n"
    "      continue;\n"
    "    }\n"
    "  }\n"
    "  uint b = t*36u;\n"
    "  float3 v0 = asfloat(InTris.Load3(b+0));\n"
    "  float3 v1 = asfloat(InTris.Load3(b+12));\n"
    "  float3 v2 = asfloat(InTris.Load3(b+24));\n"
    "  float4 r0 = asfloat(InMatrix.Load4(0));\n"
    "  float4 r1 = asfloat(InMatrix.Load4(16));\n"
    "  float4 r2 = asfloat(InMatrix.Load4(32));\n"
    "  float4 r3 = asfloat(InMatrix.Load4(48));\n"
    "  float hw = 0.5*(float)pc.w, hh = 0.5*(float)pc.h;\n"
    "  precise float sx[3], sy[3], sz[3], sw[3];\n"
    "  [unroll] for (uint k = 0u; k < 3u; ++k) {\n"
    "    float3 p = (k == 0u) ? v0 : ((k == 1u) ? v1 : v2);\n"
    "    precise float cw = dot(r3.xyz, p) + r3.w;\n"
    "    sw[k] = cw;\n"
    "    if (abs(cw) < 1.0e-8 || !isfinite(cw) || cw <= 0.0) {\n"
    "      sx[k] = 0.0; sy[k] = 0.0; sz[k] = 1.0e30;\n"
    "    } else {\n"
    "      sx[k] = (dot(r0.xyz, p) + r0.w) / cw;\n"
    "      sy[k] = (dot(r1.xyz, p) + r1.w) / cw;\n"
    "      sz[k] = (dot(r2.xyz, p) + r2.w) / cw;\n"
    "    }\n"
    "  }\n"
    "  bool reject = (sw[0] <= 0.0 || sw[1] <= 0.0 || sw[2] <= 0.0)\n"
    "             || (sx[0] < -1.0 && sx[1] < -1.0 && sx[2] < -1.0)\n"
    "             || (sx[0] >  1.0 && sx[1] >  1.0 && sx[2] >  1.0)\n"
    "             || (sy[0] < -1.0 && sy[1] < -1.0 && sy[2] < -1.0)\n"
    "             || (sy[0] >  1.0 && sy[1] >  1.0 && sy[2] >  1.0);\n"
    "  precise float px[3], py[3], pz[3];\n"
    "  [unroll] for (uint k2 = 0u; k2 < 3u; ++k2) {\n"
    "    px[k2] = sx[k2]*hw + hw;\n"
    "    py[k2] = hh - sy[k2]*hh;\n"
    "    pz[k2] = 0.5*(sz[k2] + 1.0);\n"
    "  }\n"
    "  float min_x = min(px[0], min(px[1], px[2]));\n"
    "  float max_x = max(px[0], max(px[1], px[2]));\n"
    "  float min_y = min(py[0], min(py[1], py[2]));\n"
    "  float max_y = max(py[0], max(py[1], py[2]));\n"
    "  int x0 = (int)min_x;      if (x0 < 0) x0 = 0;\n"
    "  int x1 = (int)max_x + 1;  if (x1 > (int)pc.w) x1 = (int)pc.w;\n"
    "  int y0 = (int)min_y;      if (y0 < 0) y0 = 0;\n"
    "  int y1 = (int)max_y + 1;  if (y1 > (int)pc.h) y1 = (int)pc.h;\n"
    "  if (x0 >= x1 || y0 >= y1) reject = true;\n"
    "  float area = edge_fn(px[0], py[0], px[1], py[1], px[2], py[2]);\n"
    "  bool valid = !reject && abs(area) >= 1.0e-6;\n"
    "  uint flags = valid ? (1u | (area > 0.0 ? 2u : 0u)) : 0u;\n"
    "  float inv_area = valid ? 1.0 / area : 0.0;\n"
    "  float3 nrm = cross(v1 - v0, v2 - v0);\n"
    "  float  nl  = sqrt(dot(nrm, nrm));\n"
    "  nrm = (nl > 1.0e-8) ? nrm / nl : float3(0, 1, 0);\n"
    "  uint mid = valid ? InMatIds.Load(t*4u) : 0u;\n"
    "  uint aux = valid ? (((pc.flags & 1u) != 0u) ? InAux.Load(t*4u) : t) : 0u;\n"
    "  if (!valid) { px[0]=px[1]=px[2]=0.0; py[0]=py[1]=py[2]=0.0; pz[0]=pz[1]=pz[2]=0.0; x0=y0=x1=y1=0; }\n"
    "  // SoAoV planes: plane k of record t at (k*npad + t)*16 — each Store4 is\n"
    "  // contiguous across the wave (16 B per lane) instead of 64 B strided.\n"
    "  OutSetup.Store4((0u*pc.npad + t)*16u, asuint(float4(px[0], py[0], px[1], py[1])));\n"
    "  OutSetup.Store4((1u*pc.npad + t)*16u, asuint(float4(px[2], py[2], pz[0], pz[1])));\n"
    "  OutSetup.Store4((2u*pc.npad + t)*16u, uint4(asuint(pz[2]), asuint(inv_area), flags, mid));\n"
    "  OutSetup.Store4((3u*pc.npad + t)*16u, uint4(valid ? oct_pack(nrm) : 0u, aux,\n"
    "                                 (uint)x0 | ((uint)y0 << 16), (uint)x1 | ((uint)y1 << 16)));\n"
    "  if (!valid) continue;\n"
    "  uint tx0 = (uint)x0 / 16u, tx1 = (uint)(x1 - 1) / 16u;\n"
    "  uint ty0 = (uint)y0 / 16u, ty1 = (uint)(y1 - 1) / 16u;\n"
    "  bool pos = area > 0.0;\n"
    "  float extent = (float)(pc.w + pc.h);\n"
    "  for (uint ty = ty0; ty <= ty1; ++ty) {\n"
    "    for (uint tx = tx0; tx <= tx1; ++tx) {\n"
    "      // tile ∩ bbox, pixel-centre corners; conservative per-edge reject\n"
    "      int rx0 = max((int)(tx * 16u), x0), ry0 = max((int)(ty * 16u), y0);\n"
    "      int rx1 = min((int)((tx + 1u) * 16u), x1), ry1 = min((int)((ty + 1u) * 16u), y1);\n"
    "      if (rx0 >= rx1 || ry0 >= ry1) continue;\n"
    "      float cx0 = (float)rx0 + 0.5, cy0 = (float)ry0 + 0.5;\n"
    "      float cx1 = (float)(rx1 - 1) + 0.5, cy1 = (float)(ry1 - 1) + 0.5;\n"
    "      bool overlap = true;\n"
    "      [unroll] for (uint e = 0u; e < 3u; ++e) {\n"
    "        uint a = (e + 1u) % 3u, bb = (e + 2u) % 3u;\n"
    "        float ax = px[a], ay = py[a], bx = px[bb], by = py[bb];\n"
    "        float e00 = edge_fn(ax, ay, bx, by, cx0, cy0), e10 = edge_fn(ax, ay, bx, by, cx1, cy0);\n"
    "        float e01 = edge_fn(ax, ay, bx, by, cx0, cy1), e11 = edge_fn(ax, ay, bx, by, cx1, cy1);\n"
    "        float eps = 1.0e-5 * (abs(bx - ax) + abs(by - ay)) * extent;\n"
    "        float mx = max(max(e00, e10), max(e01, e11));\n"
    "        float mn = min(min(e00, e10), min(e01, e11));\n"
    "        if (pos && mx < -eps) overlap = false;\n"
    "        if (!pos && mn > eps) overlap = false;\n"
    "      }\n"
    "      if (!overlap) continue;\n"
    "      uint tile = ty * pc.tilesX + tx;\n"
    "      uint idx; OutCounts.InterlockedAdd(tile*4u, 1u, idx);\n"
    "      if (idx < pc.cap) OutBins.Store((tile * pc.cap + idx) * 4u, t);\n"
    "    }\n"
    "  }\n"
    "  }   // grid-stride loop\n"
    "}\n";
}

// =============================================================================
// VisibilityBufferPass — binned form
// =============================================================================
namespace {

void visbuf_binned_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<VisibilityBufferPass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height;
    if (W == 0 || H == 0) { ec.dispatch(0, 1, 1); return; }
    const auto* setup  = static_cast<const cardinal::u32*>(ec.map_buffer_read(st->in_setup));
    const auto* counts = static_cast<const cardinal::u32*>(ec.map_buffer_read(st->in_bin_counts));
    const auto* bins   = static_cast<const cardinal::u32*>(ec.map_buffer_read(st->in_bins));
    float*         depth    = static_cast<float*>        (ec.map_buffer_write(st->out_depth));
    cardinal::u32* prim_id  = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_prim_id));
    cardinal::u32* mat_id   = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_mat_id));
    cardinal::u32* normal   = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_normal));
    cardinal::u32* motion   = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_motion));
    cardinal::u32* aux_out  = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_aux));
    if (!depth || !prim_id || !mat_id || !normal || !motion || !aux_out) {
        ec.dispatch(0, 1, 1); return;
    }
    for (cardinal::u32 i = 0; i < W * H; ++i) {
        depth[i] = 1.0f; prim_id[i] = kInvalidPrimId; mat_id[i] = kInvalidMatId;
        normal[i] = 0u; motion[i] = 0u; aux_out[i] = 0u;
    }
    st->fragments_drawn = st->triangles_offscreen = st->overflow_tiles = 0;
    if (!setup || !counts || !bins || st->triangle_count == 0) { ec.dispatch(0, 1, 1); return; }
    const cardinal::u32 cap = st->bin_capacity, T = st->triangle_count;
    const cardinal::usize npad = setup_npad(T);
    // Stats: setup-rejected tris (the brute record's "offscreen" count).
    // flags = dword 10 = plane 2, lane 2.
    for (cardinal::u32 t = 0; t < T; ++t)
        if (!(setup[(2u * npad + t) * 4 + 2] & 1u))
            ++st->triangles_offscreen;
    // Per pixel: test one triangle's setup record. Winner = lowest depth,
    // ties → LOWER triangle id (== the reference's ascending strict-< scan).
    auto test_tri = [&](cardinal::u32 t, int x, int y, cardinal::usize pix) {
        TriSetup s;
        load_setup(setup, npad, t, s);
        if (!(s.flags & 1u)) return;
        if (x < s.x0 || x >= s.x1 || y < s.y0 || y >= s.y1) return;
        const float fx = static_cast<float>(x) + 0.5f;
        const float fy = static_cast<float>(y) + 0.5f;
        const float w0 = edge_fn(s.px[1], s.py[1], s.px[2], s.py[2], fx, fy);
        const float w1 = edge_fn(s.px[2], s.py[2], s.px[0], s.py[0], fx, fy);
        const float w2 = edge_fn(s.px[0], s.py[0], s.px[1], s.py[1], fx, fy);
        const bool inside = (s.flags & 2u)
            ? (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
            : (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
        if (!inside) return;
        const float b0 = w0 * s.inv_area, b1 = w1 * s.inv_area, b2 = w2 * s.inv_area;
        const float z  = b0 * s.pz[0] + b1 * s.pz[1] + b2 * s.pz[2];
        if (!isf(z) || z < 0.0f || z > 1.0f) return;
        const bool better = (z < depth[pix]) ||
                            (z == depth[pix] && prim_id[pix] != kInvalidPrimId && t < prim_id[pix]);
        if (!better) return;
        if (prim_id[pix] == kInvalidPrimId) ++st->fragments_drawn;
        depth[pix] = z; prim_id[pix] = t; mat_id[pix] = s.mid; normal[pix] = s.nrm; aux_out[pix] = s.aux;
    };
    const cardinal::u32 tiles_x = st->tiles_x;
    const cardinal::u32 tiles_y = (H + kVisTileSize - 1) / kVisTileSize;
    for (cardinal::u32 ty = 0; ty < tiles_y; ++ty)
        for (cardinal::u32 tx = 0; tx < tiles_x; ++tx) {
            const cardinal::u32 tile = ty * tiles_x + tx;
            const cardinal::u32 cnt  = counts[tile];
            const bool overflow = cnt > cap;
            if (overflow) ++st->overflow_tiles;
            const int px0 = static_cast<int>(tx * kVisTileSize), py0 = static_cast<int>(ty * kVisTileSize);
            const int px1 = cardinal::min(px0 + static_cast<int>(kVisTileSize), static_cast<int>(W));
            const int py1 = cardinal::min(py0 + static_cast<int>(kVisTileSize), static_cast<int>(H));
            for (int y = py0; y < py1; ++y)
                for (int x = px0; x < px1; ++x) {
                    const cardinal::usize pix = static_cast<cardinal::usize>(y) * W + static_cast<cardinal::usize>(x);
                    if (overflow) {
                        for (cardinal::u32 t = 0; t < T; ++t) test_tri(t, x, y, pix);
                    } else {
                        const cardinal::u32* list = bins + static_cast<cardinal::usize>(tile) * cap;
                        for (cardinal::u32 i = 0; i < cnt; ++i)
                            if (list[i] < T) test_tri(list[i], x, y, pix);
                    }
                }
        }
    ec.dispatch(tiles_x, tiles_y, 1);   // one 16x16 block per bin tile
}

}  // namespace

cardinal::shared_ptr<VisibilityBufferPass::State> VisibilityBufferPass::add_to_graph(
    rg::Graph& g,
    const TriangleBinPass::State& bins,
    rg::ResourceHandle in_aux_per_tri,
    cardinal::u32 width, cardinal::u32 height,
    cardinal::u32 triangle_count)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_tris        = bins.in_tris;
    st->in_mat_ids     = bins.in_mat_ids;
    st->in_matrix      = bins.in_matrix;
    st->in_aux_per_tri = in_aux_per_tri;
    st->in_setup       = bins.out_setup;
    st->in_bin_counts  = bins.out_bin_counts;
    st->in_bins        = bins.out_bins;
    st->width          = width;
    st->height         = height;
    st->triangle_count = triangle_count;
    st->bin_capacity   = bins.bin_capacity;
    st->tiles_x        = bins.tiles_x;
    st->binned         = true;
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
    // t0 setup, t1 counts, t2 bins; u0..u5 the six channels.
    pd.accesses.push_back(rg::ResourceAccess{bins.out_setup,      rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{bins.out_bin_counts, rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{bins.out_bins,       rg::AccessMode::Read,  2});
    pd.accesses.push_back(rg::ResourceAccess{st->out_depth,   rg::AccessMode::Write, 3});
    pd.accesses.push_back(rg::ResourceAccess{st->out_prim_id, rg::AccessMode::Write, 4});
    pd.accesses.push_back(rg::ResourceAccess{st->out_mat_id,  rg::AccessMode::Write, 5});
    pd.accesses.push_back(rg::ResourceAccess{st->out_normal,  rg::AccessMode::Write, 6});
    pd.accesses.push_back(rg::ResourceAccess{st->out_motion,  rg::AccessMode::Write, 7});
    pd.accesses.push_back(rg::ResourceAccess{st->out_aux,     rg::AccessMode::Write, 8});
    pd.record   = visbuf_binned_record;
    pd.user_ctx = st.get();
    // Perfect block dimensions: one 16x16 block per 16x16 bin tile, so the
    // block cooperatively stages exactly its own tile's list in shared memory.
    pd.dispatch_x = bins.tiles_x;
    pd.dispatch_y = bins.tiles_y;
    pd.dispatch_z = 1;
    pd.compute_hlsl = VisibilityBufferPass::hlsl_source_binned();
    pd.push_constant_size = 32;
    auto push_u32 = [&pd](cardinal::u32 x) {
        for (int k = 0; k < 4; ++k)
            pd.push_constants.push_back(static_cast<cardinal::u8>((x >> (8 * k)) & 0xFF));
    };
    push_u32(width); push_u32(height); push_u32(triangle_count); push_u32(bins.tiles_x);
    push_u32(bins.bin_capacity); push_u32(static_cast<cardinal::u32>(setup_npad(triangle_count)));
    push_u32(0u); push_u32(0u);
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* VisibilityBufferPass::hlsl_source_binned() noexcept {
    // One 16x16 thread block per 16x16 bin tile (perfect block dimensions).
    // The block stages its tile's bin list in SHARED MEMORY in chunks of 256
    // setup records: thread `lid` fetches record `base+lid` — four contiguous,
    // 32-byte-aligned SoAoV plane loads, fully coalesced across the wave —
    // then every pixel thread tests all staged records from groupshared
    // instead of re-reading the same 64 bytes from global memory 256 times.
    // Winner rule unchanged (lower depth, then lower id). Barriers are block-
    // uniform: the chunk loop bounds depend only on the tile, and pixels
    // outside the viewport keep participating without storing.
    return
    "// Cardinal - VisibilityBufferPass (binned).hlsl - 16x16 block per tile,\n"
    "// shared-memory staged bin list from SoAoV setup planes.\n"
    "ByteAddressBuffer   InSetup  : register(t0);   // 4 planes x npad x 16 B\n"
    "ByteAddressBuffer   InCounts : register(t1);   // u32 per tile\n"
    "ByteAddressBuffer   InBins   : register(t2);   // cap u32 per tile\n"
    "RWByteAddressBuffer OutDepth : register(u0);\n"
    "RWByteAddressBuffer OutPrim  : register(u1);\n"
    "RWByteAddressBuffer OutMat   : register(u2);\n"
    "RWByteAddressBuffer OutNrm   : register(u3);\n"
    "RWByteAddressBuffer OutMV    : register(u4);\n"
    "RWByteAddressBuffer OutAux   : register(u5);\n"
    "struct PushT { uint w; uint h; uint triCount; uint tilesX; uint cap; uint npad; uint p1; uint p2; };\n"
    "[[vk::push_constant]] ConstantBuffer<PushT> pc : register(b0);\n"
    "#define CHUNK 256u\n"
    "groupshared float4 s_q0[CHUNK];   // px0 py0 px1 py1\n"
    "groupshared float4 s_q1[CHUNK];   // px2 py2 pz0 pz1\n"
    "groupshared uint4  s_q2[CHUNK];   // pz2 inv_area flags mid\n"
    "groupshared uint4  s_q3[CHUNK];   // nrm aux bbox_min bbox_max\n"
    "groupshared uint   s_id[CHUNK];   // triangle id (0xFFFFFFFF = empty)\n"
    "float edge_fn(float ax, float ay, float bx, float by, float cx, float cy) {\n"
    "  return (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);\n"
    "}\n"
    "void test_rec(uint j, int x, int y, float fx, float fy,\n"
    "              inout float bestZ, inout uint bestPrim, inout uint bestMat,\n"
    "              inout uint bestNrm, inout uint bestAux) {\n"
    "  uint t = s_id[j];\n"
    "  if (t == 0xFFFFFFFFu) return;\n"
    "  uint4 q2 = s_q2[j];\n"
    "  uint flags = q2.z;\n"
    "  if ((flags & 1u) == 0u) return;\n"
    "  uint4 q3 = s_q3[j];\n"
    "  int x0 = (int)(q3.z & 0xFFFFu), y0 = (int)(q3.z >> 16);\n"
    "  int x1 = (int)(q3.w & 0xFFFFu), y1 = (int)(q3.w >> 16);\n"
    "  if (x < x0 || x >= x1 || y < y0 || y >= y1) return;\n"
    "  float4 q0 = s_q0[j];\n"
    "  float4 q1 = s_q1[j];\n"
    "  float px0 = q0.x, py0 = q0.y, px1 = q0.z, py1 = q0.w;\n"
    "  float px2 = q1.x, py2 = q1.y, pz0 = q1.z, pz1 = q1.w;\n"
    "  float pz2 = asfloat(q2.x), inv_area = asfloat(q2.y);\n"
    "  float w0 = edge_fn(px1, py1, px2, py2, fx, fy);\n"
    "  float w1 = edge_fn(px2, py2, px0, py0, fx, fy);\n"
    "  float w2 = edge_fn(px0, py0, px1, py1, fx, fy);\n"
    "  bool inside = ((flags & 2u) != 0u) ? (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0)\n"
    "                                     : (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0);\n"
    "  if (!inside) return;\n"
    "  precise float z = (w0*inv_area)*pz0 + (w1*inv_area)*pz1 + (w2*inv_area)*pz2;\n"
    "  if (!isfinite(z) || z < 0.0 || z > 1.0) return;\n"
    "  bool better = (z < bestZ) || (z == bestZ && bestPrim != 0xFFFFFFFFu && t < bestPrim);\n"
    "  if (!better) return;\n"
    "  bestZ = z; bestPrim = t; bestMat = q2.w; bestNrm = q3.x; bestAux = q3.y;\n"
    "}\n"
    "[numthreads(16,16,1)]\n"
    "void CSMain(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {\n"
    "  uint lid  = gtid.y * 16u + gtid.x;\n"
    "  uint tile = gid.y * pc.tilesX + gid.x;\n"
    "  uint px = gid.x * 16u + gtid.x, py = gid.y * 16u + gtid.y;\n"
    "  bool live = (px < pc.w) && (py < pc.h);\n"
    "  int x = (int)px, y = (int)py;\n"
    "  float fx = (float)px + 0.5, fy = (float)py + 0.5;\n"
    "  float bestZ = 1.0;\n"
    "  uint bestPrim = 0xFFFFFFFFu, bestMat = 0xFFFFFFFFu, bestNrm = 0u, bestAux = 0u;\n"
    "  uint cnt = InCounts.Load(tile*4u);\n"
    "  bool overflow = cnt > pc.cap;                 // walk every record instead\n"
    "  uint total = overflow ? pc.triCount : cnt;    // block-uniform\n"
    "  uint listBase = tile * pc.cap;\n"
    "  for (uint base = 0u; base < total; base += CHUNK) {\n"
    "    // Stage: thread lid fetches record base+lid from the four planes.\n"
    "    uint i = base + lid;\n"
    "    uint t = 0xFFFFFFFFu;\n"
    "    if (i < total) {\n"
    "      t = overflow ? i : InBins.Load((listBase + i)*4u);\n"
    "      if (t >= pc.triCount) t = 0xFFFFFFFFu;\n"
    "    }\n"
    "    if (t != 0xFFFFFFFFu) {\n"
    "      s_q0[lid] = asfloat(InSetup.Load4((0u*pc.npad + t)*16u));\n"
    "      s_q1[lid] = asfloat(InSetup.Load4((1u*pc.npad + t)*16u));\n"
    "      s_q2[lid] = InSetup.Load4((2u*pc.npad + t)*16u);\n"
    "      s_q3[lid] = InSetup.Load4((3u*pc.npad + t)*16u);\n"
    "    }\n"
    "    s_id[lid] = t;\n"
    "    GroupMemoryBarrierWithGroupSync();\n"
    "    uint n = min(CHUNK, total - base);\n"
    "    if (live) {\n"
    "      [loop] for (uint j = 0u; j < n; ++j)\n"
    "        test_rec(j, x, y, fx, fy, bestZ, bestPrim, bestMat, bestNrm, bestAux);\n"
    "    }\n"
    "    GroupMemoryBarrierWithGroupSync();   // chunk consumed before it is overwritten\n"
    "  }\n"
    "  if (!live) return;\n"
    "  uint pix = py*pc.w + px;\n"
    "  OutDepth.Store(pix*4u, asuint(bestZ));\n"
    "  OutPrim.Store (pix*4u, bestPrim);\n"
    "  OutMat.Store  (pix*4u, bestMat);\n"
    "  OutNrm.Store  (pix*4u, bestNrm);\n"
    "  OutMV.Store   (pix*4u, 0u);\n"
    "  OutAux.Store  (pix*4u, bestAux);\n"
    "}\n";
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

// One pass per mip. Mip 0 copies the depth (non-finite -> 1.0 = far); mip m
// reduces mip m-1 with a conservative-far max of each 2x2 footprint (edge
// texels clamp). The CPU record and the kernel both do exactly ONE mip per
// invocation, so a 1920x1080 pyramid is 11 dispatches of shrinking size.
void hiz_mip_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* cx = static_cast<HiZBuildPass::MipCtx*>(uctx);
    HiZBuildPass::State* st = cx->st;
    const cardinal::u32 m = cx->mip;
    if (!st || m >= st->mip_count) { ec.dispatch(0, 1, 1); return; }
    const cardinal::u32 cur_w = st->mip_widths[m], cur_h = st->mip_heights[m];
    const cardinal::u32 cur_off = st->mip_offsets_f[m];
    float* pyr = static_cast<float*>(ec.map_buffer_write(st->out_pyramid));
    if (!pyr) { ec.dispatch(0, 1, 1); return; }
    if (m == 0) {
        const float* depth = static_cast<const float*>(ec.map_buffer_read(st->in_depth));
        if (!depth) { ec.dispatch(0, 1, 1); return; }
        const cardinal::usize n = static_cast<cardinal::usize>(cur_w) * cur_h;
        for (cardinal::usize i = 0; i < n; ++i) pyr[cur_off + i] = fzf(depth[i], 1.0f);
        st->mips_built = 1;
    } else {
        const cardinal::u32 prev_off = st->mip_offsets_f[m - 1];
        const cardinal::u32 prev_w   = st->mip_widths[m - 1];
        const cardinal::u32 prev_h   = st->mip_heights[m - 1];
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
        st->mips_built = m + 1;
    }
    if (m + 1 == st->mip_count) st->max_depth = pyr[st->mip_offsets_f[st->mip_count - 1]];
    ec.dispatch((cur_w + 7) / 8, (cur_h + 7) / 8, 1);
}

inline void hiz_push_u32(rg::PassDesc& pd, cardinal::u32 x) {
    for (int k = 0; k < 4; ++k)
        pd.push_constants.push_back(static_cast<cardinal::u8>((x >> (8 * k)) & 0xFFu));
}

}  // namespace

cardinal::shared_ptr<HiZBuildPass::State> HiZBuildPass::declare(
    rg::Graph& g, cardinal::u32 width, cardinal::u32 height, bool persistent)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->width      = width;
    st->height     = height;
    st->persistent = persistent;
    const cardinal::u32 total_floats = compute_mip_layout(width, height, *st);

    rg::BufferDesc od;
    od.name         = "hiz.pyramid";
    od.size_bytes   = static_cast<cardinal::usize>(total_floats == 0 ? 1u : total_floats) * sizeof(float);
    od.stride_bytes = sizeof(float);
    od.persistent   = persistent;
    st->out_pyramid = g.declare_buffer(od);
    return st;
}

void HiZBuildPass::add_passes(rg::Graph& g, State& st, rg::ResourceHandle in_depth)
{
    st.in_depth = in_depth;
    st.mip_ctx.clear();
    st.mip_ctx.resize(st.mip_count);
    for (cardinal::u32 m = 0; m < st.mip_count; ++m) {
        st.mip_ctx[m].st  = &st;
        st.mip_ctx[m].mip = m;
        rg::PassDesc pd;
        pd.name = "HiZBuildPass";
        pd.kind = rg::PassKind::Compute;
        // t0 depth (read by mip 0 only — declared on every mip so the binding
        // layout is fixed); u0 pyramid read+write (mip m reads m-1).
        pd.accesses.push_back(rg::ResourceAccess{in_depth,       rg::AccessMode::Read,      0});
        pd.accesses.push_back(rg::ResourceAccess{st.out_pyramid, rg::AccessMode::ReadWrite, 1});
        pd.record   = hiz_mip_record;
        pd.user_ctx = &st.mip_ctx[m];
        pd.dispatch_x = (st.mip_widths[m]  + 7) / 8;
        pd.dispatch_y = (st.mip_heights[m] + 7) / 8;
        pd.dispatch_z = 1;
        pd.compute_hlsl = HiZBuildPass::hlsl_source();
        pd.push_constant_size = 32;
        hiz_push_u32(pd, m);
        hiz_push_u32(pd, st.mip_widths[m]);
        hiz_push_u32(pd, st.mip_heights[m]);
        hiz_push_u32(pd, m ? st.mip_widths[m - 1]  : st.mip_widths[0]);
        hiz_push_u32(pd, m ? st.mip_heights[m - 1] : st.mip_heights[0]);
        hiz_push_u32(pd, m ? st.mip_offsets_f[m - 1] : 0u);
        hiz_push_u32(pd, st.mip_offsets_f[m]);
        hiz_push_u32(pd, 0u);
        g.add_pass(cardinal::move(pd));
    }
}

cardinal::shared_ptr<HiZBuildPass::State> HiZBuildPass::add_to_graph(
    rg::Graph& g, rg::ResourceHandle in_depth,
    cardinal::u32 width, cardinal::u32 height, bool persistent)
{
    auto st = declare(g, width, height, persistent);
    add_passes(g, *st, in_depth);
    return st;
}

const char* HiZBuildPass::hlsl_source() noexcept {
    return
    "// Cardinal - HiZBuildPass.hlsl - one dispatch per mip. Mip 0 copies depth\n"
    "// (non-finite -> 1.0 = far); mip m stores the max of its 2x2 footprint in\n"
    "// mip m-1 (edge texels clamp). Mirrors hiz_mip_record op-for-op.\n"
    "ByteAddressBuffer   InDepth : register(t0);\n"
    "RWByteAddressBuffer Pyr     : register(u0);\n"
    "struct PushT { uint mip; uint mw; uint mh; uint pw; uint ph; uint poff; uint coff; uint pad; };\n"
    "[[vk::push_constant]] ConstantBuffer<PushT> pc : register(b0);\n"
    "[numthreads(8,8,1)]\n"
    "void CSMain(uint3 tid : SV_DispatchThreadID){\n"
    "  uint x = tid.x, y = tid.y;\n"
    "  if (x >= pc.mw || y >= pc.mh) return;\n"
    "  uint o = pc.coff + y * pc.mw + x;\n"
    "  if (pc.mip == 0u) {\n"
    "    float z = asfloat(InDepth.Load((y * pc.mw + x) * 4u));\n"
    "    if (!isfinite(z)) z = 1.0;\n"
    "    Pyr.Store(o * 4u, asuint(z));\n"
    "    return;\n"
    "  }\n"
    "  uint px0 = x * 2u, py0 = y * 2u;\n"
    "  uint px1 = (px0 + 1u < pc.pw) ? px0 + 1u : px0;\n"
    "  uint py1 = (py0 + 1u < pc.ph) ? py0 + 1u : py0;\n"
    "  float z00 = asfloat(Pyr.Load((pc.poff + py0 * pc.pw + px0) * 4u));\n"
    "  float z01 = asfloat(Pyr.Load((pc.poff + py0 * pc.pw + px1) * 4u));\n"
    "  float z10 = asfloat(Pyr.Load((pc.poff + py1 * pc.pw + px0) * 4u));\n"
    "  float z11 = asfloat(Pyr.Load((pc.poff + py1 * pc.pw + px1) * 4u));\n"
    "  float zmax = z00;\n"
    "  if (z01 > zmax) zmax = z01;\n"
    "  if (z10 > zmax) zmax = z10;\n"
    "  if (z11 > zmax) zmax = z11;\n"
    "  Pyr.Store(o * 4u, asuint(zmax));\n"
    "}\n";
}

// =============================================================================
// HiZOcclusionTestPass
// =============================================================================
namespace {

// Project the 8 corners of a box through a row-major view-proj. Returns false
// when any corner lands behind the camera (w <= 0 or non-finite) — the caller
// keeps that box visible (conservative). Otherwise fills the NDC rect and the
// nearest depth mapped to [0, 1].
inline bool project_box(const float* m,
                        float ax, float ay, float az, float bx, float by, float bz,
                        float& min_sx, float& max_sx, float& min_sy, float& max_sy,
                        float& min_sz) noexcept
{
    min_sx = 1.0e30f; max_sx = -1.0e30f;
    min_sy = 1.0e30f; max_sy = -1.0e30f;
    min_sz = 1.0e30f;
    for (int c = 0; c < 8; ++c) {
        const float X = (c & 1) ? bx : ax;
        const float Y = (c & 2) ? by : ay;
        const float Z = (c & 4) ? bz : az;
        const float cx = m[0]  * X + m[1]  * Y + m[2]  * Z + m[3];
        const float cy = m[4]  * X + m[5]  * Y + m[6]  * Z + m[7];
        const float cz = m[8]  * X + m[9]  * Y + m[10] * Z + m[11];
        const float cw = m[12] * X + m[13] * Y + m[14] * Z + m[15];
        if (cw <= 0.0f || !isf(cw)) return false;
        const float nx = cx / cw;
        const float ny = cy / cw;
        const float nz = 0.5f * (cz / cw + 1.0f);
        if (nx < min_sx) min_sx = nx; if (nx > max_sx) max_sx = nx;
        if (ny < min_sy) min_sy = ny; if (ny > max_sy) max_sy = ny;
        if (nz < min_sz) min_sz = nz;
    }
    return true;
}

void hiz_test_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<HiZOcclusionTestPass::State*>(uctx);
    const cardinal::u32 N = st->aabb_count;
    st->visible = st->occluded = st->frustum_culled = st->prior_culled = 0;
    st->history_used = false;
    cardinal::u8* vis = static_cast<cardinal::u8*>(ec.map_buffer_write(st->out_visibility));
    if (!vis) { ec.dispatch(0, 1, 1); return; }
    const cardinal::u32 words = (N + 3u) / 4u;
    for (cardinal::u32 i = 0; i < words * 4u; ++i) vis[i] = 0;   // pad bytes = 0 (kernel stores words)
    if (N == 0) { ec.dispatch(0, 1, 1); return; }

    const float* bounds = static_cast<const float*>(ec.map_buffer_read(st->in_aabbs));
    const float* mat    = static_cast<const float*>(ec.map_buffer_read(st->in_matrix));
    const float* pmat_in = st->in_prev_matrix.is_valid()
        ? static_cast<const float*>(ec.map_buffer_read(st->in_prev_matrix)) : mat;
    const float* pyr    = static_cast<const float*>(ec.map_buffer_read(st->in_pyramid));
    const cardinal::u8*  prior = st->in_prior_vis.is_valid()
        ? static_cast<const cardinal::u8*>(ec.map_buffer_read(st->in_prior_vis)) : nullptr;
    const cardinal::u32* hist  = st->in_history_valid.is_valid()
        ? static_cast<const cardinal::u32*>(ec.map_buffer_read(st->in_history_valid)) : nullptr;
    if (!bounds || !mat) {
        for (cardinal::u32 i = 0; i < N; ++i) vis[i] = 1;
        st->visible = N;
        ec.dispatch((words + 63) / 64, 1, 1);
        return;
    }
    float m[16], pm[16];
    for (int i = 0; i < 16; ++i) {
        m[i]  = fzf(mat[i], (i % 5 == 0) ? 1.0f : 0.0f);
        pm[i] = fzf(pmat_in ? pmat_in[i] : mat[i], (i % 5 == 0) ? 1.0f : 0.0f);
    }
    const bool use_pyramid = st->occlusion && pyr && st->mip_count > 0 &&
                             (!hist || hist[0] != 0u);
    st->history_used = use_pyramid;

    const cardinal::u32 W = st->width;
    const cardinal::u32 H = st->height;
    const float hw = 0.5f * static_cast<float>(W);
    const float hh = 0.5f * static_cast<float>(H);
    const bool meshlets = (st->layout == HiZOcclusionTestPass::Layout::MeshletRecords);

    cardinal::u32 nv = 0, no = 0;
    for (cardinal::u32 i = 0; i < N; ++i) {
        float ax, ay, az, bx, by, bz;
        if (meshlets) {
            const float* rec = bounds + static_cast<cardinal::usize>(i) * 12u;
            ax = fzf(rec[2], 0.0f); ay = fzf(rec[3], 0.0f); az = fzf(rec[4], 0.0f);
            bx = fzf(rec[5], 0.0f); by = fzf(rec[6], 0.0f); bz = fzf(rec[7], 0.0f);
        } else {
            ax = fzf(bounds[0 * N + i], 0.0f); ay = fzf(bounds[1 * N + i], 0.0f); az = fzf(bounds[2 * N + i], 0.0f);
            bx = fzf(bounds[3 * N + i], 0.0f); by = fzf(bounds[4 * N + i], 0.0f); bz = fzf(bounds[5 * N + i], 0.0f);
        }
        if (prior && prior[i] == 0u) { vis[i] = 0; ++st->prior_culled; continue; }

        float min_sx, max_sx, min_sy, max_sy, min_sz;
        // 1. Frustum reject with the CURRENT view-proj: fully outside the
        //    screen rect -> culled. A corner behind the camera -> visible.
        if (!project_box(m, ax, ay, az, bx, by, bz, min_sx, max_sx, min_sy, max_sy, min_sz)) {
            vis[i] = 1; ++nv; continue;
        }
        if (max_sx < -1.0f || min_sx > 1.0f || max_sy < -1.0f || min_sy > 1.0f) {
            vis[i] = 0; ++st->frustum_culled; continue;
        }
        if (!use_pyramid) { vis[i] = 1; ++nv; continue; }

        // 2. Occlusion against the pyramid, through the view-proj that BUILT it.
        if (!project_box(pm, ax, ay, az, bx, by, bz, min_sx, max_sx, min_sy, max_sy, min_sz)) {
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
        // Convert px rect -> mip coordinates by scaling W -> mw.
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
        // Occluded if the box's nearest depth is BEHIND the pyramid's max.
        const bool occluded = (min_sz > max_pyramid);
        if (occluded) { vis[i] = 0; ++no; }
        else          { vis[i] = 1; ++nv; }
    }
    st->visible  = nv;
    st->occluded = no;
    ec.dispatch((words + 63) / 64, 1, 1);
}

cardinal::shared_ptr<HiZOcclusionTestPass::State> make_hiz_test(
    rg::Graph& g,
    HiZOcclusionTestPass::Layout layout,
    rg::ResourceHandle in_bounds, cardinal::u32 count,
    rg::ResourceHandle in_matrix, rg::ResourceHandle in_prev_matrix,
    rg::ResourceHandle in_prior_vis, rg::ResourceHandle in_history_valid,
    const HiZBuildPass::State& hiz, bool occlusion)
{
    auto st = cardinal::shared_ptr<HiZOcclusionTestPass::State>(new HiZOcclusionTestPass::State());
    st->in_aabbs         = in_bounds;
    st->in_matrix        = in_matrix;
    st->in_prev_matrix   = in_prev_matrix;
    st->in_pyramid       = hiz.out_pyramid;
    st->in_prior_vis     = in_prior_vis;
    st->in_history_valid = in_history_valid;
    st->layout           = layout;
    st->occlusion        = occlusion;
    st->width            = hiz.width;
    st->height           = hiz.height;
    st->mip_count        = hiz.mip_count;
    for (cardinal::u32 i = 0; i < kMaxHiZMips; ++i) {
        st->mip_widths[i]    = (i < hiz.mip_count) ? hiz.mip_widths[i]    : 0u;
        st->mip_heights[i]   = (i < hiz.mip_count) ? hiz.mip_heights[i]   : 0u;
        st->mip_offsets_f[i] = (i < hiz.mip_count) ? hiz.mip_offsets_f[i] : 0u;
        st->layout_words[i]                   = st->mip_widths[i];
        st->layout_words[kMaxHiZMips + i]     = st->mip_heights[i];
        st->layout_words[2 * kMaxHiZMips + i] = st->mip_offsets_f[i];
    }
    st->aabb_count = count;
    // Mip table for the kernel (push constants are capped at 64 B). The State
    // owns the bytes; it outlives the graph (hosts hold the shared_ptr).
    st->in_layout = g.import_buffer("hiz.layout", st->layout_words, sizeof(st->layout_words), 4);

    rg::BufferDesc od;
    od.name         = "hiz.vis";
    // Rounded up to a whole u32: the kernel packs 4 visibility bytes per
    // 32-bit store, and a root-descriptor UAV has no bounds checking.
    od.size_bytes   = (static_cast<cardinal::usize>(count) + 3u) & ~static_cast<cardinal::usize>(3u);
    if (od.size_bytes == 0) od.size_bytes = 4;
    od.stride_bytes = 1;
    st->out_visibility = g.declare_buffer(od);

    rg::ResourceHandle prev_bind  = in_prev_matrix.is_valid() ? in_prev_matrix : in_matrix;
    rg::ResourceHandle prior_bind = in_prior_vis;
    rg::ResourceHandle hist_bind  = in_history_valid;
    if (!prior_bind.is_valid()) prior_bind = g.declare_buffer(rg::BufferDesc{"hiz.noprior", 4, 4, true});
    if (!hist_bind.is_valid())  hist_bind  = g.declare_buffer(rg::BufferDesc{"hiz.nohist",  4, 4, true});

    rg::PassDesc pd;
    pd.name = "HiZOcclusionTestPass";
    pd.kind = rg::PassKind::Compute;
    // t0 bounds, t1 view-proj, t2 pyramid view-proj, t3 pyramid, t4 prior
    // (or dummy), t5 history flag (or dummy), t6 mip table; u0 visibility.
    pd.accesses.push_back(rg::ResourceAccess{in_bounds,          rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_matrix,          rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{prev_bind,          rg::AccessMode::Read,  2});
    pd.accesses.push_back(rg::ResourceAccess{hiz.out_pyramid,    rg::AccessMode::Read,  3});
    pd.accesses.push_back(rg::ResourceAccess{prior_bind,         rg::AccessMode::Read,  4});
    pd.accesses.push_back(rg::ResourceAccess{hist_bind,          rg::AccessMode::Read,  5});
    pd.accesses.push_back(rg::ResourceAccess{st->in_layout,      rg::AccessMode::Read,  6});
    pd.accesses.push_back(rg::ResourceAccess{st->out_visibility, rg::AccessMode::Write, 7});
    pd.record   = hiz_test_record;
    pd.user_ctx = st.get();
    const cardinal::u32 words = (count + 3u) / 4u;
    pd.dispatch_x = (words + 63) / 64;
    if (pd.dispatch_x == 0) pd.dispatch_x = 1;
    pd.dispatch_y = 1; pd.dispatch_z = 1;
    pd.compute_hlsl = HiZOcclusionTestPass::hlsl_source();
    pd.push_constant_size = 32;
    hiz_push_u32(pd, count);
    hiz_push_u32(pd, hiz.width);
    hiz_push_u32(pd, hiz.height);
    hiz_push_u32(pd, hiz.mip_count);
    hiz_push_u32(pd, static_cast<cardinal::u32>(layout));
    hiz_push_u32(pd, (in_prior_vis.is_valid() ? 1u : 0u) |
                     (in_history_valid.is_valid() ? 2u : 0u) |
                     (occlusion ? 4u : 0u));
    hiz_push_u32(pd, 0u);
    hiz_push_u32(pd, 0u);
    g.add_pass(cardinal::move(pd));
    return st;
}

}  // namespace

cardinal::shared_ptr<HiZOcclusionTestPass::State> HiZOcclusionTestPass::add_to_graph(
    rg::Graph& g,
    rg::ResourceHandle in_aabbs,
    rg::ResourceHandle in_matrix,
    cardinal::u32 aabb_count,
    const HiZBuildPass::State& hiz)
{
    return make_hiz_test(g, Layout::AabbSoA, in_aabbs, aabb_count, in_matrix,
                         rg::ResourceHandle{}, rg::ResourceHandle{}, rg::ResourceHandle{},
                         hiz, true);
}

cardinal::shared_ptr<HiZOcclusionTestPass::State> HiZOcclusionTestPass::add_to_graph(
    rg::Graph& g,
    Layout layout,
    rg::ResourceHandle in_bounds,
    cardinal::u32 count,
    rg::ResourceHandle in_matrix,
    rg::ResourceHandle in_prev_matrix,
    rg::ResourceHandle in_prior_vis,
    rg::ResourceHandle in_history_valid,
    const HiZBuildPass::State& hiz,
    bool occlusion)
{
    return make_hiz_test(g, layout, in_bounds, count, in_matrix, in_prev_matrix,
                         in_prior_vis, in_history_valid, hiz, occlusion);
}

const char* HiZOcclusionTestPass::hlsl_source() noexcept {
    return
    "// Cardinal - HiZOcclusionTestPass.hlsl - one thread per PACKED WORD (4 boxes).\n"
    "// Per box: screen-rect frustum reject with the current view-proj, then (when\n"
    "// enabled + history valid) project through the view-proj that built the\n"
    "// pyramid, pick the mip covering the rect, compare nearest depth to its max.\n"
    "// Mirrors hiz_test_record op-for-op (no dot(): explicit products keep the\n"
    "// CPU/GPU rounding identical).\n"
    "ByteAddressBuffer   InBounds  : register(t0);   // SoA AABBs or 12-float meshlet records\n"
    "ByteAddressBuffer   InMatrix  : register(t1);   // 16 floats, current view-proj\n"
    "ByteAddressBuffer   InPrevMat : register(t2);   // 16 floats, the pyramid's view-proj\n"
    "ByteAddressBuffer   InPyramid : register(t3);   // f32 mips\n"
    "ByteAddressBuffer   InPrior   : register(t4);   // u8 per box (or dummy)\n"
    "ByteAddressBuffer   InHist    : register(t5);   // u32 (or dummy)\n"
    "ByteAddressBuffer   InLayout  : register(t6);   // widths[16], heights[16], offsets[16]\n"
    "RWByteAddressBuffer OutVis    : register(u0);   // u8 per box, packed 4 per store\n"
    "struct PushT { uint count; uint width; uint height; uint mipCount; uint layout; uint flags; uint p0; uint p1; };\n"
    "[[vk::push_constant]] ConstantBuffer<PushT> pc : register(b0);\n"
    "float fz(float v, float fb) { return isfinite(v) ? v : fb; }\n"
    "float4 load_row(ByteAddressBuffer b, uint o, float4 id) {\n"
    "  float4 r = asfloat(b.Load4(o));\n"
    "  return float4(fz(r.x, id.x), fz(r.y, id.y), fz(r.z, id.z), fz(r.w, id.w));\n"
    "}\n"
    "bool project_box(float4 r0, float4 r1, float4 r2, float4 r3, float3 a, float3 b,\n"
    "                 out float min_sx, out float max_sx, out float min_sy, out float max_sy, out float min_sz) {\n"
    "  min_sx = 1.0e30; max_sx = -1.0e30; min_sy = 1.0e30; max_sy = -1.0e30; min_sz = 1.0e30;\n"
    "  [unroll] for (uint c = 0u; c < 8u; ++c) {\n"
    "    float X = (c & 1u) ? b.x : a.x;\n"
    "    float Y = (c & 2u) ? b.y : a.y;\n"
    "    float Z = (c & 4u) ? b.z : a.z;\n"
    "    float cx = r0.x * X + r0.y * Y + r0.z * Z + r0.w;\n"
    "    float cy = r1.x * X + r1.y * Y + r1.z * Z + r1.w;\n"
    "    float cz = r2.x * X + r2.y * Y + r2.z * Z + r2.w;\n"
    "    float cw = r3.x * X + r3.y * Y + r3.z * Z + r3.w;\n"
    "    if (cw <= 0.0 || !isfinite(cw)) return false;\n"
    "    float nx = cx / cw, ny = cy / cw, nz = 0.5 * (cz / cw + 1.0);\n"
    "    if (nx < min_sx) min_sx = nx; if (nx > max_sx) max_sx = nx;\n"
    "    if (ny < min_sy) min_sy = ny; if (ny > max_sy) max_sy = ny;\n"
    "    if (nz < min_sz) min_sz = nz;\n"
    "  }\n"
    "  return true;\n"
    "}\n"
    "uint test_box(uint i, float4 m0, float4 m1, float4 m2, float4 m3,\n"
    "              float4 q0, float4 q1, float4 q2, float4 q3, bool usePyr) {\n"
    "  float3 a, b;\n"
    "  if (pc.layout == 1u) {\n"
    "    uint o = i * 48u;\n"
    "    a = asfloat(InBounds.Load3(o + 8u));\n"
    "    b = asfloat(InBounds.Load3(o + 20u));\n"
    "  } else {\n"
    "    a = float3(asfloat(InBounds.Load((0u*pc.count + i)*4u)), asfloat(InBounds.Load((1u*pc.count + i)*4u)), asfloat(InBounds.Load((2u*pc.count + i)*4u)));\n"
    "    b = float3(asfloat(InBounds.Load((3u*pc.count + i)*4u)), asfloat(InBounds.Load((4u*pc.count + i)*4u)), asfloat(InBounds.Load((5u*pc.count + i)*4u)));\n"
    "  }\n"
    "  a = float3(fz(a.x, 0.0), fz(a.y, 0.0), fz(a.z, 0.0));\n"
    "  b = float3(fz(b.x, 0.0), fz(b.y, 0.0), fz(b.z, 0.0));\n"
    "  if ((pc.flags & 1u) != 0u) {\n"
    "    uint pw = InPrior.Load(i & ~3u);\n"
    "    if (((pw >> ((i & 3u) * 8u)) & 0xFFu) == 0u) return 0u;\n"
    "  }\n"
    "  float min_sx, max_sx, min_sy, max_sy, min_sz;\n"
    "  if (!project_box(m0, m1, m2, m3, a, b, min_sx, max_sx, min_sy, max_sy, min_sz)) return 1u;\n"
    "  if (max_sx < -1.0 || min_sx > 1.0 || max_sy < -1.0 || min_sy > 1.0) return 0u;\n"
    "  if (!usePyr) return 1u;\n"
    "  if (!project_box(q0, q1, q2, q3, a, b, min_sx, max_sx, min_sy, max_sy, min_sz)) return 1u;\n"
    "  if (min_sx < -1.0) min_sx = -1.0;\n"
    "  if (max_sx >  1.0) max_sx =  1.0;\n"
    "  if (min_sy < -1.0) min_sy = -1.0;\n"
    "  if (max_sy >  1.0) max_sy =  1.0;\n"
    "  float hw = 0.5 * (float)pc.width, hh = 0.5 * (float)pc.height;\n"
    "  float px_min = min_sx * hw + hw;\n"
    "  float px_max = max_sx * hw + hw;\n"
    "  float py_min = hh - max_sy * hh;\n"
    "  float py_max = hh - min_sy * hh;\n"
    "  float rect_w = px_max - px_min;\n"
    "  float rect_h = py_max - py_min;\n"
    "  float max_dim = (rect_w > rect_h) ? rect_w : rect_h;\n"
    "  if (max_dim <= 0.0) return 1u;\n"
    "  uint mip = 0u;\n"
    "  { float d = max_dim; [loop] while (d > 1.0 && mip + 1u < pc.mipCount) { d *= 0.5; ++mip; } }\n"
    "  uint mw  = InLayout.Load(mip * 4u);\n"
    "  uint mh  = InLayout.Load((16u + mip) * 4u);\n"
    "  uint off = InLayout.Load((32u + mip) * 4u);\n"
    "  float scale_x = (float)mw / (float)pc.width;\n"
    "  float scale_y = (float)mh / (float)pc.height;\n"
    "  int mx0 = (int)(px_min * scale_x);\n"
    "  int mx1 = (int)(px_max * scale_x) + 1;\n"
    "  int my0 = (int)(py_min * scale_y);\n"
    "  int my1 = (int)(py_max * scale_y) + 1;\n"
    "  if (mx0 < 0) mx0 = 0;\n"
    "  if (my0 < 0) my0 = 0;\n"
    "  if (mx1 > (int)mw) mx1 = (int)mw;\n"
    "  if (my1 > (int)mh) my1 = (int)mh;\n"
    "  if (mx0 >= mx1 || my0 >= my1) return 1u;\n"
    "  float max_pyr = 0.0;\n"
    "  [loop] for (int y = my0; y < my1; ++y) {\n"
    "    [loop] for (int x = mx0; x < mx1; ++x) {\n"
    "      float z = asfloat(InPyramid.Load((off + (uint)y * mw + (uint)x) * 4u));\n"
    "      if (z > max_pyr) max_pyr = z;\n"
    "    }\n"
    "  }\n"
    "  return (min_sz > max_pyr) ? 0u : 1u;\n"
    "}\n"
    "[numthreads(64,1,1)]\n"
    "void CSMain(uint3 tid : SV_DispatchThreadID){\n"
    "  uint w = tid.x;\n"
    "  uint words = (pc.count + 3u) / 4u;\n"
    "  if (w >= words) return;\n"
    "  float4 id0 = float4(1,0,0,0), id1 = float4(0,1,0,0), id2 = float4(0,0,1,0), id3 = float4(0,0,0,1);\n"
    "  float4 m0 = load_row(InMatrix,  0u, id0), m1 = load_row(InMatrix,  16u, id1);\n"
    "  float4 m2 = load_row(InMatrix, 32u, id2), m3 = load_row(InMatrix,  48u, id3);\n"
    "  float4 q0 = load_row(InPrevMat, 0u, id0), q1 = load_row(InPrevMat, 16u, id1);\n"
    "  float4 q2 = load_row(InPrevMat, 32u, id2), q3 = load_row(InPrevMat, 48u, id3);\n"
    "  bool usePyr = ((pc.flags & 4u) != 0u) && (pc.mipCount > 0u) &&\n"
    "                (((pc.flags & 2u) == 0u) || InHist.Load(0u) != 0u);\n"
    "  uint word = 0u;\n"
    "  [unroll] for (uint k = 0u; k < 4u; ++k) {\n"
    "    uint i = w * 4u + k;\n"
    "    uint v = (i < pc.count) ? test_box(i, m0, m1, m2, m3, q0, q1, q2, q3, usePyr) : 0u;\n"
    "    word |= (v & 0xFFu) << (k * 8u);\n"
    "  }\n"
    "  OutVis.Store(w * 4u, word);\n"
    "}\n";
}

}  // namespace cardinal::render::gpu
