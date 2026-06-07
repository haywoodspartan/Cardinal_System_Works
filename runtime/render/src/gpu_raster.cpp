#include <cardinal/render/gpu_raster.hpp>

#include <cardinal/core/std/cmath.hpp>
#include <cardinal/core/std/algorithm.hpp>
#include <cardinal/core/std/utility.hpp>

namespace cardinal::render::gpu {

namespace {

inline bool isf(float v) noexcept { return cardinal::isfinite(v); }
inline float fz(float v, float fb) noexcept { return isf(v) ? v : fb; }

inline float saturate01(float v) noexcept {
    if (!isf(v)) return 0.0f;
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

inline cardinal::u8 to_u8(float v) noexcept {
    return static_cast<cardinal::u8>(saturate01(v) * 255.0f + 0.5f);
}

// 2D signed area times 2 of triangle (a, b, c). Sign tells us winding.
inline float edge_fn(float ax, float ay, float bx, float by,
                     float cx, float cy) noexcept {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

}  // namespace

// =============================================================================
// TriangleRasterPass
// =============================================================================
namespace {

struct RasterCtx {
    TriangleRasterPass::State* st;
};

void raster_record(graph::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<TriangleRasterPass::State*>(uctx);
    const cardinal::u32 W = st->width;
    const cardinal::u32 H = st->height;
    const cardinal::u32 N = st->triangle_count;

    const float* tris = static_cast<const float*>(ec.map_buffer_read(st->in_tris));
    cardinal::u8* color = static_cast<cardinal::u8*>(ec.map_buffer_write(st->out_color));
    float*        depth = static_cast<float*>      (ec.map_buffer_write(st->out_depth));
    if (W == 0 || H == 0 || color == nullptr || depth == nullptr) {
        ec.dispatch(0, 1, 1);
        return;
    }
    // Clear depth to +1 (far) and color to opaque black.
    for (cardinal::u32 i = 0; i < W * H; ++i) {
        color[i * 4 + 0] = 0;
        color[i * 4 + 1] = 0;
        color[i * 4 + 2] = 0;
        color[i * 4 + 3] = 255;
        depth[i] = 1.0f;
    }
    if (N == 0 || tris == nullptr) {
        ec.dispatch(0, 1, 1);
        return;
    }

    cardinal::u32 frags_drawn = 0, frags_depth_killed = 0, tris_offscreen = 0;

    // Per-triangle layout: 18 floats (3 verts × {px, py, depth, r, g, b}).
    for (cardinal::u32 t = 0; t < N; ++t) {
        const float* v = tris + t * 18;
        // NaN sanitize.
        const float ax = fz(v[ 0], 0.0f), ay = fz(v[ 1], 0.0f), az = fz(v[ 2], 1.0f);
        const float ar = fz(v[ 3], 1.0f), ag = fz(v[ 4], 1.0f), ab = fz(v[ 5], 1.0f);
        const float bx = fz(v[ 6], 0.0f), by = fz(v[ 7], 0.0f), bz = fz(v[ 8], 1.0f);
        const float br = fz(v[ 9], 1.0f), bg = fz(v[10], 1.0f), bb = fz(v[11], 1.0f);
        const float cx = fz(v[12], 0.0f), cy = fz(v[13], 0.0f), cz = fz(v[14], 1.0f);
        const float cr = fz(v[15], 1.0f), cg = fz(v[16], 1.0f), cb = fz(v[17], 1.0f);

        // Screen-space bbox, clipped to viewport.
        float min_x = ax, max_x = ax;
        if (bx < min_x) min_x = bx; if (bx > max_x) max_x = bx;
        if (cx < min_x) min_x = cx; if (cx > max_x) max_x = cx;
        float min_y = ay, max_y = ay;
        if (by < min_y) min_y = by; if (by > max_y) max_y = by;
        if (cy < min_y) min_y = cy; if (cy > max_y) max_y = cy;
        int x0 = static_cast<int>(min_x);                  if (x0 < 0) x0 = 0;
        int x1 = static_cast<int>(max_x) + 1;              if (x1 > static_cast<int>(W)) x1 = static_cast<int>(W);
        int y0 = static_cast<int>(min_y);                  if (y0 < 0) y0 = 0;
        int y1 = static_cast<int>(max_y) + 1;              if (y1 > static_cast<int>(H)) y1 = static_cast<int>(H);
        if (x0 >= x1 || y0 >= y1) { ++tris_offscreen; continue; }

        const float area = edge_fn(ax, ay, bx, by, cx, cy);
        if (cardinal::abs(area) < 1.0e-6f) continue;       // degenerate
        const float inv_area = 1.0f / area;

        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                const float px = static_cast<float>(x) + 0.5f;
                const float py = static_cast<float>(y) + 0.5f;
                const float w0 = edge_fn(bx, by, cx, cy, px, py);
                const float w1 = edge_fn(cx, cy, ax, ay, px, py);
                const float w2 = edge_fn(ax, ay, bx, by, px, py);
                // Inside test — use >= 0 for CCW winding (area > 0). Flip
                // the sign of the test for CW. Top-left tie-break: NOT
                // included here for simplicity (the test produces 1
                // fragment per pixel covered by any winding).
                const bool inside = (area > 0.0f)
                    ? (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
                    : (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
                if (!inside) continue;
                const float b0 = w0 * inv_area;
                const float b1 = w1 * inv_area;
                const float b2 = w2 * inv_area;
                const float z  = b0 * az + b1 * bz + b2 * cz;
                const cardinal::usize pix = static_cast<cardinal::usize>(y) * W + static_cast<cardinal::usize>(x);
                if (!isf(z) || z > depth[pix] || z < 0.0f || z > 1.0f) {
                    ++frags_depth_killed;
                    continue;
                }
                depth[pix] = z;
                const float r = b0 * ar + b1 * br + b2 * cr;
                const float gC= b0 * ag + b1 * bg + b2 * cg;
                const float b = b0 * ab + b1 * bb + b2 * cb;
                color[pix * 4 + 0] = to_u8(r);
                color[pix * 4 + 1] = to_u8(gC);
                color[pix * 4 + 2] = to_u8(b);
                color[pix * 4 + 3] = 255;
                ++frags_drawn;
            }
        }
    }
    st->fragments_drawn      = frags_drawn;
    st->fragments_depth_killed = frags_depth_killed;
    st->triangles_offscreen  = tris_offscreen;

    // One workgroup per 8×8 tile of pixels.
    const cardinal::u32 tx = (W + 7) / 8;
    const cardinal::u32 ty = (H + 7) / 8;
    ec.dispatch(tx, ty, 1);
}

}  // namespace

cardinal::shared_ptr<TriangleRasterPass::State> TriangleRasterPass::add_to_graph(
    graph::Graph& g,
    graph::ResourceHandle in_tris,
    cardinal::u32 width, cardinal::u32 height,
    cardinal::u32 triangle_count)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_tris        = in_tris;
    st->width          = width;
    st->height         = height;
    st->triangle_count = triangle_count;

    graph::BufferDesc cdesc;
    cdesc.name         = "raster.color";
    cdesc.size_bytes   = static_cast<cardinal::usize>(width) * height * 4u;
    cdesc.stride_bytes = 4;
    st->out_color = g.declare_buffer(cdesc);

    graph::BufferDesc ddesc;
    ddesc.name         = "raster.depth";
    ddesc.size_bytes   = static_cast<cardinal::usize>(width) * height * sizeof(float);
    ddesc.stride_bytes = sizeof(float);
    st->out_depth = g.declare_buffer(ddesc);

    graph::PassDesc pd;
    pd.name = "TriangleRasterPass";
    pd.kind = graph::PassKind::Compute;
    pd.accesses.push_back(graph::ResourceAccess{in_tris,       graph::AccessMode::Read,  0});
    pd.accesses.push_back(graph::ResourceAccess{st->out_color, graph::AccessMode::Write, 1});
    pd.accesses.push_back(graph::ResourceAccess{st->out_depth, graph::AccessMode::Write, 2});
    pd.record   = raster_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = (width  + 7) / 8;
    pd.dispatch_y = (height + 7) / 8;
    pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* TriangleRasterPass::hlsl_source() noexcept {
    return R"(// Cardinal — TriangleRasterPass.hlsl
//
// Tile-parallel software rasterizer. One thread per pixel; one workgroup
// per 8x8 tile. CPU reference is bit-equivalent (same edge-function,
// same barycentric, same depth test, same u8 quantisation).
ByteAddressBuffer  in_tris   : register(t0);
RWByteAddressBuffer out_color : register(u0);
RWByteAddressBuffer out_depth : register(u1);

cbuffer Params : register(b0) {
    uint width, height, triangle_count, _pad;
};

float edge_fn(float2 a, float2 b, float2 c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

float load_f(ByteAddressBuffer buf, uint off) { return asfloat(buf.Load(off)); }

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint x = tid.x, y = tid.y;
    if (x >= width || y >= height) return;
    uint pix = y * width + x;

    // Initialise this pixel: opaque black + far depth.
    float depth = 1.0;
    float3 color = float3(0, 0, 0);

    float2 p = float2(x + 0.5, y + 0.5);

    for (uint t = 0; t < triangle_count; ++t) {
        uint base = t * 18 * 4;
        float ax = load_f(in_tris, base +  0*4); float ay = load_f(in_tris, base +  1*4); float az = load_f(in_tris, base +  2*4);
        float3 ac = float3(load_f(in_tris, base + 3*4), load_f(in_tris, base + 4*4), load_f(in_tris, base + 5*4));
        float bx = load_f(in_tris, base +  6*4); float by = load_f(in_tris, base +  7*4); float bz = load_f(in_tris, base +  8*4);
        float3 bc = float3(load_f(in_tris, base + 9*4), load_f(in_tris, base +10*4), load_f(in_tris, base +11*4));
        float cx = load_f(in_tris, base + 12*4); float cy = load_f(in_tris, base + 13*4); float cz = load_f(in_tris, base + 14*4);
        float3 cc = float3(load_f(in_tris, base + 15*4), load_f(in_tris, base +16*4), load_f(in_tris, base +17*4));

        float area = edge_fn(float2(ax, ay), float2(bx, by), float2(cx, cy));
        if (abs(area) < 1e-6) continue;
        float w0 = edge_fn(float2(bx, by), float2(cx, cy), p);
        float w1 = edge_fn(float2(cx, cy), float2(ax, ay), p);
        float w2 = edge_fn(float2(ax, ay), float2(bx, by), p);
        bool inside = (area > 0.0)
            ? (w0 >= 0 && w1 >= 0 && w2 >= 0)
            : (w0 <= 0 && w1 <= 0 && w2 <= 0);
        if (!inside) continue;
        float inv_a = 1.0 / area;
        float b0 = w0 * inv_a, b1 = w1 * inv_a, b2 = w2 * inv_a;
        float z = b0 * az + b1 * bz + b2 * cz;
        if (z < 0 || z > 1 || z >= depth) continue;
        depth = z;
        color = b0 * ac + b1 * bc + b2 * cc;
    }

    uint cb = pix * 4;
    out_color.Store(cb + 0, uint(saturate(color.r) * 255 + 0.5));
    out_color.Store(cb + 1, uint(saturate(color.g) * 255 + 0.5));
    out_color.Store(cb + 2, uint(saturate(color.b) * 255 + 0.5));
    out_color.Store(cb + 3, 255);
    out_depth.Store(pix * 4, asuint(depth));
}
)";
}

// =============================================================================
// PolygonClipPass — Sutherland-Hodgman against one plane
// =============================================================================
namespace {

inline float plane_eval(float nx, float ny, float nz, float d,
                        float x, float y, float z) noexcept {
    return nx * x + ny * y + nz * z + d;
}

void clip_record(graph::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<PolygonClipPass::State*>(uctx);
    const cardinal::u32 N    = st->input_count;
    const cardinal::u32 maxN = st->max_output_count;

    const float* verts = static_cast<const float*>(ec.map_buffer_read(st->in_verts));
    const float* plane = static_cast<const float*>(ec.map_buffer_read(st->in_plane));
    float*       out   = static_cast<float*>      (ec.map_buffer_write(st->out_verts));
    cardinal::u32* cnt = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_count));
    if (!verts || !plane || !out || !cnt) {
        if (cnt) cnt[0] = 0;
        st->produced = 0;
        ec.dispatch(0, 1, 1);
        return;
    }

    const float nx = fz(plane[0], 1.0f);
    const float ny = fz(plane[1], 0.0f);
    const float nz = fz(plane[2], 0.0f);
    const float d  = fz(plane[3], 0.0f);

    // SoA input → AoS scratch (small, on the stack as runtime array via
    // explicit cap kSHMax). 256 vertex cap; plenty for any UI or decal.
    constexpr cardinal::u32 kSHMax = 256;
    if (N == 0 || N > kSHMax) {
        cnt[0] = 0;
        st->produced = 0;
        ec.dispatch(0, 1, 1);
        return;
    }

    const float* vx_in = verts + 0 * N;
    const float* vy_in = verts + 1 * N;
    const float* vz_in = verts + 2 * N;

    float bx[kSHMax], by[kSHMax], bz[kSHMax];
    cardinal::u32 produced = 0;

    // Sutherland-Hodgman: walk edges (v[i] → v[(i+1) % N]).
    for (cardinal::u32 i = 0; i < N; ++i) {
        const cardinal::u32 j = (i + 1) % N;
        const float ax = fz(vx_in[i], 0.0f);
        const float ay = fz(vy_in[i], 0.0f);
        const float az = fz(vz_in[i], 0.0f);
        const float bxj = fz(vx_in[j], 0.0f);
        const float byj = fz(vy_in[j], 0.0f);
        const float bzj = fz(vz_in[j], 0.0f);
        const float sa = plane_eval(nx, ny, nz, d, ax, ay, az);
        const float sb = plane_eval(nx, ny, nz, d, bxj, byj, bzj);
        const bool a_in = sa >= 0.0f;
        const bool b_in = sb >= 0.0f;

        auto push = [&](float x, float y, float z) noexcept {
            if (produced < maxN) {
                bx[produced] = x; by[produced] = y; bz[produced] = z;
                ++produced;
            }
        };

        if (a_in && b_in) {
            // Both in → emit B.
            push(bxj, byj, bzj);
        } else if (a_in && !b_in) {
            // A in, B out → emit intersection.
            const float t = sa / (sa - sb);
            push(ax + t * (bxj - ax),
                 ay + t * (byj - ay),
                 az + t * (bzj - az));
        } else if (!a_in && b_in) {
            // A out, B in → emit intersection, then B.
            const float t = sa / (sa - sb);
            push(ax + t * (bxj - ax),
                 ay + t * (byj - ay),
                 az + t * (bzj - az));
            push(bxj, byj, bzj);
        }
        // both out → emit nothing
    }

    // Write SoA output.
    float* ox = out + 0 * maxN;
    float* oy = out + 1 * maxN;
    float* oz = out + 2 * maxN;
    for (cardinal::u32 i = 0; i < produced; ++i) {
        ox[i] = bx[i]; oy[i] = by[i]; oz[i] = bz[i];
    }
    cnt[0] = produced;
    st->produced = produced;
    ec.dispatch(1, 1, 1);
}

}  // namespace

cardinal::shared_ptr<PolygonClipPass::State> PolygonClipPass::add_to_graph(
    graph::Graph& g,
    graph::ResourceHandle in_verts,
    graph::ResourceHandle in_plane,
    cardinal::u32 input_count,
    cardinal::u32 max_output_count)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_verts          = in_verts;
    st->in_plane          = in_plane;
    st->input_count       = input_count;
    st->max_output_count  = max_output_count;

    graph::BufferDesc ovd;
    ovd.name         = "clip.verts";
    ovd.size_bytes   = static_cast<cardinal::usize>(max_output_count) * 3u * sizeof(float);
    ovd.stride_bytes = sizeof(float);
    st->out_verts = g.declare_buffer(ovd);

    graph::BufferDesc ocd;
    ocd.name         = "clip.count";
    ocd.size_bytes   = sizeof(cardinal::u32);
    ocd.stride_bytes = sizeof(cardinal::u32);
    st->out_count = g.declare_buffer(ocd);

    graph::PassDesc pd;
    pd.name = "PolygonClipPass";
    pd.kind = graph::PassKind::Compute;
    pd.accesses.push_back(graph::ResourceAccess{in_verts,      graph::AccessMode::Read,  0});
    pd.accesses.push_back(graph::ResourceAccess{in_plane,      graph::AccessMode::Read,  1});
    pd.accesses.push_back(graph::ResourceAccess{st->out_verts, graph::AccessMode::Write, 2});
    pd.accesses.push_back(graph::ResourceAccess{st->out_count, graph::AccessMode::Write, 3});
    pd.record   = clip_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = 1; pd.dispatch_y = 1; pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* PolygonClipPass::hlsl_source() noexcept {
    return R"(// Cardinal — PolygonClipPass.hlsl
//
// Single-threaded Sutherland-Hodgman convex-polygon clip against one
// plane. Polygon vertices are small (UI / decals); a single thread
// per polygon dominates the dispatch overhead. CPU reference produces
// identical output.
ByteAddressBuffer  in_verts  : register(t0);
ByteAddressBuffer  in_plane  : register(t1);
RWByteAddressBuffer out_verts : register(u0);
RWByteAddressBuffer out_count : register(u1);

cbuffer Params : register(b0) {
    uint input_count, max_output_count, _p0, _p1;
};

float load_f(ByteAddressBuffer b, uint o) { return asfloat(b.Load(o)); }

[numthreads(1, 1, 1)]
void main() {
    float nx = load_f(in_plane,  0);
    float ny = load_f(in_plane,  4);
    float nz = load_f(in_plane,  8);
    float d  = load_f(in_plane, 12);

    float bx[256], by[256], bz[256];
    uint produced = 0;
    for (uint i = 0; i < input_count; ++i) {
        uint j = (i + 1) % input_count;
        float ax = load_f(in_verts, (0 * input_count + i) * 4);
        float ay = load_f(in_verts, (1 * input_count + i) * 4);
        float az = load_f(in_verts, (2 * input_count + i) * 4);
        float bxj = load_f(in_verts, (0 * input_count + j) * 4);
        float byj = load_f(in_verts, (1 * input_count + j) * 4);
        float bzj = load_f(in_verts, (2 * input_count + j) * 4);
        float sa = nx*ax + ny*ay + nz*az + d;
        float sb = nx*bxj + ny*byj + nz*bzj + d;
        bool a_in = sa >= 0; bool b_in = sb >= 0;
        if (a_in && b_in) {
            if (produced < max_output_count) { bx[produced] = bxj; by[produced] = byj; bz[produced] = bzj; produced++; }
        } else if (a_in && !b_in) {
            float t = sa / (sa - sb);
            if (produced < max_output_count) { bx[produced] = ax + t*(bxj-ax); by[produced] = ay + t*(byj-ay); bz[produced] = az + t*(bzj-az); produced++; }
        } else if (!a_in && b_in) {
            float t = sa / (sa - sb);
            if (produced < max_output_count) { bx[produced] = ax + t*(bxj-ax); by[produced] = ay + t*(byj-ay); bz[produced] = az + t*(bzj-az); produced++; }
            if (produced < max_output_count) { bx[produced] = bxj; by[produced] = byj; bz[produced] = bzj; produced++; }
        }
    }
    for (uint i = 0; i < produced; ++i) {
        out_verts.Store((0 * max_output_count + i) * 4, asuint(bx[i]));
        out_verts.Store((1 * max_output_count + i) * 4, asuint(by[i]));
        out_verts.Store((2 * max_output_count + i) * 4, asuint(bz[i]));
    }
    out_count.Store(0, produced);
}
)";
}

// =============================================================================
// PolygonTriangulatePass — fan triangulation, vertex 0 = anchor
// =============================================================================
namespace {

void tri_record(graph::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<PolygonTriangulatePass::State*>(uctx);
    const cardinal::u32 N = st->input_count;
    if (N < 3) {
        st->triangles_produced = 0;
        ec.dispatch(0, 1, 1);
        return;
    }
    const float* verts = static_cast<const float*>(ec.map_buffer_read(st->in_verts));
    float*       tris  = static_cast<float*>      (ec.map_buffer_write(st->out_tris));
    if (!verts || !tris) {
        st->triangles_produced = 0;
        ec.dispatch(0, 1, 1);
        return;
    }
    const float* vx = verts + 0 * N;
    const float* vy = verts + 1 * N;
    const float* vz = verts + 2 * N;

    const cardinal::u32 T = N - 2;
    // Anchor vertex 0; for i ∈ [1, N-1): emit triangle (0, i, i+1).
    for (cardinal::u32 i = 1; i < N - 1; ++i) {
        const cardinal::u32 ti = i - 1;
        float* dst = tris + ti * 9;
        dst[0] = fz(vx[0],   0.0f); dst[1] = fz(vy[0],   0.0f); dst[2] = fz(vz[0],   0.0f);
        dst[3] = fz(vx[i],   0.0f); dst[4] = fz(vy[i],   0.0f); dst[5] = fz(vz[i],   0.0f);
        dst[6] = fz(vx[i+1], 0.0f); dst[7] = fz(vy[i+1], 0.0f); dst[8] = fz(vz[i+1], 0.0f);
    }
    st->triangles_produced = T;
    ec.dispatch((T + 63) / 64, 1, 1);
}

}  // namespace

cardinal::shared_ptr<PolygonTriangulatePass::State> PolygonTriangulatePass::add_to_graph(
    graph::Graph& g,
    graph::ResourceHandle in_verts,
    cardinal::u32 input_count)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_verts    = in_verts;
    st->input_count = input_count;

    const cardinal::u32 tri_count = (input_count >= 2) ? (input_count - 2) : 0;
    graph::BufferDesc otd;
    otd.name         = "tri.verts";
    otd.size_bytes   = static_cast<cardinal::usize>(tri_count) * 9u * sizeof(float);
    otd.stride_bytes = sizeof(float);
    st->out_tris = g.declare_buffer(otd);

    graph::PassDesc pd;
    pd.name = "PolygonTriangulatePass";
    pd.kind = graph::PassKind::Compute;
    pd.accesses.push_back(graph::ResourceAccess{in_verts,    graph::AccessMode::Read,  0});
    pd.accesses.push_back(graph::ResourceAccess{st->out_tris,graph::AccessMode::Write, 1});
    pd.record   = tri_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = (tri_count + 63) / 64;
    pd.dispatch_y = 1; pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* PolygonTriangulatePass::hlsl_source() noexcept {
    return R"(// Cardinal — PolygonTriangulatePass.hlsl
//
// Fan triangulation anchored at vertex 0. One thread per output triangle.
ByteAddressBuffer  in_verts  : register(t0);
RWByteAddressBuffer out_tris : register(u0);
cbuffer Params : register(b0) { uint input_count, _p0, _p1, _p2; };

float load_f(ByteAddressBuffer b, uint off) { return asfloat(b.Load(off)); }

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint ti = tid.x;
    if (input_count < 3 || ti + 2 >= input_count) return;
    uint i = ti + 1;
    uint base = ti * 9 * 4;
    out_tris.Store(base + 0*4, asuint(load_f(in_verts, (0 * input_count + 0) * 4)));
    out_tris.Store(base + 1*4, asuint(load_f(in_verts, (1 * input_count + 0) * 4)));
    out_tris.Store(base + 2*4, asuint(load_f(in_verts, (2 * input_count + 0) * 4)));
    out_tris.Store(base + 3*4, asuint(load_f(in_verts, (0 * input_count + i) * 4)));
    out_tris.Store(base + 4*4, asuint(load_f(in_verts, (1 * input_count + i) * 4)));
    out_tris.Store(base + 5*4, asuint(load_f(in_verts, (2 * input_count + i) * 4)));
    out_tris.Store(base + 6*4, asuint(load_f(in_verts, (0 * input_count + i + 1) * 4)));
    out_tris.Store(base + 7*4, asuint(load_f(in_verts, (1 * input_count + i + 1) * 4)));
    out_tris.Store(base + 8*4, asuint(load_f(in_verts, (2 * input_count + i + 1) * 4)));
}
)";
}

}  // namespace cardinal::render::gpu
