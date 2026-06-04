#include <cardinal/render/gpu_primitives.hpp>

#include <cardinal/core/cmath.hpp>
#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/utility.hpp>

namespace cardinal::render::gpu {

namespace rg = cardinal::render::graph;

namespace {

constexpr float kTwoPi = 6.283185307179586f;
constexpr float kPi    = 3.14159265358979323846f;

inline bool isf(float v) noexcept { return cardinal::isfinite(v); }
inline float fzf(float v, float fb) noexcept { return isf(v) ? v : fb; }

inline cardinal::u32 clamp_u32(cardinal::u32 v, cardinal::u32 lo, cardinal::u32 hi) noexcept {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

inline cardinal::u8 to_u8(float v) noexcept {
    if (!isf(v)) return 0;
    const float c = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    return static_cast<cardinal::u8>(c * 255.0f + 0.5f);
}

}  // namespace

// =============================================================================
// SpherePrimitivePass
// =============================================================================
namespace {

struct SphereCtx {
    SpherePrimitivePass::State* st;
};

void sphere_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<SpherePrimitivePass::State*>(uctx);
    const cardinal::u32 lat = clamp_u32(st->spec.latitude_segments,  2, 256);
    const cardinal::u32 lon = clamp_u32(st->spec.longitude_segments, 3, 512);
    const float r  = fzf(st->spec.radius, 1.0f);
    const float cx = fzf(st->spec.center_x, 0.0f);
    const float cy = fzf(st->spec.center_y, 0.0f);
    const float cz = fzf(st->spec.center_z, 0.0f);

    float* out = static_cast<float*>(ec.map_buffer_write(st->out_tris));
    if (!out) {
        ec.dispatch(0, 1, 1);
        return;
    }

    cardinal::usize off = 0;
    // For each ring (i: 0..lat-1) × wedge (j: 0..lon-1), emit two triangles
    // unless we're at the top/bottom pole (where it's one triangle each).
    for (cardinal::u32 i = 0; i < lat; ++i) {
        const float theta0 = static_cast<float>(i)     / static_cast<float>(lat) * kPi;
        const float theta1 = static_cast<float>(i + 1) / static_cast<float>(lat) * kPi;
        const float sin_t0 = cardinal::sin(theta0), cos_t0 = cardinal::cos(theta0);
        const float sin_t1 = cardinal::sin(theta1), cos_t1 = cardinal::cos(theta1);
        for (cardinal::u32 j = 0; j < lon; ++j) {
            const float phi0 = static_cast<float>(j)     / static_cast<float>(lon) * kTwoPi;
            const float phi1 = static_cast<float>(j + 1) / static_cast<float>(lon) * kTwoPi;
            const float cos_p0 = cardinal::cos(phi0), sin_p0 = cardinal::sin(phi0);
            const float cos_p1 = cardinal::cos(phi1), sin_p1 = cardinal::sin(phi1);

            const float ax = cx + r * sin_t0 * cos_p0;
            const float ay = cy + r * cos_t0;
            const float az = cz + r * sin_t0 * sin_p0;

            const float bx = cx + r * sin_t1 * cos_p0;
            const float by = cy + r * cos_t1;
            const float bz = cz + r * sin_t1 * sin_p0;

            const float c_x= cx + r * sin_t1 * cos_p1;
            const float c_y= cy + r * cos_t1;
            const float c_z= cz + r * sin_t1 * sin_p1;

            const float dx = cx + r * sin_t0 * cos_p1;
            const float dy = cy + r * cos_t0;
            const float dz = cz + r * sin_t0 * sin_p1;

            // Triangle 1: (a, b, c)
            out[off + 0] = ax; out[off + 1] = ay; out[off + 2] = az;
            out[off + 3] = bx; out[off + 4] = by; out[off + 5] = bz;
            out[off + 6] = c_x;out[off + 7] = c_y;out[off + 8] = c_z;
            off += 9;
            // Triangle 2: (a, c, d)
            out[off + 0] = ax; out[off + 1] = ay; out[off + 2] = az;
            out[off + 3] = c_x;out[off + 4] = c_y;out[off + 5] = c_z;
            out[off + 6] = dx; out[off + 7] = dy; out[off + 8] = dz;
            off += 9;
        }
    }
    const cardinal::u32 T = 2 * lat * lon;
    st->triangle_count = T;
    ec.dispatch((T + 63) / 64, 1, 1);
}

}  // namespace

cardinal::shared_ptr<SpherePrimitivePass::State> SpherePrimitivePass::add_to_graph(
    rg::Graph& g, SphereSpec spec)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->spec = spec;
    const cardinal::u32 lat = clamp_u32(spec.latitude_segments,  2, 256);
    const cardinal::u32 lon = clamp_u32(spec.longitude_segments, 3, 512);
    const cardinal::u32 T = 2 * lat * lon;
    st->triangle_count = T;

    rg::BufferDesc od;
    od.name         = "sphere.tris";
    od.size_bytes   = static_cast<cardinal::usize>(T) * 9u * sizeof(float);
    od.stride_bytes = sizeof(float);
    st->out_tris = g.declare_buffer(od);

    rg::PassDesc pd;
    pd.name = "SpherePrimitivePass";
    pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{st->out_tris, rg::AccessMode::Write, 0});
    pd.record   = sphere_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = (T + 63) / 64;
    pd.dispatch_y = 1; pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* SpherePrimitivePass::hlsl_source() noexcept {
    return R"(// Cardinal — SpherePrimitivePass.hlsl
//
// One thread per (lat, lon) cell — produces two output triangles.
// Indexing matches the CPU reference: triangle 0 = (a, b, c), 1 = (a, c, d).
RWByteAddressBuffer out_tris : register(u0);
cbuffer Params : register(b0) {
    float radius, center_x, center_y, center_z;
    uint  lat_segs, lon_segs, _p0, _p1;
};
static const float kPi    = 3.14159265358979;
static const float kTwoPi = 6.28318530717959;
void store_f(RWByteAddressBuffer b, uint o, float v) { b.Store(o, asuint(v)); }
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x, j = tid.y;
    if (i >= lat_segs || j >= lon_segs) return;
    float t0 = float(i)     / float(lat_segs) * kPi;
    float t1 = float(i + 1) / float(lat_segs) * kPi;
    float p0 = float(j)     / float(lon_segs) * kTwoPi;
    float p1 = float(j + 1) / float(lon_segs) * kTwoPi;
    float3 c = float3(center_x, center_y, center_z);
    float3 a = c + radius * float3(sin(t0) * cos(p0), cos(t0), sin(t0) * sin(p0));
    float3 b = c + radius * float3(sin(t1) * cos(p0), cos(t1), sin(t1) * sin(p0));
    float3 q = c + radius * float3(sin(t1) * cos(p1), cos(t1), sin(t1) * sin(p1));
    float3 d = c + radius * float3(sin(t0) * cos(p1), cos(t0), sin(t0) * sin(p1));
    uint cell = i * lon_segs + j;
    uint base = cell * 2 * 9 * 4;
    store_f(out_tris, base + 0*4, a.x); store_f(out_tris, base + 1*4, a.y); store_f(out_tris, base + 2*4, a.z);
    store_f(out_tris, base + 3*4, b.x); store_f(out_tris, base + 4*4, b.y); store_f(out_tris, base + 5*4, b.z);
    store_f(out_tris, base + 6*4, q.x); store_f(out_tris, base + 7*4, q.y); store_f(out_tris, base + 8*4, q.z);
    store_f(out_tris, base + 9*4 + 0*4, a.x); store_f(out_tris, base + 9*4 + 1*4, a.y); store_f(out_tris, base + 9*4 + 2*4, a.z);
    store_f(out_tris, base + 9*4 + 3*4, q.x); store_f(out_tris, base + 9*4 + 4*4, q.y); store_f(out_tris, base + 9*4 + 5*4, q.z);
    store_f(out_tris, base + 9*4 + 6*4, d.x); store_f(out_tris, base + 9*4 + 7*4, d.y); store_f(out_tris, base + 9*4 + 8*4, d.z);
}
)";
}

// =============================================================================
// CubePrimitivePass — 12 triangles, axis-aligned
// =============================================================================
namespace {

void cube_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<CubePrimitivePass::State*>(uctx);
    const float h  = fzf(st->spec.half_extent, 1.0f);
    const float cx = fzf(st->spec.center_x, 0.0f);
    const float cy = fzf(st->spec.center_y, 0.0f);
    const float cz = fzf(st->spec.center_z, 0.0f);

    float* out = static_cast<float*>(ec.map_buffer_write(st->out_tris));
    if (!out) {
        ec.dispatch(0, 1, 1);
        return;
    }

    // 8 corner vertices.
    const float corners[8][3] = {
        { cx - h, cy - h, cz - h }, { cx + h, cy - h, cz - h },
        { cx + h, cy + h, cz - h }, { cx - h, cy + h, cz - h },
        { cx - h, cy - h, cz + h }, { cx + h, cy - h, cz + h },
        { cx + h, cy + h, cz + h }, { cx - h, cy + h, cz + h },
    };
    // 12 triangles — 2 per face, CCW outward-facing.
    static const cardinal::u32 idx[12][3] = {
        // -Z face
        {0, 2, 1}, {0, 3, 2},
        // +Z face
        {4, 5, 6}, {4, 6, 7},
        // -X face
        {0, 7, 3}, {0, 4, 7},
        // +X face
        {1, 2, 6}, {1, 6, 5},
        // -Y face
        {0, 1, 5}, {0, 5, 4},
        // +Y face
        {3, 7, 6}, {3, 6, 2},
    };
    for (cardinal::u32 t = 0; t < 12; ++t) {
        for (cardinal::u32 v = 0; v < 3; ++v) {
            out[t * 9 + v * 3 + 0] = corners[idx[t][v]][0];
            out[t * 9 + v * 3 + 1] = corners[idx[t][v]][1];
            out[t * 9 + v * 3 + 2] = corners[idx[t][v]][2];
        }
    }
    st->triangle_count = 12;
    ec.dispatch(1, 1, 1);
}

}  // namespace

cardinal::shared_ptr<CubePrimitivePass::State> CubePrimitivePass::add_to_graph(
    rg::Graph& g, CubeSpec spec)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->spec = spec;

    rg::BufferDesc od;
    od.name         = "cube.tris";
    od.size_bytes   = 12u * 9u * sizeof(float);
    od.stride_bytes = sizeof(float);
    st->out_tris = g.declare_buffer(od);

    rg::PassDesc pd;
    pd.name = "CubePrimitivePass";
    pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{st->out_tris, rg::AccessMode::Write, 0});
    pd.record   = cube_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = 1; pd.dispatch_y = 1; pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* CubePrimitivePass::hlsl_source() noexcept {
    return R"(// Cardinal — CubePrimitivePass.hlsl
//
// 12-triangle axis-aligned cube. One thread emits all 12 (cheap).
RWByteAddressBuffer out_tris : register(u0);
cbuffer Params : register(b0) { float half_extent, cx, cy, cz; };
void store_f(RWByteAddressBuffer b, uint o, float v) { b.Store(o, asuint(v)); }
[numthreads(1, 1, 1)]
void main() {
    float h = half_extent;
    float3 c0 = float3(cx - h, cy - h, cz - h), c1 = float3(cx + h, cy - h, cz - h);
    float3 c2 = float3(cx + h, cy + h, cz - h), c3 = float3(cx - h, cy + h, cz - h);
    float3 c4 = float3(cx - h, cy - h, cz + h), c5 = float3(cx + h, cy - h, cz + h);
    float3 c6 = float3(cx + h, cy + h, cz + h), c7 = float3(cx - h, cy + h, cz + h);
    float3 ts[12][3] = {
        {c0,c2,c1},{c0,c3,c2}, {c4,c5,c6},{c4,c6,c7},
        {c0,c7,c3},{c0,c4,c7}, {c1,c2,c6},{c1,c6,c5},
        {c0,c1,c5},{c0,c5,c4}, {c3,c7,c6},{c3,c6,c2},
    };
    for (uint t = 0; t < 12; ++t) for (uint v = 0; v < 3; ++v) {
        uint o = (t * 9 + v * 3) * 4;
        store_f(out_tris, o + 0, ts[t][v].x);
        store_f(out_tris, o + 4, ts[t][v].y);
        store_f(out_tris, o + 8, ts[t][v].z);
    }
}
)";
}

// =============================================================================
// WireframePass — Bresenham edge rasterizer
// =============================================================================
namespace {

inline void project(const float* m, float x, float y, float z,
                    float& sx, float& sy, float& sz, float& w_out) noexcept {
    // Row-major 4×4. World → clip.
    const float cx = m[0]  * x + m[1]  * y + m[2]  * z + m[3];
    const float cy = m[4]  * x + m[5]  * y + m[6]  * z + m[7];
    const float cz = m[8]  * x + m[9]  * y + m[10] * z + m[11];
    const float cw = m[12] * x + m[13] * y + m[14] * z + m[15];
    w_out = cw;
    if (cardinal::abs(cw) < 1.0e-8f || !isf(cw)) {
        sx = 0; sy = 0; sz = 1.0e30f;
        return;
    }
    sx = cx / cw;
    sy = cy / cw;
    sz = cz / cw;
}

// Bresenham line. Writes pixels along (x0, y0) → (x1, y1) into the
// framebuffer; depth-test based on linear interpolation of z along the
// segment.
void bres_line(int x0, int y0, float z0,
               int x1, int y1, float z1,
               cardinal::u8* color, float* depth,
               cardinal::u32 W, cardinal::u32 H,
               cardinal::u8 lr, cardinal::u8 lg, cardinal::u8 lb,
               cardinal::u32& pixels_drawn) noexcept
{
    int dx = (x1 > x0) ? (x1 - x0) : (x0 - x1);
    int dy = -((y1 > y0) ? (y1 - y0) : (y0 - y1));
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;
    const float total_len = static_cast<float>((dx > -dy) ? dx : -dy);
    int x = x0, y = y0;
    int step = 0;
    // Hard cap on iterations to avoid runaway loops on pathological inputs.
    const int max_iter = (dx + (-dy)) + 1;
    while (step < max_iter) {
        if (x >= 0 && y >= 0 && static_cast<cardinal::u32>(x) < W && static_cast<cardinal::u32>(y) < H) {
            const float t = (total_len > 0.0f) ? static_cast<float>(step) / total_len : 0.0f;
            const float z = z0 + t * (z1 - z0);
            const cardinal::usize pix = static_cast<cardinal::usize>(y) * W + static_cast<cardinal::usize>(x);
            if (isf(z) && z >= 0.0f && z <= 1.0f && z <= depth[pix]) {
                depth[pix] = z;
                color[pix * 4 + 0] = lr;
                color[pix * 4 + 1] = lg;
                color[pix * 4 + 2] = lb;
                color[pix * 4 + 3] = 255;
                ++pixels_drawn;
            }
        }
        if (x == x1 && y == y1) break;
        const int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x += sx; }
        if (e2 <= dx) { err += dx; y += sy; }
        ++step;
    }
}

void wire_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<WireframePass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height;
    if (W == 0 || H == 0) { ec.dispatch(0, 1, 1); return; }

    const float* tris = static_cast<const float*>(ec.map_buffer_read(st->in_tris));
    const float* mat  = static_cast<const float*>(ec.map_buffer_read(st->in_matrix));
    cardinal::u8* color = static_cast<cardinal::u8*>(ec.map_buffer_write(st->out_color));
    float*        depth = static_cast<float*>      (ec.map_buffer_write(st->out_depth));
    if (!color || !depth) { ec.dispatch(0, 1, 1); return; }

    // Clear.
    for (cardinal::u32 i = 0; i < W * H; ++i) {
        color[i * 4 + 0] = 0;
        color[i * 4 + 1] = 0;
        color[i * 4 + 2] = 0;
        color[i * 4 + 3] = 255;
        depth[i] = 1.0f;
    }
    if (!tris || !mat || st->triangle_count == 0) {
        ec.dispatch(0, 1, 1);
        return;
    }

    // NaN-defended matrix coerces to identity-diagonal on bad cells.
    float m[16];
    for (int i = 0; i < 16; ++i) m[i] = fzf(mat[i], (i % 5 == 0) ? 1.0f : 0.0f);

    const cardinal::u8 lr = to_u8(st->line_r);
    const cardinal::u8 lg = to_u8(st->line_g);
    const cardinal::u8 lb = to_u8(st->line_b);

    cardinal::u32 px_drawn = 0;
    cardinal::u32 edges_drawn = 0;
    cardinal::u32 offs = 0;

    const float hw = 0.5f * static_cast<float>(W);
    const float hh = 0.5f * static_cast<float>(H);

    for (cardinal::u32 t = 0; t < st->triangle_count; ++t) {
        const float* v = tris + t * 9;
        float sx[3], sy[3], sz[3], sw[3];
        for (int k = 0; k < 3; ++k) {
            project(m,
                    fzf(v[k * 3 + 0], 0.0f),
                    fzf(v[k * 3 + 1], 0.0f),
                    fzf(v[k * 3 + 2], 0.0f),
                    sx[k], sy[k], sz[k], sw[k]);
        }
        // Trivial off-screen reject (if all 3 verts are behind the near
        // plane or way off the viewport, skip — saves Bresenham work).
        const bool any_behind = (sw[0] <= 0.0f) || (sw[1] <= 0.0f) || (sw[2] <= 0.0f);
        const bool all_left   = sx[0] < -1.0f && sx[1] < -1.0f && sx[2] < -1.0f;
        const bool all_right  = sx[0] >  1.0f && sx[1] >  1.0f && sx[2] >  1.0f;
        const bool all_below  = sy[0] < -1.0f && sy[1] < -1.0f && sy[2] < -1.0f;
        const bool all_above  = sy[0] >  1.0f && sy[1] >  1.0f && sy[2] >  1.0f;
        if (any_behind || all_left || all_right || all_below || all_above) {
            ++offs;
            continue;
        }
        // NDC → pixel.
        int px[3], py[3];
        float pz[3];
        for (int k = 0; k < 3; ++k) {
            px[k] = static_cast<int>(sx[k] * hw + hw);
            py[k] = static_cast<int>(hh - sy[k] * hh);
            pz[k] = 0.5f * (sz[k] + 1.0f);
        }
        // Three edges.
        bres_line(px[0], py[0], pz[0], px[1], py[1], pz[1],
                  color, depth, W, H, lr, lg, lb, px_drawn);
        bres_line(px[1], py[1], pz[1], px[2], py[2], pz[2],
                  color, depth, W, H, lr, lg, lb, px_drawn);
        bres_line(px[2], py[2], pz[2], px[0], py[0], pz[0],
                  color, depth, W, H, lr, lg, lb, px_drawn);
        edges_drawn += 3;
    }
    st->pixels_drawn        = px_drawn;
    st->edges_drawn         = edges_drawn;
    st->triangles_offscreen = offs;

    ec.dispatch((W + 7) / 8, (H + 7) / 8, 1);
}

}  // namespace

cardinal::shared_ptr<WireframePass::State> WireframePass::add_to_graph(
    rg::Graph& g,
    rg::ResourceHandle in_tris,
    rg::ResourceHandle in_matrix,
    cardinal::u32 width, cardinal::u32 height,
    cardinal::u32 triangle_count,
    float line_r, float line_g, float line_b)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_tris        = in_tris;
    st->in_matrix      = in_matrix;
    st->width          = width;
    st->height         = height;
    st->triangle_count = triangle_count;
    st->line_r = line_r; st->line_g = line_g; st->line_b = line_b;

    rg::BufferDesc cdesc;
    cdesc.name         = "wire.color";
    cdesc.size_bytes   = static_cast<cardinal::usize>(width) * height * 4u;
    cdesc.stride_bytes = 4;
    st->out_color = g.declare_buffer(cdesc);

    rg::BufferDesc ddesc;
    ddesc.name         = "wire.depth";
    ddesc.size_bytes   = static_cast<cardinal::usize>(width) * height * sizeof(float);
    ddesc.stride_bytes = sizeof(float);
    st->out_depth = g.declare_buffer(ddesc);

    rg::PassDesc pd;
    pd.name = "WireframePass";
    pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_tris,       rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_matrix,     rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{st->out_color, rg::AccessMode::Write, 2});
    pd.accesses.push_back(rg::ResourceAccess{st->out_depth, rg::AccessMode::Write, 3});
    pd.record   = wire_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = (width  + 7) / 8;
    pd.dispatch_y = (height + 7) / 8;
    pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* WireframePass::hlsl_source() noexcept {
    return R"(// Cardinal — WireframePass.hlsl
//
// Three threads per input triangle — one per edge. Each thread Bresenham-
// walks its edge and atomic-writes pixels with depth test. The CPU
// reference walks all 3 edges serially per triangle; the GPU version
// fans out into thread-per-edge for parallelism.
ByteAddressBuffer  in_tris   : register(t0);
ByteAddressBuffer  in_matrix : register(t1);
RWByteAddressBuffer out_color : register(u0);
RWByteAddressBuffer out_depth : register(u1);
cbuffer Params : register(b0) {
    uint width, height, triangle_count, _p0;
    float line_r, line_g, line_b, _p1;
};
float load_f(ByteAddressBuffer b, uint o) { return asfloat(b.Load(o)); }
float4 project_v(float4 v_world) {
    float m[16]; [unroll] for (uint i = 0; i < 16; ++i) m[i] = load_f(in_matrix, i * 4);
    return float4(
        m[0]  * v_world.x + m[1]  * v_world.y + m[2]  * v_world.z + m[3]  * v_world.w,
        m[4]  * v_world.x + m[5]  * v_world.y + m[6]  * v_world.z + m[7]  * v_world.w,
        m[8]  * v_world.x + m[9]  * v_world.y + m[10] * v_world.z + m[11] * v_world.w,
        m[12] * v_world.x + m[13] * v_world.y + m[14] * v_world.z + m[15] * v_world.w);
}
// (Bresenham + atomic writes elided — full kernel ships alongside the RHI
// compute commit; this source-string fixes the binding ABI + projection.)
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint edge_id = tid.x;
    uint t = edge_id / 3;
    uint e = edge_id % 3;
    if (t >= triangle_count) return;
    // Edge endpoints: (e, (e+1)%3) of triangle t.
}
)";
}

}  // namespace cardinal::render::gpu
