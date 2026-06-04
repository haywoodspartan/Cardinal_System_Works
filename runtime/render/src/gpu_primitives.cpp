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

// =============================================================================
// PlanePrimitivePass — subdivided XZ grid (the "rainbow floor")
// =============================================================================
namespace {

void plane_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<PlanePrimitivePass::State*>(uctx);
    const cardinal::u32 sx = clamp_u32(st->spec.subdivisions_x, 1, 1024);
    const cardinal::u32 sz = clamp_u32(st->spec.subdivisions_z, 1, 1024);
    const float ex = fzf(st->spec.half_size_x, 1.0f);
    const float ez = fzf(st->spec.half_size_z, 1.0f);
    const float cx = fzf(st->spec.center_x, 0.0f);
    const float cy = fzf(st->spec.center_y, 0.0f);
    const float cz = fzf(st->spec.center_z, 0.0f);

    float* out = static_cast<float*>(ec.map_buffer_write(st->out_tris));
    if (!out) { ec.dispatch(0, 1, 1); return; }

    const float step_x = (2.0f * ex) / static_cast<float>(sx);
    const float step_z = (2.0f * ez) / static_cast<float>(sz);
    cardinal::usize off = 0;
    for (cardinal::u32 j = 0; j < sz; ++j) {
        const float z0 = cz - ez + static_cast<float>(j)     * step_z;
        const float z1 = cz - ez + static_cast<float>(j + 1) * step_z;
        for (cardinal::u32 i = 0; i < sx; ++i) {
            const float x0 = cx - ex + static_cast<float>(i)     * step_x;
            const float x1 = cx - ex + static_cast<float>(i + 1) * step_x;
            // Quad (CCW when viewed from +Y): (x0, z0), (x1, z0), (x1, z1), (x0, z1)
            // Triangle 1: (x0,z0) (x1,z0) (x1,z1)
            out[off + 0] = x0; out[off + 1] = cy; out[off + 2] = z0;
            out[off + 3] = x1; out[off + 4] = cy; out[off + 5] = z0;
            out[off + 6] = x1; out[off + 7] = cy; out[off + 8] = z1;
            off += 9;
            // Triangle 2: (x0,z0) (x1,z1) (x0,z1)
            out[off + 0] = x0; out[off + 1] = cy; out[off + 2] = z0;
            out[off + 3] = x1; out[off + 4] = cy; out[off + 5] = z1;
            out[off + 6] = x0; out[off + 7] = cy; out[off + 8] = z1;
            off += 9;
        }
    }
    st->triangle_count = 2 * sx * sz;
    ec.dispatch((sx + 7) / 8, (sz + 7) / 8, 1);
}

}  // namespace

cardinal::shared_ptr<PlanePrimitivePass::State> PlanePrimitivePass::add_to_graph(
    rg::Graph& g, PlaneSpec spec)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->spec = spec;
    const cardinal::u32 sx = clamp_u32(spec.subdivisions_x, 1, 1024);
    const cardinal::u32 sz = clamp_u32(spec.subdivisions_z, 1, 1024);
    const cardinal::u32 T = 2 * sx * sz;
    st->triangle_count = T;

    rg::BufferDesc od;
    od.name         = "plane.tris";
    od.size_bytes   = static_cast<cardinal::usize>(T) * 9u * sizeof(float);
    od.stride_bytes = sizeof(float);
    st->out_tris = g.declare_buffer(od);

    rg::PassDesc pd;
    pd.name = "PlanePrimitivePass";
    pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{st->out_tris, rg::AccessMode::Write, 0});
    pd.record   = plane_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = (T + 63) / 64;
    pd.dispatch_y = 1; pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* PlanePrimitivePass::hlsl_source() noexcept {
    return R"(// Cardinal — PlanePrimitivePass.hlsl
//
// Grid plane on XZ. One thread per cell, two triangles emitted.
RWByteAddressBuffer out_tris : register(u0);
cbuffer Params : register(b0) {
    float half_x, half_z, cx, cy;
    float cz, _p0, _p1, _p2;
    uint  sx, sz, _p3, _p4;
};
void store_f(RWByteAddressBuffer b, uint o, float v) { b.Store(o, asuint(v)); }
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x, j = tid.y;
    if (i >= sx || j >= sz) return;
    float step_x = (2 * half_x) / float(sx);
    float step_z = (2 * half_z) / float(sz);
    float x0 = cx - half_x + i * step_x, x1 = x0 + step_x;
    float z0 = cz - half_z + j * step_z, z1 = z0 + step_z;
    uint cell = j * sx + i;
    uint base = cell * 2 * 9 * 4;
    // T1
    store_f(out_tris, base + 0*4, x0); store_f(out_tris, base + 1*4, cy); store_f(out_tris, base + 2*4, z0);
    store_f(out_tris, base + 3*4, x1); store_f(out_tris, base + 4*4, cy); store_f(out_tris, base + 5*4, z0);
    store_f(out_tris, base + 6*4, x1); store_f(out_tris, base + 7*4, cy); store_f(out_tris, base + 8*4, z1);
    // T2
    store_f(out_tris, base + 9*4 + 0*4, x0); store_f(out_tris, base + 9*4 + 1*4, cy); store_f(out_tris, base + 9*4 + 2*4, z0);
    store_f(out_tris, base + 9*4 + 3*4, x1); store_f(out_tris, base + 9*4 + 4*4, cy); store_f(out_tris, base + 9*4 + 5*4, z1);
    store_f(out_tris, base + 9*4 + 6*4, x0); store_f(out_tris, base + 9*4 + 7*4, cy); store_f(out_tris, base + 9*4 + 8*4, z1);
}
)";
}

// =============================================================================
// PyramidPrimitivePass
// =============================================================================
namespace {

void pyramid_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<PyramidPrimitivePass::State*>(uctx);
    const float e  = fzf(st->spec.base_half_extent, 1.0f);
    const float h  = fzf(st->spec.height, 1.5f);
    const float cx = fzf(st->spec.center_x, 0.0f);
    const float cy = fzf(st->spec.center_y, 0.0f);
    const float cz = fzf(st->spec.center_z, 0.0f);

    float* out = static_cast<float*>(ec.map_buffer_write(st->out_tris));
    if (!out) { ec.dispatch(0, 1, 1); return; }

    cardinal::usize off = 0;
    auto emit = [&](float ax, float ay, float az,
                    float bx, float by, float bz,
                    float c_x, float c_y, float c_z) noexcept {
        out[off + 0] = ax; out[off + 1] = ay; out[off + 2] = az;
        out[off + 3] = bx; out[off + 4] = by; out[off + 5] = bz;
        out[off + 6] = c_x;out[off + 7] = c_y;out[off + 8] = c_z;
        off += 9;
    };

    const float apex_x = cx, apex_y = cy + h, apex_z = cz;
    if (st->spec.square_base) {
        // Four base corners (CCW from +Y).
        const float bx0 = cx - e, bz0 = cz - e;
        const float bx1 = cx + e, bz1 = cz - e;
        const float bx2 = cx + e, bz2 = cz + e;
        const float bx3 = cx - e, bz3 = cz + e;
        // Four side faces (CCW outward).
        emit(apex_x, apex_y, apex_z, bx0, cy, bz0, bx1, cy, bz1);
        emit(apex_x, apex_y, apex_z, bx1, cy, bz1, bx2, cy, bz2);
        emit(apex_x, apex_y, apex_z, bx2, cy, bz2, bx3, cy, bz3);
        emit(apex_x, apex_y, apex_z, bx3, cy, bz3, bx0, cy, bz0);
        // Base (CCW from -Y so it's outward-facing).
        emit(bx0, cy, bz0, bx2, cy, bz2, bx1, cy, bz1);
        emit(bx0, cy, bz0, bx3, cy, bz3, bx2, cy, bz2);
        st->triangle_count = 6;
    } else {
        // Tetrahedron: equilateral triangle base + apex.
        const float r = e;
        const float bx0 = cx, bz0 = cz - r;
        const float bx1 = cx + r * 0.866f, bz1 = cz + r * 0.5f;
        const float bx2 = cx - r * 0.866f, bz2 = cz + r * 0.5f;
        emit(apex_x, apex_y, apex_z, bx0, cy, bz0, bx1, cy, bz1);
        emit(apex_x, apex_y, apex_z, bx1, cy, bz1, bx2, cy, bz2);
        emit(apex_x, apex_y, apex_z, bx2, cy, bz2, bx0, cy, bz0);
        emit(bx0, cy, bz0, bx2, cy, bz2, bx1, cy, bz1);
        st->triangle_count = 4;
    }
    ec.dispatch(1, 1, 1);
}

}  // namespace

cardinal::shared_ptr<PyramidPrimitivePass::State> PyramidPrimitivePass::add_to_graph(
    rg::Graph& g, PyramidSpec spec)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->spec = spec;
    st->triangle_count = spec.square_base ? 6u : 4u;

    rg::BufferDesc od;
    od.name         = "pyramid.tris";
    od.size_bytes   = static_cast<cardinal::usize>(st->triangle_count) * 9u * sizeof(float);
    od.stride_bytes = sizeof(float);
    st->out_tris = g.declare_buffer(od);

    rg::PassDesc pd;
    pd.name = "PyramidPrimitivePass";
    pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{st->out_tris, rg::AccessMode::Write, 0});
    pd.record   = pyramid_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = 1; pd.dispatch_y = 1; pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* PyramidPrimitivePass::hlsl_source() noexcept {
    return R"(// Cardinal — PyramidPrimitivePass.hlsl
// 4 or 6 triangles (tetrahedron / square-base pyramid). Single-thread.
RWByteAddressBuffer out_tris : register(u0);
cbuffer Params : register(b0) {
    float e, h, cx, cy;
    float cz, _p0, _p1, _p2;
    uint  square_base, _p3, _p4, _p5;
};
void store_f(RWByteAddressBuffer b, uint o, float v) { b.Store(o, asuint(v)); }
[numthreads(1, 1, 1)]
void main() {
    // Body deferred to RHI compute commit alongside the AS build path —
    // CPU reference is what graph::CpuBackend executes today.
}
)";
}

// =============================================================================
// PolygonViewportPass — hash-color flat-fill triangle rasterizer
// =============================================================================
namespace {

// Deterministic per-triangle colour: splitmix32(id + seed) → HSV→RGB.
inline void triangle_hash_color(cardinal::u32 tri_id, cardinal::u32 seed,
                                cardinal::u8& r, cardinal::u8& g_, cardinal::u8& b) noexcept
{
    cardinal::u32 s = tri_id + seed;
    s += 0x9E3779B9u;
    cardinal::u32 z = s;
    z = (z ^ (z >> 16)) * 0x21f0aaadu;
    z = (z ^ (z >> 15)) * 0x735a2d97u;
    z =  z ^ (z >> 15);
    const float hue_deg = static_cast<float>(z % 3600u) * 0.1f;   // 0..360 in 0.1° steps
    const float h6  = hue_deg / 60.0f;
    const int   i6  = static_cast<int>(h6);
    const float f   = h6 - static_cast<float>(i6);
    const float q   = 1.0f - f;
    const float t   = f;
    float fr = 0, fg = 0, fb = 0;
    switch (i6 % 6) {
        case 0: fr = 1.0f; fg = t;    fb = 0;    break;
        case 1: fr = q;    fg = 1.0f; fb = 0;    break;
        case 2: fr = 0;    fg = 1.0f; fb = t;    break;
        case 3: fr = 0;    fg = q;    fb = 1.0f; break;
        case 4: fr = t;    fg = 0;    fb = 1.0f; break;
        case 5: fr = 1.0f; fg = 0;    fb = q;    break;
    }
    r  = to_u8(fr);
    g_ = to_u8(fg);
    b  = to_u8(fb);
}

inline float edge_fn_2d(float ax, float ay, float bx, float by,
                        float cx, float cy) noexcept {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

void polygon_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<PolygonViewportPass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height;
    if (W == 0 || H == 0) { ec.dispatch(0, 1, 1); return; }

    const float* tris = static_cast<const float*>(ec.map_buffer_read(st->in_tris));
    const float* mat  = static_cast<const float*>(ec.map_buffer_read(st->in_matrix));
    cardinal::u8* color = static_cast<cardinal::u8*>(ec.map_buffer_write(st->out_color));
    float*        depth = static_cast<float*>      (ec.map_buffer_write(st->out_depth));
    if (!color || !depth) { ec.dispatch(0, 1, 1); return; }

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

    float m[16];
    for (int i = 0; i < 16; ++i) m[i] = fzf(mat[i], (i % 5 == 0) ? 1.0f : 0.0f);

    const float hw = 0.5f * static_cast<float>(W);
    const float hh = 0.5f * static_cast<float>(H);

    cardinal::u32 frags = 0, tris_drawn = 0, tris_off = 0;

    for (cardinal::u32 t = 0; t < st->triangle_count; ++t) {
        const float* v = tris + t * 9;
        float sx_v[3], sy_v[3], sz_v[3], sw_v[3];
        for (int k = 0; k < 3; ++k) {
            const float x = fzf(v[k * 3 + 0], 0.0f);
            const float y = fzf(v[k * 3 + 1], 0.0f);
            const float z = fzf(v[k * 3 + 2], 0.0f);
            const float cx = m[0]  * x + m[1]  * y + m[2]  * z + m[3];
            const float cy = m[4]  * x + m[5]  * y + m[6]  * z + m[7];
            const float cz = m[8]  * x + m[9]  * y + m[10] * z + m[11];
            const float cw = m[12] * x + m[13] * y + m[14] * z + m[15];
            sw_v[k] = cw;
            if (cardinal::abs(cw) < 1.0e-8f || !isf(cw)) {
                sx_v[k] = 0; sy_v[k] = 0; sz_v[k] = 1.0e30f;
            } else {
                sx_v[k] = cx / cw;
                sy_v[k] = cy / cw;
                sz_v[k] = cz / cw;
            }
        }
        const bool any_behind = (sw_v[0] <= 0.0f) || (sw_v[1] <= 0.0f) || (sw_v[2] <= 0.0f);
        const bool all_left   = sx_v[0] < -1.0f && sx_v[1] < -1.0f && sx_v[2] < -1.0f;
        const bool all_right  = sx_v[0] >  1.0f && sx_v[1] >  1.0f && sx_v[2] >  1.0f;
        const bool all_below  = sy_v[0] < -1.0f && sy_v[1] < -1.0f && sy_v[2] < -1.0f;
        const bool all_above  = sy_v[0] >  1.0f && sy_v[1] >  1.0f && sy_v[2] >  1.0f;
        if (any_behind || all_left || all_right || all_below || all_above) {
            ++tris_off;
            continue;
        }
        // Per-vertex pixel-space + linear depth.
        float px[3], py[3], pz[3];
        for (int k = 0; k < 3; ++k) {
            px[k] = sx_v[k] * hw + hw;
            py[k] = hh - sy_v[k] * hh;
            pz[k] = 0.5f * (sz_v[k] + 1.0f);
        }
        // Bounding box clipped to viewport.
        float min_x = px[0], max_x = px[0];
        if (px[1] < min_x) min_x = px[1]; if (px[1] > max_x) max_x = px[1];
        if (px[2] < min_x) min_x = px[2]; if (px[2] > max_x) max_x = px[2];
        float min_y = py[0], max_y = py[0];
        if (py[1] < min_y) min_y = py[1]; if (py[1] > max_y) max_y = py[1];
        if (py[2] < min_y) min_y = py[2]; if (py[2] > max_y) max_y = py[2];
        int x0 = static_cast<int>(min_x);          if (x0 < 0) x0 = 0;
        int x1 = static_cast<int>(max_x) + 1;      if (x1 > static_cast<int>(W)) x1 = static_cast<int>(W);
        int y0 = static_cast<int>(min_y);          if (y0 < 0) y0 = 0;
        int y1 = static_cast<int>(max_y) + 1;      if (y1 > static_cast<int>(H)) y1 = static_cast<int>(H);
        if (x0 >= x1 || y0 >= y1) { ++tris_off; continue; }

        const float area = edge_fn_2d(px[0], py[0], px[1], py[1], px[2], py[2]);
        if (cardinal::abs(area) < 1.0e-6f) continue;
        const float inv_area = 1.0f / area;

        cardinal::u8 cr, cg, cb;
        triangle_hash_color(t, st->color_seed, cr, cg, cb);

        bool any_drawn = false;
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                const float fx = static_cast<float>(x) + 0.5f;
                const float fy = static_cast<float>(y) + 0.5f;
                const float w0 = edge_fn_2d(px[1], py[1], px[2], py[2], fx, fy);
                const float w1 = edge_fn_2d(px[2], py[2], px[0], py[0], fx, fy);
                const float w2 = edge_fn_2d(px[0], py[0], px[1], py[1], fx, fy);
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
                if (z > depth[pix]) continue;
                depth[pix] = z;
                color[pix * 4 + 0] = cr;
                color[pix * 4 + 1] = cg;
                color[pix * 4 + 2] = cb;
                color[pix * 4 + 3] = 255;
                ++frags;
                any_drawn = true;
            }
        }
        if (any_drawn) ++tris_drawn;
    }
    st->fragments_drawn     = frags;
    st->triangles_drawn     = tris_drawn;
    st->triangles_offscreen = tris_off;
    ec.dispatch((W + 7) / 8, (H + 7) / 8, 1);
}

}  // namespace

cardinal::shared_ptr<PolygonViewportPass::State> PolygonViewportPass::add_to_graph(
    rg::Graph& g,
    rg::ResourceHandle in_tris,
    rg::ResourceHandle in_matrix,
    cardinal::u32 width, cardinal::u32 height,
    cardinal::u32 triangle_count,
    cardinal::u32 color_seed)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_tris        = in_tris;
    st->in_matrix      = in_matrix;
    st->width          = width;
    st->height         = height;
    st->triangle_count = triangle_count;
    st->color_seed     = color_seed;

    rg::BufferDesc cdesc;
    cdesc.name         = "polyview.color";
    cdesc.size_bytes   = static_cast<cardinal::usize>(width) * height * 4u;
    cdesc.stride_bytes = 4;
    st->out_color = g.declare_buffer(cdesc);

    rg::BufferDesc ddesc;
    ddesc.name         = "polyview.depth";
    ddesc.size_bytes   = static_cast<cardinal::usize>(width) * height * sizeof(float);
    ddesc.stride_bytes = sizeof(float);
    st->out_depth = g.declare_buffer(ddesc);

    rg::PassDesc pd;
    pd.name = "PolygonViewportPass";
    pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_tris,       rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_matrix,     rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{st->out_color, rg::AccessMode::Write, 2});
    pd.accesses.push_back(rg::ResourceAccess{st->out_depth, rg::AccessMode::Write, 3});
    pd.record   = polygon_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = (width  + 7) / 8;
    pd.dispatch_y = (height + 7) / 8;
    pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* PolygonViewportPass::hlsl_source() noexcept {
    return R"(// Cardinal — PolygonViewportPass.hlsl
//
// Tile-parallel. One thread per pixel; per-pixel walk of every visible
// triangle, edge-function inside test, hash-color flat fill, depth test.
// CPU reference produces identical output bytes.
ByteAddressBuffer   in_tris    : register(t0);
ByteAddressBuffer   in_matrix  : register(t1);
RWByteAddressBuffer out_color  : register(u0);
RWByteAddressBuffer out_depth  : register(u1);
cbuffer Params : register(b0) { uint width, height, triangle_count, color_seed; };
float load_f(ByteAddressBuffer b, uint o) { return asfloat(b.Load(o)); }
uint splitmix(uint x) { x += 0x9E3779B9; x = (x ^ (x >> 16)) * 0x21f0aaad; x = (x ^ (x >> 15)) * 0x735a2d97; return x ^ (x >> 15); }
float3 hash_color(uint t, uint seed) {
    uint z = splitmix(t + seed);
    float hue = (float)(z % 3600) * 0.1;
    float h6 = hue / 60.0;
    uint i6 = uint(h6) % 6;
    float f = h6 - floor(h6);
    float q = 1 - f, tt = f;
    if (i6 == 0) return float3(1, tt, 0);
    if (i6 == 1) return float3(q, 1, 0);
    if (i6 == 2) return float3(0, 1, tt);
    if (i6 == 3) return float3(0, q, 1);
    if (i6 == 4) return float3(tt, 0, 1);
    return float3(1, 0, q);
}
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint x = tid.x, y = tid.y;
    if (x >= width || y >= height) return;
    // (Per-pixel triangle walk + edge-fn rasterize body matches the CPU
    // reference; HLSL body lands alongside the RHI compute commit.)
}
)";
}

// =============================================================================
// WorldLabelPass — projects world anchors → screen-space records
// =============================================================================
namespace {

void world_label_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<WorldLabelPass::State*>(uctx);
    const cardinal::u32 N = st->label_count;
    const cardinal::u32 W = st->width;
    const cardinal::u32 H = st->height;

    const float* pts  = static_cast<const float*>(ec.map_buffer_read (st->in_world_points));
    const cardinal::u32* ids = static_cast<const cardinal::u32*>(
        ec.map_buffer_read(st->in_label_ids));
    const float* mat  = static_cast<const float*>(ec.map_buffer_read (st->in_matrix));
    float*       out  = static_cast<float*>      (ec.map_buffer_write(st->out_records));
    cardinal::u8* color = nullptr;
    if (st->draw_anchors && st->out_color.is_valid()) {
        color = static_cast<cardinal::u8*>(ec.map_buffer_write(st->out_color));
    }

    if (N == 0 || !pts || !ids || !mat || !out) {
        ec.dispatch(0, 1, 1);
        return;
    }

    // NaN-defended matrix.
    float m[16];
    for (int i = 0; i < 16; ++i) m[i] = fzf(mat[i], (i % 5 == 0) ? 1.0f : 0.0f);

    const float hw = 0.5f * static_cast<float>(W);
    const float hh = 0.5f * static_cast<float>(H);

    const float* px = pts + 0 * N;
    const float* py = pts + 1 * N;
    const float* pz = pts + 2 * N;

    const cardinal::u8 ar = to_u8(st->anchor_r);
    const cardinal::u8 ag = to_u8(st->anchor_g);
    const cardinal::u8 ab = to_u8(st->anchor_b);

    cardinal::u32 visible = 0, culled = 0;

    for (cardinal::u32 i = 0; i < N; ++i) {
        const float x = fzf(px[i], 0.0f);
        const float y = fzf(py[i], 0.0f);
        const float z = fzf(pz[i], 0.0f);
        const float cx = m[0]  * x + m[1]  * y + m[2]  * z + m[3];
        const float cy = m[4]  * x + m[5]  * y + m[6]  * z + m[7];
        const float cz = m[8]  * x + m[9]  * y + m[10] * z + m[11];
        const float cw = m[12] * x + m[13] * y + m[14] * z + m[15];

        float sx = 0, sy = 0, sd = 1.0e30f;
        bool vis = false;
        if (cardinal::abs(cw) > 1.0e-8f && isf(cw) && cw > 0.0f) {
            const float ndc_x = cx / cw;
            const float ndc_y = cy / cw;
            const float ndc_z = cz / cw;
            sx = ndc_x * hw + hw;
            sy = hh - ndc_y * hh;
            sd = 0.5f * (ndc_z + 1.0f);
            vis = (ndc_x >= -1.0f && ndc_x <= 1.0f &&
                   ndc_y >= -1.0f && ndc_y <= 1.0f &&
                   isf(sd) && sd >= 0.0f && sd <= 1.0f);
        }

        // Record layout: sx, sy, depth, visible (as float bits of u32), label_id
        const cardinal::usize base = i * kWorldLabelRecordFloats;
        out[base + 0] = sx;
        out[base + 1] = sy;
        out[base + 2] = sd;
        const cardinal::u32 visu = vis ? 1u : 0u;
        out[base + 3] = *reinterpret_cast<const float*>(&visu);
        const cardinal::u32 id  = ids[i];
        out[base + 4] = *reinterpret_cast<const float*>(&id);

        if (vis) {
            ++visible;
            // Optional anchor marker: small 3×3 cross at (px, py).
            if (color && W > 0 && H > 0) {
                const int ix = static_cast<int>(sx);
                const int iy = static_cast<int>(sy);
                auto stamp = [&](int x_, int y_) {
                    if (x_ < 0 || y_ < 0 ||
                        static_cast<cardinal::u32>(x_) >= W ||
                        static_cast<cardinal::u32>(y_) >= H) return;
                    const cardinal::usize pix = static_cast<cardinal::usize>(y_) * W +
                                                static_cast<cardinal::usize>(x_);
                    color[pix * 4 + 0] = ar;
                    color[pix * 4 + 1] = ag;
                    color[pix * 4 + 2] = ab;
                    color[pix * 4 + 3] = 255;
                };
                stamp(ix,     iy);
                stamp(ix - 1, iy);
                stamp(ix + 1, iy);
                stamp(ix, iy - 1);
                stamp(ix, iy + 1);
            }
        } else {
            ++culled;
        }
    }
    st->visible_count = visible;
    st->culled_count  = culled;

    ec.dispatch((N + 63) / 64, 1, 1);
}

}  // namespace

cardinal::shared_ptr<WorldLabelPass::State> WorldLabelPass::add_to_graph(
    rg::Graph& g,
    rg::ResourceHandle in_world_points,
    rg::ResourceHandle in_label_ids,
    rg::ResourceHandle in_matrix,
    cardinal::u32 label_count,
    cardinal::u32 width, cardinal::u32 height,
    bool draw_anchors)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_world_points = in_world_points;
    st->in_label_ids    = in_label_ids;
    st->in_matrix       = in_matrix;
    st->label_count     = label_count;
    st->width           = width;
    st->height          = height;
    st->draw_anchors    = draw_anchors;

    rg::BufferDesc rd;
    rd.name         = "label.records";
    rd.size_bytes   = static_cast<cardinal::usize>(label_count) * kWorldLabelRecordFloats * sizeof(float);
    rd.stride_bytes = sizeof(float);
    st->out_records = g.declare_buffer(rd);

    rg::PassDesc pd;
    pd.name = "WorldLabelPass";
    pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_world_points, rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_label_ids,    rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{in_matrix,       rg::AccessMode::Read,  2});
    pd.accesses.push_back(rg::ResourceAccess{st->out_records, rg::AccessMode::Write, 3});

    if (draw_anchors && width > 0 && height > 0) {
        rg::BufferDesc cd_;
        cd_.name         = "label.color";
        cd_.size_bytes   = static_cast<cardinal::usize>(width) * height * 4u;
        cd_.stride_bytes = 4;
        st->out_color = g.declare_buffer(cd_);
        pd.accesses.push_back(rg::ResourceAccess{st->out_color, rg::AccessMode::ReadWrite, 4});
    }
    pd.record   = world_label_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = (label_count + 63) / 64;
    pd.dispatch_y = 1; pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* WorldLabelPass::hlsl_source() noexcept {
    return R"(// Cardinal — WorldLabelPass.hlsl
//
// One thread per label anchor. Project, visibility-test, write one
// LabelRecord. Optional 3×3 anchor stamp via atomic byte writes.
ByteAddressBuffer  in_pts    : register(t0);
ByteAddressBuffer  in_ids    : register(t1);
ByteAddressBuffer  in_matrix : register(t2);
RWByteAddressBuffer out_rec  : register(u0);
RWByteAddressBuffer out_col  : register(u1);  // optional
cbuffer Params : register(b0) {
    uint N, width, height, draw_anchors;
    float ar, ag, ab, _pad;
};
float load_f(ByteAddressBuffer b, uint o) { return asfloat(b.Load(o)); }
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= N) return;
    float x = load_f(in_pts, (0 * N + i) * 4);
    float y = load_f(in_pts, (1 * N + i) * 4);
    float z = load_f(in_pts, (2 * N + i) * 4);
    float m[16]; [unroll] for (uint k = 0; k < 16; ++k) m[k] = load_f(in_matrix, k * 4);
    float cx = m[0]*x + m[1]*y + m[2] *z + m[3];
    float cy = m[4]*x + m[5]*y + m[6] *z + m[7];
    float cz = m[8]*x + m[9]*y + m[10]*z + m[11];
    float cw = m[12]*x + m[13]*y + m[14]*z + m[15];
    float sx = 0, sy = 0, sd = 1e30; uint vis = 0;
    if (cw > 1e-8) {
        float nx = cx / cw, ny = cy / cw, nz = cz / cw;
        sx = nx * (width * 0.5) + width * 0.5;
        sy = height * 0.5 - ny * (height * 0.5);
        sd = 0.5 * (nz + 1.0);
        if (nx >= -1 && nx <= 1 && ny >= -1 && ny <= 1 && sd >= 0 && sd <= 1) vis = 1;
    }
    uint base = i * 5 * 4;
    out_rec.Store(base + 0, asuint(sx));
    out_rec.Store(base + 4, asuint(sy));
    out_rec.Store(base + 8, asuint(sd));
    out_rec.Store(base +12, vis);
    out_rec.Store(base +16, asuint(in_ids.Load(i * 4)));
    // Optional anchor stamp (atomic byte writes) elided — ships with the
    // RHI compute commit alongside the rest of the WorldLabel kernel.
}
)";
}

// =============================================================================
// WorldAxisGizmoPass — RGB X/Y/Z axes at world origin
// =============================================================================
namespace {

void axis_gizmo_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<WorldAxisGizmoPass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height;
    if (W == 0 || H == 0) { ec.dispatch(0, 1, 1); return; }

    const float* mat = static_cast<const float*>(ec.map_buffer_read(st->in_matrix));
    cardinal::u8* color = static_cast<cardinal::u8*>(ec.map_buffer_write(st->out_color));
    float*        depth = static_cast<float*>      (ec.map_buffer_write(st->out_depth));
    if (!mat || !color || !depth) { ec.dispatch(0, 1, 1); return; }

    // The pass owns its color + depth output (declared on add_to_graph),
    // so we clear here the same way WireframePass does. Hosts composite
    // this onto the polygon-viewport output externally (alpha-test the
    // gizmo color != 0 over the under-image).
    for (cardinal::u32 i = 0; i < W * H; ++i) {
        color[i * 4 + 0] = 0;
        color[i * 4 + 1] = 0;
        color[i * 4 + 2] = 0;
        color[i * 4 + 3] = 0;          // alpha 0 = "no gizmo pixel here"
        depth[i] = 1.0f;
    }

    float m[16];
    for (int i = 0; i < 16; ++i) m[i] = fzf(mat[i], (i % 5 == 0) ? 1.0f : 0.0f);

    const float ox = fzf(st->origin_x, 0.0f);
    const float oy = fzf(st->origin_y, 0.0f);
    const float oz = fzf(st->origin_z, 0.0f);
    const float lx = fzf(st->length_x, 1.0f);
    const float ly = fzf(st->length_y, 1.0f);
    const float lz = fzf(st->length_z, 1.0f);

    const float hw = 0.5f * static_cast<float>(W);
    const float hh = 0.5f * static_cast<float>(H);

    auto project_to_pixel = [&](float x, float y, float z,
                                int& out_px, int& out_py, float& out_pz, bool& on_screen) {
        const float cx = m[0]  * x + m[1]  * y + m[2]  * z + m[3];
        const float cy = m[4]  * x + m[5]  * y + m[6]  * z + m[7];
        const float cz = m[8]  * x + m[9]  * y + m[10] * z + m[11];
        const float cw = m[12] * x + m[13] * y + m[14] * z + m[15];
        on_screen = false;
        if (cardinal::abs(cw) < 1.0e-8f || !isf(cw) || cw <= 0.0f) {
            out_px = 0; out_py = 0; out_pz = 1.0e30f;
            return;
        }
        const float nx = cx / cw;
        const float ny = cy / cw;
        const float nz = cz / cw;
        out_px = static_cast<int>(nx * hw + hw);
        out_py = static_cast<int>(hh - ny * hh);
        out_pz = 0.5f * (nz + 1.0f);
        on_screen = (nx >= -1.5f && nx <= 1.5f && ny >= -1.5f && ny <= 1.5f);
    };

    // Project the origin once.
    int ox_px, ox_py; float ox_pz; bool ox_on;
    project_to_pixel(ox, oy, oz, ox_px, ox_py, ox_pz, ox_on);

    auto draw_segment = [&](float ex, float ey, float ez,
                            cardinal::u8 cr, cardinal::u8 cg, cardinal::u8 cb,
                            cardinal::u32& px_drawn) {
        int ex_px, ex_py; float ex_pz; bool ex_on;
        project_to_pixel(ox + ex, oy + ey, oz + ez, ex_px, ex_py, ex_pz, ex_on);
        if (!ox_on && !ex_on) return;

        // Inline Bresenham (deduped from WireframePass's helper to keep
        // the gizmo pass self-contained).
        int x0 = ox_px, y0 = ox_py, x1 = ex_px, y1 = ex_py;
        const int dx = (x1 > x0) ? (x1 - x0) : (x0 - x1);
        const int dy = -((y1 > y0) ? (y1 - y0) : (y0 - y1));
        const int sx = (x0 < x1) ? 1 : -1;
        const int sy = (y0 < y1) ? 1 : -1;
        int err = dx + dy;
        const float total_len = static_cast<float>((dx > -dy) ? dx : -dy);
        int x = x0, y = y0, step = 0;
        const int max_iter = (dx + (-dy)) + 1;
        while (step < max_iter) {
            if (x >= 0 && y >= 0 &&
                static_cast<cardinal::u32>(x) < W &&
                static_cast<cardinal::u32>(y) < H) {
                const float t = (total_len > 0.0f)
                    ? static_cast<float>(step) / total_len : 0.0f;
                const float z = ox_pz + t * (ex_pz - ox_pz);
                const cardinal::usize pix = static_cast<cardinal::usize>(y) * W +
                                            static_cast<cardinal::usize>(x);
                if (isf(z) && z >= 0.0f && z <= 1.0f && z <= depth[pix]) {
                    depth[pix] = z;
                    color[pix * 4 + 0] = cr;
                    color[pix * 4 + 1] = cg;
                    color[pix * 4 + 2] = cb;
                    color[pix * 4 + 3] = 255;
                    ++px_drawn;
                }
            }
            if (x == x1 && y == y1) break;
            const int e2 = 2 * err;
            if (e2 >= dy) { err += dy; x += sx; }
            if (e2 <= dx) { err += dx; y += sy; }
            ++step;
        }
    };

    cardinal::u32 px_drawn = 0;
    draw_segment(lx, 0, 0, 255, 0,   0,   px_drawn);   // +X = red
    draw_segment(0, ly, 0, 0,   255, 0,   px_drawn);   // +Y = green
    draw_segment(0, 0, lz, 0,   0,   255, px_drawn);   // +Z = blue
    st->axis_pixels_drawn = px_drawn;

    ec.dispatch(1, 1, 1);
}

}  // namespace

cardinal::shared_ptr<WorldAxisGizmoPass::State> WorldAxisGizmoPass::add_to_graph(
    rg::Graph& g,
    rg::ResourceHandle in_matrix,
    cardinal::u32 width, cardinal::u32 height,
    float origin_x, float origin_y, float origin_z,
    float length_x, float length_y, float length_z)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_matrix = in_matrix;
    st->width = width; st->height = height;
    st->origin_x = origin_x; st->origin_y = origin_y; st->origin_z = origin_z;
    st->length_x = length_x; st->length_y = length_y; st->length_z = length_z;

    rg::BufferDesc cd;
    cd.name         = "gizmo.color";
    cd.size_bytes   = static_cast<cardinal::usize>(width) * height * 4u;
    cd.stride_bytes = 4;
    st->out_color = g.declare_buffer(cd);

    rg::BufferDesc dd;
    dd.name         = "gizmo.depth";
    dd.size_bytes   = static_cast<cardinal::usize>(width) * height * sizeof(float);
    dd.stride_bytes = sizeof(float);
    st->out_depth = g.declare_buffer(dd);

    rg::PassDesc pd;
    pd.name = "WorldAxisGizmoPass";
    pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_matrix,     rg::AccessMode::Read,      0});
    pd.accesses.push_back(rg::ResourceAccess{st->out_color, rg::AccessMode::ReadWrite, 1});
    pd.accesses.push_back(rg::ResourceAccess{st->out_depth, rg::AccessMode::ReadWrite, 2});
    pd.record   = axis_gizmo_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = 1; pd.dispatch_y = 1; pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* WorldAxisGizmoPass::hlsl_source() noexcept {
    return R"(// Cardinal — WorldAxisGizmoPass.hlsl
// 3 threads — one per axis. Each runs Bresenham + atomic depth on its
// segment. CPU reference walks all 3 serially.
ByteAddressBuffer  in_matrix : register(t0);
RWByteAddressBuffer out_color : register(u0);
RWByteAddressBuffer out_depth : register(u1);
cbuffer Params : register(b0) {
    uint width, height, _p0, _p1;
    float ox, oy, oz, _p2;
    float lx, ly, lz, _p3;
};
[numthreads(3, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // Axis = tid.x ∈ {0, 1, 2}. Body deferred to the RHI compute commit.
}
)";
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
