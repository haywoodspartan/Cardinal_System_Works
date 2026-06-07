#include <cardinal/render/gpu_passes.hpp>

#include <cardinal/core/math/simd_math.hpp>      // simd::frustum_cull_aabbs
#include <cardinal/core/std/cmath.hpp>          // sqrt + isfinite
#include <cardinal/core/std/utility.hpp>        // cardinal::move

namespace cardinal::render::gpu {

namespace {

inline bool isf(float v) noexcept { return cardinal::isfinite(v); }
inline float fz(float v, float fb) noexcept { return isf(v) ? v : fb; }

// ---------------------------------------------------------------------------
// FrustumCullPass — record() implementation
// ---------------------------------------------------------------------------
void cull_record(graph::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<FrustumCullPass::State*>(uctx);
    const cardinal::u32 n = st->count;
    if (n == 0) {
        ec.dispatch(0, 1, 1);
        return;
    }

    const float* aabbs  = static_cast<const float*>(ec.map_buffer_read(st->in_aabbs));
    const float* planes = static_cast<const float*>(ec.map_buffer_read(st->in_planes));
    cardinal::u8* bits  = static_cast<cardinal::u8*>(ec.map_buffer_write(st->out_bits));
    if (!aabbs || !planes || !bits) {
        // Missing binding — leave bits zero (all culled), don't dispatch.
        ec.dispatch(0, 1, 1);
        return;
    }

    // SoA layout: aabbs[0..n)        = min_x
    //             aabbs[n..2n)       = min_y
    //             aabbs[2n..3n)      = min_z
    //             aabbs[3n..4n)      = max_x
    //             aabbs[4n..5n)      = max_y
    //             aabbs[5n..6n)      = max_z
    const float* min_x = aabbs +     0;
    const float* min_y = aabbs + 1 * n;
    const float* min_z = aabbs + 2 * n;
    const float* max_x = aabbs + 3 * n;
    const float* max_y = aabbs + 4 * n;
    const float* max_z = aabbs + 5 * n;

    // Delegate to the engine's AVX-512 cull — bit-for-bit identical to
    // what the existing ForwardRenderer does on the CPU today. Under the
    // future RhiBackend the same Pass would dispatch the HLSL kernel
    // below; the CPU reference here is what regression tests pin.
    cardinal::core::simd::frustum_cull_aabbs(
        bits, planes, min_x, min_y, min_z, max_x, max_y, max_z, n);

    // Stats. Cheap walk — same cache line we just wrote.
    cardinal::u32 vis = 0;
    for (cardinal::u32 i = 0; i < n; ++i) { if (bits[i] != 0) ++vis; }
    st->visible = vis;
    st->culled  = n - vis;

    // One thread per AABB; 64 AABBs per workgroup.
    ec.dispatch((n + 63) / 64, 1, 1);
}

}  // namespace

cardinal::shared_ptr<FrustumCullPass::State> FrustumCullPass::add_to_graph(
    graph::Graph& g,
    graph::ResourceHandle in_aabbs,
    graph::ResourceHandle in_planes,
    cardinal::u32 count)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_aabbs  = in_aabbs;
    st->in_planes = in_planes;
    st->count     = count;

    graph::BufferDesc out_desc;
    out_desc.name         = "cull.visibility";
    out_desc.size_bytes   = static_cast<cardinal::usize>(count);   // 1 byte per AABB
    out_desc.stride_bytes = 1;
    st->out_bits = g.declare_buffer(out_desc);

    graph::PassDesc pd;
    pd.name = "FrustumCullPass";
    pd.kind = graph::PassKind::Compute;
    pd.accesses.push_back(graph::ResourceAccess{in_aabbs,    graph::AccessMode::Read,  0});
    pd.accesses.push_back(graph::ResourceAccess{in_planes,   graph::AccessMode::Read,  1});
    pd.accesses.push_back(graph::ResourceAccess{st->out_bits,graph::AccessMode::Write, 2});
    pd.record   = cull_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = (count + 63) / 64;
    pd.dispatch_y = 1;
    pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));

    return st;
}

const char* FrustumCullPass::hlsl_source() noexcept {
    return R"(// Cardinal — FrustumCullPass.hlsl
//
// One thread per AABB. SoA reads → six plane rejects → 1 byte out.
// Matches cardinal::core::simd::frustum_cull_aabbs bit-for-bit.
ByteAddressBuffer  in_aabbs   : register(t0);
ByteAddressBuffer  in_planes  : register(t1);
RWByteAddressBuffer out_bits  : register(u0);

cbuffer Params : register(b0) {
    uint count;
};

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= count) return;

    // SoA gather: each axis starts at axis * count * 4 bytes.
    uint base = 4u * i;
    float min_x = asfloat(in_aabbs.Load(base + 0u * count * 4u));
    float min_y = asfloat(in_aabbs.Load(base + 1u * count * 4u));
    float min_z = asfloat(in_aabbs.Load(base + 2u * count * 4u));
    float max_x = asfloat(in_aabbs.Load(base + 3u * count * 4u));
    float max_y = asfloat(in_aabbs.Load(base + 4u * count * 4u));
    float max_z = asfloat(in_aabbs.Load(base + 5u * count * 4u));

    uint vis = 1;
    [unroll] for (uint p = 0; p < 6; ++p) {
        float nx = asfloat(in_planes.Load((p * 4 + 0) * 4));
        float ny = asfloat(in_planes.Load((p * 4 + 1) * 4));
        float nz = asfloat(in_planes.Load((p * 4 + 2) * 4));
        float d  = asfloat(in_planes.Load((p * 4 + 3) * 4));
        // Positive-vertex test: pick the AABB corner farthest along the
        // plane normal. If it's still on the negative side, the whole
        // AABB is outside this plane.
        float px = (nx >= 0.0f) ? max_x : min_x;
        float py = (ny >= 0.0f) ? max_y : min_y;
        float pz = (nz >= 0.0f) ? max_z : min_z;
        if (nx * px + ny * py + nz * pz + d < 0.0f) { vis = 0; break; }
    }
    out_bits.Store(i, vis);
}
)";
}

// =============================================================================
// VertexTransformPass
// =============================================================================
namespace {

inline void mat4_apply_point(const float* m, float x, float y, float z,
                             float& out_x, float& out_y, float& out_z) noexcept
{
    // Row-major 4×4; w = 1.
    const float w = m[12] * x + m[13] * y + m[14] * z + m[15];
    const float inv_w = (cardinal::isfinite(w) && cardinal::abs(w) > 1.0e-8f)
        ? 1.0f / w : 1.0f;
    out_x = (m[0] * x + m[1] * y + m[2]  * z + m[3])  * inv_w;
    out_y = (m[4] * x + m[5] * y + m[6]  * z + m[7])  * inv_w;
    out_z = (m[8] * x + m[9] * y + m[10] * z + m[11]) * inv_w;
}

inline void mat4_rotate_vec(const float* m, float x, float y, float z,
                            float& out_x, float& out_y, float& out_z) noexcept
{
    // Upper-left 3×3 (no translation, no perspective). Uniform-scale
    // assumed; the renderer's existing path makes the same assumption.
    out_x = m[0] * x + m[1] * y + m[2]  * z;
    out_y = m[4] * x + m[5] * y + m[6]  * z;
    out_z = m[8] * x + m[9] * y + m[10] * z;
    const float ln2 = out_x * out_x + out_y * out_y + out_z * out_z;
    if (ln2 > 1.0e-12f) {
        const float inv = 1.0f / cardinal::sqrt(ln2);
        out_x *= inv; out_y *= inv; out_z *= inv;
    } else {
        out_x = 0; out_y = 1; out_z = 0;
    }
}

void xform_record(graph::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<VertexTransformPass::State*>(uctx);
    const cardinal::u32 n = st->vertex_count;
    if (n == 0) {
        ec.dispatch(0, 1, 1);
        return;
    }
    const float* in_v  = static_cast<const float*>(ec.map_buffer_read (st->in_local));
    const float* in_m  = static_cast<const float*>(ec.map_buffer_read (st->in_matrix));
    float*       out_v = static_cast<float*>      (ec.map_buffer_write(st->out_world));
    if (!in_v || !in_m || !out_v) {
        ec.dispatch(0, 1, 1);
        return;
    }

    const float* px = in_v + 0 * n;
    const float* py = in_v + 1 * n;
    const float* pz = in_v + 2 * n;
    const float* nx = in_v + 3 * n;
    const float* ny = in_v + 4 * n;
    const float* nz = in_v + 5 * n;

    float* out_px = out_v + 0 * n;
    float* out_py = out_v + 1 * n;
    float* out_pz = out_v + 2 * n;
    float* out_nx = out_v + 3 * n;
    float* out_ny = out_v + 4 * n;
    float* out_nz = out_v + 5 * n;

    // NaN-defended at the matrix: any NaN entry collapses to identity
    // (won't crash, won't poison every output vertex).
    float m[16];
    for (int i = 0; i < 16; ++i) m[i] = fz(in_m[i], (i % 5 == 0) ? 1.0f : 0.0f);

    cardinal::u32 written = 0;
    for (cardinal::u32 i = 0; i < n; ++i) {
        // Position.
        float wx, wy, wz;
        mat4_apply_point(m, fz(px[i], 0.0f), fz(py[i], 0.0f), fz(pz[i], 0.0f),
                         wx, wy, wz);
        out_px[i] = wx;
        out_py[i] = wy;
        out_pz[i] = wz;
        // Normal.
        float rx, ry, rz;
        mat4_rotate_vec(m, fz(nx[i], 0.0f), fz(ny[i], 1.0f), fz(nz[i], 0.0f),
                        rx, ry, rz);
        out_nx[i] = rx;
        out_ny[i] = ry;
        out_nz[i] = rz;
        ++written;
    }
    st->vertices_written = written;

    // 64 vertices per workgroup.
    ec.dispatch((n + 63) / 64, 1, 1);
}

}  // namespace

cardinal::shared_ptr<VertexTransformPass::State> VertexTransformPass::add_to_graph(
    graph::Graph& g,
    graph::ResourceHandle in_local,
    graph::ResourceHandle in_matrix,
    cardinal::u32 vertex_count)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_local     = in_local;
    st->in_matrix    = in_matrix;
    st->vertex_count = vertex_count;

    graph::BufferDesc out_desc;
    out_desc.name         = "xform.world";
    out_desc.size_bytes   = static_cast<cardinal::usize>(vertex_count) * 6u * sizeof(float);
    out_desc.stride_bytes = sizeof(float);
    st->out_world = g.declare_buffer(out_desc);

    graph::PassDesc pd;
    pd.name = "VertexTransformPass";
    pd.kind = graph::PassKind::Compute;
    pd.accesses.push_back(graph::ResourceAccess{in_local,      graph::AccessMode::Read,  0});
    pd.accesses.push_back(graph::ResourceAccess{in_matrix,     graph::AccessMode::Read,  1});
    pd.accesses.push_back(graph::ResourceAccess{st->out_world, graph::AccessMode::Write, 2});
    pd.record   = xform_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = (vertex_count + 63) / 64;
    pd.dispatch_y = 1;
    pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));

    return st;
}

const char* VertexTransformPass::hlsl_source() noexcept {
    return R"(// Cardinal — VertexTransformPass.hlsl
//
// One thread per vertex. Reads SoA (x|y|z|nx|ny|nz) + a row-major 4x4
// model matrix; writes SoA world position + rotated unit normal.
ByteAddressBuffer  in_local  : register(t0);
ByteAddressBuffer  in_matrix : register(t1);
RWByteAddressBuffer out_world: register(u0);

cbuffer Params : register(b0) {
    uint count;
};

float load_f(ByteAddressBuffer buf, uint byte_off) {
    return asfloat(buf.Load(byte_off));
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= count) return;

    uint stride_bytes = 4u;
    float x  = load_f(in_local, (0u * count + i) * stride_bytes);
    float y  = load_f(in_local, (1u * count + i) * stride_bytes);
    float z  = load_f(in_local, (2u * count + i) * stride_bytes);
    float nx = load_f(in_local, (3u * count + i) * stride_bytes);
    float ny = load_f(in_local, (4u * count + i) * stride_bytes);
    float nz = load_f(in_local, (5u * count + i) * stride_bytes);

    float m[16];
    [unroll] for (uint k = 0; k < 16; ++k) m[k] = load_f(in_matrix, k * 4);

    float wx = m[0] * x + m[1] * y + m[2]  * z + m[3];
    float wy = m[4] * x + m[5] * y + m[6]  * z + m[7];
    float wz = m[8] * x + m[9] * y + m[10] * z + m[11];
    float w  = m[12]* x + m[13]* y + m[14] * z + m[15];
    if (abs(w) > 1e-8) { wx /= w; wy /= w; wz /= w; }

    float rx = m[0] * nx + m[1] * ny + m[2]  * nz;
    float ry = m[4] * nx + m[5] * ny + m[6]  * nz;
    float rz = m[8] * nx + m[9] * ny + m[10] * nz;
    float ln = rsqrt(max(rx*rx + ry*ry + rz*rz, 1e-12));
    rx *= ln; ry *= ln; rz *= ln;

    out_world.Store((0u * count + i) * stride_bytes, asuint(wx));
    out_world.Store((1u * count + i) * stride_bytes, asuint(wy));
    out_world.Store((2u * count + i) * stride_bytes, asuint(wz));
    out_world.Store((3u * count + i) * stride_bytes, asuint(rx));
    out_world.Store((4u * count + i) * stride_bytes, asuint(ry));
    out_world.Store((5u * count + i) * stride_bytes, asuint(rz));
}
)";
}

}  // namespace cardinal::render::gpu
