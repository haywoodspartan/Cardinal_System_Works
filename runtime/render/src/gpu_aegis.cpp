#include <cardinal/render/gpu_aegis.hpp>

#include <cardinal/core/std/cmath.hpp>
#include <cardinal/core/std/algorithm.hpp>
#include <cardinal/core/std/utility.hpp>

namespace cardinal::render::gpu {

namespace rg = cardinal::render::graph;

namespace {

inline bool isf(float v) noexcept { return cardinal::isfinite(v); }
inline float fzf(float v, float fb) noexcept { return isf(v) ? v : fb; }

inline cardinal::u8 to_u8(float v) noexcept {
    if (!isf(v)) return 0;
    const float c = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    return static_cast<cardinal::u8>(c * 255.0f + 0.5f);
}

inline float sat(float v) noexcept {
    if (!isf(v)) return 0.0f;
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

// Pack a b0 push block as little-endian bytes for the RhiBackend GPU path
// (PassDesc::push_constants). Scalars map 1:1 to the kernel's cbuffer fields.
inline void put_u32(cardinal::vector<cardinal::u8>& v, cardinal::u32 x) {
    v.push_back(static_cast<cardinal::u8>( x        & 0xFF));
    v.push_back(static_cast<cardinal::u8>((x >>  8) & 0xFF));
    v.push_back(static_cast<cardinal::u8>((x >> 16) & 0xFF));
    v.push_back(static_cast<cardinal::u8>((x >> 24) & 0xFF));
}
inline cardinal::u32 f32_bits(float f) noexcept {
    union { float f; cardinal::u32 u; } x; x.f = f; return x.u;
}
inline void pack_push(cardinal::vector<cardinal::u8>& v,
                      cardinal::u32 a, cardinal::u32 b, float c, cardinal::u32 d) {
    v.clear(); put_u32(v, a); put_u32(v, b); put_u32(v, f32_bits(c)); put_u32(v, d);
}

}  // namespace

// ============================================================================
// GeometryClassifyPass
// ============================================================================
namespace {

void classify_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<GeometryClassifyPass::State*>(uctx);
    const cardinal::u32 N = st->triangle_count;
    const float* tris = static_cast<const float*>(ec.map_buffer_read(st->in_tris));
    const float* curv = st->in_curvature.is_valid()
        ? static_cast<const float*>(ec.map_buffer_read(st->in_curvature)) : nullptr;
    cardinal::u32* cls = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_class));
    if (!tris || !cls) { ec.dispatch(0, 1, 1); return; }
    for (int i = 0; i < 5; ++i) st->class_counts[i] = 0;

    for (cardinal::u32 t = 0; t < N; ++t) {
        const float* v = tris + t * 9;
        // Triangle edges and area (proxy for planarity).
        const float ux = v[3] - v[0], uy = v[4] - v[1], uz = v[5] - v[2];
        const float vx = v[6] - v[0], vy = v[7] - v[1], vz = v[8] - v[2];
        const float nx = uy * vz - uz * vy;
        const float ny = uz * vx - ux * vz;
        const float nz = ux * vy - uy * vx;
        const float a2 = nx * nx + ny * ny + nz * nz;
        const float c  = curv ? fzf(curv[t], 0.0f) : 0.0f;
        // Heuristic classification:
        //   curvature > 0.5 + large area → Displaced
        //   curvature > 0.2              → Organic
        //   curvature > 0.05             → HardSurface
        //   degenerate area              → Foliage (thin geometry proxy)
        //   else                          → Planar
        cardinal::u32 k;
        if (a2 < 1.0e-4f)         k = static_cast<cardinal::u32>(GeometryClass::Foliage);
        else if (c > 0.5f)        k = static_cast<cardinal::u32>(GeometryClass::Displaced);
        else if (c > 0.2f)        k = static_cast<cardinal::u32>(GeometryClass::Organic);
        else if (c > 0.05f)       k = static_cast<cardinal::u32>(GeometryClass::HardSurface);
        else                      k = static_cast<cardinal::u32>(GeometryClass::Planar);
        cls[t] = k;
        st->class_counts[k]++;
    }
    ec.dispatch((N + 63) / 64, 1, 1);
}

}  // namespace

cardinal::shared_ptr<GeometryClassifyPass::State> GeometryClassifyPass::add_to_graph(
    rg::Graph& g, rg::ResourceHandle in_tris,
    rg::ResourceHandle in_curv, cardinal::u32 N)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_tris = in_tris; st->in_curvature = in_curv;
    st->triangle_count = N;
    rg::BufferDesc od;
    od.name = "geom.class"; od.size_bytes = static_cast<cardinal::usize>(N) * sizeof(cardinal::u32);
    od.stride_bytes = sizeof(cardinal::u32);
    st->out_class = g.declare_buffer(od);
    rg::PassDesc pd;
    pd.name = "GeometryClassifyPass"; pd.kind = rg::PassKind::Compute;
    // FIXED SRV LAYOUT: the curvature access is ALWAYS declared — RhiBackend
    // assigns t-registers in access order, so a conditional access would shift
    // out_class between t1/u0 shapes and a single kernel couldn't match both.
    // When the host has no curvature we bind a 4-byte dummy and tell the
    // kernel via the push flag; the CPU record keeps gating on the ORIGINAL
    // handle in State (it never reads the dummy).
    rg::ResourceHandle curv_bind = in_curv;
    if (!curv_bind.is_valid())
        curv_bind = g.declare_buffer(rg::BufferDesc{"geom.nocurv", 4, 4, true});
    pd.accesses.push_back(rg::ResourceAccess{in_tris,       rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{curv_bind,     rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{st->out_class, rg::AccessMode::Write, 2});
    pd.record = classify_record; pd.user_ctx = st.get();
    pd.dispatch_x = (N + 63) / 64;
    pd.compute_hlsl = GeometryClassifyPass::hlsl_source();
    pd.push_constant_size = 16;
    pack_push(pd.push_constants, N, in_curv.is_valid() ? 1u : 0u, 0.0f, 0u);
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* GeometryClassifyPass::hlsl_source() noexcept {
    return
    "// Cardinal - GeometryClassifyPass.hlsl - one thread per triangle.\n"
    "// Mirrors classify_record: 5-class heuristic from area^2 + curvature.\n"
    "ByteAddressBuffer   InTris   : register(t0);   // 9 floats per tri\n"
    "ByteAddressBuffer   InCurv   : register(t1);   // 1 float per tri (or dummy)\n"
    "RWByteAddressBuffer OutClass : register(u0);   // u32 class per tri\n"
    "struct PushT { uint n; uint hasCurv; uint p0; uint p1; };\n"
    "[[vk::push_constant]] ConstantBuffer<PushT> pc : register(b0);\n"
    "#define gN pc.n\n"
    "#define gHasCurv pc.hasCurv\n"
    "[numthreads(64,1,1)]\n"
    "void CSMain(uint3 tid : SV_DispatchThreadID){\n"
    "  uint t = tid.x; if (t >= gN) return;\n"
    "  uint b = t*36u;\n"
    "  float3 v0 = asfloat(InTris.Load3(b+0));\n"
    "  float3 v1 = asfloat(InTris.Load3(b+12));\n"
    "  float3 v2 = asfloat(InTris.Load3(b+24));\n"
    "  float3 n  = cross(v1-v0, v2-v0);\n"
    "  float a2  = dot(n,n);\n"
    "  float c   = 0.0;\n"
    "  if (gHasCurv != 0u) { c = asfloat(InCurv.Load(t*4u)); if (!isfinite(c)) c = 0.0; }\n"
    "  uint k;\n"
    "  if      (a2 < 1.0e-4) k = 4u;   // Foliage (degenerate area)\n"
    "  else if (c > 0.5)     k = 3u;   // Displaced\n"
    "  else if (c > 0.2)     k = 2u;   // Organic\n"
    "  else if (c > 0.05)    k = 1u;   // HardSurface\n"
    "  else                  k = 0u;   // Planar\n"
    "  OutClass.Store(t*4u, k);\n"
    "}\n";
}

// ============================================================================
// MeshletBuildPass
// ============================================================================
namespace {

void meshlet_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<MeshletBuildPass::State*>(uctx);
    const cardinal::u32 N = st->triangle_count;
    const float* tris = static_cast<const float*>(ec.map_buffer_read(st->in_tris));
    float*       out  = static_cast<float*>      (ec.map_buffer_write(st->out_meshlets));
    cardinal::u32* cnt = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_meshlet_count));
    if (!tris || !out || !cnt) { ec.dispatch(0, 1, 1); return; }

    const cardinal::u32 M = (N + kMaxTrisPerMeshlet - 1) / kMaxTrisPerMeshlet;
    for (cardinal::u32 m = 0; m < M; ++m) {
        const cardinal::u32 first = m * kMaxTrisPerMeshlet;
        const cardinal::u32 last  = (first + kMaxTrisPerMeshlet > N) ? N : (first + kMaxTrisPerMeshlet);
        const cardinal::u32 count = last - first;
        // Bounds + normal-cone accumulation.
        float min_x = 1e30f, min_y = 1e30f, min_z = 1e30f;
        float max_x = -1e30f, max_y = -1e30f, max_z = -1e30f;
        float nax = 0, nay = 0, naz = 0;
        for (cardinal::u32 t = first; t < last; ++t) {
            const float* v = tris + t * 9;
            for (int k = 0; k < 3; ++k) {
                const float x = fzf(v[k * 3 + 0], 0.0f);
                const float y = fzf(v[k * 3 + 1], 0.0f);
                const float z = fzf(v[k * 3 + 2], 0.0f);
                if (x < min_x) min_x = x; if (x > max_x) max_x = x;
                if (y < min_y) min_y = y; if (y > max_y) max_y = y;
                if (z < min_z) min_z = z; if (z > max_z) max_z = z;
            }
            const float ux = v[3] - v[0], uy = v[4] - v[1], uz = v[5] - v[2];
            const float vx = v[6] - v[0], vy = v[7] - v[1], vz = v[8] - v[2];
            const float nx = uy * vz - uz * vy;
            const float ny = uz * vx - ux * vz;
            const float nz = ux * vy - uy * vx;
            const float nl = cardinal::sqrt(nx * nx + ny * ny + nz * nz);
            if (nl > 1e-8f) { nax += nx / nl; nay += ny / nl; naz += nz / nl; }
        }
        const float al = cardinal::sqrt(nax * nax + nay * nay + naz * naz);
        if (al > 1e-8f) { nax /= al; nay /= al; naz /= al; } else { nax = 0; nay = 1; naz = 0; }
        // Cone cutoff: half-angle of the spread. Simple: dot of each face normal
        // with the average; min of those is the cone cosine.
        float cone_cos = 1.0f;
        for (cardinal::u32 t = first; t < last; ++t) {
            const float* v = tris + t * 9;
            const float ux = v[3] - v[0], uy = v[4] - v[1], uz = v[5] - v[2];
            const float vx = v[6] - v[0], vy = v[7] - v[1], vz = v[8] - v[2];
            float nx = uy * vz - uz * vy;
            float ny = uz * vx - ux * vz;
            float nz = ux * vy - uy * vx;
            const float nl = cardinal::sqrt(nx * nx + ny * ny + nz * nz);
            if (nl > 1e-8f) { nx /= nl; ny /= nl; nz /= nl; }
            const float d = nax * nx + nay * ny + naz * nz;
            if (d < cone_cos) cone_cos = d;
        }
        float* r = out + m * kMeshletRecordFloats;
        *reinterpret_cast<cardinal::u32*>(&r[0]) = first;
        *reinterpret_cast<cardinal::u32*>(&r[1]) = count;
        r[2] = min_x; r[3] = min_y; r[4] = min_z;
        r[5] = max_x; r[6] = max_y; r[7] = max_z;
        r[8] = nax;   r[9] = nay;   r[10] = naz;
        r[11] = cone_cos;
    }
    cnt[0] = M;
    st->meshlet_count = M;
    ec.dispatch((M + 63) / 64, 1, 1);
}

}  // namespace

cardinal::shared_ptr<MeshletBuildPass::State> MeshletBuildPass::add_to_graph(
    rg::Graph& g, rg::ResourceHandle in_tris, cardinal::u32 N)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_tris = in_tris; st->triangle_count = N;
    const cardinal::u32 M = (N + kMaxTrisPerMeshlet - 1) / kMaxTrisPerMeshlet;
    st->meshlet_count = M;
    rg::BufferDesc md;
    md.name = "meshlets"; md.size_bytes = static_cast<cardinal::usize>(M) * kMeshletRecordFloats * sizeof(float);
    md.stride_bytes = sizeof(float);
    st->out_meshlets = g.declare_buffer(md);
    rg::BufferDesc cd;
    cd.name = "meshlet.count"; cd.size_bytes = sizeof(cardinal::u32);
    cd.stride_bytes = sizeof(cardinal::u32);
    st->out_meshlet_count = g.declare_buffer(cd);
    rg::PassDesc pd;
    pd.name = "MeshletBuildPass"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_tris,                rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{st->out_meshlets,       rg::AccessMode::Write, 1});
    pd.accesses.push_back(rg::ResourceAccess{st->out_meshlet_count,  rg::AccessMode::Write, 2});
    pd.record = meshlet_record; pd.user_ctx = st.get();
    pd.dispatch_x = (M + 63) / 64;
    pd.compute_hlsl = MeshletBuildPass::hlsl_source();
    pd.push_constant_size = 16;
    pack_push(pd.push_constants, N, M, 0.0f, 0u);
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* MeshletBuildPass::hlsl_source() noexcept {
    return
    "// Cardinal - MeshletBuildPass.hlsl - one thread per 64-tri meshlet.\n"
    "// Mirrors meshlet_record: AABB + averaged-normal cone axis + min-dot cone.\n"
    "// Record (48B): u32 first, u32 count, f3 min, f3 max, f3 axis, f cone_cos.\n"
    "ByteAddressBuffer   InTris      : register(t0);\n"
    "RWByteAddressBuffer OutMeshlets : register(u0);\n"
    "RWByteAddressBuffer OutCount    : register(u1);\n"
    "struct PushT { uint triCount; uint meshletCount; uint p0; uint p1; };\n"
    "[[vk::push_constant]] ConstantBuffer<PushT> pc : register(b0);\n"
    "#define gTriCount pc.triCount\n"
    "#define gMeshletCount pc.meshletCount\n"
    "[numthreads(64,1,1)]\n"
    "void CSMain(uint3 tid : SV_DispatchThreadID){\n"
    "  uint m = tid.x; if (m >= gMeshletCount) return;\n"
    "  if (m == 0u) OutCount.Store(0, gMeshletCount);\n"
    "  uint first = m*64u;\n"
    "  uint last  = min(first + 64u, gTriCount);\n"
    "  float3 mn = float3( 1e30, 1e30, 1e30);\n"
    "  float3 mx = float3(-1e30,-1e30,-1e30);\n"
    "  float3 acc = float3(0,0,0);\n"
    "  uint t;\n"
    "  [loop] for (t = first; t < last; ++t) {\n"
    "    uint b = t*36u;\n"
    "    float3 p0 = asfloat(InTris.Load3(b+0));\n"
    "    float3 p1 = asfloat(InTris.Load3(b+12));\n"
    "    float3 p2 = asfloat(InTris.Load3(b+24));\n"
    "    mn = min(mn, min(p0, min(p1, p2)));\n"
    "    mx = max(mx, max(p0, max(p1, p2)));\n"
    "    float3 fn = cross(p1-p0, p2-p0);\n"
    "    float  nl = sqrt(dot(fn,fn));\n"
    "    if (nl > 1e-8) acc += fn / nl;\n"
    "  }\n"
    "  float  al   = sqrt(dot(acc,acc));\n"
    "  float3 axis = (al > 1e-8) ? acc / al : float3(0,1,0);\n"
    "  float cone = 1.0;\n"
    "  [loop] for (t = first; t < last; ++t) {\n"
    "    uint b = t*36u;\n"
    "    float3 p0 = asfloat(InTris.Load3(b+0));\n"
    "    float3 p1 = asfloat(InTris.Load3(b+12));\n"
    "    float3 p2 = asfloat(InTris.Load3(b+24));\n"
    "    float3 fn = cross(p1-p0, p2-p0);\n"
    "    float  nl = sqrt(dot(fn,fn));\n"
    "    if (nl > 1e-8) fn /= nl;\n"
    "    cone = min(cone, dot(axis, fn));\n"
    "  }\n"
    "  uint base = m*48u;\n"
    "  OutMeshlets.Store (base+0,  first);\n"
    "  OutMeshlets.Store (base+4,  last - first);\n"
    "  OutMeshlets.Store3(base+8,  asuint(mn));\n"
    "  OutMeshlets.Store3(base+20, asuint(mx));\n"
    "  OutMeshlets.Store3(base+32, asuint(axis));\n"
    "  OutMeshlets.Store (base+44, asuint(cone));\n"
    "}\n";
}

// ============================================================================
// MeshletCullPass — frustum + normal-cone backface reject
// ============================================================================
namespace {

void meshlet_cull_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<MeshletCullPass::State*>(uctx);
    const cardinal::u32 N = st->meshlet_count;
    const float* ml  = static_cast<const float*>(ec.map_buffer_read(st->in_meshlets));
    const float* mat = static_cast<const float*>(ec.map_buffer_read(st->in_matrix));
    const float* cdir= static_cast<const float*>(ec.map_buffer_read(st->in_camera_dir));
    cardinal::u8* vis = static_cast<cardinal::u8*>(ec.map_buffer_write(st->out_visibility));
    if (!ml || !mat || !cdir || !vis) { ec.dispatch(0, 1, 1); return; }
    float m[16]; for (int i = 0; i < 16; ++i) m[i] = fzf(mat[i], (i % 5 == 0) ? 1.0f : 0.0f);
    const float vx = fzf(cdir[0], 0.0f), vy = fzf(cdir[1], 0.0f), vz = fzf(cdir[2], 1.0f);
    cardinal::u32 nv = 0, fc = 0, bc = 0;
    for (cardinal::u32 i = 0; i < N; ++i) {
        const float* r = ml + i * kMeshletRecordFloats;
        const float mn_x = r[2], mn_y = r[3], mn_z = r[4];
        const float mx_x = r[5], mx_y = r[6], mx_z = r[7];
        const float nax = r[8], nay = r[9], naz = r[10];
        const float cone = r[11];
        // Frustum reject — 8-corner projection.
        const float corners[8][3] = {
            {mn_x, mn_y, mn_z}, {mx_x, mn_y, mn_z}, {mn_x, mx_y, mn_z}, {mx_x, mx_y, mn_z},
            {mn_x, mn_y, mx_z}, {mx_x, mn_y, mx_z}, {mn_x, mx_y, mx_z}, {mx_x, mx_y, mx_z},
        };
        bool any_in = false;
        for (int c = 0; c < 8; ++c) {
            const float cx = m[0]  * corners[c][0] + m[1]  * corners[c][1] + m[2]  * corners[c][2] + m[3];
            const float cy = m[4]  * corners[c][0] + m[5]  * corners[c][1] + m[6]  * corners[c][2] + m[7];
            const float cz = m[8]  * corners[c][0] + m[9]  * corners[c][1] + m[10] * corners[c][2] + m[11];
            const float cw = m[12] * corners[c][0] + m[13] * corners[c][1] + m[14] * corners[c][2] + m[15];
            if (cw <= 0) continue;
            const float nx = cx / cw, ny = cy / cw, nz = cz / cw;
            if (nx >= -1 && nx <= 1 && ny >= -1 && ny <= 1 && nz >= 0 && nz <= 1) {
                any_in = true; break;
            }
        }
        if (!any_in) { vis[i] = 0; ++fc; continue; }
        // Backface reject: if cone fully points away from camera.
        const float face_dot = nax * vx + nay * vy + naz * vz;
        // If face_dot > 0 AND cone is tight enough → all faces point away.
        if (face_dot > 0.0f && (face_dot - (1.0f - cone)) > 0.5f) {
            vis[i] = 0; ++bc; continue;
        }
        vis[i] = 1; ++nv;
    }
    st->visible_count   = nv;
    st->frustum_culled  = fc;
    st->backface_culled = bc;
    ec.dispatch((N + 63) / 64, 1, 1);
}

}  // namespace

cardinal::shared_ptr<MeshletCullPass::State> MeshletCullPass::add_to_graph(
    rg::Graph& g, rg::ResourceHandle in_meshlets, rg::ResourceHandle in_mat,
    rg::ResourceHandle in_dir, cardinal::u32 N)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_meshlets = in_meshlets; st->in_matrix = in_mat; st->in_camera_dir = in_dir;
    st->meshlet_count = N;
    rg::BufferDesc od; od.name = "meshlet.vis"; od.stride_bytes = 1;
    // Rounded up to a whole u32: the GPU kernel packs 4 visibility bytes per
    // 32-bit store (ByteAddressBuffer stores are 4-byte wide), and a root-
    // descriptor UAV has no bounds checking to forgive a tail overhang.
    od.size_bytes = (static_cast<cardinal::usize>(N) + 3u) & ~static_cast<cardinal::usize>(3u);
    st->out_visibility = g.declare_buffer(od);
    rg::PassDesc pd;
    pd.name = "MeshletCullPass"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_meshlets,        rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_mat,             rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{in_dir,             rg::AccessMode::Read,  2});
    pd.accesses.push_back(rg::ResourceAccess{st->out_visibility, rg::AccessMode::Write, 3});
    pd.record = meshlet_cull_record; pd.user_ctx = st.get();
    // GPU: one thread per PACKED WORD (4 meshlets), see kernel.
    pd.dispatch_x = (((N + 3) / 4) + 63) / 64;
    if (pd.dispatch_x == 0) pd.dispatch_x = 1;
    pd.compute_hlsl = MeshletCullPass::hlsl_source();
    pd.push_constant_size = 16;
    pack_push(pd.push_constants, N, 0u, 0.0f, 0u);
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* MeshletCullPass::hlsl_source() noexcept {
    return
    "// Cardinal - MeshletCullPass.hlsl - frustum + normal-cone backface reject.\n"
    "// One thread per packed WORD of 4 visibility bytes (u8 out channel; BAB\n"
    "// stores are 4-byte wide, so each thread culls 4 meshlets + stores 1 u32).\n"
    "ByteAddressBuffer   InMeshlets : register(t0);   // 48B records\n"
    "ByteAddressBuffer   InMatrix   : register(t1);   // 16 floats row-major VP\n"
    "ByteAddressBuffer   InCamDir   : register(t2);   // 3 floats\n"
    "RWByteAddressBuffer OutVis     : register(u0);   // 1 byte per meshlet\n"
    "struct PushT { uint n; uint p0; uint p1; uint p2; };\n"
    "[[vk::push_constant]] ConstantBuffer<PushT> pc : register(b0);\n"
    "#define gN pc.n\n"
    "[numthreads(64,1,1)]\n"
    "void CSMain(uint3 tid : SV_DispatchThreadID){\n"
    "  uint w = tid.x;\n"
    "  uint words = (gN + 3u) / 4u;\n"
    "  if (w >= words) return;\n"
    "  float4 r0 = asfloat(InMatrix.Load4(0));\n"
    "  float4 r1 = asfloat(InMatrix.Load4(16));\n"
    "  float4 r2 = asfloat(InMatrix.Load4(32));\n"
    "  float4 r3 = asfloat(InMatrix.Load4(48));\n"
    "  float3 cam = asfloat(InCamDir.Load3(0));\n"
    "  uint word = 0u;\n"
    "  [loop] for (uint k = 0u; k < 4u; ++k) {\n"
    "    uint i = w*4u + k;\n"
    "    if (i >= gN) break;\n"
    "    uint b = i*48u;\n"
    "    float3 mn   = asfloat(InMeshlets.Load3(b+8));\n"
    "    float3 mx   = asfloat(InMeshlets.Load3(b+20));\n"
    "    float3 axis = asfloat(InMeshlets.Load3(b+32));\n"
    "    float  cone = asfloat(InMeshlets.Load(b+44));\n"
    "    bool any_in = false;\n"
    "    [loop] for (uint c = 0u; c < 8u; ++c) {\n"
    "      float3 p = float3((c & 1u) ? mx.x : mn.x,\n"
    "                        (c & 2u) ? mx.y : mn.y,\n"
    "                        (c & 4u) ? mx.z : mn.z);\n"
    "      float cw = dot(r3.xyz, p) + r3.w;\n"
    "      if (cw <= 0.0) continue;\n"
    "      float nx = (dot(r0.xyz, p) + r0.w) / cw;\n"
    "      float ny = (dot(r1.xyz, p) + r1.w) / cw;\n"
    "      float nz = (dot(r2.xyz, p) + r2.w) / cw;\n"
    "      if (nx >= -1.0 && nx <= 1.0 && ny >= -1.0 && ny <= 1.0 &&\n"
    "          nz >= 0.0 && nz <= 1.0) { any_in = true; break; }\n"
    "    }\n"
    "    uint vis = 0u;\n"
    "    if (any_in) {\n"
    "      float face_dot = dot(axis, cam);\n"
    "      vis = (face_dot > 0.0 && (face_dot - (1.0 - cone)) > 0.5) ? 0u : 1u;\n"
    "    }\n"
    "    word |= vis << (k*8u);\n"
    "  }\n"
    "  OutVis.Store(w*4u, word);\n"
    "}\n";
}

// ============================================================================
// ScreenSpaceErrorPass
// ============================================================================
namespace {

void sse_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<ScreenSpaceErrorPass::State*>(uctx);
    const cardinal::u32 N = st->meshlet_count;
    const float* ml = static_cast<const float*>(ec.map_buffer_read(st->in_meshlets));
    const float* mat= static_cast<const float*>(ec.map_buffer_read(st->in_matrix));
    cardinal::u32* lod = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_lod));
    if (!ml || !mat || !lod) { ec.dispatch(0, 1, 1); return; }
    float m[16]; for (int i = 0; i < 16; ++i) m[i] = fzf(mat[i], (i % 5 == 0) ? 1.0f : 0.0f);
    for (int i = 0; i < 8; ++i) st->lod_counts[i] = 0;
    const float vp_h = static_cast<float>(st->viewport_height);
    for (cardinal::u32 i = 0; i < N; ++i) {
        const float* r = ml + i * kMeshletRecordFloats;
        const float cx = 0.5f * (r[2] + r[5]);
        const float cy = 0.5f * (r[3] + r[6]);
        const float cz = 0.5f * (r[4] + r[7]);
        const float rx = 0.5f * (r[5] - r[2]);
        const float ry = 0.5f * (r[6] - r[3]);
        const float rz = 0.5f * (r[7] - r[4]);
        const float radius = cardinal::sqrt(rx * rx + ry * ry + rz * rz);
        // Project the center, derive the screen-space radius.
        const float pw = m[12] * cx + m[13] * cy + m[14] * cz + m[15];
        if (pw <= 1e-4f) { lod[i] = 0; ++st->lod_counts[0]; continue; }
        const float screen_radius_px = radius * vp_h / (pw * 2.0f);
        // LOD level = max(0, log2(screen_radius / error_threshold))
        cardinal::u32 chosen = 0;
        float r_at = screen_radius_px / fzf(st->error_threshold_px, 2.0f);
        while (r_at < 1.0f && chosen < 7) { r_at *= 2.0f; ++chosen; }
        lod[i] = chosen;
        st->lod_counts[chosen]++;
    }
    ec.dispatch((N + 63) / 64, 1, 1);
}

}  // namespace

cardinal::shared_ptr<ScreenSpaceErrorPass::State> ScreenSpaceErrorPass::add_to_graph(
    rg::Graph& g, rg::ResourceHandle in_meshlets, rg::ResourceHandle in_mat,
    cardinal::u32 N, cardinal::u32 vp_h)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_meshlets = in_meshlets; st->in_matrix = in_mat;
    st->meshlet_count = N; st->viewport_height = vp_h;
    rg::BufferDesc od; od.name = "sse.lod"; od.size_bytes = N * sizeof(cardinal::u32);
    od.stride_bytes = sizeof(cardinal::u32);
    st->out_lod = g.declare_buffer(od);
    rg::PassDesc pd;
    pd.name = "ScreenSpaceErrorPass"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_meshlets, rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_mat,      rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{st->out_lod, rg::AccessMode::Write, 2});
    pd.record = sse_record; pd.user_ctx = st.get();
    pd.dispatch_x = (N + 63) / 64;
    pd.compute_hlsl = ScreenSpaceErrorPass::hlsl_source();
    pd.push_constant_size = 16;
    // b0 = { uint gN, uint _pad, float gVpH, float gThresh } — pack manually
    // (pack_push's fixed shape is u32,u32,f32,u32).
    pd.push_constants.clear();
    put_u32(pd.push_constants, N);
    put_u32(pd.push_constants, 0u);
    put_u32(pd.push_constants, f32_bits(static_cast<float>(vp_h)));
    put_u32(pd.push_constants, f32_bits(cardinal::isfinite(st->error_threshold_px)
                                            ? st->error_threshold_px : 2.0f));
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* ScreenSpaceErrorPass::hlsl_source() noexcept {
    return
    "// Cardinal - ScreenSpaceErrorPass.hlsl - per-meshlet screen radius -> LOD.\n"
    "ByteAddressBuffer   InMeshlets : register(t0);   // 48B records\n"
    "ByteAddressBuffer   InMatrix   : register(t1);   // 16 floats row-major VP\n"
    "RWByteAddressBuffer OutLod     : register(u0);   // u32 LOD 0..7 per meshlet\n"
    "struct PushT { uint n; uint p0; float vpH; float thresh; };\n"
    "[[vk::push_constant]] ConstantBuffer<PushT> pc : register(b0);\n"
    "#define gN pc.n\n"
    "#define gVpH pc.vpH\n"
    "#define gThresh pc.thresh\n"
    "[numthreads(64,1,1)]\n"
    "void CSMain(uint3 tid : SV_DispatchThreadID){\n"
    "  uint i = tid.x; if (i >= gN) return;\n"
    "  uint b = i*48u;\n"
    "  float3 mn = asfloat(InMeshlets.Load3(b+8));\n"
    "  float3 mx = asfloat(InMeshlets.Load3(b+20));\n"
    "  float3 c  = 0.5*(mn + mx);\n"
    "  float3 e  = 0.5*(mx - mn);\n"
    "  float  radius = sqrt(dot(e,e));\n"
    "  float4 r3 = asfloat(InMatrix.Load4(48));\n"
    "  float  pw = dot(r3.xyz, c) + r3.w;\n"
    "  uint chosen = 0u;\n"
    "  if (pw > 1e-4) {\n"
    "    float r_at = (radius * gVpH / (pw * 2.0)) / gThresh;\n"
    "    [loop] while (r_at < 1.0 && chosen < 7u) { r_at *= 2.0; ++chosen; }\n"
    "  }\n"
    "  OutLod.Store(i*4u, chosen);\n"
    "}\n";
}

// ============================================================================
// DrawIndirectGenPass
// ============================================================================
namespace {

void indirect_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<DrawIndirectGenPass::State*>(uctx);
    const cardinal::u32 N = st->meshlet_count;
    const cardinal::u8* vis = static_cast<const cardinal::u8*>(ec.map_buffer_read(st->in_visibility));
    const float* ml = static_cast<const float*>(ec.map_buffer_read(st->in_meshlets));
    IndirectDrawCmd* cmds = static_cast<IndirectDrawCmd*>(ec.map_buffer_write(st->out_commands));
    cardinal::u32* cnt = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_count));
    if (!vis || !ml || !cmds || !cnt) { ec.dispatch(0, 1, 1); return; }
    cardinal::u32 emitted = 0;
    for (cardinal::u32 i = 0; i < N; ++i) {
        if (!vis[i]) continue;
        const float* r = ml + i * kMeshletRecordFloats;
        cmds[emitted].first_triangle = *reinterpret_cast<const cardinal::u32*>(&r[0]);
        cmds[emitted].triangle_count = *reinterpret_cast<const cardinal::u32*>(&r[1]);
        cmds[emitted].meshlet_id     = i;
        cmds[emitted]._pad           = 0;
        ++emitted;
    }
    cnt[0] = emitted;
    st->commands_emitted = emitted;
    ec.dispatch((N + 63) / 64, 1, 1);
}

}  // namespace

cardinal::shared_ptr<DrawIndirectGenPass::State> DrawIndirectGenPass::add_to_graph(
    rg::Graph& g, rg::ResourceHandle in_vis, rg::ResourceHandle in_meshlets, cardinal::u32 N)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_visibility = in_vis; st->in_meshlets = in_meshlets; st->meshlet_count = N;
    rg::BufferDesc cd; cd.name = "draw.cmds"; cd.size_bytes = N * sizeof(IndirectDrawCmd);
    cd.stride_bytes = sizeof(IndirectDrawCmd);
    st->out_commands = g.declare_buffer(cd);
    rg::BufferDesc nd; nd.name = "draw.count"; nd.size_bytes = sizeof(cardinal::u32);
    nd.stride_bytes = sizeof(cardinal::u32);
    st->out_count = g.declare_buffer(nd);
    rg::PassDesc pd;
    pd.name = "DrawIndirectGenPass"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_vis,           rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_meshlets,      rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{st->out_commands, rg::AccessMode::Write, 2});
    pd.accesses.push_back(rg::ResourceAccess{st->out_count,    rg::AccessMode::Write, 3});
    pd.record = indirect_record; pd.user_ctx = st.get();
    // GPU: ONE workgroup (256 threads) sweeps the stream with an LDS scan —
    // order-preserving compaction, chunked over ceil(N/256) (see kernel).
    pd.dispatch_x = 1;
    pd.compute_hlsl = DrawIndirectGenPass::hlsl_source();
    pd.push_constant_size = 16;
    pack_push(pd.push_constants, N, 0u, 0.0f, 0u);
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* DrawIndirectGenPass::hlsl_source() noexcept {
    // ORDER-PRESERVING PARALLEL COMPACTION — a hand-rolled single-workgroup
    // Hillis-Steele inclusive scan in groupshared memory. 256 threads sweep
    // the visibility stream in chunks; each chunk's scan assigns every
    // visible meshlet the exact slot the CPU's serial loop would (ascending
    // i), so the harness's byte-exact CmdPrefix compare still holds while the
    // work runs 256-wide with coalesced loads. s_base carries the running
    // total across chunks; thread 0 publishes the final count — the pass is
    // fully self-contained (no pre-clear dispatch, no cross-group atomics).
    return
    "// Cardinal - DrawIndirectGenPass.hlsl - LDS-scan stream compaction.\n"
    "ByteAddressBuffer   InVis      : register(t0);   // 1 byte per meshlet\n"
    "ByteAddressBuffer   InMeshlets : register(t1);   // 48B records\n"
    "RWByteAddressBuffer OutCmds    : register(u0);   // 16B IndirectDrawCmd\n"
    "RWByteAddressBuffer OutCount   : register(u1);   // u32 emitted\n"
    "struct PushT { uint n; uint p0; uint p1; uint p2; };\n"
    "[[vk::push_constant]] ConstantBuffer<PushT> pc : register(b0);\n"
    "#define gN pc.n\n"
    "groupshared uint s_scan[256];\n"
    "groupshared uint s_base;\n"
    "[numthreads(256,1,1)]\n"
    "void CSMain(uint3 gtid : SV_GroupThreadID){\n"
    "  uint tid = gtid.x;               // NO early returns: barriers below\n"
    "  if (tid == 0u) s_base = 0u;      // groupshared starts undefined\n"
    "  uint chunks = (gN + 255u) / 256u;\n"
    "  [loop] for (uint c = 0u; c < chunks; ++c) {\n"
    "    uint i = c*256u + tid;\n"
    "    uint v = 0u;\n"
    "    if (i < gN) {\n"
    "      // Guard the LOAD itself: root descriptors have no bounds checking.\n"
    "      v = (InVis.Load((i/4u)*4u) >> ((i%4u)*8u)) & 0xFFu;\n"
    "      if (v != 0u) v = 1u;         // any nonzero byte = visible-once\n"
    "    }\n"
    "    s_scan[tid] = v;\n"
    "    GroupMemoryBarrierWithGroupSync();   // publish stores (+ s_base on c==0)\n"
    "    [unroll] for (uint off = 1u; off < 256u; off <<= 1u) {\n"
    "      uint t = (tid >= off) ? s_scan[tid - off] : 0u;\n"
    "      GroupMemoryBarrierWithGroupSync();  // all reads before any write\n"
    "      s_scan[tid] += t;\n"
    "      GroupMemoryBarrierWithGroupSync();\n"
    "    }\n"
    "    if (v != 0u) {\n"
    "      uint slot = s_base + s_scan[tid] - 1u;   // inclusive scan -> 0-based\n"
    "      uint mb = i*48u;\n"
    "      OutCmds.Store4(slot*16u,\n"
    "                     uint4(InMeshlets.Load(mb+0), InMeshlets.Load(mb+4), i, 0u));\n"
    "    }\n"
    "    GroupMemoryBarrierWithGroupSync();   // writes done before s_base bump\n"
    "    if (tid == 0u) s_base += s_scan[255];\n"
    "    GroupMemoryBarrierWithGroupSync();\n"
    "  }\n"
    "  if (tid == 0u) OutCount.Store(0, s_base);\n"
    "}\n";
}

// ============================================================================
// TiledLightCullPass
// ============================================================================
namespace {

void tile_light_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<TiledLightCullPass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height, L = st->light_count;
    const float* lights = static_cast<const float*>(ec.map_buffer_read(st->in_lights));
    cardinal::u32* idx  = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_tile_lights));
    cardinal::u32* cnt  = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_tile_counts));
    if (!lights || !idx || !cnt) { ec.dispatch(0, 1, 1); return; }
    const cardinal::u32 tx = st->tiles_x, ty = st->tiles_y;
    cardinal::u32 assigns = 0, trunc = 0;
    for (cardinal::u32 t = 0; t < tx * ty; ++t) cnt[t] = 0;
    // Simplified: every light is considered to affect every tile within
    // its range. A proper tile cull uses depth bounds + light frustum;
    // this is a regression-pinnable reference, not a perf-optimal version.
    for (cardinal::u32 ti = 0; ti < tx * ty; ++ti) {
        cardinal::u32 c = 0;
        for (cardinal::u32 l = 0; l < L && c < kMaxLightsPerTile; ++l) {
            const float intensity = fzf(lights[l * 16 + 11], 0.0f);
            if (intensity <= 0.0f) continue;
            idx[ti * kMaxLightsPerTile + c] = l;
            ++c; ++assigns;
        }
        if (L > kMaxLightsPerTile) trunc += (L - kMaxLightsPerTile);
        cnt[ti] = c;
    }
    st->total_light_assignments = assigns;
    st->truncated_tiles = trunc;
    (void)W; (void)H;
    ec.dispatch(tx, ty, 1);
}

}  // namespace

cardinal::shared_ptr<TiledLightCullPass::State> TiledLightCullPass::add_to_graph(
    rg::Graph& g, rg::ResourceHandle in_lights, rg::ResourceHandle in_depth,
    rg::ResourceHandle in_mat, cardinal::u32 W, cardinal::u32 H, cardinal::u32 L)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_lights = in_lights; st->in_depth = in_depth; st->in_matrix = in_mat;
    st->width = W; st->height = H; st->light_count = L;
    st->tiles_x = (W + kLightTileSize - 1) / kLightTileSize;
    st->tiles_y = (H + kLightTileSize - 1) / kLightTileSize;
    const cardinal::u32 T = st->tiles_x * st->tiles_y;
    rg::BufferDesc id; id.name = "tile.lights"; id.size_bytes = T * kMaxLightsPerTile * sizeof(cardinal::u32);
    id.stride_bytes = sizeof(cardinal::u32);
    st->out_tile_lights = g.declare_buffer(id);
    rg::BufferDesc cd; cd.name = "tile.counts"; cd.size_bytes = T * sizeof(cardinal::u32);
    cd.stride_bytes = sizeof(cardinal::u32);
    st->out_tile_counts = g.declare_buffer(cd);
    rg::PassDesc pd;
    pd.name = "TiledLightCullPass"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_lights,           rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_depth,            rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{in_mat,              rg::AccessMode::Read,  2});
    pd.accesses.push_back(rg::ResourceAccess{st->out_tile_lights, rg::AccessMode::Write, 3});
    pd.accesses.push_back(rg::ResourceAccess{st->out_tile_counts, rg::AccessMode::Write, 4});
    pd.record = tile_light_record; pd.user_ctx = st.get();
    pd.dispatch_x = st->tiles_x; pd.dispatch_y = st->tiles_y;
    pd.compute_hlsl = TiledLightCullPass::hlsl_source();
    pd.push_constant_size = 16;
    pack_push(pd.push_constants, st->tiles_x, st->tiles_y, 0.0f, L);
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* TiledLightCullPass::hlsl_source() noexcept {
    return
    "// Cardinal - TiledLightCullPass.hlsl - one thread group per 16x16 tile.\n"
    "// Mirrors tile_light_record's simplified reference: every intensity>0\n"
    "// light lands in every tile (capped 64) — depth-bounds culling arrives\n"
    "// with the real depth wiring. Depth/matrix are declared to keep the\n"
    "// t-register layout in access order, but unused (as on the CPU).\n"
    "ByteAddressBuffer   InLights      : register(t0);  // 16 floats per light\n"
    "ByteAddressBuffer   InDepth       : register(t1);  // unused (layout only)\n"
    "ByteAddressBuffer   InMatrix      : register(t2);  // unused (layout only)\n"
    "RWByteAddressBuffer OutTileLights : register(u0);  // 64 u32 per tile\n"
    "RWByteAddressBuffer OutTileCounts : register(u1);  // u32 per tile\n"
    "struct PushT { uint tilesX; uint tilesY; float p0; uint lightCount; };\n"
    "[[vk::push_constant]] ConstantBuffer<PushT> pc : register(b0);\n"
    "#define gTilesX pc.tilesX\n"
    "#define gTilesY pc.tilesY\n"
    "#define gLightCount pc.lightCount\n"
    "[numthreads(1,1,1)]\n"
    "void CSMain(uint3 gid : SV_GroupID){\n"
    "  if (gid.x >= gTilesX || gid.y >= gTilesY) return;\n"
    "  uint ti = gid.y*gTilesX + gid.x;\n"
    "  uint c = 0u;\n"
    "  [loop] for (uint l = 0u; l < gLightCount && c < 64u; ++l) {\n"
    "    float inten = asfloat(InLights.Load(l*64u + 44u));\n"
    "    if (!(inten > 0.0)) continue;   // NaN compares false — matches fzf->0\n"
    "    OutTileLights.Store((ti*64u + c)*4u, l);\n"
    "    ++c;\n"
    "  }\n"
    "  OutTileCounts.Store(ti*4u, c);\n"
    "}\n";
}

// ============================================================================
// VBufResolvePass — samples V-Buf, shades each pixel
// ============================================================================
namespace {

void resolve_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<VBufResolvePass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height, MC = st->material_count;
    const cardinal::u32* prim = static_cast<const cardinal::u32*>(ec.map_buffer_read(st->in_prim_id));
    const cardinal::u32* matid = static_cast<const cardinal::u32*>(ec.map_buffer_read(st->in_mat_id));
    const float* mats = static_cast<const float*>(ec.map_buffer_read(st->in_materials));
    const float* amb  = static_cast<const float*>(ec.map_buffer_read(st->in_ambient));
    float* rad = static_cast<float*>(ec.map_buffer_write(st->out_radiance));
    if (!prim || !matid || !mats || !amb || !rad) { ec.dispatch(0, 1, 1); return; }
    const float ar = fzf(amb[0], 0.0f), ag = fzf(amb[1], 0.0f), ab = fzf(amb[2], 0.0f);
    cardinal::u32 shaded = 0, sky = 0;
    for (cardinal::u32 i = 0; i < W * H; ++i) {
        const cardinal::u32 pid = prim[i];
        if (pid == kInvalidPrimId) {
            rad[i * 3 + 0] = ar; rad[i * 3 + 1] = ag; rad[i * 3 + 2] = ab;
            ++sky; continue;
        }
        const cardinal::u32 mid = matid[i];
        const float* m = (mid < MC) ? (mats + mid * 8) : nullptr;
        const float base_r = m ? fzf(m[0], 0.5f) : 0.5f;
        const float base_g = m ? fzf(m[1], 0.5f) : 0.5f;
        const float base_b = m ? fzf(m[2], 0.5f) : 0.5f;
        // Diffuse-only ambient shading proxy. Tile light loop deferred to
        // the GPU compute commit.
        rad[i * 3 + 0] = base_r * (ar + 1.0f);
        rad[i * 3 + 1] = base_g * (ag + 1.0f);
        rad[i * 3 + 2] = base_b * (ab + 1.0f);
        ++shaded;
    }
    st->pixels_shaded = shaded;
    st->pixels_sky    = sky;
    ec.dispatch((W + 7) / 8, (H + 7) / 8, 1);
}

}  // namespace

cardinal::shared_ptr<VBufResolvePass::State> VBufResolvePass::add_to_graph(
    rg::Graph& g,
    rg::ResourceHandle in_depth, rg::ResourceHandle in_prim, rg::ResourceHandle in_mat,
    rg::ResourceHandle in_norm, rg::ResourceHandle in_mats,
    rg::ResourceHandle in_tl, rg::ResourceHandle in_tc,
    rg::ResourceHandle in_lights, rg::ResourceHandle in_amb,
    cardinal::u32 W, cardinal::u32 H, cardinal::u32 MC, cardinal::u32 LC)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_depth = in_depth; st->in_prim_id = in_prim; st->in_mat_id = in_mat;
    st->in_normal = in_norm; st->in_materials = in_mats;
    st->in_tile_lights = in_tl; st->in_tile_counts = in_tc;
    st->in_lights = in_lights; st->in_ambient = in_amb;
    st->width = W; st->height = H; st->material_count = MC; st->light_count = LC;
    rg::BufferDesc od; od.name = "resolve.rad"; od.size_bytes = W * H * 3 * sizeof(float);
    od.stride_bytes = sizeof(float);
    st->out_radiance = g.declare_buffer(od);
    rg::PassDesc pd;
    pd.name = "VBufResolvePass"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_depth,      rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_prim,       rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{in_mat,        rg::AccessMode::Read,  2});
    pd.accesses.push_back(rg::ResourceAccess{in_norm,       rg::AccessMode::Read,  3});
    pd.accesses.push_back(rg::ResourceAccess{in_mats,       rg::AccessMode::Read,  4});
    pd.accesses.push_back(rg::ResourceAccess{in_tl,         rg::AccessMode::Read,  5});
    pd.accesses.push_back(rg::ResourceAccess{in_tc,         rg::AccessMode::Read,  6});
    pd.accesses.push_back(rg::ResourceAccess{in_lights,     rg::AccessMode::Read,  7});
    pd.accesses.push_back(rg::ResourceAccess{in_amb,        rg::AccessMode::Read,  8});
    pd.accesses.push_back(rg::ResourceAccess{st->out_radiance, rg::AccessMode::Write, 9});
    pd.record = resolve_record; pd.user_ctx = st.get();
    pd.dispatch_x = (W + 7) / 8; pd.dispatch_y = (H + 7) / 8;
    pd.compute_hlsl = VBufResolvePass::hlsl_source();
    pd.push_constant_size = 16;
    pack_push(pd.push_constants, W, H, 0.0f, MC);
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* VBufResolvePass::hlsl_source() noexcept {
    return
    "// Cardinal - VBufResolvePass.hlsl - per-pixel ambient-proxy shade.\n"
    "// Mirrors resolve_record: sky pixels take ambient; hit pixels take\n"
    "// material base * (ambient + 1). The tile-light loop arrives with the\n"
    "// real lighting commit. 9 SRVs — declared in access order (t0..t8) even\n"
    "// where unused, to keep the register layout stable.\n"
    "ByteAddressBuffer   InDepth      : register(t0);  // unused (layout only)\n"
    "ByteAddressBuffer   InPrim       : register(t1);  // u32 per pixel\n"
    "ByteAddressBuffer   InMatId      : register(t2);  // u32 per pixel\n"
    "ByteAddressBuffer   InNormal     : register(t3);  // unused (layout only)\n"
    "ByteAddressBuffer   InMaterials  : register(t4);  // 8 floats per material\n"
    "ByteAddressBuffer   InTileLights : register(t5);  // unused (layout only)\n"
    "ByteAddressBuffer   InTileCounts : register(t6);  // unused (layout only)\n"
    "ByteAddressBuffer   InLights     : register(t7);  // unused (layout only)\n"
    "ByteAddressBuffer   InAmbient    : register(t8);  // 3 floats\n"
    "RWByteAddressBuffer OutRadiance  : register(u0);  // float3 per pixel\n"
    "struct PushT { uint w; uint h; float p0; uint matCount; };\n"
    "[[vk::push_constant]] ConstantBuffer<PushT> pc : register(b0);\n"
    "#define gW pc.w\n"
    "#define gH pc.h\n"
    "#define gMatCount pc.matCount\n"
    "float fz(float v, float fb) { return isfinite(v) ? v : fb; }\n"
    "[numthreads(8,8,1)]\n"
    "void CSMain(uint3 tid : SV_DispatchThreadID){\n"
    "  if (tid.x >= gW || tid.y >= gH) return;\n"
    "  uint i = tid.y*gW + tid.x;\n"
    "  float3 ambr = asfloat(InAmbient.Load3(0));\n"
    "  float3 amb  = float3(fz(ambr.x,0.0), fz(ambr.y,0.0), fz(ambr.z,0.0));\n"
    "  uint pid = InPrim.Load(i*4u);\n"
    "  float3 radv;\n"
    "  if (pid == 0xFFFFFFFFu) {\n"
    "    radv = amb;\n"
    "  } else {\n"
    "    uint  mid  = InMatId.Load(i*4u);\n"
    "    float3 base = float3(0.5, 0.5, 0.5);\n"
    "    if (mid < gMatCount) {\n"
    "      float3 m = asfloat(InMaterials.Load3(mid*32u));\n"
    "      base = float3(fz(m.x,0.5), fz(m.y,0.5), fz(m.z,0.5));\n"
    "    }\n"
    "    radv = base * (amb + 1.0);\n"
    "  }\n"
    "  OutRadiance.Store3(i*12u, asuint(radv));\n"
    "}\n";
}

// ============================================================================
// MotionVectorPass
// ============================================================================
namespace {

void motion_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<MotionVectorPass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height;
    const float* depth = static_cast<const float*>(ec.map_buffer_read(st->in_depth));
    const float* vp    = static_cast<const float*>(ec.map_buffer_read(st->in_view_proj));
    const float* vp_p  = static_cast<const float*>(ec.map_buffer_read(st->in_view_proj_prev));
    cardinal::u32* mv  = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_motion));
    if (!depth || !vp || !vp_p || !mv) { ec.dispatch(0, 1, 1); return; }
    // For each pixel, depth → NDC z → reconstructed world pos via inv vp.
    // Here we approximate: the depth-only delta is sufficient for a CPU
    // reference. The host wires the full reconstruction with the proper
    // inverse-VP buffer when the RHI compute path lands.
    cardinal::u32 px_mv = 0;
    for (cardinal::u32 i = 0; i < W * H; ++i) {
        const float d = depth[i];
        if (!isf(d) || d >= 1.0f) { mv[i] = 0; continue; }
        // Snorm16 zero motion → 0x80008000-ish; using zero packs as zero delta.
        mv[i] = 0;
        ++px_mv;
    }
    st->pixels_with_motion = px_mv;
    (void)vp; (void)vp_p;
    ec.dispatch((W + 7) / 8, (H + 7) / 8, 1);
}

}  // namespace

cardinal::shared_ptr<MotionVectorPass::State> MotionVectorPass::add_to_graph(
    rg::Graph& g, rg::ResourceHandle in_depth,
    rg::ResourceHandle in_vp, rg::ResourceHandle in_vp_prev,
    cardinal::u32 W, cardinal::u32 H)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_depth = in_depth; st->in_view_proj = in_vp; st->in_view_proj_prev = in_vp_prev;
    st->width = W; st->height = H;
    rg::BufferDesc od; od.name = "motion.mv"; od.size_bytes = W * H * sizeof(cardinal::u32);
    od.stride_bytes = sizeof(cardinal::u32);
    st->out_motion = g.declare_buffer(od);
    rg::PassDesc pd;
    pd.name = "MotionVectorPass"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_depth,      rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_vp,         rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{in_vp_prev,    rg::AccessMode::Read,  2});
    pd.accesses.push_back(rg::ResourceAccess{st->out_motion, rg::AccessMode::Write, 3});
    pd.record = motion_record; pd.user_ctx = st.get();
    pd.dispatch_x = (W + 7) / 8; pd.dispatch_y = (H + 7) / 8;
    pd.compute_hlsl = MotionVectorPass::hlsl_source();
    pd.push_constant_size = 16;
    pack_push(pd.push_constants, W, H, 0.0f, 0u);
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* MotionVectorPass::hlsl_source() noexcept {
    return
    "// Cardinal - MotionVectorPass.hlsl - zero-motion placeholder, exactly\n"
    "// like the CPU reference (real reprojection lands with the inverse-VP\n"
    "// wiring; the CPU record writes 0 for every pixel today, so parity = 0).\n"
    "ByteAddressBuffer   InDepth  : register(t0);  // unused (layout only)\n"
    "ByteAddressBuffer   InVP     : register(t1);  // unused (layout only)\n"
    "ByteAddressBuffer   InVPPrev : register(t2);  // unused (layout only)\n"
    "RWByteAddressBuffer OutMV    : register(u0);  // u32 snorm16x2 per pixel\n"
    "struct PushT { uint w; uint h; float p0; uint p1; };\n"
    "[[vk::push_constant]] ConstantBuffer<PushT> pc : register(b0);\n"
    "#define gW pc.w\n"
    "#define gH pc.h\n"
    "[numthreads(8,8,1)]\n"
    "void CSMain(uint3 tid : SV_DispatchThreadID){\n"
    "  if (tid.x >= gW || tid.y >= gH) return;\n"
    "  OutMV.Store((tid.y*gW + tid.x)*4u, 0u);\n"
    "}\n";
}

// ============================================================================
// TonemapPass — ACES filmic, sRGB encode
// ============================================================================
namespace {

inline float aces(float x) noexcept {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return sat((x * (a * x + b)) / (x * (c * x + d) + e));
}

inline float srgb_encode(float v) noexcept {
    const float c = sat(v);
    return (c <= 0.0031308f) ? 12.92f * c : 1.055f * cardinal::pow(c, 1.0f / 2.4f) - 0.055f;
}

void tonemap_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<TonemapPass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height;
    const float* rad = static_cast<const float*>(ec.map_buffer_read(st->in_radiance));
    cardinal::u8* rgba = static_cast<cardinal::u8*>(ec.map_buffer_write(st->out_rgba));
    if (!rad || !rgba) { ec.dispatch(0, 1, 1); return; }
    const float ex = fzf(st->exposure, 1.0f);
    cardinal::u32 clipped = 0;
    for (cardinal::u32 i = 0; i < W * H; ++i) {
        const float r = aces(fzf(rad[i * 3 + 0], 0.0f) * ex);
        const float g = aces(fzf(rad[i * 3 + 1], 0.0f) * ex);
        const float b = aces(fzf(rad[i * 3 + 2], 0.0f) * ex);
        rgba[i * 4 + 0] = to_u8(srgb_encode(r));
        rgba[i * 4 + 1] = to_u8(srgb_encode(g));
        rgba[i * 4 + 2] = to_u8(srgb_encode(b));
        rgba[i * 4 + 3] = 255;
        if (r >= 0.999f || g >= 0.999f || b >= 0.999f) ++clipped;
    }
    st->pixels_clipped = clipped;
    ec.dispatch((W + 7) / 8, (H + 7) / 8, 1);
}

}  // namespace

cardinal::shared_ptr<TonemapPass::State> TonemapPass::add_to_graph(
    rg::Graph& g, rg::ResourceHandle in_rad,
    cardinal::u32 W, cardinal::u32 H, float exposure)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_radiance = in_rad; st->width = W; st->height = H; st->exposure = exposure;
    rg::BufferDesc od; od.name = "tone.rgba"; od.size_bytes = W * H * 4;
    od.stride_bytes = 4;
    st->out_rgba = g.declare_buffer(od);
    rg::PassDesc pd;
    pd.name = "TonemapPass"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_rad,        rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{st->out_rgba,  rg::AccessMode::Write, 1});
    pd.record = tonemap_record; pd.user_ctx = st.get();
    pd.dispatch_x = (W + 7) / 8; pd.dispatch_y = (H + 7) / 8;
    // GPU path: thread the kernel + pack the b0 push block {W, H, exposure, pad}.
    pd.compute_hlsl = TonemapPass::hlsl_source();
    pd.push_constant_size = 16;
    pack_push(pd.push_constants, W, H, exposure, 0u);
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* TonemapPass::hlsl_source() noexcept {
    return
    "// Cardinal - TonemapPass.hlsl - ACES filmic + sRGB encode per pixel.\n"
    "ByteAddressBuffer   InRadiance : register(t0);   // float3 per pixel (12B)\n"
    "RWByteAddressBuffer OutRGBA    : register(u0);   // RGBA8 per pixel (4B)\n"
    "struct PushT { uint w; uint h; float exposure; uint pad; };\n"
    "[[vk::push_constant]] ConstantBuffer<PushT> pc : register(b0);\n"
    "#define gW pc.w\n"
    "#define gH pc.h\n"
    "#define gExposure pc.exposure\n"
    "float aces(float x){ float a=2.51,b=0.03,c=2.43,d=0.59,e=0.14;\n"
    "  return saturate((x*(a*x+b))/(x*(c*x+d)+e)); }\n"
    "float srgb(float c){ return c<=0.0031308 ? 12.92*c : 1.055*pow(c,1.0/2.4)-0.055; }\n"
    "[numthreads(8,8,1)]\n"
    "void CSMain(uint3 tid : SV_DispatchThreadID){\n"
    "  if (tid.x>=gW || tid.y>=gH) return;\n"
    "  uint i = tid.y*gW + tid.x;\n"
    "  float r = aces(asfloat(InRadiance.Load(i*12+0))*gExposure);\n"
    "  float g = aces(asfloat(InRadiance.Load(i*12+4))*gExposure);\n"
    "  float b = aces(asfloat(InRadiance.Load(i*12+8))*gExposure);\n"
    "  uint R=(uint)(saturate(srgb(r))*255.0+0.5);\n"
    "  uint G=(uint)(saturate(srgb(g))*255.0+0.5);\n"
    "  uint B=(uint)(saturate(srgb(b))*255.0+0.5);\n"
    "  OutRGBA.Store(i*4, R | (G<<8) | (B<<16) | (255u<<24));\n"
    "}\n";
}

// ============================================================================
// CompositePresentPass
// ============================================================================
namespace {

void composite_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<CompositePresentPass::State*>(uctx);
    const cardinal::u32 W = st->width, H = st->height;
    const cardinal::u8* scene = static_cast<const cardinal::u8*>(ec.map_buffer_read(st->in_scene));
    const cardinal::u8* ui    = st->in_ui.is_valid()
        ? static_cast<const cardinal::u8*>(ec.map_buffer_read(st->in_ui)) : nullptr;
    const cardinal::u8* giz   = st->in_gizmo.is_valid()
        ? static_cast<const cardinal::u8*>(ec.map_buffer_read(st->in_gizmo)) : nullptr;
    cardinal::u8* out = static_cast<cardinal::u8*>(ec.map_buffer_write(st->out_presentation));
    if (!scene || !out) { ec.dispatch(0, 1, 1); return; }
    cardinal::u32 over = 0;
    for (cardinal::u32 i = 0; i < W * H; ++i) {
        cardinal::u8 r = scene[i * 4 + 0];
        cardinal::u8 g_ = scene[i * 4 + 1];
        cardinal::u8 b = scene[i * 4 + 2];
        cardinal::u8 a = scene[i * 4 + 3];
        if (giz) {
            const cardinal::u8 ga = giz[i * 4 + 3];
            if (ga != 0) { r = giz[i*4+0]; g_ = giz[i*4+1]; b = giz[i*4+2]; ++over; }
        }
        if (ui) {
            const cardinal::u8 ua = ui[i * 4 + 3];
            if (ua != 0) { r = ui[i*4+0]; g_ = ui[i*4+1]; b = ui[i*4+2]; ++over; }
        }
        out[i * 4 + 0] = r; out[i * 4 + 1] = g_;
        out[i * 4 + 2] = b; out[i * 4 + 3] = a;
    }
    st->pixels_overdrawn = over;
    ec.dispatch((W + 7) / 8, (H + 7) / 8, 1);
}

}  // namespace

cardinal::shared_ptr<CompositePresentPass::State> CompositePresentPass::add_to_graph(
    rg::Graph& g, rg::ResourceHandle in_scene,
    rg::ResourceHandle in_ui, rg::ResourceHandle in_giz,
    cardinal::u32 W, cardinal::u32 H)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_scene = in_scene; st->in_ui = in_ui; st->in_gizmo = in_giz;
    st->width = W; st->height = H;
    rg::BufferDesc od; od.name = "present.rgba"; od.size_bytes = W * H * 4;
    od.stride_bytes = 4;
    st->out_presentation = g.declare_buffer(od);
    rg::PassDesc pd;
    pd.name = "CompositePresentPass"; pd.kind = rg::PassKind::Compute;
    // FIXED SRV LAYOUT (the long-standing "special SRV-layout issue"): the UI
    // and gizmo accesses are ALWAYS declared so the kernel's t0/t1/t2 layout
    // never shifts with overlay availability. Missing overlays bind 4-byte
    // dummies; push flags tell the kernel which are live. The CPU record
    // still gates on the ORIGINAL handles in State.
    rg::ResourceHandle ui_bind  = in_ui;
    rg::ResourceHandle giz_bind = in_giz;
    if (!ui_bind.is_valid())
        ui_bind  = g.declare_buffer(rg::BufferDesc{"present.noui",  4, 4, true});
    if (!giz_bind.is_valid())
        giz_bind = g.declare_buffer(rg::BufferDesc{"present.nogiz", 4, 4, true});
    pd.accesses.push_back(rg::ResourceAccess{in_scene, rg::AccessMode::Read, 0});
    pd.accesses.push_back(rg::ResourceAccess{ui_bind,  rg::AccessMode::Read, 1});
    pd.accesses.push_back(rg::ResourceAccess{giz_bind, rg::AccessMode::Read, 2});
    pd.accesses.push_back(rg::ResourceAccess{st->out_presentation, rg::AccessMode::Write, 3});
    pd.record = composite_record; pd.user_ctx = st.get();
    pd.dispatch_x = (W + 7) / 8; pd.dispatch_y = (H + 7) / 8;
    pd.compute_hlsl = CompositePresentPass::hlsl_source();
    pd.push_constant_size = 16;
    pack_push(pd.push_constants, W, H, 0.0f,
              (in_ui.is_valid() ? 1u : 0u) | (in_giz.is_valid() ? 2u : 0u));
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* CompositePresentPass::hlsl_source() noexcept {
    return
    "// Cardinal - CompositePresentPass.hlsl - alpha-TEST composite (a != 0),\n"
    "// gizmo first then UI (UI wins), output alpha = scene alpha. Mirrors\n"
    "// composite_record byte-for-byte on RGBA8 words.\n"
    "ByteAddressBuffer   InScene : register(t0);\n"
    "ByteAddressBuffer   InUi    : register(t1);   // dummy when gFlags bit0 = 0\n"
    "ByteAddressBuffer   InGiz   : register(t2);   // dummy when gFlags bit1 = 0\n"
    "RWByteAddressBuffer OutP    : register(u0);\n"
    "struct PushT { uint w; uint h; float p0; uint flags; };\n"
    "[[vk::push_constant]] ConstantBuffer<PushT> pc : register(b0);\n"
    "#define gW pc.w\n"
    "#define gH pc.h\n"
    "#define gFlags pc.flags\n"
    "[numthreads(8,8,1)]\n"
    "void CSMain(uint3 tid : SV_DispatchThreadID){\n"
    "  if (tid.x >= gW || tid.y >= gH) return;\n"
    "  uint i = tid.y*gW + tid.x;\n"
    "  uint s = InScene.Load(i*4u);\n"
    "  uint r = s & 0xFFu, g = (s >> 8) & 0xFFu, b = (s >> 16) & 0xFFu;\n"
    "  uint a = (s >> 24) & 0xFFu;\n"
    "  if ((gFlags & 2u) != 0u) {\n"
    "    uint z = InGiz.Load(i*4u);\n"
    "    if ((z >> 24) != 0u) { r = z & 0xFFu; g = (z >> 8) & 0xFFu; b = (z >> 16) & 0xFFu; }\n"
    "  }\n"
    "  if ((gFlags & 1u) != 0u) {\n"
    "    uint u = InUi.Load(i*4u);\n"
    "    if ((u >> 24) != 0u) { r = u & 0xFFu; g = (u >> 8) & 0xFFu; b = (u >> 16) & 0xFFu; }\n"
    "  }\n"
    "  OutP.Store(i*4u, r | (g << 8) | (b << 16) | (a << 24));\n"
    "}\n";
}

// ============================================================================
// AegisPipeline orchestrator
// ============================================================================
cardinal::shared_ptr<AegisPipeline> AegisPipeline::create(AegisConfig cfg) {
    auto p = cardinal::shared_ptr<AegisPipeline>(new AegisPipeline());
    p->cfg_ = cfg;
    return p;
}

void AegisPipeline::build(rg::Graph& g, const AegisSceneInputs& in,
                          AegisOutputs& out, AegisStageRefs& s)
{
    const cardinal::u32 W = cfg_.width;
    const cardinal::u32 H = cfg_.height;
    const cardinal::u32 M = in.triangle_count;

    // Block 3 — Virtual Geometry 2.0
    s.classify     = GeometryClassifyPass::add_to_graph(g, in.tris, in.curvature, M);
    s.meshlets     = MeshletBuildPass::add_to_graph(g, in.tris, M);
    s.meshlet_cull = MeshletCullPass::add_to_graph(g, s.meshlets->out_meshlets,
                                                   in.view_proj, in.camera_dir,
                                                   s.meshlets->meshlet_count);
    s.sse          = ScreenSpaceErrorPass::add_to_graph(g, s.meshlets->out_meshlets,
                                                        in.view_proj,
                                                        s.meshlets->meshlet_count, H);

    // Block 4 — Visibility & Culling + Indirect generation
    s.hiz          = HiZBuildPass::add_to_graph(g, /*depth fed from V-Buf later*/
                                                /*placeholder zero-depth*/
                                                g.declare_buffer(graph::BufferDesc{
                                                    "hiz.placeholder",
                                                    static_cast<cardinal::usize>(W) * H * sizeof(float),
                                                    sizeof(float), true}),
                                                W, H);
    s.indirect_gen = DrawIndirectGenPass::add_to_graph(g, s.meshlet_cull->out_visibility,
                                                       s.meshlets->out_meshlets,
                                                       s.meshlets->meshlet_count);

    // Block 5 — Geometry pipeline. Adaptive math-division picks the tier.
    s.adaptive = AdaptiveGeometryPass::add_to_graph(g, in.tris, M, cfg_.caps, cfg_.max_tier);
    selected_tier_ = s.adaptive->selected_tier;

    // Block 6 — V-Buffer
    s.vbuf = VisibilityBufferPass::add_to_graph(g, in.tris, in.material_ids, in.view_proj,
                                                {}, W, H, M);
    out.vbuf_depth  = s.vbuf->out_depth;
    out.vbuf_prim   = s.vbuf->out_prim_id;
    out.vbuf_mat    = s.vbuf->out_mat_id;
    out.vbuf_normal = s.vbuf->out_normal;
    out.vbuf_motion = s.vbuf->out_motion;

    // Block 7 — Tiled light cull
    s.light_cull = TiledLightCullPass::add_to_graph(g, in.lights, s.vbuf->out_depth,
                                                    in.view_proj, W, H, cfg_.light_count);

    // Block 7 — ReSTIR DI: Sample → SpatialReuse → TemporalReuse.
    // Only wired when the host supplies the reconstructed world-pos +
    // world-normal + seed buffers. Without these, the orchestrator
    // falls back to plain tile-light-list shading in VBufResolvePass.
    const bool restir_active =
        in.restir_world_pos.is_valid() &&
        in.restir_world_normal.is_valid() &&
        in.restir_seeds.is_valid();
    if (restir_active) {
        s.restir_sample = ReSTIRSamplePass::add_to_graph(
            g, in.restir_world_pos, in.restir_world_normal,
            in.lights, in.restir_seeds, W, H, cfg_.light_count);
        s.restir_spatial = ReSTIRSpatialPass::add_to_graph(
            g, s.restir_sample->out_reservoirs,
            in.restir_world_pos, in.restir_world_normal,
            in.restir_seeds, W, H);
        s.restir_temporal = ReSTIRTemporalPass::add_to_graph(
            g, s.restir_spatial->out_reservoirs,
            in.restir_prev_reservoirs,        // optional handle — pass {} for first frame
            s.vbuf->out_motion,
            in.restir_world_normal,
            in.restir_prev_world_normal,
            W, H);
    }

    // V-Buf resolve consumes the tile light list (and optionally the
    // ReSTIR reservoir buffer when it's wired). Resolve's interface
    // doesn't take the reservoir handle yet — that wiring lands when
    // the resolver gets a "use ReSTIR for direct" knob; for now the
    // ReSTIR passes run as a parallel chain and their results are
    // available to the host via stages.restir_temporal->out_reservoirs.
    s.resolve = VBufResolvePass::add_to_graph(g, s.vbuf->out_depth, s.vbuf->out_prim_id,
                                              s.vbuf->out_mat_id, s.vbuf->out_normal,
                                              in.materials,
                                              s.light_cull->out_tile_lights,
                                              s.light_cull->out_tile_counts,
                                              in.lights, in.ambient,
                                              W, H, cfg_.material_count, cfg_.light_count);
    out.radiance_hdr = s.resolve->out_radiance;

    // Block 12 — Post: motion vectors + tonemap
    if (in.view_proj_prev.is_valid()) {
        s.motion = MotionVectorPass::add_to_graph(g, s.vbuf->out_depth,
                                                  in.view_proj, in.view_proj_prev, W, H);
    }
    s.tonemap = TonemapPass::add_to_graph(g, s.resolve->out_radiance, W, H, cfg_.exposure);
    out.tonemapped = s.tonemap->out_rgba;

    // Block 13 — Composite + present
    s.composite = CompositePresentPass::add_to_graph(g, s.tonemap->out_rgba,
                                                     in.ui_overlay, in.gizmo_overlay, W, H);
    out.presentation = s.composite->out_presentation;
}

}  // namespace cardinal::render::gpu
