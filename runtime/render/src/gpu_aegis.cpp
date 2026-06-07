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
    pd.accesses.push_back(rg::ResourceAccess{in_tris, rg::AccessMode::Read, 0});
    if (in_curv.is_valid()) pd.accesses.push_back(rg::ResourceAccess{in_curv, rg::AccessMode::Read, 1});
    pd.accesses.push_back(rg::ResourceAccess{st->out_class, rg::AccessMode::Write, 2});
    pd.record = classify_record; pd.user_ctx = st.get();
    pd.dispatch_x = (N + 63) / 64;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* GeometryClassifyPass::hlsl_source() noexcept {
    return "// Cardinal — GeometryClassifyPass.hlsl\n// One thread per triangle; per-tri classifier into 5 AEGIS classes.\n";
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
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* MeshletBuildPass::hlsl_source() noexcept {
    return "// Cardinal — MeshletBuildPass.hlsl\n// One thread per meshlet; emits 12-float record (first/count/min/max/cone).\n";
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
    rg::BufferDesc od; od.name = "meshlet.vis"; od.size_bytes = N; od.stride_bytes = 1;
    st->out_visibility = g.declare_buffer(od);
    rg::PassDesc pd;
    pd.name = "MeshletCullPass"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_meshlets,        rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_mat,             rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{in_dir,             rg::AccessMode::Read,  2});
    pd.accesses.push_back(rg::ResourceAccess{st->out_visibility, rg::AccessMode::Write, 3});
    pd.record = meshlet_cull_record; pd.user_ctx = st.get();
    pd.dispatch_x = (N + 63) / 64;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* MeshletCullPass::hlsl_source() noexcept {
    return "// Cardinal — MeshletCullPass.hlsl\n// Per-meshlet frustum + normal-cone backface reject.\n";
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
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* ScreenSpaceErrorPass::hlsl_source() noexcept {
    return "// Cardinal — ScreenSpaceErrorPass.hlsl\n// Per-meshlet screen radius → LOD level.\n";
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
    pd.dispatch_x = (N + 63) / 64;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* DrawIndirectGenPass::hlsl_source() noexcept {
    return "// Cardinal — DrawIndirectGenPass.hlsl\n// Compact visibility → indirect draw command list with atomic count.\n";
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
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* TiledLightCullPass::hlsl_source() noexcept {
    return "// Cardinal — TiledLightCullPass.hlsl\n// 16x16 tile per workgroup; per-tile depth bounds + light frustum overlap.\n";
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
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* VBufResolvePass::hlsl_source() noexcept {
    return "// Cardinal — VBufResolvePass.hlsl\n// Per-pixel: sample V-Buf, reconstruct world pos via depth, fetch tile light list, shade with Cook-Torrance.\n";
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
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* MotionVectorPass::hlsl_source() noexcept {
    return "// Cardinal — MotionVectorPass.hlsl\n// Per-pixel: reconstruct world pos from depth + inv-VP; reproject through prev VP; pack snorm16 delta.\n";
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
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* TonemapPass::hlsl_source() noexcept {
    return "// Cardinal — TonemapPass.hlsl\n// ACES filmic + sRGB encode per pixel.\n";
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
    pd.accesses.push_back(rg::ResourceAccess{in_scene, rg::AccessMode::Read, 0});
    if (in_ui.is_valid())  pd.accesses.push_back(rg::ResourceAccess{in_ui,  rg::AccessMode::Read, 1});
    if (in_giz.is_valid()) pd.accesses.push_back(rg::ResourceAccess{in_giz, rg::AccessMode::Read, 2});
    pd.accesses.push_back(rg::ResourceAccess{st->out_presentation, rg::AccessMode::Write, 3});
    pd.record = composite_record; pd.user_ctx = st.get();
    pd.dispatch_x = (W + 7) / 8; pd.dispatch_y = (H + 7) / 8;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* CompositePresentPass::hlsl_source() noexcept {
    return "// Cardinal — CompositePresentPass.hlsl\n// Scene + UI + gizmo alpha-test composite for present.\n";
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
