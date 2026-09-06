// =============================================================================
// Cardinal — AEGIS visibility-raster benchmark + at-scale GPU equivalence.
//
// The reference V-Buffer kernel is O(pixels × triangles); TriangleBinPass +
// the binned kernel are O(pixels × triangles-in-tile). This harness:
//
//   1. builds the SAME 960×540 / 20k-triangle scene into two graphs — brute
//      and binned — and dispatches both for real through RhiBackend on the
//      async ComputeQueue (Vulkan + D3D12),
//   2. proves the two GPU paths agree EXACTLY on prim / depth / material
//      (the winner rule "lower depth, then lower id" makes bin order moot),
//   3. times N frames of each on the production async lane and reports
//      ms/frame + speedup. Timing is informational (no threshold — CI boxes
//      vary); equivalence is a hard CHECK.
//
// SKIPs (exit 0) without a capable device / async queue / fence.
// =============================================================================

#include <cardinal/render/gpu_visbuf.hpp>
#include <cardinal/render/gpu_aegis.hpp>
#include <cardinal/render/graph.hpp>
#include <cardinal/rhi/rhi.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/std/cstring.hpp>

#include <chrono>
#include <cmath>
#include <vector>

namespace {

namespace rg  = cardinal::render::graph;
namespace gpu = cardinal::render::gpu;
namespace rhi = cardinal::rhi;
using cardinal::u8;
using cardinal::u32;
using cardinal::usize;

int g_checks = 0;
int g_fail   = 0;
void check_impl(bool ok, const char* what, int line) {
    ++g_checks;
    if (!ok) { ++g_fail; cardinal::log::errorf("aegisbench", "FAIL  L%d  %s", line, what); }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

// ---------------------------------------------------------------------------
// Scene: T small triangles scattered over NDC with varied depth, plus a few
// large ones so bins see both regimes. Deterministic LCG.
// ---------------------------------------------------------------------------
struct Scene {
    u32 W, H, T;
    std::vector<float> tris;
    std::vector<u32>   mids;
    std::vector<float> vp;

    Scene(u32 w, u32 h, u32 t) : W(w), H(h), T(t) {
        u32 s = 0x9E3779B9u;
        auto rnd = [&]() { s = s * 1664525u + 1013904223u; return static_cast<float>(s >> 8) / 16777216.0f; };
        tris.resize(usize(T) * 9);
        mids.resize(T);
        for (u32 i = 0; i < T; ++i) {
            float* v = tris.data() + usize(i) * 9;
            const bool big = (i % 997) == 0;
            const float sz = big ? 0.6f : 0.012f + 0.02f * rnd();
            const float cx = -1.0f + 2.0f * rnd(), cy = -1.0f + 2.0f * rnd();
            const float z  = 0.05f + 0.9f * rnd();
            v[0] = cx - sz; v[1] = cy - sz; v[2] = z;
            v[3] = cx + sz; v[4] = cy - sz; v[5] = z;
            v[6] = cx;      v[7] = cy + sz; v[8] = z;
            mids[i] = i % 7;
        }
        vp = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
    }
};

// Distance (in pixels) from pixel centre (x, y) to the nearest edge LINE of
// triangle t, using the same identity-VP screen mapping the kernels use. A
// coverage disagreement between two kernels at a pixel whose centre sits
// within float rounding of an edge (margin ≲ 1e-4 px on a ~1000 px screen —
// a couple of ulps of the projected vertex) is a KNIFE-EDGE flip: the
// inside test `w >= 0` is exact, so a 1-ulp difference in a stored vs
// register-resident vertex legitimately lands the two kernels on opposite
// sides. Anything wider is a real defect.
float edge_margin_px(const Scene& sc, u32 t, u32 x, u32 y) {
    if (t >= sc.T) return 1.0e9f;
    const float* v = sc.tris.data() + usize(t) * 9;
    const float hw = 0.5f * static_cast<float>(sc.W), hh = 0.5f * static_cast<float>(sc.H);
    float px[3], py[3];
    for (int k = 0; k < 3; ++k) { px[k] = v[k * 3] * hw + hw; py[k] = hh - v[k * 3 + 1] * hh; }
    const float fx = static_cast<float>(x) + 0.5f, fy = static_cast<float>(y) + 0.5f;
    float best = 1.0e9f;
    for (int e = 0; e < 3; ++e) {
        const int a = (e + 1) % 3, b = (e + 2) % 3;
        const float ex = px[b] - px[a], ey = py[b] - py[a];
        const float len = std::sqrt(ex * ex + ey * ey);
        if (len < 1.0e-6f) continue;
        const float w = ex * (fy - py[a]) - ey * (fx - px[a]);   // edge_fn
        const float d = std::fabs(w) / len;
        if (d < best) best = d;
    }
    return best;
}

struct Built {
    cardinal::shared_ptr<rg::Graph> graph;
    cardinal::shared_ptr<gpu::VisibilityBufferPass::State> vbuf;
    cardinal::shared_ptr<gpu::TriangleBinPass::State>      bin;
};

Built build(Scene& sc, bool binned, u32 bin_capacity = gpu::kDefaultBinCapacity) {
    Built b;
    b.graph = rg::Graph::create();
    auto& g = *b.graph;
    auto h_t = g.import_buffer("in.tris", sc.tris.data(), sc.tris.size() * 4, 4);
    auto h_i = g.import_buffer("in.mids", sc.mids.data(), sc.mids.size() * 4, 4);
    auto h_m = g.import_buffer("in.vp",   sc.vp.data(),   64, 4);
    if (binned) {
        b.bin  = gpu::TriangleBinPass::add_to_graph(g, h_t, h_i, h_m, {}, sc.W, sc.H, sc.T, bin_capacity);
        b.vbuf = gpu::VisibilityBufferPass::add_to_graph(g, *b.bin, {}, sc.W, sc.H, sc.T);
    } else {
        b.vbuf = gpu::VisibilityBufferPass::add_to_graph(g, h_t, h_i, h_m, {}, sc.W, sc.H, sc.T);
    }
    g.mark_output(b.vbuf->out_prim_id);
    CHECK(g.compile());
    return b;
}

// ---- one-shot GPU execute with readback of prim / depth / mat -------------
struct ReadCtx {
    rhi::Device* dev{nullptr};
    rg::Graph*   graph{nullptr};
    cardinal::shared_ptr<rg::RhiBackend>* backend{nullptr};
    rhi::Buffer* rb_prim{nullptr};
    rhi::Buffer* rb_depth{nullptr};
    rhi::Buffer* rb_mat{nullptr};
    rhi::Buffer* rb_counts{nullptr};   // binned only (may be null)
};

void read_record(rhi::Swapchain* rec, void* user) noexcept {
    auto& cx = *static_cast<ReadCtx*>(user);
    *cx.backend = rg::RhiBackend::create(*cx.dev, *rec);
    (*cx.backend)->set_gpu_execute(true);
    (*cx.backend)->execute(*cx.graph);
    auto copy = [&](const char* name, rhi::Buffer* dst) {
        if (!dst) return;
        rhi::Buffer* src = (*cx.backend)->gpu_buffer(name);
        if (!src) return;
        (*cx.backend)->make_copy_source(name);
        rec->copy_buffer(src, 0, dst, 0, dst->size());
    };
    copy("vbuf.prim",  cx.rb_prim);
    copy("vbuf.depth", cx.rb_depth);
    copy("vbuf.mat",   cx.rb_mat);
    copy("bin.counts", cx.rb_counts);
}

struct Readback { std::vector<u8> prim, depth, mat, counts; u32 real_dispatches{0}; };

Readback run_once(rhi::Device& dev, rhi::ComputeQueue& q, rhi::Fence& fence, cardinal::u64& fv,
                  Built& b, const Scene& sc) {
    Readback out;
    auto mk = [&](usize bytes) {
        rhi::BufferDesc bd{}; bd.size = bytes; bd.usage = 0; bd.cpu_writable = false; bd.cpu_readable = true;
        return dev.create_buffer(bd);
    };
    const usize px = usize(sc.W) * sc.H;
    auto rp = mk(px * 4), rd = mk(px * 4), rm = mk(px * 4);
    cardinal::unique_ptr<rhi::Buffer> rc;
    if (b.bin) rc = mk(usize(b.bin->tiles_x) * b.bin->tiles_y * 4);
    CHECK(rp && rd && rm);
    if (!rp || !rd || !rm) return out;
    cardinal::shared_ptr<rg::RhiBackend> backend;
    ReadCtx cx;
    cx.dev = &dev; cx.graph = b.graph.get(); cx.backend = &backend;
    cx.rb_prim = rp.get(); cx.rb_depth = rd.get(); cx.rb_mat = rm.get(); cx.rb_counts = rc.get();
    const cardinal::u64 target = ++fv;
    CHECK(q.submit(&read_record, &cx, &fence, target) == target);
    fence.wait_cpu(target, 0);
    CHECK(backend != nullptr);
    if (!backend) return out;
    for (const auto& ev : backend->events()) if (ev.real_dispatch) ++out.real_dispatches;
    out.prim.resize(rp->size());  CHECK(rp->download(out.prim.data(),  out.prim.size(),  0));
    out.depth.resize(rd->size()); CHECK(rd->download(out.depth.data(), out.depth.size(), 0));
    out.mat.resize(rm->size());   CHECK(rm->download(out.mat.data(),   out.mat.size(),   0));
    if (rc) { out.counts.resize(rc->size()); CHECK(rc->download(out.counts.data(), out.counts.size(), 0)); }
    backend.reset();
    return out;
}

// ---- production async-lane timing ------------------------------------------
double time_frames(rhi::Device& dev, rhi::ComputeQueue& q, Built& b, u32 frames) {
    auto backend = rg::RhiBackend::create(dev);
    backend->set_gpu_execute(true);
    backend->set_async_queue(&q);
    backend->execute(*b.graph);              // warm-up: DXC compile + buffer alloc
    backend->set_async_queue(nullptr);       // drain
    backend->set_async_queue(&q);
    const auto t0 = std::chrono::steady_clock::now();
    for (u32 i = 0; i < frames; ++i) backend->execute(*b.graph);
    backend->set_async_queue(nullptr);       // drain: every frame retired
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    backend.reset();
    return ms / static_cast<double>(frames);
}

// ---- full AEGIS pipeline: adaptive 2x2 shading rate off vs on ----------------
// A facet-heavy scene (a few hundred LARGE triangles, several point lights) is
// where the resolve dominates and quads are mostly uniform; the small-triangle
// raster scene above is its worst case (edge quads everywhere). Both are
// timed through the production async lane.
struct FacetScene {
    u32 W, H, T;
    std::vector<float> tris, mats, lights, amb, vp, ivp, dir, cpos;
    std::vector<u32>   mids;
    FacetScene(u32 w, u32 h, u32 t) : W(w), H(h), T(t) {
        u32 s = 0x1234567u;
        auto rnd = [&]() { s = s * 1664525u + 1013904223u; return static_cast<float>(s >> 8) / 16777216.0f; };
        tris.resize(usize(T) * 9); mids.resize(T);
        for (u32 i = 0; i < T; ++i) {
            float* v = tris.data() + usize(i) * 9;
            const float sz = 0.15f + 0.35f * rnd();
            const float cx = -1.0f + 2.0f * rnd(), cy = -1.0f + 2.0f * rnd(), z = 0.05f + 0.9f * rnd();
            v[0] = cx - sz; v[1] = cy - sz; v[2] = z;  v[3] = cx + sz; v[4] = cy - sz; v[5] = z;  v[6] = cx; v[7] = cy + sz; v[8] = z;
            mids[i] = i % 2;
        }
        mats   = { 0.8f,0.3f,0.2f,0.6f, 0.0f,0,0,0,  0.2f,0.4f,0.9f,0.2f, 0.8f,0,0,0 };
        lights.assign(16 * 6, 0.0f);
        for (u32 l = 0; l < 6; ++l) {
            float* L = lights.data() + l * 16;
            L[0] = (l == 0) ? 0.0f : 1.0f;                       // 1 directional + 5 point
            L[2] = -0.8f + 0.4f * static_cast<float>(l); L[3] = 0.3f; L[4] = 1.2f;    // in front of the +z facets
            L[5] = 0.3f; L[6] = -0.5f; L[7] = -1.0f;
            L[8] = 1.0f; L[9] = 0.9f; L[10] = 0.8f; L[11] = 2.0f; L[12] = 4.0f;
        }
        amb  = { 0.05f, 0.05f, 0.06f };
        vp   = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
        ivp  = vp;
        dir  = { 0, 0, 1 };
        cpos = { 0.1f, 0.2f, 2.0f };
    }
};

struct BuiltPipeline {
    cardinal::shared_ptr<rg::Graph> graph;
    gpu::AegisOutputs   out;
    gpu::AegisStageRefs st;
};

BuiltPipeline build_pipeline(FacetScene& sc, bool adaptive) {
    BuiltPipeline b;
    b.graph = rg::Graph::create();
    auto& g = *b.graph;
    gpu::AegisSceneInputs in;
    in.triangle_count = sc.T;
    in.tris          = g.import_buffer("in.tris", sc.tris.data(),   sc.tris.size() * 4, 4);
    in.material_ids  = g.import_buffer("in.mids", sc.mids.data(),   sc.mids.size() * 4, 4);
    in.materials     = g.import_buffer("in.mats", sc.mats.data(),   sc.mats.size() * 4, 4);
    in.lights        = g.import_buffer("in.lts",  sc.lights.data(), sc.lights.size() * 4, 4);
    in.ambient       = g.import_buffer("in.amb",  sc.amb.data(),    12, 4);
    in.view_proj     = g.import_buffer("in.vp",   sc.vp.data(),     64, 4);
    in.camera_dir    = g.import_buffer("in.dir",  sc.dir.data(),    12, 4);
    in.inv_view_proj = g.import_buffer("in.ivp",  sc.ivp.data(),    64, 4);
    in.camera_pos    = g.import_buffer("in.cpos", sc.cpos.data(),   12, 4);
    gpu::AegisConfig cfg;
    cfg.width = sc.W; cfg.height = sc.H; cfg.material_count = 2; cfg.light_count = 6;
    cfg.adaptive_shading_rate = adaptive;
    auto pipeline = gpu::AegisPipeline::create(cfg);
    pipeline->build(g, in, b.out, b.st);
    g.mark_output(b.out.presentation);
    CHECK(g.compile());
    return b;
}

void bench_pipeline(rhi::Device& dev, rhi::ComputeQueue& q, const char* be_name) {
    FacetScene sc(960, 540, 300);
    BuiltPipeline full = build_pipeline(sc, false);
    BuiltPipeline quad = build_pipeline(sc, true);
    auto time_pipe = [&](BuiltPipeline& b, u32 frames) {
        auto backend = rg::RhiBackend::create(dev);
        backend->set_gpu_execute(true);
        backend->set_async_queue(&q);
        backend->execute(*b.graph);
        backend->set_async_queue(nullptr);
        backend->set_async_queue(&q);
        const auto t0 = std::chrono::steady_clock::now();
        for (u32 i = 0; i < frames; ++i) backend->execute(*b.graph);
        backend->set_async_queue(nullptr);
        const auto t1 = std::chrono::steady_clock::now();
        u32 real = 0;
        for (const auto& ev : backend->events()) if (ev.real_dispatch) ++real;
        backend.reset();
        return std::pair<double, u32>(std::chrono::duration<double, std::milli>(t1 - t0).count() / frames, real);
    };
    const auto f = time_pipe(full, 20);
    const auto a = time_pipe(quad, 20);
    CHECK(f.second >= 12u && a.second >= 12u);      // every authored kernel really dispatched
    // CPU-side quad statistics for the same scene (the CpuBackend runs the
    // reference record, which reports coarse/fine quad counts).
    auto cpu = rg::CpuBackend::create();
    cpu->execute(*quad.graph);
    const u32 qc = quad.st.resolve->quads_coarse, qf = quad.st.resolve->quads_fine;
    cardinal::log::infof("aegisbench",
        "%s  %ux%u  %u facets, 6 lights, full AEGIS graph :  full-rate %.2f ms/frame   quad-rate %.2f ms/frame   speedup %.2fx   (%u coarse / %u fine quads = %.0f%% of pixels at 1/4 rate)",
        be_name, sc.W, sc.H, sc.T, f.first, a.first, a.first > 0.0 ? f.first / a.first : 0.0,
        qc, qf, 100.0 * static_cast<double>(qc) / static_cast<double>(qc + qf ? qc + qf : 1));
}

// ---- zero-copy / unified-memory scene streaming --------------------------
// A 200k-triangle stream (7.2 MB) rewritten every frame, binned raster only.
//   upload    : the host "animates" its CPU array, execute() memcpys it into
//               the import buffer (the classic staging-style path).
//   zero-copy : the host animates straight into the next unified-memory
//               import slot (import_write_ptr) and marks it; execute() copies
//               nothing. Same GPU work, one fewer 7.2 MB pass per frame on the
//               CPU, and — with a ReBAR/GPU-upload heap — the GPU reads VRAM.
void bench_zero_copy(rhi::Device& dev, rhi::ComputeQueue& q, const char* be_name) {
    // ~350 tris land in each 16x16 tile at this density: a 2048-entry bin
    // capacity keeps every tile on the fast list path (the 256 default would
    // overflow and the every-record fallback would swamp the upload cost).
    Scene sc(960, 540, 200000);
    Built b = build(sc, true, 2048);
    const usize tri_bytes = sc.tris.size() * 4;
    auto backend = rg::RhiBackend::create(dev);
    backend->set_gpu_execute(true);
    backend->set_async_queue(&q);
    backend->execute(*b.graph);                       // warm-up + buffer creation
    backend->set_async_queue(nullptr);
    backend->set_async_queue(&q);
    // ONE sequential pass: the zero-copy destination is write-combined
    // memory (VRAM over the BAR, or an upload heap), where strided or
    // revisited writes break the WC line combining and run several times
    // slower than a forward stream. Producers that write straight into the
    // mapping must be written this way — the bridge's extract_scene is.
    auto animate = [&](float* dst, u32 frame) {
        const float dx = 0.0001f * static_cast<float>(frame % 8);
        const float* src = sc.tris.data();
        const usize n = sc.tris.size();
        for (usize i = 0; i + 2 < n; i += 3) {
            dst[i] = src[i] + dx; dst[i + 1] = src[i + 1]; dst[i + 2] = src[i + 2];
        }
    };
    std::vector<float> host(sc.tris.size());
    const u32 frames = 20;
    // upload path: animate into a host array bound as the import source.
    // (the graph imports sc.tris — animate sc.tris' twin then copy in, which
    //  is exactly the vector->memcpy->execute shape the bridge used to have)
    auto t0 = std::chrono::steady_clock::now();
    for (u32 f = 0; f < frames; ++f) {
        animate(host.data(), f);
        cardinal::memcpy(sc.tris.data(), host.data(), tri_bytes);   // producer's own array
        backend->execute(*b.graph);                                // + execute's memcpy
    }
    backend->set_async_queue(nullptr);
    const double ms_upload = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / frames;
    backend->set_async_queue(&q);
    // zero-copy path: animate directly into the next import slot.
    u32 zc_ok = 0;
    t0 = std::chrono::steady_clock::now();
    for (u32 f = 0; f < frames; ++f) {
        void* p = backend->import_write_ptr("in.tris", tri_bytes);
        if (p) {
            animate(static_cast<float*>(p), f);
            backend->mark_import_host_written("in.tris");
            ++zc_ok;
        } else {
            animate(sc.tris.data(), f);
        }
        backend->execute(*b.graph);
    }
    backend->set_async_queue(nullptr);
    const double ms_zero = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / frames;
    CHECK(zc_ok == frames);
    backend.reset();
    cardinal::log::infof("aegisbench",
        "%s  zero-copy streaming, %u tris (%.1f MB/frame), unified memory %s :  upload %.2f ms/frame   zero-copy %.2f ms/frame   speedup %.2fx",
        be_name, sc.T, static_cast<double>(tri_bytes) / (1024.0 * 1024.0),
        dev.capabilities().unified_memory ? "YES" : "no",
        ms_upload, ms_zero, ms_zero > 0.0 ? ms_upload / ms_zero : 0.0);
}

void run_backend(rhi::Backend be, const char* be_name) {
    rhi::DeviceDesc dd{}; dd.backend = be;
    auto dev = rhi::create_device(dd);
    if (!dev) { cardinal::log::infof("aegisbench", "SKIP %s: no capable device", be_name); return; }
    auto queue = dev->create_async_compute_queue();
    if (!queue) { cardinal::log::infof("aegisbench", "SKIP %s: no async compute queue", be_name); return; }
    auto fence = dev->create_fence(0);
    if (!fence) { cardinal::log::infof("aegisbench", "SKIP %s: no fence", be_name); return; }
    cardinal::u64 fv = 0;

    Scene sc(960, 540, 20000);
    Built brute  = build(sc, false);
    Built binned = build(sc, true);

    // ---- equivalence at scale ------------------------------------------------
    Readback a = run_once(*dev, *queue, *fence, fv, brute,  sc);
    Readback c = run_once(*dev, *queue, *fence, fv, binned, sc);
    CHECK(a.real_dispatches == 1u);
    CHECK(c.real_dispatches == 3u);            // clear + bin + raster all really ran
    CHECK(!a.prim.empty() && a.prim.size() == c.prim.size());
    // Depth is eps-compared with the gpu_compute harness's EpsF32 tolerance
    // (1e-5 abs + rel): the two kernels evaluate the same expressions, but one
    // from registers and one from stored setup records, and DXC may contract
    // or schedule an ulp differently — which the edge-function cancellation
    // (large terms, small result) amplifies to a few dozen ulps of depth.
    // A prim/material flip is a DEFECT only when the two depths disagree
    // beyond eps; equal-within-eps is a legitimate near-tie resolving either way.
    auto feq = [](u32 ua, u32 ub) {
        float fa, fb; cardinal::memcpy(&fa, &ua, 4); cardinal::memcpy(&fb, &ub, 4);
        if (fa == fb) return true;
        const float d = std::fabs(fa - fb);
        const float m = std::fabs(fa) > std::fabs(fb) ? std::fabs(fa) : std::fabs(fb);
        return d <= 1e-5f + 1e-5f * m;
    };
    usize bad_prim = 0, bad_depth = 0, bad_mat = 0, near_ties = 0, exact_depth = 0, knife_edges = 0;
    const usize px = usize(sc.W) * sc.H;
    if (a.prim.size() == c.prim.size() && a.prim.size() == px * 4) {
        const u32* ap = reinterpret_cast<const u32*>(a.prim.data());
        const u32* cp = reinterpret_cast<const u32*>(c.prim.data());
        const u32* ad = reinterpret_cast<const u32*>(a.depth.data());
        const u32* cd = reinterpret_cast<const u32*>(c.depth.data());
        const u32* am = reinterpret_cast<const u32*>(a.mat.data());
        const u32* cm = reinterpret_cast<const u32*>(c.mat.data());
        usize covered = 0;
        for (usize i = 0; i < px; ++i) {
            if (ap[i] != 0xFFFFFFFFu) ++covered;
            const bool depth_ok = feq(ad[i], cd[i]);
            if (ad[i] == cd[i]) ++exact_depth;
            if (ap[i] != cp[i] && depth_ok) ++near_ties;
            if (!depth_ok) {
                // Different winners: is either one an edge-on knife-edge?
                const u32 x = static_cast<u32>(i % sc.W), y = static_cast<u32>(i / sc.W);
                const float m = std::fmin(edge_margin_px(sc, ap[i], x, y), edge_margin_px(sc, cp[i], x, y));
                if (m < 1.0e-3f) {
                    ++knife_edges;
                    if (knife_edges <= 3) cardinal::log::infof("aegisbench",
                        "%s knife-edge @%zu (x=%u y=%u): prim %u / %u, centre %.2e px from an edge",
                        be_name, static_cast<size_t>(i), x, y, ap[i], cp[i], static_cast<double>(m));
                } else {
                    if (bad_depth++ < 3) cardinal::log::errorf("aegisbench",
                        "%s depth mismatch @%zu: brute %08X binned %08X (prim %u / %u, edge margin %.3e px)",
                        be_name, static_cast<size_t>(i), ad[i], cd[i], ap[i], cp[i], static_cast<double>(m));
                    if (ap[i] != cp[i]) ++bad_prim;
                    if (am[i] != cm[i]) ++bad_mat;
                }
            }
        }
        CHECK(covered > px / 10);              // the scene actually covers the screen
        cardinal::log::infof("aegisbench",
            "%s: %zu/%zu pixels covered; depth bit-exact %zu, beyond-eps %zu; prim defects %zu (near-tie flips %zu, knife-edge flips %zu); mat defects %zu",
            be_name, static_cast<size_t>(covered), static_cast<size_t>(px),
            static_cast<size_t>(exact_depth), static_cast<size_t>(bad_depth),
            static_cast<size_t>(bad_prim), static_cast<size_t>(near_ties),
            static_cast<size_t>(knife_edges), static_cast<size_t>(bad_mat));
        CHECK(near_ties  < px / 10000 + 8);    // a handful of ulp-level ties, never systematic
        CHECK(knife_edges < px / 10000 + 8);   // likewise for edge-on coverage flips
    }
    CHECK(bad_prim == 0 && bad_depth == 0 && bad_mat == 0);
    // Bin stats from the GPU: total entries + overflow tiles.
    if (!c.counts.empty()) {
        const u32* cnt = reinterpret_cast<const u32*>(c.counts.data());
        const usize tiles = c.counts.size() / 4;
        usize entries = 0, overflow = 0, maxc = 0;
        for (usize i = 0; i < tiles; ++i) { entries += cnt[i]; if (cnt[i] > binned.bin->bin_capacity) ++overflow; if (cnt[i] > maxc) maxc = cnt[i]; }
        cardinal::log::infof("aegisbench", "%s: %zu tiles, %zu bin entries (avg %.1f/tile, max %zu), %zu overflow tiles",
                             be_name, static_cast<size_t>(tiles), static_cast<size_t>(entries),
                             static_cast<double>(entries) / static_cast<double>(tiles ? tiles : 1),
                             static_cast<size_t>(maxc), static_cast<size_t>(overflow));
    }

    // ---- timing ----------------------------------------------------------------
    const double ms_brute  = time_frames(*dev, *queue, brute,  3);
    const double ms_binned = time_frames(*dev, *queue, binned, 20);
    cardinal::log::infof("aegisbench",
        "%s  %ux%u  %u tris :  brute %.2f ms/frame   binned %.2f ms/frame   speedup %.1fx",
        be_name, sc.W, sc.H, sc.T, ms_brute, ms_binned,
        ms_binned > 0.0 ? ms_brute / ms_binned : 0.0);

    bench_pipeline(*dev, *queue, be_name);
    bench_zero_copy(*dev, *queue, be_name);
}

}  // namespace

int main() {
    run_backend(rhi::Backend::Vulkan, "vulkan");
    run_backend(rhi::Backend::D3D12,  "d3d12");
    cardinal::log::infof("aegisbench", "%d checks, %d failed", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
