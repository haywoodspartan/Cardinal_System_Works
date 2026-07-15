// =============================================================================
// Cardinal — AEGIS GPU compute numerical validation harness.
//
// The render_graph suite pins every AEGIS pass bit-level on the CPU; this
// harness closes the other half: it creates a REAL rhi::Device WINDOWLESSLY
// (neither backend needs a surface until swapchain creation), builds the
// production AEGIS graph over synthetic scene data, executes it twice —
//   1. CpuBackend            → reference bytes per output resource
//   2. RhiBackend over the async ComputeQueue recorder → real GPU dispatch
//      of every authored kernel, outputs copied to readback buffers
// — then compares. Integer outputs must match EXACTLY; float outputs get a
// tight epsilon; u8-quantised outputs (tonemap/composite/oct-normals) allow
// ±1 in the last place for legal FP contraction differences.
//
// Scenario A binds every optional input (curvature / aux / overlays /
// prev-VP); scenario B omits them all, exercising the fixed-SRV-layout
// dummy-binding paths. Both run on BOTH backends when available.
//
// SKIP semantics: no capable GPU (rhi::create_device null) or no dedicated
// async-compute queue → the affected backend is skipped with a log line and
// the test still exits 0, keeping the GPU-less CI green.
// =============================================================================

#include <cardinal/render/gpu_aegis.hpp>
#include <cardinal/render/graph.hpp>
#include <cardinal/render/aegis_runner.hpp>
#include <cardinal/rhi/rhi.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/std/cstring.hpp>

#include <string>
#include <vector>
#include <cmath>

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
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("gputest", "FAIL  L%d  %s", line, what);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

// ---------------------------------------------------------------------------
// Synthetic scene — host storage must outlive the graph (import_buffer keeps
// the pointer). Deterministic, and deliberately clear of every classifier /
// LOD / rounding knife-edge so 1-ulp GPU differences can't flip a compare.
// ---------------------------------------------------------------------------
struct SceneData {
    static constexpr u32 W = 32, H = 32, T = 200;

    std::vector<float> tris;        // T * 9, authored directly in NDC (VP = I)
    std::vector<float> curvature;   // T
    std::vector<u32>   mat_ids;     // T
    std::vector<u32>   aux;         // T
    std::vector<float> materials;   // 2 * 8
    std::vector<float> lights;      // 3 * 16 (PackedLight; [11] = intensity)
    std::vector<float> ambient;     // 3
    std::vector<float> vp;          // 16 identity (row-major)
    std::vector<float> vp_prev;     // 16 identity
    std::vector<float> cam_dir;     // 3
    std::vector<u8>    ui;          // W*H*4
    std::vector<u8>    gizmo;       // W*H*4

    void build() {
        tris.resize(usize(T) * 9);
        curvature.resize(T);
        mat_ids.resize(T);
        aux.resize(T);
        // 10x10 grid of small triangles; two depth layers (i and i+100 share a
        // cell but z always differs by >= 2*0.2237 — no depth knife-edges).
        const float curv_vals[5] = { 0.0f, 0.01f, 0.1f, 0.25f, 0.6f };
        for (u32 i = 0; i < T; ++i) {
            const u32   gx = i % 10, gy = (i / 10) % 10;
            float cx = -0.9f + static_cast<float>(gx) * 0.2f;
            float cy = -0.9f + static_cast<float>(gy) * 0.2f;
            float z  = -0.8f + static_cast<float>(i % 7) * 0.2237f;
            const bool degenerate = (i % 13) == 12;
            const bool offscreen  = (i % 11) == 10;
            if (offscreen) cx += 5.0f;
            float* v = tris.data() + usize(i) * 9;
            if (degenerate) {
                v[0] = cx; v[1] = cy; v[2] = z;
                v[3] = cx; v[4] = cy; v[5] = z;
                v[6] = cx; v[7] = cy; v[8] = z;
            } else if ((i % 17) != 16) {  // CCW (dominant winding)
                v[0] = cx - 0.07f; v[1] = cy - 0.07f; v[2] = z;
                v[3] = cx + 0.07f; v[4] = cy - 0.07f; v[5] = z;
                v[6] = cx;         v[7] = cy + 0.09f; v[8] = z;
            } else {                      // rare CW (two-sided raster coverage).
                // Deliberately RARE: alternating windings made the meshlet
                // cone-axis accumulator sum ~equal +z/-z normals — catastrophic
                // cancellation whose normalize() amplifies 1-ulp GPU/CPU
                // differences into O(1) axis flips. A dominant winding keeps
                // the accumulator well-conditioned.
                v[0] = cx - 0.07f; v[1] = cy - 0.07f; v[2] = z;
                v[3] = cx;         v[4] = cy + 0.09f; v[5] = z;
                v[6] = cx + 0.07f; v[7] = cy - 0.07f; v[8] = z;
            }
            curvature[i] = curv_vals[i % 5];
            mat_ids[i]   = i % 3;        // 2 -> out-of-range: 0.5 fallback path
            aux[i]       = i * 10u + 1u;
        }
        materials = { 0.8f, 0.2f, 0.1f, 0.5f, 0.5f, 0.0f, 0.0f, 0.0f,
                      0.1f, 0.9f, 0.3f, 0.2f, 0.8f, 0.0f, 0.0f, 0.0f };
        lights.assign(3 * 16, 0.0f);
        lights[0 * 16 + 11] = 2.0f;      // light 0: on
        lights[1 * 16 + 11] = 0.0f;      // light 1: off (skipped)
        lights[2 * 16 + 11] = 1.0f;      // light 2: on
        ambient = { 0.1f, 0.2f, 0.3f };
        vp.assign(16, 0.0f);
        vp[0] = vp[5] = vp[10] = vp[15] = 1.0f;   // identity: NDC passthrough
        vp_prev = vp;
        cam_dir = { 0.0f, 0.0f, 1.0f };
        ui.assign(usize(W) * H * 4, 0);
        gizmo.assign(usize(W) * H * 4, 0);
        for (u32 x = 0; x < 6; ++x) {             // red UI strip, top-left
            ui[(0 * W + x) * 4 + 0] = 200;
            ui[(0 * W + x) * 4 + 3] = 255;
        }
        for (u32 x = 0; x < 12; ++x) {            // green gizmo strip, bottom
            gizmo[((H - 1) * W + x) * 4 + 1] = 180;
            gizmo[((H - 1) * W + x) * 4 + 3] = 255;
        }
    }
};

// ---------------------------------------------------------------------------
// Output comparison modes
// ---------------------------------------------------------------------------
enum class Cmp {
    Exact,        // byte-identical
    EpsF32,       // per-float |a-b| <= max(1e-5, 1e-5*|a|)
    Byte1,        // per-byte |a-b| <= 1 (u8 quantisation of float math)
    Lane16,       // per-16-bit lane |a-b| <= 1 (oct-packed snorm16 normals)
    Meshlet,      // 48B records: [0..1] u32 exact, [2..7] f32 exact, [8..11] eps
    CmdPrefix,    // draw.cmds: compare only count*16 bytes (rest = GPU garbage)
    TilePrefix,   // tile.lights: per tile compare only count entries
};

struct Ref {
    const char*        name;    // BufferDesc name (RhiBackend cache key)
    rg::ResourceHandle handle;  // for CpuBackend::buffer_contents
    Cmp                mode;
};

bool feq(float a, float b) {
    if (a == b) return true;
    const float d  = std::fabs(a - b);
    const float m  = std::fabs(a) > std::fabs(b) ? std::fabs(a) : std::fabs(b);
    return d <= 1e-5f + 1e-5f * m;
}

// Compare one output; `extra` carries the sibling count buffer for prefix modes.
bool compare_output(const Ref& r,
                    const std::vector<u8>& cpu, const std::vector<u8>& gpu_bytes,
                    const std::vector<u8>& extra, std::string& why) {
    if (cpu.size() > gpu_bytes.size()) { why = "gpu buffer smaller than cpu"; return false; }
    const usize n = cpu.size();
    auto as_f32 = [](const std::vector<u8>& v, usize i) {
        float f; cardinal::memcpy(&f, v.data() + i * 4, 4); return f;
    };
    auto as_u32 = [](const std::vector<u8>& v, usize i) {
        u32 x; cardinal::memcpy(&x, v.data() + i * 4, 4); return x;
    };
    switch (r.mode) {
    case Cmp::Exact:
        for (usize i = 0; i < n; ++i)
            if (cpu[i] != gpu_bytes[i]) { why = "byte " + std::to_string(i); return false; }
        return true;
    case Cmp::Byte1:
        for (usize i = 0; i < n; ++i) {
            const int d = int(cpu[i]) - int(gpu_bytes[i]);
            if (d < -1 || d > 1) { why = "byte " + std::to_string(i); return false; }
        }
        return true;
    case Cmp::EpsF32:
        for (usize i = 0; i < n / 4; ++i)
            if (!feq(as_f32(cpu, i), as_f32(gpu_bytes, i))) {
                why = "float " + std::to_string(i); return false;
            }
        return true;
    case Cmp::Lane16:
        for (usize i = 0; i < n / 4; ++i) {
            const u32 a = as_u32(cpu, i), b = as_u32(gpu_bytes, i);
            const int lo = int(a & 0xFFFF) - int(b & 0xFFFF);
            const int hi = int(a >> 16)    - int(b >> 16);
            if (lo < -1 || lo > 1 || hi < -1 || hi > 1) {
                why = "lane16 " + std::to_string(i); return false;
            }
        }
        return true;
    case Cmp::Meshlet:
        for (usize m = 0; m < n / 48; ++m) {
            const usize f = m * 12;
            for (usize k = 0; k < 8; ++k)          // first/count + exact bounds
                if (as_u32(cpu, f + k) != as_u32(gpu_bytes, f + k)) {
                    why = "meshlet " + std::to_string(m) + " word " + std::to_string(k);
                    return false;
                }
            // Cone axis + cos accumulate 64 normalize/cross chains — FP
            // contraction differences compound, so compare with an absolute
            // tolerance sized for unit-length values (still catches any real
            // logic error; the consumer applies a 0.5 slack on cone tests).
            for (usize k = 8; k < 12; ++k) {
                const float d = std::fabs(as_f32(cpu, f + k) - as_f32(gpu_bytes, f + k));
                if (!(d <= 2e-3f)) {
                    why = "meshlet " + std::to_string(m) + " float " + std::to_string(k);
                    return false;
                }
            }
        }
        return true;
    case Cmp::CmdPrefix: {
        if (extra.size() < 4) { why = "no count"; return false; }
        u32 count; cardinal::memcpy(&count, extra.data(), 4);
        const usize bytes = usize(count) * 16;
        if (bytes > n) { why = "count too big"; return false; }
        for (usize i = 0; i < bytes; ++i)
            if (cpu[i] != gpu_bytes[i]) { why = "cmd byte " + std::to_string(i); return false; }
        return true;
    }
    case Cmp::TilePrefix: {
        const usize tiles = extra.size() / 4;
        for (usize t = 0; t < tiles; ++t) {
            u32 c; cardinal::memcpy(&c, extra.data() + t * 4, 4);
            if (c > 64) { why = "tile count corrupt"; return false; }
            for (usize e = 0; e < c; ++e) {
                const usize i = (t * 64 + e);
                if (as_u32(cpu, i) != as_u32(gpu_bytes, i)) {
                    why = "tile " + std::to_string(t) + " entry " + std::to_string(e);
                    return false;
                }
            }
        }
        return true;
    }
    }
    why = "unknown mode";
    return false;
}

// ---------------------------------------------------------------------------
// The GPU half: RhiBackend recorded onto the async ComputeQueue.
// ---------------------------------------------------------------------------
struct GpuRunCtx {
    rhi::Device*                                dev {nullptr};
    rg::Graph*                                  graph {nullptr};
    cardinal::shared_ptr<rg::RhiBackend>*       backend {nullptr};  // kept alive past submit
    const std::vector<Ref>*                     refs {nullptr};
    std::vector<rhi::Buffer*>*                  readbacks {nullptr};  // parallel to refs
};

void gpu_record(rhi::Swapchain* rec, void* user) noexcept {
    auto& cx = *static_cast<GpuRunCtx*>(user);
    // The production dispatch path, verbatim: buffer map + lazy DXC compile +
    // hazard barriers + bind + push + dispatch — recorded onto the async list.
    *cx.backend = rg::RhiBackend::create(*cx.dev, *rec);
    (*cx.backend)->set_gpu_execute(true);
    (*cx.backend)->execute(*cx.graph);
    for (usize i = 0; i < cx.refs->size(); ++i) {
        const Ref& r = (*cx.refs)[i];
        rhi::Buffer* src = (*cx.backend)->gpu_buffer(r.name);
        rhi::Buffer* dst = (*cx.readbacks)[i];
        if (!src || !dst) continue;
        (*cx.backend)->make_copy_source(r.name);
        rec->copy_buffer(src, 0, dst, 0, dst->size());
    }
}

// Runs both scenarios on one backend. Returns false only on real failures.
void run_backend(rhi::Backend be, const char* be_name) {
    rhi::DeviceDesc dd{};
    dd.backend = be;
    auto dev = rhi::create_device(dd);
    if (!dev) {
        cardinal::log::infof("gputest", "SKIP %s: no capable device", be_name);
        return;
    }
    auto queue = dev->create_async_compute_queue();
    if (!queue) {
        cardinal::log::infof("gputest", "SKIP %s: no async compute queue", be_name);
        return;
    }
    auto fence = dev->create_fence(0);
    if (!fence) {
        cardinal::log::infof("gputest", "SKIP %s: no fence", be_name);
        return;
    }

    for (int scenario = 0; scenario < 2; ++scenario) {
        const bool with_opt = (scenario == 0);
        SceneData sd;
        sd.build();

        auto graph = rg::Graph::create();
        gpu::AegisSceneInputs in;
        in.triangle_count = SceneData::T;
        in.tris         = graph->import_buffer("in.tris",  sd.tris.data(),
                              sd.tris.size() * 4, 4);
        in.material_ids = graph->import_buffer("in.mids",  sd.mat_ids.data(),
                              sd.mat_ids.size() * 4, 4);
        in.materials    = graph->import_buffer("in.mats",  sd.materials.data(),
                              sd.materials.size() * 4, 4);
        in.lights       = graph->import_buffer("in.lts",   sd.lights.data(),
                              sd.lights.size() * 4, 4);
        in.ambient      = graph->import_buffer("in.amb",   sd.ambient.data(), 12, 4);
        in.view_proj    = graph->import_buffer("in.vp",    sd.vp.data(), 64, 4);
        in.camera_dir   = graph->import_buffer("in.dir",   sd.cam_dir.data(), 12, 4);
        if (with_opt) {
            in.curvature      = graph->import_buffer("in.curv", sd.curvature.data(),
                                    sd.curvature.size() * 4, 4);
            in.view_proj_prev = graph->import_buffer("in.vpp",  sd.vp_prev.data(), 64, 4);
            in.ui_overlay     = graph->import_buffer("in.ui",   sd.ui.data(),
                                    sd.ui.size(), 4);
            in.gizmo_overlay  = graph->import_buffer("in.giz",  sd.gizmo.data(),
                                    sd.gizmo.size(), 4);
        }

        gpu::AegisConfig cfg{};
        cfg.width  = SceneData::W;
        cfg.height = SceneData::H;
        cfg.material_count = 2;
        cfg.light_count    = 3;
        cfg.exposure       = 1.0f;
        auto pipeline = gpu::AegisPipeline::create(cfg);
        gpu::AegisOutputs   out;
        gpu::AegisStageRefs st;
        pipeline->build(*graph, in, out, st);
        CHECK(graph->compile());

        // ---- CPU reference ------------------------------------------------
        auto cpu = rg::CpuBackend::create();
        cpu->execute(*graph);

        std::vector<Ref> refs = {
            // Imported inputs first — if these don't round-trip bit-exact the
            // upload/binding path is broken and every kernel compare is noise.
            { "in.tris",      in.tris,                           Cmp::Exact     },
            { "in.lts",       in.lights,                         Cmp::Exact     },
            { "in.vp",        in.view_proj,                      Cmp::Exact     },
            { "geom.class",   st.classify->out_class,            Cmp::Exact     },
            { "meshlets",     st.meshlets->out_meshlets,         Cmp::Meshlet   },
            { "meshlet.count",st.meshlets->out_meshlet_count,    Cmp::Exact     },
            { "meshlet.vis",  st.meshlet_cull->out_visibility,   Cmp::Exact     },
            { "sse.lod",      st.sse->out_lod,                   Cmp::Exact     },
            { "draw.count",   st.indirect_gen->out_count,        Cmp::Exact     },
            { "draw.cmds",    st.indirect_gen->out_commands,     Cmp::CmdPrefix },
            { "tile.counts",  st.light_cull->out_tile_counts,    Cmp::Exact     },
            { "tile.lights",  st.light_cull->out_tile_lights,    Cmp::TilePrefix},
            { "vbuf.depth",   st.vbuf->out_depth,                Cmp::EpsF32    },
            { "vbuf.prim",    st.vbuf->out_prim_id,              Cmp::Exact     },
            { "vbuf.mat",     st.vbuf->out_mat_id,               Cmp::Exact     },
            { "vbuf.normal",  st.vbuf->out_normal,               Cmp::Lane16    },
            { "vbuf.motion",  st.vbuf->out_motion,               Cmp::Exact     },
            { "vbuf.aux",     st.vbuf->out_aux,                  Cmp::Exact     },
            { "resolve.rad",  st.resolve->out_radiance,          Cmp::EpsF32    },
            { "tone.rgba",    st.tonemap->out_rgba,              Cmp::Byte1     },
            { "present.rgba", st.composite->out_presentation,    Cmp::Byte1     },
        };
        if (with_opt && st.motion)
            refs.push_back({ "motion.mv", st.motion->out_motion, Cmp::Exact });

        // ---- GPU dispatch + readback --------------------------------------
        std::vector<cardinal::unique_ptr<rhi::Buffer>> rb_owned;
        std::vector<rhi::Buffer*>                      rb_ptrs;
        for (const Ref& r : refs) {
            const usize sz = graph->buffer(r.handle).size_bytes;
            rhi::BufferDesc bd{};
            bd.size         = sz;
            bd.usage        = 0;
            bd.cpu_writable = false;
            bd.cpu_readable = true;
            rb_owned.push_back(dev->create_buffer(bd));
            rb_ptrs.push_back(rb_owned.back().get());
            CHECK(rb_ptrs.back() != nullptr);
        }

        cardinal::shared_ptr<rg::RhiBackend> backend;
        GpuRunCtx cx;
        cx.dev = dev.get(); cx.graph = graph.get(); cx.backend = &backend;
        cx.refs = &refs;    cx.readbacks = &rb_ptrs;
        // Fence targets must INCREASE across submits — a timeline semaphore
        // cannot re-signal an old value, and wait_cpu(old) returns instantly,
        // racing the readback against in-flight GPU work.
        const cardinal::u64 target_value = static_cast<cardinal::u64>(scenario) + 1u;
        const cardinal::u64 target = queue->submit(&gpu_record, &cx, fence.get(),
                                                   target_value);
        CHECK(target == target_value);
        fence->wait_cpu(target_value, 0);
        CHECK(backend != nullptr);

        // Every kernel pass must have REALLY dispatched (no silent stub-fallback).
        u32 real = 0, stub = 0;
        for (const auto& ev : backend->events()) {
            if (ev.real_dispatch) ++real; else ++stub;
        }
        const auto stats = backend->stats();
        cardinal::log::infof("gputest",
            "%s scenario %c: %u real dispatches, %u stubs, %u pipelines",
            be_name, with_opt ? 'A' : 'B', real, stub, stats.pipelines_compiled);
        // 10 kernels in A (motion built), 9 in B; HiZ + Adaptive stay stubs.
        CHECK(real >= (with_opt ? 10u : 9u));

        for (usize i = 0; i < refs.size(); ++i) {
            const Ref& r = refs[i];
            const auto expect = cpu->buffer_contents(r.handle);
            CHECK(!expect.empty());
            std::vector<u8> got(rb_ptrs[i]->size());
            CHECK(rb_ptrs[i]->download(got.data(), got.size(), 0));

            std::vector<u8> extra;
            if (r.mode == Cmp::CmdPrefix) {
                const auto e = cpu->buffer_contents(st.indirect_gen->out_count);
                extra.assign(e.begin(), e.end());
            } else if (r.mode == Cmp::TilePrefix) {
                const auto e = cpu->buffer_contents(st.light_cull->out_tile_counts);
                extra.assign(e.begin(), e.end());
            }
            std::vector<u8> expect_std(expect.begin(), expect.end());
            std::string why;
            const bool ok = compare_output(r, expect_std, got, extra, why);
            if (!ok) {
                cardinal::log::errorf("gputest", "%s/%c MISMATCH %s: %s",
                                      be_name, with_opt ? 'A' : 'B', r.name, why.c_str());
                // First 16 bytes of each side, as u32 hex — enough to see
                // zero-vs-value, off-by-slot, endianness, or 1-ulp patterns.
                u32 c[4]{}, g[4]{};
                cardinal::memcpy(c, expect_std.data(),
                                 expect_std.size() < 16 ? expect_std.size() : 16);
                cardinal::memcpy(g, got.data(), got.size() < 16 ? got.size() : 16);
                cardinal::log::errorf("gputest",
                    "  cpu: %08X %08X %08X %08X", c[0], c[1], c[2], c[3]);
                cardinal::log::errorf("gputest",
                    "  gpu: %08X %08X %08X %08X", g[0], g[1], g[2], g[3]);
            }
            CHECK(ok);
        }

        // Backend (and its GPU buffers) can die now — the fence retired.
        backend.reset();

        // ---- Productized async routing smoke --------------------------------
        // The manual recorder path above validates NUMERICS; this validates
        // the PRODUCTION plumbing: a swapchain-less RhiBackend given the
        // async ComputeQueue records + submits the whole graph internally
        // (set_async_queue), and its destructor drains in-flight work.
        if (with_opt) {
            auto abackend = rg::RhiBackend::create(*dev);
            abackend->set_gpu_execute(true);
            abackend->set_async_queue(queue.get());
            abackend->execute(*graph);
            u32 areal = 0;
            for (const auto& ev : abackend->events())
                if (ev.real_dispatch) ++areal;
            CHECK(areal >= 10u);                    // every kernel routed async
            abackend.reset();                       // dtor drain must not hang
            cardinal::log::infof("gputest",
                "%s async routing: %u real dispatches via ComputeQueue", be_name, areal);
        }
    }
}

}  // namespace

int main() {
    run_backend(rhi::Backend::Vulkan, "vulkan");
#if defined(_WIN32)
    run_backend(rhi::Backend::D3D12, "d3d12");
#endif

    if (g_fail == 0) {
        cardinal::log::infof("gputest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("gputest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
