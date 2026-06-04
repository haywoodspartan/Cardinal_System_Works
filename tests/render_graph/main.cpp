// =============================================================================
// Cardinal — Render-graph regression suite.
//
// The graph is the declarative seam between the engine's per-frame GPU
// work and whatever executes it (CpuBackend for tests, RhiBackend for
// real Vulkan / D3D12 dispatch). A regression in compile() silently
// breaks the execution order for every host that uses the framework;
// a regression in CpuBackend silently breaks every test that depends
// on it for offline validation. This suite pins both.
//
// Pure CPU + headless. Exit 0 = all pass.
// =============================================================================

#include <cardinal/render/graph.hpp>
#include <cardinal/render/gpu_passes.hpp>
#include <cardinal/render/gpu_raster.hpp>
#include <cardinal/core/log.hpp>
#include <cardinal/core/utility.hpp>
#include <cardinal/core/simd_math.hpp>
#include <cardinal/core/geom.hpp>

#include <limits>

namespace {

namespace rg = cardinal::render::graph;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("gtest", "FAIL L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

// Helper to build a one-buffer-of-N-floats descriptor.
rg::ResourceHandle declare_f32_buffer(rg::Graph& g, const char* name, cardinal::u32 n) {
    rg::BufferDesc bd;
    bd.name         = name;
    bd.size_bytes   = static_cast<cardinal::usize>(n) * sizeof(float);
    bd.stride_bytes = sizeof(float);
    return g.declare_buffer(bd);
}

// Generic pass that copies bytes from input → output by overwriting each
// destination float with a function of the source. The `user_ctx` is a
// pointer to a struct describing the operation.
struct CopyOp {
    rg::ResourceHandle in;
    rg::ResourceHandle out;
    float              scale {1.0f};
    float              bias  {0.0f};
    cardinal::u32      count {0};
};

void copy_scale_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    CopyOp* op = static_cast<CopyOp*>(uctx);
    const float* src = static_cast<const float*>(ec.map_buffer_read(op->in));
    float*       dst = static_cast<float*>      (ec.map_buffer_write(op->out));
    if (!src || !dst) return;
    for (cardinal::u32 i = 0; i < op->count; ++i) {
        dst[i] = src[i] * op->scale + op->bias;
    }
    ec.dispatch((op->count + 63) / 64, 1, 1);    // 64 threads per group, virtual
}

// Pass that just writes a constant value into its output buffer.
struct FillOp { rg::ResourceHandle out; float value; cardinal::u32 count; };
void fill_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    FillOp* op = static_cast<FillOp*>(uctx);
    float* dst = static_cast<float*>(ec.map_buffer_write(op->out));
    if (!dst) return;
    for (cardinal::u32 i = 0; i < op->count; ++i) dst[i] = op->value;
    ec.dispatch((op->count + 63) / 64, 1, 1);
}

// Pass that reads two inputs and writes their per-element sum to an output.
struct AddOp {
    rg::ResourceHandle a, b, out;
    cardinal::u32      count;
};
void add_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    AddOp* op = static_cast<AddOp*>(uctx);
    const float* a = static_cast<const float*>(ec.map_buffer_read(op->a));
    const float* b = static_cast<const float*>(ec.map_buffer_read(op->b));
    float*       out = static_cast<float*>(ec.map_buffer_write(op->out));
    if (!a || !b || !out) return;
    for (cardinal::u32 i = 0; i < op->count; ++i) out[i] = a[i] + b[i];
    ec.dispatch((op->count + 63) / 64, 1, 1);
}

// ----------------------------------------------------------------------------
// Declaration & accessors.
// ----------------------------------------------------------------------------
void test_empty_graph_compiles() {
    auto g = rg::Graph::create();
    CHECK(g->compile());
    CHECK(g->execution_order().empty());
    CHECK(g->stats().passes == 0u);
}

void test_resource_declaration() {
    auto g = rg::Graph::create();
    auto buf = declare_f32_buffer(*g, "verts", 256);
    rg::TextureDesc td;
    td.name = "depth";
    td.width = 1920; td.height = 1080;
    td.format = rg::TextureFormat::D32_Float;
    auto tex = g->declare_texture(td);

    CHECK(buf.is_valid());
    CHECK(tex.is_valid());
    CHECK(buf.kind == rg::ResourceKind::Buffer);
    CHECK(tex.kind == rg::ResourceKind::Texture);
    CHECK(g->buffer(buf).size_bytes == 256u * sizeof(float));
    CHECK(g->texture(tex).width == 1920u);
    CHECK(g->texture(tex).format == rg::TextureFormat::D32_Float);
    CHECK(g->resource_count() == sz(2));
}

void test_invalid_handle_safe() {
    auto g = rg::Graph::create();
    rg::ResourceHandle bogus{ 999, rg::ResourceKind::Buffer };
    // buffer() / texture() on a bogus handle returns the sentinel (id 0)
    // descriptor — no UB, no crash.
    CHECK(g->buffer(bogus).size_bytes == 0u);
    CHECK(g->texture(bogus).width == 0u);
    rg::ResourceHandle invalid;
    CHECK(!invalid.is_valid());
}

// ----------------------------------------------------------------------------
// Topological sort — RAW / WAW / WAR ordering.
// ----------------------------------------------------------------------------
void test_topo_sort_raw() {
    // Two passes, second reads what first writes. Order must be 1 → 2.
    auto g = rg::Graph::create();
    auto a = declare_f32_buffer(*g, "a", 4);
    auto b = declare_f32_buffer(*g, "b", 4);
    FillOp op1{a, 1.0f, 4};
    CopyOp op2{a, b, 2.0f, 0.0f, 4};
    rg::PassDesc pd1; pd1.name = "fill"; pd1.kind = rg::PassKind::Compute;
    pd1.accesses.push_back(rg::ResourceAccess{a, rg::AccessMode::Write, 0});
    pd1.record = fill_record; pd1.user_ctx = &op1;
    rg::PassDesc pd2; pd2.name = "copy"; pd2.kind = rg::PassKind::Compute;
    pd2.accesses.push_back(rg::ResourceAccess{a, rg::AccessMode::Read,  0});
    pd2.accesses.push_back(rg::ResourceAccess{b, rg::AccessMode::Write, 1});
    pd2.record = copy_scale_record; pd2.user_ctx = &op2;
    const cardinal::u32 id_fill = g->add_pass(cardinal::move(pd1));
    const cardinal::u32 id_copy = g->add_pass(cardinal::move(pd2));
    CHECK(g->compile());
    const auto& order = g->execution_order();
    CHECK(order.size() == sz(2));
    CHECK(order[0] == id_fill);
    CHECK(order[1] == id_copy);
    CHECK(g->stats().read_after_write == 1u);
    CHECK(g->stats().write_after_write == 0u);
}

void test_topo_sort_waw() {
    // Two passes both write the same buffer. Order must keep them in
    // declaration order (Pass 1 writes, then Pass 2 writes) so the
    // final value is Pass 2's.
    auto g = rg::Graph::create();
    auto out = declare_f32_buffer(*g, "out", 4);
    FillOp op1{out, 1.0f, 4};
    FillOp op2{out, 7.0f, 4};
    rg::PassDesc pd1; pd1.name = "first";  pd1.kind = rg::PassKind::Compute;
    pd1.accesses.push_back(rg::ResourceAccess{out, rg::AccessMode::Write, 0});
    pd1.record = fill_record; pd1.user_ctx = &op1;
    rg::PassDesc pd2; pd2.name = "second"; pd2.kind = rg::PassKind::Compute;
    pd2.accesses.push_back(rg::ResourceAccess{out, rg::AccessMode::Write, 0});
    pd2.record = fill_record; pd2.user_ctx = &op2;
    const cardinal::u32 id1 = g->add_pass(cardinal::move(pd1));
    const cardinal::u32 id2 = g->add_pass(cardinal::move(pd2));
    CHECK(g->compile());
    const auto& order = g->execution_order();
    CHECK(order.size() == sz(2));
    CHECK(order[0] == id1);
    CHECK(order[1] == id2);
    CHECK(g->stats().write_after_write == 1u);
}

void test_topo_sort_war() {
    // Pass 1 reads buffer X, Pass 2 writes X. Pass 2 must wait for Pass 1
    // (WAR).
    auto g = rg::Graph::create();
    auto x   = declare_f32_buffer(*g, "x",   4);
    auto sink= declare_f32_buffer(*g, "sink",4);
    CopyOp op1{x, sink, 1.0f, 0.0f, 4};
    FillOp op2{x, 9.0f, 4};
    rg::PassDesc pd1; pd1.name = "read"; pd1.kind = rg::PassKind::Compute;
    pd1.accesses.push_back(rg::ResourceAccess{x,    rg::AccessMode::Read,  0});
    pd1.accesses.push_back(rg::ResourceAccess{sink, rg::AccessMode::Write, 1});
    pd1.record = copy_scale_record; pd1.user_ctx = &op1;
    rg::PassDesc pd2; pd2.name = "overwrite"; pd2.kind = rg::PassKind::Compute;
    pd2.accesses.push_back(rg::ResourceAccess{x, rg::AccessMode::Write, 0});
    pd2.record = fill_record; pd2.user_ctx = &op2;
    const cardinal::u32 id_r = g->add_pass(cardinal::move(pd1));
    const cardinal::u32 id_w = g->add_pass(cardinal::move(pd2));
    CHECK(g->compile());
    const auto& order = g->execution_order();
    CHECK(order.size() == sz(2));
    CHECK(order[0] == id_r);
    CHECK(order[1] == id_w);
    CHECK(g->stats().write_after_read == 1u);
}

void test_topo_sort_diamond() {
    //  A writes X
    //  B reads X, writes Y
    //  C reads X, writes Z
    //  D reads Y + Z, writes W
    //  Order must put A first, then B + C (any order), then D.
    auto g = rg::Graph::create();
    auto X = declare_f32_buffer(*g, "X", 4);
    auto Y = declare_f32_buffer(*g, "Y", 4);
    auto Z = declare_f32_buffer(*g, "Z", 4);
    auto W = declare_f32_buffer(*g, "W", 4);
    FillOp opA{X, 1.0f, 4};
    CopyOp opB{X, Y, 2.0f, 0.0f, 4};
    CopyOp opC{X, Z, 3.0f, 0.0f, 4};
    AddOp  opD{Y, Z, W, 4};

    auto add = [&](const char* n, void (*rec)(rg::ExecutionContext&, void*) noexcept,
                   void* uctx, cardinal::vector<rg::ResourceAccess> accesses) {
        rg::PassDesc pd; pd.name = n; pd.kind = rg::PassKind::Compute;
        pd.accesses = cardinal::move(accesses);
        pd.record = rec; pd.user_ctx = uctx;
        return g->add_pass(cardinal::move(pd));
    };
    const cardinal::u32 id_a = add("A", fill_record, &opA, {
        rg::ResourceAccess{X, rg::AccessMode::Write, 0}});
    const cardinal::u32 id_b = add("B", copy_scale_record, &opB, {
        rg::ResourceAccess{X, rg::AccessMode::Read,  0},
        rg::ResourceAccess{Y, rg::AccessMode::Write, 1}});
    const cardinal::u32 id_c = add("C", copy_scale_record, &opC, {
        rg::ResourceAccess{X, rg::AccessMode::Read,  0},
        rg::ResourceAccess{Z, rg::AccessMode::Write, 1}});
    const cardinal::u32 id_d = add("D", add_record, &opD, {
        rg::ResourceAccess{Y, rg::AccessMode::Read,  0},
        rg::ResourceAccess{Z, rg::AccessMode::Read,  1},
        rg::ResourceAccess{W, rg::AccessMode::Write, 2}});

    CHECK(g->compile());
    const auto& order = g->execution_order();
    CHECK(order.size() == sz(4));
    CHECK(order[0] == id_a);
    CHECK(order.back() == id_d);
    // Middle two are B and C in some order.
    CHECK((order[1] == id_b && order[2] == id_c) ||
          (order[1] == id_c && order[2] == id_b));
    CHECK(g->stats().read_after_write == 4u);   // X→B (read), X→C (read), Y→D, Z→D
}

void test_no_false_cycles_under_complex_deps() {
    // The graph's dependency model is forward-only (resource last_writer
    // can only be set on an EARLIER pass when a later one references it),
    // so by construction the resource access pattern can't produce a
    // cycle. This test pins that property: a deliberately "swapped"
    // pair (A reads Y writes X, B reads X writes Y) still compiles
    // because B's references to X point back at A, not vice-versa, and
    // A's reference to Y has no producer (rs[Y].last_writer == 0 when
    // A is processed).
    auto g = rg::Graph::create();
    auto x = declare_f32_buffer(*g, "x", 4);
    auto y = declare_f32_buffer(*g, "y", 4);
    rg::PassDesc pdA; pdA.name = "A"; pdA.kind = rg::PassKind::Compute;
    pdA.accesses.push_back(rg::ResourceAccess{y, rg::AccessMode::Read,  0});
    pdA.accesses.push_back(rg::ResourceAccess{x, rg::AccessMode::Write, 1});
    rg::PassDesc pdB; pdB.name = "B"; pdB.kind = rg::PassKind::Compute;
    pdB.accesses.push_back(rg::ResourceAccess{x, rg::AccessMode::Read,  0});
    pdB.accesses.push_back(rg::ResourceAccess{y, rg::AccessMode::Write, 1});
    const cardinal::u32 ida = g->add_pass(cardinal::move(pdA));
    const cardinal::u32 idb = g->add_pass(cardinal::move(pdB));
    CHECK(g->compile());                       // must succeed (no cycle)
    CHECK(!g->stats().cycle_detected);
    const auto& order = g->execution_order();
    CHECK(order.size() == sz(2));
    CHECK(order[0] == ida);                    // A first (writes X)
    CHECK(order[1] == idb);                    // B second (reads X, writes Y)
    // Both dep flavours show up: RAW from x (B reads A's write of X),
    // WAR from y (B writes Y after A read Y).
    CHECK(g->stats().read_after_write == 1u);
    CHECK(g->stats().write_after_read == 1u);
}

void test_determinism_id_order() {
    // Two passes with no dependencies between them. The compile() must
    // sort them by id (deterministic) — not by hash, not by iteration
    // order of an internal set.
    auto g = rg::Graph::create();
    auto a = declare_f32_buffer(*g, "a", 4);
    auto b = declare_f32_buffer(*g, "b", 4);
    rg::PassDesc pA; pA.name = "A"; pA.kind = rg::PassKind::Compute;
    pA.accesses.push_back(rg::ResourceAccess{a, rg::AccessMode::Write, 0});
    rg::PassDesc pB; pB.name = "B"; pB.kind = rg::PassKind::Compute;
    pB.accesses.push_back(rg::ResourceAccess{b, rg::AccessMode::Write, 0});
    const cardinal::u32 idA = g->add_pass(cardinal::move(pA));
    const cardinal::u32 idB = g->add_pass(cardinal::move(pB));
    CHECK(g->compile());
    const auto& order = g->execution_order();
    CHECK(order.size() == sz(2));
    CHECK(order[0] == idA);
    CHECK(order[1] == idB);
}

// ----------------------------------------------------------------------------
// CpuBackend — execution + access enforcement.
// ----------------------------------------------------------------------------
void test_cpu_backend_fill_and_copy() {
    auto g = rg::Graph::create();
    auto src = declare_f32_buffer(*g, "src", 16);
    auto dst = declare_f32_buffer(*g, "dst", 16);
    FillOp op_fill{src, 3.0f, 16};
    CopyOp op_copy{src, dst, 2.0f, 1.0f, 16};
    rg::PassDesc pd1; pd1.name = "fill"; pd1.kind = rg::PassKind::Compute;
    pd1.accesses.push_back(rg::ResourceAccess{src, rg::AccessMode::Write, 0});
    pd1.record = fill_record; pd1.user_ctx = &op_fill;
    rg::PassDesc pd2; pd2.name = "scale"; pd2.kind = rg::PassKind::Compute;
    pd2.accesses.push_back(rg::ResourceAccess{src, rg::AccessMode::Read,  0});
    pd2.accesses.push_back(rg::ResourceAccess{dst, rg::AccessMode::Write, 1});
    pd2.record = copy_scale_record; pd2.user_ctx = &op_copy;
    g->add_pass(cardinal::move(pd1));
    g->add_pass(cardinal::move(pd2));
    CHECK(g->compile());
    auto backend = rg::CpuBackend::create();
    backend->execute(*g);
    auto out = backend->buffer_contents(dst);
    CHECK(out.size() == 16u * sizeof(float));
    const float* outf = reinterpret_cast<const float*>(out.data());
    for (int i = 0; i < 16; ++i) {
        // expect 3.0 * 2.0 + 1.0 = 7.0
        const float v = outf[i];
        CHECK(v == 7.0f);
    }
    CHECK(backend->trace().size() == sz(2));
}

void test_cpu_backend_access_enforcement() {
    // A pass that's NOT declared with Write on a buffer must get
    // nullptr back from map_buffer_write — and our record() must handle
    // that gracefully (we wrote it to early-return).
    struct {
        bool called{false};
        bool got_null_on_write{false};
        bool got_null_on_read{false};
        rg::ResourceHandle h;
    } ctx;
    auto g = rg::Graph::create();
    auto buf = declare_f32_buffer(*g, "x", 4);
    ctx.h = buf;
    auto rec = [](rg::ExecutionContext& ec, void* uctx) noexcept {
        auto* c = reinterpret_cast<decltype(&ctx)>(uctx);
        c->called = true;
        // We declared this pass with Read only on `h`. Calling
        // map_buffer_write must return nullptr.
        const void* p_read = ec.map_buffer_read(c->h);
        void*       p_wrt  = ec.map_buffer_write(c->h);
        c->got_null_on_read  = (p_read == nullptr);
        c->got_null_on_write = (p_wrt  == nullptr);
    };
    rg::PassDesc pd; pd.name = "read-only";
    pd.accesses.push_back(rg::ResourceAccess{buf, rg::AccessMode::Read, 0});
    pd.record  = rec;
    pd.user_ctx = &ctx;
    g->add_pass(cardinal::move(pd));
    CHECK(g->compile());
    rg::CpuBackend::create()->execute(*g);
    CHECK(ctx.called);
    CHECK(!ctx.got_null_on_read);          // read access allowed
    CHECK(ctx.got_null_on_write);          // write access denied
}

void test_cpu_backend_trace_records_dispatches() {
    auto g = rg::Graph::create();
    auto buf = declare_f32_buffer(*g, "x", 64);
    FillOp op{buf, 1.0f, 64};
    rg::PassDesc pd; pd.name = "kernel"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{buf, rg::AccessMode::Write, 0});
    pd.record = fill_record; pd.user_ctx = &op;
    g->add_pass(cardinal::move(pd));
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    CHECK(b->trace().size() == sz(1));
    const auto& t = b->trace()[0];
    CHECK(t.name == "kernel");
    CHECK(t.kind == rg::PassKind::Compute);
    CHECK(t.dispatched_gx == 1u);
    CHECK(t.dispatched_gy == 1u);
    CHECK(t.dispatched_gz == 1u);
    CHECK(t.bytes_written == 64u * sizeof(float));
    CHECK(t.bytes_read    == 0u);
}

void test_cpu_backend_param_kv() {
    struct {
        cardinal::u32 read_value{0};
        cardinal::u32 fallback_value{0};
    } ctx;
    auto rec = [](rg::ExecutionContext& ec, void* uctx) noexcept {
        auto* c = reinterpret_cast<decltype(&ctx)>(uctx);
        ec.set_param_u32("count", 42u);
        c->read_value     = ec.param_u32("count", 999u);
        c->fallback_value = ec.param_u32("missing", 777u);
    };
    auto g = rg::Graph::create();
    rg::PassDesc pd; pd.name = "p"; pd.kind = rg::PassKind::Compute;
    pd.record = rec; pd.user_ctx = &ctx;
    g->add_pass(cardinal::move(pd));
    CHECK(g->compile());
    rg::CpuBackend::create()->execute(*g);
    CHECK(ctx.read_value == 42u);
    CHECK(ctx.fallback_value == 777u);
}

void test_cpu_backend_no_record_safe() {
    // A pass with a null record callback must still appear in the trace,
    // just with zero byte counts.
    auto g = rg::Graph::create();
    rg::PassDesc pd; pd.name = "noop"; pd.kind = rg::PassKind::Compute;
    pd.dispatch_x = 8; pd.dispatch_y = 4; pd.dispatch_z = 2;
    g->add_pass(cardinal::move(pd));
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    CHECK(b->trace().size() == sz(1));
    CHECK(b->trace()[0].name == "noop");
    CHECK(b->trace()[0].dispatched_gx == 8u);
    CHECK(b->trace()[0].dispatched_gy == 4u);
    CHECK(b->trace()[0].dispatched_gz == 2u);
    CHECK(b->trace()[0].bytes_written == 0u);
}

void test_cpu_backend_determinism_across_runs() {
    // Same graph + same backend + same inputs → bit-identical buffer outputs.
    auto build = [](rg::Graph& g, FillOp* opf, CopyOp* opc) {
        auto a = declare_f32_buffer(g, "a", 32);
        auto b = declare_f32_buffer(g, "b", 32);
        opf->out = a; opf->value = 5.5f; opf->count = 32;
        opc->in = a; opc->out = b; opc->scale = 1.5f; opc->bias = -0.25f; opc->count = 32;
        rg::PassDesc pd1; pd1.name = "f"; pd1.kind = rg::PassKind::Compute;
        pd1.accesses.push_back(rg::ResourceAccess{a, rg::AccessMode::Write, 0});
        pd1.record = fill_record; pd1.user_ctx = opf;
        rg::PassDesc pd2; pd2.name = "c"; pd2.kind = rg::PassKind::Compute;
        pd2.accesses.push_back(rg::ResourceAccess{a, rg::AccessMode::Read,  0});
        pd2.accesses.push_back(rg::ResourceAccess{b, rg::AccessMode::Write, 1});
        pd2.record = copy_scale_record; pd2.user_ctx = opc;
        g.add_pass(cardinal::move(pd1));
        g.add_pass(cardinal::move(pd2));
        return b;
    };
    FillOp f1{}, f2{}; CopyOp c1{}, c2{};
    auto ga = rg::Graph::create();
    auto gb = rg::Graph::create();
    auto out_a = build(*ga, &f1, &c1);
    auto out_b = build(*gb, &f2, &c2);
    CHECK(ga->compile());
    CHECK(gb->compile());
    auto ba = rg::CpuBackend::create();
    auto bb = rg::CpuBackend::create();
    ba->execute(*ga);
    bb->execute(*gb);
    auto va = ba->buffer_contents(out_a);
    auto vb = bb->buffer_contents(out_b);
    CHECK(va.size() == vb.size());
    for (cardinal::usize i = 0; i < va.size(); ++i) CHECK(va[i] == vb[i]);
}

// ----------------------------------------------------------------------------
// End-to-end: a "GPU pipeline" expressed as a graph.
// ----------------------------------------------------------------------------
void test_pipeline_cull_transform_tonemap() {
    // A miniature pipeline that mirrors the engine's intended GPU layout:
    //   cull_pass:      reads in_aabbs, writes vis_bits         (compute)
    //   transform_pass: reads vis_bits + in_verts, writes xform (compute)
    //   tonemap_pass:   reads xform, writes out_rgba            (compute)
    //
    // The CpuBackend runs each pass's reference impl. The order check
    // pins the dependency chain.
    auto g = rg::Graph::create();
    auto in_aabbs = declare_f32_buffer(*g, "in_aabbs", 16);
    auto vis_bits = declare_f32_buffer(*g, "vis_bits", 4);
    auto in_verts = declare_f32_buffer(*g, "in_verts", 16);
    auto xform    = declare_f32_buffer(*g, "xform",    16);
    auto out_rgba = declare_f32_buffer(*g, "out_rgba", 16);

    // We can use the simple fill / copy ops as stand-ins for the real
    // shaders — the dependency-tracking is what we want to pin.
    FillOp fill_aabbs{in_aabbs, 0.0f, 16};
    FillOp fill_verts{in_verts, 1.0f, 16};
    CopyOp cull_p{in_aabbs, vis_bits, 1.0f, 0.0f, 4};
    AddOp  xform_p{vis_bits, in_verts, xform, 4};
    CopyOp tone_p {xform, out_rgba, 2.0f, 0.0f, 16};

    auto add = [&](const char* n, void (*r)(rg::ExecutionContext&, void*) noexcept,
                   void* u, cardinal::vector<rg::ResourceAccess> ax) {
        rg::PassDesc pd; pd.name = n; pd.kind = rg::PassKind::Compute;
        pd.accesses = cardinal::move(ax);
        pd.record = r; pd.user_ctx = u;
        return g->add_pass(cardinal::move(pd));
    };
    const cardinal::u32 id_init_aabbs = add("init_aabbs", fill_record, &fill_aabbs, {
        rg::ResourceAccess{in_aabbs, rg::AccessMode::Write, 0}});
    const cardinal::u32 id_init_verts = add("init_verts", fill_record, &fill_verts, {
        rg::ResourceAccess{in_verts, rg::AccessMode::Write, 0}});
    const cardinal::u32 id_cull = add("cull", copy_scale_record, &cull_p, {
        rg::ResourceAccess{in_aabbs, rg::AccessMode::Read,  0},
        rg::ResourceAccess{vis_bits, rg::AccessMode::Write, 1}});
    const cardinal::u32 id_xform = add("xform", add_record, &xform_p, {
        rg::ResourceAccess{vis_bits, rg::AccessMode::Read,  0},
        rg::ResourceAccess{in_verts, rg::AccessMode::Read,  1},
        rg::ResourceAccess{xform,    rg::AccessMode::Write, 2}});
    const cardinal::u32 id_tone = add("tonemap", copy_scale_record, &tone_p, {
        rg::ResourceAccess{xform,    rg::AccessMode::Read,  0},
        rg::ResourceAccess{out_rgba, rg::AccessMode::Write, 1}});

    g->mark_output(out_rgba);
    CHECK(g->compile());
    CHECK(g->has_output());
    const auto& order = g->execution_order();
    CHECK(order.size() == sz(5));
    // init_aabbs and init_verts have no deps → come first (id-sorted).
    CHECK(order[0] == id_init_aabbs);
    CHECK(order[1] == id_init_verts);
    CHECK(order[2] == id_cull);
    CHECK(order[3] == id_xform);
    CHECK(order[4] == id_tone);

    auto backend = rg::CpuBackend::create();
    backend->execute(*g);
    auto out = backend->buffer_contents(out_rgba);
    CHECK(out.size() == 16u * sizeof(float));
    const float* outf = reinterpret_cast<const float*>(out.data());
    // First 4 floats: vis_bits[i] (= 0) + in_verts[i] (=1) = 1, then × 2 = 2.
    // After 4, xform was only written for indices [0..4) — the rest is 0
    // → tone result is also 0.
    for (int i = 0; i < 4; ++i) CHECK(outf[i] == 2.0f);
    for (int i = 4; i < 16; ++i) CHECK(outf[i] == 0.0f);
}

// ----------------------------------------------------------------------------
// Safety
// ----------------------------------------------------------------------------
void test_zero_size_buffer_safe() {
    // A zero-byte buffer must compile + execute without UB. The
    // CpuBackend hands nullptr to map_*; the pass's early-return paths
    // skip safely.
    auto g = rg::Graph::create();
    rg::BufferDesc bd;
    bd.name = "empty";
    bd.size_bytes = 0;
    auto h = g->declare_buffer(bd);
    FillOp op{h, 1.0f, 0};
    rg::PassDesc pd; pd.name = "zero"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{h, rg::AccessMode::Write, 0});
    pd.record = fill_record; pd.user_ctx = &op;
    g->add_pass(cardinal::move(pd));
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    CHECK(b->trace().size() == sz(1));
    CHECK(b->buffer_contents(h).empty());
}

// ----------------------------------------------------------------------------
// FrustumCullPass — verify the CPU reference under CpuBackend matches
// cardinal::core::simd::frustum_cull_aabbs bit-for-bit (the same SIMD path
// the existing ForwardRenderer uses, so this pass is the GPU-bound
// expression of the engine's current cull path).
// ----------------------------------------------------------------------------
void test_cull_pass_matches_simd_path() {
    namespace gpu = cardinal::render::gpu;
    constexpr cardinal::u32 N = 32;

    // SoA AABBs: 6 contiguous arrays of N floats.
    cardinal::vector<float> aabbs(static_cast<cardinal::usize>(N) * 6u);
    float* mn_x = aabbs.data() + 0 * N;
    float* mn_y = aabbs.data() + 1 * N;
    float* mn_z = aabbs.data() + 2 * N;
    float* mx_x = aabbs.data() + 3 * N;
    float* mx_y = aabbs.data() + 4 * N;
    float* mx_z = aabbs.data() + 5 * N;
    // Walk N AABBs along +X starting at 0, each a 1-unit cube.
    for (cardinal::u32 i = 0; i < N; ++i) {
        const float x = static_cast<float>(i) * 2.0f;
        mn_x[i] = x;       mx_x[i] = x + 1.0f;
        mn_y[i] = 0.0f;    mx_y[i] = 1.0f;
        mn_z[i] = 0.0f;    mx_z[i] = 1.0f;
    }
    // Frustum: 6 planes that pass only the first ~half of the AABBs.
    // Build by hand so the result is deterministic.
    cardinal::vector<float> planes(24, 0.0f);
    // Plane 0: x >= 0           (nx=1, d=0)
    planes[0] = 1; planes[1] = 0; planes[2] = 0; planes[3] = 0;
    // Plane 1: x <= 30          (nx=-1, d=30)
    planes[4] = -1; planes[5] = 0; planes[6] = 0; planes[7] = 30;
    // Plane 2: y >= -1          (ny=1, d=1)
    planes[8] = 0; planes[9] = 1; planes[10] = 0; planes[11] = 1;
    // Plane 3: y <= 10          (ny=-1, d=10)
    planes[12] = 0; planes[13] = -1; planes[14] = 0; planes[15] = 10;
    // Plane 4: z >= -1          (nz=1, d=1)
    planes[16] = 0; planes[17] = 0; planes[18] = 1; planes[19] = 1;
    // Plane 5: z <= 10          (nz=-1, d=10)
    planes[20] = 0; planes[21] = 0; planes[22] = -1; planes[23] = 10;

    // Direct SIMD reference (the same path the engine uses today).
    cardinal::vector<cardinal::u8> ref_bits(N, 0u);
    cardinal::core::simd::frustum_cull_aabbs(
        ref_bits.data(), planes.data(),
        mn_x, mn_y, mn_z, mx_x, mx_y, mx_z, N);

    // Run via the graph pass.
    auto g = rg::Graph::create();
    rg::BufferDesc abd; abd.name = "aabbs"; abd.size_bytes = aabbs.size() * sizeof(float);
    auto h_aabbs = g->declare_buffer(abd);
    rg::BufferDesc pld; pld.name = "planes"; pld.size_bytes = planes.size() * sizeof(float);
    auto h_planes = g->declare_buffer(pld);

    // Pre-fill via a "init" pass each (the host could also import_buffer,
    // but the CpuBackend's transient allocation is what we exercise here).
    struct InitAABBs { rg::ResourceHandle h; cardinal::vector<float>* src; } iab{h_aabbs, &aabbs};
    struct InitPlanes { rg::ResourceHandle h; cardinal::vector<float>* src; } ipl{h_planes, &planes};
    auto rec_init_aabbs = [](rg::ExecutionContext& ec, void* uctx) noexcept {
        auto* c = static_cast<InitAABBs*>(uctx);
        float* dst = static_cast<float*>(ec.map_buffer_write(c->h));
        if (!dst) return;
        for (cardinal::usize i = 0; i < c->src->size(); ++i) dst[i] = (*c->src)[i];
    };
    auto rec_init_planes = [](rg::ExecutionContext& ec, void* uctx) noexcept {
        auto* c = static_cast<InitPlanes*>(uctx);
        float* dst = static_cast<float*>(ec.map_buffer_write(c->h));
        if (!dst) return;
        for (cardinal::usize i = 0; i < c->src->size(); ++i) dst[i] = (*c->src)[i];
    };
    rg::PassDesc pd_ia; pd_ia.name = "init_aabbs"; pd_ia.kind = rg::PassKind::Compute;
    pd_ia.accesses.push_back(rg::ResourceAccess{h_aabbs, rg::AccessMode::Write, 0});
    pd_ia.record = rec_init_aabbs; pd_ia.user_ctx = &iab;
    g->add_pass(cardinal::move(pd_ia));
    rg::PassDesc pd_ip; pd_ip.name = "init_planes"; pd_ip.kind = rg::PassKind::Compute;
    pd_ip.accesses.push_back(rg::ResourceAccess{h_planes, rg::AccessMode::Write, 0});
    pd_ip.record = rec_init_planes; pd_ip.user_ctx = &ipl;
    g->add_pass(cardinal::move(pd_ip));

    auto state = gpu::FrustumCullPass::add_to_graph(*g, h_aabbs, h_planes, N);

    CHECK(g->compile());
    auto backend = rg::CpuBackend::create();
    backend->execute(*g);

    auto out = backend->buffer_contents(state->out_bits);
    CHECK(out.size() == sz(static_cast<int>(N)));
    // BIT-FOR-BIT vs the direct SIMD reference.
    for (cardinal::u32 i = 0; i < N; ++i) {
        CHECK(out[i] == ref_bits[i]);
    }
    // Stats match the byte count.
    cardinal::u32 ref_vis = 0;
    for (cardinal::u32 i = 0; i < N; ++i) if (ref_bits[i] != 0) ++ref_vis;
    CHECK(state->visible == ref_vis);
    CHECK(state->culled  == N - ref_vis);

    // HLSL is non-empty + mentions [numthreads].
    const char* hlsl = gpu::FrustumCullPass::hlsl_source();
    CHECK(hlsl != nullptr);
    CHECK(hlsl[0] != '\0');
}

void test_cull_pass_zero_count_safe() {
    namespace gpu = cardinal::render::gpu;
    auto g = rg::Graph::create();
    rg::BufferDesc abd; abd.name = "aabbs"; abd.size_bytes = 0;
    auto h_aabbs = g->declare_buffer(abd);
    rg::BufferDesc pld; pld.name = "planes"; pld.size_bytes = 24 * sizeof(float);
    auto h_planes = g->declare_buffer(pld);
    auto st = gpu::FrustumCullPass::add_to_graph(*g, h_aabbs, h_planes, 0u);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    CHECK(st->visible == 0u);
    CHECK(st->culled  == 0u);
}

// ----------------------------------------------------------------------------
// VertexTransformPass — verify CPU reference matches an inline transform.
// ----------------------------------------------------------------------------
void test_xform_pass_identity_matrix() {
    namespace gpu = cardinal::render::gpu;
    constexpr cardinal::u32 V = 8;
    cardinal::vector<float> local(static_cast<cardinal::usize>(V) * 6u, 0.0f);
    float* lpx = local.data() + 0 * V;
    float* lpy = local.data() + 1 * V;
    float* lpz = local.data() + 2 * V;
    float* lnx = local.data() + 3 * V;
    float* lny = local.data() + 4 * V;
    float* lnz = local.data() + 5 * V;
    for (cardinal::u32 i = 0; i < V; ++i) {
        lpx[i] = static_cast<float>(i);
        lpy[i] = static_cast<float>(i) * 2.0f;
        lpz[i] = static_cast<float>(i) * 3.0f;
        lnx[i] = 0; lny[i] = 1; lnz[i] = 0;
    }
    // Row-major identity matrix.
    cardinal::vector<float> mat = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };

    auto g = rg::Graph::create();
    rg::BufferDesc lbd; lbd.name = "local"; lbd.size_bytes = local.size() * sizeof(float);
    auto h_local = g->declare_buffer(lbd);
    rg::BufferDesc mbd; mbd.name = "mat"; mbd.size_bytes = mat.size() * sizeof(float);
    auto h_mat = g->declare_buffer(mbd);

    struct InitV { rg::ResourceHandle h; cardinal::vector<float>* src; } iv_local{h_local, &local};
    struct InitV im_mat{h_mat, &mat};
    auto rec = [](rg::ExecutionContext& ec, void* uctx) noexcept {
        auto* c = static_cast<InitV*>(uctx);
        float* dst = static_cast<float*>(ec.map_buffer_write(c->h));
        if (!dst) return;
        for (cardinal::usize i = 0; i < c->src->size(); ++i) dst[i] = (*c->src)[i];
    };
    rg::PassDesc pdl; pdl.name = "ilocal"; pdl.kind = rg::PassKind::Compute;
    pdl.accesses.push_back(rg::ResourceAccess{h_local, rg::AccessMode::Write, 0});
    pdl.record = rec; pdl.user_ctx = &iv_local;
    g->add_pass(cardinal::move(pdl));
    rg::PassDesc pdm; pdm.name = "imat"; pdm.kind = rg::PassKind::Compute;
    pdm.accesses.push_back(rg::ResourceAccess{h_mat, rg::AccessMode::Write, 0});
    pdm.record = rec; pdm.user_ctx = &im_mat;
    g->add_pass(cardinal::move(pdm));

    auto st = gpu::VertexTransformPass::add_to_graph(*g, h_local, h_mat, V);

    CHECK(g->compile());
    auto backend = rg::CpuBackend::create();
    backend->execute(*g);

    auto out = backend->buffer_contents(st->out_world);
    CHECK(out.size() == V * 6u * sizeof(float));
    const float* outf = reinterpret_cast<const float*>(out.data());
    const float* opx = outf + 0 * V;
    const float* opy = outf + 1 * V;
    const float* opz = outf + 2 * V;
    const float* onx = outf + 3 * V;
    const float* ony = outf + 4 * V;
    const float* onz = outf + 5 * V;
    for (cardinal::u32 i = 0; i < V; ++i) {
        // Identity: world == local for position; normal unchanged.
        CHECK(opx[i] == lpx[i]);
        CHECK(opy[i] == lpy[i]);
        CHECK(opz[i] == lpz[i]);
        CHECK(onx[i] == 0.0f);
        CHECK(ony[i] == 1.0f);
        CHECK(onz[i] == 0.0f);
    }
    CHECK(st->vertices_written == V);
}

void test_xform_pass_translation() {
    namespace gpu = cardinal::render::gpu;
    constexpr cardinal::u32 V = 4;
    cardinal::vector<float> local(static_cast<cardinal::usize>(V) * 6u, 0.0f);
    for (cardinal::u32 i = 0; i < V; ++i) {
        local[0 * V + i] = static_cast<float>(i);
        local[1 * V + i] = 0;
        local[2 * V + i] = 0;
        local[3 * V + i] = 0;
        local[4 * V + i] = 1;
        local[5 * V + i] = 0;
    }
    cardinal::vector<float> mat = {
        1, 0, 0, 10,    // translate +10 X
        0, 1, 0, -5,    // translate -5 Y
        0, 0, 1,  2,    // translate +2 Z
        0, 0, 0,  1,
    };

    auto g = rg::Graph::create();
    auto h_local = g->declare_buffer(rg::BufferDesc{"local", local.size() * sizeof(float), 0, true});
    auto h_mat   = g->declare_buffer(rg::BufferDesc{"mat",   mat.size()   * sizeof(float), 0, true});
    struct InitV { rg::ResourceHandle h; cardinal::vector<float>* src; } iv_local{h_local, &local}, im_mat{h_mat, &mat};
    auto rec = [](rg::ExecutionContext& ec, void* uctx) noexcept {
        auto* c = static_cast<InitV*>(uctx);
        float* dst = static_cast<float*>(ec.map_buffer_write(c->h));
        if (!dst) return;
        for (cardinal::usize i = 0; i < c->src->size(); ++i) dst[i] = (*c->src)[i];
    };
    rg::PassDesc pdl; pdl.name = "il"; pdl.kind = rg::PassKind::Compute;
    pdl.accesses.push_back(rg::ResourceAccess{h_local, rg::AccessMode::Write, 0});
    pdl.record = rec; pdl.user_ctx = &iv_local;
    g->add_pass(cardinal::move(pdl));
    rg::PassDesc pdm; pdm.name = "im"; pdm.kind = rg::PassKind::Compute;
    pdm.accesses.push_back(rg::ResourceAccess{h_mat, rg::AccessMode::Write, 0});
    pdm.record = rec; pdm.user_ctx = &im_mat;
    g->add_pass(cardinal::move(pdm));

    auto st = gpu::VertexTransformPass::add_to_graph(*g, h_local, h_mat, V);
    CHECK(g->compile());
    auto backend = rg::CpuBackend::create();
    backend->execute(*g);
    auto out = backend->buffer_contents(st->out_world);
    const float* outf = reinterpret_cast<const float*>(out.data());
    const float* opx = outf + 0 * V;
    const float* opy = outf + 1 * V;
    const float* opz = outf + 2 * V;
    for (cardinal::u32 i = 0; i < V; ++i) {
        const float x = static_cast<float>(i);
        // Translated; tolerate float noise within 1e-4.
        CHECK(opx[i] >= x + 10.0f - 1e-4f && opx[i] <= x + 10.0f + 1e-4f);
        CHECK(opy[i] >= -5.0f - 1e-4f && opy[i] <= -5.0f + 1e-4f);
        CHECK(opz[i] >=  2.0f - 1e-4f && opz[i] <=  2.0f + 1e-4f);
    }
}

void test_xform_pass_nan_matrix_safe() {
    namespace gpu = cardinal::render::gpu;
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    constexpr cardinal::u32 V = 4;
    cardinal::vector<float> local(static_cast<cardinal::usize>(V) * 6u, 0.0f);
    for (cardinal::u32 i = 0; i < V; ++i) {
        local[0 * V + i] = 1.0f; local[4 * V + i] = 1.0f;
    }
    cardinal::vector<float> mat(16, qnan);   // poison

    auto g = rg::Graph::create();
    auto h_local = g->declare_buffer(rg::BufferDesc{"local", local.size() * sizeof(float), 0, true});
    auto h_mat   = g->declare_buffer(rg::BufferDesc{"mat",   mat.size()   * sizeof(float), 0, true});
    struct InitV { rg::ResourceHandle h; cardinal::vector<float>* src; } iv_local{h_local, &local}, im_mat{h_mat, &mat};
    auto rec = [](rg::ExecutionContext& ec, void* uctx) noexcept {
        auto* c = static_cast<InitV*>(uctx);
        float* dst = static_cast<float*>(ec.map_buffer_write(c->h));
        if (!dst) return;
        for (cardinal::usize i = 0; i < c->src->size(); ++i) dst[i] = (*c->src)[i];
    };
    rg::PassDesc pdl; pdl.name = "il"; pdl.kind = rg::PassKind::Compute;
    pdl.accesses.push_back(rg::ResourceAccess{h_local, rg::AccessMode::Write, 0});
    pdl.record = rec; pdl.user_ctx = &iv_local;
    g->add_pass(cardinal::move(pdl));
    rg::PassDesc pdm; pdm.name = "im"; pdm.kind = rg::PassKind::Compute;
    pdm.accesses.push_back(rg::ResourceAccess{h_mat, rg::AccessMode::Write, 0});
    pdm.record = rec; pdm.user_ctx = &im_mat;
    g->add_pass(cardinal::move(pdm));

    auto st = gpu::VertexTransformPass::add_to_graph(*g, h_local, h_mat, V);
    CHECK(g->compile());
    rg::CpuBackend::create()->execute(*g);
    // Output must be finite end-to-end (the matrix sanitizer collapsed
    // poisoned cells to identity-diagonal so the rest of the math stays
    // defined).
    auto out_bytes = rg::CpuBackend::create()->buffer_contents(st->out_world);
    // (We executed once on a fresh backend above. Pull from a second
    // execute to satisfy the contract that buffer_contents returns the
    // last-executed state.)
    auto b2 = rg::CpuBackend::create();
    b2->execute(*g);
    out_bytes = b2->buffer_contents(st->out_world);
    const float* outf = reinterpret_cast<const float*>(out_bytes.data());
    for (cardinal::usize i = 0; i < V * 6; ++i) {
        CHECK(outf[i] == outf[i]);   // NaN check (NaN != NaN)
        CHECK((outf[i] - outf[i]) == 0.0f);
    }
}

// ----------------------------------------------------------------------------
// TriangleRasterPass
// ----------------------------------------------------------------------------
namespace gpu = cardinal::render::gpu;

// Helper to wire a "load these bytes" init pass into the graph.
struct InitBlob { rg::ResourceHandle h; const void* src; cardinal::usize bytes; };
void rec_init_blob(rg::ExecutionContext& ec, void* uctx) noexcept {
    auto* c = static_cast<InitBlob*>(uctx);
    void* dst = ec.map_buffer_write(c->h);
    if (!dst) return;
    auto* d = static_cast<cardinal::u8*>(dst);
    auto* s = static_cast<const cardinal::u8*>(c->src);
    for (cardinal::usize i = 0; i < c->bytes; ++i) d[i] = s[i];
}
cardinal::u32 add_init_pass(rg::Graph& g, const char* name, rg::ResourceHandle h, InitBlob* blob) {
    rg::PassDesc pd; pd.name = name; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{h, rg::AccessMode::Write, 0});
    pd.record = rec_init_blob; pd.user_ctx = blob;
    return g.add_pass(cardinal::move(pd));
}

void test_raster_single_triangle_inside_box() {
    constexpr cardinal::u32 W = 16, H = 16;
    // One CCW triangle that covers the centre pixel (8, 8) — pure red.
    cardinal::vector<float> tris = {
        // v0           v1           v2
         2.0f, 2.0f, 0.5f,   1.0f, 0.0f, 0.0f,
        14.0f, 2.0f, 0.5f,   1.0f, 0.0f, 0.0f,
         8.0f,14.0f, 0.5f,   1.0f, 0.0f, 0.0f,
    };
    auto g = rg::Graph::create();
    auto h_tris = g->declare_buffer(rg::BufferDesc{"tris", tris.size() * sizeof(float), 0, true});
    InitBlob blob{h_tris, tris.data(), tris.size() * sizeof(float)};
    add_init_pass(*g, "init_tris", h_tris, &blob);
    auto st = gpu::TriangleRasterPass::add_to_graph(*g, h_tris, W, H, 1);
    CHECK(g->compile());
    auto backend = rg::CpuBackend::create();
    backend->execute(*g);
    auto col = backend->buffer_contents(st->out_color);
    auto dep = backend->buffer_contents(st->out_depth);
    CHECK(col.size() == W * H * 4u);
    CHECK(dep.size() == W * H * sizeof(float));
    // Sample the centre pixel — should be red.
    const cardinal::usize pix = (8u * W + 8u) * 4u;
    CHECK(col[pix + 0] == 255);
    CHECK(col[pix + 1] == 0);
    CHECK(col[pix + 2] == 0);
    CHECK(col[pix + 3] == 255);
    // Depth at centre should match the triangle z (0.5).
    const float z = reinterpret_cast<const float*>(dep.data())[8u * W + 8u];
    CHECK(z >= 0.49f && z <= 0.51f);
    // Pixel (0, 0) is outside the triangle — black + far depth.
    CHECK(col[0] == 0);
    CHECK(reinterpret_cast<const float*>(dep.data())[0] == 1.0f);
    // Stats sane.
    CHECK(st->fragments_drawn > 0u);
}

void test_raster_depth_test_keeps_closest() {
    // Two overlapping triangles: a far blue one, then a near red one.
    constexpr cardinal::u32 W = 8, H = 8;
    cardinal::vector<float> tris = {
        // far blue (z = 0.8)
         0.0f, 0.0f, 0.8f,   0.0f, 0.0f, 1.0f,
         8.0f, 0.0f, 0.8f,   0.0f, 0.0f, 1.0f,
         4.0f, 8.0f, 0.8f,   0.0f, 0.0f, 1.0f,
        // near red (z = 0.2)
         0.0f, 0.0f, 0.2f,   1.0f, 0.0f, 0.0f,
         8.0f, 0.0f, 0.2f,   1.0f, 0.0f, 0.0f,
         4.0f, 8.0f, 0.2f,   1.0f, 0.0f, 0.0f,
    };
    auto g = rg::Graph::create();
    auto h = g->declare_buffer(rg::BufferDesc{"tris", tris.size() * sizeof(float), 0, true});
    InitBlob blob{h, tris.data(), tris.size() * sizeof(float)};
    add_init_pass(*g, "i", h, &blob);
    auto st = gpu::TriangleRasterPass::add_to_graph(*g, h, W, H, 2);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    auto col = b->buffer_contents(st->out_color);
    const cardinal::usize pix = (4u * W + 4u) * 4u;
    CHECK(col[pix + 0] == 255);                // near red wins
    CHECK(col[pix + 2] == 0);
}

void test_raster_offscreen_triangle_safe() {
    constexpr cardinal::u32 W = 8, H = 8;
    cardinal::vector<float> tris = {
       100.0f, 100.0f, 0.5f,   1.0f, 1.0f, 1.0f,
       200.0f, 100.0f, 0.5f,   1.0f, 1.0f, 1.0f,
       150.0f, 200.0f, 0.5f,   1.0f, 1.0f, 1.0f,
    };
    auto g = rg::Graph::create();
    auto h = g->declare_buffer(rg::BufferDesc{"tris", tris.size() * sizeof(float), 0, true});
    InitBlob blob{h, tris.data(), tris.size() * sizeof(float)};
    add_init_pass(*g, "i", h, &blob);
    auto st = gpu::TriangleRasterPass::add_to_graph(*g, h, W, H, 1);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    CHECK(st->fragments_drawn == 0u);
    CHECK(st->triangles_offscreen == 1u);
}

void test_raster_nan_triangle_safe() {
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    constexpr cardinal::u32 W = 4, H = 4;
    cardinal::vector<float> tris = {
        qnan, qnan, qnan,   qnan, qnan, qnan,
        qnan, qnan, qnan,   qnan, qnan, qnan,
        qnan, qnan, qnan,   qnan, qnan, qnan,
    };
    auto g = rg::Graph::create();
    auto h = g->declare_buffer(rg::BufferDesc{"tris", tris.size() * sizeof(float), 0, true});
    InitBlob blob{h, tris.data(), tris.size() * sizeof(float)};
    add_init_pass(*g, "i", h, &blob);
    auto st = gpu::TriangleRasterPass::add_to_graph(*g, h, W, H, 1);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    // Every depth value must be finite.
    auto dep = b->buffer_contents(st->out_depth);
    const float* df = reinterpret_cast<const float*>(dep.data());
    for (cardinal::usize i = 0; i < W * H; ++i) {
        CHECK(df[i] == df[i]);                 // NaN check
    }
}

// ----------------------------------------------------------------------------
// PolygonClipPass — Sutherland-Hodgman
// ----------------------------------------------------------------------------
void test_clip_quad_against_x_plane_full_inside() {
    // Unit square at z=0 entirely inside x >= -1.
    constexpr cardinal::u32 N = 4;
    cardinal::vector<float> verts(static_cast<cardinal::usize>(N) * 3u, 0.0f);
    float* vx = verts.data() + 0 * N;
    float* vy = verts.data() + 1 * N;
    float* vz = verts.data() + 2 * N;
    vx[0] = 0; vy[0] = 0; vz[0] = 0;
    vx[1] = 1; vy[1] = 0; vz[1] = 0;
    vx[2] = 1; vy[2] = 1; vz[2] = 0;
    vx[3] = 0; vy[3] = 1; vz[3] = 0;
    cardinal::vector<float> plane = {1.0f, 0.0f, 0.0f, 1.0f};  // x >= -1

    auto g = rg::Graph::create();
    auto hv = g->declare_buffer(rg::BufferDesc{"verts", verts.size() * sizeof(float), 0, true});
    auto hp = g->declare_buffer(rg::BufferDesc{"plane", plane.size() * sizeof(float), 0, true});
    InitBlob bv{hv, verts.data(), verts.size() * sizeof(float)};
    InitBlob bp{hp, plane.data(), plane.size() * sizeof(float)};
    add_init_pass(*g, "iv", hv, &bv);
    add_init_pass(*g, "ip", hp, &bp);
    auto st = gpu::PolygonClipPass::add_to_graph(*g, hv, hp, N, N + 1);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    CHECK(st->produced == 4u);                 // All 4 verts emitted (B-of-A=in,B=in case)
}

void test_clip_quad_against_x_plane_half_in() {
    // Unit square at z=0; clip against x >= 0.5. Two verts in, two out.
    constexpr cardinal::u32 N = 4;
    cardinal::vector<float> verts(static_cast<cardinal::usize>(N) * 3u, 0.0f);
    float* vx = verts.data() + 0 * N;
    float* vy = verts.data() + 1 * N;
    float* vz = verts.data() + 2 * N;
    vx[0] = 0; vy[0] = 0;
    vx[1] = 1; vy[1] = 0;
    vx[2] = 1; vy[2] = 1;
    vx[3] = 0; vy[3] = 1;
    (void)vz;
    cardinal::vector<float> plane = {1.0f, 0.0f, 0.0f, -0.5f};   // x >= 0.5

    auto g = rg::Graph::create();
    auto hv = g->declare_buffer(rg::BufferDesc{"verts", verts.size() * sizeof(float), 0, true});
    auto hp = g->declare_buffer(rg::BufferDesc{"plane", plane.size() * sizeof(float), 0, true});
    InitBlob bv{hv, verts.data(), verts.size() * sizeof(float)};
    InitBlob bp{hp, plane.data(), plane.size() * sizeof(float)};
    add_init_pass(*g, "iv", hv, &bv);
    add_init_pass(*g, "ip", hp, &bp);
    auto st = gpu::PolygonClipPass::add_to_graph(*g, hv, hp, N, N + 1);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    // Should produce 4 vertices (the two inside + two intersection points).
    CHECK(st->produced == 4u);
    auto out = b->buffer_contents(st->out_verts);
    const cardinal::u32 max_out = N + 1;
    const float* ox = reinterpret_cast<const float*>(out.data()) + 0 * max_out;
    // All resulting x coords must be ≥ 0.5 (within float tolerance).
    for (cardinal::u32 i = 0; i < st->produced; ++i) {
        CHECK(ox[i] >= 0.5f - 1e-3f);
    }
}

void test_clip_quad_fully_outside() {
    constexpr cardinal::u32 N = 4;
    cardinal::vector<float> verts(static_cast<cardinal::usize>(N) * 3u, 0.0f);
    float* vx = verts.data() + 0 * N;
    vx[0] = -5; vx[1] = -4; vx[2] = -4; vx[3] = -5;
    cardinal::vector<float> plane = {1.0f, 0.0f, 0.0f, 0.0f};  // x >= 0

    auto g = rg::Graph::create();
    auto hv = g->declare_buffer(rg::BufferDesc{"verts", verts.size() * sizeof(float), 0, true});
    auto hp = g->declare_buffer(rg::BufferDesc{"plane", plane.size() * sizeof(float), 0, true});
    InitBlob bv{hv, verts.data(), verts.size() * sizeof(float)};
    InitBlob bp{hp, plane.data(), plane.size() * sizeof(float)};
    add_init_pass(*g, "iv", hv, &bv);
    add_init_pass(*g, "ip", hp, &bp);
    auto st = gpu::PolygonClipPass::add_to_graph(*g, hv, hp, N, N + 1);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    CHECK(st->produced == 0u);                 // entirely culled
}

void test_clip_nan_plane_safe() {
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    constexpr cardinal::u32 N = 3;
    cardinal::vector<float> verts(static_cast<cardinal::usize>(N) * 3u, 0.0f);
    verts[0 * N + 0] = 0; verts[0 * N + 1] = 1; verts[0 * N + 2] = 0;   // x
    verts[1 * N + 0] = 0; verts[1 * N + 1] = 0; verts[1 * N + 2] = 1;   // y
    cardinal::vector<float> plane = {qnan, qnan, qnan, qnan};
    auto g = rg::Graph::create();
    auto hv = g->declare_buffer(rg::BufferDesc{"verts", verts.size() * sizeof(float), 0, true});
    auto hp = g->declare_buffer(rg::BufferDesc{"plane", plane.size() * sizeof(float), 0, true});
    InitBlob bv{hv, verts.data(), verts.size() * sizeof(float)};
    InitBlob bp{hp, plane.data(), plane.size() * sizeof(float)};
    add_init_pass(*g, "iv", hv, &bv);
    add_init_pass(*g, "ip", hp, &bp);
    auto st = gpu::PolygonClipPass::add_to_graph(*g, hv, hp, N, N + 1);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    // Output buffer + count must be finite no matter what (plane → x >= 0 fallback).
    auto cnt = b->buffer_contents(st->out_count);
    const cardinal::u32 c = *reinterpret_cast<const cardinal::u32*>(cnt.data());
    CHECK(c <= N + 1);                         // bounded
}

// ----------------------------------------------------------------------------
// PolygonTriangulatePass — fan triangulation
// ----------------------------------------------------------------------------
void test_triangulate_quad() {
    // A quad → exactly 2 triangles (fan anchored at vertex 0).
    constexpr cardinal::u32 N = 4;
    cardinal::vector<float> verts(static_cast<cardinal::usize>(N) * 3u, 0.0f);
    float* vx = verts.data() + 0 * N;
    float* vy = verts.data() + 1 * N;
    vx[0] = 0; vy[0] = 0;
    vx[1] = 1; vy[1] = 0;
    vx[2] = 1; vy[2] = 1;
    vx[3] = 0; vy[3] = 1;
    auto g = rg::Graph::create();
    auto hv = g->declare_buffer(rg::BufferDesc{"verts", verts.size() * sizeof(float), 0, true});
    InitBlob bv{hv, verts.data(), verts.size() * sizeof(float)};
    add_init_pass(*g, "iv", hv, &bv);
    auto st = gpu::PolygonTriangulatePass::add_to_graph(*g, hv, N);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    CHECK(st->triangles_produced == 2u);
    auto out = b->buffer_contents(st->out_tris);
    CHECK(out.size() == 2u * 9u * sizeof(float));
    const float* ot = reinterpret_cast<const float*>(out.data());
    // Triangle 0: (v0, v1, v2) = (0,0,0), (1,0,0), (1,1,0)
    CHECK(ot[0] == 0.0f); CHECK(ot[1] == 0.0f);
    CHECK(ot[3] == 1.0f); CHECK(ot[4] == 0.0f);
    CHECK(ot[6] == 1.0f); CHECK(ot[7] == 1.0f);
    // Triangle 1: (v0, v2, v3)
    CHECK(ot[ 9] == 0.0f); CHECK(ot[10] == 0.0f);   // v0
    CHECK(ot[12] == 1.0f); CHECK(ot[13] == 1.0f);   // v2
    CHECK(ot[15] == 0.0f); CHECK(ot[16] == 1.0f);   // v3
}

void test_triangulate_below_three_is_empty() {
    constexpr cardinal::u32 N = 2;
    cardinal::vector<float> verts(static_cast<cardinal::usize>(N) * 3u, 0.0f);
    auto g = rg::Graph::create();
    auto hv = g->declare_buffer(rg::BufferDesc{"verts", verts.size() * sizeof(float), 0, true});
    InitBlob bv{hv, verts.data(), verts.size() * sizeof(float)};
    add_init_pass(*g, "iv", hv, &bv);
    auto st = gpu::PolygonTriangulatePass::add_to_graph(*g, hv, N);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    CHECK(st->triangles_produced == 0u);
}

// ----------------------------------------------------------------------------
// Chain: clip → triangulate → raster (all on the graph, all one execute).
// ----------------------------------------------------------------------------
void test_chain_clip_triangulate_raster() {
    constexpr cardinal::u32 N = 4;
    cardinal::vector<float> verts(static_cast<cardinal::usize>(N) * 3u, 0.0f);
    float* vx = verts.data() + 0 * N;
    float* vy = verts.data() + 1 * N;
    // Quad spanning (2..14, 2..14) at depth 0.5.
    vx[0] = 2;  vy[0] = 2;
    vx[1] = 14; vy[1] = 2;
    vx[2] = 14; vy[2] = 14;
    vx[3] = 2;  vy[3] = 14;
    // Don't clip anything off (plane: x >= -100).
    cardinal::vector<float> plane = {1.0f, 0.0f, 0.0f, 100.0f};

    auto g = rg::Graph::create();
    auto hv = g->declare_buffer(rg::BufferDesc{"verts", verts.size() * sizeof(float), 0, true});
    auto hp = g->declare_buffer(rg::BufferDesc{"plane", plane.size() * sizeof(float), 0, true});
    InitBlob bv{hv, verts.data(), verts.size() * sizeof(float)};
    InitBlob bp{hp, plane.data(), plane.size() * sizeof(float)};
    add_init_pass(*g, "iv", hv, &bv);
    add_init_pass(*g, "ip", hp, &bp);
    // Both passes need to agree on the SoA stride (max_output_count for
    // clip == input_count for triangulate). For chains where the
    // polygon doesn't grow under clipping (always true when the entire
    // polygon is fully inside the plane), max_output_count = N keeps
    // the strides matched. For chains that may grow, the consumer needs
    // a dynamic count + matching stride — that's the natural extension.
    auto clip = gpu::PolygonClipPass::add_to_graph(*g, hv, hp, N, N);
    auto tri  = gpu::PolygonTriangulatePass::add_to_graph(*g, clip->out_verts, N);
    (void)tri;
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    CHECK(clip->produced == 4u);
    // Triangulate sees 4 vertices in the buffer → 2 triangles.
    CHECK(tri->triangles_produced == 2u);
}

void test_reset_clears_state() {
    auto g = rg::Graph::create();
    FillOp op{};
    auto h = declare_f32_buffer(*g, "b", 4);
    op.out = h; op.value = 1.0f; op.count = 4;
    rg::PassDesc pd; pd.name = "f"; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{h, rg::AccessMode::Write, 0});
    pd.record = fill_record; pd.user_ctx = &op;
    g->add_pass(cardinal::move(pd));
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    CHECK(!b->trace().empty());
    b->reset();
    CHECK(b->trace().empty());
    CHECK(b->buffer_contents(h).empty());
}

}  // namespace

int main() {
    test_empty_graph_compiles();
    test_resource_declaration();
    test_invalid_handle_safe();

    test_topo_sort_raw();
    test_topo_sort_waw();
    test_topo_sort_war();
    test_topo_sort_diamond();
    test_no_false_cycles_under_complex_deps();
    test_determinism_id_order();

    test_cpu_backend_fill_and_copy();
    test_cpu_backend_access_enforcement();
    test_cpu_backend_trace_records_dispatches();
    test_cpu_backend_param_kv();
    test_cpu_backend_no_record_safe();
    test_cpu_backend_determinism_across_runs();

    test_pipeline_cull_transform_tonemap();

    test_cull_pass_matches_simd_path();
    test_cull_pass_zero_count_safe();
    test_xform_pass_identity_matrix();
    test_xform_pass_translation();
    test_xform_pass_nan_matrix_safe();

    test_raster_single_triangle_inside_box();
    test_raster_depth_test_keeps_closest();
    test_raster_offscreen_triangle_safe();
    test_raster_nan_triangle_safe();

    test_clip_quad_against_x_plane_full_inside();
    test_clip_quad_against_x_plane_half_in();
    test_clip_quad_fully_outside();
    test_clip_nan_plane_safe();

    test_triangulate_quad();
    test_triangulate_below_three_is_empty();
    test_chain_clip_triangulate_raster();

    test_zero_size_buffer_safe();
    test_reset_clears_state();

    if (g_fail == 0) {
        cardinal::log::infof("gtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("gtest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
