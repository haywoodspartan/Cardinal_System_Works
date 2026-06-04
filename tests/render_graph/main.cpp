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
#include <cardinal/core/log.hpp>
#include <cardinal/core/utility.hpp>

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

    test_zero_size_buffer_safe();
    test_reset_clears_state();

    if (g_fail == 0) {
        cardinal::log::infof("gtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("gtest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
