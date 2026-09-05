// =============================================================================
// Cardinal — Skeletal skinning regression suite.
//
// Pins the skinning core (scene::skin): influence sanitising, skeleton
// validation, the topological global-transform walk, palette construction
// (identity at bind), and the linear-blend-skin kernel on hand-computed
// rigs. Then pins the render-graph SkinPass: its CpuBackend record must be
// BYTE-IDENTICAL to skin_vertices, and — when a capable GPU is present —
// the authored HLSL kernel dispatched through RhiBackend on both Vulkan and
// D3D12 must match the CPU reference to FP-contraction epsilon (colour + UV
// pass-through exact). No GPU → the GPU half SKIPs and the test stays green.
//
// Exit 0 = all pass.
// =============================================================================

#include <cardinal/scene/skin.hpp>
#include <cardinal/scene/scene.hpp>
#include <cardinal/render/gpu_skin.hpp>
#include <cardinal/render/graph.hpp>
#include <cardinal/rhi/rhi.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/std/cstring.hpp>

#include <vector>
#include <cmath>

namespace {

namespace sk  = cardinal::scene::skin;
namespace rg  = cardinal::render::graph;
namespace gpu = cardinal::render::gpu;
namespace rhi = cardinal::rhi;
using cardinal::scene::Vertex;
using cardinal::scene::Vec3;
using cardinal::scene::Mat4;
using cardinal::u8;
using cardinal::u32;
using cardinal::usize;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("skintest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-4f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
bool ap3(const Vec3& a, const Vec3& b, float e = 1e-4f) {
    return ap(a.x, b.x, e) && ap(a.y, b.y, e) && ap(a.z, b.z, e);
}

constexpr float kPi = 3.14159265358979f;

// ---- fixtures ------------------------------------------------------------
// Two-joint chain: root at origin, "arm" one unit up. Inverse binds derived.
sk::Skeleton make_arm() {
    sk::Skeleton s;
    sk::Joint root; root.name = "root"; root.parent = -1;
    sk::Joint arm;  arm.name  = "arm";  arm.parent  = 0;
    arm.local_bind = Mat4::translation({0.0f, 1.0f, 0.0f});
    s.joints.push_back(root);
    s.joints.push_back(arm);
    s.derive_inverse_bind();
    return s;
}

sk::Influence rigid(u32 joint) {
    sk::Influence i;
    i.joint[0] = static_cast<cardinal::u16>(joint);
    i.weight[0] = 1.0f; i.weight[1] = i.weight[2] = i.weight[3] = 0.0f;
    return i;
}

// ---- Influence::normalized -------------------------------------------------
void test_normalized() {
    sk::Influence a;
    a.weight[0] = 2.0f; a.weight[1] = 2.0f; a.weight[2] = 0.0f; a.weight[3] = 0.0f;
    sk::Influence n = sk::normalized(a);
    CHECK(ap(n.weight[0], 0.5f) && ap(n.weight[1], 0.5f));
    CHECK(n.weight[2] == 0.0f && n.weight[3] == 0.0f);

    // NaN + negative weights are dropped, the rest renormalised.
    sk::Influence b;
    b.weight[0] = std::nanf(""); b.weight[1] = -1.0f; b.weight[2] = 0.25f; b.weight[3] = 0.75f;
    sk::Influence nb = sk::normalized(b);
    CHECK(nb.weight[0] == 0.0f && nb.weight[1] == 0.0f);
    CHECK(ap(nb.weight[2], 0.25f) && ap(nb.weight[3], 0.75f));

    // All-zero → rigid to joint[0] (never a vanishing vertex).
    sk::Influence z;
    z.joint[0] = 7; z.weight[0] = z.weight[1] = z.weight[2] = z.weight[3] = 0.0f;
    sk::Influence nz = sk::normalized(z);
    CHECK(nz.joint[0] == 7 && nz.weight[0] == 1.0f && nz.weight[1] == 0.0f);
}

// ---- Skeleton -------------------------------------------------------------
void test_skeleton() {
    sk::Skeleton s = make_arm();
    CHECK(s.joint_count() == 2u);
    CHECK(s.validate());
    CHECK(s.find("arm") == 1);
    CHECK(s.find("root") == 0);
    CHECK(s.find("nope") == -1);

    // Parent must precede child.
    sk::Skeleton bad = s;
    bad.joints[0].parent = 1;
    CHECK(!bad.validate());
    bad.joints[0].parent = 0;            // self-parent
    CHECK(!bad.validate());
    bad.joints[0].parent = -2;
    CHECK(!bad.validate());

    // Derived inverse bind of "arm" undoes the +1 y translation.
    const Mat4& ib = s.joints[1].inverse_bind;
    CHECK(ap(ib.m[3][1], -1.0f));
    CHECK(ap(ib.m[0][0], 1.0f) && ap(ib.m[1][1], 1.0f) && ap(ib.m[2][2], 1.0f));
}

// ---- compute_global / palette --------------------------------------------
void test_globals_and_palette() {
    // Three-link chain, each +1 y: globals at y = 1, 2, 3.
    sk::Skeleton s;
    for (int i = 0; i < 3; ++i) {
        sk::Joint j; j.parent = i - 1;
        j.local_bind = Mat4::translation({0.0f, 1.0f, 0.0f});
        s.joints.push_back(j);
    }
    s.derive_inverse_bind();
    CHECK(s.validate());

    sk::Pose bind;
    sk::bind_pose(s, bind);
    CHECK(bind.local.size() == 3u);

    std::vector<Mat4> g_std;
    cardinal::vector<Mat4> g;
    sk::compute_global(s, bind, g);
    CHECK(g.size() == 3u);
    CHECK(ap(g[0].m[3][1], 1.0f));
    CHECK(ap(g[1].m[3][1], 2.0f));
    CHECK(ap(g[2].m[3][1], 3.0f));

    // Bind pose ⇒ identity palette.
    cardinal::vector<Mat4> pal;
    sk::compute_palette(s, g, pal);
    CHECK(pal.size() == 3u);
    for (usize j = 0; j < pal.size(); ++j)
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r)
                CHECK(ap(pal[j].m[c][r], (c == r) ? 1.0f : 0.0f, 1e-5f));

    // Empty pose == bind pose (missing joints fall back to local_bind).
    sk::Pose empty;
    cardinal::vector<Mat4> scratch, pal2;
    sk::compute_palette(s, empty, scratch, pal2);
    CHECK(pal2.size() == 3u);
    CHECK(ap(pal2[2].m[3][1], 0.0f, 1e-5f));

    // Pose the ROOT +2 x: every descendant palette carries the +2 x.
    sk::Pose posed = bind;
    posed.local[0] = Mat4::translation({2.0f, 1.0f, 0.0f});
    sk::compute_palette(s, posed, scratch, pal2);
    CHECK(ap(pal2[0].m[3][0], 2.0f));
    CHECK(ap(pal2[2].m[3][0], 2.0f));
    CHECK(ap(pal2[2].m[3][1], 0.0f, 1e-5f));   // y unchanged from bind
}

// ---- skin_point -----------------------------------------------------------
void test_skin_point() {
    sk::Skeleton s = make_arm();
    sk::Pose pose;
    sk::bind_pose(s, pose);
    cardinal::vector<Mat4> scratch, pal;

    // Bend the arm 90° about +Z (in its own local frame, after its bind offset).
    pose.local[1] = Mat4::translation({0.0f, 1.0f, 0.0f})
                  * Mat4::rotation_xyz({0.0f, 0.0f, kPi * 0.5f});
    sk::compute_palette(s, pose, scratch, pal);

    // Point rigid on the arm, one unit above the arm joint: (0,2,0) → arm
    // space (0,1,0) → rotated (-1,0,0) → back to model space (-1,1,0).
    Vec3 p{0.0f, 2.0f, 0.0f}, n{0.0f, 1.0f, 0.0f}, op, on;
    sk::skin_point(rigid(1), pal.data(), static_cast<u32>(pal.size()), p, n, op, on);
    CHECK(ap3(op, {-1.0f, 1.0f, 0.0f}));
    CHECK(ap3(on, {-1.0f, 0.0f, 0.0f}));

    // Rigid on the (unposed) root: untouched.
    sk::skin_point(rigid(0), pal.data(), static_cast<u32>(pal.size()), p, n, op, on);
    CHECK(ap3(op, p) && ap3(on, n));

    // 50/50 blend root ↔ a purely translated arm: half the translation.
    pose.local[1] = Mat4::translation({0.0f, 1.0f, 2.0f});    // +2 z from bind
    sk::compute_palette(s, pose, scratch, pal);
    sk::Influence half;
    half.joint[0] = 0; half.joint[1] = 1;
    half.weight[0] = 0.5f; half.weight[1] = 0.5f; half.weight[2] = half.weight[3] = 0.0f;
    sk::skin_point(half, pal.data(), static_cast<u32>(pal.size()), p, n, op, on);
    CHECK(ap3(op, {0.0f, 2.0f, 1.0f}));
    CHECK(ap3(on, n));

    // Out-of-range joint contributes identity for its weight.
    sk::Influence oob = half;
    oob.joint[1] = 900;
    sk::skin_point(oob, pal.data(), static_cast<u32>(pal.size()), p, n, op, on);
    CHECK(ap3(op, p));

    // Normal is re-normalised under a scaled palette.
    Mat4 big[1] = { Mat4::scaling({3.0f, 3.0f, 3.0f}) };
    sk::skin_point(rigid(0), big, 1, p, n, op, on);
    CHECK(ap3(op, {0.0f, 6.0f, 0.0f}));
    CHECK(ap(on.x*on.x + on.y*on.y + on.z*on.z, 1.0f));

    // Zero total weight → pass-through (NaN weight dropped).
    sk::Influence none;
    none.weight[0] = 0.0f; none.weight[1] = std::nanf("");
    none.weight[2] = 0.0f; none.weight[3] = 0.0f;
    sk::skin_point(none, big, 1, p, n, op, on);
    CHECK(ap3(op, p) && ap3(on, n));
}

// ---- skin_vertices --------------------------------------------------------
void test_skin_vertices() {
    Mat4 pal[2] = { Mat4::identity(), Mat4::translation({1.0f, 0.0f, 0.0f}) };
    Vertex in[3] = {
        {{0,0,0},{0,1,0},{1,0,0},{0.1f,0.2f}},
        {{1,0,0},{0,1,0},{0,1,0},{0.3f,0.4f}},
        {{0,1,0},{0,1,0},{0,0,1},{0.5f,0.6f}},
    };
    sk::Influence infl[3] = { rigid(0), rigid(1), rigid(1) };
    Vertex out[3];
    sk::skin_vertices(in, infl, 3, pal, 2, out);
    CHECK(ap3(out[0].position, {0,0,0}));
    CHECK(ap3(out[1].position, {2,0,0}));
    CHECK(ap3(out[2].position, {1,1,0}));
    // Colour + UV pass through untouched.
    CHECK(out[2].color.z == 1.0f && out[2].texcoord.x == 0.5f && out[2].texcoord.y == 0.6f);

    // In-place (aliasing) gives the same result.
    Vertex alias[3] = { in[0], in[1], in[2] };
    sk::skin_vertices(alias, infl, 3, pal, 2, alias);
    CHECK(ap3(alias[1].position, {2,0,0}) && ap3(alias[2].position, {1,1,0}));

    // Null palette → verbatim copy.
    Vertex copy[3];
    sk::skin_vertices(in, infl, 3, nullptr, 0, copy);
    CHECK(cardinal::memcmp(copy, in, sizeof(in)) == 0);
}

// ---- SkinPass on CpuBackend: byte-identical to the reference -------------
struct SkinScene {
    std::vector<Vertex>        verts;
    std::vector<sk::Influence> infl;
    std::vector<Mat4>          palette;
    std::vector<Vertex>        expect;

    void build(u32 n) {
        sk::Skeleton s = make_arm();
        sk::Pose pose;
        sk::bind_pose(s, pose);
        pose.local[1] = Mat4::translation({0.0f, 1.0f, 0.0f})
                      * Mat4::rotation_xyz({0.3f, 0.0f, kPi * 0.5f});
        cardinal::vector<Mat4> scratch, pal;
        sk::compute_palette(s, pose, scratch, pal);
        palette.assign(pal.begin(), pal.end());

        verts.resize(n); infl.resize(n);
        for (u32 i = 0; i < n; ++i) {
            const float t = static_cast<float>(i) / static_cast<float>(n);
            Vertex& v = verts[i];
            v.position = { 0.5f - t, t * 2.0f, 0.25f * static_cast<float>(i % 5) };
            const float nx = 0.3f, ny = 1.0f - t, nz = t;
            const float l  = std::sqrt(nx*nx + ny*ny + nz*nz);
            v.normal   = { nx / l, ny / l, nz / l };
            v.color    = { t, 1.0f - t, 0.5f };
            v.texcoord = { t, 1.0f - t };
            sk::Influence& f = infl[i];
            f.joint[0] = 0; f.joint[1] = 1; f.joint[2] = 0; f.joint[3] = 0;
            // Mix of rigid-root, rigid-arm and blends; the blend weights are
            // exact binary fractions so CPU/GPU sums agree bit-for-bit.
            switch (i % 4) {
                case 0:  f.weight[0] = 1.0f;  f.weight[1] = 0.0f;  break;
                case 1:  f.weight[0] = 0.0f;  f.weight[1] = 1.0f;  break;
                case 2:  f.weight[0] = 0.5f;  f.weight[1] = 0.5f;  break;
                default: f.weight[0] = 0.25f; f.weight[1] = 0.75f; break;
            }
            f.weight[2] = 0.0f; f.weight[3] = 0.0f;
        }
        expect.resize(n);
        sk::skin_vertices(verts.data(), infl.data(), n,
                          palette.data(), static_cast<u32>(palette.size()),
                          expect.data());
    }

    cardinal::shared_ptr<gpu::SkinPass::State> declare(rg::Graph& g) {
        auto hv = g.import_buffer("skin.verts", verts.data(),
                                  verts.size() * sizeof(Vertex), sizeof(Vertex));
        auto hi = g.import_buffer("skin.infl", infl.data(),
                                  infl.size() * sizeof(sk::Influence), sizeof(sk::Influence));
        auto hp = g.import_buffer("skin.palette", palette.data(),
                                  palette.size() * sizeof(Mat4), sizeof(Mat4));
        return gpu::SkinPass::add_to_graph(g, hv, hi, hp,
                                           static_cast<u32>(verts.size()),
                                           static_cast<u32>(palette.size()));
    }
};

void test_skin_pass_cpu() {
    SkinScene sc;
    sc.build(200);
    auto g  = rg::Graph::create();
    auto st = sc.declare(*g);
    CHECK(st != nullptr);
    if (!st) return;
    g->mark_output(st->out_verts);
    CHECK(g->compile());
    auto be = rg::CpuBackend::create();
    be->execute(*g);
    const auto out = be->buffer_contents(st->out_verts);
    CHECK(out.size() == sc.expect.size() * sizeof(Vertex));
    if (out.size() == sc.expect.size() * sizeof(Vertex))
        CHECK(cardinal::memcmp(out.data(), sc.expect.data(), out.size()) == 0);
    CHECK(be->trace().size() == 1u);
    if (!be->trace().empty()) CHECK(be->trace()[0].dispatched_gx == (200u + 63u) / 64u);

    // Invalid inputs are rejected up front.
    auto g2 = rg::Graph::create();
    CHECK(gpu::SkinPass::add_to_graph(*g2, {}, {}, {}, 10, 1) == nullptr);
    auto hv = g2->import_buffer("v", sc.verts.data(), sizeof(Vertex), sizeof(Vertex));
    CHECK(gpu::SkinPass::add_to_graph(*g2, hv, hv, hv, 0, 1) == nullptr);
}

// ---- SkinPass on a REAL GPU (both backends; SKIP without a device) ------
struct GpuRun {
    rhi::Device*                          dev{nullptr};
    rg::Graph*                            graph{nullptr};
    cardinal::shared_ptr<rg::RhiBackend>* backend{nullptr};
    rhi::Buffer*                          readback{nullptr};
};

void gpu_record(rhi::Swapchain* rec, void* user) noexcept {
    auto& cx = *static_cast<GpuRun*>(user);
    *cx.backend = rg::RhiBackend::create(*cx.dev, *rec);
    (*cx.backend)->set_gpu_execute(true);
    (*cx.backend)->execute(*cx.graph);
    rhi::Buffer* src = (*cx.backend)->gpu_buffer("skin.out");
    if (src && cx.readback) {
        (*cx.backend)->make_copy_source("skin.out");
        rec->copy_buffer(src, 0, cx.readback, 0, cx.readback->size());
    }
}

void test_skin_pass_gpu(rhi::Backend be, const char* be_name) {
    rhi::DeviceDesc dd{};
    dd.backend = be;
    auto dev = rhi::create_device(dd);
    if (!dev) { cardinal::log::infof("skintest", "SKIP %s: no capable device", be_name); return; }
    auto queue = dev->create_async_compute_queue();
    if (!queue) { cardinal::log::infof("skintest", "SKIP %s: no async compute queue", be_name); return; }
    auto fence = dev->create_fence(0);
    if (!fence) { cardinal::log::infof("skintest", "SKIP %s: no fence", be_name); return; }

    SkinScene sc;
    sc.build(1000);
    auto g  = rg::Graph::create();
    auto st = sc.declare(*g);
    CHECK(st != nullptr);
    if (!st) return;
    g->mark_output(st->out_verts);
    CHECK(g->compile());

    rhi::BufferDesc bd{};
    bd.size         = g->buffer(st->out_verts).size_bytes;
    bd.usage        = 0;
    bd.cpu_writable = false;
    bd.cpu_readable = true;
    auto rb = dev->create_buffer(bd);
    CHECK(rb != nullptr);
    if (!rb) return;

    cardinal::shared_ptr<rg::RhiBackend> backend;
    GpuRun cx;
    cx.dev = dev.get(); cx.graph = g.get(); cx.backend = &backend; cx.readback = rb.get();
    const cardinal::u64 target = queue->submit(&gpu_record, &cx, fence.get(), 1u);
    CHECK(target == 1u);
    fence->wait_cpu(1u, 0);
    CHECK(backend != nullptr);
    if (!backend) return;

    // The kernel must have REALLY compiled + dispatched (no stub fallback).
    u32 real = 0;
    for (const auto& ev : backend->events()) if (ev.real_dispatch) ++real;
    CHECK(real == 1u);

    std::vector<u8> got(rb->size());
    CHECK(rb->download(got.data(), got.size(), 0));
    CHECK(got.size() == sc.expect.size() * sizeof(Vertex));
    if (got.size() != sc.expect.size() * sizeof(Vertex)) return;

    const auto* gv = reinterpret_cast<const Vertex*>(got.data());
    u32 bad = 0;
    for (usize i = 0; i < sc.expect.size(); ++i) {
        const Vertex& e = sc.expect[i];
        const Vertex& a = gv[i];
        const bool ok = ap3(a.position, e.position, 1e-4f) && ap3(a.normal, e.normal, 1e-4f)
                     && cardinal::memcmp(&a.color, &e.color, sizeof(Vec3)) == 0
                     && cardinal::memcmp(&a.texcoord, &e.texcoord, sizeof(e.texcoord)) == 0;
        if (!ok && bad++ < 3)
            cardinal::log::errorf("skintest", "%s vertex %zu: gpu (%f %f %f) cpu (%f %f %f)",
                be_name, static_cast<size_t>(i), a.position.x, a.position.y, a.position.z,
                e.position.x, e.position.y, e.position.z);
    }
    CHECK(bad == 0);
    cardinal::log::infof("skintest", "%s: SkinPass GPU vs CPU — %u/%zu mismatches",
                         be_name, bad, static_cast<size_t>(sc.expect.size()));
    backend.reset();
}

}  // namespace

int main() {
    test_normalized();
    test_skeleton();
    test_globals_and_palette();
    test_skin_point();
    test_skin_vertices();
    test_skin_pass_cpu();
    test_skin_pass_gpu(rhi::Backend::Vulkan, "vulkan");
    test_skin_pass_gpu(rhi::Backend::D3D12,  "d3d12");

    cardinal::log::infof("skintest", "%d checks, %d failed", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
