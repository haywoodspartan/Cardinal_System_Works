// =============================================================================
// Cardinal — SkinPass (GPU linear-blend skinning). See gpu_skin.hpp.
// =============================================================================
#include <cardinal/render/gpu_skin.hpp>
#include <cardinal/scene/scene.hpp>
#include <cardinal/scene/skin.hpp>
#include <cardinal/core/std/cstring.hpp>
#include <cardinal/core/std/utility.hpp>

namespace cardinal::render::gpu {
namespace rg = cardinal::render::graph;

namespace {

static_assert(sizeof(cardinal::scene::Vertex) == SkinPass::kVertexBytes,
              "SkinPass HLSL assumes the 44-byte scene::Vertex layout");
static_assert(sizeof(cardinal::scene::skin::Influence) == SkinPass::kInfluenceBytes,
              "SkinPass HLSL assumes the 24-byte Influence layout");
static_assert(sizeof(cardinal::scene::Mat4) == SkinPass::kMat4Bytes,
              "SkinPass HLSL assumes a 64-byte Mat4");

inline void put_u32(cardinal::vector<cardinal::u8>& v, cardinal::u32 x) {
    cardinal::u8 b[4];
    cardinal::memcpy(b, &x, 4);
    v.push_back(b[0]); v.push_back(b[1]); v.push_back(b[2]); v.push_back(b[3]);
}

void skin_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<SkinPass::State*>(uctx);
    using cardinal::scene::Vertex;
    using cardinal::scene::Mat4;
    using cardinal::scene::skin::Influence;

    const auto* in   = static_cast<const Vertex*>   (ec.map_buffer_read (st->in_verts));
    const auto* infl = static_cast<const Influence*>(ec.map_buffer_read (st->in_influences));
    const auto* pal  = static_cast<const Mat4*>     (ec.map_buffer_read (st->in_palette));
    auto*       out  = static_cast<Vertex*>         (ec.map_buffer_write(st->out_verts));
    if (!in || !infl || !pal || !out) { ec.dispatch(0, 1, 1); return; }

    // Clamp to what the buffers actually hold — the record is the trust
    // boundary for host-declared counts.
    cardinal::u32 n = st->vertex_count;
    const cardinal::usize cap_in   = ec.buffer_size(st->in_verts)      / sizeof(Vertex);
    const cardinal::usize cap_infl = ec.buffer_size(st->in_influences) / sizeof(Influence);
    const cardinal::usize cap_out  = ec.buffer_size(st->out_verts)     / sizeof(Vertex);
    if (cap_in   < n) n = static_cast<cardinal::u32>(cap_in);
    if (cap_infl < n) n = static_cast<cardinal::u32>(cap_infl);
    if (cap_out  < n) n = static_cast<cardinal::u32>(cap_out);
    cardinal::u32 jc = st->joint_count;
    const cardinal::usize cap_pal = ec.buffer_size(st->in_palette) / sizeof(Mat4);
    if (cap_pal < jc) jc = static_cast<cardinal::u32>(cap_pal);

    cardinal::scene::skin::skin_vertices(in, infl, n, pal, jc, out);
    ec.dispatch(SkinPass::grid_groups(n), 1, 1);
}

}  // namespace

cardinal::shared_ptr<SkinPass::State> SkinPass::add_to_graph(
    rg::Graph& g,
    rg::ResourceHandle in_verts,
    rg::ResourceHandle in_influences,
    rg::ResourceHandle in_palette,
    cardinal::u32 vertex_count,
    cardinal::u32 joint_count) {
    if (!in_verts.is_valid() || !in_influences.is_valid() || !in_palette.is_valid())
        return nullptr;
    if (vertex_count == 0) return nullptr;

    auto st = cardinal::shared_ptr<State>(new State());
    st->in_verts      = in_verts;
    st->in_influences = in_influences;
    st->in_palette    = in_palette;
    st->vertex_count  = vertex_count;
    st->joint_count   = joint_count;

    rg::BufferDesc od;
    od.name         = "skin.out";
    od.size_bytes   = static_cast<cardinal::usize>(vertex_count) * kVertexBytes;
    od.stride_bytes = kVertexBytes;
    st->out_verts   = g.declare_buffer(od);

    rg::PassDesc pd;
    pd.name = "SkinPass";
    pd.kind = rg::PassKind::Compute;
    // Binding order == access order: t0 verts, t1 influences, t2 palette, u0 out.
    pd.accesses.push_back(rg::ResourceAccess{in_verts,      rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{in_influences, rg::AccessMode::Read,  1});
    pd.accesses.push_back(rg::ResourceAccess{in_palette,    rg::AccessMode::Read,  2});
    pd.accesses.push_back(rg::ResourceAccess{st->out_verts, rg::AccessMode::Write, 3});
    pd.record     = skin_record;
    pd.user_ctx   = st.get();
    pd.dispatch_x = grid_groups(vertex_count);      // 256-thread blocks, grid-stride
    pd.compute_hlsl = hlsl_source();
    pd.push_constant_size = 16;
    put_u32(pd.push_constants, vertex_count);
    put_u32(pd.push_constants, joint_count);
    put_u32(pd.push_constants, pd.dispatch_x);      // groups
    put_u32(pd.push_constants, 0u);
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* SkinPass::hlsl_source() noexcept {
    return
    "// Cardinal - SkinPass.hlsl - one thread per vertex.\n"
    "// Mirrors scene::skin::skin_point op-for-op (blend palette, then transform).\n"
    "ByteAddressBuffer   InVerts   : register(t0);  // 44 B/vertex\n"
    "ByteAddressBuffer   InInfl    : register(t1);  // 24 B/vertex: uint2 joints + float4 w\n"
    "ByteAddressBuffer   InPalette : register(t2);  // 64 B/joint, column-major\n"
    "RWByteAddressBuffer OutVerts  : register(u0);  // 44 B/vertex\n"
    "struct PushT { uint n; uint joints; uint groups; uint p1; };\n"
    "[[vk::push_constant]] ConstantBuffer<PushT> pc : register(b0);\n"
    "void load_cols(uint j, out float4 c0, out float4 c1, out float4 c2, out float4 c3) {\n"
    "  if (j < pc.joints) {\n"
    "    uint b = j * 64u;\n"
    "    c0 = asfloat(InPalette.Load4(b));\n"
    "    c1 = asfloat(InPalette.Load4(b + 16u));\n"
    "    c2 = asfloat(InPalette.Load4(b + 32u));\n"
    "    c3 = asfloat(InPalette.Load4(b + 48u));\n"
    "  } else {\n"
    "    c0 = float4(1,0,0,0); c1 = float4(0,1,0,0);\n"
    "    c2 = float4(0,0,1,0); c3 = float4(0,0,0,1);\n"
    "  }\n"
    "}\n"
    "// 256-thread blocks + grid-stride loop over the vertex range.\n"
    "[numthreads(256,1,1)]\n"
    "void CSMain(uint3 tid : SV_DispatchThreadID) {\n"
    "  uint stride = max(pc.groups, 1u) * 256u;\n"
    "  for (uint v = tid.x; v < pc.n; v += stride) {\n"
    "  uint vb = v * 44u;\n"
    "  float3 pos = asfloat(InVerts.Load3(vb));\n"
    "  float3 nrm = asfloat(InVerts.Load3(vb + 12u));\n"
    "  float3 col = asfloat(InVerts.Load3(vb + 24u));\n"
    "  float2 uv  = asfloat(InVerts.Load2(vb + 36u));\n"
    "  uint ib = v * 24u;\n"
    "  uint2  jj = InInfl.Load2(ib);\n"
    "  float4 w  = asfloat(InInfl.Load4(ib + 8u));\n"
    "  uint j[4]; j[0] = jj.x & 0xFFFFu; j[1] = jj.x >> 16; j[2] = jj.y & 0xFFFFu; j[3] = jj.y >> 16;\n"
    "  float4 a0 = 0, a1 = 0, a2 = 0, a3 = 0;\n"
    "  float total = 0.0;\n"
    "  [unroll] for (uint k = 0; k < 4; ++k) {\n"
    "    float wk = w[k];\n"
    "    if (!isfinite(wk) || wk == 0.0) continue;\n"
    "    total += wk;\n"
    "    float4 c0, c1, c2, c3; load_cols(j[k], c0, c1, c2, c3);\n"
    "    a0 += wk * c0; a1 += wk * c1; a2 += wk * c2; a3 += wk * c3;\n"
    "  }\n"
    "  float3 opos = pos, onrm = nrm;\n"
    "  if (total != 0.0) {\n"
    "    opos = a0.xyz * pos.x + a1.xyz * pos.y + a2.xyz * pos.z + a3.xyz;\n"
    "    float3 n = a0.xyz * nrm.x + a1.xyz * nrm.y + a2.xyz * nrm.z;\n"
    "    float l2 = dot(n, n);\n"
    "    if (l2 > 1.0e-12) onrm = n * (1.0 / sqrt(l2));\n"
    "  }\n"
    "  OutVerts.Store3(vb,       asuint(opos));\n"
    "  OutVerts.Store3(vb + 12u, asuint(onrm));\n"
    "  OutVerts.Store3(vb + 24u, asuint(col));\n"
    "  OutVerts.Store2(vb + 36u, asuint(uv));\n"
    "  }   // grid-stride loop\n"
    "}\n";
}

}  // namespace cardinal::render::gpu
