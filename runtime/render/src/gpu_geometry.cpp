#include <cardinal/render/gpu_geometry.hpp>

#include <cardinal/core/cmath.hpp>
#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/utility.hpp>

namespace cardinal::render::gpu {

namespace rg = cardinal::render::graph;

namespace {

inline bool isf(float v) noexcept { return cardinal::isfinite(v); }
inline float fzf(float v, float fb) noexcept { return isf(v) ? v : fb; }

struct V3 { float x, y, z; };
inline V3 mid(const V3& a, const V3& b) noexcept {
    return { 0.5f * (a.x + b.x), 0.5f * (a.y + b.y), 0.5f * (a.z + b.z) };
}
inline float dist2(const V3& a, const V3& b) noexcept {
    const float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

inline V3 sanitize(const V3& v) noexcept {
    return { fzf(v.x, 0.0f), fzf(v.y, 0.0f), fzf(v.z, 0.0f) };
}

// Quantize each component through the tier's precision::Format. FP32 is
// pass-through. Returns the dequantized value + reports the L1 error
// (for stats).
inline V3 quantize_v3(const V3& v, precision::Format f, float& err_acc) noexcept {
    const V3 s = sanitize(v);
    if (f == precision::Format::FP32) return s;
    const float qx = precision::quantise(f, s.x);
    const float qy = precision::quantise(f, s.y);
    const float qz = precision::quantise(f, s.z);
    err_acc += cardinal::abs(qx - s.x) + cardinal::abs(qy - s.y) + cardinal::abs(qz - s.z);
    return { qx, qy, qz };
}

}  // namespace

// =============================================================================
// Tier properties + selector
// =============================================================================
TierProperties properties_of(GeometryTier tier) noexcept {
    switch (tier) {
        case GeometryTier::Fp32:
            return { precision::Format::FP32,     32, 1, 0.0f,         "FP32"     };
        case GeometryTier::Fp16:
            // FP16 RMS error on [-1, 1] ≤ 2^-11 ≈ 4.9e-4
            return { precision::Format::FP16,     16, 2, 5.0e-4f,      "FP16"     };
        case GeometryTier::Fp8:
            // FP8 E4M3 worst-case |x - q(x)| on [-1, 1] ≤ 2^-3 / 2 ≈ 0.0625
            return { precision::Format::FP8_E4M3,  8, 4, 0.0625f,      "FP8 E4M3" };
        case GeometryTier::Fp4:
            // FP4 E2M1 only encodes {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6};
            // worst-case |x - q(x)| on [-1, 1] is 0.5 (e.g. 0.25 → 0 or 0.5).
            return { precision::Format::FP4_E2M1,  4, 8, 0.5f,         "FP4 E2M1" };
    }
    return { precision::Format::FP32, 32, 1, 0.0f, "FP32" };
}

const char* tier_name(GeometryTier tier) noexcept {
    return properties_of(tier).name;
}

GeometryTier select_tier(const GpuCapabilities& caps,
                         GeometryTier max_tier) noexcept
{
    // Walk down from FP4 → FP32 looking for the highest tier the device
    // supports AND that fits under the per-input micro-triangle budget.
    const cardinal::u32 budget = caps.max_micro_triangles_per_input;
    const GeometryTier order[4] = {
        GeometryTier::Fp4, GeometryTier::Fp8,
        GeometryTier::Fp16, GeometryTier::Fp32,
    };
    for (GeometryTier t : order) {
        if (static_cast<cardinal::u32>(t) > static_cast<cardinal::u32>(max_tier)) continue;
        const auto props = properties_of(t);
        if (props.micro_triangles_per_input > budget) continue;
        switch (t) {
            case GeometryTier::Fp32: return t;            // always available
            case GeometryTier::Fp16: if (caps.fp16_supported) return t; break;
            case GeometryTier::Fp8:  if (caps.fp8_supported)  return t; break;
            case GeometryTier::Fp4:  if (caps.fp4_supported)  return t; break;
        }
    }
    return GeometryTier::Fp32;   // unreachable; FP32 always satisfies above
}

// =============================================================================
// Subdivision — emit N micro-triangles from one input
// =============================================================================
namespace {

// Emit one quantized micro-triangle into `out` at `*out_offset`,
// advancing the offset by 9 floats.
inline void emit_tri(float* out, cardinal::usize& off,
                     const V3& a, const V3& b, const V3& c,
                     precision::Format fmt, float& err_acc) noexcept
{
    const V3 qa = quantize_v3(a, fmt, err_acc);
    const V3 qb = quantize_v3(b, fmt, err_acc);
    const V3 qc = quantize_v3(c, fmt, err_acc);
    out[off + 0] = qa.x; out[off + 1] = qa.y; out[off + 2] = qa.z;
    out[off + 3] = qb.x; out[off + 4] = qb.y; out[off + 5] = qb.z;
    out[off + 6] = qc.x; out[off + 7] = qc.y; out[off + 8] = qc.z;
    off += 9;
}

// Longest-edge split: midpoint M of the longest edge of (a, b, c).
// Emit two children sharing M as the new vertex. The opposite vertex
// becomes the third corner of each child.
void emit_split_longest(float* out, cardinal::usize& off,
                        const V3& a, const V3& b, const V3& c,
                        precision::Format fmt, float& err_acc) noexcept
{
    const float dab = dist2(a, b);
    const float dbc = dist2(b, c);
    const float dca = dist2(c, a);
    if (dab >= dbc && dab >= dca) {
        const V3 m = mid(a, b);
        emit_tri(out, off, a, m, c, fmt, err_acc);
        emit_tri(out, off, m, b, c, fmt, err_acc);
    } else if (dbc >= dab && dbc >= dca) {
        const V3 m = mid(b, c);
        emit_tri(out, off, a, b, m, fmt, err_acc);
        emit_tri(out, off, a, m, c, fmt, err_acc);
    } else {
        const V3 m = mid(c, a);
        emit_tri(out, off, a, b, m, fmt, err_acc);
        emit_tri(out, off, m, b, c, fmt, err_acc);
    }
}

// Full 4-1 split. Midpoints of all three edges → four children.
void emit_split_four(float* out, cardinal::usize& off,
                     const V3& a, const V3& b, const V3& c,
                     precision::Format fmt, float& err_acc) noexcept
{
    const V3 mab = mid(a, b);
    const V3 mbc = mid(b, c);
    const V3 mca = mid(c, a);
    emit_tri(out, off, a,   mab, mca, fmt, err_acc);
    emit_tri(out, off, mab, b,   mbc, fmt, err_acc);
    emit_tri(out, off, mca, mbc, c,   fmt, err_acc);
    emit_tri(out, off, mab, mbc, mca, fmt, err_acc);
}

// 4-1 split then longest-edge on each child → 8 micro-triangles.
void emit_split_eight(float* out, cardinal::usize& off,
                      const V3& a, const V3& b, const V3& c,
                      precision::Format fmt, float& err_acc) noexcept
{
    const V3 mab = mid(a, b);
    const V3 mbc = mid(b, c);
    const V3 mca = mid(c, a);
    emit_split_longest(out, off, a,   mab, mca, fmt, err_acc);
    emit_split_longest(out, off, mab, b,   mbc, fmt, err_acc);
    emit_split_longest(out, off, mca, mbc, c,   fmt, err_acc);
    emit_split_longest(out, off, mab, mbc, mca, fmt, err_acc);
}

struct GeomCtx {
    AdaptiveGeometryPass::State* st;
};

void geom_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<AdaptiveGeometryPass::State*>(uctx);
    const cardinal::u32 M = st->input_triangle_count;
    if (M == 0) {
        cardinal::u32* cnt = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_count));
        if (cnt) cnt[0] = 0;
        st->micro_triangles_emitted = 0;
        st->total_quantization_error = 0.0f;
        ec.dispatch(0, 1, 1);
        return;
    }
    const float* in_tris = static_cast<const float*>(ec.map_buffer_read(st->in_triangles));
    float*       out_micro = static_cast<float*>(ec.map_buffer_write(st->out_micro_tris));
    cardinal::u32* cnt = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_count));
    if (!in_tris || !out_micro || !cnt) {
        if (cnt) cnt[0] = 0;
        st->micro_triangles_emitted = 0;
        st->total_quantization_error = 0.0f;
        ec.dispatch(0, 1, 1);
        return;
    }
    const TierProperties props = properties_of(st->selected_tier);
    const precision::Format fmt = props.format;
    const cardinal::u32 N = props.micro_triangles_per_input;

    float err_acc = 0.0f;
    cardinal::usize off = 0;
    for (cardinal::u32 t = 0; t < M; ++t) {
        const float* v = in_tris + t * 9;
        V3 a{ v[0], v[1], v[2] };
        V3 b{ v[3], v[4], v[5] };
        V3 c{ v[6], v[7], v[8] };
        a = sanitize(a); b = sanitize(b); c = sanitize(c);
        switch (N) {
            case 1: emit_tri          (out_micro, off, a, b, c, fmt, err_acc); break;
            case 2: emit_split_longest(out_micro, off, a, b, c, fmt, err_acc); break;
            case 4: emit_split_four   (out_micro, off, a, b, c, fmt, err_acc); break;
            case 8: emit_split_eight  (out_micro, off, a, b, c, fmt, err_acc); break;
            default: emit_tri         (out_micro, off, a, b, c, fmt, err_acc); break;
        }
    }
    const cardinal::u32 total = M * N;
    cnt[0] = total;
    st->micro_triangles_emitted   = total;
    st->total_quantization_error  = err_acc;

    // 64 micro-triangles per workgroup.
    ec.dispatch((total + 63) / 64, 1, 1);
}

}  // namespace

cardinal::shared_ptr<AdaptiveGeometryPass::State> AdaptiveGeometryPass::add_to_graph(
    rg::Graph& g,
    rg::ResourceHandle in_triangles,
    cardinal::u32 input_triangle_count,
    const GpuCapabilities& caps,
    GeometryTier max_tier)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_triangles         = in_triangles;
    st->input_triangle_count = input_triangle_count;
    st->selected_tier        = select_tier(caps, max_tier);
    const auto props = properties_of(st->selected_tier);
    st->micro_triangles_per_input = props.micro_triangles_per_input;

    const cardinal::u32 total = input_triangle_count * props.micro_triangles_per_input;
    rg::BufferDesc od;
    od.name         = "geom.micro";
    od.size_bytes   = static_cast<cardinal::usize>(total) * 9u * sizeof(float);
    od.stride_bytes = sizeof(float);
    st->out_micro_tris = g.declare_buffer(od);

    rg::BufferDesc cd;
    cd.name         = "geom.count";
    cd.size_bytes   = sizeof(cardinal::u32);
    cd.stride_bytes = sizeof(cardinal::u32);
    st->out_count = g.declare_buffer(cd);

    rg::PassDesc pd;
    pd.name = "AdaptiveGeometryPass";
    pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_triangles,         rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{st->out_micro_tris,   rg::AccessMode::Write, 1});
    pd.accesses.push_back(rg::ResourceAccess{st->out_count,        rg::AccessMode::Write, 2});
    pd.record   = geom_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = (total + 63) / 64;
    pd.dispatch_y = 1; pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* AdaptiveGeometryPass::hlsl_source() noexcept {
    return R"(// Cardinal — AdaptiveGeometryPass.hlsl
//
// One thread per input triangle. Subdivision pattern is selected at
// dispatch time via a #define from the host (TIER_N = 1 / 2 / 4 / 8).
// The shader produces TIER_N micro-triangles per input, each component
// quantized to the matching precision format and immediately re-promoted
// to FP32 (so downstream passes consume a uniform 32-bit stream).
//
// Quantize/dequantize helpers ship in shaders/include/cardinal_packed_math.hlsli
// and match cardinal::render::precision bit-for-bit.
#include "cardinal_packed_math.hlsli"

ByteAddressBuffer  in_tris   : register(t0);
RWByteAddressBuffer out_micro : register(u0);
RWByteAddressBuffer out_count : register(u1);

cbuffer Params : register(b0) {
    uint  input_count;    // M
    uint  format;         // precision::Format encoding
    uint  micro_per_input;// N (1/2/4/8)
    uint  _pad;
};

float load_f(ByteAddressBuffer b, uint o) { return asfloat(b.Load(o)); }
void  store_f(RWByteAddressBuffer b, uint o, float v) { b.Store(o, asuint(v)); }

float3 quantize_v(float3 v) {
    // Per-component quantize → dequantize via the format-keyed helper.
    return float3(quantise_fmt(format, v.x), quantise_fmt(format, v.y), quantise_fmt(format, v.z));
}

void emit(uint slot, float3 a, float3 b, float3 c) {
    uint base = slot * 9 * 4;
    a = quantize_v(a); b = quantize_v(b); c = quantize_v(c);
    store_f(out_micro, base + 0*4, a.x); store_f(out_micro, base + 1*4, a.y); store_f(out_micro, base + 2*4, a.z);
    store_f(out_micro, base + 3*4, b.x); store_f(out_micro, base + 4*4, b.y); store_f(out_micro, base + 5*4, b.z);
    store_f(out_micro, base + 6*4, c.x); store_f(out_micro, base + 7*4, c.y); store_f(out_micro, base + 8*4, c.z);
}

float3 mid(float3 a, float3 b) { return 0.5 * (a + b); }
float  d2 (float3 a, float3 b) { float3 d = a - b; return dot(d, d); }

void split_longest(uint slot, float3 a, float3 b, float3 c) {
    float dab = d2(a, b), dbc = d2(b, c), dca = d2(c, a);
    if (dab >= dbc && dab >= dca) { float3 m = mid(a, b); emit(slot, a, m, c); emit(slot + 1, m, b, c); }
    else if (dbc >= dca)          { float3 m = mid(b, c); emit(slot, a, b, m); emit(slot + 1, a, m, c); }
    else                          { float3 m = mid(c, a); emit(slot, a, b, m); emit(slot + 1, m, b, c); }
}

void split_four(uint slot, float3 a, float3 b, float3 c) {
    float3 mab = mid(a, b), mbc = mid(b, c), mca = mid(c, a);
    emit(slot + 0, a,   mab, mca);
    emit(slot + 1, mab, b,   mbc);
    emit(slot + 2, mca, mbc, c  );
    emit(slot + 3, mab, mbc, mca);
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x;
    if (t >= input_count) return;
    uint base = t * 9 * 4;
    float3 a = float3(load_f(in_tris, base + 0*4), load_f(in_tris, base + 1*4), load_f(in_tris, base + 2*4));
    float3 b = float3(load_f(in_tris, base + 3*4), load_f(in_tris, base + 4*4), load_f(in_tris, base + 5*4));
    float3 c = float3(load_f(in_tris, base + 6*4), load_f(in_tris, base + 7*4), load_f(in_tris, base + 8*4));
    uint slot = t * micro_per_input;
    if      (micro_per_input == 1) { emit         (slot, a, b, c); }
    else if (micro_per_input == 2) { split_longest(slot, a, b, c); }
    else if (micro_per_input == 4) { split_four   (slot, a, b, c); }
    else {  // 8
        float3 mab = mid(a, b), mbc = mid(b, c), mca = mid(c, a);
        split_longest(slot + 0, a,   mab, mca);
        split_longest(slot + 2, mab, b,   mbc);
        split_longest(slot + 4, mca, mbc, c  );
        split_longest(slot + 6, mab, mbc, mca);
    }
    // Total count is published by the host (input_count * micro_per_input);
    // one thread writes it.
    if (t == 0) out_count.Store(0, input_count * micro_per_input);
}
)";
}

}  // namespace cardinal::render::gpu
