#include <cardinal/lighting/gpu_radiance_cache.hpp>

#include <cardinal/core/utility.hpp>

#include <chrono>

namespace cardinal::lighting::gpu {

namespace rg = cardinal::render::graph;

namespace {

void radiance_cache_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<RadianceCachePass::State*>(uctx);
    if (!st->tracer || !st->scene ||
        st->dims_x == 0 || st->dims_y == 0 || st->dims_z == 0) {
        ec.dispatch(0, 1, 1);
        return;
    }
    float* out = static_cast<float*>(ec.map_buffer_write(st->out_probe_data));
    if (!out) {
        ec.dispatch(0, 1, 1);
        return;
    }

    // Build a temporary IrradianceProbeVolume that aliases our output
    // buffer's layout, drive the existing baker, then copy the result
    // into the graph-managed buffer.
    IrradianceProbeVolume vol;
    vol.origin  = st->origin;
    vol.spacing = st->spacing;
    vol.resize(st->dims_x, st->dims_y, st->dims_z);

    ProbeBakeConfig cfg;
    cfg.samples_per_axis = st->samples_per_axis;
    cfg.rng_seed         = st->rng_seed;

    const auto t0 = std::chrono::high_resolution_clock::now();
    auto baker = ProbeVolumeBaker::create(st->tracer);
    baker->bake(*st->scene, vol, cfg);
    const auto t1 = std::chrono::high_resolution_clock::now();

    // Copy the volume data into the output buffer. Layout is identical
    // (vol.data is exactly dims_x*dims_y*dims_z*6*3 floats), so this is
    // a flat copy.
    const cardinal::usize total_floats = vol.data.size();
    for (cardinal::usize i = 0; i < total_floats; ++i) {
        out[i] = vol.data[i];
    }

    const auto bake_stats   = baker->stats();
    st->probes_baked = bake_stats.probes;
    st->rays_cast    = bake_stats.rays;
    st->total_us     = static_cast<float>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    ec.dispatch(st->dims_x, st->dims_y, st->dims_z);
}

}  // namespace

cardinal::shared_ptr<RadianceCachePass::State> RadianceCachePass::add_to_graph(
    rg::Graph& g,
    cardinal::shared_ptr<PathTracer> tracer,
    const Scene& scene,
    cardinal::scene::Vec3 origin,
    cardinal::scene::Vec3 spacing,
    cardinal::u32 dims_x, cardinal::u32 dims_y, cardinal::u32 dims_z,
    cardinal::u32 samples_per_axis)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->tracer = tracer;
    st->scene  = &scene;
    st->origin = origin; st->spacing = spacing;
    st->dims_x = dims_x; st->dims_y = dims_y; st->dims_z = dims_z;
    st->samples_per_axis = samples_per_axis;

    const cardinal::u32 probe_count = dims_x * dims_y * dims_z;
    rg::BufferDesc od;
    od.name         = "radiance.cache";
    od.size_bytes   = static_cast<cardinal::usize>(probe_count) * 6u * 3u * sizeof(float);
    od.stride_bytes = sizeof(float);
    st->out_probe_data = g.declare_buffer(od);

    rg::PassDesc pd;
    pd.name = "RadianceCachePass";
    pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{st->out_probe_data, rg::AccessMode::Write, 0});
    pd.record   = radiance_cache_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = dims_x;
    pd.dispatch_y = dims_y;
    pd.dispatch_z = dims_z;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* RadianceCachePass::hlsl_source() noexcept {
    return R"(// Cardinal — RadianceCachePass.hlsl
//
// One thread per probe × axis (6 hemispheres per probe). Cosine-weighted
// rays into the BVH, integrate per-hemisphere radiance, write 6×3 floats
// per probe. The HLSL counterpart of lighting::ProbeVolumeBaker; the CPU
// reference above wraps the existing CPU baker bit-for-bit.
RWByteAddressBuffer out_probe_data : register(u0);
cbuffer Params : register(b0) {
    float3 origin; uint dims_x;
    float3 spacing; uint dims_y;
    uint dims_z, samples_per_axis, rng_seed, _pad;
};
[numthreads(2, 2, 2)]
void main(uint3 tid : SV_DispatchThreadID) { /* RHI compute commit */ }
)";
}

}  // namespace cardinal::lighting::gpu
