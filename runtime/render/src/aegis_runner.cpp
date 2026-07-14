#include <cardinal/render/aegis_runner.hpp>

namespace cardinal::render {

cardinal::shared_ptr<AegisPipelineRunner> AegisPipelineRunner::create(
    gpu::AegisConfig cfg, AegisBackendMode mode)
{
    auto r = cardinal::shared_ptr<AegisPipelineRunner>(new AegisPipelineRunner());
    r->cfg_       = cfg;
    r->mode_      = mode;
    r->pipeline_  = gpu::AegisPipeline::create(cfg);
    r->graph_     = graph::Graph::create();
    switch (mode) {
        case AegisBackendMode::Cpu:
            r->backend_ = graph::CpuBackend::create();
            break;
        case AegisBackendMode::Null:
            r->backend_ = graph::NullBackend::create();
            break;
        case AegisBackendMode::ThreadedCpu:
            r->backend_ = graph::ThreadedCpuBackend::create();
            break;
        case AegisBackendMode::Rhi:
            // RhiBackend requires a Device + Swapchain to bind against —
            // available only when the runner is owned by a Pipeline that
            // has them. Hosts constructing the runner without device
            // refs get the NullBackend fallback so they don't crash; the
            // AegisGraphPipeline bridge replaces the runner's backend
            // with a real RhiBackend at construction time when device +
            // swapchain are in scope. See aegis_pipeline_bridge.cpp.
            r->backend_ = graph::NullBackend::create();
            break;
    }
    return r;
}

void AegisPipelineRunner::reconfigure(gpu::AegisConfig cfg) {
    cfg_      = cfg;
    pipeline_ = gpu::AegisPipeline::create(cfg);   // caches cfg; must rebuild
    built_    = false;                             // force the next build()
    // graph_ + backend_ are deliberately preserved (build() recreates graph_;
    // the live backend was installed via set_backend).
}

// --- Pure config resolution (host-driven clamp; see header) ----------------
gpu::AegisConfig aegis_resolve_config(gpu::AegisConfig cfg,
                                      const AegisDeviceSupport& dev,
                                      const AegisFeatureRequest& req,
                                      cardinal::u32 width,
                                      cardinal::u32 height) noexcept
{
    cfg.caps.fp16_supported = dev.fp16;
    cfg.caps.fp8_supported  = dev.fp8;
    cfg.caps.fp4_supported  = dev.fp4;

    const bool allow8 = dev.fp8 && req.allow_fp8;
    const bool allow4 = dev.fp4 && req.allow_fp4;
    int want = req.max_tier_want;
    if (want < 0) want = 0;
    if (want > 3) want = 3;
    cfg.max_tier = (want >= 3 && allow4) ? gpu::GeometryTier::Fp4
                 : (want >= 2 && allow8) ? gpu::GeometryTier::Fp8
                 : (want >= 1)           ? gpu::GeometryTier::Fp16
                 :                         gpu::GeometryTier::Fp32;

    cfg.features.async_compute         = dev.async_compute         && req.async_compute;
    cfg.features.variable_rate_shading = dev.variable_rate_shading && req.variable_rate_shading;
    cfg.features.bindless_resources    = dev.bindless_resources    && req.bindless_resources;
    cfg.features.direct_storage        = dev.direct_storage        && req.direct_storage;

    if (width)  cfg.width  = width;
    if (height) cfg.height = height;
    return cfg;
}

bool aegis_config_equal(const gpu::AegisConfig& a, const gpu::AegisConfig& b) noexcept {
    return a.width == b.width && a.height == b.height &&
           a.material_count == b.material_count && a.light_count == b.light_count &&
           a.exposure == b.exposure &&
           a.caps.fp16_supported == b.caps.fp16_supported &&
           a.caps.fp8_supported  == b.caps.fp8_supported &&
           a.caps.fp4_supported  == b.caps.fp4_supported &&
           a.caps.max_micro_triangles_per_input == b.caps.max_micro_triangles_per_input &&
           a.max_tier == b.max_tier &&
           a.features.async_compute         == b.features.async_compute &&
           a.features.variable_rate_shading == b.features.variable_rate_shading &&
           a.features.bindless_resources    == b.features.bindless_resources &&
           a.features.direct_storage        == b.features.direct_storage;
}

graph::Graph& AegisPipelineRunner::begin_build() {
    graph_ = graph::Graph::create();
    graph_fresh_ = true;
    return *graph_;
}

bool AegisPipelineRunner::build(const gpu::AegisSceneInputs& inputs) {
    if (!graph_ || !pipeline_) return false;
    // Fresh graph per build — the AEGIS topology is rebuilt every frame
    // because resource handles change with the scene. When the host used
    // begin_build() (declaring its scene-input buffers on the new graph),
    // consume that graph; otherwise create one now (legacy flow — valid
    // only when `inputs` carries no handles minted from graph()).
    if (!graph_fresh_) graph_ = graph::Graph::create();
    graph_fresh_ = false;
    outputs_ = gpu::AegisOutputs{};
    stages_  = gpu::AegisStageRefs{};
    pipeline_->build(*graph_, inputs, outputs_, stages_);
    const bool ok = graph_->compile();
    last_compile_ = graph_->stats();
    built_ = ok;
    return ok;
}

void AegisPipelineRunner::execute() {
    if (!built_ || !backend_ || !graph_) return;
    backend_->execute(*graph_);
}

graph::Graph& AegisPipelineRunner::graph() noexcept {
    return *graph_;
}

void AegisPipelineRunner::set_backend(cardinal::shared_ptr<graph::Backend> backend) noexcept {
    backend_ = cardinal::move(backend);
}

cardinal::vector<graph::NullBackend::PassEvent>
AegisPipelineRunner::null_trace() const {
    // Returns events when the runner's current backend is actually a
    // NullBackend (irrespective of the originally-configured mode_) —
    // hosts may swap backends at runtime via set_backend, so the mode
    // enum doesn't necessarily reflect the live backend's type.
    cardinal::vector<graph::NullBackend::PassEvent> out;
    auto* nb = dynamic_cast<graph::NullBackend*>(backend_.get());
    if (!nb) return out;
    for (const auto& ev : nb->events()) out.push_back(ev);
    return out;
}

cardinal::vector<cardinal::u8>
AegisPipelineRunner::read_buffer(graph::ResourceHandle h) const {
    if (mode_ != AegisBackendMode::Cpu) return {};
    auto* cb = dynamic_cast<graph::CpuBackend*>(backend_.get());
    if (!cb) return {};
    return cb->buffer_contents(h);
}

}  // namespace cardinal::render
