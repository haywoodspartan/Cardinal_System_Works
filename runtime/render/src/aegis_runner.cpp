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

bool AegisPipelineRunner::build(const gpu::AegisSceneInputs& inputs) {
    if (!graph_ || !pipeline_) return false;
    // Fresh graph per build — the AEGIS topology is rebuilt every frame
    // because resource handles change with the scene. Cheap: graph nodes
    // are flat vectors, no allocator churn beyond the per-frame inputs.
    graph_ = graph::Graph::create();
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
