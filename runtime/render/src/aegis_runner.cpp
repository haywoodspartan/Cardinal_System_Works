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
        case AegisBackendMode::Rhi:
            // RhiBackend not yet implemented; fall through to NullBackend so
            // hosts targeting the future RHI path don't crash today.
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

cardinal::vector<graph::NullBackend::PassEvent>
AegisPipelineRunner::null_trace() const {
    cardinal::vector<graph::NullBackend::PassEvent> out;
    if (mode_ != AegisBackendMode::Null && mode_ != AegisBackendMode::Rhi) return out;
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
