#pragma once

// =============================================================================
// Cardinal — AegisPipelineRunner: bridge between AegisPipeline and Backend.
//
// The audit identified that the new graph-pass family ships as a "parallel
// universe" — declared on render::graph but with no consumer outside the
// test harness. AegisPipelineRunner is the bridge: hosts construct one
// once (config + caps fixed), then call build(scene_inputs) → execute()
// each frame. The runner:
//
//   * owns a render::graph::Graph instance + the AegisPipeline orchestrator
//   * owns a graph::Backend (CpuBackend by default; NullBackend for
//     topology-only runs; RhiBackend when compute lands)
//   * exposes AegisStageRefs so the editor can read per-frame stats
//   * normalises the rhi::GpuCapabilities → PrecisionCaps mapping so
//     hosts don't have to convert manually
//
// This is the single entry point the engine's editor / Studio uses to
// drive the AEGIS pipeline. Once render::graph::RhiBackend lands, hosts
// switch backends with no other call-site changes.
// =============================================================================

#include <cardinal/render/graph.hpp>
#include <cardinal/render/gpu_aegis.hpp>
#include <cardinal/core/types.hpp>

namespace cardinal::render {

// Backend-mode knob — picks which backend the runner executes against.
enum class AegisBackendMode : cardinal::u32 {
    Cpu  = 0,   // graph::CpuBackend — full reference execution, allocates buffers
    Null,       // graph::NullBackend — topology only, no buffers, no records
    ThreadedCpu,// graph::ThreadedCpuBackend — parallel pass execution per wave
    Rhi,        // graph::RhiBackend — real compute dispatch (when RHI compute lands)
};

class AegisPipelineRunner {
public:
    static cardinal::shared_ptr<AegisPipelineRunner> create(
        gpu::AegisConfig cfg,
        AegisBackendMode mode = AegisBackendMode::Cpu);

    // Wire the AEGIS graph against `inputs` and compile. Returns false if
    // the graph contains a cycle (shouldn't happen in the AEGIS topology
    // but the runner reports it cleanly anyway).
    bool build(const gpu::AegisSceneInputs& inputs);

    // Execute the compiled graph against the configured backend. Pre:
    // build() returned true. Post: outputs() reflects the most recent
    // execute; stages() reflects per-pass stats.
    void execute();

    const gpu::AegisOutputs&    outputs() const noexcept { return outputs_; }
    const gpu::AegisStageRefs&  stages()  const noexcept { return stages_; }
    gpu::AegisConfig            config()  const noexcept { return cfg_; }
    AegisBackendMode            backend_mode() const noexcept { return mode_; }

    // Graph access for hosts that want to inject extra passes alongside
    // (e.g. an editor adding a debug visualiser pass after composite).
    graph::Graph&               graph() noexcept;

    // Replace the runner's Backend at runtime — used by the
    // AegisGraphPipeline bridge to install a real graph::RhiBackend
    // (which needs an rhi::Device + Swapchain in scope) over the
    // default Cpu/Null one created in create(). Safe to call between
    // execute() invocations; the previous backend is released.
    void set_backend(cardinal::shared_ptr<graph::Backend> backend) noexcept;

    // Replace the runner's AegisConfig at runtime (precision caps + max_tier +
    // features). MUST rebuild the AegisPipeline because it caches the config at
    // create() (gpu_aegis.cpp) and reads caps/max_tier when it declares the
    // adaptive-geometry pass at build() — so select_tier re-runs against the new
    // caps/max_tier on the next build(). The live Backend is preserved (only
    // set_backend changes it). Diff-gate at the call site: rebuilding the
    // pipeline every frame is wasteful.
    void reconfigure(gpu::AegisConfig cfg);

    // Editor surface — graph stats from the most recent build().
    graph::CompileStats         compile_stats() const noexcept { return last_compile_; }

    // NullBackend telemetry — populated only when backend_mode == Null.
    cardinal::vector<graph::NullBackend::PassEvent> null_trace() const;

    // CpuBackend buffer readback — only valid in Cpu mode. Returns empty
    // for invalid handles or non-Cpu backends.
    cardinal::vector<cardinal::u8> read_buffer(graph::ResourceHandle h) const;

private:
    AegisPipelineRunner() = default;

    gpu::AegisConfig                         cfg_;
    AegisBackendMode                         mode_ {AegisBackendMode::Cpu};
    cardinal::shared_ptr<gpu::AegisPipeline> pipeline_;
    cardinal::shared_ptr<graph::Graph>       graph_;
    cardinal::shared_ptr<graph::Backend>     backend_;
    gpu::AegisOutputs                        outputs_;
    gpu::AegisStageRefs                      stages_;
    graph::CompileStats                      last_compile_;
    bool                                     built_ {false};
};

// ---------------------------------------------------------------------------
// AegisConfig resolution — pure, host-driven clamping of REQUESTED features +
// precision tier against what the DEVICE actually supports. Kept free of rhi
// and Knob types (the bridge maps rhi::GpuCapabilities -> AegisDeviceSupport
// and the pipeline-panel knobs -> AegisFeatureRequest) so the clamp logic is
// directly unit-testable.
// ---------------------------------------------------------------------------
struct AegisDeviceSupport {
    bool fp16 {true};
    bool fp8  {false};
    bool fp4  {false};
    bool async_compute         {false};
    bool variable_rate_shading {false};
    bool bindless_resources    {false};
    bool direct_storage        {false};
};
struct AegisFeatureRequest {
    int  max_tier_want {1};        // 0..3 = FP32 / FP16 / FP8 / FP4 (from the knob)
    bool allow_fp8 {false};
    bool allow_fp4 {false};
    bool async_compute         {false};
    bool variable_rate_shading {false};
    bool bindless_resources    {false};
    bool direct_storage        {false};
};

// Returns `base` with caps (fp16/fp8/fp4 = device truth), max_tier (the
// requested ceiling, clamped down to what the device supports + the user
// allows), and features (requested AND device-supported) applied. width/height,
// when non-zero, override base (resize fold-in).
gpu::AegisConfig aegis_resolve_config(gpu::AegisConfig base,
                                      const AegisDeviceSupport& dev,
                                      const AegisFeatureRequest& req,
                                      cardinal::u32 width  = 0,
                                      cardinal::u32 height = 0) noexcept;

// True when two configs are equal across every field the resolver can change
// (so the bridge only reconfigures on a real change, not every frame).
bool aegis_config_equal(const gpu::AegisConfig& a, const gpu::AegisConfig& b) noexcept;

}  // namespace cardinal::render
