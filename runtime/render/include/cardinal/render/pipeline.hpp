#pragma once

// =============================================================================
// Cardinal — runtime-switchable render pipelines.
//
// A `Pipeline` is one self-contained way of rendering the scene (forward,
// forward+, deferred, path-traced, debug visualiser, …). Pipelines expose
// their feature set as a list of `Knob`s — typed, grouped, optionally
// gated by GPU capabilities — so the editor can render a generic settings
// UI without knowing anything about the implementation.
//
// `Registry` owns one instance of every available pipeline and dispatches
// the active one's render() each frame. Switching is O(1) and lossless:
// the previous pipeline's settings are retained so users can flip back.
//
// Today's two shipping pipelines:
//
//   ForwardBaseline  — the existing CPU-transformed forward renderer with
//                      ViewMode + tonemap + render-scale knobs.
//   DebugVisualizer  — same renderer, locked-down channel picker
//                      (Normals / Polygons / Heightmap / Wireframe) for
//                      content-debugging workflows.
//
// Real Forward+, Deferred, and Path-traced pipelines slot in by
// implementing the same interface — each is a multi-week project; the
// framework is here for the day they land.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/budget/memory.hpp>
#include <cardinal/rhi/rhi.hpp>
#include <cardinal/scene/light.hpp>
#include <cardinal/scene/scene.hpp>

#include <cardinal/core/std/containers.hpp>

namespace cardinal::render {

// ---------------------------------------------------------------------------
// Knob — a typed, UI-introspectable setting on a pipeline.
// ---------------------------------------------------------------------------
enum class KnobKind : u32 {
    Bool,     // checkbox
    Int,      // slider with int range
    Float,    // slider with float range
    Enum,     // combobox with string labels
};

struct Knob {
    cardinal::string  id;                        // stable identifier ("tonemap")
    cardinal::string  label;                     // human-readable
    cardinal::string  group;                     // "Lighting" / "AA" / "Performance"
    cardinal::string  tooltip;                   // longer description
    KnobKind     kind{KnobKind::Bool};

    // Storage. Only one slot is meaningful per knob (chosen by `kind`).
    bool         b{false};
    int          i{0};
    float        f{0.0f};
    int          e{0};                      // index into enum_labels

    // Range for Int / Float kinds.
    int          i_min{0},  i_max{1};
    float        f_min{0.f}, f_max{1.f};
    float        f_step{0.01f};

    // Enum labels (one per option). Index into them via `e`.
    cardinal::vector<cardinal::string> enum_labels;

    // Availability — set by Pipeline::on_caps. When unavailable the panel
    // greys the knob out and shows the reason as a tooltip.
    bool         available{true};
    cardinal::string  unavailable_reason;
};

// ---------------------------------------------------------------------------
// Pipeline — one render strategy.
// ---------------------------------------------------------------------------
enum class PipelineId : u32 {
    ForwardBaseline = 0,
    DebugVisualizer,
    ForwardClustered,    // meshlet/cluster culling baseline (Phase 6)
    Aegis,               // AEGIS Pipeline 2.0 — full graph-driven GPU
                         // pipeline (Virtual Geometry → V-Buf → Resolve →
                         // Tonemap → Composite). Registered slot reserved
                         // for the AegisRenderPipeline that lands when
                         // render::graph::RhiBackend is wired.
    // Future: ForwardPlus, Deferred, PathTraced …
};

// Stats a pipeline can publish per-frame for the editor to display.
// Optional — pipelines that don't compute these leave the values at zero.
struct FrameStats {
    u32 clusters_total{0};
    u32 clusters_culled{0};
    u32 clusters_drawn{0};
    u32 triangles_drawn{0};
    u32 tessellation_factor_avg{0};
    // Entity-level frustum cull stats from the underlying scene
    // ForwardRenderer. ForwardBaseline + DebugVisualizer pipelines
    // populate these from scene::ForwardRenderer::frame_stats().
    u32 entities_total{0};
    u32 entities_drawn{0};
    u32 entities_culled{0};
    u32 draw_calls{0};          // GPU draw submissions this frame
    u32 pc_pushes{0};           // push-constant writes (deduped at entity boundaries)
    // vgeom aggregates summed across all selected meshes (see
    // scene::FrameStats for field semantics).
    u32 vgeom_master_tris{0};
    u32 vgeom_drawn_tris{0};
    u32 vgeom_cut_clusters{0};
    u32 vgeom_culled_clusters{0};
    // Per-phase wall-clock cost in microseconds (cull, Phase 1 sizing,
    // Phase 2 parallel transform, painter sort, GPU submit). Populated
    // from scene::ForwardRenderer::frame_stats() when the underlying
    // pipeline is one of the forward variants. Hi-Z / cluster-shader
    // pipelines may leave these at 0.
    f32 cull_us{0.0f};
    f32 phase1_us{0.0f};
    f32 phase2_us{0.0f};
    f32 sort_us{0.0f};
    f32 submit_us{0.0f};
};

class Pipeline {
public:
    virtual ~Pipeline() = default;
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    virtual PipelineId  id()           const noexcept = 0;
    virtual const char* name()         const noexcept = 0;
    virtual const char* description()  const noexcept = 0;

    // Mutable so the editor can write directly. Persistent across switches.
    virtual cardinal::vector<Knob>& knobs() noexcept = 0;

    // Called once after the device + swapchain are ready (and again if the
    // user hot-swaps the GPU someday). Pipelines mark each knob's
    // availability based on the live capability bits.
    virtual void on_caps(const rhi::GpuCapabilities& caps) = 0;

    // Issue the per-frame render. Must be called between
    // swapchain->begin_frame() and swapchain->end_frame().
    virtual void render(scene::Scene& scene, float aspect) = 0;

    // Optional per-frame lighting input. Pipelines that own a
    // scene::ForwardRenderer forward this to the underlying renderer's
    // set_light_set(); other pipelines may ignore it (default no-op).
    virtual void set_light_set(const scene::LightSet* /*lights*/) {}

    // Editor gizmo toggles — forward to the underlying scene renderer
    // when present. Default no-op so non-forward pipelines compile.
    virtual void set_show_aabbs  (bool /*on*/) {}
    virtual void set_show_frustum(bool /*on*/) {}
    virtual void set_show_axes   (bool /*on*/) {}
    virtual void set_show_lights (bool /*on*/) {}    // light-volume wireframes

    // Pipelines that compute cluster/triangle counts (e.g. ForwardClustered)
    // override this; the default returns zeros for everything.
    virtual FrameStats stats() const noexcept { return {}; }

protected:
    Pipeline() = default;
};

// ---------------------------------------------------------------------------
// Registry — owns every available pipeline, dispatches the active one.
// ---------------------------------------------------------------------------
class Registry {
public:
    static cardinal::unique_ptr<Registry> create(rhi::Device& dev,
                                            rhi::Swapchain& sw);
    virtual ~Registry() = default;
    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;

    virtual cardinal::vector<Pipeline*> all() = 0;
    virtual Pipeline*              active() = 0;
    virtual PipelineId             active_id() const noexcept = 0;
    virtual void                   set_active(PipelineId id) = 0;

    virtual void render(scene::Scene& scene, float aspect) = 0;

    // Forwarded to the active pipeline each frame.
    virtual void set_light_set(const scene::LightSet* lights) = 0;

protected:
    Registry() = default;
};

// ---------------------------------------------------------------------------
// FeatureGate — read-only utility that maps GpuCapabilities to a set of
// feature flags + human-readable reasons. Pipelines use this to set Knob
// availability; the editor can render a "GPU Features" matrix from the
// same source.
// ---------------------------------------------------------------------------
enum class Feature : u32 {
    HardwareRT = 0,            // ray-query or ray-tracing-pipeline
    DLSS_Upscale,              // NVIDIA DLSS Super Resolution
    DLSS_FrameGen,             // NVIDIA DLSS-G
    DLSS_RayReconstruct,       // NVIDIA DLSS-D
    FSR3_Upscale,              // AMD FSR 3 quality / balanced / perf
    FSR3_FrameGen,
    XeSS_Upscale,              // Intel XeSS
    MeshShaders,
    TaskShaders,
    VRS,                       // variable rate shading (DX12 / VK_KHR_fragment_shading_rate)
    BindlessDescriptors,
    FP16Math,                  // Rapid Packed Math
    INT8Math,
    AsyncCompute,
    ReflexLowLatency,
    DLAA,                      // DLSS native-res AA (RTX only)
    FP8Math,                   // hardware FP8 tensor ops (Hopper / Ada / Blackwell-class)
    FP4Math,                   // hardware FP4 tensor ops (Blackwell-class)
    Count_
};

struct FeatureStatus {
    bool        available{false};
    cardinal::string reason;        // when unavailable, why
};

class FeatureGate {
public:
    explicit FeatureGate(const rhi::GpuCapabilities& caps);
    FeatureStatus query(Feature f) const noexcept;
    static const char* feature_name(Feature f) noexcept;
    static const char* feature_group(Feature f) noexcept;
private:
    const rhi::GpuCapabilities& caps_;
};

}  // namespace cardinal::render
