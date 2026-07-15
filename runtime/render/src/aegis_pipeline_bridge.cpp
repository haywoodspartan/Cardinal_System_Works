// =============================================================================
// Cardinal — AegisGraphPipeline.
//
// The fourth concrete render::Pipeline (PipelineId::Aegis), registered in
// RegistryImpl alongside ForwardBaseline / DebugVisualizer / ForwardClustered.
// Studio's pipeline-selector dropdown shows "AEGIS Pipeline 2.0" as a
// first-class option once this commit lands.
//
// Until graph::RhiBackend exists, the bridge runs as a HYBRID:
//
//   * AegisPipelineRunner builds + executes the full AEGIS graph each
//     frame on the NullBackend (topology-only — zero buffer allocation,
//     constant cost per frame). The orchestrator's per-stage stats are
//     published to the editor: pass counts, wave decomposition, selected
//     GeometryTier, etc.
//   * scene::ForwardRenderer issues the actual GPU draws — the on-screen
//     image is identical to ForwardBaseline today.
//
// When graph::RhiBackend lands the host flips backend_mode to Rhi and
// the on-screen image switches to AEGIS output without touching the
// Pipeline subclass, the Studio selector, or any host code.
// =============================================================================

#include <cardinal/render/pipeline.hpp>
#include <cardinal/render/aegis_runner.hpp>

#include <initializer_list>
#include <cardinal/render/gpu_aegis.hpp>
#include <cardinal/scene/renderer.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/std/cmath.hpp>   // cardinal::sqrt (camera_dir normalize)
#include <cardinal/core/std/utility.hpp>

namespace cardinal::render {

namespace {

class AegisGraphPipeline final : public Pipeline {
public:
    AegisGraphPipeline(rhi::Device& dev, rhi::Swapchain& sw)
        : dev_(&dev)
        , sw_(&sw)
        , renderer_(scene::ForwardRenderer::create(dev, sw))
    {
        gpu::AegisConfig cfg;
        cfg.width  = sw.width();
        cfg.height = sw.height();
        // No precision ceiling hardcode: the AegisConfig defaults (fp8/fp4
        // unsupported, so select_tier maxes at FP16) hold until on_caps() runs
        // at registry construction, which applies the real device caps + the
        // knob intent via apply_config() -> reconfigure().

        // Construct with a placeholder Null backend (create() has no device,
        // so it can't build the real RhiBackend). The backend_mode Knob below
        // DEFAULTS to RhiBackend, and current_mode_ starts at Null — so the
        // first render() detects the mismatch and swaps in a real
        // graph::RhiBackend bound to this pipeline's device + swapchain
        // (update_backend_from_knob). RhiBackend degrades to recording-only
        // telemetry until the device implements compute pipelines, and the
        // on-screen image is drawn by the ForwardRenderer either way, so this
        // is safe as the default. Null / Cpu / ThreadedCpu stay selectable.
        runner_ = AegisPipelineRunner::create(cfg, AegisBackendMode::Null);

        build_knobs();
        cache_knob_indices();
    }

    // ---- Hot-knob index cache -------------------------------------------
    // render() consults ~10 knobs EVERY frame; scanning the ~80-entry knobs_
    // vector with string compares for each cost ~800 compares/frame. knobs_
    // is only (re)built in build_knobs(), and per-frame writes mutate values
    // in place (no resize), so indices resolved once stay valid.
    struct KnobIdx {
        int backend_mode {-1}, max_tier {-1}, max_tier_fp8 {-1},
            max_tier_fp4 {-1}, async_compute {-1}, variable_rate_shading {-1},
            bindless_resources {-1}, direct_storage {-1}, gpu_execute {-1},
            view_mode {-1};
    };
    void cache_knob_indices() {
        auto find = [&](const char* id) -> int {
            for (cardinal::usize i = 0; i < knobs_.size(); ++i)
                if (knobs_[i].id == id) return static_cast<int>(i);
            return -1;
        };
        kidx_.backend_mode          = find("backend_mode");
        kidx_.max_tier              = find("max_tier");
        kidx_.max_tier_fp8          = find("max_tier_fp8");
        kidx_.max_tier_fp4          = find("max_tier_fp4");
        kidx_.async_compute         = find("async_compute");
        kidx_.variable_rate_shading = find("variable_rate_shading");
        kidx_.bindless_resources    = find("bindless_resources");
        kidx_.direct_storage        = find("direct_storage");
        kidx_.gpu_execute           = find("gpu_execute");
        kidx_.view_mode             = find("view_mode");
    }
    bool kb(int idx) const {
        return idx >= 0 && knobs_[static_cast<cardinal::usize>(idx)].kind == KnobKind::Bool
             ? knobs_[static_cast<cardinal::usize>(idx)].b : false;
    }
    int ke(int idx, int dflt) const {
        return idx >= 0 && knobs_[static_cast<cardinal::usize>(idx)].kind == KnobKind::Enum
             ? knobs_[static_cast<cardinal::usize>(idx)].e : dflt;
    }

    // Read the backend_mode Knob value at the top of render() and swap
    // the runner's backend if the user changed it in the editor.
    void update_backend_from_knob() {
        const Knob* k = (kidx_.backend_mode >= 0)
            ? &knobs_[static_cast<cardinal::usize>(kidx_.backend_mode)] : nullptr;
        if (!k || k->kind != KnobKind::Enum) return;
        if (k->e < 0 || k->e > 3) return;   // guard the int -> enum cast (4 backends)
        const auto desired = static_cast<AegisBackendMode>(k->e);
        if (desired == current_mode_) return;
        current_mode_ = desired;
        rhi_backend_ = nullptr;   // only set for the Rhi backend below
        switch (desired) {
            case AegisBackendMode::Null:
                runner_->set_backend(graph::NullBackend::create());
                break;
            case AegisBackendMode::Cpu:
                runner_->set_backend(graph::CpuBackend::create());
                break;
            case AegisBackendMode::ThreadedCpu:
                runner_->set_backend(graph::ThreadedCpuBackend::create());
                break;
            case AegisBackendMode::Rhi:
                if (dev_ && sw_) {
                    // Keep our own handle so the gpu_execute knob can drive the
                    // backend's opt-in GPU dispatch (default off — recording-only).
                    rhi_backend_ = graph::RhiBackend::create(*dev_, *sw_);
                    runner_->set_backend(rhi_backend_);
                } else {
                    runner_->set_backend(graph::NullBackend::create());
                }
                break;
        }
    }

    // Read the precision + GPU-Feature knobs into an AegisFeatureRequest.
    // Index-cached — this runs every frame via apply_config().
    AegisFeatureRequest read_request() const {
        AegisFeatureRequest r;
        r.max_tier_want         = ke(kidx_.max_tier, 0);   // 0..3 = FP32/16/8/4
        r.allow_fp8             = kb(kidx_.max_tier_fp8);
        r.allow_fp4             = kb(kidx_.max_tier_fp4);
        r.async_compute         = kb(kidx_.async_compute);
        r.variable_rate_shading = kb(kidx_.variable_rate_shading);
        r.bindless_resources    = kb(kidx_.bindless_resources);
        r.direct_storage        = kb(kidx_.direct_storage);
        return r;
    }

    // Resolve the desired AegisConfig from the cached device support + the
    // current knob request, and reconfigure the runner ONLY when it changed
    // (rebuilding the AEGIS pipeline every frame would be wasteful). This is
    // what makes FP8/FP4 escalation + the feature flags actually take effect.
    void apply_config() {
        const cardinal::u32 w = sw_ ? sw_->width()  : 0u;
        const cardinal::u32 h = sw_ ? sw_->height() : 0u;
        gpu::AegisConfig want =
            aegis_resolve_config(runner_->config(), dev_caps_, read_request(), w, h);
        if (!aegis_config_equal(want, runner_->config()))
            runner_->reconfigure(want);
    }

    PipelineId  id()          const noexcept override { return PipelineId::Aegis; }
    const char* name()        const noexcept override { return "AEGIS Pipeline 2.0"; }
    const char* description() const noexcept override {
        return "Graph-driven GPU pipeline matching the AEGIS Render Pipeline 2.0 "
               "spec: Virtual Geometry classify/meshlet/SSE + frustum/Hi-Z cull + "
               "adaptive math-division tier + V-Buffer + tile light cull + ReSTIR DI "
               "(opt-in) + TAA + Tonemap + Composite. Topology runs on the graph "
               "backend (selectable via knob); on-screen draw delegates to the "
               "ForwardRenderer until RhiBackend lands.";
    }
    cardinal::vector<Knob>& knobs() noexcept override { return knobs_; }

    void on_caps(const rhi::GpuCapabilities& caps) override {
        // Cache the device's AEGIS-relevant support so the per-frame
        // apply_config() can clamp the knob requests against real hardware.
        dev_caps_.fp16 = caps.shader_float16;
        dev_caps_.fp8  = caps.fp8_math;
        dev_caps_.fp4  = caps.fp4_math;
        dev_caps_.async_compute         = caps.async_compute;
        dev_caps_.variable_rate_shading = caps.variable_rate_shading;
        dev_caps_.bindless_resources    = caps.bindless_resources;
        dev_caps_.direct_storage        = caps.direct_storage_capable;

        // Per-knob availability gating. Walks every knob whose ID matches a
        // GpuCapabilities feature flag and sets `available` + a friendly
        // unavailable_reason. The Render Pipeline panel greys out greyed-
        // out knobs and shows the reason on hover.
        const auto gate = [&](const char* knob_id, bool supported,
                              const char* reason) {
            for (auto& k : knobs_) {
                if (k.id == knob_id) {
                    k.available = supported;
                    k.unavailable_reason = supported ? "" : reason;
                    return;
                }
            }
        };
        gate("async_compute",         caps.async_compute,
             "Device has no separate async compute queue");
        gate("variable_rate_shading", caps.variable_rate_shading,
             "Device lacks VRS Tier 1 support");
        gate("mesh_shaders",          caps.mesh_shader,
             "Device has no mesh-shader pipeline");
        gate("bindless_resources",    caps.bindless_resources,
             "Device lacks bindless descriptor heaps");
        gate("ray_tracing",           caps.ray_tracing_pipeline,
             "Device has no ray-tracing pipeline");
        gate("ray_traced_shadows",    caps.ray_tracing_pipeline,
             "Ray-traced shadows need a ray-tracing GPU");
        gate("rt_reflections",        caps.ray_tracing_pipeline,
             "Ray-traced reflections need a ray-tracing GPU");
        gate("restir_di",             caps.ray_tracing_pipeline,
             "ReSTIR DI needs a ray-tracing GPU");
        gate("dlss_upscaler",         caps.nvidia_dlss_capable,
             "DLSS requires an NVIDIA RTX 20-series or newer GPU");
        gate("fsr_upscaler",          caps.amd_fsr3_capable,
             "FSR 3 requires an AMD RX 6000-series or newer GPU");
        gate("direct_storage",        caps.direct_storage_capable,
             "DirectStorage / RTX IO not present on this device");
        gate("max_tier_fp8",          caps.fp8_math,
             "Device lacks FP8 math (Hopper / Ada / RDNA4 required)");
        gate("max_tier_fp4",          caps.fp4_math,
             "Device lacks FP4 math (Blackwell / RDNA5 required)");

        // Device support just resolved — push the clamped config into the runner.
        apply_config();
    }

    void render(scene::Scene& scn, float aspect) override {
        if (!renderer_) return;

        // Pick up any editor-driven backend change before this frame's
        // build + execute.
        update_backend_from_knob();
        // Push knob/caps changes (precision tier + GPU features) into the
        // runner config — diff-gated, so it only reconfigures on a real change.
        apply_config();

        // Honour the experimental GPU-compute-execute toggle: when on, the
        // RhiBackend actually dispatches the AEGIS compute graph on the GPU;
        // when off (default) it stays recording-only telemetry while the
        // ForwardRenderer draws the screen (the stable path).
        if (rhi_backend_) {
            rhi_backend_->set_gpu_execute(kb(kidx_.gpu_execute));
            // Async-compute lane: when the resolved config carries
            // async_compute (device-supported AND requested), the whole AEGIS
            // compute graph is recorded + submitted on the dedicated async
            // queue — concurrent with this frame's graphics work. The queue
            // member is declared ABOVE rhi_backend_ so it destructs AFTER the
            // backend's drain (set_async_queue holds a non-owning pointer).
            const bool want_async = runner_->config().features.async_compute;
            if (want_async && async_queue_ == nullptr && dev_ != nullptr)
                async_queue_ = dev_->create_async_compute_queue();
            rhi_backend_->set_async_queue(
                want_async ? async_queue_.get() : nullptr);
        }

        // PERSISTENT GRAPH: the AEGIS topology is static for a given config,
        // so the graph is built ONCE and retained — reconfigure() (resize /
        // precision / feature change) invalidates it via is_built(). Only the
        // imported camera bytes below change per frame; every backend re-reads
        // imports at execute. NOTE: the AegisSceneInputs SHAPE (triangle_count,
        // which optional handles are valid) is baked at build — if this bridge
        // ever feeds real scene streams, it must force a rebuild when the
        // shape changes (set a dirty flag; see aegis_runner.hpp is_built()).
        if (!runner_->is_built()) {
            graph::Graph& gr = runner_->begin_build();
            gpu::AegisSceneInputs in;
            in.triangle_count = 0;
            in.tris         = gr.declare_buffer(graph::BufferDesc{"tris",  0, 0, true});
            in.material_ids = gr.declare_buffer(graph::BufferDesc{"mids",  0, 0, true});
            in.materials    = gr.declare_buffer(graph::BufferDesc{"mats",  0, 0, true});
            in.lights       = gr.declare_buffer(graph::BufferDesc{"lts",   0, 0, true});
            in.ambient      = gr.import_buffer("amb", amb_data_, 12, 4);
            in.view_proj    = gr.import_buffer("vp",  vp_data_,  64, 4);
            in.camera_dir   = gr.import_buffer("dir", dir_data_, 12, 4);
            // PBR pair: real inverse view-proj + camera position switch the
            // V-Buf resolve from the ambient proxy to tile-lit Cook-Torrance.
            in.inv_view_proj = gr.import_buffer("ivp",  ivp_data_,    64, 4);
            in.camera_pos    = gr.import_buffer("cpos", campos_data_, 12, 4);
            runner_->build(in);
        }

        // Fresh camera bytes every frame. AEGIS passes read the matrix
        // ROW-major (4 consecutive floats = one row, w-row at [12..15]);
        // cardinal Mat4 storage is COLUMN-major (m[col][row]) — transpose on
        // pack. A raw memcpy would compile and silently mis-cull (identity
        // matrices in tests are transpose-symmetric and hide it).
        {
            const float safe_aspect = (aspect > 0.001f) ? aspect : 1.0f;
            const auto vp = scn.camera().proj(safe_aspect) * scn.camera().view();
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    vp_data_[r * 4 + c] = vp.m[c][r];
            const auto ivp = vp.inverse();
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    ivp_data_[r * 4 + c] = ivp.m[c][r];
            const scene::Vec3 eye = scn.camera().position;
            const scene::Vec3 tgt = scn.camera().target;
            campos_data_[0] = eye.x; campos_data_[1] = eye.y; campos_data_[2] = eye.z;
            float fx = tgt.x - eye.x, fy = tgt.y - eye.y, fz = tgt.z - eye.z;
            const float len2 = fx * fx + fy * fy + fz * fz;
            if (len2 > 1.0e-12f) {
                const float inv = 1.0f / cardinal::sqrt(len2);
                dir_data_[0] = fx * inv; dir_data_[1] = fy * inv; dir_data_[2] = fz * inv;
            } else {
                dir_data_[0] = 0.0f; dir_data_[1] = 0.0f; dir_data_[2] = 1.0f;
            }
        }

        runner_->execute();

        // Delegate the actual draw to the ForwardRenderer, honouring the
        // per-viewport view mode the Studio pushed into our "view_mode" knob
        // (Solid / Wireframe / Polygons / Heightmap / Normals / RTX Preview).
        // When graph::RhiBackend ships its own raster output this fallthrough
        // goes away, but the view-mode contract stays the same.
        const int vm_e_raw = ke(kidx_.view_mode, 0);
        const int vm_e     = (vm_e_raw < 0) ? 0 : (vm_e_raw > 5 ? 5 : vm_e_raw);
        renderer_->render(scn, static_cast<scene::ViewMode>(vm_e), aspect);
    }

    void set_light_set(const scene::LightSet* lights) override {
        if (renderer_) renderer_->set_light_set(lights);
    }
    void set_show_aabbs  (bool on) override { if (renderer_) renderer_->set_show_aabbs(on);   }
    void set_show_frustum(bool on) override { if (renderer_) renderer_->set_show_frustum(on); }
    void set_show_axes   (bool on) override { if (renderer_) renderer_->set_show_axes(on);    }
    void set_show_lights (bool on) override { if (renderer_) renderer_->set_show_lights(on);  }

    FrameStats stats() const noexcept override {
        FrameStats s{};
        if (renderer_) {
            const auto srs = renderer_->frame_stats();
            s.entities_total  = srs.entities_total;
            s.entities_drawn  = srs.entities_drawn;
            s.entities_culled = srs.entities_culled;
            s.draw_calls      = srs.draw_calls;
            s.pc_pushes       = srs.pc_pushes;
            s.cull_us         = srs.cull_us;
            s.phase1_us       = srs.phase1_us;
            s.phase2_us       = srs.phase2_us;
            s.sort_us         = srs.sort_us;
            s.submit_us       = srs.submit_us;
        }
        // Surface the AEGIS graph's pass count as cluster_total so the
        // editor's existing "clusters" widget can display it without
        // schema changes — the editor reads cluster_* fields uniformly
        // across pipelines and clusters here is "AEGIS graph nodes".
        s.clusters_total = runner_ ? runner_->compile_stats().passes : 0;
        s.clusters_drawn = s.clusters_total;
        return s;
    }

private:
    // ---- Knob factory helpers -----------------------------------------
    // Keep build_knobs() readable — every knob would otherwise be a 6-line
    // brace block. These helpers fold the boilerplate into one-liners.
    Knob& add_bool_(const char* id, const char* label, const char* group,
                    const char* tip, bool def) {
        Knob k;
        k.id = id; k.label = label; k.group = group; k.tooltip = tip;
        k.kind = KnobKind::Bool; k.b = def;
        knobs_.push_back(cardinal::move(k));
        return knobs_.back();
    }
    Knob& add_int_(const char* id, const char* label, const char* group,
                   const char* tip, int def, int lo, int hi) {
        Knob k;
        k.id = id; k.label = label; k.group = group; k.tooltip = tip;
        k.kind = KnobKind::Int; k.i = def; k.i_min = lo; k.i_max = hi;
        knobs_.push_back(cardinal::move(k));
        return knobs_.back();
    }
    Knob& add_float_(const char* id, const char* label, const char* group,
                     const char* tip, float def, float lo, float hi,
                     float step = 0.01f) {
        Knob k;
        k.id = id; k.label = label; k.group = group; k.tooltip = tip;
        k.kind = KnobKind::Float;
        k.f = def; k.f_min = lo; k.f_max = hi; k.f_step = step;
        knobs_.push_back(cardinal::move(k));
        return knobs_.back();
    }
    Knob& add_enum_(const char* id, const char* label, const char* group,
                    const char* tip, int def,
                    std::initializer_list<const char*> labels) {
        Knob k;
        k.id = id; k.label = label; k.group = group; k.tooltip = tip;
        k.kind = KnobKind::Enum; k.e = def;
        for (const char* s : labels) k.enum_labels.push_back(s);
        knobs_.push_back(cardinal::move(k));
        return knobs_.back();
    }

    void build_knobs() {
        knobs_.clear();

        // ---- Visualisation -------------------------------------------
        // Per-viewport view mode. The Studio pushes the focused viewport's
        // scene::ViewMode into this knob each frame (by id "view_mode", the
        // same contract ForwardBaseline uses), and render() forwards it to the
        // on-screen ForwardRenderer — so Solid / Wireframe / Polygons /
        // Heightmap / Normals / RTX Preview all work under AEGIS. Indices
        // match scene::ViewMode exactly.
        add_enum_("view_mode", "View Mode", "Visualisation",
            "How the on-screen forward draw shades each fragment. Driven by the "
            "viewport toolbar (per-viewport); mirrored here for reference.",
            0, {"Solid", "Wireframe", "Polygons", "Heightmap", "Normals", "RTX Preview"});

        // ---- AEGIS core ----------------------------------------------
        add_enum_("backend_mode", "Graph Backend", "AEGIS",
            "Which graph::Backend the AEGIS runner executes against. "
            "Null = topology only (per-frame stats, no buffers). "
            "Cpu / ThreadedCpu = full virtual-GPU simulation. "
            "Rhi = real GPU compute dispatch (default; degrades to recording-"
            "only telemetry until the device implements compute pipelines).",
            3, {"CpuBackend", "Null (topology)", "ThreadedCpuBackend", "RhiBackend"});
        add_enum_("max_tier", "Max Geometry Tier", "AEGIS",
            "Maximum precision tier the math-division engine is allowed to "
            "escalate to. FP4 (Blackwell) = 8 micro-tris per source triangle; "
            "FP32 = no subdivision.",
            1, {"FP32", "FP16", "FP8", "FP4"});
        add_bool_("virtual_geometry", "Virtual Geometry", "AEGIS",
            "GPU-driven cluster-LOD geometry pipeline (Nanite-style): classify -> "
            "meshlet -> screen-space-error LOD select per cluster.", true);
        add_bool_("meshlet_culling", "Meshlet Culling", "AEGIS",
            "Per-cluster frustum + backface-cone + small-feature culling before "
            "the V-Buffer rasterizer.", true);
        add_bool_("hiz_occlusion", "Hi-Z Occlusion Culling", "AEGIS",
            "Two-pass hierarchical-Z occlusion test against last frame's depth so "
            "hidden clusters never reach the rasterizer.", true);
        add_bool_("gpu_execute", "GPU Compute Execute (experimental)", "AEGIS",
            "Run the AEGIS compute graph on the GPU through RhiBackend instead of "
            "recording-only telemetry. EXPERIMENTAL + unvalidated on real hardware: "
            "enable to test GPU dispatch (buffers are now cached across frames). "
            "Leave OFF for the stable ForwardRenderer path.", false);

        // ---- Quality / Resolution -------------------------------------
        add_float_("resolution_scale", "Resolution Scale", "Quality",
            "Internal render-target resolution multiplier before upscaling. "
            "0.5 = quarter pixels (huge perf); 2.0 = SSAA x2 (huge quality cost).",
            1.0f, 0.50f, 2.0f, 0.05f);
        add_enum_("msaa_samples", "MSAA Samples", "Quality",
            "Multi-sample anti-aliasing for the forward path. Ignored when a "
            "temporal upscaler (DLSS / FSR) is active.",
            0, {"1x (off)", "2x", "4x", "8x"});
        add_enum_("anisotropic_filter", "Anisotropic Filter", "Quality",
            "Texture filtering for sampled material textures.",
            3, {"Bilinear", "Trilinear", "Aniso 2x", "Aniso 4x", "Aniso 8x", "Aniso 16x"});
        // ---- Shadows --------------------------------------------------
        add_enum_("shadow_resolution", "Shadow Atlas", "Shadows",
            "Per-light shadow-map dimensions. 8k is for cinematic captures.",
            2, {"512", "1024", "2048", "4096", "8192"});
        add_int_("shadow_cascades", "Shadow Cascades", "Shadows",
            "Number of cascaded shadow-map slices for the sun directional.",
            3, 1, 4);
        add_enum_("shadow_filter", "Shadow Filtering", "Shadows",
            "Hard = aliased edges (cheapest). PCF = percentage-closer soft edges. "
            "PCSS = contact-hardening soft shadows (penumbra grows with distance).",
            1, {"Hard", "PCF", "PCSS (soft)"});
        add_float_("shadow_distance", "Shadow Distance (m)", "Shadows",
            "Max world distance the sun cascades cover; beyond it geometry is unshadowed.",
            150.0f, 10.0f, 2000.0f, 10.0f);
        add_bool_("contact_shadows", "Contact Shadows", "Shadows",
            "Screen-space short-range ray-march that fills the gaps small geometry "
            "leaves in cascaded shadow maps.", true);
        add_bool_("ray_traced_shadows", "Ray-Traced Shadows", "Shadows",
            "Hardware-RT shadow rays for pixel-accurate penumbra. Needs an RT GPU.",
            false);

        // ---- Frame pacing ---------------------------------------------
        add_enum_("vsync_mode", "VSync", "Frame Pacing",
            "Off = uncapped (tear allowed). On = wait for vblank. Adaptive = "
            "off when below refresh, on when above (G-Sync / FreeSync feel).",
            0, {"Off", "On", "Adaptive"});
        add_enum_("fps_limit", "FPS Limit", "Frame Pacing",
            "Hard cap on rendered frames per second. None = no cap (relies on "
            "VSync or vendor-side limiter).",
            0, {"None", "30", "60", "90", "120", "144", "165", "240"});
        add_float_("frame_pacing_smoothing", "Pacing Smoothing", "Frame Pacing",
            "How aggressively the engine smooths frame-time jitter. 0 = raw, "
            "1 = heavy averaging (smoother but laggier).",
            0.3f, 0.0f, 1.0f, 0.05f);

        // ---- Post-FX --------------------------------------------------
        add_float_("exposure", "Exposure (EV)", "Post-FX",
            "TonemapPass exposure multiplier (linear stops above middle grey).",
            1.0f, 0.0f, 4.0f, 0.05f);
        add_enum_("tonemap_operator", "Tonemap Operator", "Post-FX",
            "How linear HDR is mapped to LDR display range. AgX is the modern "
            "preference; ACES is the film-industry standard; Reinhard is the "
            "classic; Uncharted2 is Hable's curve.",
            1, {"Reinhard", "ACES Fitted", "AgX", "Uncharted 2", "PBR Neutral"});
        add_bool_("bloom_enabled", "Bloom", "Post-FX",
            "Bright-pass blur composited back over the HDR target.", true);
        add_float_("bloom_threshold", "Bloom Threshold", "Post-FX",
            "Minimum luminance (cd/m²-ish) that contributes to bloom.",
            1.0f, 0.0f, 5.0f, 0.05f);
        add_float_("bloom_intensity", "Bloom Intensity", "Post-FX",
            "Scale on the bloom mip-chain composite.",
            0.5f, 0.0f, 2.0f, 0.05f);
        add_float_("sharpness", "CAS Sharpness", "Post-FX",
            "AMD Contrast-Adaptive Sharpening strength applied post-upscale.",
            0.25f, 0.0f, 1.0f, 0.05f);
        add_float_("chromatic_aberration", "Chromatic Aberration", "Post-FX",
            "Per-channel screen-space offset (in screen units × 1000).",
            0.0f, 0.0f, 1.0f, 0.05f);
        add_float_("vignette", "Vignette", "Post-FX",
            "Edge-darkening intensity.",
            0.0f, 0.0f, 1.0f, 0.05f);
        add_bool_("motion_blur", "Motion Blur", "Post-FX",
            "Per-object + camera velocity blur reconstructed from the motion-vector "
            "buffer.", false);
        add_float_("motion_blur_amount", "Motion Blur Amount", "Post-FX",
            "Shutter-angle scale on the velocity blur.",
            0.5f, 0.0f, 1.0f, 0.05f);
        add_bool_("depth_of_field", "Depth of Field", "Post-FX",
            "Physically-based bokeh DoF (circle-of-confusion from focus + aperture).",
            false);
        add_float_("dof_focus_distance", "DoF Focus (m)", "Post-FX",
            "World distance the lens is focused at.",
            10.0f, 0.1f, 1000.0f, 0.5f);
        add_float_("dof_aperture", "DoF Aperture (f-stop)", "Post-FX",
            "Lens f-number — smaller = shallower depth of field + larger bokeh.",
            5.6f, 1.0f, 22.0f, 0.1f);
        add_float_("film_grain", "Film Grain", "Post-FX",
            "Animated luminance noise overlaid for a filmic feel.",
            0.0f, 0.0f, 1.0f, 0.05f);

        // ---- Lighting -------------------------------------------------
        add_bool_("ibl_diffuse",  "IBL Diffuse", "Lighting",
            "Image-based diffuse from the irradiance probe volume.", true);
        add_bool_("ibl_specular", "IBL Specular", "Lighting",
            "Image-based specular from the pre-filtered environment cube.", true);
        add_bool_("ssao_enabled", "SSAO", "Lighting",
            "Screen-space ambient occlusion (GTAO-style).", true);
        add_enum_("ssao_quality", "SSAO Quality", "Lighting",
            "Sample count + radius per AO probe.",
            1, {"Low (4 taps)", "Med (8 taps)", "High (16 taps)", "Ultra (32 taps)"});
        // ---- Reflections ----------------------------------------------
        add_bool_("ssr_enabled", "Screen-Space Reflections", "Reflections",
            "Screen-space reflections (linear ray-march + Hi-Z accelerator).", false);
        add_enum_("ssr_quality", "SSR Quality", "Reflections",
            "Reflection ray step count + roughness importance sampling.",
            1, {"Low", "Med", "High"});
        add_bool_("rt_reflections", "Ray-Traced Reflections", "Reflections",
            "Hardware-RT reflections for off-screen + rough surfaces SSR can't reach. "
            "Needs an RT GPU.", false);
        add_enum_("reflection_quality", "RT Reflection Quality", "Reflections",
            "Ray count + denoiser strength for ray-traced reflections.",
            1, {"Low", "Med", "High", "Ultra"});

        // ---- Global Illumination --------------------------------------
        add_enum_("gi_mode", "Global Illumination", "Global Illumination",
            "Indirect-light technique. SSGI is screen-space (no HW RT). RTGI + "
            "ReSTIR GI trace real rays (need an RT GPU) for off-screen bounce light.",
            1, {"Off", "SSGI", "RTGI", "ReSTIR GI"});
        add_int_("gi_bounces", "GI Bounces", "Global Illumination",
            "Indirect light bounces traced per frame (RT modes).",
            1, 1, 4);
        add_float_("gi_intensity", "GI Intensity", "Global Illumination",
            "Scale on the gathered indirect irradiance.",
            1.0f, 0.0f, 4.0f, 0.05f);
        add_bool_("restir_di", "ReSTIR DI", "Global Illumination",
            "Reservoir spatiotemporal importance resampling for many-light direct "
            "lighting (the AEGIS DI pass). Needs an RT GPU.", false);

        // ---- Upscaler -------------------------------------------------
        add_enum_("dlss_upscaler", "DLSS", "Upscaler",
            "NVIDIA Deep Learning Super Sampling. Quality > Balanced > "
            "Performance trade screen-space samples for speed.",
            0, {"Off", "DLAA (native)", "Quality", "Balanced", "Performance", "Ultra Performance"});
        add_enum_("fsr_upscaler", "FSR 3", "Upscaler",
            "AMD FidelityFX Super Resolution. Quality > Balanced > Performance.",
            0, {"Off", "Native AA", "Quality", "Balanced", "Performance"});
        add_bool_("frame_generation", "DLSS / FSR Frame Generation", "Upscaler",
            "Synthesise an interpolated frame between every rendered pair. "
            "Doubles displayed FPS at the cost of input latency.", false);

        // ---- GPU Features (feature-gated via on_caps()) ---------------
        add_bool_("async_compute", "Async Compute", "GPU Features",
            "Run compute passes on a dedicated GPU queue concurrent with the "
            "graphics queue. Effective on hardware with multiple compute engines.",
            false);
        add_bool_("variable_rate_shading", "Variable Rate Shading", "GPU Features",
            "Coarse-shade screen regions the eye doesn't focus on. Big win in "
            "VR foveated rendering and post-FX shaders.", false);
        add_bool_("mesh_shaders", "Mesh Shaders", "GPU Features",
            "Use mesh + task shaders instead of vertex + geometry pipelines. "
            "Required for the Block 4 GPU-driven cluster pipeline.", false);
        add_bool_("bindless_resources", "Bindless Resources", "GPU Features",
            "Replace per-draw descriptor binding with a single huge descriptor "
            "heap indexed in-shader. Required for Block 8 material extensions.",
            false);
        add_bool_("ray_tracing", "Ray Tracing Pipeline", "GPU Features",
            "Hardware RT — required for ReSTIR DI / GI and ray-traced shadows.",
            false);
        add_bool_("direct_storage", "DirectStorage / RTX IO", "GPU Features",
            "GPU-decompressed asset streaming straight from NVMe to VRAM, "
            "bypassing the CPU.", false);
        add_bool_("max_tier_fp8", "Allow FP8 Tier", "GPU Features",
            "Permit the math-division engine to escalate to FP8 (Hopper / Ada / RDNA4).",
            false);
        add_bool_("max_tier_fp4", "Allow FP4 Tier", "GPU Features",
            "Permit the math-division engine to escalate to FP4 (Blackwell / RDNA5).",
            false);

        // ---- Anti-Aliasing --------------------------------------------
        add_enum_("aa_mode", "Anti-Aliasing", "Anti-Aliasing",
            "Edge AA technique. FXAA = cheap post blur. SMAA = sharper morphological. "
            "TAA = temporal accumulation (best, but needs motion vectors + jitter).",
            3, {"Off", "FXAA", "SMAA", "TAA"});
        add_float_("taa_sharpness", "TAA Sharpness", "Anti-Aliasing",
            "Post-resolve sharpening to counter TAA's inherent softness.",
            0.5f, 0.0f, 1.0f, 0.05f);
        add_bool_("taa_upsampling", "TAA Upsampling", "Anti-Aliasing",
            "Let TAA also upscale from the internal resolution scale (TAAU).",
            false);

        // ---- Color Grading --------------------------------------------
        add_float_("cg_temperature", "Temperature", "Color Grading",
            "White-balance Kelvin shift: negative = cooler/blue, positive = warmer/orange.",
            0.0f, -1.0f, 1.0f, 0.05f);
        add_float_("cg_tint", "Tint", "Color Grading",
            "White-balance green<->magenta axis.",
            0.0f, -1.0f, 1.0f, 0.05f);
        add_float_("cg_saturation", "Saturation", "Color Grading",
            "Chroma scale: 0 = greyscale, 1 = unchanged, 2 = vivid.",
            1.0f, 0.0f, 2.0f, 0.05f);
        add_float_("cg_contrast", "Contrast", "Color Grading",
            "Tonal contrast about middle grey.",
            1.0f, 0.0f, 2.0f, 0.05f);
        add_bool_("cg_lut_enabled", "Apply Color LUT", "Color Grading",
            "Apply the project's 3D colour lookup table after grading.", false);

        // ---- Atmosphere -----------------------------------------------
        add_bool_("fog_enabled", "Fog", "Atmosphere",
            "Exponential height + distance fog.", false);
        add_float_("fog_density", "Fog Density", "Atmosphere",
            "Extinction coefficient for the distance fog.",
            0.02f, 0.0f, 1.0f, 0.005f);
        add_bool_("volumetric_fog", "Volumetric Fog", "Atmosphere",
            "Froxel-based volumetric fog that scatters light through participating media.",
            false);
        add_bool_("volumetric_lighting", "Volumetric Lighting", "Atmosphere",
            "Light-shaft / god-ray scattering from shadowed light sources.", false);

        // ---- Detail & Streaming ---------------------------------------
        add_int_("lod_bias", "LOD Bias", "Detail & Streaming",
            "Shifts mesh LOD selection: negative = hold higher detail longer "
            "(costlier), positive = drop sooner (faster).",
            0, -3, 3);
        add_enum_("texture_quality", "Texture Quality", "Detail & Streaming",
            "Highest resident mip level for streamed textures.",
            3, {"Low", "Medium", "High", "Ultra"});
        add_float_("mip_bias", "Mip Bias", "Detail & Streaming",
            "Global sampler mip LOD bias; negative sharpens (more aliasing), "
            "positive softens.",
            0.0f, -2.0f, 2.0f, 0.1f);
        add_int_("texture_streaming_budget", "Texture Pool (MB)", "Detail & Streaming",
            "VRAM budget for the streamed-texture pool.",
            2048, 256, 16384);
        add_bool_("virtual_texturing", "Virtual Texturing", "Detail & Streaming",
            "Stream only the texture tiles visible this frame (cardinal::vt) — "
            "decouples texture memory from texture resolution.", true);

        // ---- Debug (pipeline-global only) -----------------------------
        // The per-viewport visualisation CHOICE is made in the VIEWPORT
        // TOOLBAR — each viewport carries its own scene::ViewMode, pushed
        // into the "view_mode" knob above each frame (see render()). These
        // Debug knobs, by contrast, affect the whole engine simulation /
        // cull state regardless of viewport.
        add_bool_("pause_simulation", "Pause Simulation", "Debug",
            "Freeze the scene update; rendering continues.", false);
        add_bool_("freeze_culling", "Freeze Culling Frustum", "Debug",
            "Hold the previous-frame cull frustum so you can fly the camera "
            "OUTSIDE it to see what's being culled.", false);
    }

    rhi::Device*                                      dev_  {nullptr};
    rhi::Swapchain*                                   sw_   {nullptr};
    cardinal::shared_ptr<scene::ForwardRenderer>      renderer_;
    cardinal::shared_ptr<AegisPipelineRunner>          runner_;
    cardinal::vector<Knob>                             knobs_;
    KnobIdx                                            kidx_;   // hot-knob indices
    AegisBackendMode                                   current_mode_ {AegisBackendMode::Null};
    AegisDeviceSupport                                 dev_caps_ {};   // from on_caps
    // Async lane BEFORE the backend: members destruct in reverse order, and
    // the RhiBackend dtor drains through this queue (non-owning pointer).
    cardinal::unique_ptr<cardinal::rhi::ComputeQueue>  async_queue_;
    cardinal::shared_ptr<graph::RhiBackend>            rhi_backend_;   // when mode==Rhi
    // Persistent host storage for the graph's imported camera inputs — the
    // graph keeps raw pointers into these across frames (persistent graph).
    float amb_data_[3] {0.1f, 0.1f, 0.12f};
    float vp_data_[16] {};
    float dir_data_[3] {0.0f, 0.0f, 1.0f};
    float ivp_data_[16] {};
    float campos_data_[3] {};
};

}  // namespace

// Factory exposed to RegistryImpl in pipelines.cpp.
cardinal::unique_ptr<Pipeline> create_aegis_pipeline(rhi::Device& dev,
                                                     rhi::Swapchain& sw) {
    return cardinal::make_unique<AegisGraphPipeline>(dev, sw);
}

}  // namespace cardinal::render
