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
#include <cardinal/core/log.hpp>
#include <cardinal/core/utility.hpp>

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
        cfg.caps.fp16_supported = true;
        cfg.caps.fp8_supported  = false;
        cfg.caps.fp4_supported  = false;
        cfg.max_tier            = gpu::GeometryTier::Fp16;

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
    }

    // Read the backend_mode Knob value at the top of render() and swap
    // the runner's backend if the user changed it in the editor.
    void update_backend_from_knob() {
        const Knob* k = nullptr;
        for (auto& kn : knobs_) {
            if (kn.id == "backend_mode") { k = &kn; break; }
        }
        if (!k || k->kind != KnobKind::Enum) return;
        const auto desired = static_cast<AegisBackendMode>(k->e);
        if (desired == current_mode_) return;
        current_mode_ = desired;
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
                    runner_->set_backend(graph::RhiBackend::create(*dev_, *sw_));
                } else {
                    runner_->set_backend(graph::NullBackend::create());
                }
                break;
        }
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
        // Wire RHI device caps into the runner's PrecisionCaps so the
        // tier selector knows what FP16/8/4 the device actually supports.
        gpu::AegisConfig cfg = runner_->config();
        cfg.caps.fp16_supported = caps.shader_float16;

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
    }

    void render(scene::Scene& scn, float aspect) override {
        if (!renderer_) return;

        // Pick up any editor-driven backend change before this frame's
        // build + execute.
        update_backend_from_knob();

        // Run the AEGIS graph for telemetry. The runner manages its own
        // graph + backend; we just hand it minimal inputs each frame.
        // The actual scene-derived buffer wiring (real triangle stream,
        // material palette, etc.) lands when RhiBackend can consume them.
        gpu::AegisSceneInputs in;
        in.triangle_count = 0;
        in.tris         = runner_->graph().declare_buffer(graph::BufferDesc{"tris",  0, 0, true});
        in.material_ids = runner_->graph().declare_buffer(graph::BufferDesc{"mids",  0, 0, true});
        in.materials    = runner_->graph().declare_buffer(graph::BufferDesc{"mats",  0, 0, true});
        in.lights       = runner_->graph().declare_buffer(graph::BufferDesc{"lts",   0, 0, true});
        in.ambient      = runner_->graph().declare_buffer(graph::BufferDesc{"amb",  12, 0, true});
        in.view_proj    = runner_->graph().declare_buffer(graph::BufferDesc{"vp",   64, 0, true});
        in.camera_dir   = runner_->graph().declare_buffer(graph::BufferDesc{"dir",  12, 0, true});
        if (runner_->build(in)) runner_->execute();

        // Delegate the actual draw to the ForwardRenderer, honouring the
        // per-viewport view mode the Studio pushed into our "view_mode" knob
        // (Solid / Wireframe / Polygons / Heightmap / Normals / RTX Preview).
        // When graph::RhiBackend ships its own raster output this fallthrough
        // goes away, but the view-mode contract stays the same.
        scene::ViewMode vm = scene::ViewMode::Solid;
        for (const auto& k : knobs_) {
            if (k.id == "view_mode" && k.kind == KnobKind::Enum) {
                const int e = (k.e < 0) ? 0 : (k.e > 5 ? 5 : k.e);
                vm = static_cast<scene::ViewMode>(e);
                break;
            }
        }
        renderer_->render(scn, vm, aspect);
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
            3, {"Null (topology)", "CpuBackend", "ThreadedCpuBackend", "RhiBackend"});
        add_enum_("max_tier", "Max Geometry Tier", "AEGIS",
            "Maximum precision tier the math-division engine is allowed to "
            "escalate to. FP4 (Blackwell) = 8 micro-tris per source triangle; "
            "FP32 = no subdivision.",
            1, {"FP32", "FP16", "FP8", "FP4"});

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
        add_enum_("shadow_resolution", "Shadow Atlas", "Quality",
            "Per-light shadow-map dimensions. 8k is for cinematic captures.",
            2, {"512", "1024", "2048", "4096", "8192"});
        add_int_("shadow_cascades", "Shadow Cascades", "Quality",
            "Number of cascaded shadow-map slices for the sun directional.",
            3, 1, 4);

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
        add_bool_("ssr_enabled", "SSR", "Lighting",
            "Screen-space reflections (linear ray-march + Hi-Z accelerator).", false);
        add_enum_("ssr_quality", "SSR Quality", "Lighting",
            "Reflection ray step count + roughness importance sampling.",
            1, {"Low", "Med", "High"});

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
    AegisBackendMode                                   current_mode_ {AegisBackendMode::Null};
};

}  // namespace

// Factory exposed to RegistryImpl in pipelines.cpp.
cardinal::unique_ptr<Pipeline> create_aegis_pipeline(rhi::Device& dev,
                                                     rhi::Swapchain& sw) {
    return cardinal::make_unique<AegisGraphPipeline>(dev, sw);
}

}  // namespace cardinal::render
