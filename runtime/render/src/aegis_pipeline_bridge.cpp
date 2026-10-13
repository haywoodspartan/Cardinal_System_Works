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
#include <cardinal/core/std/cstring.hpp>
#include <cardinal/core/std/utility.hpp>
namespace cardinal::render {
namespace {
// Fullscreen present of the AEGIS composite buffer (RGBA8, W×H, row 0 = top)
// into the active viewport: bufferless triangle from SV_VertexID, the PS maps
// each viewport pixel to the buffer with a nearest stretch. Storage slot 0.
const char* kAegisBlitShader =
    "ByteAddressBuffer InPresent : register(t0);\n"
    "struct Push { uint4 dims; };   // vpW, vpH, W, H\n"
    "#ifdef __spirv__\n"
    "[[vk::push_constant]] Push pc;\n"
    "#else\n"
    "ConstantBuffer<Push> pc : register(b0);\n"
    "#endif\n"
    "float4 VSMain(uint vid : SV_VertexID) : SV_Position {\n"
    "  float2 uv = float2((vid << 1) & 2, vid & 2);\n"
    "  return float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);\n"
    "}\n"
    "float4 PSMain(float4 pos : SV_Position) : SV_Target {\n"
    "  uint vpW = max(pc.dims.x, 1u), vpH = max(pc.dims.y, 1u);\n"
    "  uint W = max(pc.dims.z, 1u),   H = max(pc.dims.w, 1u);\n"
    "  uint sx = min((uint)pos.x * W / vpW, W - 1u);\n"
    "  uint sy = min((uint)pos.y * H / vpH, H - 1u);\n"
    "  uint p = InPresent.Load((sy * W + sx) * 4u);\n"
    "  return float4((p & 0xFFu) / 255.0, ((p >> 8) & 0xFFu) / 255.0,\n"
    "                ((p >> 16) & 0xFFu) / 255.0, 1.0);\n"
    "}\n";


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
            view_mode {-1}, aegis_present {-1},
            ssgi_enabled {-1}, ssgi_radius {-1}, ssgi_intensity {-1},
            ssgi_ao_strength {-1}, ssgi_steps {-1}, hiz_occlusion {-1};
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
        kidx_.aegis_present         = find("aegis_present");
        kidx_.ssgi_enabled          = find("ssgi_enabled");
        kidx_.ssgi_radius           = find("ssgi_radius");
        kidx_.ssgi_intensity        = find("ssgi_intensity");
        kidx_.ssgi_ao_strength      = find("ssgi_ao_strength");
        kidx_.ssgi_steps            = find("ssgi_steps");
        kidx_.hiz_occlusion         = find("hiz_occlusion");
    }
    bool kb(int idx) const {
        return idx >= 0 && knobs_[static_cast<cardinal::usize>(idx)].kind == KnobKind::Bool
             ? knobs_[static_cast<cardinal::usize>(idx)].b : false;
    }
    int ke(int idx, int dflt) const {
        return idx >= 0 && knobs_[static_cast<cardinal::usize>(idx)].kind == KnobKind::Enum
             ? knobs_[static_cast<cardinal::usize>(idx)].e : dflt;
    }
    float kf(int idx, float dflt) const {
        return idx >= 0 && knobs_[static_cast<cardinal::usize>(idx)].kind == KnobKind::Float
             ? knobs_[static_cast<cardinal::usize>(idx)].f : dflt;
    }
    int ki(int idx, int dflt) const {
        return idx >= 0 && knobs_[static_cast<cardinal::usize>(idx)].kind == KnobKind::Int
             ? knobs_[static_cast<cardinal::usize>(idx)].i : dflt;
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
        hiz_valid_data_ = 0u;     // a fresh backend has no pyramid history yet
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
        r.screen_space_gi       = kidx_.ssgi_enabled >= 0 ? kb(kidx_.ssgi_enabled) : true;
        r.ssgi_radius           = kf(kidx_.ssgi_radius,      1.0f);
        r.ssgi_intensity        = kf(kidx_.ssgi_intensity,   1.0f);
        r.ssgi_ao_strength      = kf(kidx_.ssgi_ao_strength, 1.0f);
        r.ssgi_steps            = ki(kidx_.ssgi_steps,       4);
        r.hiz_occlusion         = kidx_.hiz_occlusion >= 0 ? kb(kidx_.hiz_occlusion) : true;
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
        // Real scene feed: the palette sizes are baked into the graph
        // (buffer sizes + push constants), so a change reconfigures — which
        // invalidates the persistent graph and forces the rebuild below.
        want.material_count = mat_count_;
        want.light_count    = light_count_;
        if (!aegis_config_equal(want, runner_->config()))
            runner_->reconfigure(want);
    }
    // ---- Real scene feed --------------------------------------------------
    // Flattens the scene into the AEGIS input streams every frame, in place
    // (the persistent graph keeps raw pointers into these vectors; they only
    // reallocate when a COUNT changes, which also forces a graph rebuild):
    //   tris_   : world-space triangle list, 9 f32 / tri (skinned meshes are
    //             LBS-deformed on the CPU exactly like the forward path)
    //   mids_   : material index per tri (palette deduped on tint/rough/metal)
    //   mats_   : 8 f32 / material — base rgb, roughness, metalness, 3 spare
    //   lights_ : 16 f32 PackedLight — kind, _, pos×3, dir×3, colour×3,
    //             intensity, range, spot inner/outer cos, _
    // Empty palettes get one placeholder entry: a 0-byte imported buffer is
    // skipped by the RhiBackend and every pass touching it would stub.
    void extract_scene(scene::Scene& scn, bool feed) {
        struct MatKey { float r, g, b, ro, me; };
        cardinal::vector<MatKey> keys;
        mats_.clear();
        if (!feed) {
            // Telemetry-only frames skip the flatten (it duplicates the forward
            // renderer's transform work) and feed placeholder palettes.
            tris_.clear(); mids_.clear(); tri_count_ = 0;
            const float m[8] = { 0.8f, 0.8f, 0.8f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f };
            for (float f : m) mats_.push_back(f);
            mat_count_ = 1;
            lights_.assign(16, 0.0f);
            light_count_ = 1;
            return;
        }
        auto mat_index = [&](const scene::Entity& e) -> cardinal::u32 {
            const MatKey k{ e.tint.x, e.tint.y, e.tint.z,
                            e.material.roughness, e.material.metalness };
            for (cardinal::usize i = 0; i < keys.size(); ++i)
                if (cardinal::memcmp(&keys[i], &k, sizeof k) == 0) return static_cast<cardinal::u32>(i);
            keys.push_back(k);
            const float m[8] = { k.r, k.g, k.b, k.ro, k.me, 0.0f, 0.0f, 0.0f };
            for (float f : m) mats_.push_back(f);
            return static_cast<cardinal::u32>(keys.size() - 1);
        };
        cardinal::usize tri_total = 0;
        for (const auto& e : scn.entities())
            if (e.visible && e.mesh && e.mesh->cpu_vertices())
                tri_total += e.mesh->vertex_count() / 3;
        // ZERO-COPY: when the GPU graph is live and the stream shape is
        // unchanged, write straight into the RhiBackend's next unified-memory
        // import slot (VRAM the CPU can map) instead of into tris_/mids_ and
        // letting execute() memcpy them across. The host arrays keep their
        // (stale) size for the CPU backends + the rebuild frame.
        float*         tri_dst = nullptr;
        cardinal::u32* mid_dst = nullptr;
        bool zero_copy = false;
        if (zero_copy_ok_ && rhi_backend_ && tri_total > 0 && built_tri_count_ == tri_total) {
            tri_dst = static_cast<float*>(rhi_backend_->import_write_ptr("tris", tri_total * 36u));
            mid_dst = static_cast<cardinal::u32*>(rhi_backend_->import_write_ptr("mids", tri_total * 4u));
            zero_copy = (tri_dst != nullptr) && (mid_dst != nullptr);
        }
        if (!zero_copy) {
            tris_.resize(tri_total * 9);
            mids_.resize(tri_total);
            tri_dst = tris_.data();
            mid_dst = mids_.data();
        }
        cardinal::usize t = 0;
        for (const auto& e : scn.entities()) {
            if (!e.visible || !e.mesh) continue;
            const scene::Vertex* src = e.mesh->cpu_vertices();
            const cardinal::u32 vc = (e.mesh->vertex_count() / 3) * 3;
            if (!src || vc == 0) continue;
            const cardinal::u32 mid = mat_index(e);
            const scene::Mat4 model = e.transform.matrix();
            const scene::Mat4* pal = nullptr;
            cardinal::u32 palc = 0;
            const scene::skin::Influence* infl = nullptr;
            if (e.skeleton && e.mesh->skinned()) {
                scene::skin::compute_palette(*e.skeleton, e.pose, skin_scratch_global_, skin_scratch_palette_);
                pal = skin_scratch_palette_.data();
                palc = static_cast<cardinal::u32>(skin_scratch_palette_.size());
                infl = e.mesh->skin_influences();
            }
            for (cardinal::u32 v = 0; v < vc; ++v) {
                scene::Vec3 p = src[v].position;
                if (pal && infl) {
                    scene::Vec3 n;
                    scene::skin::skin_point(infl[v], pal, palc, src[v].position, src[v].normal, p, n);
                }
                const scene::Vec4 w = model * scene::Vec4{ p.x, p.y, p.z, 1.0f };
                float* o = tri_dst + t * 9 + (v % 3) * 3;
                o[0] = w.x; o[1] = w.y; o[2] = w.z;
                if ((v % 3) == 2) { mid_dst[t] = mid; ++t; }
            }
        }
        tri_count_ = static_cast<cardinal::u32>(t);
        if (zero_copy) {
            rhi_backend_->mark_import_host_written("tris");
            rhi_backend_->mark_import_host_written("mids");
            ++zero_copy_frames_;
        }
        if (mats_.empty()) {
            const float m[8] = { 0.8f, 0.8f, 0.8f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f };
            for (float f : m) mats_.push_back(f);
        }
        mat_count_ = static_cast<cardinal::u32>(mats_.size() / 8);
        lights_.clear();
        if (light_set_) {
            for (const auto& l : light_set_->lights()) {
                float L[16] = {};
                L[0]  = (l.kind == scene::LightKind::Directional) ? 0.0f : 1.0f;
                L[2]  = l.position.x;  L[3]  = l.position.y;  L[4]  = l.position.z;
                L[5]  = l.direction.x; L[6]  = l.direction.y; L[7]  = l.direction.z;
                L[8]  = l.color.x;     L[9]  = l.color.y;     L[10] = l.color.z;
                L[11] = l.intensity;   L[12] = l.range;
                L[13] = l.spot_inner_cos; L[14] = l.spot_outer_cos;
                for (float f : L) lights_.push_back(f);
            }
            const scene::Vec3 a = light_set_->ambient();
            amb_data_[0] = a.x; amb_data_[1] = a.y; amb_data_[2] = a.z;
        }
        if (lights_.empty()) lights_.assign(16, 0.0f);   // one zero-intensity placeholder
        light_count_ = static_cast<cardinal::u32>(lights_.size() / 16);
    }
    // ---- On-screen present ------------------------------------------------
    // Inside the async recording (post-record hook): fence + transition the
    // composite output, copy it into this frame's queue-shared ring slot.
    static void post_record_thunk(rhi::Swapchain* rec, graph::RhiBackend& be, void* user) noexcept {
        static_cast<AegisGraphPipeline*>(user)->post_record(rec, be);
    }
    void post_record(rhi::Swapchain* rec, graph::RhiBackend& be) noexcept {
        if (!hook_dst_ || !rec) return;
        rhi::Buffer* src = be.gpu_buffer("present.rgba");
        if (!src || src->size() < hook_dst_->size()) return;
        be.make_copy_source("present.rgba");
        rec->copy_buffer(src, 0, hook_dst_, 0, hook_dst_->size());
        hook_copied_ = true;
    }
    bool ensure_blit_pso() {
        if (blit_pso_) return true;
        if (blit_tried_ || !dev_ || !sw_) return false;
        blit_tried_ = true;
        auto vs = dev_->compile_shader(rhi::ShaderStage::Vertex,   kAegisBlitShader, "VSMain");
        auto fs = dev_->compile_shader(rhi::ShaderStage::Fragment, kAegisBlitShader, "PSMain");
        if (!vs.ok() || !fs.ok()) {
            cardinal::log::errorf("render/aegis", "present blit shader compile failed");
            return false;
        }
        rhi::PipelineDesc pd{};
        pd.vertex_shader   = cardinal::move(vs);
        pd.fragment_shader = cardinal::move(fs);
        pd.vertex_stride   = 0;                       // bufferless fullscreen triangle
        pd.color_format    = sw_->color_format();
        pd.depth_format    = sw_->depth_format();     // attachment-compatible, test off
        pd.depth_test      = false;
        pd.depth_write     = false;
        pd.topology        = rhi::PrimitiveTopology::TriangleList;
        pd.push_constant_size    = 16;
        pd.storage_buffer_slots  = 1;
        pd.sampled_texture_slots = 0;
        blit_pso_ = dev_->create_pipeline(pd);
        if (!blit_pso_) cardinal::log::errorf("render/aegis", "present blit PSO creation failed");
        return blit_pso_ != nullptr;
    }
    // Ring slot for this render(): (re)create when the composite size changed.
    // Old buffers are retired, not destroyed — a frame in flight may still be
    // reading them — and dropped a few frames later.
    rhi::Buffer* acquire_present_slot(cardinal::usize bytes) {
        PresentSlot& slot = ring_[ring_next_ % kRingSlots];
        if (!slot.buf || slot.size != bytes) {
            if (slot.buf) retired_.push_back(Retired{ cardinal::move(slot.buf), frame_counter_ });
            rhi::BufferDesc bd{};
            bd.size         = bytes;
            bd.usage        = static_cast<u32>(rhi::BufferUsage::Storage);
            bd.cpu_writable = false;
            bd.queue_shared = true;
            slot.buf  = dev_->create_buffer(bd);
            slot.size = slot.buf ? bytes : 0;
        }
        return slot.buf.get();
    }
    void retire_old_buffers() {
        cardinal::usize keep = 0;
        for (cardinal::usize i = 0; i < retired_.size(); ++i) {
            if (frame_counter_ - retired_[i].frame < kRetireFrames) {
                if (keep != i) retired_[keep] = cardinal::move(retired_[i]);
                ++keep;
            }
        }
        retired_.resize(keep);
    }
    void draw_present(rhi::Buffer* buf) {
        const cardinal::u32 vp = sw_->active_viewport();
        cardinal::u32 vw = sw_->viewport_width(vp), vh = sw_->viewport_height(vp);
        if (vw == 0 || vh == 0) { vw = sw_->width(); vh = sw_->height(); }
        const cardinal::u32 push[4] = { vw, vh, runner_->config().width, runner_->config().height };
        sw_->bind_pipeline(blit_pso_.get());
        sw_->set_push_constants(0, push, sizeof push);
        sw_->bind_storage_buffer(0, buf);
        sw_->draw(3);
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
        unified_memory_ = caps.unified_memory;

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
        ++frame_counter_;
        retire_old_buffers();
        // Pick up any editor-driven backend change before this frame's
        // build + execute.
        update_backend_from_knob();
        // Real scene feed BEFORE apply_config: the palette counts it derives
        // are part of the config (they size graph buffers). Fed whenever the
        // graph will actually consume it (GPU execute requested, or a CPU
        // simulation backend is selected); telemetry-only frames skip it.
        const bool feed = kb(kidx_.gpu_execute) ||
                          current_mode_ == AegisBackendMode::Cpu ||
                          current_mode_ == AegisBackendMode::ThreadedCpu;
        zero_copy_ok_ = rhi_backend_ && kb(kidx_.gpu_execute) && rhi_backend_->async_lane();
        extract_scene(scn, feed);
        // Push knob/caps changes (precision tier + GPU features) into the
        // runner config — diff-gated, so it only reconfigures on a real change.
        apply_config();
        // GPU compute execute. The AEGIS graph is dispatched for real ONLY on
        // the async compute lane: render() runs inside the frame's render
        // pass, and a compute dispatch inside an active dynamic-rendering
        // instance is illegal on Vulkan — so without an async queue (device
        // support AND the Async Compute knob) the backend stays recording-only
        // telemetry and the ForwardRenderer draws the screen.
        bool gpu_on = false;
        if (rhi_backend_) {
            const bool want_async = runner_->config().features.async_compute;
            if (want_async && async_queue_ == nullptr && dev_ != nullptr)
                async_queue_ = dev_->create_async_compute_queue();
            rhi_backend_->set_async_queue(
                want_async ? async_queue_.get() : nullptr);
            const bool want_gpu = kb(kidx_.gpu_execute);
            gpu_on = want_gpu && rhi_backend_->async_lane();
            rhi_backend_->set_gpu_execute(gpu_on);
            if (want_gpu && !gpu_on && !gpu_gate_logged_) {
                gpu_gate_logged_ = true;
                cardinal::log::warnf("render/aegis",
                    "GPU Compute Execute needs the async compute lane (enable the "
                    "Async Compute knob on a device with a compute queue); staying "
                    "recording-only");
            }
        }

        // PERSISTENT GRAPH: the AEGIS topology is static for a given config,
        // so the graph is built ONCE and retained — reconfigure() (resize /
        // precision / feature change) invalidates it via is_built(). Only the
        // imported camera bytes below change per frame; every backend re-reads
        // imports at execute. NOTE: the AegisSceneInputs SHAPE (triangle_count,
        // which optional handles are valid) is baked at build — if this bridge
        // ever feeds real scene streams, it must force a rebuild when the
        // shape changes (set a dirty flag; see aegis_runner.hpp is_built()).
        // Rebuild when invalidated (reconfigure) OR when the triangle count
        // changed: the count is baked into buffer sizes + push constants.
        if (!runner_->is_built() || built_tri_count_ != tri_count_) {
            graph::Graph& gr = runner_->begin_build();
            gpu::AegisSceneInputs in;
            in.triangle_count = tri_count_;
            if (tri_count_ > 0) {
                in.tris         = gr.import_buffer("tris", tris_.data(), tris_.size() * 4, 4);
                in.material_ids = gr.import_buffer("mids", mids_.data(), mids_.size() * 4, 4);
            } else {
                in.tris         = gr.declare_buffer(graph::BufferDesc{"tris",  0, 0, true});
                in.material_ids = gr.declare_buffer(graph::BufferDesc{"mids",  0, 0, true});
            }
            in.materials    = gr.import_buffer("mats", mats_.data(),   mats_.size() * 4, 4);
            in.lights       = gr.import_buffer("lts",  lights_.data(), lights_.size() * 4, 4);
            built_tri_count_ = tri_count_;
            in.ambient      = gr.import_buffer("amb", amb_data_, 12, 4);
            in.view_proj    = gr.import_buffer("vp",  vp_data_,  64, 4);
            in.camera_dir   = gr.import_buffer("dir", dir_data_, 12, 4);
            // PBR pair: real inverse view-proj + camera position switch the
            // V-Buf resolve from the ambient proxy to tile-lit Cook-Torrance.
            in.inv_view_proj = gr.import_buffer("ivp",  ivp_data_,    64, 4);
            in.camera_pos    = gr.import_buffer("cpos", campos_data_, 12, 4);
            // Previous-frame VP (motion vectors + the Hi-Z occlusion test
            // against last frame's pyramid) and the pyramid's validity flag:
            // 0 on the first execute after any (re)build, 1 from then on.
            in.view_proj_prev    = gr.import_buffer("vpp",  vpp_data_, 64, 4);
            in.hiz_history_valid = gr.import_buffer("hizv", &hiz_valid_data_, 4, 4);
            hiz_valid_data_ = 0u;
            runner_->build(in);
        }

        // Fresh camera bytes every frame. AEGIS passes read the matrix
        // ROW-major (4 consecutive floats = one row, w-row at [12..15]);
        // cardinal Mat4 storage is COLUMN-major (m[col][row]) — transpose on
        // pack. A raw memcpy would compile and silently mis-cull (identity
        // matrices in tests are transpose-symmetric and hide it).
        {
            // Last frame's VP first (the Hi-Z pyramid was built with it);
            // the first frame has no "last", so it sees zero motion.
            if (vpp_primed_) for (int k = 0; k < 16; ++k) vpp_data_[k] = vp_data_[k];
            const float safe_aspect = (aspect > 0.001f) ? aspect : 1.0f;
            const auto vp = scn.camera().proj(safe_aspect) * scn.camera().view();
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    vp_data_[r * 4 + c] = vp.m[c][r];
            if (!vpp_primed_) { for (int k = 0; k < 16; ++k) vpp_data_[k] = vp_data_[k]; vpp_primed_ = true; }
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

        // On-screen present: arm the post-record hook so the async recording
        // copies the composite into this frame's queue-shared ring slot.
        const bool present_want = gpu_on && kb(kidx_.aegis_present) && tri_count_ > 0 && dev_ && sw_;
        hook_dst_ = nullptr;
        hook_copied_ = false;
        if (present_want) {
            const cardinal::usize bytes = static_cast<cardinal::usize>(runner_->config().width) *
                                          runner_->config().height * 4u;
            hook_dst_ = acquire_present_slot(bytes);
            rhi_backend_->set_post_record(hook_dst_ ? &post_record_thunk : nullptr, this);
        } else if (rhi_backend_) {
            rhi_backend_->set_post_record(nullptr, nullptr);
        }
        runner_->execute();
        hiz_valid_data_ = 1u;   // the persistent pyramid now holds this frame -> valid history next frame
        real_dispatches_ = 0;
        if (rhi_backend_)
            for (const auto& ev : rhi_backend_->events()) if (ev.real_dispatch) ++real_dispatches_;
        // Present the AEGIS frame when the copy was really recorded: make the
        // graphics queue wait (GPU-side) for this submission's fence value,
        // then blit the ring slot into the active viewport.
        if (present_want && hook_copied_ && real_dispatches_ > 0 && ensure_blit_pso()) {
            sw_->wait_fence(rhi_backend_->async_fence(), rhi_backend_->async_fence_value());
            draw_present(hook_dst_);
            ++ring_next_;
            if (!present_logged_) {
                present_logged_ = true;
                cardinal::log::infof("render/aegis",
                    "AEGIS on-screen: %u tris, %u materials, %u lights, %u real dispatches, %ux%u composite",
                    tri_count_, mat_count_, light_count_, real_dispatches_,
                    runner_->config().width, runner_->config().height);
            }
            if (zero_copy_frames_ == 1)
                cardinal::log::infof("render/aegis",
                    "zero-copy scene streaming active: %u tris written straight into the "
                    "double-buffered import mapping (unified memory: %s)",
                    tri_count_, unified_memory_ ? "yes, VRAM-resident" : "no, host memory");
            return;
        }
        // Delegate the actual draw to the ForwardRenderer, honouring the
        // per-viewport view mode the Studio pushed into our "view_mode" knob
        // (Solid / Wireframe / Polygons / Heightmap / Normals / RTX Preview).
        const int vm_e_raw = ke(kidx_.view_mode, 0);
        const int vm_e     = (vm_e_raw < 0) ? 0 : (vm_e_raw > 5 ? 5 : vm_e_raw);
        renderer_->render(scn, static_cast<scene::ViewMode>(vm_e), aspect);
    }
    void set_light_set(const scene::LightSet* lights) override {
        light_set_ = lights;
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
        s.clusters_drawn = real_dispatches_;          // passes that really dispatched on the GPU
        s.triangles_drawn = built_tri_count_;         // triangles fed to the AEGIS graph
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
        add_bool_("gpu_execute", "GPU Compute Execute", "AEGIS",
            "Dispatch the AEGIS compute graph on the GPU (real scene triangles, "
            "materials and lights) through RhiBackend on the async compute lane. "
            "Requires the Async Compute knob; without it the graph stays "
            "recording-only telemetry.", false);
        add_bool_("aegis_present", "Present AEGIS Output", "AEGIS",
            "Show the AEGIS composite (tile-binned V-Buffer -> tile-lit PBR resolve "
            "-> tonemap) in the viewport instead of the ForwardRenderer draw. Needs "
            "GPU Compute Execute + Async Compute. Editor gizmos are not drawn in "
            "this mode yet.", false);

        // ---- Global Illumination (AEGIS Block 8) ----------------------
        add_bool_("ssgi_enabled", "Screen-Space GI + AO", "Global Illumination",
            "Screen-space final gather after the V-Buffer resolve: marches 8 "
            "directions through the depth buffer, darkens the flat ambient term "
            "by ambient occlusion and adds one bounce of the direct radiance "
            "already shaded on screen (colour bleed). One compute dispatch; needs "
            "the PBR resolve (inverse view-proj + camera position).", true);
        add_float_("ssgi_radius", "GI Gather Radius (m)", "Global Illumination",
            "World-space radius a pixel gathers occlusion and bounce from. Larger "
            "= softer, wider contact darkening and bleed, more off-screen misses.",
            1.0f, 0.05f, 10.0f, 0.05f);
        add_float_("ssgi_intensity", "GI Bounce Intensity", "Global Illumination",
            "Scale on the gathered one-bounce diffuse. 0 leaves only the ambient "
            "occlusion.", 1.0f, 0.0f, 4.0f, 0.05f);
        add_float_("ssgi_ao_strength", "AO Strength", "Global Illumination",
            "How far ambient occlusion can darken the flat ambient term: 0 = none, "
            "1 = fully occluded pixels lose all ambient.", 1.0f, 0.0f, 1.0f, 0.05f);
        add_int_("ssgi_steps", "GI Steps per Direction", "Global Illumination",
            "Taps per direction (8 directions). 4 = 32 taps per pixel; 8 = 64.",
            4, 1, 8);

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

public:
    ~AegisGraphPipeline() override {
        // Retire the async lane while the queue is alive, then drop the
        // owners of the backend before the queue goes (see async_queue_).
        if (rhi_backend_) rhi_backend_->set_async_queue(nullptr);
        runner_.reset();
        rhi_backend_.reset();
        async_queue_.reset();
    }

private:
    rhi::Device*                                      dev_  {nullptr};
    rhi::Swapchain*                                   sw_   {nullptr};
    cardinal::shared_ptr<scene::ForwardRenderer>      renderer_;
    cardinal::shared_ptr<AegisPipelineRunner>          runner_;
    cardinal::vector<Knob>                             knobs_;
    KnobIdx                                            kidx_;   // hot-knob indices
    AegisBackendMode                                   current_mode_ {AegisBackendMode::Null};
    AegisDeviceSupport                                 dev_caps_ {};   // from on_caps
    // The RhiBackend holds a NON-owning pointer to this queue and drains
    // through it in its destructor. runner_ (above) co-owns the backend and
    // is destroyed after this member, so member order alone cannot keep the
    // queue alive long enough: ~AegisGraphPipeline detaches explicitly.
    cardinal::unique_ptr<cardinal::rhi::ComputeQueue>  async_queue_;
    cardinal::shared_ptr<graph::RhiBackend>            rhi_backend_;   // when mode==Rhi
    // Persistent host storage for the graph's imported camera inputs — the
    // graph keeps raw pointers into these across frames (persistent graph).
    float amb_data_[3] {0.1f, 0.1f, 0.12f};
    float vp_data_[16] {};
    float vpp_data_[16] {};              // last frame's view-proj (motion vectors + Hi-Z history)
    bool  vpp_primed_ {false};
    cardinal::u32 hiz_valid_data_ {0};   // 1 once the persistent pyramid holds a frame built on this graph
    float dir_data_[3] {0.0f, 0.0f, 1.0f};
    float ivp_data_[16] {};
    float campos_data_[3] {};
    // ---- Real scene feed (see extract_scene) -------------------------------
    const scene::LightSet*            light_set_ {nullptr};
    cardinal::vector<float>           tris_;
    cardinal::vector<cardinal::u32>   mids_;
    cardinal::vector<float>           mats_;
    cardinal::vector<float>           lights_;
    cardinal::u32                     tri_count_ {0}, mat_count_ {0}, light_count_ {0};
    cardinal::u32                     built_tri_count_ {0};
    cardinal::vector<scene::Mat4>     skin_scratch_global_, skin_scratch_palette_;
    bool                              gpu_gate_logged_ {false};
    cardinal::u32                     real_dispatches_ {0};
    cardinal::u32                     frame_counter_ {0};
    bool                              zero_copy_ok_ {false};        // per frame: GPU lane live
    cardinal::u32                     zero_copy_frames_ {0};
    bool                              unified_memory_ {false};      // from on_caps
    // ---- On-screen present (see post_record / draw_present) ----------------
    static constexpr cardinal::u32 kRingSlots    = 4;   // > frames in flight
    static constexpr cardinal::u32 kRetireFrames = 16;
    struct PresentSlot { cardinal::unique_ptr<rhi::Buffer> buf; cardinal::usize size {0}; };
    struct Retired     { cardinal::unique_ptr<rhi::Buffer> buf; cardinal::u32 frame {0}; };
    PresentSlot                       ring_[kRingSlots];
    cardinal::u32                     ring_next_ {0};
    cardinal::vector<Retired>         retired_;
    cardinal::unique_ptr<rhi::Pipeline> blit_pso_;
    bool                              blit_tried_ {false};
    rhi::Buffer*                      hook_dst_ {nullptr};
    bool                              hook_copied_ {false};
    bool                              present_logged_ {false};
};

}  // namespace

// Factory exposed to RegistryImpl in pipelines.cpp.
cardinal::unique_ptr<Pipeline> create_aegis_pipeline(rhi::Device& dev,
                                                     rhi::Swapchain& sw) {
    return cardinal::make_unique<AegisGraphPipeline>(dev, sw);
}

}  // namespace cardinal::render
