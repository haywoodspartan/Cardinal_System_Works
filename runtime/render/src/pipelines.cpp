// =============================================================================
// Cardinal — concrete render pipelines.
//
// Both shipping pipelines today wrap the same underlying scene::ForwardRenderer
// (CPU-transformed forward path). They differ in which knobs they expose and
// what range of behaviour the user can drive from the panel:
//
//   ForwardBaselinePipeline  — full editor-grade controls (view mode + a few
//                              global knobs). The "default" pipeline.
//   DebugVisualizerPipeline  — locked to debugging channels with a single
//                              "Channel" enum knob, plus a couple of overlay
//                              toggles. Designed for content authoring.
//
// When real Forward+, Deferred or Path-traced pipelines land they implement
// the same Pipeline interface, declare their own Knob list, and slot into
// the Registry — no other engine code changes.
// =============================================================================
#include <cardinal/render/pipeline.hpp>

#include <cardinal/core/diag/log.hpp>
#include <cardinal/render/algos.hpp>
#include <cardinal/render/geo.hpp>
#include <cardinal/render/tess.hpp>
#include <cardinal/render/tex.hpp>
#include <cardinal/scene/renderer.hpp>

#include <cardinal/core/std/algorithm.hpp>
#include <cardinal/core/std/atomic.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/core/std/utility.hpp>

namespace cardinal::render {

// Forward declaration of the AEGIS bridge pipeline factory — defined in
// aegis_pipeline_bridge.cpp at cardinal::render scope (external linkage).
// Placed outside the anonymous namespace below so the linker picks up
// the right symbol from the bridge TU.
cardinal::unique_ptr<Pipeline> create_aegis_pipeline(rhi::Device& dev,
                                                     rhi::Swapchain& sw);

namespace {

// Find a knob by id. Returns nullptr if missing.
Knob* knob_by(cardinal::vector<Knob>& v, const char* id) {
    for (auto& k : v) if (k.id == id) return &k;
    return nullptr;
}

// Helper to read enum-knob value as a typed enum.
template<typename E>
E enum_value(const Knob& k, E fallback) {
    return (k.kind == KnobKind::Enum && k.e >= 0)
        ? static_cast<E>(k.e) : fallback;
}

// Build a Knob backed by the AlgoRegistry's catalogue for the given
// category. The Knob's enum_labels list is whatever's currently
// registered — including any user / plugin algos added at runtime.
Knob make_algo_knob(const char* id, const char* label, const char* group,
                    const char* tooltip, algo::CategoryId cat) {
    Knob k;
    k.id = id; k.label = label; k.group = group; k.tooltip = tooltip;
    k.kind = KnobKind::Enum;
    k.enum_labels = algo::labels_for(cat);
    k.e = algo::default_index_for(cat);
    return k;
}

// ---------------------------------------------------------------------------
// ForwardBaselinePipeline
// ---------------------------------------------------------------------------
class ForwardBaseline final : public Pipeline {
public:
    ForwardBaseline(rhi::Device& dev, rhi::Swapchain& sw)
        : renderer_(scene::ForwardRenderer::create(dev, sw))
    {
        // Build the knob list. We populate availability later in on_caps().
        knobs_ = make_knobs();
    }

    PipelineId  id()          const noexcept override { return PipelineId::ForwardBaseline; }
    const char* name()        const noexcept override { return "Forward Baseline"; }
    const char* description() const noexcept override {
        return "Single-pass forward shading with CPU-transformed vertices. "
               "View-mode picker, tonemap, render-scale, optional fake-DLSS upscale.";
    }
    cardinal::vector<Knob>& knobs() noexcept override { return knobs_; }

    void on_caps(const rhi::GpuCapabilities& caps) override {
        FeatureGate gate(caps);
        gate_knob("dlss_upscale", gate.query(Feature::DLSS_Upscale));
        gate_knob("dlss_framegen",gate.query(Feature::DLSS_FrameGen));
        gate_knob("rt_shadows",   gate.query(Feature::HardwareRT));
        gate_knob("mesh_shaders", gate.query(Feature::MeshShaders));
        gate_knob("vrs",          gate.query(Feature::VRS));
        gate_knob("reflex",       gate.query(Feature::ReflexLowLatency));
        gate_knob("fp16_math",    gate.query(Feature::FP16Math));
    }

    void render(scene::Scene& scn, float aspect) override {
        if (renderer_ == nullptr) return;
        const auto* km = knob_by(knobs_, "view_mode");
        const auto vm = km ? enum_value(*km, scene::ViewMode::Solid)
                           : scene::ViewMode::Solid;
        renderer_->render(scn, vm, aspect);
    }
    void set_light_set(const scene::LightSet* lights) override {
        if (renderer_) renderer_->set_light_set(lights);
    }
    void set_show_aabbs  (bool on) override { if (renderer_) renderer_->set_show_aabbs(on);   }
    void set_show_frustum(bool on) override { if (renderer_) renderer_->set_show_frustum(on); }
    void set_show_axes   (bool on) override { if (renderer_) renderer_->set_show_axes(on);    }
    void set_show_lights (bool on) override { if (renderer_) renderer_->set_show_lights(on);  }
    // Surface the underlying scene renderer's frustum-cull stats.
    // Cluster fields stay zero — this pipeline doesn't slice meshes
    // into clusters; the entity is its rendering primitive.
    FrameStats stats() const noexcept override {
        FrameStats s{};
        if (renderer_ != nullptr) {
            const auto srs = renderer_->frame_stats();
            s.entities_total  = srs.entities_total;
            s.entities_drawn  = srs.entities_drawn;
            s.entities_culled = srs.entities_culled;
            s.draw_calls      = srs.draw_calls;
            s.pc_pushes       = srs.pc_pushes;
            s.vgeom_master_tris     = srs.vgeom_master_tris;
            s.vgeom_drawn_tris      = srs.vgeom_drawn_tris;
            s.vgeom_cut_clusters    = srs.vgeom_cut_clusters;
            s.vgeom_culled_clusters = srs.vgeom_culled_clusters;
            s.cull_us         = srs.cull_us;
            s.phase1_us       = srs.phase1_us;
            s.phase2_us       = srs.phase2_us;
            s.sort_us         = srs.sort_us;
            s.submit_us       = srs.submit_us;
        }
        return s;
    }

private:
    static cardinal::vector<Knob> make_knobs() {
        cardinal::vector<Knob> k;

        // ---- Visualisation -----------------------------------------------
        Knob vm; vm.id="view_mode"; vm.label="View Mode"; vm.group="Visualisation";
        vm.tooltip = "How the forward renderer shades each fragment.";
        vm.kind = KnobKind::Enum;
        vm.enum_labels = {"Solid","Wireframe","Polygons","Heightmap","Normals","RTX Preview"};
        vm.e = 0;
        k.push_back(cardinal::move(vm));

        // ---- Tonemapping --------------------------------------------------
        // Backed by the AlgoRegistry catalogue (7 operators today; user-
        // added ones appear here automatically).
        k.push_back(make_algo_knob("tonemap_algo", "Tonemap", "Post",
            "Tone-mapping operator. Each registered algo has its own roll-off + colour response.",
            algo::CategoryId::Tonemap));

        // Hash / RNG used by stochastic post-process passes (TAA jitter,
        // dithering, etc.). Selectable so the user can swap in their own.
        k.push_back(make_algo_knob("hash_algo", "Hash / RNG", "Shader Optimisation",
            "Pseudo-random function for shader noise + sample-pattern derivation.",
            algo::CategoryId::HashRng));

        // Mip kernel — feeds the (future) compute mip generator.
        k.push_back(make_algo_knob("mip_filter_algo", "Mip filter", "Textures",
            "Down-sample kernel used when generating mip levels at load time.",
            algo::CategoryId::MipFilter));

        // Normal encoding — matters for any GBuffer / packed normal storage.
        k.push_back(make_algo_knob("normal_encode_algo", "Normal encoding", "Shader Optimisation",
            "Compression scheme for storing surface normals in the GBuffer / vertex stream.",
            algo::CategoryId::NormalEncode));

        // Sampling pattern — drives TAA jitter + GI ambient.
        k.push_back(make_algo_knob("sampling_algo", "Sampling pattern", "Performance",
            "Low-discrepancy sequence for AA jitter + ambient sampling.",
            algo::CategoryId::Sampling));

        // Compute precision — selects FP32 / FP16 / FP8 / FP4 for the
        // pipeline's bandwidth-bound intermediates (probes, neural buffers,
        // attention compute). Forward-looking; toggling this is currently
        // informational, but the algo registry's HLSL function name flows
        // into the shader specialiser when shaders that use packed math land.
        k.push_back(make_algo_knob("precision_algo", "Compute precision",
            "Shader Optimisation",
            "Numeric format used for compute kernels + bandwidth-bound "
            "intermediates. FP4 / FP8 cut memory traffic dramatically; "
            "the editor preview rounds your input through the format so "
            "you can see the loss before flipping the knob.",
            algo::CategoryId::PrecisionMath));

        Knob ex; ex.id="exposure"; ex.label="Exposure (EV)"; ex.group="Post";
        ex.tooltip = "Stops added before tonemapping. 0 = unmodified.";
        ex.kind = KnobKind::Float;
        ex.f_min = -8.0f; ex.f_max = 8.0f; ex.f_step = 0.1f; ex.f = 0.0f;
        k.push_back(cardinal::move(ex));

        // ---- Performance --------------------------------------------------
        Knob rs; rs.id="render_scale"; rs.label="Render Scale"; rs.group="Performance";
        rs.tooltip = "Internal render resolution as a fraction of the viewport. "
                     "0.5 = quarter-area framebuffer (much faster, blurrier).";
        rs.kind = KnobKind::Float;
        rs.f_min = 0.25f; rs.f_max = 2.0f; rs.f_step = 0.05f; rs.f = 1.0f;
        k.push_back(cardinal::move(rs));

        Knob ms; ms.id="mesh_shaders"; ms.label="Use Mesh Shaders";
        ms.group="Performance";
        ms.tooltip = "Replace the legacy vertex pipeline with mesh + task shaders.";
        ms.kind = KnobKind::Bool; ms.b = false;
        k.push_back(cardinal::move(ms));

        Knob vrs; vrs.id="vrs"; vrs.label="Variable Rate Shading";
        vrs.group="Performance";
        vrs.tooltip = "Reduce shading rate in low-detail screen regions.";
        vrs.kind = KnobKind::Bool; vrs.b = false;
        k.push_back(cardinal::move(vrs));

        Knob fp; fp.id="fp16_math"; fp.label="Prefer FP16 Math"; fp.group="Performance";
        fp.tooltip = "Pick min16 paths in shaders (Rapid Packed Math).";
        fp.kind = KnobKind::Bool; fp.b = true;
        k.push_back(cardinal::move(fp));

        // ---- Upscaling / frame-gen ----------------------------------------
        Knob du; du.id="dlss_upscale"; du.label="DLSS Super Resolution";
        du.group="Upscaling";
        du.tooltip = "NVIDIA DLSS — render below native, upscale via NGX.";
        du.kind = KnobKind::Enum;
        du.enum_labels = {"Off","Quality","Balanced","Performance","Ultra-Perf","DLAA"};
        du.e = 0;
        k.push_back(cardinal::move(du));

        Knob dg; dg.id="dlss_framegen"; dg.label="DLSS Frame Generation";
        dg.group="Upscaling";
        dg.tooltip = "Synthesise intermediate frames using the Optical Flow Accelerator. "
                     "Requires RTX 40-series.";
        dg.kind = KnobKind::Bool; dg.b = false;
        k.push_back(cardinal::move(dg));

        // ---- Lighting / shadows -------------------------------------------
        Knob sh; sh.id="shadow_method"; sh.label="Shadow Method"; sh.group="Lighting";
        sh.tooltip = "Cascaded shadow maps for directional lights.";
        sh.kind = KnobKind::Enum;
        sh.enum_labels = {"Off","PCF","PCSS","RTX (hardware)"};
        sh.e = 1;
        k.push_back(cardinal::move(sh));

        Knob rs2; rs2.id="rt_shadows"; rs2.label="RT Hardware Shadows";
        rs2.group="Lighting";
        rs2.tooltip = "Use hardware ray tracing for sharp + contact-shadow accuracy.";
        rs2.kind = KnobKind::Bool; rs2.b = false;
        k.push_back(cardinal::move(rs2));

        Knob ssa; ssa.id="ssao_strength"; ssa.label="SSAO Strength"; ssa.group="Lighting";
        ssa.kind = KnobKind::Float;
        ssa.f_min = 0.0f; ssa.f_max = 2.0f; ssa.f_step = 0.05f; ssa.f = 0.6f;
        k.push_back(cardinal::move(ssa));

        // ---- Latency ------------------------------------------------------
        Knob rf; rf.id="reflex"; rf.label="NVIDIA Reflex";
        rf.group="Latency";
        rf.tooltip = "Drive the GPU just-in-time so PC latency stays low.";
        rf.kind = KnobKind::Bool; rf.b = true;
        k.push_back(cardinal::move(rf));

        return k;
    }

    void gate_knob(const char* id, FeatureStatus s) {
        if (auto* k = knob_by(knobs_, id)) {
            k->available           = s.available;
            k->unavailable_reason  = s.reason;
            if (!s.available && k->kind == KnobKind::Enum) k->e = 0;
            if (!s.available && k->kind == KnobKind::Bool) k->b = false;
        }
    }

    cardinal::unique_ptr<scene::ForwardRenderer> renderer_;
    cardinal::vector<Knob>                       knobs_;
};

// ---------------------------------------------------------------------------
// DebugVisualizerPipeline
// ---------------------------------------------------------------------------
class DebugVisualizer final : public Pipeline {
public:
    DebugVisualizer(rhi::Device& dev, rhi::Swapchain& sw)
        : renderer_(scene::ForwardRenderer::create(dev, sw))
    {
        knobs_ = make_knobs();
    }

    PipelineId  id()          const noexcept override { return PipelineId::DebugVisualizer; }
    const char* name()        const noexcept override { return "Debug Visualizer"; }
    const char* description() const noexcept override {
        return "Debug-channel renderer for content authoring. "
               "Picks a single visualisation channel; no lighting, no post.";
    }
    cardinal::vector<Knob>& knobs() noexcept override { return knobs_; }

    void on_caps(const rhi::GpuCapabilities&) override { /* nothing gated */ }

    void render(scene::Scene& scn, float aspect) override {
        if (renderer_ == nullptr) return;
        const auto* km = knob_by(knobs_, "channel");
        // Map the simplified channel enum back to ViewMode.
        scene::ViewMode vm = scene::ViewMode::Wireframe;
        if (km && km->kind == KnobKind::Enum) {
            switch (km->e) {
                case 0: vm = scene::ViewMode::Wireframe; break;
                case 1: vm = scene::ViewMode::Polygons;  break;
                case 2: vm = scene::ViewMode::Heightmap; break;
                case 3: vm = scene::ViewMode::Normals;   break;
                case 4: vm = scene::ViewMode::RTXPreview;break;
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
        if (renderer_ != nullptr) {
            const auto srs = renderer_->frame_stats();
            s.entities_total  = srs.entities_total;
            s.entities_drawn  = srs.entities_drawn;
            s.entities_culled = srs.entities_culled;
            s.draw_calls      = srs.draw_calls;
            s.pc_pushes       = srs.pc_pushes;
            s.vgeom_master_tris     = srs.vgeom_master_tris;
            s.vgeom_drawn_tris      = srs.vgeom_drawn_tris;
            s.vgeom_cut_clusters    = srs.vgeom_cut_clusters;
            s.vgeom_culled_clusters = srs.vgeom_culled_clusters;
            s.cull_us         = srs.cull_us;
            s.phase1_us       = srs.phase1_us;
            s.phase2_us       = srs.phase2_us;
            s.sort_us         = srs.sort_us;
            s.submit_us       = srs.submit_us;
        }
        return s;
    }

private:
    static cardinal::vector<Knob> make_knobs() {
        cardinal::vector<Knob> k;
        Knob ch; ch.id="channel"; ch.label="Channel"; ch.group="Visualisation";
        ch.tooltip = "Which debug visualisation to render.";
        ch.kind = KnobKind::Enum;
        ch.enum_labels = {"Wireframe","Polygons (random)","Heightmap","Normals","RTX Preview"};
        ch.e = 0;
        k.push_back(cardinal::move(ch));

        Knob bg; bg.id="show_grid"; bg.label="Show ground grid"; bg.group="Overlay";
        bg.kind = KnobKind::Bool; bg.b = true;
        k.push_back(cardinal::move(bg));

        Knob ax; ax.id="show_axes"; ax.label="Show world axes"; ax.group="Overlay";
        ax.kind = KnobKind::Bool; ax.b = true;
        k.push_back(cardinal::move(ax));
        return k;
    }

    cardinal::unique_ptr<scene::ForwardRenderer> renderer_;
    cardinal::vector<Knob>                       knobs_;
};

// ---------------------------------------------------------------------------
// ForwardClusteredPipeline — meshlet/cluster baseline.
//
// On first render of each scene, slice every entity's mesh into meshlets
// (cardinal::render::geo::build_meshlets). Each frame:
//   1) Build a frustum from the camera VP
//   2) Walk every entity's meshlet list, run frustum + backface-cone tests
//   3) Compute LOD per surviving cluster
//   4) Report stats (total / culled / drawn / triangles)
//   5) Hand the underlying scene::ForwardRenderer the same scene to draw
//
// Because the RHI doesn't yet expose draw-meshlet primitives, the actual
// rasterisation is the same forward pass as the baseline pipeline; the
// novel work is the cluster bookkeeping that's ready for either:
//   (a) hardware mesh shaders when the RHI adds them
//   (b) GPU-driven indirect draw with a compute culling pass
// Either path consumes the same ClusterBounds list this code populates.
// ---------------------------------------------------------------------------
class ForwardClustered final : public Pipeline {
public:
    ForwardClustered(rhi::Device& dev, rhi::Swapchain& sw)
        : renderer_(scene::ForwardRenderer::create(dev, sw))
    {
        knobs_ = make_knobs();
    }

    PipelineId  id()          const noexcept override { return PipelineId::ForwardClustered; }
    const char* name()        const noexcept override { return "Forward Clustered (meshlets)"; }
    const char* description() const noexcept override {
        return "Meshlet baseline (Phase 6). Slices meshes into 64-tri clusters with "
               "bounding sphere + backface cone, then runs CPU frustum + cone "
               "culling each frame. Hardware mesh-shader dispatch lands when the "
               "RHI adds the primitive — until then the surviving clusters drive "
               "the same forward pass as the baseline.";
    }
    cardinal::vector<Knob>& knobs() noexcept override { return knobs_; }

    void on_caps(const rhi::GpuCapabilities& caps) override {
        FeatureGate gate(caps);
        gate_knob("hw_mesh_shaders",  gate.query(Feature::MeshShaders));
        gate_knob("aniso_quality",    gate.query(Feature::FP16Math));   // proxy
        gate_knob("wave_intrinsics",  gate.query(Feature::AsyncCompute));// proxy
        gate_knob("rt_shadows",       gate.query(Feature::HardwareRT));
    }

    FrameStats stats() const noexcept override { return last_stats_; }

    void set_light_set(const scene::LightSet* lights) override {
        if (renderer_) renderer_->set_light_set(lights);
    }
    void set_show_aabbs  (bool on) override { if (renderer_) renderer_->set_show_aabbs(on);   }
    void set_show_frustum(bool on) override { if (renderer_) renderer_->set_show_frustum(on); }
    void set_show_axes   (bool on) override { if (renderer_) renderer_->set_show_axes(on);    }
    void set_show_lights (bool on) override { if (renderer_) renderer_->set_show_lights(on);  }

    void render(scene::Scene& scn, float aspect) override {
        if (renderer_ == nullptr) return;
        ensure_meshlets(scn);

        // Build frustum from the active camera.
        const auto view = scn.camera().view();
        const auto proj = scn.camera().proj(aspect > 0.001f ? aspect : 1.0f);
        const auto vp   = proj * view;
        const auto frustum = geo::frustum_from_vp(vp);

        const bool   enable_culling = read_bool("enable_cluster_culling", true);
        const bool   enable_lod     = read_bool("enable_lod", true);
        const u32    quality_idx    = static_cast<u32>(read_enum("tess_quality", 1));
        const float  tess_scale     = tess::quality_scale(static_cast<tess::Quality>(quality_idx));

        FrameStats s{};
        for (const auto& e : scn.entities()) {
            if (!e.visible || e.mesh == nullptr) continue;
            auto it = meshlets_.find(e.mesh.get());
            if (it == meshlets_.end()) continue;

            for (const auto& m : it->second.meshlets) {
                ++s.clusters_total;
                // Translate the meshlet's local-space sphere into world space
                // by adding the entity translation (uniform-scale only for now).
                geo::ClusterBounds wb = m.bounds;
                wb.sphere_center += e.transform.translation;
                wb.cone_apex     += e.transform.translation;

                if (enable_culling && !geo::cluster_visible(wb, frustum, scn.camera().position)) {
                    ++s.clusters_culled;
                    continue;
                }
                u32 lod = 0;
                if (enable_lod) {
                    lod = geo::cluster_lod(wb, scn.camera().position);
                    if (lod >= 3) { ++s.clusters_culled; continue; }
                }
                ++s.clusters_drawn;
                s.triangles_drawn       += m.index_count / 3;
                s.tessellation_factor_avg += static_cast<u32>(tess_scale);
            }
        }
        if (s.clusters_drawn > 0) {
            s.tessellation_factor_avg /= s.clusters_drawn;
        }
        last_stats_ = s;

        // Issue the actual draw via the existing forward renderer; cluster
        // visibility is bookkeeping today (becomes per-cluster indirect when
        // the GPU side lands).
        const auto* km = knob_by(knobs_, "view_mode");
        const auto vm = km ? enum_value(*km, scene::ViewMode::Solid)
                           : scene::ViewMode::Solid;
        renderer_->render(scn, vm, aspect);

        // Fold the underlying scene renderer's entity-cull stats into
        // last_stats_ so the editor can see the bottom-of-funnel count.
        // Cluster numbers above are independent — we report both layers.
        const auto srs = renderer_->frame_stats();
        last_stats_.entities_total  = srs.entities_total;
        last_stats_.entities_drawn  = srs.entities_drawn;
        last_stats_.entities_culled = srs.entities_culled;
        last_stats_.draw_calls            = srs.draw_calls;
        last_stats_.pc_pushes             = srs.pc_pushes;
        last_stats_.vgeom_master_tris     = srs.vgeom_master_tris;
        last_stats_.vgeom_drawn_tris      = srs.vgeom_drawn_tris;
        last_stats_.vgeom_cut_clusters    = srs.vgeom_cut_clusters;
        last_stats_.vgeom_culled_clusters = srs.vgeom_culled_clusters;
        last_stats_.cull_us         = srs.cull_us;
        last_stats_.phase1_us       = srs.phase1_us;
        last_stats_.phase2_us       = srs.phase2_us;
        last_stats_.sort_us         = srs.sort_us;
        last_stats_.submit_us       = srs.submit_us;
    }

private:
    static cardinal::vector<Knob> make_knobs() {
        cardinal::vector<Knob> k;

        Knob vm; vm.id="view_mode"; vm.label="View Mode"; vm.group="Visualisation";
        vm.kind = KnobKind::Enum;
        vm.enum_labels = {"Solid","Wireframe","Polygons","Heightmap","Normals","RTX Preview"};
        vm.e = 0;
        k.push_back(cardinal::move(vm));

        // ---- Cluster culling -----------------------------------------
        Knob cc; cc.id="enable_cluster_culling"; cc.label="Cluster culling";
        cc.group="Clusters";
        cc.tooltip = "Run frustum + backface-cone tests per meshlet on the CPU. "
                     "When disabled, every cluster passes through (useful to "
                     "compare Drawn vs Culled in the stats panel).";
        cc.kind = KnobKind::Bool; cc.b = true;
        k.push_back(cardinal::move(cc));

        Knob lod; lod.id="enable_lod"; lod.label="Distance LOD";
        lod.group="Clusters";
        lod.tooltip = "Drop high-detail clusters when the camera is far. Coarse "
                      "LOD selection is a stand-in for the Nanite-style hierarchical "
                      "DAG that lands later.";
        lod.kind = KnobKind::Bool; lod.b = true;
        k.push_back(cardinal::move(lod));

        Knob hms; hms.id="hw_mesh_shaders"; hms.label="Hardware Mesh Shader Dispatch";
        hms.group="Clusters";
        hms.tooltip = "Drive the surviving clusters through HW mesh shaders. "
                      "Currently a forward-looking knob — toggling it has no "
                      "effect until the RHI adds the mesh-shader primitive.";
        hms.kind = KnobKind::Bool; hms.b = false;
        k.push_back(cardinal::move(hms));

        // Cluster cull algorithm — registry-backed so user / plugin culling
        // strategies (e.g. occlusion + Hi-Z) appear here without recompile.
        k.push_back(make_algo_knob("cluster_cull_algo", "Cluster culling algorithm",
            "Clusters",
            "How visible clusters are picked from the meshlet list each frame.",
            algo::CategoryId::ClusterCull));

        // ---- Tessellation --------------------------------------------
        Knob tq; tq.id="tess_quality"; tq.label="Tessellation Quality";
        tq.group="Tessellation";
        tq.tooltip = "Sets the world-space scale that drives the hull-shader "
                     "factor. Higher = denser tessellation closer to the camera.";
        tq.kind = KnobKind::Enum;
        tq.enum_labels = {"Off","Low (8x)","Medium (16x)","High (32x)"};
        tq.e = 1;
        k.push_back(cardinal::move(tq));

        // Tessellation policy — algo-backed.
        k.push_back(make_algo_knob("tess_policy_algo", "Tessellation policy",
            "Tessellation",
            "Hull-shader policy that picks the per-edge tess factor.",
            algo::CategoryId::TessFactor));

        // ---- Texture math --------------------------------------------
        Knob aq; aq.id="aniso_quality"; aq.label="Anisotropic Filter";
        aq.group="Textures";
        aq.tooltip = "Maximum anisotropy level for sampler descriptors. "
                     "16x is free on RTX-class GPUs; lower presets save bandwidth.";
        aq.kind = KnobKind::Enum;
        aq.enum_labels = {"Off (1x)","Low (2x)","Medium (4x)","High (8x)","Ultra (16x)"};
        aq.e = 4;
        k.push_back(cardinal::move(aq));

        Knob bc; bc.id="texture_compression"; bc.label="Block Compression";
        bc.group="Textures";
        bc.tooltip = "Preferred block-compressed format for VRAM-resident textures. "
                     "BC7 is the modern default for colour; BC5 for normal maps.";
        bc.kind = KnobKind::Enum;
        bc.enum_labels = {"None","BC1","BC3","BC5","BC6H (HDR)","BC7","ASTC 4x4","ASTC 6x6","ASTC 8x8"};
        bc.e = 5;
        k.push_back(cardinal::move(bc));

        Knob ts; ts.id="streaming_budget_mb"; ts.label="Streaming Budget (MB)";
        ts.group="Textures";
        ts.tooltip = "VRAM ceiling for the texture streaming planner. The planner "
                     "drops top mips across the residency set until it fits.";
        ts.kind = KnobKind::Int;
        ts.i_min = 64; ts.i_max = 16384; ts.i = 2048;
        k.push_back(cardinal::move(ts));

        // ---- Shader optimisations ------------------------------------
        Knob fp; fp.id="prefer_fp16"; fp.label="Prefer FP16 (Rapid Packed Math)";
        fp.group="Shader Optimisation";
        fp.kind = KnobKind::Bool; fp.b = true;
        k.push_back(cardinal::move(fp));

        Knob wo; wo.id="wave_intrinsics"; wo.label="Wave intrinsics";
        wo.group="Shader Optimisation";
        wo.tooltip = "Use SM 6.0 wave reductions (WaveActiveSum/Max/Min) for "
                     "in-shader compaction + lighting accumulation.";
        wo.kind = KnobKind::Bool; wo.b = true;
        k.push_back(cardinal::move(wo));

        Knob bn; bn.id="bindless"; bn.label="Bindless descriptor arrays";
        bn.group="Shader Optimisation";
        bn.tooltip = "Treat all textures as a single huge descriptor array indexed "
                     "by material ID. Drops binding overhead to zero.";
        bn.kind = KnobKind::Bool; bn.b = true;
        k.push_back(cardinal::move(bn));

        // Compute precision — the meshlet pipeline benefits a lot from
        // FP4/FP8 because the per-cluster transformed-vertex storage is
        // inherently bandwidth-bound.
        k.push_back(make_algo_knob("precision_algo", "Compute precision",
            "Shader Optimisation",
            "Numeric format for per-cluster intermediates and the cluster "
            "bookkeeping buffer. FP4 cuts the buffer to 1/8th vs. FP32; "
            "the panel preview shows quantisation error live.",
            algo::CategoryId::PrecisionMath));

        // ---- RT --------------------------------------------------------
        Knob rs; rs.id="rt_shadows"; rs.label="RT Hardware Shadows";
        rs.group="Lighting"; rs.kind = KnobKind::Bool; rs.b = false;
        k.push_back(cardinal::move(rs));

        return k;
    }

    void gate_knob(const char* id, FeatureStatus s) {
        if (auto* k = knob_by(knobs_, id)) {
            k->available = s.available;
            k->unavailable_reason = s.reason;
        }
    }

    bool read_bool(const char* id, bool fb) const {
        for (const auto& k : knobs_)
            if (k.id == id && k.kind == KnobKind::Bool) return k.b;
        return fb;
    }
    int read_enum(const char* id, int fb) const {
        for (const auto& k : knobs_)
            if (k.id == id && k.kind == KnobKind::Enum) return k.e;
        return fb;
    }

    // Build meshlets for any mesh we haven't seen before. We keep a one-shot
    // copy of the index/position data extracted from the scene::Mesh; meshlet
    // bounds are local-space (we add entity translation per frame).
    void ensure_meshlets(scene::Scene& scn) {
        for (const auto& e : scn.entities()) {
            if (!e.visible || e.mesh == nullptr) continue;
            if (meshlets_.count(e.mesh.get())) continue;

            const auto* verts = e.mesh->cpu_vertices();
            const u32   n     = e.mesh->vertex_count();
            if (verts == nullptr || n < 3) continue;

            // Our scene::Mesh today is a non-indexed triangle soup; build a
            // throwaway identity index list to feed the meshlet generator.
            cardinal::vector<u32> indices(n);
            for (u32 i = 0; i < n; ++i) indices[i] = i;

            geo::Mesh m = geo::build_meshlets(
                indices.data(), n,
                &verts[0].position.x, n, sizeof(scene::Vertex),
                &verts[0].normal.x,   sizeof(scene::Vertex));
            cardinal::log::infof("render",
                "ForwardClustered: meshlet '%s' → %u clusters",
                e.mesh->name().c_str(), static_cast<u32>(m.meshlets.size()));
            meshlets_.emplace(e.mesh.get(), cardinal::move(m));
        }
    }

    cardinal::unique_ptr<scene::ForwardRenderer>          renderer_;
    cardinal::vector<Knob>                                knobs_;
    cardinal::unordered_map<const scene::Mesh*, geo::Mesh> meshlets_;
    FrameStats                                       last_stats_{};
};

// ---------------------------------------------------------------------------
// RegistryImpl
// ---------------------------------------------------------------------------
class RegistryImpl final : public Registry {
public:
    RegistryImpl(rhi::Device& dev, rhi::Swapchain& sw) {
        pipelines_.push_back(cardinal::make_unique<ForwardBaseline>  (dev, sw));
        pipelines_.push_back(cardinal::make_unique<DebugVisualizer>  (dev, sw));
        pipelines_.push_back(cardinal::make_unique<ForwardClustered> (dev, sw));
        pipelines_.push_back(create_aegis_pipeline                   (dev, sw));
        for (auto& p : pipelines_) p->on_caps(dev.capabilities());
        // Default to AEGIS Pipeline 2.0 on every start (its graph runs on the
        // RhiBackend by default; on-screen draw delegates to the ForwardRenderer
        // until RhiBackend rasterises). Hosts can still set_active() any other
        // pipeline at runtime, and the studio_settings pipeline.active CVar
        // overrides this when the user has saved a different choice.
        for (size_t i = 0; i < pipelines_.size(); ++i) {
            if (pipelines_[i]->id() == PipelineId::Aegis) { active_idx_ = i; break; }
        }
        cardinal::log::infof("render",
            "Pipeline registry online — %zu pipelines, default '%s'",
            pipelines_.size(), pipelines_[active_idx_]->name());
    }

    cardinal::vector<Pipeline*> all() override {
        cardinal::vector<Pipeline*> out; out.reserve(pipelines_.size());
        for (auto& p : pipelines_) out.push_back(p.get());
        return out;
    }
    Pipeline*  active()                override { return pipelines_[active_idx_].get(); }
    PipelineId active_id() const noexcept override { return pipelines_[active_idx_]->id(); }

    void set_active(PipelineId id) override {
        for (size_t i = 0; i < pipelines_.size(); ++i) {
            if (pipelines_[i]->id() == id) {
                if (i != active_idx_) {
                    cardinal::log::infof("render",
                        "Pipeline switched: %s → %s",
                        pipelines_[active_idx_]->name(), pipelines_[i]->name());
                }
                active_idx_ = i;
                return;
            }
        }
    }

    void render(scene::Scene& scene, float aspect) override {
        if (auto* p = active()) p->render(scene, aspect);
    }

    void set_light_set(const scene::LightSet* lights) override {
        for (auto& p : pipelines_) p->set_light_set(lights);
    }

private:
    cardinal::vector<cardinal::unique_ptr<Pipeline>> pipelines_;
    size_t                                 active_idx_{0};
};

}  // namespace

cardinal::unique_ptr<Registry> Registry::create(rhi::Device& dev, rhi::Swapchain& sw) {
    return cardinal::make_unique<RegistryImpl>(dev, sw);
}

}  // namespace cardinal::render
