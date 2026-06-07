// =============================================================================
// Cardinal_System_StudioMin
//
// Minimal editor host built on cardinal::ui::StudioEngine — the "roof" of
// the concrete / wood / roof layering. The legacy Cardinal_System_Studio
// sample drives Window + Device + Swapchain + Studio + JobSystem manually
// (~2700 lines). This sample asks StudioEngine::run() to do all of that
// for us and only provides the bits that are application-specific:
//
//   - A 1-frame spawn of a scene
//   - on_simulate (game logic — empty here)
//   - on_ui (custom panel — RHI controls demo)
//   - on_menu_bar (a couple of menu items)
//
// Also demonstrates two facilities that landed this turn:
//   - cardinal::crash::install()    — minidump on uncaught exception
//   - cardinal::trace::begin_capture / end_capture — chrome-tracing dump
//
// Build target: Cardinal_System_StudioMin.exe
// =============================================================================

#include <cardinal/ui/studio_engine.hpp>
#include <cardinal/ui/studio.hpp>
#include <cardinal/ui/simd_panel.hpp>
#include <cardinal/ui/vgeom_panel.hpp>

#include <cardinal/core/sync/async.hpp>
#include <cardinal/core/diag/crash.hpp>
#include <cardinal/core/sync/jobs.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/diag/trace_export.hpp>
#include <cardinal/engine/engine.hpp>
#include <cardinal/render/pipeline.hpp>
#include <cardinal/rhi/rhi.hpp>
#include <cardinal/scene/light.hpp>
#include <cardinal/scene/scene.hpp>
#include <cardinal/actor/component.hpp>
#include <cardinal/actor/world.hpp>
#include <cardinal/game/game.hpp>
#include <cardinal/game/game_actor.hpp>
#include <cardinal/game/reflection.hpp>
#include <cardinal/audio/audio.hpp>
#include <cardinal/net/net.hpp>
#include <cardinal/net/replication.hpp>
#include <cardinal/physics/physics.hpp>
#include <cardinal/serial/serial.hpp>
#include <cardinal/sim/sim.hpp>
#include <cardinal/sky/sky.hpp>

#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <string>

namespace {

// Convert a quaternion to intrinsic XYZ Euler angles (radians) matching
// the convention of cardinal::Mat4::rotation_xyz — i.e. the full
// rotation matrix R = Rz(rz) · Ry(ry) · Rx(rx) when applied to a column
// vector. Used to push physics::Body orientation back into a
// scene::Transform that uses rotation_euler.
//
// Derivation (column-vector convention):
//   R[2][0] = -sin(ry)                            → ry = asin(-R[2][0])
//   R[2][1] / R[2][2] = (cy·sx)/(cy·cx) = tan(rx) → rx = atan2(R[2][1], R[2][2])
//   R[1][0] / R[0][0] = (sz·cy)/(cz·cy) = tan(rz) → rz = atan2(R[1][0], R[0][0])
// Quaternion-to-matrix formulas substituted in directly so we avoid
// building a Mat3 intermediate. Clamped to [-1, 1] for numerical safety
// at the gimbal-lock poles.
// NOTE: distinct from cardinal::core::quat_to_euler_xyz — this variant
// is derived against the renderer's Mat4::rotation_xyz column-vector
// convention so a physics body's quat maps to the exact rotation_euler
// the scene rebuilds. Kept local + renamed to avoid an ADL-ambiguous
// call now that the canonical helper exists (unifying the two euler
// conventions is a separate, careful task).
inline cardinal::scene::Vec3 phys_quat_to_render_euler(const cardinal::physics::Quat& q) {
    const float xx = q.x*q.x, yy = q.y*q.y, zz = q.z*q.z;
    const float xy = q.x*q.y, xz = q.x*q.z, yz = q.y*q.z;
    const float wx = q.w*q.x, wy = q.w*q.y, wz = q.w*q.z;
    const float sy_arg = std::clamp(2.0f * (wy - xz), -1.0f, 1.0f);
    const float ry = std::asin(sy_arg);
    const float rx = std::atan2(2.0f * (yz + wx), 1.0f - 2.0f * (xx + yy));
    const float rz = std::atan2(2.0f * (xy + wz), 1.0f - 2.0f * (yy + zz));
    return cardinal::scene::Vec3{rx, ry, rz};
}

// Fully-saturated HSV→RGB (s=v=1). Used to give the demo "Extra
// lights" ring a rainbow of distinct hues so the arbitrary-count
// StructuredBuffer<Light> path is visually obvious.
inline cardinal::scene::Vec3 hue_rgb(float h01) {
    const float h = (h01 - std::floor(h01)) * 6.0f;
    const float x = 1.0f - std::fabs(std::fmod(h, 2.0f) - 1.0f);
    switch (static_cast<int>(h)) {
        case 0:  return {1.0f, x,    0.0f};
        case 1:  return {x,    1.0f, 0.0f};
        case 2:  return {0.0f, 1.0f, x   };
        case 3:  return {0.0f, x,    1.0f};
        case 4:  return {x,    0.0f, 1.0f};
        default: return {1.0f, 0.0f, x   };
    }
}

// 60-sample rolling window for the per-phase µs counters in the RHI
// panel. File-scope because MSVC forbids static data members in a
// class defined inside a function (C2246).
struct PhaseRing {
    static constexpr int kCapacity = 60;
    float samples[kCapacity]{};
    int   write_idx{0};
    int   filled{0};
    void push(float v) {
        samples[write_idx] = v;
        write_idx = (write_idx + 1) % kCapacity;
        if (filled < kCapacity) ++filled;
    }
    void avg_peak(float& out_avg, float& out_peak) const {
        if (filled == 0) { out_avg = 0; out_peak = 0; return; }
        float sum = 0, peak = 0;
        for (int i = 0; i < filled; ++i) {
            sum += samples[i];
            if (samples[i] > peak) peak = samples[i];
        }
        out_avg  = sum / static_cast<float>(filled);
        out_peak = peak;
    }
};

// Backend dropdown + validation toggle. Lives here in the sample because
// it leans on the engine's request_backend_swap() API which the legacy
// sample doesn't have access to (it doesn't use cardinal::engine::Engine).
void draw_rhi_panel(cardinal::engine::Engine& e,
                    cardinal::sky::Sky&  sky,
                    cardinal::game::Game& game,
                    const cardinal::serial::MeshRegistry& mesh_registry,
                    bool* p_open) {
    if (!ImGui::Begin("RHI", p_open, ImGuiWindowFlags_AlwaysAutoResize))
    { ImGui::End(); return; }

    const auto current = e.active_backend();
    const char* current_name =
        current == cardinal::engine::EngineDesc::Backend::Vulkan ? "Vulkan" :
        current == cardinal::engine::EngineDesc::Backend::D3D12  ? "D3D12"  : "Auto";

    ImGui::Text("Current backend: %s", current_name);
    ImGui::Text("Swap pending  : %s", e.backend_swap_pending() ? "yes" : "no");
    ImGui::Separator();

    if (ImGui::Button("Vulkan", ImVec2(96, 0))) {
        e.request_backend_swap(cardinal::engine::EngineDesc::Backend::Vulkan);
    }
    ImGui::SameLine();
    if (ImGui::Button("D3D12", ImVec2(96, 0))) {
        e.request_backend_swap(cardinal::engine::EngineDesc::Backend::D3D12);
    }
    ImGui::SameLine();
    if (ImGui::Button("Toggle", ImVec2(96, 0))) {
        // Auto resolves to "the opposite of current" inside request_backend_swap.
        e.request_backend_swap(cardinal::engine::EngineDesc::Backend::Auto);
    }

    ImGui::Separator();
    // Validation layer is a per-create flag (vkCreateInstance time on
    // Vulkan; ID3D12Debug at device-create time on D3D12), so toggling
    // it requires a device re-create — handled by the engine's swap
    // pipeline. The checkbox updates the pending flag immediately and
    // schedules the re-create on the next safe boundary.
    {
        bool validation = e.validation_enabled();
        if (ImGui::Checkbox("Validation layer", &validation)) {
            e.set_validation_enabled(validation);
            e.request_device_recreate();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(re-creates device on next safe frame)");
    }

    ImGui::Separator();
    // Trace capture button — saves a chrome://tracing JSON next to the exe.
    static bool capturing = false;
    if (!capturing) {
        if (ImGui::Button("Start 120-frame trace")) {
            cardinal::trace::begin_capture(120);
            capturing = true;
        }
    } else {
        if (ImGui::Button("Stop + save trace.json")) {
            cardinal::trace::end_capture("trace.json");
            capturing = false;
        }
    }
    if (!cardinal::crash::last_dump_path().empty()) {
        ImGui::Separator();
        ImGui::TextWrapped("Last crash dump: %s",
            cardinal::crash::last_dump_path().c_str());
    }

    // ----- Sky save/load (validates the panel-button → serial pattern) --
    // Sky is the simplest of the three subsystems serial::save_* knows how
    // to round-trip (no entity/mesh-graph re-create needed). MinApp owns
    // the Sky instance; the engine doesn't. Save writes "sky.snapshot"
    // next to the exe; Load reads the same file back in.
    ImGui::Separator();
    ImGui::TextDisabled("Sky snapshot (hour=%.1f, %zu keys):",
                        sky.hour(), sky.keys().size());
    if (ImGui::Button("Save Sky")) {
        std::string err;
        if (cardinal::serial::save_sky(sky, "sky.snapshot", &err)) {
            cardinal::log::infof("studio_min", "Saved sky.snapshot");
        } else {
            cardinal::log::errorf("studio_min", "save_sky failed: %s", err.c_str());
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Load Sky")) {
        std::string err;
        if (cardinal::serial::load_sky(sky, "sky.snapshot", &err)) {
            cardinal::log::infof("studio_min", "Loaded sky.snapshot");
        } else {
            cardinal::log::errorf("studio_min", "load_sky failed: %s", err.c_str());
        }
    }

    // ----- World save/load (mirrors the Sky pattern) --------------------
    // World is the actor::World that game::Game wraps. Save reads, load
    // mutates via the Game (it spawns actors back via the class registry).
    // Empty by default in the StudioMin demo — buttons round-trip cleanly
    // even with zero actors, validating the wiring.
    ImGui::Separator();
    ImGui::TextDisabled("World snapshot (%u game actors):",
                        game.game_actor_count());
    if (ImGui::Button("Save World")) {
        std::string err;
        const auto stats = cardinal::serial::save_world(
            game.world(), "world.snapshot", &err);
        if (err.empty()) {
            cardinal::log::infof("studio_min",
                "Saved world.snapshot (%u actors, %u props, %llu bytes)",
                stats.actors_written, stats.properties_written,
                static_cast<unsigned long long>(stats.bytes_written));
        } else {
            cardinal::log::errorf("studio_min", "save_world failed: %s", err.c_str());
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Load World")) {
        std::string err;
        const auto stats = cardinal::serial::load_world(
            game, "world.snapshot", /*replace_existing*/ true, &err);
        if (err.empty()) {
            cardinal::log::infof("studio_min",
                "Loaded world.snapshot (%u spawned, %u props, %u skipped)",
                stats.actors_spawned, stats.properties_applied,
                stats.actors_skipped);
        } else {
            cardinal::log::errorf("studio_min", "load_world failed: %s", err.c_str());
        }
    }

    // ----- Scene save/load (rendering primitives) -----------------------
    // Saves scene::Entity[] with mesh refs as Mesh::name() strings;
    // load reattaches via the host-provided MeshRegistry. Entities with
    // unnamed or unregistered meshes are skipped (logged in stats).
    ImGui::Separator();
    ImGui::TextDisabled("Scene snapshot (%zu entities, %zu meshes registered):",
                        e.scene().entities().size(), mesh_registry.size());
    if (ImGui::Button("Save Scene")) {
        std::string err;
        const auto stats = cardinal::serial::save_scene(
            e.scene(), "scene.snapshot", &err);
        if (err.empty()) {
            cardinal::log::infof("studio_min",
                "Saved scene.snapshot (%u written, %u skipped meshless, "
                "%u skipped unnamed, %llu bytes)",
                stats.entities_written, stats.entities_skipped_meshless,
                stats.entities_skipped_unnamed,
                static_cast<unsigned long long>(stats.bytes_written));
        } else {
            cardinal::log::errorf("studio_min", "save_scene failed: %s", err.c_str());
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Load Scene")) {
        std::string err;
        const auto stats = cardinal::serial::load_scene(
            e.scene(), mesh_registry, "scene.snapshot",
            /*replace_existing*/ true, &err);
        if (err.empty()) {
            cardinal::log::infof("studio_min",
                "Loaded scene.snapshot (%u spawned, %u skipped unknown mesh)",
                stats.entities_spawned, stats.entities_skipped_unknown_mesh);
            for (const auto& w : stats.warnings) {
                cardinal::log::warnf("studio_min", "scene load: %s", w.c_str());
            }
        } else {
            cardinal::log::errorf("studio_min", "load_scene failed: %s", err.c_str());
        }
    }

    // ----- Gizmo overlays (AABBs + camera frustum + lights) -------------
    // Routes through the active pipeline → underlying scene::ForwardRenderer.
    // Default off; toggling pushes one extra wireframe draw + a couple
    // dozen verts per AABB / 24 verts per frustum / ~40 verts per cone
    // / ~144 verts per sphere.
    ImGui::Separator();
    {
        static bool show_aabbs   = false;
        static bool show_frustum = false;
        static bool show_axes    = false;
        static bool show_lights  = false;
        if (ImGui::Checkbox("AABBs", &show_aabbs)) {
            if (auto* p = e.pipelines().active()) p->set_show_aabbs(show_aabbs);
        }
        ImGui::SameLine();
        if (ImGui::Checkbox("Frustum", &show_frustum)) {
            if (auto* p = e.pipelines().active()) p->set_show_frustum(show_frustum);
        }
        ImGui::SameLine();
        if (ImGui::Checkbox("Axes", &show_axes)) {
            if (auto* p = e.pipelines().active()) p->set_show_axes(show_axes);
        }
        ImGui::SameLine();
        if (ImGui::Checkbox("Lights", &show_lights)) {
            if (auto* p = e.pipelines().active()) p->set_show_lights(show_lights);
        }
        ImGui::TextDisabled("  (green=visible, red=culled, white=frustum,");
        ImGui::TextDisabled("   RGB=world axes, light-colored=light volumes)");
    }

    // ----- Render-pipeline frame stats (entity frustum cull) ------------
    // Pulled from the active pipeline's stats() — populated by the
    // forward renderer's per-frame frustum_cull_aabbs pass. When the
    // camera excludes part of the scene, drawn < total visible.
    ImGui::Separator();
    if (auto* pipe = e.pipelines().active()) {
        const auto rs = pipe->stats();
        // Cull counts are already per-frame snapshots (not monotonic
        // counters), so they go straight into PhaseRing without a
        // delta step. Same 60-sample window as the µs breakdown +
        // JobSystem totals — one consistent smoothing horizon across
        // the panel.
        static PhaseRing r_ent_total, r_ent_drawn, r_ent_culled, r_draws, r_pushes;
        r_ent_total .push(static_cast<float>(rs.entities_total));
        r_ent_drawn .push(static_cast<float>(rs.entities_drawn));
        r_ent_culled.push(static_cast<float>(rs.entities_culled));
        r_draws     .push(static_cast<float>(rs.draw_calls));
        r_pushes    .push(static_cast<float>(rs.pc_pushes));
        float a_total = 0, p_total = 0;     r_ent_total .avg_peak(a_total,  p_total);
        float a_drawn = 0, p_drawn = 0;     r_ent_drawn .avg_peak(a_drawn,  p_drawn);
        float a_culled = 0, p_culled = 0;   r_ent_culled.avg_peak(a_culled, p_culled);
        float a_draws = 0, p_draws = 0;     r_draws     .avg_peak(a_draws,  p_draws);
        float a_pushes = 0, p_pushes = 0;   r_pushes    .avg_peak(a_pushes, p_pushes);

        ImGui::TextDisabled("Frustum cull (entities, avg / peak over %d frames):",
                            PhaseRing::kCapacity);
        ImGui::Text("  total  %5.1f / %-5.0f", a_total,  p_total);
        ImGui::Text("  drawn  %5.1f / %-5.0f", a_drawn,  p_drawn);
        ImGui::Text("  culled %5.1f / %-5.0f", a_culled, p_culled);
        if (a_total > 0.0f) {
            const float pct = 100.0f * a_culled / a_total;
            ImGui::Text("  %.1f%% culled (avg)", pct);
        }
        ImGui::Text("  draws  %5.1f / %-5.0f  (one per EntityWork; GPU MVP via push constants)",
                    a_draws, p_draws);
        ImGui::Text("  pushes %5.1f / %-5.0f  (push_constants writes; deduped at entity boundaries)",
                    a_pushes, p_pushes);
        // Explicit dedup % so users don't have to do mental math against
        // draws/pushes — surfaces the saving when split entities + morton-
        // grouped vgeom clusters share an MVP across consecutive draws.
        if (a_draws > 0.0f) {
            const float dedup_pct = 100.0f * (1.0f - a_pushes / a_draws);
            const ImVec4 col = (dedup_pct >= 25.0f)
                ? ImVec4(0.40f, 0.85f, 0.45f, 1.0f)   // green: meaningful savings
                : ImVec4(0.65f, 0.65f, 0.65f, 1.0f);  // grey: little to dedup
            ImGui::TextColored(col, "  push dedup: %.1f%% saved", dedup_pct);
        }

        // vgeom aggregate — only rendered when at least one mesh in the
        // scene has a cooked + enabled vgeom hierarchy this frame
        // (master == 0 otherwise). Separately smoothed because the
        // numbers grow much larger (millions of master tris) than the
        // entity counts above.
        if (rs.vgeom_master_tris > 0) {
            static PhaseRing r_vmaster, r_vdrawn, r_vcut, r_vculled;
            r_vmaster.push(static_cast<float>(rs.vgeom_master_tris));
            r_vdrawn .push(static_cast<float>(rs.vgeom_drawn_tris));
            r_vcut   .push(static_cast<float>(rs.vgeom_cut_clusters));
            r_vculled.push(static_cast<float>(rs.vgeom_culled_clusters));
            float a_master = 0, p_master = 0; r_vmaster.avg_peak(a_master, p_master);
            float a_vdrawn = 0, p_vdrawn = 0; r_vdrawn .avg_peak(a_vdrawn, p_vdrawn);
            float a_vcut = 0,   p_vcut = 0;   r_vcut   .avg_peak(a_vcut,   p_vcut);
            float a_vculled=0,  p_vculled=0;  r_vculled.avg_peak(a_vculled,p_vculled);

            ImGui::Separator();
            ImGui::TextDisabled("vgeom (avg / peak over %d frames):",
                                PhaseRing::kCapacity);
            const float reduction = (a_master > 0.0f)
                ? 100.0f * (1.0f - a_vdrawn / a_master) : 0.0f;
            ImGui::Text("  drawn:    %.0f / %.0f tris  (master %.0f, %.1f%% reduction)",
                        a_vdrawn, p_vdrawn, a_master, reduction);
            ImGui::Text("  clusters: %.0f cut, %.0f culled (avg)",
                        a_vcut, a_vculled);
        }

        // Per-phase wall-clock breakdown — measured around the cull
        // pre-pass, Phase 1 sizing, Phase 2 parallel transform, painter
        // sort, and GPU submit. Phase 2 reports the longest worker's
        // runtime (parallel_for blocks on its slowest task), so it's
        // the wall-clock span, not the summed CPU time.
        //
        // Smoothed over a 60-sample ring (~1s at 60 FPS, ~0.5s at
        // 120 FPS). Per-frame raw values jitter too much to read; the
        // ring shows the steady-state cost + the worst frame in the
        // window so transient stalls stay visible without the eye-melt.
        const float total_us = rs.cull_us + rs.phase1_us + rs.phase2_us
                             + rs.sort_us + rs.submit_us;
        if (total_us > 0.0f) {
            static PhaseRing r_cull, r_p1, r_p2, r_sort, r_submit, r_total;
            r_cull  .push(rs.cull_us);
            r_p1    .push(rs.phase1_us);
            r_p2    .push(rs.phase2_us);
            r_sort  .push(rs.sort_us);
            r_submit.push(rs.submit_us);
            r_total .push(total_us);

            ImGui::Separator();
            ImGui::TextDisabled("Frame time breakdown (\xc2\xb5s, avg / peak over %d frames):",
                                PhaseRing::kCapacity);
            float avg_total = 0.0f, peak_total = 0.0f;
            r_total.avg_peak(avg_total, peak_total);
            auto row = [&](const char* label, const PhaseRing& r) {
                float a = 0.0f, p = 0.0f;
                r.avg_peak(a, p);
                const float pct = (avg_total > 0.0f)
                    ? (100.0f * a / avg_total) : 0.0f;
                ImGui::Text("  %-8s %7.1f / %-7.1f  (%.1f%%)",
                            label, a, p, pct);
            };
            row("cull",   r_cull);
            row("phase1", r_p1);
            row("phase2", r_p2);
            row("sort",   r_sort);
            row("submit", r_submit);
            ImGui::Text("  total    %7.1f / %-7.1f", avg_total, peak_total);
        }
    }

    // ----- Scene info ---------------------------------------------------
    // Scene snapshot save/load is a known gap: cardinal::serial today
    // operates on cardinal::actor::World (gameplay state) + Sky, but
    // Engine exposes cardinal::scene::Scene (rendering primitives) and
    // doesn't surface World/Sky. Wiring serial through requires either:
    //   (a) Engine::world() / sky() accessors + serial::save_world/sky
    //       hooked to panel buttons, OR
    //   (b) A new scene::serialize / deserialize that walks Entity[]
    //       (mesh refs need a re-create story — CookDesc cache?).
    // Until either lands, surface a read-only Scene snapshot below so
    // users can at least SEE what's loaded without dropping into the
    // dedicated outliner panel.
    {
        ImGui::Separator();
        ImGui::TextDisabled("Scene info:");
        const auto& ents = e.scene().entities();
        cardinal::u32 visible = 0, with_mesh = 0;
        cardinal::u64 total_verts = 0;
        for (const auto& en : ents) {
            if (en.visible) ++visible;
            if (en.mesh) {
                ++with_mesh;
                total_verts += en.mesh->vertex_count();
            }
        }
        ImGui::Text("  entities: %zu  (%u visible, %u with mesh)",
                    ents.size(), visible, with_mesh);
        ImGui::Text("  total mesh verts: %llu",
                    static_cast<unsigned long long>(total_verts));
    }

    // ----- JobSystem stats ----------------------------------------------
    // Snapshot all worker counters in one call. Lets the user see HOW
    // the parallel_for in Phase 2 is utilising workers — high steal
    // ratio means good load-balancing across an uneven work mix; lots
    // of sleep events means the pool is idle most of the time (Phase 2
    // wasn't very parallelisable, or there was no work to do).
    //
    // Counters are MONOTONIC so per-frame DELTAS are what's interesting
    // (jobs run THIS frame, not since boot). Same 60-sample PhaseRing
    // pattern as the frame-time breakdown — push the delta on each
    // draw, display avg + peak over the window.
    if (auto* js = cardinal::async::pool()) {
        const auto js_stats = js->stats();
        // Previous-frame snapshot for delta computation. First call
        // has prev == 0 so the delta is the boot-to-now total — that
        // single sample is harmless (peak will dominate, avg fills in
        // over the next ~60 frames).
        static cardinal::u64 prev_jobs    = 0;
        static cardinal::u64 prev_stolen  = 0;
        static cardinal::u64 prev_attempts= 0;
        static PhaseRing r_jobs, r_stolen, r_attempts, r_max_queue;
        const float d_jobs     = static_cast<float>(js_stats.total_jobs_executed - prev_jobs);
        const float d_stolen   = static_cast<float>(js_stats.total_jobs_stolen   - prev_stolen);
        const float d_attempts = static_cast<float>(js_stats.total_steal_attempts- prev_attempts);
        prev_jobs     = js_stats.total_jobs_executed;
        prev_stolen   = js_stats.total_jobs_stolen;
        prev_attempts = js_stats.total_steal_attempts;
        r_jobs     .push(d_jobs);
        r_stolen   .push(d_stolen);
        r_attempts .push(d_attempts);
        r_max_queue.push(static_cast<float>(js_stats.max_queue_depth));

        float avg_jobs = 0, pk_jobs = 0;       r_jobs     .avg_peak(avg_jobs, pk_jobs);
        float avg_stolen = 0, pk_stolen = 0;   r_stolen   .avg_peak(avg_stolen, pk_stolen);
        float avg_attempts = 0, pk_attempts = 0;
                                               r_attempts .avg_peak(avg_attempts, pk_attempts);
        float avg_q = 0, pk_q = 0;             r_max_queue.avg_peak(avg_q, pk_q);

        ImGui::Separator();
        ImGui::TextDisabled("JobSystem (workers=%u perf=%u gen=%u bg=%u, "
                            "avg / peak over %d frames):",
                            js->worker_count(),
                            js->perf_worker_count(),
                            js->general_worker_count(),
                            js->background_worker_count(),
                            PhaseRing::kCapacity);
        ImGui::Text("  jobs/frame:  %7.1f / %-7.0f", avg_jobs, pk_jobs);
        ImGui::Text("  steal hits:  %7.1f / %-7.0f  of %.1f attempts",
                    avg_stolen, pk_stolen, avg_attempts);
        if (avg_attempts > 0.0f) {
            const float hit_pct = 100.0f * avg_stolen / avg_attempts;
            ImGui::SameLine(); ImGui::Text("  (%.1f%% hit)", hit_pct);
        }
        ImGui::Text("  max queue:   %7.1f / %-7.0f  (own-deque depth)",
                    avg_q, pk_q);

        // Per-worker breakdown — collapsed by default so the panel
        // doesn't get pushed off-screen on hosts with 32 workers.
        // Tier-coloured ID column shows which workers are perf vs
        // general vs background at a glance; the row's job count
        // reveals load distribution (a single-worker run shows a
        // huge value in one row + zeros elsewhere = poor parallelism).
        if (ImGui::TreeNode("Per-worker breakdown")) {
            if (ImGui::BeginTable("##js_workers", 7,
                    ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                    ImGuiTableFlags_SizingStretchProp))
            {
                ImGui::TableSetupColumn("ID",       ImGuiTableColumnFlags_WidthFixed, 32.0f);
                ImGui::TableSetupColumn("Tier",     ImGuiTableColumnFlags_WidthFixed, 76.0f);
                ImGui::TableSetupColumn("Jobs",     ImGuiTableColumnFlags_WidthFixed, 80.0f);
                ImGui::TableSetupColumn("Stolen",   ImGuiTableColumnFlags_WidthFixed, 80.0f);
                ImGui::TableSetupColumn("Steal%",   ImGuiTableColumnFlags_WidthFixed, 60.0f);
                ImGui::TableSetupColumn("Sleep",    ImGuiTableColumnFlags_WidthFixed, 80.0f);
                ImGui::TableSetupColumn("Queue",    ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableHeadersRow();

                const auto tier_label = [](cardinal::WorkerTier t) -> const char* {
                    switch (t) {
                        case cardinal::WorkerTier::Performance: return "perf";
                        case cardinal::WorkerTier::General:     return "gen";
                        case cardinal::WorkerTier::Background:  return "bg";
                    }
                    return "?";
                };
                const auto tier_color = [](cardinal::WorkerTier t) -> ImVec4 {
                    switch (t) {
                        case cardinal::WorkerTier::Performance:
                            return ImVec4(0.40f, 0.85f, 0.45f, 1.0f);   // green
                        case cardinal::WorkerTier::General:
                            return ImVec4(0.45f, 0.70f, 0.95f, 1.0f);   // blue
                        case cardinal::WorkerTier::Background:
                            return ImVec4(0.65f, 0.65f, 0.65f, 1.0f);   // grey
                    }
                    return ImVec4(1, 1, 1, 1);
                };

                for (cardinal::u32 wi = 0; wi < js->worker_count(); ++wi) {
                    const auto& w = js_stats.per_worker[wi];
                    const auto tier = js->worker_tier(wi);
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::Text("%u", wi);
                    ImGui::TableSetColumnIndex(1);
                    ImGui::TextColored(tier_color(tier), "%s", tier_label(tier));
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%llu", static_cast<unsigned long long>(w.jobs_executed));
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%llu", static_cast<unsigned long long>(w.jobs_stolen));
                    ImGui::TableSetColumnIndex(4);
                    if (w.steal_attempts > 0) {
                        ImGui::Text("%.0f%%",
                            100.0f * static_cast<float>(w.jobs_stolen) /
                                     static_cast<float>(w.steal_attempts));
                    } else {
                        ImGui::TextDisabled("\xe2\x80\x94");   // em-dash UTF-8
                    }
                    ImGui::TableSetColumnIndex(5);
                    ImGui::Text("%llu", static_cast<unsigned long long>(w.sleep_events));
                    ImGui::TableSetColumnIndex(6);
                    ImGui::Text("%u", w.queue_depth);
                }
                ImGui::EndTable();
            }
            ImGui::TreePop();
        }
    }

    ImGui::End();
}

// MinSpinner — tiny demo GameActor so Save/Load World writes a non-
// empty file. Mirrors the SpinnerActor pattern from the legacy
// `samples/03_studio` sample. Reflected properties get persisted by
// serial::save_world; load re-spawns by class name + replays values.
class MinSpinner final : public cardinal::game::GameActor {
public:
    float                 rpm{30.0f};
    cardinal::scene::Vec3 axis{0.0f, 1.0f, 0.0f};
    bool                  reverse{false};

    void on_tick(float dt) override {
        if (owner() == nullptr) return;
        auto* tr = owner()->get_component<cardinal::actor::TransformComponent>();
        if (tr == nullptr) return;
        const float omega = (reverse ? -1.0f : 1.0f) * rpm * 6.2831853f / 60.0f;
        tr->rotation_euler.x += axis.x * omega * dt;
        tr->rotation_euler.y += axis.y * omega * dt;
        tr->rotation_euler.z += axis.z * omega * dt;
    }
};

CARDINAL_REGISTER_GAME_CLASS(MinSpinner, "Demo/MinSpinner",
    PROP_FLOAT(rpm,     0.0f, 600.0f, "Revolutions per minute")
    PROP_VEC3 (axis,                  "Rotation axis (world space)")
    PROP_BOOL (reverse,               "Spin in the opposite direction"))

class MinApp final : public cardinal::ui::StudioApplication {
public:
    void on_init(cardinal::engine::Engine& e) override {
        cardinal::log::infof("studio_min",
            "StudioApplication online — engine reports backend=%s",
            e.active_backend() == cardinal::engine::EngineDesc::Backend::Vulkan
                ? "Vulkan" : "D3D12");

        // Spawn two MinSpinner instances so Save/Load World has actors
        // to serialise. Different rpm/reverse values to make the
        // round-trip visible (post-load, the reflected properties
        // should match what we set here).
        if (auto* a = game_.spawn_class("MinSpinner", "SpinnerA")) {
            if (auto* s = a->get_component<MinSpinner>()) {
                s->rpm = 60.0f; s->reverse = false;
            }
        }
        if (auto* a = game_.spawn_class("MinSpinner", "SpinnerB")) {
            if (auto* s = a->get_component<MinSpinner>()) {
                s->rpm = 15.0f; s->reverse = true;
            }
        }
        cardinal::log::infof("studio_min",
            "spawned %u game actor(s) for save/load demo", game_.game_actor_count());

        // Spawn two scene::Entity instances backed by named meshes, and
        // mirror those meshes into mesh_registry_ so Save/Load Scene
        // round-trips them. Named meshes are the contract:
        //   - save_scene writes Mesh::name() per entity
        //   - load_scene looks up the name in the registry to reattach
        // Different transforms + tints make the round-trip visually
        // verifiable (post-load, the entities should reappear in the
        // same spots with the same colours).
        auto& scene = e.scene();
        auto& dev   = e.device();
        // Box + sphere meshes are cached on the host so the runtime
        // Spawn Cube / Spawn Sphere buttons (Physics panel) can reuse
        // the same shared_ptr — adding 100 cubes still costs 1 mesh.
        box_mesh_ = cardinal::scene::Mesh::make_box(dev, 1.0f);
        if (box_mesh_) {
            box_mesh_->set_name("primitives/box");
            mesh_registry_["primitives/box"] = box_mesh_;
        }
        sphere_mesh_ = cardinal::scene::Mesh::make_sphere(dev, 0.5f, 24);
        if (sphere_mesh_) {
            sphere_mesh_->set_name("primitives/sphere");
            mesh_registry_["primitives/sphere"] = sphere_mesh_;
        }
        auto& box    = box_mesh_;
        auto& sphere = sphere_mesh_;
        // Ground plane — 20×20 m, 8×8 subdivisions, normal +Y. Gives
        // the Spot cone footprint somewhere to land + the warm
        // Directional key a flat reference surface (the cube and
        // sphere alone left the cone illuminating empty space). Sits
        // at y=0 so the cube (centre y=1, size 1 → bottom y=0.5) and
        // sphere (centre y=1, radius 0.5 → bottom y=0.5) hover 0.5m
        // above it, leaving a clean shadow-shaped gap when shadow
        // mapping arrives.
        auto plane = cardinal::scene::Mesh::make_plane(dev, 20.0f, 8);
        if (plane) {
            plane->set_name("primitives/plane_ground");
            mesh_registry_["primitives/plane_ground"] = plane;
        }
        // Ground plane is a one-shot static entity — no physics body
        // for it as a scene entity (the physics ground is the infinite
        // y=0 plane created below). Tint left at white because
        // make_plane bakes the grey directly into the vertex colours.
        if (plane) {
            auto& en = scene.add_entity("DemoGround");
            en.mesh = plane;
            en.transform.translation = cardinal::scene::Vec3{0.0f, 0.0f, 0.0f};
            en.tint = cardinal::scene::Vec3{1.0f, 1.0f, 1.0f};   // make_plane's grey baked in
            // Semi-glossy dielectric floor — mid roughness so the
            // moving lights streak a soft highlight across the plane.
            // Distinct from the chrome sphere (metal, sharp) and rough
            // cube: three different PBR materials live at once via the
            // slot-1 storage buffer.
            en.material.specular_intensity = 0.6f;
            en.material.roughness          = 0.35f;
            en.material.metalness          = 0.0f;
        }
        // Cube + sphere entities are spawned by the physics block below
        // via spawn_demo_body(), which pairs each new scene::Entity with
        // a physics::Body. Same path runs when the user clicks Spawn
        // Cube / Spawn Sphere in the Physics panel — one factory, one
        // bodies_ vector, one sync loop.
        // Attach vgeom hierarchies on the box + sphere demo meshes —
        // small enough that the cook is instant + each becomes a
        // single-cluster hierarchy, but the round-trip exercises the
        // vgeom save/load path (Save Scene → close → Load Scene should
        // restore the attached state visible in the vgeom panel). The
        // ground plane is intentionally left un-vgeom'd; it's a
        // single-LOD reference surface, not a content asset under test.
        if (box)    cardinal::scene::vgeom_attach(*box);
        if (sphere) cardinal::scene::vgeom_attach(*sphere);
        cardinal::log::infof("studio_min",
            "spawned %zu scene entities + %zu mesh registry entries for Save/Load Scene "
            "(vgeom attached on box + sphere)",
            scene.entities().size(), mesh_registry_.size());

        // Lighting — three simultaneous lights of all three kinds:
        // Directional key + Spot accent + Point fill. Since the
        // StructuredBuffer<Light> migration the renderer has no fixed
        // slot cap (up to kMaxLights=64), so all three light at once —
        // this is the visible proof the push-block→SSBO arc landed.
        // The "Extra lights" toggle in the Sky panel piles on a ring of
        // coloured point lights to exercise the arbitrary-count path.
        lights_.set_ambient(cardinal::scene::Vec3{0.15f, 0.15f, 0.18f});

        cardinal::scene::Light key{};
        key.kind      = cardinal::scene::LightKind::Directional;
        key.direction = cardinal::scene::Vec3{-0.4f, -1.0f, -0.3f};   // points AT scene
        key.color     = cardinal::scene::Vec3{1.0f, 0.95f, 0.85f};    // warm
        key.intensity = 1.0f;
        lights_.add(key);

        // Second slot: a Spot light angled in from above-front. Cool
        // teal so the cone outline is obvious against the warm key's
        // ambient contribution. Direction is "FROM light TO target"
        // (engine convention) — straight down with a slight tilt so the
        // cone falloff is visible on the cube's near face and the
        // sphere's curvature rather than hidden under their tops.
        cardinal::scene::Light spot{};
        spot.kind           = cardinal::scene::LightKind::Spot;
        spot.position       = cardinal::scene::Vec3{1.5f, 4.0f, 1.0f};
        spot.direction      = cardinal::scene::Vec3{-0.25f, -1.0f, -0.2f};  // normalized in PS
        spot.color          = cardinal::scene::Vec3{0.30f, 0.85f, 1.00f};   // cool teal
        spot.intensity      = 5.0f;
        spot.range          = 10.0f;                                         // metres
        spot.spot_inner_cos = 0.95f;                                         // ~18° inner cone
        spot.spot_outer_cos = 0.80f;                                         // ~37° outer cone half-angle
        lights_.add(spot);

        // Third light — a warm-orange Point fill from the back-left,
        // now LIVE (was capped out under the old 2-slot push array).
        // It rims the side of the cube/sphere the cool Spot doesn't
        // reach, so the three-kind setup reads clearly.
        cardinal::scene::Light point{};
        point.kind      = cardinal::scene::LightKind::Point;
        point.position  = cardinal::scene::Vec3{-2.0f, 2.5f, -1.5f};
        point.color     = cardinal::scene::Vec3{1.0f, 0.55f, 0.30f};   // warm orange
        point.intensity = 4.0f;                                          // bright at source
        point.range     = 6.0f;                                          // metres
        lights_.add(point);
        // Remember how many "base" lights precede any dynamically-added
        // ring lights so the Extra-lights toggle can append/trim without
        // disturbing the Directional/Spot/Point the rest of on_simulate
        // mutates by fixed index (0 = sun, 1 = orbiting spot).
        base_light_count_ = static_cast<cardinal::u32>(lights_.lights().size());

        // ----- Physics ---------------------------------------------
        // Spawn a static ground plane + dynamic cube + dynamic sphere
        // mirroring the visual scene 1:1. Bodies use the same world-
        // space positions as the entities; the per-frame phys → entity
        // sync below pushes integrator output back into the scene
        // Transforms. Materials picked for a satisfying-bouncy demo:
        // sphere bounces a bit higher than the cube (restitution 0.6
        // vs 0.3); both with moderate friction so they don't slide
        // forever after settling.
        phys_ = cardinal::physics::World::create();
        if (phys_) {
            phys_->set_gravity(cardinal::physics::Vec3{0, -9.81f, 0});

            // Static ground — n·x + d = 0 with n=(0,1,0), d=0 puts the
            // plane at y=0 exactly under the DemoGround entity.
            cardinal::physics::BodyDesc gd{};
            gd.type     = cardinal::physics::BodyType::Static;
            gd.position = cardinal::physics::Vec3{0, 0, 0};
            gd.collider = cardinal::physics::Collider::make_plane(
                cardinal::physics::Vec3{0, 1, 0}, 0.0f);
            gd.material.restitution = 0.4f;
            gd.material.friction    = 0.6f;
            phys_->create_body(gd);

            // Initial cube — 20° tilt around an oblique axis + ~3 rad/s
            // mixed-XYZ spin so each face shows during the fall.
            spawn_demo_body(e,
                DemoKind::Cube,
                cardinal::physics::Vec3{-1.5f, 3.0f, 0.0f},
                cardinal::core::axis_angle(
                    cardinal::physics::Vec3{0.3f, 1.0f, 0.4f},
                    20.0f * 3.14159265f / 180.0f),
                cardinal::physics::Vec3{2.0f, 1.5f, 1.0f},
                cardinal::scene::Vec3{0.85f, 0.65f, 0.30f});

            // Initial sphere — slightly higher drop, no spin (rolling a
            // smooth sphere out of rest isn't visually informative).
            spawn_demo_body(e,
                DemoKind::Sphere,
                cardinal::physics::Vec3{1.5f, 3.5f, 0.0f},
                cardinal::physics::Quat::identity(),
                cardinal::physics::Vec3{0, 0, 0},
                cardinal::scene::Vec3{0.55f, 0.85f, 0.95f});

            cardinal::log::infof("studio_min",
                "physics: %zu bodies (ground + %zu dynamic), g=-9.81 m/s²",
                phys_->body_count(), bodies_.size());
        }

        // ----- Audio -----------------------------------------------
        // First real sound out of the engine: a short sine "impact"
        // cue triggered from the physics contact callback, mixed by
        // the device-agnostic audio::Engine and pushed to the default
        // output device by the WASAPI backend. Listener tracks the
        // camera each frame (on_simulate). Silent-degrading: if no
        // device, start_default_output returns null and play_3d is
        // just bookkeeping.
        audio_ = cardinal::audio::Engine::create();
        if (audio_) {
            cardinal::audio::Cue impact;
            impact.id                = "impact";
            impact.kind              = cardinal::audio::CueKind::SineWave;
            impact.duration_s        = 0.10f;          // short blip
            impact.sine_frequency_hz = 150.0f;         // low thud
            impact.gain              = 0.8f;
            audio_->register_cue(impact);
            audio_out_ = cardinal::audio::start_default_output(audio_);

            // Contact callback fires inside phys_->step() (same thread
            // as on_simulate). Gate hard against the settling pile:
            // a global cooldown PLUS a real-impact velocity threshold
            // (resting contacts have ~0 speed and stay silent).
            phys_->set_contact_callback(
                [this](const cardinal::physics::ContactEvent& c) {
                    if (audio_ == nullptr || !audio_enabled_) return;
                    if (audio_impact_cd_ > 0.0f)              return;
                    const float va = vlen(phys_->velocity(c.a));
                    const float vb = vlen(phys_->velocity(c.b));
                    if (std::max(va, vb) < 1.5f)             return;
                    audio_->play_3d("impact",
                        cardinal::scene::Vec3{c.point.x, c.point.y, c.point.z},
                        cardinal::audio::kChannelSfx, 1.0f, 1.0f, false);
                    audio_impact_cd_ = 0.08f;   // ~12 hits/s ceiling
                });
        }

        // ----- Networking -------------------------------------------
        // In-process loopback transport (SP path). listen() synthesises
        // the single virtual peer; the Replicator broadcasts/echoes
        // snapshots over it. The exact same code becomes MMO by
        // swapping create_loopback() → create_udp() — gameplay above
        // never branches on "is multiplayer".
        net_ = cardinal::net::Transport::create_loopback();
        if (net_) {
            net_->listen(0);
            repl_ = cardinal::make_unique<cardinal::net::Replicator>(*net_);
            net_events_.clear();
            net_->poll(net_events_);          // drain the Connected event
            cardinal::log::infof("studio_min",
                "net: loopback transport up, replicator ready");
        }
    }

    // |v| helper for the contact-velocity gate.
    static float vlen(const cardinal::physics::Vec3& v) {
        return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    // Demo-body factory — creates one scene::Entity + one physics::Body
    // for either a unit cube or a unit-diameter sphere at `pos`, with
    // the given starting orientation + angular velocity. Captures the
    // pairing in bodies_ so the per-frame phys → transform sync and the
    // Reset / Clear-Spawned UI affordances find it. Same code path runs
    // for the initial demo bodies (on_init) and the user-spawned ones
    // (Physics panel buttons) so there's one definition of "what a
    // demo body is".
    enum class DemoKind { Cube, Sphere };
    struct DemoBody {
        cardinal::physics::BodyHandle body;
        cardinal::u32                 entity_id;
        DemoKind                      kind;
        cardinal::physics::Vec3       initial_pos;
        cardinal::physics::Quat       initial_orient;
        cardinal::physics::Vec3       initial_angvel;
        // Spawn tint, kept so the per-frame pick-highlight blend has
        // a stable base to lerp back to (the highlight overrides
        // entity.tint each frame, so we can't read it back from there).
        cardinal::scene::Vec3         base_tint{1, 1, 1};
    };
    void spawn_demo_body(cardinal::engine::Engine& e,
                         DemoKind kind,
                         const cardinal::physics::Vec3& pos,
                         const cardinal::physics::Quat& orient,
                         const cardinal::physics::Vec3& angvel,
                         const cardinal::scene::Vec3& tint)
    {
        if (phys_ == nullptr) return;
        auto mesh = (kind == DemoKind::Cube) ? box_mesh_ : sphere_mesh_;
        if (!mesh) return;

        auto& scene = e.scene();
        // Auto-generated name keeps the Outliner readable + makes the
        // Save Scene round-trip emit a unique entity per spawned body.
        const char* prefix = (kind == DemoKind::Cube) ? "Cube_" : "Sphere_";
        std::string name = prefix + std::to_string(bodies_.size());
        auto& en = scene.add_entity(std::move(name));
        en.mesh = mesh;
        en.transform.translation =
            cardinal::scene::Vec3{pos.x, pos.y, pos.z};
        en.tint = tint;

        cardinal::physics::BodyDesc bd{};
        bd.type     = cardinal::physics::BodyType::Dynamic;
        bd.position = pos;
        if (kind == DemoKind::Cube) {
            bd.collider = cardinal::physics::Collider::make_box(
                cardinal::physics::Vec3{0.5f, 0.5f, 0.5f});
            bd.material.restitution = 0.3f;
            bd.material.friction    = 0.6f;
        } else {
            bd.collider = cardinal::physics::Collider::make_sphere(0.5f);
            bd.material.restitution = 0.6f;
            bd.material.friction    = 0.4f;
        }
        bd.mass            = 1.0f;
        bd.linear_damping  = 0.05f;
        bd.angular_damping = 0.10f;
        bd.orientation     = orient;
        bd.angular_velocity = angvel;
        const auto h = phys_->create_body(bd);

        bodies_.push_back(
            DemoBody{h, en.id, kind, pos, orient, angvel, tint});

        // Replicate the spawn on the ReliableOrdered lifecycle channel
        // (only while the net path is active — the initial on_init
        // bodies predate the transport, so they're never queued).
        if (net_ != nullptr && net_replicate_) {
            cardinal::net::RepEvent ev;
            ev.kind            = cardinal::net::RepEventKind::Spawn;
            ev.id              = en.id;
            ev.archetype       = (kind == DemoKind::Cube) ? 0u : 1u;
            ev.state.id        = en.id;
            ev.state.position  =
                cardinal::scene::Vec3{pos.x, pos.y, pos.z};
            net_evtq_.push_back(ev);
        }
    }

    void on_simulate(cardinal::engine::Engine& e, float dt) override {
        // Tick the sky so its hour advances. Default time_scale is 120
        // (12 minutes = 24h day), so visual change is slow but the
        // Save/Load Sky buttons round-trip the live state.
        sky_.tick(dt);
        // Feed the active sky state into the scene's first Directional
        // light + ambient — the renderer's "key" light tracks the sun
        // each frame. sky_.state().sun_dir already points AT the scene
        // (engine convention for Directional lights), so it drops in
        // without negation. Sun intensity multiplies the per-key
        // sun_color so dawn/dusk fade smoothly through the LightSet.
        // Sky_track_ toggle in the panel lets the user freeze the sun
        // to inspect a single time-of-day.
        if (sky_track_lights_ && !lights_.lights().empty()) {
            const auto& s = sky_.state();
            auto& key = lights_.lights().front();
            if (key.kind == cardinal::scene::LightKind::Directional) {
                key.direction = s.sun_dir;
                key.color     = s.sun_color;
                key.intensity = s.sun_intensity;
            }
            lights_.set_ambient(s.ambient);
            // Swapchain clear color tracks sky horizon — the bg
            // tints to dawn-orange / midday-blue / dusk-red / night-
            // purple as the clock advances. Reuses the engine's new
            // set_clear_color() API so the value applies on the next
            // begin_frame and survives a backend swap.
            if (sky_track_clear_color_) {
                e.set_clear_color(s.horizon.x, s.horizon.y, s.horizon.z, 1.0f);
            }
        }

        // Spot-light orbit — slot 1 (the Spot demo light) circles the
        // origin at radius/height/period set by the Sky panel. Direction
        // always re-aims at the scene-centre target so the cone tracks
        // the cube + sphere as it sweeps around them. Period guarded
        // against zero so the panel slider can't divide-by-zero.
        if (spot_orbit_enabled_ && lights_.lights().size() >= 2) {
            spot_orbit_time_ += dt;
            const float period =
                (spot_orbit_period_ > 0.01f) ? spot_orbit_period_ : 0.01f;
            const float theta =
                (spot_orbit_time_ / period) * 6.28318530717958647692f;
            auto& spot = lights_.lights()[1];
            if (spot.kind == cardinal::scene::LightKind::Spot) {
                spot.position = cardinal::scene::Vec3{
                    std::cos(theta) * spot_orbit_radius_,
                    spot_orbit_height_,
                    std::sin(theta) * spot_orbit_radius_};
                // Aim at the scene-centre target — half-metre above the
                // ground so the cone hits the cube + sphere directly
                // rather than splattering across the floor.
                const cardinal::scene::Vec3 target{0.0f, 0.5f, 0.0f};
                cardinal::scene::Vec3 d{
                    target.x - spot.position.x,
                    target.y - spot.position.y,
                    target.z - spot.position.z};
                const float len = std::sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
                if (len > 1e-6f) {
                    d.x /= len; d.y /= len; d.z /= len;
                }
                spot.direction = d;
            }
        }

        // Extra-lights demo — proof the StructuredBuffer<Light> path
        // handles an arbitrary count (the old push array hard-capped at
        // 2). A ring of distinctly-hued Point lights orbits the scene
        // just above the ground; the toggle/count live in the Sky
        // panel. Lights live in [base_light_count_, base+N) so the
        // fixed-index sun (0) / spot (1) mutations above are untouched.
        {
            auto& L = lights_.lights();
            const cardinal::u32 want =
                extra_lights_enabled_
                    ? static_cast<cardinal::u32>(extra_lights_count_) : 0u;
            const cardinal::usize target_total =
                static_cast<cardinal::usize>(base_light_count_) + want;
            if (L.size() != target_total) L.resize(target_total);
            if (want > 0) {
                extra_lights_time_ += dt;
                const float base_a =
                    extra_lights_time_ * 0.5f;   // ~0.08 Hz ring spin
                for (cardinal::u32 k = 0; k < want; ++k) {
                    const float frac =
                        static_cast<float>(k) / static_cast<float>(want);
                    const float a =
                        base_a + frac * 6.28318530717958647692f;
                    auto& rl = L[base_light_count_ + k];
                    rl.kind      = cardinal::scene::LightKind::Point;
                    rl.position  = cardinal::scene::Vec3{
                        std::cos(a) * 6.0f, 1.2f, std::sin(a) * 6.0f};
                    rl.color     = hue_rgb(frac);
                    rl.intensity = 3.0f;
                    rl.range     = 5.0f;
                }
            }
        }

        // Tick game — Stopped state still drives sim_ pre/post hooks
        // so the world is in a consistent state for save/load.
        game_.tick(dt);

        // Step physics + push body positions back to the matching scene
        // Entities. World::step() internally accumulates and runs
        // 0..K fixed-timestep substeps so the sim stays deterministic
        // across frame-rate jitter. Skip when the UI toggle is off so
        // the user can freeze the scene to inspect lighting / gizmos.
        if (phys_ != nullptr && physics_enabled_) {
            // Wind field — sync the registered ForceField with current
            // UI state. Done before step() so the integrator sees the
            // up-to-date field this tick. Only acts when the UI marked
            // the params dirty; idle frames are zero-cost.
            if (wind_dirty_) {
                if (wind_field_active_) {
                    phys_->remove_force_field(wind_field_handle_);
                    wind_field_active_ = false;
                }
                if (wind_enabled_) {
                    cardinal::physics::World::ForceField f{};
                    f.kind        = cardinal::physics::World::FieldKind::UniformAccel;
                    f.direction   = wind_dir_;
                    f.strength    = wind_strength_;
                    f.radius      = 0.0f;             // unbounded
                    f.layer_mask  = 0xFFFFFFFFu;
                    wind_field_handle_ = phys_->add_force_field(f);
                    wind_field_active_ = true;
                    // Wake every body so the wind takes effect immediately
                    // on a settled pile rather than waiting for the next
                    // collision to wake them up.
                    for (const auto& db : bodies_) {
                        phys_->wake(db.body);
                    }
                }
                wind_dirty_ = false;
            }

            // Time the step on the host side — physics::World doesn't
            // expose its own per-step accessor. high_resolution_clock
            // is a 100 ns tick on Windows-MSVC, plenty of precision
            // for sub-millisecond solver work.
            using clk = std::chrono::high_resolution_clock;
            const auto t0 = clk::now();
            phys_->step(dt);
            const auto t1 = clk::now();
            phys_step_us_ring_.push(
                static_cast<float>(std::chrono::duration_cast<
                    std::chrono::microseconds>(t1 - t0).count()));
            phys_contacts_last_ =
                static_cast<cardinal::u32>(phys_->last_contacts().size());

            auto& scene = e.scene();
            // Advance the pick-highlight envelope once per frame. The
            // pulse phase free-runs (≈1.3 Hz throb); the time budget
            // decays so the gold fades back to base_tint.
            if (pick_highlight_time_ > 0.0f) {
                pick_highlight_time_ =
                    std::max(0.0f, pick_highlight_time_ - dt);
                pick_highlight_pulse_ += dt * 8.0f;
            }
            // Per-body sync: position + orientation pushed into the
            // matching scene::Entity. ID-based lookup (rather than a
            // cached pointer) survives Scene's entity-vector reallocations
            // when the Spawn buttons add more bodies at runtime. Tint is
            // recomputed from base_tint every frame so highlight cleanup
            // is automatic (Reset / Clear-Spawned / re-pick all just
            // work — no per-body restore bookkeeping).
            for (std::size_t i = 0; i < bodies_.size(); ++i) {
                const auto& db = bodies_[i];
                if (db.entity_id == 0) continue;
                auto* en = scene.find_by_id(db.entity_id);
                if (en == nullptr) continue;
                const auto p = phys_->position(db.body);
                en->transform.translation =
                    cardinal::scene::Vec3{p.x, p.y, p.z};
                // Quaternion → Euler each frame — cheap (a handful of
                // trig ops). Demo bodies don't hit gimbal-lock poles in
                // practice; the helper still clamps asin's arg for
                // numerical safety.
                const auto q = phys_->orientation(db.body);
                en->transform.rotation_euler = phys_quat_to_render_euler(q);

                // Highlight blend: gold, sine-throbbed, time-enveloped.
                float blend = 0.0f;
                if (static_cast<int>(i) == pick_highlight_index_ &&
                    pick_highlight_time_ > 0.0f) {
                    const float env =
                        pick_highlight_time_ / kPickHighlightDuration;
                    blend = (0.55f + 0.30f *
                             std::sin(pick_highlight_pulse_)) * env;
                    blend = std::clamp(blend, 0.0f, 1.0f);
                }
                const cardinal::scene::Vec3 hl{1.0f, 0.85f, 0.30f};
                en->tint = cardinal::scene::Vec3{
                    db.base_tint.x + (hl.x - db.base_tint.x) * blend,
                    db.base_tint.y + (hl.y - db.base_tint.y) * blend,
                    db.base_tint.z + (hl.z - db.base_tint.z) * blend};

                // Per-kind PBR material (slot-1 storage buffer). Driven
                // by the live Sky-panel sliders: every sphere reads the
                // chrome (metal, low-roughness) params, every cube the
                // rough-plastic ones, simultaneously in one frame — the
                // metallic-roughness BRDF proven per-entity.
                if (db.kind == DemoKind::Sphere) {
                    en->material.specular_intensity = mat_sphere_intensity_;
                    en->material.roughness          = mat_sphere_roughness_;
                    en->material.metalness          = mat_sphere_metalness_;
                } else {
                    en->material.specular_intensity = mat_cube_intensity_;
                    en->material.roughness          = mat_cube_roughness_;
                    en->material.metalness          = mat_cube_metalness_;
                }
            }
        }

        // ----- Networking replication (SP = MMO one-path proof) ----
        // When enabled, the rendered transforms are driven THROUGH the
        // net stack — the exact Replicator/Transport API a real MMO
        // client uses, here echoed in-proc by the loopback transport.
        //
        // Snapshots broadcast at a THROTTLED rate (net_snap_hz_), not
        // the frame rate, because real netcode does exactly that — and
        // that gap is what makes "snap to newest" visibly steppy. Two
        // client paths, switchable live, make the contrast the payoff:
        //   • snap        — client_ingest: apply newest, freeze between
        //                    snapshots → choppy at low snapshot Hz.
        //   • interpolate  — client_buffer + client_sample at
        //                    (now - delay): lerp between the two
        //                    snapshots bracketing a slightly-past time
        //                    → smooth, packet-loss tolerant.
        // The net proxy is the SOLE driver of the replicated entities
        // while on (it overrides this frame's phys→entity sync), so the
        // freeze-vs-smooth difference is honest, not masked by the
        // local physics still running underneath. Toggle off → entities
        // stay physics-driven (default); base demo undisturbed.
        if (net_replicate_ && repl_ != nullptr && net_ != nullptr) {
            net_time_ += dt;                  // host-supplied monotonic clock
            auto& rscene = e.scene();

            // SERVER: flush queued lifecycle events on the Reliable-
            // Ordered channel (rare, independent of snapshot cadence).
            if (!net_evtq_.empty()) {
                repl_->server_events(net_evtq_);
                net_evtq_.clear();
            }

            // SERVER: gather authoritative transforms + broadcast, but
            // only every snap_interval seconds (throttled cadence).
            const float snap_interval =
                1.0f / std::max(1.0f, net_snap_hz_);
            net_snap_accum_ += dt;
            if (net_snap_accum_ >= snap_interval) {
                // Consume one interval; never let a long frame spiral
                // the accumulator into a broadcast storm.
                net_snap_accum_ =
                    (net_snap_accum_ - snap_interval > snap_interval)
                        ? 0.0f
                        : net_snap_accum_ - snap_interval;
                std::vector<cardinal::net::RepState> tx;
                tx.reserve(bodies_.size());
                for (const auto& db : bodies_) {
                    if (db.entity_id == 0) continue;
                    auto* en = rscene.find_by_id(db.entity_id);
                    if (en == nullptr) continue;
                    cardinal::net::RepState s;
                    s.id             = db.entity_id;
                    s.position       = en->transform.translation;
                    s.rotation_euler = en->transform.rotation_euler;
                    s.scale          = en->transform.scale;
                    tx.push_back(s);
                }
                repl_->server_broadcast(tx);
            }

            // CLIENT: drain the transport every frame regardless of the
            // snapshot cadence.
            net_events_.clear();
            net_->poll(net_events_);

            // CLIENT: decode lifecycle events first. ReliableOrdered →
            // these arrive exactly + in order even with Loss maxed,
            // whereas the snapshot recv counter visibly lags. Despawn
            // tally drives the panel; the held-set prune below uses it.
            std::vector<cardinal::net::RepEvent> evs;
            repl_->client_events(net_events_, evs);
            for (const auto& ev : evs) {
                if (ev.kind == cardinal::net::RepEventKind::Despawn)
                    ++net_despawns_;
                else
                    ++net_spawns_;
            }

            std::vector<cardinal::net::RepState> rx;
            if (net_interp_) {
                repl_->client_buffer(net_events_, net_time_);
                repl_->client_sample(net_time_ - net_interp_delay_, rx);
            } else {
                repl_->client_ingest(net_events_, rx);
            }
            // Hold the last applied set so "snap" freezes (rather than
            // falling back to live physics) between snapshots — that's
            // the steppiness interpolation is there to remove.
            if (!rx.empty()) net_last_ = std::move(rx);
            // Drop despawned ids from the held set so a removed entity
            // can't linger in the snap-hold path (the reliable Despawn
            // is the authority — snapshots just stop mentioning it).
            if (!evs.empty() && !net_last_.empty()) {
                std::vector<cardinal::net::RepState> keep;
                keep.reserve(net_last_.size());
                for (const auto& s : net_last_) {
                    bool dead = false;
                    for (const auto& ev : evs)
                        if (ev.kind ==
                                cardinal::net::RepEventKind::Despawn &&
                            ev.id == s.id) { dead = true; break; }
                    if (!dead) keep.push_back(s);
                }
                net_last_.swap(keep);
            }
            for (const auto& s : net_last_) {
                if (auto* en = rscene.find_by_id(s.id)) {
                    en->transform.translation    = s.position;
                    en->transform.rotation_euler = s.rotation_euler;
                    en->transform.scale          = s.scale;
                }
            }
            net_applied_ = static_cast<cardinal::u32>(net_last_.size());
            net_hist_    = repl_->history_size();
        } else if (!net_evtq_.empty()) {
            // Replication disabled mid-queue: don't strand stale events
            // to flush on a later re-enable.
            net_evtq_.clear();
        }

        // ----- Audio per-frame -------------------------------------
        // Runs unconditionally (sound continues even with physics
        // paused). Listener rides the camera so 3D impacts pan/atten
        // correctly; master channel folds the UI volume/enable. The
        // WASAPI thread pulls Engine::render() asynchronously — tick()
        // just advances play-heads + culls finished instances.
        if (audio_ != nullptr) {
            if (audio_impact_cd_ > 0.0f) audio_impact_cd_ -= dt;
            const auto& cam = e.scene().camera();
            cardinal::scene::Vec3 fwd{
                cam.target.x - cam.position.x,
                cam.target.y - cam.position.y,
                cam.target.z - cam.position.z};
            const float fl = std::sqrt(fwd.x*fwd.x + fwd.y*fwd.y + fwd.z*fwd.z);
            if (fl > 1e-5f) { fwd.x/=fl; fwd.y/=fl; fwd.z/=fl; }
            cardinal::audio::Listener lis;
            lis.position = cam.position;
            lis.forward  = fwd;
            lis.up       = cam.up;
            audio_->set_listener(lis);
            audio_->set_channel_volume(cardinal::audio::kChannelMaster,
                                       audio_enabled_ ? audio_volume_ : 0.0f);
            audio_->tick(dt);
        }

        // Bind our LightSet to whichever pipeline is currently active —
        // cheap pointer-store, but doing it every frame keeps the lights
        // surviving any backend swap (the new pipeline's renderer starts
        // with lights_ == nullptr until set_light_set is called again).
        if (auto* p = e.pipelines().active()) {
            p->set_light_set(&lights_);
        }
    }

    // Camera-ray pick. The StudioMin scene renders to the full window
    // behind the ImGui panels (no viewport-RTT panel), so we work in
    // ImGui main-viewport coordinates: cursor → panel-local pixels →
    // NDC → unproject through inverse(proj*view) → world ray →
    // physics raycast. WantCaptureMouse gates out clicks that ImGui
    // already consumed (a panel/menu), so picking only fires on the
    // "empty" scene area. NDC-Z 0=near, 1=far matches the renderer's
    // add_frustum_edges() convention (cardinal::Mat4::perspective is
    // 0..1 depth).
    void do_viewport_pick(cardinal::engine::Engine& e) {
        ImGuiIO& io = ImGui::GetIO();
        if (!ImGui::IsMouseClicked(ImGuiMouseButton_Left)) return;
        if (io.WantCaptureMouse) return;          // click landed on a panel
        if (phys_ == nullptr) return;

        const ImGuiViewport* vp = ImGui::GetMainViewport();
        const ImVec2 m = ImGui::GetMousePos();
        const float lx = m.x - vp->Pos.x;
        const float ly = m.y - vp->Pos.y;
        const float W  = vp->Size.x, H = vp->Size.y;
        if (W < 2.0f || H < 2.0f) return;
        if (lx < 0.0f || ly < 0.0f || lx > W || ly > H) return;

        const float ndc_x = 2.0f * (lx / W) - 1.0f;
        const float ndc_y = 1.0f - 2.0f * (ly / H);   // screen Y down → NDC Y up

        const auto& cam = e.scene().camera();
        const cardinal::scene::Mat4 inv =
            (cam.proj(W / H) * cam.view()).inverse();
        auto unproject = [&](float z) -> cardinal::physics::Vec3 {
            const cardinal::scene::Vec4 c{ndc_x, ndc_y, z, 1.0f};
            const cardinal::scene::Vec4 w = inv * c;
            const float iw =
                (std::fabs(w.w) > 1e-6f) ? (1.0f / w.w) : 1.0f;
            return cardinal::physics::Vec3{w.x * iw, w.y * iw, w.z * iw};
        };
        const cardinal::physics::Vec3 np = unproject(0.0f);
        const cardinal::physics::Vec3 fp = unproject(1.0f);

        cardinal::physics::Ray ray{};
        ray.origin = np;
        cardinal::physics::Vec3 d{fp.x - np.x, fp.y - np.y, fp.z - np.z};
        const float dl = std::sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
        if (dl > 1e-6f) { d.x /= dl; d.y /= dl; d.z /= dl; }
        ray.direction = d;

        const auto rh = phys_->raycast(ray, 1.0e6f, 0xFFFFFFFFu);
        pick_.valid      = true;
        pick_.hit        = rh.hit;
        pick_.body_index = -1;
        if (rh.hit) {
            pick_.point    = rh.point;
            pick_.distance = rh.t;
            for (std::size_t i = 0; i < bodies_.size(); ++i) {
                if (bodies_[i].body == rh.body) {
                    pick_.body_index = static_cast<int>(i);
                    break;
                }
            }
            // Arm the highlight on the struck body. Full duration so
            // the gold throb is obvious; on_simulate decays it.
            pick_highlight_index_ = pick_.body_index;
            pick_highlight_time_  = kPickHighlightDuration;
            // Poke along the ray so the click reads as a deliberate
            // shove. Skip when impulse is ~0 (inspect-only mode).
            if (pick_impulse_ > 1e-3f) {
                phys_->apply_impulse(rh.body, cardinal::physics::Vec3{
                    ray.direction.x * pick_impulse_,
                    ray.direction.y * pick_impulse_,
                    ray.direction.z * pick_impulse_});
                phys_->wake(rh.body);
            }
        }
    }

    void on_ui(cardinal::ui::StudioEngine& se) override {
        do_viewport_pick(se.engine());
        draw_rhi_panel(se.engine(), sky_, game_, mesh_registry_, &show_rhi_);
        if (show_simd_)    cardinal::ui::draw_simd_panel(&show_simd_);
        if (show_vgeom_)   cardinal::ui::draw_vgeom_panel(&se.engine().scene(), &show_vgeom_);
        if (show_physics_) draw_physics_panel(se.engine());
        if (show_sky_)     draw_sky_panel();
        // Studio's standard panels (Stats, Log, Profiler, etc.) draw
        // themselves through the host's own calls. The minimal sample
        // doesn't open any — Studio's built-in FPS overlay is enough
        // to validate the loop is alive. Hosts that want more would
        // call se.studio().draw_log_panel(...) etc. here.
    }

    // Physics control panel — pause toggle, body reset, runtime spawn
    // buttons, live readout per body. Takes Engine& so the Spawn
    // buttons can reach scene() through spawn_demo_body. Kept inside
    // MinApp so it has direct access to phys_ + bodies_ without piping
    // everything through a free function.
    // Sky control panel — time-of-day clock, freeze toggle, time-scale
    // slider, sky → lights bridge toggle, plus a read-only summary of
    // the current SkyState (sun direction / color / intensity, zenith
    // / horizon, ambient). Mirrors how the gameplay user would tune
    // lighting at runtime.
    void draw_sky_panel() {
        if (!ImGui::Begin("Sky", &show_sky_,
                          ImGuiWindowFlags_AlwaysAutoResize))
        { ImGui::End(); return; }

        // Hour slider — direct mutation of the clock. Float in [0, 24);
        // ImGui's SliderFloat is good enough for a debug panel.
        float h = sky_.hour();
        if (ImGui::SliderFloat("Hour", &h, 0.0f, 24.0f, "%.2f h")) {
            sky_.set_hour(h);
        }
        bool frozen = sky_.frozen();
        if (ImGui::Checkbox("Frozen", &frozen)) {
            sky_.set_frozen(frozen);
        }
        ImGui::SameLine();
        ImGui::Checkbox("Sky → Lights", &sky_track_lights_);
        ImGui::SameLine();
        ImGui::Checkbox("Sky → BG", &sky_track_clear_color_);

        float ts = sky_.time_scale();
        if (ImGui::SliderFloat("time_scale", &ts, 0.0f, 1440.0f, "%.1f")) {
            sky_.set_time_scale(ts);
        }
        ImGui::TextDisabled("  (time_scale=120 → 12 min wall = 24 h game)");
        ImGui::Separator();

        const auto& s = sky_.state();
        const int hh = static_cast<int>(std::floor(s.hour));
        const int mm = static_cast<int>(std::floor((s.hour - hh) * 60.0f));
        ImGui::Text("clock     : %02d:%02d", hh, mm);
        ImGui::Text("sun_dir   : (%.2f, %.2f, %.2f)",
                    s.sun_dir.x, s.sun_dir.y, s.sun_dir.z);
        ImGui::Text("sun_color : (%.2f, %.2f, %.2f)  x %.2f",
                    s.sun_color.x, s.sun_color.y, s.sun_color.z,
                    s.sun_intensity);
        ImGui::Text("zenith    : (%.2f, %.2f, %.2f)",
                    s.zenith.x, s.zenith.y, s.zenith.z);
        ImGui::Text("horizon   : (%.2f, %.2f, %.2f)",
                    s.horizon.x, s.horizon.y, s.horizon.z);
        ImGui::Text("ambient   : (%.2f, %.2f, %.2f)",
                    s.ambient.x, s.ambient.y, s.ambient.z);
        ImGui::Separator();

        // Spot-orbit controls. Independent of the sky → lights bridge:
        // the spot lives in slot 1 (the sky animation only touches the
        // Directional in slot 0). Live readout of the current angle so
        // the user can see the orbit phase.
        ImGui::Text("Spot orbit (slot 1):");
        ImGui::Checkbox("Orbiting", &spot_orbit_enabled_);
        ImGui::DragFloat("radius",  &spot_orbit_radius_, 0.1f, 0.5f, 20.0f, "%.1f m");
        ImGui::DragFloat("height",  &spot_orbit_height_, 0.1f, 0.5f, 20.0f, "%.1f m");
        ImGui::DragFloat("period",  &spot_orbit_period_, 0.5f, 1.0f, 120.0f, "%.1f s");
        const float period_guard =
            (spot_orbit_period_ > 0.01f) ? spot_orbit_period_ : 0.01f;
        const float theta_deg =
            std::fmod(spot_orbit_time_ / period_guard, 1.0f) * 360.0f;
        ImGui::TextDisabled("  phase: %.0f°", theta_deg);

        ImGui::Separator();
        // Extra-lights demo — the visible proof the push-block→SSBO
        // migration lifted the old hard 2-light cap. Toggle on and the
        // count slider piles a rainbow ring of orbiting Point lights
        // into the LightSet (capped at the renderer's kMaxLights=64,
        // minus the 3 base lights).
        ImGui::Text("Extra lights (SSBO proof):");
        ImGui::Checkbox("Ring on", &extra_lights_enabled_);
        ImGui::SliderInt("count", &extra_lights_count_, 0, 60);
        ImGui::TextDisabled("  total lights now: %zu",
                            lights_.lights().size());

        ImGui::Separator();
        // Per-entity materials (slot-1 storage buffer). Drag these and
        // every sphere vs every cube responds independently in the same
        // frame — the proof the per-material path works. Ground keeps
        // its own glossy material (set once in on_init).
        ImGui::Text("PBR materials (SSBO slot 1):");
        ImGui::TextDisabled("  Sphere (metal)");
        ImGui::SliderFloat("S spec",  &mat_sphere_intensity_, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("S rough", &mat_sphere_roughness_, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("S metal", &mat_sphere_metalness_, 0.0f, 1.0f, "%.2f");
        ImGui::TextDisabled("  Cube (dielectric)");
        ImGui::SliderFloat("C spec",  &mat_cube_intensity_, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("C rough", &mat_cube_roughness_, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("C metal", &mat_cube_metalness_, 0.0f, 1.0f, "%.2f");

        ImGui::Separator();
        // HDR exposure → ACES tonemap pre-scale. The bright multi-light
        // ring blows out at 1.0; pull this down to recover highlight
        // detail, push it up for a dim single-light scene. Pushed to
        // the shader via LightSet::set_exposure → material_pad.y.
        ImGui::Text("Tonemap (ACES):");
        if (ImGui::SliderFloat("exposure", &mat_exposure_, 0.05f, 4.0f, "%.2f")) {
            lights_.set_exposure(mat_exposure_);
        }

        ImGui::Separator();
        // Directional shadow map. The shadow-casting light is slot 0 —
        // the same Directional the sky→lights bridge drives — so the
        // cube/sphere shadows sweep across the ground as the sun moves
        // through the day. Off → renderer skips the depth pass + PCF.
        ImGui::Text("Shadows (directional):");
        if (ImGui::Checkbox("Cast shadows", &shadows_enabled_)) {
            lights_.set_shadows_enabled(shadows_enabled_);
        }
        ImGui::TextDisabled("  caster = slot-0 sun (tracks time-of-day)");

        ImGui::End();
    }

    void draw_physics_panel(cardinal::engine::Engine& e) {
        if (!ImGui::Begin("Physics", &show_physics_,
                          ImGuiWindowFlags_AlwaysAutoResize))
        { ImGui::End(); return; }

        if (phys_ == nullptr) {
            ImGui::TextDisabled("World failed to construct on_init.");
            ImGui::End();
            return;
        }

        ImGui::Checkbox("Enabled", &physics_enabled_);
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            // Each body returns to its captured initial state — works
            // identically for the original demo cube + sphere and for
            // anything the user has Spawn-ed. wake() ensures the bodies
            // aren't left asleep at the new position.
            for (const auto& db : bodies_) {
                phys_->teleport(db.body, db.initial_pos,
                                cardinal::physics::Vec3{0, 0, 0});
                phys_->set_orientation(db.body, db.initial_orient);
                phys_->set_angular_velocity(db.body, db.initial_angvel);
                phys_->wake(db.body);
            }
        }
        ImGui::SameLine();
        // Spawn buttons drop a fresh body at a randomized elevated
        // position. Lateral spread fits within the 20×20 m ground
        // plane (±7 m gives a 1.5 m margin to the plane edge so bodies
        // don't tunnel off the side of the world). Cubes spawn with a
        // small spin so the tumble carries the visual style of the
        // initial demo body.
        if (ImGui::Button("Spawn Cube")) {
            std::uniform_real_distribution<float> xz(-7.0f, 7.0f);
            std::uniform_real_distribution<float> y (5.0f, 8.0f);
            std::uniform_real_distribution<float> ax(-2.5f, 2.5f);
            const cardinal::physics::Vec3 pos{xz(rng_), y(rng_), xz(rng_)};
            const cardinal::physics::Vec3 av {ax(rng_), ax(rng_), ax(rng_)};
            spawn_demo_body(e, DemoKind::Cube, pos,
                cardinal::physics::Quat::identity(), av,
                cardinal::scene::Vec3{0.85f, 0.65f, 0.30f});
        }
        ImGui::SameLine();
        if (ImGui::Button("Spawn Sphere")) {
            std::uniform_real_distribution<float> xz(-7.0f, 7.0f);
            std::uniform_real_distribution<float> y (5.0f, 8.0f);
            const cardinal::physics::Vec3 pos{xz(rng_), y(rng_), xz(rng_)};
            spawn_demo_body(e, DemoKind::Sphere, pos,
                cardinal::physics::Quat::identity(),
                cardinal::physics::Vec3{0, 0, 0},
                cardinal::scene::Vec3{0.55f, 0.85f, 0.95f});
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear Spawned")) {
            // Destroy every body past the original two (indices 0 + 1)
            // plus their backing scene entities. Iterate in reverse so
            // remove_entity's index shifts don't invalidate later
            // iterations. bodies_ shrinks back to its on_init size.
            auto& scene = e.scene();
            while (bodies_.size() > 2) {
                const auto& db = bodies_.back();
                phys_->destroy_body(db.body);
                if (db.entity_id != 0) {
                    if (net_ != nullptr && net_replicate_) {
                        cardinal::net::RepEvent ev;
                        ev.kind = cardinal::net::RepEventKind::Despawn;
                        ev.id   = db.entity_id;
                        net_evtq_.push_back(ev);
                    }
                    scene.remove_entity(db.entity_id);
                }
                bodies_.pop_back();
            }
        }
        ImGui::Separator();

        const auto g = phys_->gravity();
        ImGui::Text("gravity   : (%.2f, %.2f, %.2f) m/s²", g.x, g.y, g.z);
        ImGui::Text("bodies    : %zu total (1 static + %zu dynamic)",
                    phys_->body_count(), bodies_.size());
        ImGui::Text("fixed dt  : %.4f s  (= %.0f Hz)",
                    phys_->fixed_timestep(),
                    1.0f / phys_->fixed_timestep());
        ImGui::Text("solver it.: %u", phys_->solver_iterations());
        // Per-frame solver cost + contact count. Wall time is the
        // full step() including the internal substep accumulator
        // (so frames where substep count > 1 will read higher than
        // the underlying single-substep solver work).
        float avg_us = 0.0f, peak_us = 0.0f;
        phys_step_us_ring_.avg_peak(avg_us, peak_us);
        ImGui::Text("step time : %.1f µs avg  /  %.1f µs peak  (60-sample)",
                    avg_us, peak_us);
        // Count sleeping bodies so the user can tell when the pile
        // has come to rest (and the solver should drop near-zero).
        cardinal::u32 sleeping = 0;
        for (const auto& db : bodies_) {
            if (phys_->is_sleeping(db.body)) ++sleeping;
        }
        ImGui::Text("contacts  : %u this step  /  %u sleeping bodies",
                    phys_contacts_last_, sleeping);
        ImGui::Separator();

        // Force fields — currently just one configurable Wind. The dirty
        // flag is set on any control change; on_simulate's pre-step
        // block consumes it to remove + re-add the live ForceField.
        ImGui::Text("Force fields:");
        if (ImGui::Checkbox("Wind", &wind_enabled_)) {
            wind_dirty_ = true;
        }
        if (ImGui::DragFloat3("dir", &wind_dir_.x, 0.05f, -1.0f, 1.0f)) {
            wind_dirty_ = true;
        }
        if (ImGui::DragFloat("strength", &wind_strength_, 0.5f, 0.0f, 50.0f,
                             "%.1f m/s²")) {
            wind_dirty_ = true;
        }
        ImGui::TextDisabled("  active fields: %zu",
                            phys_->force_field_count());
        ImGui::Separator();

        // Click-to-pick — left-click the scene (not a panel) to cast a
        // camera ray. Shows the last result + tunes the poke impulse.
        ImGui::Text("Pick (left-click scene):");
        ImGui::DragFloat("poke impulse", &pick_impulse_, 0.25f, 0.0f, 50.0f,
                         "%.1f N·s");
        if (!pick_.valid) {
            ImGui::TextDisabled("  no click yet");
        } else if (!pick_.hit) {
            ImGui::TextDisabled("  last click: missed (no body)");
        } else {
            const char* kind = "unknown";
            if (pick_.body_index >= 0 &&
                pick_.body_index < static_cast<int>(bodies_.size())) {
                kind = (bodies_[static_cast<cardinal::usize>(
                            pick_.body_index)].kind == DemoKind::Cube)
                       ? "Cube" : "Sphere";
            }
            ImGui::Text("  hit [%d] %s @ (%.2f, %.2f, %.2f)  dist %.2f m",
                        pick_.body_index, kind,
                        pick_.point.x, pick_.point.y, pick_.point.z,
                        pick_.distance);
        }
        ImGui::Separator();

        // Per-body row — truncate to first 8 to keep the panel sane
        // when the user spams Spawn. Summary tail shows the remainder.
        const cardinal::usize show_n =
            std::min<cardinal::usize>(bodies_.size(), 8);
        for (cardinal::usize i = 0; i < show_n; ++i) {
            const auto& db = bodies_[i];
            const auto p   = phys_->position(db.body);
            const auto v   = phys_->velocity(db.body);
            const bool slp = phys_->is_sleeping(db.body);
            const char* k  = (db.kind == DemoKind::Cube) ? "Cube  " : "Sphere";
            ImGui::Text("[%zu] %s  pos (%.2f, %.2f, %.2f)  vel %.2f m/s  %s",
                        i, k, p.x, p.y, p.z,
                        std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z),
                        slp ? "sleeping" : "awake");
        }
        if (bodies_.size() > show_n) {
            ImGui::TextDisabled("  ... + %zu more (truncated)",
                                bodies_.size() - show_n);
        }

        ImGui::Separator();
        // Audio — impacts come from the physics contact callback, so
        // the controls live here. Shows the WASAPI device state + the
        // live mixer instance count.
        ImGui::Text("Audio:");
        ImGui::Checkbox("Enabled##audio", &audio_enabled_);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120.0f);
        ImGui::SliderFloat("volume", &audio_volume_, 0.0f, 1.0f, "%.2f");
        if (audio_ == nullptr) {
            ImGui::TextDisabled("  engine unavailable");
        } else {
            const auto as = audio_->stats();
            ImGui::TextDisabled("  device: %s  |  voices: %u  played: %llu",
                audio_out_ ? "WASAPI" : "none (silent)",
                as.active_instances,
                static_cast<unsigned long long>(as.total_played));
        }

        ImGui::Separator();
        // Networking — replicates the physics bodies through the net
        // stack. ON routes transforms physics → server_broadcast →
        // loopback → client_ingest → render (the exact MMO path; here
        // looped back in-proc, 1-frame delay). Same code is real
        // multiplayer by swapping the transport factory.
        ImGui::Text("Networking (replication):");
        ImGui::Checkbox("Replicate via net path", &net_replicate_);
        if (repl_ == nullptr) {
            ImGui::TextDisabled("  transport unavailable");
        } else {
            // Snapshot cadence + client path. Drop the Hz toward ~5–8
            // and watch "snap" judder while "interpolate" stays smooth;
            // an interp delay of ~1.5–2 snapshot intervals is the sweet
            // spot (too small re-introduces the steppiness).
            ImGui::Checkbox("Interpolate (off = snap to newest)",
                            &net_interp_);
            ImGui::SliderFloat("Snapshot Hz", &net_snap_hz_,
                               2.0f, 60.0f, "%.0f");
            ImGui::SliderFloat("Interp delay", &net_interp_delay_,
                               0.02f, 0.40f, "%.3f s");
            // Adverse-link injector (loopback sim). This is what makes
            // the choices above *provable*: crank Loss/Jitter and the
            // "snap" path visibly hitches while "interpolate" rides
            // through it — the seq-gating + history buffer earning
            // their keep. Time is in net polls (≈ sim frames);
            // deterministic (seeded), so it's a real regression knob.
            ImGui::SliderInt  ("Latency (polls)", &net_lat_, 0, 30);
            ImGui::SliderInt  ("Jitter (+/-)",    &net_jit_, 0, 15);
            ImGui::SliderFloat("Loss",            &net_loss_,
                               0.0f, 0.6f, "%.2f");
            if (net_) {
                cardinal::net::NetConditions nc;
                nc.latency_polls = static_cast<cardinal::u32>(net_lat_);
                nc.jitter_polls  = static_cast<cardinal::u32>(net_jit_);
                nc.loss          = net_loss_;
                net_->set_conditions(nc);
            }
            const float ivl = 1.0f / std::max(1.0f, net_snap_hz_);
            ImGui::TextDisabled(
                "  mode %s | interval %.0f ms | delay %.2f intervals",
                net_interp_ ? "INTERP" : "snap",
                ivl * 1000.0f, net_interp_delay_ / ivl);
            ImGui::TextDisabled(
                "  link: lat %d | jit +/-%d | loss %.0f%%%s",
                net_lat_, net_jit_, net_loss_ * 100.0f,
                (net_lat_ || net_jit_ || net_loss_ > 0.0f)
                    ? "  (loopback sim active)" : "  (perfect link)");
            ImGui::TextDisabled(
                "  loopback (SP=MMO path) | peers %zu | history %zu | applied %u",
                net_ ? net_->peer_count() : (cardinal::usize)0,
                net_hist_, net_applied_);
            ImGui::TextDisabled(
                "  snapshots sent %llu / recv %llu | seq %u",
                static_cast<unsigned long long>(repl_->snapshots_sent()),
                static_cast<unsigned long long>(repl_->snapshots_recv()),
                repl_->last_seq());
            // Real wire size: quantized to 28 B/entity (rot i16 + scale
            // u16, pos still f32) vs 40 B raw — bandwidth at the live
            // snapshot rate. The cost MMO scaling actually pays.
            const cardinal::u32 snap_b = repl_->last_snapshot_bytes();
            ImGui::TextDisabled(
                "  wire %u B/snap (28 B/ent vs 40, -30%%) | ~%.1f KB/s @ %.0f Hz",
                snap_b,
                static_cast<double>(snap_b) * net_snap_hz_ / 1024.0,
                net_snap_hz_);
            // Lifecycle on ReliableOrdered: sent==recv ALWAYS, even at
            // max Loss (where snapshots recv visibly trails sent). The
            // channel split, made tangible — Spawn/Clear with Loss up.
            ImGui::TextDisabled(
                "  lifecycle: +%u spawn / -%u despawn | evt sent %llu / recv %llu",
                net_spawns_, net_despawns_,
                static_cast<unsigned long long>(repl_->events_sent()),
                static_cast<unsigned long long>(repl_->events_recv()));
        }

        ImGui::End();
    }

    void on_menu_bar(cardinal::ui::StudioEngine& /*se*/) override {
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("RHI",     nullptr, &show_rhi_);
            ImGui::MenuItem("SIMD",    nullptr, &show_simd_);
            ImGui::MenuItem("VGeom",   nullptr, &show_vgeom_);
            ImGui::MenuItem("Physics", nullptr, &show_physics_);
            ImGui::MenuItem("Sky",     nullptr, &show_sky_);
            ImGui::EndMenu();
        }
    }

    void on_shutdown(cardinal::engine::Engine& /*e*/) override {
        cardinal::log::infof("studio_min", "shutdown");
    }

private:
    bool show_rhi_{true};
    bool show_simd_{false};       // off-by-default; opened from View menu
    bool show_vgeom_{false};      // off-by-default; opened from View menu
    bool show_physics_{false};    // off-by-default; opened from View menu
    bool show_sky_{false};        // off-by-default; opened from View menu
    // Sky lives on the host (MinApp) rather than the engine — engine
    // doesn't currently expose a Sky accessor, so we own it here and
    // pass it into draw_rhi_panel for the Save/Load Sky buttons.
    // Default-constructed Sky has reasonable mid-day defaults.
    cardinal::sky::Sky sky_;
    // Game owns a SimWorld which owns the actor::World. Same hosting
    // pattern as sky_: engine doesn't expose these, so MinApp owns
    // them. DECLARATION ORDER matters — Game holds a reference to
    // SimWorld, so sim_ must outlive game_; with class-member init
    // that means sim_ is declared first.
    cardinal::sim::SimWorld sim_{};
    cardinal::game::Game    game_{sim_};
    // Mesh-name → shared_ptr<Mesh> registry consumed by serial::load_scene.
    // Empty until the host populates it (no demo entities in StudioMin
    // today). When the host adds named meshes to the scene, mirror them
    // here so Save/Load Scene can round-trip.
    cardinal::serial::MeshRegistry mesh_registry_;
    // Lighting bound to the active pipeline each frame. on_init seeds
    // one warm directional + cool ambient; the renderer's color_for
    // routes through LightSet::shade() when the set is non-empty.
    cardinal::scene::LightSet      lights_;
    // Physics — owns a World containing the ground plane plus a
    // growing list of dynamic demo bodies. bodies_ pairs each
    // physics::Body with the matching scene::Entity (by id, not
    // pointer, so a Scene::add_entity reallocation can't invalidate
    // the mapping when the Spawn buttons fire at runtime).
    // physics_enabled_ gates the per-frame step() so the user can
    // freeze the scene to inspect lighting / gizmos / save-load.
    // box_mesh_ / sphere_mesh_ are cached at on_init time so every
    // spawned cube/sphere shares the same shared_ptr<Mesh> — 100
    // cubes cost 1 mesh.
    std::unique_ptr<cardinal::physics::World> phys_;
    std::vector<DemoBody>                     bodies_;
    std::shared_ptr<cardinal::scene::Mesh>    box_mesh_;
    std::shared_ptr<cardinal::scene::Mesh>    sphere_mesh_;
    std::mt19937                              rng_{42};
    bool                                      physics_enabled_{true};
    // When true, on_simulate copies sky_.state().sun_dir/color/intensity
    // into the LightSet's first Directional each frame + sky.ambient
    // into LightSet.ambient. Switch off to inspect a fixed lighting
    // setup while sky's clock keeps ticking for Save/Load tests.
    bool                                      sky_track_lights_{true};
    // Companion toggle — when true (and sky_track_lights_ is also true),
    // engine.set_clear_color() is driven from sky_.state().horizon each
    // frame so the swapchain bg tints with time of day.
    bool                                      sky_track_clear_color_{true};
    // Spot-orbit demo — Slot-1 (the Spot light) circles the origin
    // each frame when enabled. Independent of sky_track_lights_, which
    // only mutates slot-0 (Directional). Phase accumulator in seconds;
    // 2π is reached at spot_orbit_period_ wall seconds.
    bool                                      spot_orbit_enabled_{true};
    float                                     spot_orbit_time_{0.0f};
    float                                     spot_orbit_radius_{4.0f};
    float                                     spot_orbit_height_{4.0f};
    float                                     spot_orbit_period_{12.0f};
    // Number of fixed "base" lights (Directional+Spot+Point) added in
    // on_init. The Extra-lights ring is appended after these so the
    // fixed-index sun(0)/spot(1) per-frame mutations stay valid.
    cardinal::u32                             base_light_count_{0};
    bool                                      extra_lights_enabled_{false};
    int                                       extra_lights_count_{12};
    float                                     extra_lights_time_{0.0f};
    // Live per-kind PBR material (slot-1 storage buffer). Sphere
    // defaults read polished chrome (metal, near-mirror); cube reads
    // rough plastic (dielectric, broad). Applied every frame to all
    // bodies of that kind in the on_simulate sync loop; Sky-panel
    // sliders drive them so the metallic-roughness BRDF is provable in
    // real time. intensity = spec_scale (overall specular strength).
    float                                     mat_sphere_intensity_{1.0f};
    float                                     mat_sphere_roughness_{0.08f};
    float                                     mat_sphere_metalness_{1.0f};
    float                                     mat_cube_intensity_{0.6f};
    float                                     mat_cube_roughness_{0.85f};
    float                                     mat_cube_metalness_{0.0f};
    // HDR exposure (ACES pre-scale). Default 1.0 matches LightSet's
    // own default so an untouched slider == neutral. Sky-panel slider
    // pushes it via lights_.set_exposure() → material_pad.y.
    float                                     mat_exposure_{1.0f};
    // Directional shadow toggle (Sky panel). Default on; mirrors
    // LightSet::shadows_enabled_ so the renderer's pass-1 gate sees it.
    bool                                      shadows_enabled_{true};
    // Audio: device-agnostic mixer + WASAPI output handle. The
    // physics contact callback plays a 3D "impact" cue; the listener
    // tracks the camera. audio_impact_cd_ throttles the settling-pile
    // contact spam (decremented per frame in on_simulate).
    std::shared_ptr<cardinal::audio::Engine>          audio_;
    std::unique_ptr<cardinal::audio::OutputBackend>   audio_out_;
    float                                     audio_volume_{0.6f};
    bool                                      audio_enabled_{true};
    float                                     audio_impact_cd_{0.0f};
    // Networking: in-proc loopback transport + server-authoritative
    // Replicator. When net_replicate_ is on, the demo routes body
    // transforms through the SAME net stack a real MMO uses (the
    // loopback echoes them, 1-frame delay) — proving the SP = MMO
    // one-netcode-path design end to end. net_events_ reused per
    // frame; net_applied_ = states ingested last frame (panel readout).
    cardinal::unique_ptr<cardinal::net::Transport>   net_;
    cardinal::unique_ptr<cardinal::net::Replicator>  repl_;
    std::vector<cardinal::net::NetEvent>             net_events_;
    bool                                             net_replicate_{false};
    cardinal::u32                                    net_applied_{0};
    // Interpolation demo state. net_time_ is the host-supplied monotonic
    // clock the clock-free Replicator interpolates against (accumulated
    // dt — deterministic, no <chrono>). Snapshots broadcast every
    // 1/net_snap_hz_ s (throttled below the frame rate on purpose);
    // net_interp_ picks the smooth (buffer+sample at now-delay) vs the
    // steppy (ingest newest) client path; net_last_ holds the applied
    // set so "snap" freezes between snapshots; net_hist_ = buffered
    // snapshot depth (panel readout).
    double                                           net_time_{0.0};
    float                                            net_snap_accum_{0.0f};
    float                                            net_snap_hz_{10.0f};
    bool                                             net_interp_{true};
    float                                            net_interp_delay_{0.12f};
    std::vector<cardinal::net::RepState>             net_last_;
    cardinal::usize                                  net_hist_{0};
    // Adverse-link sim knobs pushed to the loopback each frame via
    // Transport::set_conditions — turns the perfect in-proc link into
    // a lossy/jittery one so the interp-vs-snap contrast is provable.
    int                                              net_lat_{0};
    int                                              net_jit_{0};
    float                                            net_loss_{0.0f};
    // Reliable lifecycle channel: server-side outbound queue (flushed
    // via server_events while replicating) + client-side observed
    // spawn/despawn tallies. ReliableOrdered, so these stay exact even
    // with Loss maxed — the channel split made visible next to the
    // lossy snapshot counters.
    std::vector<cardinal::net::RepEvent>             net_evtq_;
    cardinal::u32                                    net_spawns_{0};
    cardinal::u32                                    net_despawns_{0};
    // Rolling 60-sample window for physics::World::step() wall time
    // (microseconds). Same PhaseRing the RHI panel uses for its
    // per-phase render µs — one smoothing horizon across the engine.
    // phys_contacts_last_ holds the most-recent contact event count
    // (cleared at the next step start, so we snapshot it here).
    PhaseRing                                 phys_step_us_ring_{};
    cardinal::u32                             phys_contacts_last_{0};
    // Wind force field — UniformAccel ForceField applied to every
    // dynamic body. Lives behind a UI toggle in the Physics panel so
    // the user can flip it on and watch the cubes slide. The UI sets
    // wind_dirty_ on any control change; on_simulate consumes that flag
    // to remove + re-add the field so the live state always matches
    // the UI. No allocations on idle frames.
    bool                                      wind_enabled_{false};
    cardinal::physics::Vec3                   wind_dir_{1.0f, 0.0f, 0.0f};
    float                                     wind_strength_{5.0f};
    cardinal::physics::World::FieldHandle     wind_field_handle_{};
    bool                                      wind_field_active_{false};
    bool                                      wind_dirty_{false};
    // Click-to-pick — left-click anywhere over the scene (not over an
    // ImGui panel) casts a ray from the camera through the cursor into
    // the physics world. On a hit the body is "poked" with an impulse
    // along the ray so the interaction is immediately visible, and the
    // hit is summarised in the Physics panel. pick_impulse_ scales the
    // poke; 0 = inspect-only (no force).
    struct PickResult {
        bool                    valid{false};   // a click was processed
        bool                    hit{false};     // the ray struck a body
        int                     body_index{-1}; // index into bodies_, or -1
        cardinal::physics::Vec3 point{0, 0, 0};
        float                   distance{0.0f};
    };
    PickResult                                pick_{};
    float                                     pick_impulse_{4.0f};
    // Pick highlight — the last-picked body's entity tint is blended
    // toward a gold highlight for kPickHighlightDuration seconds with a
    // sine throb, then decays back to its base_tint. Index is into
    // bodies_; -1 = nothing highlighted. pulse_ is a free-running phase
    // accumulator so the throb is framerate-independent.
    static constexpr float kPickHighlightDuration = 1.2f;
    int                                       pick_highlight_index_{-1};
    float                                     pick_highlight_time_{0.0f};
    float                                     pick_highlight_pulse_{0.0f};
};

}  // namespace

int main(int argc, char** argv) {
    // Arm the crash reporter FIRST — anything that crashes after this
    // point lands a .dmp + .log in <exe-dir>/crashes/.
    cardinal::crash::install();

    MinApp app;
    return cardinal::ui::StudioEngine::run(argc, argv, app);
}
