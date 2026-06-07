// =============================================================================
// Cardinal — StudioEngine implementation (the "roof" layer).
//
// Wraps a cardinal::engine::Engine with Studio integration:
//   - ImGui dockspace + default panel layout
//   - Menu bar (with File→Quit + on_menu_bar passthrough)
//   - FlyCamera input pumped from ImGui state
//   - Per-viewport FPS quota slots (fed into the engine's FramePacer)
//   - Studio create / begin_frame / end_frame / update_platform_windows
//   - Backend-swap chaining (Studio re-binds to the new device after a swap)
// =============================================================================
#include <cardinal/ui/studio_engine.hpp>
#include <cardinal/ui/studio.hpp>

#include <cardinal/core/sync/frame_pacer.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/plugin/plugin.hpp>
#include <cardinal/rhi/rhi.hpp>
#include <cardinal/scene/fly_camera.hpp>
#include <cardinal/scene/scene.hpp>
#include <cardinal/window/window.hpp>

#include <cardinal/ui/imgui.hpp>
#include <imgui_internal.h>

#include <cardinal/core/std/cstring.hpp>
#include <cardinal/core/std/utility.hpp>

namespace cardinal::ui {

namespace {

void build_default_dock_layout(ImGuiID dockspace_id, const ImVec2& size) {
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, size);

    ImGuiID center = dockspace_id;
    ImGuiID left   = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left,  0.18f, nullptr, &center);
    ImGuiID right  = ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.24f, nullptr, &center);
    ImGuiID bottom = ImGui::DockBuilderSplitNode(center, ImGuiDir_Down,  0.30f, nullptr, &center);

    ImGui::DockBuilderDockWindow("Viewport 1",      center);
    ImGui::DockBuilderDockWindow("Viewport 2",      center);
    ImGui::DockBuilderDockWindow("Viewport 3",      center);
    ImGui::DockBuilderDockWindow("Hierarchy",       left);
    ImGui::DockBuilderDockWindow("Assets",          left);
    ImGui::DockBuilderDockWindow("Asset Palette",   left);
    ImGui::DockBuilderDockWindow("Inspector",       right);
    ImGui::DockBuilderDockWindow("Stats",           right);
    ImGui::DockBuilderDockWindow("Render Pipeline", right);
    ImGui::DockBuilderDockWindow("Log",             bottom);
    ImGui::DockBuilderDockWindow("Console",         bottom);
    ImGui::DockBuilderDockWindow("Stack Tracer",    bottom);
    ImGui::DockBuilderDockWindow("Function Tracer", bottom);
    ImGui::DockBuilderDockWindow("Script Debugger", bottom);
    ImGui::DockBuilderFinish(dockspace_id);
}

void draw_dockspace(bool& first_time) {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::SetNextWindowViewport(vp->ID);

    constexpr ImGuiWindowFlags host_flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize   | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoDocking  | ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_MenuBar;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin("##cardinal_dockhost", nullptr, host_flags);
    ImGui::PopStyleVar(3);

    const ImGuiID dock_id = ImGui::GetID("CardinalDockSpace");
    if (first_time) {
        first_time = false;
        if (ImGui::DockBuilderGetNode(dock_id) == nullptr) {
            build_default_dock_layout(dock_id, vp->WorkSize);
        }
    }
    ImGui::DockSpace(dock_id, ImVec2(0, 0), ImGuiDockNodeFlags_PassthruCentralNode);
}

}  // namespace

// =============================================================================
// StudioEngineImpl
// =============================================================================
class StudioEngineImpl final : public StudioEngine {
public:
    bool initialize(const cardinal::engine::EngineDesc& desc) {
        // Build the wood-layer engine. We let it own all the lifetime-tracked
        // subsystems (window, device, swapchain, scene, pipelines, lua, …).
        engine_ = cardinal::engine::Engine::create(desc);
        if (!engine_) {
            cardinal::log::errorf("studio_engine", "engine::Engine::create failed");
            return false;
        }

        // Studio sits on top of the engine's RHI + window. Studio owns its
        // own ImGui context, swapchain overlay registration, and panel state.
        studio_ = Studio::create(engine_->device(), engine_->swapchain(),
                                 engine_->window());
        if (!studio_) {
            cardinal::log::errorf("studio_engine", "Studio::create failed");
            return false;
        }

        // Hook the engine's swapchain-resize signal so Studio rebuilds its
        // per-image framebuffers without us having to plumb OS events here.
        // The hook target is an internal API on EngineImpl; it's safe to
        // cast because we just constructed it via Engine::create().
        wire_engine_callbacks_();

        // Per-viewport FPS state defaults.
        for (auto& f : vp_fps_) f = 0.0f;

        return true;
    }

    int run_loop(StudioApplication& app) override {
        app.on_init(*engine_);
        u64 reached_60 = 0;
        while (!engine_->wants_quit() && !engine_->window().should_close()) {
            tick_one_frame(app);
            if (engine_->frame_index() == 60 && reached_60 == 0) {
                cardinal::log::infof("studio_engine", "Reached frame 60");
                reached_60 = 1;
            }
        }
        app.on_shutdown(*engine_);
        plugin::Registry::instance().shutdown();
        cardinal::log::infof("studio_engine",
            "Shutting down (rendered %llu frames)",
            (unsigned long long)engine_->frame_index());
        return 0;
    }

    bool tick_one_frame(StudioApplication& app) override {
        // Phase 1 — pump OS, simulate, render scene.
        if (!engine_->tick_pre_render(app)) return false;

        // Phase 2 — Studio frame: dockspace, menu bar, app's on_ui, FPS overlay.
        studio_->begin_frame();
        draw_dockspace(first_layout_);

        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Quit")) engine_->quit();
                ImGui::EndMenu();
            }
            app.on_menu_bar(*this);
            ImGui::EndMenuBar();
        }
        ImGui::End();   // ##cardinal_dockhost

        // Fly-camera input — driven by ImGui state because Studio grabs
        // the keyboard/mouse first. Headless apps drive the FlyCamera
        // through the input system instead.
        pump_fly_camera_(static_cast<f32>(engine_->frame_timer().dt_seconds()));

        // App's own UI (panels, custom menus, gizmos, etc.).
        app.on_ui(*this);

        // Push per-viewport FPS quotas into the pacer.
        for (u32 i = 0; i < kViewports; ++i) {
            core::FrameLimit fl{};
            fl.fps_foreground = vp_fps_[i];
            fl.fps_background = 0.0f;
            engine_->pacer().set_limit(1u + i, fl);
        }

        studio_->draw_fps_overlay();
        studio_->end_frame();

        // Phase 3 — present + plugin tick + frame counter.
        engine_->tick_post_render();

        // Multi-viewport dock-out windows (after present so platform
        // windows show this frame's contents).
        studio_->update_platform_windows();
        return true;
    }

    cardinal::engine::Engine& engine() override { return *engine_; }
    Studio&                   studio() override { return *studio_; }

    void quit() noexcept override        { engine_->quit(); }
    bool wants_quit() const noexcept override { return engine_->wants_quit(); }

    f32* viewport_fps_slot(u32 idx) noexcept override {
        return idx < kViewports ? &vp_fps_[idx] : nullptr;
    }

private:
    void wire_engine_callbacks_() {
        // External resize path: window message → engine consumes resize →
        // engine calls swapchain->resize() → engine fires the hook below.
        // Studio rebuilds its per-image framebuffers (Vulkan) / no-op
        // (D3D12).
        engine_->set_on_swapchain_resized(
            [this](cardinal::engine::Engine& /*e*/) {
                studio_->on_swapchain_resized();
            });

        // Internal rebuild path: swapchain itself tears down + rebuilds
        // (vsync change, future hot-swap). The engine doesn't observe
        // these — they happen inside swapchain->begin_frame. Hook the
        // swapchain's own on_rebuilt callback so Studio's framebuffers
        // get re-created before the next overlay record references the
        // freshly-allocated image views. Without this, changing vsync
        // mode at runtime crashed the process: Studio's VkFramebuffers
        // pointed at views that destroy_swapchain_objects() had freed.
        engine_->swapchain().set_on_rebuilt(
            [this]() { studio_->on_swapchain_resized(); });

        // Backend swap: Studio re-binds its ImGui backend + descriptor
        // heaps onto the new device. We intentionally do NOT touch
        // Studio in the "before" hook — its own reinit() handles old-
        // device teardown, and we want to preserve ImGui context, tool
        // windows, console state, and the log sink across the swap.
        engine_->set_backend_swap_hooks(
            [this](cardinal::engine::Engine& /*e*/) { /* no-op */ },
            [this](cardinal::engine::Engine& e) {
                if (!studio_->reinit(e.device(), e.swapchain())) {
                    cardinal::log::errorf("studio_engine",
                        "Backend swap FAILED: Studio::reinit returned false.");
                    e.quit();
                }
            });
    }

    void pump_fly_camera_(f32 dt) {
        scene::FlyInput in{};
        const ImGuiIO& io = ImGui::GetIO();
        const ImVec2 mp   = io.MousePos;
        in.accept_input = !ImGui::IsAnyItemActive()
                       && ImGui::IsMouseHoveringRect(ImVec2(0,0), io.DisplaySize, false)
                       && !io.WantCaptureKeyboard;
        in.look = io.MouseDown[1];
        if (in.look) {
            in.mouse_dx = mp.x - last_mouse_.x;
            in.mouse_dy = mp.y - last_mouse_.y;
            in.accept_input = true;
        }
        last_mouse_ = mp;
        in.forward  = ImGui::IsKeyDown(ImGuiKey_W);
        in.backward = ImGui::IsKeyDown(ImGuiKey_S);
        in.left     = ImGui::IsKeyDown(ImGuiKey_A);
        in.right    = ImGui::IsKeyDown(ImGuiKey_D);
        in.up       = ImGui::IsKeyDown(ImGuiKey_E);
        in.down     = ImGui::IsKeyDown(ImGuiKey_Q);
        in.sprint   = ImGui::IsKeyDown(ImGuiKey_LeftShift) ||
                      ImGui::IsKeyDown(ImGuiKey_RightShift);
        in.scroll   = io.MouseWheel;
        engine_->fly_camera().tick(engine_->scene().camera(), in, dt);
    }

    cardinal::unique_ptr<cardinal::engine::Engine> engine_;
    cardinal::unique_ptr<Studio>                   studio_;
    bool                                      first_layout_{true};
    ImVec2                                    last_mouse_{0, 0};
    f32                                       vp_fps_[kViewports]{0.0f, 0.0f, 0.0f};
};

// ----- Public factory + run() ----------------------------------------------
cardinal::unique_ptr<StudioEngine> StudioEngine::create(
    const cardinal::engine::EngineDesc& desc)
{
    auto e = cardinal::make_unique<StudioEngineImpl>();
    if (!e->initialize(desc)) return nullptr;
    return e;
}

int StudioEngine::run(int argc, char** argv, StudioApplication& app) {
    cardinal::engine::EngineDesc desc{};
    // Reuse the engine's --backend= parser by going through Engine::run
    // semantics here. Inline rather than duplicate the parser.
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (cardinal::strncmp(a, "--backend=", 10) != 0) continue;
        const char* val = a + 10;
        if (cardinal::strcmp(val, "vulkan") == 0)
            desc.backend = cardinal::engine::EngineDesc::Backend::Vulkan;
        else if (cardinal::strcmp(val, "d3d12") == 0)
            desc.backend = cardinal::engine::EngineDesc::Backend::D3D12;
        else if (cardinal::strcmp(val, "auto") == 0)
            desc.backend = cardinal::engine::EngineDesc::Backend::Auto;
    }
    auto se = create(desc);
    if (!se) return 1;
    return se->run_loop(app);
}

}  // namespace cardinal::ui
