// =============================================================================
// Cardinal — Engine Core implementation (pure "wood" layer).
//
// Owns Window, RHI, Scene, Render, Physics, Lua, Plugins, and the frame
// loop. NO Studio, NO ImGui — those live one floor up in cardinal::ui::
// StudioEngine, which wraps this and slots its own pump between
// tick_pre_render() and tick_post_render().
// =============================================================================
#include <cardinal/engine/engine.hpp>

#include <cardinal/core/log.hpp>
#include <cardinal/physics/physics.hpp>
#include <cardinal/plugin/plugin.hpp>
#include <cardinal/render/pipeline.hpp>
#include <cardinal/rhi/rhi.hpp>
#include <cardinal/scene/assets.hpp>
#include <cardinal/scene/fly_camera.hpp>
#include <cardinal/scene/scene.hpp>
#include <cardinal/scene/terrain.hpp>
#include <cardinal/script/engine.hpp>
#include <cardinal/window/window.hpp>

#include <cardinal/core/chrono.hpp>
#include <cardinal/core/cstring.hpp>
#include <cardinal/core/filesystem.hpp>
#include <cardinal/core/thread.hpp>
#include <cardinal/core/utility.hpp>

namespace cardinal::engine {

namespace {

rhi::Backend desc_backend_to_rhi(EngineDesc::Backend b) noexcept {
    switch (b) {
        case EngineDesc::Backend::Vulkan: return rhi::Backend::Vulkan;
        case EngineDesc::Backend::D3D12:  return rhi::Backend::D3D12;
        case EngineDesc::Backend::Auto:   return rhi::Backend::Auto;
    }
    return rhi::Backend::Auto;
}

rhi::Swapchain::ReflexMode desc_reflex_to_rhi(EngineDesc::ReflexMode m) noexcept {
    switch (m) {
        case EngineDesc::ReflexMode::Off:         return rhi::Swapchain::ReflexMode::Off;
        case EngineDesc::ReflexMode::On:          return rhi::Swapchain::ReflexMode::On;
        case EngineDesc::ReflexMode::OnPlusBoost: return rhi::Swapchain::ReflexMode::OnPlusBoost;
    }
    return rhi::Swapchain::ReflexMode::On;
}

EngineDesc::Backend parse_backend_arg(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (cardinal::strncmp(a, "--backend=", 10) != 0) continue;
        const char* val = a + 10;
        if (cardinal::strcmp(val, "vulkan") == 0) return EngineDesc::Backend::Vulkan;
        if (cardinal::strcmp(val, "d3d12")  == 0) return EngineDesc::Backend::D3D12;
        if (cardinal::strcmp(val, "auto")   == 0) return EngineDesc::Backend::Auto;
    }
    return EngineDesc::Backend::Auto;
}

}  // namespace

// =============================================================================
// EngineImpl
// =============================================================================
class EngineImpl final : public Engine {
public:
    ~EngineImpl() override {
        // Release the process-static asset catalog WHILE the rhi::Device is
        // still alive. The catalog's AssetFactory closures capture
        // shared_ptr<Mesh>, and each Mesh owns a device-bound rhi::Buffer.
        // Left to C-runtime static teardown, those buffers free after the
        // device is gone and ~VulkanBuffer dereferences freed VulkanDevice
        // memory — the 0xC0000005 READ access violation seen when the
        // Vulkan-backed editor exits. This destructor body runs before any
        // member (device_ included) is destroyed, so the buffers are torn
        // down against a live device. Symmetric with the
        // register_default_assets() call in initialize().
        scene::AssetCatalog::instance().clear();
    }

    bool initialize(const EngineDesc& desc) {
        desc_ = desc;

        // Window.
        window::WindowDesc wdesc{};
        wdesc.title  = desc.window_title.c_str();
        wdesc.width  = desc.window_width;
        wdesc.height = desc.window_height;
        window_ = window::Window::create(wdesc);
        if (!window_) {
            cardinal::log::errorf("engine", "Window::create failed"); return false;
        }

        // RHI. Boot the runtime-mutable validation flag from the boot
        // descriptor — subsequent set_validation_enabled() calls update
        // it and take effect on the next device re-create.
        validation_enabled_     = desc.rhi_validation;
        rhi::DeviceDesc ddesc{};
        ddesc.backend           = desc_backend_to_rhi(desc.backend);
        ddesc.enable_validation = validation_enabled_;
        device_ = rhi::create_device(ddesc);
        if (!device_) {
            cardinal::log::errorf("engine", "create_device failed"); return false;
        }
        cardinal::log::infof("engine", "RHI: %s [%s]", device_->adapter_name(),
            device_->backend() == rhi::Backend::Vulkan ? "Vulkan" : "D3D12");

        swapchain_ = device_->create_swapchain(
            window_->native_handle(), window_->width(), window_->height());
        if (!swapchain_) {
            cardinal::log::errorf("engine", "create_swapchain failed"); return false;
        }
        swapchain_->set_viewport_size(desc.viewport_width, desc.viewport_height);
        swapchain_->set_vsync_interval(desc.vsync_interval);

        // Asset catalog + terrain profiles (registered before first asset spawn).
        if (desc.with_assets)            scene::register_default_assets(*device_);
        if (desc.with_terrain_profiles)  scene::register_default_terrain_profiles();

        // Scene.
        if (desc.with_scene) scene_ = cardinal::make_unique<scene::Scene>();

        // Pipelines.
        if (desc.with_pipelines) {
            pipelines_ = render::Registry::create(*device_, *swapchain_);
            if (!pipelines_) {
                cardinal::log::errorf("engine", "render::Registry::create failed"); return false;
            }
        }

        // Lua.
        if (desc.with_lua) {
            lua_ = script::Engine::create();
            if (lua_) lua_->run_string("cardinal.log_info('Lua', 'engine ready')");
        }

        // Plugins.
        if (desc.with_plugins) {
            cardinal::fs::path pdir =
                desc.plugin_dir.empty()
                ? (cardinal::fs::current_path() / "plugins")
                : cardinal::fs::path(desc.plugin_dir);
            (void)pdir;  // Registry::load_directory logs the scan result.
            plugin::Registry::instance().load_directory(pdir.string().c_str());
        }

        // Physics.
        if (desc.with_physics) physics_ = physics::World::create();

        // Frame pacer.
        pacer_ = core::FramePacer::create();
        pacer_->set_window(window_->native_handle());
        core::FrameLimit lim{};
        lim.fps_foreground = desc.fps_foreground;
        lim.fps_background = desc.fps_background;
        pacer_->set_limit(core::kPaceMainLoop, lim);

        // Reflex defaults — only meaningful when supported on the adapter.
        if (swapchain_->reflex_supported()) {
            swapchain_->set_reflex_mode(desc_reflex_to_rhi(desc.reflex));
            cardinal::log::infof("engine",
                "Reflex: %s (auto-enabled at boot)",
                desc.reflex == EngineDesc::ReflexMode::Off          ? "Off" :
                desc.reflex == EngineDesc::ReflexMode::OnPlusBoost  ? "On + Boost" : "On");
        }

        // Fly camera (for whichever scene the app populates). The pure
        // engine doesn't pump input into it — UI hosts (StudioEngine) or
        // the host's on_simulate are responsible for that.
        if (scene_) fly_cam_.sync_from(scene_->camera());

        cardinal::log::infof("engine",
            "Engine online — backend=%s, viewport=%ux%u, vsync=%u",
            device_->backend() == rhi::Backend::Vulkan ? "Vulkan" : "D3D12",
            desc.viewport_width, desc.viewport_height, desc.vsync_interval);
        return true;
    }

    int run_loop(Application& app) override {
        app.on_init(*this);
        u64 reached_60 = 0;
        while (!wants_quit_ && !window_->should_close()) {
            tick_one_frame(app);
            if (frame_index_ == 60 && reached_60 == 0) {
                cardinal::log::infof("engine", "Reached frame 60");
                reached_60 = 1;
            }
        }
        app.on_shutdown(*this);
        plugin::Registry::instance().shutdown();
        cardinal::log::infof("engine", "Shutting down (rendered %llu frames)",
                             (unsigned long long)frame_index_);
        return 0;
    }

    bool tick_one_frame(Application& app) override {
        if (!tick_pre_render(app)) return false;
        tick_post_render();
        return true;
    }

    bool tick_pre_render(Application& app) override {
        // 0. Pending backend swap? Process it BEFORE the new frame's reflex
        //    sleep / acquire — the swap rebuilds device + swapchain so any
        //    in-flight begin_frame state from the OLD backend would alias
        //    against the NEW.
        if (swap_pending_) perform_backend_swap();

        // 1. Reflex sleep (driver-level latency-optimal pause)
        swapchain_->reflex_sleep();
        // 2. Pacer sleep (user-set FPS quotas: foreground / background)
        pacer_->wait_for_next_frame();
        // 3. SimulationStart marker
        swapchain_->reflex_marker(rhi::Swapchain::ReflexMarker::SimulationStart, frame_index_);

        // OS event pump.
        window_->poll_events();
        swapchain_->reflex_marker(rhi::Swapchain::ReflexMarker::InputSample, frame_index_);

        if (window_->is_minimized()) {
            using namespace cardinal::chrono_literals;
            cardinal::this_thread::sleep_for(16ms);
            in_frame_ = false;
            return false;       // Skip the rest of the frame
        }
        u32 nw = 0, nh = 0;
        if (window_->consume_resize(&nw, &nh)) {
            if (swapchain_->resize(nw, nh)) {
                if (on_swapchain_resized_) on_swapchain_resized_(*this);
            }
        }

        const f64 dt = timer_.tick();
        last_dt_ = static_cast<f32>(dt);

        // Application sim hook + physics step.
        app.on_simulate(*this, last_dt_);
        if (physics_) physics_->step(last_dt_);

        swapchain_->reflex_marker(rhi::Swapchain::ReflexMarker::SimulationEnd,    frame_index_);
        swapchain_->reflex_marker(rhi::Swapchain::ReflexMarker::RenderSubmitStart, frame_index_);

        // Begin GPU frame + scene render. Clear color is hot-swappable
        // each frame via set_clear_color() — most hosts wire this to a
        // sky-horizon source from on_simulate.
        swapchain_->begin_frame(clear_color_[0], clear_color_[1],
                                clear_color_[2], clear_color_[3]);

        const u32 vw = swapchain_->viewport_width();
        const u32 vh = swapchain_->viewport_height();
        const f32 aspect = (vh > 0) ? float(vw) / float(vh) : 1.0f;
        if (pipelines_ && scene_) pipelines_->render(*scene_, aspect);

        in_frame_ = true;
        return true;
    }

    void tick_post_render() override {
        if (!in_frame_) return;     // pre_render returned false (skipped frame)

        // Plugin tick lives between scene render and swapchain end so
        // plugins can do per-frame work that affects the next frame's
        // simulate (e.g. console commands they injected).
        plugin::Registry::instance().tick(last_dt_);

        swapchain_->reflex_marker(rhi::Swapchain::ReflexMarker::RenderSubmitEnd, frame_index_);
        swapchain_->end_frame();

        ++frame_index_;
        in_frame_ = false;
    }

    void quit() noexcept override        { wants_quit_ = true; }
    bool wants_quit() const noexcept override { return wants_quit_ || (window_ && window_->should_close()); }

    rhi::Device&       device()       override { return *device_; }
    rhi::Swapchain&    swapchain()    override { return *swapchain_; }
    window::Window&    window()       override { return *window_; }
    scene::Scene&      scene()        override { return *scene_; }
    scene::FlyCamera&  fly_camera()   override { return fly_cam_; }
    render::Registry&  pipelines()    override { return *pipelines_; }
    script::Engine*    lua()          override { return lua_.get(); }
    physics::World*    physics()      override { return physics_.get(); }
    core::FramePacer&  pacer()        override { return *pacer_; }
    core::FrameTimer&  frame_timer()  override { return timer_; }

    u32  selected_entity() const noexcept override { return selected_id_; }
    void set_selected_entity(u32 id) noexcept override { selected_id_ = id; }
    u64  frame_index()     const noexcept override { return frame_index_; }

    void set_on_swapchain_resized(OnSwapchainResized cb) override {
        on_swapchain_resized_ = cardinal::move(cb);
    }

    // ---- Backend hot-swap orchestration ---------------------------------

    void set_backend_swap_hooks(OnBeforeBackendSwap before,
                                OnAfterBackendSwap  after) override {
        on_before_swap_ = cardinal::move(before);
        on_after_swap_  = cardinal::move(after);
    }

    bool request_backend_swap(EngineDesc::Backend target) override {
        const auto current = active_backend();
        if (target == current) return false;        // already there
        if (swap_pending_)     return false;        // one in flight
        // Resolve Auto to whatever the OPPOSITE of the current backend is.
        if (target == EngineDesc::Backend::Auto) {
            target = (current == EngineDesc::Backend::Vulkan)
                ? EngineDesc::Backend::D3D12
                : EngineDesc::Backend::Vulkan;
        }
        swap_target_   = target;
        swap_pending_  = true;
        cardinal::log::infof("engine",
            "Backend swap scheduled: %s -> %s",
            backend_name(current), backend_name(target));
        return true;
    }

    EngineDesc::Backend active_backend() const noexcept override {
        if (device_ == nullptr) return EngineDesc::Backend::Auto;
        return device_->backend() == rhi::Backend::Vulkan
            ? EngineDesc::Backend::Vulkan
            : EngineDesc::Backend::D3D12;
    }

    bool backend_swap_pending() const noexcept override { return swap_pending_; }

    bool validation_enabled() const noexcept override { return validation_enabled_; }
    void set_validation_enabled(bool enabled) override { validation_enabled_ = enabled; }

    void set_clear_color(f32 r, f32 g, f32 b, f32 a) noexcept override {
        clear_color_[0] = r; clear_color_[1] = g;
        clear_color_[2] = b; clear_color_[3] = a;
    }
    void clear_color(f32 out_rgba[4]) const noexcept override {
        out_rgba[0] = clear_color_[0]; out_rgba[1] = clear_color_[1];
        out_rgba[2] = clear_color_[2]; out_rgba[3] = clear_color_[3];
    }

    bool request_device_recreate() override {
        if (swap_pending_) return false;        // one in flight
        // Same backend, but the swap pipeline rebuilds the device with
        // the current validation_enabled_ and any other settings the
        // panel has tweaked. Reuses the same hook chain so Studio
        // re-binds its overlay transparently.
        swap_target_  = active_backend();
        swap_pending_ = true;
        cardinal::log::infof("engine",
            "Device re-create scheduled on %s (settings change; "
            "validation=%s)",
            backend_name(swap_target_),
            validation_enabled_ ? "on" : "off");
        return true;
    }

private:
    static const char* backend_name(EngineDesc::Backend b) noexcept {
        switch (b) {
            case EngineDesc::Backend::Vulkan: return "Vulkan";
            case EngineDesc::Backend::D3D12:  return "D3D12";
            case EngineDesc::Backend::Auto:   return "Auto";
        }
        return "?";
    }

    // The actual backend swap. Runs at the top of tick_pre_render so no
    // in-flight GPU work bridges the device boundary.
    void perform_backend_swap() {
        const auto from = active_backend();
        const auto to   = swap_target_;
        swap_pending_   = false;

        cardinal::log::infof("engine",
            "Backend swap executing: %s -> %s",
            backend_name(from), backend_name(to));

        // Step 1: host releases app-owned + UI-owned GPU resources. The UI
        // host (StudioEngine) chains under this hook so Studio gets a
        // chance to drop its overlay before the device dies.
        if (on_before_swap_) on_before_swap_(*this);

        // Step 2: drop pipelines (releases backend-owned PSO/shader objs).
        pipelines_.reset();

        // Step 3: tear down old swapchain + device.
        swapchain_.reset();
        device_.reset();

        // Step 4: create new device on the target backend. Validation
        // flag comes from the runtime-mutable validation_enabled_ —
        // not from the original boot descriptor — so set_validation_
        // enabled() takes effect here.
        rhi::DeviceDesc ddesc{};
        ddesc.backend           = (to == EngineDesc::Backend::Vulkan)
            ? rhi::Backend::Vulkan : rhi::Backend::D3D12;
        ddesc.enable_validation = validation_enabled_;
        device_ = rhi::create_device(ddesc);
        if (!device_) {
            cardinal::log::errorf("engine",
                "Backend swap FAILED: create_device(%s) returned null. "
                "Engine is in an unrecoverable state — quitting.",
                backend_name(to));
            wants_quit_ = true;
            return;
        }
        cardinal::log::infof("engine", "Swap: new device = %s [%s]",
            device_->adapter_name(), backend_name(to));

        // Step 5: create new swapchain + restore viewport size + vsync.
        swapchain_ = device_->create_swapchain(
            window_->native_handle(), window_->width(), window_->height());
        if (!swapchain_) {
            cardinal::log::errorf("engine",
                "Backend swap FAILED: create_swapchain on %s returned null. Quitting.",
                backend_name(to));
            wants_quit_ = true;
            return;
        }
        swapchain_->set_viewport_size(desc_.viewport_width, desc_.viewport_height);
        swapchain_->set_vsync_interval(desc_.vsync_interval);

        // Step 6: pipeline registry rebuilt against the new device.
        if (desc_.with_pipelines) {
            pipelines_ = render::Registry::create(*device_, *swapchain_);
            if (!pipelines_) {
                cardinal::log::errorf("engine",
                    "Backend swap FAILED: render::Registry::create on %s returned null.",
                    backend_name(to));
                wants_quit_ = true;
                return;
            }
        }

        // Step 7: host re-uploads app-owned + UI-owned GPU resources. The
        // UI host re-binds Studio onto the new device here.
        if (on_after_swap_) on_after_swap_(*this);

        cardinal::log::infof("engine",
            "Backend swap complete — running on %s", backend_name(to));
    }

    OnBeforeBackendSwap on_before_swap_{};
    OnAfterBackendSwap  on_after_swap_{};
    OnSwapchainResized  on_swapchain_resized_{};
    bool                swap_pending_{false};
    EngineDesc::Backend swap_target_{EngineDesc::Backend::Auto};
    // Pending validation flag — applied to the next rhi::create_device.
    // Initialised from desc_.rhi_validation in initialize() and mutated
    // via set_validation_enabled(); read by perform_backend_swap when
    // building DeviceDesc for the new device.
    bool                validation_enabled_{false};

    EngineDesc                          desc_{};
    cardinal::unique_ptr<window::Window>     window_;
    cardinal::unique_ptr<rhi::Device>        device_;
    cardinal::unique_ptr<rhi::Swapchain>     swapchain_;
    cardinal::unique_ptr<scene::Scene>       scene_;
    cardinal::unique_ptr<render::Registry>   pipelines_;
    cardinal::unique_ptr<script::Engine>     lua_;
    cardinal::unique_ptr<physics::World>     physics_;
    cardinal::unique_ptr<core::FramePacer>   pacer_;

    scene::FlyCamera                    fly_cam_;
    core::FrameTimer                    timer_;
    bool                                wants_quit_{false};
    bool                                in_frame_{false};
    f32                                 last_dt_{0.0f};
    u32                                 selected_id_{0};
    u64                                 frame_index_{1};
    // Swapchain clear-color. Default is the original hardcoded
    // dark blue-grey. Hosts override via set_clear_color() — typically
    // each frame in on_simulate from sky/horizon state.
    f32                                 clear_color_[4]{0.07f, 0.08f, 0.10f, 1.0f};
};

// ----- Public factory + run() ----------------------------------------------
cardinal::unique_ptr<Engine> Engine::create(const EngineDesc& desc) {
    auto e = cardinal::make_unique<EngineImpl>();
    if (!e->initialize(desc)) return nullptr;
    return e;
}

int Engine::run(int argc, char** argv, Application& app) {
    EngineDesc desc{};
    desc.backend = parse_backend_arg(argc, argv);
    auto eng = create(desc);
    if (!eng) return 1;
    return eng->run_loop(app);
}

}  // namespace cardinal::engine
