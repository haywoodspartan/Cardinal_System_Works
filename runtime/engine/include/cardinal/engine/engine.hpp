#pragma once

// =============================================================================
// Cardinal — Engine Core (the "wood" layer).
//
// Layering model — concrete / wood / roof:
//   cardinal::core           ← concrete foundation (types, math, jobs, hal,
//                              io, time, log, framepacer, …). Everything
//                              builds on this.
//   cardinal::engine         ← the wood. Window, RHI, Scene, Render, Physics,
//                              Lua, Plugins, frame loop. Pure runtime — no
//                              editor / UI dependency.
//   cardinal::ui::StudioEngine ← the roof. Wraps an Engine and adds Studio
//                              (ImGui dockspace, menu bar, panel draw,
//                              FlyCamera input). Lives in runtime/ui because
//                              it depends on Studio.
//
// Engine deliberately does NOT include or link ImGui or cardinal::ui. A
// shipped game links cardinal::engine and never pays for the editor; an
// editor-flavoured host links cardinal::ui::StudioEngine which transitively
// brings the Engine.
//
// Typical headless / shipped-game host:
//
//     class MyApp : public cardinal::engine::Application {
//         void on_init     (Engine&) override { ... spawn entities ... }
//         void on_simulate (Engine&, f32 dt) override { ... }
//         void on_shutdown (Engine&) override { ... }
//     };
//
//     int main(int argc, char** argv) {
//         MyApp app;
//         return cardinal::engine::Engine::run(argc, argv, app);
//     }
//
// Editor-flavoured host:  use cardinal::ui::StudioEngine instead.
// =============================================================================

#include <cardinal/core/frame_pacer.hpp>
#include <cardinal/core/time.hpp>
#include <cardinal/core/types.hpp>

#include <functional>
#include <memory>
#include <string>

namespace cardinal::rhi      { class Device; class Swapchain; }
namespace cardinal::window   { class Window; }
namespace cardinal::scene    { class Scene; class FlyCamera; }
namespace cardinal::render   { class Registry; }
namespace cardinal::script   { class Engine; }
namespace cardinal::physics  { class World; }

namespace cardinal::engine {

// Boot configuration. Pass to Engine::create / Engine::run.
struct EngineDesc {
    // Window.
    std::string window_title{"Cardinal"};
    u32         window_width{1600};
    u32         window_height{900};

    // RHI.
    enum class Backend : u32 { Auto, Vulkan, D3D12 };
    Backend     backend{Backend::Auto};
    bool        rhi_validation{false};

    // Off-screen viewport size (StudioEngine grows this each frame; pure
    // headless hosts can leave it at the default).
    u32         viewport_width{1280};
    u32         viewport_height{720};

    // VSync interval (0 = uncapped + DXGI tearing). FramePacer caps below.
    u32         vsync_interval{0};

    // FramePacer foreground / background quotas (FPS; 0 = uncapped).
    f32         fps_foreground{0.0f};
    f32         fps_background{30.0f};

    // Reflex (D3D12 only on supported NVIDIA hardware). Off / On / OnPlusBoost.
    enum class ReflexMode : u32 { Off, On, OnPlusBoost };
    ReflexMode  reflex{ReflexMode::On};

    // Subsystem opt-outs — for headless tools (Lua-only, physics-only, …).
    bool        with_pipelines      {true};
    bool        with_scene          {true};
    bool        with_lua            {true};
    bool        with_physics        {true};
    bool        with_plugins        {true};
    bool        with_assets         {true};
    bool        with_terrain_profiles{true};

    // Plugin search dir; empty → <exe>/plugins/.
    std::string plugin_dir{};
};

// Application interface — override to plug in app-specific logic. Every
// hook is optional; default impls are no-ops.
//
// NOTE: this is the *headless* application interface. Editor-flavoured
// applications subclass cardinal::ui::StudioApplication instead, which
// adds on_ui() / on_menu_bar() hooks and keeps everything below.
class Application;
class Engine;

class Application {
public:
    virtual ~Application() = default;

    // Called once after subsystems are up + before the first frame.
    virtual void on_init(Engine&) {}

    // Called every frame — game state updates that aren't physics-stepped
    // (e.g. AI ticks, audio cues). dt is the wall-clock delta capped by
    // FrameTimer (default 250 ms).
    virtual void on_simulate(Engine&, f32 /*dt*/) {}

    // Called once after the last frame, before subsystems tear down.
    virtual void on_shutdown(Engine&) {}
};

// Engine — owns + drives every subsystem.
class Engine {
public:
    // Convenience top-level: parses --backend= from argv, creates the
    // engine, runs until quit, returns process exit code.
    static int run(int argc, char** argv, Application& app);

    // Lower-level: hand-rolled boot. Returns nullptr on failure.
    static std::unique_ptr<Engine> create(const EngineDesc& desc);
    virtual ~Engine() = default;
    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    // Run the loop with an Application until quit().
    virtual int  run_loop(Application& app) = 0;

    // ----- Three-phase frame tick ---------------------------------------
    //
    // The pure Engine ticks in three discrete phases so a UI host (e.g.
    // StudioEngine) can interleave its own work between the GPU begin_frame
    // and end_frame:
    //
    //   tick_pre_render(app)  : OS pump, reflex sleep, FramePacer wait,
    //                           on_simulate, physics step, swapchain
    //                           begin_frame, render scene to viewport.
    //                           Returns false when the engine wants to
    //                           exit (window closed / quit() called); true
    //                           on a normal-or-skipped frame.
    //
    //   tick_post_render()    : swapchain end_frame, present, plugin tick,
    //                           frame_index++.
    //
    // Headless apps just call run_loop() and never see these.
    // UI hosts call: tick_pre_render(); /* ImGui + Studio + panels */;
    //                tick_post_render();
    virtual bool tick_pre_render(Application& app) = 0;
    virtual void tick_post_render() = 0;

    // Manual single-frame pump (for hosts that just want a flat tick).
    // Equivalent to tick_pre_render(app) followed by tick_post_render().
    virtual bool tick_one_frame(Application& app) = 0;

    virtual void quit() noexcept = 0;
    virtual bool wants_quit() const noexcept = 0;

    // Subsystem accessors. Asserts when the corresponding subsystem
    // was disabled in EngineDesc.
    virtual rhi::Device&         device()       = 0;
    virtual rhi::Swapchain&      swapchain()    = 0;
    virtual window::Window&      window()       = 0;
    virtual scene::Scene&        scene()        = 0;
    virtual scene::FlyCamera&    fly_camera()   = 0;
    virtual render::Registry&    pipelines()    = 0;
    virtual script::Engine*      lua()          = 0;     // null if disabled
    virtual physics::World*      physics()      = 0;     // null if disabled
    virtual core::FramePacer&    pacer()        = 0;
    virtual core::FrameTimer&    frame_timer()  = 0;

    // Selection helpers — Studio + viewport interactions write into this,
    // app code reads it from on_ui / on_simulate.
    virtual u32  selected_entity() const noexcept = 0;
    virtual void set_selected_entity(u32 id) noexcept = 0;

    // Frame counters (drive Reflex frame-id + analytics).
    virtual u64  frame_index() const noexcept = 0;

    // ---- Runtime backend hot-swap (DX12 ↔ Vulkan) -----------------------
    //
    // Tear down every backend-bound resource (swapchain, device, render
    // pipelines), recreate them on the requested backend, then call back
    // into the host's resource-rebuild hooks so app-owned GPU state (scene
    // meshes, custom textures, async-compute buffers, etc.) gets re-uploaded
    // to the new device.
    //
    // The swap blocks for the duration — typically a frame or two — and is
    // safe to call ONLY between frames. Use `request_backend_swap` to
    // schedule it onto the next safe boundary.
    //
    // The host registers callbacks via set_backend_swap_hooks(). Both are
    // optional. The UI host (StudioEngine) chains its own swap hooks under
    // these so Studio gets re-bound to the new device transparently.
    using OnBeforeBackendSwap = std::function<void(Engine&)>;
    using OnAfterBackendSwap  = std::function<void(Engine&)>;
    virtual void set_backend_swap_hooks(OnBeforeBackendSwap before,
                                        OnAfterBackendSwap  after) = 0;

    // Schedule a backend swap for the next safe boundary. Returns false if
    // the backend is identical to the active one (no-op) or if a swap is
    // already pending.
    virtual bool request_backend_swap(EngineDesc::Backend target) = 0;

    // Read-only — what backend is the engine currently rendering on.
    virtual EngineDesc::Backend active_backend() const noexcept = 0;

    // True between request_backend_swap() and the actual swap completing.
    virtual bool backend_swap_pending() const noexcept = 0;

    // Runtime-mutable validation flag — the value that will be passed
    // to rhi::create_device on the next swap. The live device cannot
    // toggle validation; both Vulkan validation layers and D3D12
    // debug-layer attach at create_instance / create_device time.
    // Pair with request_device_recreate() to apply immediately.
    virtual bool validation_enabled() const noexcept = 0;
    virtual void set_validation_enabled(bool enabled) = 0;

    // Schedule a device re-creation on the SAME backend. Used to apply
    // settings that need device-level re-init (validation flag today;
    // per-device feature gates in future). Reuses the same hook path
    // as request_backend_swap (on_before_swap → drop → create →
    // on_after_swap). Returns false if a swap is already pending.
    virtual bool request_device_recreate() = 0;

    // ---- UI host integration --------------------------------------------
    //
    // Notify the host that the swapchain just resized (after window
    // resize → swapchain->resize). UI layers like StudioEngine wire this
    // to rebuild per-image framebuffers / overlay state. The hook fires
    // AFTER the swapchain has accepted the new size.
    using OnSwapchainResized = std::function<void(Engine&)>;
    virtual void set_on_swapchain_resized(OnSwapchainResized cb) = 0;

    // ---- Clear color ----------------------------------------------------
    // Color the swapchain clears to each frame, before any pipeline
    // renders. Default {0.07, 0.08, 0.10, 1.0} (dark blue-grey).
    // Hosts that want a sky-driven background call this from on_simulate
    // each tick. Stored as an internal float[4] — applied at the next
    // begin_frame() boundary, so a mid-frame change still takes effect
    // on the NEXT frame, not the current one.
    virtual void set_clear_color(f32 r, f32 g, f32 b, f32 a) noexcept = 0;
    virtual void clear_color(f32 out_rgba[4]) const noexcept       = 0;

protected:
    Engine() = default;
};

}  // namespace cardinal::engine
