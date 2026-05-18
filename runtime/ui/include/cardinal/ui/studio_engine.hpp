#pragma once

// =============================================================================
// Cardinal — StudioEngine (the "roof" layer).
//
// Layering model — concrete / wood / roof:
//   cardinal::core            (concrete — types, math, jobs, hal, io, …)
//   cardinal::engine          (wood — Window, RHI, Scene, Render, Physics,
//                              Lua, Plugins, frame loop. No UI.)
//   cardinal::ui::StudioEngine (roof — wraps an Engine and adds the ImGui
//                              dockspace, menu bar, Studio panel pump,
//                              FlyCamera input, per-viewport FPS quotas.)
//
// Use this when you want an editor-flavoured app — Studio panels, dockable
// viewports, the works. Use cardinal::engine::Engine directly when you want
// a shipped headless game with no editor cost.
//
// Typical editor host:
//
//     class MyApp : public cardinal::ui::StudioApplication {
//         void on_init     (engine::Engine&) override { ... }
//         void on_simulate (engine::Engine&, f32 dt) override { ... }
//         void on_ui       (StudioEngine&) override { ... ImGui windows ... }
//         void on_menu_bar (StudioEngine&) override { ... ImGui::MenuItem ... }
//     };
//
//     int main(int argc, char** argv) {
//         MyApp app;
//         return cardinal::ui::StudioEngine::run(argc, argv, app);
//     }
// =============================================================================

#include <cardinal/engine/engine.hpp>

#include <memory>

namespace cardinal::ui {

class Studio;
class StudioEngine;

// Editor-flavoured Application — adds UI hooks on top of the headless
// engine::Application contract. The StudioEngine frame pump invokes
// on_ui() between begin_frame() and end_frame() inside each frame, and
// on_menu_bar() inside the dockhost menu bar.
//
// on_init / on_simulate / on_shutdown still come from engine::Application
// and take an engine::Engine& — that's deliberate. Sim code lives in the
// wood layer and shouldn't reach into the roof.
class StudioApplication : public cardinal::engine::Application {
public:
    virtual void on_ui(StudioEngine& /*se*/) {}
    virtual void on_menu_bar(StudioEngine& /*se*/) {}
};

class StudioEngine {
public:
    // Convenience top-level: parses --backend= from argv, boots an
    // editor-flavoured engine, runs until quit, returns process exit code.
    static int run(int argc, char** argv, StudioApplication& app);

    // Lower-level: hand-rolled boot. Returns nullptr on failure.
    static std::unique_ptr<StudioEngine> create(
        const cardinal::engine::EngineDesc& desc);

    virtual ~StudioEngine() = default;
    StudioEngine(const StudioEngine&)            = delete;
    StudioEngine& operator=(const StudioEngine&) = delete;

    // Run the loop with a StudioApplication until quit().
    virtual int  run_loop(StudioApplication& app) = 0;

    // Manual single-frame pump (for hosts that drive their own outer loop).
    virtual bool tick_one_frame(StudioApplication& app) = 0;

    // Underlying wood-layer engine. Apps reach through here for device,
    // swapchain, scene, frame counters, backend swap, etc.
    virtual cardinal::engine::Engine& engine()       = 0;
    virtual Studio&                   studio()       = 0;

    virtual void quit() noexcept = 0;
    virtual bool wants_quit() const noexcept = 0;

    // Per-viewport FPS quotas — Studio's viewport panels read/write these.
    // Returns nullptr for an out-of-range index. The pure engine has its
    // own pacer; these slots feed the per-viewport pace ids on top of it.
    static constexpr u32 kViewports = 3;
    virtual f32* viewport_fps_slot(u32 idx) noexcept = 0;

protected:
    StudioEngine() = default;
};

}  // namespace cardinal::ui
