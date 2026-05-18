#pragma once

// =============================================================================
// 03_studio sample — types shared between main.cpp and console_setup.cpp.
//
// The sample's console setup grew large enough to deserve its own .cpp
// (200+ lines of CARDINAL_CVAR_*/CARDINAL_CCOMMAND macros). To keep main.cpp
// readable AND let the registration function reach into the live engine
// state without a 14-argument signature, we pass everything via this small
// "context" struct of raw pointers. The pointers are owned by main(); the
// console-setup file only borrows them, so there's no lifetime hand-off.
//
// ViewportSlot lives here too because both files mutate the vector
// (main.cpp's draw loop reads it, console_setup.cpp's `viewport.add` /
// `viewport.maximize` commands write it).
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/scene/scene.hpp>

#include <string>

namespace cardinal::core    { class FramePacer; }
namespace cardinal::rhi     { class Swapchain;  class Device; }
namespace cardinal::render  { class Registry; }
namespace cardinal::scene   { class Scene; }
namespace cardinal::world   { class WorldGrid; class WorldStreamer; }
namespace cardinal::ui        { class Studio; }
namespace cardinal::cppscript { class Engine; }

namespace sample_studio {

// Per-viewport editor state — title (the ImGui window id), visibility,
// view mode, FPS cap, and maximize/minimize toggles. Mirror of the
// ViewportSlot definition that used to live inside main(); kept here so
// console_setup.cpp can mutate the vector in `viewport.add` / `viewport.remove`.
struct ViewportSlot {
    std::string                  title;
    bool                         show{true};
    bool                         maximized{false};
    bool                         minimized{false};
    cardinal::scene::ViewMode    mode{cardinal::scene::ViewMode::Solid};
    float                        fps_cap{0.0f};   // 0 = Unlimited
};

// Hard-coded ceiling matching the editor's MenuItem labels + dock layout.
inline constexpr int kMaxViewports = 16;

// Borrowed handles + state pointers. Lifetime: caller owns everything;
// the console-setup function captures by value into lambdas, so the
// pointers must outlive the cardinal::console::Registry (i.e. the
// process; we never tear the registry down).
struct ConsoleSetupContext {
    cardinal::rhi::Swapchain*           sw{nullptr};
    cardinal::core::FramePacer*         pacer{nullptr};
    cardinal::render::Registry*         pipelines{nullptr};
    cardinal::world::WorldGrid*         world_grid{nullptr};
    cardinal::world::WorldStreamer*     world_streamer{nullptr};
    cardinal::scene::Scene*             scene{nullptr};
    cardinal::rhi::Device*              device{nullptr};
    cardinal::ui::Studio*               studio{nullptr};
    cardinal::cppscript::Engine*        cppscript{nullptr};

    // Mutable host-side editor state. The console reads + writes these.
    std::vector<ViewportSlot>*          viewports{nullptr};
    int*                                next_viewport_serial{nullptr};
    cardinal::u32*                      selected_id{nullptr};
    bool*                               want_quit{nullptr};
};

// One-shot: registers every CVar + CCommand the sample exposes through
// the engine console. Idempotent only in the sense that double-calling
// will hit Registry::register_*'s "already registered" guard and silently
// drop duplicates — don't rely on that.
void register_engine_console_commands(const ConsoleSetupContext& ctx);

}  // namespace sample_studio
