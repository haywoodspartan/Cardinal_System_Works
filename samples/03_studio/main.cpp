// =============================================================================
// Cardinal Studio — Phase 5 editor sample.
//
// Wires up everything Studio currently exposes:
//   - Backend selection via `--backend=<n>` (set by cardinal_launcher.exe)
//   - Off-screen viewport RTT
//   - 3D scene with primitive meshes (cardinal::scene)
//   - Forward renderer with selectable view modes
//   - Two viewport panels (each with its own ViewMode dropdown — proves
//     the "multiple detachable viewports with selectable analysis" feature)
//   - Scene hierarchy + inspector wired to the live scene
//   - Asset browser pointed at the project root
//   - Console panel hosting a Lua REPL via cardinal::script
//   - Plugin registry that scans `<exe>/plugins/`
// =============================================================================
#include "sample_types.hpp"

#include <cardinal/console/console.hpp>
#include <cardinal/core/async.hpp>
#include <cardinal/core/budget.hpp>
#include <cardinal/core/crash.hpp>
#include <cardinal/core/frame_pacer.hpp>
#include <cardinal/core/jobs.hpp>
#include <cardinal/core/log.hpp>
#include <cardinal/core/memory.hpp>
#include <cardinal/core/topology.hpp>
#include <cardinal/cppscript/cppscript.hpp>
#include <cardinal/edit/brush.hpp>
#include <cardinal/edit/editor_mode.hpp>
#include <cardinal/edit/mesh_ops.hpp>
#include <cardinal/edit/undo.hpp>
#include <cardinal/cmd/command.hpp>
#include <cardinal/vt/vt.hpp>
#include <cardinal/actor/world.hpp>
#include <cardinal/actor/builtin_prefabs.hpp>
#include <cardinal/sim/sim.hpp>
#include <cardinal/anim/anim.hpp>
#include <cardinal/audio/audio.hpp>
#include <cardinal/cine/cine.hpp>
#include <cardinal/game/game.hpp>
#include <cardinal/game/game_actor.hpp>
#include <cardinal/net/net.hpp>
#include <cardinal/net/replication.hpp>
#include <cardinal/game/reflection.hpp>
#include <cardinal/sky/sky.hpp>
#include <cardinal/scene/light.hpp>
#include <cardinal/core/autoscaler.hpp>
#include <cardinal/input/input.hpp>
#include <cardinal/serial/serial.hpp>
#include <cardinal/particles/particles.hpp>
#include <cardinal/nav/nav.hpp>
#include <cardinal/hud/hud.hpp>
#include <cardinal/project/project.hpp>
#include <cardinal/cook/cook.hpp>
#include <cardinal/pack/pack.hpp>
#include <cardinal/shader/shader.hpp>
#include <cardinal/asset/asset.hpp>
#include <cardinal/core/hal.hpp>
#include <cardinal/core/io.hpp>
#include <cardinal/core/geom.hpp>
#include <cardinal/partition/partition.hpp>
#include <cardinal/level/level.hpp>
#include <cardinal/mass/mass.hpp>
#include <cardinal/navmesh/navmesh.hpp>
#include <cardinal/ai/ai.hpp>

// Forward decl of the Win32 input message hook (defined in input_windows.cpp).
namespace cardinal::input {
    extern "C" bool win32_message_hook(void* hwnd, u32 msg, u64 wparam, i64 lparam, void* user);
}
#include <cardinal/world/world.hpp>
#include <cardinal/plugin/plugin.hpp>
#include <cardinal/render/pipeline.hpp>
#include <cardinal/rhi/rhi.hpp>
#include <cardinal/scene/assets.hpp>
#include <cardinal/scene/fly_camera.hpp>
#include <cardinal/scene/scene.hpp>
#include <cardinal/scene/terrain.hpp>
#include <cardinal/script/engine.hpp>
#include <cardinal/ui/studio.hpp>
#include <cardinal/window/window.hpp>

#include <imgui.h>
#include <imgui_internal.h>     // for DockBuilder

#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32) && !defined(NDEBUG)
    #include <crtdbg.h>
#endif

using cardinal::u32;
using cardinal::u64;
using cardinal::i32;
using cardinal::i64;
using cardinal::f32;
using cardinal::f64;
using cardinal::usize;

namespace ui    = cardinal::ui;
namespace rhi   = cardinal::rhi;
namespace win   = cardinal::window;
namespace scn   = cardinal::scene;
namespace plg   = cardinal::plugin;
namespace scr   = cardinal::script;
namespace rnd   = cardinal::render;
namespace clog  = cardinal::log;
namespace edit  = cardinal::edit;
namespace wld   = cardinal::world;
namespace cv    = cardinal::console;

// =============================================================================
// Sample game classes — demonstrate the CARDINAL_REGISTER_GAME_CLASS macro.
//
// Each class derives from cardinal::game::GameActor, gets editor-reflected
// properties via PROP_*, and overrides begin_play/on_tick/end_play. The
// Game::start_play() call fires begin_play across every spawned instance.
// =============================================================================

// SpinnerActor — rotates its host actor's Y axis at `rpm` revolutions per
// minute. Demonstrates Float properties + on_tick.
class SpinnerActor final : public cardinal::game::GameActor {
public:
    float                  rpm{30.0f};
    cardinal::scene::Vec3  axis{0.0f, 1.0f, 0.0f};
    bool                   reverse{false};

    void begin_play() override {
        clog::infof("game/spinner", "[%s] begin_play (rpm=%.1f)",
            owner() ? owner()->name().c_str() : "?", rpm);
    }
    void on_tick(float dt) override {
        if (owner() == nullptr) return;
        auto* tr = owner()->get_component<cardinal::actor::TransformComponent>();
        if (tr == nullptr) return;
        const float omega = (reverse ? -1.0f : 1.0f) * rpm * 6.2831853f / 60.0f;
        tr->rotation_euler.x += axis.x * omega * dt;
        tr->rotation_euler.y += axis.y * omega * dt;
        tr->rotation_euler.z += axis.z * omega * dt;
    }
    void end_play() override {
        clog::infof("game/spinner", "[%s] end_play",
            owner() ? owner()->name().c_str() : "?");
    }
};

CARDINAL_REGISTER_GAME_CLASS(SpinnerActor, "Demo/Spinner",
    PROP_FLOAT(rpm,     0.0f, 600.0f, "Revolutions per minute")
    PROP_VEC3 (axis,                  "Rotation axis (world space)")
    PROP_BOOL (reverse,               "Spin in the opposite direction"))

// FloaterActor — bobs up and down with a configurable amplitude/period.
// Demonstrates how a GameActor can persist its own state (start_y_).
class FloaterActor final : public cardinal::game::GameActor {
public:
    float amplitude{1.5f};
    float period_s {2.0f};
    bool  attenuate_when_paused{true};

    void begin_play() override {
        if (owner()) {
            if (auto* tr = owner()->get_component<cardinal::actor::TransformComponent>())
                start_y_ = tr->translation.y;
        }
        accum_ = 0.0f;
    }
    void on_tick(float dt) override {
        if (owner() == nullptr) return;
        auto* tr = owner()->get_component<cardinal::actor::TransformComponent>();
        if (tr == nullptr) return;
        accum_ += dt;
        const float w = period_s > 1e-3f ? (6.2831853f / period_s) : 0.0f;
        tr->translation.y = start_y_ + amplitude * std::sin(accum_ * w);
    }

private:
    float start_y_{0.0f};
    float accum_  {0.0f};
};

CARDINAL_REGISTER_GAME_CLASS(FloaterActor, "Demo/Floater",
    PROP_FLOAT(amplitude, 0.0f, 20.0f, "Vertical bob amplitude (units)")
    PROP_FLOAT(period_s,  0.1f, 30.0f, "Period of one bob in seconds")
    PROP_BOOL (attenuate_when_paused, "Cosmetic — informational"))

// PlayerActor — the controllable pawn the Play button possesses. It owns a
// reusable cardinal::actor::PlayerControllerComponent on its host actor and
// just forwards the editor-reflected tunables into it on begin_play. The
// per-frame input is fed host-side (main loop) exactly like the editor
// fly-camera — see the "Possessed player vs editor fly-cam" block below —
// because input + the scene camera live on the host, not in the sim.
class PlayerActor final : public cardinal::game::GameActor {
public:
    float move_speed       {6.0f};
    float sprint_multiplier{2.0f};
    float mouse_sensitivity{0.0035f};
    float eye_height       {1.7f};
    bool  fly_mode         {false};

    cardinal::actor::PlayerControllerComponent* controller() const noexcept {
        return owner() ? owner()->get_component<
            cardinal::actor::PlayerControllerComponent>() : nullptr;
    }

    void begin_play() override {
        if (owner() == nullptr) return;
        auto* pc = owner()->get_component<
            cardinal::actor::PlayerControllerComponent>();
        if (pc == nullptr)
            pc = owner()->add_component<
                cardinal::actor::PlayerControllerComponent>();
        pc->move_speed        = move_speed;
        pc->sprint_multiplier = sprint_multiplier;
        pc->look_sensitivity  = mouse_sensitivity;
        pc->eye_height        = eye_height;
        pc->fly_mode          = fly_mode;
        clog::infof("game/player", "[%s] possessed (speed=%.1f, fly=%d)",
            owner()->name().c_str(), move_speed, fly_mode ? 1 : 0);
    }
    void on_tick(float /*dt*/) override {
        // Intentionally empty: the host drives controller()->tick() with
        // real input + applies the first-person pose to scene::Camera.
    }
    void end_play() override {
        clog::infof("game/player", "[%s] released",
            owner() ? owner()->name().c_str() : "?");
    }
};

CARDINAL_REGISTER_GAME_CLASS(PlayerActor, "Demo/Player",
    PROP_FLOAT(move_speed,        0.5f, 50.0f, "Walk speed (units/sec)")
    PROP_FLOAT(sprint_multiplier, 1.0f, 6.0f,  "Shift sprint multiplier")
    PROP_FLOAT(mouse_sensitivity, 0.0005f, 0.02f, "Look radians per pixel")
    PROP_FLOAT(eye_height,        0.1f, 3.0f,  "Camera height above feet")
    PROP_BOOL (fly_mode,                       "Noclip: no gravity, free vertical"))

// -----------------------------------------------------------------------------
// CLI parsing (--backend=<vulkan|d3d12|auto>)
// -----------------------------------------------------------------------------
static rhi::Backend parse_backend_arg(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strncmp(a, "--backend=", 10) != 0) continue;
        const char* val = a + 10;
        if (std::strcmp(val, "vulkan") == 0) return rhi::Backend::Vulkan;
        if (std::strcmp(val, "d3d12")  == 0) return rhi::Backend::D3D12;
        if (std::strcmp(val, "auto")   == 0) return rhi::Backend::Auto;
    }
    return rhi::Backend::Auto;
}
static const char* backend_label(rhi::Backend b) {
    switch (b) {
        case rhi::Backend::Vulkan: return "Vulkan";
        case rhi::Backend::D3D12:  return "D3D12";
        case rhi::Backend::Auto:   return "Auto";
    }
    return "?";
}

// -----------------------------------------------------------------------------
// Default DockSpace layout — splits the host window into 4 docks the first
// time the user runs. `initial_viewport_titles` lists every viewport slot
// the sample owns at boot; each is docked into the central area as a tab.
// New viewports added at runtime via the View menu's "Add Viewport" button
// dock themselves into the same node when their window first opens (ImGui
// docks new windows under whatever node was last focused inside the central
// area, which is exactly the viewport tab strip).
// -----------------------------------------------------------------------------
static void build_default_layout(ImGuiID dockspace_id, const ImVec2& size,
                                 const std::vector<std::string>& initial_viewport_titles) {
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, size);

    ImGuiID center = dockspace_id;
    ImGuiID left   = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left,  0.18f, nullptr, &center);
    ImGuiID right  = ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.24f, nullptr, &center);
    ImGuiID bottom = ImGui::DockBuilderSplitNode(center, ImGuiDir_Down,  0.30f, nullptr, &center);

    // Viewports docked into the same node → ImGui auto-renders them as a
    // tab strip at the top of that node. Drag a tab off → it becomes its
    // own dock node (or its own OS window when ConfigFlags_ViewportsEnable
    // is on). That's the Chrome-tabs UX, end-to-end.
    // Toolbar — narrow strip across the top of the central area.
    ImGuiID top    = ImGui::DockBuilderSplitNode(center, ImGuiDir_Up, 0.05f, nullptr, &center);
    ImGui::DockBuilderDockWindow("Toolbar",              top);
    for (const auto& t : initial_viewport_titles) {
        ImGui::DockBuilderDockWindow(t.c_str(),          center);
    }
    ImGui::DockBuilderDockWindow("Hierarchy",            left);
    ImGui::DockBuilderDockWindow("Assets",               left);
    ImGui::DockBuilderDockWindow("Asset Palette",        left);
    ImGui::DockBuilderDockWindow("Inspector",            right);
    ImGui::DockBuilderDockWindow("Stats",                right);
    ImGui::DockBuilderDockWindow("Render Pipeline",      right);
    ImGui::DockBuilderDockWindow("World",                right);
    ImGui::DockBuilderDockWindow("Log",                  bottom);
    ImGui::DockBuilderDockWindow("Console",              bottom);
    ImGui::DockBuilderDockWindow("Code Sandbox",         bottom);
    ImGui::DockBuilderDockWindow("Stack Tracer",         bottom);
    ImGui::DockBuilderDockWindow("Function Tracer",      bottom);
    ImGui::DockBuilderDockWindow("Script Debugger",      bottom);
    ImGui::DockBuilderFinish(dockspace_id);
}

static void draw_dockspace(bool& first_time,
                           const std::vector<std::string>& initial_viewport_titles) {
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
    ImGui::Begin("##dockhost", nullptr, host_flags);
    ImGui::PopStyleVar(3);

    const ImGuiID dock_id = ImGui::GetID("CardinalDockSpace");
    if (first_time) {
        first_time = false;
        if (ImGui::DockBuilderGetNode(dock_id) == nullptr) {
            build_default_layout(dock_id, vp->WorkSize, initial_viewport_titles);
        }
    }
    ImGui::DockSpace(dock_id, ImVec2(0, 0), ImGuiDockNodeFlags_PassthruCentralNode);
}

int main(int argc, char** argv) {
    std::setbuf(stdout, nullptr);

    // Arm the crash reporter before anything else. A SEH violation
    // after this point lands a minidump + diagnostic .log in
    // <exe-dir>/crashes/ so we can debug the "click X and Vulkan
    // asserts" class of bug from a real stack trace next time.
    cardinal::crash::install();

#if defined(_WIN32) && !defined(NDEBUG)
    // Diagnostics: log every abort path so when the user sees the CRT
    // "abort() has been called" dialog there's a stderr line above it
    // identifying *what* called abort(). The CRT's own dialog still
    // appears — we just make sure we get a chance to log first.
    std::set_terminate([]() {
        clog::errorf("sample", "std::terminate() called — unhandled exception or noexcept violation");
        try {
            const auto cur = std::current_exception();
            if (cur) std::rethrow_exception(cur);
        } catch (const std::exception& e) {
            clog::errorf("sample", "  what(): %s", e.what());
        } catch (...) {
            clog::errorf("sample", "  (non-std exception)");
        }
        std::fflush(stderr);
        std::abort();
    });
    std::signal(SIGABRT, [](int) {
        clog::errorf("sample", "SIGABRT raised — abort() was called");
        std::fflush(stderr);
    });
    // CRT _ASSERT/_ASSERTE: route to stderr in addition to the dialog
    // so we always get the file:line reason in the log.
    _CrtSetReportMode(_CRT_ASSERT,
        _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG | _CRTDBG_MODE_WNDW);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR,
        _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG | _CRTDBG_MODE_WNDW);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    // Pure-virtual call handler — fires if a destructor calls a virtual
    // method that's been overridden by a now-destroyed derived class.
    _set_purecall_handler([]() {
        clog::errorf("sample", "pure virtual function call detected");
        std::fflush(stderr);
    });
    // Invalid CRT parameter handler — fires for things like sprintf with
    // null format strings, which would otherwise abort silently.
    _set_invalid_parameter_handler([](const wchar_t* expr,
                                      const wchar_t* func,
                                      const wchar_t* file,
                                      unsigned int line,
                                      uintptr_t) {
        char ebuf[128]{}, ffunc[64]{}, ffile[256]{};
        if (expr) std::wcstombs(ebuf, expr, sizeof(ebuf) - 1);
        if (func) std::wcstombs(ffunc, func, sizeof(ffunc) - 1);
        if (file) std::wcstombs(ffile, file, sizeof(ffile) - 1);
        clog::errorf("sample",
            "invalid CRT parameter: %s in %s (%s:%u)",
            ebuf[0] ? ebuf : "?",
            ffunc[0] ? ffunc : "?",
            ffile[0] ? ffile : "?", line);
        std::fflush(stderr);
    });
#endif

    const rhi::Backend chosen = parse_backend_arg(argc, argv);
    clog::infof("sample", "Backend (CLI): %s", backend_label(chosen));

    // ---- Window ------------------------------------------------------------
    win::WindowDesc wdesc{};
    wdesc.title  = "Cardinal Studio";
    wdesc.width  = 1600;
    wdesc.height = 900;
    auto window = win::Window::create(wdesc);
    if (window == nullptr) { std::fprintf(stderr, "Window failed\n"); return 1; }

    // ---- RHI ---------------------------------------------------------------
    rhi::DeviceDesc ddesc{};
    ddesc.backend = chosen;
    auto device = rhi::create_device(ddesc);
    if (device == nullptr) {
        clog::errorf("sample", "Device creation failed for backend %s",
                     backend_label(chosen));
        return 1;
    }
    clog::infof("sample", "RHI: %s [%s]", device->adapter_name(),
                device->backend() == rhi::Backend::Vulkan ? "Vulkan" : "D3D12");

    auto sw = device->create_swapchain(
        window->native_handle(), window->width(), window->height());
    if (sw == nullptr) { std::fprintf(stderr, "Swapchain failed\n"); return 1; }

    // Initial viewport size — Studio's Viewport panel grows the off-screen
    // RTT to its own size each frame, so this is just the bootstrap.
    constexpr u32 kViewportW = 1280;
    constexpr u32 kViewportH = 720;
    sw->set_viewport_size(kViewportW, kViewportH);

    // Default to uncapped framerate on the editor build — Stats panel has
    // a "VSync" toggle to flip back when the user wants pacing.
    sw->set_vsync(false);

    // ---- Frame pacer ------------------------------------------------------
    // Centralises FPS limits: foreground / background main-loop targets
    // PLUS per-viewport quotas (each viewport panel gets its own context).
    // The pacer auto-detects window focus via Win32 GetForegroundWindow.
    auto pacer = cardinal::core::FramePacer::create();
    pacer->set_window(window->native_handle());
    {
        // Default policy: uncapped while focused, 30 FPS while another
        // window has focus. The user can change this in Settings → FPS.
        cardinal::core::FrameLimit main_limit{};
        main_limit.fps_foreground = 0.0f;     // 0 = uncapped
        main_limit.fps_background = 30.0f;
        pacer->set_limit(cardinal::core::kPaceMainLoop, main_limit);
    }
    // ----- Variable viewport set -----------------------------------------
    //
    // The sample owns N viewport panels (default 4, max 16). Each carries
    // its own visualisation mode + FPS cap + visibility toggle + window
    // title. The View menu lets the user add or remove slots at runtime;
    // newly-added windows dock themselves into the central area's tab strip
    // when they first open. The most-recently-modified ViewMode is mirrored
    // into the active render::Pipeline so the global render reflects the
    // viewport the user is interacting with (true per-viewport RTT is the
    // next iteration — needs the swapchain's viewport state vectorised).
    //
    // ViewportSlot lives in sample_types.hpp so console_setup.cpp's
    // viewport.add / viewport.maximize commands can mutate the same vector.
    using ViewportSlot = sample_studio::ViewportSlot;
    constexpr int kMaxViewports = sample_studio::kMaxViewports;
    std::vector<ViewportSlot> viewports;
    viewports.reserve(kMaxViewports);
    // Tracks the next title number to assign so closing+re-adding doesn't
    // reuse a still-docked title (which would steal the docked slot).
    int next_viewport_serial = 1;
    auto make_slot = [&](scn::ViewMode m) {
        ViewportSlot s;
        s.title = "Viewport " + std::to_string(next_viewport_serial++);
        s.mode  = m;
        return s;
    };
    viewports.push_back(make_slot(scn::ViewMode::Solid));
    viewports.push_back(make_slot(scn::ViewMode::Wireframe));
    viewports.push_back(make_slot(scn::ViewMode::Heightmap));
    viewports.push_back(make_slot(scn::ViewMode::Normals));

    // ---- Asset catalog ----------------------------------------------------
    // Register the engine-shipped primitives + tree + terrain chunk so the
    // Asset Palette has something to show. Plugins can call register_asset
    // later to extend.
    scn::register_default_assets(*device);

    // ---- Terrain profile library -----------------------------------------
    // Engine-shipped curated biome profiles. The Studio's Terrain modal
    // picks from / edits this library. Saved profiles can be re-loaded
    // from disk via the modal's Load lib button.
    scn::register_default_terrain_profiles();

    // ---- Scene -------------------------------------------------------------
    scn::Scene scene;
    // Bootstrap a tiny demo scene by spawning straight from the catalog —
    // proves the editor is dogfooding its own asset registry on day one.
    {
        auto& cat = scn::AssetCatalog::instance();
        scn::AssetSpawnContext ctx{};
        ctx.scene = &scene; ctx.device = device.get();
        ctx.position = { 0.0f, 0.0f, 0.0f };
        cat.spawn("primitive.plane.m", ctx);
        ctx.position = { -1.4f, 0.0f, 0.0f }; cat.spawn("primitive.cube",       ctx);
        ctx.position = {  1.0f, 0.0f, 0.5f }; cat.spawn("primitive.sphere",     ctx);
        ctx.position = {  0.0f, 0.0f,-1.6f }; cat.spawn("primitive.cube",       ctx);
    }

    // Build the runtime-switchable render pipeline registry. Sample doesn't
    // talk to any specific pipeline — it just calls registry->render(scene)
    // each frame. The active pipeline is picked from the Render Pipeline
    // panel; switching is hot.
    auto pipelines = rnd::Registry::create(*device, *sw);
    if (pipelines == nullptr) { clog::errorf("sample", "render::Registry init failed"); return 1; }

    // ---- Studio ------------------------------------------------------------
    auto studio = ui::Studio::create(*device, *sw, *window);
    if (studio == nullptr) { std::fprintf(stderr, "Studio failed\n"); return 1; }
    clog::infof("sample", "Studio running — viewport %ux%u", kViewportW, kViewportH);

    // Hook the swapchain's internal-rebuild callback. The external resize
    // path (window resize) below still calls studio->on_swapchain_resized()
    // explicitly. THIS hook covers the cases where the swapchain rebuilds
    // itself without the host knowing — vsync mode change today, future
    // backend hot-swap. Without it, changing vsync at runtime crashed
    // because Studio kept VkFramebuffers tied to freed image views.
    sw->set_on_rebuilt([&studio]() { studio->on_swapchain_resized(); });

    // ---- Scripting ---------------------------------------------------------
    auto lua = scr::Engine::create();
    lua->run_string("cardinal.log_info('Lua', 'engine ready')");

    // C++ scripting engine — compiles arbitrary .cpp files at runtime via a
    // child clang-cl/cl.exe process and loads the result through the plugin
    // Registry. Runtime crash isolation matches the rest of the plugin
    // surface (SEH-tier today; cardinal::sandbox subprocess mode planned).
    //
    // Compose include paths from the running .exe's location so `script.compile`
    // works regardless of cwd (cwd would be `bin/`; argv[0]'s parent is the
    // same when launched directly + through cardinal_launcher.exe).
    cardinal::cppscript::Desc csdesc{};
    {
        const std::filesystem::path exe_path  = argv[0];
        // <repo>/build/<preset>/bin/Cardinal_System_Studio.exe → 3 levels up = repo root.
        const std::filesystem::path repo_root =
            (exe_path.parent_path() / ".." / ".." / "..").lexically_normal();
        const std::filesystem::path runtime   = repo_root / "runtime";
        for (const auto* mod : {"core", "plugin", "render"}) {
            csdesc.include_dirs.push_back((runtime / mod / "include").string());
        }
    }
    auto cppscript_engine = cardinal::cppscript::Engine::create(csdesc);

    // ---- Plugins -----------------------------------------------------------
    // Look in <exe-dir>/plugins. Cardinal_System_HelloPlugin (and any other
    // sample/user plugin DLLs) drop there at build time.
    {
        char exe[260]; std::filesystem::path dir;
#if defined(_WIN32)
        // Use argv[0]'s directory as a CWD-independent fallback.
        std::strncpy(exe, argv[0], sizeof(exe) - 1); exe[sizeof(exe) - 1] = '\0';
        dir = std::filesystem::path(exe).parent_path() / "plugins";
#else
        dir = std::filesystem::path("plugins");
#endif
        plg::Registry::instance().load_directory(dir.string().c_str());
    }

    // ---- Reflex defaults --------------------------------------------------
    // If Reflex is supported on this adapter, default to ON (no boost, no
    // cap) — gives latency benefits without fighting any user-set FPS cap.
    if (sw->reflex_supported()) {
        sw->set_reflex_mode(rhi::Swapchain::ReflexMode::On);
        clog::infof("sample", "Reflex: ON by default (no boost, no FPS cap)");
    } else {
        clog::infof("sample", "Reflex: not supported on this adapter");
    }

    // ---- Frame loop --------------------------------------------------------
    bool show_demo      = false;
    bool show_log       = true;
    bool show_stats     = true;
    bool show_hierarchy = true;
    bool show_inspector = true;
    bool show_assets    = true;
    bool show_palette   = true;
    bool show_console   = true;
    bool show_cppscript = true;
    bool show_stack_tr  = true;
    bool show_func_tr   = true;
    bool show_dbg       = true;
    bool show_pipeline  = true;
    bool show_memory    = true;
    bool show_profiler  = true;
    bool show_modes     = true;
    bool show_brush     = false;
    bool show_mesh_tools= false;
    bool show_tex_tools = false;
    bool show_vt        = true;
    bool show_sim       = true;
    bool show_actors    = true;
    bool show_prefabs   = true;
    bool show_curve     = false;
    bool show_sequencer = true;
    bool show_mixer     = true;
    bool show_game      = true;
    bool show_classes   = true;
    bool show_sky       = true;
    bool show_cmd_palette = false;   // Ctrl+P — unified command-bus palette
    bool show_input     = false;
    bool show_save_load = false;
    bool show_particles = false;
    bool show_nav       = false;
    // HUD is the in-game overlay — health bar, crosshair, minimap, etc.
    // Off by default so the editor isn't cluttered when you're not actually
    // playing. Toggle from View → HUD overlay, or just hit Play and it
    // shows itself.
    bool show_hud       = false;
    bool show_project   = true;
    bool show_cook_pack = true;
    bool show_shader_cc = false;
    bool show_world_sys = true;
    bool first_layout   = true;
    bool want_quit      = false;
    u32  selected_id    = 0;
    bool open_terrain_modal = false;

    // Fly camera replaces the demo orbit. It only takes input when the
    // Viewport panel is hovered (gated below).
    scn::FlyCamera fly_cam;
    fly_cam.sync_from(scene.camera());
    ImVec2 last_mouse_pos{0, 0};

    // Play-as-player possession: when the game is Playing and a PlayerActor
    // exists, the host drives its PlayerControllerComponent from real input
    // and copies the first-person pose onto scene::Camera. `player_id` is
    // assigned where the pawn is spawned (game-setup block below);
    // `was_possessing` tracks the edge so we hand the camera off smoothly
    // (sync the controller's look from the editor cam on Play; sync the
    // fly-cam back from the player's eye on Stop).
    cardinal::actor::ActorId player_id{0};
    bool was_possessing{false};

    // Optional "play as a CLIENT player": route the possessed pawn's
    // transform through the real net stack in-process (loopback) — the
    // exact server_broadcast -> transport -> client_ingest path an MMO
    // build takes, just with create_loopback() instead of create_udp().
    // Off by default (local possession); toggled live from View -> "Net
    // Loopback (client player)". Lazily created on first enable.
    cardinal::unique_ptr<cardinal::net::Transport>  net_loop;
    cardinal::unique_ptr<cardinal::net::Replicator> net_repl;
    cardinal::vector<cardinal::net::NetEvent>       net_events;
    cardinal::vector<cardinal::net::RepState>       net_tx, net_rx;
    bool net_loopback{false};

    // Editor tool — what an LMB click in the viewport does.
    //   Select     : pick the entity under the cursor
    //   PlaceAsset : spawn `placement_asset_id` at the Y=0 ray-plane hit
    enum class Tool : int { Select = 0, PlaceAsset };
    Tool        tool = Tool::Select;
    std::string placement_asset_id;     // empty when tool != PlaceAsset

    // Undo / redo stack — wired to the toolbar buttons + Ctrl+Z / Ctrl+Y
    // hotkeys + the gizmo-drag-release edge.
    //
    // SCOPE: this is the SCENE render-graph undo (scene::Entity transforms via
    // the gizmo, viewport place/delete). It is DISTINCT from the Game bar's
    // WorldHistory, which snapshots the ACTOR world (inspector / outliner /
    // layout edits). The two cover two different representations, so they are
    // intentionally separate authorities: a single Ctrl+Z drives THIS scene
    // stack (the Game bar's undo is button-driven, not chord-bound, to avoid a
    // double-fire). A truly unified single-history undo would require capturing
    // BOTH the scene graph AND the actor world in one ordered snapshot stream
    // (serialize_scene + serialize_world together) — a larger effort tracked
    // separately; a naive merge would let one undo desync the other half.
    edit::UndoStack undo;

    // Unified command bus — engine ops registered as named commands that UI
    // entry points dispatch by id (Phase 1: world.place_asset). Plus the
    // per-viewport captured (view, proj) the picking commands unproject with.
    cardinal::cmd::CommandRegistry commands;
    cardinal::cmd::register_builtin_commands(commands);
    std::vector<cardinal::cmd::ViewProj> rendered_vp;   // one per viewport, this frame

    // Gizmo-drag coalescing: capture the entity's position the moment the
    // drag starts so when the drag releases we can record one "Move
    // entity" undo with old → new positions, not one per frame.
    bool       gizmo_drag_was_active{false};

    // Multi-select set. `selected_id` stays the PRIMARY (== selection
    // back(), or 0) so the ~20 single-id call sites (inspector, context
    // menu, toolbar focus) keep working unchanged; the gizmo + hierarchy
    // + marquee drive the full set. Per-entity start transforms are
    // snapshotted on the drag-start edge so every mode (move/rotate/
    // scale) fans out drift-free and a single composite undo can revert
    // the whole multi-edit.
    std::vector<u32> selection;
    struct GizmoSnap { u32 id; scn::Vec3 p, r, s; };
    std::vector<GizmoSnap> gizmo_drag_snaps;
    scn::Vec3  gizmo_prim_p0{}, gizmo_prim_r0{}, gizmo_prim_s0{1,1,1};

    // Toolbar persistent state (mirrors the in-app Tool / snap / play flags).
    ui::Studio::ToolbarState tbar{};
    tbar.tool = 0;
    bool     snap_enabled = false;
    float    snap_step    = 1.0f;

    // ----- World grid + chunk streamer (3D) -----------------------------
    // Default world: 32-unit chunks, ±1024 chunks horizontal × ±64 vertical
    // (≈ 65 km × 4 km × 65 km), 8-chunk horizontal radius / 4-chunk vertical
    // half-height. (0,0,0) is the world centre; chunks below ground or
    // west of origin have negative coords. The user can crank extents up
    // toward INT32_MAX/4 in the World panel for effectively unbounded worlds.
    wld::WorldGridDesc world_desc{};
    world_desc.chunk_size         = 32.0f;
    world_desc.extent_x           = 1024;
    world_desc.extent_y           = 64;
    world_desc.extent_z           = 1024;
    world_desc.render_distance_xz = 8;
    world_desc.render_distance_y  = 4;
    wld::WorldGrid     world_grid(world_desc);
    wld::WorldStreamer world_streamer(world_grid);
    world_streamer.set_on_load([](wld::ChunkCoord c) {
        clog::infof("world", "load chunk (%d, %d, %d)", c.x, c.y, c.z);
    });
    world_streamer.set_on_unload([](wld::ChunkCoord c) {
        clog::infof("world", "unload chunk (%d, %d, %d)", c.x, c.y, c.z);
    });
    // World-panel mirror state.
    ui::Studio::WorldPanelState wpanel{};
    wpanel.chunk_size          = world_grid.chunk_size();
    wpanel.extent_x            = world_grid.extent_x();
    wpanel.extent_y            = world_grid.extent_y();
    wpanel.extent_z            = world_grid.extent_z();
    wpanel.render_distance_xz  = world_grid.render_distance_xz();
    wpanel.render_distance_y   = world_grid.render_distance_y();
    wpanel.show_grid_overlay   = true;
    wpanel.cull_outside_render_distance = true;
    bool show_world = true;

    // ----- Job system + async pool -------------------------------------
    //
    // Topology-aware: one worker per physical core, perf cores (V-Cache or
    // P-cores) come first. cardinal::async wraps the pool in
    // parallel_for / Future so subsystems don't have to thread a JobSystem*
    // through every call site. The Profiler panel reads worker stats from
    // here.
    cardinal::JobSystem job_system;
    {
        const auto cpu  = cardinal::topology::detect();
        const auto plan = cardinal::derive_worker_plan(cpu);
        job_system.start(plan);
        cardinal::async::set_pool(&job_system);
        // Register topology with the async profiler so the panel can label
        // each worker by tier (perf / general / background) and NUMA node.
        for (cardinal::u32 i = 0; i < plan.worker_cores.size(); ++i) {
            cardinal::async::detail::register_worker_topology(
                i,
                plan.worker_cores[i],
                static_cast<cardinal::async::WorkerTier>(plan.worker_tiers[i]),
                plan.worker_numa_nodes[i]);
        }
        clog::infof("sample",
            "JobSystem: %u workers (%u perf, %u general, %u background) "
            "on %u physical cores; hybrid=%s",
            static_cast<unsigned>(plan.worker_cores.size()),
            static_cast<unsigned>(plan.perf_worker_count),
            static_cast<unsigned>(plan.general_worker_count),
            static_cast<unsigned>(plan.background_worker_count),
            static_cast<unsigned>(cpu.physical_core_count),
            cpu.is_hybrid ? "yes" : "no");
    }

    // ----- Editor state machine ----------------------------------------
    //
    // Single mode at a time (UE5-style). Panels + viewport input check
    // .mode() to decide whether to consume an event vs forward it.
    cardinal::edit::EditorState editor_state;
    editor_state.on_mode_change(
        [](cardinal::edit::EditorMode old, cardinal::edit::EditorMode now) {
            clog::infof("editor", "mode %s -> %s",
                cardinal::edit::editor_mode_name(old),
                cardinal::edit::editor_mode_name(now));
        });

    // ----- Active brush (sculpt / paint / foliage) ---------------------
    cardinal::edit::brush::Brush active_brush{};
    active_brush.radius_world = 4.0f;
    active_brush.strength     = 2.0f;

    // The mesh-tools and texture-tools panels own their own State (held
    // inside Studio); the host reads the per-frame "request" snapshot
    // returned from the draw call.
    float last_frame_ms_for_profiler = 16.6f;

    // ----- Simulation, actors, animation, audio, cinematics ------------
    //
    // SimWorld owns the actor::World and drives tick groups (PrePhysics,
    // Physics, Update, Render, ...). We seed it with two demo actors:
    //   "Hero"  — RigidBody + AudioEmitter + Tag(player)
    //   "Light" — directional LightComponent
    // Plus a Blueprint "spawnable_box" the user can hit from the Outliner
    // panel to drop physics cubes into the scene.
    cardinal::sim::SimDesc simdesc{};
    simdesc.fixed_dt   = 1.0f / 120.0f;
    // simdesc.parallel_handlers stays at its default (false). The sample's
    // handlers are short (audio_engine.tick, bob_player.tick, cine_player
    // .tick, game's per-actor sweep) so the JobSystem dispatch + main-
    // thread wait round-trip dominates. With 7 tick groups per frame each
    // potentially fanning out, we were burning ~1-2 ms / frame in
    // dispatch overhead alone. Flip it back on once individual handlers
    // grow heavy enough to justify the overhead (>~100 µs each).
    cardinal::sim::SimWorld sim_world(simdesc);
    auto& aworld = sim_world.world();

    aworld.register_blueprint({
        "spawnable_box",
        [](cardinal::actor::Actor& a) {
            auto* m = a.add_component<cardinal::actor::MeshComponent>();
            m->asset_id = "primitives/box";
            m->tint     = {0.95f, 0.85f, 0.45f};
            auto* rb = a.add_component<cardinal::actor::RigidBodyComponent>();
            rb->use_gravity = true;
            rb->velocity    = {0.0f, 4.0f, 0.0f};
        }
    });
    aworld.register_blueprint({
        "ambient_light",
        [](cardinal::actor::Actor& a) {
            auto* l = a.add_component<cardinal::actor::LightComponent>();
            l->kind = cardinal::actor::LightKind::Directional;
            l->color = {1.0f, 0.95f, 0.85f};
            l->intensity = 1.6f;
        }
    });

    // Starter prefab library — lights, a physics prop, a camera, gameplay
    // markers — ready to stamp from the Studio Prefab panel.
    cardinal::actor::register_builtin_prefabs(aworld);

    cardinal::actor::Actor* hero = aworld.spawn("Hero");
    hero->add_component<cardinal::actor::TagComponent>()->add("player");
    {
        auto* m = hero->add_component<cardinal::actor::MeshComponent>();
        m->asset_id = "primitives/sphere";
        m->tint = {0.55f, 0.85f, 1.0f};
        auto* ae = hero->add_component<cardinal::actor::AudioEmitterComponent>();
        ae->cue_id = "footstep";
        ae->is_3d  = true;
        ae->loop   = false;
    }
    aworld.spawn_blueprint("ambient_light");

    // ---- Animation: a clip that bobs the Hero's Y -------------------
    cardinal::actor::ActorId hero_id = hero->id();
    auto bob_clip = std::make_shared<cardinal::anim::Clip>("hero_bob");
    auto bob_track = std::make_unique<cardinal::anim::Track<float>>(
        "hero.y",
        [hero_id, &aworld](const float& v) {
            if (auto* a = aworld.find(hero_id))
                if (auto* tr = a->get_component<cardinal::actor::TransformComponent>())
                    tr->translation.y = v;
        });
    bob_track->curve().add_key(0.0f, 0.0f, cardinal::anim::InterpMode::CubicHermite);
    bob_track->curve().add_key(1.0f, 1.5f, cardinal::anim::InterpMode::CubicHermite);
    bob_track->curve().add_key(2.0f, 0.0f, cardinal::anim::InterpMode::CubicHermite);
    bob_track->curve().wrap = cardinal::anim::WrapMode::Loop;
    auto bob_curve_handle = &bob_track->curve();   // for the Curve Editor panel
    bob_clip->add_track(std::move(bob_track));
    auto bob_player = std::make_shared<cardinal::anim::Player>(bob_clip);
    bob_player->set_loop(true);
    bob_player->play();

    // ---- Audio engine + a few cues ----------------------------------
    cardinal::audio::EngineDesc adesc{};
    adesc.distance_min = 1.0f;
    adesc.distance_max = 40.0f;
    adesc.rolloff      = 1.5f;
    auto audio_engine = cardinal::audio::Engine::create(adesc);
    {
        cardinal::audio::Cue c{};
        c.id = "footstep";
        c.kind = cardinal::audio::CueKind::SineWave;
        c.duration_s        = 0.10f;
        c.sine_frequency_hz = 220.0f;
        c.gain              = 0.35f;
        audio_engine->register_cue(std::move(c));
    }
    {
        cardinal::audio::Cue c{};
        c.id = "ambient_pad";
        c.kind = cardinal::audio::CueKind::SineWave;
        c.duration_s        = 8.0f;
        c.sine_frequency_hz = 110.0f;
        c.gain              = 0.15f;
        audio_engine->register_cue(std::move(c));
    }
    audio_engine->play_2d("ambient_pad", cardinal::audio::kChannelMusic, 0.7f, 1.0f, true);

    // ---- Cinematic sequence -----------------------------------------
    cardinal::cine::Sequence demo_seq{};
    demo_seq.name       = "demo_intro";
    demo_seq.duration_s = 8.0f;
    demo_seq.looping    = true;
    {
        cardinal::cine::AnimationTrack at{};
        at.name        = "Hero Bob";
        at.player      = bob_player;
        at.start_time_s= 0.0f;
        at.length_s    = 8.0f;
        demo_seq.tracks.emplace_back(std::move(at));

        cardinal::cine::AudioTrack aud{};
        aud.name = "Stings";
        aud.events.push_back({1.0f, "footstep", cardinal::audio::kChannelSfx, 1.0f, false, {0,0,0}});
        aud.events.push_back({3.0f, "footstep", cardinal::audio::kChannelSfx, 1.0f, false, {0,0,0}});
        aud.events.push_back({5.0f, "footstep", cardinal::audio::kChannelSfx, 1.0f, false, {0,0,0}});
        aud.events.push_back({7.0f, "footstep", cardinal::audio::kChannelSfx, 1.0f, false, {0,0,0}});
        demo_seq.tracks.emplace_back(std::move(aud));

        cardinal::cine::EventTrack ev{};
        ev.name = "Story";
        ev.events.push_back({0.5f, "scene.start", ""});
        ev.events.push_back({4.0f, "scene.midpoint", "halfway"});
        ev.events.push_back({7.5f, "scene.outro", ""});
        demo_seq.tracks.emplace_back(std::move(ev));
    }
    demo_seq.markers.push_back({"Intro",   0.5f, 0xFFFFFFFFu});
    demo_seq.markers.push_back({"Midpoint",4.0f, 0xFFFFFFFFu});
    cardinal::cine::Player cine_player(&demo_seq, audio_engine.get());
    cine_player.set_event_dispatch([&](const std::string& name, const std::string& payload) {
        clog::infof("cine/event", "fired: %s '%s'", name.c_str(), payload.c_str());
    });
    cine_player.play();

    // Wire the animation player into the simulation's Update group so
    // pause/step/time-scale all flow through it.
    sim_world.add_handler(cardinal::sim::TickGroup::Update,
        [bob_player](float dt) { bob_player->tick(dt); });
    sim_world.add_handler(cardinal::sim::TickGroup::LateUpdate,
        [&cine_player](float dt) { cine_player.tick(dt); });
    sim_world.add_handler(cardinal::sim::TickGroup::Update,
        [&audio_engine](float dt) { if (audio_engine) audio_engine->tick(dt); });

    cardinal::u32 selected_actor_id = hero ? hero->id() : 0u;

    // ----- Input ------------------------------------------------------
    // Process-wide manager + Win32 hook installed on the host window.
    // Sample bindings: Move.X = A/D, Move.Z = W/S, Jump = Space, Fire = LMB.
    auto input_mgr = cardinal::input::Manager::create();
    input_mgr->attach_to_window(window.get());
    // add (don't replace) — Studio already attached its ImGui hook in
    // Studio::create() above. The chain runs Studio first (so ImGui can
    // consume text input / panel clicks); the input manager's hook never
    // consumes (always returns false) so it just snapshots state.
    window->add_message_hook(cardinal::input::win32_message_hook, input_mgr.get());
    {
        using namespace cardinal::input;
        input_mgr->bind_axis("Move.X",
            Binding{BindKind::Key, static_cast<u32>(KeyCode::D),  +1.0f});
        input_mgr->bind_axis("Move.X",
            Binding{BindKind::Key, static_cast<u32>(KeyCode::A),  -1.0f});
        input_mgr->bind_axis("Move.Z",
            Binding{BindKind::Key, static_cast<u32>(KeyCode::W),  -1.0f});
        input_mgr->bind_axis("Move.Z",
            Binding{BindKind::Key, static_cast<u32>(KeyCode::S),  +1.0f});
        input_mgr->bind_action("Jump",
            Binding{BindKind::Key, static_cast<u32>(KeyCode::Space), 1.0f});
        input_mgr->bind_action("Fire",
            Binding{BindKind::MouseButton, static_cast<u32>(MouseButton::Left), 1.0f});
        input_mgr->bind_action("QuickSave",
            Binding{BindKind::Key, static_cast<u32>(KeyCode::F2), 1.0f});
        input_mgr->bind_action("QuickLoad",
            Binding{BindKind::Key, static_cast<u32>(KeyCode::F3), 1.0f});
    }

    // ----- Particles --------------------------------------------------
    auto particles_sys = cardinal::particles::System::create();
    {
        cardinal::particles::EmitterDesc d{};
        d.name = "Sparkles";
        d.origin = {0.0f, 1.0f, 0.0f};
        d.rate_per_second = 80.0f;
        d.max_particles   = 2048;
        d.lifetime_min = 0.6f; d.lifetime_max = 1.6f;
        d.size_min = 0.04f;    d.size_max = 0.10f;
        d.velocity_min = {-1.0f, 1.0f, -1.0f};
        d.velocity_max = { 1.0f, 3.0f,  1.0f};
        d.start_rgba = 0xFF80E0FFu;
        d.end_rgba   = 0x00FF80E0u;
        particles_sys->add(d);
    }

    // ----- Navigation grid -------------------------------------------
    cardinal::nav::Grid nav_grid(48, 32, 1.0f);
    // Drop a few walls so the demo path has obstacles to find a way around.
    for (cardinal::i32 x = 8;  x < 22; ++x) nav_grid.set_blocked(x, 12);
    for (cardinal::i32 y = 12; y < 22; ++y) nav_grid.set_blocked(28, y);
    cardinal::ui::Studio::NavState nav_state{};
    nav_state.start_x = 2;  nav_state.start_y = 2;
    nav_state.goal_x  = 40; nav_state.goal_y  = 28;

    // ----- HUD --------------------------------------------------------
    cardinal::hud::Hud hud;

    // ----- Project + Cook + Pack + Shader compiler + Asset registry ---
    //
    // Five new modules end-to-end:
    //   project: open / create projects from Studio
    //   shader : caches compiled HLSL bytecode keyed by source hash
    //   cook   : walks assets/ + bakes per-extension cookers' output
    //   pack   : binary archive format the game streams from at runtime
    //   asset  : runtime registry — game code asks for keys, gets typed assets
    auto shader_compiler = cardinal::shader::Compiler::create(*device,
        "G:/Cardinal_System_Works/build/shader_cache");
    cardinal::cook::CookerRegistry cooker_registry;
    cooker_registry.register_builtin(shader_compiler.get());

    // Asset registry — mounts directories and/or pack archives. Game code
    // looks up "key" (e.g. "meshes/teapot") and gets back a typed asset.
    auto asset_registry = cardinal::asset::Registry::create();

    // Pre-register a couple of in-memory materials so the renderer has
    // something to resolve even before the user cooks anything.
    {
        cardinal::asset::MaterialAsset gold{};
        gold.base_color = {1.00f, 0.78f, 0.34f};
        gold.metallic   = 1.0f;
        gold.roughness  = 0.20f;
        asset_registry->register_material("materials/gold", gold);
        cardinal::asset::MaterialAsset emissive{};
        emissive.base_color = {0.20f, 0.40f, 1.00f};
        emissive.emission   = {0.20f, 0.55f, 1.00f};
        emissive.emission_strength = 4.0f;
        asset_registry->register_material("materials/glow", emissive);
    }

    // The "current" project the editor is editing — null until the user
    // opens one or creates one via the Project panel.
    std::shared_ptr<cardinal::project::Project> current_project;
    cardinal::project::RecentProjects recent_projects(
        "G:/Cardinal_System_Works/build/recent_projects.txt");
    recent_projects.load();

    // ----- World systems: HAL + IO + Partition + Level/HLOD + Mass +
    //                      Navmesh + AI ----------------------------------
    //
    // HAL is process-static (free functions); we just touch it once so the
    // CPU features get detected and cached.
    (void)cardinal::hal::cpu_features();
    (void)cardinal::hal::os_info();
    cardinal::hal::set_current_thread_name("main");

    // IO Dispatcher — configurable per-priority concurrency caps.
    cardinal::io::DispatcherDesc io_desc{};
    auto io_dispatcher = cardinal::io::Dispatcher::create(io_desc);

    // World Partition — 4 demo cells covering a small area, mixed strategies.
    auto world_partition = cardinal::partition::WorldPartition::create({});
    {
        cardinal::partition::CellDesc origin{};
        origin.name   = "Origin Hub";
        origin.bounds = cardinal::core::geom::AABB::from_center_extent(
            {0, 0, 0}, {16, 8, 16});
        origin.mode   = cardinal::partition::StreamMode::Always;
        origin.priority = 100;
        world_partition->add_cell(origin);

        cardinal::partition::CellDesc north{};
        north.name   = "North District";
        north.bounds = cardinal::core::geom::AABB::from_center_extent(
            {0, 0, -64}, {32, 12, 32});
        north.mode   = cardinal::partition::StreamMode::Distance;
        north.load_radius = 80.0f; north.unload_radius = 110.0f;
        world_partition->add_cell(north);

        cardinal::partition::CellDesc east{};
        east.name    = "East Vista";
        east.bounds  = cardinal::core::geom::AABB::from_center_extent(
            {64, 0, 0}, {32, 12, 32});
        east.mode    = cardinal::partition::StreamMode::Vision;
        world_partition->add_cell(east);

        cardinal::partition::CellDesc remote{};
        remote.name  = "Remote Sector";
        remote.bounds= cardinal::core::geom::AABB::from_center_extent(
            {200, 0, 200}, {64, 16, 64});
        remote.mode  = cardinal::partition::StreamMode::DistanceOrVision;
        remote.load_radius = 150.0f; remote.unload_radius = 200.0f;
        world_partition->add_cell(remote);
    }
    world_partition->set_on_load(
        [](cardinal::partition::CellId id, const cardinal::partition::CellDesc& d) {
            clog::infof("partition", "LOAD   cell %u (%s)", id, d.name.c_str());
        });
    world_partition->set_on_unload(
        [](cardinal::partition::CellId id, const cardinal::partition::CellDesc& d) {
            clog::infof("partition", "UNLOAD cell %u (%s)", id, d.name.c_str());
        });
    const u32 main_viewer_id = world_partition->add_viewer({});

    // Level Manager — built on top of actor::World. Spawn a sample
    // LevelInstance with a couple of decorative actor templates.
    auto level_manager = cardinal::level::LevelManager::create(aworld);
    {
        cardinal::level::LevelInstanceDesc inst_desc{};
        inst_desc.name = "decoration_pack";
        cardinal::level::ActorTemplate t1{};
        t1.name = "Pillar A"; t1.translation = { -3, 0, 0 };
        t1.mesh_asset = "primitives/box"; t1.tint = {0.7f, 0.7f, 0.85f};
        cardinal::level::ActorTemplate t2{};
        t2.name = "Pillar B"; t2.translation = {  3, 0, 0 };
        t2.mesh_asset = "primitives/box"; t2.tint = {0.85f, 0.7f, 0.7f};
        inst_desc.actors = { t1, t2 };
        auto inst = cardinal::level::LevelInstance::create(inst_desc);
        level_manager->spawn(inst, {0, 0, -10});
    }

    // Asset placement — the bridge that makes "place an asset" produce a
    // first-class actor (visible to the Actor Outliner / Game / serial
    // save-load / level) while a scene::Entity stays the render mirror.
    // Every palette drop + "spawn here" goes through this; sync keeps the
    // authoritative actor and the rendered entity coherent each frame.
    auto placement = cardinal::level::AssetPlacement::create(aworld, scene);

    // HLOD — build a tree from the scene's mesh entities so the panel has
    // something to show. Small scenes get small trees; that's fine.
    cardinal::level::HlodTree hlod_tree;
    {
        std::vector<cardinal::level::HlodInput> inputs;
        u32 next_id = 1;
        for (const auto& e : scene.entities()) {
            if (!e.visible || e.mesh == nullptr) continue;
            cardinal::level::HlodInput hi{};
            hi.id       = next_id++;
            hi.position = e.transform.translation;
            hi.bounds   = cardinal::core::geom::AABB::from_center_extent(
                hi.position, {0.5f, 0.5f, 0.5f});
            inputs.push_back(hi);
        }
        if (!inputs.empty()) hlod_tree = cardinal::level::build_hlod(inputs);
    }

    // Mass ECS — register sample components + spawn a swarm.
    auto mass_world = cardinal::mass::World::create();
    const u32 ctf_bit = mass_world->register_component<cardinal::mass::CTransform>("Transform");
    const u32 cvl_bit = mass_world->register_component<cardinal::mass::CVelocity> ("Velocity");
    const u32 chl_bit = mass_world->register_component<cardinal::mass::CHealth>   ("Health");
    const u32 cfa_bit = mass_world->register_component<cardinal::mass::CFaction>  ("Faction");
    (void)ctf_bit; (void)cvl_bit; (void)chl_bit; (void)cfa_bit;
    for (cardinal::u32 i = 0; i < 256; ++i) {
        const cardinal::mass::EntityId e = mass_world->create_entity();
        cardinal::mass::CTransform tr{};
        tr.position = { static_cast<f32>(i % 16) * 1.5f - 12.0f, 0.0f,
                        static_cast<f32>(i / 16) * 1.5f - 12.0f };
        cardinal::mass::CVelocity v{};
        v.v = { 0.05f, 0.0f, 0.05f }; v.speed = 1.0f;
        cardinal::mass::CHealth   h{}; h.hp = 100.0f; h.max_hp = 100.0f;
        cardinal::mass::CFaction  f{}; f.id = i & 1;
        mass_world->add(e, tr);
        mass_world->add(e, v);
        mass_world->add(e, h);
        mass_world->add(e, f);
    }

    // AI Perception — one sensor at the camera + a couple of stimuli so
    // the AI tab has live events to show.
    cardinal::ai::PerceptionWorld ai_perception;
    cardinal::ai::SensorDesc cam_sensor{};
    cam_sensor.position = scene.camera().position;
    cam_sensor.forward  = {0, 0, -1};
    cam_sensor.sight_radius    = 30.0f;
    cam_sensor.sight_cos_angle = std::cos(60.0f * 0.0174532925f);
    cam_sensor.hearing_radius  = 20.0f;
    const auto cam_sensor_id = ai_perception.add_sensor(cam_sensor);
    {
        cardinal::ai::StimulusDesc s{};
        s.kind = cardinal::ai::StimulusKind::Sight;
        s.position = {5, 1, -5};
        s.lifetime_seconds = 0.0f;   // persistent
        ai_perception.add_stimulus(s);
        s.kind = cardinal::ai::StimulusKind::Sound;
        s.position = {-8, 1, 8};
        s.strength = 1.5f;
        s.lifetime_seconds = 0.0f;
        ai_perception.add_stimulus(s);
    }

    // Navmesh — a flat 4x4 quad plane around the origin, triangulated.
    cardinal::navmesh::Mesh nav_mesh;
    {
        std::vector<cardinal::scene::Vec3> verts;
        std::vector<u32>                   tris;
        const i32 N = 4;
        const f32 cell = 4.0f;
        const f32 origin = -static_cast<f32>(N) * cell * 0.5f;
        for (i32 y = 0; y <= N; ++y) {
            for (i32 x = 0; x <= N; ++x) {
                verts.push_back({ origin + x * cell, 0.0f, origin + y * cell });
            }
        }
        for (i32 y = 0; y < N; ++y) {
            for (i32 x = 0; x < N; ++x) {
                const u32 a = y * (N + 1) + x;
                const u32 b = a + 1;
                const u32 c = a + (N + 1);
                const u32 d = c + 1;
                tris.push_back(a); tris.push_back(b); tris.push_back(d);
                tris.push_back(a); tris.push_back(d); tris.push_back(c);
            }
        }
        nav_mesh.build_from_triangles(verts, tris);
    }

    // ----- Game lifecycle ---------------------------------------------
    // The Game wraps SimWorld with a Stopped/Playing/Paused state machine.
    // While Stopped, the editor still ticks (cameras, UI, profiler) but
    // GameActor::on_tick() is gated. The "Play" button transitions to
    // Playing and fires begin_play across every GameActor instance.
    cardinal::game::Game game(sim_world);
    game.set_on_state_change([](cardinal::game::GameState old_,
                                cardinal::game::GameState now) {
        clog::infof("game", "state %s -> %s",
            cardinal::game::game_state_name(old_),
            cardinal::game::game_state_name(now));
    });
    // Pre-spawn one of each demo class so the user can immediately see
    // them in the Outliner before pressing Play.
    if (auto* a = game.spawn_class("SpinnerActor", "Spinner")) {
        if (auto* tr = a->get_component<cardinal::actor::TransformComponent>()) {
            tr->translation = {0.0f, 1.5f, 0.0f};
        }
        auto* m = a->add_component<cardinal::actor::MeshComponent>();
        m->asset_id = "primitives/box";
        m->tint = {0.85f, 0.65f, 0.30f};
    }
    if (auto* a = game.spawn_class("FloaterActor", "Floater")) {
        if (auto* tr = a->get_component<cardinal::actor::TransformComponent>()) {
            tr->translation = {3.0f, 1.0f, 0.0f};
        }
        auto* m = a->add_component<cardinal::actor::MeshComponent>();
        m->asset_id = "primitives/sphere";
        m->tint = {0.55f, 0.85f, 0.95f};
    }
    // The pawn the Play button possesses. Visible in the Outliner before
    // Play; the host (camera/input block) drives it while Playing.
    if (auto* a = game.spawn_class("PlayerActor", "Player")) {
        player_id = a->id();
        if (auto* tr = a->get_component<cardinal::actor::TransformComponent>())
            tr->translation = {0.0f, 0.0f, 6.0f};
        auto* m = a->add_component<cardinal::actor::MeshComponent>();
        m->asset_id = "primitives/box";
        m->tint = {0.30f, 0.75f, 0.95f};
        // Ensure the controller exists pre-Play so the inspector shows it.
        a->add_component<cardinal::actor::PlayerControllerComponent>();
    }

    // ----- Sky / Time-of-Day ------------------------------------------
    // 12-minute day cycle by default; Sky::tick uses unscaled wall time so
    // the sky keeps moving even when the game is paused (matches editor
    // expectations — pausing shouldn't freeze the lighting reference).
    cardinal::sky::Sky sky;
    sky.set_day_length_seconds(12.0f * 60.0f);
    sky.set_hour(8.5f);

    // ----- LightSet (renderer-facing) --------------------------------
    // Mirrors the sky's sun + every actor LightComponent into a flat list
    // the renderer reads each frame.
    cardinal::scene::LightSet light_set;
    pipelines->set_light_set(&light_set);

    // (Autoscalers are constructed after budget_broker — see below.)

    // ----- Virtual Texture system --------------------------------------
    //
    // One process-shared System owns the physical tile pool + decode
    // pipeline. We register one demo VT (4096 x 4096 tiles at mip 0,
    // 13 mips, procedurally generated) so the panel + sample have
    // something to chew on. Real games would register one per material
    // atlas or per terrain layer.
    cardinal::vt::SystemDesc vtdesc{};
    vtdesc.pool_slots          = 2048;   // ~128 MiB at 64 KiB/tile
    vtdesc.request_ring_size   = 8192;
    vtdesc.prefetch_neighbors  = 2;      // 4-neighbours + parent mip
    auto vt_system = cardinal::vt::System::create(vtdesc);

    cardinal::vt::VirtualTextureDesc demo_vtdesc{};
    demo_vtdesc.id           = 0;
    demo_vtdesc.width_tiles  = 4096;
    demo_vtdesc.height_tiles = 4096;
    demo_vtdesc.mip_count    = 13;       // 4096 -> 1
    demo_vtdesc.decoder      = std::make_shared<cardinal::vt::ProceduralDecoder>(0xC0FFEEu);
    auto demo_vt = cardinal::vt::VirtualTexture::create(demo_vtdesc);
    vt_system->register_vt(demo_vt);

    // Camera-driven sampling: each frame we take the camera's XZ position
    // and sweep a small AABB of tiles around it (in world units mapped to
    // VT tile space). LOD comes from a height-based ramp — flying high =
    // sample coarser mips. This is the "predictive prefetch" path; UE5
    // does this with a GPU feedback texture, we do it analytically.
    auto vt_request_around = [&](float wx, float wz, float altitude) {
        if (vt_system == nullptr) return;
        // Map world X/Z into VT tile coordinates. Demo VT is 4096 tiles
        // per side, ~1 tile per world unit centred on origin.
        const float scale  = 1.0f;
        const i32   center_x = static_cast<i32>(std::floor(wx * scale)) + 2048;
        const i32   center_y = static_cast<i32>(std::floor(wz * scale)) + 2048;
        // LOD from altitude — every 16 units up halves resolution.
        const u32 mip = std::min<u32>(12u,
            static_cast<u32>(std::max(0.0f, std::log2(std::max(1.0f, altitude * 0.0625f + 1.0f)))));
        const u32 wt = demo_vt->page_table().width_tiles(mip);
        const u32 ht = demo_vt->page_table().height_tiles(mip);
        // 5x5 tile sweep at the chosen mip.
        for (i32 dy = -2; dy <= 2; ++dy) {
            for (i32 dx = -2; dx <= 2; ++dx) {
                const i32 tx = (center_x >> mip) + dx;
                const i32 ty = (center_y >> mip) + dy;
                if (tx < 0 || ty < 0 ||
                    tx >= static_cast<i32>(wt) || ty >= static_cast<i32>(ht)) continue;
                demo_vt->request(mip, static_cast<u32>(ty), static_cast<u32>(tx));
            }
        }
    };

    // ----- Memory pressure broker --------------------------------------
    //
    // Cross-cuts CPU + GPU. Subsystems voluntarily honour the pressure tier
    // it publishes; the broker queries the live OS counters + the bound
    // rhi::Device's VRAM telemetry once per frame.
    //
    // The world streamer is our first consumer: under High pressure we
    // halve its render distance, under Critical we drop it to 1. The user
    // sees the chunk count collapse and frees a chunk's worth of VRAM
    // per slot eviction. The original render distance is restored when
    // pressure drops back.
    cardinal::budget::Broker budget_broker;
    {
        auto* dev_for_query = device.get();
        budget_broker.bind_gpu_query([dev_for_query]() {
            cardinal::budget::GpuMemorySample s{};
            if (dev_for_query != nullptr) {
                const auto vs = dev_for_query->query_vram_usage();
                s.budget_bytes        = vs.budget_bytes;
                s.current_usage_bytes = vs.current_usage_bytes;
            }
            return s;
        });
    }

    // ----- Autoscalers -------------------------------------------------
    // Three scalers register with the broker and react on tier transitions:
    //   - render_scale       — adjusts rhi::RenderSettings.render_scale
    //   - shadow_resolution  — informational (would resize shadow maps)
    //   - mesh_cache         — informational (would cap scene mesh cache)
    // These need to outlive the broker callbacks but die before the broker
    // itself, so unique_ptr in main() works (everything tears down LIFO at
    // function exit).
    rhi::Device* rhi_device_ptr = device.get();
    auto render_scale_scaler = std::make_unique<cardinal::budget::AutoScaler<float>>(
        budget_broker,
        cardinal::budget::AutoScaler<float>::Config{
            "render_scale",
            cardinal::budget::Domain::Gpu,
            /*low=*/1.00f, /*med=*/1.00f, /*high=*/0.85f, /*crit=*/0.65f,
            /*min=*/0, /*max=*/0,
            [rhi_device_ptr](const float& v, cardinal::memory::Pressure p) {
                if (rhi_device_ptr == nullptr) return;
                cardinal::rhi::RenderSettings s = rhi_device_ptr->settings();
                s.render_scale = v;
                rhi_device_ptr->apply_settings(s);
                clog::infof("budget/render_scale",
                    "pressure=%s -> render_scale=%.2f",
                    cardinal::memory::pressure_name(p), v);
            }
        });
    auto shadow_res_scaler = std::make_unique<cardinal::budget::AutoScaler<u32>>(
        budget_broker,
        cardinal::budget::AutoScaler<u32>::Config{
            "shadow_resolution",
            cardinal::budget::Domain::Gpu,
            /*low=*/4096u, /*med=*/2048u, /*high=*/1024u, /*crit=*/512u,
            /*min=*/0, /*max=*/0,
            [](const u32& v, cardinal::memory::Pressure p) {
                clog::infof("budget/shadow_res",
                    "pressure=%s -> shadow map %ux%u",
                    cardinal::memory::pressure_name(p), v, v);
            }
        });
    auto mesh_cache_scaler = std::make_unique<cardinal::budget::AutoScaler<u32>>(
        budget_broker,
        cardinal::budget::AutoScaler<u32>::Config{
            "mesh_cache",
            cardinal::budget::Domain::System,
            /*low=*/1024u, /*med=*/768u, /*high=*/384u, /*crit=*/96u,
            /*min=*/16ull  * 1024 * 1024,
            /*max=*/512ull * 1024 * 1024,
            [](const u32& v, cardinal::memory::Pressure p) {
                clog::infof("budget/mesh_cache",
                    "pressure=%s -> %u cached meshes",
                    cardinal::memory::pressure_name(p), v);
            }
        });
    // Remember the user's intended distances so we can restore them when
    // pressure subsides. The world panel may also mutate these mid-session;
    // we re-snapshot whenever the user changes them via the panel.
    int  ws_baseline_xz = world_grid.render_distance_xz();
    int  ws_baseline_y  = world_grid.render_distance_y();
    bool ws_overridden_by_pressure = false;
    // VT cache as a GPU-budget subsystem — under High pressure we shrink
    // the pool by 25%; under Critical we purge entirely and let the
    // residency rebuild as the camera moves. The host owns the VT system
    // pointer, so we capture it directly.
    if (vt_system) {
        cardinal::budget::Subsystem vt_sub{};
        vt_sub.name               = "vt_cache";
        vt_sub.domain             = cardinal::budget::Domain::Gpu;
        vt_sub.advisory_min_bytes = 32ull  * 1024ull * 1024ull;
        vt_sub.advisory_max_bytes = static_cast<cardinal::u64>(vtdesc.pool_slots) *
                                    cardinal::vt::kTileBytesRGBA;
        vt_sub.on_pressure_change =
            [vts = vt_system.get(), max_slots = vtdesc.pool_slots]
            (cardinal::memory::Pressure p) {
                using P = cardinal::memory::Pressure;
                switch (p) {
                    case P::Low:
                    case P::Medium:
                        vts->cache().resize(max_slots);
                        break;
                    case P::High:
                        vts->cache().resize(std::max(256u, (max_slots * 3u) / 4u));
                        break;
                    case P::Critical:
                        vts->purge();
                        vts->cache().resize(std::max(128u, max_slots / 4u));
                        break;
                }
                clog::infof("budget/vt_cache",
                    "pressure=%s -> resize to %u slots",
                    cardinal::memory::pressure_name(p),
                    static_cast<unsigned>(vts->cache().slot_count()));
            };
        budget_broker.register_subsystem(std::move(vt_sub));
    }

    {
        cardinal::budget::Subsystem ws_sub{};
        ws_sub.name               = "world_streamer";
        ws_sub.domain             = cardinal::budget::Domain::Gpu;
        // Advisory range — purely cosmetic for the panel's progress bar.
        ws_sub.advisory_min_bytes = 64ull  * 1024ull * 1024ull;   //  64 MB
        ws_sub.advisory_max_bytes = 512ull * 1024ull * 1024ull;   // 512 MB
        ws_sub.on_pressure_change =
            [&world_grid, &world_streamer, &ws_baseline_xz, &ws_baseline_y,
             &ws_overridden_by_pressure, &wpanel](cardinal::memory::Pressure p) {
            using P = cardinal::memory::Pressure;
            int new_xz = ws_baseline_xz;
            int new_y  = ws_baseline_y;
            switch (p) {
                case P::Low:
                case P::Medium:
                    new_xz = ws_baseline_xz;
                    new_y  = ws_baseline_y;
                    ws_overridden_by_pressure = false;
                    break;
                case P::High:
                    new_xz = std::max(2, ws_baseline_xz / 2);
                    new_y  = std::max(1, ws_baseline_y  / 2);
                    ws_overridden_by_pressure = true;
                    break;
                case P::Critical:
                    new_xz = 1;
                    new_y  = 1;
                    ws_overridden_by_pressure = true;
                    break;
            }
            if (new_xz != world_grid.render_distance_xz() ||
                new_y  != world_grid.render_distance_y())
            {
                world_grid.set_render_distance_xz(new_xz);
                world_grid.set_render_distance_y (new_y);
                world_streamer.invalidate();
                wpanel.render_distance_xz = new_xz;
                wpanel.render_distance_y  = new_y;
                clog::infof("budget/world_streamer",
                    "pressure=%s -> render_distance_xz=%d, _y=%d",
                    cardinal::memory::pressure_name(p), new_xz, new_y);
            }
        };
        budget_broker.register_subsystem(std::move(ws_sub));
    }

    // -------------------------------------------------------------------
    // Engine console — register a starter pack of CVars / CCommands.
    // Implementation lives in console_setup.cpp so this main() stays
    // focused on the frame loop. The context struct hands every relevant
    // pointer to the registration function; lambdas inside capture by
    // value so the ctx itself can go out of scope right after.
    // -------------------------------------------------------------------
    {
        sample_studio::ConsoleSetupContext cctx{};
        cctx.sw                   = sw.get();
        cctx.pacer                = pacer.get();
        cctx.pipelines            = pipelines.get();
        cctx.world_grid           = &world_grid;
        cctx.world_streamer       = &world_streamer;
        cctx.scene                = &scene;
        cctx.device               = device.get();
        cctx.viewports            = &viewports;
        cctx.next_viewport_serial = &next_viewport_serial;
        cctx.selected_id          = &selected_id;
        cctx.want_quit            = &want_quit;
        cctx.studio               = studio.get();
        cctx.cppscript            = cppscript_engine.get();
        sample_studio::register_engine_console_commands(cctx);
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto       prev_t = t0;
    u32        frame  = 0;

    u64 reflex_frame = 1;
    while (!window->should_close() && !want_quit) {
        // Reflex sleep MUST come first — it's where the driver decides
        // when to begin the new frame for minimum input-to-photon
        // latency. Pacer's own sleep follows (honours user-side caps).
        sw->reflex_sleep();
        pacer->wait_for_next_frame();

        // SimulationStart marker — must be the first marker of each
        // frame. InputSample is between Sim Start/End and brackets
        // user-input reads. RenderSubmit Start/End wraps all GPU
        // command submission.
        sw->reflex_marker(rhi::Swapchain::ReflexMarker::SimulationStart, reflex_frame);

        window->poll_events();
        sw->reflex_marker(rhi::Swapchain::ReflexMarker::InputSample, reflex_frame);

        // ---- Resize handling -----------------------------------------------
        // The OS window may have been resized; if so, recreate the swapchain
        // to the new client size and let Studio rebuild its per-image state.
        // While the window is minimised (zero-area client) we skip the whole
        // frame so we don't try to drive a 0x0 swapchain.
        if (window->is_minimized()) {
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(16ms);
            continue;
        }
        u32 nw = 0, nh = 0;
        if (window->consume_resize(&nw, &nh)) {
            if (sw->resize(nw, nh)) studio->on_swapchain_resized();
        }

        const auto now = std::chrono::steady_clock::now();
        const float dt = std::chrono::duration<float>(now - prev_t).count();
        prev_t = now;
        last_frame_ms_for_profiler = dt * 1000.0f;
        // End-of-frame tick for cardinal::async — rolls per-phase running
        // totals into the published snapshot the Profiler panel reads.
        cardinal::async::frame_advance();

        // SimulationEnd → RenderSubmitStart — boundary between CPU-side
        // sim work and GPU command recording, exactly where Reflex wants
        // its measurement split.
        sw->reflex_marker(rhi::Swapchain::ReflexMarker::SimulationEnd,   reflex_frame);
        sw->reflex_marker(rhi::Swapchain::ReflexMarker::RenderSubmitStart, reflex_frame);

        // ---- Frame ---------------------------------------------------------
        const float bg_r = 0.07f, bg_g = 0.08f, bg_b = 0.10f;

        // Tell the swapchain how many viewport RTTs we need this frame —
        // the editor owns the viewport vector, so the count is the live
        // size of `viewports`. Slot ids align 1:1 with vector indices.
        sw->set_viewport_count(static_cast<u32>(viewports.size()));
        sw->begin_frame(bg_r, bg_g, bg_b, 1.0f);

        // Maximize gating: if any viewport is maximized, only render that
        // one. Otherwise render every visible, non-minimised slot.
        int maximized_idx = -1;
        for (int i = 0; i < static_cast<int>(viewports.size()); ++i) {
            if (viewports[i].maximized) { maximized_idx = i; break; }
        }

        // 3D scene draw — dispatches to whichever pipeline the user picked
        // in the Render Pipeline panel. We loop once per visible viewport,
        // each time:
        //   1. Push that viewport's ViewMode into the active pipeline's
        //      "view_mode" knob so the render shades exactly as the panel
        //      requests (Solid / Wireframe / Polygons / Heightmap / etc.).
        //   2. Tell the swapchain to redirect the next render into that
        //      viewport's RTT (set_active_viewport).
        //   3. Compute aspect from the viewport's allocated extent.
        // The shared scene::Camera means every viewport sees the same
        // world from the same angle — only the visualisation differs.
        // Per-viewport Camera is a follow-up (would need split fly-cam state
        // mapped to the focused panel).
        // Reset the per-viewport captured matrices each frame (entries stay
        // invalid for viewports not rendered below, so a pick over them is
        // rejected rather than silently using a stale/foreign projection).
        rendered_vp.assign(viewports.size(), cardinal::cmd::ViewProj{});
        for (int i = 0; i < static_cast<int>(viewports.size()); ++i) {
            const auto& vps = viewports[i];
            if (!vps.show)      continue;
            if (vps.minimized)  continue;
            if (maximized_idx >= 0 && i != maximized_idx) continue;

            auto* active = pipelines->active();
            if (active != nullptr) {
                // Push the panel's chosen mode into whichever knob the
                // active pipeline uses for visualisation. ForwardBaseline
                // names it "view_mode" with 6 entries (Solid…RTXPreview);
                // DebugVisualizer names it "channel" with 5 entries
                // (Wireframe / Polygons / Heightmap / Normals / RTXPreview).
                // For DebugVisualizer we map ViewMode::Solid back to
                // Wireframe so a Solid-tagged panel still shows something
                // when DebugVisualizer is active.
                const int vm_idx = static_cast<int>(vps.mode);
                for (auto& kn : active->knobs()) {
                    if (kn.kind != rnd::KnobKind::Enum) continue;
                    if (kn.id == "view_mode") { kn.e = vm_idx; break; }
                    if (kn.id == "channel") {
                        // ViewMode 0=Solid (no channel equivalent → 0),
                        // 1=Wireframe, 2=Polygons, 3=Heightmap, 4=Normals,
                        // 5=RTX → channel index = vm-1 clamped to [0,4].
                        const int ch = (vm_idx == 0) ? 0
                                     : std::min(vm_idx - 1, 4);
                        kn.e = ch;
                        break;
                    }
                }
            }
            sw->set_active_viewport(static_cast<u32>(i));
            const u32   vw_i = sw->viewport_width (static_cast<u32>(i));
            const u32   vh_i = sw->viewport_height(static_cast<u32>(i));
            const float aspect_i = (vh_i > 0)
                ? static_cast<float>(vw_i) / static_cast<float>(vh_i)
                : 1.0f;
            pipelines->render(scene, aspect_i);
            // Capture the EXACT view/proj this viewport was rasterised with,
            // so a later viewport click unprojects with bit-identical matrices
            // (the placement-offset fix — no aspect/camera re-derivation).
            if (i < static_cast<int>(rendered_vp.size()))
                rendered_vp[static_cast<usize>(i)] =
                    { scene.camera().view(), scene.camera().proj(aspect_i), true };
        }
        // NOTE: set_viewport_camera does NOT re-derive aspect — it just
        // stores the view/proj it is given. The per-panel overlay camera
        // is set inside the viewport-panel draw loop below (each panel
        // gets proj(aspect_i) for its own extent) so detached-window
        // overlays match their RTT + the pick.

        // ---- Editor UI -----------------------------------------------------
        studio->begin_frame();
        // Hand the dockspace builder the boot-time viewport list so the
        // first-run layout docks every initial viewport into the central
        // tab strip. Slots added later via the View menu inherit that node
        // because ImGui parents undocked windows under the last-focused
        // node when first opened.
        std::vector<std::string> initial_viewport_titles;
        if (first_layout) {
            initial_viewport_titles.reserve(viewports.size());
            for (const auto& vps : viewports) initial_viewport_titles.push_back(vps.title);
        }
        draw_dockspace(first_layout, initial_viewport_titles);

        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Quit")) want_quit = true;
                ImGui::EndMenu();
            }
            // ---- Create — every primitive / composite the catalog ships
            // is reachable from this menu (spawned at the camera's look-at
            // point), and Terrain Grid… opens the chunked-terrain modal.
            if (ImGui::BeginMenu("Create")) {
                const auto fwd = scn::normalize(scene.camera().target - scene.camera().position);
                scn::Vec3 spawn_at = scene.camera().position + fwd * 2.5f;
                spawn_at.y = std::max(0.0f, spawn_at.y);

                auto& cat = scn::AssetCatalog::instance();
                for (const auto& cat_name : cat.categories()) {
                    if (ImGui::BeginMenu(cat_name.c_str())) {
                        for (const auto& a : cat.all()) {
                            if (a.category != cat_name) continue;
                            if (ImGui::MenuItem(a.label.c_str())) {
                                // Through AssetPlacement: spawns the render
                                // entity AND a first-class actor (Outliner
                                // / Game / serial / level all see it).
                                auto pr = placement->place(
                                    a.id.c_str(), device.get(), spawn_at);
                                if (pr.primary_entity)
                                    selected_id = pr.primary_entity;
                            }
                            if (ImGui::IsItemHovered() && !a.tooltip.empty()) {
                                ImGui::SetTooltip("%s", a.tooltip.c_str());
                            }
                        }
                        ImGui::EndMenu();
                    }
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Terrain Grid…")) open_terrain_modal = true;
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                // Viewports — one MenuItem per slot, plus an Add / Remove
                // pair. New slots default to Solid mode, Unlimited cap, and
                // visible; ImGui docks the new window into whichever node
                // last hosted a viewport (i.e. the central tab strip), so
                // they appear right beside the existing tabs.
                for (auto& vps : viewports) {
                    ImGui::MenuItem(vps.title.c_str(), nullptr, &vps.show);
                }
                ImGui::Separator();
                {
                    const bool can_add = static_cast<int>(viewports.size()) < kMaxViewports;
                    if (!can_add) ImGui::BeginDisabled();
                    if (ImGui::MenuItem("Add Viewport")) {
                        viewports.push_back(make_slot(scn::ViewMode::Solid));
                    }
                    if (!can_add) {
                        ImGui::EndDisabled();
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Maximum %d viewports.", kMaxViewports);
                        }
                    }
                    const bool can_remove = viewports.size() > 1;
                    if (!can_remove) ImGui::BeginDisabled();
                    if (ImGui::MenuItem("Remove Last Viewport")) {
                        viewports.pop_back();
                    }
                    if (!can_remove) {
                        ImGui::EndDisabled();
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("At least one viewport must remain.");
                        }
                    }
                    ImGui::TextDisabled("  %zu / %d viewports", viewports.size(), kMaxViewports);
                }
                ImGui::Separator();
                ImGui::MenuItem("Hierarchy",           nullptr, &show_hierarchy);
                ImGui::MenuItem("Inspector",           nullptr, &show_inspector);
                ImGui::MenuItem("Assets",              nullptr, &show_assets);
                ImGui::MenuItem("Asset Palette",       nullptr, &show_palette);
                ImGui::MenuItem("Stats",               nullptr, &show_stats);
                ImGui::MenuItem("Log",                 nullptr, &show_log);
                ImGui::MenuItem("Console",             nullptr, &show_console);
                ImGui::MenuItem("Code Sandbox",        nullptr, &show_cppscript);
                ImGui::Separator();
                ImGui::MenuItem("Render Pipeline",     nullptr, &show_pipeline);
                ImGui::MenuItem("Memory && Budgets",   nullptr, &show_memory);
                ImGui::MenuItem("Profiler",            nullptr, &show_profiler);
                ImGui::MenuItem("Virtual Textures",    nullptr, &show_vt);
                ImGui::Separator();
                ImGui::MenuItem("Game",                nullptr, &show_game);
                ImGui::MenuItem("Net Loopback (client player)", nullptr, &net_loopback);
                ImGui::MenuItem("Game Classes",        nullptr, &show_classes);
                ImGui::MenuItem("Sky / Time of Day",   nullptr, &show_sky);
                ImGui::MenuItem("Simulation",          nullptr, &show_sim);
                ImGui::MenuItem("Actors",              nullptr, &show_actors);
                ImGui::MenuItem("Prefabs",             nullptr, &show_prefabs);
                ImGui::MenuItem("Sequencer",           nullptr, &show_sequencer);
                ImGui::MenuItem("Mixer",               nullptr, &show_mixer);
                ImGui::MenuItem("Curve Editor",        nullptr, &show_curve);
                ImGui::Separator();
                ImGui::MenuItem("Input",               nullptr, &show_input);
                ImGui::MenuItem("Save / Load",         nullptr, &show_save_load);
                ImGui::MenuItem("Particles",           nullptr, &show_particles);
                ImGui::MenuItem("Navigation",          nullptr, &show_nav);
                ImGui::MenuItem("HUD overlay",         nullptr, &show_hud);
                ImGui::Separator();
                ImGui::MenuItem("Project",             nullptr, &show_project);
                ImGui::MenuItem("Cook && Pack",        nullptr, &show_cook_pack);
                ImGui::MenuItem("Shader Compiler",     nullptr, &show_shader_cc);
                ImGui::MenuItem("World Systems",       nullptr, &show_world_sys);
                ImGui::Separator();
                ImGui::MenuItem("Editor Modes",        nullptr, &show_modes);
                ImGui::MenuItem("Brush",               nullptr, &show_brush);
                ImGui::MenuItem("Mesh Tools",          nullptr, &show_mesh_tools);
                ImGui::MenuItem("Texture Tools",       nullptr, &show_tex_tools);
                ImGui::Separator();
                ImGui::MenuItem("World",               nullptr, &show_world);
                ImGui::MenuItem("Stack Tracer",        nullptr, &show_stack_tr);
                ImGui::MenuItem("Function Tracer",     nullptr, &show_func_tr);
                ImGui::MenuItem("Script Debugger",     nullptr, &show_dbg);
                ImGui::Separator();
                ImGui::MenuItem("ImGui Demo",          nullptr, &show_demo);
                ImGui::EndMenu();
            }
            // ---- Settings — quick toggles for things you fiddle with hourly.
            if (ImGui::BeginMenu("Settings")) {
                ImGui::SeparatorText("VSync");
                // Sync interval combo: 0 = uncapped (DXGI tearing path),
                // 1 = locked to display, 2 = half, 3 = third, 4 = quarter.
                // The tearing flag and the present-interval are paired up
                // inside the swapchain (interval > 0 ⇒ no tearing flag).
                const char* vs_labels[] = { "Off (uncapped)", "On (1:1 refresh)",
                                            "Half rate (1:2)", "Third (1:3)", "Quarter (1:4)" };
                int vs_idx = static_cast<int>(std::min<u32>(sw->vsync_interval(), 4u));
                ImGui::SetNextItemWidth(180.0f);
                if (ImGui::Combo("VSync mode", &vs_idx, vs_labels, IM_ARRAYSIZE(vs_labels))) {
                    sw->set_vsync_interval(static_cast<u32>(vs_idx));
                }
                ImGui::TextDisabled("  off uses DXGI_PRESENT_ALLOW_TEARING for true uncapped");

                ImGui::SeparatorText("NVIDIA Reflex");
                if (sw->reflex_supported()) {
                    const char* rfx_labels[] = { "Off", "On", "On + Boost" };
                    int rfx_idx = static_cast<int>(sw->reflex_mode());
                    ImGui::SetNextItemWidth(180.0f);
                    if (ImGui::Combo("Reflex mode", &rfx_idx, rfx_labels, IM_ARRAYSIZE(rfx_labels))) {
                        sw->set_reflex_mode(static_cast<rhi::Swapchain::ReflexMode>(rfx_idx));
                    }
                    int cap = static_cast<int>(sw->reflex_fps_cap());
                    bool cap_on = cap > 0;
                    if (ImGui::Checkbox("Reflex FPS cap", &cap_on)) {
                        sw->set_reflex_fps_cap(cap_on ? 144u : 0u);
                    }
                    if (cap_on) {
                        ImGui::SameLine(); ImGui::SetNextItemWidth(120.0f);
                        if (ImGui::SliderInt("##rfx_cap", &cap, 30, 480, "%d FPS")) {
                            sw->set_reflex_fps_cap(static_cast<u32>(cap));
                        }
                    }
                    auto lat = sw->reflex_latency();
                    if (lat.valid) {
                        ImGui::TextDisabled("  total %.2f ms = sim %.2f + render %.2f + present %.2f",
                                            lat.total_latency_ms, lat.sim_ms, lat.render_ms, lat.present_ms);
                        ImGui::TextDisabled("  driver %.2f / queue %.2f / GPU %.2f ms",
                                            lat.driver_ms, lat.os_render_queue_ms, lat.gpu_ms);
                    } else {
                        ImGui::TextDisabled("  (gathering latency samples — needs ~90 frames)");
                    }
                } else {
                    ImGui::TextDisabled("  Reflex unavailable (needs NVIDIA + NvAPI)");
                }
                ImGui::Separator();

                // FPS pacing controls — main-loop foreground / background.
                // A value of 0 means "uncapped" for that profile.
                ImGui::SeparatorText("Frame pacing");
                cardinal::core::FrameLimit cur = pacer->limit(cardinal::core::kPaceMainLoop);
                bool fg_unl = cur.fps_foreground <= 0.0f;
                if (ImGui::Checkbox("Foreground: Unlimited", &fg_unl)) {
                    cur.fps_foreground = fg_unl ? 0.0f : 144.0f;
                    pacer->set_limit(cardinal::core::kPaceMainLoop, cur);
                }
                if (!fg_unl) {
                    ImGui::SetNextItemWidth(180.0f);
                    if (ImGui::SliderFloat("Foreground FPS", &cur.fps_foreground, 15.0f, 480.0f, "%.0f")) {
                        pacer->set_limit(cardinal::core::kPaceMainLoop, cur);
                    }
                }
                bool bg_unl = cur.fps_background <= 0.0f;
                if (ImGui::Checkbox("Background: Unlimited", &bg_unl)) {
                    cur.fps_background = bg_unl ? 0.0f : 30.0f;
                    pacer->set_limit(cardinal::core::kPaceMainLoop, cur);
                }
                if (!bg_unl) {
                    ImGui::SetNextItemWidth(180.0f);
                    if (ImGui::SliderFloat("Background FPS", &cur.fps_background, 5.0f, 60.0f, "%.0f")) {
                        pacer->set_limit(cardinal::core::kPaceMainLoop, cur);
                    }
                }
                ImGui::TextDisabled("  Window %s — capped to %.0f FPS",
                                    pacer->is_foreground() ? "FOCUSED" : "background",
                                    pacer->effective_fps_cap());

                ImGui::Separator();
                ImGui::SetNextItemWidth(180.0f);
                ImGui::SliderFloat("Camera speed", &fly_cam.speed,
                                   fly_cam.speed_min, fly_cam.speed_max,
                                   "%.2f units/s", ImGuiSliderFlags_Logarithmic);
                ImGui::SetNextItemWidth(180.0f);
                ImGui::SliderFloat("Mouse sensitivity", &fly_cam.look_sensitivity,
                                   0.0005f, 0.02f, "%.4f rad/px");
                ImGui::TextDisabled("  WASD move / Q E down up / RMB look / Shift sprint / wheel speed");
                ImGui::EndMenu();
            }

            // ---- Tool radio — Select pulls up the gizmo + click-to-pick;
            // Place uses the active asset id from the Palette and drops it
            // wherever the user clicks on the Y=0 plane.
            ImGui::TextDisabled(" | Tool:");
            if (ImGui::RadioButton("Select", tool == Tool::Select)) {
                tool = Tool::Select;
                placement_asset_id.clear();
            }
            if (ImGui::RadioButton("Place", tool == Tool::PlaceAsset)) {
                tool = Tool::PlaceAsset;
            }
            if (tool == Tool::PlaceAsset) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.4f, 1.0f), "(%s)",
                    placement_asset_id.empty() ? "pick from palette"
                                               : placement_asset_id.c_str());
            }
            ImGui::EndMenuBar();
        }
        ImGui::End();   // ##dockhost

        // ---- Fly camera input ---------------------------------------------
        //
        // The FlyCam runs in EVERY game state — Stopped, Paused, Playing.
        // It's the editor's free-look navigation camera (UE5 / Unity
        // editor convention) and the user needs it to position the camera
        // for level editing whether the sim is running or not.
        //
        // The "only capture mouse when simulating" rule applies to
        // GAMEPLAY input (the PIE input gate) — not to editor camera
        // navigation. WASD here drives the editor cam; gameplay-side
        // WASD bindings are gated separately by input_mgr->set_gameplay_
        // active() further down.
        //
        // RMB serves double duty: bare RMB-drag = camera look-mode, and
        // Alt+RMB = viewport context menu (handled inside Studio).
        {
            const ImGuiIO& io = ImGui::GetIO();
            const ImVec2  mp  = io.MousePos;

            // Possessed? (Playing + pawn alive + has a controller.)
            cardinal::actor::PlayerControllerComponent* pc = nullptr;
            if (game.state() == cardinal::game::GameState::Playing &&
                player_id != 0)
            {
                if (auto* pa = aworld.find(player_id))
                    pc = pa->get_component<
                        cardinal::actor::PlayerControllerComponent>();
            }
            const bool possessing = (pc != nullptr);

            // Smooth camera hand-off on the possession edge.
            if (possessing && !was_possessing) {
                pc->sync_look_from_forward(scn::normalize(
                    scene.camera().target - scene.camera().position));
            } else if (!possessing && was_possessing) {
                fly_cam.sync_from(scene.camera());
            }
            was_possessing = possessing;

            const bool ui_capture =
                ImGui::IsAnyItemActive() || io.WantCaptureKeyboard;

            if (possessing) {
                // ---- Possessed player — first-person control --------
                cardinal::actor::PlayerInput pin{};
                pin.accept_input = !ui_capture;
                pin.move_z = (ImGui::IsKeyDown(ImGuiKey_W) ? 1.0f : 0.0f)
                           - (ImGui::IsKeyDown(ImGuiKey_S) ? 1.0f : 0.0f);
                pin.move_x = (ImGui::IsKeyDown(ImGuiKey_D) ? 1.0f : 0.0f)
                           - (ImGui::IsKeyDown(ImGuiKey_A) ? 1.0f : 0.0f);
                pin.jump   = ImGui::IsKeyDown(ImGuiKey_Space);
                pin.sprint = ImGui::IsKeyDown(ImGuiKey_LeftShift) ||
                             ImGui::IsKeyDown(ImGuiKey_RightShift);
                pin.look   = io.MouseDown[1];   // RMB look (editor-cam UX)
                if (pin.look) {
                    pin.mouse_dx = mp.x - last_mouse_pos.x;
                    pin.mouse_dy = mp.y - last_mouse_pos.y;
                }
                last_mouse_pos = mp;
                pc->tick(dt, pin);
                scene.camera().position = pc->camera_eye();
                scene.camera().target   = pc->camera_target();
                scene.camera().up       = {0.0f, 1.0f, 0.0f};

                // ---- Optional: play as a *client* player ------------
                // Route the pawn transform through the real netcode the
                // exact way an MMO build does — server_broadcast over the
                // transport, then client_ingest back — only the factory
                // differs (create_loopback vs create_udp). Lossless
                // loopback, so the round-tripped transform == authored;
                // the point is to exercise + prove the client path live.
                if (net_loopback) {
                    if (!net_loop) {
                        net_loop = cardinal::net::Transport::create_loopback();
                        if (net_loop) {
                            net_loop->listen(0);
                            net_repl = cardinal::make_unique<
                                cardinal::net::Replicator>(*net_loop);
                        }
                    }
                    if (net_repl) {
                        if (auto* pa = aworld.find(player_id)) {
                            auto* tr = pa->get_component<
                                cardinal::actor::TransformComponent>();
                            if (tr != nullptr) {
                                // Receive last frame's snapshot (client).
                                net_events.clear();
                                net_loop->poll(net_events);
                                net_rx.clear();
                                if (net_repl->client_ingest(net_events,
                                                            net_rx) > 0) {
                                    for (const auto& rs : net_rx) {
                                        if (rs.id != player_id) continue;
                                        tr->translation    = rs.position;
                                        tr->rotation_euler = rs.rotation_euler;
                                        tr->scale          = rs.scale;
                                    }
                                    scene.camera().position = {
                                        tr->translation.x,
                                        tr->translation.y + pc->eye_height,
                                        tr->translation.z };
                                }
                                // Broadcast this frame's authoritative
                                // state (server) for next tick.
                                net_tx.clear();
                                cardinal::net::RepState s;
                                s.id             = player_id;
                                s.position       = tr->translation;
                                s.rotation_euler = tr->rotation_euler;
                                s.scale          = tr->scale;
                                net_tx.push_back(s);
                                net_repl->server_broadcast(net_tx);
                            }
                        }
                    }
                } else if (net_loop) {
                    net_repl.reset();   // toggled off — tear loopback down
                    net_loop.reset();
                }
            } else {
                // ---- Editor fly-cam (unchanged behaviour) -----------
                scn::FlyInput in{};
                in.accept_input = ui_capture ? false :
                    ImGui::IsMouseHoveringRect(ImVec2(0,0),
                                               io.DisplaySize, false);
                in.look = io.MouseDown[1];
                if (in.look) {
                    in.mouse_dx = mp.x - last_mouse_pos.x;
                    in.mouse_dy = mp.y - last_mouse_pos.y;
                    in.accept_input = true;   // RMB always grabs input
                }
                last_mouse_pos = mp;
                in.forward  = ImGui::IsKeyDown(ImGuiKey_W);
                in.backward = ImGui::IsKeyDown(ImGuiKey_S);
                in.left     = ImGui::IsKeyDown(ImGuiKey_A);
                in.right    = ImGui::IsKeyDown(ImGuiKey_D);
                in.up       = ImGui::IsKeyDown(ImGuiKey_E);
                in.down     = ImGui::IsKeyDown(ImGuiKey_Q);
                in.sprint   = ImGui::IsKeyDown(ImGuiKey_LeftShift) ||
                              ImGui::IsKeyDown(ImGuiKey_RightShift);
                in.scroll   = io.MouseWheel;
                fly_cam.tick(scene.camera(), in, dt);
            }
        }
        // Each viewport panel renders its OWN per-id RTT now — set
        // earlier in this frame inside the per-viewport render loop, so
        // a panel set to Wireframe stays Wireframe regardless of what the
        // others show. Drag a tab off → its window pops into its own dock
        // node (or its own OS window when ConfigFlags_ViewportsEnable is
        // on). When a viewport is maximized, hide the others; the host
        // also collapses side panels in that mode (handled in the
        // "any_maximized" gate around side-panel calls below).
        const bool any_maximized = (maximized_idx >= 0);
        for (int i = 0; i < static_cast<int>(viewports.size()); ++i) {
            auto& vps = viewports[i];
            if (!vps.show) continue;
            if (any_maximized && i != maximized_idx) continue;
            // Studio's gizmo / world-grid hover crosshair / chunk-readout
            // overlays project with the camera set here, consumed
            // synchronously inside draw_viewport_panel. Give EACH panel
            // its OWN aspect — the same per-viewport extent its RTT was
            // rendered with (render loop above) and that the click pick
            // uses (pick.aspect) — so a panel in its own OS window
            // (different aspect than viewport 0) has its overlays line up
            // with the scene and with where objects actually place. The
            // single global-aspect set_viewport_camera later in the frame
            // skewed every non-main viewport's overlays; correct docked.
            {
                const u32   vw_i = sw->viewport_width (static_cast<u32>(i));
                const u32   vh_i = sw->viewport_height(static_cast<u32>(i));
                const float aspect_i = (vh_i > 0)
                    ? static_cast<float>(vw_i) / static_cast<float>(vh_i)
                    : 1.0f;
                studio->set_viewport_camera(scene.camera().view(),
                                            scene.camera().proj(aspect_i));
            }
            studio->draw_viewport_panel(
                vps.title.c_str(),
                static_cast<u32>(i),
                &vps.mode,
                &vps.fps_cap,
                &vps.maximized,
                &vps.minimized,
                &vps.show);
        }

        // Pump each viewport's chosen FPS into the pacer. We only set the
        // *foreground* cap for viewports — the background cap is the main
        // loop's responsibility (set once at boot, edited via Settings).
        // A value of 0 (Unlimited) means "no contribution to the min cap".
        // First clear every slot ID we *might* have used last frame so a
        // dropped viewport stops contributing; then push limits for every
        // currently-visible slot.
        {
            auto push_vp_limit = [&](cardinal::core::PaceContextId id, float fps, bool visible) {
                if (!visible) { pacer->clear_limit(id); return; }
                cardinal::core::FrameLimit fl{};
                fl.fps_foreground = fps;          // 0 = unlimited
                fl.fps_background = 0.0f;         // background = main loop's domain
                pacer->set_limit(id, fl);
            };
            // Sweep every slot in [1, kMaxViewports]; current slots push
            // their cap, missing/closed slots get cleared.
            for (int i = 0; i < kMaxViewports; ++i) {
                const auto id = static_cast<cardinal::core::PaceContextId>(i + 1);
                if (i < static_cast<int>(viewports.size()) && viewports[i].show) {
                    push_vp_limit(id, viewports[i].fps_cap, true);
                } else {
                    pacer->clear_limit(id);
                }
            }
        }

        // Side panels — suppressed entirely when any viewport is
        // maximized so the focused viewport really is fullscreen.
        if (!any_maximized && show_hierarchy) {
            studio->draw_scene_hierarchy(scene, &selection, "Hierarchy", &show_hierarchy);
            selected_id = selection.empty() ? 0u : selection.back();
        }
        if (!any_maximized && show_inspector) studio->draw_inspector(scene, selected_id, "Inspector", &show_inspector);
        if (!any_maximized && show_assets)    studio->draw_asset_browser(
                                std::filesystem::current_path().string().c_str(), "Assets", &show_assets);

        // Helper used by Asset Palette right-click menu, viewport context
        // menu, and elsewhere: spawn an asset at a given world point and
        // record the action in the undo stack so the user can take it back.
        auto spawn_asset_at = [&](const char* asset_id, scn::Vec3 pos) {
            if (asset_id == nullptr || *asset_id == '\0') return;
            // Route through AssetPlacement: the spawn is now a first-class
            // actor (Actor Outliner / Game / serial save-load / level all
            // see it) with a scene::Entity render mirror — not a
            // scene-only entity that the rest of the toolchain can't touch.
            auto pr = placement->place(asset_id, device.get(), pos);
            if (pr.actor == 0u) return;
            selected_id = pr.primary_entity;

            // Undo/redo removes + re-places the whole placement (actor +
            // render entities) as a unit. A shared cell tracks the live
            // actor id because each redo mints a fresh actor/entity set.
            auto live = std::make_shared<cardinal::actor::ActorId>(pr.actor);
            std::string  id_copy = asset_id;
            scn::Vec3    at      = pos;
            auto*        pl      = placement.get();
            rhi::Device* dev     = device.get();
            edit::Command cmd;
            cmd.label  = "Spawn " + id_copy;
            cmd.revert = [live, pl]() {
                if (*live != 0u) { pl->remove(*live); *live = 0u; }
            };
            cmd.apply  = [live, pl, id_copy, at, dev, &selected_id]() {
                auto rr = pl->place(id_copy.c_str(), dev, at);
                *live = rr.actor;
                if (rr.primary_entity) selected_id = rr.primary_entity;
            };
            undo.push_executed(std::move(cmd));
        };

        // Asset Palette — left-click sets place tool, right-click context
        // menu lets the user spawn directly at camera or world origin.
        if (!any_maximized && show_palette) {
            studio->draw_asset_palette(
                [&](const char* asset_id) {
                    placement_asset_id = asset_id ? asset_id : "";
                    tool = Tool::PlaceAsset;
                },
                [&](const char* asset_id, int where) {
                    scn::Vec3 pos{ 0, 0, 0 };
                    if (where == 0) {
                        const auto fwd = scn::normalize(scene.camera().target - scene.camera().position);
                        pos = scene.camera().position + fwd * 2.5f;
                        pos.y = std::max(0.0f, pos.y);
                    }
                    spawn_asset_at(asset_id, pos);
                },
                placement_asset_id.empty() ? nullptr : placement_asset_id.c_str(),
                "Asset Palette", &show_palette);
        }

        // ----- Game-engine toolbar (Tool / Snap / Undo / Play / Camera).
        {
            tbar.tool = (tool == Tool::PlaceAsset) ? 1 : 0;
            tbar.snap_enabled = snap_enabled;
            tbar.snap_step    = snap_step;

            ui::Studio::UndoButtons ub{};
            ub.can_undo    = undo.can_undo();
            ub.can_redo    = undo.can_redo();
            ub.undo_label  = undo.next_undo_label();
            ub.redo_label  = undo.next_redo_label();

            studio->draw_main_toolbar(&tbar, &ub, "Toolbar", nullptr);

            // Toolbar → app state.
            tool          = (tbar.tool == 1) ? Tool::PlaceAsset : Tool::Select;
            snap_enabled  = tbar.snap_enabled;
            snap_step     = tbar.snap_step;
            if (ub.undo_clicked || ub.redo_clicked) {
                cardinal::cmd::CommandContext ucx{};
                ucx.scene_undo = &undo;
                commands.dispatch(ub.undo_clicked ? "edit.undo" : "edit.redo", ucx);
            }
            if (tbar.focus_selection && selected_id != 0) {
                if (auto* e = scene.find_by_id(selected_id)) {
                    scene.camera().target = e->transform.translation;
                }
            }
            if (tbar.reset_camera) {
                scene.camera().position = { 0.0f, 1.5f, 4.0f };
                scene.camera().target   = { 0.0f, 0.5f, 0.0f };
                fly_cam.sync_from(scene.camera());
            }
        }

        // Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y hotkeys (only when not typing).
        {
            const ImGuiIO& io = ImGui::GetIO();
            if (!io.WantCaptureKeyboard && (io.KeyCtrl || io.KeySuper)) {
                const bool z = ImGui::IsKeyPressed(ImGuiKey_Z, false);
                const bool y = ImGui::IsKeyPressed(ImGuiKey_Y, false);
                // Undo/redo route through the command bus (edit.undo/redo) —
                // one dispatch path shared by the toolbar buttons + palette.
                cardinal::cmd::CommandContext ucx{};
                ucx.scene_undo = &undo;
                if (z && io.KeyShift)      commands.dispatch("edit.redo", ucx);
                else if (z)                commands.dispatch("edit.undo", ucx);
                else if (y)                commands.dispatch("edit.redo", ucx);
                if (ImGui::IsKeyPressed(ImGuiKey_P, false))
                    show_cmd_palette = !show_cmd_palette;          // Ctrl+P
            }
        }

        // ---- Command palette (Ctrl+P) — the unified command bus surfaced as
        // a searchable list. EVERY registered command is reachable + callable
        // here through the one registry, with a host-built CommandContext
        // (actor world + scene + placement + current selection). Commands that
        // need inputs the palette can't supply (e.g. a viewport click for
        // world.place_asset) fail gracefully with a message rather than crash.
        if (show_cmd_palette) {
            ImGui::SetNextWindowSize(ImVec2(440, 380), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Command Palette", &show_cmd_palette)) {
                static char cmd_filter[64] = "";
                ImGui::SetNextItemWidth(-1.0f);
                ImGui::InputTextWithHint("##cmdfilter", "filter commands…",
                                         cmd_filter, sizeof(cmd_filter));
                ImGui::TextDisabled("context: %s selected · %zu commands",
                                    selected_actor_id ? "1 actor" : "no",
                                    commands.ids().size());
                ImGui::Separator();
                auto lower = [](cardinal::string s) {
                    for (auto& ch : s)
                        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
                    return s;
                };
                const cardinal::string flt = lower(cmd_filter);
                for (const auto& id : commands.ids()) {
                    const auto* c = commands.find(id);
                    const cardinal::string label = c ? c->label : id;
                    if (!flt.empty() &&
                        lower(id).find(flt) == cardinal::string::npos &&
                        lower(label).find(flt) == cardinal::string::npos)
                        continue;
                    const cardinal::string row = label + "   (" + id + ")";
                    if (ImGui::Selectable(row.c_str())) {
                        cardinal::cmd::CommandContext cx{};
                        cx.world      = &aworld;
                        cx.scene      = &scene;
                        cx.placement  = placement.get();
                        cx.scene_undo = &undo;
                        cx.device     = device.get();
                        if (selected_actor_id != 0)
                            cx.selection = { selected_actor_id };
                        const auto r = commands.dispatch(id, cx);
                        clog::infof("palette", "%s -> %s (%s) [n=%u]",
                            id.c_str(), r.ok ? "ok" : "FAIL",
                            r.message.empty() ? "-" : r.message.c_str(),
                            cx.result_count);
                    }
                }
            }
            ImGui::End();
        }

        // Terrain Grid spawn modal — opens on Create→Terrain Grid…, executes
        // the spawn the frame the user hits Generate.
        {
            ui::Studio::TerrainSpawnRequest req{};
            studio->draw_terrain_modal(open_terrain_modal, &req);
            open_terrain_modal = false;       // one-shot
            if (req.requested) {
                // Look the chosen profile up in the library and route through
                // the multi-layer profile spawner. If the user un-typed the
                // name we fall back to the first registered profile.
                auto& lib = scn::TerrainProfileLibrary::instance();
                const scn::TerrainProfile* p = lib.find(req.profile_name.c_str());
                if (p == nullptr && !lib.all().empty()) p = &lib.all().front();
                if (p != nullptr) {
                    scn::TerrainGridProfileDesc d{};
                    d.profile    = *p;
                    d.chunks_x   = req.chunks_x;
                    d.chunks_z   = req.chunks_z;
                    d.resolution = req.resolution;
                    d.chunk_size = req.chunk_size;
                    auto ids = scn::spawn_terrain_grid_profile(scene, *device, d);
                    clog::infof("sample",
                        "terrain grid spawned — profile '%s', %zu chunks (%dx%d, %u verts/side, %.1fu/chunk)",
                        p->name.c_str(), ids.size(), req.chunks_x, req.chunks_z,
                        req.resolution, req.chunk_size);
                    if (!ids.empty()) selected_id = ids.front();
                } else {
                    clog::warnf("sample", "terrain spawn: no profile in library");
                }
            }
        }

        // Push camera + gizmo target into Studio so its viewport overlay
        // can render the 3-axis translate handle on the selected entity.
        {
            // Camera/projection is set PER PANEL in the viewport draw loop
            // above — scene.camera().proj(aspect_i) for each panel's own
            // extent, immediately before that panel's draw_viewport_panel,
            // which is what the gizmo / world-grid / hover-chunk /
            // gizmo-drag overlays consume. The single global-aspect
            // set_viewport_camera(proj(aspect_g)) that used to live here
            // was the source of the detached-window overlay skew (a
            // torn-out panel's aspect != viewport 0's); it is redundant
            // (always overwritten by the per-panel call before any
            // consumer reads gizmo_proj_) and is removed. Do NOT reinstate
            // a global-aspect camera here.
            studio->set_gizmo_snap(snap_enabled, snap_step);
            // Reconcile: the context menu / spawn / toolbar paths still
            // mutate the single `selected_id`. Fold that back into the
            // multi-select set so the gizmo + fan-out stay coherent
            // without touching all ~10 single-id call sites.
            if (selected_id == 0) {
                selection.clear();
            } else {
                bool present = false;
                for (u32 s : selection) if (s == selected_id) { present = true; break; }
                if (!present) { selection.clear(); selection.push_back(selected_id); }
            }
            // Pivot = selection centroid; rotate/scale anchor uses the
            // PRIMARY entity's orientation/scale so the gizmo orients to
            // it. Empty selection → no gizmo.
            const auto* prim = scene.find_by_id(selected_id);
            if (prim != nullptr && !selection.empty()) {
                scn::Vec3 c{0,0,0};
                u32 n = 0;
                for (u32 id : selection)
                    if (auto* e = scene.find_by_id(id)) {
                        c.x += e->transform.translation.x;
                        c.y += e->transform.translation.y;
                        c.z += e->transform.translation.z;
                        ++n;
                    }
                if (n > 0) {
                    const float inv = 1.0f / static_cast<float>(n);
                    c.x *= inv; c.y *= inv; c.z *= inv;
                }
                studio->set_gizmo_transform(c, prim->transform.rotation_euler,
                                            prim->transform.scale, true);
            } else {
                studio->set_gizmo_transform({0,0,0}, {0,0,0}, {1,1,1}, false);
            }
        }

        if (!any_maximized && show_console) {
            // Two-tier dispatch: try the engine console (cvars/commands)
            // first; on miss, fall through to the Lua REPL. The output sink
            // accumulates each line so the panel sees the full transcript.
            studio->draw_console_panel(
                [&](const char* line) -> std::string {
                    if (line == nullptr || *line == '\0') return {};
                    std::string acc;
                    cv::Output sink([&](const std::string& s) {
                        if (!acc.empty()) acc.push_back('\n');
                        acc += s;
                    });
                    if (cv::Registry::instance().execute(line, sink)) {
                        return acc;
                    }
                    // Fall-through: hand the unrecognised line to Lua so
                    // scripted commands still work alongside CVars.
                    return lua->repl_eval(line);
                },
                "Console", &show_console);
        }

        if (!any_maximized && show_log)      studio->draw_log_panel  ("Log",   &show_log);
        if (!any_maximized && show_stats)    studio->draw_stats_panel("Stats", &show_stats);
        if (!any_maximized && show_pipeline) studio->draw_render_pipeline_panel(*pipelines, "Render Pipeline", &show_pipeline);
        if (!any_maximized && show_memory)   studio->draw_memory_panel(&budget_broker, "Memory & Budgets", &show_memory);
        if (!any_maximized && show_profiler) studio->draw_profiler_panel(last_frame_ms_for_profiler, "Profiler", &show_profiler);
        if (!any_maximized && show_vt)       studio->draw_vt_panel(vt_system.get(), "Virtual Textures", &show_vt);
        // Game logic (auto-checkpoint debounce + Play/Pause/Stop hotkeys + PIE
        // lifecycle) runs EVERY frame, unconditionally — NOT gated by the Game
        // panel's visibility — so undo checkpoints + hotkeys keep working when
        // the panel is closed or a viewport is maximized. Only the bar's
        // widgets are visibility-gated.
        studio->update_game_logic(&game);
        if (!any_maximized && show_game)     studio->draw_game_bar_panel(&game, "Game", &show_game);
        if (!any_maximized && show_classes)  studio->draw_class_picker_panel(&game, &selected_actor_id, "Game Classes", &show_classes);
        if (!any_maximized && show_sky)      studio->draw_sky_panel(&sky, "Sky / Time of Day", &show_sky);
        if (!any_maximized && show_sim)      studio->draw_sim_bar_panel(&sim_world, "Simulation", &show_sim);
        if (!any_maximized && show_actors)   studio->draw_actor_outliner_panel(&aworld, &selected_actor_id, "Actors", &show_actors);
        if (!any_maximized && show_prefabs)  studio->draw_prefab_panel(&aworld, &selected_actor_id, "Prefabs", &show_prefabs);
        if (!any_maximized && show_curve)    studio->draw_curve_editor_panel(bob_curve_handle, "Curve Editor", &show_curve);
        if (!any_maximized && show_sequencer)studio->draw_sequencer_panel(&cine_player, "Sequencer", &show_sequencer);
        if (!any_maximized && show_mixer)    studio->draw_mixer_panel(audio_engine.get(), "Mixer", &show_mixer);
        if (!any_maximized && show_input)    studio->draw_input_panel(input_mgr.get(), "Input", &show_input);
        if (!any_maximized && show_save_load)studio->draw_save_load_panel(&game, &sky, "Save / Load", &show_save_load);
        if (!any_maximized && show_particles)studio->draw_particles_panel(particles_sys.get(), "Particles", &show_particles);
        if (!any_maximized && show_nav)      studio->draw_nav_panel(&nav_grid, &nav_state, "Navigation", &show_nav);

        // ----- Project / Cook & Pack / Shader Compiler ----------------
        if (!any_maximized && show_project) {
            const auto act = studio->draw_project_panel(&current_project,
                                                        &recent_projects,
                                                        "Project", &show_project);
            // After "Cook & Pack" button, optionally auto-mount the new
            // pack into the asset registry so subsequent loads pull from it.
            if (act.cook_and_pack_clicked && current_project) {
                const std::string pack_path = current_project->dirs().pack
                    + "/" + current_project->info().default_pack_name + ".cpk";
                auto archive = cardinal::pack::Archive::open(pack_path);
                if (archive) {
                    asset_registry->clear_mounts();
                    asset_registry->mount_archive(archive);
                    clog::infof("sample",
                        "asset registry now mounted from %s (%zu entries)",
                        pack_path.c_str(), archive->entries().size());
                }
            }
            // Opening / creating a project loads its runtime world snapshot
            // (ProjectInfo::startup_world) straight into the live game, so
            // a freshly created project is immediately runnable — press
            // Play and the loaded actors simulate. Anonymous transform-only
            // actors load on any runtime; class-bound actors bind once the
            // project's game DLL is present (skipped gracefully otherwise).
            if (act.opened) {
                const auto& pinfo = act.opened->info();
                const std::string world_path =
                    act.opened->dirs().root + "/" + pinfo.startup_world;
                std::string werr;
                const auto ls = cardinal::serial::load_world(
                    game, world_path, /*replace_existing=*/true, &werr);
                if (werr.empty())
                    clog::infof("sample",
                        "project '%s' opened — startup world '%s': "
                        "%u actors, %u props, %u skipped",
                        pinfo.name.c_str(), pinfo.startup_world.c_str(),
                        ls.actors_spawned, ls.properties_applied,
                        ls.actors_skipped);
                else
                    clog::warnf("sample",
                        "project world load failed (%s): %s",
                        world_path.c_str(), werr.c_str());
            }
        }
        if (!any_maximized && show_cook_pack) {
            studio->draw_cook_pack_panel(current_project.get(),
                                          &cooker_registry,
                                          shader_compiler.get(),
                                          "Cook & Pack", &show_cook_pack);
        }
        if (!any_maximized && show_shader_cc) {
            studio->draw_shader_compiler_panel(shader_compiler.get(),
                                                "Shader Compiler",
                                                &show_shader_cc);
        }
        // Drive shader hot-reload polling once per frame.
        if (shader_compiler) shader_compiler->tick();

        // ----- World systems tick + panel ------------------------------
        if (io_dispatcher) io_dispatcher->tick();

        // Update the partition's main viewer from the camera.
        if (world_partition) {
            cardinal::partition::Viewer v{};
            v.position = scene.camera().position;
            v.forward.x = scene.camera().target.x - scene.camera().position.x;
            v.forward.y = scene.camera().target.y - scene.camera().position.y;
            v.forward.z = scene.camera().target.z - scene.camera().position.z;
            const f32 aspect = (sw->viewport_height() > 0)
                ? static_cast<f32>(sw->viewport_width()) /
                  static_cast<f32>(sw->viewport_height()) : 1.0f;
            v.frustum = cardinal::core::geom::Frustum::from_view_proj(
                scene.camera().proj(aspect) * scene.camera().view());
            world_partition->update_viewer(main_viewer_id, v);
            world_partition->tick();
        }

        // ----- Independent per-frame ticks: phase A -----------------------
        //
        // These two ticks (mass drift + AI perception) WERE wrapped in a
        // parallel_invoke earlier, but for sample-scale work they're each
        // microseconds — well below the JobSystem's dispatch + main-thread
        // wait round-trip (~50-200 µs). Sequential is faster here. When
        // mass entity counts grow into the thousands, the right move is
        // to parallelise INSIDE mass_world->for_each via parallel_for over
        // chunks (the chunked archetype layout is built for that), not to
        // fan out at this level.
        cam_sensor.position = scene.camera().position;
        cam_sensor.forward.x = scene.camera().target.x - scene.camera().position.x;
        cam_sensor.forward.y = scene.camera().target.y - scene.camera().position.y;
        cam_sensor.forward.z = scene.camera().target.z - scene.camera().position.z;
        ai_perception.update_sensor(cam_sensor_id, cam_sensor);

        if (mass_world) {
            mass_world->for_each(
                (1u << ctf_bit) | (1u << cvl_bit),
                [&](cardinal::mass::EntityId e) {
                    auto* tr = mass_world->get<cardinal::mass::CTransform>(e);
                    auto* vl = mass_world->get<cardinal::mass::CVelocity>(e);
                    if (tr && vl) {
                        tr->position.x += vl->v.x * dt;
                        tr->position.z += vl->v.z * dt;
                    }
                });
        }
        ai_perception.tick(dt);

        if (!any_maximized && show_world_sys) {
            ui::Studio::WorldSystemsInputs inputs{};
            inputs.io_dispatcher = io_dispatcher.get();
            inputs.partition     = world_partition.get();
            inputs.level_manager = level_manager.get();
            inputs.hlod_tree     = &hlod_tree;
            inputs.mass_world    = mass_world.get();
            inputs.ai_perception = &ai_perception;
            inputs.navmesh       = &nav_mesh;
            studio->draw_world_systems_panel(inputs, "World Systems", &show_world_sys);
        }
        if (!any_maximized && show_modes)    studio->draw_editor_modes_panel(&editor_state, "Modes", &show_modes);
        if (!any_maximized && show_brush)    studio->draw_brush_panel(&active_brush, "Brush", &show_brush);
        if (!any_maximized && show_mesh_tools) {
            const cardinal::scene::Entity* sel_e = scene.find_by_id(selected_id);
            const bool has_mesh = (sel_e != nullptr && sel_e->mesh != nullptr);
            const ui::Studio::MeshToolsRequest mtr =
                studio->draw_mesh_tools_panel(has_mesh, "Mesh Tools", &show_mesh_tools);

            // Apply: generate primitive into a fresh entity.
            if (mtr.generate) {
                std::vector<cardinal::scene::Vertex> verts;
                using K = ui::Studio::PrimitiveKind;
                using namespace cardinal::edit::mesh_ops;
                switch (mtr.prim) {
                    case K::Box:      verts = make_box     (mtr.size); break;
                    case K::Plane:    verts = make_plane   (mtr.size, mtr.segments); break;
                    case K::Sphere:   verts = make_sphere  (mtr.radius, mtr.segments); break;
                    case K::Cylinder: verts = make_cylinder(mtr.radius, mtr.height, mtr.segments); break;
                    case K::Cone:     verts = make_cone    (mtr.radius, mtr.height, mtr.segments); break;
                    case K::Torus:    verts = make_torus   (mtr.radius, mtr.minor, mtr.segments,
                                                            std::max(4u, mtr.segments / 2u)); break;
                    case K::Disk:     verts = make_disk    (mtr.radius, mtr.segments); break;
                }
                if (!verts.empty()) {
                    auto m = cardinal::scene::Mesh::from_vertices(*device,
                        verts.data(), static_cast<u32>(verts.size()));
                    if (m != nullptr) {
                        const char* mname = "Primitive";
                        switch (mtr.prim) {
                            case K::Box:      mname = "Box"; break;
                            case K::Plane:    mname = "Plane"; break;
                            case K::Sphere:   mname = "Sphere"; break;
                            case K::Cylinder: mname = "Cylinder"; break;
                            case K::Cone:     mname = "Cone"; break;
                            case K::Torus:    mname = "Torus"; break;
                            case K::Disk:     mname = "Disk"; break;
                        }
                        m->set_name(mname);
                        auto& e = scene.add_entity(mname);
                        e.mesh = std::move(m);
                        e.transform.translation = scene.camera().position;
                        selected_id = e.id;
                    }
                }
            }

            // Apply mesh ops to selected entity. We pull the mesh's CPU
            // shadow, run the op, upload as a fresh Mesh, and swap it in.
            auto apply_op = [&](auto&& op_fn) {
                cardinal::scene::Entity* e = scene.find_by_id(selected_id);
                if (e == nullptr || e->mesh == nullptr) return;
                const cardinal::scene::Vertex* src = e->mesh->cpu_vertices();
                if (src == nullptr) return;
                const u32 vc = e->mesh->vertex_count();
                std::vector<cardinal::scene::Vertex> in(src, src + vc);
                std::vector<cardinal::scene::Vertex> out = op_fn(std::move(in));
                if (out.empty()) return;
                auto nm = cardinal::scene::Mesh::from_vertices(*device, out.data(),
                    static_cast<u32>(out.size()));
                if (nm != nullptr) e->mesh = std::move(nm);
            };
            if (mtr.subdivide) apply_op([&](auto v) {
                return cardinal::edit::mesh_ops::subdivide(v, mtr.subdivide_levels); });
            if (mtr.mirror)    apply_op([&](auto v) {
                return cardinal::edit::mesh_ops::mirror(v, mtr.mirror_axis); });
            if (mtr.decimate)  apply_op([&](auto v) {
                return cardinal::edit::mesh_ops::decimate_cluster(v, mtr.decimate_cell); });
            if (mtr.smooth)    apply_op([&](auto v) {
                return cardinal::edit::mesh_ops::smooth_laplacian(v,
                    mtr.smooth_iters, mtr.smooth_lambda); });
            if (mtr.recompute_normals) apply_op([&](auto v) {
                return cardinal::edit::mesh_ops::recompute_normals(v, mtr.recompute_smooth); });
        }
        if (!any_maximized && show_tex_tools) {
            const auto ttr = studio->draw_texture_tools_panel("Texture Tools", &show_tex_tools);
            if (ttr.export_now) {
                clog::infof("sample", "exported texture: %s", ttr.export_path.c_str());
            }
        }
        if (!any_maximized && show_cppscript) {
            // Code Sandbox panel — surfaces the cardinal::cppscript::Engine
            // (compiler discovery, job table, diagnostics, compile/unload/
            // reload buttons). The sandbox-status callback is nullptr for
            // now: this sample doesn't yet bind a cardinal::sandbox per
            // script (the next iteration adds a checkbox in the panel for
            // "subprocess mode" + the host bridge that constructs sandboxes
            // when the user picks it).
            studio->draw_cppscript_panel(cppscript_engine.get(),
                                         "Code Sandbox", &show_cppscript,
                                         nullptr);
        }

        // ----- Input frame (resolves edges; must run BEFORE any query) ---
        input_mgr->begin_frame(dt);

        // PIE gate — gameplay actions / axes only fire while the game is
        // actually Playing (mirrors UE5 Play-In-Editor). The studio still
        // sees its own hotkeys (F5/F6/F8/ESC) because those go through
        // ImGui::IsKeyPressed, not the action map.
        input_mgr->set_gameplay_active(
            game.state() == cardinal::game::GameState::Playing);

        // ----- Quick-save / quick-load hotkeys ---------------------------
        if (input_mgr->action_pressed("QuickSave")) {
            std::string err;
            cardinal::serial::save_world(aworld,
                "save/quicksave.cardinalworld", &err);
        }
        if (input_mgr->action_pressed("QuickLoad")) {
            std::string err;
            cardinal::serial::load_world(game,
                "save/quicksave.cardinalworld", true, &err);
        }

        // ----- Particles + Sky --------------------------------------------
        //
        // Same reasoning as phase A above: each tick is microseconds, so
        // dispatching them as separate jobs costs more than it saves at
        // sample scale. Sequential. For real workloads, parallelism
        // belongs INSIDE particles_sys->tick (parallel_for over emitters)
        // and inside sky's lighting rebuild (parallel_for over actor
        // light components), not at this two-task fan-out layer.
        if (!particles_sys->emitters().empty()) {
            auto& d = particles_sys->emitters().front()->desc();
            d.origin.x = scene.camera().position.x;
            d.origin.z = scene.camera().position.z;
            d.origin.y = scene.camera().position.y - 1.0f;
        }
        particles_sys->tick(dt);
        sky.tick(dt);

        // ----- Sync the LightSet from sky + actors --------------------
        // Renderer reads light_set every frame; we rebuild here from the
        // current sky state + every actor with a LightComponent.
        light_set.clear();
        light_set.set_ambient(sky.state().ambient);
        light_set.clear_color() = sky.state().horizon;
        // Sun (skip when below the horizon — sun_intensity is 0 at night).
        if (sky.state().sun_intensity > 1e-3f) {
            cardinal::scene::Light sun{};
            sun.kind      = cardinal::scene::LightKind::Directional;
            sun.direction = sky.state().sun_dir;
            sun.color     = sky.state().sun_color;
            sun.intensity = sky.state().sun_intensity;
            light_set.add(sun);
        }
        // Mirror actor LightComponents into scene::Light entries.
        for (const auto& aptr : aworld.actors()) {
            if (!aptr->alive()) continue;
            auto* lc = aptr->get_component<cardinal::actor::LightComponent>();
            if (lc == nullptr) continue;
            auto* tr = aptr->get_component<cardinal::actor::TransformComponent>();
            cardinal::scene::Light L{};
            L.kind      = static_cast<cardinal::scene::LightKind>(lc->kind);
            L.color     = lc->color;
            L.intensity = lc->intensity;
            L.range     = lc->range;
            L.spot_inner_cos = lc->spot_inner_cos;
            L.spot_outer_cos = lc->spot_outer_cos;
            L.position  = tr ? tr->translation : cardinal::scene::Vec3{0,0,0};
            // For directional/spot lights, use the actor's forward (rotated
            // -Z). Cheap: derive from euler.
            if (tr) {
                const float cy = std::cos(tr->rotation_euler.y);
                const float sy = std::sin(tr->rotation_euler.y);
                const float cp = std::cos(tr->rotation_euler.x);
                const float sp = std::sin(tr->rotation_euler.x);
                L.direction = { -sy*cp, sp, -cy*cp };
            }
            light_set.add(L);
        }

        // ----- Audio listener follows the camera ---------------------
        if (audio_engine) {
            cardinal::audio::Listener lis{};
            lis.position = scene.camera().position;
            lis.forward.x = scene.camera().target.x - scene.camera().position.x;
            lis.forward.y = scene.camera().target.y - scene.camera().position.y;
            lis.forward.z = scene.camera().target.z - scene.camera().position.z;
            audio_engine->set_listener(lis);
        }

        // ----- Drive the simulation through the Game --------------------
        // While game.state() is Stopped, the SimWorld still ticks (so the
        // editor camera works) but PreUpdate/PrePhysics/Physics/Update
        // groups are gated off — gameplay code doesn't run. Pause leaves
        // the actors in place but freezes their on_tick.
        game.tick(dt);

        // ----- Memory pressure broker: poll OS counters + GPU budget. -----
        // Tier-transition callbacks fire from inside tick(); the world
        // streamer subsystem we registered earlier reacts by adjusting
        // render_distance.
        budget_broker.tick(/*sample_interval_ms=*/250);

        // ----- Virtual texture: predict + tick ----------------------------
        // 1. Camera-driven prefetch: synthesise tile requests around the
        //    current camera position at an altitude-derived mip.
        // 2. tick() drains the request queue, fans decode jobs to the
        //    worker pool, and commits resident tiles into the page table.
        if (vt_system) {
            const auto cp = scene.camera().position;
            vt_request_around(cp.x, cp.z, std::abs(cp.y));
            const u64 now_ms = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()).count());
            vt_system->tick(now_ms);
        }

        // Approximate per-subsystem usage so the panel's bar fills.
        // World streamer: bytes ~= active_count * (estimated bytes/chunk).
        // Real number requires a chunk-meshes-bytes accountant which lives
        // in the renderer; for now use a coarse heuristic.
        constexpr cardinal::u64 kEstBytesPerChunk = 256ull * 1024ull;
        // VT id was assigned 1 on register; world_streamer was assigned 2.
        // We reverse-look-up so the bars always match what's actually in
        // the broker.
        if (vt_system) {
            const auto vts = vt_system->stats();
            budget_broker.report_used(/*vt_id=*/1u,
                static_cast<cardinal::u64>(vts.cache.resident) *
                    cardinal::vt::kTileBytesRGBA);
            budget_broker.report_used(/*ws_id=*/2u,
                static_cast<cardinal::u64>(world_streamer.active_count()) *
                    kEstBytesPerChunk);
        } else {
            budget_broker.report_used(/*ws_id=*/1u,
                static_cast<cardinal::u64>(world_streamer.active_count()) *
                    kEstBytesPerChunk);
        }

        // ----- World streaming: tick + panel + grid overlay --------------
        // 1. Update entities' (chunk_x, chunk_y, chunk_z) membership.
        // 2. Compute camera chunk + advance the streamer (fires load/unload).
        // 3. Push the active 3D chunk set into the scene (renderer culls).
        // 4. Update the World panel readouts + apply any UI mutations back
        //    to the WorldGrid (chunk size, per-axis extents, render dists).
        // 5. Push the overlay state into Studio so the viewport draws the
        //    grid lines on the camera's current Y plane.
        cardinal::async::FrameScope _phase_world("World Streaming");
        scene.assign_chunks(world_grid.chunk_size());
        const auto cam_p = scene.camera().position;
        world_streamer.tick(cam_p.x, cam_p.y, cam_p.z);

        scene.clear_visible_chunks();
        if (wpanel.cull_outside_render_distance) {
            for (const auto& c : world_streamer.active())
                scene.mark_chunk_visible(c.x, c.y, c.z);
        }

        wpanel.camera_chunk_x     = world_streamer.camera_chunk().x;
        wpanel.camera_chunk_y     = world_streamer.camera_chunk().y;
        wpanel.camera_chunk_z     = world_streamer.camera_chunk().z;
        wpanel.active_chunk_count = static_cast<int>(world_streamer.active_count());
        if (!any_maximized && show_world) studio->draw_world_panel(&wpanel, "World", &show_world);

        // Apply panel-driven config changes (no-ops when unchanged).
        // When the user moves the render-distance slider directly we treat
        // that as the new "intended" value (the baseline the broker will
        // restore to once pressure subsides). The pressure-driven path
        // never touches `wpanel` directly — we don't want to feedback-loop
        // here.
        if (wpanel.chunk_size         != world_grid.chunk_size())         { world_grid.set_chunk_size(wpanel.chunk_size);                 world_streamer.invalidate(); }
        if (wpanel.render_distance_xz != world_grid.render_distance_xz()) {
            world_grid.set_render_distance_xz(wpanel.render_distance_xz);
            world_streamer.invalidate();
            if (!ws_overridden_by_pressure) ws_baseline_xz = wpanel.render_distance_xz;
        }
        if (wpanel.render_distance_y  != world_grid.render_distance_y())  {
            world_grid.set_render_distance_y(wpanel.render_distance_y);
            world_streamer.invalidate();
            if (!ws_overridden_by_pressure) ws_baseline_y = wpanel.render_distance_y;
        }
        if (wpanel.extent_x           != world_grid.extent_x())           { world_grid.set_extent_x(wpanel.extent_x);                     world_streamer.invalidate(); }
        if (wpanel.extent_y           != world_grid.extent_y())           { world_grid.set_extent_y(wpanel.extent_y);                     world_streamer.invalidate(); }
        if (wpanel.extent_z           != world_grid.extent_z())           { world_grid.set_extent_z(wpanel.extent_z);                     world_streamer.invalidate(); }
        if (wpanel.evict_all_clicked) world_streamer.evict_all();

        // Push the overlay state into Studio every frame.
        ui::Studio::WorldGridOverlay ov{};
        ov.show               = wpanel.show_grid_overlay;
        ov.chunk_size         = world_grid.chunk_size();
        ov.render_distance_xz = world_grid.render_distance_xz();
        ov.render_distance_y  = world_grid.render_distance_y();
        ov.camera_chunk_x     = world_streamer.camera_chunk().x;
        ov.camera_chunk_y     = world_streamer.camera_chunk().y;
        ov.camera_chunk_z     = world_streamer.camera_chunk().z;
        studio->set_world_grid_overlay(ov);

        if (show_stack_tr) studio->draw_stack_tracer_panel   ("Stack Tracer",    &show_stack_tr);
        if (show_func_tr)  studio->draw_function_tracer_panel("Function Tracer", &show_func_tr);
        if (show_dbg)      studio->draw_script_debugger_panel(*lua, "Script Debugger", &show_dbg);
        if (show_demo)     ImGui::ShowDemoWindow(&show_demo);

        studio->draw_fps_overlay();

        // Plugin tick — happens between begin_frame and end_frame so plugins
        // can interact with editor state if they want to.
        plg::Registry::instance().tick(dt);
        // Drain cppscript completions: any compile that finished on the
        // worker thread between frames gets promoted to a Registry::load
        // here, so the new plugin's on_attach runs on this thread (where
        // GPU + ImGui state is safe to touch).
        if (cppscript_engine) cppscript_engine->tick();

        if (frame == 60) clog::infof("sample", "Reached frame %u", frame);

        // ----- HUD overlay -------------------------------------------
        // Sits on top of every Studio panel + game viewport via the ImGui
        // foreground draw list. Purely visual — doesn't capture clicks —
        // but the bars + crosshair can visually obscure editor UI when the
        // game isn't actually running. Two gates:
        //   1) show_hud — user can toggle the whole overlay off entirely
        //   2) game.game_active() — even when toggled on, only render
        //      while the game is Playing or Paused. Stopped (editor mode)
        //      always gets a clean canvas.
        if (show_hud && game.game_active()) {
            const u32 sw_w = sw->width();
            const u32 sw_h = sw->height();
            hud.begin_frame(sw_w, sw_h);
            hud.bar_with_label(20.0f, sw_h - 40.0f, 220.0f, 20.0f,
                /*health=*/0.72f, "HEALTH", {200, 60, 60, 230});
            hud.bar_with_label(20.0f, sw_h - 64.0f, 180.0f, 14.0f,
                /*stamina=*/0.40f, "STAM",   {220, 200, 80, 220});
            hud.crosshair(sw_w * 0.5f, sw_h * 0.5f, 6.0f, 14.0f);

            // Minimap of nav-grid + camera marker.
            std::vector<cardinal::hud::Hud::MapMarker> markers;
            markers.push_back({static_cast<float>(nav_state.start_x),
                               static_cast<float>(nav_state.start_y),
                               {90, 220, 90, 255}});
            markers.push_back({static_cast<float>(nav_state.goal_x),
                               static_cast<float>(nav_state.goal_y),
                               {230, 110, 110, 255}});
            hud.minimap(static_cast<float>(sw_w) - 180.0f, 40.0f, 160.0f,
                        0.0f, 0.0f,
                        static_cast<float>(std::max(nav_grid.width(), nav_grid.height())),
                        markers);

            // FPS in the top-left.
            char fps_buf[32];
            std::snprintf(fps_buf, sizeof(fps_buf),
                "%.1f FPS  %.2f ms",
                dt > 0.0f ? 1.0f / dt : 0.0f, dt * 1000.0f);
            hud.text(20.0f, 20.0f, fps_buf, {200, 220, 255, 230});

            // Game-state badge.
            const char* sn = cardinal::game::game_state_name(game.state());
            hud.text(20.0f, 40.0f,
                     std::string("Game: ") + sn,
                     {200, 220, 200, 230});

            hud.tick_toasts(dt);
            hud.render_toasts(static_cast<float>(sw_w) - 280.0f,
                              static_cast<float>(sw_h) - 200.0f);
            hud.render(ImGui::GetForegroundDrawList());
        }

        studio->end_frame();

        // Writeback: the Studio gizmo / inspector edit the scene::Entity
        // (the render surface). Push those edits onto each placement's
        // authoritative actor so the Actor Outliner, Game, level, and
        // serial save-load all see the up-to-date transform/tint. Runs
        // once per frame after every panel/gizmo edit is committed.
        placement->sync_from_scene();

        // RenderSubmitEnd — closes the render-side bracket. The Present
        // Start/End markers live inside the swapchain's end_frame()/
        // present pair (they wrap DXGI's Present call to capture OS-block
        // time). After end_frame, this frame's Reflex slot is complete;
        // bump the frame_id for the next iteration.
        sw->reflex_marker(rhi::Swapchain::ReflexMarker::RenderSubmitEnd, reflex_frame);
        sw->end_frame();
        ++reflex_frame;

        // Pump ImGui's secondary OS windows (panels the user dragged out
        // of the main window become real native windows under multi-viewport).
        // Done AFTER end_frame so the main swap-present has flushed first.
        studio->update_platform_windows();

        // ---- Viewport interactions: click-to-select / click-to-place /
        //      gizmo-drag / Delete. Done AFTER end_frame so scene mutations
        //      land outside the cmd-buffer record window — they take effect
        //      on the next frame.
        {
            // ---- Gizmo: multi-mode (Move / Rotate / Scale), multi-
            //      select fan-out, drift-free, undo-coalesced. The Studio
            //      library does the ray-solve + gizmo-side snap; the host
            //      applies the absolute result by mode and fans the
            //      change across the whole selection from per-entity
            //      drag-start snapshots so one composite undo reverts it.
            const auto giz = studio->last_gizmo_drag();
            const bool drag_active = studio->gizmo_drag_in_progress();

            if (drag_active && !gizmo_drag_was_active && !selection.empty()) {
                gizmo_drag_snaps.clear();
                scn::Vec3 c0{0,0,0};
                u32 n0 = 0;
                for (u32 id : selection)
                    if (auto* e = scene.find_by_id(id)) {
                        gizmo_drag_snaps.push_back({ id,
                            e->transform.translation,
                            e->transform.rotation_euler,
                            e->transform.scale });
                        c0.x += e->transform.translation.x;
                        c0.y += e->transform.translation.y;
                        c0.z += e->transform.translation.z;
                        ++n0;
                    }
                if (n0 > 0) {
                    const float iv = 1.0f / static_cast<float>(n0);
                    c0.x *= iv; c0.y *= iv; c0.z *= iv;
                }
                gizmo_prim_p0 = c0;                       // pivot @ drag start
                if (auto* p = scene.find_by_id(selected_id)) {
                    gizmo_prim_r0 = p->transform.rotation_euler;
                    gizmo_prim_s0 = p->transform.scale;
                }
            }
            gizmo_drag_was_active = drag_active;

            if (giz.active && !gizmo_drag_snaps.empty()) {
                if (giz.mode == ui::Studio::GizmoMode::Translate) {
                    const scn::Vec3 d{
                        giz.target_world.x - gizmo_prim_p0.x,
                        giz.target_world.y - gizmo_prim_p0.y,
                        giz.target_world.z - gizmo_prim_p0.z };
                    for (auto& s : gizmo_drag_snaps)
                        if (auto* e = scene.find_by_id(s.id))
                            e->transform.translation =
                                { s.p.x+d.x, s.p.y+d.y, s.p.z+d.z };
                } else if (giz.mode == ui::Studio::GizmoMode::Rotate) {
                    const scn::Vec3 dr{
                        giz.target_rotation_euler.x - gizmo_prim_r0.x,
                        giz.target_rotation_euler.y - gizmo_prim_r0.y,
                        giz.target_rotation_euler.z - gizmo_prim_r0.z };
                    for (auto& s : gizmo_drag_snaps)
                        if (auto* e = scene.find_by_id(s.id))
                            e->transform.rotation_euler =
                                { s.r.x+dr.x, s.r.y+dr.y, s.r.z+dr.z };
                } else {  // Scale — per-axis ratio vs the primary's start.
                    auto rr = [](float a, float b) {
                        return (std::fabs(b) < 1e-6f) ? 1.0f : a / b;
                    };
                    const scn::Vec3 k{
                        rr(giz.target_scale.x, gizmo_prim_s0.x),
                        rr(giz.target_scale.y, gizmo_prim_s0.y),
                        rr(giz.target_scale.z, gizmo_prim_s0.z) };
                    for (auto& s : gizmo_drag_snaps)
                        if (auto* e = scene.find_by_id(s.id))
                            e->transform.scale =
                                { s.s.x*k.x, s.s.y*k.y, s.s.z*k.z };
                }
                // Re-anchor the gizmo to the live centroid + primary
                // orientation/scale so it tracks the moving selection.
                scn::Vec3 c{0,0,0};
                u32 n = 0;
                for (u32 id : selection)
                    if (auto* e = scene.find_by_id(id)) {
                        c.x += e->transform.translation.x;
                        c.y += e->transform.translation.y;
                        c.z += e->transform.translation.z;
                        ++n;
                    }
                if (n > 0) {
                    const float iv = 1.0f / static_cast<float>(n);
                    c.x *= iv; c.y *= iv; c.z *= iv;
                }
                if (auto* p = scene.find_by_id(selected_id))
                    studio->set_gizmo_transform(
                        c, p->transform.rotation_euler,
                        p->transform.scale, true);
            }

            // Drag-release edge → ONE composite undo for the whole
            // multi-edit (per-entity before/after full transforms).
            if (studio->gizmo_drag_release_consume()
                && !gizmo_drag_snaps.empty())
            {
                struct TX { u32 id; scn::Vec3 p, r, s; };
                std::vector<TX> before, after;
                bool changed = false;
                for (auto& sn : gizmo_drag_snaps) {
                    if (auto* e = scene.find_by_id(sn.id)) {
                        const auto& t = e->transform;
                        before.push_back({ sn.id, sn.p, sn.r, sn.s });
                        after.push_back({ sn.id, t.translation,
                                          t.rotation_euler, t.scale });
                        if (t.translation.x!=sn.p.x || t.translation.y!=sn.p.y ||
                            t.translation.z!=sn.p.z ||
                            t.rotation_euler.x!=sn.r.x || t.rotation_euler.y!=sn.r.y ||
                            t.rotation_euler.z!=sn.r.z ||
                            t.scale.x!=sn.s.x || t.scale.y!=sn.s.y ||
                            t.scale.z!=sn.s.z)
                            changed = true;
                    }
                }
                if (changed) {
                    edit::Command cmd;
                    cmd.label  = "Transform entities";
                    cmd.apply  = [after, &scene]() {
                        for (auto& a : after)
                            if (auto* x = scene.find_by_id(a.id)) {
                                x->transform.translation    = a.p;
                                x->transform.rotation_euler = a.r;
                                x->transform.scale          = a.s;
                            }
                    };
                    cmd.revert = [before, &scene]() {
                        for (auto& b : before)
                            if (auto* x = scene.find_by_id(b.id)) {
                                x->transform.translation    = b.p;
                                x->transform.rotation_euler = b.r;
                                x->transform.scale          = b.s;
                            }
                    };
                    undo.push_executed(std::move(cmd));
                }
                gizmo_drag_snaps.clear();
            }

            // ---- Viewport context menu — Studio renders the popup inside
            // draw_viewport_panel (popup must live in the viewport's
            // ImGui Begin/End scope); we just read the snapshot here.
            ui::Studio::ViewportCtxAction vca = studio->last_viewport_ctx_action();
            if (vca.delete_selected && selected_id != 0) {
                // Snapshot before delete for undo (mesh + transform + tint).
                if (auto* e = scene.find_by_id(selected_id)) {
                    const u32   eid   = selected_id;
                    const std::string name = e->name;
                    const auto  pos   = e->transform.translation;
                    const auto  scl   = e->transform.scale;
                    const auto  tnt   = e->tint;
                    auto        mesh  = e->mesh;
                    edit::Command cmd;
                    cmd.label  = "Delete " + name;
                    cmd.apply  = [eid, &scene]() { scene.remove_entity(eid); };
                    cmd.revert = [name, pos, scl, tnt, mesh, &scene]() {
                        auto& ne = scene.add_entity(name);
                        ne.transform.translation = pos;
                        ne.transform.scale       = scl;
                        ne.tint                  = tnt;
                        ne.mesh                  = mesh;
                    };
                    cmd.apply();
                    undo.push_executed(std::move(cmd));
                    selected_id = 0;
                }
            }
            if (vca.duplicate_selected && selected_id != 0) {
                if (auto* e = scene.find_by_id(selected_id)) {
                    auto& ne = scene.add_entity(e->name + " (copy)");
                    ne.transform = e->transform;
                    ne.transform.translation.x += 1.0f;
                    ne.tint = e->tint;
                    ne.mesh = e->mesh;
                    selected_id = ne.id;
                }
            }
            if (vca.focus_selected && selected_id != 0) {
                if (auto* e = scene.find_by_id(selected_id)) {
                    scene.camera().target = e->transform.translation;
                }
            }
            if (vca.clear_selection) selected_id = 0;
            if (vca.spawn_active_asset_at_cursor && !placement_asset_id.empty()) {
                // Prefer spawning at the world-grid chunk under the cursor
                // (level-editor feel) when the grid overlay is on; fall
                // back to the camera-forward position otherwise.
                const auto hover = studio->last_hover_chunk();
                scn::Vec3 pos;
                if (hover.valid) {
                    const float cs = world_grid.chunk_size();
                    pos.x = (static_cast<float>(hover.x) + 0.5f) * cs;
                    pos.y =  static_cast<float>(hover.y)         * cs;
                    pos.z = (static_cast<float>(hover.z) + 0.5f) * cs;
                } else {
                    const auto fwd = scn::normalize(scene.camera().target -
                                                     scene.camera().position);
                    pos = scene.camera().position + fwd * 2.5f;
                    pos.y = std::max(0.0f, pos.y);
                }
                spawn_asset_at(placement_asset_id.c_str(), pos);
            }

            // ---- Click-to-select / click-to-place (existing behaviour).
            // Use the HOVERED panel's own aspect — Studio reports it with
            // the pick (pick.aspect / pick.viewport_id). A viewport dragged
            // into its own OS window has a different aspect than viewport
            // 0; pairing its panel-local NDC with viewport 0's aspect
            // skewed the unprojected ray so placement landed offset. Fall
            // back to viewport 0 only when no panel was hovered.
            const auto  pick = studio->last_viewport_pick();
            float aspect_pick = pick.aspect;
            if (!(aspect_pick > 0.0f)) {
                const u32 v0w = sw->viewport_width(0u);
                const u32 v0h = sw->viewport_height(0u);
                aspect_pick = (v0w > 0 && v0h > 0)
                    ? static_cast<float>(v0w) / static_cast<float>(v0h)
                    : 1.0f;
            }
            // Resolve the CAPTURED view/proj for the hovered viewport — the
            // exact matrices the scene was rasterised with this frame (stored
            // in the render loop). Unprojecting the click with THESE instead of
            // re-deriving proj(aspect)/view from the live camera at click time
            // is what makes placement land under the cursor: no aspect-vs-NDC
            // or camera-staleness drift can creep in. Fall back to viewport 0,
            // then skip if nothing was rendered.
            cardinal::cmd::ViewProj pick_vp{};
            if (pick.viewport_id < rendered_vp.size() &&
                rendered_vp[pick.viewport_id].valid)
                pick_vp = rendered_vp[pick.viewport_id];
            else if (!rendered_vp.empty() && rendered_vp[0].valid)
                pick_vp = rendered_vp[0];

            if (pick.click_left && pick_vp.valid) {
                const auto  ray = scn::unproject_ndc_ray(pick.ndc_x, pick.ndc_y,
                                                         pick_vp.view, pick_vp.proj);

                if (tool == Tool::Select) {
                    const u32 hit_id = scn::pick_entity(scene, ray);
                    if (hit_id == 0) {
                        if (!pick.additive) selection.clear();
                        clog::infof("sample", "click missed — no entity hit");
                    } else if (pick.additive) {
                        std::vector<u32> nx;
                        nx.reserve(selection.size() + 1);
                        bool rm = false;
                        for (u32 s : selection) {
                            if (s == hit_id) { rm = true; continue; }
                            nx.push_back(s);
                        }
                        if (!rm) nx.push_back(hit_id);
                        selection.swap(nx);
                    } else {
                        selection.clear();
                        selection.push_back(hit_id);
                    }
                    selected_id = selection.empty() ? 0u : selection.back();
                } else if (tool == Tool::PlaceAsset && !placement_asset_id.empty()) {
                    // Route placement through the unified command bus. The
                    // command unprojects with the captured pick_vp + places;
                    // we keep the undo recording here (Phase-2 will fold it in).
                    cardinal::cmd::CommandContext cx{};
                    cx.placement       = placement.get();
                    cx.device          = device.get();
                    cx.viewport        = pick_vp;
                    cx.ndc_x           = pick.ndc_x;
                    cx.ndc_y           = pick.ndc_y;
                    cx.active_asset_id = placement_asset_id;
                    cx.snap_enabled    = snap_enabled;
                    cx.snap_step       = snap_step;
                    const auto r = commands.dispatch("world.place_asset", cx);
                    if (r.ok) {
                        selected_id = cx.result_entity;
                        // Undo removes the placement; redo re-places at the
                        // captured hit (mirrors spawn_asset_at).
                        auto live = std::make_shared<cardinal::actor::ActorId>(cx.result_actor);
                        std::string  id_copy = placement_asset_id;
                        scn::Vec3    at      = cx.result_hit;
                        auto*        pl      = placement.get();
                        rhi::Device* dev     = device.get();
                        edit::Command cmd;
                        cmd.label  = "Place " + id_copy;
                        cmd.revert = [live, pl]() {
                            if (*live != 0u) { pl->remove(*live); *live = 0u; }
                        };
                        cmd.apply  = [live, pl, id_copy, at, dev, &selected_id]() {
                            auto rr = pl->place(id_copy.c_str(), dev, at);
                            *live = rr.actor;
                            if (rr.primary_entity) selected_id = rr.primary_entity;
                        };
                        undo.push_executed(std::move(cmd));
                    } else if (!r.message.empty()) {
                        clog::infof("sample", "place_asset: %s", r.message.c_str());
                    }
                }
            }
            // Marquee box-select — Studio reports a panel-local NDC rect;
            // we test each entity's projected centre against it (Studio
            // stays scene-agnostic). Ctrl/Shift during the box extends.
            if (pick.marquee && aspect_pick > 0.0f) {
                const auto  mvp = scene.camera().proj(aspect_pick) * scene.camera().view();
                if (!pick.marquee_additive) selection.clear();
                for (auto& e : scene.entities()) {
                    const auto& t = e.transform.translation;
                    const scn::Vec4 c =
                        mvp * scn::Vec4{ t.x, t.y, t.z, 1.0f };
                    if (c.w <= 0.0001f) continue;
                    const float nx = c.x / c.w, ny = c.y / c.w;
                    if (nx >= pick.marquee_min_x && nx <= pick.marquee_max_x &&
                        ny >= pick.marquee_min_y && ny <= pick.marquee_max_y) {
                        bool has = false;
                        for (u32 s : selection) if (s == e.id) { has = true; break; }
                        if (!has) selection.push_back(e.id);
                    }
                }
                selected_id = selection.empty() ? 0u : selection.back();
            }

            // Delete via Del/Backspace — operates on the whole selection,
            // one composite undo record.
            if (pick.delete_pressed && !selection.empty()) {
                struct Del {
                    std::string name;
                    scn::Vec3 pos, scl, tnt;
                    std::shared_ptr<scn::Mesh> mesh;
                };
                std::vector<Del> dels;
                for (u32 id : selection)
                    if (auto* e = scene.find_by_id(id))
                        dels.push_back({ e->name, e->transform.translation,
                                         e->transform.scale, e->tint, e->mesh });
                const std::vector<u32> ids = selection;
                if (!ids.empty()) {
                    edit::Command cmd;
                    cmd.label  = "Delete selection";
                    cmd.apply  = [ids, &scene]() {
                        for (u32 id : ids) scene.remove_entity(id);
                    };
                    cmd.revert = [dels, &scene]() {
                        for (auto& d : dels) {
                            auto& ne = scene.add_entity(d.name);
                            ne.transform.translation = d.pos;
                            ne.transform.scale       = d.scl;
                            ne.tint                  = d.tnt;
                            ne.mesh                  = d.mesh;
                        }
                    };
                    cmd.apply();
                    undo.push_executed(std::move(cmd));
                }
                selection.clear();
                selected_id = 0;
            }
        }
        ++frame;
    }

    plg::Registry::instance().shutdown();
    // Drain async work + tear down the JobSystem before destroying everything
    // it might have referenced.
    cardinal::async::set_pool(nullptr);
    job_system.shutdown();

    // Detach the swapchain rebuild callback we installed earlier. The
    // callback captured `studio` by reference; once `studio` starts
    // destructing (a few lines below, when locals unwind) any residual
    // call to the callback would dereference freed memory. Clearing it
    // here closes the window. Studio's own shutdown also clears the
    // callback — this is defence-in-depth in case Studio's destructor
    // hasn't run yet at the moment a stray notify fires.
    if (sw) sw->set_on_rebuilt(nullptr);

    // Release the process-static AssetCatalog WHILE `device` is still alive.
    // register_default_assets(*device) (above) filled the catalog with
    // AssetFactory closures that capture shared_ptr<scn::Mesh>, and each
    // Mesh owns a device-bound rhi::Buffer. The catalog is a function-local
    // static, so left alone it is destroyed during C-runtime teardown —
    // AFTER main() returns and `device` (a main-scope local) is gone —
    // making ~VulkanBuffer free its VMA allocation against a dead allocator:
    //   "Assertion failed: m_pMetadata->IsEmpty() && Some allocations were
    //    not freed before destruction of this memory block" (vk_mem_alloc.h)
    // on Vulkan exit. Clearing here tears those buffers down against the
    // live device. (Engine-based apps get the symmetric clear in
    // ~EngineImpl; this manual-main sample needs its own.)
    scn::AssetCatalog::instance().clear();

    clog::infof("sample", "Shutting down (rendered %u frames)", frame);
    return 0;
}
