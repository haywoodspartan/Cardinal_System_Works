#include <cardinal/ui/studio.hpp>

#include "studio_vk_diag.hpp"      // detail::check_vk (decomposition step 1)
#include "studio_gizmo_math.hpp"   // detail::project_/mouse_world_ray_/ray_plane_ (step 2)

// Panels — grouped by Studio concern; one folder per area (src/panels/<area>/).
#include "panels/scene/actor_outliner.hpp"
#include "panels/scene/hierarchy.hpp"
#include "panels/scene/inspector.hpp"
#include "panels/scene/prefab.hpp"
#include "panels/scene/class_picker.hpp"
#include "panels/assets/project_panel.hpp"
#include "panels/assets/cook_pack.hpp"
#include "panels/assets/save_load.hpp"
#include "panels/gameplay/game_bar.hpp"
#include "panels/gameplay/sim_bar.hpp"
#include "panels/gameplay/world_systems.hpp"
#include "panels/gameplay/input_panel.hpp"
#include "panels/gameplay/nav_panel.hpp"
#include "panels/animation/sequencer.hpp"
#include "panels/animation/curve_editor.hpp"
#include "panels/animation/mixer.hpp"
#include "panels/tools/brush.hpp"
#include "panels/tools/mesh_tools.hpp"
#include "panels/tools/texture_tools.hpp"
#include "panels/tools/editor_modes.hpp"
#include "panels/rendering/sky_panel.hpp"
#include "panels/rendering/vt.hpp"
#include "panels/rendering/particles_panel.hpp"
#include "panels/scripting/cppscript.hpp"
#include "panels/scripting/shader_compiler.hpp"
#include "panels/diagnostics/log.hpp"
#include "panels/diagnostics/console.hpp"
#include "panels/diagnostics/stats.hpp"
#include "panels/diagnostics/memory.hpp"
#include "panels/diagnostics/profiler.hpp"

#include <cardinal/sandbox/sandbox.hpp>   // for Status (used by panel callback adapter)

#include <cardinal/console/console.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/platform.hpp>
#include <cardinal/render/pipeline.hpp>
#include <cardinal/rhi/rhi.hpp>
#include <cardinal/rhi/vulkan_interop.hpp>
#include <cardinal/scene/assets.hpp>
#include <cardinal/scene/scene.hpp>
#include <cardinal/scene/terrain.hpp>
#include <cardinal/script/engine.hpp>
#include <cardinal/trace/stack.hpp>
#include <cardinal/trace/timeline.hpp>
#include <cardinal/window/window.hpp>

#include <cardinal/core/std/filesystem.hpp>

#if CARDINAL_PLATFORM_WINDOWS
    #include <Windows.h>
    #include <d3d12.h>
    #include <wrl/client.h>
    #include <cardinal/rhi/d3d12_interop.hpp>
    using Microsoft::WRL::ComPtr;
#endif

#include <volk.h>
#if CARDINAL_PLATFORM_WINDOWS
    // Win32 surface KHR types — volk doesn't pull these in under VK_NO_PROTOTYPES.
    #include <vulkan/vulkan_win32.h>
#endif
#include <cardinal/ui/imgui.hpp>
#include <imgui_internal.h>          // ImGuiWindow / ImGuiContext for layout-sanity pass
#include <imgui_impl_vulkan.h>
#if CARDINAL_PLATFORM_WINDOWS
    #include <imgui_impl_win32.h>
    #include <imgui_impl_dx12.h>
#endif
// ImGuizmo — 3D translate / rotate / scale handles + ViewManipulate orbit cube
// layered on top of ImGui. Provides Manipulate(view, proj, op, mode, matrix)
// for the in-viewport gizmo and ViewManipulate(view, length, pos, size, color)
// for the corner-cube camera orbiter. Snap, local/world toggle and Universal-
// op (T+R+S in one widget) all built in. Wired alongside Cardinal's hand-rolled
// gizmo so the user can A/B them via the runtime use_imguizmo_ flag.
#include <ImGuizmo.h>

#include <cardinal/core/std/algorithm.hpp>
#include <cardinal/core/std/cctype.hpp>
#include <cardinal/core/std/chrono.hpp>
#include <cardinal/core/std/climits.hpp>
#include <cardinal/core/std/cmath.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/core/std/cstdio.hpp>
#include <cardinal/core/std/cstring.hpp>
#include <cardinal/core/std/ctime.hpp>
#include <cardinal/core/std/thread.hpp>
#include <cardinal/core/std/utility.hpp>

#if CARDINAL_PLATFORM_WINDOWS
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(
    HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
#endif

namespace cardinal::ui {

namespace {

// vk_result_name + check_vk were carved out to studio_vk_diag.{hpp,cpp}
// (Studio decomposition step 1 — zero StudioImpl coupling, behaviour
// identical). Referenced below as detail::check_vk.

// LogStore + level_style moved to panels/log.{hpp,cpp}.

// =============================================================================
// StudioImpl
//
// Core flow:
//  1) Create our own VkRenderPass (single colour attachment matching the
//     swapchain format, LOAD_OP_CLEAR, finalLayout = COLOR_ATTACHMENT). This
//     is what lets ImGui's pipeline init succeed — `UseDynamicRendering=false`
//     + a real RenderPass dodges the SDK build's broken dynamic-rendering
//     pipeline path that segfaulted previously.
//  2) Build a VkFramebuffer per swapchain image view.
//  3) Hand ImGui the descriptor pool, render pass, queue, image counts.
//  4) Register an overlay callback with the Swapchain. That callback runs
//     between scene rendering and PRESENT — it begins our render pass on
//     the current swapchain image's framebuffer and runs ImGui's draw data.
//  5) Bind the swapchain's off-screen viewport image as an ImGui texture
//     so callers can do `ImGui::Image(studio->viewport_texture(), sz)` to
//     embed the rendered scene in an editor panel.
// =============================================================================
class StudioImpl final : public Studio {
public:
    StudioImpl() = default;
    ~StudioImpl() override { shutdown(); }

    bool initialize(rhi::Device& dev, rhi::Swapchain& sw, window::Window& win) {
        device_ = &dev; swapchain_ = &sw; window_ = &win;
        gpu_    = (dev.backend() == rhi::Backend::Vulkan) ? Gpu::Vulkan : Gpu::D3D12;

        // ImGui core setup happens once regardless of backend.
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        // Windows move ONLY by dragging their title bar — never by
        // click-dragging the body. ImGui's default is "move from
        // anywhere", which turns any in-panel drag (marquee-select or
        // drag-place inside a viewport) into a window move; for a panel
        // popped into its own OS window (ViewportsEnable, below) that
        // drags the ENTIRE platform window instead of interacting with
        // its contents. Title-bar-only move is also the expected
        // pro-editor behaviour (UE / Unity / Blender all do this).
        io.ConfigWindowsMoveFromTitleBarOnly = true;
        // DO NOT add `io.ConfigDockingAlwaysTabBar = true;` here. It DOES
        // fix the residual "a SOLO torn-out panel still drags from its
        // body" issue (with the node host window hidden, that window has
        // NoTitleBar + DockIsActive==false so ImGui's move-cancel gate is
        // skipped). BUT with ConfigDockingAlwaysTabBar the solo detached
        // node keeps a separate DockNodeHost window, which becomes an
        // EXTRA ImGui platform viewport → an additional secondary Vulkan
        // swapchain via Platform_CreateVkSurface. That re-enters the
        // fragile secondary-submit path documented below and aborts the
        // device (VK_ERROR_DEVICE_LOST in ImGui_ImplVulkan_RenderWindow)
        // when the user manipulates the separated window. A Vulkan crash
        // strictly outweighs body-drag-moves-window, which is a known,
        // NON-fatal limitation for solo torn-out panels only. A safe
        // per-ImGuiWindowClass alternative (collapse torn-out viewports
        // to one swapchain while other panels keep the tab bar) is
        // tracked separately — do not re-enable the global flag until
        // the Vulkan secondary-swapchain path is hardened.
        // Master window system: any panel can be dragged outside the main
        // window and become its own OS-native window. ImGui_ImplWin32 spawns
        // those secondary windows; the renderer backend creates a per-window
        // swapchain (Vulkan via Platform_CreateVkSurface, DX12 via DXGI).
        //
        // Both backends now support OS-level pop-out viewports. The Vulkan
        // path uses a dedicated `render_pass_secondary_` (UNDEFINED →
        // PRESENT_SRC_KHR) for its per-secondary-window framebuffers — the
        // main render pass (COLOR_ATTACHMENT_OPTIMAL → COLOR_ATTACHMENT_OPTIMAL)
        // matches what cardinal's swapchain hands us before / after the
        // main-window overlay, but those layouts are wrong for ImGui's
        // self-managed secondary swapchain (DEVICE_LOST every frame on
        // vkWaitForFences). The two passes are render-pass-compatible so
        // ImGui's draw-list pipelines work against either.
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
        ImGui::StyleColorsDark();

        // ImGuizmo: set projection mode once at init. The engine's viewport
        // camera is always perspective; ortho-preview panels can flip this
        // per-Manipulate call if needed in the future.
        ImGuizmo::SetOrthographic(false);

        // When viewports are enabled, ImGui style for free-floating windows
        // looks better with no rounding + opaque alpha — matches host chrome.
        ImGuiStyle& style = ImGui::GetStyle();
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            style.WindowRounding              = 0.0f;
            style.Colors[ImGuiCol_WindowBg].w = 1.0f;

            // Multi-monitor usability: give every popped-out panel a real
            // OS window frame (title bar + minimize/maximize/close + system
            // menu) and its own taskbar entry. Without this ImGui creates
            // borderless secondary windows that can't be OS-maximized or
            // Win+Arrow / Snap-tiled, so a panel dragged onto a second
            // monitor can't fill it. This is a platform-backend (Win32)
            // setting — identical behaviour whether the render backend is
            // Vulkan or D3D12 (both just draw into the window ImGui_ImplWin32
            // creates). Users can now maximize/snap each dock-out window to
            // whatever monitor it lives on. (Double-click the title bar to
            // maximize-to-monitor; drag to a screen edge to half-tile.)
            io.ConfigViewportsNoDecoration   = false;
            io.ConfigViewportsNoTaskBarIcon  = false;
        }
        if (resolve_ini_path(ini_path_, sizeof(ini_path_))) io.IniFilename = ini_path_;

#if CARDINAL_PLATFORM_WINDOWS
        if (!ImGui_ImplWin32_Init(win.native_handle())) {
            cardinal::log::errorf("ui/studio", "ImGui_ImplWin32_Init failed");
            return false;
        }
        // Append, don't replace — the integrator may also attach an
        // input::Manager hook. Both need to see every event.
        win.add_message_hook(&StudioImpl::message_hook_thunk, this);
#endif

        bool ok = (gpu_ == Gpu::Vulkan) ? init_vulkan() : init_d3d12();
        if (!ok) return false;

        // Try to bind viewport 0 up front; later panels lazily bind as
        // their ids are first drawn (try_bind_viewport_texture(id)).
        try_bind_viewport_texture(0u);
        cardinal::log::add_sink(&log_state_.cstore);
        sw.set_overlay(&StudioImpl::overlay_thunk, this);

        cardinal::log::infof("ui/studio", "Studio online (%s, %ux%u, %s)",
            device_->adapter_name(), swapchain_->width(), swapchain_->height(),
            gpu_ == Gpu::Vulkan ? "Vulkan" : "D3D12");
        return true;
    }

    bool init_vulkan() {
        const auto dh = rhi::vulkan_handles(device_);
        const auto sh = rhi::vulkan_handles(swapchain_);
        color_format_ = sh.color_format;
        image_count_  = sh.image_count;

        if (!create_descriptor_pool(dh.device))         { cardinal::log::errorf("ui/studio","desc_pool failed");   return false; }
        if (!create_render_pass(dh.device))             { cardinal::log::errorf("ui/studio","render_pass failed"); return false; }
        if (!create_framebuffers(dh.device, swapchain_)){ cardinal::log::errorf("ui/studio","framebuffers failed");return false; }
        if (!create_sampler(dh.device))                 { cardinal::log::errorf("ui/studio","sampler failed");     return false; }

        if (!ImGui_ImplVulkan_LoadFunctions(VK_API_VERSION_1_3, &load_volk, dh.instance)) {
            cardinal::log::errorf("ui/studio", "ImGui_ImplVulkan_LoadFunctions failed");
            return false;
        }

        // Multi-viewport for Vulkan: ImGui_ImplVulkan_Init asserts that
        // Platform_CreateVkSurface is wired when ConfigFlags_ViewportsEnable
        // is set. ImGui_ImplWin32 doesn't ship a Vulkan-aware surface
        // creator, so plumb our own using vkCreateWin32SurfaceKHR.
#if CARDINAL_PLATFORM_WINDOWS
        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            ImGui::GetPlatformIO().Platform_CreateVkSurface = &create_vk_surface_thunk;
        }
#endif

        ImGui_ImplVulkan_InitInfo info{};
        info.ApiVersion          = VK_API_VERSION_1_3;
        info.Instance            = dh.instance;
        info.PhysicalDevice      = dh.physical_device;
        info.Device              = dh.device;
        info.QueueFamily         = dh.graphics_queue_family;
        info.Queue               = dh.graphics_queue;
        info.DescriptorPool      = desc_pool_;
        // Pass the SECONDARY render pass to ImGui — it uses this to
        // create framebuffers for OS-level pop-out windows AND to compile
        // its draw-list pipelines. The main-window overlay path we drive
        // ourselves still uses render_pass_ at vkCmdBeginRenderPass time;
        // both passes are render-pass-compatible so the same pipelines
        // work in either context (Vulkan only requires matching attachment
        // formats / sample counts / counts, not matching layouts).
        info.RenderPass          = render_pass_secondary_;
        info.Subpass             = 0;
        info.MinImageCount       = sh.min_image_count;
        info.ImageCount          = sh.image_count;
        info.MSAASamples         = VK_SAMPLE_COUNT_1_BIT;
        info.UseDynamicRendering = false;
        info.CheckVkResultFn     = &detail::check_vk;
        info.MinAllocationSize   = 1024 * 1024;
        if (!ImGui_ImplVulkan_Init(&info)) {
            cardinal::log::errorf("ui/studio", "ImGui_ImplVulkan_Init failed");
            return false;
        }
        return true;
    }

#if CARDINAL_PLATFORM_WINDOWS
    bool init_d3d12() {
        const auto dh = rhi::d3d12_handles(device_);
        const auto sh = rhi::d3d12_handles(swapchain_);
        if (dh.device == nullptr) { cardinal::log::errorf("ui/studio","d3d12 device handle null"); return false; }
        image_count_ = sh.back_buffer_count;

        // SRV descriptor heap for ImGui's font + any textures we AddTexture.
        D3D12_DESCRIPTOR_HEAP_DESC dh_desc{};
        dh_desc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        dh_desc.NumDescriptors = dx12_srv_capacity_;
        dh_desc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (FAILED(dh.device->CreateDescriptorHeap(&dh_desc, IID_PPV_ARGS(&dx12_srv_heap_)))) {
            cardinal::log::errorf("ui/studio", "D3D12 SRV heap creation failed");
            return false;
        }
        dx12_srv_stride_ = dh.device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        // All slots are free initially.
        dx12_srv_free_.reserve(dx12_srv_capacity_);
        for (UINT i = 0; i < dx12_srv_capacity_; ++i) dx12_srv_free_.push_back(i);

        ImGui_ImplDX12_InitInfo info{};
        info.Device              = dh.device;
        info.CommandQueue        = dh.gfx_queue;
        info.NumFramesInFlight   = static_cast<int>(sh.back_buffer_count);
        info.RTVFormat           = sh.back_buffer_format;
        info.DSVFormat           = DXGI_FORMAT_UNKNOWN;
        info.SrvDescriptorHeap   = dx12_srv_heap_.Get();
        info.UserData            = this;
        info.SrvDescriptorAllocFn = &StudioImpl::dx12_srv_alloc_thunk;
        info.SrvDescriptorFreeFn  = &StudioImpl::dx12_srv_free_thunk;
        if (!ImGui_ImplDX12_Init(&info)) {
            cardinal::log::errorf("ui/studio", "ImGui_ImplDX12_Init failed");
            return false;
        }
        return true;
    }
#else
    bool init_d3d12() { return false; }
#endif

    void begin_frame() override {
        if (gpu_ == Gpu::Vulkan) ImGui_ImplVulkan_NewFrame();
#if CARDINAL_PLATFORM_WINDOWS
        else                     ImGui_ImplDX12_NewFrame();
        ImGui_ImplWin32_NewFrame();
#endif
        ImGui::NewFrame();
        // ImGuizmo per-frame setup. Must be called before any panel renders
        // a gizmo so the internal projection state (display size, mouse
        // position cache, draw-list defaults) is fresh. The 96 x 96 invisible
        // "gizmo" window BeginFrame creates is harmless w/ NoInputs — the
        // earlier suspicion that it broke docking turned out to be wrong;
        // the real cause was per-panel ImGuiWindowFlags_NoMove flags in
        // panels/*.cpp (fixed in 62bdcb2). Removing this call broke gizmo
        // handle alignment with the manipulated object.
        ImGuizmo::BeginFrame();
        ++frame_count_;
        // Pick state is populated by draw_viewport_panel(); reset at the
        // top of each frame so a stale click doesn't leak across frames.
        vp_pick_ = ViewportPick{};
        vp_ctx_  = ViewportCtxAction{};
        // World-grid hover-chunk feedback — repopulated by draw_world_grid_overlay
        // when the cursor is over the grid. Reset here so when the cursor
        // leaves the viewport, the chunk-coord badge disappears.
        hover_chunk_valid_ = false;
        // Gizmo drag delta is per-frame; the host integrates it into the
        // entity translation, then reads zero again until the next move.
        gizmo_drag_ = GizmoDrag{};

        // First-frame layout sanity pass — clamp every restored ImGui
        // window position to fit on a real monitor. ImGui persists window
        // positions to its .ini; if the user closes the editor with a
        // panel dragged to (-2000, 500) (e.g. monitor disconnected since,
        // or the panel was off-screen at exit), the next launch restores
        // it there and ImGui spawns a secondary OS-level viewport to host
        // it. On Vulkan, secondary-window submission is fragile and the
        // first secondary submit faults the device (DEVICE_LOST). Clamp
        // here so stale .ini state can never produce off-screen panels —
        // the user just sees their layout snap back into the host on the
        // first frame after a problematic .ini.
        clamp_imgui_windows_to_monitor_once();

        // Focused-panel maximize — ImGui input is valid post-NewFrame.
        // Gives every docked panel/tab a maximize it otherwise lacks
        // (only the viewport panels carry a toolbar maximize button).
        {
            const ImGuiIO& io = ImGui::GetIO();
            if (!io.WantTextInput &&
                ImGui::IsKeyDown(ImGuiKey_LeftShift) &&
                ImGui::IsKeyPressed(ImGuiKey_F11, false)) {
                toggle_maximize_focused_panel();
            }
            if (max_restore_pending_) {
                if (!max_restore_win_.empty() && max_restore_dock_ != 0) {
                    ImGui::DockBuilderDockWindow(max_restore_win_.c_str(),
                                                 max_restore_dock_);
                }
                max_restore_pending_ = false;
                max_restore_win_.clear();
                max_restore_dock_ = 0;
            }
            if (!max_win_.empty()) {
                if (max_enter_pending_) {
                    // Undock so the panel can float over everything.
                    ImGui::DockBuilderDockWindow(max_win_.c_str(), 0);
                    max_enter_pending_ = false;
                }
                const ImGuiViewport* vp = ImGui::GetMainViewport();
                ImGui::SetWindowCollapsed(max_win_.c_str(), false);
                ImGui::SetWindowPos (max_win_.c_str(), vp->WorkPos);
                ImGui::SetWindowSize(max_win_.c_str(), vp->WorkSize);
                ImGui::SetWindowFocus(max_win_.c_str());
            }
        }
    }

    void end_frame() override {
        // Just finalise the draw lists — the GPU recording happens inside
        // overlay_render() when the swapchain reaches the right point.
        ImGui::Render();
    }

    void toggle_maximize_focused_panel() override {
        ImGuiContext* g = ImGui::GetCurrentContext();
        if (g == nullptr) return;
        if (!max_win_.empty()) {
            // Already maximized → request restore to its old dock node.
            max_restore_win_     = max_win_;
            max_restore_dock_    = max_prev_dock_;
            max_restore_pending_ = true;
            max_win_.clear();
            return;
        }
        ImGuiWindow* w = g->NavWindow;          // the focused panel
        if (w == nullptr || w->Name == nullptr) return;
        if ((w->Flags & (ImGuiWindowFlags_Popup | ImGuiWindowFlags_Tooltip |
                         ImGuiWindowFlags_ChildWindow)) != 0) return;
        if (w->Name[0] == '#') return;          // dockhost / internal
        max_win_           = w->Name;
        max_prev_dock_     = w->DockId;
        max_enter_pending_ = true;
    }

    void on_swapchain_resized() override {
        if (gpu_ == Gpu::Vulkan) {
            const auto dh = rhi::vulkan_handles(device_);
            if (dh.device == VK_NULL_HANDLE) return;
            // Drain so we can free old framebuffers safely.
            vkDeviceWaitIdle(dh.device);
            for (auto fb : framebuffers_) {
                if (fb != VK_NULL_HANDLE) vkDestroyFramebuffer(dh.device, fb, nullptr);
            }
            framebuffers_.clear();
            create_framebuffers(dh.device, swapchain_);
        }
        // D3D12 needs no per-image rebuild — RTV heap entries are recreated
        // by D3D12Swapchain::resize() against the new back buffers and we
        // re-fetch back_buffer_rtv() each frame inside begin_frame().
    }

    void update_platform_windows() override {
        // Commit any deferred per-viewport resizes first — runs strictly
        // AFTER swapchain->end_frame so the GPU work that referenced the
        // old images is fully retired before we tear them down. Then drop
        // each affected panel's ImGui descriptor and let
        // try_bind_viewport_texture rebind on the next frame.
        //
        // Both backends now own a real per-id RTT, so a viewport[id]
        // resize only invalidates that one panel's descriptor (no more
        // global vk_invalidate_all sweep).
        if (swapchain_ != nullptr) {
            for (u32 id = 0; id < vp_panels_.size(); ++id) {
                auto& vp = vp_panels_[id];
                if (vp.commit_w == 0 && vp.commit_h == 0) continue;
                swapchain_->set_viewport_size(id, vp.commit_w, vp.commit_h);
                if (gpu_ == Gpu::Vulkan && vp.vk_descriptor != VK_NULL_HANDLE) {
                    ImGui_ImplVulkan_RemoveTexture(vp.vk_descriptor);
                }
                vp.vk_descriptor = VK_NULL_HANDLE;
#if CARDINAL_PLATFORM_WINDOWS
                if (vp.dx12_srv_slot != 0xFFFFFFFFu) {
                    dx12_srv_free_.push_back(vp.dx12_srv_slot);
                    vp.dx12_srv_slot = 0xFFFFFFFFu;
                }
                vp.dx12_imtex = nullptr;
#endif
                vp.commit_w = vp.commit_h = 0;
            }
        }

        ImGuiIO& io = ImGui::GetIO();
        if ((io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) == 0) return;
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
    }

    void draw_fps_overlay() override {
        // Per the editor's UX brief: framerate is shown PER-VIEWPORT in the
        // viewport-panel toolbar (`Title · 60 FPS`), not as a floating
        // top-left card. We keep this method as a no-op draw so the host's
        // existing `studio->draw_fps_overlay()` call site stays valid;
        // it's purely a sampler now — updates the EMA that
        // draw_viewport_panel reads each frame.
        const auto now = cardinal::chrono::steady_clock::now();
        const float dt = cardinal::chrono::duration<float>(now - last_sample_).count();
        if (dt >= 0.25f) {
            const auto frames = frame_count_ - last_sample_frame_;
            ema_fps_ = static_cast<float>(frames) / dt;
            last_sample_       = now;
            last_sample_frame_ = frame_count_;
        }
        // (No ImGui::Begin / End. The per-viewport toolbar is the only
        // place the counter appears now.)
    }

    void* viewport_texture() override {
        return viewport_texture(0u);
    }
    void* viewport_texture(u32 id) override {
        ensure_panel_slots(id);
        // Re-bind if the underlying view changed (e.g. set_viewport_size
        // was called). The view pointer changes on reallocation.
        try_bind_viewport_texture(id);
        auto& vp = vp_panels_[id];
        if (gpu_ == Gpu::Vulkan) return reinterpret_cast<void*>(vp.vk_descriptor);
#if CARDINAL_PLATFORM_WINDOWS
        return vp.dx12_imtex;
#else
        return nullptr;
#endif
    }

    // Grow vp_panels_ on demand so callers can index by an arbitrary id.
    void ensure_panel_slots(u32 id) {
        if (id >= vp_panels_.size()) vp_panels_.resize(id + 1u);
    }

    // First-frame layout sanity. Walks the .ini-loaded ImGuiWindowSettings
    // (NOT ctx->Windows — those don't exist until each window is first
    // Begin'd this session) and snaps any with a stale ViewportPos
    // outside every known monitor back inside the main monitor.
    //
    // Why bother: ImGui persists per-window ViewportPos to its .ini for
    // any window the user dragged into its own OS-level secondary window.
    // If the user then closes the editor, swaps monitors, or shrinks the
    // host, the next launch restores those positions OFF-SCREEN, ImGui
    // spawns secondary OS windows there in update_platform_windows, and
    // on Vulkan the first secondary submit faults the device with
    // VK_ERROR_DEVICE_LOST. We dock-snap before update_platform_windows
    // ever sees them so no off-screen secondary is ever created.
    void clamp_imgui_windows_to_monitor_once() {
        if (clamp_done_) return;
        clamp_done_ = true;

        const ImGuiPlatformIO& pio  = ImGui::GetPlatformIO();
        const ImGuiViewport*   main = ImGui::GetMainViewport();
        if (main == nullptr) return;

        // Build a union of monitor rects. Multi-monitor users with a panel
        // dragged onto a second display should NOT get snapped back; the
        // .ini stays valid because the secondary monitor is still listed.
        // Falls back to the main viewport's WorkRect if the platform
        // backend didn't enumerate monitors (rare; should always be set).
        struct Rect { float l, t, r, b; };
        cardinal::vector<Rect> monitors;
        if (pio.Monitors.Size > 0) {
            monitors.reserve(static_cast<usize>(pio.Monitors.Size));
            for (int i = 0; i < pio.Monitors.Size; ++i) {
                const auto& m = pio.Monitors[i];
                monitors.push_back({ m.WorkPos.x, m.WorkPos.y,
                                     m.WorkPos.x + m.WorkSize.x,
                                     m.WorkPos.y + m.WorkSize.y });
            }
        } else {
            monitors.push_back({ main->WorkPos.x, main->WorkPos.y,
                                 main->WorkPos.x + main->WorkSize.x,
                                 main->WorkPos.y + main->WorkSize.y });
        }
        // Considered visible if at least 64px of the window's title bar
        // overlaps any monitor's WorkRect.
        auto pos_visible = [&](float x, float y, float w, float h) {
            const float pad = 64.0f;
            for (const auto& m : monitors) {
                const float ox = cardinal::min(x + w, m.r) - cardinal::max(x, m.l);
                const float oy = cardinal::min(y + h, m.b) - cardinal::max(y, m.t);
                if (ox >= pad && oy >= pad) return true;
            }
            return false;
        };

        ImGuiContext* ctx = ImGui::GetCurrentContext();
        if (ctx == nullptr) return;
        int snapped = 0;
        // Walk every persisted window-setting block. Each entry's
        // ViewportPos is what triggers the off-screen secondary creation
        // on first Begin — clearing ViewportId snaps it back into the
        // main viewport (so it docks into the dockspace as normal).
        for (ImGuiWindowSettings* s = ctx->SettingsWindows.begin();
             s != nullptr;
             s = ctx->SettingsWindows.next_chunk(s))
        {
            if (s == nullptr) continue;
            // Only check windows that asked for an explicit OS-window
            // viewport. Plain Pos values are inside the host already.
            if (s->ViewportId == 0) continue;
            const float w = (s->Size.x > 0) ? static_cast<float>(s->Size.x) : 320.0f;
            const float h = (s->Size.y > 0) ? static_cast<float>(s->Size.y) : 240.0f;
            if (pos_visible(static_cast<float>(s->ViewportPos.x),
                            static_cast<float>(s->ViewportPos.y), w, h))
                continue;

            // Off-screen — clear the secondary-viewport binding so this
            // window re-attaches to the main host on first Begin. Also
            // zero the position so the docking system places it sensibly.
            cardinal::log::infof("ui/studio",
                "layout-sanity: snapping '%s' from off-screen ViewportPos=(%d,%d) "
                "back into main host",
                s->GetName(),
                static_cast<int>(s->ViewportPos.x),
                static_cast<int>(s->ViewportPos.y));
            s->ViewportId   = 0;
            s->ViewportPos  = ImVec2ih(0, 0);
            s->Pos          = ImVec2ih(0, 0);
            ++snapped;
        }
        if (snapped > 0) {
            cardinal::log::infof("ui/studio",
                "layout-sanity: %d window(s) snapped back from off-screen "
                "(stale .ini state). Settings will be rewritten on exit.",
                snapped);
            ImGui::MarkIniSettingsDirty();
        }
    }

    // Public API: wipe all persisted layout state so the next frame rebuilds
    // the default dock layout from scratch. Useful when the .ini gets stuck
    // in a state where panels can't be reached. Bound to the
    // `editor.reset_layout` console command.
    void reset_layout() override {
        // Tell ImGui to forget every window's settings. ImGui will reset
        // dock state too because we never persist a layout file beyond
        // imgui_ini.
        ImGui::ClearIniSettings();
        // Force the layout-sanity pass to fire again next frame so anything
        // ImGui re-creates with a default position gets snapped to main.
        clamp_done_ = false;
        cardinal::log::infof("ui/studio",
            "layout reset — every window will reposition on the next frame.");
    }

    ViewportPick last_viewport_pick() const noexcept override { return vp_pick_; }

    HoverChunk last_hover_chunk() const noexcept override {
        HoverChunk h;
        // Prefer the context-menu freeze when it's pinned — that's the
        // chunk the user right-clicked on, which is what "Spawn active
        // asset here" wants. Fall back to the live per-frame hover for
        // any non-menu reader.
        if (ctx_hover_chunk_valid_) {
            h.valid = true;
            h.x = ctx_hover_chunk_x_;
            h.y = ctx_hover_chunk_y_;
            h.z = ctx_hover_chunk_z_;
        } else {
            h.valid = hover_chunk_valid_;
            h.x = hover_chunk_x_;
            h.y = hover_chunk_y_;
            h.z = hover_chunk_z_;
        }
        return h;
    }

    // ----- Gizmo plumbing ------------------------------------------------
    // Host pushes camera + selection state in; Studio renders the gizmo
    // overlay during draw_viewport_panel and accumulates a per-frame
    // world-space translation delta.
    void set_viewport_camera(const cardinal::scene::Mat4& view,
                             const cardinal::scene::Mat4& proj) override
    {
        gizmo_view_ = view;
        gizmo_proj_ = proj;
        gizmo_have_camera_ = true;
    }
    void set_gizmo_target(const cardinal::scene::Vec3& world_pos,
                          bool has_target) override
    {
        gizmo_target_ = world_pos;
        gizmo_has_target_ = has_target;
        // If the user un-selected mid-drag, abort the drag so the next
        // selection doesn't inherit a stale axis/handle.
        if (!has_target) { gizmo_axis_active_ = -1; gizmo_handle_active_ = -1; }
    }
    void set_gizmo_transform(const cardinal::scene::Vec3& position,
                             const cardinal::scene::Vec3& rotation_euler,
                             const cardinal::scene::Vec3& scale,
                             bool has_target) override
    {
        gizmo_target_     = position;
        gizmo_target_rot_ = rotation_euler;
        gizmo_target_scl_ = scale;
        gizmo_has_target_ = has_target;
        if (!has_target) { gizmo_axis_active_ = -1; gizmo_handle_active_ = -1; }
    }
    void set_gizmo_mode(GizmoMode m) noexcept override {
        if (m != gizmo_mode_) {            // mode change aborts any drag
            gizmo_mode_ = m;
            gizmo_axis_active_ = -1;
            gizmo_handle_active_ = -1;
        }
    }
    GizmoMode gizmo_mode() const noexcept override { return gizmo_mode_; }

    // ---- ImGuizmo toggle --------------------------------------------
    void set_use_imguizmo(bool enabled) noexcept override { use_imguizmo_ = enabled; }
    bool use_imguizmo() const noexcept override            { return use_imguizmo_; }
    void set_imguizmo_local_space(bool local) noexcept override { imguizmo_local_space_ = local; }
    bool imguizmo_local_space() const noexcept override    { return imguizmo_local_space_; }
    void set_imguizmo_show_view_cube(bool show) noexcept override { imguizmo_show_view_cube_ = show; }
    void set_gizmo_snap(bool enabled, float step) noexcept override {
        gizmo_snap_on_   = enabled;
        gizmo_snap_step_ = step;
    }

    void set_selection(const u32* ids, cardinal::usize count) override {
        selection_.clear();
        if (ids != nullptr)
            for (cardinal::usize i = 0; i < count; ++i) selection_.push_back(ids[i]);
    }
    const cardinal::vector<u32>& selection() const noexcept override {
        return selection_;
    }

    GizmoDrag last_gizmo_drag() const noexcept override { return gizmo_drag_; }
    bool gizmo_drag_in_progress() const noexcept override { return gizmo_drag_in_progress_; }
    bool gizmo_drag_release_consume() override {
        const bool r = gizmo_drag_release_;
        gizmo_drag_release_ = false;
        return r;
    }
    ViewportCtxAction last_viewport_ctx_action() const noexcept override {
        return vp_ctx_;
    }

    void draw_log_panel(const char* title, bool* p_open) override {
        panels::log_panel::draw(title, p_open, log_state_);
    }

    void draw_cppscript_panel(cardinal::cppscript::Engine* engine,
                              const char* title,
                              bool* p_open,
                              SandboxStatusLookup lookup) override
    {
        // Adapt the void*-erased lookup to the panel's typed signature.
        // Studio's public header doesn't pull in sandbox.hpp; the panel's
        // own header does, so the cast lives here.
        panels::cppscript_panel::SandboxLookup typed_lookup;
        if (lookup) {
            typed_lookup = [lookup](const cardinal::string& src,
                                    cardinal::sandbox::Status* out) -> bool {
                return lookup(src, static_cast<void*>(out));
            };
        }
        panels::cppscript_panel::draw(engine, title, p_open,
                                      cppscript_panel_state_, typed_lookup);
    }

    void draw_stats_panel(const char* title, bool* p_open) override {
        panels::stats_panel::Inputs in;
        in.device      = device_;
        in.swapchain   = swapchain_;
        in.ema_fps     = ema_fps_;
        in.frame_count = frame_count_;
        panels::stats_panel::draw(title, p_open, in);
    }

    void draw_memory_panel(cardinal::budget::Broker* broker,
                           const char* title, bool* p_open) override {
        panels::memory_panel::draw(broker, title, p_open);
    }

    void draw_profiler_panel(float frame_ms_now, const char* title,
                             bool* p_open) override {
        panels::profiler_panel::draw(profiler_state_, frame_ms_now, title, p_open);
    }

    void draw_vt_panel(cardinal::vt::System* sys, const char* title,
                       bool* p_open) override {
        panels::vt_panel::draw(sys, title, p_open);
    }

    void draw_sim_bar_panel(cardinal::sim::SimWorld* sim,
                            const char* title, bool* p_open) override {
        panels::sim_bar_panel::draw(sim, title, p_open);
    }
    void draw_actor_outliner_panel(cardinal::actor::World* world,
                                   u32* selected_actor_id_inout,
                                   const char* title, bool* p_open) override {
        panels::actor_outliner_panel::draw(world, selected_actor_id_inout,
                                           title, p_open);
    }
    void draw_prefab_panel(cardinal::actor::World* world,
                           u32* selected_actor_id_inout,
                           const char* title, bool* p_open) override {
        panels::prefab_panel::draw(prefab_panel_state_, world,
                                   selected_actor_id_inout, title, p_open);
    }
    void draw_curve_editor_panel(cardinal::anim::Curve<float>* curve,
                                 const char* title, bool* p_open) override {
        panels::curve_editor_panel::draw(curve, title, p_open);
    }
    void draw_sequencer_panel(cardinal::cine::Player* player,
                              const char* title, bool* p_open) override {
        panels::sequencer_panel::draw(player, title, p_open);
    }
    void draw_mixer_panel(cardinal::audio::Engine* engine,
                          const char* title, bool* p_open) override {
        panels::mixer_panel::draw(engine, title, p_open);
    }

    void update_game_logic(cardinal::game::Game* game) override {
        panels::game_bar_panel::update(game_bar_state_, game);
    }
    void draw_game_bar_panel(cardinal::game::Game* game,
                             const char* title, bool* p_open) override {
        panels::game_bar_panel::draw(game_bar_state_, game, title, p_open);
    }
    void draw_class_picker_panel(cardinal::game::Game* game,
                                 u32* selected_actor_id_inout,
                                 const char* title, bool* p_open) override {
        panels::class_picker_panel::draw(game, selected_actor_id_inout,
                                         title, p_open);
    }
    void draw_sky_panel(cardinal::sky::Sky* sky,
                        const char* title, bool* p_open) override {
        panels::sky_panel::draw(sky, title, p_open);
    }

    void draw_input_panel(cardinal::input::Manager* mgr,
                          const char* title, bool* p_open) override {
        panels::input_panel::draw(mgr, title, p_open);
    }
    void draw_save_load_panel(cardinal::game::Game* game,
                              cardinal::sky::Sky*  sky,
                              const char* title, bool* p_open) override {
        panels::save_load_panel::draw(game, sky, title, p_open);
    }
    void draw_particles_panel(cardinal::particles::System* sys,
                              const char* title, bool* p_open) override {
        panels::particles_panel::draw(sys, title, p_open);
    }
    ProjectAction draw_project_panel(
        cardinal::shared_ptr<cardinal::project::Project>* current_project,
        cardinal::project::RecentProjects* recents,
        const char* title, bool* p_open) override
    {
        const auto a = panels::project_panel::draw(current_project, recents,
                                                    title, p_open);
        ProjectAction r{};
        r.opened                = a.opened_project;
        r.save_clicked          = a.save_clicked;
        r.cook_clicked          = a.cook_clicked;
        r.pack_clicked          = a.pack_clicked;
        r.cook_and_pack_clicked = a.cook_and_pack_clicked;
        return r;
    }
    void draw_cook_pack_panel(cardinal::project::Project* project,
                              cardinal::cook::CookerRegistry* registry,
                              cardinal::shader::Compiler* shader_compiler,
                              const char* title, bool* p_open) override
    {
        panels::cook_pack_panel::draw(project, registry, shader_compiler,
                                       &cook_pack_state_, title, p_open);
    }
    void draw_shader_compiler_panel(cardinal::shader::Compiler* compiler,
                                    const char* title, bool* p_open) override
    {
        panels::shader_compiler_panel::draw(compiler, title, p_open);
    }

    void draw_world_systems_panel(const WorldSystemsInputs& in,
                                  const char* title, bool* p_open) override
    {
        panels::world_systems_panel::Inputs i{};
        i.io_dispatcher = in.io_dispatcher;
        i.partition     = in.partition;
        i.level_manager = in.level_manager;
        i.hlod_tree     = in.hlod_tree;
        i.mass_world    = in.mass_world;
        i.ai_perception = in.ai_perception;
        i.navmesh       = in.navmesh;
        panels::world_systems_panel::draw(i, title, p_open);
    }

    void draw_nav_panel(cardinal::nav::Grid* grid, NavState* state,
                        const char* title, bool* p_open) override {
        // Re-shape the public NavState into the panel's local State —
        // they're the same fields, just kept separate so the panel header
        // doesn't leak into studio.hpp.
        panels::nav_panel::State s{};
        if (state) {
            s.start_x = state->start_x; s.start_y = state->start_y;
            s.goal_x  = state->goal_x;  s.goal_y  = state->goal_y;
            s.allow_diagonal = state->allow_diagonal;
            s.paint_blocked  = state->paint_blocked;
            s.cell_pixels    = state->cell_pixels;
        }
        panels::nav_panel::draw(grid, s, title, p_open);
        if (state) {
            state->start_x = s.start_x; state->start_y = s.start_y;
            state->goal_x  = s.goal_x;  state->goal_y  = s.goal_y;
            state->allow_diagonal = s.allow_diagonal;
            state->paint_blocked  = s.paint_blocked;
            state->cell_pixels    = s.cell_pixels;
        }
    }

    void draw_editor_modes_panel(cardinal::edit::EditorState* state,
                                 const char* title, bool* p_open) override {
        panels::editor_modes_panel::draw(state, title, p_open);
    }

    void draw_brush_panel(cardinal::edit::brush::Brush* brush,
                          const char* title, bool* p_open) override {
        panels::brush_panel::draw(brush, title, p_open);
    }

    MeshToolsRequest draw_mesh_tools_panel(bool selection_has_mesh,
                                           const char* title, bool* p_open) override
    {
        panels::mesh_tools_panel::draw(&mesh_tools_state_,
                                       selection_has_mesh, title, p_open);
        MeshToolsRequest r{};
        r.generate           = mesh_tools_state_.generate_clicked;
        r.prim               = static_cast<PrimitiveKind>(
            static_cast<u32>(mesh_tools_state_.prim_kind));
        r.size               = mesh_tools_state_.prim_size;
        r.radius             = mesh_tools_state_.prim_radius;
        r.height             = mesh_tools_state_.prim_height;
        r.minor              = mesh_tools_state_.prim_minor;
        r.segments           = mesh_tools_state_.prim_segments;
        r.subdivide          = mesh_tools_state_.subdivide_clicked;
        r.subdivide_levels   = mesh_tools_state_.subdivide_levels;
        r.mirror             = mesh_tools_state_.mirror_clicked;
        r.mirror_axis        = mesh_tools_state_.mirror_axis;
        r.decimate           = mesh_tools_state_.decimate_clicked;
        r.decimate_cell      = mesh_tools_state_.decimate_cell;
        r.smooth             = mesh_tools_state_.smooth_clicked;
        r.smooth_iters       = mesh_tools_state_.smooth_iterations;
        r.smooth_lambda      = mesh_tools_state_.smooth_lambda;
        r.recompute_normals  = mesh_tools_state_.recompute_normals_clicked;
        r.recompute_smooth   = mesh_tools_state_.recompute_normals_smooth;
        return r;
    }

    TextureToolsRequest draw_texture_tools_panel(const char* title,
                                                 bool* p_open) override
    {
        panels::texture_tools_panel::draw(&tex_tools_state_, title, p_open);
        TextureToolsRequest r{};
        r.export_now  = tex_tools_state_.export_clicked;
        r.export_path = tex_tools_state_.export_path;
        return r;
    }

    // -------------------------------------------------------------------
    // Detachable viewport panel — shows the off-screen RTT and lets the
    // user pick the active view-mode via a toolbar combobox. The Studio
    // hands the chosen mode back to the caller via mode_inout; the actual
    // rendering pass (cardinal::scene::ForwardRenderer) is driven by the
    // app, not Studio. That keeps Studio depending only on the ViewMode
    // *enum*, not the renderer implementation.
    // -------------------------------------------------------------------
    void draw_viewport_panel(const char* title, scene::ViewMode* mode_inout,
                             bool* p_open) override {
        // Legacy 3-arg form — single viewport (id 0), no FPS dropdown,
        // no maximize / minimize toolbar buttons.
        draw_viewport_panel(title, 0u, mode_inout, nullptr, nullptr, nullptr, p_open);
    }

    void draw_viewport_panel(const char* title, scene::ViewMode* mode_inout,
                             float* fps_limit_inout, bool* p_open) override
    {
        // Legacy 4-arg form — single viewport (id 0), FPS dropdown but
        // no maximize / minimize buttons.
        draw_viewport_panel(title, 0u, mode_inout, fps_limit_inout,
                            nullptr, nullptr, p_open);
    }

    void draw_viewport_panel(const char* title, u32 viewport_id,
                             scene::ViewMode* mode_inout,
                             float* fps_limit_inout,
                             bool* maximized_inout,
                             bool* minimized_inout,
                             bool* p_open) override
    {
        ensure_panel_slots(viewport_id);

        // Maximize: pin the panel to the host viewport's full work area
        // so it covers everything else. The host is responsible for
        // hiding / suppressing other viewports + side panels when any
        // viewport is maximized — Studio just owns the geometry.
        if (maximized_inout != nullptr && *maximized_inout) {
            const ImGuiViewport* hv = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos (hv->WorkPos);
            ImGui::SetNextWindowSize(hv->WorkSize);
            ImGui::SetNextWindowViewport(hv->ID);
            ImGui::SetNextWindowFocus();
        }
        // Minimize: defer to ImGui's window-collapse so the panel folds
        // to its title bar. Apply once on the rising edge of *minimized
        // (sticky collapse otherwise breaks user-driven uncollapse via
        // the ▾ button).
        if (minimized_inout != nullptr && *minimized_inout) {
            ImGui::SetNextWindowCollapsed(true, ImGuiCond_Once);
        }

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        // The body-drag-moves-the-window bug is now fixed GLOBALLY at the
        // IO layer (ConfigWindowsMoveFromTitleBarOnly + the
        // ConfigDockingAlwaysTabBar bypass-closer in init). The viewport
        // panel therefore needs NO per-window NoMove — and must not have
        // one, or it would be the only panel the user can't reposition by
        // its title bar/tab, contradicting "move from the top". NoMove
        // stays only while maximized (the panel is pinned to the host
        // work area then and must not be dragged or undocked).
        const ImGuiWindowFlags wflags =
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
            ((maximized_inout != nullptr && *maximized_inout)
                ? (ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoDocking)
                : 0);
        if (!ImGui::Begin(title ? title : "Viewport", p_open, wflags)) {
            ImGui::End(); ImGui::PopStyleVar(); return;
        }

        // Toolbar.
        //
        // We render the toolbar inside a fixed-height child region. Two
        // reasons:
        //   1) Layout determinism — the image area below gets exactly
        //      `panel_h - toolbar_h` regardless of whether the panel is
        //      tall or short. Without this, when a user docks four
        //      viewports into a thin horizontal strip ImGui clips the
        //      bottom of the toolbar against the dock node's bottom edge.
        //   2) The widgets share an explicit row, so SameLine'd items
        //      can never wrap onto a phantom second row.
        //
        // Combos scale with the viewport panel width — fixed pixel widths
        // overflowed when the user split the central area into 4+ viewports.
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, 3));
        const float toolbar_h = ImGui::GetFrameHeight() + 6.0f;  // 1 row + 6px breathing
        char tb_label[32];
        cardinal::snprintf(tb_label, sizeof(tb_label), "##vp_toolbar_%u", viewport_id);
        ImGui::BeginChild(tb_label, ImVec2(0.0f, toolbar_h),
                          ImGuiChildFlags_None,
                          ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        ImGui::SetCursorPos(ImVec2(6.0f, 3.0f));
        const float toolbar_avail = ImGui::GetContentRegionAvail().x;
        // Per-viewport view mode. These are the visualisation choices that
        // live ON THE VIEWPORT, not in the global Render Pipeline panel.
        // Every panel can choose its own mode independently.
        const char* labels[] = { "Solid", "Wireframe", "Polygons",
                                 "Heightmap", "Normals", "Ray-Traced Lighting" };
        int sel = mode_inout ? static_cast<int>(*mode_inout) : 0;
        const float vm_w = cardinal::clamp(toolbar_avail * 0.30f, 90.0f, 200.0f);
        char vm_label[32];
        cardinal::snprintf(vm_label, sizeof(vm_label), "##viewmode_%u", viewport_id);
        ImGui::SetNextItemWidth(vm_w);
        if (ImGui::Combo(vm_label, &sel, labels, IM_ARRAYSIZE(labels)) && mode_inout) {
            *mode_inout = static_cast<scene::ViewMode>(sel);
        }
        ImGui::SameLine();

        // Per-viewport FPS cap — passes the chosen value back to the host
        // through *fps_limit_inout. The host wires it into a FramePacer
        // context so the main loop's sleep honours the strictest cap.
        if (fps_limit_inout != nullptr) {
            const float vals[]    = { 0.0f, 240.0f, 144.0f, 120.0f, 60.0f, 30.0f, 15.0f };
            const char* names[]   = { "Unlimited", "240", "144", "120", "60", "30", "15" };
            int idx = 0;
            for (int i = 0; i < IM_ARRAYSIZE(vals); ++i) {
                if (cardinal::abs(vals[i] - *fps_limit_inout) < 0.5f) { idx = i; break; }
            }
            const float fps_w = cardinal::clamp(toolbar_avail * 0.16f, 70.0f, 130.0f);
            char fps_label[32];
            cardinal::snprintf(fps_label, sizeof(fps_label), "##fps_cap_%u", viewport_id);
            ImGui::SetNextItemWidth(fps_w);
            if (ImGui::Combo(fps_label, &idx, names, IM_ARRAYSIZE(names))) {
                *fps_limit_inout = vals[idx];
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Per-viewport FPS cap.\n"
                                  "The main loop runs at min() of all active viewports'\n"
                                  "caps - Unlimited contributes nothing to the min.\n"
                                  "Caps below the GPU bottleneck are immediately effective;\n"
                                  "above it the GPU rate is shown.");
            }
            ImGui::SameLine();
        }

        // Maximize / Minimize toolbar buttons. Live to the right of the
        // combos so they don't compete for the FPS readout's space.
        if (minimized_inout != nullptr) {
            const float bw = ImGui::GetFrameHeight();
            if (ImGui::Button("_", ImVec2(bw, 0))) {
                *minimized_inout = true;
                ImGui::SetWindowCollapsed(true);
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Minimize");
            ImGui::SameLine();
        }
        if (maximized_inout != nullptr) {
            const float bw = ImGui::GetFrameHeight();
            const char* glyph = *maximized_inout ? "[]" : "[ ]";
            if (ImGui::Button(glyph, ImVec2(bw + 4.0f, 0))) {
                *maximized_inout = !*maximized_inout;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(*maximized_inout ? "Restore (un-maximize)" : "Maximize");
            }
            ImGui::SameLine();
        }

        // Live effective FPS readout — sampled per-panel so the user can see
        // the *actual* rate this viewport is rendering at, independent of
        // the global FPS overlay top-left. Truncated when the panel is
        // narrow (combos + buttons always win real estate).
        const float text_avail = ImGui::GetContentRegionAvail().x;
        if (text_avail > 60.0f) {
            ImGui::TextDisabled("%s | %.0f FPS", title ? title : "Viewport", ema_fps_);
        } else if (text_avail > 24.0f) {
            ImGui::TextDisabled("%.0f", ema_fps_);
        }
        ImGui::EndChild();
        ImGui::PopStyleVar();

        // Image area: the off-screen RTT auto-scales to the panel size.
        // To avoid thrashing the GPU resource as the user drags a window
        // edge, we debounce: track the last requested size, and only
        // commit a resize when the panel has been stable at that size for
        // a few frames AND the delta from the live RTT is meaningful. The
        // debounce is per-panel so each viewport resizes independently.
        //
        // Both backends now own a real per-id RTT (vectorised on D3D12 and
        // Vulkan), so every panel debounces + commits its own resize
        // independently. No backend-special-case here anymore.
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        const u32    desired_w = static_cast<u32>(cardinal::max(1.0f, avail.x));
        const u32    desired_h = static_cast<u32>(cardinal::max(1.0f, avail.y));
        auto& panel = vp_panels_[viewport_id];
        if (swapchain_ != nullptr) {
            const u32 cur_w = swapchain_->viewport_width(viewport_id);
            const u32 cur_h = swapchain_->viewport_height(viewport_id);
            const u32 dx = (desired_w > cur_w) ? desired_w - cur_w : cur_w - desired_w;
            const u32 dy = (desired_h > cur_h) ? desired_h - cur_h : cur_h - desired_h;
            const bool first_alloc = (cur_w == 0 || cur_h == 0);
            const bool meaningful  = (dx >= 16 || dy >= 16);

            if (desired_w == panel.pending_w && desired_h == panel.pending_h) {
                ++panel.stable_frames;
            } else {
                panel.pending_w     = desired_w;
                panel.pending_h     = desired_h;
                panel.stable_frames = 0;
            }

            if ((first_alloc && desired_w > 1 && desired_h > 1) ||
                (meaningful && panel.stable_frames >= 6)) {
                // Defer the actual resize to AFTER end_frame. Calling
                // set_viewport_size now would invalidate descriptors that
                // ImGui's already-queued draw lists still reference,
                // producing DEVICE_LOST when the overlay submits.
                panel.commit_w = desired_w;
                panel.commit_h = desired_h;
                panel.stable_frames = INT_MAX / 2;   // don't fire again until size changes
            }
        }

        void* tex = viewport_texture(viewport_id);
        const ImVec2 image_pos = ImGui::GetCursorScreenPos();
        if (tex != nullptr && avail.x > 1 && avail.y > 1) {
            // Pixel-perfect 1:1 — RTT is the panel's exact size.
            ImGui::Image(reinterpret_cast<ImTextureID>(tex), avail);
        } else if (avail.x > 1 && avail.y > 1) {
            ImGui::TextDisabled("(allocating viewport %ux%u…)", desired_w, desired_h);
        }

        // Translate gizmo overlay — drawn on top of the RTT image. Returns
        // true while the user is dragging an axis (so the click-to-select
        // path knows to ignore the click and avoid swapping selection
        // mid-drag). Also accumulates this frame's drag delta into
        // gizmo_drag_ which the host reads after end_frame.
        // Chunk grid overlay — drawn UNDER the gizmo so handles aren't
        // hidden by the lines. No-op when disabled or no camera pushed.
        draw_world_grid_overlay(image_pos, avail);

        // Gizmo overlay. ImGuizmo is the default; the hand-rolled overlay
        // stays in tree as a fallback under use_imguizmo_=false. Both
        // produce the same drag output (gizmo_drag_.target_world /
        // target_rotation_euler / target_scale) so the host loop reads it
        // the same way regardless of which renderer fired.
        const bool gizmo_busy =
            use_imguizmo_
                ? draw_imguizmo_overlay_(image_pos, avail, viewport_id)
                : draw_gizmo_overlay(image_pos, avail, viewport_id);

        // Per-frame interaction snapshot. Items are populated only when this
        // viewport is the one actually hovered; the host loop reads
        // last_viewport_pick() after end_frame to do picking / placement.
        if (ImGui::IsItemHovered() && avail.x > 1.0f && avail.y > 1.0f) {
            vp_pick_.hovered = true;
            vp_last_hovered_id_ = viewport_id;
            // Travel the panel id + the aspect with the pick so the host
            // unprojects with a projection matching THIS panel.
            //
            // CRITICAL: report the aspect the scene was actually RENDERED
            // with — the swapchain viewport's COMMITTED pixel size — NOT the
            // live ImGui panel `avail`. The RTT is rendered at the committed
            // size (proj = committed W/H) and then drawn STRETCHED to fill
            // `avail`; the host must unproject the click with that same
            // committed aspect or the ray is skewed and placement lands at a
            // constant offset from the cursor. The two sizes differ whenever
            // the deferred/debounced resize (see above) hasn't caught up —
            // a steady-state gap of up to the 16px commit threshold, and the
            // gap is frame-rate sensitive (so it surfaced in release but not
            // debug). Fall back to `avail` only before the RTT exists.
            vp_pick_.viewport_id = viewport_id;
            {
                const u32 cw = (swapchain_ != nullptr)
                             ? swapchain_->viewport_width(viewport_id)  : 0u;
                const u32 ch = (swapchain_ != nullptr)
                             ? swapchain_->viewport_height(viewport_id) : 0u;
                vp_pick_.aspect = (cw > 0u && ch > 0u)
                    ? static_cast<float>(cw) / static_cast<float>(ch)
                    : (avail.x / avail.y);
            }
            const ImGuiIO& io = ImGui::GetIO();

            const bool ctrl_or_shift = io.KeyCtrl || io.KeyShift;

            // LMB single-click select — low drag, no gizmo. `additive`
            // (Ctrl/Shift) tells the host to toggle into the selection
            // set rather than replace it.
            if (!gizmo_busy &&
                ImGui::IsMouseReleased(ImGuiMouseButton_Left) &&
                io.MouseDragMaxDistanceSqr[ImGuiMouseButton_Left] < 9.0f /*3px*/)
            {
                const float local_x = io.MousePos.x - image_pos.x;
                const float local_y = io.MousePos.y - image_pos.y;
                vp_pick_.click_left = true;
                vp_pick_.additive   = ctrl_or_shift;
                vp_pick_.ndc_x =       (local_x / avail.x) * 2.0f - 1.0f;
                vp_pick_.ndc_y = 1.0f - (local_y / avail.y) * 2.0f;
            }

            // Marquee box-select — left-drag over empty space while no
            // gizmo handle is engaged. Rubber-band drawn live; on release
            // the panel-local NDC rect is reported for the host to test
            // its entities against (Studio stays scene-agnostic).
            if (!gizmo_busy && gizmo_axis_active_ < 0 &&
                gizmo_handle_active_ < 0)
            {
                if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 4.0f)) {
                    if (!marquee_dragging_) {
                        // First frame of the drag — pin to this panel so
                        // the release-commit converts NDC using the same
                        // image_pos/avail the drag started in.
                        marquee_viewport_id_ = viewport_id;
                        marquee_dragging_    = true;
                    }
                    // Live-draw in any hovered panel (visual continuity);
                    // ImGui clips to the window when the rect spans panels.
                    const ImVec2 dd = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                    const ImVec2 cur = io.MousePos;
                    const ImVec2 a0{ cur.x - dd.x, cur.y - dd.y };
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    dl->AddRectFilled(a0, cur, IM_COL32(120,170,255,40));
                    dl->AddRect      (a0, cur, IM_COL32(150,190,255,200));
                } else if (marquee_dragging_ &&
                           ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                    if (viewport_id == marquee_viewport_id_) {
                        // Originator commits the rect using its own
                        // image_pos/avail — the only frame in which `a0`
                        // (drag-start in OS desktop coords) → panel-local
                        // NDC is meaningful.
                        const ImVec2 dd = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                        const ImVec2 cur = io.MousePos;
                        const ImVec2 a0{ cur.x - dd.x, cur.y - dd.y };
                        auto ndc = [&](float px, float py, float& nx, float& ny) {
                            nx =        ((px - image_pos.x) / avail.x) * 2.0f - 1.0f;
                            ny = 1.0f - ((py - image_pos.y) / avail.y) * 2.0f;
                        };
                        float ax, ay, bx, by;
                        ndc(a0.x, a0.y, ax, ay);
                        ndc(cur.x, cur.y, bx, by);
                        vp_pick_.marquee          = true;
                        vp_pick_.marquee_min_x    = cardinal::min(ax, bx);
                        vp_pick_.marquee_max_x    = cardinal::max(ax, bx);
                        vp_pick_.marquee_min_y    = cardinal::min(ay, by);
                        vp_pick_.marquee_max_y    = cardinal::max(ay, by);
                        vp_pick_.marquee_additive = ctrl_or_shift;
                    }
                    // Always clear — cross-panel release just cancels.
                    marquee_dragging_    = false;
                    marquee_viewport_id_ = 0xFFFFFFFFu;
                }
            }

            // Delete / Backspace + W/E/R gizmo-mode toggle. Gated on no
            // RMB (RMB = fly-camera look; W/E/R are modal only when not
            // navigating, matching Blender/UE) and not typing elsewhere.
            if (!io.WantCaptureKeyboard) {
                if (ImGui::IsKeyPressed(ImGuiKey_Delete) ||
                    ImGui::IsKeyPressed(ImGuiKey_Backspace)) {
                    vp_pick_.delete_pressed = true;
                }
                if (!io.MouseDown[1]) {
                    if (ImGui::IsKeyPressed(ImGuiKey_W))
                        set_gizmo_mode(GizmoMode::Translate);
                    else if (ImGui::IsKeyPressed(ImGuiKey_E))
                        set_gizmo_mode(GizmoMode::Rotate);
                    else if (ImGui::IsKeyPressed(ImGuiKey_R))
                        set_gizmo_mode(GizmoMode::Scale);
                }
            }
        }

        // Right-click context menu — must live inside the viewport's
        // Begin/End scope (BeginPopupContextWindow attaches to the
        // current window). Action is captured into vp_ctx_; the host
        // reads via Studio::last_viewport_ctx_action() AFTER end_frame.
        draw_viewport_context_menu(&vp_ctx_);

        ImGui::End();
        ImGui::PopStyleVar();
    }

    // -------------------------------------------------------------------
    // -------------------------------------------------------------------
    // World-grid overlay — draws XZ chunk-boundary lines on the Y=0 plane
    // (and a brighter rectangle around the camera's render-distance disc).
    // Each line is a world-space segment projected through the current
    // camera matrices and clipped at the panel rect. No-op when the host
    // didn't push a grid via set_world_grid_overlay or the toggle is off.
    // -------------------------------------------------------------------
    void draw_world_grid_overlay(const ImVec2& image_pos, const ImVec2& avail) {
        if (!world_overlay_.show || !gizmo_have_camera_) return;
        if (world_overlay_.chunk_size <= 0.0f)            return;
        if (avail.x < 4.0f || avail.y < 4.0f)             return;

        // Draw a square footprint of (2*render_distance + extra) chunks
        // around the camera so the user sees the actual streaming area.
        // The grid sits on the y plane of the camera's current chunk
        // (so it visibly tracks the camera's vertical level).
        const f32 cs       = world_overlay_.chunk_size;
        const i32 r        = cardinal::max<i32>(2, world_overlay_.render_distance_xz + 2);
        const i32 cx       = world_overlay_.camera_chunk_x;
        const i32 cy       = world_overlay_.camera_chunk_y;
        const i32 cz       = world_overlay_.camera_chunk_z;
        const f32 grid_y   = static_cast<f32>(cy) * cs;

        const cardinal::scene::Mat4 vp = gizmo_proj_ * gizmo_view_;
        auto project_y0 = [&](f32 wx, f32 wz, ImVec2& out_pix) -> bool {
            const cardinal::scene::Vec4 c =
                vp * cardinal::scene::Vec4{ wx, grid_y, wz, 1.0f };
            if (c.w <= 0.001f) return false;
            const f32 ndc_x = c.x / c.w;
            const f32 ndc_y = c.y / c.w;
            out_pix.x = image_pos.x + (ndc_x * 0.5f + 0.5f) * avail.x;
            out_pix.y = image_pos.y + (1.0f - (ndc_y * 0.5f + 0.5f)) * avail.y;
            return true;
        };

        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 rect_min = image_pos;
        const ImVec2 rect_max{ image_pos.x + avail.x, image_pos.y + avail.y };
        dl->PushClipRect(rect_min, rect_max, true);
        const f32 origin_x = static_cast<f32>(cx - r) * cs;
        const f32 origin_z = static_cast<f32>(cz - r) * cs;
        const f32 end_x    = static_cast<f32>(cx + r + 1) * cs;
        const f32 end_z    = static_cast<f32>(cz + r + 1) * cs;
        const i32 lines    = 2 * r + 2;

        const ImU32 col_grid   = IM_COL32(180, 180, 200, 60);
        const ImU32 col_axis   = IM_COL32(220, 220, 255, 130);
        const ImU32 col_active = IM_COL32(110, 200, 255, 180);

        // Vertical (constant X) lines.
        for (i32 i = 0; i < lines; ++i) {
            const f32 wx = origin_x + static_cast<f32>(i) * cs;
            ImVec2 a, b;
            if (project_y0(wx, origin_z, a) && project_y0(wx, end_z, b)) {
                const i32 chunk_x = cx - r + i;
                const ImU32 col = (chunk_x == 0) ? col_axis : col_grid;
                dl->AddLine(a, b, col, 1.0f);
            }
        }
        // Horizontal (constant Z) lines.
        for (i32 j = 0; j < lines; ++j) {
            const f32 wz = origin_z + static_cast<f32>(j) * cs;
            ImVec2 a, b;
            if (project_y0(origin_x, wz, a) && project_y0(end_x, wz, b)) {
                const i32 chunk_z = cz - r + j;
                const ImU32 col = (chunk_z == 0) ? col_axis : col_grid;
                dl->AddLine(a, b, col, 1.0f);
            }
        }

        // Highlight the render-distance square around the camera chunk.
        {
            const f32 ax = static_cast<f32>(cx - world_overlay_.render_distance_xz) * cs;
            const f32 az = static_cast<f32>(cz - world_overlay_.render_distance_xz) * cs;
            const f32 bx = static_cast<f32>(cx + world_overlay_.render_distance_xz + 1) * cs;
            const f32 bz = static_cast<f32>(cz + world_overlay_.render_distance_xz + 1) * cs;
            ImVec2 corners[5];
            bool ok = true;
            ok &= project_y0(ax, az, corners[0]);
            ok &= project_y0(bx, az, corners[1]);
            ok &= project_y0(bx, bz, corners[2]);
            ok &= project_y0(ax, bz, corners[3]);
            corners[4] = corners[0];
            if (ok) dl->AddPolyline(corners, 5, col_active, ImDrawFlags_Closed, 2.0f);
        }
        // Highlight the camera's current chunk.
        {
            const f32 ax = static_cast<f32>(cx)     * cs;
            const f32 az = static_cast<f32>(cz)     * cs;
            const f32 bx = static_cast<f32>(cx + 1) * cs;
            const f32 bz = static_cast<f32>(cz + 1) * cs;
            ImVec2 corners[5];
            bool ok = true;
            ok &= project_y0(ax, az, corners[0]);
            ok &= project_y0(bx, az, corners[1]);
            ok &= project_y0(bx, bz, corners[2]);
            ok &= project_y0(ax, bz, corners[3]);
            corners[4] = corners[0];
            if (ok) dl->AddPolyline(corners, 5,
                IM_COL32(255, 200, 90, 220), ImDrawFlags_Closed, 2.5f);
        }

        // ---- Hover-chunk highlight + tooltip (level-editor feel) --------
        //
        // Project the cursor's screen position into a world ray, intersect
        // with the grid plane (y == grid_y), figure out which chunk the
        // hit point falls into, then outline + label that chunk. Mirrors
        // what UE5 does in the world outliner's drag-place workflow:
        // hovering the grid shows which cell the next click would land on.
        const ImGuiIO& io = ImGui::GetIO();
        const ImVec2 mp = io.MousePos;
        if (mp.x >= image_pos.x && mp.x < image_pos.x + avail.x &&
            mp.y >= image_pos.y && mp.y < image_pos.y + avail.y &&
            avail.x > 1.0f && avail.y > 1.0f)
        {
            const float ndc_x =       (mp.x - image_pos.x) / avail.x  * 2.0f - 1.0f;
            const float ndc_y = 1.0f - (mp.y - image_pos.y) / avail.y  * 2.0f;
            const auto ray = cardinal::scene::unproject_ndc_ray(
                ndc_x, ndc_y, gizmo_view_, gizmo_proj_);
            cardinal::scene::Vec3 hit{};
            // Same float→i32 UB-cast pattern fixed in 8 runtime modules
            // (tex_ops/mesh_ops/world/scene/physics/brush/vgeom/render
            // — c420695 et al.). The grid-overlay chunk_size cs is
            // user-set on world_overlay_; the ray intersection hit can
            // be at extreme distance / non-finite if the camera matrices
            // come from a torn-out / DPI-scaled platform window with
            // degenerate aspect. Any non-finite hit.x/hit.z or non-
            // finite/zero cs would produce `floor(NaN) = NaN` →
            // `static_cast<i32>(NaN)` UB. Gate the hover-chunk reporting
            // on a fresh isfinite/positive-cs check (canNOT early-
            // return — the enclosing function has a matching
            // PopClipRect at the bottom).
            const bool cast_safe = cardinal::scene::ray_plane_y_intersect(ray, grid_y, &hit)
                                && cardinal::isfinite(hit.x)
                                && cardinal::isfinite(hit.z)
                                && cardinal::isfinite(cs) && cs > 0.0f;
            if (cast_safe) {
                const i32 hcx = static_cast<i32>(cardinal::floor(hit.x / cs));
                const i32 hcz = static_cast<i32>(cardinal::floor(hit.z / cs));
                // Outline the hovered chunk.
                const f32 ax = static_cast<f32>(hcx)     * cs;
                const f32 az = static_cast<f32>(hcz)     * cs;
                const f32 bx = static_cast<f32>(hcx + 1) * cs;
                const f32 bz = static_cast<f32>(hcz + 1) * cs;
                ImVec2 corners[5];
                bool ok = true;
                ok &= project_y0(ax, az, corners[0]);
                ok &= project_y0(bx, az, corners[1]);
                ok &= project_y0(bx, bz, corners[2]);
                ok &= project_y0(ax, bz, corners[3]);
                corners[4] = corners[0];
                if (ok) {
                    // Translucent fill + bright outline so the cell pops
                    // without obscuring whatever's under it.
                    ImVec2 fill[4] = { corners[0], corners[1],
                                       corners[2], corners[3] };
                    dl->AddConvexPolyFilled(fill, 4,
                        IM_COL32(110, 220, 255, 40));
                    dl->AddPolyline(corners, 5,
                        IM_COL32(110, 220, 255, 240),
                        ImDrawFlags_Closed, 2.0f);

                    // Coordinate badge near the cursor — "(cx, cy, cz)".
                    char lbl[64];
                    cardinal::snprintf(lbl, sizeof(lbl), "(%d, %d, %d)",
                                  hcx, cy, hcz);
                    const ImVec2 ts = ImGui::CalcTextSize(lbl);
                    const ImVec2 tp{ mp.x + 14.0f, mp.y + 14.0f };
                    dl->AddRectFilled(
                        ImVec2(tp.x - 4, tp.y - 2),
                        ImVec2(tp.x + ts.x + 4, tp.y + ts.y + 2),
                        IM_COL32(0, 0, 0, 170), 3.0f);
                    dl->AddText(tp, IM_COL32(220, 240, 255, 255), lbl);
                }
                // Stash for the right-click context menu (which fires
                // outside this fn) so it can read "the chunk under the
                // cursor at time of click".
                hover_chunk_x_     = hcx;
                hover_chunk_y_     = cy;
                hover_chunk_z_     = hcz;
                hover_chunk_valid_ = true;
            }
        }

        dl->PopClipRect();
    }

    // Translate gizmo — 3 single-axis arrow handles. Proven drag math:
    // at mouse-down capture the world axis ray; per frame unproject the
    // mouse to a ray and take the closest-point parameter between the
    // two — invariant to camera angle/distance/entity motion. Plane +
    // screen handles and snap are layered on by the dispatcher; rotate
    // and scale modes have their own helpers. Returns true while an axis
    // is actively dragged (suppresses click-select).
    // -------------------------------------------------------------------
    bool draw_translate_axes_(const ImVec2& image_pos, const ImVec2& avail,
                              u32 viewport_id) {
        if (!gizmo_has_target_ || !gizmo_have_camera_) return false;
        if (avail.x < 4.0f || avail.y < 4.0f)          return false;

        // Project an arbitrary world-space point into THIS panel's pixels.
        const cardinal::scene::Mat4 vp = gizmo_proj_ * gizmo_view_;
        auto project = [&](const cardinal::scene::Vec3& wp,
                           ImVec2& out_pixel, float& out_ndc_w) -> bool {
            const cardinal::scene::Vec4 c = vp * cardinal::scene::Vec4{ wp.x, wp.y, wp.z, 1.0f };
            if (c.w <= 0.001f) return false;
            const float ndc_x =  c.x / c.w;
            const float ndc_y =  c.y / c.w;
            out_pixel.x = image_pos.x + (ndc_x * 0.5f + 0.5f) * avail.x;
            out_pixel.y = image_pos.y + (1.0f - (ndc_y * 0.5f + 0.5f)) * avail.y;
            out_ndc_w   = c.w;
            return true;
        };

        // Axis lengths scale with depth so the gizmo stays a constant
        // screen size regardless of how far the camera is from the entity.
        ImVec2 origin_pix; float origin_w = 0.0f;
        if (!project(gizmo_target_, origin_pix, origin_w)) return false;
        // Pick a world-space axis length that yields ~80 pixels at the
        // current depth. perspective: pixels = world * (avail.y / w).
        const float target_pixels = 80.0f;
        const float world_len = cardinal::max(0.001f,
                                         target_pixels * origin_w / avail.y);

        const cardinal::scene::Vec3 ax_world[3] = {
            { gizmo_target_.x + world_len, gizmo_target_.y, gizmo_target_.z },
            { gizmo_target_.x, gizmo_target_.y + world_len, gizmo_target_.z },
            { gizmo_target_.x, gizmo_target_.y, gizmo_target_.z + world_len },
        };
        ImVec2 ax_pix[3]; float ax_w[3];
        for (int i = 0; i < 3; ++i) {
            if (!project(ax_world[i], ax_pix[i], ax_w[i])) return false;
        }

        // Hit-test against each axis line. Pick the closest within 8px.
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImGuiIO& io = ImGui::GetIO();
        const ImVec2 mp = io.MousePos;
        const bool   hovered_image = ImGui::IsItemHovered();

        auto dist_to_segment = [](ImVec2 p, ImVec2 a, ImVec2 b) {
            const float ax_ = b.x - a.x, ay_ = b.y - a.y;
            const float pax = p.x - a.x, pay = p.y - a.y;
            const float len2 = ax_ * ax_ + ay_ * ay_;
            if (len2 < 1e-6f) {
                const float dx_ = p.x - a.x, dy_ = p.y - a.y;
                return cardinal::sqrt(dx_ * dx_ + dy_ * dy_);
            }
            float t = (pax * ax_ + pay * ay_) / len2;
            t = cardinal::clamp(t, 0.0f, 1.0f);
            const float qx = a.x + t * ax_, qy = a.y + t * ay_;
            const float dx_ = p.x - qx, dy_ = p.y - qy;
            return cardinal::sqrt(dx_ * dx_ + dy_ * dy_);
        };

        // Determine hover axis (only meaningful while the panel is hovered
        // and we're not currently dragging another handle).
        int hover_axis = -1;
        if (hovered_image && gizmo_axis_active_ < 0) {
            float best_d = 8.0f;
            for (int i = 0; i < 3; ++i) {
                const float d = dist_to_segment(mp, origin_pix, ax_pix[i]);
                if (d < best_d) { best_d = d; hover_axis = i; }
            }
        }

        // ----- Drag state machine -----------------------------------------
        // The OLD math projected the mouse pixel-delta onto the axis's
        // screen-space direction and multiplied by a depth-derived
        // `world_len` scale factor. Bug: when dragging Z (or any axis with
        // a depth component), the entity's projected depth changes, which
        // changes `world_len` mid-drag, which changes the axis_world scale,
        // which makes the entity rubberband — and if depth → 0, world_len
        // → ∞ and the entity disappears off the world map.
        //
        // The fix is the textbook engine approach: at drag-start capture
        // the world-space axis ray; per frame unproject the mouse pixel to
        // a world-space ray, find the closest-point parameter `t` between
        // the two rays, and that's where the gizmo should be along its
        // axis. Result: pure axis dragging that is invariant to camera
        // angle, distance, and the entity's own movement during the drag.
        auto closest_t_on_axis_from_mouse = [&](ImVec2 mouse_local_to_image)
            -> cardinal::pair<bool, float>
        {
            const float ndc_x =        (mouse_local_to_image.x / avail.x) * 2.0f - 1.0f;
            const float ndc_y = 1.0f - (mouse_local_to_image.y / avail.y) * 2.0f;
            const cardinal::scene::Mat4 inv_vp = vp.inverse();
            const cardinal::scene::Vec4 nclip{ ndc_x, ndc_y, 0.0f, 1.0f };
            const cardinal::scene::Vec4 fclip{ ndc_x, ndc_y, 1.0f, 1.0f };
            const cardinal::scene::Vec4 nw_  = inv_vp * nclip;
            const cardinal::scene::Vec4 fw_  = inv_vp * fclip;
            if (cardinal::fabs(nw_.w) < 1e-6f || cardinal::fabs(fw_.w) < 1e-6f)
                return { false, 0.0f };
            const cardinal::scene::Vec3 r_origin{ nw_.x/nw_.w, nw_.y/nw_.w, nw_.z/nw_.w };
            const cardinal::scene::Vec3 r_far   { fw_.x/fw_.w, fw_.y/fw_.w, fw_.z/fw_.w };
            const cardinal::scene::Vec3 r_dir   = cardinal::scene::normalize(
                cardinal::scene::Vec3{ r_far.x - r_origin.x,
                                       r_far.y - r_origin.y,
                                       r_far.z - r_origin.z });
            // Closest point between mouse ray and axis ray. Axis ray is
            // captured at drag-start (origin_world + axis_dir * t).
            const cardinal::scene::Vec3& a0 = gizmo_drag_origin_world_;
            const cardinal::scene::Vec3& da = gizmo_drag_axis_dir_;
            const cardinal::scene::Vec3  diff{ a0.x - r_origin.x,
                                               a0.y - r_origin.y,
                                               a0.z - r_origin.z };
            const float aa = cardinal::scene::dot(da, da);   // = 1 (unit)
            const float bb = cardinal::scene::dot(da, r_dir);
            const float cc = cardinal::scene::dot(r_dir, r_dir); // = 1
            const float dd = cardinal::scene::dot(da, diff);
            const float ee = cardinal::scene::dot(r_dir, diff);
            const float denom = aa * cc - bb * bb;
            if (cardinal::fabs(denom) < 1e-6f) return { false, 0.0f };  // parallel
            const float t_axis = (bb * ee - cc * dd) / denom;
            return { true, t_axis };
        };

        if (gizmo_axis_active_ < 0) {
            if (hover_axis >= 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                gizmo_axis_active_       = hover_axis;
                gizmo_drag_origin_pix_   = mp;
                gizmo_drag_origin_world_ = gizmo_target_;
                gizmo_drag_axis_dir_     =
                    (hover_axis == 0) ? cardinal::scene::Vec3{ 1.0f, 0.0f, 0.0f } :
                    (hover_axis == 1) ? cardinal::scene::Vec3{ 0.0f, 1.0f, 0.0f } :
                                        cardinal::scene::Vec3{ 0.0f, 0.0f, 1.0f };
                // Capture the axis-parameter t the user "grabbed" so the
                // gizmo doesn't snap to mouse on click — it stays where
                // the user clicked, and movement is always relative.
                const ImVec2 click_local{ mp.x - image_pos.x, mp.y - image_pos.y };
                auto r = closest_t_on_axis_from_mouse(click_local);
                gizmo_drag_t_origin_ = r.first ? r.second : 0.0f;
                gizmo_drag_in_progress_ = true;     // for undo coalescing
                // Pin the drag to THIS panel so multi-viewport setups
                // don't have every visible panel re-run the drag math
                // each frame with its own matrices (last-drawn wins,
                // teleports the object).
                gizmo_drag_viewport_id_ = viewport_id;
            }
        } else {
            if (!io.MouseDown[0]) {
                // Mouse-up handled in ANY panel — if the owning panel is
                // closed mid-drag the release would otherwise never fire.
                gizmo_axis_active_ = -1;
                gizmo_drag_in_progress_ = false;
                gizmo_drag_release_ = true;          // host fires the undo on this edge
                gizmo_drag_viewport_id_ = 0xFFFFFFFFu;
            } else if (viewport_id != gizmo_drag_viewport_id_) {
                // Drag is live in a sibling panel — render only, never
                // overwrite gizmo_target_ from this panel's frame.
            } else {
                const ImVec2 mouse_local{ mp.x - image_pos.x, mp.y - image_pos.y };
                auto r = closest_t_on_axis_from_mouse(mouse_local);
                if (r.first) {
                    const float dt = r.second - gizmo_drag_t_origin_;
                    cardinal::scene::Vec3 new_pos{
                        gizmo_drag_origin_world_.x + gizmo_drag_axis_dir_.x * dt,
                        gizmo_drag_origin_world_.y + gizmo_drag_axis_dir_.y * dt,
                        gizmo_drag_origin_world_.z + gizmo_drag_axis_dir_.z * dt,
                    };
                    // Reject NaN / inf — should not happen with the new
                    // closest-point math, but defend against degenerate
                    // camera setups (entity behind camera, etc).
                    if (cardinal::isfinite(new_pos.x) && cardinal::isfinite(new_pos.y)
                        && cardinal::isfinite(new_pos.z))
                    {
                        gizmo_drag_.delta = {
                            new_pos.x - gizmo_target_.x,
                            new_pos.y - gizmo_target_.y,
                            new_pos.z - gizmo_target_.z,
                        };
                        // Absolute path — recomputed each frame from the
                        // FIXED drag origin + axis * dt. No integration,
                        // so no float drift no matter how far the entity
                        // travels. Hosts should prefer this over delta.
                        gizmo_drag_.target_world = new_pos;
                        gizmo_drag_.active       = true;
                        gizmo_target_            = new_pos;
                        // Re-project for the same-frame visual update.
                        if (project(gizmo_target_, origin_pix, origin_w)) {
                            for (int i = 0; i < 3; ++i) {
                                cardinal::scene::Vec3 endp = gizmo_target_;
                                if      (i == 0) endp.x += world_len;
                                else if (i == 1) endp.y += world_len;
                                else             endp.z += world_len;
                                project(endp, ax_pix[i], ax_w[i]);
                            }
                        }
                    }
                }
            }
        }

        // Render — three 2D arrows in axis colours. X=red, Y=green, Z=blue.
        // Highlight the hovered/dragged axis with a thicker, brighter line.
        const ImU32 col_x = IM_COL32(220,  60,  60, 255);
        const ImU32 col_y = IM_COL32( 60, 200,  80, 255);
        const ImU32 col_z = IM_COL32( 70, 130, 235, 255);
        const ImU32 col_hot[3] = { IM_COL32(255,200,120,255),
                                   IM_COL32(255,200,120,255),
                                   IM_COL32(255,200,120,255) };
        const ImU32 base[3] = { col_x, col_y, col_z };

        for (int i = 0; i < 3; ++i) {
            const bool hot = (i == hover_axis) || (i == gizmo_axis_active_);
            const ImU32 c  = hot ? col_hot[i] : base[i];
            const float th = hot ? 4.0f : 2.5f;
            dl->AddLine(origin_pix, ax_pix[i], c, th);
            // Arrowhead — small triangle perpendicular to the line.
            const ImVec2 d{ ax_pix[i].x - origin_pix.x,
                            ax_pix[i].y - origin_pix.y };
            const float  ln = cardinal::sqrt(d.x * d.x + d.y * d.y);
            if (ln > 1e-3f) {
                const ImVec2 dn{ d.x / ln, d.y / ln };
                const ImVec2 pn{ -dn.y, dn.x };
                const float head = hot ? 12.0f : 9.0f;
                const ImVec2 b{ ax_pix[i].x - dn.x * head + pn.x * head * 0.5f,
                                ax_pix[i].y - dn.y * head + pn.y * head * 0.5f };
                const ImVec2 c2{ ax_pix[i].x - dn.x * head - pn.x * head * 0.5f,
                                 ax_pix[i].y - dn.y * head - pn.y * head * 0.5f };
                dl->AddTriangleFilled(ax_pix[i], b, c2, c);
            }
        }
        // Origin handle — a small filled square so it's clear where the
        // gizmo's pivot is.
        dl->AddRectFilled(ImVec2(origin_pix.x - 4, origin_pix.y - 4),
                          ImVec2(origin_pix.x + 4, origin_pix.y + 4),
                          IM_COL32(230, 230, 230, 255));

        return gizmo_axis_active_ >= 0;
    }

    // ===================================================================
    // Multi-mode gizmo: shared math + Rotate/Scale/Plane helpers and the
    // public dispatcher. All drags are drift-free (results recomputed
    // each frame from a fixed mouse-down anchor) and undo-coalesced
    // (drag_in_progress/release edges fire for every mode).
    // ===================================================================
    // euler<->quat + Hamilton product now live in cardinal::core::math
    // (canonical, deterministically tested). The rotate gizmo uses
    // cardinal::core::quat_from_euler_xyz / quat_to_euler_xyz and
    // core::Quat::operator* directly — no private duplicates.

    // project_ / mouse_world_ray_ / ray_plane_ are pure gizmo-math
    // primitives — their bodies moved VERBATIM to
    // studio_gizmo_math.{hpp,cpp} (Studio decomposition step 2). These
    // thin forwarders keep the exact signatures so every existing
    // (unqualified) call site is unchanged and behaviour is identical;
    // the heavy math now lives in one small, independently-editable file.
    bool project_(const cardinal::scene::Mat4& vp, const ImVec2& ip,
                  const ImVec2& av, const cardinal::scene::Vec3& wp,
                  ImVec2& out) const {
        return detail::project_(vp, ip, av, wp, out);
    }
    bool mouse_world_ray_(const ImVec2& ip, const ImVec2& av,
                          const cardinal::scene::Mat4& inv_vp,
                          cardinal::scene::Vec3& ro,
                          cardinal::scene::Vec3& rd) const {
        return detail::mouse_world_ray_(ip, av, inv_vp, ro, rd);
    }
    static bool ray_plane_(const cardinal::scene::Vec3& ro,
                           const cardinal::scene::Vec3& rd,
                           const cardinal::scene::Vec3& p0,
                           const cardinal::scene::Vec3& n,
                           cardinal::scene::Vec3& hit) {
        return detail::ray_plane_(ro, rd, p0, n, hit);
    }
    // Camera world position = inverse(view) translation column.
    cardinal::scene::Vec3 camera_pos_() const {
        const cardinal::scene::Mat4 iv = gizmo_view_.inverse();
        return { iv.m[3][0], iv.m[3][1], iv.m[3][2] };
    }
    float snapf_(float v, float step) const {
        if (!gizmo_snap_on_ || step <= 0.0f) return v;
        return cardinal::round(v / step) * step;
    }

    // ----- Translate: 3 plane handles (YZ/XZ/XY) + screen-space square --
    // Engages only when no single-axis drag is live. handle ids:
    //   3=YZ(normal X) 4=XZ(normal Y) 5=XY(normal Z) 6=screen(camera-facing)
    bool draw_translate_planes_screen_(const ImVec2& ip, const ImVec2& av,
                                       u32 viewport_id) {
        const cardinal::scene::Mat4 vp = gizmo_proj_ * gizmo_view_;
        const cardinal::scene::Mat4 ivp = vp.inverse();
        ImVec2 o_pix;
        if (!project_(vp, ip, av, gizmo_target_, o_pix)) return false;
        float ow = 0.0f;
        { const cardinal::scene::Vec4 c = vp * cardinal::scene::Vec4{
              gizmo_target_.x, gizmo_target_.y, gizmo_target_.z, 1.0f };
          ow = c.w; }
        const float wlen = cardinal::max(0.001f, 80.0f * ow / av.y);
        const float quad = wlen * 0.32f;     // plane patch half-size

        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImGuiIO& io = ImGui::GetIO();
        const ImVec2 mp = io.MousePos;
        const bool hov = ImGui::IsItemHovered();

        // Plane definitions: {handle, normalAxis, uAxis, vAxis, colour}.
        struct Pl { int h; cardinal::scene::Vec3 n, u, v; ImU32 col; };
        const Pl pls[3] = {
            { 3, {1,0,0}, {0,1,0}, {0,0,1}, IM_COL32(220, 70, 70,110) },
            { 4, {0,1,0}, {0,0,1}, {1,0,0}, IM_COL32( 70,200, 90,110) },
            { 5, {0,0,1}, {1,0,0}, {0,1,0}, IM_COL32( 80,140,235,110) },
        };
        auto quad_pts = [&](const Pl& p, ImVec2 out[4]) -> bool {
            const cardinal::scene::Vec3 c[4] = {
                { gizmo_target_.x + p.u.x*quad,           gizmo_target_.y + p.u.y*quad,           gizmo_target_.z + p.u.z*quad },
                { gizmo_target_.x + (p.u.x+p.v.x)*quad,   gizmo_target_.y + (p.u.y+p.v.y)*quad,   gizmo_target_.z + (p.u.z+p.v.z)*quad },
                { gizmo_target_.x + p.v.x*quad,           gizmo_target_.y + p.v.y*quad,           gizmo_target_.z + p.v.z*quad },
                {  gizmo_target_.x, gizmo_target_.y, gizmo_target_.z },
            };
            for (int i = 0; i < 4; ++i)
                if (!project_(vp, ip, av, c[i], out[i])) return false;
            return true;
        };
        auto in_quad = [](const ImVec2& q, const ImVec2 v4[4]) {
            int cnt = 0;
            for (int i = 0, j = 3; i < 4; j = i++) {
                if (((v4[i].y > q.y) != (v4[j].y > q.y)) &&
                    (q.x < (v4[j].x - v4[i].x) * (q.y - v4[i].y) /
                           (v4[j].y - v4[i].y) + v4[i].x))
                    cnt = !cnt;
            }
            return cnt != 0;
        };

        int hover_h = -1;
        ImVec2 qv[3][4];
        bool qok[3];
        for (int i = 0; i < 3; ++i) {
            qok[i] = quad_pts(pls[i], qv[i]);
            if (qok[i] && hov && gizmo_handle_active_ < 0 &&
                gizmo_axis_active_ < 0 && in_quad(mp, qv[i]))
                hover_h = pls[i].h;
        }
        // Screen handle: small square at the gizmo origin.
        const float sq = 9.0f;
        const bool over_screen =
            hov && gizmo_handle_active_ < 0 && gizmo_axis_active_ < 0 &&
            cardinal::fabs(mp.x - o_pix.x) < sq && cardinal::fabs(mp.y - o_pix.y) < sq;
        if (over_screen && hover_h < 0) hover_h = 6;

        auto plane_normal = [&](int h) -> cardinal::scene::Vec3 {
            if (h == 3) return { 1,0,0 };
            if (h == 4) return { 0,1,0 };
            if (h == 5) return { 0,0,1 };
            const cardinal::scene::Vec3 cp = camera_pos_();
            return cardinal::scene::normalize(cardinal::scene::Vec3{
                cp.x - gizmo_target_.x, cp.y - gizmo_target_.y,
                cp.z - gizmo_target_.z });
        };

        if (gizmo_handle_active_ < 0) {
            if (hover_h >= 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                cardinal::scene::Vec3 ro, rd;
                if (mouse_world_ray_(ip, av, ivp, ro, rd)) {
                    cardinal::scene::Vec3 hit;
                    if (ray_plane_(ro, rd, gizmo_target_,
                                   plane_normal(hover_h), hit)) {
                        gizmo_handle_active_     = hover_h;
                        gizmo_drag_origin_world_ = gizmo_target_;
                        gizmo_drag_plane_hit0_   = hit;
                        gizmo_drag_in_progress_  = true;
                        gizmo_drag_viewport_id_  = viewport_id;   // pin to this panel
                    }
                }
            }
        } else if (gizmo_handle_active_ >= 3) {
            if (!io.MouseDown[0]) {
                gizmo_handle_active_ = -1;
                gizmo_drag_in_progress_ = false;
                gizmo_drag_release_ = true;
                gizmo_drag_viewport_id_ = 0xFFFFFFFFu;
            } else if (viewport_id != gizmo_drag_viewport_id_) {
                // Sibling panel — render only, never overwrite drag state.
            } else {
                cardinal::scene::Vec3 ro, rd;
                if (mouse_world_ray_(ip, av, ivp, ro, rd)) {
                    cardinal::scene::Vec3 hit;
                    if (ray_plane_(ro, rd, gizmo_drag_origin_world_,
                                   plane_normal(gizmo_handle_active_), hit)) {
                        cardinal::scene::Vec3 np{
                            gizmo_drag_origin_world_.x + (hit.x - gizmo_drag_plane_hit0_.x),
                            gizmo_drag_origin_world_.y + (hit.y - gizmo_drag_plane_hit0_.y),
                            gizmo_drag_origin_world_.z + (hit.z - gizmo_drag_plane_hit0_.z) };
                        if (cardinal::isfinite(np.x) && cardinal::isfinite(np.y) &&
                            cardinal::isfinite(np.z)) {
                            np.x = snapf_(np.x, gizmo_snap_step_);
                            np.y = snapf_(np.y, gizmo_snap_step_);
                            np.z = snapf_(np.z, gizmo_snap_step_);
                            gizmo_drag_.delta = { np.x-gizmo_target_.x,
                                                  np.y-gizmo_target_.y,
                                                  np.z-gizmo_target_.z };
                            gizmo_drag_.target_world = np;
                            gizmo_drag_.active = true;
                            gizmo_drag_.mode = GizmoMode::Translate;
                            gizmo_target_ = np;
                            project_(vp, ip, av, gizmo_target_, o_pix);
                        }
                    }
                }
            }
        }

        // Render the plane patches + screen square.
        for (int i = 0; i < 3; ++i) {
            if (!qok[i]) continue;
            const bool hot = (pls[i].h == hover_h) ||
                             (pls[i].h == gizmo_handle_active_);
            const ImU32 fill = hot ? IM_COL32(255,200,120,150) : pls[i].col;
            dl->AddQuadFilled(qv[i][0], qv[i][1], qv[i][2], qv[i][3], fill);
            dl->AddQuad(qv[i][0], qv[i][1], qv[i][2], qv[i][3],
                        IM_COL32(255,255,255,150), 1.0f);
        }
        const bool sc_hot = (hover_h == 6) || (gizmo_handle_active_ == 6);
        dl->AddRectFilled(ImVec2(o_pix.x - sq, o_pix.y - sq),
                          ImVec2(o_pix.x + sq, o_pix.y + sq),
                          sc_hot ? IM_COL32(255,200,120,180)
                                 : IM_COL32(210,210,210,90));
        return gizmo_handle_active_ >= 3;
    }

    // ----- Rotate: 3 world-axis rings -----------------------------------
    bool draw_rotate_gizmo_(const ImVec2& ip, const ImVec2& av,
                            u32 viewport_id) {
        const cardinal::scene::Mat4 vp  = gizmo_proj_ * gizmo_view_;
        const cardinal::scene::Mat4 ivp = vp.inverse();
        ImVec2 o_pix;
        if (!project_(vp, ip, av, gizmo_target_, o_pix)) return false;
        float ow = 0.0f;
        { const cardinal::scene::Vec4 c = vp * cardinal::scene::Vec4{
              gizmo_target_.x, gizmo_target_.y, gizmo_target_.z, 1.0f };
          ow = c.w; }
        const float radius = cardinal::max(0.001f, 92.0f * ow / av.y);

        struct Rg { cardinal::scene::Vec3 n, u, v; ImU32 col; };
        const Rg rings[3] = {
            { {1,0,0}, {0,1,0}, {0,0,1}, IM_COL32(220, 70, 70,255) },
            { {0,1,0}, {0,0,1}, {1,0,0}, IM_COL32( 70,200, 90,255) },
            { {0,0,1}, {1,0,0}, {0,1,0}, IM_COL32( 80,140,235,255) },
        };
        constexpr int SEG = 56;
        ImVec2 ring_pix[3][SEG];
        bool   ring_ok[3];
        for (int a = 0; a < 3; ++a) {
            ring_ok[a] = true;
            for (int s = 0; s < SEG; ++s) {
                const float th =
                    static_cast<float>(s) / static_cast<float>(SEG) * 6.2831853f;
                const float cu = cardinal::cos(th) * radius;
                const float cv = cardinal::sin(th) * radius;
                const cardinal::scene::Vec3 wp{
                    gizmo_target_.x + rings[a].u.x*cu + rings[a].v.x*cv,
                    gizmo_target_.y + rings[a].u.y*cu + rings[a].v.y*cv,
                    gizmo_target_.z + rings[a].u.z*cu + rings[a].v.z*cv };
                if (!project_(vp, ip, av, wp, ring_pix[a][s]))
                    ring_ok[a] = false;
            }
        }
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImGuiIO& io = ImGui::GetIO();
        const ImVec2 mp = io.MousePos;
        const bool hov = ImGui::IsItemHovered();

        auto seg_d = [](ImVec2 p, ImVec2 a, ImVec2 b) {
            const float ex = b.x-a.x, ey = b.y-a.y;
            const float l2 = ex*ex + ey*ey;
            float t = (l2 < 1e-6f) ? 0.0f
                : cardinal::clamp(((p.x-a.x)*ex + (p.y-a.y)*ey) / l2, 0.0f, 1.0f);
            const float qx = a.x+t*ex, qy = a.y+t*ey;
            return cardinal::sqrt((p.x-qx)*(p.x-qx) + (p.y-qy)*(p.y-qy));
        };
        int hover_a = -1;
        if (hov && gizmo_handle_active_ < 0) {
            float best = 7.0f;
            for (int a = 0; a < 3; ++a) {
                if (!ring_ok[a]) continue;
                for (int s = 0; s < SEG; ++s) {
                    const float d = seg_d(mp, ring_pix[a][s],
                                          ring_pix[a][(s+1)%SEG]);
                    if (d < best) { best = d; hover_a = a; }
                }
            }
        }
        auto angle_on = [&](int a, float& ang) -> bool {
            cardinal::scene::Vec3 ro, rd, hit;
            if (!mouse_world_ray_(ip, av, ivp, ro, rd)) return false;
            if (!ray_plane_(ro, rd, gizmo_target_, rings[a].n, hit))
                return false;
            const cardinal::scene::Vec3 w{ hit.x-gizmo_target_.x,
                hit.y-gizmo_target_.y, hit.z-gizmo_target_.z };
            ang = cardinal::atan2(cardinal::scene::dot(w, rings[a].v),
                             cardinal::scene::dot(w, rings[a].u));
            return true;
        };

        if (gizmo_handle_active_ < 0) {
            if (hover_a >= 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                float a0;
                if (angle_on(hover_a, a0)) {
                    gizmo_handle_active_   = hover_a;
                    gizmo_drag_angle0_     = a0;
                    gizmo_drag_origin_rot_ = gizmo_target_rot_;
                    gizmo_drag_in_progress_= true;
                    gizmo_drag_viewport_id_ = viewport_id;   // pin to this panel
                }
            }
        } else {
            if (!io.MouseDown[0]) {
                gizmo_handle_active_ = -1;
                gizmo_drag_in_progress_ = false;
                gizmo_drag_release_ = true;
                gizmo_drag_viewport_id_ = 0xFFFFFFFFu;
            } else if (viewport_id != gizmo_drag_viewport_id_) {
                // Sibling panel — render only, never overwrite rotation.
            } else {
                float a1;
                if (angle_on(gizmo_handle_active_, a1)) {
                    float d = a1 - gizmo_drag_angle0_;
                    d = cardinal::atan2(cardinal::sin(d), cardinal::cos(d));   // shortest
                    if (gizmo_snap_on_ && gizmo_snap_step_ > 0.0f) {
                        const float st = gizmo_snap_step_ *
                                         0.01745329252f;        // deg→rad
                        d = cardinal::round(d / st) * st;
                    }
                    const cardinal::scene::Vec3& ax =
                        rings[gizmo_handle_active_].n;
                    const float hs = cardinal::sin(d*0.5f), hc = cardinal::cos(d*0.5f);
                    // world-axis incremental rotation ⊗ origin orientation
                    const cardinal::core::Quat qa{
                        ax.x*hs, ax.y*hs, ax.z*hs, hc };
                    const cardinal::core::Quat oq =
                        cardinal::core::quat_from_euler_xyz(gizmo_drag_origin_rot_);
                    const cardinal::scene::Vec3 e =
                        cardinal::core::quat_to_euler_xyz(qa * oq);
                    if (cardinal::isfinite(e.x) && cardinal::isfinite(e.y) &&
                        cardinal::isfinite(e.z)) {
                        gizmo_target_rot_ = e;
                        gizmo_drag_.active = true;
                        gizmo_drag_.mode = GizmoMode::Rotate;
                        gizmo_drag_.target_rotation_euler = e;
                    }
                }
            }
        }

        for (int a = 0; a < 3; ++a) {
            if (!ring_ok[a]) continue;
            const bool hot = (a == hover_a) || (a == gizmo_handle_active_);
            const ImU32 c = hot ? IM_COL32(255,200,120,255) : rings[a].col;
            for (int s = 0; s < SEG; ++s)
                dl->AddLine(ring_pix[a][s], ring_pix[a][(s+1)%SEG], c,
                            hot ? 3.0f : 2.0f);
        }
        dl->AddCircleFilled(o_pix, 3.0f, IM_COL32(230,230,230,255));
        return gizmo_handle_active_ >= 0;
    }

    // ----- Scale: 3 axis box-handles + uniform centre box ---------------
    bool draw_scale_gizmo_(const ImVec2& ip, const ImVec2& av,
                           u32 viewport_id) {
        const cardinal::scene::Mat4 vp  = gizmo_proj_ * gizmo_view_;
        const cardinal::scene::Mat4 ivp = vp.inverse();
        ImVec2 o_pix;
        if (!project_(vp, ip, av, gizmo_target_, o_pix)) return false;
        float ow = 0.0f;
        { const cardinal::scene::Vec4 c = vp * cardinal::scene::Vec4{
              gizmo_target_.x, gizmo_target_.y, gizmo_target_.z, 1.0f };
          ow = c.w; }
        const float wlen = cardinal::max(0.001f, 78.0f * ow / av.y);
        const cardinal::scene::Vec3 axd[3] = {
            {1,0,0},{0,1,0},{0,0,1} };
        const ImU32 axc[3] = { IM_COL32(220,70,70,255),
                               IM_COL32(70,200,90,255),
                               IM_COL32(80,140,235,255) };
        ImVec2 tip[3]; bool tok[3];
        for (int a = 0; a < 3; ++a) {
            const cardinal::scene::Vec3 wp{
                gizmo_target_.x + axd[a].x*wlen,
                gizmo_target_.y + axd[a].y*wlen,
                gizmo_target_.z + axd[a].z*wlen };
            tok[a] = project_(vp, ip, av, wp, tip[a]);
        }
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImGuiIO& io = ImGui::GetIO();
        const ImVec2 mp = io.MousePos;
        const bool hov = ImGui::IsItemHovered();

        int hover_h = -1;
        if (hov && gizmo_handle_active_ < 0) {
            for (int a = 0; a < 3; ++a)
                if (tok[a] && cardinal::fabs(mp.x-tip[a].x) < 8.0f &&
                    cardinal::fabs(mp.y-tip[a].y) < 8.0f) hover_h = a;
            if (hover_h < 0 && cardinal::fabs(mp.x-o_pix.x) < 9.0f &&
                cardinal::fabs(mp.y-o_pix.y) < 9.0f) hover_h = 6;   // uniform
        }
        auto axis_t = [&](int a, float& t) -> bool {
            cardinal::scene::Vec3 ro, rd;
            if (!mouse_world_ray_(ip, av, ivp, ro, rd)) return false;
            const cardinal::scene::Vec3& da = axd[a];
            const cardinal::scene::Vec3 df{ gizmo_target_.x-ro.x,
                gizmo_target_.y-ro.y, gizmo_target_.z-ro.z };
            const float bb = cardinal::scene::dot(da, rd);
            const float dd = cardinal::scene::dot(da, df);
            const float ee = cardinal::scene::dot(rd, df);
            const float den = 1.0f - bb*bb;
            if (cardinal::fabs(den) < 1e-6f) return false;
            t = (bb*ee - dd) / den;
            return cardinal::isfinite(t);
        };

        if (gizmo_handle_active_ < 0) {
            if (hover_h >= 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                gizmo_handle_active_    = hover_h;
                gizmo_drag_origin_scl_  = gizmo_target_scl_;
                gizmo_drag_in_progress_ = true;
                gizmo_drag_viewport_id_ = viewport_id;   // pin to this panel
                if (hover_h == 6) {
                    gizmo_drag_uniform0_ = mp.y;
                } else {
                    float t0 = 0.0f; axis_t(hover_h, t0);
                    gizmo_drag_t_origin_ = t0;
                }
            }
        } else {
            if (!io.MouseDown[0]) {
                gizmo_handle_active_ = -1;
                gizmo_drag_in_progress_ = false;
                gizmo_drag_release_ = true;
                gizmo_drag_viewport_id_ = 0xFFFFFFFFu;
            } else if (viewport_id != gizmo_drag_viewport_id_) {
                // Sibling panel — render only, never overwrite scale.
            } else {
                cardinal::scene::Vec3 ns = gizmo_drag_origin_scl_;
                bool ok = false;
                if (gizmo_handle_active_ == 6) {
                    const float f = cardinal::pow(2.0f,
                        (gizmo_drag_uniform0_ - mp.y) / 120.0f);
                    ns = { gizmo_drag_origin_scl_.x*f,
                           gizmo_drag_origin_scl_.y*f,
                           gizmo_drag_origin_scl_.z*f };
                    ok = cardinal::isfinite(f);
                } else {
                    float t1 = 0.0f;
                    if (axis_t(gizmo_handle_active_, t1)) {
                        const float f = cardinal::clamp(
                            1.0f + (t1 - gizmo_drag_t_origin_) /
                                   cardinal::max(0.001f, wlen),
                            0.01f, 100.0f);
                        if (gizmo_handle_active_ == 0) ns.x = gizmo_drag_origin_scl_.x*f;
                        if (gizmo_handle_active_ == 1) ns.y = gizmo_drag_origin_scl_.y*f;
                        if (gizmo_handle_active_ == 2) ns.z = gizmo_drag_origin_scl_.z*f;
                        ok = cardinal::isfinite(f);
                    }
                }
                if (ok) {
                    if (gizmo_snap_on_ && gizmo_snap_step_ > 0.0f) {
                        auto sn = [&](float v){
                            return cardinal::max(0.01f,
                                cardinal::round(v/gizmo_snap_step_)*gizmo_snap_step_); };
                        ns = { sn(ns.x), sn(ns.y), sn(ns.z) };
                    }
                    if (cardinal::isfinite(ns.x) && cardinal::isfinite(ns.y) &&
                        cardinal::isfinite(ns.z)) {
                        gizmo_target_scl_ = ns;
                        gizmo_drag_.active = true;
                        gizmo_drag_.mode = GizmoMode::Scale;
                        gizmo_drag_.target_scale = ns;
                    }
                }
            }
        }

        for (int a = 0; a < 3; ++a) {
            if (!tok[a]) continue;
            const bool hot = (a == hover_h) || (a == gizmo_handle_active_);
            const ImU32 c = hot ? IM_COL32(255,200,120,255) : axc[a];
            dl->AddLine(o_pix, tip[a], c, hot ? 3.5f : 2.5f);
            dl->AddRectFilled(ImVec2(tip[a].x-5, tip[a].y-5),
                              ImVec2(tip[a].x+5, tip[a].y+5), c);
        }
        const bool uhot = (hover_h == 6) || (gizmo_handle_active_ == 6);
        dl->AddRectFilled(ImVec2(o_pix.x-5, o_pix.y-5),
                          ImVec2(o_pix.x+5, o_pix.y+5),
                          uhot ? IM_COL32(255,200,120,255)
                               : IM_COL32(230,230,230,255));
        return gizmo_handle_active_ >= 0;
    }

    // ----- ImGuizmo-backed gizmo overlay --------------------------------
    // Industry-standard renderer (the gizmo you see in Unity / Unreal /
    // every Blender-inspired editor). Drives the same gizmo_drag_ output
    // as the hand-rolled path so the host doesn't care which renderer
    // fired. Translate / Rotate / Scale picked off gizmo_mode_; local-vs-
    // world space from imguizmo_local_space_; snap from gizmo_snap_on_ +
    // gizmo_snap_step_ (unit semantics match the existing path).
    bool draw_imguizmo_overlay_(const ImVec2& image_pos, const ImVec2& avail,
                                u32 viewport_id) {
        if (!gizmo_has_target_ || !gizmo_have_camera_) return false;
        if (avail.x < 4.0f || avail.y < 4.0f)          return false;

        // Only ONE viewport runs ImGuizmo per frame — the active-drag owner,
        // else the last-hovered panel. ImGuizmo keeps a single global context;
        // if every visible panel called Manipulate, each would re-solve the
        // drag from its OWN camera and the last-drawn panel would win,
        // teleporting the object (the severe multi-viewport offset). Gating to
        // the owning panel keeps the gizmo in the focused viewport and the
        // manipulation correct — with a single viewport this is always true,
        // so normal single-viewport editing is unaffected.
        const u32 gizmo_owner = gizmo_drag_in_progress_ ? gizmo_drag_viewport_id_
                                                        : vp_last_hovered_id_;
        if (viewport_id != gizmo_owner) return false;

        // Per-frame defaults — host reads these every frame.
        gizmo_drag_.active                = false;
        gizmo_drag_.mode                  = gizmo_mode_;
        gizmo_drag_.target_rotation_euler = gizmo_target_rot_;
        gizmo_drag_.target_scale          = gizmo_target_scl_;

        // ImGuizmo draws onto the current window's foreground draw list and
        // clips to the rect set here. Setting both each frame is required:
        // ImGui's draw list pointer is per-window and the rect can change
        // when the user docks / resizes the viewport panel.
        ImGuizmo::SetDrawlist();
        ImGuizmo::SetRect(image_pos.x, image_pos.y, avail.x, avail.y);

        // Build the gizmo's input model matrix from the host-published
        // target position + euler rotation + scale. ImGuizmo wants a
        // column-major float[16]; Cardinal scene::Mat4 stores the same
        // layout. RecomposeMatrixFromComponents handles the YPR ordering.
        float matrix[16];
        float t_in[3] = { gizmo_target_.x,     gizmo_target_.y,     gizmo_target_.z };
        // ImGuizmo wants degrees for the recompose helper.
        constexpr float kRad2Deg = 57.29577951308232f;
        float r_in[3] = { gizmo_target_rot_.x * kRad2Deg,
                          gizmo_target_rot_.y * kRad2Deg,
                          gizmo_target_rot_.z * kRad2Deg };
        float s_in[3] = { gizmo_target_scl_.x, gizmo_target_scl_.y, gizmo_target_scl_.z };
        ImGuizmo::RecomposeMatrixFromComponents(t_in, r_in, s_in, matrix);

        // Operation + space.
        ImGuizmo::OPERATION op = ImGuizmo::TRANSLATE;
        switch (gizmo_mode_) {
            case GizmoMode::Translate: op = ImGuizmo::TRANSLATE; break;
            case GizmoMode::Rotate:    op = ImGuizmo::ROTATE;    break;
            case GizmoMode::Scale:     op = ImGuizmo::SCALE;     break;
        }
        const ImGuizmo::MODE mode = imguizmo_local_space_
                                        ? ImGuizmo::LOCAL
                                        : ImGuizmo::WORLD;

        // Snap. ImGuizmo expects a 3-float array even when the unit is
        // scalar (rotation, scale) — same value in every slot. Disable
        // by passing nullptr.
        float snap_buf[3] = { 0.0f, 0.0f, 0.0f };
        float* snap_ptr   = nullptr;
        if (gizmo_snap_on_ && gizmo_snap_step_ > 0.0f) {
            snap_buf[0] = snap_buf[1] = snap_buf[2] = gizmo_snap_step_;
            snap_ptr = snap_buf;
        }

        // Drive ImGuizmo. Returns true while the user is actively
        // manipulating; we don't depend on the return — we read the
        // mutated matrix back unconditionally and only commit the drag
        // when IsUsing() reports the widget owns the input.
        ImGuizmo::Manipulate(reinterpret_cast<const float*>(&gizmo_view_),
                             reinterpret_cast<const float*>(&gizmo_proj_),
                             op, mode, matrix, /*deltaMatrix*/ nullptr,
                             snap_ptr);

        const bool is_using = ImGuizmo::IsUsing();
        if (is_using) {
            // Decompose mutated matrix back into TRS for the host.
            float t_out[3], r_out[3], s_out[3];
            ImGuizmo::DecomposeMatrixToComponents(matrix, t_out, r_out, s_out);
            constexpr float kDeg2Rad = 0.01745329251994329f;

            cardinal::scene::Vec3 new_pos{t_out[0], t_out[1], t_out[2]};
            cardinal::scene::Vec3 new_rot{r_out[0]*kDeg2Rad,
                                          r_out[1]*kDeg2Rad,
                                          r_out[2]*kDeg2Rad};
            cardinal::scene::Vec3 new_scl{s_out[0], s_out[1], s_out[2]};

            // Populate the drag output the host loop reads.
            gizmo_drag_.active                  = true;
            gizmo_drag_.mode                    = gizmo_mode_;
            gizmo_drag_.target_world            = new_pos;
            gizmo_drag_.target_rotation_euler   = new_rot;
            gizmo_drag_.target_scale            = new_scl;
            gizmo_drag_.delta                   = { new_pos.x - gizmo_target_.x,
                                                    new_pos.y - gizmo_target_.y,
                                                    new_pos.z - gizmo_target_.z };

            // Update Studio's internal anchors so the next frame's gizmo
            // draws at the new transform until the host writes its own.
            gizmo_target_     = new_pos;
            gizmo_target_rot_ = new_rot;
            gizmo_target_scl_ = new_scl;
            gizmo_drag_viewport_id_ = viewport_id;

            // Maintain the drag-in-progress edge for undo coalescing.
            if (!gizmo_drag_in_progress_) gizmo_drag_in_progress_ = true;
        } else if (gizmo_drag_in_progress_) {
            // Edge: was-using → not-using. Signal release for the next
            // gizmo_drag_release_consume() poll. Only the owning panel reaches
            // here (the per-frame owner gate at the top returns siblings early).
            gizmo_drag_in_progress_ = false;
            gizmo_drag_release_     = true;
        }

        // Corner-cube camera orbiter — ViewManipulate. The widget mutates
        // the view matrix in place if the user drags it; downstream the
        // host can pick up the new view via gizmo_view_ next frame.
        // 96-pixel square pinned to the top-right of the panel.
        if (imguizmo_show_view_cube_) {
            const float kCubeSize = 96.0f;
            const ImVec2 cube_pos{
                image_pos.x + avail.x - kCubeSize - 8.0f,
                image_pos.y + 8.0f
            };
            ImGuizmo::ViewManipulate(reinterpret_cast<float*>(&gizmo_view_),
                                     /*distance*/ 8.0f,
                                     cube_pos,
                                     ImVec2{kCubeSize, kCubeSize},
                                     /*backgroundColor*/ 0x10101080u);
        }

        return is_using;
    }

    // ----- Public dispatcher (call site unchanged) ----------------------
    bool draw_gizmo_overlay(const ImVec2& image_pos, const ImVec2& avail,
                            u32 viewport_id) {
        if (!gizmo_has_target_ || !gizmo_have_camera_) return false;
        if (avail.x < 4.0f || avail.y < 4.0f)          return false;

        // Per-frame defaults — host reads these every frame.
        gizmo_drag_.active = false;
        gizmo_drag_.mode   = gizmo_mode_;
        gizmo_drag_.target_rotation_euler = gizmo_target_rot_;
        gizmo_drag_.target_scale          = gizmo_target_scl_;

        bool busy = false;
        if (gizmo_mode_ == GizmoMode::Rotate) {
            busy = draw_rotate_gizmo_(image_pos, avail, viewport_id);
        } else if (gizmo_mode_ == GizmoMode::Scale) {
            busy = draw_scale_gizmo_(image_pos, avail, viewport_id);
        } else {
            // Single-axis path (proven) — skip while a plane/screen drag
            // owns the handle so the two can't fight.
            if (gizmo_handle_active_ < 0)
                busy = draw_translate_axes_(image_pos, avail, viewport_id);
            // Gizmo-side snap for the axis result (idempotent if the host
            // also snaps — back-compat safe).
            if (gizmo_drag_.active &&
                gizmo_drag_.mode == GizmoMode::Translate &&
                gizmo_snap_on_ && gizmo_snap_step_ > 0.0f) {
                cardinal::scene::Vec3 t = gizmo_drag_.target_world;
                t.x = snapf_(t.x, gizmo_snap_step_);
                t.y = snapf_(t.y, gizmo_snap_step_);
                t.z = snapf_(t.z, gizmo_snap_step_);
                gizmo_drag_.target_world = t;
                gizmo_drag_.delta = { t.x-gizmo_drag_origin_world_.x,
                                      t.y-gizmo_drag_origin_world_.y,
                                      t.z-gizmo_drag_origin_world_.z };
                gizmo_target_ = t;
            }
            // Plane + screen handles only when no axis drag is live.
            if (gizmo_axis_active_ < 0)
                busy = draw_translate_planes_screen_(image_pos, avail, viewport_id) || busy;
        }
        return busy;
    }

    // -------------------------------------------------------------------
    // Scene hierarchy — flat list with selection.
    // -------------------------------------------------------------------
    void draw_scene_hierarchy(scene::Scene& scene, u32* selected_id_inout,
                              const char* title, bool* p_open) override {
        // Rich single-select panel lives in panels/hierarchy.cpp.
        panels::draw(scene, selected_id_inout, title, p_open, hierarchy_state_);
        // Mirror id ⇄ the shared selection set so the multi-aware
        // viewport / gizmo and legacy callers stay coherent.
        if (selected_id_inout != nullptr) {
            const u32 id = *selected_id_inout;
            const bool same = (selection_.size() == 1 && selection_.front() == id);
            if (id == 0) selection_.clear();
            else if (!same) { selection_.clear(); selection_.push_back(id); }
        }
    }

    // Multi-select flat-list overload. Ctrl = toggle, Shift = range from
    // the anchor (last selected), plain click = replace. Host passes its
    // own vector (or nullptr to drive Studio's shared set) and reads it
    // back after end_frame; the gizmo pivot/fan-out stay host-side.
    void draw_scene_hierarchy(scene::Scene& scene,
                              cardinal::vector<u32>* selection_inout,
                              const char* title, bool* p_open) override {
        cardinal::vector<u32>& sel =
            (selection_inout != nullptr) ? *selection_inout : selection_;
        if (!ImGui::Begin(title ? title : "Hierarchy", p_open,
                          0)) {
            ImGui::End();
            return;
        }
        const ImGuiIO& io = ImGui::GetIO();
        const u32 anchor = sel.empty() ? 0u : sel.back();
        auto in_sel = [&](u32 id) {
            for (u32 s : sel) if (s == id) return true;
            return false;
        };
        auto& ents = scene.entities();
        ImGui::TextDisabled("%zu entit%s | %zu selected",
                            ents.size(), ents.size() == 1 ? "y" : "ies",
                            sel.size());
        hier_filter_.Draw("Filter", -1.0f);
        ImGui::Separator();
        for (auto& e : ents) {
            if (hier_filter_.IsActive() && !hier_filter_.PassFilter(
                    e.name.empty() ? "(entity)" : e.name.c_str())) continue;
            ImGui::PushID(static_cast<int>(e.id));
            // Per-row visibility toggle (honoured by the renderer via
            // is_entity_visible) — like UE5's outliner eye.
            bool vis = e.visible;
            if (ImGui::Checkbox("##vis", &vis)) e.visible = vis;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Toggle visibility in viewport");
            ImGui::SameLine();
            // Inline rename — double-click a row (below) swaps its label for an
            // editor; Enter / Escape / click-away commits.
            if (rename_id_ == e.id) {
                ImGui::SetNextItemWidth(-1.0f);
                if (rename_focus_) { ImGui::SetKeyboardFocusHere(); rename_focus_ = false; }
                const bool committed = ImGui::InputText("##rename", rename_buf_,
                    sizeof(rename_buf_),
                    ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
                if (committed || ImGui::IsItemDeactivated()) {
                    e.name = rename_buf_;
                    rename_id_ = 0;
                }
                ImGui::PopID();
                continue;
            }
            char label[176];
            cardinal::snprintf(label, sizeof(label), "%s##he_%u",
                          e.name.empty() ? "(entity)" : e.name.c_str(),
                          e.id);
            if (ImGui::Selectable(label, in_sel(e.id))) {
                if (io.KeyCtrl) {
                    cardinal::vector<u32> next;
                    next.reserve(sel.size() + 1);
                    bool found = false;
                    for (u32 s : sel) {
                        if (s == e.id) { found = true; continue; }
                        next.push_back(s);
                    }
                    if (!found) next.push_back(e.id);
                    sel.swap(next);
                } else if (io.KeyShift && anchor != 0u) {
                    int ia = -1, ib = -1, idx = 0;
                    for (auto& q : ents) {
                        if (q.id == anchor) ia = idx;
                        if (q.id == e.id)   ib = idx;
                        ++idx;
                    }
                    if (ia >= 0 && ib >= 0) {
                        if (ia > ib) { const int t = ia; ia = ib; ib = t; }
                        sel.clear();
                        idx = 0;
                        for (auto& q : ents) {
                            if (idx >= ia && idx <= ib) sel.push_back(q.id);
                            ++idx;
                        }
                    } else {
                        sel.clear();
                        sel.push_back(e.id);
                    }
                } else {
                    sel.clear();
                    sel.push_back(e.id);
                }
            }
            if (ImGui::IsItemHovered() &&
                ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                rename_id_ = e.id;
                cardinal::snprintf(rename_buf_, sizeof(rename_buf_), "%s", e.name.c_str());
                rename_focus_ = true;
            }
            ImGui::PopID();
        }
        ImGui::End();
        if (selection_inout != nullptr && &sel != &selection_)
            selection_ = sel;     // keep Studio's mirror coherent
    }


    // -------------------------------------------------------------------
    // Inspector — properties of the selected entity. Uses scene API for
    // mutation (so undo/redo can hook into it later).
    // -------------------------------------------------------------------
    void draw_inspector(scene::Scene& scene, u32 selected_id,
                        const char* title, bool* p_open) override {
        panels::inspector_panel::draw(scene, selected_id, title, p_open);
    }

    // -------------------------------------------------------------------
    // Asset browser — directory tree on the left, file grid on the right.
    // Uses cardinal::fs; recognises common extensions for icon labels.
    // -------------------------------------------------------------------
    void draw_asset_browser(const char* root_path, const char* title, bool* p_open) override {
        if (!ImGui::Begin(title ? title : "Assets", p_open,
                          0)) { ImGui::End(); return; }

        if (root_path == nullptr || *root_path == '\0') {
            ImGui::TextDisabled("(no asset root provided)");
            ImGui::End();
            return;
        }

        // Initialise lazily — we cache the root path and a rolling scan.
        if (asset_root_ != root_path) {
            asset_root_ = root_path;
            asset_cwd_  = root_path;
            asset_filter_.Clear();
            scan_asset_dir();
        }

        // Toolbar: refresh + filter.
        if (ImGui::Button("Refresh")) scan_asset_dir();
        ImGui::SameLine();
        ImGui::SetNextItemWidth(200.0f);
        if (asset_filter_.Draw("Filter", 200.0f)) {} // user typed
        ImGui::SameLine();
        ImGui::TextDisabled("%s", asset_cwd_.c_str());
        ImGui::Separator();

        // Two columns: tree | grid.
        ImGui::Columns(2, "##assets_split", true);
        ImGui::SetColumnWidth(0, 220.0f);
        if (ImGui::BeginChild("##tree", ImVec2(0, 0))) {
            draw_asset_tree(asset_root_);
        }
        ImGui::EndChild();
        ImGui::NextColumn();

        if (ImGui::BeginChild("##grid", ImVec2(0, 0))) {
            for (const auto& f : asset_files_) {
                if (asset_filter_.IsActive() && !asset_filter_.PassFilter(f.name.c_str())) continue;
                const ImU32 icon_col = asset_color_for(f.kind);
                ImGui::PushID(f.path.c_str());
                ImGui::PushStyleColor(ImGuiCol_Text, icon_col);
                ImGui::TextUnformatted(asset_icon_for(f.kind));
                ImGui::PopStyleColor();
                ImGui::SameLine();
                ImGui::Selectable(f.name.c_str(), false, ImGuiSelectableFlags_None);
                // Right-click a file -> content-browser actions.
                if (ImGui::BeginPopupContextItem()) {
                    ImGui::TextDisabled("%s", f.name.c_str());
                    ImGui::Separator();
                    if (ImGui::MenuItem("Copy path")) ImGui::SetClipboardText(f.path.c_str());
                    if (ImGui::MenuItem("Copy name")) ImGui::SetClipboardText(f.name.c_str());
                    ImGui::EndPopup();
                }
                ImGui::PopID();
            }
            if (asset_files_.empty()) {
                ImGui::TextDisabled("(empty directory)");
            }
        }
        ImGui::EndChild();

        ImGui::Columns(1);
        ImGui::End();
    }

    // -------------------------------------------------------------------
    // Asset palette — the editor's drop-anything-into-the-scene panel.
    // Lists every entry in scene::AssetCatalog grouped by category. The
    // currently-selected asset (passed in by the host) is highlighted so
    // it's obvious what an LMB click in the viewport will spawn.
    // -------------------------------------------------------------------
    void draw_asset_palette(const AssetPickFn& on_pick,
                            const char* current_asset_id,
                            const char* title, bool* p_open) override
    {
        // Forward to the spawn-at-aware overload with a no-op spawn callback.
        draw_asset_palette(on_pick, AssetSpawnAt{}, current_asset_id, title, p_open);
    }

    void draw_asset_palette(const AssetPickFn& on_pick,
                            const AssetSpawnAt& on_spawn_at,
                            const char* current_asset_id,
                            const char* title, bool* p_open) override
    {
        if (!ImGui::Begin(title ? title : "Asset Palette", p_open,
                          0)) {
            ImGui::End(); return;
        }

        const auto& cat = cardinal::scene::AssetCatalog::instance();
        if (cat.all().empty()) {
            ImGui::TextDisabled("(no assets registered yet — call register_default_assets)");
            ImGui::End();
            return;
        }

        ImGui::Text("Active: %s", current_asset_id ? current_asset_id : "(none — Select tool)");
        ImGui::Separator();
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::InputTextWithHint("##palette_filter", "filter…",
                                 palette_filter_, sizeof(palette_filter_));
        ImGui::Separator();

        const cardinal::string filter = palette_filter_;
        for (const auto& cat_name : cat.categories()) {
            // Count visible entries in this category given the filter.
            int visible = 0;
            for (const auto& a : cat.all()) {
                if (a.category != cat_name) continue;
                if (!filter.empty() &&
                    a.label.find(filter)   == cardinal::string::npos &&
                    a.id.find   (filter)   == cardinal::string::npos) continue;
                ++visible;
            }
            if (visible == 0) continue;

            if (!ImGui::CollapsingHeader(cat_name.c_str(),
                                         ImGuiTreeNodeFlags_DefaultOpen)) continue;

            // Draw a button per entry. Highlight the currently-selected one
            // with a coloured frame.
            for (const auto& a : cat.all()) {
                if (a.category != cat_name) continue;
                if (!filter.empty() &&
                    a.label.find(filter) == cardinal::string::npos &&
                    a.id.find   (filter) == cardinal::string::npos) continue;

                const bool active = (current_asset_id != nullptr &&
                                     a.id == current_asset_id);
                ImGui::PushID(a.id.c_str());
                if (active) {
                    ImGui::PushStyleColor(ImGuiCol_Button,        IM_COL32( 70, 110, 200, 255));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, IM_COL32( 90, 140, 230, 255));
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  IM_COL32(110, 170, 255, 255));
                }
                if (ImGui::Button(a.label.c_str(), ImVec2(-1.0f, 0.0f))) {
                    if (on_pick) on_pick(a.id.c_str());
                }
                if (active) ImGui::PopStyleColor(3);
                if (ImGui::IsItemHovered() && !ImGui::IsItemActive()) {
                    ImGui::BeginTooltip();
                    ImGui::Text("%s", a.label.c_str());
                    ImGui::TextDisabled("id: %s", a.id.c_str());
                    if (!a.tooltip.empty()) {
                        ImGui::Separator();
                        ImGui::TextUnformatted(a.tooltip.c_str());
                    }
                    ImGui::EndTooltip();
                }
                // Right-click context menu for direct spawning bypassing
                // the place-tool. Available when the host wired the
                // on_spawn_at callback (3-arg overload caller).
                if (on_spawn_at) {
                    if (ImGui::BeginPopupContextItem("##palette_ctx")) {
                        if (ImGui::MenuItem("Pick (set as place tool)"))
                            if (on_pick) on_pick(a.id.c_str());
                        ImGui::Separator();
                        if (ImGui::MenuItem("Spawn at camera"))
                            on_spawn_at(a.id.c_str(), 0);
                        if (ImGui::MenuItem("Spawn at world origin"))
                            on_spawn_at(a.id.c_str(), 1);
                        ImGui::EndPopup();
                    }
                }
                ImGui::PopID();
            }
            ImGui::Spacing();
        }

        ImGui::Separator();
        ImGui::TextDisabled("Click → set place tool ; Right-click → spawn directly");
        ImGui::End();
    }

    // -------------------------------------------------------------------
    // Terrain world-engine modal.
    //
    // Opens on-demand from the host's Create→Terrain Grid… menu. Owns its
    // own state across frames, so re-opening keeps the user's previous
    // selection + layer edits.
    //
    // Sections (top to bottom):
    //   1. Profile picker — every entry in TerrainProfileLibrary
    //   2. Library file ops — Save / Load to disk + filename input
    //   3. Grid extent — chunks_x/z + chunk_size + resolution
    //   4. Profile editor — name/description/global params + tint ramp
    //                       + per-layer editor (add/remove/duplicate/reorder)
    //   5. Generate / Cancel
    //
    // Profile edits write back to the library in real time so the next
    // dropdown switch shows the updated state. Hitting Generate fills
    // request_out with the picked profile name + grid params.
    // -------------------------------------------------------------------

    // -------------------------------------------------------------------
    // Game-engine toolbar — horizontal strip across the top of the editor.
    // Layout (left to right):
    //   [ Select ] [ Place ]  |  [ Snap □ ] [ step: 1.00 ]  |  [ ↶ ] [ ↷ ]
    //   |  [ ▶ Play ] [ ⏸ Pause ] [ ⏭ Step ]
    //   |  [ Focus ] [ Reset cam ]
    // Most controls write back through `state` / `undo`; the host reads
    // them after this call returns.
    // -------------------------------------------------------------------
    void draw_main_toolbar(ToolbarState* state, UndoButtons* undo,
                           const char* title, bool* p_open) override
    {
        if (state == nullptr) return;
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 6));
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoScrollbar
                               | ImGuiWindowFlags_NoCollapse;
        if (!ImGui::Begin(title ? title : "Toolbar", p_open, flags)) {
            ImGui::End(); ImGui::PopStyleVar(); return;
        }

        // ----- Tool radio -----
        const char* tool_names[] = { "Select", "Place" };
        for (int i = 0; i < IM_ARRAYSIZE(tool_names); ++i) {
            if (i > 0) ImGui::SameLine();
            const bool sel = (state->tool == i);
            if (sel) ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(70,110,200,255));
            if (ImGui::Button(tool_names[i])) state->tool = i;
            if (sel) ImGui::PopStyleColor();
        }

        ImGui::SameLine(); ImGui::TextDisabled("|");

        // ----- Snap -----
        ImGui::SameLine();
        ImGui::Checkbox("Snap", &state->snap_enabled);
        if (state->snap_enabled) {
            ImGui::SameLine();
            ImGui::SetNextItemWidth(80.0f);
            ImGui::DragFloat("##snap_step", &state->snap_step,
                             0.05f, 0.01f, 32.0f, "%.2fu");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Snap-to-grid step in world units");
        }

        ImGui::SameLine(); ImGui::TextDisabled("|");

        // ----- Undo / Redo -----
        if (undo != nullptr) {
            ImGui::SameLine();
            if (!undo->can_undo) ImGui::BeginDisabled();
            if (ImGui::Button("Undo")) undo->undo_clicked = true;
            if (undo->can_undo && undo->undo_label && undo->undo_label[0])
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Undo: %s  (Ctrl+Z)", undo->undo_label);
            if (!undo->can_undo) ImGui::EndDisabled();

            ImGui::SameLine();
            if (!undo->can_redo) ImGui::BeginDisabled();
            if (ImGui::Button("Redo")) undo->redo_clicked = true;
            if (undo->can_redo && undo->redo_label && undo->redo_label[0])
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Redo: %s  (Ctrl+Y)", undo->redo_label);
            if (!undo->can_redo) ImGui::EndDisabled();

            ImGui::SameLine(); ImGui::TextDisabled("|");
        }

        // ----- Play / Pause / Step -----
        ImGui::SameLine();
        if (state->play) {
            ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(60,170,90,255));
            if (ImGui::Button("Stop")) { state->play = false; state->paused = false; }
            ImGui::PopStyleColor();
        } else {
            if (ImGui::Button("Play")) { state->play = true;  state->paused = false; }
        }
        ImGui::SameLine();
        if (!state->play) ImGui::BeginDisabled();
        if (ImGui::Button(state->paused ? "Resume" : "Pause"))
            state->paused = !state->paused;
        ImGui::SameLine();
        if (ImGui::Button("Step")) { state->step_once = true; state->paused = true; }
        if (!state->play) ImGui::EndDisabled();

        ImGui::SameLine(); ImGui::TextDisabled("|");

        // ----- Camera commands -----
        ImGui::SameLine();
        if (ImGui::Button("Focus selection")) state->focus_selection = true;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Pivot the fly-camera around the selected entity (F)");
        ImGui::SameLine();
        if (ImGui::Button("Reset cam")) state->reset_camera = true;

        ImGui::End();
        ImGui::PopStyleVar();
    }

    // -------------------------------------------------------------------
    // Right-click context menu invoked from inside the host's viewport
    // panel. Call right after `Studio::draw_viewport_panel` while the
    // viewport's content rectangle is still the implicit ImGui parent.
    // -------------------------------------------------------------------
    void draw_viewport_context_menu(ViewportCtxAction* out) override {
        if (out == nullptr) return;
        *out = {};

        // Modifier-gated open: bare RMB in the viewport is reserved for
        // FlyCam look-mode (when the game is Playing). Holding ALT while
        // right-clicking opens this menu — matches Maya / Blender's "alt+
        // mouse for editor commands" convention and keeps RMB-drag clean
        // for camera control.
        //
        // We also require the cursor be over the current window AND not
        // over a child item so a right-click on a docked panel inside
        // the viewport region doesn't accidentally trigger this.
        const ImGuiIO& io = ImGui::GetIO();
        const bool modifier_held = io.KeyAlt;
        if (modifier_held
            && ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup)
            && !ImGui::IsAnyItemHovered()
            && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
        {
            ImGui::OpenPopup("##viewport_ctx");
            // Freeze the chunk under the right-click cursor — the live
            // hover_chunk_* is reset every frame and won't be valid by
            // the time the user clicks a menu item (cursor is over the
            // menu by then, not the panel). Without this, "Spawn active
            // asset here" always fell back to camera-forward.
            ctx_hover_chunk_x_     = hover_chunk_x_;
            ctx_hover_chunk_y_     = hover_chunk_y_;
            ctx_hover_chunk_z_     = hover_chunk_z_;
            ctx_hover_chunk_valid_ = hover_chunk_valid_;
        }

        if (ImGui::BeginPopup("##viewport_ctx")) {
            ImGui::TextDisabled("(Alt+RMB to open)");
            ImGui::Separator();
            if (ImGui::MenuItem("Spawn active asset here", "Click in Place mode")) {
                out->spawn_active_asset_at_cursor = true;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Focus selection",     "F"))     out->focus_selected = true;
            if (ImGui::MenuItem("Duplicate selection", "Ctrl+D")) out->duplicate_selected = true;
            if (ImGui::MenuItem("Delete selection",    "Del"))   out->delete_selected = true;
            ImGui::Separator();
            if (ImGui::MenuItem("Clear selection", "Esc"))       out->clear_selection = true;
            ImGui::EndPopup();
        } else if (ctx_hover_chunk_valid_) {
            // Popup closed (item clicked, click-outside, or Esc) — drop
            // the freeze so the next right-click captures fresh. The host
            // reads last_hover_chunk() AFTER end_frame on the same frame
            // the item is clicked; ImGui closes the popup on item-click,
            // so BeginPopup returned false here, but the spawn handler
            // still gets the captured chunk because clear happens on the
            // NEXT frame's call (the action is consumed before then).
            ctx_hover_chunk_valid_ = false;
        }
    }

    // -------------------------------------------------------------------
    // Right-click on a hierarchy entry. Call from inside the host's
    // hierarchy panel after rendering each row's selectable.
    // -------------------------------------------------------------------
    void draw_hierarchy_context_menu(HierarchyCtxAction* out) override {
        if (out == nullptr) return;
        *out = {};
        if (ImGui::BeginPopupContextItem("##hierarchy_ctx",
                ImGuiPopupFlags_MouseButtonRight))
        {
            if (ImGui::MenuItem("Rename",     "F2"))     out->rename = true;
            if (ImGui::MenuItem("Duplicate",  "Ctrl+D")) out->duplicate = true;
            if (ImGui::MenuItem("Delete",     "Del"))    out->delete_entity = true;
            ImGui::Separator();
            if (ImGui::MenuItem("Focus camera", "F"))    out->focus_camera = true;
            ImGui::EndPopup();
        }
    }

    // -------------------------------------------------------------------
    // World streaming panel — 3D streaming controls. Host owns the
    // WorldGrid; we just read/write the desc fields through `state`.
    // -------------------------------------------------------------------
    void draw_world_panel(WorldPanelState* state, const char* title,
                          bool* p_open) override
    {
        if (state == nullptr) return;
        state->evict_all_clicked = false;
        if (!ImGui::Begin(title ? title : "World", p_open,
                          0)) {
            ImGui::End(); return;
        }

        ImGui::SeparatorText("Grid");
        ImGui::SliderFloat("Chunk size (units)", &state->chunk_size,
                           1.0f, 256.0f, "%.1f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("World units per chunk side (uniform XYZ).");

        // Per-axis extents — drag-int because users routinely want very
        // large numbers (millions of chunks) which a slider can't reach.
        // Range goes up near INT32_MAX/2 so the world is effectively
        // unbounded; default safe values are 1024 horizontal / 64 vertical.
        ImGui::DragInt("Extent X (chunks)", &state->extent_x, 1.0f, 1, INT32_MAX / 4);
        ImGui::DragInt("Extent Y (chunks)", &state->extent_y, 1.0f, 1, INT32_MAX / 4);
        ImGui::DragInt("Extent Z (chunks)", &state->extent_z, 1.0f, 1, INT32_MAX / 4);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Half-extents — world spans ±N chunks per axis.\n"
                              "0,0,0 is the centre, NOT the limit.\n"
                              "Crank to ~5×10⁸ for effectively unbounded worlds.");

        const f64 sx_units = static_cast<f64>(state->chunk_size) * 2.0 * state->extent_x;
        const f64 sy_units = static_cast<f64>(state->chunk_size) * 2.0 * state->extent_y;
        const f64 sz_units = static_cast<f64>(state->chunk_size) * 2.0 * state->extent_z;
        const i64 total_chunks = static_cast<i64>(state->extent_x) * 2
                               * static_cast<i64>(state->extent_y) * 2
                               * static_cast<i64>(state->extent_z) * 2;
        ImGui::TextDisabled("World: %.0f × %.0f × %.0f units", sx_units, sy_units, sz_units);
        ImGui::TextDisabled("Total chunks: %lld",
                            static_cast<long long>(total_chunks));

        ImGui::SeparatorText("Streaming");
        ImGui::SliderInt("Render dist XZ", &state->render_distance_xz, 0, 64);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Horizontal chunk radius around the camera (XZ disc).");
        ImGui::SliderInt("Render dist Y",  &state->render_distance_y,  0, 64);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Vertical chunk half-height (Y slab). Set equal to XZ for a sphere.");

        ImGui::Checkbox("Cull entities outside render distance",
                        &state->cull_outside_render_distance);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("When ON, the renderer hides entities whose chunk isn't in the active set.\n"
                              "When OFF, all entities draw — useful for editing far-away chunks.");
        ImGui::Checkbox("Show chunk grid in viewport", &state->show_grid_overlay);

        ImGui::SeparatorText("Live");
        ImGui::Text("Camera chunk: (%d, %d, %d)",
                    state->camera_chunk_x, state->camera_chunk_y, state->camera_chunk_z);
        ImGui::Text("Active chunks: %d / %lld total",
                    state->active_chunk_count,
                    static_cast<long long>(total_chunks));

        // Cylinder-shape estimate: π·rxz² × (2·ry + 1).
        const i32 rxz = state->render_distance_xz;
        const i32 ry  = state->render_distance_y;
        const f32 est = 3.14159f * static_cast<f32>(rxz * rxz)
                                 * static_cast<f32>(2 * ry + 1);
        ImGui::TextDisabled("Estimated active per-frame cap: ~%.0f", est);

        ImGui::Separator();
        if (ImGui::Button("Force evict all")) state->evict_all_clicked = true;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Fires on_unload for every active chunk + clears state.\n"
                              "Useful for testing the load callback path.");
        ImGui::End();
    }

    void set_world_grid_overlay(const WorldGridOverlay& o) override {
        world_overlay_ = o;
    }

    void draw_terrain_modal(bool open_now, TerrainSpawnRequest* request_out) override {
        if (open_now) {
            terrain_modal_open_ = true;
            ImGui::OpenPopup("Terrain World Engine##terrain_modal");
        }
        if (request_out) request_out->requested = false;

        ImVec2 centre = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(centre, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(560.0f, 720.0f), ImGuiCond_Appearing);

        if (!ImGui::BeginPopupModal("Terrain World Engine##terrain_modal",
                                    &terrain_modal_open_,
                                    ImGuiWindowFlags_None))
        {
            return;
        }

        TerrainSpawnRequest& S = terrain_modal_state_;
        auto& lib = cardinal::scene::TerrainProfileLibrary::instance();

        // -------- 1. Profile picker --------
        ImGui::SeparatorText("Profile");
        const auto profile_names = lib.names();
        if (S.profile_name.empty() && !profile_names.empty())
            S.profile_name = profile_names.front();

        const char* preview = S.profile_name.empty() ? "(none)" : S.profile_name.c_str();
        ImGui::SetNextItemWidth(-130.0f);
        if (ImGui::BeginCombo("##profile_pick", preview)) {
            for (const auto& n : profile_names) {
                const bool sel = (n == S.profile_name);
                if (ImGui::Selectable(n.c_str(), sel)) S.profile_name = n;
                if (sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        if (ImGui::Button("Duplicate##profile")) {
            if (auto p = lib.find(S.profile_name.c_str())) {
                cardinal::scene::TerrainProfile np = *p;
                np.name += " (copy)";
                lib.set(np);
                S.profile_name = np.name;
            }
        }

        // -------- 2. Library file ops --------
        ImGui::SeparatorText("Library");
        ImGui::SetNextItemWidth(-200.0f);
        ImGui::InputTextWithHint("##terr_lib_path", "terrain_profiles.txt",
                                 terrain_lib_path_, sizeof(terrain_lib_path_));
        ImGui::SameLine();
        if (ImGui::Button("Save lib")) {
            if (terrain_lib_path_[0] != '\0') {
                if (lib.save_to_file(terrain_lib_path_)) {
                    cardinal::log::infof("ui/studio",
                        "Terrain library saved to '%s' (%zu profiles)",
                        terrain_lib_path_, lib.all().size());
                }
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Load lib")) {
            if (terrain_lib_path_[0] != '\0') lib.load_from_file(terrain_lib_path_);
        }
        ImGui::TextDisabled("Library has %zu profile(s)", lib.all().size());

        // -------- 3. Grid extent --------
        ImGui::SeparatorText("Grid");
        ImGui::SliderInt("Chunks X", &S.chunks_x, 1, 16);
        ImGui::SliderInt("Chunks Z", &S.chunks_z, 1, 16);
        int res = static_cast<int>(S.resolution);
        if (ImGui::SliderInt("Resolution (verts/side)", &res, 8, 256))
            S.resolution = static_cast<u32>(res);
        ImGui::SliderFloat("Chunk size (world units)", &S.chunk_size, 4.0f, 256.0f, "%.1f");
        const u64 quads = static_cast<u64>(S.resolution - 1)
                        * static_cast<u64>(S.resolution - 1);
        const u64 tris  = quads * 2ull
                        * static_cast<u64>(S.chunks_x) * static_cast<u64>(S.chunks_z);
        const float world_w = S.chunk_size * static_cast<float>(S.chunks_x);
        const float world_d = S.chunk_size * static_cast<float>(S.chunks_z);
        ImGui::TextDisabled("World footprint: %.1f × %.1f units, ≈ %llu triangles",
                            world_w, world_d, static_cast<unsigned long long>(tris));

        // -------- 4. Profile editor (writes through to library) --------
        ImGui::SeparatorText("Profile editor");
        cardinal::scene::TerrainProfile edited;
        bool have_profile = false;
        if (auto p = lib.find(S.profile_name.c_str())) {
            edited = *p;
            have_profile = true;
        }

        if (have_profile) {
            bool dirty = false;

            // Name + description live in input fields. Renaming a profile
            // re-keys it in the library (we save under the new name and
            // remove the old).
            char name_buf[96]; cardinal::strncpy(name_buf, edited.name.c_str(), sizeof(name_buf) - 1);
            name_buf[sizeof(name_buf) - 1] = '\0';
            if (ImGui::InputText("Name", name_buf, sizeof(name_buf))) {
                cardinal::string old_name = edited.name;
                edited.name = name_buf;
                if (!edited.name.empty() && edited.name != old_name) {
                    lib.remove(old_name.c_str());
                    lib.set(edited);
                    S.profile_name = edited.name;
                    dirty = false;          // already committed
                }
            }
            char desc_buf[256]; cardinal::strncpy(desc_buf, edited.description.c_str(), sizeof(desc_buf) - 1);
            desc_buf[sizeof(desc_buf) - 1] = '\0';
            if (ImGui::InputText("Description", desc_buf, sizeof(desc_buf))) {
                edited.description = desc_buf;
                dirty = true;
            }

            if (ImGui::CollapsingHeader("Global", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::SliderFloat("Base height",  &edited.base_height,  -32.0f, 32.0f)) dirty = true;
                if (ImGui::SliderFloat("Height scale", &edited.height_scale,   0.1f, 128.0f)) dirty = true;
            }
            if (ImGui::CollapsingHeader("Tint ramp", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::ColorEdit3("Low (valley)", &edited.tint_low.x))   dirty = true;
                if (ImGui::ColorEdit3("High (peak)",  &edited.tint_high.x))  dirty = true;
                if (ImGui::ColorEdit3("Cliff",        &edited.tint_cliff.x)) dirty = true;
                if (ImGui::DragFloat("Tint height low",  &edited.tint_height_low,  0.1f)) dirty = true;
                if (ImGui::DragFloat("Tint height high", &edited.tint_height_high, 0.1f)) dirty = true;
                if (ImGui::SliderFloat("Cliff blend (1−n.y)",
                                       &edited.cliff_blend_threshold, 0.0f, 0.99f)) dirty = true;
            }

            // -------- Layers --------
            ImGui::Separator();
            ImGui::Text("Layers (%zu)", edited.layers.size());
            ImGui::SameLine();
            if (ImGui::Button("+ Add layer")) {
                cardinal::scene::NoiseLayer L{};
                L.id = "layer_" + cardinal::to_string(edited.layers.size() + 1);
                edited.layers.push_back(L);
                dirty = true;
            }

            const char* op_names[]    = { "Add", "Multiply", "Max", "Min" };
            const char* shape_names[] = { "Smooth", "Ridge", "Billow" };

            for (size_t i = 0; i < edited.layers.size(); ++i) {
                cardinal::scene::NoiseLayer& L = edited.layers[i];
                ImGui::PushID(static_cast<int>(i));
                char header[96];
                cardinal::snprintf(header, sizeof(header), "[%zu] %s — %s · %s · w=%.2f",
                              i, L.id.c_str(),
                              op_names[static_cast<int>(L.op) & 3],
                              shape_names[static_cast<int>(L.shape) % 3],
                              L.weight);
                if (ImGui::CollapsingHeader(header)) {
                    if (ImGui::Checkbox("Enabled", &L.enabled)) dirty = true;
                    ImGui::SameLine();
                    if (ImGui::Button("Up") && i > 0) {
                        cardinal::swap(edited.layers[i], edited.layers[i - 1]); dirty = true;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Down") && i + 1 < edited.layers.size()) {
                        cardinal::swap(edited.layers[i], edited.layers[i + 1]); dirty = true;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Remove")) {
                        edited.layers.erase(edited.layers.begin() + i);
                        dirty = true;
                        ImGui::PopID();
                        break;
                    }

                    char id_buf[96];
                    cardinal::strncpy(id_buf, L.id.c_str(), sizeof(id_buf) - 1);
                    id_buf[sizeof(id_buf) - 1] = '\0';
                    if (ImGui::InputText("Id", id_buf, sizeof(id_buf))) {
                        L.id = id_buf; dirty = true;
                    }

                    int op  = static_cast<int>(L.op);
                    int sh  = static_cast<int>(L.shape);
                    if (ImGui::Combo("Op",    &op, op_names,    IM_ARRAYSIZE(op_names))) {
                        L.op    = static_cast<cardinal::scene::NoiseOp>(op);    dirty = true;
                    }
                    if (ImGui::Combo("Shape", &sh, shape_names, IM_ARRAYSIZE(shape_names))) {
                        L.shape = static_cast<cardinal::scene::NoiseShape>(sh); dirty = true;
                    }
                    if (ImGui::SliderFloat("Weight",     &L.weight,     0.0f, 2.0f))   dirty = true;
                    if (ImGui::SliderFloat("Frequency",  &L.frequency,  0.001f, 0.5f, "%.4f",
                                           ImGuiSliderFlags_Logarithmic))               dirty = true;
                    if (ImGui::SliderInt  ("Octaves",    &L.octaves,    1, 8))          dirty = true;
                    if (ImGui::SliderFloat("Lacunarity", &L.lacunarity, 1.5f, 4.0f))    dirty = true;
                    if (ImGui::SliderFloat("Gain",       &L.gain,       0.1f, 0.9f))    dirty = true;
                    int seed = static_cast<int>(L.seed);
                    if (ImGui::InputInt("Seed", &seed))  { L.seed = static_cast<u32>(seed); dirty = true; }
                    if (ImGui::DragFloat("Offset X",  &L.offset_x, 0.5f))    dirty = true;
                    if (ImGui::DragFloat("Offset Z",  &L.offset_z, 0.5f))    dirty = true;
                    if (ImGui::SliderFloat("Warp amount",    &L.warp_amount,    0.0f, 80.0f, "%.2f")) dirty = true;
                    if (ImGui::SliderFloat("Warp frequency", &L.warp_frequency, 0.001f, 0.5f, "%.4f",
                                           ImGuiSliderFlags_Logarithmic))    dirty = true;
                }
                ImGui::PopID();
            }

            if (dirty) lib.set(edited);
        } else {
            ImGui::TextDisabled("(no profile selected — register some via "
                                "register_default_terrain_profiles)");
        }

        // -------- 5. Generate / Cancel --------
        ImGui::Separator();
        const bool can_generate = have_profile;
        if (!can_generate) ImGui::BeginDisabled();
        if (ImGui::Button("Generate", ImVec2(140, 0))) {
            if (request_out) {
                *request_out = S;
                request_out->requested = true;
            }
            ImGui::CloseCurrentPopup();
            terrain_modal_open_ = false;
        }
        if (!can_generate) ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Close", ImVec2(140, 0))) {
            ImGui::CloseCurrentPopup();
            terrain_modal_open_ = false;
        }
        ImGui::EndPopup();
    }

    // -------------------------------------------------------------------
    // Tool-window registry — apps add custom panels (without modifying
    // Studio) by handing us a draw callback + show pointer + dock hint.
    // We hold a vector of registrations; draw_tool_windows iterates it
    // each frame, draw_tool_windows_menu lists them in the View menu.
    // The host's dockspace builder reads `registered_tool_windows()` to
    // place each tool the first time the layout is built.
    // -------------------------------------------------------------------
    void register_tool_window(ToolWindow tw) override {
        for (auto& w : tool_windows_) {
            if (w.title == tw.title) { w = cardinal::move(tw); return; }
        }
        tool_windows_.push_back(cardinal::move(tw));
    }
    void unregister_tool_window(const char* title) override {
        if (title == nullptr) return;
        for (auto it = tool_windows_.begin(); it != tool_windows_.end(); ++it) {
            if (it->title == title) { tool_windows_.erase(it); return; }
        }
    }
    void draw_tool_windows() override {
        for (auto& w : tool_windows_) {
            if (w.show != nullptr && !*w.show) continue;
            if (w.draw) w.draw(w.show);
        }
    }
    void draw_tool_windows_menu() override {
        for (auto& w : tool_windows_) {
            if (w.show != nullptr) ImGui::MenuItem(w.title.c_str(), nullptr, w.show);
            else                   ImGui::TextDisabled("%s", w.title.c_str());
        }
    }
    cardinal::vector<ToolWindow> registered_tool_windows() const override {
        return tool_windows_;
    }

    // -------------------------------------------------------------------
    // Console — command line at the bottom that calls the user-supplied
    // evaluator. Used to host the script REPL.
    // -------------------------------------------------------------------
    void draw_console_panel(const ConsoleEval& eval, const char* title, bool* p_open) override {
        // Implementation lives in panels/console.cpp.
        panels::draw(eval, title, p_open, console_state_);
    }

    // -------------------------------------------------------------------
    // Stack Tracer panel — captures + displays the calling thread's
    // stack on demand, plus the most recent crash stack handed to us
    // through cardinal::log (or hold a snapshot of the user's choice).
    // -------------------------------------------------------------------
    void draw_stack_tracer_panel(const char* title, bool* p_open) override {
        if (!ImGui::Begin(title ? title : "Stack Tracer", p_open,
                          0)) { ImGui::End(); return; }

        if (ImGui::Button("Capture now")) {
            captured_stack_ = cardinal::trace::capture(/*skip*/ 1, /*max_depth*/ 64);
            captured_stack_label_ = "main thread @ " + current_timestamp();
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            captured_stack_.clear();
            captured_stack_label_.clear();
        }

        ImGui::TextDisabled("%s", captured_stack_label_.empty()
            ? "(no capture yet — click Capture now)"
            : captured_stack_label_.c_str());
        ImGui::Separator();

        if (ImGui::BeginChild("##stack_scroll", ImVec2(0, 0))) {
            for (size_t i = 0; i < captured_stack_.size(); ++i) {
                const auto& f = captured_stack_[i];
                ImGui::Text("%2zu  ", i);
                ImGui::SameLine(0.0f, 0.0f);
                ImGui::TextColored(ImVec4(0.6f, 0.85f, 1.0f, 1.0f), "%s",
                    f.module.empty() ? "?" : f.module.c_str());
                ImGui::SameLine(0.0f, 0.0f);
                ImGui::TextUnformatted("!");
                ImGui::SameLine(0.0f, 0.0f);
                ImGui::TextColored(ImVec4(0.95f, 0.95f, 0.65f, 1.0f), "%s",
                    f.symbol.empty() ? "?" : f.symbol.c_str());
                if (!f.file.empty()) {
                    ImGui::SameLine();
                    ImGui::TextDisabled("(%s:%u)", f.file.c_str(), f.line);
                }
            }
            if (captured_stack_.empty()) {
                ImGui::TextDisabled("(empty)");
            }
        }
        ImGui::EndChild();
        ImGui::End();
    }

    // -------------------------------------------------------------------
    // Function Tracer panel — flame-style timeline of CALL/RET events
    // pulled from the global trace ring. Zoom + pan via toolbar; rows
    // are per-thread.
    // -------------------------------------------------------------------
    void draw_function_tracer_panel(const char* title, bool* p_open) override {
        if (!ImGui::Begin(title ? title : "Function Tracer", p_open,
                          0)) { ImGui::End(); return; }

        if (ImGui::Button("Snapshot")) {
            cardinal::trace::Timeline::instance().snapshot(trace_view_);
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear ring")) {
            cardinal::trace::Timeline::instance().clear();
            trace_view_.clear();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("%zu events buffered (cap %u)",
            trace_view_.size(), cardinal::trace::Timeline::instance().capacity());

        ImGui::Separator();

        // Pair CALL with matching RET to compute durations. Build a list of
        // {start_ns, end_ns, name, depth, thread, category}.
        struct Span { u64 t0, t1; u32 depth, tid; const char* name; const char* cat; };
        cardinal::vector<Span> spans;
        spans.reserve(trace_view_.size() / 2);

        cardinal::vector<size_t> call_stack;
        for (size_t i = 0; i < trace_view_.size(); ++i) {
            const auto& e = trace_view_[i];
            if (e.kind == cardinal::trace::EventKind::Call) {
                call_stack.push_back(i);
            } else if (e.kind == cardinal::trace::EventKind::Return && !call_stack.empty()) {
                const auto& s = trace_view_[call_stack.back()];
                spans.push_back({ s.t_ns, e.t_ns, s.depth, s.thread_id, s.name, s.category });
                call_stack.pop_back();
            }
        }

        if (spans.empty()) {
            ImGui::TextDisabled("(no completed spans — click Snapshot after exercising code)");
            ImGui::End();
            return;
        }

        const u64 t_min = trace_view_.front().t_ns;
        const u64 t_max = trace_view_.back().t_ns;
        const u64 t_span = (t_max > t_min) ? (t_max - t_min) : 1;

        ImGui::Text("Window: %.3f ms",
            static_cast<double>(t_span) / 1.0e6);

        const float row_h = ImGui::GetFontSize() + 4.0f;
        ImVec2 canvas = ImGui::GetContentRegionAvail();
        if (canvas.x < 50.0f || canvas.y < row_h * 2) { ImGui::End(); return; }

        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 origin = ImGui::GetCursorScreenPos();
        const float w = canvas.x;
        const float scale = w / static_cast<float>(t_span);

        for (const auto& s : spans) {
            const float x0 = origin.x + (s.t0 - t_min) * scale;
            const float x1 = origin.x + (s.t1 - t_min) * scale;
            const float y0 = origin.y + s.depth * row_h;
            const float y1 = y0 + row_h - 2.0f;
            ImU32 col = (cardinal::strcmp(s.cat, "lua") == 0)
                ? IM_COL32(110, 180, 230, 200)
                : IM_COL32(170, 220, 130, 200);
            dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), col, 2.0f);
            dl->AddRect      (ImVec2(x0, y0), ImVec2(x1, y1),
                              IM_COL32(20, 20, 20, 200), 2.0f);
            if (x1 - x0 > 30.0f) {
                dl->AddText(ImVec2(x0 + 3.0f, y0 + 1.0f),
                            IM_COL32(20, 20, 20, 255), s.name);
            }
        }
        ImGui::Dummy(canvas);
        ImGui::End();
    }

    // -------------------------------------------------------------------
    // Script Debugger panel — controls the Lua engine's debugger surface.
    // -------------------------------------------------------------------
    void draw_script_debugger_panel(script::Engine& engine,
                                    const char* title, bool* p_open) override {
        if (!ImGui::Begin(title ? title : "Script Debugger", p_open,
                          0)) { ImGui::End(); return; }

        // Toolbar.
        bool dbg = engine.debug_enabled();
        if (ImGui::Checkbox("Enable", &dbg)) engine.set_debug_enabled(dbg);
        ImGui::SameLine();
        const char* state_str =
            engine.state() == script::DebugState::Idle         ? "running" :
            engine.state() == script::DebugState::AtBreakpoint ? "PAUSED"  :
                                                                 "stepping";
        const ImVec4 col = engine.state() == script::DebugState::AtBreakpoint
            ? ImVec4(1.0f, 0.55f, 0.2f, 1.0f) : ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
        ImGui::SameLine();
        ImGui::TextColored(col, "[%s]", state_str);

        ImGui::SameLine();
        if (ImGui::Button("Continue")) engine.resume(script::StepMode::Continue);
        ImGui::SameLine();
        if (ImGui::Button("Step Over")) engine.resume(script::StepMode::Over);
        ImGui::SameLine();
        if (ImGui::Button("Step In"))   engine.resume(script::StepMode::In);
        ImGui::SameLine();
        if (ImGui::Button("Step Out"))  engine.resume(script::StepMode::Out);

        ImGui::Separator();

        // Breakpoints.
        ImGui::SeparatorText("Breakpoints");
        ImGui::SetNextItemWidth(220.0f);
        ImGui::InputText("source",  dbg_bp_source_, sizeof(dbg_bp_source_));
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80.0f);
        ImGui::InputInt ("line", &dbg_bp_line_, 0, 0);
        ImGui::SameLine();
        if (ImGui::Button("Add")) {
            if (dbg_bp_source_[0] && dbg_bp_line_ > 0) {
                engine.add_breakpoint(dbg_bp_source_, static_cast<u32>(dbg_bp_line_));
            }
        }
        for (auto& bp : engine.breakpoints()) {
            ImGui::PushID(&bp);
            ImGui::Text("%s:%u", bp.first.c_str(), bp.second);
            ImGui::SameLine();
            if (ImGui::SmallButton("x")) {
                engine.remove_breakpoint(bp.first.c_str(), bp.second);
            }
            ImGui::PopID();
        }

        // Stack + locals (only meaningful while paused).
        ImGui::SeparatorText("Stack");
        if (engine.state() == script::DebugState::AtBreakpoint) {
            for (const auto& f : engine.stack()) {
                ImGui::Text("%s  %s:%u  [%s]",
                    f.name.empty() ? "?" : f.name.c_str(),
                    f.source.c_str(), f.current_line, f.what.c_str());
            }
            ImGui::SeparatorText("Locals");
            for (const auto& v : engine.locals()) {
                ImGui::Text("%-16s %-10s = %s",
                    v.name.c_str(), v.type.c_str(), v.value.c_str());
            }
            ImGui::SeparatorText("Upvalues");
            for (const auto& v : engine.upvalues()) {
                ImGui::Text("%-16s %-10s = %s",
                    v.name.c_str(), v.type.c_str(), v.value.c_str());
            }
        } else {
            ImGui::TextDisabled("(not paused)");
        }
        ImGui::End();
    }

    // -------------------------------------------------------------------
    // Render Pipeline panel — picker + per-pipeline knobs + GPU features.
    // -------------------------------------------------------------------
    void draw_render_pipeline_panel(render::Registry& reg,
                                    const char* title, bool* p_open) override
    {
        if (!ImGui::Begin(title ? title : "Render Pipeline", p_open,
                          0)) {
            ImGui::End(); return;
        }

        // Active-pipeline picker.
        auto pipelines = reg.all();
        const char* current_name = reg.active() ? reg.active()->name() : "?";
        if (ImGui::BeginCombo("Active pipeline", current_name)) {
            for (auto* p : pipelines) {
                const bool is_sel = (reg.active() == p);
                if (ImGui::Selectable(p->name(), is_sel)) reg.set_active(p->id());
                if (is_sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        if (auto* p = reg.active()) {
            ImGui::TextWrapped("%s", p->description());
        }
        ImGui::Separator();

        // Optional: per-frame statistics published by the active pipeline.
        // Pipelines that don't compute stats return zeros; we hide the box
        // when nothing meaningful is reported.
        if (auto* p = reg.active()) {
            const auto fs = p->stats();
            if (fs.clusters_total > 0) {
                ImGui::SeparatorText("Frame Stats");
                ImGui::Text("Clusters total : %u", fs.clusters_total);
                const float culled_pct = fs.clusters_total > 0
                    ? 100.0f * static_cast<float>(fs.clusters_culled) /
                                static_cast<float>(fs.clusters_total)
                    : 0.0f;
                ImGui::Text("Clusters culled: %u  (%.1f%%)",
                            fs.clusters_culled, static_cast<double>(culled_pct));
                ImGui::Text("Clusters drawn : %u", fs.clusters_drawn);
                ImGui::Text("Triangles drawn: %u", fs.triangles_drawn);
                if (fs.tessellation_factor_avg > 0)
                    ImGui::Text("Tess factor avg: %u", fs.tessellation_factor_avg);
            }
        }

        // Knobs grouped by Knob::group.
        if (auto* p = reg.active()) {
            auto& knobs = p->knobs();
            cardinal::string current_group;
            for (auto& k : knobs) {
                if (k.group != current_group) {
                    ImGui::SeparatorText(k.group.c_str());
                    current_group = k.group;
                }
                draw_knob(k);
            }
        }

        ImGui::Separator();
        ImGui::SeparatorText("GPU Feature Matrix");
        ImGui::TextDisabled("Knobs above are auto-greyed when their feature isn't supported.");
        if (device_) {
            render::FeatureGate gate(device_->capabilities());
            cardinal::string cur_group;
            for (u32 i = 0; i < static_cast<u32>(render::Feature::Count_); ++i) {
                const auto f   = static_cast<render::Feature>(i);
                const auto st  = gate.query(f);
                const auto* g  = render::FeatureGate::feature_group(f);
                if (cur_group != g) { ImGui::SeparatorText(g); cur_group = g; }
                ImGui::TextDisabled("  %-26s", render::FeatureGate::feature_name(f));
                ImGui::SameLine(280.0f);
                ImGui::TextColored(st.available
                    ? ImVec4(0.40f, 1.0f, 0.40f, 1.0f)
                    : ImVec4(0.55f, 0.55f, 0.55f, 1.0f),
                    "%s", st.available ? "yes" : "no");
                if (!st.available && !st.reason.empty()) {
                    ImGui::SameLine(0.0f, 16.0f);
                    ImGui::TextDisabled("(%s)", st.reason.c_str());
                }
            }
        }
        ImGui::End();
    }

    void draw_knob(render::Knob& k) {
        const bool was_enabled = k.available;
        if (!was_enabled) ImGui::BeginDisabled();
        ImGui::PushID(k.id.c_str());
        switch (k.kind) {
            case render::KnobKind::Bool:
                ImGui::Checkbox(k.label.c_str(), &k.b);
                break;
            case render::KnobKind::Int:
                ImGui::SliderInt(k.label.c_str(), &k.i, k.i_min, k.i_max);
                break;
            case render::KnobKind::Float:
                ImGui::SliderFloat(k.label.c_str(), &k.f, k.f_min, k.f_max,
                                   "%.3f", ImGuiSliderFlags_AlwaysClamp);
                break;
            case render::KnobKind::Enum: {
                const char* preview = (k.e >= 0 && k.e < (int)k.enum_labels.size())
                    ? k.enum_labels[k.e].c_str() : "?";
                if (ImGui::BeginCombo(k.label.c_str(), preview)) {
                    for (int i = 0; i < (int)k.enum_labels.size(); ++i) {
                        const bool sel = (k.e == i);
                        if (ImGui::Selectable(k.enum_labels[i].c_str(), sel)) k.e = i;
                        if (sel) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                break;
            }
        }
        ImGui::PopID();
        if (!was_enabled) ImGui::EndDisabled();

        // Tooltip — both the Knob's own + the unavailability reason.
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::BeginTooltip();
            if (!k.tooltip.empty()) ImGui::TextWrapped("%s", k.tooltip.c_str());
            if (!was_enabled && !k.unavailable_reason.empty()) {
                ImGui::Separator();
                ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.4f, 1.0f),
                    "Unavailable: %s", k.unavailable_reason.c_str());
            }
            ImGui::EndTooltip();
        }
    }

    // Used by the Stack Tracer's "captured at" line.
    static cardinal::string current_timestamp() {
        const auto t = cardinal::time(nullptr);
        char buf[32];
        cardinal::strftime(buf, sizeof(buf), "%H:%M:%S", cardinal::localtime(&t));
        return buf;
    }

private:
    // ---- Asset browser helpers --------------------------------------
    enum class AssetKind { Folder, Mesh, Texture, Shader, Script, Audio, Other };

    struct AssetFile { cardinal::string name; cardinal::string path; AssetKind kind; };

    static AssetKind classify(const cardinal::fs::path& p) {
        if (cardinal::fs::is_directory(p)) return AssetKind::Folder;
        const auto e = p.extension().string();
        cardinal::string ext; ext.reserve(e.size());
        for (char c : e) ext.push_back(static_cast<char>(::tolower(c)));
        if (ext == ".gltf" || ext == ".glb" || ext == ".obj" || ext == ".fbx") return AssetKind::Mesh;
        if (ext == ".png"  || ext == ".jpg" || ext == ".jpeg"|| ext == ".tga" ||
            ext == ".dds"  || ext == ".bmp" || ext == ".hdr" || ext == ".exr") return AssetKind::Texture;
        if (ext == ".hlsl" || ext == ".glsl"|| ext == ".vert"|| ext == ".frag"||
            ext == ".comp" || ext == ".spv" || ext == ".dxil") return AssetKind::Shader;
        if (ext == ".lua"  || ext == ".py"  || ext == ".js"  || ext == ".cpp"||
            ext == ".hpp"  || ext == ".h"   || ext == ".c") return AssetKind::Script;
        if (ext == ".wav"  || ext == ".ogg" || ext == ".mp3" || ext == ".flac") return AssetKind::Audio;
        return AssetKind::Other;
    }
    static const char* asset_icon_for(AssetKind k) {
        switch (k) {
            case AssetKind::Folder:  return "[D]";
            case AssetKind::Mesh:    return "[M]";
            case AssetKind::Texture: return "[T]";
            case AssetKind::Shader:  return "[S]";
            case AssetKind::Script:  return "[L]";
            case AssetKind::Audio:   return "[A]";
            default:                 return "[ ]";
        }
    }
    static ImU32 asset_color_for(AssetKind k) {
        switch (k) {
            case AssetKind::Folder:  return IM_COL32(180, 180, 220, 255);
            case AssetKind::Mesh:    return IM_COL32(160, 220, 160, 255);
            case AssetKind::Texture: return IM_COL32(220, 180, 120, 255);
            case AssetKind::Shader:  return IM_COL32(200, 160, 220, 255);
            case AssetKind::Script:  return IM_COL32(220, 220, 140, 255);
            case AssetKind::Audio:   return IM_COL32(140, 200, 220, 255);
            default:                 return IM_COL32(170, 170, 170, 255);
        }
    }
    void scan_asset_dir() {
        asset_files_.clear();
        cardinal::error_code ec;
        if (!cardinal::fs::exists(asset_cwd_, ec)) return;
        for (auto& entry : cardinal::fs::directory_iterator(asset_cwd_, ec)) {
            AssetFile f;
            f.path = entry.path().string();
            f.name = entry.path().filename().string();
            f.kind = classify(entry.path());
            asset_files_.push_back(cardinal::move(f));
        }
        cardinal::sort(asset_files_.begin(), asset_files_.end(),
                  [](const AssetFile& a, const AssetFile& b){
                      if (a.kind != b.kind) return a.kind < b.kind;
                      return a.name < b.name;
                  });
    }
    void draw_asset_tree(const cardinal::string& dir) {
        cardinal::error_code ec;
        if (!cardinal::fs::exists(dir, ec) || !cardinal::fs::is_directory(dir, ec)) return;
        for (auto& entry : cardinal::fs::directory_iterator(dir, ec)) {
            if (!entry.is_directory(ec)) continue;
            const cardinal::string p = entry.path().string();
            const cardinal::string n = entry.path().filename().string();
            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;
            if (asset_cwd_ == p) flags |= ImGuiTreeNodeFlags_Selected;
            const bool open = ImGui::TreeNodeEx(p.c_str(), flags, "[D] %s", n.c_str());
            if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                asset_cwd_ = p;
                scan_asset_dir();
            }
            if (open) { draw_asset_tree(p); ImGui::TreePop(); }
        }
    }

private:
    bool create_descriptor_pool(VkDevice device) {
        // ImGui's font + each AddTexture() call wants COMBINED_IMAGE_SAMPLER.
        // Generous pool covers ImGui + headroom for app textures.
        const cardinal::array<VkDescriptorPoolSize, 1> pool_sizes = {{
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024 },
        }};
        VkDescriptorPoolCreateInfo pci{};
        pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pci.maxSets       = 1024;
        pci.poolSizeCount = static_cast<u32>(pool_sizes.size());
        pci.pPoolSizes    = pool_sizes.data();
        return vkCreateDescriptorPool(device, &pci, nullptr, &desc_pool_) == VK_SUCCESS;
    }

    bool create_render_pass(VkDevice device) {
        // Both render passes are render-pass-compatible (same attachment
        // format / sample count / count) so ImGui's single set of pipelines
        // works against either at vkCmdBeginRenderPass time. They differ
        // only in initial/final image layout: the main one matches what
        // cardinal's Vulkan swapchain hands us before/after the overlay,
        // and the secondary one matches what vkAcquireNextImageKHR gives
        // to ImGui's per-secondary-window swapchain (UNDEFINED in, ready
        // for vkQueuePresentKHR out).
        auto build = [&](VkImageLayout initial, VkImageLayout final_layout,
                         VkRenderPass& out) -> bool {
            VkAttachmentDescription att{};
            att.format         = color_format_;
            att.samples        = VK_SAMPLE_COUNT_1_BIT;
            att.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
            att.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
            att.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            att.initialLayout  = initial;
            att.finalLayout    = final_layout;

            VkAttachmentReference ref{};
            ref.attachment = 0;
            ref.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkSubpassDescription sub{};
            sub.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount = 1;
            sub.pColorAttachments    = &ref;

            VkSubpassDependency dep{};
            dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
            dep.dstSubpass    = 0;
            dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dep.srcAccessMask = 0;
            dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

            VkRenderPassCreateInfo rpc{};
            rpc.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            rpc.attachmentCount = 1;
            rpc.pAttachments    = &att;
            rpc.subpassCount    = 1;
            rpc.pSubpasses      = &sub;
            rpc.dependencyCount = 1;
            rpc.pDependencies   = &dep;

            return vkCreateRenderPass(device, &rpc, nullptr, &out) == VK_SUCCESS;
        };

        // Main: cardinal's swapchain transitions the back buffer into
        // COLOR_ATTACHMENT_OPTIMAL before invoking our overlay, and
        // expects it back in the same layout so it can do a final
        // transition to PRESENT_SRC_KHR in end_frame.
        if (!build(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   render_pass_)) return false;

        // Secondary: ImGui owns the per-window swapchain end-to-end. The
        // image arrives at UNDEFINED (loadOp=CLEAR makes that safe) and
        // must finish at PRESENT_SRC_KHR for vkQueuePresentKHR.
        if (!build(VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                   render_pass_secondary_)) return false;

        return true;
    }

    bool create_framebuffers(VkDevice device, rhi::Swapchain* sw) {
        framebuffers_.resize(image_count_, VK_NULL_HANDLE);
        framebuffer_extent_.width  = sw->width();
        framebuffer_extent_.height = sw->height();
        for (u32 i = 0; i < image_count_; ++i) {
            VkImageView v = rhi::vulkan_swapchain_view(sw, i);
            if (v == VK_NULL_HANDLE) return false;

            VkFramebufferCreateInfo fb{};
            fb.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fb.renderPass      = render_pass_;
            fb.attachmentCount = 1;
            fb.pAttachments    = &v;
            fb.width           = framebuffer_extent_.width;
            fb.height          = framebuffer_extent_.height;
            fb.layers          = 1;
            if (vkCreateFramebuffer(device, &fb, nullptr, &framebuffers_[i]) != VK_SUCCESS) {
                return false;
            }
        }
        return true;
    }

    bool create_sampler(VkDevice device) {
        VkSamplerCreateInfo si{};
        si.sType     = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode= VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.minLod    = 0.0f;
        si.maxLod    = 1.0f;
        return vkCreateSampler(device, &si, nullptr, &viewport_sampler_) == VK_SUCCESS;
    }

    void try_bind_viewport_texture(u32 id) {
        if (swapchain_ == nullptr) return;
        ensure_panel_slots(id);
        const u32 vw = swapchain_->viewport_width(id);
        const u32 vh = swapchain_->viewport_height(id);
        if (vw == 0 || vh == 0) return;
        auto& vp = vp_panels_[id];

        if (gpu_ == Gpu::Vulkan) {
            if (vp.vk_descriptor != VK_NULL_HANDLE) return;
            VkImageView vv = vulkan_viewport_view(swapchain_, id);
            if (vv == VK_NULL_HANDLE) return;
            vp.vk_descriptor = ImGui_ImplVulkan_AddTexture(
                viewport_sampler_, vv, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            cardinal::log::infof("ui/studio",
                "Vulkan viewport[%u] SRV bound (descriptor=%p, view=%p)",
                id, static_cast<void*>(vp.vk_descriptor), static_cast<void*>(vv));
            return;
        }

#if CARDINAL_PLATFORM_WINDOWS
        // D3D12 — wait for the swapchain to lazily create the viewport
        // resource on its first begin_frame, then create an SRV for it
        // in our heap. The GPU descriptor handle is what ImGui treats as
        // the ImTextureID for ImGui::Image() calls. The actual per-frame
        // state transition (RENDER_TARGET ↔ PIXEL_SHADER_RESOURCE) lives
        // in D3D12Swapchain::end_frame around the overlay callback.
        if (vp.dx12_imtex != nullptr) return;     // already bound
        ID3D12Resource* vimg = rhi::d3d12_viewport_image(swapchain_, id);
        if (vimg == nullptr || dx12_srv_heap_ == nullptr ||
            dx12_srv_free_.empty()) return;

        const auto dh = rhi::d3d12_handles(device_);
        if (dh.device == nullptr) return;

        const UINT slot = dx12_srv_free_.back();
        dx12_srv_free_.pop_back();
        vp.dx12_srv_slot = slot;

        D3D12_CPU_DESCRIPTOR_HANDLE cpu = dx12_srv_heap_->GetCPUDescriptorHandleForHeapStart();
        D3D12_GPU_DESCRIPTOR_HANDLE gpu = dx12_srv_heap_->GetGPUDescriptorHandleForHeapStart();
        cpu.ptr += static_cast<SIZE_T>(slot) * dx12_srv_stride_;
        gpu.ptr += static_cast<UINT64>(slot) * dx12_srv_stride_;

        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format                  = DXGI_FORMAT_B8G8R8A8_UNORM;
        srv.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels     = 1;
        dh.device->CreateShaderResourceView(vimg, &srv, cpu);

        // ImGui DX12 backend treats ImTextureID as the gpu_handle.ptr value.
        vp.dx12_imtex = reinterpret_cast<void*>(static_cast<uintptr_t>(gpu.ptr));
        cardinal::log::infof("ui/studio",
            "D3D12 viewport[%u] SRV bound (slot %u, gpu.ptr=0x%llx)",
            id, slot, static_cast<unsigned long long>(gpu.ptr));
#endif
    }

    // --- Overlay callback — called by Swapchain::end_frame -------------------
    static void overlay_thunk(void* user) {
        static_cast<StudioImpl*>(user)->overlay_render();
    }
    void overlay_render() {
        ImDrawData* dd = ImGui::GetDrawData();
        if (dd == nullptr || dd->DisplaySize.x <= 0 || dd->DisplaySize.y <= 0) return;

        if (gpu_ == Gpu::Vulkan) {
            VkCommandBuffer cmd = rhi::current_command_buffer(swapchain_);
            const u32       img = rhi::vulkan_acquired_image_index(swapchain_);
            if (cmd == VK_NULL_HANDLE || img >= framebuffers_.size()) return;

            VkClearValue clear{};
            clear.color = {{0.07f, 0.08f, 0.10f, 1.0f}};
            VkRenderPassBeginInfo rpbi{};
            rpbi.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rpbi.renderPass        = render_pass_;
            rpbi.framebuffer       = framebuffers_[img];
            rpbi.renderArea.extent = framebuffer_extent_;
            rpbi.clearValueCount   = 1;
            rpbi.pClearValues      = &clear;
            vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
            ImGui_ImplVulkan_RenderDrawData(dd, cmd);
            vkCmdEndRenderPass(cmd);
            return;
        }
#if CARDINAL_PLATFORM_WINDOWS
        // D3D12 — the swapchain has the back buffer in RENDER_TARGET state
        // and OMSetRenderTargets has already been issued for it. We just
        // need to bind our SRV heap so ImGui can find its font + textures
        // and dispatch the draw.
        ID3D12GraphicsCommandList* cmd = rhi::d3d12_current_cmd(swapchain_);
        if (cmd == nullptr) return;
        ID3D12DescriptorHeap* heaps[] = { dx12_srv_heap_.Get() };
        cmd->SetDescriptorHeaps(1, heaps);
        ImGui_ImplDX12_RenderDrawData(dd, cmd);
#endif
    }

    void shutdown() {
        if (!initialized()) return;
        cardinal::log::remove_sink(&log_state_.cstore);

        // Detach EVERY callback we ever registered on the swapchain
        // before we tear ourselves down. Anything that fires after
        // this point (a stray rebuild during teardown, late OS event)
        // sees a null callback instead of dereferencing into our
        // freed state. Order matters: do this first so the rest of
        // shutdown can't race with a callback re-entering studio.
        if (swapchain_ != nullptr) {
            swapchain_->set_overlay(nullptr, nullptr);
            swapchain_->set_on_rebuilt(nullptr);
        }

        // Destroy any multi-viewport secondary platform windows BEFORE we
        // shut down the renderer backend. ImGui owns Win32 child HWNDs +
        // per-viewport DXGI swapchains; if we tear down the renderer first
        // those secondary swapchains would be destroyed by the WM_DESTROY
        // path mid-shutdown, tripping IM_ASSERTs in the backend that expect
        // the GPU to be idle. DestroyPlatformWindows synchronously walks
        // every viewport, calls Renderer_DestroyWindow + Platform_DestroyWindow,
        // and waits on each viewport's fence. Idempotent + cheap if there
        // are no torn-out windows.
        ImGui::DestroyPlatformWindows();

        if (gpu_ == Gpu::Vulkan) {
            const auto dh = rhi::vulkan_handles(device_);
            if (dh.device != VK_NULL_HANDLE) {
                vkDeviceWaitIdle(dh.device);
                ImGui_ImplVulkan_Shutdown();
#if CARDINAL_PLATFORM_WINDOWS
                ImGui_ImplWin32_Shutdown();
                // Detach JUST our hook — leave any sibling hooks (input
                // manager, debug overlays) attached so they keep working.
                if (window_ != nullptr) {
                    window_->remove_message_hook(&StudioImpl::message_hook_thunk, this);
                }
#endif
                ImGui::DestroyContext();

                if (viewport_sampler_ != VK_NULL_HANDLE)
                    vkDestroySampler(dh.device, viewport_sampler_, nullptr);
                for (auto fb : framebuffers_) {
                    if (fb != VK_NULL_HANDLE) vkDestroyFramebuffer(dh.device, fb, nullptr);
                }
                framebuffers_.clear();
                if (render_pass_ != VK_NULL_HANDLE)
                    vkDestroyRenderPass(dh.device, render_pass_, nullptr);
                if (render_pass_secondary_ != VK_NULL_HANDLE)
                    vkDestroyRenderPass(dh.device, render_pass_secondary_, nullptr);
                if (desc_pool_ != VK_NULL_HANDLE)
                    vkDestroyDescriptorPool(dh.device, desc_pool_, nullptr);
            }
        }
#if CARDINAL_PLATFORM_WINDOWS
        else {
            // D3D12 — drain the GPU before tearing ImGui down.
            const auto dh = rhi::d3d12_handles(device_);
            if (dh.gfx_queue) {
                ComPtr<ID3D12Fence> fence;
                if (SUCCEEDED(dh.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) {
                    HANDLE evt = CreateEventA(nullptr, FALSE, FALSE, nullptr);
                    dh.gfx_queue->Signal(fence.Get(), 1);
                    if (fence->GetCompletedValue() < 1) {
                        fence->SetEventOnCompletion(1, evt);
                        WaitForSingleObject(evt, INFINITE);
                    }
                    if (evt) CloseHandle(evt);
                }
            }
            ImGui_ImplDX12_Shutdown();
            ImGui_ImplWin32_Shutdown();
            // Detach JUST our hook (see Vulkan path above for rationale).
            if (window_ != nullptr) {
                window_->remove_message_hook(&StudioImpl::message_hook_thunk, this);
            }
            ImGui::DestroyContext();
            dx12_srv_heap_.Reset();
            dx12_srv_free_.clear();
        }
#endif

        device_ = nullptr; swapchain_ = nullptr; window_ = nullptr;
        desc_pool_ = VK_NULL_HANDLE;
        render_pass_ = VK_NULL_HANDLE;
        render_pass_secondary_ = VK_NULL_HANDLE;
        viewport_sampler_ = VK_NULL_HANDLE;
        // Per-panel viewport bindings — drop in place; the underlying GPU
        // descriptors are torn down by the backend shutdown above.
        for (auto& vp : vp_panels_) vp.vk_descriptor = VK_NULL_HANDLE;
    }

    // ----- Backend hot-swap (DX12 ↔ Vulkan at runtime) ---------------------
    //
    // Releases everything tied to the OLD device + swapchain (renderer
    // backend, descriptor heaps, render passes, sampler, viewport SRV
    // bindings, framebuffers) but leaves the ImGui CONTEXT, log sink,
    // tool-window registry, console state, panels' state, and Win32
    // platform backend untouched. Then re-runs init_vulkan / init_d3d12
    // against the new pair.
    //
    // Engine-level orchestration drains the OLD device (vkDeviceWaitIdle /
    // d3d12 fence wait) BEFORE this is called, then runs the host's
    // before-hook (release app-owned GPU buffers), destroys the old
    // device + swapchain, creates the new ones, calls reinit() here,
    // then runs the after-hook (re-upload meshes / textures).
    bool reinit(rhi::Device& new_device, rhi::Swapchain& new_swapchain) override {
        if (!initialized()) return false;

        // 1. Drop swapchain overlay registration on the OLD swapchain so
        //    nothing re-enters ImGui during teardown.
        if (swapchain_ != nullptr) swapchain_->set_overlay(nullptr, nullptr);

        // 2. Tear down the renderer backend + per-panel descriptor bindings.
        //    Caller is responsible for having drained the GPU first.
        if (gpu_ == Gpu::Vulkan) {
            ImGui_ImplVulkan_Shutdown();
            const auto dh_old = rhi::vulkan_handles(device_);
            if (dh_old.device != VK_NULL_HANDLE) {
                if (viewport_sampler_ != VK_NULL_HANDLE)
                    vkDestroySampler(dh_old.device, viewport_sampler_, nullptr);
                for (auto fb : framebuffers_) {
                    if (fb != VK_NULL_HANDLE) vkDestroyFramebuffer(dh_old.device, fb, nullptr);
                }
                framebuffers_.clear();
                if (render_pass_ != VK_NULL_HANDLE)
                    vkDestroyRenderPass(dh_old.device, render_pass_, nullptr);
                if (render_pass_secondary_ != VK_NULL_HANDLE)
                    vkDestroyRenderPass(dh_old.device, render_pass_secondary_, nullptr);
                if (desc_pool_ != VK_NULL_HANDLE)
                    vkDestroyDescriptorPool(dh_old.device, desc_pool_, nullptr);
            }
            viewport_sampler_      = VK_NULL_HANDLE;
            render_pass_           = VK_NULL_HANDLE;
            render_pass_secondary_ = VK_NULL_HANDLE;
            desc_pool_             = VK_NULL_HANDLE;
        }
#if CARDINAL_PLATFORM_WINDOWS
        else {
            ImGui_ImplDX12_Shutdown();
            dx12_srv_heap_.Reset();
            dx12_srv_free_.clear();
            dx12_srv_stride_ = 0;
        }
#endif
        // Per-panel descriptors / SRV slots are now dead (they pointed at
        // the OLD device's resources). Reset so try_bind_viewport_texture
        // re-creates them on first viewport_texture(id) call after swap.
        for (auto& vp : vp_panels_) {
            vp.vk_descriptor = VK_NULL_HANDLE;
#if CARDINAL_PLATFORM_WINDOWS
            vp.dx12_srv_slot = 0xFFFFFFFFu;
            vp.dx12_imtex    = nullptr;
#endif
        }

        // 3. Switch to the new device + swapchain. window_ persists (we
        //    reuse the same OS HWND across the swap; only the GPU device
        //    changes).
        device_    = &new_device;
        swapchain_ = &new_swapchain;
        gpu_       = (new_device.backend() == rhi::Backend::Vulkan)
            ? Gpu::Vulkan : Gpu::D3D12;

        // 4. Re-run backend init against the new pair.
        const bool ok = (gpu_ == Gpu::Vulkan) ? init_vulkan() : init_d3d12();
        if (!ok) {
            cardinal::log::errorf("ui/studio",
                "reinit: backend init failed (gpu=%s)",
                gpu_ == Gpu::Vulkan ? "Vulkan" : "D3D12");
            return false;
        }

        // 5. Re-register overlay on the new swapchain.
        new_swapchain.set_overlay(&StudioImpl::overlay_thunk, this);

        // 6. Bind viewport 0's SRV up front so the first frame after swap
        //    has a valid ImTextureID for the main viewport panel. Other
        //    panel ids lazy-bind on their next viewport_texture(id) call.
        try_bind_viewport_texture(0u);

        cardinal::log::infof("ui/studio",
            "reinit complete — running on %s",
            gpu_ == Gpu::Vulkan ? "Vulkan" : "D3D12");
        return true;
    }

    bool initialized() const { return device_ != nullptr; }

    static PFN_vkVoidFunction load_volk(const char* name, void* user) {
        auto instance = static_cast<VkInstance>(user);
        return vkGetInstanceProcAddr(instance, name);
    }

#if CARDINAL_PLATFORM_WINDOWS
    // Multi-viewport surface creator. ImGui supplies a viewport whose
    // PlatformHandle is the HWND it just created; we wrap it in a
    // VkSurfaceKHR via vkCreateWin32SurfaceKHR.
    static int create_vk_surface_thunk(ImGuiViewport* vp, ImU64 vk_inst,
                                       const void* allocator, ImU64* out_surface)
    {
        HWND hwnd = static_cast<HWND>(vp->PlatformHandleRaw
                                       ? vp->PlatformHandleRaw
                                       : vp->PlatformHandle);
        if (hwnd == nullptr) return -1;
        auto pCreate = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
            vkGetInstanceProcAddr(reinterpret_cast<VkInstance>(vk_inst),
                                  "vkCreateWin32SurfaceKHR"));
        if (pCreate == nullptr) return -1;
        VkWin32SurfaceCreateInfoKHR ci{};
        ci.sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        ci.hwnd      = hwnd;
        ci.hinstance = GetModuleHandleW(nullptr);
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        VkResult r = pCreate(reinterpret_cast<VkInstance>(vk_inst), &ci,
                             static_cast<const VkAllocationCallbacks*>(allocator),
                             &surface);
        *out_surface = reinterpret_cast<ImU64>(surface);
        return r == VK_SUCCESS ? 0 : -1;
    }
#endif

    // Computes <exe-dir>/cardinal_studio.ini and writes it into out_buf.
    // Returns false (out_buf untouched) if the path doesn't fit.
    static bool resolve_ini_path(char* out_buf, size_t cap) {
#if CARDINAL_PLATFORM_WINDOWS
        char  exe_path[MAX_PATH];
        DWORD n = GetModuleFileNameA(nullptr, exe_path, sizeof(exe_path));
        if (n == 0 || n >= sizeof(exe_path)) return false;
        // Strip the file name to get the directory.
        for (DWORD i = n; i > 0; --i) {
            if (exe_path[i - 1] == '\\' || exe_path[i - 1] == '/') {
                exe_path[i] = '\0';
                int written = cardinal::snprintf(out_buf, cap, "%scardinal_studio.ini", exe_path);
                return written > 0 && static_cast<size_t>(written) < cap;
            }
        }
        return false;
#else
        (void)out_buf; (void)cap;
        return false;  // POSIX path resolution can land here when WSI is wired
#endif
    }

#if CARDINAL_PLATFORM_WINDOWS
    static bool message_hook_thunk(void* hwnd, u32 msg, u64 wp, i64 lp, void* /*user*/) {
        return ImGui_ImplWin32_WndProcHandler(
            static_cast<HWND>(hwnd), static_cast<UINT>(msg),
            static_cast<WPARAM>(wp), static_cast<LPARAM>(lp)) != 0;
    }
#endif

    rhi::Device*            device_{nullptr};
    rhi::Swapchain*         swapchain_{nullptr};
    window::Window*         window_{nullptr};

    VkDescriptorPool        desc_pool_{VK_NULL_HANDLE};
    VkRenderPass            render_pass_{VK_NULL_HANDLE};
    // Secondary-window render pass — used by ImGui for OS-level pop-out
    // viewports. Differs from render_pass_ only in initialLayout
    // (UNDEFINED — secondary swapchain images arrive that way from
    // vkAcquireNextImageKHR) and finalLayout (PRESENT_SRC_KHR — ImGui
    // submits + presents the secondary swapchain itself, so we transition
    // to present-ready in the render pass instead of a separate barrier).
    // Render-pass-compatible with render_pass_ (same attachment format,
    // sample count, attachment count) so ImGui's single set of pipelines
    // works against both passes.
    VkRenderPass            render_pass_secondary_{VK_NULL_HANDLE};
    cardinal::vector<VkFramebuffer> framebuffers_;
    VkExtent2D              framebuffer_extent_{0, 0};
    VkFormat                color_format_{VK_FORMAT_UNDEFINED};
    u32                     image_count_{0};

    VkSampler               viewport_sampler_{VK_NULL_HANDLE};
    // (Per-viewport descriptor sets live in vp_panels_[id].vk_descriptor.)

    cardinal::chrono::steady_clock::time_point last_sample_{cardinal::chrono::steady_clock::now()};
    u64                                   last_sample_frame_{0};
    u64                                   frame_count_{0};
    float                                 ema_fps_{0.0f};

    // Focused-panel maximize state (toggle_maximize_focused_panel + Shift+F11).
    cardinal::string                      max_win_;           // "" = none
    ImGuiID                               max_prev_dock_{0};  // dock to restore
    bool                                  max_enter_pending_{false};
    bool                                  max_restore_pending_{false};
    cardinal::string                      max_restore_win_;
    ImGuiID                               max_restore_dock_{0};

    // Log panel state — sink + scratch buffers + filter + min level. The
    // panel rendering lives in panels/log.cpp; we just hold the state and
    // the cardinal::log sink lifetime here. log_state_.cstore is
    // registered as a sink in initialize() and removed in shutdown().
    panels::log_panel::State       log_state_{};
    // Code Sandbox panel state — input path the user is typing + selected
    // job id for the diagnostics pane. Bound to a cardinal::cppscript::Engine
    // by the host on each draw call.
    panels::cppscript_panel::State cppscript_panel_state_{};
    // Profiler panel — frame-time history is owned here so it persists
    // across frames; per-phase + per-worker data is queried fresh from
    // cardinal::async on each draw.
    panels::profiler_panel::State        profiler_state_{};
    panels::mesh_tools_panel::State      mesh_tools_state_{};
    panels::texture_tools_panel::State   tex_tools_state_{};
    panels::cook_pack_panel::State       cook_pack_state_{};
    panels::prefab_panel::State          prefab_panel_state_{};
    panels::game_bar_panel::State        game_bar_state_{};

    // World-grid level-editor: which chunk the cursor is over (filled by
    // draw_world_grid_overlay). Read by draw_viewport_context_menu so a
    // right-click can offer "Spawn at chunk (cx, cy, cz)". Reset to false
    // at the top of each frame so a stale hit doesn't leak.
    i32  hover_chunk_x_{0};
    i32  hover_chunk_y_{0};
    i32  hover_chunk_z_{0};
    bool hover_chunk_valid_{false};

    // Context-menu freeze of hover_chunk_*. The live state is reset to
    // valid=false at the top of every frame and only set when the cursor
    // is over a viewport panel's grid-overlay rect — but the Alt+RMB
    // context menu PERSISTS across frames, and by the time the user
    // moves onto the menu item and clicks "Spawn active asset here",
    // the cursor is over the popup (not the panel), live hover is
    // invalid, and the host's spawn-at-cursor fell back to camera-
    // forward instead of the chunk the user actually right-clicked.
    // Captured on OpenPopup (when the right-click happens and the live
    // hover is correct for that exact cursor); cleared when the popup
    // closes (item-clicked / clicked-outside / Esc) so the next right-
    // click captures fresh. Preferred by last_hover_chunk() when valid.
    i32  ctx_hover_chunk_x_{0};
    i32  ctx_hover_chunk_y_{0};
    i32  ctx_hover_chunk_z_{0};
    bool ctx_hover_chunk_valid_{false};

    // Backing storage for ImGuiIO::IniFilename (which holds a non-owning
    // const char*). 1 KB covers any reasonable absolute path.
    char                           ini_path_[1024]{};

    // Which GPU backend is hosting the editor — selected in initialize().
    enum class Gpu { Vulkan, D3D12 };
    Gpu                            gpu_{Gpu::Vulkan};

#if CARDINAL_PLATFORM_WINDOWS
    // D3D12-only state: the SRV descriptor heap ImGui shares with us, plus
    // a free list of slot indices for the Alloc/Free callbacks.
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> dx12_srv_heap_;
    UINT                                         dx12_srv_stride_{0};
    UINT                                         dx12_srv_capacity_{1024};
    cardinal::vector<UINT>                            dx12_srv_free_;

    // (Per-viewport SRV bindings live in vp_panels_[id].dx12_srv_slot /
    // .dx12_imtex now — moved out of single-viewport scalars so each
    // editor panel can sample its own RTT.)

    static void dx12_srv_alloc_thunk(ImGui_ImplDX12_InitInfo* info,
        D3D12_CPU_DESCRIPTOR_HANDLE* out_cpu, D3D12_GPU_DESCRIPTOR_HANDLE* out_gpu)
    {
        auto* self = static_cast<StudioImpl*>(info->UserData);
        if (self == nullptr || self->dx12_srv_free_.empty()) {
            *out_cpu = {}; *out_gpu = {}; return;
        }
        const UINT idx = self->dx12_srv_free_.back();
        self->dx12_srv_free_.pop_back();
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = self->dx12_srv_heap_->GetCPUDescriptorHandleForHeapStart();
        D3D12_GPU_DESCRIPTOR_HANDLE gpu = self->dx12_srv_heap_->GetGPUDescriptorHandleForHeapStart();
        cpu.ptr += static_cast<SIZE_T>(idx) * self->dx12_srv_stride_;
        gpu.ptr += static_cast<UINT64>(idx) * self->dx12_srv_stride_;
        *out_cpu = cpu;
        *out_gpu = gpu;
    }
    static void dx12_srv_free_thunk(ImGui_ImplDX12_InitInfo* info,
        D3D12_CPU_DESCRIPTOR_HANDLE cpu, D3D12_GPU_DESCRIPTOR_HANDLE /*gpu*/)
    {
        auto* self = static_cast<StudioImpl*>(info->UserData);
        if (self == nullptr || self->dx12_srv_heap_ == nullptr) return;
        const SIZE_T base = self->dx12_srv_heap_->GetCPUDescriptorHandleForHeapStart().ptr;
        const SIZE_T off  = cpu.ptr - base;
        const UINT   idx  = static_cast<UINT>(off / self->dx12_srv_stride_);
        if (idx < self->dx12_srv_capacity_) self->dx12_srv_free_.push_back(idx);
    }
#endif

    // Asset browser state.
    cardinal::string                    asset_root_;
    cardinal::string                    asset_cwd_;
    cardinal::vector<AssetFile>         asset_files_;
    ImGuiTextFilter                asset_filter_;

    // Outliner inline-rename state (double-click a row to rename in place).
    u32                            rename_id_    {0};
    bool                           rename_focus_ {false};
    char                           rename_buf_[128] {};
    ImGuiTextFilter                hier_filter_;   // Outliner name search

    // Console / REPL state — actual panel rendering lives in
    // panels/console.cpp. We hold the scrollback + input + focus/submit
    // edges across frames and pass them through on each draw.
    panels::ConsolePanelState      console_state_{};

    // Stack Tracer panel state.
    cardinal::vector<cardinal::trace::StackFrame> captured_stack_;
    cardinal::string                              captured_stack_label_;

    // Function Tracer panel state.
    cardinal::vector<cardinal::trace::Event>      trace_view_;

    // Script Debugger panel state.
    char                                     dbg_bp_source_[256]{};
    int                                      dbg_bp_line_{1};

    // Per-viewport panel state — one entry per editor viewport panel.
    //
    // Studio caches all the per-panel bookkeeping (resize debounce, ImGui
    // texture bindings, etc.) here so each draw_viewport_panel(id, …) can
    // index into its own slot. Slots are lazily grown by ensure_panel_slots
    // on first use of a given id.
    //
    // Resize debounce — avoid thrashing the off-screen RTT every frame the
    // user is dragging a window edge. pending_w/h = candidate size we're
    // watching for stability; commit_w/h = stable size waiting for
    // end_frame so the resize doesn't dangle live ImGui descriptors.
    //
    // SRV bindings — D3D12 burns a slot in dx12_srv_heap_ for the viewport
    // image's SRV; the GPU handle.ptr packed into a void* is the
    // ImTextureID for ImGui::Image. Vulkan adds the per-image-view
    // descriptor through ImGui_ImplVulkan_AddTexture and stores the
    // descriptor set. Both rebind on size change (the resource pointer
    // moves when the swapchain reallocates).
    struct ViewportPanelState {
        u32   pending_w{0};
        u32   pending_h{0};
        u32   commit_w {0};
        u32   commit_h {0};
        int   stable_frames{0};
#if CARDINAL_PLATFORM_WINDOWS
        UINT  dx12_srv_slot{0xFFFFFFFFu};   // sentinel = unbound
        void* dx12_imtex   {nullptr};
#endif
        VkDescriptorSet vk_descriptor{VK_NULL_HANDLE};
    };
    cardinal::vector<ViewportPanelState>          vp_panels_{};

    // Track which viewport id was hovered last so the legacy single-pick
    // ViewportPick still makes sense in a multi-viewport world. The host
    // reads last_viewport_pick() once per frame.
    u32                                      vp_last_hovered_id_{0u};

    // Per-frame pick / interaction snapshot (see ViewportPick struct in
    // studio.hpp). Reset in begin_frame, populated by draw_viewport_panel,
    // read by the host loop after end_frame.
    ViewportPick                             vp_pick_{};
    ViewportCtxAction                        vp_ctx_{};

    // ----- Gizmo state (host-pushed each frame) ----------------------------
    // gizmo_view_/proj_  : current camera matrices for projecting the gizmo
    // gizmo_target_      : world-space pivot — usually the selected entity
    // gizmo_axis_active_ : -1 idle, 0/1/2 = X/Y/Z while a drag is live
    // gizmo_drag_origin_*: anchor captured on mouse-down so the per-frame
    //                      world delta can be derived from the absolute
    //                      mouse position rather than accumulating per-frame
    //                      deltas (avoids drift from rounding).
    cardinal::scene::Mat4                    gizmo_view_{};
    cardinal::scene::Mat4                    gizmo_proj_{};
    bool                                     gizmo_have_camera_{false};
    cardinal::scene::Vec3                    gizmo_target_{};
    bool                                     gizmo_has_target_{false};
    int                                      gizmo_axis_active_{-1};
    ImVec2                                   gizmo_drag_origin_pix_{};
    cardinal::scene::Vec3                    gizmo_drag_origin_world_{};
    cardinal::scene::Vec3                    gizmo_drag_axis_dir_{};   // unit (1,0,0) etc
    float                                    gizmo_drag_t_origin_{0.0f};
    bool                                     gizmo_drag_in_progress_{false};
    bool                                     gizmo_drag_release_{false};
    // Which viewport panel owns the active drag. When multiple panels are
    // visible (split-dock, torn-out), every panel runs the gizmo dispatcher
    // each frame; without this gate, every panel's drag-update fires with
    // its OWN matrices + local mouse and the LAST panel drawn overwrites
    // gizmo_target_ — when the click-panel isn't drawn last, the object
    // teleports per-frame to wherever the last panel's ray happens to
    // intersect the captured axis. Captured at drag-start, cleared on
    // mouse-up (mouse-up handled in ANY panel so a closed/torn-out owning
    // panel can't strand the drag). 0xFFFFFFFFu = no drag owner.
    u32                                      gizmo_drag_viewport_id_{0xFFFFFFFFu};
    GizmoDrag                                gizmo_drag_{};

    // ----- Extended gizmo state (modes / rotate / scale / snap) ---------
    // gizmo_handle_active_ encodes which handle is being dragged:
    //   -1 idle | 0/1/2 X/Y/Z axis | 3/4/5 YZ/XZ/XY plane | 6 screen|uniform
    GizmoMode                                gizmo_mode_{GizmoMode::Translate};
    bool                                     gizmo_snap_on_{false};
    float                                    gizmo_snap_step_{1.0f};
    cardinal::scene::Vec3                    gizmo_target_rot_{};   // euler rad
    cardinal::scene::Vec3                    gizmo_target_scl_{1,1,1};

    // ----- ImGuizmo integration -----------------------------------------
    // When true, draw_viewport_panel routes the gizmo through ImGuizmo's
    // Manipulate/ViewManipulate. When false, falls back to Cardinal's
    // hand-rolled overlay (still in tree for A/B). Default true — the
    // ImGuizmo widget is the industry-standard look + supports Local/World
    // toggle, Universal mode (T+R+S in one widget), and per-axis snap out
    // of the box.
    bool                                     use_imguizmo_{true};
    // World vs local-space transform — drives ImGuizmo::MODE on Manipulate.
    bool                                     imguizmo_local_space_{false};
    // ViewManipulate orbit-cube widget in the corner of the viewport. Default
    // OFF — the cube renders a per-frame 96 x 96 input-capturing area in the
    // viewport's top-right which can confuse hit testing on adjacent docked
    // panels. The host opts in via set_imguizmo_show_view_cube(true) when it
    // actually wants the orbit widget.
    bool                                     imguizmo_show_view_cube_{false};
    int                                      gizmo_handle_active_{-1};
    // Drag-start anchors (captured on mouse-down; results derived from
    // these + the current mouse ray every frame → no integration drift).
    cardinal::scene::Vec3                    gizmo_drag_origin_rot_{};
    cardinal::scene::Vec3                    gizmo_drag_origin_scl_{1,1,1};
    cardinal::scene::Vec3                    gizmo_drag_plane_hit0_{};
    float                                    gizmo_drag_angle0_{0.0f};
    cardinal::scene::Vec3                    gizmo_drag_scl_ref_{}; // screen-uniform ref
    float                                    gizmo_drag_uniform0_{0.0f};

    // ----- Multi-selection set (Studio-owned; host reads back) ----------
    cardinal::vector<u32>                         selection_{};
    bool                                     marquee_dragging_{false};
    // Pin marquee box-select to the panel where it started. The marquee
    // block is gated by IsItemHovered, so on cross-panel drags the panel
    // currently under the cursor processes the live rect AND the
    // release commit — but `a0 = cur - dd` is the drag-start point in
    // OS desktop coords (originally in the originator panel) and would
    // be converted to NDC using the WRONG panel's image_pos, producing
    // nonsense selection. Capture viewport_id on drag-start; only the
    // originator commits on release. (Live-draw still happens in any
    // hovered panel for visual continuity — ImGui clips to the window.)
    u32                                      marquee_viewport_id_{0xFFFFFFFFu};

    // Asset-palette UI state — preserves filter text + collapsed groups
    // across frames.
    char                                     palette_filter_[64]{};

    // Hierarchy panel state — filter buffer + expand pulse. The panel
    // body lives in panels/hierarchy.cpp; we just hold the state across
    // frames and pass it through on each draw.
    panels::HierarchyPanelState              hierarchy_state_{};

    // Set true after the first-frame layout sanity pass clamps any stale
    // off-screen window positions; reset by reset_layout().
    bool                                     clamp_done_{false};

    // Terrain modal — last-used parameters carry over so the user can
    // re-open and tweak. terrain_lib_path_ holds the on-disk file path
    // for Save lib / Load lib (defaults to a sensible filename in the cwd).
    bool                                     terrain_modal_open_{false};
    TerrainSpawnRequest                      terrain_modal_state_{};
    char                                     terrain_lib_path_[260]{ "terrain_profiles.txt" };

    // World grid overlay state — host pushes via set_world_grid_overlay each
    // frame; viewport panel reads it during draw_world_grid_overlay.
    WorldGridOverlay                         world_overlay_{};

    // Tool-window registry — apps register custom dockable panels via
    // register_tool_window(). draw_tool_windows() iterates this each frame,
    // draw_tool_windows_menu() emits one MenuItem per entry, and
    // registered_tool_windows() exposes the list to the host's dockspace
    // builder so it can place each tool at its dock_hint the first time
    // the layout is constructed.
    cardinal::vector<ToolWindow>                  tool_windows_{};
};

}  // namespace

cardinal::unique_ptr<Studio> Studio::create(
    rhi::Device& device, rhi::Swapchain& swapchain, window::Window& window)
{
    auto s = cardinal::make_unique<StudioImpl>();
    if (!s->initialize(device, swapchain, window)) return nullptr;
    return s;
}

}  // namespace cardinal::ui
