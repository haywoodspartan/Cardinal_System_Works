#pragma once

// =============================================================================
// Cardinal RHI - Vulkan backend internal header.
//
// Mirrors the D3D12 split: shared includes + inline helpers + the leaf
// classes (Buffer / OneShotCmd / Texture / Pipeline / AccelerationStructure)
// + the big-class declarations (VulkanSwapchain, VulkanDevice). Method bodies
// live in vulkan.cpp (device), swapchain.cpp, commands.cpp, compute.cpp.
// Private to the vulkan/ subdirectory.
// =============================================================================
#include <cardinal/rhi/rhi.hpp>

#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/platform.hpp>
#include <cardinal/core/alloc/linear_allocator.hpp>
#include <cardinal/core/alloc/arena_allocator.hpp>

#include "modern.h"

// WSI platform selection — picked up by vulkan headers for the right
// Vk*SurfaceCreateInfoKHR struct.
#if CARDINAL_PLATFORM_WINDOWS
    #define VK_USE_PLATFORM_WIN32_KHR 1
#elif CARDINAL_PLATFORM_LINUX
    #define VK_USE_PLATFORM_XCB_KHR 1
#endif

#include <volk.h>

// VMA (vulkan-memory-allocator). Header-only; the implementation TU is in
// rhi_vulkan_alloc.cpp. Suppress its noisier warnings here too just in case
// any inline functions trigger them under /W4.
#if CARDINAL_COMPILER_MSVC
    #pragma warning(push)
    #pragma warning(disable: 4127 4324 4505)
#endif
#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>
#if CARDINAL_COMPILER_MSVC
    #pragma warning(pop)
#endif

#if CARDINAL_PLATFORM_WINDOWS
    #include <Windows.h>
    #include <unknwn.h>  // IUnknown — not auto-included with WIN32_LEAN_AND_MEAN
#endif

// DXC for runtime HLSL -> SPIR-V. dxcapi.h drags in Windows COM headers, so
// it must come AFTER Windows.h + unknwn.h above.
#include <dxcapi.h>

#include <cardinal/core/std/algorithm.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/core/std/cstring.hpp>
#include <cardinal/core/std/utility.hpp>


namespace cardinal::rhi {

class VulkanDevice;   // fwd (VulkanSwapchain holds a VulkanDevice&)


inline constexpr u32 frames_in_flight = 2;

// Per-thread transient arena used by VulkanSwapchain::end_frame() to back
// the per-frame barrier staging vector (VkImageMemoryBarrier2 batch for
// viewport→SHADER_READ + swapchain UNDEFINED→COLOR_ATTACHMENT). Reset
// at the top of each end_frame() — every push_back goes through a
// single atomic fetch_add. Sized for worst-case 64 viewports + 1
// swapchain barrier × ~100 B per VkImageMemoryBarrier2 = ~7 KB; 16 KB
// for headroom. Each rendering thread that calls end_frame() lazily
// allocates its own arena on first use.
inline constexpr cardinal::usize kVkEndFrameArenaBytes = 16u * 1024u;

inline cardinal::core::LinearAllocator& vk_end_frame_arena() noexcept {
    thread_local cardinal::core::LinearAllocator arena(kVkEndFrameArenaBytes);
    return arena;
}

// Convert a VkResult to its enum-name string. Tiny switch — we only name
// the codes the swapchain / device init actually hands back. Everything
// else falls through to a generic "(VkResult NN)" so logs still carry the
// numeric code for unknown values.
inline const char* vk_result_string(VkResult r) noexcept {
    switch (r) {
        case VK_SUCCESS:                            return "VK_SUCCESS";
        case VK_NOT_READY:                          return "VK_NOT_READY";
        case VK_TIMEOUT:                            return "VK_TIMEOUT";
        case VK_INCOMPLETE:                         return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY:           return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:         return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED:        return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST:                  return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED:            return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT:            return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT:        return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT:          return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER:          return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS:             return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED:         return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL:              return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_OUT_OF_DATE_KHR:              return "VK_ERROR_OUT_OF_DATE_KHR";
        case VK_ERROR_SURFACE_LOST_KHR:             return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:     return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
        case VK_SUBOPTIMAL_KHR:                     return "VK_SUBOPTIMAL_KHR";
        case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
                                                    return "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
        default:                                    return "(unknown)";
    }
}

// Convenience: log + return false on a failed VkResult. Routes through
// cardinal::log so the Studio Log panel mirrors the diagnostic.
inline bool vk_check(VkResult r, const char* what) {
    if (r == VK_SUCCESS) return true;
    cardinal::log::errorf("rhi/vk", "%s failed: %s (%d)",
        what, vk_result_string(r), static_cast<int>(r));
    return false;
}

inline VkFormat to_vk_format(Format f) {
    switch (f) {
        case Format::R32_Float:           return VK_FORMAT_R32_SFLOAT;
        case Format::R32G32_Float:        return VK_FORMAT_R32G32_SFLOAT;
        case Format::R32G32B32_Float:     return VK_FORMAT_R32G32B32_SFLOAT;
        case Format::R32G32B32A32_Float:  return VK_FORMAT_R32G32B32A32_SFLOAT;
        case Format::R8G8B8A8_UNORM:      return VK_FORMAT_R8G8B8A8_UNORM;
        case Format::R8G8B8A8_SRGB:       return VK_FORMAT_R8G8B8A8_SRGB;
        case Format::B8G8R8A8_UNORM:      return VK_FORMAT_B8G8R8A8_UNORM;
        case Format::B8G8R8A8_SRGB:       return VK_FORMAT_B8G8R8A8_SRGB;
        case Format::D32_Float:           return VK_FORMAT_D32_SFLOAT;
        case Format::Unknown:
        default:                          return VK_FORMAT_UNDEFINED;
    }
}

inline Format from_vk_format(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R32_SFLOAT:          return Format::R32_Float;
        case VK_FORMAT_R32G32_SFLOAT:       return Format::R32G32_Float;
        case VK_FORMAT_R32G32B32_SFLOAT:    return Format::R32G32B32_Float;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return Format::R32G32B32A32_Float;
        case VK_FORMAT_R8G8B8A8_UNORM:      return Format::R8G8B8A8_UNORM;
        case VK_FORMAT_R8G8B8A8_SRGB:       return Format::R8G8B8A8_SRGB;
        case VK_FORMAT_B8G8R8A8_UNORM:      return Format::B8G8R8A8_UNORM;
        case VK_FORMAT_B8G8R8A8_SRGB:       return Format::B8G8R8A8_SRGB;
        default:                            return Format::Unknown;
    }
}

inline VkPrimitiveTopology to_vk_topology(PrimitiveTopology t) {
    switch (t) {
        case PrimitiveTopology::TriangleList:  return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        case PrimitiveTopology::TriangleStrip: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        case PrimitiveTopology::LineList:      return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case PrimitiveTopology::LineStrip:     return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        case PrimitiveTopology::PointList:     return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    }
    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

inline VkBufferUsageFlags to_vk_buffer_usage(u32 usage) {
    VkBufferUsageFlags out = 0;
    if (usage & static_cast<u32>(BufferUsage::Vertex))  out |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if (usage & static_cast<u32>(BufferUsage::Index))   out |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (usage & static_cast<u32>(BufferUsage::Uniform)) out |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (usage & static_cast<u32>(BufferUsage::Storage)) out |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (usage & static_cast<u32>(BufferUsage::AccelerationStructureBuild)) {
        out |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    }
    if (usage & static_cast<u32>(BufferUsage::ShaderDeviceAddress)) {
        out |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    // RT extension consumers also need TRANSFER_DST so we can copy into them.
    out |= VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    // Storage buffers may also back GPU-driven indirect dispatch/draw args
    // (AEGIS DrawIndirectGen) — cheap to always allow.
    if (usage & static_cast<u32>(BufferUsage::Storage)) {
        out |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    }
    return out;
}

// =============================================================================
// VulkanSwapchain
// =============================================================================
class VulkanSwapchain final : public Swapchain {
public:
    VulkanSwapchain(VulkanDevice& dev) : dev_(dev) {}
    ~VulkanSwapchain() override;

    bool initialize(void* native_window, u32 width, u32 height);

    u32    width()  const noexcept override { return extent_.width; }
    u32    height() const noexcept override { return extent_.height; }
    Format color_format() const noexcept override { return from_vk_format(format_); }

    // Multi-viewport interface — fully implemented on Vulkan now (was a
    // degraded "alias-to-0" path before). Each viewport id owns its own
    // image / view / VMA allocation / depth (when added). begin_frame
    // doesn't open any render pass; the host calls set_active_viewport(id)
    // once per visible panel, which closes the previous panel's dynamic
    // render and opens one on viewport[id]'s view. end_frame transitions
    // every rendered viewport to SHADER_READ_ONLY_OPTIMAL so ImGui can
    // sample them inside the overlay's render pass.
    void set_viewport_size(u32 w, u32 h) override {
        set_viewport_size(0u, w, h);
    }
    u32  viewport_width()  const noexcept override { return viewport_width(0u);  }
    u32  viewport_height() const noexcept override { return viewport_height(0u); }
    void set_viewport_size(u32 id, u32 w, u32 h) override {
        if (id >= viewport_count_) set_viewport_count(id + 1u);
        viewports_[id].pending = { w, h };
    }
    u32  viewport_width (u32 id) const noexcept override {
        return id < viewport_count_ ? viewports_[id].extent.width  : 0u;
    }
    u32  viewport_height(u32 id) const noexcept override {
        return id < viewport_count_ ? viewports_[id].extent.height : 0u;
    }
    void set_viewport_count(u32 n) override;
    u32  viewport_count() const noexcept override { return viewport_count_; }
    void set_active_viewport(u32 id) override;
    u32  active_viewport() const noexcept override { return active_viewport_id_; }

    // Vulkan path: each per-viewport off-screen RTT carries a matching
    // VK_FORMAT_D32_SFLOAT depth attachment. Returning a real format
    // here flips the scene renderer's painter's-algorithm CPU sort OFF
    // (it only fires when the swapchain has no depth — see
    // ForwardRendererImpl::render). For terrain scenes this is the
    // single biggest CPU win on Vulkan: sort cost was O(N log N) over
    // every visible triangle, which dominated once a 64×64 chunk
    // (~7400 tris) showed up in the scene.
    Format depth_format() const noexcept override { return Format::D32_Float; }
    void   set_vsync(bool on) override         { request_vsync_interval(on ? 1u : 0u); }
    bool   vsync() const noexcept override     { return vsync_interval_ > 0u; }
    void   set_vsync_interval(u32 i) override  { request_vsync_interval(i); }
    u32    vsync_interval() const noexcept override { return vsync_interval_; }
    // Reflex on Vulkan needs VK_NV_low_latency2 + NvLL_VK init — slated
    // for the same Phase 5.5 work as DLSS dispatch. The base class
    // defaults already report Off / unsupported / no-op, which is what
    // we want until that lands.

    u32  begin_frame(float r, float g, float b, float a) override;
    void end_frame() override;
    bool resize(u32 new_w, u32 new_h) override;

    void set_on_rebuilt(OnRebuilt cb) override { on_rebuilt_ = cardinal::move(cb); }

    void bind_pipeline(Pipeline* p) override;
    void bind_vertex_buffer(Buffer* b, usize offset = 0) override;
    void draw(u32 vertex_count, u32 instance_count, u32 first_vertex, u32 first_instance) override;
    void set_push_constants(u32 offset, const void* data, u32 size) override;
    void bind_storage_buffer(u32 slot, Buffer* b) override;
    void bind_storage_buffer_uav(u32 slot, Buffer* b) override;   // compute RW UAV
    void dispatch(u32 gx, u32 gy = 1, u32 gz = 1) override;       // compute dispatch
    void dispatch_indirect(Buffer* args, u32 args_offset) override;  // GPU-driven
    void uav_barrier(Buffer* b) override;                        // RAW/WAW flush
    void transition_buffer_state(Buffer* b, ResourceState before,
                                 ResourceState after) override;
    void begin_shadow_pass(Texture* depth) override;
    void end_shadow_pass() override;
    void bind_sampled_texture(u32 slot, Texture* tex) override;

    // Multi-queue sync (AEGIS Block 10): signal/wait a timeline-semaphore Fence
    // on the GRAPHICS queue — the graphics<->async-compute handshake. Defined
    // after the VulkanFence class. Empty submits used purely for the timeline op.
    u64  signal_fence(Fence* f) override;
    void wait_fence  (Fence* f, u64 value) override;

    void set_overlay(OverlayCallback cb, void* user_data) override {
        overlay_cb_ = cb; overlay_user_ = user_data;
    }

    // Interop accessors (see VulkanDevice equivalents).
    VkSwapchainKHR  vk_swapchain()       const noexcept { return swapchain_; }
    VkFormat        vk_color_format()    const noexcept { return format_; }
    u32             vk_image_count()     const noexcept { return static_cast<u32>(images_.size()); }
    u32             vk_min_image_count() const noexcept { return min_image_count_; }
    VkCommandBuffer vk_current_cmd()     const noexcept { return frames_[frame_index_].cmd; }
    u32             vk_acquired_image_index() const noexcept { return acquired_image_; }
    VkImage         vk_image(u32 i)      const noexcept { return i < images_.size() ? images_[i] : VK_NULL_HANDLE; }
    VkImageView     vk_view (u32 i)      const noexcept { return i < views_.size()  ? views_[i]  : VK_NULL_HANDLE; }
    VkFormat        vk_viewport_format() const noexcept { return format_; }
    VkImage         vk_viewport_image()       const noexcept { return vk_viewport_image(0u); }
    VkImageView     vk_viewport_view()        const noexcept { return vk_viewport_view (0u); }
    VkImage         vk_viewport_image(u32 id) const noexcept {
        return id < viewport_count_ ? viewports_[id].image : VK_NULL_HANDLE;
    }
    VkImageView     vk_viewport_view (u32 id) const noexcept {
        return id < viewport_count_ ? viewports_[id].view  : VK_NULL_HANDLE;
    }
    u32             vk_viewport_count()       const noexcept { return viewport_count_; }

    // Called once per begin_frame to commit any pending vsync change.
    // We can't tear the swapchain down inline with set_vsync_interval()
    // because there are GPU submissions in flight; instead set a flag
    // that the next frame consults at a safe point.
    void apply_pending_vsync_change();

private:
    bool create_surface(void* native_window);
    bool create_swapchain_objects(u32 width, u32 height);
    bool create_per_frame_objects();
    void destroy_swapchain_objects();
    void destroy_per_frame_objects();
    void request_vsync_interval(u32 interval) noexcept {
        // Vulkan WSI's presentMode is one of IMMEDIATE / MAILBOX / FIFO /
        // FIFO_RELAXED — there's no "1:N vsync" knob like DXGI's
        // SyncInterval. Anything > 1 has no native equivalent; the
        // closest user-visible behaviour is FIFO + a half-rate FPS cap
        // applied via cardinal::core::FramePacer (which the engine
        // already exposes). We clamp here so we don't lie about what
        // the swapchain is doing. Warn ONCE so the log isn't spammed
        // when a UI slider is dragged.
        if (interval > 1u) {
            if (!warned_vsync_interval_clamped_) {
                cardinal::log::warnf("rhi/vk",
                    "vsync_interval=%u not supported by Vulkan WSI — "
                    "clamping to 1. For half/third rate use the engine's "
                    "FramePacer FPS cap instead.", interval);
                warned_vsync_interval_clamped_ = true;
            }
            interval = 1u;
        }
        if (interval == vsync_interval_) return;
        vsync_interval_      = interval;
        vsync_change_pending_ = true;     // committed at top of begin_frame
    }
    static const char* present_mode_name(VkPresentModeKHR m) noexcept {
        switch (m) {
            case VK_PRESENT_MODE_IMMEDIATE_KHR:    return "IMMEDIATE (uncapped)";
            case VK_PRESENT_MODE_MAILBOX_KHR:      return "MAILBOX (capped)";
            case VK_PRESENT_MODE_FIFO_KHR:         return "FIFO (vsync)";
            case VK_PRESENT_MODE_FIFO_RELAXED_KHR: return "FIFO_RELAXED";
            default:                               return "?";
        }
    }

    // Per-id viewport (off-screen render target) lifecycle. Each id holds
    // its own image + view + VMA allocation. ensure_viewport_image grows
    // / resizes a single slot; destroy_viewport_image releases one;
    // destroy_all_viewport_images is the shutdown sweep.
    bool ensure_viewport_image(u32 id, u32 w, u32 h);
    void destroy_viewport_image(u32 id);
    void destroy_all_viewport_images();

    VulkanDevice& dev_;

    VkSurfaceKHR    surface_{VK_NULL_HANDLE};
    VkSwapchainKHR  swapchain_{VK_NULL_HANDLE};
    VkFormat        format_{VK_FORMAT_UNDEFINED};
    VkColorSpaceKHR color_space_{VK_COLORSPACE_SRGB_NONLINEAR_KHR};
    VkExtent2D      extent_{};
    VkPresentModeKHR present_mode_{VK_PRESENT_MODE_FIFO_KHR};
    u32             min_image_count_{0};

    cardinal::vector<VkImage>     images_;
    cardinal::vector<VkImageView> views_;

    struct FrameSync {
        VkSemaphore     image_available{VK_NULL_HANDLE};
        VkSemaphore     render_finished{VK_NULL_HANDLE};
        VkFence         in_flight{VK_NULL_HANDLE};
        VkCommandPool   pool{VK_NULL_HANDLE};
        VkCommandBuffer cmd{VK_NULL_HANDLE};
        // Transient descriptor pool for this frame's storage-buffer
        // binds. Reset wholesale at begin_frame (after the in_flight
        // fence wait guarantees the GPU is done with frame-N-k's sets)
        // so per-frame allocations recycle without per-set frees. Grows
        // on demand: a multi-viewport editor blows past any fixed set
        // count, so a fresh block is appended whenever the frame's draws
        // exhaust the current pools (canonical transient-pool-chain).
        // Empty until the first bind needs it (push-only apps pay zero).
        cardinal::vector<VkDescriptorPool> desc_pools;
        u32                                desc_pool_cur{0};
    };
    cardinal::array<FrameSync, frames_in_flight> frames_{};
    u32 frame_index_{0};
    u32 acquired_image_{0};

    // Tracks the most recently bound pipeline so set_push_constants()
    // can find the layout + declared push-constant size without the
    // caller having to plumb the pipeline pointer through every call.
    // Stored as base Pipeline* to avoid forward-decl ordering with the
    // full VulkanPipeline class definition (which lives ~900 lines
    // below); set_push_constants does a static_cast on demand.
    Pipeline* bound_pipeline_{nullptr};

    // Pending storage-buffer bindings for the currently-bound pipeline.
    // bind_storage_buffer accumulates here, then (re)writes a single
    // descriptor set covering every populated slot and binds it — so a
    // 2-slot pipeline (lights @0 + materials @1) gets ONE set with both
    // bindings, not two sets that clobber each other. Cleared on
    // bind_pipeline (a new pipeline starts with nothing bound).
    static constexpr u32 kMaxStorageSlots = 8;
    Buffer*  pending_sb_[kMaxStorageSlots]{};
    Texture* pending_st_[kMaxStorageSlots]{};   // sampled textures (graphics)
    Buffer*  pending_uav_[kMaxStorageSlots]{};  // read-write UAV buffers (compute)
    void rebuild_and_bind_descriptor_set_();

    // Active shadow-pass depth target between begin/end_shadow_pass.
    // end transitions it DEPTH_ATTACHMENT → SHADER_READ_ONLY so the
    // main pass can sample it. nullptr when no shadow pass is open.
    Texture* shadow_tex_{nullptr};
    // True when begin_shadow_pass suspended an open viewport scope;
    // end_shadow_pass resumes it via set_active_viewport.
    bool     shadow_suspended_{false};

    // Per-viewport off-screen RTT slot. Each slot owns an image / view /
    // VMA allocation; the host calls set_active_viewport(id) between
    // begin_frame and end_frame to direct subsequent draws into that
    // slot. `rendered_this_frame` lets end_frame's overlay-prep pass know
    // which slots actually need a SHADER_READ_ONLY transition (panels
    // that weren't drawn this frame stay in whatever layout they had).
    struct ViewportSlot {
        VkImage         image{VK_NULL_HANDLE};
        VkImageView     view{VK_NULL_HANDLE};
        VmaAllocation   alloc{VK_NULL_HANDLE};
        VkExtent2D      extent{0, 0};
        VkExtent2D      pending{0, 0};
        // Layout we left this image in at the END of last set_active_viewport
        // (or end_frame). Used by the next render's barrier so we don't
        // assume UNDEFINED on every transition (which discards content).
        VkImageLayout   current_layout{VK_IMAGE_LAYOUT_UNDEFINED};
        bool            rendered_this_frame{false};

        // Depth attachment matching the colour image. Cleared on every
        // begin_rendering — there's no reason to preserve depth across
        // frames in the editor (each frame redraws the whole scene).
        // Layout starts UNDEFINED, transitions to DEPTH_ATTACHMENT_OPTIMAL
        // before each render, and we leave it there (no later sample of
        // the depth — overlay reads colour only).
        VkImage         depth_image{VK_NULL_HANDLE};
        VkImageView     depth_view{VK_NULL_HANDLE};
        VmaAllocation   depth_alloc{VK_NULL_HANDLE};
        VkImageLayout   depth_layout{VK_IMAGE_LAYOUT_UNDEFINED};
    };
    cardinal::vector<ViewportSlot> viewports_{};
    u32                       viewport_count_{0u};
    u32                       active_viewport_id_{0u};
    // Whether vkCmdBeginRendering is currently open on a viewport (so
    // end_frame / next set_active_viewport knows to close it first).
    bool                      rendering_open_{false};
    float                     last_clear_color_[4]{0, 0, 0, 1};

    OverlayCallback overlay_cb_{nullptr};
    void*           overlay_user_{nullptr};

    // VSync interval. Maps to a VkPresentModeKHR at swapchain creation
    // time (see create_swapchain_objects). Changing this at runtime sets
    // vsync_change_pending_; apply_pending_vsync_change() rebuilds the
    // swapchain at the start of the next frame.
    u32             vsync_interval_{0};
    bool            vsync_change_pending_{false};
    bool            warned_vsync_interval_clamped_{false};

    // Fires after any rebuild of the swapchain images (resize, vsync
    // change, future hot-swap). Consumers re-create their per-image state.
    OnRebuilt       on_rebuilt_{};

    // Cached probe results — refreshed on every create_swapchain_objects.
    // Used by Settings UI / diagnostics; "yes/no" reported in the boot log.
    bool            immediate_supported_{false};
    bool            mailbox_supported_{false};
    bool            fifo_relaxed_supported_{false};
};
// =============================================================================
// VulkanDevice
// =============================================================================
class VulkanDevice final : public Device {
public:
    Backend backend() const noexcept override { return Backend::Vulkan; }
    const char* adapter_name() const noexcept override { return adapter_name_; }

    const GpuCapabilities& capabilities() const noexcept override { return caps_; }
    const RenderSettings&  settings()     const noexcept override { return settings_; }

    void apply_settings(const RenderSettings& s) override {
        // Clamp every setting against capabilities — log when a request is
        // not satisfiable. The engine continues with the clamped values.
        settings_ = s;
        auto clamp = [&](bool& field, bool cap, const char* what) {
            if (field && !cap) {
                cardinal::log::warnf("rhi/vk",
                    "settings: %s requested but unsupported on this device — disabled",
                    what);
                field = false;
            }
        };
        clamp(settings_.enable_ray_tracing,     caps_.ray_tracing_pipeline,     "ray tracing");
        clamp(settings_.enable_ray_query,       caps_.ray_query,                "ray query");
        clamp(settings_.enable_path_tracing,    caps_.ray_tracing_pipeline,     "path tracing");
        clamp(settings_.enable_mesh_shaders,    caps_.mesh_shader,              "mesh shaders");
        clamp(settings_.prefer_fp16,            caps_.shader_float16,           "FP16 / RPM");
        clamp(settings_.prefer_int8,            caps_.shader_int8,              "INT8 / RPM");

        // Frame generation requires DLSS 3 (Ada+) OR FSR 3.
        if (settings_.enable_frame_generation &&
            !caps_.nvidia_framegen_capable && !caps_.amd_fsr3_capable) {
            cardinal::log::warnf("rhi/vk",
                "settings: frame generation requested but no capable path — disabled");
            settings_.enable_frame_generation = false;
        }

        // Upscaler validity check.
        switch (settings_.upscaler) {
            case UpscalerMode::DLSS_Quality:
            case UpscalerMode::DLSS_Balanced:
            case UpscalerMode::DLSS_Performance:
            case UpscalerMode::DLSS_UltraPerformance:
            case UpscalerMode::DLAA:
                if (!caps_.nvidia_dlss_capable) {
                    cardinal::log::warnf("rhi/vk",
                        "settings: DLSS/DLAA requested but no NGX-capable GPU — Off");
                    settings_.upscaler = UpscalerMode::Off;
                }
                break;
            case UpscalerMode::FSR3_Quality:
            case UpscalerMode::FSR3_Balanced:
            case UpscalerMode::FSR3_Performance:
                if (!caps_.amd_fsr3_capable) {
                    // FSR can run on any GPU technically, but the FSR3 frame-
                    // gen path expects a capable backend. Allow on AMD only.
                    cardinal::log::warnf("rhi/vk",
                        "settings: FSR3 selected on a non-AMD path — Off");
                    settings_.upscaler = UpscalerMode::Off;
                }
                break;
            case UpscalerMode::TAA:
            case UpscalerMode::Off:
                break;
        }
    }

    ~VulkanDevice() override {
        // Drain every queue BEFORE we let VMA tear down. Without this,
        // any in-flight GPU command that references a VMA allocation
        // makes vmaDestroyAllocator's debug-leak assertion fire (the
        // allocation is "in use" from VMA's POV until the GPU finishes).
        // That assertion is the "Debug Assertion Failed" dialog the
        // user sees when closing the Vulkan-backed window — it doesn't
        // happen on D3D12 because D3D12 doesn't run a CPU-side leak
        // tracker like VMA does.
        if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);

        if (dxc_compiler_ != nullptr) dxc_compiler_->Release();
        if (dxc_utils_    != nullptr) dxc_utils_->Release();
        if (default_sampler_ != VK_NULL_HANDLE)
            vkDestroySampler(device_, default_sampler_, nullptr);
        if (allocator_ != VK_NULL_HANDLE) vmaDestroyAllocator(allocator_);
        if (device_    != VK_NULL_HANDLE) vkDestroyDevice(device_, nullptr);
        if (instance_  != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
        modern::shutdown_reflex();
        modern::shutdown_streamline();
    }

    bool initialize(const DeviceDesc& desc);

    cardinal::unique_ptr<Swapchain> create_swapchain(
        void* native_window, u32 w, u32 h) override
    {
        auto sw = cardinal::make_unique<VulkanSwapchain>(*this);
        if (!sw->initialize(native_window, w, h)) return nullptr;
        return sw;
    }

    cardinal::unique_ptr<Buffer>   create_buffer(const BufferDesc& desc) override;
    cardinal::unique_ptr<Pipeline> create_pipeline(const PipelineDesc& desc) override;
    cardinal::unique_ptr<Pipeline> create_compute_pipeline(const ComputePipelineDesc& desc) override;
    cardinal::unique_ptr<Texture>  create_texture(const TextureDesc& desc) override;
    cardinal::unique_ptr<AccelerationStructure> create_blas(const BlasDesc& desc) override;
    cardinal::unique_ptr<AccelerationStructure> create_tlas(const TlasDesc& desc) override;

    // Async compute (AEGIS Block 10). create_fence makes a timeline-semaphore
    // Fence; create_async_compute_queue returns a VulkanComputeQueue bound to
    // the dedicated compute family (nullptr when none / caps.async_compute is
    // false). Defined below, after the VulkanFence/VulkanComputeQueue classes.
    cardinal::unique_ptr<Fence>        create_fence(u64 initial_value = 0) override;
    cardinal::unique_ptr<ComputeQueue> create_async_compute_queue() override;

    ShaderBlob compile_shader(
        ShaderStage stage,
        const char* hlsl_source,
        const char* entry_point) override;

    VramSnapshot query_vram_usage() const noexcept override;

    // Raw handle accessors — used by sibling RHI types AND the interop
    // shim in vulkan_interop.hpp. Names are short because they're only
    // visible inside this TU (anonymous namespace).
    VkInstance       vk_instance()             const noexcept { return instance_; }
    VkPhysicalDevice vk_physical()             const noexcept { return physical_; }
    VkDevice         vk_device()               const noexcept { return device_; }
    VkQueue          vk_graphics_queue()       const noexcept { return graphics_queue_; }
    u32              vk_graphics_queue_family() const noexcept { return graphics_queue_family_; }
    VkQueue          vk_compute_queue()        const noexcept { return compute_queue_; }
    u32              vk_compute_queue_family() const noexcept { return compute_queue_family_; }
    VmaAllocator     vk_allocator()            const noexcept { return allocator_; }

    // Friends so the swapchain / buffer / pipeline can reach the handles cleanly.
    friend class VulkanSwapchain;
    friend class VulkanBuffer;
    friend class VulkanTexture;
    friend class VulkanPipeline;
    friend class VulkanAccelerationStructure;

private:
    bool init_dxc();
    bool init_vma();

    VkInstance       instance_{VK_NULL_HANDLE};
    VkPhysicalDevice physical_{VK_NULL_HANDLE};
    VkDevice         device_{VK_NULL_HANDLE};
    VkQueue          graphics_queue_{VK_NULL_HANDLE};
    u32              graphics_queue_family_{static_cast<u32>(-1)};
    VkQueue          compute_queue_{VK_NULL_HANDLE};            // dedicated async-compute lane
    u32              compute_queue_family_{static_cast<u32>(-1)};
    char             adapter_name_[256]{};

    VmaAllocator     allocator_{VK_NULL_HANDLE};
    // Shared sampler for all sampled-texture bindings (shadow map +
    // future RTs). Clamp-to-edge + linear — fine for a depth shadow
    // lookup; per-texture samplers can come later if a use-case needs
    // wrap/anisotropy. Created post-device, destroyed before it.
    VkSampler        default_sampler_{VK_NULL_HANDLE};

    // DXC runtime compiler (lazy-init).
    IDxcCompiler3* dxc_compiler_{nullptr};
    IDxcUtils*     dxc_utils_{nullptr};

    GpuCapabilities caps_{};
    RenderSettings     settings_{};

    // True iff VK_EXT_memory_budget was enabled at device creation; gates the
    // pNext chain in query_vram_usage().
    bool             has_memory_budget_{false};
    // True iff VK_NV_low_latency2 was enabled. Gates NvLL_VK Reflex calls
    // (NvLL_VK_InitLowLatencyDevice / LatencySleep / SetLatencyMarker)
    // — without the extension those calls return an error on the driver
    // side and the swapchain falls back to no-op reflex.
    bool             has_nv_low_latency2_{false};
};
// =============================================================================
// VulkanBuffer
// =============================================================================
class VulkanBuffer final : public Buffer {
public:
    VulkanBuffer(VulkanDevice& dev) : dev_(dev) {}

    ~VulkanBuffer() override {
        if (alloc_ != VK_NULL_HANDLE) {
            vmaDestroyBuffer(dev_.allocator_, buffer_, alloc_);
        }
    }

    bool initialize(const BufferDesc& desc) {
        size_ = desc.size;

        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = desc.size;
        bci.usage       = to_vk_buffer_usage(desc.usage);
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        if (desc.cpu_writable) {
            aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }

        VmaAllocationInfo info{};
        if (!vk_check(
                vmaCreateBuffer(dev_.allocator_, &bci, &aci, &buffer_, &alloc_, &info),
                "vmaCreateBuffer")) {
            return false;
        }
        mapped_ptr_ = info.pMappedData;
        cpu_writable_ = desc.cpu_writable;

        // Cache device address if requested.
        if ((desc.usage & static_cast<u32>(BufferUsage::ShaderDeviceAddress)) ||
            (desc.usage & static_cast<u32>(BufferUsage::AccelerationStructureBuild))) {
            VkBufferDeviceAddressInfo bdai{};
            bdai.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            bdai.buffer = buffer_;
            device_address_ = vkGetBufferDeviceAddress(dev_.device_, &bdai);
        }
        return true;
    }

    usize size() const noexcept override { return size_; }
    u64   device_address() const noexcept override { return device_address_; }

    void upload(const void* data, usize size, usize offset) override {
        if (!cpu_writable_) {
            cardinal::log::errorf("rhi/vk", "upload() to non-cpu_writable buffer");
            return;
        }
        if (offset + size > size_) {
            cardinal::log::errorf("rhi/vk",
                "upload out of range (offset=%zu size=%zu buffer=%zu)",
                offset, size, size_);
            return;
        }
        if (mapped_ptr_ != nullptr) {
            cardinal::memcpy(static_cast<u8*>(mapped_ptr_) + offset, data, size);
        } else {
            void* p = nullptr;
            vmaMapMemory(dev_.allocator_, alloc_, &p);
            cardinal::memcpy(static_cast<u8*>(p) + offset, data, size);
            vmaUnmapMemory(dev_.allocator_, alloc_);
        }
    }

    VkBuffer handle() const noexcept { return buffer_; }

private:
    VulkanDevice& dev_;
    VkBuffer       buffer_{VK_NULL_HANDLE};
    VmaAllocation  alloc_{VK_NULL_HANDLE};
    void*          mapped_ptr_{nullptr};
    usize          size_{0};
    bool           cpu_writable_{false};
    u64            device_address_{0};
};
struct OneShotCmd {
    VulkanDevice&   dev;
    VkCommandPool   pool{VK_NULL_HANDLE};
    VkCommandBuffer cmd{VK_NULL_HANDLE};

    explicit OneShotCmd(VulkanDevice& d) : dev(d) {
        VkCommandPoolCreateInfo pci{};
        pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pci.queueFamilyIndex = dev.vk_graphics_queue_family();
        vkCreateCommandPool(dev.vk_device(), &pci, nullptr, &pool);

        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = pool;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        vkAllocateCommandBuffers(dev.vk_device(), &ai, &cmd);

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
    }

    void submit_and_wait() {
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmd;
        vkQueueSubmit(dev.vk_graphics_queue(), 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(dev.vk_graphics_queue());
    }

    ~OneShotCmd() {
        if (pool != VK_NULL_HANDLE) vkDestroyCommandPool(dev.vk_device(), pool, nullptr);
    }
};
// =============================================================================
// VulkanTexture — GPU 2D image. First consumer: the shadow-map depth
// render target (DepthRenderTarget | Sampled). `layout_` is mutable
// state the swapchain advances as it transitions the image between the
// depth-write pass and the sampled main pass.
// =============================================================================
class VulkanTexture final : public Texture {
public:
    explicit VulkanTexture(VulkanDevice& dev) : dev_(dev) {}
    ~VulkanTexture() override {
        if (view_  != VK_NULL_HANDLE) vkDestroyImageView(dev_.device_, view_, nullptr);
        if (image_ != VK_NULL_HANDLE) vmaDestroyImage(dev_.allocator_, image_, alloc_);
    }

    bool initialize(const TextureDesc& d) {
        w_ = d.width; h_ = d.height; fmt_ = d.format;
        if (w_ == 0 || h_ == 0) return false;
        depth_ = (d.format == Format::D32_Float);

        VkImageUsageFlags usage = 0;
        if (d.usage & static_cast<u32>(TextureUsage::DepthRenderTarget))
            usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        if (d.usage & static_cast<u32>(TextureUsage::ColorRenderTarget))
            usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (d.usage & static_cast<u32>(TextureUsage::Sampled))
            usage |= VK_IMAGE_USAGE_SAMPLED_BIT;

        VkImageCreateInfo ic{};
        ic.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ic.imageType     = VK_IMAGE_TYPE_2D;
        ic.format        = to_vk_format(d.format);
        ic.extent        = { w_, h_, 1 };
        ic.mipLevels     = 1;
        ic.arrayLayers   = 1;
        ic.samples       = VK_SAMPLE_COUNT_1_BIT;
        ic.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ic.usage         = usage;
        ic.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ic.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;             // GPU-local
        if (!vk_check(
                vmaCreateImage(dev_.allocator_, &ic, &aci,
                               &image_, &alloc_, nullptr),
                "vmaCreateImage (texture)")) {
            return false;
        }

        VkImageViewCreateInfo iv{};
        iv.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        iv.image                       = image_;
        iv.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
        iv.format                      = to_vk_format(d.format);
        iv.subresourceRange.aspectMask = depth_
            ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        iv.subresourceRange.levelCount = 1;
        iv.subresourceRange.layerCount = 1;
        if (!vk_check(
                vkCreateImageView(dev_.device_, &iv, nullptr, &view_),
                "vkCreateImageView (texture)")) {
            return false;
        }
        // A Sampled texture must not be left in UNDEFINED: the pipeline
        // statically declares its sampler binding, so the main pass binds
        // + samples it (descriptor imageLayout = SHADER_READ_ONLY_OPTIMAL)
        // EVERY frame — even when the producing pass is skipped (the
        // shadow map in wireframe / no-directional-light / shadows-off /
        // first frames). Sampling an image whose real layout (UNDEFINED)
        // doesn't match the descriptor is undefined behaviour + a layout
        // validation error. Put it in the sampleable layout up front; the
        // shadow pass legitimately cycles it SHADER_READ_ONLY → DEPTH_
        // ATTACHMENT → SHADER_READ_ONLY when it does run.
        if (d.usage & static_cast<u32>(TextureUsage::Sampled)) {
            OneShotCmd one(dev_);
            VkImageMemoryBarrier b{};
            b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image               = image_;
            b.subresourceRange.aspectMask     = depth_
                ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
            b.subresourceRange.baseMipLevel   = 0;
            b.subresourceRange.levelCount     = 1;
            b.subresourceRange.baseArrayLayer = 0;
            b.subresourceRange.layerCount     = 1;
            b.srcAccessMask       = 0;
            b.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(one.cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &b);
            one.submit_and_wait();
            layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        } else {
            layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        }
        return true;
    }

    // Texture interface.
    u32    width()  const noexcept override { return w_; }
    u32    height() const noexcept override { return h_; }
    Format format() const noexcept override { return fmt_; }

    // Swapchain-facing handles + mutable layout tracker.
    VkImage       image()    const noexcept { return image_; }
    VkImageView   view()     const noexcept { return view_; }
    VkExtent2D    extent()   const noexcept { return { w_, h_ }; }
    bool          is_depth() const noexcept { return depth_; }
    VkImageLayout layout_{VK_IMAGE_LAYOUT_UNDEFINED};

private:
    VulkanDevice&  dev_;
    VkImage        image_{VK_NULL_HANDLE};
    VkImageView    view_{VK_NULL_HANDLE};
    VmaAllocation  alloc_{VK_NULL_HANDLE};
    u32            w_{0}, h_{0};
    Format         fmt_{Format::D32_Float};
    bool           depth_{true};
};
// =============================================================================
// VulkanPipeline
// =============================================================================
class VulkanPipeline final : public Pipeline {
public:
    VulkanPipeline(VulkanDevice& dev) : dev_(dev) {}

    ~VulkanPipeline() override {
        if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(dev_.device_, pipeline_, nullptr);
        if (layout_   != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev_.device_, layout_, nullptr);
        if (dsl_      != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev_.device_, dsl_, nullptr);
        if (vs_module_ != VK_NULL_HANDLE) vkDestroyShaderModule(dev_.device_, vs_module_, nullptr);
        if (fs_module_ != VK_NULL_HANDLE) vkDestroyShaderModule(dev_.device_, fs_module_, nullptr);
        if (cs_module_ != VK_NULL_HANDLE) vkDestroyShaderModule(dev_.device_, cs_module_, nullptr);
    }

    bool initialize(const PipelineDesc& desc);
    // Compute variant (AEGIS graph::RhiBackend). One descriptor set: storage
    // bindings [0,storage) read-only + [storage, storage+uav) read-write (Vulkan
    // has no separate UAV type — both are STORAGE_BUFFER), stage = COMPUTE.
    bool initialize_compute(const ComputePipelineDesc& desc);

    bool                  is_compute()          const noexcept override { return is_compute_; }
    VkPipeline            handle()              const noexcept { return pipeline_; }
    VkPipelineLayout      layout()              const noexcept { return layout_; }
    u32                   push_constant_size()  const noexcept { return push_constant_size_; }
    // VK_NULL_HANDLE when storage_buffer_slots_ == 0 (push-only pipeline).
    VkDescriptorSetLayout descriptor_set_layout() const noexcept { return dsl_; }
    u32                   storage_buffer_slots() const noexcept { return storage_buffer_slots_; }
    u32                   uav_slots()            const noexcept { return uav_slots_; }

private:
    VulkanDevice&         dev_;
    VkPipelineLayout      layout_{VK_NULL_HANDLE};
    VkPipeline            pipeline_{VK_NULL_HANDLE};
    VkDescriptorSetLayout dsl_{VK_NULL_HANDLE};
    VkShaderModule        vs_module_{VK_NULL_HANDLE};
    VkShaderModule        fs_module_{VK_NULL_HANDLE};
    VkShaderModule        cs_module_{VK_NULL_HANDLE};
    u32                   push_constant_size_{0};
    u32                   storage_buffer_slots_{0};
    u32                   sampled_texture_slots_{0};
    u32                   uav_slots_{0};
    bool                  is_compute_{false};
public:
    u32 sampled_texture_slots() const noexcept { return sampled_texture_slots_; }
};
// =============================================================================
// VulkanAccelerationStructure (Phase 5 — BLAS only; TLAS lands next session)
//
// Build flow:
//   1) Translate BlasDesc to VkAccelerationStructureGeometryKHR + range info
//   2) vkGetAccelerationStructureBuildSizesKHR -> result + scratch sizes
//   3) Allocate result buffer (AS_STORAGE + DEVICE_ADDRESS) and scratch
//      (STORAGE + DEVICE_ADDRESS) via VMA
//   4) vkCreateAccelerationStructureKHR on the result buffer
//   5) Allocate one-shot command buffer, vkCmdBuildAccelerationStructuresKHR,
//      submit + vkQueueWaitIdle
//   6) vkGetAccelerationStructureDeviceAddressKHR -> cache device address
//
// RTXMU pooling lands in Phase 5.1 — for now each BLAS owns its own buffers.
// =============================================================================
class VulkanAccelerationStructure final : public AccelerationStructure {
public:
    VulkanAccelerationStructure(VulkanDevice& dev) : dev_(dev) {}

    ~VulkanAccelerationStructure() override {
        if (as_handle_ != VK_NULL_HANDLE) {
            vkDestroyAccelerationStructureKHR(dev_.device_, as_handle_, nullptr);
        }
        if (result_buffer_ != VK_NULL_HANDLE) {
            vmaDestroyBuffer(dev_.allocator_, result_buffer_, result_alloc_);
        }
    }

    bool initialize_blas(const BlasDesc& desc);

    u64 device_address() const noexcept override { return device_address_; }

private:
    VulkanDevice&             dev_;
    VkAccelerationStructureKHR as_handle_{VK_NULL_HANDLE};
    VkBuffer                  result_buffer_{VK_NULL_HANDLE};
    VmaAllocation             result_alloc_{VK_NULL_HANDLE};
    u64                       device_address_{0};
};

}  // namespace cardinal::rhi
