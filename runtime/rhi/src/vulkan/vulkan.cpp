#include <cardinal/rhi/rhi.hpp>

#include <cardinal/core/log.hpp>
#include <cardinal/core/platform.hpp>
#include <cardinal/core/linear_allocator.hpp>
#include <cardinal/core/arena_allocator.hpp>

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

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/containers.hpp>
#include <cardinal/core/cstring.hpp>
#include <cardinal/core/utility.hpp>

namespace cardinal::rhi {

// =============================================================================
// Forward / shared
// =============================================================================
class VulkanDevice;

namespace {

constexpr u32 frames_in_flight = 2;

// Per-thread transient arena used by VulkanSwapchain::end_frame() to back
// the per-frame barrier staging vector (VkImageMemoryBarrier2 batch for
// viewport→SHADER_READ + swapchain UNDEFINED→COLOR_ATTACHMENT). Reset
// at the top of each end_frame() — every push_back goes through a
// single atomic fetch_add. Sized for worst-case 64 viewports + 1
// swapchain barrier × ~100 B per VkImageMemoryBarrier2 = ~7 KB; 16 KB
// for headroom. Each rendering thread that calls end_frame() lazily
// allocates its own arena on first use.
constexpr cardinal::usize kVkEndFrameArenaBytes = 16u * 1024u;

cardinal::core::LinearAllocator& vk_end_frame_arena() noexcept {
    thread_local cardinal::core::LinearAllocator arena(kVkEndFrameArenaBytes);
    return arena;
}

// Convert a VkResult to its enum-name string. Tiny switch — we only name
// the codes the swapchain / device init actually hands back. Everything
// else falls through to a generic "(VkResult NN)" so logs still carry the
// numeric code for unknown values.
const char* vk_result_string(VkResult r) noexcept {
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
bool vk_check(VkResult r, const char* what) {
    if (r == VK_SUCCESS) return true;
    cardinal::log::errorf("rhi/vk", "%s failed: %s (%d)",
        what, vk_result_string(r), static_cast<int>(r));
    return false;
}

VkFormat to_vk_format(Format f) {
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

Format from_vk_format(VkFormat f) {
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

VkPrimitiveTopology to_vk_topology(PrimitiveTopology t) {
    switch (t) {
        case PrimitiveTopology::TriangleList:  return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        case PrimitiveTopology::TriangleStrip: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        case PrimitiveTopology::LineList:      return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case PrimitiveTopology::LineStrip:     return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        case PrimitiveTopology::PointList:     return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    }
    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

VkBufferUsageFlags to_vk_buffer_usage(u32 usage) {
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
    return out;
}

}  // namespace

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
    Texture* pending_st_[kMaxStorageSlots]{};   // sampled textures
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

bool VulkanDevice::initialize(const DeviceDesc& desc) {
    // Modern SDKs first — Streamline wants to be initialised before any
    // Vulkan resources exist (it doesn't intercept in our manual-hooking
    // mode, but it caches global state at init).
    modern::log_linked_sdks();
    const bool sl_ok      = modern::init_streamline();
    const bool reflex_ok  = modern::init_reflex();
    (void)sl_ok;
    (void)reflex_ok;

    if (volkInitialize() != VK_SUCCESS) {
        cardinal::log::errorf("rhi/vk", "vulkan-1 loader not found — install GPU drivers");
        return false;
    }

    u32 loader_version = volkGetInstanceVersion();
    if (loader_version < VK_API_VERSION_1_3) {
        cardinal::log::errorf("rhi/vk",
            "driver only supports Vulkan %u.%u; need 1.3+",
            VK_VERSION_MAJOR(loader_version), VK_VERSION_MINOR(loader_version));
        return false;
    }

    // Instance.
    VkApplicationInfo app{};
    app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName   = "Cardinal";
    app.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
    app.pEngineName        = "Cardinal Engine";
    app.engineVersion      = VK_MAKE_VERSION(0, 0, 1);
    app.apiVersion         = VK_API_VERSION_1_3;

    cardinal::vector<const char*> layers;
    cardinal::vector<const char*> exts = {
        VK_KHR_SURFACE_EXTENSION_NAME,
#if CARDINAL_PLATFORM_WINDOWS
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
#elif CARDINAL_PLATFORM_LINUX
        VK_KHR_XCB_SURFACE_EXTENSION_NAME,
#endif
    };
    if (desc.enable_validation) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    VkInstanceCreateInfo ic{};
    ic.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ic.pApplicationInfo        = &app;
    ic.enabledLayerCount       = static_cast<u32>(layers.size());
    ic.ppEnabledLayerNames     = layers.empty() ? nullptr : layers.data();
    ic.enabledExtensionCount   = static_cast<u32>(exts.size());
    ic.ppEnabledExtensionNames = exts.empty() ? nullptr : exts.data();

    if (!vk_check(vkCreateInstance(&ic, nullptr, &instance_), "vkCreateInstance")) {
        return false;
    }
    volkLoadInstance(instance_);

    // Physical device pick (discrete preferred + max VRAM).
    u32 phys_count = 0;
    vkEnumeratePhysicalDevices(instance_, &phys_count, nullptr);
    if (phys_count == 0) {
        cardinal::log::errorf("rhi/vk", "no Vulkan-capable GPUs");
        return false;
    }
    cardinal::vector<VkPhysicalDevice> phys(phys_count);
    vkEnumeratePhysicalDevices(instance_, &phys_count, phys.data());

    u64 best_score = 0;
    for (auto p : phys) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(p, &props);

        VkPhysicalDeviceMemoryProperties mem{};
        vkGetPhysicalDeviceMemoryProperties(p, &mem);
        u64 vram = 0;
        for (u32 i = 0; i < mem.memoryHeapCount; ++i) {
            if ((mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                vram += mem.memoryHeaps[i].size;
            }
        }
        u64 score = vram / (1024 * 1024);
        if (desc.prefer_discrete_gpu &&
            props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 1'000'000;
        }
        if (score > best_score) {
            best_score = score;
            physical_  = p;
            cardinal::strncpy(adapter_name_, props.deviceName, sizeof(adapter_name_) - 1);
        }
    }
    if (physical_ == VK_NULL_HANDLE) return false;

    // Pick a graphics queue family. (We don't yet check surface support here —
    // Win32/X11 surfaces always work on the graphics queue in practice; we'll
    // verify on swapchain creation and bail if not.)
    u32 qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &qf_count, nullptr);
    cardinal::vector<VkQueueFamilyProperties> qfs(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &qf_count, qfs.data());
    // First graphics family + a DEDICATED compute-only family (COMPUTE without
    // GRAPHICS) — the latter is the async-compute lane that overlaps the
    // graphics queue on real hardware (AEGIS Block 10). A shared graphics+
    // compute family is NOT used for async (it time-slices, not overlaps), so
    // when no dedicated family exists async compute stays disabled.
    for (u32 i = 0; i < qf_count; ++i) {
        const bool gfx = (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        const bool cmp = (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT)  != 0;
        if (gfx && graphics_queue_family_ == static_cast<u32>(-1))
            graphics_queue_family_ = i;
        if (cmp && !gfx && compute_queue_family_ == static_cast<u32>(-1))
            compute_queue_family_ = i;
    }
    if (graphics_queue_family_ == static_cast<u32>(-1)) {
        cardinal::log::errorf("rhi/vk", "no graphics queue family");
        return false;
    }

    // Identity for the capabilities struct.
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical_, &props);
        const char* vendor = "Unknown";
        switch (props.vendorID) {
            case 0x10DE: vendor = "NVIDIA Corporation"; break;
            case 0x1002: vendor = "AMD";                break;
            case 0x8086: vendor = "Intel";              break;
            default:                                    break;
        }
        cardinal::strncpy(caps_.vendor_name, vendor, sizeof(caps_.vendor_name) - 1);
        // Best-effort architecture string from device name.
        const char* arch = "Unknown";
        if (props.vendorID == 0x10DE) {
            // RTX 40 = Ada Lovelace, RTX 30 = Ampere, RTX 20 = Turing.
            if (cardinal::strstr(props.deviceName, "RTX 40") || cardinal::strstr(props.deviceName, "RTX 50"))
                arch = "Ada Lovelace+";
            else if (cardinal::strstr(props.deviceName, "RTX 30")) arch = "Ampere";
            else if (cardinal::strstr(props.deviceName, "RTX 20") || cardinal::strstr(props.deviceName, "GTX 16"))
                arch = "Turing";
        } else if (props.vendorID == 0x1002) {
            if (cardinal::strstr(props.deviceName, "RX 7"))      arch = "RDNA 3";
            else if (cardinal::strstr(props.deviceName, "RX 6")) arch = "RDNA 2";
            else if (cardinal::strstr(props.deviceName, "RX 5")) arch = "RDNA";
        }
        cardinal::strncpy(caps_.gpu_arch, arch, sizeof(caps_.gpu_arch) - 1);

        VkPhysicalDeviceMemoryProperties mem{};
        vkGetPhysicalDeviceMemoryProperties(physical_, &mem);
        for (u32 i = 0; i < mem.memoryHeapCount; ++i) {
            if ((mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                caps_.vram_bytes += mem.memoryHeaps[i].size;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Query supported features via the pNext chain. We ask for everything we
    // know how to use; whatever the driver reports back is what we'll
    // actually enable below.
    // ------------------------------------------------------------------------
    VkPhysicalDeviceVulkan11Features         q11{};  q11.sType  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    VkPhysicalDeviceVulkan12Features         q12{};  q12.sType  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceVulkan13Features         q13{};  q13.sType  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    VkPhysicalDeviceMeshShaderFeaturesEXT    qmesh{}; qmesh.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR qas{}; qas.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR    qrtp{}; qrtp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    VkPhysicalDeviceRayQueryFeaturesKHR              qrq{};  qrq.sType  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;

    // Chain: f2 -> 11 -> 12 -> 13 -> mesh -> as -> rtp -> rq
    qrtp.pNext  = &qrq;
    qas.pNext   = &qrtp;
    qmesh.pNext = &qas;
    q13.pNext   = &qmesh;
    q12.pNext   = &q13;
    q11.pNext   = &q12;

    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &q11;
    vkGetPhysicalDeviceFeatures2(physical_, &f2);

    // ------------------------------------------------------------------------
    // Enumerate device extensions to gate optional ones.
    // ------------------------------------------------------------------------
    u32 ext_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_, nullptr, &ext_count, nullptr);
    cardinal::vector<VkExtensionProperties> avail_exts(ext_count);
    vkEnumerateDeviceExtensionProperties(physical_, nullptr, &ext_count, avail_exts.data());

    auto has_ext = [&](const char* name) {
        for (const auto& e : avail_exts) {
            if (cardinal::strcmp(e.extensionName, name) == 0) return true;
        }
        return false;
    };

    // Required.
    cardinal::vector<const char*> dev_exts = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    const bool ext_dyn_rendering = has_ext(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    if (ext_dyn_rendering) dev_exts.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);

    const bool ext_mesh = has_ext(VK_EXT_MESH_SHADER_EXTENSION_NAME) && qmesh.meshShader;
    if (ext_mesh) dev_exts.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);

    const bool ext_def_host = has_ext(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    const bool ext_as       = has_ext(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
                              qas.accelerationStructure && ext_def_host;
    const bool ext_rtp      = has_ext(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) &&
                              qrtp.rayTracingPipeline && ext_as;
    const bool ext_rq       = has_ext(VK_KHR_RAY_QUERY_EXTENSION_NAME) && qrq.rayQuery && ext_as;

    if (ext_def_host) dev_exts.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    if (ext_as)       dev_exts.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    if (ext_rtp)      dev_exts.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    if (ext_rq)       dev_exts.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);

    // VK_EXT_memory_budget: enables vkGetPhysicalDeviceMemoryProperties2 with a
    // VkPhysicalDeviceMemoryBudgetPropertiesEXT pNext, which gives us per-heap
    // budget + usage. Universally supported on Vulkan 1.1+ desktop drivers.
    const bool ext_mem_budget = has_ext(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    if (ext_mem_budget) dev_exts.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    has_memory_budget_ = ext_mem_budget;

    // VK_NV_low_latency2: prerequisite for NvLL_VK Reflex calls
    // (NvLL_VK_InitLowLatencyDevice, NvLL_VK_LatencySleep,
    // NvLL_VK_SetLatencyMarker). Only available on NVIDIA drivers.
    // Probed unconditionally — falls through silently on AMD / Intel.
    const bool ext_nv_low_latency2 = has_ext("VK_NV_low_latency2");
    if (ext_nv_low_latency2) dev_exts.push_back("VK_NV_low_latency2");
    has_nv_low_latency2_ = ext_nv_low_latency2;

    // ------------------------------------------------------------------------
    // Build the enable chain — only set bits the driver reported as TRUE.
    // ------------------------------------------------------------------------
    VkPhysicalDeviceVulkan11Features e11{};  e11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    VkPhysicalDeviceVulkan12Features e12{};  e12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceVulkan13Features e13{};  e13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    VkPhysicalDeviceMeshShaderFeaturesEXT emesh{}; emesh.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR eas{};  eas.sType  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR    ertp{}; ertp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    VkPhysicalDeviceRayQueryFeaturesKHR              erq{};  erq.sType  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;

    // 1.3 core
    e13.dynamicRendering = q13.dynamicRendering;
    e13.synchronization2 = q13.synchronization2;
    e13.maintenance4     = q13.maintenance4;
    e13.subgroupSizeControl = q13.subgroupSizeControl;
    // 1.2 — bindless, BDA, scalar layout, FP16/INT8, descriptor indexing, timeline
    e12.bufferDeviceAddress                      = q12.bufferDeviceAddress;
    e12.descriptorIndexing                       = q12.descriptorIndexing;
    e12.runtimeDescriptorArray                   = q12.runtimeDescriptorArray;
    e12.descriptorBindingPartiallyBound          = q12.descriptorBindingPartiallyBound;
    e12.descriptorBindingVariableDescriptorCount = q12.descriptorBindingVariableDescriptorCount;
    e12.descriptorBindingSampledImageUpdateAfterBind  = q12.descriptorBindingSampledImageUpdateAfterBind;
    e12.descriptorBindingStorageBufferUpdateAfterBind = q12.descriptorBindingStorageBufferUpdateAfterBind;
    e12.shaderSampledImageArrayNonUniformIndexing     = q12.shaderSampledImageArrayNonUniformIndexing;
    e12.scalarBlockLayout                        = q12.scalarBlockLayout;
    e12.shaderFloat16                            = q12.shaderFloat16;
    e12.shaderInt8                               = q12.shaderInt8;
    e12.timelineSemaphore                        = q12.timelineSemaphore;
    e12.storageBuffer8BitAccess                  = q12.storageBuffer8BitAccess;
    e12.uniformAndStorageBuffer8BitAccess        = q12.uniformAndStorageBuffer8BitAccess;
    // 1.1 — wave/subgroup ops
    e11.storageBuffer16BitAccess                 = q11.storageBuffer16BitAccess;
    e11.uniformAndStorageBuffer16BitAccess       = q11.uniformAndStorageBuffer16BitAccess;
    e11.shaderDrawParameters                     = q11.shaderDrawParameters;

    // Mesh shaders
    if (ext_mesh) {
        emesh.meshShader = qmesh.meshShader;
        emesh.taskShader = qmesh.taskShader;
    }
    // Ray tracing
    if (ext_as)  eas.accelerationStructure  = qas.accelerationStructure;
    if (ext_rtp) ertp.rayTracingPipeline    = qrtp.rayTracingPipeline;
    if (ext_rq)  erq.rayQuery               = qrq.rayQuery;

    // Build the chain — only attach what we're actually using.
    void* chain_head = &e13;
    e13.pNext = &e12;
    e12.pNext = &e11;
    void** tail_pnext = &e11.pNext;
    if (ext_mesh) { *tail_pnext = &emesh; tail_pnext = &emesh.pNext; }
    if (ext_as)   { *tail_pnext = &eas;   tail_pnext = &eas.pNext;   }
    if (ext_rtp)  { *tail_pnext = &ertp;  tail_pnext = &ertp.pNext;  }
    if (ext_rq)   { *tail_pnext = &erq;   tail_pnext = &erq.pNext;   }

    // ------------------------------------------------------------------------
    // Logical device.
    // ------------------------------------------------------------------------
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qcs[2]{};
    qcs[0].sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qcs[0].queueFamilyIndex = graphics_queue_family_;
    qcs[0].queueCount       = 1;
    qcs[0].pQueuePriorities = &prio;
    u32 qc_count = 1;
    // Request the dedicated async-compute queue if a compute-only family exists.
    if (compute_queue_family_ != static_cast<u32>(-1) &&
        compute_queue_family_ != graphics_queue_family_) {
        qcs[1].sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qcs[1].queueFamilyIndex = compute_queue_family_;
        qcs[1].queueCount       = 1;
        qcs[1].pQueuePriorities = &prio;
        qc_count = 2;
    } else {
        compute_queue_family_ = static_cast<u32>(-1);   // no dedicated lane
    }

    VkDeviceCreateInfo dc{};
    dc.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dc.pNext                   = chain_head;
    dc.queueCreateInfoCount    = qc_count;
    dc.pQueueCreateInfos       = qcs;
    dc.enabledExtensionCount   = static_cast<u32>(dev_exts.size());
    dc.ppEnabledExtensionNames = dev_exts.data();

    if (!vk_check(vkCreateDevice(physical_, &dc, nullptr, &device_), "vkCreateDevice")) {
        return false;
    }
    volkLoadDevice(device_);
    vkGetDeviceQueue(device_, graphics_queue_family_, 0, &graphics_queue_);
    if (compute_queue_family_ != static_cast<u32>(-1))
        vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);

    // ------------------------------------------------------------------------
    // Populate the public-facing capability struct.
    // ------------------------------------------------------------------------
    caps_.dynamic_rendering        = q13.dynamicRendering;
    caps_.synchronization2         = q13.synchronization2;
    caps_.buffer_device_address    = q12.bufferDeviceAddress;
    caps_.descriptor_indexing      = q12.descriptorIndexing;
    caps_.runtime_descriptor_array = q12.runtimeDescriptorArray;
    caps_.scalar_block_layout      = q12.scalarBlockLayout;
    caps_.timeline_semaphore       = q12.timelineSemaphore;
    // Async compute needs both the dedicated queue AND timeline semaphores
    // (the Fence + cross-queue handshake are timeline-based).
    caps_.async_compute            = (compute_queue_ != VK_NULL_HANDLE) && q12.timelineSemaphore;
    caps_.max_async_compute_queues = caps_.async_compute ? 1u : 0u;
    caps_.shader_float16           = q12.shaderFloat16;
    caps_.shader_int8              = q12.shaderInt8;
    caps_.shader_int16             = q11.storageBuffer16BitAccess;  // proxy
    caps_.storage_buffer_16bit     = q11.storageBuffer16BitAccess;
    caps_.wave_intrinsics          = true;  // Vulkan 1.1+ has subgroup ops baseline
    caps_.mesh_shader              = ext_mesh && qmesh.meshShader;
    caps_.task_shader              = ext_mesh && qmesh.taskShader;
    caps_.acceleration_structure   = ext_as  && qas.accelerationStructure;
    caps_.ray_tracing_pipeline     = ext_rtp && qrtp.rayTracingPipeline;
    caps_.ray_query                = ext_rq  && qrq.rayQuery;

    // Vendor-specific paths.
    const bool is_nvidia = cardinal::strcmp(caps_.vendor_name, "NVIDIA Corporation") == 0;
    const bool is_amd    = cardinal::strcmp(caps_.vendor_name, "AMD") == 0;
    const bool is_ada    = cardinal::strstr(caps_.gpu_arch, "Ada") != nullptr;
    caps_.nvidia_dlss_capable     = is_nvidia && caps_.acceleration_structure;
    caps_.nvidia_framegen_capable = caps_.nvidia_dlss_capable && is_ada;
    caps_.amd_fsr3_capable        = is_amd;

    if (!init_vma()) return false;

    // Apply initial settings (clamps to capabilities).
    apply_settings(desc.initial_settings);

    return true;
}

bool VulkanDevice::init_vma() {
    VmaVulkanFunctions fns{};
    fns.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    fns.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo aci{};
    aci.vulkanApiVersion = VK_API_VERSION_1_3;
    aci.physicalDevice   = physical_;
    aci.device           = device_;
    aci.instance         = instance_;
    aci.pVulkanFunctions = &fns;
    // Required so vmaCreateBuffer honors VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    // which we need for ray tracing inputs and acceleration-structure scratch
    // buffers.
    aci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    if (vmaCreateAllocator(&aci, &allocator_) != VK_SUCCESS || allocator_ == VK_NULL_HANDLE) {
        cardinal::log::errorf("rhi/vk", "vmaCreateAllocator failed");
        return false;
    }

    // Shared sampler for sampled-texture bindings. Clamp-to-edge so a
    // shadow lookup just outside the map reads the edge texel rather
    // than wrapping; linear for cheap 2x2 PCF-ish softening.
    VkSamplerCreateInfo sci{};
    sci.sType         = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter     = VK_FILTER_LINEAR;
    sci.minFilter     = VK_FILTER_LINEAR;
    sci.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod        = 1.0f;
    if (vkCreateSampler(device_, &sci, nullptr, &default_sampler_) != VK_SUCCESS) {
        cardinal::log::errorf("rhi/vk", "vkCreateSampler (default) failed");
        return false;
    }
    return true;
}

// =============================================================================
// DXC runtime shader compilation
//
// Single source-of-truth: HLSL with SM 6.x. DXC outputs SPIR-V here for
// Vulkan; the same source compiles to DXIL for the (future) D3D12 backend
// by dropping the `-spirv` flag. This is the "runtime selectable" pivot —
// shaders aren't baked at build time, they're text the engine compiles
// when it needs them (enabling hot reload, runtime permutations, etc.).
// =============================================================================
bool VulkanDevice::init_dxc() {
    if (dxc_compiler_ != nullptr) return true;
    HRESULT hr = DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxc_compiler_));
    if (FAILED(hr) || dxc_compiler_ == nullptr) {
        cardinal::log::errorf("rhi/vk", "DxcCreateInstance(compiler) failed (0x%08lX)",
                              static_cast<unsigned long>(hr));
        return false;
    }
    hr = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxc_utils_));
    if (FAILED(hr) || dxc_utils_ == nullptr) {
        cardinal::log::errorf("rhi/vk", "DxcCreateInstance(utils) failed (0x%08lX)",
                              static_cast<unsigned long>(hr));
        return false;
    }
    return true;
}

ShaderBlob VulkanDevice::compile_shader(
    ShaderStage stage, const char* hlsl_source, const char* entry_point)
{
    if (!init_dxc()) return {};
    if (hlsl_source == nullptr || entry_point == nullptr) return {};

    LPCWSTR profile = nullptr;
    switch (stage) {
        case ShaderStage::Vertex:   profile = L"vs_6_6"; break;
        case ShaderStage::Fragment: profile = L"ps_6_6"; break;
        case ShaderStage::Compute:  profile = L"cs_6_6"; break;
    }
    if (profile == nullptr) return {};

    wchar_t entry_w[128]{};
    MultiByteToWideChar(CP_UTF8, 0, entry_point, -1, entry_w, 128);

    DxcBuffer src{};
    src.Ptr      = hlsl_source;
    src.Size     = cardinal::strlen(hlsl_source);
    src.Encoding = DXC_CP_UTF8;

    // Args: SM 6.6 baseline so HLSL 2021 + new wave/inline RT ops are valid.
    // 16-bit native types only enabled when both the device supports FP16
    // *and* the runtime setting requests it — that pair is what drives RPM.
    cardinal::vector<LPCWSTR> args = {
        L"-T", profile,
        L"-E", entry_w,
        L"-spirv",                       // SPIR-V output (Vulkan)
        L"-fvk-use-dx-layout",           // HLSL-style cbuffer layout
        L"-fspv-target-env=vulkan1.3",   // emit Vulkan 1.3 SPIR-V
        L"-HV", L"2021",                 // HLSL 2021
        L"-O3",                          // optimize
        // Matrix packing MUST match the D3D12 backend: the renderer
        // pushes the SAME scene::Mat4 bytes to both, and the shared
        // HLSL does mul(pc.mvp, pos). D3D12 compiles with DXC's default
        // (column-major); the old -Zpr (row-major) here transposed
        // every transform on Vulkan only, so placed geometry projected
        // to garbage clip space and never rendered as 3D. Pin
        // column-major explicitly so the two backends can't diverge.
        L"-Zpc",                         // column-major (match D3D12)
    };
    if (caps_.shader_float16 && settings_.prefer_fp16) {
        args.push_back(L"-enable-16bit-types");
    }

    IDxcResult* result = nullptr;
    HRESULT hr = dxc_compiler_->Compile(
        &src, args.data(), static_cast<u32>(args.size()),
        nullptr, IID_PPV_ARGS(&result));
    if (FAILED(hr) || result == nullptr) {
        cardinal::log::errorf("rhi/vk", "DXC Compile call failed (0x%08lX)",
                              static_cast<unsigned long>(hr));
        if (result != nullptr) result->Release();
        return {};
    }

    HRESULT status = E_FAIL;
    result->GetStatus(&status);
    if (FAILED(status)) {
        IDxcBlobUtf8* errs = nullptr;
        result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errs), nullptr);
        if (errs != nullptr && errs->GetStringLength() > 0) {
            cardinal::log::errorf("rhi/vk", "HLSL compile error:\n%s",
                                  errs->GetStringPointer());
        } else {
            cardinal::log::errorf("rhi/vk", "HLSL compile failed with no diagnostics");
        }
        if (errs != nullptr)   errs->Release();
        result->Release();
        return {};
    }

    IDxcBlob* obj = nullptr;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&obj), nullptr);
    if (obj == nullptr || obj->GetBufferSize() == 0) {
        cardinal::log::errorf("rhi/vk", "DXC produced empty object");
        if (obj != nullptr) obj->Release();
        result->Release();
        return {};
    }

    ShaderBlob blob;
    auto* p = static_cast<const u8*>(obj->GetBufferPointer());
    blob.bytes.assign(p, p + obj->GetBufferSize());

    obj->Release();
    result->Release();
    return blob;
}

// Live VRAM telemetry — VK_EXT_memory_budget gives us per-heap budget +
// usage; we sum the device-local heaps (== "VRAM" on a discrete GPU; on
// integrated parts this is the shared heap, which is also the right
// answer). Fallback when the extension isn't present: return budget=total,
// usage=0 so the broker treats us as Low pressure rather than spamming.
Device::VramSnapshot VulkanDevice::query_vram_usage() const noexcept {
    VramSnapshot s{};
    if (physical_ == VK_NULL_HANDLE) return s;

    if (has_memory_budget_) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT bp{};
        bp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;

        VkPhysicalDeviceMemoryProperties2 mp2{};
        mp2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        mp2.pNext = &bp;
        vkGetPhysicalDeviceMemoryProperties2(physical_, &mp2);

        for (u32 i = 0; i < mp2.memoryProperties.memoryHeapCount; ++i) {
            if ((mp2.memoryProperties.memoryHeaps[i].flags &
                 VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                s.budget_bytes        += static_cast<u64>(bp.heapBudget[i]);
                s.current_usage_bytes += static_cast<u64>(bp.heapUsage[i]);
            }
        }
    } else {
        // Best effort: total VRAM as budget, no usage info.
        s.budget_bytes = caps_.vram_bytes;
    }
    return s;
}

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

cardinal::unique_ptr<Buffer> VulkanDevice::create_buffer(const BufferDesc& desc) {
    auto b = cardinal::make_unique<VulkanBuffer>(*this);
    if (!b->initialize(desc)) return nullptr;
    return b;
}

// Transient single-submit command buffer for one-off GPU work (initial
// image layout transitions, acceleration-structure builds). submit_and_wait
// stalls the queue — only for setup-time, never per-frame.
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

cardinal::unique_ptr<Texture> VulkanDevice::create_texture(const TextureDesc& desc) {
    auto t = cardinal::make_unique<VulkanTexture>(*this);
    if (!t->initialize(desc)) return nullptr;
    return t;
}

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
    }

    bool initialize(const PipelineDesc& desc);

    VkPipeline            handle()              const noexcept { return pipeline_; }
    VkPipelineLayout      layout()              const noexcept { return layout_; }
    u32                   push_constant_size()  const noexcept { return push_constant_size_; }
    // VK_NULL_HANDLE when storage_buffer_slots_ == 0 (push-only pipeline).
    VkDescriptorSetLayout descriptor_set_layout() const noexcept { return dsl_; }
    u32                   storage_buffer_slots() const noexcept { return storage_buffer_slots_; }

private:
    VulkanDevice&         dev_;
    VkPipelineLayout      layout_{VK_NULL_HANDLE};
    VkPipeline            pipeline_{VK_NULL_HANDLE};
    VkDescriptorSetLayout dsl_{VK_NULL_HANDLE};
    VkShaderModule        vs_module_{VK_NULL_HANDLE};
    VkShaderModule        fs_module_{VK_NULL_HANDLE};
    u32                   push_constant_size_{0};
    u32                   storage_buffer_slots_{0};
    u32                   sampled_texture_slots_{0};
public:
    u32 sampled_texture_slots() const noexcept { return sampled_texture_slots_; }
};

namespace {

VkShaderModule make_shader_module(VkDevice dev, const ShaderBlob& spirv) {
    if (!spirv.ok() || (spirv.bytes.size() % 4) != 0) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = spirv.bytes.size();
    ci.pCode    = reinterpret_cast<const u32*>(spirv.bytes.data());
    VkShaderModule mod = VK_NULL_HANDLE;
    vkCreateShaderModule(dev, &ci, nullptr, &mod);
    return mod;
}

}  // namespace

bool VulkanPipeline::initialize(const PipelineDesc& desc) {
    vs_module_ = make_shader_module(dev_.device_, desc.vertex_shader);
    fs_module_ = make_shader_module(dev_.device_, desc.fragment_shader);
    if (vs_module_ == VK_NULL_HANDLE || fs_module_ == VK_NULL_HANDLE) {
        cardinal::log::errorf("rhi/vk", "vkCreateShaderModule failed");
        return false;
    }

    push_constant_size_    = desc.push_constant_size;
    storage_buffer_slots_  = desc.storage_buffer_slots;
    sampled_texture_slots_ = desc.sampled_texture_slots;

    // Descriptor-set 0, visible to all graphics stages:
    //   bindings [0, storage_buffer_slots_)            STORAGE_BUFFER
    //   bindings [storage_buffer_slots_, +sampled_n)   COMBINED_IMAGE_SAMPLER
    // Only built when the pipeline declares at least one of either, so
    // push-constant-only pipelines keep an empty (set-less) layout.
    cardinal::vector<VkDescriptorSetLayoutBinding> dsl_bindings;
    if (storage_buffer_slots_ > 0 || sampled_texture_slots_ > 0) {
        for (u32 s = 0; s < storage_buffer_slots_; ++s) {
            VkDescriptorSetLayoutBinding b{};
            b.binding         = s;
            b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b.descriptorCount = 1;
            b.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT
                              | VK_SHADER_STAGE_FRAGMENT_BIT;
            dsl_bindings.push_back(b);
        }
        for (u32 s = 0; s < sampled_texture_slots_; ++s) {
            VkDescriptorSetLayoutBinding b{};
            b.binding         = storage_buffer_slots_ + s;
            b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b.descriptorCount = 1;
            b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
            dsl_bindings.push_back(b);
        }
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<u32>(dsl_bindings.size());
        dlci.pBindings    = dsl_bindings.data();
        if (!vk_check(
                vkCreateDescriptorSetLayout(dev_.device_, &dlci, nullptr, &dsl_),
                "vkCreateDescriptorSetLayout")) {
            return false;
        }
    }

    VkPipelineLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    if (dsl_ != VK_NULL_HANDLE) {
        lci.setLayoutCount = 1;
        lci.pSetLayouts    = &dsl_;
    }
    VkPushConstantRange pc_range{};
    if (push_constant_size_ > 0) {
        // Visible to both stages so the renderer can put MVP in VS and
        // per-draw scalars in FS without two separate ranges.
        //
        // Vulkan only *guarantees* maxPushConstantsSize >= 128 B; real
        // devices vary (commonly 128 or 256). The scene block is 240 B,
        // so on a 128-byte device vkCreatePipelineLayout would fail
        // opaquely and no scene PSO would come up — the exact twin of
        // the D3D12 root-signature overflow. Fail loudly + cleanly with
        // an actionable message instead (the higher layers already
        // degrade gracefully on a null pipeline). The portable fix —
        // demote large blocks to a per-frame UBO, mirroring the D3D12
        // backend's root-CBV fallback — is the follow-up; the common
        // NVIDIA/AMD desktop limit of 256 B covers the 240 B block.
        VkPhysicalDeviceProperties pdp{};
        vkGetPhysicalDeviceProperties(dev_.vk_physical(), &pdp);
        if (push_constant_size_ > pdp.limits.maxPushConstantsSize) {
            cardinal::log::errorf("rhi/vk",
                "push constant block %u B exceeds device maxPushConstantsSize "
                "%u B — pipeline not created (needs a UBO fallback)",
                push_constant_size_, pdp.limits.maxPushConstantsSize);
            return false;
        }
        pc_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT
                            | VK_SHADER_STAGE_FRAGMENT_BIT;
        pc_range.offset     = 0;
        pc_range.size       = push_constant_size_;
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges    = &pc_range;
    }
    if (!vk_check(
            vkCreatePipelineLayout(dev_.device_, &lci, nullptr, &layout_),
            "vkCreatePipelineLayout")) {
        return false;
    }

    cardinal::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs_module_;
    stages[0].pName  = desc.vertex_entry;
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs_module_;
    stages[1].pName  = desc.fragment_entry;

    // Vertex input.
    VkVertexInputBindingDescription binding{};
    binding.binding   = 0;
    binding.stride    = desc.vertex_stride;
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    cardinal::vector<VkVertexInputAttributeDescription> attribs;
    attribs.reserve(desc.vertex_attribs.size());
    for (const auto& a : desc.vertex_attribs) {
        VkVertexInputAttributeDescription ad{};
        ad.location = a.location;
        ad.binding  = 0;
        ad.format   = to_vk_format(a.format);
        ad.offset   = a.offset;
        attribs.push_back(ad);
    }

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    if (desc.vertex_stride > 0 && !attribs.empty()) {
        vi.vertexBindingDescriptionCount   = 1;
        vi.pVertexBindingDescriptions      = &binding;
        vi.vertexAttributeDescriptionCount = static_cast<u32>(attribs.size());
        vi.pVertexAttributeDescriptions    = attribs.data();
    }

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = to_vk_topology(desc.topology);

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode    = VK_CULL_MODE_NONE;
    rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // color_format == Unknown ⇒ depth-only pipeline (0 colour
    // attachments) — the shadow-map depth pass. Drives both the blend
    // state below and the dynamic-rendering formats further down.
    const bool depth_only = (desc.color_format == Format::Unknown);

    VkPipelineColorBlendAttachmentState att{};
    att.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = depth_only ? 0u : 1u;
    cb.pAttachments    = depth_only ? nullptr : &att;

    cardinal::array<VkDynamicState, 2> dyn_states = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = static_cast<u32>(dyn_states.size());
    dyn.pDynamicStates    = dyn_states.data();

    VkFormat color_fmt = to_vk_format(desc.color_format);
    // Depth-attachment format declaration. When Unknown, prc reports
    // VK_FORMAT_UNDEFINED and the pipeline expects no depth attachment
    // in the rendering pass. When set (currently D32_SFLOAT for the
    // viewport RTTs), the pipeline must be bound in a context that
    // attaches a matching-format depth image.
    VkFormat depth_fmt = (desc.depth_format == Format::Unknown)
        ? VK_FORMAT_UNDEFINED
        : to_vk_format(desc.depth_format);

    VkPipelineRenderingCreateInfo prc{};
    prc.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    prc.colorAttachmentCount    = depth_only ? 0u : 1u;
    prc.pColorAttachmentFormats = depth_only ? nullptr : &color_fmt;
    prc.depthAttachmentFormat   = depth_fmt;

    // Depth-stencil state. Only attach when a depth format is declared;
    // otherwise omit the struct entirely so Vulkan doesn't validate
    // depth state against an absent attachment.
    VkPipelineDepthStencilStateCreateInfo dss{};
    dss.sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    dss.depthTestEnable       = (desc.depth_format != Format::Unknown && desc.depth_test) ? VK_TRUE : VK_FALSE;
    dss.depthWriteEnable      = (desc.depth_format != Format::Unknown && desc.depth_write) ? VK_TRUE : VK_FALSE;
    // LESS — entries with smaller depth pass. Matches the scene's
    // 1.0 = far / 0.0 = near convention (see CLEAR depth above).
    dss.depthCompareOp        = VK_COMPARE_OP_LESS;
    dss.depthBoundsTestEnable = VK_FALSE;
    dss.stencilTestEnable     = VK_FALSE;

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.pNext               = &prc;
    pci.stageCount          = static_cast<u32>(stages.size());
    pci.pStages             = stages.data();
    pci.pVertexInputState   = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState      = &vp;
    pci.pRasterizationState = &rs;
    pci.pMultisampleState   = &ms;
    pci.pColorBlendState    = &cb;
    pci.pDynamicState       = &dyn;
    pci.pDepthStencilState  = (desc.depth_format != Format::Unknown) ? &dss : nullptr;
    pci.layout              = layout_;
    pci.subpass             = 0;

    return vk_check(
        vkCreateGraphicsPipelines(
            dev_.device_, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline_),
        "vkCreateGraphicsPipelines");
}

cardinal::unique_ptr<Pipeline> VulkanDevice::create_pipeline(const PipelineDesc& desc) {
    auto p = cardinal::make_unique<VulkanPipeline>(*this);
    if (!p->initialize(desc)) return nullptr;
    return p;
}

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

namespace {

// One-shot command buffer for AS builds. Submits to graphics queue and
// waits on the queue (sync). Suitable for one-time setup; per-frame builds
// will use a dedicated transfer queue + fence in a future phase.
// Allocate a transient buffer for AS scratch / result storage.
struct ScratchBuffer {
    VmaAllocator     allocator{VK_NULL_HANDLE};
    VkBuffer         buffer{VK_NULL_HANDLE};
    VmaAllocation    alloc{VK_NULL_HANDLE};
    u64              device_address{0};

    ~ScratchBuffer() {
        if (buffer != VK_NULL_HANDLE) vmaDestroyBuffer(allocator, buffer, alloc);
    }

    bool create(VulkanDevice& dev, VkDeviceSize size, VkBufferUsageFlags usage) {
        allocator = dev.vk_allocator();
        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = size;
        bci.usage       = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        if (vmaCreateBuffer(allocator, &bci, &aci, &buffer, &alloc, nullptr) != VK_SUCCESS) {
            return false;
        }
        VkBufferDeviceAddressInfo bdai{};
        bdai.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        bdai.buffer = buffer;
        device_address = vkGetBufferDeviceAddress(dev.vk_device(), &bdai);
        return true;
    }
};

}  // namespace

bool VulkanAccelerationStructure::initialize_blas(const BlasDesc& desc) {
    if (desc.geometries.empty()) {
        cardinal::log::errorf("rhi/vk/blas", "no geometries");
        return false;
    }

    // 1) Build the geometry / range info arrays.
    cardinal::vector<VkAccelerationStructureGeometryKHR>      geoms;
    cardinal::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
    cardinal::vector<u32>                                      max_prim_counts;
    geoms.reserve(desc.geometries.size());
    ranges.reserve(desc.geometries.size());
    max_prim_counts.reserve(desc.geometries.size());

    for (const auto& g : desc.geometries) {
        if (g.vertex_buffer == nullptr || g.vertex_count == 0) {
            cardinal::log::errorf("rhi/vk/blas", "geometry missing vertex buffer");
            return false;
        }
        auto* vbuf = static_cast<VulkanBuffer*>(g.vertex_buffer);
        if (vbuf->device_address() == 0) {
            cardinal::log::errorf("rhi/vk/blas",
                "vertex buffer must be created with ShaderDeviceAddress");
            return false;
        }

        VkAccelerationStructureGeometryTrianglesDataKHR tris{};
        tris.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        tris.vertexFormat = to_vk_format(g.vertex_format);
        tris.vertexData.deviceAddress = vbuf->device_address() + g.vertex_offset;
        tris.vertexStride             = g.vertex_stride;
        tris.maxVertex                = g.vertex_count - 1;
        tris.indexType                = VK_INDEX_TYPE_NONE_KHR;

        u32 prim_count = g.vertex_count / 3;
        if (g.index_buffer != nullptr && g.index_count > 0) {
            auto* ibuf = static_cast<VulkanBuffer*>(g.index_buffer);
            if (ibuf->device_address() == 0) {
                cardinal::log::errorf("rhi/vk/blas",
                    "index buffer must be ShaderDeviceAddress");
                return false;
            }
            tris.indexType                = g.indices_are_u32 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
            tris.indexData.deviceAddress  = ibuf->device_address() + g.index_offset;
            prim_count                    = g.index_count / 3;
        }

        VkAccelerationStructureGeometryKHR geom{};
        geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom.geometry.triangles = tris;
        geom.flags        = g.opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0u;
        geoms.push_back(geom);

        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = prim_count;
        ranges.push_back(range);

        max_prim_counts.push_back(prim_count);
    }

    // 2) Query build sizes.
    VkAccelerationStructureBuildGeometryInfoKHR build_info{};
    build_info.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build_info.flags         = (desc.prefer_fast_trace
                                ? VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR);
    if (desc.allow_compaction) {
        build_info.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    }
    build_info.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = static_cast<u32>(geoms.size());
    build_info.pGeometries   = geoms.data();

    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(
        dev_.device_,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info,
        max_prim_counts.data(),
        &sizes);

    // 3) Result + scratch buffers.
    {
        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = sizes.accelerationStructureSize;
        bci.usage       = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (!vk_check(
                vmaCreateBuffer(dev_.allocator_, &bci, &aci,
                                &result_buffer_, &result_alloc_, nullptr),
                "vmaCreateBuffer (BLAS result)")) {
            return false;
        }
    }

    ScratchBuffer scratch;
    if (!scratch.create(dev_,
                        sizes.buildScratchSize,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        cardinal::log::errorf("rhi/vk/blas", "scratch buffer alloc failed");
        return false;
    }

    // 4) Create the AS handle on top of the result buffer.
    {
        VkAccelerationStructureCreateInfoKHR aci{};
        aci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        aci.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        aci.size   = sizes.accelerationStructureSize;
        aci.buffer = result_buffer_;
        if (!vk_check(
                vkCreateAccelerationStructureKHR(dev_.device_, &aci, nullptr, &as_handle_),
                "vkCreateAccelerationStructureKHR")) {
            return false;
        }
    }

    // 5) Build via one-shot command buffer.
    {
        OneShotCmd one(dev_);
        build_info.dstAccelerationStructure  = as_handle_;
        build_info.scratchData.deviceAddress = scratch.device_address;

        cardinal::vector<const VkAccelerationStructureBuildRangeInfoKHR*> range_ptrs;
        range_ptrs.reserve(ranges.size());
        for (const auto& r : ranges) range_ptrs.push_back(&r);

        vkCmdBuildAccelerationStructuresKHR(
            one.cmd, 1, &build_info, range_ptrs.data());

        // Barrier so the AS is readable for subsequent uses.
        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(one.cmd,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0, 1, &mb, 0, nullptr, 0, nullptr);

        one.submit_and_wait();
    }

    // 6) Cache the device address.
    {
        VkAccelerationStructureDeviceAddressInfoKHR ai{};
        ai.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        ai.accelerationStructure = as_handle_;
        device_address_ = vkGetAccelerationStructureDeviceAddressKHR(dev_.device_, &ai);
    }
    return true;
}

cardinal::unique_ptr<AccelerationStructure>
VulkanDevice::create_blas(const BlasDesc& desc) {
    if (!caps_.acceleration_structure) {
        cardinal::log::errorf("rhi/vk/blas",
            "device does not support VK_KHR_acceleration_structure");
        return nullptr;
    }
    auto blas = cardinal::make_unique<VulkanAccelerationStructure>(*this);
    if (!blas->initialize_blas(desc)) return nullptr;
    return blas;
}

cardinal::unique_ptr<AccelerationStructure>
VulkanDevice::create_tlas(const TlasDesc& /*desc*/) {
    // TODO(phase-5.1): symmetric to BLAS but with VkAccelerationStructureInstanceKHR
    // packed into an instances buffer + VK_GEOMETRY_TYPE_INSTANCES_KHR.
    cardinal::log::errorf("rhi/vk/tlas", "not yet implemented");
    return nullptr;
}

// =============================================================================
// VulkanSwapchain implementation
// =============================================================================
VulkanSwapchain::~VulkanSwapchain() {
    if (dev_.device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(dev_.device_);
        destroy_all_viewport_images();
        destroy_per_frame_objects();
        destroy_swapchain_objects();
    }
    if (surface_ != VK_NULL_HANDLE && dev_.instance_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(dev_.instance_, surface_, nullptr);
    }
}

// ---------------------------------------------------------------------------
// Viewport (RTT) lifecycle
// ---------------------------------------------------------------------------
bool VulkanSwapchain::ensure_viewport_image(u32 id, u32 w, u32 h) {
    if (id >= viewport_count_) return false;
    auto& vp = viewports_[id];
    if (vp.extent.width == w && vp.extent.height == h && vp.image != VK_NULL_HANDLE) {
        return true;
    }
    // Idle the device so we don't free an image the GPU is still sampling
    // from a previous frame's overlay submit. Safe because per-viewport
    // resizes are rare (panel-drag debounced inside Studio).
    vkDeviceWaitIdle(dev_.device_);
    destroy_viewport_image(id);
    if (w == 0 || h == 0) return true;

    VkImageCreateInfo ic{};
    ic.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ic.imageType     = VK_IMAGE_TYPE_2D;
    ic.format        = format_;
    ic.extent        = { w, h, 1 };
    ic.mipLevels     = 1;
    ic.arrayLayers   = 1;
    ic.samples       = VK_SAMPLE_COUNT_1_BIT;
    ic.tiling        = VK_IMAGE_TILING_OPTIMAL;
    // ColorAttachment: render into it. TransferSrc: blit out of it (legacy
    // single-viewport blit-to-backbuffer path; harmless on per-id slots).
    // Sampled: bind for ImGui::Image inside the editor's viewport panel.
    ic.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                       VK_IMAGE_USAGE_SAMPLED_BIT;
    ic.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ic.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (!vk_check(
            vmaCreateImage(dev_.allocator_, &ic, &aci,
                           &vp.image, &vp.alloc, nullptr),
            "vmaCreateImage (viewport[id])")) {
        return false;
    }

    VkImageViewCreateInfo iv{};
    iv.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    iv.image                       = vp.image;
    iv.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
    iv.format                      = format_;
    iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    iv.subresourceRange.levelCount = 1;
    iv.subresourceRange.layerCount = 1;
    if (!vk_check(
            vkCreateImageView(dev_.device_, &iv, nullptr, &vp.view),
            "vkCreateImageView (viewport[id])")) {
        return false;
    }

    // ----- Depth attachment ----------------------------------------------
    // Same extent as the colour image; D32_SFLOAT is universally
    // supported and matches what depth_format() reports. CLEAR loadOp
    // each frame in set_active_viewport, so we never sample previous
    // depth contents — initialLayout UNDEFINED is fine.
    VkImageCreateInfo dc{};
    dc.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    dc.imageType     = VK_IMAGE_TYPE_2D;
    dc.format        = VK_FORMAT_D32_SFLOAT;
    dc.extent        = { w, h, 1 };
    dc.mipLevels     = 1;
    dc.arrayLayers   = 1;
    dc.samples       = VK_SAMPLE_COUNT_1_BIT;
    dc.tiling        = VK_IMAGE_TILING_OPTIMAL;
    dc.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    dc.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    dc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (!vk_check(
            vmaCreateImage(dev_.allocator_, &dc, &aci,
                           &vp.depth_image, &vp.depth_alloc, nullptr),
            "vmaCreateImage (depth[id])")) {
        return false;
    }

    VkImageViewCreateInfo dv{};
    dv.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    dv.image                       = vp.depth_image;
    dv.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
    dv.format                      = VK_FORMAT_D32_SFLOAT;
    dv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    dv.subresourceRange.levelCount = 1;
    dv.subresourceRange.layerCount = 1;
    if (!vk_check(
            vkCreateImageView(dev_.device_, &dv, nullptr, &vp.depth_view),
            "vkCreateImageView (depth[id])")) {
        return false;
    }

    vp.extent         = { w, h };
    vp.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;   // fresh image
    vp.depth_layout   = VK_IMAGE_LAYOUT_UNDEFINED;
    return true;
}

void VulkanSwapchain::destroy_viewport_image(u32 id) {
    if (id >= viewports_.size()) return;
    auto& vp = viewports_[id];
    if (vp.view        != VK_NULL_HANDLE) vkDestroyImageView(dev_.device_, vp.view,        nullptr);
    if (vp.depth_view  != VK_NULL_HANDLE) vkDestroyImageView(dev_.device_, vp.depth_view,  nullptr);
    if (vp.image       != VK_NULL_HANDLE) vmaDestroyImage(dev_.allocator_, vp.image,       vp.alloc);
    if (vp.depth_image != VK_NULL_HANDLE) vmaDestroyImage(dev_.allocator_, vp.depth_image, vp.depth_alloc);
    vp.view         = VK_NULL_HANDLE;
    vp.depth_view   = VK_NULL_HANDLE;
    vp.image        = VK_NULL_HANDLE;
    vp.depth_image  = VK_NULL_HANDLE;
    vp.alloc        = VK_NULL_HANDLE;
    vp.depth_alloc  = VK_NULL_HANDLE;
    vp.extent = {0, 0};
    vp.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    vp.depth_layout   = VK_IMAGE_LAYOUT_UNDEFINED;
}

void VulkanSwapchain::destroy_all_viewport_images() {
    for (u32 i = 0; i < viewports_.size(); ++i) destroy_viewport_image(i);
    viewports_.clear();
    viewport_count_ = 0u;
}

void VulkanSwapchain::set_viewport_count(u32 n) {
    if (n == viewport_count_) return;
    if (n < viewport_count_) {
        // Shrinking — drain the GPU then release trailing slots' resources.
        vkDeviceWaitIdle(dev_.device_);
        for (u32 i = n; i < viewport_count_; ++i) destroy_viewport_image(i);
        viewports_.resize(n);
    } else {
        viewports_.resize(n);
    }
    viewport_count_ = n;
    if (active_viewport_id_ >= viewport_count_ && viewport_count_ > 0u) {
        active_viewport_id_ = viewport_count_ - 1u;
    }
}

void VulkanSwapchain::set_active_viewport(u32 id) {
    if (id >= viewport_count_) {
        cardinal::log::errorf("rhi/vk",
            "set_active_viewport(%u) out of range (count=%u)", id, viewport_count_);
        return;
    }
    active_viewport_id_ = id;

    FrameSync& f = frames_[frame_index_];
    if (f.cmd == VK_NULL_HANDLE) return;   // outside a begin_frame/end_frame pair

    auto& vp = viewports_[id];
    // Lazy resize against the most recent set_viewport_size request.
    if (vp.pending.width != vp.extent.width || vp.pending.height != vp.extent.height) {
        ensure_viewport_image(id, vp.pending.width, vp.pending.height);
    }
    if (vp.image == VK_NULL_HANDLE || vp.extent.width == 0 || vp.extent.height == 0) {
        return;   // size 0 = panel hidden / not yet sized
    }

    // Close any in-flight render pass on the previous viewport before
    // opening a new one. (Vulkan disallows nested vkCmdBeginRendering.)
    if (rendering_open_) {
        vkCmdEndRendering(f.cmd);
        rendering_open_ = false;
    }

    // Transition viewport[id]'s COLOUR image to COLOR_ATTACHMENT_OPTIMAL
    // using whatever layout it ended last in (UNDEFINED on first use;
    // SHADER_READ_ONLY after end_frame's overlay-prep). Source-layout-
    // aware so we don't discard the previous frame's contents prematurely.
    //
    // We batch the colour barrier with the DEPTH barrier into a single
    // vkCmdPipelineBarrier2 call. Depth always transitions
    // {UNDEFINED|DEPTH_ATTACHMENT_OPTIMAL} → DEPTH_ATTACHMENT_OPTIMAL —
    // it gets cleared each frame so we don't care about preserving its
    // contents.
    cardinal::array<VkImageMemoryBarrier2, 2> bb{};
    bb[0].sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    bb[0].srcStageMask     = (vp.current_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
        : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    bb[0].srcAccessMask    = (vp.current_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
        : 0;
    bb[0].dstStageMask     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    bb[0].dstAccessMask    = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    bb[0].oldLayout        = (vp.current_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        : VK_IMAGE_LAYOUT_UNDEFINED;
    bb[0].newLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    bb[0].image            = vp.image;
    bb[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    bb[1].sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    bb[1].srcStageMask     = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
                           | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    bb[1].srcAccessMask    = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    bb[1].dstStageMask     = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
                           | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    bb[1].dstAccessMask    = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                           | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    bb[1].oldLayout        = (vp.depth_layout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
        ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
        : VK_IMAGE_LAYOUT_UNDEFINED;
    bb[1].newLayout        = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    bb[1].image            = vp.depth_image;
    bb[1].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 2;
    dep.pImageMemoryBarriers    = bb.data();
    vkCmdPipelineBarrier2(f.cmd, &dep);
    vp.current_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    vp.depth_layout   = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;

    // Open dynamic rendering on viewport[id] with colour + depth.
    // Colour: CLEAR to begin_frame's stash. Depth: CLEAR to 1.0 (far
    // plane), STORE off (we never sample the depth — overlay reads
    // colour only and the next frame clears depth again).
    VkRenderingAttachmentInfo cat{};
    cat.sType            = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    cat.imageView        = vp.view;
    cat.imageLayout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    cat.loadOp           = VK_ATTACHMENT_LOAD_OP_CLEAR;
    cat.storeOp          = VK_ATTACHMENT_STORE_OP_STORE;
    cat.clearValue.color = {{ last_clear_color_[0], last_clear_color_[1],
                              last_clear_color_[2], last_clear_color_[3] }};

    VkRenderingAttachmentInfo dat{};
    dat.sType            = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    dat.imageView        = vp.depth_view;
    dat.imageLayout      = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    dat.loadOp           = VK_ATTACHMENT_LOAD_OP_CLEAR;
    dat.storeOp          = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    dat.clearValue.depthStencil = { 1.0f, 0u };

    VkRenderingInfo ri{};
    ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea           = {{0, 0}, vp.extent};
    ri.layerCount           = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments    = &cat;
    ri.pDepthAttachment     = &dat;
    vkCmdBeginRendering(f.cmd, &ri);
    rendering_open_ = true;

    // Re-issue viewport / scissor for this slot's size (HLSL +Y up convention).
    VkViewport vkvp{};
    vkvp.x        = 0.0f;
    vkvp.y        = static_cast<float>(vp.extent.height);
    vkvp.width    = static_cast<float>(vp.extent.width);
    vkvp.height   = -static_cast<float>(vp.extent.height);
    vkvp.minDepth = 0.0f;
    vkvp.maxDepth = 1.0f;
    vkCmdSetViewport(f.cmd, 0, 1, &vkvp);
    VkRect2D scissor{{0, 0}, vp.extent};
    vkCmdSetScissor(f.cmd, 0, 1, &scissor);

    vp.rendered_this_frame = true;
}

bool VulkanSwapchain::create_surface(void* native_window) {
#if CARDINAL_PLATFORM_WINDOWS
    // vcpkg's volk doesn't unconditionally export the platform-specific
    // surface entrypoints, so resolve manually via the loader. (volk's own
    // symbol table only includes them when volk.c was compiled with the
    // corresponding VK_USE_PLATFORM_* defines.)
    auto vkCreateWin32SurfaceKHR_fn =
        reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
            vkGetInstanceProcAddr(dev_.instance_, "vkCreateWin32SurfaceKHR"));
    if (vkCreateWin32SurfaceKHR_fn == nullptr) {
        cardinal::log::errorf("rhi/vk", "vkCreateWin32SurfaceKHR not exposed by loader");
        return false;
    }

    VkWin32SurfaceCreateInfoKHR sci{};
    sci.sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    sci.hinstance = GetModuleHandleW(nullptr);
    sci.hwnd      = static_cast<HWND>(native_window);
    return vk_check(
        vkCreateWin32SurfaceKHR_fn(dev_.instance_, &sci, nullptr, &surface_),
        "vkCreateWin32SurfaceKHR");
#else
    (void)native_window;
    cardinal::log::errorf("rhi/vk", "WSI surface not implemented for this platform");
    return false;
#endif
}

bool VulkanSwapchain::create_swapchain_objects(u32 width, u32 height) {
    // Verify the chosen graphics queue supports presentation on this surface.
    VkBool32 supported = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(
        dev_.physical_, dev_.graphics_queue_family_, surface_, &supported);
    if (supported == VK_FALSE) {
        cardinal::log::errorf("rhi/vk", "graphics queue does not support presentation");
        return false;
    }

    // Pick format: prefer 32-bit BGRA SRGB, else first available.
    u32 fmt_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev_.physical_, surface_, &fmt_count, nullptr);
    cardinal::vector<VkSurfaceFormatKHR> formats(fmt_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev_.physical_, surface_, &fmt_count, formats.data());
    VkSurfaceFormatKHR chosen = formats[0];
    for (auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
            f.colorSpace == VK_COLORSPACE_SRGB_NONLINEAR_KHR) {
            chosen = f;
            break;
        }
    }
    format_      = chosen.format;
    color_space_ = chosen.colorSpace;

    // Pick present mode based on the requested vsync_interval. The
    // mapping mirrors what D3D12 does with sync_interval + ALLOW_TEARING:
    //
    //   vsync_interval = 0 (uncapped)   → IMMEDIATE if available, else MAILBOX,
    //                                     else FIFO. IMMEDIATE allows tearing
    //                                     and is the only Vulkan mode that
    //                                     can exceed display refresh rate.
    //   vsync_interval = 1 (vsync)      → FIFO (locked to refresh, no tearing).
    //   vsync_interval > 1 (half/third) → FIFO_RELAXED if available + we'll
    //                                     emit the same frame multiple times
    //                                     by skipping presents (handled at the
    //                                     pacer/loop level, not here).
    //
    // Without IMMEDIATE Vulkan caps Present at the display refresh rate
    // (FIFO behaves like vsync-on, MAILBOX is triple-buffered but still
    // capped). This was the equivalent of the D3D12 ALLOW_TEARING bug.
    u32 pm_count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(dev_.physical_, surface_, &pm_count, nullptr);
    cardinal::vector<VkPresentModeKHR> modes(pm_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(dev_.physical_, surface_, &pm_count, modes.data());

    auto has = [&](VkPresentModeKHR m) {
        for (auto x : modes) if (x == m) return true;
        return false;
    };
    immediate_supported_ = has(VK_PRESENT_MODE_IMMEDIATE_KHR);
    mailbox_supported_   = has(VK_PRESENT_MODE_MAILBOX_KHR);
    fifo_relaxed_supported_ = has(VK_PRESENT_MODE_FIFO_RELAXED_KHR);

    if (vsync_interval_ == 0u) {
        // Uncapped — IMMEDIATE first, MAILBOX as a fallback (still capped,
        // but at least triple-buffered low-latency), FIFO last resort.
        present_mode_ = immediate_supported_ ? VK_PRESENT_MODE_IMMEDIATE_KHR
                       : mailbox_supported_  ? VK_PRESENT_MODE_MAILBOX_KHR
                                             : VK_PRESENT_MODE_FIFO_KHR;
    } else {
        // VSync requested — FIFO is universally available; FIFO_RELAXED
        // tears on late frames which is the better default once vsync is on.
        present_mode_ = fifo_relaxed_supported_ ? VK_PRESENT_MODE_FIFO_RELAXED_KHR
                                                : VK_PRESENT_MODE_FIFO_KHR;
    }
    cardinal::log::infof("rhi/vk",
        "Present mode: %s  (interval=%u, IMMEDIATE=%s MAILBOX=%s FIFO_RELAXED=%s)",
        present_mode_name(present_mode_), vsync_interval_,
        immediate_supported_   ? "yes" : "no",
        mailbox_supported_     ? "yes" : "no",
        fifo_relaxed_supported_? "yes" : "no");

    // Surface capabilities.
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev_.physical_, surface_, &caps);

    extent_ = caps.currentExtent;
    if (extent_.width == 0xFFFFFFFFu) {
        extent_.width  = cardinal::clamp(width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
        extent_.height = cardinal::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }

    min_image_count_ = caps.minImageCount;
    u32 image_count  = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && image_count > caps.maxImageCount) {
        image_count = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR sc{};
    sc.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sc.surface          = surface_;
    sc.minImageCount    = image_count;
    sc.imageFormat      = format_;
    sc.imageColorSpace  = color_space_;
    sc.imageExtent      = extent_;
    sc.imageArrayLayers = 1;
    sc.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sc.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sc.preTransform     = caps.currentTransform;
    sc.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sc.presentMode      = present_mode_;
    sc.clipped          = VK_TRUE;

    if (!vk_check(
            vkCreateSwapchainKHR(dev_.device_, &sc, nullptr, &swapchain_),
            "vkCreateSwapchainKHR")) {
        return false;
    }

    u32 actual = 0;
    vkGetSwapchainImagesKHR(dev_.device_, swapchain_, &actual, nullptr);
    images_.resize(actual);
    vkGetSwapchainImagesKHR(dev_.device_, swapchain_, &actual, images_.data());

    views_.resize(actual);
    for (u32 i = 0; i < actual; ++i) {
        VkImageViewCreateInfo iv{};
        iv.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        iv.image                       = images_[i];
        iv.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
        iv.format                      = format_;
        iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        iv.subresourceRange.levelCount = 1;
        iv.subresourceRange.layerCount = 1;
        if (!vk_check(
                vkCreateImageView(dev_.device_, &iv, nullptr, &views_[i]),
                "vkCreateImageView")) {
            return false;
        }
    }
    return true;
}

bool VulkanSwapchain::create_per_frame_objects() {
    for (auto& f : frames_) {
        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (!vk_check(vkCreateSemaphore(dev_.device_, &si, nullptr, &f.image_available), "vkCreateSemaphore")) return false;
        if (!vk_check(vkCreateSemaphore(dev_.device_, &si, nullptr, &f.render_finished), "vkCreateSemaphore")) return false;

        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // start signalled so first wait returns immediately
        if (!vk_check(vkCreateFence(dev_.device_, &fi, nullptr, &f.in_flight), "vkCreateFence")) return false;

        VkCommandPoolCreateInfo pi{};
        pi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pi.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                              VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pi.queueFamilyIndex = dev_.graphics_queue_family_;
        if (!vk_check(vkCreateCommandPool(dev_.device_, &pi, nullptr, &f.pool), "vkCreateCommandPool")) return false;

        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = f.pool;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (!vk_check(vkAllocateCommandBuffers(dev_.device_, &ai, &f.cmd), "vkAllocateCommandBuffers")) return false;
    }
    return true;
}

void VulkanSwapchain::destroy_per_frame_objects() {
    for (auto& f : frames_) {
        for (VkDescriptorPool dp : f.desc_pools)
            vkDestroyDescriptorPool(dev_.device_, dp, nullptr);
        if (f.cmd  != VK_NULL_HANDLE) vkFreeCommandBuffers(dev_.device_, f.pool, 1, &f.cmd);
        if (f.pool != VK_NULL_HANDLE) vkDestroyCommandPool(dev_.device_, f.pool, nullptr);
        if (f.in_flight       != VK_NULL_HANDLE) vkDestroyFence(dev_.device_, f.in_flight, nullptr);
        if (f.render_finished != VK_NULL_HANDLE) vkDestroySemaphore(dev_.device_, f.render_finished, nullptr);
        if (f.image_available != VK_NULL_HANDLE) vkDestroySemaphore(dev_.device_, f.image_available, nullptr);
        f = {};
    }
}

void VulkanSwapchain::destroy_swapchain_objects() {
    for (auto v : views_) {
        if (v != VK_NULL_HANDLE) vkDestroyImageView(dev_.device_, v, nullptr);
    }
    views_.clear();
    images_.clear();
    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(dev_.device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

// ---- resize ---------------------------------------------------------------
// Idle the device, tear down the existing VkSwapchainKHR + image views, then
// recreate at the new size. The surface stays — only the swapchain changes.
// Per-frame command buffers / sync objects are still valid (they target the
// device, not specific images), so we keep them.
bool VulkanSwapchain::resize(u32 new_w, u32 new_h) {
    if (new_w == 0 || new_h == 0) return false;
    if (new_w == extent_.width && new_h == extent_.height) return true;

    vkDeviceWaitIdle(dev_.device_);
    destroy_swapchain_objects();

    if (!create_swapchain_objects(new_w, new_h)) {
        cardinal::log::errorf("rhi/vk", "swapchain resize failed at %ux%u", new_w, new_h);
        return false;
    }
    extent_.width  = new_w;
    extent_.height = new_h;
    cardinal::log::infof("rhi/vk", "swapchain resized to %ux%u", new_w, new_h);
    if (on_rebuilt_) on_rebuilt_();
    return true;
}

// Tear down + rebuild the swapchain at the current size with the new
// vsync_interval. Called from begin_frame when vsync_change_pending_ is
// set — guarantees the device is between submits at that point.
void VulkanSwapchain::apply_pending_vsync_change() {
    if (!vsync_change_pending_) return;
    vsync_change_pending_ = false;
    if (extent_.width == 0 || extent_.height == 0) return;

    const u32 w = extent_.width;
    const u32 h = extent_.height;
    vkDeviceWaitIdle(dev_.device_);
    destroy_swapchain_objects();
    if (!create_swapchain_objects(w, h)) {
        cardinal::log::errorf("rhi/vk",
            "swapchain rebuild after vsync change failed (interval=%u)",
            vsync_interval_);
        return;
    }
    extent_.width  = w;
    extent_.height = h;
    cardinal::log::infof("rhi/vk",
        "Swapchain rebuilt for vsync_interval=%u (present mode now %s)",
        vsync_interval_, present_mode_name(present_mode_));
    // Notify consumers — Studio rebuilds its per-image VkFramebuffers,
    // future per-image RTV/SRV consumers re-attach. Without this fire
    // the next overlay record dereferences freed image views and
    // crashes the process. This was the "1:2 vsync mode crashes" path
    // before the fix landed.
    if (on_rebuilt_) on_rebuilt_();
}

// =============================================================================
// Recording API on the swapchain (Phase 2.5-D).
// =============================================================================
void VulkanSwapchain::bind_pipeline(Pipeline* p) {
    auto* vp = static_cast<VulkanPipeline*>(p);
    bound_pipeline_ = p;
    // New pipeline → its descriptor set is independent; drop any
    // storage-buffer state accumulated for the previous pipeline so a
    // stale slot can't leak into the next set.
    for (auto& s : pending_sb_) s = nullptr;
    for (auto& t : pending_st_) t = nullptr;
    vkCmdBindPipeline(frames_[frame_index_].cmd,
                      VK_PIPELINE_BIND_POINT_GRAPHICS, vp->handle());
}

void VulkanSwapchain::bind_vertex_buffer(Buffer* b, usize offset) {
    auto* vb = static_cast<VulkanBuffer*>(b);
    VkBuffer       buf = vb->handle();
    VkDeviceSize   off = offset;
    vkCmdBindVertexBuffers(frames_[frame_index_].cmd, 0, 1, &buf, &off);
}

void VulkanSwapchain::draw(u32 vertex_count, u32 instance_count,
                           u32 first_vertex, u32 first_instance) {
    vkCmdDraw(frames_[frame_index_].cmd,
              vertex_count, instance_count, first_vertex, first_instance);
}

void VulkanSwapchain::set_push_constants(u32 offset, const void* data, u32 size) {
    if (bound_pipeline_ == nullptr) return;
    auto* vp = static_cast<VulkanPipeline*>(bound_pipeline_);
    const u32 declared = vp->push_constant_size();
    if (declared == 0) return;                       // no-op for pipelines without a block
    if (offset + size > declared) {
        // Range out of bounds — drop and log once. Matches the doc'd
        // contract; avoids a vkCmdPushConstants validation-layer error.
        static bool warned = false;
        if (!warned) {
            cardinal::log::warnf("rhi/vk",
                "set_push_constants out-of-range: offset=%u size=%u declared=%u (dropped)",
                offset, size, declared);
            warned = true;
        }
        return;
    }
    vkCmdPushConstants(frames_[frame_index_].cmd,
                       vp->layout(),
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       offset, size, data);
}

void VulkanSwapchain::begin_shadow_pass(Texture* depth) {
    auto* vt = static_cast<VulkanTexture*>(depth);
    if (vt == nullptr || vt->image() == VK_NULL_HANDLE) return;
    VkCommandBuffer cmd = frames_[frame_index_].cmd;

    // Suspend the main viewport's dynamic-rendering scope (Vulkan
    // disallows nested vkCmdBeginRendering). The renderer calls this
    // FIRST in render() — before any scene draw — so nothing's been
    // written to the viewport yet and end_shadow_pass can re-open it
    // cleanly via set_active_viewport (CLEAR ops reproduce the exact
    // pre-shadow state).
    shadow_suspended_ = rendering_open_;
    if (rendering_open_) {
        vkCmdEndRendering(cmd);
        rendering_open_ = false;
    }

    // Transition whatever layout the depth image is in → depth
    // attachment optimal for the depth-only write pass.
    VkImageMemoryBarrier toAtt{};
    toAtt.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toAtt.oldLayout        = vt->layout_;
    toAtt.newLayout        = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    toAtt.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toAtt.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toAtt.image            = vt->image();
    toAtt.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
    toAtt.srcAccessMask    = VK_ACCESS_SHADER_READ_BIT;
    toAtt.dstAccessMask    = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
            | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toAtt);
    vt->layout_ = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;

    const VkExtent2D ext = vt->extent();
    VkRenderingAttachmentInfo dat{};
    dat.sType                       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    dat.imageView                   = vt->view();
    dat.imageLayout                 = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    dat.loadOp                      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    dat.storeOp                     = VK_ATTACHMENT_STORE_OP_STORE;
    dat.clearValue.depthStencil     = { 1.0f, 0 };

    VkRenderingInfo ri{};
    ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.extent    = ext;
    ri.layerCount           = 1;
    ri.colorAttachmentCount = 0;             // depth-only
    ri.pDepthAttachment     = &dat;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vpr{ 0.0f, 0.0f,
                    static_cast<float>(ext.width),
                    static_cast<float>(ext.height), 0.0f, 1.0f };
    VkRect2D   sci{ {0, 0}, ext };
    vkCmdSetViewport(cmd, 0, 1, &vpr);
    vkCmdSetScissor (cmd, 0, 1, &sci);

    shadow_tex_     = depth;
    rendering_open_ = true;
}

void VulkanSwapchain::end_shadow_pass() {
    if (shadow_tex_ == nullptr) return;
    VkCommandBuffer cmd = frames_[frame_index_].cmd;
    vkCmdEndRendering(cmd);
    rendering_open_ = false;

    // Depth-write → shader-readable so the main pass can sample it.
    auto* vt = static_cast<VulkanTexture*>(shadow_tex_);
    VkImageMemoryBarrier toRead{};
    toRead.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toRead.oldLayout        = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    toRead.newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toRead.image            = vt->image();
    toRead.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
    toRead.srcAccessMask    = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    toRead.dstAccessMask    = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toRead);
    vt->layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    shadow_tex_ = nullptr;

    // Resume the main viewport scope we suspended in begin_shadow_pass.
    // set_active_viewport re-issues the correct colour+depth barriers
    // and re-opens dynamic rendering (CLEAR ops) — valid here because
    // the shadow pass ran before any scene geometry hit the viewport.
    if (shadow_suspended_) {
        shadow_suspended_ = false;
        set_active_viewport(active_viewport_id_);
    }
}

// Allocate one transient descriptor set covering set 0 and (re)write
// every pending storage buffer + sampled texture, then bind it. Called
// after each bind_storage_buffer / bind_sampled_texture so a pipeline
// using both (e.g. lights@0 + materials@1 + shadow-map sampled@2) ends
// up with a single complete set, not partial sets clobbering each
// other. The last call before the draw wins.
void VulkanSwapchain::rebuild_and_bind_descriptor_set_() {
    if (bound_pipeline_ == nullptr) return;
    auto* vp = static_cast<VulkanPipeline*>(bound_pipeline_);
    const VkDescriptorSetLayout dsl = vp->descriptor_set_layout();
    if (dsl == VK_NULL_HANDLE) return;             // push-only pipeline

    FrameSync& f = frames_[frame_index_];

    // Per-frame transient pool *chain*. A single fixed pool can't cover a
    // multi-viewport editor's per-frame set count — and the engine uses
    // up to kMaxStorageSlots storage + kMaxStorageSlots samplers per set,
    // so the old 64-set / 64-descriptor pool actually ran dry at ~32 sets
    // (2 storage each) and then silently dropped the draw's descriptors,
    // GPU-faulting. Append a fresh block whenever the current one is
    // exhausted; all blocks are reset (not freed) at begin_frame, so the
    // chain self-tunes to the workload and never silently under-allocates.
    constexpr u32 kBlockSets = 64;
    auto make_pool = [&]() -> VkDescriptorPool {
        VkDescriptorPoolSize ps[2]{};
        ps[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps[0].descriptorCount = kBlockSets * kMaxStorageSlots;
        ps[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps[1].descriptorCount = kBlockSets * kMaxStorageSlots;
        VkDescriptorPoolCreateInfo pci{};
        pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets       = kBlockSets;
        pci.poolSizeCount = 2;
        pci.pPoolSizes    = ps;
        VkDescriptorPool p = VK_NULL_HANDLE;
        if (!vk_check(vkCreateDescriptorPool(dev_.device_, &pci, nullptr, &p),
                      "vkCreateDescriptorPool")) {
            return VK_NULL_HANDLE;
        }
        return p;
    };

    if (f.desc_pools.empty()) {
        VkDescriptorPool p = make_pool();
        if (p == VK_NULL_HANDLE) return;
        f.desc_pools.push_back(p);
    }

    VkDescriptorSetAllocateInfo dai{};
    dai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts        = &dsl;
    VkDescriptorSet set = VK_NULL_HANDLE;
    for (;;) {
        dai.descriptorPool = f.desc_pools[f.desc_pool_cur];
        const VkResult r = vkAllocateDescriptorSets(dev_.device_, &dai, &set);
        if (r == VK_SUCCESS) break;
        if (r != VK_ERROR_OUT_OF_POOL_MEMORY &&
            r != VK_ERROR_FRAGMENTED_POOL) {
            vk_check(r, "vkAllocateDescriptorSets");
            return;
        }
        // Current block is full this frame — advance to / append the next.
        ++f.desc_pool_cur;
        if (f.desc_pool_cur >= f.desc_pools.size()) {
            VkDescriptorPool p = make_pool();
            if (p == VK_NULL_HANDLE) return;
            f.desc_pools.push_back(p);
        }
    }

    const u32 nstore = vp->storage_buffer_slots();
    const u32 ntex   = vp->sampled_texture_slots();
    VkDescriptorBufferInfo dbi[kMaxStorageSlots]{};
    VkDescriptorImageInfo  dii[kMaxStorageSlots]{};
    VkWriteDescriptorSet   wr [kMaxStorageSlots * 2]{};
    u32 nw = 0;
    for (u32 s = 0; s < nstore && s < kMaxStorageSlots; ++s) {
        if (pending_sb_[s] == nullptr) continue;
        auto* vbuf      = static_cast<VulkanBuffer*>(pending_sb_[s]);
        dbi[s].buffer   = vbuf->handle();
        dbi[s].offset   = 0;
        dbi[s].range    = VK_WHOLE_SIZE;
        wr[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[nw].dstSet          = set;
        wr[nw].dstBinding      = s;
        wr[nw].descriptorCount = 1;
        wr[nw].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[nw].pBufferInfo     = &dbi[s];
        ++nw;
    }
    for (u32 s = 0; s < ntex && s < kMaxStorageSlots; ++s) {
        if (pending_st_[s] == nullptr) continue;
        auto* vtex      = static_cast<VulkanTexture*>(pending_st_[s]);
        dii[s].sampler     = dev_.default_sampler_;
        dii[s].imageView   = vtex->view();
        dii[s].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        wr[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[nw].dstSet          = set;
        wr[nw].dstBinding      = nstore + s;        // textures follow buffers
        wr[nw].descriptorCount = 1;
        wr[nw].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        wr[nw].pImageInfo      = &dii[s];
        ++nw;
    }
    if (nw > 0) {
        vkUpdateDescriptorSets(dev_.device_, nw, wr, 0, nullptr);
    }
    vkCmdBindDescriptorSets(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            vp->layout(), 0, 1, &set, 0, nullptr);
}

void VulkanSwapchain::bind_storage_buffer(u32 slot, Buffer* b) {
    if (bound_pipeline_ == nullptr || b == nullptr) return;
    auto* vp = static_cast<VulkanPipeline*>(bound_pipeline_);
    if (vp->descriptor_set_layout() == VK_NULL_HANDLE) return;
    if (slot >= vp->storage_buffer_slots() || slot >= kMaxStorageSlots) {
        static bool warned = false;
        if (!warned) {
            cardinal::log::warnf("rhi/vk",
                "bind_storage_buffer slot=%u >= declared=%u (dropped)",
                slot, vp->storage_buffer_slots());
            warned = true;
        }
        return;
    }
    pending_sb_[slot] = b;
    rebuild_and_bind_descriptor_set_();
}

void VulkanSwapchain::bind_sampled_texture(u32 slot, Texture* tex) {
    if (bound_pipeline_ == nullptr || tex == nullptr) return;
    auto* vp = static_cast<VulkanPipeline*>(bound_pipeline_);
    if (vp->descriptor_set_layout() == VK_NULL_HANDLE) return;
    if (slot >= vp->sampled_texture_slots() || slot >= kMaxStorageSlots) {
        static bool warned = false;
        if (!warned) {
            cardinal::log::warnf("rhi/vk",
                "bind_sampled_texture slot=%u >= declared=%u (dropped)",
                slot, vp->sampled_texture_slots());
            warned = true;
        }
        return;
    }
    pending_st_[slot] = tex;
    rebuild_and_bind_descriptor_set_();
}

bool VulkanSwapchain::initialize(void* native_window, u32 width, u32 height) {
    if (!create_surface(native_window))            return false;
    if (!create_swapchain_objects(width, height))  return false;
    if (!create_per_frame_objects())               return false;
    return true;
}

u32 VulkanSwapchain::begin_frame(float r, float g, float b, float a) {
    // Apply pending vsync change BEFORE acquiring an image. Doing so here
    // means the rebuilt swapchain is what the rest of the frame uses.
    apply_pending_vsync_change();

    FrameSync& f = frames_[frame_index_];

    vkWaitForFences(dev_.device_, 1, &f.in_flight, VK_TRUE, UINT64_MAX);
    vkResetFences(dev_.device_, 1, &f.in_flight);

    // The fence wait above guarantees the GPU finished this frame
    // slot's previous submission, so every transient descriptor set
    // allocated from it (kFramesInFlight ago) is now safe to recycle.
    // One reset is far cheaper than per-set frees.
    for (VkDescriptorPool dp : f.desc_pools)
        vkResetDescriptorPool(dev_.device_, dp, 0);
    f.desc_pool_cur = 0;          // refill from the first block again

    vkAcquireNextImageKHR(dev_.device_, swapchain_, UINT64_MAX,
        f.image_available, VK_NULL_HANDLE, &acquired_image_);

    vkResetCommandBuffer(f.cmd, 0);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(f.cmd, &bi);

    // Stash the clear color for set_active_viewport() to reuse on each
    // per-viewport clear (each panel gets a fresh CLEAR loadOp).
    last_clear_color_[0] = r; last_clear_color_[1] = g;
    last_clear_color_[2] = b; last_clear_color_[3] = a;

    // Reset per-viewport rendered-this-frame flags so end_frame's overlay
    // prep only transitions the slots the host actually drew into.
    for (auto& vp : viewports_) vp.rendered_this_frame = false;
    rendering_open_ = false;

    // Open viewport 0 by default so legacy hosts (single-viewport renderers
    // that just call begin_frame → render → end_frame, no set_active_viewport)
    // keep working unchanged. Multi-viewport hosts immediately call
    // set_active_viewport(id) per panel; that closes this default rendering
    // and opens its own.
    if (viewport_count_ > 0u) {
        set_active_viewport(0u);
    } else {
        // No viewports configured — render straight into the swapchain image
        // (legacy headless / no-editor path). COLOUR ONLY — no depth
        // attachment here. Pipelines created with PipelineDesc::depth_format
        // != Unknown CANNOT be bound in this context (Vulkan validation
        // requires the pipeline's declared attachment formats to match the
        // active rendering pass). The scene renderer always goes through
        // viewports, so its depth-test pipelines never reach this path.
        // Headless tests that bind only colour-only pipelines work fine.
        VkImageMemoryBarrier2 b1{};
        b1.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b1.srcStageMask     = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        b1.dstStageMask     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        b1.dstAccessMask    = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        b1.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
        b1.newLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b1.image            = images_[acquired_image_];
        b1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b1;
        vkCmdPipelineBarrier2(f.cmd, &dep);

        VkRenderingAttachmentInfo att{};
        att.sType            = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        att.imageView        = views_[acquired_image_];
        att.imageLayout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        att.loadOp           = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att.storeOp          = VK_ATTACHMENT_STORE_OP_STORE;
        att.clearValue.color = {{r, g, b, a}};

        VkRenderingInfo ri{};
        ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea           = {{0, 0}, extent_};
        ri.layerCount           = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments    = &att;
        vkCmdBeginRendering(f.cmd, &ri);
        rendering_open_ = true;

        VkViewport vp{};
        vp.x        = 0.0f;
        vp.y        = static_cast<float>(extent_.height);
        vp.width    = static_cast<float>(extent_.width);
        vp.height   = -static_cast<float>(extent_.height);
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(f.cmd, 0, 1, &vp);
        VkRect2D scissor{{0, 0}, extent_};
        vkCmdSetScissor(f.cmd, 0, 1, &scissor);
    }

    return acquired_image_;
}

void VulkanSwapchain::end_frame() {
    FrameSync& f = frames_[frame_index_];

    // Close any open dynamic render (the last set_active_viewport's
    // vkCmdBeginRendering, or the no-viewports default that targets the
    // swapchain image directly).
    if (rendering_open_) {
        vkCmdEndRendering(f.cmd);
        rendering_open_ = false;
    }

    const bool any_viewport_rendered = [&]{
        for (const auto& vp : viewports_) if (vp.rendered_this_frame) return true;
        return false;
    }();

    if (any_viewport_rendered && overlay_cb_ != nullptr) {
        // Multi-viewport + overlay: transition every rendered viewport to
        // SHADER_READ_ONLY so ImGui can sample them inside the overlay's
        // render pass. Batch all transitions into one vkCmdPipelineBarrier2.
        // Per-frame transient — back with the thread-local end-frame arena
        // so push_back is a lock-free atomic bump, not a malloc.
        auto& arena = vk_end_frame_arena();
        arena.reset();
        using BarrierAlloc = cardinal::core::ArenaAllocator<VkImageMemoryBarrier2>;
        cardinal::vector<VkImageMemoryBarrier2, BarrierAlloc> barriers(BarrierAlloc{arena});
        barriers.reserve(viewport_count_ + 1);
        for (auto& vp : viewports_) {
            if (!vp.rendered_this_frame || vp.image == VK_NULL_HANDLE) continue;
            VkImageMemoryBarrier2 bv{};
            bv.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            bv.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            bv.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            bv.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            bv.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            bv.oldLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            bv.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            bv.image         = vp.image;
            bv.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            barriers.push_back(bv);
            vp.current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        // Plus the swapchain image: UNDEFINED -> COLOR_ATTACHMENT (overlay
        // clears + draws on top).
        VkImageMemoryBarrier2 bs{};
        bs.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        bs.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        bs.dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        bs.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        bs.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        bs.newLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        bs.image         = images_[acquired_image_];
        bs.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barriers.push_back(bs);

        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = static_cast<u32>(barriers.size());
        dep.pImageMemoryBarriers    = barriers.data();
        vkCmdPipelineBarrier2(f.cmd, &dep);
    } else if (overlay_cb_ == nullptr && !any_viewport_rendered) {
        // No overlay AND no per-viewport render — direct path where the
        // swapchain image was the render target itself. Already in
        // COLOR_ATTACHMENT_OPTIMAL from begin_frame's no-viewports branch;
        // transition to PRESENT_SRC for vkQueuePresentKHR.
        VkImageMemoryBarrier2 b2{};
        b2.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b2.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        b2.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        b2.dstStageMask  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        b2.oldLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b2.newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        b2.image         = images_[acquired_image_];
        b2.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b2;
        vkCmdPipelineBarrier2(f.cmd, &dep);
    }
    // (overlay + no-viewport-rendered: nothing to do — overlay will run
    // and the swapchain image already in COLOR_ATTACHMENT_OPTIMAL from
    // the no-viewports begin_frame branch.)

    // Run overlay (e.g. ImGui) on the swapchain image while it's in
    // COLOR_ATTACHMENT_OPTIMAL. Overlay must end with image in
    // COLOR_ATTACHMENT_OPTIMAL — we transition to PRESENT_SRC after.
    if (overlay_cb_ != nullptr) {
        overlay_cb_(overlay_user_);

        VkImageMemoryBarrier2 bp{};
        bp.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        bp.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        bp.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        bp.dstStageMask  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        bp.dstAccessMask = 0;
        bp.oldLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        bp.newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        bp.image         = images_[acquired_image_];
        bp.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &bp;
        vkCmdPipelineBarrier2(f.cmd, &dep);
    }

    vkEndCommandBuffer(f.cmd);

    // Submit via synchronization2.
    VkSemaphoreSubmitInfo wait{};
    wait.sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    wait.semaphore = f.image_available;
    wait.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSemaphoreSubmitInfo signal{};
    signal.sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signal.semaphore = f.render_finished;
    signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkCommandBufferSubmitInfo cmd{};
    cmd.sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cmd.commandBuffer = f.cmd;

    VkSubmitInfo2 si{};
    si.sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    si.waitSemaphoreInfoCount   = 1;
    si.pWaitSemaphoreInfos      = &wait;
    si.commandBufferInfoCount   = 1;
    si.pCommandBufferInfos      = &cmd;
    si.signalSemaphoreInfoCount = 1;
    si.pSignalSemaphoreInfos    = &signal;

    vkQueueSubmit2(dev_.graphics_queue_, 1, &si, f.in_flight);

    VkPresentInfoKHR pi{};
    pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &f.render_finished;
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &swapchain_;
    pi.pImageIndices      = &acquired_image_;
    vkQueuePresentKHR(dev_.graphics_queue_, &pi);

    frame_index_ = (frame_index_ + 1) % frames_in_flight;
}

// =============================================================================
// Async compute (AEGIS Block 10) — timeline-semaphore Fence, the compute-list
// recorder, and the ComputeQueue that submits onto the dedicated compute queue.
//
// NOTE: compute *dispatch* (bind compute pipeline / dispatch) is a separate
// feature still pending in both backends, so the recorder's dispatch() stays
// the base no-op for now — buffer copies, UAV barriers and the cross-queue
// timeline handshake are functional. Dispatch lights up here with no async-lane
// changes once compute pipelines land.
// =============================================================================

// Timeline-semaphore Fence. wait_cpu blocks on vkWaitSemaphores; current_value
// reads the counter. bump() hands signal_fence an auto-incrementing value.
class VulkanFence final : public Fence {
public:
    bool initialize(VkDevice dev, u64 initial) {
        dev_ = dev;
        VkSemaphoreTypeCreateInfo ti{};
        ti.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        ti.initialValue  = initial;
        VkSemaphoreCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        ci.pNext = &ti;
        next_    = initial + 1u;
        return vkCreateSemaphore(dev, &ci, nullptr, &sem_) == VK_SUCCESS;
    }
    ~VulkanFence() override { if (sem_) vkDestroySemaphore(dev_, sem_, nullptr); }

    u64 wait_cpu(u64 value, u64 timeout_ns) override {
        if (!sem_) return 0;
        VkSemaphoreWaitInfo wi{};
        wi.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wi.semaphoreCount = 1;
        wi.pSemaphores    = &sem_;
        wi.pValues        = &value;
        vkWaitSemaphores(dev_, &wi, timeout_ns == 0 ? ~0ull : timeout_ns);
        return current_value();
    }
    u64 current_value() const noexcept override {
        u64 v = 0;
        if (sem_) vkGetSemaphoreCounterValue(dev_, sem_, &v);
        return v;
    }
    VkSemaphore sem()  const noexcept { return sem_; }
    u64         bump()       noexcept { return next_++; }

private:
    VkDevice    dev_{VK_NULL_HANDLE};
    VkSemaphore sem_{VK_NULL_HANDLE};
    u64         next_{1};
};

// Records compute-queue-valid ops (buffer copies, UAV barriers) onto `cmd_`.
class VulkanComputeRecorder final : public Swapchain {
public:
    VulkanComputeRecorder(VkDevice dev, VkCommandBuffer cmd) : dev_(dev), cmd_(cmd) {}

    void copy_buffer(Buffer* src, usize src_off,
                     Buffer* dst, usize dst_off, usize size) override {
        auto* s = static_cast<VulkanBuffer*>(src);
        auto* d = static_cast<VulkanBuffer*>(dst);
        if (!cmd_ || !s || !d) return;
        VkBufferCopy region{ src_off, dst_off, size };
        vkCmdCopyBuffer(cmd_, s->handle(), d->handle(), 1, &region);
    }
    void uav_barrier(Buffer*) override {
        if (!cmd_) return;
        VkMemoryBarrier2 mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo dep{};
        dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers    = &mb;
        vkCmdPipelineBarrier2(cmd_, &dep);
    }

    // ---- present / frame / viewport surface — never reached here ----
    u32    width()  const noexcept override { return 0; }
    u32    height() const noexcept override { return 0; }
    Format color_format() const noexcept override { return Format::B8G8R8A8_UNORM; }
    void   set_vsync(bool) override {}
    bool   vsync() const noexcept override { return false; }
    void   set_vsync_interval(u32) override {}
    u32    vsync_interval() const noexcept override { return 0; }
    bool   resize(u32, u32) override { return false; }
    void   set_on_rebuilt(OnRebuilt) override {}
    void   set_viewport_size(u32, u32) override {}
    u32    viewport_width()  const noexcept override { return 0; }
    u32    viewport_height() const noexcept override { return 0; }
    void   set_viewport_size(u32, u32, u32) override {}
    u32    viewport_width (u32) const noexcept override { return 0; }
    u32    viewport_height(u32) const noexcept override { return 0; }
    void   set_viewport_count(u32) override {}
    u32    viewport_count() const noexcept override { return 0; }
    void   set_active_viewport(u32) override {}
    u32    active_viewport() const noexcept override { return 0; }
    u32    begin_frame(float, float, float, float) override { return 0; }
    void   end_frame() override {}
    void   set_overlay(OverlayCallback, void*) override {}
    void   bind_pipeline(Pipeline*) override {}
    void   bind_vertex_buffer(Buffer*, usize) override {}
    void   draw(u32, u32, u32, u32) override {}
    void   set_push_constants(u32, const void*, u32) override {}

private:
    VkDevice        dev_{VK_NULL_HANDLE};
    VkCommandBuffer cmd_{VK_NULL_HANDLE};
};

// Owns a command pool + buffer on the compute family; submit() records the
// caller's work and submits onto the async-compute queue with a timeline signal.
class VulkanComputeQueue final : public ComputeQueue {
public:
    bool initialize(VkDevice dev, VkQueue queue, u32 family) {
        if (!dev || queue == VK_NULL_HANDLE) return false;
        dev_ = dev; queue_ = queue;
        VkCommandPoolCreateInfo pci{};
        pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = family;
        if (vkCreateCommandPool(dev, &pci, nullptr, &pool_) != VK_SUCCESS) return false;
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = pool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(dev, &ai, &cmd_) != VK_SUCCESS) return false;
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        return vkCreateFence(dev, &fci, nullptr, &done_) == VK_SUCCESS;
    }
    ~VulkanComputeQueue() override {
        if (done_) vkDestroyFence(dev_, done_, nullptr);
        if (pool_) vkDestroyCommandPool(dev_, pool_, nullptr);   // frees cmd_
    }

    u64 submit(RecordFn record, void* user,
               Fence* signal_fence, u64 signal_value) override {
        if (queue_ == VK_NULL_HANDLE || cmd_ == VK_NULL_HANDLE) return 0;

        // Wait for the PREVIOUS submission to retire before reusing the buffer.
        if (submitted_) {
            vkWaitForFences(dev_, 1, &done_, VK_TRUE, ~0ull);
            vkResetFences(dev_, 1, &done_);
        }

        vkResetCommandBuffer(cmd_, 0);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd_, &bi);
        if (record) {
            VulkanComputeRecorder rec(dev_, cmd_);
            record(&rec, user);
        }
        vkEndCommandBuffer(cmd_);

        VkCommandBufferSubmitInfo csi{};
        csi.sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        csi.commandBuffer = cmd_;

        VkSemaphoreSubmitInfo sig{};
        bool have_sig = false;
        if (auto* vf = static_cast<VulkanFence*>(signal_fence); vf && vf->sem()) {
            sig.sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            sig.semaphore = vf->sem();
            sig.value     = signal_value;     // timeline target value
            sig.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            have_sig      = true;
        }

        VkSubmitInfo2 si{};
        si.sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        si.commandBufferInfoCount   = 1;
        si.pCommandBufferInfos      = &csi;
        si.signalSemaphoreInfoCount = have_sig ? 1u : 0u;
        si.pSignalSemaphoreInfos    = have_sig ? &sig : nullptr;
        vkQueueSubmit2(queue_, 1, &si, done_);
        submitted_ = true;
        return signal_value;
    }

private:
    VkDevice        dev_{VK_NULL_HANDLE};
    VkQueue         queue_{VK_NULL_HANDLE};   // device-owned compute queue
    VkCommandPool   pool_{VK_NULL_HANDLE};
    VkCommandBuffer cmd_{VK_NULL_HANDLE};
    VkFence         done_{VK_NULL_HANDLE};    // internal reuse fence
    bool            submitted_{false};
};

// ---- VulkanDevice async-compute factories ----
cardinal::unique_ptr<Fence> VulkanDevice::create_fence(u64 initial_value) {
    auto f = cardinal::make_unique<VulkanFence>();
    if (!f->initialize(device_, initial_value)) return nullptr;
    return f;
}

cardinal::unique_ptr<ComputeQueue> VulkanDevice::create_async_compute_queue() {
    if (compute_queue_ == VK_NULL_HANDLE) return nullptr;   // caps.async_compute == false
    auto q = cardinal::make_unique<VulkanComputeQueue>();
    if (!q->initialize(device_, compute_queue_, compute_queue_family_)) return nullptr;
    return q;
}

// ---- VulkanSwapchain graphics-queue side of the timeline handshake ----
u64 VulkanSwapchain::signal_fence(Fence* f) {
    auto* vf = static_cast<VulkanFence*>(f);
    if (!vf || !vf->sem()) return 0;
    const u64 v = vf->bump();
    VkSemaphoreSubmitInfo sig{};
    sig.sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    sig.semaphore = vf->sem();
    sig.value     = v;
    sig.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSubmitInfo2 si{};
    si.sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    si.signalSemaphoreInfoCount = 1;
    si.pSignalSemaphoreInfos    = &sig;
    vkQueueSubmit2(dev_.graphics_queue_, 1, &si, VK_NULL_HANDLE);
    return v;
}

void VulkanSwapchain::wait_fence(Fence* f, u64 value) {
    auto* vf = static_cast<VulkanFence*>(f);
    if (!vf || !vf->sem()) return;
    VkSemaphoreSubmitInfo wait{};
    wait.sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    wait.semaphore = vf->sem();
    wait.value     = value;
    wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSubmitInfo2 si{};
    si.sType                  = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    si.waitSemaphoreInfoCount = 1;
    si.pWaitSemaphoreInfos    = &wait;
    vkQueueSubmit2(dev_.graphics_queue_, 1, &si, VK_NULL_HANDLE);
}

// =============================================================================
// Factory entry point referenced by rhi.cpp.
// =============================================================================
cardinal::unique_ptr<Device> create_vulkan_device(const DeviceDesc& desc) {
    auto dev = cardinal::make_unique<VulkanDevice>();
    if (!dev->initialize(desc)) return nullptr;
    return dev;
}

// =============================================================================
// Vulkan interop — implementations of cardinal/rhi/vulkan_interop.hpp.
// Live here because they need access to the (anonymous-namespace) concrete
// classes. The header forward-declares; we cast based on backend().
// =============================================================================
}  // namespace cardinal::rhi

#include <cardinal/rhi/vulkan_interop.hpp>

namespace cardinal::rhi {

VulkanDeviceHandles vulkan_handles(Device* dev) {
    VulkanDeviceHandles h{};
    if (dev == nullptr || dev->backend() != Backend::Vulkan) return h;
    auto* vd = static_cast<VulkanDevice*>(dev);
    h.instance              = vd->vk_instance();
    h.physical_device       = vd->vk_physical();
    h.device                = vd->vk_device();
    h.graphics_queue        = vd->vk_graphics_queue();
    h.graphics_queue_family = vd->vk_graphics_queue_family();
    return h;
}

VulkanSwapchainHandles vulkan_handles(Swapchain* sw) {
    VulkanSwapchainHandles h{};
    if (sw == nullptr) return h;
    auto* vs = static_cast<VulkanSwapchain*>(sw);
    h.swapchain        = vs->vk_swapchain();
    h.color_format     = vs->vk_color_format();
    h.image_count      = vs->vk_image_count();
    h.min_image_count  = vs->vk_min_image_count();
    return h;
}

VkCommandBuffer current_command_buffer(Swapchain* sw) {
    if (sw == nullptr) return VK_NULL_HANDLE;
    auto* vs = static_cast<VulkanSwapchain*>(sw);
    return vs->vk_current_cmd();
}

u32 vulkan_acquired_image_index(Swapchain* sw) {
    if (sw == nullptr) return 0;
    auto* vs = static_cast<VulkanSwapchain*>(sw);
    return vs->vk_acquired_image_index();
}

VkImage vulkan_swapchain_image(Swapchain* sw, u32 index) {
    if (sw == nullptr) return VK_NULL_HANDLE;
    auto* vs = static_cast<VulkanSwapchain*>(sw);
    return vs->vk_image(index);
}

VkImageView vulkan_swapchain_view(Swapchain* sw, u32 index) {
    if (sw == nullptr) return VK_NULL_HANDLE;
    auto* vs = static_cast<VulkanSwapchain*>(sw);
    return vs->vk_view(index);
}

VkFormat vulkan_viewport_format(Swapchain* sw) {
    if (sw == nullptr) return VK_FORMAT_UNDEFINED;
    auto* vs = static_cast<VulkanSwapchain*>(sw);
    return vs->vk_viewport_format();
}

VkImage vulkan_viewport_image(Swapchain* sw) {
    if (sw == nullptr) return VK_NULL_HANDLE;
    auto* vs = static_cast<VulkanSwapchain*>(sw);
    return vs->vk_viewport_image();
}

VkImage vulkan_viewport_image(Swapchain* sw, u32 id) {
    // Per-id viewport image. Each editor panel binds its own SRV via the
    // matching VkImageView returned by vulkan_viewport_view(sw, id).
    if (sw == nullptr) return VK_NULL_HANDLE;
    return static_cast<VulkanSwapchain*>(sw)->vk_viewport_image(id);
}

VkImageView vulkan_viewport_view(Swapchain* sw, u32 id) {
    // Per-id viewport view. The host renders into viewport[id] via
    // set_active_viewport(id) + scene render, then ImGui samples this view
    // inside the overlay's render pass.
    if (sw == nullptr) return VK_NULL_HANDLE;
    return static_cast<VulkanSwapchain*>(sw)->vk_viewport_view(id);
}

VkImageView vulkan_viewport_view(Swapchain* sw) {
    if (sw == nullptr) return VK_NULL_HANDLE;
    auto* vs = static_cast<VulkanSwapchain*>(sw);
    return vs->vk_viewport_view();
}

}  // namespace cardinal::rhi
