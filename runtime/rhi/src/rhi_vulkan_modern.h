#pragma once

#include <cardinal/core/types.hpp>

// Internal header for VulkanDevice <-> modern-SDK shims. Only consumed
// inside the cardinal_rhi target; not part of the public RHI surface.

// Forward-declare Vulkan handle types as opaque so the rest of the RHI
// doesn't have to pull <vulkan/vulkan.h> just to talk to the SDK shims.
struct VkDevice_T;        using VkDevice_Opaque    = VkDevice_T*;
struct VkSwapchainKHR_T;  using VkSwapchainOpaque  = VkSwapchainKHR_T*;

namespace cardinal::rhi::modern {

bool init_streamline();    // returns false if SDK absent or init failed
void shutdown_streamline();

bool init_reflex();
void shutdown_reflex();

void log_linked_sdks();    // diagnostic output to stdout

// ---- Reflex per-device wiring (Vulkan) ------------------------------------
//
// Called from VulkanDevice once the VK_NV_low_latency2 extension has been
// enabled. Returns true on success; false when the SDK isn't linked or
// the call fails (driver too old, non-NVIDIA GPU, etc.). The swapchain
// queries reflex_device_ready() before emitting any markers.
bool init_reflex_for_device(void* vk_device);
void shutdown_reflex_for_device(void* vk_device);
bool reflex_device_ready() noexcept;

// Per-swapchain wiring — registers a swapchain handle with Reflex so
// NvLL_VK_LatencySleep / NvLL_VK_SetLatencyMarker can target it.
// `mode_off` is true when the user has set ReflexMode::Off — sleeps
// become no-ops, but markers still fire (useful for trace tools that
// want stage timings without the latency penalty).
void register_reflex_swapchain  (void* vk_device, void* vk_swapchain, bool mode_off);
void unregister_reflex_swapchain(void* vk_device, void* vk_swapchain);

// Per-frame marker emit. `marker_type` is the NvLL_VK marker enum
// integer (we don't include NvLL header here — see rhi_vulkan_modern.cpp
// for the mapping). `frame_id` is the engine's monotonic frame counter.
void reflex_set_marker(void* vk_device, void* vk_swapchain,
                       u32 marker_type, u64 frame_id) noexcept;

// Blocks until the driver says "now's the latency-optimal moment for
// the next simulate". No-op when not registered or not supported.
void reflex_latency_sleep(void* vk_device, void* vk_swapchain) noexcept;

// Marker-type enum mirror (kept in sync with NvLL_VK_LATENCY_MARKER_TYPE
// inside the .cpp). Exposed so the swapchain can pass a u32 without
// needing the NVAPI header.
enum class ReflexMarker : u32 {
    SimulationStart   = 0,
    SimulationEnd     = 1,
    RenderSubmitStart = 2,
    RenderSubmitEnd   = 3,
    PresentStart      = 4,
    PresentEnd        = 5,
    InputSample       = 6,
};

}  // namespace cardinal::rhi::modern
