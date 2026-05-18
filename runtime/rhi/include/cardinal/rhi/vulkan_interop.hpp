#pragma once

// =============================================================================
// Vulkan interop — escape hatch for middleware (Dear ImGui, RenderDoc helper
// libs, profilers) that needs raw Vulkan handles. The main rhi.hpp keeps
// abstract types; this header is opt-in and brings in <vulkan/vulkan.h>.
//
// Returns zero-initialised handles if the device/swapchain is not Vulkan-
// backed. Callers must check Device::backend() == Backend::Vulkan first.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/rhi/rhi.hpp>

// Use volk's headers — they re-export vulkan.h with VK_NO_PROTOTYPES so
// callers don't accidentally pull in libvulkan-1 prototypes.
#include <volk.h>

namespace cardinal::rhi {

struct VulkanDeviceHandles {
    VkInstance       instance{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    VkDevice         device{VK_NULL_HANDLE};
    VkQueue          graphics_queue{VK_NULL_HANDLE};
    u32              graphics_queue_family{0};
};

struct VulkanSwapchainHandles {
    VkSwapchainKHR swapchain{VK_NULL_HANDLE};
    VkFormat       color_format{VK_FORMAT_UNDEFINED};
    u32            image_count{0};
    u32            min_image_count{0};
};

VulkanDeviceHandles    vulkan_handles(Device* dev);
VulkanSwapchainHandles vulkan_handles(Swapchain* sw);

// Returns the VkCommandBuffer that the swapchain is currently recording into,
// i.e. valid between begin_frame() and end_frame() of THIS frame. Use to
// inject middleware draws (ImGui, debug overlays).
VkCommandBuffer current_command_buffer(Swapchain* sw);

// Swapchain image array + index of the most recently acquired image. UI
// systems use these to build per-image VkFramebuffers for a render pass
// and pick the right one in end_frame.
u32         vulkan_acquired_image_index(Swapchain* sw);
VkImage     vulkan_swapchain_image(Swapchain* sw, u32 index);
VkImageView vulkan_swapchain_view (Swapchain* sw, u32 index);

// Off-screen viewport image (created when set_viewport_size != 0). The view
// is in SHADER_READ_ONLY_OPTIMAL layout at the moment Studio's overlay
// callback runs (the swapchain transitions it before invoking the overlay
// when one is registered). Returns VK_NULL_HANDLE if no viewport exists.
VkFormat    vulkan_viewport_format(Swapchain* sw);
VkImage     vulkan_viewport_image (Swapchain* sw);
VkImageView vulkan_viewport_view  (Swapchain* sw);
// Multi-viewport variants — return the Nth viewport's resource. The Vulkan
// backend currently aliases all non-zero ids to viewport 0 (degraded path);
// d3d12 has the full per-viewport implementation. Out-of-range ids return
// VK_NULL_HANDLE so callers can probe.
VkImage     vulkan_viewport_image (Swapchain* sw, u32 id);
VkImageView vulkan_viewport_view  (Swapchain* sw, u32 id);

}  // namespace cardinal::rhi
