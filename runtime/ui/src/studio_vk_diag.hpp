#pragma once

// =============================================================================
// Cardinal UI — Vulkan diagnostic helper (Studio decomposition, step 1).
//
// Carved verbatim out of the studio.cpp monolith. Zero coupling to
// StudioImpl / the renderer / viewport / gizmo state — it is purely a
// VkResult logger, so moving it here cannot change any runtime
// behaviour. studio.cpp wires this as
// ImGui_ImplVulkan_InitInfo::CheckVkResultFn.
//
// Private UI-internal header (lives in runtime/ui/src, NOT exported).
// =============================================================================

#include <volk.h>   // VkResult — GPU SDK header (sanctioned exception)

namespace cardinal::ui::detail {

// ImGui's Vulkan backend invokes this for every Vk* call result. The
// first occurrence of an error code logs a full captured stack; identical
// repeats are heavily rate-limited (one line per ~1024 hits with a running
// count); VK_SUCCESS / VK_SUBOPTIMAL_KHR / VK_ERROR_OUT_OF_DATE_KHR are
// treated as benign (the backend handles swapchain rebuilds and they are
// noisy on resize).
void check_vk(VkResult r);

}  // namespace cardinal::ui::detail
