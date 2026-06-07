#include "vulkan_internal.hpp"

// VulkanSwapchain core: surface / swapchain objects / viewports / per-frame / resize / init / begin_frame / end_frame.
namespace cardinal::rhi {

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


}  // namespace cardinal::rhi
