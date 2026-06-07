#include "vulkan_internal.hpp"

// VulkanSwapchain recording surface: bind / draw / push / shadow pass / descriptor set / storage / sampled.
namespace cardinal::rhi {

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


}  // namespace cardinal::rhi
