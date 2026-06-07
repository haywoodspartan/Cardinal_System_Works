#include "vulkan_internal.hpp"

// Async compute (AEGIS Block 10): timeline-semaphore Fence, compute recorder, ComputeQueue, Device/Swapchain fence ops.
namespace cardinal::rhi {

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


}  // namespace cardinal::rhi
