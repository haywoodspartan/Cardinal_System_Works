#include "vulkan_internal.hpp"

// Async compute (AEGIS Block 10): timeline-semaphore Fence, compute recorder, ComputeQueue, Device/Swapchain fence ops.
namespace cardinal::rhi {

// =============================================================================
// Async compute (AEGIS Block 10) — timeline-semaphore Fence, the compute-list
// recorder, and the ComputeQueue that submits onto the dedicated compute queue.
//
// The recorder is dispatch-complete: bind_pipeline, storage/UAV descriptor
// binds (per-submit pool owned by the queue), push constants, dispatch,
// copies, UAV barriers and the cross-queue timeline handshake — enough to run
// any AEGIS kernel off the graphics queue (and for the headless GPU
// validation harness in tests/gpu_compute).
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

// Records compute-queue-valid ops (buffer copies, UAV barriers, full compute
// bind/push/dispatch) onto `cmd_`. Descriptor sets are allocated from the
// owning queue's per-submit pool (`pool`, reset by the queue after the
// previous submission retires) — sets stay alive until the GPU is done.
class VulkanComputeRecorder final : public Swapchain {
public:
    VulkanComputeRecorder(VkDevice dev, VkCommandBuffer cmd, VkDescriptorPool pool)
        : dev_(dev), cmd_(cmd), pool_(pool) {}

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
    void transition_buffer_state(Buffer* b, ResourceState before,
                                 ResourceState after) override {
        auto* vb = static_cast<VulkanBuffer*>(b);
        if (!cmd_ || !vb || before == after) return;
        auto access = [](ResourceState s) -> VkAccessFlags2 {
            switch (s) {
                case ResourceState::UnorderedAccess:
                    return VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                case ResourceState::ShaderResource: return VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                case ResourceState::CopySource:     return VK_ACCESS_2_TRANSFER_READ_BIT;
                case ResourceState::CopyDest:       return VK_ACCESS_2_TRANSFER_WRITE_BIT;
                default:                            return 0;
            }
        };
        auto stage = [](ResourceState s) -> VkPipelineStageFlags2 {
            switch (s) {
                case ResourceState::CopySource:
                case ResourceState::CopyDest: return VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                default:                      return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            }
        };
        VkBufferMemoryBarrier2 bb{};
        bb.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        bb.srcStageMask        = stage(before);
        bb.srcAccessMask       = access(before);
        bb.dstStageMask        = stage(after);
        bb.dstAccessMask       = access(after);
        bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.buffer              = vb->handle();
        bb.offset              = 0;
        bb.size                = VK_WHOLE_SIZE;
        VkDependencyInfo dep{};
        dep.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.bufferMemoryBarrierCount = 1;
        dep.pBufferMemoryBarriers    = &bb;
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
    // Real compute recording on the async list: bind/push/dispatch plus
    // storage-buffer descriptor binds via the queue's per-submit pool.
    void   bind_pipeline(Pipeline* p) override {
        auto* vp = static_cast<VulkanPipeline*>(p);
        if (cmd_ == VK_NULL_HANDLE || vp == nullptr || !vp->is_compute()) return;
        bound_ = vp;
        for (auto& b : pending_sb_)  b = nullptr;   // new pipeline, clean slate
        for (auto& b : pending_uav_) b = nullptr;
        desc_dirty_ = false;
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, vp->handle());
    }
    void   bind_storage_buffer(u32 slot, Buffer* b) override {
        if (bound_ == nullptr || b == nullptr) return;
        if (slot >= bound_->storage_buffer_slots() || slot >= kMaxSlots_) return;
        pending_sb_[slot] = b;
        desc_dirty_ = true;
    }
    void   bind_storage_buffer_uav(u32 slot, Buffer* b) override {
        if (bound_ == nullptr || b == nullptr) return;
        if (slot >= bound_->uav_slots() || slot >= kMaxSlots_) return;
        pending_uav_[slot] = b;
        desc_dirty_ = true;
    }
    void   dispatch(u32 gx, u32 gy = 1, u32 gz = 1) override {
        if (cmd_ == VK_NULL_HANDLE) return;
        flush_descriptors_();
        // A failed descriptor flush leaves the set unbound/stale — dispatching
        // through it is undefined. Drop the dispatch instead (logged once by
        // the flush) rather than record garbage.
        if (desc_dirty_) return;
        vkCmdDispatch(cmd_, gx, gy, gz);
    }
    void   set_push_constants(u32 offset, const void* data, u32 size) override {
        if (cmd_ == VK_NULL_HANDLE || bound_ == nullptr ||
            bound_->push_constant_size() == 0) return;
        vkCmdPushConstants(cmd_, bound_->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                           offset, size, data);
    }
    void   bind_vertex_buffer(Buffer*, usize) override {}
    void   draw(u32, u32, u32, u32) override {}

private:
    // Build + bind one descriptor set from the pending tables — deferred to
    // dispatch so a run of bind calls costs one set, mirroring the main-queue
    // layout: read-only SSBOs at bindings [0, nstore), UAVs at [nstore, +uav).
    void flush_descriptors_() {
        if (!desc_dirty_ || bound_ == nullptr || pool_ == VK_NULL_HANDLE) return;
        VkDescriptorSetLayout dsl = bound_->descriptor_set_layout();
        if (dsl == VK_NULL_HANDLE) { desc_dirty_ = false; return; }  // push-only kernel

        VkDescriptorSetAllocateInfo dai{};
        dai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool     = pool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts        = &dsl;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(dev_, &dai, &set) != VK_SUCCESS) {
            cardinal::log::errorf("rhi/vk", "async descriptor-set alloc failed");
            return;
        }

        const u32 nstore = bound_->storage_buffer_slots();
        const u32 nuav   = bound_->uav_slots();
        VkDescriptorBufferInfo dbi[kMaxSlots_]{};
        VkDescriptorBufferInfo dbu[kMaxSlots_]{};
        VkWriteDescriptorSet   wr [kMaxSlots_ * 2]{};
        u32 nw = 0;
        for (u32 s = 0; s < nstore && s < kMaxSlots_; ++s) {
            if (pending_sb_[s] == nullptr) continue;
            dbi[s].buffer = static_cast<VulkanBuffer*>(pending_sb_[s])->handle();
            dbi[s].range  = VK_WHOLE_SIZE;
            wr[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[nw].dstSet          = set;
            wr[nw].dstBinding      = s;
            wr[nw].descriptorCount = 1;
            wr[nw].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[nw].pBufferInfo     = &dbi[s];
            ++nw;
        }
        for (u32 s = 0; s < nuav && s < kMaxSlots_; ++s) {
            if (pending_uav_[s] == nullptr) continue;
            dbu[s].buffer = static_cast<VulkanBuffer*>(pending_uav_[s])->handle();
            dbu[s].range  = VK_WHOLE_SIZE;
            wr[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[nw].dstSet          = set;
            wr[nw].dstBinding      = kUavBindingBase + s;   // matches -fvk-u-shift
            wr[nw].descriptorCount = 1;
            wr[nw].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[nw].pBufferInfo     = &dbu[s];
            ++nw;
        }
        if (nw > 0) vkUpdateDescriptorSets(dev_, nw, wr, 0, nullptr);
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                bound_->layout(), 0, 1, &set, 0, nullptr);
        desc_dirty_ = false;
    }

    static constexpr u32 kMaxSlots_ = 16;   // matches swapchain kMaxStorageSlots

    VkDevice         dev_{VK_NULL_HANDLE};
    VkCommandBuffer  cmd_{VK_NULL_HANDLE};
    VkDescriptorPool pool_{VK_NULL_HANDLE};
    VulkanPipeline*  bound_{nullptr};
    Buffer*          pending_sb_ [kMaxSlots_]{};
    Buffer*          pending_uav_[kMaxSlots_]{};
    bool             desc_dirty_{false};
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
        if (vkCreateFence(dev, &fci, nullptr, &done_) != VK_SUCCESS) return false;
        // Per-submit descriptor pool for the recorder's storage-buffer binds.
        // Reset after each previous-submission wait, so sets live exactly as
        // long as the GPU may read them.
        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = 256u * 32u;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = 256;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes    = &ps;
        return vkCreateDescriptorPool(dev, &dpci, nullptr, &desc_pool_) == VK_SUCCESS;
    }
    ~VulkanComputeQueue() override {
        if (desc_pool_) vkDestroyDescriptorPool(dev_, desc_pool_, nullptr);
        if (done_) vkDestroyFence(dev_, done_, nullptr);
        if (pool_) vkDestroyCommandPool(dev_, pool_, nullptr);   // frees cmd_
    }

    u64 submit(RecordFn record, void* user,
               Fence* signal_fence, u64 signal_value) override {
        if (queue_ == VK_NULL_HANDLE || cmd_ == VK_NULL_HANDLE) return 0;

        // Wait for the PREVIOUS submission to retire before reusing the buffer
        // (and its descriptor sets — safe to reset the pool only after this).
        if (submitted_) {
            vkWaitForFences(dev_, 1, &done_, VK_TRUE, ~0ull);
            vkResetFences(dev_, 1, &done_);
        }
        if (desc_pool_) vkResetDescriptorPool(dev_, desc_pool_, 0);

        vkResetCommandBuffer(cmd_, 0);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd_, &bi);
        if (record) {
            VulkanComputeRecorder rec(dev_, cmd_, desc_pool_);
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
    VkDevice         dev_{VK_NULL_HANDLE};
    VkQueue          queue_{VK_NULL_HANDLE};   // device-owned compute queue
    VkCommandPool    pool_{VK_NULL_HANDLE};
    VkCommandBuffer  cmd_{VK_NULL_HANDLE};
    VkFence          done_{VK_NULL_HANDLE};    // internal reuse fence
    VkDescriptorPool desc_pool_{VK_NULL_HANDLE};  // recorder SSBO sets, per-submit
    bool             submitted_{false};
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
