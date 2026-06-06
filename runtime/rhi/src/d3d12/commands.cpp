// =============================================================================
// Cardinal RHI — D3D12Swapchain command-recording surface.
//
// Split out of swapchain.cpp: everything the renderer records INTO the
// current frame's command list — pipeline/vertex/constant/resource binding,
// draws, the shadow depth pass, and GPU debug-marker regions. The swapchain
// core (init / present / resize / per-viewport RTT management) stays in
// swapchain.cpp; this TU is the per-draw hot path.
//
// Windows-only.
// =============================================================================

#if defined(_WIN32) && !defined(CARDINAL_NO_D3D12)

#include "internal.hpp"

#if CARDINAL_PLATFORM_WINDOWS

namespace cardinal::rhi {

// PIX / RenderDoc recognise ID3D12GraphicsCommandList::BeginEvent /
// SetMarker when the metadata word selects a string encoding: 1 = ANSI,
// 2 = UTF-16. We pass a plain ANSI label so captures show named regions
// without taking a WinPixEventRuntime dependency.
namespace { constexpr UINT kPixEventAnsiVersion = 1u; }

// -----------------------------------------------------------------------------
// Pipeline + resource binding
// -----------------------------------------------------------------------------
void D3D12Swapchain::bind_pipeline(Pipeline* p) {
    auto* dp = static_cast<D3D12Pipeline*>(p);
    if (dp == nullptr || dp == bound_pipeline_) return;
    Frame& f = frames_[frame_index_];
    f.cmd->SetGraphicsRootSignature(dp->root_signature());
    f.cmd->SetPipelineState(dp->pso());
    f.cmd->IASetPrimitiveTopology(dp->topology());
    bound_pipeline_ = p;
}

void D3D12Swapchain::bind_vertex_buffer(Buffer* b, usize offset) {
    bound_vbuf_        = b;
    bound_vbuf_offset_ = offset;
    if (b == nullptr || bound_pipeline_ == nullptr) return;
    auto* db = static_cast<D3D12Buffer*>(b);
    auto* dp = static_cast<D3D12Pipeline*>(bound_pipeline_);
    D3D12_VERTEX_BUFFER_VIEW vb{};
    vb.BufferLocation = db->resource()->GetGPUVirtualAddress() + offset;
    vb.SizeInBytes    = static_cast<UINT>(db->size() - offset);
    vb.StrideInBytes  = dp->vertex_stride();
    frames_[frame_index_].cmd->IASetVertexBuffers(0, 1, &vb);
}

void D3D12Swapchain::draw(u32 vc, u32 ic, u32 fv, u32 fi) {
    frames_[frame_index_].cmd->DrawInstanced(vc, ic, fv, fi);
}

void D3D12Swapchain::set_push_constants(u32 offset, const void* data, u32 size) {
    if (bound_pipeline_ == nullptr) return;
    auto* dp = static_cast<D3D12Pipeline*>(bound_pipeline_);
    const u32 declared = dp->push_constant_size();
    if (declared == 0) return;                    // no-op for pipelines without a block
    if (offset + size > declared) {
        static bool warned = false;
        if (!warned) {
            cardinal::log::warnf("rhi/d3d12",
                "set_push_constants out-of-range: offset=%u size=%u declared=%u (dropped)",
                offset, size, declared);
            warned = true;
        }
        return;
    }
    if (dp->push_is_cbv()) {
        // Large block → root CBV. Patch the CPU shadow at [offset,offset+size)
        // (supports partial / multi-call updates), then publish the whole
        // declared block into a fresh 256 B-aligned ring slice and point
        // root param 0 (b0) at it. A new slice per call keeps each draw's
        // constants live until the GPU consumes them.
        if (declared > kMaxPushBytes) {
            static bool warned = false;
            if (!warned) {
                cardinal::log::warnf("rhi/d3d12",
                    "push block %uB exceeds CBV ring slice cap %uB (dropped)",
                    declared, kMaxPushBytes);
                warned = true;
            }
            return;
        }
        cardinal::memcpy(push_shadow_.data() + offset, data, size);
        const u32 slot = (declared + (kCbAlign - 1u)) & ~(kCbAlign - 1u);
        if (cb_ring_offset_ + slot > kCbRingBytes) {
            cb_ring_offset_ = 0;          // defensive wrap (very busy frame)
        }
        u8* cpu = cb_ring_cpu_[frame_index_] + cb_ring_offset_;
        const D3D12_GPU_VIRTUAL_ADDRESS gpu =
            cb_ring_gpu_[frame_index_] + cb_ring_offset_;
        cardinal::memcpy(cpu, push_shadow_.data(), declared);
        cb_ring_offset_ += slot;
        frames_[frame_index_].cmd->SetGraphicsRootConstantBufferView(0, gpu);
        return;
    }

    // Small block → inline root constants, written in DWORDs. We round size
    // up for the count and pass byte-offset/4 for the destination index.
    // Caller is expected to align offset to a 4-byte boundary.
    const UINT num_values   = (size + 3u) / 4u;
    const UINT dest_dword   = offset / 4u;
    frames_[frame_index_].cmd->SetGraphicsRoot32BitConstants(
        0, num_values, data, dest_dword);
}

void D3D12Swapchain::bind_storage_buffer(u32 slot, Buffer* b) {
    if (bound_pipeline_ == nullptr || b == nullptr) return;
    auto* dp = static_cast<D3D12Pipeline*>(bound_pipeline_);
    if (slot >= dp->storage_buffer_slots()) {
        static bool warned = false;
        if (!warned) {
            cardinal::log::warnf("rhi/d3d12",
                "bind_storage_buffer slot=%u >= declared=%u (dropped)",
                slot, dp->storage_buffer_slots());
            warned = true;
        }
        return;
    }
    // Root SRV — binds straight from the buffer's GPU virtual address,
    // no descriptor heap (mirrors the Vulkan STORAGE_BUFFER descriptor).
    // Root index = storage_root_base() + slot (push block, when present,
    // owns index 0). HLSL side: StructuredBuffer<T> : register(t<slot>).
    auto* db = static_cast<D3D12Buffer*>(b);
    const UINT root_idx = dp->storage_root_base() + slot;
    frames_[frame_index_].cmd->SetGraphicsRootShaderResourceView(
        root_idx, db->resource()->GetGPUVirtualAddress());
}

void D3D12Swapchain::bind_sampled_texture(u32 slot, Texture* tex) {
    if (bound_pipeline_ == nullptr || tex == nullptr) return;
    auto* dp = static_cast<D3D12Pipeline*>(bound_pipeline_);
    if (slot >= dp->sampled_texture_slots()) {
        static bool warned = false;
        if (!warned) {
            cardinal::log::warnf("rhi/d3d12",
                "bind_sampled_texture slot=%u >= declared=%u (dropped)",
                slot, dp->sampled_texture_slots());
            warned = true;
        }
        return;
    }
    // Each D3D12Texture owns its own 1-entry shader-visible SRV heap.
    // Single-sampled-texture case (the shadow map = slot 0) is exact;
    // multiple distinct sampled textures in one pipeline would need a
    // shared heap — deferred until a use-case needs it. The descriptor
    // table root param (sampled_table_root) was appended after the push
    // block + storage root SRVs in create_pipeline.
    auto* dt   = static_cast<D3D12Texture*>(tex);
    auto* cmd  = frames_[frame_index_].cmd.Get();
    ID3D12DescriptorHeap* heaps[] = { dt->srv_heap() };
    cmd->SetDescriptorHeaps(1, heaps);
    cmd->SetGraphicsRootDescriptorTable(dp->sampled_table_root(),
                                        dt->srv_gpu_handle());
}

// -----------------------------------------------------------------------------
// Shadow depth pass
// -----------------------------------------------------------------------------
void D3D12Swapchain::begin_shadow_pass(Texture* depth) {
    auto* dt = static_cast<D3D12Texture*>(depth);
    if (dt == nullptr || dt->resource() == nullptr) return;
    auto* cmd = frames_[frame_index_].cmd.Get();

    // Whatever state the depth resource is in → DEPTH_WRITE for the pass.
    if (dt->state_ != D3D12_RESOURCE_STATE_DEPTH_WRITE) {
        D3D12_RESOURCE_BARRIER bar{};
        bar.Type  = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bar.Transition.pResource   = dt->resource();
        bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        bar.Transition.StateBefore = dt->state_;
        bar.Transition.StateAfter  = D3D12_RESOURCE_STATE_DEPTH_WRITE;
        cmd->ResourceBarrier(1, &bar);
        dt->state_ = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    }

    const D3D12_CPU_DESCRIPTOR_HANDLE dsv = dt->dsv_handle();
    cmd->OMSetRenderTargets(0, nullptr, FALSE, &dsv);   // depth-only
    cmd->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0,
                               0, nullptr);

    D3D12_VIEWPORT vpr{ 0.0f, 0.0f,
                        static_cast<float>(dt->width()),
                        static_cast<float>(dt->height()), 0.0f, 1.0f };
    D3D12_RECT     sci{ 0, 0,
                        static_cast<LONG>(dt->width()),
                        static_cast<LONG>(dt->height()) };
    cmd->RSSetViewports(1, &vpr);
    cmd->RSSetScissorRects(1, &sci);
    shadow_tex_ = depth;
}

void D3D12Swapchain::end_shadow_pass() {
    if (shadow_tex_ == nullptr) return;
    auto* dt  = static_cast<D3D12Texture*>(shadow_tex_);
    auto* cmd = frames_[frame_index_].cmd.Get();
    // DEPTH_WRITE → PIXEL_SHADER_RESOURCE so the main pass can sample it.
    D3D12_RESOURCE_BARRIER bar{};
    bar.Type  = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bar.Transition.pResource   = dt->resource();
    bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    bar.Transition.StateBefore = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    bar.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    cmd->ResourceBarrier(1, &bar);
    dt->state_  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    shadow_tex_ = nullptr;

    // Resume the main viewport target the shadow pass replaced via
    // OMSetRenderTargets(0,…). D3D12 has no dynamic-rendering nesting
    // rule (unlike Vulkan), so this is just a rebind — but routing
    // through set_active_viewport keeps the two backends symmetric
    // and re-establishes the RTV/DSV + viewport/scissor correctly.
    set_active_viewport(viewport_active_id_);
}

// -----------------------------------------------------------------------------
// GPU debug-marker regions — show as named scopes in PIX / RenderDoc /
// Nsight captures. begin/end_event must be balanced; insert_marker is a
// zero-duration label. No-ops harmlessly outside a frame (null cmd list).
// -----------------------------------------------------------------------------
void D3D12Swapchain::begin_event(const char* name) {
    auto* cmd = frames_[frame_index_].cmd.Get();
    if (cmd == nullptr || name == nullptr) return;
    cmd->BeginEvent(kPixEventAnsiVersion, name,
                    static_cast<UINT>(cardinal::strlen(name) + 1u));
}

void D3D12Swapchain::end_event() {
    auto* cmd = frames_[frame_index_].cmd.Get();
    if (cmd != nullptr) cmd->EndEvent();
}

void D3D12Swapchain::insert_marker(const char* name) {
    auto* cmd = frames_[frame_index_].cmd.Get();
    if (cmd == nullptr || name == nullptr) return;
    cmd->SetMarker(kPixEventAnsiVersion, name,
                   static_cast<UINT>(cardinal::strlen(name) + 1u));
}

}  // namespace cardinal::rhi

#endif  // CARDINAL_PLATFORM_WINDOWS
#endif  // _WIN32 && !CARDINAL_NO_D3D12
