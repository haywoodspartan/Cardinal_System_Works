#pragma once

// =============================================================================
// D3D12 interop — escape hatch for middleware (Dear ImGui, RenderDoc helpers,
// profilers) that needs raw D3D12 / DXGI handles. Mirrors vulkan_interop.hpp
// in spirit but is opt-in: include only when the consumer can handle the
// extra include surface (d3d12.h + dxgi1_6.h).
//
// All accessors return null/zero when the underlying device/swapchain is
// not D3D12-backed; callers should `if (dev->backend() == Backend::D3D12)`
// before reaching for these.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/rhi/rhi.hpp>

#include <d3d12.h>
#include <dxgi1_6.h>

namespace cardinal::rhi {

struct D3D12DeviceHandles {
    ID3D12Device*       device{nullptr};
    ID3D12CommandQueue* gfx_queue{nullptr};
    IDXGIFactory6*      factory{nullptr};
};

struct D3D12SwapchainHandles {
    IDXGISwapChain3* swapchain{nullptr};
    DXGI_FORMAT      back_buffer_format{DXGI_FORMAT_UNKNOWN};
    u32              back_buffer_count{0};
};

D3D12DeviceHandles    d3d12_handles(Device*    dev);
D3D12SwapchainHandles d3d12_handles(Swapchain* sw);

// Per-frame plumbing — only valid between begin_frame() and end_frame() of
// THIS frame. Used by middleware that records its own commands.
ID3D12GraphicsCommandList* d3d12_current_cmd       (Swapchain* sw);
u32                        d3d12_acquired_image    (Swapchain* sw);
ID3D12Resource*            d3d12_back_buffer       (Swapchain* sw, u32 index);
D3D12_CPU_DESCRIPTOR_HANDLE d3d12_back_buffer_rtv  (Swapchain* sw, u32 index);

// Off-screen viewport (set_viewport_size != 0,0). Returns nullptr / zero
// handle if no viewport is configured.
//
// The single-arg form returns viewport 0 (the default / legacy slot); the
// two-arg form returns viewport `id` from the multi-viewport pool. Out-of-
// range ids return nullptr.
ID3D12Resource*             d3d12_viewport_image   (Swapchain* sw);
ID3D12Resource*             d3d12_viewport_image   (Swapchain* sw, u32 id);
u32                         d3d12_viewport_count   (Swapchain* sw);

}  // namespace cardinal::rhi
