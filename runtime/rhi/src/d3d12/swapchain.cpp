// =============================================================================
// Cardinal RHI — D3D12Swapchain implementation.
//
// Out-of-line methods of the D3D12Swapchain class. Class declaration lives
// in internal.hpp; this TU provides the bodies. Methods include:
//
//   * initialize()              — IDXGISwapChain3 creation, RTV heap,
//                                  per-frame command allocators + lists
//   * begin_frame / end_frame   — Wait fence, swap RTV, transition states,
//                                  Present
//   * Reflex (NvAPI) integration: set_reflex_mode, reflex_marker, …
//   * Per-viewport off-screen RTT: ensure_viewport_objects, set_active_…
//   * Recording: bind_pipeline, draw, set_push_constants, bind_storage_buffer,
//                bind_sampled_texture, begin/end_shadow_pass, resize
//
// Windows-only.
// =============================================================================

#if defined(_WIN32) && !defined(CARDINAL_NO_D3D12)

#include "internal.hpp"

#if CARDINAL_PLATFORM_WINDOWS

namespace cardinal::rhi {

// D3D12Swapchain implementation
// -----------------------------------------------------------------------------
bool D3D12Swapchain::initialize(HWND hwnd, u32 w, u32 h) {
    hwnd_   = hwnd;
    extent_ = { w, h };

    DXGI_SWAP_CHAIN_DESC1 scd{};
    scd.Width            = w;
    scd.Height           = h;
    scd.Format           = DXGI_FORMAT_B8G8R8A8_UNORM;
    scd.SampleDesc.Count = 1;
    scd.BufferUsage      = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.BufferCount      = kFramesInFlight;
    scd.SwapEffect       = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scd.Scaling          = DXGI_SCALING_STRETCH;
    scd.AlphaMode        = DXGI_ALPHA_MODE_UNSPECIFIED;
    // ALLOW_TEARING — required for true uncapped present. Without it,
    // Present(0,0) still composites at the display refresh rate via DWM.
    // Cache the flag both at creation time AND on present (DXGI requires
    // the *same* flag at both call sites or it returns DXGI_ERROR_INVALID_CALL).
    scd.Flags            = dev_.tearing_supported()
                         ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
                         : 0u;
    swap_flags_          = scd.Flags;

    ComPtr<IDXGISwapChain1> sw1;
    HRESULT hr = dev_.factory()->CreateSwapChainForHwnd(
        dev_.gfx_queue(), hwnd, &scd, nullptr, nullptr, &sw1);
    if (FAILED(hr)) {
        cardinal::log::errorf("rhi/d3d12", "CreateSwapChainForHwnd failed (%s)", hr_str(hr));
        return false;
    }
    if (FAILED(sw1.As(&swap_))) return false;
    dev_.factory()->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
    frame_index_ = swap_->GetCurrentBackBufferIndex();

    // RTV heap (2 entries, one per back buffer).
    D3D12_DESCRIPTOR_HEAP_DESC rtvd{};
    rtvd.NumDescriptors = kFramesInFlight;
    rtvd.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    if (FAILED(dev_.device()->CreateDescriptorHeap(&rtvd, IID_PPV_ARGS(&rtv_heap_)))) {
        cardinal::log::errorf("rhi/d3d12", "RTV heap creation failed");
        return false;
    }
    rtv_stride_ = dev_.device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    auto rtv_handle = rtv_heap_->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < kFramesInFlight; ++i) {
        if (FAILED(swap_->GetBuffer(i, IID_PPV_ARGS(&back_buffers_[i])))) return false;
        dev_.device()->CreateRenderTargetView(back_buffers_[i].Get(), nullptr, rtv_handle);
        rtv_handle.ptr += rtv_stride_;
    }

    if (!create_per_frame_objects()) return false;
    if (!ensure_depth_image(w, h))    return false;

    // Fence for CPU/GPU sync.
    if (FAILED(dev_.device()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)))) {
        cardinal::log::errorf("rhi/d3d12", "CreateFence failed");
        return false;
    }
    fence_event_ = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    if (fence_event_ == nullptr) return false;

    // Reflex availability — NvAPI must be initialised AND the adapter
    // must be NVIDIA (NvAPI_D3D_SetSleepMode returns NoImplementation on
    // non-NVIDIA cards, but probing here keeps the UI honest).
#if CARDINAL_HAS_NVAPI
    if (dev_.nvapi_initialised()) {
        // Issue a no-op SetSleepMode (mode = current) just to get a
        // success/failure signal from the driver for this exact ID3D12Device.
        NV_SET_SLEEP_MODE_PARAMS p{};
        p.version = NV_SET_SLEEP_MODE_PARAMS_VER;
        p.bLowLatencyMode  = FALSE;
        p.bLowLatencyBoost = FALSE;
        p.minimumIntervalUs = 0;
        const NvAPI_Status s = NvAPI_D3D_SetSleepMode(
            reinterpret_cast<IUnknown*>(dev_.device()), &p);
        reflex_supported_ = (s == NVAPI_OK);
        cardinal::log::infof("rhi/d3d12",
            "Reflex (D3D12) %s — NvAPI_D3D_SetSleepMode returned %d",
            reflex_supported_ ? "available" : "unsupported on this adapter",
            static_cast<int>(s));
    }
#endif

    cardinal::log::infof("rhi/d3d12", "Swapchain ready (%ux%u, %u buffers)", w, h, kFramesInFlight);
    return true;
}

// =============================================================================
// Reflex (NvAPI) implementation. Compiles to no-ops without CARDINAL_HAS_NVAPI
// or when NvAPI returned non-OK during init.
// =============================================================================
void D3D12Swapchain::set_reflex_mode(ReflexMode m) {
#if CARDINAL_HAS_NVAPI
    if (!reflex_supported_) { reflex_mode_ = m; return; }
    NV_SET_SLEEP_MODE_PARAMS p{};
    p.version           = NV_SET_SLEEP_MODE_PARAMS_VER;
    p.bLowLatencyMode   = (m != ReflexMode::Off);
    p.bLowLatencyBoost  = (m == ReflexMode::OnPlusBoost);
    p.minimumIntervalUs = (reflex_fps_cap_ > 0u)
                          ? (1'000'000u / reflex_fps_cap_) : 0u;
    p.bUseMarkersToOptimize = TRUE;
    const NvAPI_Status s = NvAPI_D3D_SetSleepMode(
        reinterpret_cast<IUnknown*>(dev_.device()), &p);
    if (s == NVAPI_OK) {
        reflex_mode_ = m;
    } else {
        cardinal::log::warnf("rhi/d3d12",
            "NvAPI_D3D_SetSleepMode(mode=%u) failed (%d)",
            static_cast<u32>(m), static_cast<int>(s));
    }
#else
    reflex_mode_ = m;
#endif
}

void D3D12Swapchain::set_reflex_fps_cap(u32 fps) {
    reflex_fps_cap_ = fps;
    set_reflex_mode(reflex_mode_);   // re-push (encodes minimumIntervalUs)
}

void D3D12Swapchain::reflex_sleep() {
#if CARDINAL_HAS_NVAPI
    if (!reflex_supported_) return;
    // It's recommended to call this every frame even when the mode is Off —
    // the driver may still apply control-panel-level FPS caps via this hook.
    NvAPI_D3D_Sleep(reinterpret_cast<IUnknown*>(dev_.device()));
#endif
}

void D3D12Swapchain::reflex_marker(ReflexMarker m, u64 frame_id) {
#if CARDINAL_HAS_NVAPI
    if (!reflex_supported_) return;
    NV_LATENCY_MARKER_PARAMS p{};
    p.version = NV_LATENCY_MARKER_PARAMS_VER;
    p.frameID = frame_id;
    switch (m) {
        case ReflexMarker::SimulationStart:   p.markerType = SIMULATION_START;   break;
        case ReflexMarker::SimulationEnd:     p.markerType = SIMULATION_END;     break;
        case ReflexMarker::RenderSubmitStart: p.markerType = RENDERSUBMIT_START; break;
        case ReflexMarker::RenderSubmitEnd:   p.markerType = RENDERSUBMIT_END;   break;
        case ReflexMarker::PresentStart:      p.markerType = PRESENT_START;      break;
        case ReflexMarker::PresentEnd:        p.markerType = PRESENT_END;        break;
        case ReflexMarker::InputSample:       p.markerType = INPUT_SAMPLE;       break;
        default: return;
    }
    NvAPI_D3D_SetLatencyMarker(
        reinterpret_cast<IUnknown*>(dev_.device()), &p);
#else
    (void)m; (void)frame_id;
#endif
}

Swapchain::ReflexLatency D3D12Swapchain::reflex_latency() const {
    Swapchain::ReflexLatency out{};
#if CARDINAL_HAS_NVAPI
    if (!reflex_supported_) return out;
    NV_LATENCY_RESULT_PARAMS r{};
    r.version = NV_LATENCY_RESULT_PARAMS_VER;
    if (NvAPI_D3D_GetLatency(
            reinterpret_cast<IUnknown*>(dev_.device()), &r) != NVAPI_OK) {
        return out;
    }
    // Average over the most recent 8 valid frames (the buffer holds 64,
    // newest at the end). NvAPI returns timestamps in microseconds; we
    // emit milliseconds for the UI.
    constexpr int kAvg = 8;
    int      n = 0;
    double   tot = 0, sim = 0, render = 0, present = 0, drv = 0, osq = 0, gpu = 0;
    for (int i = 64 - kAvg; i < 64; ++i) {
        const auto& f = r.frameReport[i];
        if (f.frameID == 0) continue;
        if (f.inputSampleTime == 0 || f.gpuRenderEndTime == 0) continue;
        tot     += static_cast<double>(f.gpuRenderEndTime  - f.inputSampleTime);
        sim     += static_cast<double>(f.simEndTime        - f.simStartTime);
        render  += static_cast<double>(f.renderSubmitEndTime - f.renderSubmitStartTime);
        present += static_cast<double>(f.presentEndTime    - f.presentStartTime);
        drv     += static_cast<double>(f.driverEndTime     - f.driverStartTime);
        osq     += static_cast<double>(f.osRenderQueueEndTime - f.osRenderQueueStartTime);
        gpu     += static_cast<double>(f.gpuRenderEndTime  - f.gpuRenderStartTime);
        ++n;
    }
    if (n == 0) return out;
    const double inv = 1.0 / (n * 1000.0);   // µs → ms
    out.valid              = true;
    out.total_latency_ms   = static_cast<float>(tot     * inv);
    out.sim_ms             = static_cast<float>(sim     * inv);
    out.render_ms          = static_cast<float>(render  * inv);
    out.present_ms         = static_cast<float>(present * inv);
    out.driver_ms          = static_cast<float>(drv     * inv);
    out.os_render_queue_ms = static_cast<float>(osq     * inv);
    out.gpu_ms             = static_cast<float>(gpu     * inv);
#endif
    return out;
}

// Allocate a D32_FLOAT depth resource + DSV. Used for both the back-buffer
// pass and (separately) the off-screen viewport image.
bool D3D12Swapchain::ensure_depth_image(u32 w, u32 h) {
    if (w == 0 || h == 0) return true;
    depth_image_.Reset();

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension          = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Alignment          = 0;
    rd.Width              = w;
    rd.Height             = h;
    rd.DepthOrArraySize   = 1;
    rd.MipLevels          = 1;
    rd.Format             = DXGI_FORMAT_D32_FLOAT;
    rd.SampleDesc.Count   = 1;
    rd.Layout             = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags              = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    D3D12_CLEAR_VALUE cv{};
    cv.Format               = DXGI_FORMAT_D32_FLOAT;
    cv.DepthStencil.Depth   = 1.0f;
    cv.DepthStencil.Stencil = 0;

    HRESULT hr = dev_.device()->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_DEPTH_WRITE, &cv, IID_PPV_ARGS(&depth_image_));
    if (FAILED(hr)) {
        cardinal::log::errorf("rhi/d3d12", "depth image alloc failed (%s)", hr_str(hr));
        return false;
    }

    if (dsv_heap_ == nullptr) {
        D3D12_DESCRIPTOR_HEAP_DESC dh{};
        dh.NumDescriptors = 1;
        dh.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        if (FAILED(dev_.device()->CreateDescriptorHeap(&dh, IID_PPV_ARGS(&dsv_heap_)))) {
            cardinal::log::errorf("rhi/d3d12", "depth DSV heap alloc failed");
            return false;
        }
    }

    D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
    dsv.Format        = DXGI_FORMAT_D32_FLOAT;
    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    dev_.device()->CreateDepthStencilView(depth_image_.Get(), &dsv,
        dsv_heap_->GetCPUDescriptorHandleForHeapStart());
    return true;
}

bool D3D12Swapchain::ensure_viewport_depth(u32 id, u32 w, u32 h) {
    if (id >= viewport_count_) return false;
    auto& vp = viewports_[id];

    // Same dimensions as the existing depth → no-op.
    if (vp.depth_image) {
        D3D12_RESOURCE_DESC d = vp.depth_image->GetDesc();
        if (static_cast<u32>(d.Width) == w &&
            static_cast<u32>(d.Height) == h) return true;
    }
    if (w == 0 || h == 0) { vp.depth_image.Reset(); return true; }
    // Idle the GPU before freeing the old depth.
    for (UINT i = 0; i < kFramesInFlight; ++i) wait_for_frame(i);
    vp.depth_image.Reset();

    D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC   rd{};
    rd.Dimension          = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width              = w;
    rd.Height             = h;
    rd.DepthOrArraySize   = 1;
    rd.MipLevels          = 1;
    rd.Format             = DXGI_FORMAT_D32_FLOAT;
    rd.SampleDesc.Count   = 1;
    rd.Layout             = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags              = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    D3D12_CLEAR_VALUE cv{};
    cv.Format = DXGI_FORMAT_D32_FLOAT;
    cv.DepthStencil.Depth = 1.0f;
    HRESULT hr = dev_.device()->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_DEPTH_WRITE, &cv, IID_PPV_ARGS(&vp.depth_image));
    if (FAILED(hr)) {
        cardinal::log::errorf("rhi/d3d12", "viewport[%u] depth alloc failed (%s)",
                              id, hr_str(hr));
        return false;
    }

    if (vp.dsv_heap == nullptr) {
        D3D12_DESCRIPTOR_HEAP_DESC dh{};
        dh.NumDescriptors = 1;
        dh.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        if (FAILED(dev_.device()->CreateDescriptorHeap(&dh, IID_PPV_ARGS(&vp.dsv_heap))))
            return false;
    }

    D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
    dsv.Format        = DXGI_FORMAT_D32_FLOAT;
    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    dev_.device()->CreateDepthStencilView(vp.depth_image.Get(), &dsv,
        vp.dsv_heap->GetCPUDescriptorHandleForHeapStart());
    return true;
}

bool D3D12Swapchain::create_per_frame_objects() {
    for (UINT i = 0; i < kFramesInFlight; ++i) {
        if (FAILED(dev_.device()->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&frames_[i].alloc)))) return false;
        if (FAILED(dev_.device()->CreateCommandList(
                0, D3D12_COMMAND_LIST_TYPE_DIRECT, frames_[i].alloc.Get(), nullptr,
                IID_PPV_ARGS(&frames_[i].cmd)))) return false;
        frames_[i].cmd->Close();

        // Per-frame constant ring (UPLOAD heap, persistently mapped).
        D3D12_HEAP_PROPERTIES cbh{};
        cbh.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC cbd{};
        cbd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
        cbd.Width            = kCbRingBytes;
        cbd.Height           = 1;
        cbd.DepthOrArraySize = 1;
        cbd.MipLevels        = 1;
        cbd.Format           = DXGI_FORMAT_UNKNOWN;
        cbd.SampleDesc.Count = 1;
        cbd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(dev_.device()->CreateCommittedResource(
                &cbh, D3D12_HEAP_FLAG_NONE, &cbd,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                IID_PPV_ARGS(&cb_ring_[i])))) return false;
        D3D12_RANGE no_read{0, 0};
        void* mapped = nullptr;
        if (FAILED(cb_ring_[i]->Map(0, &no_read, &mapped))) return false;
        cb_ring_cpu_[i] = static_cast<u8*>(mapped);
        cb_ring_gpu_[i] = cb_ring_[i]->GetGPUVirtualAddress();
    }
    return true;
}

void D3D12Swapchain::destroy_per_frame_objects() {
    for (auto& f : frames_) { f.cmd.Reset(); f.alloc.Reset(); }
    for (UINT i = 0; i < kFramesInFlight; ++i) {
        if (cb_ring_[i]) { cb_ring_[i]->Unmap(0, nullptr); cb_ring_[i].Reset(); }
        cb_ring_cpu_[i] = nullptr;
        cb_ring_gpu_[i] = 0;
    }
    cb_ring_offset_ = 0;
}

bool D3D12Swapchain::ensure_viewport_objects(u32 id, u32 w, u32 h) {
    if (id >= viewport_count_) return false;
    auto& vp = viewports_[id];
    if (vp.extent.width == w && vp.extent.height == h && vp.image) return true;
    // Idle every in-flight frame before destroying the old resources — the
    // GPU may still be sampling them from the previous frame's command list.
    for (UINT i = 0; i < kFramesInFlight; ++i) wait_for_frame(i);
    destroy_viewport_objects(id);
    if (w == 0 || h == 0) return true;

    D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC   rd{};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width            = w;
    rd.Height           = h;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.Format           = DXGI_FORMAT_B8G8R8A8_UNORM;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags            = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    D3D12_CLEAR_VALUE cv{};
    cv.Format = rd.Format;
    HRESULT hr = dev_.device()->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_RENDER_TARGET, &cv, IID_PPV_ARGS(&vp.image));
    if (FAILED(hr)) {
        cardinal::log::errorf("rhi/d3d12", "viewport[%u] RTT alloc failed (%s)",
                              id, hr_str(hr));
        return false;
    }
    D3D12_DESCRIPTOR_HEAP_DESC dh{}; dh.NumDescriptors = 1; dh.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    if (FAILED(dev_.device()->CreateDescriptorHeap(&dh, IID_PPV_ARGS(&vp.rtv_heap)))) {
        return false;
    }
    dev_.device()->CreateRenderTargetView(vp.image.Get(), nullptr,
        vp.rtv_heap->GetCPUDescriptorHandleForHeapStart());

    vp.extent = { w, h };
    return true;
}

void D3D12Swapchain::destroy_viewport_objects(u32 id) {
    if (id >= viewports_.size()) return;
    auto& vp = viewports_[id];
    vp.image.Reset();
    vp.rtv_heap.Reset();
    vp.depth_image.Reset();
    vp.dsv_heap.Reset();
    vp.extent = {};
}

void D3D12Swapchain::destroy_all_viewport_objects() {
    for (u32 i = 0; i < viewports_.size(); ++i) destroy_viewport_objects(i);
    viewports_.clear();
    viewport_count_ = 0u;
}

void D3D12Swapchain::set_viewport_count(u32 n) {
    if (n == viewport_count_) return;
    if (n < viewport_count_) {
        // Shrinking — release the trailing slots first, then shrink the vector.
        for (UINT i = 0; i < kFramesInFlight; ++i) wait_for_frame(i);
        for (u32 i = n; i < viewport_count_; ++i) destroy_viewport_objects(i);
        viewports_.resize(n);
    } else {
        viewports_.resize(n);
    }
    viewport_count_ = n;
    if (viewport_active_id_ >= viewport_count_ && viewport_count_ > 0u) {
        viewport_active_id_ = viewport_count_ - 1u;
    }
}

void D3D12Swapchain::set_active_viewport(u32 id) {
    if (id >= viewport_count_) {
        cardinal::log::errorf("rhi/d3d12",
            "set_active_viewport(%u) out of range (count=%u)", id, viewport_count_);
        return;
    }
    viewport_active_id_ = id;

    // Lazily allocate this viewport's image + depth using its pending size.
    auto& vp = viewports_[id];
    if (vp.pending.width != vp.extent.width || vp.pending.height != vp.extent.height) {
        ensure_viewport_objects(id, vp.pending.width, vp.pending.height);
    }
    if (vp.extent.width == 0 || vp.extent.height == 0) return;
    ensure_viewport_depth(id, vp.extent.width, vp.extent.height);

    Frame& f = frames_[frame_index_];
    if (f.cmd == nullptr) return;   // not inside a begin_frame/end_frame pair

    D3D12_CPU_DESCRIPTOR_HANDLE vrtv = vp.rtv_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE vdsv = (vp.dsv_heap != nullptr)
        ? vp.dsv_heap->GetCPUDescriptorHandleForHeapStart()
        : D3D12_CPU_DESCRIPTOR_HANDLE{};

    f.cmd->OMSetRenderTargets(1, &vrtv, FALSE,
        vp.dsv_heap != nullptr ? &vdsv : nullptr);
    f.cmd->ClearRenderTargetView(vrtv, last_clear_color_, 0, nullptr);
    if (vp.dsv_heap != nullptr) {
        f.cmd->ClearDepthStencilView(vdsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
    }

    D3D12_VIEWPORT d3dvp{ 0, 0, static_cast<float>(vp.extent.width),
                                static_cast<float>(vp.extent.height), 0.0f, 1.0f };
    D3D12_RECT     sc{ 0, 0, static_cast<LONG>(vp.extent.width),
                              static_cast<LONG>(vp.extent.height) };
    f.cmd->RSSetViewports(1, &d3dvp);
    f.cmd->RSSetScissorRects(1, &sc);

    vp.rendered_this_frame = true;
    bound_pipeline_ = nullptr;   // pipeline state is per-RT, force re-bind
    bound_vbuf_     = nullptr;
}

void D3D12Swapchain::wait_for_frame(u32 i) {
    if (fence_->GetCompletedValue() < frames_[i].fence_value) {
        fence_->SetEventOnCompletion(frames_[i].fence_value, fence_event_);
        WaitForSingleObject(fence_event_, INFINITE);
    }
}

u32 D3D12Swapchain::begin_frame(float r, float g, float b, float a) {
    // Stash the clear color for set_active_viewport() to reuse on each
    // per-viewport clear (every panel gets cleared to the same color so
    // they share a uniform "void" backdrop).
    last_clear_color_[0] = r; last_clear_color_[1] = g;
    last_clear_color_[2] = b; last_clear_color_[3] = a;

    // Reset per-viewport rendered-this-frame flags so end_frame only
    // transitions the ones the host actually drew into.
    for (auto& vp : viewports_) vp.rendered_this_frame = false;

    frame_index_ = swap_->GetCurrentBackBufferIndex();
    Frame& f = frames_[frame_index_];
    wait_for_frame(frame_index_);
    cb_ring_offset_ = 0;          // frame's prior GPU work retired -> ring reusable
    f.alloc->Reset();
    f.cmd->Reset(f.alloc.Get(), nullptr);

    ID3D12Resource* backbuf = back_buffers_[frame_index_].Get();

    // Step 1: back buffer goes to RENDER_TARGET state.
    {
        D3D12_RESOURCE_BARRIER bar{};
        bar.Type  = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bar.Transition.pResource   = backbuf;
        bar.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        bar.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
        bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        f.cmd->ResourceBarrier(1, &bar);
    }

    // Step 2: ALWAYS clear the back buffer + its depth attachment.
    // FLIP_DISCARD discards previous contents on Present, so without this
    // the back buffer is undefined and the editor flickers / shows garbage
    // when the off-screen viewport is active. The overlay (ImGui) will
    // draw on top of this clear later. Depth is cleared to 1.0 so the
    // depth-test passes for the first triangle drawn.
    {
        D3D12_CPU_DESCRIPTOR_HANDLE bb_rtv = rtv_heap_->GetCPUDescriptorHandleForHeapStart();
        bb_rtv.ptr += static_cast<SIZE_T>(frame_index_) * rtv_stride_;
        D3D12_CPU_DESCRIPTOR_HANDLE dsv = (dsv_heap_ != nullptr)
            ? dsv_heap_->GetCPUDescriptorHandleForHeapStart() : D3D12_CPU_DESCRIPTOR_HANDLE{};
        const float clear[4] = { r, g, b, a };
        f.cmd->OMSetRenderTargets(1, &bb_rtv, FALSE,
            dsv_heap_ != nullptr ? &dsv : nullptr);
        f.cmd->ClearRenderTargetView(bb_rtv, clear, 0, nullptr);
        if (dsv_heap_ != nullptr) {
            f.cmd->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        }
    }

    // Step 3: if at least one viewport exists, switch to viewport[0] as
    // the default render target so the legacy single-viewport flow (host
    // does begin_frame → render → end_frame, no set_active_viewport call)
    // keeps working. Multi-viewport hosts call set_active_viewport(id)
    // before each render and override this default. If no viewports are
    // configured (zero count), scene rendering hits the back buffer
    // directly — that's the headless / no-editor path.
    if (viewport_count_ > 0u) {
        set_active_viewport(0u);
    } else {
        D3D12_VIEWPORT vp{ 0, 0, static_cast<float>(extent_.width),
                                  static_cast<float>(extent_.height), 0.0f, 1.0f };
        D3D12_RECT     sc{ 0, 0, static_cast<LONG>(extent_.width),
                                  static_cast<LONG>(extent_.height) };
        f.cmd->RSSetViewports(1, &vp);
        f.cmd->RSSetScissorRects(1, &sc);
    }

    bound_pipeline_ = nullptr;
    bound_vbuf_     = nullptr;
    return frame_index_;
}

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

void D3D12Swapchain::end_frame() {
    Frame& f = frames_[frame_index_];
    ID3D12Resource* backbuf = back_buffers_[frame_index_].Get();

    // Transition EVERY viewport that was rendered this frame to
    // PIXEL_SHADER_RESOURCE so the overlay (ImGui::Image inside each
    // Viewport panel) can sample them. Flipped back to RENDER_TARGET
    // below for next frame. We batch all transitions into one
    // ResourceBarrier call to minimise driver overhead.
    cardinal::vector<D3D12_RESOURCE_BARRIER> to_srv;
    to_srv.reserve(viewport_count_);
    for (u32 i = 0; i < viewport_count_; ++i) {
        auto& vp = viewports_[i];
        if (!vp.rendered_this_frame || vp.image == nullptr) continue;
        D3D12_RESOURCE_BARRIER bar{};
        bar.Type  = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bar.Transition.pResource   = vp.image.Get();
        bar.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        bar.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        to_srv.push_back(bar);
    }
    if (!to_srv.empty()) {
        f.cmd->ResourceBarrier(static_cast<UINT>(to_srv.size()), to_srv.data());
    }

    // Switch the active RTV back to the back buffer + reset viewport/scissor
    // to the full back buffer size, then run the overlay so ImGui's draw
    // lands on the user-visible image. We do this UNCONDITIONALLY — the
    // overlay must always render on the back buffer, regardless of whether
    // the off-screen viewport image was used during scene rendering.
    {
        D3D12_CPU_DESCRIPTOR_HANDLE bb_rtv = rtv_heap_->GetCPUDescriptorHandleForHeapStart();
        bb_rtv.ptr += static_cast<SIZE_T>(frame_index_) * rtv_stride_;
        f.cmd->OMSetRenderTargets(1, &bb_rtv, FALSE, nullptr);

        D3D12_VIEWPORT vp{ 0, 0,
            static_cast<float>(extent_.width), static_cast<float>(extent_.height),
            0.0f, 1.0f };
        D3D12_RECT sc{ 0, 0,
            static_cast<LONG>(extent_.width), static_cast<LONG>(extent_.height) };
        f.cmd->RSSetViewports(1, &vp);
        f.cmd->RSSetScissorRects(1, &sc);
    }

    if (overlay_cb_) overlay_cb_(overlay_user_);

    // Restore each rendered-this-frame viewport image to RENDER_TARGET so
    // next frame's set_active_viewport can clear + draw into it.
    cardinal::vector<D3D12_RESOURCE_BARRIER> to_rt;
    to_rt.reserve(to_srv.size());
    for (const auto& bar : to_srv) {
        D3D12_RESOURCE_BARRIER nb = bar;
        nb.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        nb.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
        to_rt.push_back(nb);
    }
    if (!to_rt.empty()) {
        f.cmd->ResourceBarrier(static_cast<UINT>(to_rt.size()), to_rt.data());
    }

    {
        D3D12_RESOURCE_BARRIER bar{};
        bar.Type  = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bar.Transition.pResource   = backbuf;
        bar.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        bar.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
        bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        f.cmd->ResourceBarrier(1, &bar);
    }

    f.cmd->Close();
    ID3D12CommandList* lists[] = { f.cmd.Get() };
    dev_.gfx_queue()->ExecuteCommandLists(1, lists);

    // Sync interval: 0 = uncapped, 1 = locked, 2 = half-refresh, etc.
    // DXGI_PRESENT_ALLOW_TEARING is required at Present time AND the swap
    // chain must have been created with DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.
    // The flag is *only* legal when sync_interval == 0 — DXGI returns
    // INVALID_CALL if you pair it with sync_interval > 0.
    const UINT sync_interval = vsync_interval_;
    const UINT present_flags =
        (sync_interval == 0u && (swap_flags_ & DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING))
            ? DXGI_PRESENT_ALLOW_TEARING
            : 0u;
    // Reflex: PresentStart wraps the OS-side block on Present. Marker
    // pairs are no-ops when Reflex isn't initialised.
    reflex_marker(ReflexMarker::PresentStart, reflex_frame_id_);
    swap_->Present(sync_interval, present_flags);
    reflex_marker(ReflexMarker::PresentEnd,   reflex_frame_id_);
    ++reflex_frame_id_;

    f.fence_value = next_fence_value_++;
    dev_.gfx_queue()->Signal(fence_.Get(), f.fence_value);
}

// ---- resize ---------------------------------------------------------------
// DXGI's ResizeBuffers does the heavy lifting; we just have to release any
// references to the old back buffers (and idle the GPU) before calling it.
bool D3D12Swapchain::resize(u32 new_w, u32 new_h) {
    if (new_w == 0 || new_h == 0) return false;
    if (new_w == extent_.width && new_h == extent_.height) return true;

    // Idle every frame slot so we know nothing is referencing the buffers.
    for (UINT i = 0; i < kFramesInFlight; ++i) wait_for_frame(i);

    for (auto& bb : back_buffers_) bb.Reset();

    HRESULT hr = swap_->ResizeBuffers(
        kFramesInFlight, new_w, new_h, DXGI_FORMAT_B8G8R8A8_UNORM, swap_flags_);
    if (FAILED(hr)) {
        cardinal::log::errorf("rhi/d3d12", "ResizeBuffers failed (%s)", hr_str(hr));
        return false;
    }

    extent_      = { new_w, new_h };
    frame_index_ = swap_->GetCurrentBackBufferIndex();

    auto rtv_handle = rtv_heap_->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < kFramesInFlight; ++i) {
        if (FAILED(swap_->GetBuffer(i, IID_PPV_ARGS(&back_buffers_[i])))) return false;
        dev_.device()->CreateRenderTargetView(back_buffers_[i].Get(), nullptr, rtv_handle);
        rtv_handle.ptr += rtv_stride_;
    }

    // Re-allocate the depth image at the new size.
    if (!ensure_depth_image(new_w, new_h)) return false;

    cardinal::log::infof("rhi/d3d12", "Swapchain resized to %ux%u", new_w, new_h);
    if (on_rebuilt_) on_rebuilt_();
    return true;
}

void D3D12Swapchain::destroy() {
    if (fence_event_) {
        // Idle the device so we can safely tear everything down.
        const u64 v = next_fence_value_++;
        if (dev_.gfx_queue()) dev_.gfx_queue()->Signal(fence_.Get(), v);
        if (fence_->GetCompletedValue() < v) {
            fence_->SetEventOnCompletion(v, fence_event_);
            WaitForSingleObject(fence_event_, INFINITE);
        }
        CloseHandle(fence_event_); fence_event_ = nullptr;
    }
    destroy_all_viewport_objects();
    destroy_per_frame_objects();
    for (auto& bb : back_buffers_) bb.Reset();
    rtv_heap_.Reset();
    swap_.Reset();
    fence_.Reset();
}


}  // namespace cardinal::rhi

#endif  // CARDINAL_PLATFORM_WINDOWS
#endif  // _WIN32 && !CARDINAL_NO_D3D12
