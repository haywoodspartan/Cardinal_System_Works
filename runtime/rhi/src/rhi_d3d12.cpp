// =============================================================================
// Cardinal RHI — Direct3D 12 backend.
//
// Functional baseline (Phase 2.6 first cut):
//   ✓ Device + adapter selection (preferring discrete, with fallback)
//   ✓ Swapchain (DXGI flip-model, double-buffered) with viewport RTT
//   ✓ Per-frame command allocator/list with fence-based sync
//   ✓ Buffer (UPLOAD heap for cpu_writable; DEFAULT heap otherwise)
//   ✓ Graphics pipeline (root sig + PSO from DXC-compiled DXIL)
//   ✓ Clear color, bind pipeline, draw, overlay callback
//   ✓ Capability surface populated from D3D12_FEATURE_* probes
//
// Stubs deliberately deferred to a follow-up phase:
//   ✗ Acceleration structures (create_blas / create_tlas return nullptr)
//   ✗ Rapid Packed Math intrinsics surfaced via root constants
//   ✗ NVIDIA SDK plumbing (Streamline / Reflex are Vulkan-first today)
//
// API parity with the Vulkan backend means the same engine code path drives
// both. Differences are localised to this file via the Device/Swapchain
// abstract bases.
// =============================================================================

#include <cardinal/rhi/rhi.hpp>
#include <cardinal/core/log.hpp>
#include <cardinal/core/platform.hpp>

#if defined(_WIN32)
#include <cardinal/rhi/d3d12_interop.hpp>
#endif

#if !CARDINAL_PLATFORM_WINDOWS

namespace cardinal::rhi {
cardinal::unique_ptr<Device> create_d3d12_device(const DeviceDesc&) {
    cardinal::log::warnf("rhi/d3d12", "D3D12 backend is Windows-only");
    return nullptr;
}
}  // namespace cardinal::rhi

#else  // CARDINAL_PLATFORM_WINDOWS

#include <Windows.h>
#include <unknwn.h>
#include <wrl/client.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <dxgidebug.h>
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dxguid.lib")

// DXC for HLSL → DXIL compilation. The Vulkan backend uses the same library
// to emit SPIR-V — we just request the DXIL profile here.
#include <dxcapi.h>

// NvAPI — used here ONLY for Reflex on D3D12 (NvAPI_D3D_SetSleepMode /
// Sleep / SetLatencyMarker / GetLatency). The header is noisy under /WX
// so it gets local-suppression. Compiles to no-ops when the SDK is absent.
#if CARDINAL_HAS_NVAPI
    #if defined(_MSC_VER)
        #pragma warning(push)
        #pragma warning(disable: 4100 4189 4244 4245 4365 4458 4505 4828)
    #endif
    #include <nvapi.h>
    #if defined(_MSC_VER)
        #pragma warning(pop)
    #endif
#endif

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/containers.hpp>
#include <cardinal/core/cstdio.hpp>
#include <cardinal/core/cstring.hpp>
#include <cardinal/core/utility.hpp>

using Microsoft::WRL::ComPtr;

namespace cardinal::rhi {

namespace {

constexpr u32 kFramesInFlight = 2;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
const char* hr_str(HRESULT hr) {
    static thread_local char buf[64];
    cardinal::snprintf(buf, sizeof(buf), "0x%08lx", hr);
    return buf;
}

#define D3D_CHECK(expr) do { HRESULT _hr = (expr);                            \
    if (FAILED(_hr)) {                                                        \
        cardinal::log::errorf("rhi/d3d12", "%s failed (%s)", #expr, hr_str(_hr)); \
        return false;                                                         \
    } } while (0)

DXGI_FORMAT to_dxgi(Format f) {
    switch (f) {
        case Format::R8G8B8A8_UNORM:    return DXGI_FORMAT_R8G8B8A8_UNORM;
        case Format::B8G8R8A8_UNORM:    return DXGI_FORMAT_B8G8R8A8_UNORM;
        case Format::R32G32_Float:      return DXGI_FORMAT_R32G32_FLOAT;
        case Format::R32G32B32_Float:   return DXGI_FORMAT_R32G32B32_FLOAT;
        case Format::R32G32B32A32_Float:return DXGI_FORMAT_R32G32B32A32_FLOAT;
    }
    return DXGI_FORMAT_UNKNOWN;
}

D3D12_PRIMITIVE_TOPOLOGY_TYPE to_d3d_topology_type(PrimitiveTopology t) {
    switch (t) {
        case PrimitiveTopology::TriangleList:
        case PrimitiveTopology::TriangleStrip: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        case PrimitiveTopology::LineList:
        case PrimitiveTopology::LineStrip:     return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
        case PrimitiveTopology::PointList:     return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
    }
    return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
}

D3D12_PRIMITIVE_TOPOLOGY to_d3d_topology(PrimitiveTopology t) {
    switch (t) {
        case PrimitiveTopology::TriangleList:  return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        case PrimitiveTopology::TriangleStrip: return D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
        case PrimitiveTopology::LineList:      return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
        case PrimitiveTopology::LineStrip:     return D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
        case PrimitiveTopology::PointList:     return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
    }
    return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
}

// -----------------------------------------------------------------------------
// D3D12Buffer
// -----------------------------------------------------------------------------
class D3D12Buffer final : public Buffer {
public:
    D3D12Buffer(ComPtr<ID3D12Resource> r, usize size, bool cpu_writable)
        : res_(cardinal::move(r)), size_(size), cpu_writable_(cpu_writable) {
        gpu_va_ = res_->GetGPUVirtualAddress();
    }

    usize size() const noexcept override { return size_; }

    void upload(const void* data, usize size, usize offset) override {
        if (!cpu_writable_) {
            cardinal::log::errorf("rhi/d3d12", "upload() on non-cpu_writable buffer");
            return;
        }
        if (offset + size > size_) {
            cardinal::log::errorf("rhi/d3d12", "upload() out of range");
            return;
        }
        void* mapped = nullptr;
        D3D12_RANGE read_range{0, 0};   // we don't read
        if (FAILED(res_->Map(0, &read_range, &mapped))) {
            cardinal::log::errorf("rhi/d3d12", "upload Map failed");
            return;
        }
        cardinal::memcpy(static_cast<u8*>(mapped) + offset, data, size);
        res_->Unmap(0, nullptr);
    }

    u64 device_address() const noexcept override { return gpu_va_; }

    ID3D12Resource* resource() const noexcept { return res_.Get(); }

private:
    ComPtr<ID3D12Resource> res_;
    usize                  size_{0};
    bool                   cpu_writable_{false};
    u64                    gpu_va_{0};
};

// -----------------------------------------------------------------------------
// D3D12Texture — depth render target that's also a shader resource. The
// resource is R32_TYPELESS so it can carry BOTH a D32_FLOAT DSV (write
// pass) and an R32_FLOAT SRV (sample pass) — the standard depth-as-
// texture trick. Owns its own 1-entry DSV heap + 1-entry shader-visible
// SRV heap. `state_` tracks the resource state for the DEPTH_WRITE ↔
// PIXEL_SHADER_RESOURCE barrier the swapchain issues around the pass.
// -----------------------------------------------------------------------------
class D3D12Texture final : public Texture {
public:
    bool initialize(ID3D12Device* dev, const TextureDesc& d) {
        w_ = d.width; h_ = d.height; fmt_ = d.format;
        if (w_ == 0 || h_ == 0) return false;
        const bool is_depth = (d.format == Format::D32_Float);

        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width            = w_;
        rd.Height           = h_;
        rd.DepthOrArraySize = 1;
        rd.MipLevels        = 1;
        rd.Format           = is_depth ? DXGI_FORMAT_R32_TYPELESS
                                       : DXGI_FORMAT_R8G8B8A8_UNORM;
        rd.SampleDesc.Count = 1;
        rd.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        rd.Flags            = (d.usage & static_cast<u32>(TextureUsage::DepthRenderTarget))
                            ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
                            : D3D12_RESOURCE_FLAG_NONE;

        D3D12_CLEAR_VALUE cv{};
        cv.Format             = DXGI_FORMAT_D32_FLOAT;
        cv.DepthStencil.Depth = 1.0f;

        state_ = D3D12_RESOURCE_STATE_DEPTH_WRITE;
        if (FAILED(dev->CreateCommittedResource(
                &heap, D3D12_HEAP_FLAG_NONE, &rd, state_,
                is_depth ? &cv : nullptr, IID_PPV_ARGS(&res_)))) {
            cardinal::log::errorf("rhi/d3d12", "create_texture resource failed");
            return false;
        }

        if (d.usage & static_cast<u32>(TextureUsage::DepthRenderTarget)) {
            D3D12_DESCRIPTOR_HEAP_DESC dh{};
            dh.NumDescriptors = 1;
            dh.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
            if (FAILED(dev->CreateDescriptorHeap(&dh, IID_PPV_ARGS(&dsv_heap_))))
                return false;
            D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
            dsv.Format        = DXGI_FORMAT_D32_FLOAT;
            dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
            dev->CreateDepthStencilView(res_.Get(), &dsv,
                dsv_heap_->GetCPUDescriptorHandleForHeapStart());
        }
        if (d.usage & static_cast<u32>(TextureUsage::Sampled)) {
            D3D12_DESCRIPTOR_HEAP_DESC sh{};
            sh.NumDescriptors = 1;
            sh.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            sh.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            if (FAILED(dev->CreateDescriptorHeap(&sh, IID_PPV_ARGS(&srv_heap_))))
                return false;
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format                  = is_depth ? DXGI_FORMAT_R32_FLOAT
                                                   : DXGI_FORMAT_R8G8B8A8_UNORM;
            srv.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2D.MipLevels     = 1;
            dev->CreateShaderResourceView(res_.Get(), &srv,
                srv_heap_->GetCPUDescriptorHandleForHeapStart());
        }
        return true;
    }

    u32    width()  const noexcept override { return w_; }
    u32    height() const noexcept override { return h_; }
    Format format() const noexcept override { return fmt_; }

    ID3D12Resource*       resource() const noexcept { return res_.Get(); }
    ID3D12DescriptorHeap* srv_heap() const noexcept { return srv_heap_.Get(); }
    D3D12_CPU_DESCRIPTOR_HANDLE dsv_handle() const noexcept {
        return dsv_heap_->GetCPUDescriptorHandleForHeapStart();
    }
    D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu_handle() const noexcept {
        return srv_heap_->GetGPUDescriptorHandleForHeapStart();
    }
    D3D12_RESOURCE_STATES state_{D3D12_RESOURCE_STATE_DEPTH_WRITE};

private:
    ComPtr<ID3D12Resource>       res_;
    ComPtr<ID3D12DescriptorHeap> dsv_heap_;
    ComPtr<ID3D12DescriptorHeap> srv_heap_;
    u32    w_{0}, h_{0};
    Format fmt_{Format::D32_Float};
};

// -----------------------------------------------------------------------------
// D3D12Pipeline
// -----------------------------------------------------------------------------
class D3D12Pipeline final : public Pipeline {
public:
    D3D12Pipeline(ComPtr<ID3D12RootSignature> rs, ComPtr<ID3D12PipelineState> pso,
                  PrimitiveTopology topo, u32 vertex_stride, u32 push_constant_size,
                  u32 storage_buffer_slots, u32 sampled_texture_slots,
                  bool push_is_cbv)
        : root_(cardinal::move(rs)), pso_(cardinal::move(pso))
        , topology_(to_d3d_topology(topo)), vertex_stride_(vertex_stride)
        , push_constant_size_(push_constant_size)
        , push_is_cbv_(push_is_cbv)
        , storage_buffer_slots_(storage_buffer_slots)
        , sampled_texture_slots_(sampled_texture_slots)
        // Root SRVs follow the optional 32-bit-constants param. When a
        // push block exists it occupies root index 0, so storage slots
        // start at index 1; otherwise they start at 0. bind_storage_
        // buffer adds the slot to this base.
        , storage_root_base_(push_constant_size > 0 ? 1u : 0u)
        // The sampled-texture descriptor table is the single root param
        // appended AFTER the push block + all storage root SRVs.
        , sampled_table_root_(storage_root_base_ + storage_buffer_slots) {}

    ID3D12RootSignature*   root_signature()      const noexcept { return root_.Get(); }
    ID3D12PipelineState*   pso()                 const noexcept { return pso_.Get(); }
    D3D12_PRIMITIVE_TOPOLOGY topology()          const noexcept { return topology_; }
    u32                    vertex_stride()       const noexcept { return vertex_stride_; }
    u32                    push_constant_size()  const noexcept { return push_constant_size_; }
    bool                   push_is_cbv()         const noexcept { return push_is_cbv_; }
    u32                    storage_buffer_slots() const noexcept { return storage_buffer_slots_; }
    u32                    storage_root_base()    const noexcept { return storage_root_base_; }
    u32                    sampled_texture_slots() const noexcept { return sampled_texture_slots_; }
    u32                    sampled_table_root()   const noexcept { return sampled_table_root_; }

private:
    ComPtr<ID3D12RootSignature> root_;
    ComPtr<ID3D12PipelineState> pso_;
    D3D12_PRIMITIVE_TOPOLOGY    topology_;
    u32                         vertex_stride_{0};
    u32                         push_constant_size_{0};
    bool                        push_is_cbv_{false};
    u32                         storage_buffer_slots_{0};
    u32                         storage_root_base_{0};
    u32                         sampled_texture_slots_{0};
    u32                         sampled_table_root_{0};
};

// -----------------------------------------------------------------------------
// D3D12Swapchain
// -----------------------------------------------------------------------------
class D3D12Device;   // fwd

class D3D12Swapchain final : public Swapchain {
public:
    explicit D3D12Swapchain(D3D12Device& dev) : dev_(dev) {}
    ~D3D12Swapchain() override { destroy(); }

    bool initialize(HWND hwnd, u32 w, u32 h);

    u32    width()  const noexcept override { return extent_.width; }
    u32    height() const noexcept override { return extent_.height; }
    Format color_format() const noexcept override { return Format::B8G8R8A8_UNORM; }

    void   set_viewport_size(u32 w, u32 h) override {
        set_viewport_size(0u, w, h);
    }
    u32    viewport_width()  const noexcept override { return viewport_width (0u); }
    u32    viewport_height() const noexcept override { return viewport_height(0u); }
    void   set_viewport_size(u32 id, u32 w, u32 h) override {
        if (id >= viewport_count_) set_viewport_count(id + 1u);
        viewports_[id].pending = { w, h };
    }
    u32    viewport_width (u32 id) const noexcept override {
        return id < viewport_count_ ? viewports_[id].extent.width  : 0u;
    }
    u32    viewport_height(u32 id) const noexcept override {
        return id < viewport_count_ ? viewports_[id].extent.height : 0u;
    }
    void   set_viewport_count(u32 n) override;
    u32    viewport_count() const noexcept override { return viewport_count_; }
    void   set_active_viewport(u32 id) override;
    u32    active_viewport() const noexcept override { return viewport_active_id_; }

    u32  begin_frame(float r, float g, float b, float a) override;
    void end_frame() override;
    bool resize(u32 new_w, u32 new_h) override;
    void set_on_rebuilt(OnRebuilt cb) override { on_rebuilt_ = cardinal::move(cb); }

    void   set_vsync(bool on) override         { vsync_interval_ = on ? 1u : 0u; }
    bool   vsync() const noexcept override     { return vsync_interval_ > 0u; }
    void   set_vsync_interval(u32 i) override  { vsync_interval_ = cardinal::min<u32>(i, 4u); }
    u32    vsync_interval() const noexcept override { return vsync_interval_; }
    Format depth_format() const noexcept override { return Format::D32_Float; }

    // ----- Reflex (NvAPI on D3D12) -------------------------------------
    bool reflex_supported() const noexcept override { return reflex_supported_; }
    void set_reflex_mode(ReflexMode m) override;
    ReflexMode reflex_mode() const noexcept override { return reflex_mode_; }
    void set_reflex_fps_cap(u32 fps) override;
    u32  reflex_fps_cap() const noexcept override { return reflex_fps_cap_; }
    void reflex_sleep() override;
    void reflex_marker(ReflexMarker m, u64 frame_id) override;
    ReflexLatency reflex_latency() const override;

    void set_overlay(OverlayCallback cb, void* user_data) override {
        overlay_cb_ = cb; overlay_user_ = user_data;
    }

    void bind_pipeline(Pipeline* p) override;
    void bind_vertex_buffer(Buffer* b, usize offset = 0) override;
    void draw(u32 vertex_count, u32 instance_count = 1,
              u32 first_vertex = 0, u32 first_instance = 0) override;
    void set_push_constants(u32 offset, const void* data, u32 size) override;
    void bind_storage_buffer(u32 slot, Buffer* b) override;
    void begin_shadow_pass(Texture* depth) override;
    void end_shadow_pass() override;
    void bind_sampled_texture(u32 slot, Texture* tex) override;

    // Interop accessors.
    IDXGISwapChain3*           swap_chain()       const noexcept { return swap_.Get(); }
    DXGI_FORMAT                back_buffer_format() const noexcept { return DXGI_FORMAT_B8G8R8A8_UNORM; }
    u32                        back_buffer_count() const noexcept { return kFramesInFlight; }
    ID3D12GraphicsCommandList* current_cmd()       const noexcept { return frames_[frame_index_].cmd.Get(); }
    u32                        current_index()     const noexcept { return frame_index_; }
    ID3D12Resource*            back_buffer(u32 i)  const noexcept {
        return i < kFramesInFlight ? back_buffers_[i].Get() : nullptr;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE back_buffer_rtv(u32 i) const noexcept {
        D3D12_CPU_DESCRIPTOR_HANDLE h = rtv_heap_->GetCPUDescriptorHandleForHeapStart();
        h.ptr += static_cast<SIZE_T>(i) * rtv_stride_;
        return h;
    }
    ID3D12Resource*            viewport_image()           const noexcept { return viewport_image(0u); }
    ID3D12Resource*            viewport_image(u32 id)     const noexcept {
        return (id < viewport_count_) ? viewports_[id].image.Get() : nullptr;
    }
    u32                        d3d12_viewport_count()     const noexcept { return viewport_count_; }

private:
    struct Extent { u32 width{0}, height{0}; };
    struct Frame {
        ComPtr<ID3D12CommandAllocator>      alloc;
        ComPtr<ID3D12GraphicsCommandList>   cmd;
        u64                                 fence_value{0};
    };

    // Per-viewport off-screen render target. Each editor viewport panel
    // gets one of these — the host loops between begin_frame/end_frame
    // setting set_active_viewport(id) + pipeline.render() per panel, which
    // writes into the corresponding image. ImGui samples each via its SRV.
    struct ViewportSlot {
        Extent                           extent{};   // current allocated size
        Extent                           pending{};  // last set_viewport_size request
        ComPtr<ID3D12Resource>           image;
        ComPtr<ID3D12DescriptorHeap>     rtv_heap;
        ComPtr<ID3D12Resource>           depth_image;
        ComPtr<ID3D12DescriptorHeap>     dsv_heap;
        bool                             rendered_this_frame{false};   // for end_frame transitions
    };

    void destroy();
    bool create_per_frame_objects();
    void destroy_per_frame_objects();
    bool ensure_viewport_objects(u32 id, u32 w, u32 h);
    void destroy_viewport_objects(u32 id);
    void destroy_all_viewport_objects();
    void wait_for_frame(u32 i);

    D3D12Device& dev_;

    HWND                   hwnd_{nullptr};
    Extent                 extent_{};
    ComPtr<IDXGISwapChain3>          swap_;
    ComPtr<ID3D12DescriptorHeap>     rtv_heap_;
    UINT                             rtv_stride_{0};
    cardinal::array<ComPtr<ID3D12Resource>, kFramesInFlight> back_buffers_;
    cardinal::array<Frame, kFramesInFlight> frames_;
    UINT                             frame_index_{0};

    // Per-frame upload constant ring — backs root-CBV push blocks (large
    // push-constant pipelines; see create_pipeline). Each set_push_constants
    // bump-allocates a fresh 256 B-aligned slice so successive draws in a
    // frame don't clobber each other's constants before the GPU consumes
    // them. One persistently-mapped buffer per frame-in-flight; the offset
    // resets in begin_frame after the frame's prior submission has retired.
    static constexpr u32 kCbAlign     = 256u;            // D3D12 CBV placement
    static constexpr u32 kMaxPushBytes = 256u;           // >= largest push block
    static constexpr u32 kCbRingBytes = 256u * 1024u;    // 1024 slices/frame
    cardinal::array<ComPtr<ID3D12Resource>, kFramesInFlight>     cb_ring_{};
    cardinal::array<u8*, kFramesInFlight>                        cb_ring_cpu_{};
    cardinal::array<D3D12_GPU_VIRTUAL_ADDRESS, kFramesInFlight>  cb_ring_gpu_{};
    u32                              cb_ring_offset_{0};
    cardinal::array<u8, kMaxPushBytes> push_shadow_{};   // CPU staging (partial writes)

    ComPtr<ID3D12Fence>              fence_;
    HANDLE                           fence_event_{nullptr};
    u64                              next_fence_value_{1};

    // Multi-viewport off-screen RTTs. viewports_ holds N per-panel slots
    // (count tracked by viewport_count_); viewport_active_id_ is the panel
    // the next pipeline.render() call writes into. begin_frame stashes the
    // clear color so each set_active_viewport(id) can re-clear that
    // viewport's color + depth before the host's render pass overwrites it.
    cardinal::vector<ViewportSlot>        viewports_{};
    u32                              viewport_count_{0u};
    u32                              viewport_active_id_{0u};
    float                            last_clear_color_[4]{0, 0, 0, 1};

    // Depth attachment for the main back-buffer path (sized to the window).
    // Per-viewport depth attachments live inside viewports_[i].depth_image.
    ComPtr<ID3D12Resource>           depth_image_;
    ComPtr<ID3D12DescriptorHeap>     dsv_heap_;

    // VSync sync-interval. 0 = uncapped (DXGI tearing), 1 = locked to
    // display refresh, 2 = half-rate, 3 = third, 4 = quarter. Drives
    // Present(sync_interval, …).
    u32                              vsync_interval_{0};

    // Fires after the swapchain back buffers have been recreated for any
    // reason. D3D12 only rebuilds on resize today (vsync change is just
    // a Present-time parameter, no rebuild needed); kept for symmetry
    // with the Vulkan path so consumers can register one callback.
    OnRebuilt                        on_rebuilt_{};
    // DXGI swap-chain flags cached from CreateSwapChainForHwnd. Kept so
    // ResizeBuffers can pass the same flag set (DXGI requires consistency,
    // particularly DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING which must be
    // mirrored on Present via DXGI_PRESENT_ALLOW_TEARING).
    UINT                             swap_flags_{0};

    // ----- Reflex (NvAPI) state -------------------------------------------
    // reflex_supported_ : true once NvAPI_Initialize succeeded and the
    //                     adapter is NVIDIA. Probed once at swapchain init.
    // reflex_mode_      : last value passed to SetSleepMode (mirrored for
    //                     UI read-back; the driver also caches internally).
    // reflex_fps_cap_   : minimumIntervalUs translated back to FPS.
    // reflex_frame_id_  : monotonic frame counter that goes into every
    //                     marker so Reflex can correlate stages per-frame.
    bool                             reflex_supported_{false};
    ReflexMode                       reflex_mode_{ReflexMode::Off};
    u32                              reflex_fps_cap_{0};
    mutable u64                      reflex_frame_id_{1};

    bool ensure_depth_image(u32 w, u32 h);
    bool ensure_viewport_depth(u32 id, u32 w, u32 h);

    OverlayCallback overlay_cb_{nullptr};
    void*           overlay_user_{nullptr};

    // Recording state for current frame.
    Pipeline* bound_pipeline_{nullptr};
    Buffer*   bound_vbuf_{nullptr};
    Texture*  shadow_tex_{nullptr};   // active between begin/end_shadow_pass
    usize     bound_vbuf_offset_{0};
};

// -----------------------------------------------------------------------------
// D3D12Device
// -----------------------------------------------------------------------------
class D3D12Device final : public Device {
public:
    Backend backend() const noexcept override { return Backend::D3D12; }
    const char* adapter_name() const noexcept override { return adapter_name_; }

    const GpuCapabilities& capabilities() const noexcept override { return caps_; }
    const RenderSettings&  settings()     const noexcept override { return settings_; }
    void apply_settings(const RenderSettings& s) override { settings_ = s; }

    bool initialize(const DeviceDesc& desc);

    cardinal::unique_ptr<Swapchain> create_swapchain(void* native_window, u32 w, u32 h) override {
        auto sw = cardinal::make_unique<D3D12Swapchain>(*this);
        if (!sw->initialize(static_cast<HWND>(native_window), w, h)) return nullptr;
        return sw;
    }

    cardinal::unique_ptr<Buffer>   create_buffer(const BufferDesc& desc) override;
    cardinal::unique_ptr<Pipeline> create_pipeline(const PipelineDesc& desc) override;
    cardinal::unique_ptr<Texture>  create_texture(const TextureDesc& desc) override;

    // RT not yet wired in the D3D12 backend — return nullptr so callers see
    // it as unsupported.
    cardinal::unique_ptr<AccelerationStructure> create_blas(const BlasDesc&) override { return nullptr; }
    cardinal::unique_ptr<AccelerationStructure> create_tlas(const TlasDesc&) override { return nullptr; }

    ShaderBlob compile_shader(ShaderStage stage, const char* hlsl, const char* entry) override;

    VramSnapshot query_vram_usage() const noexcept override;

    // Interop accessors.
    ID3D12Device*       device()    const noexcept { return device_.Get(); }
    ID3D12CommandQueue* gfx_queue() const noexcept { return gfx_queue_.Get(); }
    IDXGIFactory6*      factory()   const noexcept { return factory_.Get(); }

private:
    bool create_factory();
    bool select_adapter(bool prefer_discrete);
    bool create_device();
    bool create_queues();
    bool init_dxc();
    void populate_capabilities();

    ComPtr<IDXGIFactory6>       factory_;
    ComPtr<IDXGIAdapter1>       adapter_;
    ComPtr<ID3D12Device>        device_;
    ComPtr<ID3D12CommandQueue>  gfx_queue_;

    // DXC (HLSL → DXIL).
    ComPtr<IDxcUtils>           dxc_utils_;
    ComPtr<IDxcCompiler3>       dxc_compiler_;
    ComPtr<IDxcIncludeHandler>  dxc_includes_;

    char            adapter_name_[160]{};
    GpuCapabilities caps_{};
    RenderSettings  settings_{};
public:
    // Tearing flag — DXGI 1.5 / Win10+. Required for true uncapped present
    // (Present(0,0) without this still composites at the display refresh
    // rate via DWM, which is the "60 cap when foreground" the editor was
    // hitting). Probed once at device creation.
    bool tearing_supported() const noexcept { return tearing_supported_; }
    // True iff NvAPI_Initialize succeeded (regardless of vendor). The
    // swapchain probes vendor before declaring Reflex usable.
    bool nvapi_initialised() const noexcept { return nvapi_initialised_; }
private:
    bool            tearing_supported_{false};
    bool            nvapi_initialised_{false};
public:
    ~D3D12Device() override {
#if CARDINAL_HAS_NVAPI
        if (nvapi_initialised_) NvAPI_Unload();
#endif
    }
};

// -----------------------------------------------------------------------------
// D3D12Device implementation
// -----------------------------------------------------------------------------
bool D3D12Device::initialize(const DeviceDesc& desc) {
    if (desc.enable_validation) {
        ComPtr<ID3D12Debug> dbg;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) {
            dbg->EnableDebugLayer();
            cardinal::log::infof("rhi/d3d12", "D3D12 debug layer enabled");
        }
    }

    if (!create_factory())                          return false;
    if (!select_adapter(desc.prefer_discrete_gpu))  return false;
    if (!create_device())                           return false;
    if (!create_queues())                           return false;
    if (!init_dxc())                                return false;
    populate_capabilities();
    settings_ = desc.initial_settings;

    // NvAPI — global init. Must be paired with NvAPI_Unload at shutdown.
    // Failure is non-fatal: Reflex just stays unavailable on this adapter.
#if CARDINAL_HAS_NVAPI
    {
        const NvAPI_Status s = NvAPI_Initialize();
        nvapi_initialised_ = (s == NVAPI_OK);
        if (nvapi_initialised_) {
            cardinal::log::infof("rhi/d3d12",
                "NvAPI initialised — Reflex available on NVIDIA adapters");
        } else {
            cardinal::log::infof("rhi/d3d12",
                "NvAPI_Initialize returned %d (Reflex disabled)",
                static_cast<int>(s));
        }
    }
#endif

    cardinal::log::infof("rhi/d3d12", "Device initialised (%s)", adapter_name_);
    return true;
}

bool D3D12Device::create_factory() {
    UINT flags = 0;
    D3D_CHECK(CreateDXGIFactory2(flags, IID_PPV_ARGS(&factory_)));

    // Probe DXGI tearing support (Win10+ / variable refresh). Without this
    // flag the swapchain caps Present(0,0) to the display refresh rate via
    // DWM composition, regardless of what the application asks for.
    {
        ComPtr<IDXGIFactory5> f5;
        if (SUCCEEDED(factory_.As(&f5))) {
            BOOL supp = FALSE;
            if (SUCCEEDED(f5->CheckFeatureSupport(
                    DXGI_FEATURE_PRESENT_ALLOW_TEARING, &supp, sizeof(supp))))
            {
                tearing_supported_ = (supp == TRUE);
            }
        }
        cardinal::log::infof("rhi/d3d12",
            "DXGI tearing support: %s (uncapped present requires this)",
            tearing_supported_ ? "YES" : "NO");
    }
    return true;
}

bool D3D12Device::select_adapter(bool prefer_discrete) {
    // EnumAdapterByGpuPreference is the modern path — we ask DXGI to sort by
    // performance preference. Fall back to manual enum if it's unavailable.
    DXGI_GPU_PREFERENCE pref = prefer_discrete
        ? DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE
        : DXGI_GPU_PREFERENCE_UNSPECIFIED;

    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> adp;
        if (factory_->EnumAdapterByGpuPreference(i, pref, IID_PPV_ARGS(&adp))
                == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        DXGI_ADAPTER_DESC1 d;
        if (FAILED(adp->GetDesc1(&d))) continue;
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (SUCCEEDED(D3D12CreateDevice(adp.Get(), D3D_FEATURE_LEVEL_11_0,
                                        __uuidof(ID3D12Device), nullptr))) {
            adapter_ = cardinal::move(adp);
            WideCharToMultiByte(CP_UTF8, 0, d.Description, -1,
                                adapter_name_, sizeof(adapter_name_) - 1,
                                nullptr, nullptr);
            caps_.vram_bytes = d.DedicatedVideoMemory;
            cardinal::strncpy(caps_.vendor_name,
                d.VendorId == 0x10DE ? "NVIDIA Corporation" :
                d.VendorId == 0x1002 ? "Advanced Micro Devices" :
                d.VendorId == 0x8086 ? "Intel" : "Unknown",
                sizeof(caps_.vendor_name) - 1);
            return true;
        }
    }
    cardinal::log::errorf("rhi/d3d12", "no D3D12-capable adapter found");
    return false;
}

bool D3D12Device::create_device() {
    D3D_CHECK(D3D12CreateDevice(adapter_.Get(), D3D_FEATURE_LEVEL_11_0,
                                IID_PPV_ARGS(&device_)));
    return true;
}

bool D3D12Device::create_queues() {
    D3D12_COMMAND_QUEUE_DESC q{};
    q.Type     = D3D12_COMMAND_LIST_TYPE_DIRECT;
    q.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    q.Flags    = D3D12_COMMAND_QUEUE_FLAG_NONE;
    D3D_CHECK(device_->CreateCommandQueue(&q, IID_PPV_ARGS(&gfx_queue_)));
    return true;
}

bool D3D12Device::init_dxc() {
    D3D_CHECK(DxcCreateInstance(CLSID_DxcUtils,    IID_PPV_ARGS(&dxc_utils_)));
    D3D_CHECK(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxc_compiler_)));
    D3D_CHECK(dxc_utils_->CreateDefaultIncludeHandler(&dxc_includes_));
    return true;
}

void D3D12Device::populate_capabilities() {
    // Probe the bits the engine cares about. D3D12 makes most of what we want
    // mandatory at FL_12_0+ (mesh shaders need _2 options struct), so we just
    // expose what's actually queryable.
    D3D12_FEATURE_DATA_D3D12_OPTIONS  o{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 o7{};
    D3D12_FEATURE_DATA_SHADER_MODEL   sm{D3D_SHADER_MODEL_6_8};

    device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS,  &o,  sizeof(o));
    device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof(o5));
    device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &o7, sizeof(o7));
    device_->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL,   &sm, sizeof(sm));

    caps_.dynamic_rendering        = true;   // D3D12 has always had OMSetRenderTargets
    caps_.synchronization2         = true;
    caps_.buffer_device_address    = true;   // GPU VAs are first-class in D3D12
    caps_.descriptor_indexing      = true;
    caps_.runtime_descriptor_array = true;
    caps_.scalar_block_layout      = true;
    caps_.timeline_semaphore       = true;   // ID3D12Fence is timeline-equivalent
    caps_.shader_int16             = true;
    caps_.shader_float16           = (sm.HighestShaderModel >= D3D_SHADER_MODEL_6_2);
    caps_.shader_int8              = (sm.HighestShaderModel >= D3D_SHADER_MODEL_6_4);
    caps_.storage_buffer_16bit     = true;
    caps_.wave_intrinsics          = (sm.HighestShaderModel >= D3D_SHADER_MODEL_6_0);
    caps_.mesh_shader              = (o7.MeshShaderTier >= D3D12_MESH_SHADER_TIER_1);
    caps_.task_shader              = caps_.mesh_shader;     // amplification
    caps_.acceleration_structure   = (o5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0);
    caps_.ray_tracing_pipeline     = (o5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0);
    caps_.ray_query                = (o5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1);
    caps_.nvidia_dlss_capable      = (cardinal::strstr(caps_.vendor_name, "NVIDIA") != nullptr)
                                  && caps_.ray_tracing_pipeline;
    caps_.nvidia_framegen_capable  = caps_.nvidia_dlss_capable;
    caps_.amd_fsr3_capable         = (cardinal::strstr(caps_.vendor_name, "AMD") != nullptr ||
                                      cardinal::strstr(caps_.vendor_name, "Advanced Micro") != nullptr);

    cardinal::strncpy(caps_.gpu_arch,
        caps_.mesh_shader && caps_.ray_query ? "RTX 30+ class" :
        caps_.acceleration_structure         ? "RT-capable"   :
                                               "raster-only",
        sizeof(caps_.gpu_arch) - 1);
}

cardinal::unique_ptr<Buffer> D3D12Device::create_buffer(const BufferDesc& desc) {
    if (desc.size == 0) return nullptr;

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = desc.cpu_writable ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width              = desc.size;
    rd.Height             = 1;
    rd.DepthOrArraySize   = 1;
    rd.MipLevels          = 1;
    rd.Format             = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count   = 1;
    rd.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags              = D3D12_RESOURCE_FLAG_NONE;
    // ALLOW_UNORDERED_ACCESS is *illegal* on an UPLOAD-heap resource and is
    // only meaningful for GPU-writable (UAV) buffers. Our cpu_writable
    // storage buffers (per-frame light/material uploads) live on the UPLOAD
    // heap and are bound read-only as root SRVs (StructuredBuffer<T>), so
    // requesting UAV here makes CreateCommittedResource fail with
    // E_INVALIDARG. Only DEFAULT-heap storage buffers may be UAVs.
    if (!desc.cpu_writable &&
        (static_cast<u32>(desc.usage) & static_cast<u32>(BufferUsage::Storage)))
        rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    const D3D12_RESOURCE_STATES initial =
        desc.cpu_writable ? D3D12_RESOURCE_STATE_GENERIC_READ
                          : D3D12_RESOURCE_STATE_COMMON;

    ComPtr<ID3D12Resource> res;
    HRESULT hr = device_->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &rd, initial, nullptr, IID_PPV_ARGS(&res));
    if (FAILED(hr)) {
        cardinal::log::errorf("rhi/d3d12", "CreateCommittedResource (buffer) failed (%s)",
            hr_str(hr));
        return nullptr;
    }
    return cardinal::make_unique<D3D12Buffer>(cardinal::move(res), desc.size, desc.cpu_writable);
}

cardinal::unique_ptr<Texture> D3D12Device::create_texture(const TextureDesc& desc) {
    auto t = cardinal::make_unique<D3D12Texture>();
    if (!t->initialize(device_.Get(), desc)) return nullptr;
    return t;
}

ShaderBlob D3D12Device::compile_shader(ShaderStage stage, const char* hlsl, const char* entry) {
    ShaderBlob out;
    if (hlsl == nullptr || entry == nullptr) return out;

    LPCWSTR target = L"vs_6_5";
    switch (stage) {
        case ShaderStage::Vertex:   target = L"vs_6_5"; break;
        case ShaderStage::Fragment: target = L"ps_6_5"; break;
        case ShaderStage::Compute:  target = L"cs_6_5"; break;
    }
    // Convert entry point char → wide.
    wchar_t entry_w[64]{};
    MultiByteToWideChar(CP_UTF8, 0, entry, -1, entry_w, 63);

    DxcBuffer src{};
    src.Ptr      = hlsl;
    src.Size     = cardinal::strlen(hlsl);
    src.Encoding = DXC_CP_UTF8;

    cardinal::array<LPCWSTR, 8> args = {
        L"-E", entry_w,
        L"-T", target,
        L"-Zi",
        L"-Qstrip_reflect",
        L"-HV", L"2021",
    };

    ComPtr<IDxcResult> result;
    HRESULT hr = dxc_compiler_->Compile(&src, args.data(), static_cast<UINT32>(args.size()),
                                        dxc_includes_.Get(), IID_PPV_ARGS(&result));
    if (FAILED(hr)) {
        cardinal::log::errorf("rhi/d3d12", "DXC Compile call failed (%s)", hr_str(hr));
        return out;
    }
    HRESULT compile_status = S_OK;
    result->GetStatus(&compile_status);
    if (FAILED(compile_status)) {
        ComPtr<IDxcBlobUtf8> errors;
        result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
        if (errors && errors->GetStringLength() > 0) {
            cardinal::log::errorf("rhi/d3d12", "DXC error: %s", errors->GetStringPointer());
        } else {
            cardinal::log::errorf("rhi/d3d12", "DXC failed (%s)", hr_str(compile_status));
        }
        return out;
    }
    ComPtr<IDxcBlob> code;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&code), nullptr);
    if (!code || code->GetBufferSize() == 0) {
        cardinal::log::errorf("rhi/d3d12", "DXC returned empty object");
        return out;
    }
    out.bytes.resize(code->GetBufferSize());
    cardinal::memcpy(out.bytes.data(), code->GetBufferPointer(), code->GetBufferSize());
    return out;
}

// Live VRAM telemetry — DXGI 4 path. QueryVideoMemoryInfo is essentially
// free (one user-mode call into dxgi.dll); safe per-frame.
Device::VramSnapshot D3D12Device::query_vram_usage() const noexcept {
    VramSnapshot s{};
    if (!adapter_) return s;
    ComPtr<IDXGIAdapter4> a4;
    if (SUCCEEDED(adapter_.As(&a4))) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        if (SUCCEEDED(a4->QueryVideoMemoryInfo(
                /*NodeIndex=*/0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
            s.budget_bytes                    = static_cast<u64>(info.Budget);
            s.current_usage_bytes             = static_cast<u64>(info.CurrentUsage);
            s.available_for_reservation_bytes = static_cast<u64>(info.AvailableForReservation);
            s.current_reservation_bytes       = static_cast<u64>(info.CurrentReservation);
        }
    }
    return s;
}

cardinal::unique_ptr<Pipeline> D3D12Device::create_pipeline(const PipelineDesc& desc) {
    if (!desc.vertex_shader.ok() || !desc.fragment_shader.ok()) {
        cardinal::log::errorf("rhi/d3d12", "create_pipeline: shaders not compiled");
        return nullptr;
    }

    // Root signature: optional 32-bit constants block at slot 0 when the
    // pipeline declares one. push_constant_size is in BYTES; the D3D12
    // root-constants slot counts in DWORDs (4 B each), so we round up
    // for the slot size while passing the byte count up to D3D12 via
    // SetGraphicsRoot32BitConstants below. Visible to all stages so
    // both VS and PS see the block (matches the Vulkan side, which
    // enables both VK_SHADER_STAGE_VERTEX_BIT + FRAGMENT_BIT).
    // Root parameter layout:
    //   [0]            32-bit constants (b0)         — iff push_constant_size>0
    //   [base .. base+N) root SRV (t0, t1, …)        — one per storage slot
    // where base = (push_constant_size>0 ? 1 : 0). Keeping the push
    // block at index 0 preserves set_push_constants' hardcoded root
    // index 0. Root SRVs (not descriptor tables) need no descriptor
    // heap — they bind straight from a GPU virtual address, mirroring
    // the Vulkan storage-buffer descriptor on the other backend.
    // A D3D12 root signature is capped at 64 DWORDs. 32-bit root constants
    // cost one DWORD *per DWORD of payload*, so a large push block (e.g. the
    // 240 B / 60-DWORD scene block) plus root SRVs (2 DWORDs each) + a
    // descriptor table (1 DWORD) blows the budget and CreateRootSignature
    // fails with E_INVALIDARG. Demote big blocks to a root CBV (b0, a flat
    // 2 DWORDs) backed by the per-frame constant ring; keep the cheaper
    // inline root-constants path for genuinely small blocks (<=16 DWORDs,
    // e.g. the shadow pass's single mat4). HLSL `cbuffer Push : register(b0)`
    // is identical either way, so no shader change is needed.
    const bool push_as_cbv = desc.push_constant_size > 64;
    cardinal::vector<D3D12_ROOT_PARAMETER> root_params;
    if (desc.push_constant_size > 0) {
        D3D12_ROOT_PARAMETER rp{};
        if (push_as_cbv) {
            rp.ParameterType             = D3D12_ROOT_PARAMETER_TYPE_CBV;
            rp.Descriptor.ShaderRegister = 0;             // b0
            rp.Descriptor.RegisterSpace  = 0;
        } else {
            rp.ParameterType            = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            rp.Constants.ShaderRegister = 0;              // b0
            rp.Constants.RegisterSpace  = 0;
            rp.Constants.Num32BitValues = (desc.push_constant_size + 3) / 4;
        }
        rp.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        root_params.push_back(rp);
    }
    for (u32 s = 0; s < desc.storage_buffer_slots; ++s) {
        D3D12_ROOT_PARAMETER rp{};
        rp.ParameterType             = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rp.Descriptor.ShaderRegister = s;                 // t0, t1, …
        rp.Descriptor.RegisterSpace  = 0;
        rp.ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;
        root_params.push_back(rp);
    }
    // Sampled textures (shadow map): one descriptor-table root param
    // holding a contiguous SRV range at registers t[storage_slots ..],
    // matching the Vulkan binding convention (textures follow buffers).
    // `range` must outlive D3D12SerializeRootSignature below — it's a
    // synchronous call, so a function-local is fine.
    D3D12_DESCRIPTOR_RANGE srv_range{};
    if (desc.sampled_texture_slots > 0) {
        srv_range.RangeType          = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        srv_range.NumDescriptors     = desc.sampled_texture_slots;
        srv_range.BaseShaderRegister = desc.storage_buffer_slots;   // t after storage
        srv_range.RegisterSpace      = 0;
        srv_range.OffsetInDescriptorsFromTableStart =
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        D3D12_ROOT_PARAMETER rp{};
        rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rp.DescriptorTable.NumDescriptorRanges = 1;
        rp.DescriptorTable.pDescriptorRanges   = &srv_range;
        rp.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        root_params.push_back(rp);
    }
    // Static sampler s0 for the sampled textures — clamp + linear, same
    // policy as the Vulkan device default sampler. Static samplers live
    // in the root signature and consume no root-parameter slot.
    D3D12_STATIC_SAMPLER_DESC ss{};
    ss.Filter           = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    ss.AddressU         = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    ss.AddressV         = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    ss.AddressW         = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    ss.MaxLOD           = D3D12_FLOAT32_MAX;
    ss.ShaderRegister   = 0;                               // s0
    ss.RegisterSpace    = 0;
    ss.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    if (!root_params.empty()) {
        rsd.NumParameters = static_cast<UINT>(root_params.size());
        rsd.pParameters   = root_params.data();
    }
    if (desc.sampled_texture_slots > 0) {
        rsd.NumStaticSamplers = 1;
        rsd.pStaticSamplers   = &ss;
    }

    // Fail loudly *here* rather than via an opaque CreateRootSignature
    // E_INVALIDARG if some future slot configuration still overflows the
    // 64-DWORD ceiling. push: CBV=2, else ceil(bytes/4); root SRV=2 each;
    // sampled-texture descriptor table=1.
    {
        const u32 push_dwords = (desc.push_constant_size == 0) ? 0u
                              : (push_as_cbv ? 2u : (desc.push_constant_size + 3u) / 4u);
        const u32 total_dwords = push_dwords
                               + 2u * desc.storage_buffer_slots
                               + (desc.sampled_texture_slots > 0 ? 1u : 0u);
        if (total_dwords > 64u) {
            cardinal::log::errorf("rhi/d3d12",
                "root signature too large: %u DWORDs > 64 "
                "(push=%uB storage=%u tex=%u)",
                total_dwords, desc.push_constant_size,
                desc.storage_buffer_slots, desc.sampled_texture_slots);
            return nullptr;
        }
    }

    ComPtr<ID3DBlob> blob, err;
    HRESULT hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1,
                                             &blob, &err);
    if (FAILED(hr)) {
        cardinal::log::errorf("rhi/d3d12", "SerializeRootSignature failed (%s%s)",
            hr_str(hr), err ? static_cast<const char*>(err->GetBufferPointer()) : "");
        return nullptr;
    }
    ComPtr<ID3D12RootSignature> root;
    hr = device_->CreateRootSignature(0, blob->GetBufferPointer(),
                                      blob->GetBufferSize(), IID_PPV_ARGS(&root));
    if (FAILED(hr)) {
        cardinal::log::errorf("rhi/d3d12", "CreateRootSignature failed (%s)", hr_str(hr));
        return nullptr;
    }

    // Input layout from PipelineDesc::vertex_attribs.
    cardinal::vector<D3D12_INPUT_ELEMENT_DESC> ie;
    ie.reserve(desc.vertex_attribs.size());
    static const char* semantic_names[] = { "POSITION", "COLOR", "TEXCOORD", "NORMAL", "TANGENT" };
    for (const auto& a : desc.vertex_attribs) {
        D3D12_INPUT_ELEMENT_DESC el{};
        // Match the HLSL we use: POSITION0 / COLOR0 / TEXCOORD<n> by location.
        const u32 sem_idx = a.location;
        const char* sem   = sem_idx < cardinal::size(semantic_names)
                          ? semantic_names[sem_idx] : "TEXCOORD";
        // Heuristic: location 0 -> POSITION, 1 -> COLOR, 2+ -> TEXCOORD with
        // increasing semantic indices. Matches what our triangle sample uses.
        u32 sem_index = 0;
        if (sem_idx >= 2) { sem = "TEXCOORD"; sem_index = sem_idx - 2; }
        el.SemanticName      = sem;
        el.SemanticIndex     = sem_index;
        el.Format            = to_dxgi(a.format);
        el.InputSlot         = 0;
        el.AlignedByteOffset = a.offset;
        el.InputSlotClass    = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
        ie.push_back(el);
    }

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature                = root.Get();
    psoDesc.VS.pShaderBytecode            = desc.vertex_shader.bytes.data();
    psoDesc.VS.BytecodeLength             = desc.vertex_shader.bytes.size();
    psoDesc.PS.pShaderBytecode            = desc.fragment_shader.bytes.data();
    psoDesc.PS.BytecodeLength             = desc.fragment_shader.bytes.size();
    psoDesc.InputLayout                   = { ie.data(), static_cast<UINT>(ie.size()) };
    psoDesc.PrimitiveTopologyType         = to_d3d_topology_type(desc.topology);
    psoDesc.SampleMask                    = UINT_MAX;
    // color_format == Unknown ⇒ depth-only PSO (0 render targets) — the
    // shadow-map depth pass binds OMSetRenderTargets(0,...). The PSO's
    // RT count must match, so honour Unknown rather than forcing 1.
    const bool depth_only = (desc.color_format == Format::Unknown);
    psoDesc.NumRenderTargets              = depth_only ? 0u : 1u;
    if (!depth_only) psoDesc.RTVFormats[0] = to_dxgi(desc.color_format);
    psoDesc.SampleDesc.Count              = 1;

    // Sensible raster defaults — matches the Vulkan path for parity.
    psoDesc.RasterizerState.FillMode      = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode      = D3D12_CULL_MODE_NONE;
    psoDesc.RasterizerState.FrontCounterClockwise = TRUE;
    psoDesc.RasterizerState.DepthClipEnable       = TRUE;

    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

    // Depth — opt-in per pipeline. When the caller sets depth_format we
    // tell the PSO about the matching DSV format and enable depth test +
    // write. Function = LESS so the nearer fragment wins.
    if (desc.depth_format != Format::Unknown) {
        psoDesc.DepthStencilState.DepthEnable    = desc.depth_test  ? TRUE : FALSE;
        psoDesc.DepthStencilState.DepthWriteMask = desc.depth_write
            ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
        psoDesc.DepthStencilState.DepthFunc      = D3D12_COMPARISON_FUNC_LESS;
        psoDesc.DepthStencilState.StencilEnable  = FALSE;
        psoDesc.DSVFormat = (desc.depth_format == Format::D32_Float)
            ? DXGI_FORMAT_D32_FLOAT : DXGI_FORMAT_UNKNOWN;
    } else {
        psoDesc.DepthStencilState.DepthEnable    = FALSE;
        psoDesc.DepthStencilState.StencilEnable  = FALSE;
    }

    ComPtr<ID3D12PipelineState> pso;
    hr = device_->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso));
    if (FAILED(hr)) {
        cardinal::log::errorf("rhi/d3d12", "CreateGraphicsPipelineState failed (%s)", hr_str(hr));
        return nullptr;
    }
    return cardinal::make_unique<D3D12Pipeline>(cardinal::move(root), cardinal::move(pso),
                                           desc.topology, desc.vertex_stride,
                                           desc.push_constant_size,
                                           desc.storage_buffer_slots,
                                           desc.sampled_texture_slots,
                                           push_as_cbv);
}

// -----------------------------------------------------------------------------
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

}  // namespace

// -----------------------------------------------------------------------------
// Factory entry point
// -----------------------------------------------------------------------------
cardinal::unique_ptr<Device> create_d3d12_device(const DeviceDesc& desc) {
    auto d = cardinal::make_unique<D3D12Device>();
    if (!d->initialize(desc)) return nullptr;
    return d;
}

// -----------------------------------------------------------------------------
// d3d12_interop.hpp implementations — only valid when the underlying object
// is actually D3D12. Backend mismatch returns null/zero.
// -----------------------------------------------------------------------------
D3D12DeviceHandles d3d12_handles(Device* dev) {
    D3D12DeviceHandles h{};
    if (dev == nullptr || dev->backend() != Backend::D3D12) return h;
    auto* d = static_cast<D3D12Device*>(dev);
    h.device    = d->device();
    h.gfx_queue = d->gfx_queue();
    h.factory   = d->factory();
    return h;
}

D3D12SwapchainHandles d3d12_handles(Swapchain* sw) {
    D3D12SwapchainHandles h{};
    if (sw == nullptr) return h;
    auto* s = static_cast<D3D12Swapchain*>(sw);
    h.swapchain          = s->swap_chain();
    h.back_buffer_format = s->back_buffer_format();
    h.back_buffer_count  = s->back_buffer_count();
    return h;
}

ID3D12GraphicsCommandList* d3d12_current_cmd(Swapchain* sw) {
    if (sw == nullptr) return nullptr;
    return static_cast<D3D12Swapchain*>(sw)->current_cmd();
}
u32 d3d12_acquired_image(Swapchain* sw) {
    if (sw == nullptr) return 0;
    return static_cast<D3D12Swapchain*>(sw)->current_index();
}
ID3D12Resource* d3d12_back_buffer(Swapchain* sw, u32 i) {
    if (sw == nullptr) return nullptr;
    return static_cast<D3D12Swapchain*>(sw)->back_buffer(i);
}
D3D12_CPU_DESCRIPTOR_HANDLE d3d12_back_buffer_rtv(Swapchain* sw, u32 i) {
    if (sw == nullptr) { D3D12_CPU_DESCRIPTOR_HANDLE h{}; return h; }
    return static_cast<D3D12Swapchain*>(sw)->back_buffer_rtv(i);
}
ID3D12Resource* d3d12_viewport_image(Swapchain* sw) {
    if (sw == nullptr) return nullptr;
    return static_cast<D3D12Swapchain*>(sw)->viewport_image();
}
ID3D12Resource* d3d12_viewport_image(Swapchain* sw, u32 id) {
    if (sw == nullptr) return nullptr;
    return static_cast<D3D12Swapchain*>(sw)->viewport_image(id);
}
u32 d3d12_viewport_count(Swapchain* sw) {
    if (sw == nullptr) return 0u;
    return static_cast<D3D12Swapchain*>(sw)->d3d12_viewport_count();
}

}  // namespace cardinal::rhi

#endif  // CARDINAL_PLATFORM_WINDOWS
