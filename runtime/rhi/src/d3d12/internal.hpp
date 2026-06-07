#pragma once

// =============================================================================
// Cardinal RHI — D3D12 backend internal header.
//
// Phase 2 of the per-backend split: hosts the includes + ComPtr alias +
// inline helpers + the three small "leaf" classes (D3D12Buffer /
// Texture / Pipeline — fully inline, ~40-100 lines each) + forward
// declarations for the two big classes (D3D12Swapchain, D3D12Device)
// whose method bodies live in their own TUs (swapchain.cpp,
// device.cpp). This split lets the linker rebuild only the changed
// TU instead of the whole 2,072-line monolith on every edit.
//
// File is *.hpp not *.h to match the rest of the engine's header
// convention; it is private to the d3d12/ subdirectory (no consumer
// outside this folder).
//
// Windows-only — the non-Windows create_d3d12_device fallback lives
// in device.cpp wrapped in #if !CARDINAL_PLATFORM_WINDOWS.
// =============================================================================

#include <cardinal/rhi/rhi.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/platform.hpp>

#if defined(_WIN32)
#include <cardinal/rhi/d3d12_interop.hpp>
#endif

#if CARDINAL_PLATFORM_WINDOWS

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

// NvAPI — used here ONLY for Reflex on D3D12. The header is noisy under /WX
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

#include <cardinal/core/std/algorithm.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/core/std/cstdio.hpp>
#include <cardinal/core/std/cstring.hpp>
#include <cardinal/core/std/utility.hpp>

namespace cardinal::rhi {

using ::Microsoft::WRL::ComPtr;

// -----------------------------------------------------------------------------
// Common helpers — inline so multiple TUs sharing this header don't
// hit ODR. The D3D_CHECK macro returns from the enclosing function on
// FAILED(hr); intended for the bool-returning init paths.
// -----------------------------------------------------------------------------
inline constexpr u32 kFramesInFlight = 2;

inline const char* hr_str(HRESULT hr) {
    static thread_local char buf[64];
    cardinal::snprintf(buf, sizeof(buf), "0x%08lx", hr);
    return buf;
}

#define D3D_CHECK(expr) do { HRESULT _hr = (expr);                            \
    if (FAILED(_hr)) {                                                        \
        cardinal::log::errorf("rhi/d3d12", "%s failed (%s)", #expr, hr_str(_hr)); \
        return false;                                                         \
    } } while (0)

inline DXGI_FORMAT to_dxgi(Format f) {
    switch (f) {
        case Format::R8G8B8A8_UNORM:    return DXGI_FORMAT_R8G8B8A8_UNORM;
        case Format::B8G8R8A8_UNORM:    return DXGI_FORMAT_B8G8R8A8_UNORM;
        case Format::R32G32_Float:      return DXGI_FORMAT_R32G32_FLOAT;
        case Format::R32G32B32_Float:   return DXGI_FORMAT_R32G32B32_FLOAT;
        case Format::R32G32B32A32_Float:return DXGI_FORMAT_R32G32B32A32_FLOAT;
    }
    return DXGI_FORMAT_UNKNOWN;
}

inline D3D12_PRIMITIVE_TOPOLOGY_TYPE to_d3d_topology_type(PrimitiveTopology t) {
    switch (t) {
        case PrimitiveTopology::TriangleList:
        case PrimitiveTopology::TriangleStrip: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        case PrimitiveTopology::LineList:
        case PrimitiveTopology::LineStrip:     return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
        case PrimitiveTopology::PointList:     return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
    }
    return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
}

inline D3D12_PRIMITIVE_TOPOLOGY to_d3d_topology(PrimitiveTopology t) {
    switch (t) {
        case PrimitiveTopology::TriangleList:  return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        case PrimitiveTopology::TriangleStrip: return D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
        case PrimitiveTopology::LineList:      return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
        case PrimitiveTopology::LineStrip:     return D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
        case PrimitiveTopology::PointList:     return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
    }
    return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
}

}  // namespace cardinal::rhi

// -----------------------------------------------------------------------------
// Inline "leaf" classes (Buffer / Texture / Pipeline) extracted verbatim from
// the original monolith. They have no out-of-line methods, so the full class
// bodies live in this header.
// -----------------------------------------------------------------------------
namespace cardinal::rhi {

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
        // ALLOW_DEPTH_STENCIL — and therefore the DEPTH_WRITE create state
        // + an optimised depth clear value — is keyed on the *usage*, not
        // the format. A plain Sampled colour texture has neither flag, and
        // D3D12 rejects DEPTH_WRITE (or a clear value) on a resource that
        // lacks ALLOW_DEPTH_STENCIL with E_INVALIDARG. Depth targets must
        // still start in DEPTH_WRITE so the shadow-pass barrier cycle
        // (begin/end_shadow_pass) stays balanced.
        const bool is_depth_target =
            (d.usage & static_cast<u32>(TextureUsage::DepthRenderTarget)) != 0u;

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

        state_ = is_depth_target ? D3D12_RESOURCE_STATE_DEPTH_WRITE
                                 : D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        if (FAILED(dev->CreateCommittedResource(
                &heap, D3D12_HEAP_FLAG_NONE, &rd, state_,
                is_depth_target ? &cv : nullptr, IID_PPV_ARGS(&res_)))) {
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

    // Compute pipeline ctor (Device::create_compute_pipeline). Root layout:
    //   [0]          push block (b0)        — iff push_constant_size>0
    //   [base..)     root SRV (t0..)        — read-only storage_buffer_slots
    //   [uav_base..) root UAV (u0..)        — read-write uav_slots
    // No input layout / sampler table (compute has neither). is_compute_ flips
    // the recorder onto SetComputeRoot*.
    D3D12Pipeline(ComPtr<ID3D12RootSignature> rs, ComPtr<ID3D12PipelineState> pso,
                  u32 push_constant_size, bool push_is_cbv,
                  u32 storage_buffer_slots, u32 uav_slots)
        : root_(cardinal::move(rs)), pso_(cardinal::move(pso))
        , topology_(D3D_PRIMITIVE_TOPOLOGY_UNDEFINED)
        , push_constant_size_(push_constant_size)
        , push_is_cbv_(push_is_cbv)
        , storage_buffer_slots_(storage_buffer_slots)
        , storage_root_base_(push_constant_size > 0 ? 1u : 0u)
        , uav_slots_(uav_slots)
        , uav_root_base_(storage_root_base_ + storage_buffer_slots)
        , is_compute_(true) {}

    bool                   is_compute()          const noexcept override { return is_compute_; }
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
    u32                    uav_slots()            const noexcept { return uav_slots_; }
    u32                    uav_root_base()        const noexcept { return uav_root_base_; }

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
    u32                         uav_slots_{0};
    u32                         uav_root_base_{0};
    bool                        is_compute_{false};
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
    void bind_storage_buffer_uav(u32 slot, Buffer* b) override;   // compute RW UAV
    void dispatch(u32 gx, u32 gy = 1, u32 gz = 1) override;       // compute dispatch
    void dispatch_indirect(Buffer* args, u32 args_offset) override;  // GPU-driven
    void uav_barrier(Buffer* b) override;                        // RAW/WAW flush
    void transition_buffer_state(Buffer* b, ResourceState before,
                                 ResourceState after) override;
    void set_shading_rate(ShadingRate rate) override;            // VRS (Block 12)
    void bind_shading_rate_image(Texture* per_tile_rate_image) override;
    void begin_shadow_pass(Texture* depth) override;
    void end_shadow_pass() override;
    void bind_sampled_texture(u32 slot, Texture* tex) override;

    // GPU debug-marker regions (PIX / RenderDoc / Nsight). Implemented in
    // commands.cpp via ID3D12GraphicsCommandList::BeginEvent/EndEvent/SetMarker.
    void begin_event(const char* name) override;
    void end_event() override;
    void insert_marker(const char* name) override;

    // Multi-queue sync (AEGIS Block 10). signal_fence signals the GRAPHICS
    // queue's progress to `f` (returns the value signalled); wait_fence makes
    // the graphics queue wait for `f` to reach `value` before proceeding —
    // the graphics<->async-compute handshake. Implemented in compute.cpp.
    u64  signal_fence(Fence* f) override;
    void wait_fence  (Fence* f, u64 value) override;

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
    ComPtr<ID3D12CommandSignature>   dispatch_sig_;   // lazily-built DISPATCH indirect sig

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
    cardinal::unique_ptr<Pipeline> create_compute_pipeline(const ComputePipelineDesc& desc) override;
    cardinal::unique_ptr<Texture>  create_texture(const TextureDesc& desc) override;

    // RT not yet wired in the D3D12 backend — return nullptr so callers see
    // it as unsupported.
    cardinal::unique_ptr<AccelerationStructure> create_blas(const BlasDesc&) override { return nullptr; }
    cardinal::unique_ptr<AccelerationStructure> create_tlas(const TlasDesc&) override { return nullptr; }

    ShaderBlob compile_shader(ShaderStage stage, const char* hlsl, const char* entry) override;

    VramSnapshot query_vram_usage() const noexcept override;

    // Async compute (AEGIS Block 10) — implemented in compute.cpp. create_fence
    // makes an ID3D12Fence-backed timeline Fence; create_async_compute_queue
    // returns a D3D12ComputeQueue bound to the dedicated COMPUTE queue (nullptr
    // when the device has none / caps.async_compute is false).
    cardinal::unique_ptr<Fence>        create_fence(u64 initial_value = 0) override;
    cardinal::unique_ptr<ComputeQueue> create_async_compute_queue() override;

    // Interop accessors.
    ID3D12Device*       device()        const noexcept { return device_.Get(); }
    ID3D12CommandQueue* gfx_queue()     const noexcept { return gfx_queue_.Get(); }
    ID3D12CommandQueue* compute_queue() const noexcept { return compute_queue_.Get(); }
    IDXGIFactory6*      factory()       const noexcept { return factory_.Get(); }

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
    ComPtr<ID3D12CommandQueue>  compute_queue_;   // dedicated async-compute lane

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

}  // namespace cardinal::rhi

#endif  // CARDINAL_PLATFORM_WINDOWS
