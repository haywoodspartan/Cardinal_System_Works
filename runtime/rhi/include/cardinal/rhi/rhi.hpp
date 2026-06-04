#pragma once

#include <cardinal/core/types.hpp>
#include <cardinal/core/containers.hpp>

// =============================================================================
// Cardinal RHI — Render Hardware Interface.
//
// Phase 2.5 status:
//   - Vulkan backend: instance, device, surface, swapchain, frame sync,
//     dynamic-rendering clear loop, runtime HLSL→SPIR-V via DXC, public
//     Buffer + Pipeline + bind/draw API. ✓
//   - D3D12 backend: not yet implemented (Phase 2.6).
//   - Pipeline API surface is intentionally minimal — shapes itself when
//     we add textures, descriptors, push constants, multiple render
//     targets etc. Right now: vertex+fragment shaders, vertex attribs,
//     single color target.
// =============================================================================
namespace cardinal::rhi {

// ---------------------------------------------------------------------------
// Backend / device
// ---------------------------------------------------------------------------
enum class Backend : u32 {
    Auto = 0,
    Vulkan,
    D3D12,
};

enum class ShaderStage : u32 {
    Vertex,
    Fragment,
    Compute,
};

// Backend-native compiled bytecode:
//   Vulkan : SPIR-V words (cast to u32* on use)
//   D3D12  : DXIL bytes
struct ShaderBlob {
    cardinal::vector<u8> bytes;
    bool ok() const noexcept { return !bytes.empty(); }
};

// ---------------------------------------------------------------------------
// Vertex / format
// ---------------------------------------------------------------------------
enum class Format : u32 {
    Unknown = 0,
    R32_Float,
    R32G32_Float,
    R32G32B32_Float,
    R32G32B32A32_Float,
    R8G8B8A8_UNORM,
    R8G8B8A8_SRGB,
    B8G8R8A8_UNORM,
    B8G8R8A8_SRGB,
    D32_Float,             // 32-bit float depth — used by depth attachments
};

struct VertexAttrib {
    u32    location;   // matches HLSL semantic order via SV_POSITION/COLOR0 etc
    Format format;
    u32    offset;     // bytes from start of vertex
};

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------
enum class BufferUsage : u32 {
    Vertex                  = 1u << 0,
    Index                   = 1u << 1,
    Uniform                 = 1u << 2,
    Storage                 = 1u << 3,
    AccelerationStructureBuild = 1u << 4,  // input geometry to AS build (vert/idx)
    ShaderDeviceAddress     = 1u << 5,     // expose buffer's GPU virtual address
};

struct BufferDesc {
    usize size;
    u32   usage;          // bitmask of BufferUsage
    bool  cpu_writable;   // true: host-visible (mappable). false: GPU-only.
                          //    GPU-only currently still allocates host-visible —
                          //    proper staging upload lands in Phase 2.5-E.
};

class Buffer {
public:
    virtual ~Buffer() = default;

    Buffer(const Buffer&)            = delete;
    Buffer& operator=(const Buffer&) = delete;

    virtual usize size() const noexcept = 0;

    // Copy `size` bytes from `data` into the buffer at `offset`. Requires
    // BufferDesc::cpu_writable == true. Currently each call maps + memcpy +
    // unmaps; Phase 2.5-E will add a persistent-mapped fast path.
    virtual void upload(const void* data, usize size, usize offset = 0) = 0;

    // GPU virtual address — only valid if the buffer was created with
    // BufferUsage::ShaderDeviceAddress. Returns 0 otherwise. Used for
    // ray-tracing acceleration structure inputs and bindless / buffer
    // descriptor patterns.
    virtual u64 device_address() const noexcept = 0;

protected:
    Buffer() = default;
};

// ---------------------------------------------------------------------------
// Texture — a 2D GPU image. Added for the shadow-mapping arc; the first
// consumer is a depth render target the renderer draws the scene into
// from the light's POV (pass 1) then samples in the main PS (pass 2)
// for the shadow test. Kept deliberately minimal — like the storage-
// buffer primitive, it grows as needs land (mip levels, array layers,
// cube faces, colour RTs for post-process, etc.).
//
// Backend mapping:
//   Vulkan : VkImage + VkImageView + VMA alloc. DepthRenderTarget →
//            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT; Sampled →
//            VK_IMAGE_USAGE_SAMPLED_BIT (the renderer transitions
//            DEPTH_ATTACHMENT → SHADER_READ_ONLY between the two passes).
//   D3D12  : ID3D12Resource (D32_FLOAT) with a DSV and an SRV; the
//            resource-state barrier does the DEPTH_WRITE → PIXEL_
//            SHADER_RESOURCE transition between passes.
// ---------------------------------------------------------------------------
enum class TextureUsage : u32 {
    DepthRenderTarget = 1u << 0,   // can be the depth attachment of a pass
    Sampled           = 1u << 1,   // can be read in a shader (SRV / sampled)
    ColorRenderTarget = 1u << 2,   // future: off-screen colour (post-process)
};

struct TextureDesc {
    u32    width{0};
    u32    height{0};
    Format format{Format::D32_Float};
    u32    usage{0};               // bitmask of TextureUsage
};

class Texture {
public:
    virtual ~Texture() = default;
    Texture(const Texture&)            = delete;
    Texture& operator=(const Texture&) = delete;

    virtual u32    width()  const noexcept = 0;
    virtual u32    height() const noexcept = 0;
    virtual Format format() const noexcept = 0;

protected:
    Texture() = default;
};

// ---------------------------------------------------------------------------
// Acceleration Structures (Phase 5 — RT primitives)
//
// BLAS  = bottom-level: per-mesh, references vertex+index buffers
// TLAS  = top-level:   per-scene, references BLAS instances with transforms
//
// Both opaque to the engine; the RHI builds them, RTXMU (when wired in
// Phase 5.1) will manage suballocation + compaction for high BLAS counts.
// ---------------------------------------------------------------------------
struct BlasGeometryDesc {
    Buffer* vertex_buffer{nullptr};
    Format  vertex_format{Format::R32G32B32_Float};
    u32     vertex_stride{0};
    u32     vertex_count{0};
    u32     vertex_offset{0};   // bytes into vertex_buffer

    Buffer* index_buffer{nullptr};   // optional; nullptr => use sequential vertices
    bool    indices_are_u32{true};   // false => u16 indices
    u32     index_count{0};
    u32     index_offset{0};         // bytes into index_buffer

    bool opaque{true};
};

struct BlasDesc {
    cardinal::vector<BlasGeometryDesc> geometries;
    bool prefer_fast_trace{true};   // false => prefer fast build
    bool allow_compaction{false};   // hand off to RTXMU when true (Phase 5.1)
};

struct TlasInstance {
    class AccelerationStructure* blas{nullptr};
    float    transform[12]{1,0,0,0, 0,1,0,0, 0,0,1,0};  // 3x4 row-major (identity)
    u32      instance_id{0};
    u32      hit_group_index{0};
    u8       mask{0xFF};
};

struct TlasDesc {
    cardinal::vector<TlasInstance> instances;
    bool prefer_fast_trace{true};
};

class AccelerationStructure {
public:
    virtual ~AccelerationStructure() = default;

    AccelerationStructure(const AccelerationStructure&)            = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;

    // GPU virtual address of the AS — what shaders / TLAS-builds consume.
    virtual u64 device_address() const noexcept = 0;

protected:
    AccelerationStructure() = default;
};

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------
enum class PrimitiveTopology : u32 {
    TriangleList,
    TriangleStrip,
    LineList,
    LineStrip,
    PointList,
};

struct PipelineDesc {
    ShaderBlob               vertex_shader;
    ShaderBlob               fragment_shader;
    const char*              vertex_entry{"VSMain"};
    const char*              fragment_entry{"PSMain"};
    cardinal::vector<VertexAttrib> vertex_attribs;
    u32                      vertex_stride{0};       // 0 if no vertex buffer (uses SV_VertexID)
    PrimitiveTopology        topology{PrimitiveTopology::TriangleList};
    Format                   color_format{Format::B8G8R8A8_UNORM};

    // Depth-test opt-in. When depth_format != Unknown the swapchain MUST
    // currently have a depth attachment of the same format; the pipeline
    // honours depth_test (sampler) + depth_write (RT).
    Format                   depth_format{Format::Unknown};
    bool                     depth_test{false};
    bool                     depth_write{false};

    // Push-constant block size in BYTES, 0 to disable. The block is
    // visible to BOTH vertex + fragment stages (Vulkan VK_SHADER_STAGE_
    // ALL_GRAPHICS-style; D3D12 root constants visible to both stages
    // by default). Cap at 128 bytes for portability — Vulkan guarantees
    // that minimum, D3D12 root signatures fit far more but the engine's
    // common shape is one Mat4 (64 B) + a small per-draw scalar block.
    //
    // The shader declares the matching layout:
    //   Vulkan / SPIR-V:  layout(push_constant) uniform Block { ... } pc;
    //   HLSL via DXC:     [[vk::push_constant]] cbuffer Block { ... };
    //   HLSL native D3D:  cbuffer Block : register(b0) { ... };  (root constants)
    // Caller must keep the host struct in sync with the shader layout.
    u32                      push_constant_size{0};

    // ---- Storage-buffer bindings (read-only SSBO / structured SRV) ------
    //
    // Number of read-only storage buffers this pipeline binds, slots
    // [0 .. storage_buffer_slots). 0 (default) keeps the legacy
    // push-constant-only layout — existing pipelines are unaffected.
    //
    // Why this exists: the push-constant block is hard-capped (Vulkan's
    // portable minimum is 128 B; the engine's scene shader is already at
    // the 256 B every-modern-desktop ceiling). Lighting + per-material
    // arrays don't fit and can't grow. A bound storage buffer lifts that
    // cap entirely — the shader reads an arbitrarily large light/material
    // array behind a single binding while the push block keeps only the
    // per-draw essentials (MVP, model, a few scalars).
    //
    // Backend mapping (each slot s in [0, storage_buffer_slots)):
    //   Vulkan : descriptor-set 0, binding s, VK_DESCRIPTOR_TYPE_
    //            STORAGE_BUFFER, visible to all graphics stages. Bound
    //            per-frame via Swapchain::bind_storage_buffer(s, buf).
    //   D3D12  : a root SRV parameter (D3D12_ROOT_PARAMETER_TYPE_SRV)
    //            per slot, set via SetGraphicsRootShaderResourceView with
    //            the buffer's GPU virtual address. No descriptor heap.
    //
    // Shader declaration (DXC HLSL, portable to both):
    //   struct Light { float4 a; float4 b; ... };
    //   [[vk::binding(0,0)]] StructuredBuffer<Light> g_lights : register(t0);
    //
    // The buffer must be created with BufferUsage::Storage (and, on the
    // Vulkan side, ShaderDeviceAddress is NOT required for the descriptor
    // path). Caller keeps the element struct in sync with the shader.
    u32                      storage_buffer_slots{0};

    // ---- Sampled-texture bindings (shadow map / future RTs) ------------
    //
    // Number of sampled 2D textures this pipeline reads, slots
    // [0 .. sampled_texture_slots). 0 (default) leaves the layout
    // unchanged. The shadow-mapping arc's first consumer binds the
    // depth shadow map at slot 0.
    //
    // Binding convention (descriptor set 0): storage buffers occupy
    // bindings [0, storage_buffer_slots); sampled textures follow at
    // bindings [storage_buffer_slots + slot]. Each is a single
    // combined-image-sampler (the backend pairs the texture's view with
    // a device default sampler — clamp, linear).
    //   Vulkan : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER.
    //   D3D12  : SRV + static sampler in the root signature.
    // Shader (DXC HLSL):
    //   [[vk::binding(<storage_slots>,0)]] Texture2D shadow : register(t<n>);
    //   SamplerState shadow_smp : register(s0);
    u32                      sampled_texture_slots{0};
};

// ---------------------------------------------------------------------------
// ComputePipelineDesc — compute-shader pipeline (AEGIS graph::RhiBackend
// surface). Mirrors the storage_buffer / push_constant story of
// PipelineDesc above, minus the rasterizer state. uav_slots count
// read-write storage buffers; storage_buffer_slots count read-only ones.
// ---------------------------------------------------------------------------
struct ComputePipelineDesc {
    ShaderBlob   compute_shader;
    const char*  compute_entry{"CSMain"};
    u32          push_constant_size{0};
    u32          storage_buffer_slots{0};   // read-only SSBOs (bind_storage_buffer)
    u32          uav_slots{0};              // read-write UAVs (bind_storage_buffer_uav)
};

class Pipeline {
public:
    virtual ~Pipeline() = default;

    Pipeline(const Pipeline&)            = delete;
    Pipeline& operator=(const Pipeline&) = delete;

protected:
    Pipeline() = default;
};

// ---------------------------------------------------------------------------
// Swapchain — per-frame command recording.
// ---------------------------------------------------------------------------
class Swapchain {
public:
    virtual ~Swapchain() = default;

    Swapchain(const Swapchain&)            = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    virtual u32    width() const noexcept = 0;     // present resolution
    virtual u32    height() const noexcept = 0;
    virtual Format color_format() const noexcept = 0;

    // Depth attachment paired with the back buffer + the off-screen viewport
    // image. Returns Format::Unknown when depth is disabled (legacy path —
    // Vulkan/D3D12 implementations always provision one today).
    virtual Format depth_format() const noexcept { return Format::D32_Float; }

    // VSync interval. 0 = uncapped (DXGI tearing path on D3D12), 1 = sync
    // to display refresh, 2 = half-rate, 3 = third-rate, … Anything > 0 is
    // a divisor on the display refresh: at 60 Hz, interval 2 → 30 FPS, 3 → 20.
    // The legacy bool API maps to interval = 1/0 for back-compat.
    virtual void   set_vsync(bool on) = 0;
    virtual bool   vsync() const noexcept = 0;
    virtual void   set_vsync_interval(u32 interval) = 0;
    virtual u32    vsync_interval() const noexcept = 0;

    // ----- NVIDIA Reflex (D3D12 path uses NvAPI; Vulkan path is wired
    //       separately in rhi_vulkan_modern). The swapchain owns the
    //       D3D-device handle Reflex needs, so the API lives here.
    //
    //   set_reflex_mode(off | on | on_plus_boost) — enable / disable
    //   set_reflex_fps_cap(fps) — built-in framerate limiter that runs at
    //                             the optimal point in the frame for
    //                             latency. 0 = no cap.
    //   reflex_supported() — true when the SDK is initialised AND running
    //                        on supported hardware (NVIDIA + Win10+).
    //   reflex_sleep() — host calls this at the very top of each frame.
    //   reflex_marker(stage, frame_id) — host emits per-stage markers.
    enum class ReflexMode : u32 { Off = 0, On = 1, OnPlusBoost = 2 };
    enum class ReflexMarker : u32 {
        SimulationStart = 0,
        SimulationEnd,
        RenderSubmitStart,
        RenderSubmitEnd,
        PresentStart,
        PresentEnd,
        InputSample,
    };
    virtual bool reflex_supported() const noexcept { return false; }
    virtual void set_reflex_mode(ReflexMode m)     { (void)m; }
    virtual ReflexMode reflex_mode() const noexcept { return ReflexMode::Off; }
    virtual void set_reflex_fps_cap(u32 fps)       { (void)fps; }
    virtual u32  reflex_fps_cap() const noexcept   { return 0u; }
    virtual void reflex_sleep()                    {}
    virtual void reflex_marker(ReflexMarker, u64 /*frame_id*/) {}

    // Latest Reflex latency report (averaged over the last few frames).
    // Zero-filled when Reflex is disabled or no frames have completed.
    struct ReflexLatency {
        bool   valid{false};
        float  total_latency_ms{0.0f};       // input-to-photon
        float  sim_ms{0.0f};
        float  render_ms{0.0f};
        float  present_ms{0.0f};
        float  driver_ms{0.0f};
        float  os_render_queue_ms{0.0f};
        float  gpu_ms{0.0f};
    };
    virtual ReflexLatency reflex_latency() const { return {}; }

    // Recreate the back buffers at a new size. Idempotent when called with
    // the current size. Idles the GPU before destroying the old back
    // buffers so dependent state (RTV heaps, command lists referencing
    // them, etc.) can be safely rebuilt right after.
    //
    // Returns true on success. On failure (e.g. zero-size, allocation
    // error) the swapchain is left in its previous state and the caller
    // should skip the next frame.
    virtual bool   resize(u32 new_width, u32 new_height) = 0;

    // ---- Rebuild notification --------------------------------------------
    //
    // Fires AFTER the swapchain images / views have been recreated for ANY
    // reason: external resize(), internal vsync-mode change, future
    // backend hot-swap. Consumers that hold image-view references
    // (Studio's per-image VkFramebuffers, custom render passes) MUST
    // listen here and rebuild — otherwise the next frame's overlay
    // recording dereferences freed views and crashes.
    //
    // Single-callback (last writer wins). The engine sets it; UI hosts
    // chain through the engine's set_on_swapchain_resized hook.
    using OnRebuilt = cardinal::function<void()>;
    virtual void set_on_rebuilt(OnRebuilt cb) = 0;

    // Viewport (render-to-texture) configuration.
    //
    // Multi-viewport model
    // --------------------
    // The swapchain owns N off-screen viewport render targets — one per
    // editor viewport panel. The host calls `set_viewport_count(n)` to grow
    // / shrink the pool, then sizes each one independently with
    // `set_viewport_size(id, w, h)`. Each viewport has its own colour image,
    // RTV/DSV heap entry, depth buffer, and ImGui-visible SRV.
    //
    // The render path uses a tiny state machine: the host calls
    // `set_active_viewport(id)` once before each `pipeline.render()`. The
    // pipeline writes into viewport[id]'s color + depth attachments. Repeat
    // for every visible viewport between `begin_frame` and `end_frame`.
    //
    // Each panel shows its own RTT, so a Wireframe viewport stays Wireframe
    // even when the user flips a different viewport to Polygons. The shared
    // scene::Camera means all viewports view the same world from the same
    // angle — only the visualisation differs.
    //
    // Legacy single-viewport sugar
    // ----------------------------
    //   set_viewport_size(0,0)  — disable viewport 0 (the legacy RTT);
    //                             begin_frame renders directly into the
    //                             back-buffer (the pre-multi-viewport path).
    //   set_viewport_size(w,h)  — equivalent to set_viewport_size(0,w,h).
    //   viewport_width()        — viewport_width(0).
    //
    // Backend notes
    // -------------
    //   D3D12  — full multi-viewport: per-id images / depth / RTV / DSV.
    //   Vulkan — degraded multi-viewport: viewport_count ≥ 1 is accepted but
    //            only viewport 0 is real; non-zero ids alias to viewport 0
    //            and SRVs return null. Full Vulkan vectorisation is a
    //            follow-up.
    virtual void set_viewport_size(u32 width, u32 height) = 0;
    virtual u32  viewport_width()  const noexcept = 0;   // 0 if disabled
    virtual u32  viewport_height() const noexcept = 0;
    virtual void set_viewport_size(u32 id, u32 width, u32 height) = 0;
    virtual u32  viewport_width (u32 id) const noexcept = 0;
    virtual u32  viewport_height(u32 id) const noexcept = 0;
    // Resize the per-viewport vector. Growing zero-initialises new slots
    // (callers must follow up with set_viewport_size before they're useful);
    // shrinking releases the trailing slots' GPU resources.
    virtual void set_viewport_count(u32 n) = 0;
    virtual u32  viewport_count() const noexcept = 0;
    // Pick which viewport the next render writes into. The pipeline reads
    // viewport_image_[active_id_] in its render call. Default is 0; you must
    // re-set this before each render between begin_frame and end_frame.
    virtual void set_active_viewport(u32 id) = 0;
    virtual u32  active_viewport() const noexcept = 0;

    // Acquire next image, transition to color attachment, begin dynamic
    // rendering with the given clear color. Returns the acquired image
    // index.
    virtual u32  begin_frame(float r, float g, float b, float a) = 0;
    // End rendering, transition to PRESENT, submit, present.
    virtual void end_frame() = 0;

    // Overlay callback — invoked inside end_frame AFTER scene rendering ends
    // (and after the viewport blit, if a viewport is active), but BEFORE the
    // swapchain image transitions to PRESENT. The callback runs with the
    // swapchain image in COLOR_ATTACHMENT_OPTIMAL and the current command
    // buffer recording. Use this to draw editor UI on top of the rendered
    // scene. Pass nullptr to clear.
    using OverlayCallback = void (*)(void* user_data);
    virtual void set_overlay(OverlayCallback cb, void* user_data) = 0;

    // Recording commands — call between begin_frame() and end_frame().
    virtual void bind_pipeline(Pipeline* p) = 0;
    virtual void bind_vertex_buffer(Buffer* b, usize offset = 0) = 0;
    virtual void draw(u32 vertex_count, u32 instance_count = 1,
                      u32 first_vertex = 0, u32 first_instance = 0) = 0;

    // Bind a read-only storage buffer to shader slot `slot` for the
    // currently-bound pipeline. `slot` must be < the bound pipeline's
    // PipelineDesc::storage_buffer_slots. Recorded into the active
    // command buffer — call BETWEEN bind_pipeline() and draw().
    //
    // Default no-op: backends that haven't implemented the storage-
    // buffer path yet (or pipelines that declared storage_buffer_slots
    // == 0) silently ignore the call, so existing push-constant-only
    // render paths are unaffected. Vulkan/D3D12 overrides land in
    // follow-on iterations (descriptor set / root SRV respectively).
    virtual void bind_storage_buffer(u32 /*slot*/, Buffer* /*b*/) {}

    // Update the currently-bound pipeline's push-constant block. `offset`
    // and `size` are in BYTES. Both must respect the pipeline's
    // declared push_constant_size; out-of-range writes are dropped (and
    // logged once). The update is recorded into the active command
    // buffer — must be called BETWEEN bind_pipeline() and draw().
    //
    // No-op when the bound pipeline declared push_constant_size == 0;
    // useful for callers that want to share a draw helper across
    // pipelines that may or may not use a push block.
    virtual void set_push_constants(u32 offset, const void* data, u32 size) = 0;

    // ---- Compute dispatch (AEGIS graph::RhiBackend surface) -------------
    //
    // Compute pipelines created via Device::create_compute_pipeline are
    // bound via the same bind_pipeline() entry point as graphics
    // pipelines, then receive dispatches via the methods below. Bound
    // storage buffers split into read-only (bind_storage_buffer, above)
    // and read-write UAV (bind_storage_buffer_uav).
    //
    // The default implementations are no-ops so existing Vulkan / D3D12
    // backends continue to compile + run unchanged while compute support
    // is incrementally wired. graph::RhiBackend calls these unconditionally
    // — when the backend implements them, dispatches start firing on the
    // real device with no graph::RhiBackend changes.
    virtual void bind_storage_buffer_uav(u32 /*slot*/, Buffer* /*b*/) {}
    virtual void dispatch(u32 /*gx*/, u32 /*gy*/ = 1, u32 /*gz*/ = 1) {}

    // ---- Shadow / depth-only pass (shadow-mapping arc) ------------------
    //
    // Bracket a depth-only render INTO `depth` (a Texture created with
    // TextureUsage::DepthRenderTarget). Between begin/end the caller
    // binds a depth-only pipeline + draws the scene from the light's
    // POV; end_shadow_pass transitions `depth` to a shader-readable
    // state so the main pass can sample it via bind_storage_buffer-style
    // resource binding. Default no-ops so the current single-pass path
    // is unaffected until the backends implement it (same safe-
    // incremental rollout as the storage-buffer primitive).
    virtual void begin_shadow_pass(Texture* /*depth*/) {}
    virtual void end_shadow_pass() {}
    // Bind a sampled texture (e.g. the shadow map) for the bound
    // pipeline at `slot`. Distinct slot space from storage buffers.
    // Default no-op until backends implement the sampled-image path.
    virtual void bind_sampled_texture(u32 /*slot*/, Texture* /*tex*/) {}

    // =====================================================================
    // AEGIS Render Pipeline 2.0 surface — every entry below is declared
    // with a default no-op implementation so existing Vulkan / D3D12
    // backends continue to compile + run unchanged. Real implementations
    // land per-feature as they're validated against real devices.
    //
    // graph::RhiBackend uses these declarations to dispatch the AEGIS
    // graph's per-Pass work onto the device. When a method's default
    // no-op is overridden by a backend, that feature comes online with
    // no AEGIS-side or call-site changes.
    // =====================================================================

    // ---- Indirect dispatch / drawing (AEGIS Block 4 → DrawIndirectGen) --
    //
    // GPU-driven culling produces a compacted command buffer (an array of
    // IndirectDrawCmd or IndirectDispatchCmd structs declared by the
    // host) that the GPU consumes via these entries. No CPU roundtrip
    // for the visibility → draw pipeline.
    //
    // Buffer layout matches the backend's expectation:
    //   D3D12  : D3D12_DRAW_INDEXED_ARGUMENTS / D3D12_DISPATCH_ARGUMENTS
    //            tightly packed at `args_offset`
    //   Vulkan : VkDrawIndirectCommand / VkDispatchIndirectCommand
    virtual void draw_indirect(Buffer* /*args*/, u32 /*args_offset*/,
                               u32 /*draw_count*/, u32 /*stride*/) {}
    virtual void draw_indexed_indirect(Buffer* /*args*/, u32 /*args_offset*/,
                                        u32 /*draw_count*/, u32 /*stride*/) {}
    virtual void dispatch_indirect(Buffer* /*args*/, u32 /*args_offset*/) {}

    // ---- Mesh / Task shader dispatch (AEGIS Block 5) --------------------
    //
    // Modern geometry pipeline — replaces vertex+hull+domain+geom with a
    // mesh shader that emits the meshlet directly. dispatch_mesh kicks off
    // a workgroup grid; the mesh shader emits the triangles itself.
    // Backends that don't have GpuCapabilities::mesh_shader set ignore
    // the call (no-op).
    virtual void dispatch_mesh(u32 /*gx*/, u32 /*gy*/ = 1, u32 /*gz*/ = 1) {}
    virtual void dispatch_mesh_indirect(Buffer* /*args*/, u32 /*args_offset*/) {}

    // ---- Ray tracing dispatch (AEGIS Block 7 → PathTracePass) -----------
    //
    // Inline ray-tracing (ray_query) calls live inside compute/mesh
    // shaders; the bracket version below is for ray-tracing PIPELINES
    // (raygen + miss + closesthit shader binding tables). Backends
    // without GpuCapabilities::ray_tracing_pipeline ignore the call.
    virtual void trace_rays(u32 /*width*/, u32 /*height*/, u32 /*depth*/ = 1) {}
    virtual void trace_rays_indirect(Buffer* /*args*/, u32 /*args_offset*/) {}

    // ---- Variable Rate Shading (AEGIS Block 12 — perf knob) -------------
    //
    // VRS lets the rasterizer shade fewer fragments in low-importance
    // screen regions (motion blur, periphery, etc.). Two paths:
    //   * Per-draw "all pixels at rate X" via set_shading_rate
    //   * Per-tile via bind_shading_rate_image (small u8 texture mapping
    //     screen tiles to VRS rates)
    // Default no-ops when GpuCapabilities::variable_rate_shading is false.
    enum class ShadingRate : u32 {
        Rate_1x1 = 0,  // full shading (default)
        Rate_1x2,
        Rate_2x1,
        Rate_2x2,
        Rate_2x4,
        Rate_4x2,
        Rate_4x4,
    };
    virtual void set_shading_rate(ShadingRate /*rate*/) {}
    virtual void bind_shading_rate_image(Texture* /*per_tile_rate_image*/) {}

    // ---- Bindless descriptor heap (AEGIS Block 8 → Bindless Materials) --
    //
    // Binds a single large descriptor heap that contains 1000s of
    // textures / buffers indexable by the shader via NonUniformResourceIndex
    // (DX12) / VK_EXT_descriptor_indexing (Vulkan). One bind covers
    // every shader for the rest of the frame.
    virtual void bind_bindless_heap(Buffer* /*heap*/) {}

    // ---- Resource barriers (graph::RhiBackend explicit barriers) --------
    //
    // The AEGIS graph framework infers RAW/WAW/WAR between passes; the
    // RhiBackend uses these primitives to emit the matching pipeline
    // barriers / resource transitions on the real device.
    enum class ResourceState : u32 {
        Common               = 0,
        VertexBuffer,
        IndexBuffer,
        ConstantBuffer,
        ShaderResource,      // SRV / sampled storage
        UnorderedAccess,     // UAV / storage with writes
        IndirectArgument,
        CopySource,
        CopyDest,
        Present,
        DepthRead,
        DepthWrite,
        RenderTarget,
        ResolveSource,
        ResolveDest,
    };
    virtual void transition_buffer_state(Buffer* /*b*/,
                                          ResourceState /*before*/,
                                          ResourceState /*after*/) {}
    virtual void transition_texture_state(Texture* /*t*/,
                                          ResourceState /*before*/,
                                          ResourceState /*after*/) {}
    // Single-call equivalent for "I just wrote to this UAV, now I want
    // to read it" — emits a UAV-flush barrier without a state change.
    virtual void uav_barrier(Buffer* /*b*/) {}
    virtual void uav_barrier_texture(Texture* /*t*/) {}

    // ---- Explicit copies (graph::RhiBackend Copy passes) ----------------
    virtual void copy_buffer(Buffer* /*src*/, usize /*src_offset*/,
                              Buffer* /*dst*/, usize /*dst_offset*/,
                              usize /*size*/) {}
    virtual void copy_texture(Texture* /*src*/, Texture* /*dst*/) {}

    // ---- GPU timestamp queries (Block 13 → frame-pacing telemetry) ------
    //
    // Issue a timestamp into a query heap. The host reads the heap N
    // frames later (after GPU completion) for per-stage µs cost.
    virtual void write_timestamp(u32 /*query_heap_id*/, u32 /*index*/) {}

    // ---- Debug markers / events (PIX, RenderDoc, Nsight) ----------------
    //
    // begin_event / end_event nest; insert_marker is a one-shot label.
    // Default no-op — backends emit PIX events (D3D12 PIXBeginEvent /
    // EndEvent) or Vulkan debug markers (vkCmdBeginDebugUtilsLabelEXT).
    virtual void begin_event(const char* /*name*/) {}
    virtual void end_event() {}
    virtual void insert_marker(const char* /*name*/) {}

    // ---- Predication / conditional rendering (Block 4 — late occlusion) -
    //
    // Predicate subsequent draws on a counter in a Buffer (e.g. visible-
    // primitive count from Hi-Z occlusion). Drawn iff *(args_offset:u32)
    // != 0 unless invert == true. D3D12 SetPredication / Vulkan
    // VK_EXT_conditional_rendering.
    virtual void set_predication(Buffer* /*args*/, u32 /*args_offset*/,
                                  bool /*invert*/ = false) {}
    virtual void clear_predication() {}

    // ---- Multi-queue submission (AEGIS Block 10 → Async Compute) --------
    //
    // The host obtains an async compute queue via Device::async_compute_
    // queue() (when GpuCapabilities::async_compute is true) and submits
    // independent work that overlaps with the graphics queue. These are
    // sync primitives the host uses to fence between queues.
    virtual u64  signal_fence(class Fence* /*f*/) { return 0; }
    virtual void wait_fence  (class Fence* /*f*/, u64 /*value*/) {}

protected:
    Swapchain() = default;
};

// ---------------------------------------------------------------------------
// Fence — explicit GPU sync object for multi-queue submission (AEGIS
// async compute story). Created by Device::create_fence; signalled +
// waited via Swapchain::signal_fence / wait_fence on either the
// graphics queue or the async compute queue.
// ---------------------------------------------------------------------------
class Fence {
public:
    virtual ~Fence() = default;
    Fence(const Fence&)            = delete;
    Fence& operator=(const Fence&) = delete;

    // Host-side wait for the fence to reach `value`. Returns the
    // achieved value (>= `value`) or 0 on timeout.
    virtual u64 wait_cpu(u64 /*value*/, u64 /*timeout_ns*/ = 1'000'000'000ull) { return 0; }
    virtual u64 current_value() const noexcept { return 0; }

protected:
    Fence() = default;
};

// ---------------------------------------------------------------------------
// QueryHeap — GPU timestamp / occlusion query container. Created via
// Device::create_query_heap; entries written by Swapchain::write_timestamp.
// ---------------------------------------------------------------------------
enum class QueryKind : u32 { Timestamp = 0, Occlusion, PipelineStatistics };

class QueryHeap {
public:
    virtual ~QueryHeap() = default;
    QueryHeap(const QueryHeap&)            = delete;
    QueryHeap& operator=(const QueryHeap&) = delete;

    // Read back N consecutive query results into `out` (host pointer).
    // For timestamps the unit is ticks; the host divides by
    // GpuCapabilities::timestamp_ticks_per_second.
    virtual void readback(u32 /*first*/, u32 /*count*/, u64* /*out*/) {}

protected:
    QueryHeap() = default;
};

// ---------------------------------------------------------------------------
// ComputeQueue — async compute submission lane (AEGIS Block 10).
// Owns its own command list pool; the host records work via the same
// Swapchain-like recording interface and submits it independently of
// the graphics queue.
// ---------------------------------------------------------------------------
class ComputeQueue {
public:
    virtual ~ComputeQueue() = default;
    ComputeQueue(const ComputeQueue&)            = delete;
    ComputeQueue& operator=(const ComputeQueue&) = delete;

    // Record + submit one self-contained compute submission. The
    // backend's implementation acquires a command list, calls
    // `record(cmd)`, closes it, submits to the async queue, and
    // signals `signal_fence` to `signal_value` on completion.
    using RecordFn = void(*)(Swapchain* cmd_recorder, void* user) noexcept;
    virtual u64 submit(RecordFn /*record*/, void* /*user*/,
                       Fence* /*signal_fence*/ = nullptr,
                       u64    /*signal_value*/ = 0) { return 0; }

protected:
    ComputeQueue() = default;
};

// ---------------------------------------------------------------------------
// Capabilities — IMMUTABLE: what the GPU + driver actually support. Queried
// at device creation. Apps consult these to decide whether to expose a
// runtime setting.
// ---------------------------------------------------------------------------
struct GpuCapabilities {
    // Core modern Vulkan / DX12 (most are required for the engine to run)
    bool dynamic_rendering{false};
    bool synchronization2{false};
    bool buffer_device_address{false};
    bool descriptor_indexing{false};
    bool runtime_descriptor_array{false};
    bool scalar_block_layout{false};
    bool timeline_semaphore{false};

    // Shader compute power
    bool shader_float16{false};       // FP16 / RPM math
    bool shader_int8{false};          // INT8 / RPM math
    bool shader_int16{false};
    bool storage_buffer_16bit{false}; // FP16 in storage buffers
    bool wave_intrinsics{false};      // subgroup ops

    // Mesh / task shaders (modern geometry pipeline)
    bool mesh_shader{false};
    bool task_shader{false};

    // Ray tracing (BLAS/TLAS, ray pipeline, ray query for inline RT)
    bool acceleration_structure{false};
    bool ray_tracing_pipeline{false};
    bool ray_query{false};

    // Vendor-specific paths (presence implies the CPU-side SDK can be used,
    // not that it's currently active — see RenderSettings to enable).
    bool nvidia_dlss_capable{false};      // RTX 20-series and newer (Tensor cores)
    bool nvidia_framegen_capable{false};  // RTX 40-series (Optical Flow Accelerator)
    bool amd_fsr3_capable{false};         // RX 6000+ for FSR 3 frame interpolation

    // ---- AEGIS surface caps (graph::RhiBackend dispatch-target gates) ---
    bool variable_rate_shading{false};    // VRS Tier 1+ (Block 12 perf)
    bool variable_rate_image  {false};    // VRS Tier 2 (per-tile rates image)
    bool async_compute        {false};    // separate compute queue (Block 10)
    bool indirect_dispatch    {false};    // GPU-driven dispatch (Block 4)
    bool indirect_draw_count  {false};    // ExecuteIndirect with count buffer (Block 4)
    bool predication          {false};    // SetPredication / conditional rendering
    bool bindless_resources   {false};    // 1M+ descriptors via heap indexing (Block 8)
    bool sparse_tiled_resources{false};   // tiled / sparse for virtual texturing (Block 11)
    bool sampler_feedback     {false};    // texture-usage feedback for streaming (Block 11)
    bool multi_gpu_explicit   {false};    // explicit multi-adapter (NVLink / Crossfire)

    // ---- Compute / RT power knobs ---------------------------------------
    bool fp8_math             {false};    // Hopper / Ada / RDNA4 (math-division tier)
    bool fp4_math             {false};    // Blackwell / RDNA5 (math-division tier)
    bool tensor_cores         {false};    // NVIDIA Tensor cores (DLSS / ML)
    bool optical_flow_accel   {false};    // DLSS 3 Frame Gen accelerator
    bool dgc_compute          {false};    // device-generated commands for compute
    bool ray_tracing_invocation_reorder{false};  // SER (RTX 40-series)
    bool opacity_micromap     {false};    // RT alpha-test acceleration

    // ---- Streaming / IO (Block 11) --------------------------------------
    bool direct_storage_capable{false};   // DirectStorage / RTX IO present
    bool gdeflate_decompress  {false};    // GPU GDEFLATE decode (RTX 40+ HW path)

    // ---- Timestamp / profiling ------------------------------------------
    u64  timestamp_ticks_per_second{0};   // divide timestamp delta by this for seconds
    u32  max_async_compute_queues  {0};   // typically 1 on D3D12, 1-2 on Vulkan
    u32  max_copy_queues           {0};

    // Identity
    char vendor_name[32]{};      // "NVIDIA Corporation", "AMD", "Intel"
    char gpu_arch[32]{};         // best-effort: "Ada Lovelace", "RDNA 3", etc.
    u64  vram_bytes{0};
};

// ---------------------------------------------------------------------------
// RenderSettings — MUTABLE at runtime. Apps tweak these from a settings UI;
// engine subsystems consult them each frame. Toggling some (e.g. ray
// tracing on/off) may require pipeline recreation — drivers typically allow
// this within a frame boundary. The engine guarantees a clean handoff at
// the next begin_frame() after apply_settings() is called.
// ---------------------------------------------------------------------------
enum class UpscalerMode : u32 {
    Off,
    DLSS_Quality,
    DLSS_Balanced,
    DLSS_Performance,
    DLSS_UltraPerformance,
    DLAA,                  // native-resolution antialiasing via NGX
    FSR3_Quality,
    FSR3_Balanced,
    FSR3_Performance,
    TAA,                   // engine-internal TAA fallback
};

struct RenderSettings {
    // Ray tracing
    bool enable_ray_tracing{false};      // turn the RT pipeline on/off
    bool enable_ray_query{false};        // inline RT in raster shaders
    u32  ray_tracing_max_bounces{4};

    // Path tracing (built on RT — requires enable_ray_tracing)
    bool enable_path_tracing{false};
    u32  path_tracing_samples_per_pixel{1};

    // Upscaling / AA
    UpscalerMode upscaler{UpscalerMode::Off};

    // Frame interpolation (DLSS 3 / FSR 3) — independent of upscaler
    bool enable_frame_generation{false};

    // Modern geometry path
    bool enable_mesh_shaders{true};      // when off, fall back to vertex pipe

    // Math precision toggles
    bool prefer_fp16{true};              // shaders pick min16 paths
    bool prefer_int8{false};             // for inference passes only

    // Internal render scale (applied BEFORE upscaler if any)
    float render_scale{1.0f};
};

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------
struct DeviceDesc {
    Backend backend{Backend::Auto};
    bool    enable_validation{false};
    bool    prefer_discrete_gpu{true};
    // Initial settings; can be changed at runtime via Device::apply_settings.
    RenderSettings initial_settings{};
};

class Device {
public:
    virtual ~Device() = default;

    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;

    virtual Backend     backend() const noexcept     = 0;
    virtual const char* adapter_name() const noexcept = 0;

    // Capability query — never changes after device creation.
    virtual const GpuCapabilities& capabilities() const noexcept = 0;

    // Settings — runtime mutable. apply_settings() validates against
    // capabilities and silently clamps unsupported requests (logged).
    virtual const RenderSettings& settings() const noexcept = 0;
    virtual void                  apply_settings(const RenderSettings& s) = 0;

    virtual cardinal::unique_ptr<Swapchain> create_swapchain(
        void* native_window, u32 width, u32 height) = 0;

    virtual cardinal::unique_ptr<Buffer>   create_buffer(const BufferDesc& desc)     = 0;
    virtual cardinal::unique_ptr<Pipeline> create_pipeline(const PipelineDesc& desc) = 0;

    // Compute-pipeline creation (graph::RhiBackend surface). Backends
    // that haven't wired vkCreateComputePipelines / ID3D12Device::
    // CreateComputePipelineState yet return nullptr — graph::RhiBackend
    // detects the null and falls through to its NullBackend-equivalent
    // recording path. When backends implement this, AEGIS graph passes
    // dispatch on the real device with no Pipeline / runner changes.
    virtual cardinal::unique_ptr<Pipeline> create_compute_pipeline(
        const ComputePipelineDesc& /*desc*/) { return {}; }

    // Off-screen texture (shadow-mapping arc). Default returns nullptr
    // so backends that haven't implemented it yet — and the existing
    // render path — are unaffected (same safe-incremental pattern as
    // bind_storage_buffer). Vulkan/D3D12 overrides land next.
    virtual cardinal::unique_ptr<Texture> create_texture(const TextureDesc& /*d*/) {
        return nullptr;
    }

    // RT primitives — return nullptr if caps.acceleration_structure is false.
    // Build is synchronous on the graphics queue (vkQueueWaitIdle internally).
    // Async build + RTXMU pooling lands in Phase 5.1.
    virtual cardinal::unique_ptr<AccelerationStructure> create_blas(const BlasDesc& desc) = 0;
    virtual cardinal::unique_ptr<AccelerationStructure> create_tlas(const TlasDesc& desc) = 0;

    virtual ShaderBlob compile_shader(
        ShaderStage stage,
        const char* hlsl_source,
        const char* entry_point) = 0;

    // ====================================================================
    // AEGIS surface — factory methods for the rest of the pipeline. Every
    // factory defaults to nullptr so backends compile + run unchanged
    // while individual features are wired. graph::RhiBackend gates each
    // dispatch path on a non-null result.
    // ====================================================================

    // Fence — GPU sync object for multi-queue submission (AEGIS Block 10).
    virtual cardinal::unique_ptr<Fence> create_fence(u64 /*initial_value*/ = 0) {
        return {};
    }

    // QueryHeap — timestamp / occlusion / pipeline-statistics queries.
    virtual cardinal::unique_ptr<QueryHeap> create_query_heap(
        QueryKind /*kind*/, u32 /*count*/) { return {}; }

    // Async compute queue (AEGIS Block 10 → Async Compute Scheduler).
    // Returns nullptr if caps.async_compute is false. The host owns the
    // returned queue for the lifetime of the device; multiple queues
    // can be created up to caps.max_async_compute_queues.
    virtual cardinal::unique_ptr<ComputeQueue> create_async_compute_queue() {
        return {};
    }

    // ---- Mesh-shader pipeline (AEGIS Block 5) --------------------------
    //
    // Distinct PipelineDesc shape: mesh shader (no vertex/IA) +
    // optional task shader + fragment shader. Backends without
    // GpuCapabilities::mesh_shader return nullptr.
    struct MeshPipelineDesc {
        ShaderBlob   task_shader;     // optional; empty disables task stage
        ShaderBlob   mesh_shader;
        ShaderBlob   fragment_shader;
        const char*  task_entry    {"TSMain"};
        const char*  mesh_entry    {"MSMain"};
        const char*  fragment_entry{"PSMain"};
        Format       color_format  {Format::B8G8R8A8_UNORM};
        Format       depth_format  {Format::Unknown};
        bool         depth_test    {false};
        bool         depth_write   {false};
        u32          push_constant_size   {0};
        u32          storage_buffer_slots {0};
        u32          sampled_texture_slots{0};
    };
    virtual cardinal::unique_ptr<Pipeline> create_mesh_pipeline(
        const MeshPipelineDesc& /*desc*/) { return {}; }

    // ---- Ray-tracing pipeline (AEGIS Block 7 → PathTracePass) -----------
    //
    // raygen + miss + closest-hit + (optional) any-hit + intersection,
    // wrapped in a shader binding table. Backends without caps.ray_
    // tracing_pipeline return nullptr.
    struct RayTracingPipelineDesc {
        ShaderBlob   raygen;
        ShaderBlob   miss;
        ShaderBlob   closest_hit;
        ShaderBlob   any_hit;          // optional
        ShaderBlob   intersection;     // optional (procedural geom)
        const char*  raygen_entry      {"RayGen"};
        const char*  miss_entry        {"Miss"};
        const char*  closest_hit_entry {"ClosestHit"};
        const char*  any_hit_entry     {"AnyHit"};
        const char*  intersection_entry{"Intersection"};
        u32          max_recursion_depth {1};
        u32          max_payload_bytes   {16};
        u32          max_attribute_bytes {8};
        u32          push_constant_size  {0};
    };
    virtual cardinal::unique_ptr<Pipeline> create_ray_tracing_pipeline(
        const RayTracingPipelineDesc& /*desc*/) { return {}; }

    // ---- Indirect-command signature (DX12 ExecuteIndirect, Vk
    // VK_EXT_device_generated_commands) ---------------------------------
    //
    // The command signature describes the layout of an indirect-args
    // buffer (draw vs dispatch vs draw-indexed vs custom root-constants
    // prepended). Backends without caps.indirect_draw_count return
    // nullptr.
    enum class IndirectCommandType : u32 {
        Draw          = 0,
        DrawIndexed,
        Dispatch,
        DispatchMesh,
        TraceRays,
    };
    virtual cardinal::unique_ptr<Pipeline> create_indirect_signature(
        IndirectCommandType /*type*/, u32 /*stride_bytes*/) { return {}; }

    // ---- Bindless descriptor heap (Block 8) -----------------------------
    //
    // Allocates a single large descriptor heap (~1M slots) that holds
    // every shader-accessible resource. Shaders index via descriptor
    // index from a constant buffer; one heap bind per frame. Backends
    // without caps.bindless_resources return nullptr.
    struct BindlessHeapDesc {
        u32 srv_count {1u << 20};   // sampled textures + SRVs
        u32 uav_count {65536};      // RW textures + UAVs
        u32 cbv_count {65536};
        u32 sampler_count {2048};
    };
    virtual cardinal::unique_ptr<Buffer> create_bindless_heap(
        const BindlessHeapDesc& /*desc*/) { return {}; }
    // Write a sampled-texture / UAV / CBV descriptor into a heap slot.
    // Index space matches the heap layout above. Default no-ops; the
    // backends override when the heap factory returns a real heap.
    virtual void write_bindless_srv(Buffer* /*heap*/, u32 /*index*/, Texture* /*tex*/) {}
    virtual void write_bindless_uav(Buffer* /*heap*/, u32 /*index*/, Buffer* /*buf*/) {}
    virtual void write_bindless_cbv(Buffer* /*heap*/, u32 /*index*/, Buffer* /*buf*/) {}

    // ------------------------------------------------------------------
    // Live VRAM telemetry — drives the budget broker.
    //
    // budget_bytes               : what the OS / driver says we may use
    //                              right now (changes as other apps load
    //                              and unload textures). For DXGI this
    //                              is QueryVideoMemoryInfo::Budget; for
    //                              Vulkan it's VK_EXT_memory_budget's
    //                              heapBudget on the device-local heap.
    // current_usage_bytes         : how much of that budget THIS process
    //                              is currently consuming.
    // available_for_reservation_bytes / current_reservation_bytes:
    //   D3D12-only knobs (zero on Vulkan).
    //
    // Implementations should be cheap — one driver call. Safe to call
    // every frame.
    // ------------------------------------------------------------------
    struct VramSnapshot {
        u64 budget_bytes{0};
        u64 current_usage_bytes{0};
        u64 available_for_reservation_bytes{0};
        u64 current_reservation_bytes{0};
    };
    virtual VramSnapshot query_vram_usage() const noexcept = 0;

protected:
    Device() = default;
};

cardinal::unique_ptr<Device> create_device(const DeviceDesc& desc);

}  // namespace cardinal::rhi
