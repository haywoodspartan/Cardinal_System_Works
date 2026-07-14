#pragma once

// =============================================================================
// Cardinal — Render Graph.
//
// A declarative DAG of GPU passes. Every per-frame operation the engine
// wants to move from CPU to GPU — frustum cull, vertex transform, draw,
// postfx, tonemap — becomes a Pass node with explicit read/write resource
// declarations. The Graph::compile() step topologically sorts the nodes,
// validates the resource accesses, and emits the barrier list a real
// driver would need; Backend::execute() runs the compiled graph.
//
// Two backends ship in-tree:
//   * CpuBackend  — virtual-GPU memory simulator. Runs every pass on the
//                    host CPU using deterministic byte-level reads + writes
//                    so the SAME graph that drives the GPU at runtime can
//                    be validated bit-for-bit in a headless test.
//   * (later) RhiBackend — translates the graph to real vkCmdDispatch /
//                    ID3D12GraphicsCommandList4::Dispatch calls when the
//                    RHI's compute surface lands. The Backend interface
//                    is the seam; the graph + passes don't change.
//
// Design rules (carried from the rest of the engine):
//   * NaN-defensive at every public entrypoint (compile() rejects malformed
//     passes; execute() never invokes a pass with a null context).
//   * No per-frame allocation in the hot path — compile() reserves once;
//     execute() walks pre-sorted indices.
//   * Knob-introspectable settings for backend tunables stay in the host's
//     Pipeline (the graph itself is a typed data structure, not a UI).
//   * Deterministic — same graph + same backend + same inputs → same outputs.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/containers.hpp>

namespace cardinal::render::graph {

// ---------------------------------------------------------------------------
// Resource — typed handle to a GPU-side buffer or texture managed by the
// graph. Created via Graph::declare_buffer / declare_texture; passes
// receive these handles by value (id is stable across a graph's lifetime).
// ---------------------------------------------------------------------------
enum class ResourceKind : u32 {
    Buffer = 0,
    Texture,
};

enum class TextureFormat : u32 {
    Unknown      = 0,
    R8G8B8A8_UNORM,
    R8G8B8A8_SRGB,
    R32_Float,
    R32G32B32A32_Float,
    D32_Float,
};

struct BufferDesc {
    cardinal::string name;
    usize            size_bytes  {0};
    u32              stride_bytes{0};   // 0 = raw buffer; >0 = structured (for compute)
    bool             transient   {true};// true = managed by the graph
};

struct TextureDesc {
    cardinal::string name;
    u32              width  {0};
    u32              height {0};
    TextureFormat    format {TextureFormat::Unknown};
    bool             transient {true};
};

// Stable opaque handle. id == 0 reserved for "invalid"; ids 1.. are
// dense indices into the graph's internal resource table.
struct ResourceHandle {
    u32          id   {0};
    ResourceKind kind {ResourceKind::Buffer};
    bool is_valid() const noexcept { return id != 0; }
};

// ---------------------------------------------------------------------------
// Pass — one node in the graph. Declares which resources it reads + writes
// (so the compiler can topologically sort and insert barriers) and a
// record() callback that performs the actual work when the backend runs it.
// ---------------------------------------------------------------------------
enum class PassKind : u32 {
    Compute = 0,   // pure compute dispatch (CpuBackend runs the callback)
    Raster,        // graphics draw (CpuBackend records only — no rasterizer)
    Copy,          // buffer-to-buffer / texture-to-buffer copy
};

enum class AccessMode : u32 {
    Read = 0,
    Write,
    ReadWrite,
};

struct ResourceAccess {
    ResourceHandle handle;
    AccessMode     mode  {AccessMode::Read};
    u32            slot  {0};   // shader binding slot (informational for CpuBackend)
};

// Execution context handed to a pass's record() callback. Wraps the
// virtual-GPU memory the CpuBackend allocated (or, eventually, the
// real RHI command list). Resources mapped here are scoped to one pass
// invocation — pointers are invalidated on return.
class ExecutionContext {
public:
    virtual ~ExecutionContext() = default;
    ExecutionContext(const ExecutionContext&) = delete;
    ExecutionContext& operator=(const ExecutionContext&) = delete;

    // Pointer to writable bytes for `h` (which must have been declared as
    // Write or ReadWrite in this pass's access list). Returns nullptr if
    // the handle isn't writeable or isn't declared on this pass.
    virtual void*       map_buffer_write(ResourceHandle h) noexcept = 0;
    // Pointer to read-only bytes for `h` (which must have been declared as
    // Read or ReadWrite in this pass's access list).
    virtual const void* map_buffer_read (ResourceHandle h) noexcept = 0;
    virtual usize       buffer_size     (ResourceHandle h) const noexcept = 0;

    // Compute dispatch primitive. CpuBackend treats this as informational
    // (the record callback IS the compute kernel running once on the host);
    // RhiBackend translates it to vkCmdDispatch. Calling dispatch() inside
    // a Raster or Copy pass is a programming error and is silently ignored
    // by both backends.
    virtual void dispatch(u32 gx, u32 gy, u32 gz) noexcept = 0;

    // Push-constant style scratch — small per-pass key/value strings the
    // host can use to thread parameters down without globals.
    virtual void        set_param_u32(const char* key, u32 v) noexcept = 0;
    virtual u32         param_u32    (const char* key, u32 fallback) const noexcept = 0;

protected:
    ExecutionContext() = default;
};

// Record callback. user_ctx is opaque — the pass owns its meaning. The
// graph never inspects it.
using PassRecordFn = void (*)(ExecutionContext& ec, void* user_ctx) noexcept;

struct PassDesc {
    cardinal::string                 name;
    PassKind                         kind   {PassKind::Compute};
    cardinal::vector<ResourceAccess> accesses;

    PassRecordFn record  {nullptr};
    void*        user_ctx{nullptr};

    // Default compute group counts. Hosts can override per-frame inside
    // record() via ExecutionContext::dispatch.
    u32 dispatch_x{1};
    u32 dispatch_y{1};
    u32 dispatch_z{1};

    // GPU compute shader (graph::RhiBackend). When compute_hlsl is non-null the
    // RhiBackend compiles it (ShaderStage::Compute) into a real compute pipeline
    // and dispatches it; when null (or it fails to compile) it falls back to the
    // recording-only telemetry path. The kernel's bindings follow the access
    // ORDER: read-only ByteAddressBuffers at t0.. (Read accesses), read-write
    // RWByteAddressBuffers at u0.. (Write/ReadWrite accesses). push_constant_size
    // is the size of the b0 push block declared at pipeline-creation time;
    // push_constants holds the per-dispatch bytes the RhiBackend uploads (GAP 3
    // param threading — the pass packs its scalars here, e.g. width/height).
    const char*                      compute_hlsl   {nullptr};
    const char*                      compute_entry  {"CSMain"};
    u32                              push_constant_size{0};
    cardinal::vector<cardinal::u8>   push_constants;
};

// ---------------------------------------------------------------------------
// Graph — the builder + the compiled execution plan.
// ---------------------------------------------------------------------------
struct CompileStats {
    u32 passes              {0};
    u32 buffers             {0};
    u32 textures            {0};
    u32 read_after_write    {0};   // pure RAW dependencies (the barrier count)
    u32 write_after_write   {0};   // WAW (also a barrier)
    u32 write_after_read    {0};   // WAR (also a barrier)
    bool cycle_detected     {false};
};

class Graph {
public:
    static cardinal::shared_ptr<Graph> create();

    // ---- declaration phase (pre-compile) -------------------------------
    ResourceHandle declare_buffer (BufferDesc desc);
    ResourceHandle declare_texture(TextureDesc desc);

    // Import an externally-owned resource (host owns lifetime; graph just
    // tracks access). `bytes` is the host-side backing store the
    // CpuBackend will map. Bytes pointer must outlive the graph.
    ResourceHandle import_buffer (cardinal::string name, void* bytes, usize size,
                                  u32 stride_bytes = 0);

    u32 add_pass(PassDesc desc);     // returns pass id (1-based; 0 reserved)
    void mark_output(ResourceHandle h);

    // ---- compile -------------------------------------------------------
    // Returns true on success; on failure stats().cycle_detected is set
    // and execution_order() is left empty. Re-callable: clears prior
    // compile output first.
    bool compile() noexcept;
    const cardinal::vector<u32>& execution_order() const noexcept { return order_; }
    CompileStats stats() const noexcept { return stats_; }

    // Wave decomposition for parallel execution (consumed by
    // ThreadedCpuBackend). wave_of[pass_id] = max(wave_of[predecessor]) + 1.
    // Passes in the same wave share no dependency edges → safe to run
    // concurrently. Computed lazily inside compile() after the topo sort.
    const cardinal::vector<u32>& wave_of_pass() const noexcept { return wave_of_; }
    u32 wave_count() const noexcept { return wave_count_; }

    // ---- read-only accessors after compile -----------------------------
    // Sentinel at index 0 holds an invalid record so id 0 stays reserved
    // for "no handle / no pass". Subtract it from the public count.
    usize             pass_count()     const noexcept { return passes_.empty()    ? 0 : passes_.size()    - 1; }
    usize             resource_count() const noexcept { return resources_.empty() ? 0 : resources_.size() - 1; }
    const PassDesc&   pass(u32 id) const noexcept;
    const BufferDesc& buffer (ResourceHandle h) const noexcept;
    const TextureDesc&texture(ResourceHandle h) const noexcept;
    // Imported (host-supplied) bytes for a buffer resource, or nullptr for a
    // transient one. Lets a GPU backend (RhiBackend) upload initial data
    // without running the CpuBackend. Size is buffer(h).size_bytes.
    const void*       buffer_imported(ResourceHandle h) const noexcept;
    bool              has_output() const noexcept { return output_.is_valid(); }
    ResourceHandle    output() const noexcept { return output_; }

private:
    Graph() = default;

    struct Resource {
        ResourceKind kind  {ResourceKind::Buffer};
        BufferDesc   bdesc;
        TextureDesc  tdesc;
        // Imported pointer (nullptr for transient — the CpuBackend
        // allocates).
        void*        imported_bytes {nullptr};
    };

    cardinal::vector<Resource> resources_;   // [0] is sentinel (id 0 = invalid)
    cardinal::vector<PassDesc> passes_;      // [0] is sentinel
    cardinal::vector<u32>      order_;       // pass ids in execution order
    cardinal::vector<u32>      wave_of_;     // wave_of_[pass_id] = topological depth
    u32                        wave_count_   {0};
    ResourceHandle             output_;
    CompileStats               stats_;
};

// ---------------------------------------------------------------------------
// Backend interface. Plug-in point for executors.
//
// Three concrete backends are planned:
//   * CpuBackend  — virtual-GPU memory simulator. Runs every pass on the
//                    host CPU using deterministic byte-level reads + writes
//                    so the SAME graph that drives the GPU at runtime can
//                    be validated bit-for-bit in a headless test.
//   * NullBackend — records pass execution order + dispatch counts WITHOUT
//                    allocating storage or invoking record(). Useful for
//                    topology validation, frame-graph telemetry, and
//                    smoke-testing that compile() produces a sane order.
//   * RhiBackend  — translates the compiled graph to real
//                    vkCmdDispatch / ID3D12GraphicsCommandList4::Dispatch
//                    calls when the RHI's compute surface lands. Each
//                    Pass's hlsl_source() string compiles via the RHI's
//                    shader compiler at first execute(); subsequent
//                    executes bind resources + dispatch the cached
//                    pipeline. The Backend interface is the seam; the
//                    graph + passes don't change.
// ---------------------------------------------------------------------------
class Backend {
public:
    virtual ~Backend() = default;
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    // Execute the compiled graph. Pre: graph.compile() returned true.
    // Post: every pass's record() has been called in topological order.
    virtual void execute(Graph& g) noexcept = 0;

protected:
    Backend() = default;
};

// ---------------------------------------------------------------------------
// CpuBackend — virtual-GPU memory simulator. Walks the graph's execution
// order, allocates byte storage for every transient resource, hands a
// real pointer to each pass's record() callback, and records a trace of
// the operations for test inspection. Pure CPU; deterministic.
// ---------------------------------------------------------------------------
class CpuBackend : public Backend {
public:
    static cardinal::shared_ptr<CpuBackend> create();

    void execute(Graph& g) noexcept override;

    // ---- post-execute test surface -------------------------------------
    // Read the contents of any buffer (transient or imported) after the
    // last execute() call. Returns empty for unknown handles or for
    // textures (textures aren't byte-readable in the simulator).
    cardinal::vector<u8> buffer_contents(ResourceHandle h) const;

    // Op trace — one entry per pass run in execution order.
    struct PassTrace {
        cardinal::string name;
        PassKind         kind;
        u32              dispatched_gx {0}, dispatched_gy{0}, dispatched_gz{0};
        u32              bytes_written {0};
        u32              bytes_read    {0};
    };
    const cardinal::vector<PassTrace>& trace() const noexcept { return trace_; }

    // Wipes the trace + any retained allocations between executes. Called
    // automatically at the top of execute(); exposed for tests that want
    // to inspect intermediate state.
    void reset() noexcept;

private:
    CpuBackend() = default;

    cardinal::vector<cardinal::vector<u8>> storage_;   // index by resource id
    cardinal::vector<PassTrace>            trace_;
};

// ---------------------------------------------------------------------------
// RhiBackend — real GPU compute dispatch via cardinal::rhi.
//
// The third concrete Backend, alongside CpuBackend (virtual-GPU
// simulator for tests) and NullBackend (topology only). RhiBackend
// compiles each Pass's hlsl_source() once via the RHI's shader
// compiler, then on each execute() walks the graph's topological
// order and emits real bind_pipeline / bind_storage_buffer_uav /
// dispatch calls on a borrowed rhi::Swapchain command recorder.
//
// Resource binding:
//   * Output handles (graph::AccessMode::Write / ReadWrite) → bound
//     as UAVs via bind_storage_buffer_uav(slot, ...).
//   * Input handles  (graph::AccessMode::Read)              → bound
//     as read-only storage buffers via bind_storage_buffer(slot, ...).
//   * Each Pass's slot indices come from its declared ResourceAccess
//     list — same slots the CpuBackend's ExecutionContext uses, so
//     the per-pass HLSL register decls match across backends.
//
// Until rhi::Device::create_compute_pipeline returns a non-null Pipeline
// (current backends default-return {}), RhiBackend records the
// dispatch sequence into a trace (same shape as NullBackend's events)
// so hosts can verify the graph reaches the dispatch site without
// crashing. When the Vulkan / D3D12 backends wire compute, the same
// RhiBackend code starts firing real vkCmdDispatch calls with zero
// AEGIS-side changes.
// ---------------------------------------------------------------------------
}  // namespace cardinal::render::graph

namespace cardinal::rhi { class Device; class Swapchain; class Pipeline; class Buffer; }

namespace cardinal::render::graph {

class RhiBackend : public Backend {
public:
    static cardinal::shared_ptr<RhiBackend> create(cardinal::rhi::Device& dev,
                                                   cardinal::rhi::Swapchain& sw);

    void execute(Graph& g) noexcept override;

    struct PassEvent {
        cardinal::string name;
        PassKind         kind;
        u32              dispatch_x {0}, dispatch_y{0}, dispatch_z{0};
        bool             real_dispatch {false};   // true when create_compute_pipeline succeeded
    };
    const cardinal::vector<PassEvent>& events() const noexcept { return events_; }

    struct Stats {
        u32 passes_dispatched_real    {0};   // create_compute_pipeline succeeded
        u32 passes_dispatched_stub    {0};   // backend returned null pipeline
        u32 pipelines_compiled        {0};   // cumulative pipeline cache size
    };
    Stats stats() const noexcept { return stats_; }
    void reset() noexcept;

    // Opt-in GPU compute execution. OFF by default: the Studio runs this backend
    // every frame for telemetry while the ForwardRenderer draws the screen, so
    // building the AEGIS buffer map + dispatching compute here is pure overhead
    // — and until the path is validated on real hardware it regressed FPS on
    // Vulkan and hard-crashed D3D12 (per-frame buffer churn + dispatching with
    // partially-bound descriptors). When false, execute() does recording-only
    // telemetry (NullBackend-equivalent cost). Flip on to drive the real GPU path.
    void set_gpu_execute(bool e) noexcept { gpu_execute_ = e; }
    bool gpu_execute() const noexcept { return gpu_execute_; }

    // The live rhi::Buffer backing a graph resource (by BufferDesc name), or
    // null if the resource never materialised. Used by the GPU validation
    // harness to copy pass outputs into readback buffers after execute().
    cardinal::rhi::Buffer* gpu_buffer(const cardinal::string& name) const noexcept;

    // Fence + transition `name`'s buffer into CopySource based on the state
    // execute() left it in (UAV write vs SRV read — the harness can't know).
    // Call between execute() and Swapchain::copy_buffer, on the SAME
    // recording the execute targeted.
    void make_copy_source(const cardinal::string& name) noexcept;

private:
    RhiBackend() = default;

    cardinal::rhi::Device*    dev_ {nullptr};
    cardinal::rhi::Swapchain* sw_  {nullptr};
    // Pass NAME → compiled compute pipeline (lazy, cached across executes).
    // Keyed by name, not pass id: the graph is rebuilt per frame and pass ids
    // SHIFT when the topology changes (ReSTIR / MotionVector wire in
    // conditionally) — an id-keyed cache would hand a kernel-less pass a
    // stale slot and dispatch the WRONG kernel with its buffers. `tried`
    // stops a failing kernel from re-running DXC every frame.
    struct CachedPipeline {
        cardinal::unique_ptr<cardinal::rhi::Pipeline> pipe;
        bool                                          tried {false};
    };
    cardinal::unordered_map<cardinal::string, CachedPipeline> pipeline_cache_;
    // GAP-2 buffer storage. buffer_cache_ OWNS the rhi::Buffers and PERSISTS them
    // across frames, keyed by resource name + size — so a stable graph reuses its
    // buffers instead of recreating every one every execute (the per-frame churn
    // that bled FPS / crashed allocators). buffers_ is a per-frame non-owning
    // ResourceHandle.id → Buffer* view rebuilt from the cache each execute.
    struct CachedBuffer {
        cardinal::unique_ptr<cardinal::rhi::Buffer> buf;
        usize                                       size {0};
        bool host_visible {false};   // imported/UPLOAD-heap (never a UAV; on
                                     // D3D12 pinned GENERIC_READ — no barriers)
    };
    cardinal::unordered_map<cardinal::string, CachedBuffer> buffer_cache_;
    cardinal::vector<cardinal::rhi::Buffer*>                buffers_;
    // Final per-buffer use state after the last execute() (Srv/Uav), so
    // make_copy_source can issue the correct pre-copy barrier.
    enum class BufUse : u8 { Untouched, Srv, Uav };
    cardinal::unordered_map<cardinal::rhi::Buffer*, BufUse> final_use_;
    cardinal::vector<PassEvent> events_;
    Stats stats_;
    bool  gpu_execute_ {false};   // opt-in; see set_gpu_execute (default safe)
};

// ---------------------------------------------------------------------------
// ThreadedCpuBackend — parallel pass execution. Decomposes the topo
// order into "waves": passes within a wave share no dependency edges
// and can run concurrently on different worker threads. Each wave
// dispatches via cardinal::async::parallel_for_each (falls back to
// serial when no async pool is bound).
//
// This is the DX12/Vulkan "multi-threaded command recording" model
// expressed at the graph level: every Pass becomes one work item, and
// the wave decomposition guarantees no two concurrent passes share a
// resource (RAW / WAW / WAR edges all force serialisation). For an
// AEGIS frame with M=14 passes, typical waves are 1-3 passes wide
// (mostly serial), but the cull/transform/light-cull/post chain has
// 4-wide waves under realistic scene topology.
//
// Stats: wall-clock per wave, total threaded time, serial-equivalent
// estimate (= sum of all pass times), and the achieved CPU speedup.
// Hosts use the speedup metric to decide whether the cost of
// std::thread synchronisation pays off vs CpuBackend's pure-serial
// execution. Below ~3 passes per wave the speedup is typically < 1.0
// and CpuBackend wins.
// ---------------------------------------------------------------------------
class ThreadedCpuBackend : public Backend {
public:
    static cardinal::shared_ptr<ThreadedCpuBackend> create();

    void execute(Graph& g) noexcept override;

    cardinal::vector<u8> buffer_contents(ResourceHandle h) const;

    struct WaveStats {
        u32   wave_id    {0};
        u32   pass_count {0};
        float duration_us {0.0f};
    };
    const cardinal::vector<WaveStats>& waves() const noexcept { return waves_; }
    float total_us()      const noexcept { return total_us_; }
    float serial_est_us() const noexcept { return serial_est_us_; }
    // Speedup = serial_est / total. > 1 means parallel was faster.
    float speedup()       const noexcept;
    void reset() noexcept;

private:
    ThreadedCpuBackend() = default;
    cardinal::vector<cardinal::vector<u8>> storage_;
    cardinal::vector<WaveStats>            waves_;
    float                                  total_us_      {0.0f};
    float                                  serial_est_us_ {0.0f};
};

// ---------------------------------------------------------------------------
// NullBackend — topology-only executor. Walks the graph's execution order
// and records the trace WITHOUT allocating per-resource storage or
// invoking each pass's record() function. Useful for:
//   * Telemetry / frame-graph visualisation (the editor's "what passes
//     ran this frame" panel can run NullBackend on a debug graph fork)
//   * Regression-pinning the compile() output (execution order +
//     dependency counts) independent of per-pass behaviour
//   * Smoke-testing that a freshly-built AegisPipeline produces a
//     well-ordered graph before the host commits to allocating buffers
//     for the real execute()
// Much cheaper than CpuBackend: zero allocation per resource, zero
// record() invocations, deterministic.
// ---------------------------------------------------------------------------
class NullBackend : public Backend {
public:
    static cardinal::shared_ptr<NullBackend> create();

    void execute(Graph& g) noexcept override;

    struct PassEvent {
        cardinal::string name;
        PassKind         kind;
        u32              dispatch_x {0}, dispatch_y{0}, dispatch_z{0};
    };
    const cardinal::vector<PassEvent>& events() const noexcept { return events_; }
    void reset() noexcept { events_.clear(); }

    // Aggregate stats over the last execute().
    struct Stats {
        u32 passes_executed {0};
        u32 compute_passes  {0};
        u32 raster_passes   {0};
        u32 copy_passes     {0};
    };
    Stats stats() const noexcept;

private:
    NullBackend() = default;
    cardinal::vector<PassEvent> events_;
};

}  // namespace cardinal::render::graph
