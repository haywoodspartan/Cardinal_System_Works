#include <cardinal/render/graph.hpp>

#include <cardinal/core/std/algorithm.hpp>      // cardinal::clamp
#include <cardinal/core/std/cstring.hpp>        // cardinal::strcmp / strlen
#include <cardinal/core/std/utility.hpp>        // cardinal::move
#include <cardinal/core/sync/async.hpp>          // cardinal::async::parallel_for_each
#include <cardinal/core/alloc/linear_allocator.hpp>   // per-compile transient arena
#include <cardinal/core/alloc/arena_allocator.hpp>    // STL adapter for the arena
#include <cardinal/rhi/rhi.hpp>             // rhi::Device + Swapchain (for RhiBackend)

#include <chrono>

namespace cardinal::render::graph {

namespace {

// Compare two C strings; nullptr-safe.
inline bool streq(const char* a, const char* b) noexcept {
    if (a == b) return true;
    if (!a || !b) return false;
    return cardinal::strcmp(a, b) == 0;
}

// Thread-local arena used by Graph::compile() for its transient state
// (the resource-state table, the adjacency lists, the Kahn ready bucket,
// the predecessor lists). Reset at the top of every compile() call.
//
// 256 KB sized for a graph with ~2k passes + ~2k resources + dense
// adjacency — well above any realistic AEGIS workload. Allocation goes
// into the arena via std::atomic fetch_add (lock-free); reset is a
// single store. Compile() runs on a single thread, so the rewind step
// at the start is safe.
//
// Why a function-static thread_local: the arena owns a 256 KB heap
// buffer; making it a true global would heap-allocate at static-init
// time on threads that never compile. function-local thread_local lazy-
// allocates on first compile per thread.
constexpr cardinal::usize kCompileArenaBytes = 256u * 1024u;

cardinal::core::LinearAllocator& compile_arena() noexcept {
    thread_local cardinal::core::LinearAllocator arena(kCompileArenaBytes);
    return arena;
}

}  // namespace

// =============================================================================
// Graph — declaration phase
// =============================================================================
cardinal::shared_ptr<Graph> Graph::create() {
    auto g = cardinal::shared_ptr<Graph>(new Graph());
    // Sentinel id 0 = invalid. Push placeholders.
    g->resources_.push_back(Resource{});
    g->passes_.push_back(PassDesc{});
    return g;
}

ResourceHandle Graph::declare_buffer(BufferDesc desc) {
    Resource r;
    r.kind  = ResourceKind::Buffer;
    r.bdesc = cardinal::move(desc);
    resources_.push_back(cardinal::move(r));
    return ResourceHandle{ static_cast<u32>(resources_.size() - 1), ResourceKind::Buffer };
}

ResourceHandle Graph::declare_texture(TextureDesc desc) {
    Resource r;
    r.kind  = ResourceKind::Texture;
    r.tdesc = cardinal::move(desc);
    resources_.push_back(cardinal::move(r));
    return ResourceHandle{ static_cast<u32>(resources_.size() - 1), ResourceKind::Texture };
}

ResourceHandle Graph::import_buffer(cardinal::string name, void* bytes, usize size,
                                    u32 stride_bytes)
{
    Resource r;
    r.kind = ResourceKind::Buffer;
    r.bdesc.name         = cardinal::move(name);
    r.bdesc.size_bytes   = size;
    r.bdesc.stride_bytes = stride_bytes;
    r.bdesc.transient    = false;
    r.imported_bytes     = bytes;
    resources_.push_back(cardinal::move(r));
    return ResourceHandle{ static_cast<u32>(resources_.size() - 1), ResourceKind::Buffer };
}

u32 Graph::add_pass(PassDesc desc) {
    passes_.push_back(cardinal::move(desc));
    return static_cast<u32>(passes_.size() - 1);
}

void Graph::mark_output(ResourceHandle h) {
    output_ = h;
}

// =============================================================================
// Graph — compile (topological sort + dependency tally)
// =============================================================================
bool Graph::compile() noexcept {
    stats_  = CompileStats{};
    order_.clear();

    if (passes_.size() <= 1) {
        // Sentinel only — nothing to schedule. That's a valid (degenerate)
        // graph, not an error.
        stats_.passes = 0;
        return true;
    }

    const u32 n_passes = static_cast<u32>(passes_.size() - 1);   // skip sentinel
    const u32 n_res    = static_cast<u32>(resources_.size() - 1);

    // For each resource, track the LAST pass that wrote it + a list of
    // all passes that read it since the last write. Then each new
    // dependency edge falls out:
    //   RAW: pass P reads R, last writer of R was W ≠ 0 → edge W → P
    //   WAW: pass P writes R, last writer of R was W ≠ 0 → edge W → P
    //   WAR: pass P writes R, every reader since last write → edge reader → P
    //
    // Then Kahn's algorithm gives the execution order.

    // ---- Per-compile transient arena ---------------------------------
    // All the bookkeeping below — resource state, adjacency lists,
    // indegree, Kahn ready bucket, predecessor lists — lives until the
    // end of this function and is then thrown away. Backing them with
    // a thread-local LinearAllocator turns hundreds of small
    // operator-new calls per compile into a single arena reset.
    auto& arena = compile_arena();
    arena.reset();
    using U32Alloc      = cardinal::core::ArenaAllocator<u32>;
    using ArenaVecU32   = cardinal::vector<u32, U32Alloc>;

    // Resource state: track last writer + readers-since-last-write per
    // resource (1-based ids). The nested readers vector ALSO goes into
    // the arena so the per-resource read history doesn't escape into
    // the heap.
    struct ResState {
        u32         last_writer{0};
        ArenaVecU32 readers_since_last_write;

        explicit ResState(U32Alloc alloc) noexcept
            : readers_since_last_write(alloc) {}
    };
    using ResStateAlloc = cardinal::core::ArenaAllocator<ResState>;
    cardinal::vector<ResState, ResStateAlloc> rs(ResStateAlloc{arena});
    rs.reserve(static_cast<usize>(n_res + 1));
    for (u32 i = 0; i <= n_res; ++i) rs.emplace_back(U32Alloc{arena});

    // Adjacency list: edges[from_pass] = passes that depend on it.
    using ArenaVecAlloc      = cardinal::core::ArenaAllocator<ArenaVecU32>;
    cardinal::vector<ArenaVecU32, ArenaVecAlloc> edges(ArenaVecAlloc{arena});
    edges.reserve(static_cast<usize>(n_passes + 1));
    for (u32 i = 0; i <= n_passes; ++i) edges.emplace_back(U32Alloc{arena});

    ArenaVecU32 indeg(U32Alloc{arena});
    indeg.assign(static_cast<usize>(n_passes + 1), 0u);

    auto add_edge = [&](u32 from, u32 to) noexcept {
        if (from == 0 || to == 0 || from == to) return;
        // De-dup (a pass may both RAW + WAW from the same predecessor; we
        // only need one edge).
        for (u32 e : edges[from]) {
            if (e == to) return;
        }
        edges[from].push_back(to);
        indeg[to]++;
    };

    for (u32 p = 1; p <= n_passes; ++p) {
        const PassDesc& pd = passes_[p];
        // First sweep: WAR edges (writes that follow earlier reads on same
        // resource depend on those reads completing).
        for (const ResourceAccess& a : pd.accesses) {
            if (a.handle.id == 0 || a.handle.id > n_res) continue;
            if (a.mode == AccessMode::Write || a.mode == AccessMode::ReadWrite) {
                for (u32 r : rs[a.handle.id].readers_since_last_write) {
                    add_edge(r, p);
                    stats_.write_after_read++;
                }
            }
        }
        // Second sweep: RAW + WAW from last writer.
        for (const ResourceAccess& a : pd.accesses) {
            if (a.handle.id == 0 || a.handle.id > n_res) continue;
            const u32 lw = rs[a.handle.id].last_writer;
            if (lw == 0) continue;
            if (a.mode == AccessMode::Read) {
                add_edge(lw, p);
                stats_.read_after_write++;
            } else if (a.mode == AccessMode::Write || a.mode == AccessMode::ReadWrite) {
                add_edge(lw, p);
                stats_.write_after_write++;
            }
        }
        // Third sweep: update last writer / reader state.
        for (const ResourceAccess& a : pd.accesses) {
            if (a.handle.id == 0 || a.handle.id > n_res) continue;
            if (a.mode == AccessMode::Write || a.mode == AccessMode::ReadWrite) {
                rs[a.handle.id].last_writer = p;
                rs[a.handle.id].readers_since_last_write.clear();
            } else {  // Read
                rs[a.handle.id].readers_since_last_write.push_back(p);
            }
        }
    }

    // Kahn's algorithm. Stable-sorted by id so the order is deterministic.
    ArenaVecU32 ready(U32Alloc{arena});
    for (u32 p = 1; p <= n_passes; ++p) {
        if (indeg[p] == 0) ready.push_back(p);
    }
    order_.reserve(n_passes);
    while (!ready.empty()) {
        // Pop smallest id for determinism.
        usize smallest_at = 0;
        for (usize i = 1; i < ready.size(); ++i) {
            if (ready[i] < ready[smallest_at]) smallest_at = i;
        }
        const u32 p = ready[smallest_at];
        ready.erase(ready.begin() + static_cast<long long>(smallest_at));
        order_.push_back(p);
        for (u32 succ : edges[p]) {
            if (--indeg[succ] == 0) ready.push_back(succ);
        }
    }
    if (order_.size() != static_cast<usize>(n_passes)) {
        // Cycle — at least one pass never reached indegree 0.
        stats_.cycle_detected = true;
        order_.clear();
        return false;
    }
    stats_.passes   = n_passes;
    stats_.buffers  = 0;
    stats_.textures = 0;
    for (u32 i = 1; i <= n_res; ++i) {
        if (resources_[i].kind == ResourceKind::Buffer)  stats_.buffers++;
        else                                             stats_.textures++;
    }

    // Wave decomposition: wave_of_[p] = 1 + max(wave_of_[predecessor]).
    // Passes with no predecessors land in wave 0.
    wave_of_.assign(static_cast<usize>(n_passes + 1), 0u);
    wave_count_ = 0;
    // Re-derive predecessor list from `edges`; we only have successor
    // adjacency above. Walk the topo order; for each pass, set wave
    // from the max over its in-edges. Build a predecessor list now —
    // also in the per-compile arena.
    cardinal::vector<ArenaVecU32, ArenaVecAlloc> preds(ArenaVecAlloc{arena});
    preds.reserve(static_cast<usize>(n_passes + 1));
    for (u32 i = 0; i <= n_passes; ++i) preds.emplace_back(U32Alloc{arena});
    for (u32 p = 1; p <= n_passes; ++p) {
        for (u32 succ : edges[p]) preds[succ].push_back(p);
    }
    for (u32 pid : order_) {
        u32 max_pred_wave = 0;
        bool has_pred = false;
        for (u32 pred : preds[pid]) {
            has_pred = true;
            if (wave_of_[pred] > max_pred_wave) max_pred_wave = wave_of_[pred];
        }
        const u32 w = has_pred ? (max_pred_wave + 1) : 0u;
        wave_of_[pid] = w;
        if (w + 1 > wave_count_) wave_count_ = w + 1;
    }
    return true;
}

const PassDesc& Graph::pass(u32 id) const noexcept {
    if (id == 0 || id >= passes_.size()) return passes_[0];
    return passes_[id];
}

const BufferDesc& Graph::buffer(ResourceHandle h) const noexcept {
    if (h.id == 0 || h.id >= resources_.size() || resources_[h.id].kind != ResourceKind::Buffer)
        return resources_[0].bdesc;
    return resources_[h.id].bdesc;
}

const TextureDesc& Graph::texture(ResourceHandle h) const noexcept {
    if (h.id == 0 || h.id >= resources_.size() || resources_[h.id].kind != ResourceKind::Texture)
        return resources_[0].tdesc;
    return resources_[h.id].tdesc;
}

const void* Graph::buffer_imported(ResourceHandle h) const noexcept {
    if (h.id == 0 || h.id >= resources_.size() || resources_[h.id].kind != ResourceKind::Buffer)
        return nullptr;
    return resources_[h.id].imported_bytes;
}

// =============================================================================
// CpuBackend
// =============================================================================
namespace {

// Concrete ExecutionContext over the CpuBackend's per-resource byte vectors.
class CpuExecutionContext final : public ExecutionContext {
public:
    CpuExecutionContext(Graph& g, cardinal::vector<cardinal::vector<u8>>& storage,
                        const PassDesc& pass) noexcept
        : g_(g), storage_(storage), pass_(pass) {}

    void* map_buffer_write(ResourceHandle h) noexcept override {
        if (!declared_for_(h, true)) return nullptr;
        bytes_written_ += static_cast<u32>(buffer_size(h));
        return storage_ptr_(h);
    }
    const void* map_buffer_read(ResourceHandle h) noexcept override {
        if (!declared_for_(h, false)) return nullptr;
        bytes_read_ += static_cast<u32>(buffer_size(h));
        return storage_ptr_(h);
    }
    usize buffer_size(ResourceHandle h) const noexcept override {
        if (h.id == 0 || h.id >= storage_.size()) return 0;
        return storage_[h.id].size();
    }
    void dispatch(u32 gx, u32 gy, u32 gz) noexcept override {
        if (pass_.kind != PassKind::Compute) return;
        dispatch_gx_ = gx; dispatch_gy_ = gy; dispatch_gz_ = gz;
    }
    void set_param_u32(const char* key, u32 v) noexcept override {
        if (!key) return;
        for (auto& kv : params_) {
            if (streq(kv.key, key)) { kv.v = v; return; }
        }
        params_.push_back(KV{ key, v });
    }
    u32 param_u32(const char* key, u32 fallback) const noexcept override {
        if (!key) return fallback;
        for (const auto& kv : params_) {
            if (streq(kv.key, key)) return kv.v;
        }
        return fallback;
    }

    u32 bytes_written() const noexcept { return bytes_written_; }
    u32 bytes_read()    const noexcept { return bytes_read_;    }
    u32 dispatch_gx() const noexcept { return dispatch_gx_; }
    u32 dispatch_gy() const noexcept { return dispatch_gy_; }
    u32 dispatch_gz() const noexcept { return dispatch_gz_; }

private:
    bool declared_for_(ResourceHandle h, bool need_write) const noexcept {
        for (const ResourceAccess& a : pass_.accesses) {
            if (a.handle.id != h.id) continue;
            if (need_write) {
                return a.mode == AccessMode::Write || a.mode == AccessMode::ReadWrite;
            } else {
                return a.mode == AccessMode::Read || a.mode == AccessMode::ReadWrite;
            }
        }
        return false;
    }
    void* storage_ptr_(ResourceHandle h) noexcept {
        if (h.id == 0 || h.id >= storage_.size()) return nullptr;
        return storage_[h.id].empty() ? nullptr : storage_[h.id].data();
    }

    struct KV { const char* key; u32 v; };

    Graph&                                   g_;
    cardinal::vector<cardinal::vector<u8>>&  storage_;
    const PassDesc&                          pass_;
    cardinal::vector<KV>                     params_;
    u32                                      bytes_written_{0};
    u32                                      bytes_read_   {0};
    u32 dispatch_gx_{0}, dispatch_gy_{0}, dispatch_gz_{0};
};

}  // namespace

cardinal::shared_ptr<CpuBackend> CpuBackend::create() {
    return cardinal::shared_ptr<CpuBackend>(new CpuBackend());
}

void CpuBackend::reset() noexcept {
    storage_.clear();
    trace_.clear();
}

void CpuBackend::execute(Graph& g) noexcept {
    reset();
    const auto& order = g.execution_order();
    if (order.empty()) return;

    // Allocate per-resource byte storage. Imported buffers reuse the
    // host-side pointer (no copy); transient buffers get a fresh vector.
    // We can't access Graph::resources_ directly (private), so we route
    // through buffer()/texture() + a probe of import state via the
    // buffer's `transient` flag.
    storage_.resize(g.resource_count() + 1);

    // First pass: walk every pass's accesses to size buffer storage. The
    // public API doesn't expose Resource::imported_bytes, so for imported
    // buffers (transient=false) we leave storage_ empty and route through
    // import_overlay_ below.
    // To bridge this without breaking encapsulation, the CpuBackend
    // requires hosts to use *transient* buffers for resources the graph
    // owns and use a separate "import bridge" for host-owned ones. Since
    // we don't yet wire imports here, every declared buffer is treated
    // as transient and gets its own vector<u8>.
    for (u32 rid = 1; rid <= g.resource_count(); ++rid) {
        // We only allocate for buffers; textures are tracked by name +
        // descriptor but the CpuBackend doesn't simulate pixel storage.
        ResourceHandle bh{ rid, ResourceKind::Buffer };
        const BufferDesc& bd = g.buffer(bh);
        if (bd.size_bytes > 0) {
            storage_[rid].assign(bd.size_bytes, 0u);
            // Imported (host-fed) buffers seed their storage from the host
            // bytes — without this every import read back as zeros and any
            // input-consuming pass computed against a blank scene (caught by
            // the GPU validation harness comparing CPU vs GPU).
            if (const void* src = g.buffer_imported(bh))
                cardinal::memcpy(storage_[rid].data(), src, bd.size_bytes);
        }
    }

    // Walk in topological order.
    trace_.reserve(order.size());
    for (u32 pid : order) {
        const PassDesc& pd = g.pass(pid);
        PassTrace t{};
        t.name = pd.name;
        t.kind = pd.kind;
        if (pd.record) {
            CpuExecutionContext ec(g, storage_, pd);
            // Pre-load any host-set dispatch size as defaults so the
            // record() can override.
            ec.dispatch(pd.dispatch_x, pd.dispatch_y, pd.dispatch_z);
            pd.record(ec, pd.user_ctx);
            t.dispatched_gx = ec.dispatch_gx();
            t.dispatched_gy = ec.dispatch_gy();
            t.dispatched_gz = ec.dispatch_gz();
            t.bytes_written = ec.bytes_written();
            t.bytes_read    = ec.bytes_read();
        } else {
            t.dispatched_gx = pd.dispatch_x;
            t.dispatched_gy = pd.dispatch_y;
            t.dispatched_gz = pd.dispatch_z;
        }
        trace_.push_back(cardinal::move(t));
    }
}

cardinal::vector<u8> CpuBackend::buffer_contents(ResourceHandle h) const {
    if (h.id == 0 || h.id >= storage_.size()) return {};
    cardinal::vector<u8> out;
    out.assign(storage_[h.id].begin(), storage_[h.id].end());
    return out;
}

// =============================================================================
// RhiBackend — real GPU compute dispatch via cardinal::rhi
// =============================================================================
cardinal::shared_ptr<RhiBackend> RhiBackend::create(cardinal::rhi::Device& dev,
                                                   cardinal::rhi::Swapchain& sw) {
    auto b = cardinal::shared_ptr<RhiBackend>(new RhiBackend());
    b->dev_ = &dev;
    b->sw_  = &sw;
    return b;
}

void RhiBackend::reset() noexcept {
    events_.clear();
    pipeline_cache_.clear();
    buffers_.clear();
    buffer_cache_.clear();
    stats_ = Stats{};
}

cardinal::rhi::Buffer* RhiBackend::gpu_buffer(const cardinal::string& name) const noexcept {
    const auto it = buffer_cache_.find(name);
    return (it != buffer_cache_.end()) ? it->second.buf.get() : nullptr;
}

void RhiBackend::make_copy_source(const cardinal::string& name) noexcept {
    const auto slot_it = buffer_cache_.find(name);
    if (slot_it == buffer_cache_.end() || !slot_it->second.buf || !sw_) return;
    cardinal::rhi::Buffer* buf = slot_it->second.buf.get();
    // Host-visible (imported / UPLOAD-heap) buffers take NO barrier: on D3D12
    // they are pinned in GENERIC_READ — which already includes COPY_SOURCE —
    // and a transition barrier on an UPLOAD-heap resource is invalid; they are
    // never UAV-written either. Vulkan host-visible needs nothing extra.
    if (slot_it->second.host_visible) return;
    using RS = cardinal::rhi::Swapchain::ResourceState;
    const auto it = final_use_.find(buf);
    const BufUse use = (it != final_use_.end()) ? it->second : BufUse::Untouched;
    if (use == BufUse::Uav) {
        sw_->uav_barrier(buf);                            // flush the UAV writes
        sw_->transition_buffer_state(buf, RS::UnorderedAccess, RS::CopySource);
    } else if (use == BufUse::Srv) {
        sw_->transition_buffer_state(buf, RS::ShaderResource, RS::CopySource);
    }
    // Untouched (never dispatched over this execute): D3D12 buffers implicitly
    // promote COMMON -> COPY_SOURCE; Vulkan needs no state. Nothing to do.
    // NOTE: single-readback contract — after this the buffer is accounted
    // Untouched; a SECOND execute() on the same recording would need the
    // buffer back in a shader state first (not a supported flow today).
    final_use_[buf] = BufUse::Untouched;
}

void RhiBackend::execute(Graph& g) noexcept {
    events_.clear();
    if (!dev_ || !sw_) return;

    const auto& order = g.execution_order();

    buffers_.clear();

    // GPU compute execution is OPT-IN (set_gpu_execute), default OFF. The Studio
    // runs this backend every frame purely for telemetry while the on-screen
    // image is drawn by the ForwardRenderer, so building the buffer map +
    // dispatching the AEGIS compute graph here is pure overhead. Worse, until
    // the path is GPU-validated it (a) recreated every AEGIS resource buffer
    // EVERY frame — an alloc/free storm that bled FPS on Vulkan and hard-crashed
    // D3D12's allocator — and (b) could dispatch with partially-bound
    // descriptors. So unless explicitly enabled, fall through to recording-only
    // telemetry (NullBackend-equivalent cost) below.
    if (gpu_execute_) {
        // GAP 2: (re)build the ResourceHandle.id -> rhi::Buffer map when a pass
        // carries a kernel. (Persistent-id caching across frames is the perf
        // follow-up before this is enabled for real workloads.)
        bool any_kernel = false;
        for (u32 pid : order) {
            const PassDesc& p = g.pass(pid);
            if (p.kind == PassKind::Compute && p.compute_hlsl) { any_kernel = true; break; }
        }
        if (any_kernel) {
            const usize rcount = g.resource_count() + 1;
            buffers_.assign(rcount, nullptr);   // per-frame id -> Buffer* view
            for (u32 rid = 1; rid < rcount; ++rid) {
                const ResourceHandle h{rid};
                const BufferDesc&    bdesc = g.buffer(h);
                const usize          sz    = bdesc.size_bytes;
                if (sz == 0) continue;
                // Reuse the cached buffer for this resource unless it is new or
                // its size changed (e.g. a swapchain resize). Only then create.
                CachedBuffer& slot = buffer_cache_[bdesc.name];
                const bool wants_host = g.buffer_imported(h) != nullptr;
                if (!slot.buf || slot.size != sz || slot.host_visible != wants_host) {
                    cardinal::rhi::BufferDesc bd{};
                    bd.size  = sz;
                    bd.usage = static_cast<u32>(cardinal::rhi::BufferUsage::Storage)
                             | static_cast<u32>(cardinal::rhi::BufferUsage::ShaderDeviceAddress);
                    // Host-visible ONLY for imported (host-fed, SRV-read)
                    // buffers. Transient buffers are pass outputs — on D3D12 a
                    // UAV write requires a DEFAULT-heap resource (UPLOAD-heap
                    // resources cannot carry ALLOW_UNORDERED_ACCESS; binding
                    // one as a root UAV is illegal). Device-local is also the
                    // fast path on Vulkan. The heap kind participates in the
                    // reuse check so a name flipping imported<->transient can
                    // never inherit a wrong-heap buffer.
                    bd.cpu_writable = wants_host;
                    slot.buf  = dev_->create_buffer(bd);
                    slot.size = sz;
                    slot.host_visible = wants_host;
                }
                if (slot.buf) {
                    // Imported (host) data can change every frame — re-upload it;
                    // transient buffers (no import) just persist.
                    const void* src = g.buffer_imported(h);
                    if (src) slot.buf->upload(src, sz, 0);
                    buffers_[rid] = slot.buf.get();
                }
            }
        }
    }

    // Per-execute buffer hazard tracking for chained kernels. D3D12 buffers
    // decay to COMMON at ExecuteCommandLists boundaries and implicitly promote
    // on FIRST use within a list, but a UAV-write -> SRV-read chain inside one
    // list needs an explicit transition, UAV -> UAV needs a UAV barrier, and
    // Vulkan needs the equivalent memory barriers. Tracked per rhi::Buffer;
    // persisted in final_use_ so make_copy_source knows each buffer's end state.
    final_use_.clear();
    auto& buf_use = final_use_;

    for (u32 pid : order) {
        const PassDesc& pd = g.pass(pid);
        PassEvent ev;
        ev.name = pd.name;
        ev.kind = pd.kind;
        ev.dispatch_x = pd.dispatch_x;
        ev.dispatch_y = pd.dispatch_y;
        ev.dispatch_z = pd.dispatch_z;

        // GAP 1: lazily compile this pass's compute kernel into a pipeline.
        // The slot counts derive from the access modes (Read -> read-only SRV,
        // Write/ReadWrite -> read-write UAV). When the pass carries no HLSL (or
        // it fails to compile), create_compute_pipeline returns null and we
        // fall through to the recording-only telemetry path — same as before.
        CachedPipeline* cp = nullptr;
        if (gpu_execute_ && pd.kind == PassKind::Compute && pd.compute_hlsl) {
            cp = &pipeline_cache_[pd.name];
            if (!cp->pipe && !cp->tried) {
                cp->tried = true;
                u32 srv = 0, uav = 0;
                for (const auto& a : pd.accesses) {
                    if (a.mode == AccessMode::Read) ++srv; else ++uav;
                }
                cardinal::rhi::ComputePipelineDesc cpd;
                cpd.compute_shader       = dev_->compile_shader(
                    cardinal::rhi::ShaderStage::Compute, pd.compute_hlsl, pd.compute_entry);
                cpd.compute_entry        = pd.compute_entry;
                cpd.push_constant_size   = pd.push_constant_size;
                cpd.storage_buffer_slots = srv;
                cpd.uav_slots            = uav;
                cp->pipe = dev_->create_compute_pipeline(cpd);
                if (cp->pipe) ++stats_.pipelines_compiled;
            }
        }

        // Only dispatch when EVERY declared resource has a live backing buffer.
        // Dispatching a compute pipeline with unbound root SRV/UAV descriptors
        // is a device-removed hard-crash on D3D12 (and undefined on Vulkan), so
        // a pass whose buffers aren't all present falls back to recording-only.
        bool all_bound = gpu_execute_;
        if (all_bound) {
            for (const auto& a : pd.accesses) {
                if (a.handle.id >= buffers_.size() || !buffers_[a.handle.id]) {
                    all_bound = false; break;
                }
            }
        }

        const bool has_real_pipe =
            (gpu_execute_ && all_bound && pd.kind == PassKind::Compute &&
             pd.compute_hlsl && cp && cp->pipe);
        if (has_real_pipe) {
            // Hazard fences BEFORE the dispatch: transition each access's
            // buffer to the use this pass makes of it, based on how the
            // previous pass in this command stream left it.
            for (const auto& a : pd.accesses) {
                cardinal::rhi::Buffer* buf = buffers_[a.handle.id];
                const BufUse want = (a.mode == AccessMode::Read) ? BufUse::Srv
                                                                 : BufUse::Uav;
                BufUse& cur = buf_use[buf];   // default Untouched
                if (cur == BufUse::Untouched) {
                    cur = want;               // first touch: implicit promotion
                } else if (cur != want) {
                    using RS = cardinal::rhi::Swapchain::ResourceState;
                    sw_->transition_buffer_state(buf,
                        cur  == BufUse::Uav ? RS::UnorderedAccess : RS::ShaderResource,
                        want == BufUse::Uav ? RS::UnorderedAccess : RS::ShaderResource);
                    cur = want;
                } else if (want == BufUse::Uav) {
                    sw_->uav_barrier(buf);    // UAV -> UAV (WAW / RAW) flush
                }
            }

            sw_->bind_pipeline(cp->pipe.get());
            // GAP 3: upload the per-dispatch push block (the pass packed its
            // scalars — width/height/params — into pd.push_constants).
            if (!pd.push_constants.empty())
                sw_->set_push_constants(0, pd.push_constants.data(),
                                        static_cast<u32>(pd.push_constants.size()));
            // Bind accesses in order: Read -> SRV t0.., Write/RW -> UAV u0..
            // (matches the kernel's StructuredBuffer/RWStructuredBuffer order).
            u32 t = 0, u = 0;
            for (const auto& a : pd.accesses) {
                cardinal::rhi::Buffer* buf = (a.handle.id < buffers_.size())
                    ? buffers_[a.handle.id] : nullptr;
                if (a.mode == AccessMode::Read) { if (buf) sw_->bind_storage_buffer(t, buf);     ++t; }
                else                            { if (buf) sw_->bind_storage_buffer_uav(u, buf); ++u; }
            }
            sw_->dispatch(pd.dispatch_x, pd.dispatch_y, pd.dispatch_z);
            ev.real_dispatch = true;
            ++stats_.passes_dispatched_real;
        } else {
            // Recording-only: same semantics as NullBackend; GPU dispatch lights
            // up automatically once a pass carries a compilable kernel.
            ++stats_.passes_dispatched_stub;
        }
        events_.push_back(cardinal::move(ev));
    }
}

// =============================================================================
// ThreadedCpuBackend — parallel pass execution within each wave
// =============================================================================
cardinal::shared_ptr<ThreadedCpuBackend> ThreadedCpuBackend::create() {
    return cardinal::shared_ptr<ThreadedCpuBackend>(new ThreadedCpuBackend());
}

void ThreadedCpuBackend::reset() noexcept {
    storage_.clear();
    waves_.clear();
    total_us_ = 0.0f;
    serial_est_us_ = 0.0f;
}

float ThreadedCpuBackend::speedup() const noexcept {
    // When the work is too short to measure (sub-µs passes) both
    // accumulators round to 0 and the ratio is undefined; return 1.0
    // (= no speedup, no slowdown) instead of 0 so downstream knobs
    // can compare against >= 1 without special-casing tiny graphs.
    if (total_us_ <= 0.0f || serial_est_us_ <= 0.0f) return 1.0f;
    return serial_est_us_ / total_us_;
}

void ThreadedCpuBackend::execute(Graph& g) noexcept {
    reset();
    const auto& order = g.execution_order();
    const auto& wave_of = g.wave_of_pass();
    const u32   waves   = g.wave_count();
    if (order.empty() || waves == 0) return;

    // Allocate storage per resource (imported buffers seeded from the host
    // bytes — mirrors CpuBackend, so both CPU backends agree on inputs).
    storage_.resize(g.resource_count() + 1);
    for (u32 rid = 1; rid <= g.resource_count(); ++rid) {
        ResourceHandle bh{ rid, ResourceKind::Buffer };
        const BufferDesc& bd = g.buffer(bh);
        if (bd.size_bytes > 0) {
            storage_[rid].assign(bd.size_bytes, 0u);
            if (const void* src = g.buffer_imported(bh))
                cardinal::memcpy(storage_[rid].data(), src, bd.size_bytes);
        }
    }

    // Bucket passes by wave.
    cardinal::vector<cardinal::vector<u32>> by_wave(static_cast<usize>(waves));
    for (u32 pid : order) {
        const u32 w = wave_of[pid];
        by_wave[w].push_back(pid);
    }

    const auto t_start = std::chrono::high_resolution_clock::now();
    float serial_acc_us = 0.0f;

    for (u32 w = 0; w < waves; ++w) {
        const auto& wave = by_wave[w];
        if (wave.empty()) continue;
        const auto t_wave0 = std::chrono::high_resolution_clock::now();

        // Per-pass timing slot — indexed by position within the wave so
        // each thread writes its own cell with no contention.
        cardinal::vector<float> per_pass_us(wave.size(), 0.0f);

        // Run all passes in this wave concurrently via index-based
        // parallel_for. The graph guarantees passes within a wave share
        // no resources by construction (wave decomposition), so the
        // per-thread CpuExecutionContexts touch disjoint byte ranges.
        cardinal::async::parallel_for(0u, static_cast<u32>(wave.size()),
            [this, &g, &per_pass_us, &wave](u32 slot) {
                const u32 pid = wave[slot];
                const auto t0 = std::chrono::high_resolution_clock::now();
                const PassDesc& pd = g.pass(pid);
                if (pd.record) {
                    CpuExecutionContext ec(g, storage_, pd);
                    ec.dispatch(pd.dispatch_x, pd.dispatch_y, pd.dispatch_z);
                    pd.record(ec, pd.user_ctx);
                }
                const auto t1 = std::chrono::high_resolution_clock::now();
                per_pass_us[slot] = static_cast<float>(
                    std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
            });

        const auto t_wave1 = std::chrono::high_resolution_clock::now();
        WaveStats ws;
        ws.wave_id    = w;
        ws.pass_count = static_cast<u32>(wave.size());
        ws.duration_us = static_cast<float>(
            std::chrono::duration_cast<std::chrono::microseconds>(t_wave1 - t_wave0).count());
        waves_.push_back(ws);
        for (float v : per_pass_us) serial_acc_us += v;
    }
    const auto t_end = std::chrono::high_resolution_clock::now();
    total_us_      = static_cast<float>(
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count());
    serial_est_us_ = serial_acc_us;
}

cardinal::vector<u8> ThreadedCpuBackend::buffer_contents(ResourceHandle h) const {
    if (h.id == 0 || h.id >= storage_.size()) return {};
    cardinal::vector<u8> out;
    out.assign(storage_[h.id].begin(), storage_[h.id].end());
    return out;
}

// =============================================================================
// NullBackend
// =============================================================================
cardinal::shared_ptr<NullBackend> NullBackend::create() {
    return cardinal::shared_ptr<NullBackend>(new NullBackend());
}

void NullBackend::execute(Graph& g) noexcept {
    events_.clear();
    const auto& order = g.execution_order();
    events_.reserve(order.size());
    for (u32 pid : order) {
        const PassDesc& pd = g.pass(pid);
        PassEvent ev;
        ev.name = pd.name;
        ev.kind = pd.kind;
        ev.dispatch_x = pd.dispatch_x;
        ev.dispatch_y = pd.dispatch_y;
        ev.dispatch_z = pd.dispatch_z;
        events_.push_back(cardinal::move(ev));
    }
}

NullBackend::Stats NullBackend::stats() const noexcept {
    Stats s{};
    s.passes_executed = static_cast<u32>(events_.size());
    for (const auto& ev : events_) {
        switch (ev.kind) {
            case PassKind::Compute: ++s.compute_passes; break;
            case PassKind::Raster:  ++s.raster_passes;  break;
            case PassKind::Copy:    ++s.copy_passes;    break;
        }
    }
    return s;
}

}  // namespace cardinal::render::graph
