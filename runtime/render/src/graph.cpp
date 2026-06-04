#include <cardinal/render/graph.hpp>

#include <cardinal/core/algorithm.hpp>      // cardinal::clamp
#include <cardinal/core/cstring.hpp>        // cardinal::strcmp / strlen
#include <cardinal/core/utility.hpp>        // cardinal::move

namespace cardinal::render::graph {

namespace {

// Compare two C strings; nullptr-safe.
inline bool streq(const char* a, const char* b) noexcept {
    if (a == b) return true;
    if (!a || !b) return false;
    return cardinal::strcmp(a, b) == 0;
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

    // Resource state: track last writer + readers-since-last-write per
    // resource (1-based ids).
    struct ResState {
        u32                       last_writer{0};
        cardinal::vector<u32>     readers_since_last_write;
    };
    cardinal::vector<ResState> rs(static_cast<usize>(n_res + 1));

    // Adjacency list: edges[from_pass] = passes that depend on it.
    cardinal::vector<cardinal::vector<u32>> edges(static_cast<usize>(n_passes + 1));
    cardinal::vector<u32>                   indeg(static_cast<usize>(n_passes + 1), 0u);

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
    cardinal::vector<u32> ready;
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
