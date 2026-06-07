#pragma once

// =============================================================================
// Cardinal — Virtual Texturing system.
//
// One process-wide System owns the shared physical TileCache and pumps the
// decoder pipeline. Each VirtualTexture owns its own PageTable and a
// per-VT Decoder.
//
// Lifecycle:
//   System::create({ pool_slots, decoder_workers, ... })
//     System::register_vt(vt) for each VT
//   per-frame:
//     vt->request(mip, y, x) for each visible tile (predicted by camera)
//     System::tick(now_ms)   — drains the request queue, fans decode jobs
//                              to cardinal::async, evicts under budget
//   vt->lookup(mip, y, x)    — read-only, returns TileLookup
// =============================================================================

#include <cardinal/vt/decoder.hpp>
#include <cardinal/vt/page_table.hpp>
#include <cardinal/vt/tile_cache.hpp>
#include <cardinal/vt/types.hpp>

#include <cardinal/core/std/atomic.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/core/std/thread.hpp>

namespace cardinal::vt {

class System;

// ---------------------------------------------------------------------------
// VirtualTexture — a single logical raster.
//
// Constructed via VirtualTextureDesc + a Decoder. After construction, the
// integrator must register it with a System. The VT keeps a non-owning
// pointer back to its System so request() can enqueue without going through
// a global lookup.
// ---------------------------------------------------------------------------
struct VirtualTextureDesc {
    VtId           id{0};
    u32            width_tiles {1};       // tiles along X at mip 0
    u32            height_tiles{1};
    u32            mip_count   {1};
    cardinal::shared_ptr<Decoder> decoder;
};

class VirtualTexture {
public:
    static cardinal::shared_ptr<VirtualTexture> create(const VirtualTextureDesc& desc);

    // Enqueue a tile request (no-op if already Pending or Resident).
    // Safe to call from any thread; the queue is lock-free MPSC.
    void request(u32 mip, u32 y, u32 x);

    // Single-tile lookup. Bumps the cache's LRU stamp on a hit.
    TileLookup lookup(u32 mip, u32 y, u32 x) const;

    // Read-only descriptor.
    const VirtualTextureDesc& desc() const noexcept { return desc_; }
    PageTable&       page_table()       noexcept { return *pt_; }
    const PageTable& page_table() const noexcept { return *pt_; }

    // Set by System::register_vt; resists null deref by silently dropping
    // request()s before registration.
    void attach(System* sys) noexcept { system_ = sys; }

private:
    VirtualTextureDesc        desc_{};
    cardinal::unique_ptr<PageTable> pt_;
    System*                    system_{nullptr};
};

// ---------------------------------------------------------------------------
// System — process-shared.
// ---------------------------------------------------------------------------
struct SystemDesc {
    u32  pool_slots{4096};         // physical tile capacity (4096 ≈ 256 MiB)
    u32  decode_concurrency{0};    // 0 = use cardinal::async pool
    u32  request_ring_size{8192};  // capacity of the lock-free request ring
    u32  prefetch_neighbors{1};    // 0 = none, 1 = 4-neighbours, 2 = +parent mip
    bool register_with_budget_broker{true};
};

struct SystemStats {
    CacheStats cache{};
    u64 requests_seen     {0};   // pushed via VirtualTexture::request
    u64 requests_processed{0};   // popped + decoded
    u64 requests_dropped  {0};   // ring overflow (rare)
    u64 prefetch_added    {0};   // synthesised from a primary request
    u32 vt_count          {0};
};

class System {
public:
    static cardinal::shared_ptr<System> create(const SystemDesc& desc = {});
    ~System();

    System(const System&) = delete;
    System& operator=(const System&) = delete;

    // Register a VT for tracking. Idempotent — re-registering with the same
    // id replaces the previous entry. The system holds a weak_ptr, so the
    // integrator owns the VT's lifetime.
    void register_vt  (const cardinal::shared_ptr<VirtualTexture>& vt);
    void unregister_vt(VtId id);
    cardinal::shared_ptr<VirtualTexture> get_vt(VtId id) const;

    // Push a tile into the decode queue. Producer side; lock-free.
    // Returns false on ring overflow (stat-tracked).
    bool enqueue_request(TileKey key);

    // Drain the request queue, fan decode jobs out to cardinal::async,
    // commit completed tiles into the cache + page table, and apply LRU
    // eviction if we're over the desired residency target.
    //
    // `now_ms` should monotonically increase (e.g. steady_clock-derived).
    // Cheap to call every frame.
    void tick(u64 now_ms);

    // Hard-clear cache (used on Critical memory pressure).
    void purge();

    SystemStats stats() const noexcept;
    TileCache&  cache() noexcept { return *cache_; }
    const TileCache& cache() const noexcept { return *cache_; }
    const SystemDesc& desc() const noexcept { return desc_; }

private:
    System() = default;
    bool initialize_(const SystemDesc& desc);

    // Per-tick worker: pop a key, decode (sync on the worker fiber),
    // insert, update page table.
    void process_request_(TileKey key, u64 now_ms);

    // Predictive prefetch — add neighbours / parent-mip of `key` to the
    // request queue.
    void prefetch_(TileKey key);

    SystemDesc                                desc_{};
    cardinal::unique_ptr<TileCache>                cache_;
    // VT registry. Slot index = VtId. Holds weak_ptrs so we don't keep VTs
    // alive past the integrator's release.
    mutable cardinal::mutex                         vts_mtx_;
    cardinal::vector<cardinal::weak_ptr<VirtualTexture>> vts_;

    // Lock-free MPSC ring of pending requests. Written by request(); drained
    // by tick(). Sized to desc_.request_ring_size.
    struct Ring {
        cardinal::vector<cardinal::atomic<u64>> slots;
        cardinal::atomic<u64>              head{0};   // producer position
        cardinal::atomic<u64>              tail{0};   // consumer position
    };
    cardinal::unique_ptr<Ring>                      ring_;

    cardinal::atomic<u64> stat_seen_     {0};
    cardinal::atomic<u64> stat_processed_{0};
    cardinal::atomic<u64> stat_dropped_  {0};
    cardinal::atomic<u64> stat_prefetch_ {0};

    // budget broker subscription id (set when register_with_budget_broker)
    u32 budget_sub_id_{0};
};

}  // namespace cardinal::vt
