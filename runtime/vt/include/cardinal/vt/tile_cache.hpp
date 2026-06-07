#pragma once

// =============================================================================
// Cardinal — physical tile pool + LRU cache.
//
// The pool is a flat cardinal::vector<Slot> of fixed size — never grows once
// configured. Each slot holds:
//   - the bytes of one decoded tile (kTileBytesRGBA)
//   - the TileKey it represents (or kInvalidTileKey when free)
//   - last_use_ms — wall-clock-relative timestamp; LRU eviction picks the
//     entry with the oldest stamp
//   - content hash for dedup
//
// Lookups return the slot or kSlotEmpty. Dedup: when a tile is decoded, we
// compute its content hash; if another resident tile already has the same
// hash AND identical bytes (memcmp guard against hash collisions), the new
// TileKey just points at the existing slot — no new memory allocated.
//
// Eviction is amortized: we don't sort slots every frame; instead we keep a
// "high-water" timestamp and on eviction we scan for the oldest, skipping
// pinned (recently-used) slots.
// =============================================================================

#include <cardinal/vt/types.hpp>

#include <cardinal/core/std/atomic.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/core/std/thread.hpp>

namespace cardinal::vt {

struct CacheStats {
    u32 capacity{0};        // total slots in the pool
    u32 resident{0};        // slots currently holding tiles
    u32 free_slots{0};      // capacity - resident
    u64 hits{0};            // lookup found Resident
    u64 misses{0};          // lookup found NotResident or Pending
    u64 evictions{0};       // forced out by LRU
    u64 dedup_hits{0};      // new tile matched an existing one's bytes
    u64 inserts{0};         // total tile inserts attempted
};

class TileCache {
public:
    // dedup_enabled: when true, every insert hashes 64 KiB of tile bytes
    // and memcmps against any existing slot with the same hash. Useful
    // when many tiles share content (atlas-style streaming, repeated
    // mip-0 patterns). Disabled by default — the procedural decoder
    // produces unique tiles per (vt, mip, x, y), so the dedup work is
    // pure overhead AND it runs INSIDE the cache mutex which serializes
    // every parallel decode worker. Flip on only when you've measured a
    // real dedup hit rate.
    explicit TileCache(u32 slot_count, bool dedup_enabled = false) noexcept;
    ~TileCache();

    TileCache(const TileCache&) = delete;
    TileCache& operator=(const TileCache&) = delete;

    // ----- Producer side (decoder thread) ----------------------------------

    // Insert a freshly-decoded tile. Returns the physical slot it landed in.
    // If the same content (verified via hash + memcmp) is already resident,
    // this dedupes and bumps the existing slot's reference count.
    PhysicalSlot insert(TileKey key, const u8* bytes, usize len, u64 now_ms);

    // ----- Consumer side ---------------------------------------------------

    // Find the slot for `key` or kSlotEmpty if not resident. Bumps the
    // tile's last_use_ms so LRU keeps it around.
    PhysicalSlot lookup(TileKey key, u64 now_ms) noexcept;

    // Read-only peek (does NOT bump LRU). Used by stats / panel rendering.
    PhysicalSlot peek(TileKey key) const noexcept;

    // ----- Eviction --------------------------------------------------------

    // Free at least one slot — returns the freed count.
    u32 evict_lru(u32 max_to_free, u64 now_ms);

    // Resize the pool. Tiles past the new size are dropped.
    void resize(u32 new_slot_count);

    // Drop everything (used by tests + budget Critical pressure).
    void clear() noexcept;

    // ----- Inspection ------------------------------------------------------
    CacheStats stats() const noexcept;
    u32        slot_count() const noexcept { return static_cast<u32>(slots_.size()); }

    // Bytes addressable for slot `s` or nullptr if invalid / free.
    const u8* slot_bytes(PhysicalSlot s) const noexcept;

    // Returns the TileKey currently occupying slot `s`, or kInvalidTileKey.
    TileKey   slot_key(PhysicalSlot s) const noexcept;

    // Iterate every resident slot. Cheap; for stats / panel only.
    void for_each_resident(const cardinal::function<void(PhysicalSlot, TileKey, u64 last_use_ms)>& fn) const;

private:
    struct Slot {
        cardinal::vector<u8> bytes;       // size kTileBytesRGBA when resident
        TileKey         key{kInvalidTileKey};
        TileHash        hash{0};
        u64             last_use_ms{0};
        u32             ref_count{0};
    };

    PhysicalSlot pick_eviction_victim(u64 now_ms) const noexcept;
    // Pop one slot off the free stack (or kSlotEmpty if drained).
    PhysicalSlot take_free_slot() noexcept;
    void         release_to_free_stack(PhysicalSlot s) noexcept;

    mutable cardinal::mutex                       mtx_;
    cardinal::vector<Slot>                        slots_;
    // Key -> slot index for O(1) lookup. Sized at slot_count().
    cardinal::unordered_map<TileKey, PhysicalSlot> by_key_;
    // Hash -> slot index for dedup. May map to a slot whose hash matches
    // — caller verifies via memcmp. Only populated when dedup_enabled_.
    cardinal::unordered_map<TileHash, PhysicalSlot> by_hash_;
    // LIFO of currently-empty slot indices. Push on construction (every
    // slot starts free), eviction, clear; pop on insert. Replaces an
    // O(N) linear scan in find_free_slot — at 1024 slots that scan
    // dominated insert latency under load.
    cardinal::vector<PhysicalSlot>                free_slots_;
    bool                                     dedup_enabled_{false};

    cardinal::atomic<u64> hits_       {0};
    cardinal::atomic<u64> misses_     {0};
    cardinal::atomic<u64> evictions_  {0};
    cardinal::atomic<u64> dedup_hits_ {0};
    cardinal::atomic<u64> inserts_    {0};
};

}  // namespace cardinal::vt
