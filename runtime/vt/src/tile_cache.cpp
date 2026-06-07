#include <cardinal/vt/tile_cache.hpp>

#include <cardinal/core/std/algorithm.hpp>
#include <cardinal/core/std/cstring.hpp>
#include <cardinal/core/std/thread.hpp>

namespace cardinal::vt {

TileCache::TileCache(u32 slot_count, bool dedup_enabled) noexcept
    : dedup_enabled_(dedup_enabled)
{
    slots_.resize(slot_count);
    // Pre-size every slot's backing buffer once so insert doesn't need
    // to grow it (assign was triggering a realloc on every first-time
    // insert; under load that's hundreds of allocs per frame plus the
    // matching frees on eviction).
    for (auto& s : slots_) s.bytes.resize(kTileBytesRGBA);

    by_key_.reserve(slot_count * 2);
    if (dedup_enabled_) by_hash_.reserve(slot_count * 2);

    // Populate the free-slot stack with every index (highest first so the
    // first inserts land in a consistent order).
    free_slots_.reserve(slot_count);
    for (u32 i = slot_count; i-- > 0; ) free_slots_.push_back(i);
}

TileCache::~TileCache() = default;

PhysicalSlot TileCache::take_free_slot() noexcept {
    // O(1) pop from the free-slot stack. Caller holds mtx_.
    if (free_slots_.empty()) return kSlotEmpty;
    PhysicalSlot s = free_slots_.back();
    free_slots_.pop_back();
    return s;
}

void TileCache::release_to_free_stack(PhysicalSlot s) noexcept {
    free_slots_.push_back(s);
}

PhysicalSlot TileCache::pick_eviction_victim(u64 now_ms) const noexcept {
    PhysicalSlot best = kSlotEmpty;
    u64 oldest = static_cast<u64>(-1);
    for (u32 i = 0; i < slots_.size(); ++i) {
        const Slot& s = slots_[i];
        if (!s.key.valid()) continue;
        // Ref-counted slots are pinned (held by an in-flight render). Skip
        // them so we don't yank GPU memory the renderer is reading from.
        if (s.ref_count > 0) continue;
        if (s.last_use_ms < oldest) {
            oldest = s.last_use_ms;
            best   = i;
        }
    }
    (void)now_ms;
    return best;
}

PhysicalSlot TileCache::insert(TileKey key, const u8* bytes, usize len, u64 now_ms) {
    if (!key.valid() || bytes == nullptr || len == 0) return kSlotEmpty;
    inserts_.fetch_add(1, cardinal::memory_order_relaxed);

    // Hash only when dedup is enabled. The 64 KiB hash was almost pure
    // overhead with the procedural decoder (every tile unique).
    const TileHash h = dedup_enabled_ ? hash_tile(bytes, len) : 0u;

    // ----- Critical section: pick slot under the lock --------------------
    PhysicalSlot dest;
    {
        cardinal::lock_guard<cardinal::mutex> lg(mtx_);

        // Already resident under this key? Just bump LRU.
        if (auto it = by_key_.find(key); it != by_key_.end()) {
            slots_[it->second].last_use_ms = now_ms;
            return it->second;
        }

        // Dedup — same content already in cache? Only when enabled.
        if (dedup_enabled_) {
            if (auto it = by_hash_.find(h); it != by_hash_.end()) {
                Slot& other = slots_[it->second];
                // memcmp under the lock is unfortunate but the dedup-rare
                // case we're optimising for is "lots of unique content"
                // where this branch is never taken anyway.
                if (cardinal::memcmp(other.bytes.data(), bytes, len) == 0) {
                    by_key_[key]      = it->second;
                    other.last_use_ms = now_ms;
                    dedup_hits_.fetch_add(1, cardinal::memory_order_relaxed);
                    return it->second;
                }
            }
        }

        // Need a slot. Try free first; otherwise evict the oldest unpinned.
        dest = take_free_slot();
        if (dest == kSlotEmpty) {
            dest = pick_eviction_victim(now_ms);
            if (dest == kSlotEmpty) {
                // Pool fully pinned — refuse insert. Caller logs.
                return kSlotEmpty;
            }
            // Evict.
            Slot& victim = slots_[dest];
            if (dedup_enabled_) by_hash_.erase(victim.hash);
            for (auto it = by_key_.begin(); it != by_key_.end(); ) {
                if (it->second == dest) it = by_key_.erase(it); else ++it;
            }
            evictions_.fetch_add(1, cardinal::memory_order_relaxed);
        }

        // Reserve the slot (publish key) BEFORE we drop the lock. Other
        // workers won't reach for this slot now — its key.valid() is true
        // and pick_eviction_victim won't touch it (ref_count check kicks
        // in via the bump below).
        Slot& s = slots_[dest];
        s.key         = key;
        s.hash        = h;
        s.last_use_ms = now_ms;
        s.ref_count   = 1;          // hold pinned during the copy
        by_key_[key]  = dest;
        if (dedup_enabled_) by_hash_[h] = dest;
    }
    // ----- Lock released --------------------------------------------------

    // 64 KiB memcpy WITHOUT holding the cache mutex. Other workers can
    // continue inserting/looking up in parallel during this copy.
    Slot& s = slots_[dest];
    if (s.bytes.size() != len) s.bytes.resize(len);
    cardinal::memcpy(s.bytes.data(), bytes, len);

    // Drop our pin so the slot becomes evictable again.
    {
        cardinal::lock_guard<cardinal::mutex> lg(mtx_);
        if (s.ref_count > 0) --s.ref_count;
    }
    return dest;
}

PhysicalSlot TileCache::lookup(TileKey key, u64 now_ms) noexcept {
    cardinal::lock_guard<cardinal::mutex> lg(mtx_);
    auto it = by_key_.find(key);
    if (it == by_key_.end()) {
        misses_.fetch_add(1, cardinal::memory_order_relaxed);
        return kSlotEmpty;
    }
    slots_[it->second].last_use_ms = now_ms;
    hits_.fetch_add(1, cardinal::memory_order_relaxed);
    return it->second;
}

PhysicalSlot TileCache::peek(TileKey key) const noexcept {
    cardinal::lock_guard<cardinal::mutex> lg(mtx_);
    auto it = by_key_.find(key);
    return (it == by_key_.end()) ? kSlotEmpty : it->second;
}

u32 TileCache::evict_lru(u32 max_to_free, u64 now_ms) {
    cardinal::lock_guard<cardinal::mutex> lg(mtx_);
    u32 freed = 0;
    while (freed < max_to_free) {
        PhysicalSlot v = pick_eviction_victim(now_ms);
        if (v == kSlotEmpty) break;
        Slot& victim = slots_[v];
        if (dedup_enabled_) by_hash_.erase(victim.hash);
        for (auto it = by_key_.begin(); it != by_key_.end(); ) {
            if (it->second == v) it = by_key_.erase(it); else ++it;
        }
        // Keep `bytes` capacity so the next insert into this slot doesn't
        // re-allocate (we just clobber via memcpy in insert()).
        victim.key  = kInvalidTileKey;
        victim.hash = 0;
        victim.last_use_ms = 0;
        victim.ref_count   = 0;
        release_to_free_stack(v);
        ++freed;
        evictions_.fetch_add(1, cardinal::memory_order_relaxed);
    }
    return freed;
}

void TileCache::resize(u32 new_slot_count) {
    cardinal::lock_guard<cardinal::mutex> lg(mtx_);
    if (new_slot_count == slots_.size()) return;
    if (new_slot_count > slots_.size()) {
        const u32 old_size = static_cast<u32>(slots_.size());
        slots_.resize(new_slot_count);
        for (u32 i = old_size; i < new_slot_count; ++i) {
            slots_[i].bytes.resize(kTileBytesRGBA);
            free_slots_.push_back(i);
        }
        return;
    }
    // Shrink — evict every slot at index >= new_slot_count, and drop any
    // free-stack entries past the new size.
    free_slots_.erase(
        cardinal::remove_if(free_slots_.begin(), free_slots_.end(),
            [&](PhysicalSlot s){ return s >= new_slot_count; }),
        free_slots_.end());
    for (u32 i = new_slot_count; i < slots_.size(); ++i) {
        Slot& victim = slots_[i];
        if (!victim.key.valid()) continue;
        if (dedup_enabled_) by_hash_.erase(victim.hash);
        for (auto it = by_key_.begin(); it != by_key_.end(); ) {
            if (it->second == i) it = by_key_.erase(it); else ++it;
        }
        evictions_.fetch_add(1, cardinal::memory_order_relaxed);
    }
    slots_.resize(new_slot_count);
}

void TileCache::clear() noexcept {
    cardinal::lock_guard<cardinal::mutex> lg(mtx_);
    for (auto& s : slots_) {
        // Keep bytes capacity (see evict_lru rationale).
        s.key         = kInvalidTileKey;
        s.hash        = 0;
        s.last_use_ms = 0;
        s.ref_count   = 0;
    }
    by_key_.clear();
    by_hash_.clear();
    free_slots_.clear();
    free_slots_.reserve(slots_.size());
    for (u32 i = static_cast<u32>(slots_.size()); i-- > 0; ) {
        free_slots_.push_back(i);
    }
}

CacheStats TileCache::stats() const noexcept {
    CacheStats s{};
    s.capacity   = static_cast<u32>(slots_.size());
    {
        cardinal::lock_guard<cardinal::mutex> lg(mtx_);
        for (const auto& slot : slots_) if (slot.key.valid()) ++s.resident;
    }
    s.free_slots = s.capacity - s.resident;
    s.hits       = hits_       .load(cardinal::memory_order_relaxed);
    s.misses     = misses_     .load(cardinal::memory_order_relaxed);
    s.evictions  = evictions_  .load(cardinal::memory_order_relaxed);
    s.dedup_hits = dedup_hits_ .load(cardinal::memory_order_relaxed);
    s.inserts    = inserts_    .load(cardinal::memory_order_relaxed);
    return s;
}

const u8* TileCache::slot_bytes(PhysicalSlot s) const noexcept {
    cardinal::lock_guard<cardinal::mutex> lg(mtx_);
    if (s >= slots_.size()) return nullptr;
    if (!slots_[s].key.valid()) return nullptr;
    return slots_[s].bytes.data();
}

TileKey TileCache::slot_key(PhysicalSlot s) const noexcept {
    cardinal::lock_guard<cardinal::mutex> lg(mtx_);
    if (s >= slots_.size()) return kInvalidTileKey;
    return slots_[s].key;
}

void TileCache::for_each_resident(
    const cardinal::function<void(PhysicalSlot, TileKey, u64)>& fn) const
{
    if (!fn) return;
    cardinal::lock_guard<cardinal::mutex> lg(mtx_);
    for (u32 i = 0; i < slots_.size(); ++i) {
        const Slot& s = slots_[i];
        if (s.key.valid()) fn(i, s.key, s.last_use_ms);
    }
}

}  // namespace cardinal::vt
