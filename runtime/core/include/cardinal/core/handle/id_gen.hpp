#pragma once

// =============================================================================
// Cardinal — monotonic id generator + pool with generation tracking.
//
// `IdGen<T>` — atomic counter. `next()` returns a strictly-increasing T.
//              Threadsafe. T must be an unsigned integer type.
//
// `SlotPool<Tag>` — index/generation pool that vends Handle<Tag> values.
//                   Slot reuse bumps the generation, so old handles to a
//                   freed slot compare unequal even when index is recycled.
//                   Single-threaded — wrap externally for cross-thread use.
// =============================================================================

#include <cardinal/core/handle/handle.hpp>
#include <cardinal/core/types.hpp>

#include <atomic>
#include <vector>

namespace cardinal::core {

template <typename T = u64>
class IdGen {
    static_assert(std::is_unsigned_v<T>, "IdGen: T must be unsigned");
public:
    explicit IdGen(T first = T{1}) noexcept : counter_(first) {}
    T next() noexcept { return counter_.fetch_add(T{1}, std::memory_order_relaxed); }
    T peek() const noexcept { return counter_.load(std::memory_order_relaxed); }
private:
    std::atomic<T> counter_;
};

// Advance a 32-bit slot generation, skipping the reserved sentinel 0 on
// wrap. Handle generation 0 means "invalid" (Handle::valid()), so a
// generation that ++'d straight through 0xFFFFFFFF -> 0 would make a
// live recycled slot vend an UNUSABLE handle (object unreferenceable),
// and the next free would wrap it to 1 — ABA-colliding with the slot's
// original gen-1 owner (silent aliasing, the exact corruption this type
// exists to prevent). Wrapping 0xFFFFFFFF -> 1 keeps every recycled
// handle valid and distinct from the immediately-prior owner. (Full ABA
// avoidance across 2^32-1 recycles of one slot would require a 64-bit
// generation — a Handle ABI change, deliberately out of scope; this
// closes the concrete gen-0 defect.) Single source of truth so the
// pool and any future generational scheme cannot drift.
constexpr u32 next_generation(u32 g) noexcept {
    return (g == 0xFFFFFFFFu) ? 1u : (g + 1u);
}

template <typename Tag>
class SlotPool {
public:
    using H = Handle<Tag>;

    H allocate() {
        if (!free_.empty()) {
            const u32 idx = free_.back(); free_.pop_back();
            // free() already advanced this slot's generation, so the
            // recycled handle is distinct from every prior owner's.
            return H{ idx, generations_[idx] };
        }
        const u32 idx = static_cast<u32>(generations_.size());
        generations_.push_back(1);                     // gen 0 reserved → invalid
        return H{ idx, 1 };
    }

    bool free(H h) {
        if (!alive(h)) return false;                   // also rejects double-free
        // Bump the generation NOW (not lazily on reuse): a freed handle
        // must be detectably dead immediately, and the index must not be
        // pushed onto the free list twice (that would later vend one
        // slot to two live handles — silent aliasing corruption). This
        // is the use-after-free guard this type exists to provide.
        // next_generation() skips the reserved 0 on 32-bit wrap so a
        // recycled slot never vends an invalid (gen-0) handle.
        generations_[h.index] = next_generation(generations_[h.index]);
        free_.push_back(h.index);
        return true;
    }

    bool alive(H h) const noexcept {
        return h.valid()
            && h.index < generations_.size()
            && generations_[h.index] == h.generation;
    }

    usize capacity() const noexcept { return generations_.size(); }
    usize live_count() const noexcept { return generations_.size() - free_.size(); }

    // Current generation of a slot index (0 if out of range). Lets a higher-
    // level container (SlotMap) reconstruct the live Handle for a slot it
    // knows is occupied — e.g. to yield handles during iteration.
    u32 generation_at(u32 idx) const noexcept {
        return idx < generations_.size() ? generations_[idx] : 0u;
    }

    void clear() noexcept { generations_.clear(); free_.clear(); }

private:
    std::vector<u32> generations_;
    std::vector<u32> free_;
};

}  // namespace cardinal::core
