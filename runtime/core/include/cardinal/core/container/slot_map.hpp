#pragma once

// =============================================================================
// cardinal::core::SlotMap<T, Tag> — value storage keyed by a generational
// Handle (the safe resource-handle container).
//
// SlotPool<Tag> (handle/id_gen.hpp) already vends Handle<Tag> values with
// generation tracking + slot reuse, and SparseSet keys by a caller-supplied
// integer. SlotMap is the missing combination: insert() returns a Handle, and
// get(handle) returns the value ONLY while that exact handle is live — a stale
// handle to a freed-then-recycled slot (same index, bumped generation) returns
// nullptr instead of silently aliasing the new occupant. This is the pattern
// resource managers want (textures, meshes, entities, GPU objects): stable
// opaque handles, O(1) insert/get/erase, free-slot reuse, use-after-free
// detection.
//
// vs the other containers: DenseMap = hashed key→value; SparseSet = external
// dense-int key, no generation safety; SlotMap = it OWNS + vends the keys and
// detects stale ones.
//
// T must be default-constructible + move/copy-assignable (slots are pre-sized).
// Type-safety: pass a distinct empty Tag per logical handle kind so a
// MeshHandle can't be used on a TextureSlotMap.
//
// FOUNDATION RULE: lives in cardinal::core. Exposed as cardinal::slot_map.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/containers.hpp>      // cardinal::vector
#include <cardinal/core/std/utility.hpp>         // cardinal::move
#include <cardinal/core/handle/handle.hpp>   // Handle<Tag>
#include <cardinal/core/handle/id_gen.hpp>   // SlotPool<Tag>

#include <utility>     // std::forward

namespace cardinal::core {

struct DefaultSlotMapTag {};

template <class T, class Tag = DefaultSlotMapTag>
class SlotMap {
public:
    using handle_type = Handle<Tag>;
    using value_type  = T;
    using size_type   = cardinal::usize;

    // ---- capacity -----------------------------------------------------
    [[nodiscard]] bool empty() const noexcept { return pool_.live_count() == 0; }
    size_type          size()  const noexcept { return pool_.live_count(); }

    void clear() noexcept {
        pool_.clear();
        values_.clear();
        occupied_.clear();
    }

    // ---- insertion ----------------------------------------------------
    template <class... Args>
    handle_type emplace(Args&&... args) {
        const handle_type h = pool_.allocate();
        if (h.index >= values_.size()) {
            values_.resize(h.index + 1u);
            occupied_.resize(h.index + 1u, 0u);
        }
        values_[h.index]   = T(std::forward<Args>(args)...);
        occupied_[h.index] = 1u;
        return h;
    }
    handle_type insert(const T& v) { return emplace(v); }
    handle_type insert(T&& v)      { return emplace(cardinal::move(v)); }

    // ---- lookup (stale-handle-safe) -----------------------------------
    [[nodiscard]] bool contains(handle_type h) const noexcept { return pool_.alive(h); }
    T* get(handle_type h) noexcept {
        return pool_.alive(h) ? &values_[h.index] : nullptr;
    }
    const T* get(handle_type h) const noexcept {
        return pool_.alive(h) ? &values_[h.index] : nullptr;
    }

    // ---- removal ------------------------------------------------------
    // Frees the slot (bumping its generation, so `h` and every other copy of
    // it become stale). Returns false for an already-dead handle (no double
    // free). The slot's index is reused by a future insert.
    bool erase(handle_type h) {
        if (!pool_.alive(h)) return false;
        occupied_[h.index] = 0u;
        values_[h.index]   = T{};          // release the value
        pool_.free(h);
        return true;
    }

    // ---- iteration over live entries ----------------------------------
    template <class Fn> void for_each(Fn&& fn) {            // fn(handle_type, T&)
        for (cardinal::u32 i = 0; i < occupied_.size(); ++i)
            if (occupied_[i]) fn(handle_type{ i, pool_.generation_at(i) }, values_[i]);
    }
    template <class Fn> void for_each(Fn&& fn) const {      // fn(handle_type, const T&)
        for (cardinal::u32 i = 0; i < occupied_.size(); ++i)
            if (occupied_[i]) fn(handle_type{ i, pool_.generation_at(i) }, values_[i]);
    }

private:
    SlotPool<Tag>             pool_;
    cardinal::vector<T>       values_;
    cardinal::vector<cardinal::u8> occupied_;
};

}  // namespace cardinal::core

namespace cardinal {
template <class T, class Tag = core::DefaultSlotMapTag>
using slot_map = core::SlotMap<T, Tag>;
}  // namespace cardinal
