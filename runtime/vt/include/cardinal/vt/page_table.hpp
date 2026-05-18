#pragma once

// =============================================================================
// Cardinal — per-VT page table.
//
// One PageTable per VirtualTexture. Maps (mip, y, x) -> { status, slot }.
//
// Internally a single-level flat array sized to the VT's tile dimensions at
// each mip. Per-mip storage is allocated lazily — high mips are tiny (1x1,
// 2x2, ...) so allocating up front would still be cheap, but lazy alloc
// keeps memory tight when only a region of the VT is being sampled.
//
// The page table is the sampling-side input: a renderer (or CPU sample call)
// asks lookup(uv, lod) and gets back a TileLookup. Misses enqueue a request
// onto System's queue.
//
// Updates are single-writer (the System tick) so the published values can be
// stored as relaxed atomics — the renderer reads them with no locking.
// =============================================================================

#include <cardinal/vt/types.hpp>

#include <cardinal/core/atomic.hpp>
#include <cardinal/core/containers.hpp>

namespace cardinal::vt {

struct PageTableDesc {
    u32 width_tiles {1};   // # tiles along X at mip 0
    u32 height_tiles{1};   // # tiles along Y at mip 0
    u32 mip_count   {1};   // # mip levels (1..kMaxMipLevels)
};

class PageTable {
public:
    explicit PageTable(const PageTableDesc& desc) noexcept;

    PageTable(const PageTable&) = delete;
    PageTable& operator=(const PageTable&) = delete;

    // ----- Read side (renderer) --------------------------------------------

    // Look up a single tile coordinate. (mip, y, x) must be in range.
    TileLookup lookup(u32 mip, u32 y, u32 x) const noexcept;

    // ----- Write side (System tick + decoder) ------------------------------

    void mark_pending (u32 mip, u32 y, u32 x) noexcept;
    void mark_resident(u32 mip, u32 y, u32 x, PhysicalSlot slot) noexcept;
    void mark_evicted (u32 mip, u32 y, u32 x) noexcept;
    void mark_failed  (u32 mip, u32 y, u32 x) noexcept;

    // ----- Geometry --------------------------------------------------------

    u32 mip_count   () const noexcept { return desc_.mip_count;   }
    u32 width_tiles (u32 mip) const noexcept;
    u32 height_tiles(u32 mip) const noexcept;
    u32 total_tiles () const noexcept;
    u32 resident_count() const noexcept { return resident_count_.load(cardinal::memory_order_relaxed); }
    u32 pending_count () const noexcept { return pending_count_.load(cardinal::memory_order_relaxed);  }

private:
    // Pack {status, slot} into one u32: high 4 bits status, low 28 bits slot.
    static constexpr u32 pack_(TileStatus s, PhysicalSlot slot) noexcept {
        return (static_cast<u32>(s) << 28) | (slot & 0x0FFFFFFFu);
    }
    static constexpr TileStatus unpack_status_(u32 v) noexcept {
        return static_cast<TileStatus>(v >> 28);
    }
    static constexpr PhysicalSlot unpack_slot_(u32 v) noexcept {
        return v & 0x0FFFFFFFu;
    }

    u32 entry_index_(u32 mip, u32 y, u32 x) const noexcept;
    void ensure_mip_allocated_(u32 mip);

    PageTableDesc                              desc_{};
    // One entries vector per mip — built lazily on first write to that mip.
    cardinal::vector<cardinal::vector<cardinal::atomic<u32>>> entries_per_mip_;
    cardinal::atomic<u32>                           resident_count_{0};
    cardinal::atomic<u32>                           pending_count_{0};
};

}  // namespace cardinal::vt
