#include <cardinal/vt/page_table.hpp>

#include <cardinal/core/utility.hpp>

namespace cardinal::vt {

PageTable::PageTable(const PageTableDesc& desc) noexcept : desc_(desc) {
    if (desc_.mip_count == 0) desc_.mip_count = 1;
    if (desc_.mip_count > kMaxMipLevels) desc_.mip_count = kMaxMipLevels;
    if (desc_.width_tiles  == 0) desc_.width_tiles  = 1;
    if (desc_.height_tiles == 0) desc_.height_tiles = 1;
    entries_per_mip_.resize(desc_.mip_count);
}

u32 PageTable::width_tiles(u32 mip) const noexcept {
    if (mip >= desc_.mip_count) return 0;
    const u32 w = desc_.width_tiles >> mip;
    return w == 0 ? 1u : w;
}

u32 PageTable::height_tiles(u32 mip) const noexcept {
    if (mip >= desc_.mip_count) return 0;
    const u32 h = desc_.height_tiles >> mip;
    return h == 0 ? 1u : h;
}

u32 PageTable::total_tiles() const noexcept {
    u32 total = 0;
    for (u32 m = 0; m < desc_.mip_count; ++m) total += width_tiles(m) * height_tiles(m);
    return total;
}

u32 PageTable::entry_index_(u32 mip, u32 y, u32 x) const noexcept {
    return y * width_tiles(mip) + x;
}

void PageTable::ensure_mip_allocated_(u32 mip) {
    if (mip >= entries_per_mip_.size()) return;
    auto& v = entries_per_mip_[mip];
    if (!v.empty()) return;
    const u32 count = width_tiles(mip) * height_tiles(mip);
    // cardinal::atomic isn't movable; resize() on a vector of atomics needs the
    // initial-size constructor, not later reallocations. We size in one
    // shot here.
    cardinal::vector<cardinal::atomic<u32>> fresh(count);
    for (auto& a : fresh) a.store(pack_(TileStatus::NotResident, kSlotEmpty),
                                  cardinal::memory_order_relaxed);
    v = cardinal::move(fresh);
}

TileLookup PageTable::lookup(u32 mip, u32 y, u32 x) const noexcept {
    TileLookup r{};
    if (mip >= entries_per_mip_.size()) return r;
    const auto& v = entries_per_mip_[mip];
    if (v.empty()) return r;   // never written → NotResident, slot empty
    if (x >= width_tiles(mip) || y >= height_tiles(mip)) return r;
    const u32 packed = v[entry_index_(mip, y, x)].load(cardinal::memory_order_acquire);
    r.status = unpack_status_(packed);
    r.slot   = unpack_slot_(packed);
    return r;
}

void PageTable::mark_pending(u32 mip, u32 y, u32 x) noexcept {
    if (mip >= entries_per_mip_.size()) return;
    // Bounds-check (x,y) like lookup() does. Without this an out-of-range
    // coord makes entry_index_ = y*width+x either alias a DIFFERENT
    // tile's cell (silent status corruption) or, for y>=height, index
    // PAST the per-mip atomics vector (OOB) — and still bump the counter.
    if (x >= width_tiles(mip) || y >= height_tiles(mip)) return;
    if (entries_per_mip_[mip].empty()) ensure_mip_allocated_(mip);
    if (entries_per_mip_[mip].empty()) return;
    auto& cell = entries_per_mip_[mip][entry_index_(mip, y, x)];
    const u32 prev = cell.load(cardinal::memory_order_acquire);
    const TileStatus prev_status = unpack_status_(prev);
    if (prev_status == TileStatus::Pending || prev_status == TileStatus::Resident) return;
    cell.store(pack_(TileStatus::Pending, kSlotEmpty), cardinal::memory_order_release);
    pending_count_.fetch_add(1, cardinal::memory_order_relaxed);
}

void PageTable::mark_resident(u32 mip, u32 y, u32 x, PhysicalSlot slot) noexcept {
    if (mip >= entries_per_mip_.size()) return;
    if (x >= width_tiles(mip) || y >= height_tiles(mip)) return;  // see mark_pending
    if (entries_per_mip_[mip].empty()) ensure_mip_allocated_(mip);
    if (entries_per_mip_[mip].empty()) return;
    auto& cell = entries_per_mip_[mip][entry_index_(mip, y, x)];
    const u32 prev = cell.load(cardinal::memory_order_acquire);
    const TileStatus prev_status = unpack_status_(prev);
    cell.store(pack_(TileStatus::Resident, slot), cardinal::memory_order_release);
    if (prev_status == TileStatus::Pending) {
        pending_count_.fetch_sub(1, cardinal::memory_order_relaxed);
    }
    if (prev_status != TileStatus::Resident) {
        resident_count_.fetch_add(1, cardinal::memory_order_relaxed);
    }
}

void PageTable::mark_evicted(u32 mip, u32 y, u32 x) noexcept {
    if (mip >= entries_per_mip_.size()) return;
    if (x >= width_tiles(mip) || y >= height_tiles(mip)) return;  // see mark_pending
    if (entries_per_mip_[mip].empty()) return;
    auto& cell = entries_per_mip_[mip][entry_index_(mip, y, x)];
    const u32 prev = cell.load(cardinal::memory_order_acquire);
    const TileStatus prev_status = unpack_status_(prev);
    cell.store(pack_(TileStatus::NotResident, kSlotEmpty), cardinal::memory_order_release);
    if (prev_status == TileStatus::Resident) {
        resident_count_.fetch_sub(1, cardinal::memory_order_relaxed);
    }
    if (prev_status == TileStatus::Pending) {
        pending_count_.fetch_sub(1, cardinal::memory_order_relaxed);
    }
}

void PageTable::mark_failed(u32 mip, u32 y, u32 x) noexcept {
    if (mip >= entries_per_mip_.size()) return;
    if (x >= width_tiles(mip) || y >= height_tiles(mip)) return;  // see mark_pending
    if (entries_per_mip_[mip].empty()) return;
    auto& cell = entries_per_mip_[mip][entry_index_(mip, y, x)];
    const u32 prev = cell.load(cardinal::memory_order_acquire);
    const TileStatus prev_status = unpack_status_(prev);
    cell.store(pack_(TileStatus::Failed, kSlotEmpty), cardinal::memory_order_release);
    if (prev_status == TileStatus::Pending) {
        pending_count_.fetch_sub(1, cardinal::memory_order_relaxed);
    }
}

}  // namespace cardinal::vt
