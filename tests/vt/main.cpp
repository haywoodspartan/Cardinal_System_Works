// =============================================================================
// Cardinal — deterministic virtual-texturing core regression suite.
//
// TileKey is the universal 64-bit currency: vt_id(16)|mip(6)|y(21)|x(21).
// Every VT cache entry, request and page-table pointer is keyed by one;
// an aliasing regression samples the wrong texture. The bit-pack is
// constexpr so its EXACT layout is pinned at compile time (static_assert
// → drift fails the build) and mirrored at runtime. PageTable is the
// residency state machine — lookup + mark_pending/resident/evicted/
// failed with lazy per-mip allocation and resident/pending counters; its
// subtle rules (pending dedup, re-resident keeps the count, and the
// mark_failed-from-Resident quirk that does NOT decrement resident) are
// pinned exactly. std::hash<TileKey> of a zero-raw key is exactly the
// canonical SplitMix64(seed-0) vector — an exact cross-check. Pure,
// single-writer, deterministic. Exit 0 = all pass.
// =============================================================================

#include <cardinal/vt/page_table.hpp>
#include <cardinal/core/diag/log.hpp>

#include <functional>
#include <string>

namespace {

namespace vt = cardinal::vt;
using cardinal::u8;
using cardinal::u32;
using cardinal::u64;
using cardinal::usize;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("vttest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool streq(const char* a, const char* b) { return std::string(a) == b; }
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

// ---- COMPILE-TIME pins: constants + TileKey bit layout -------------
static_assert(vt::kTileEdge      == 128u,        "tile edge");
static_assert(vt::kTileBorder    == 2u,          "tile border");
static_assert(vt::kTileTexels    == 128u * 128u, "tile texels");
static_assert(vt::kTileBytesRGBA == 128u*128u*4u,"tile bytes");
static_assert(vt::kMaxMipLevels  == 22u,         "max mips");
static_assert(vt::kMaxVtId       == 65535u,      "max vt id");
static_assert(vt::kMaxAxis       == 2097151u,    "max axis");
static_assert(vt::kSlotEmpty     == static_cast<vt::PhysicalSlot>(-1), "slot empty");

static_assert(vt::TileKey().raw == static_cast<u64>(-1), "default = all ones");
static_assert(!vt::TileKey().valid(),                    "default invalid");
static_assert(vt::kInvalidTileKey.raw == static_cast<u64>(-1), "sentinel");
static_assert(vt::TileKey(0,0,0,0).raw == 0ull,          "zero key raw");
static_assert(vt::TileKey(0,0,0,0).valid(),              "zero key valid");

// Exact bit positions — the heart of the format.
static_assert(vt::TileKey(0,0,0,1).raw == 1ull,                 "x bit 0");
static_assert(vt::TileKey(0,0,1,0).raw == (1ull << 21),         "y bit 21");
static_assert(vt::TileKey(0,1,0,0).raw == (1ull << 42),         "mip bit 42");
static_assert(vt::TileKey(1,0,0,0).raw == (1ull << 48),         "vt bit 48");

// Round-trip incl. the maxima of every field (no cross-bleed).
static_assert(vt::TileKey(7,3,5,9).vt_id() == 7,  "rt vt");
static_assert(vt::TileKey(7,3,5,9).mip()   == 3u, "rt mip");
static_assert(vt::TileKey(7,3,5,9).y()     == 5u, "rt y");
static_assert(vt::TileKey(7,3,5,9).x()     == 9u, "rt x");
static_assert(vt::TileKey(65535,63,2097151,2097151).vt_id() == 65535,    "max vt");
static_assert(vt::TileKey(65535,63,2097151,2097151).mip()   == 63u,      "max mip");
static_assert(vt::TileKey(65535,63,2097151,2097151).y()     == 2097151u, "max y");
static_assert(vt::TileKey(65535,63,2097151,2097151).x()     == 2097151u, "max x");
static_assert(vt::TileKey(1,2,3,4) == vt::TileKey(1,2,3,4),  "eq");
static_assert(vt::TileKey(1,2,3,4) != vt::TileKey(1,2,3,5),  "neq");

// ---- TileKey runtime mirror + std::hash ---------------------------
void test_tilekey() {
    CHECK(vt::TileKey().raw == static_cast<u64>(-1));
    CHECK(!vt::TileKey().valid());
    CHECK(vt::kInvalidTileKey == vt::TileKey());
    CHECK(vt::TileKey(0,0,0,0).valid());
    CHECK(vt::TileKey(0,0,0,0).raw == 0ull);

    const vt::TileKey k(7,3,5,9);
    CHECK(k.vt_id() == 7 && k.mip() == 3u && k.y() == 5u && k.x() == 9u);
    // Field independence — one axis maxed leaves the others zero.
    CHECK(vt::TileKey(0,0,0,2097151u).x() == 2097151u);
    CHECK(vt::TileKey(0,0,0,2097151u).y() == 0u);
    CHECK(vt::TileKey(0,0,2097151u,0).y() == 2097151u);
    CHECK(vt::TileKey(0,0,2097151u,0).x() == 0u);
    CHECK(vt::TileKey(0,63u,0,0).mip() == 63u && vt::TileKey(0,63u,0,0).vt_id() == 0);
    CHECK(vt::TileKey(65535,0,0,0).vt_id() == 65535 && vt::TileKey(65535,0,0,0).mip() == 0u);
    CHECK(vt::TileKey(1,2,3,4) == vt::TileKey(1,2,3,4));
    CHECK(vt::TileKey(1,2,3,4) != vt::TileKey(1,2,3,5));

    // std::hash: a zero-raw key hashes to splitmix64(0x9e37..) — the
    // canonical SplitMix64(seed 0) first output (64-bit size_t).
    std::hash<vt::TileKey> H;
    CHECK(static_cast<u64>(H(vt::TileKey(0,0,0,0))) == 0xE220A8397B1DCDAFull);
    // Deterministic + injective (splitmix64 finalizer is a bijection,
    // so distinct raw ⇒ distinct hash, never just "probably").
    CHECK(H(k) == H(vt::TileKey(7,3,5,9)));
    CHECK(H(vt::TileKey(1,2,3,4)) != H(vt::TileKey(1,2,3,5)));
    CHECK(H(vt::TileKey()) != H(vt::TileKey(0,0,0,0)));
}

// ---- tile_status_name ---------------------------------------------
void test_status_name() {
    using S = vt::TileStatus;
    CHECK(streq(vt::tile_status_name(S::NotResident), "NotResident"));
    CHECK(streq(vt::tile_status_name(S::Pending),     "Pending"));
    CHECK(streq(vt::tile_status_name(S::Resident),    "Resident"));
    CHECK(streq(vt::tile_status_name(S::Failed),      "Failed"));
    CHECK(streq(vt::tile_status_name(static_cast<S>(99)), "?"));
    // TileLookup default state.
    vt::TileLookup tl{};
    CHECK(tl.status == S::NotResident && tl.slot == vt::kSlotEmpty);
}

// ---- hash_tile invariants -----------------------------------------
void test_hash_tile() {
    const u8 a[8] = { 1,2,3,4,5,6,7,8 };
    const u8 b[8] = { 1,2,3,4,5,6,7,9 };          // last byte differs
    const u8 a3[3] = { 1,2,3 };

    CHECK(vt::hash_tile(nullptr, 8) == 0ull);      // null → 0
    CHECK(vt::hash_tile(a, 0) == 0ull);            // empty → 0
    CHECK(vt::hash_tile(a, 8) == vt::hash_tile(a, 8));   // deterministic
    CHECK(vt::hash_tile(a, 8) != 0ull);
    CHECK(vt::hash_tile(a, 8) != vt::hash_tile(b, 8));   // content-sensitive
    CHECK(vt::hash_tile(a, 8) != vt::hash_tile(a, 3));   // length-sensitive
    CHECK(vt::hash_tile(a3, 3) == vt::hash_tile(a3, 3)); // tail path stable
    // Only `len` bytes are read: a3 == a's first 3 bytes ⇒ equal hash.
    CHECK(vt::hash_tile(a3, 3) == vt::hash_tile(a, 3));
}

// ---- PageTable geometry + ctor clamps -----------------------------
void test_geometry() {
    {
        vt::PageTableDesc d; d.width_tiles = 8; d.height_tiles = 4; d.mip_count = 4;
        vt::PageTable pt(d);
        CHECK(pt.mip_count() == 4u);
        CHECK(pt.width_tiles(0) == 8u  && pt.height_tiles(0) == 4u);
        CHECK(pt.width_tiles(1) == 4u  && pt.height_tiles(1) == 2u);
        CHECK(pt.width_tiles(2) == 2u  && pt.height_tiles(2) == 1u);
        CHECK(pt.width_tiles(3) == 1u  && pt.height_tiles(3) == 1u); // 4>>3=0→1
        CHECK(pt.width_tiles(4) == 0u  && pt.height_tiles(4) == 0u); // OOB mip
        CHECK(pt.total_tiles() == 8u*4u + 4u*2u + 2u*1u + 1u*1u);    // 43
        CHECK(pt.resident_count() == 0u && pt.pending_count() == 0u);
    }
    {   // Clamps: mip 0→1, width/height 0→1.
        vt::PageTableDesc d{};                       // all default 1s
        vt::PageTable pt(d);
        CHECK(pt.mip_count() == 1u);
        CHECK(pt.width_tiles(0) == 1u && pt.height_tiles(0) == 1u);
        CHECK(pt.total_tiles() == 1u);
    }
    {
        vt::PageTableDesc d; d.width_tiles = 0; d.height_tiles = 0; d.mip_count = 0;
        vt::PageTable pt(d);
        CHECK(pt.mip_count() == 1u);                 // 0 → 1
        CHECK(pt.width_tiles(0) == 1u && pt.height_tiles(0) == 1u);
    }
    {
        vt::PageTableDesc d; d.width_tiles = 4; d.height_tiles = 4; d.mip_count = 100;
        vt::PageTable pt(d);
        CHECK(pt.mip_count() == vt::kMaxMipLevels);  // 100 → 22
    }
}

// ---- lookup + mark state machine + counters -----------------------
void test_lookup_marks() {
    vt::PageTableDesc d; d.width_tiles = 4; d.height_tiles = 4; d.mip_count = 2;
    vt::PageTable pt(d);
    using S = vt::TileStatus;

    // Fresh: every in-range tile NotResident; OOB lookups safe.
    {
        vt::TileLookup l = pt.lookup(0,0,0);
        CHECK(l.status == S::NotResident && l.slot == vt::kSlotEmpty);
    }
    CHECK(pt.lookup(5,0,0).status == S::NotResident);   // OOB mip
    CHECK(pt.lookup(0,99,99).status == S::NotResident); // mip unallocated

    // The slot field is packed into 28 bits, so once a cell is WRITTEN
    // the 32-bit kSlotEmpty sentinel reads back masked. (Per the header,
    // slot is only meaningful when status == Resident — this pins the
    // documented truncation, not a usable value.)
    constexpr vt::PhysicalSlot kEmpty28 = vt::kSlotEmpty & 0x0FFFFFFFu;
    CHECK(kEmpty28 == 0x0FFFFFFFu);

    // mark_pending allocates the mip + counts once; dedup is a no-op.
    pt.mark_pending(0,1,2);
    CHECK(pt.lookup(0,1,2).status == S::Pending);
    CHECK(pt.lookup(0,1,2).slot == kEmpty28);
    CHECK(pt.pending_count() == 1u && pt.resident_count() == 0u);
    pt.mark_pending(0,1,2);                              // already Pending
    CHECK(pt.pending_count() == 1u);                     // not double-counted
    // Now the mip is allocated, the x/y range guard engages.
    CHECK(pt.lookup(0,99,99).status == S::NotResident);
    CHECK(pt.lookup(0,0,0).status == S::NotResident);    // sibling untouched

    // Pending → Resident: pending--, resident++.
    pt.mark_resident(0,1,2, /*slot*/7u);
    CHECK(pt.lookup(0,1,2).status == S::Resident);
    CHECK(pt.lookup(0,1,2).slot == 7u);
    CHECK(pt.pending_count() == 0u && pt.resident_count() == 1u);

    // Resident → Resident (new slot): slot updates, no double count.
    pt.mark_resident(0,1,2, /*slot*/9u);
    CHECK(pt.lookup(0,1,2).slot == 9u);
    CHECK(pt.resident_count() == 1u);

    // NotResident → Resident directly: resident++.
    pt.mark_resident(0,3,3, /*slot*/5u);
    CHECK(pt.resident_count() == 2u);
    CHECK(pt.lookup(0,3,3).slot == 5u);
    CHECK(pt.lookup(0,1,2).slot == 9u);                  // independent cell

    // Resident → evicted: resident--; second evict is a no-op.
    pt.mark_evicted(0,1,2);
    CHECK(pt.lookup(0,1,2).status == S::NotResident);
    CHECK(pt.lookup(0,1,2).slot == kEmpty28);            // written → masked
    CHECK(pt.resident_count() == 1u);
    pt.mark_evicted(0,1,2);                               // NotResident now
    CHECK(pt.resident_count() == 1u);

    // Pending → Failed: pending--; resident untouched.
    pt.mark_pending(0,0,0);
    CHECK(pt.pending_count() == 1u);
    pt.mark_failed(0,0,0);
    CHECK(pt.lookup(0,0,0).status == S::Failed);
    CHECK(pt.pending_count() == 0u && pt.resident_count() == 1u);

    // QUIRK: Resident → Failed does NOT decrement resident_count
    // (mark_failed only adjusts the pending counter). (0,3,3) was the
    // sole Resident tile (count 1); failing it leaves the count STALE at
    // 1 instead of dropping to 0.
    pt.mark_failed(0,3,3);
    CHECK(pt.lookup(0,3,3).status == S::Failed);
    CHECK(pt.resident_count() == 1u);                    // NOT decremented

    // mark_evicted / mark_failed on an UNALLOCATED mip are no-ops and
    // do not allocate it; mark_pending does allocate.
    pt.mark_evicted(1,0,0);
    CHECK(pt.lookup(1,0,0).status == S::NotResident);
    pt.mark_failed(1,0,0);
    CHECK(pt.lookup(1,0,0).status == S::NotResident);
    CHECK(pt.pending_count() == 0u);
    pt.mark_pending(1,0,0);                               // allocates mip 1
    CHECK(pt.lookup(1,0,0).status == S::Pending);
    CHECK(pt.pending_count() == 1u);

    // Out-of-range mip writes are silently dropped.
    pt.mark_pending(9,0,0);
    CHECK(pt.lookup(9,0,0).status == S::NotResident);
    CHECK(pt.pending_count() == 1u);                     // unchanged
}

// ---- mark_* must bounds-check (mip,y,x) like lookup() does ---------
// Regression: lookup() guards `x>=width_tiles(mip) || y>=height_tiles
// (mip)` but mark_pending/resident/evicted/failed did NOT. An
// out-of-range coord either ALIASED a different tile's status cell
// (entry_index_ = y*width+x lands on some other valid cell — silent
// deterministic corruption) or, for y>=height, indexed PAST the
// per-mip atomics vector (OOB) — and still bumped pending_/resident_
// count_. The header says "(mip,y,x) must be in range"; mark_* must
// enforce it too. No existing test exercises out-of-range mark_*.
void test_mark_out_of_range() {
    using S = vt::TileStatus;
    vt::PageTableDesc d; d.width_tiles = 4; d.height_tiles = 4;
    d.mip_count = 2;
    vt::PageTable pt(d);

    // x == width_tiles(0) (out of range; valid x in [0,4)). entry_index_
    // would be 0*4+4 == 4 == the cell of in-range tile (mip0,y1,x0):
    // pre-fix this Pending-marks (0,1,0) and bumps the counter.
    pt.mark_pending(0, /*y*/0, /*x*/4);
    CHECK(pt.pending_count() == 0u);                    // pre-fix: 1
    CHECK(pt.lookup(0, 1, 0).status == S::NotResident); // pre-fix: Pending

    // y >= height_tiles(0): entry_index_ = 8*4+0 = 32 >= 16 → OOB on the
    // size-16 atomics vector pre-fix. Post-fix: guarded no-op.
    pt.mark_resident(0, /*y*/8, /*x*/0, /*slot*/3u);
    CHECK(pt.resident_count() == 0u);                   // pre-fix: 1 (+OOB)
    CHECK(pt.lookup(0, 0, 0).status == S::NotResident);

    // mark_evicted / mark_failed out of range: clean no-ops too.
    pt.mark_evicted(0, /*y*/99, /*x*/0);
    pt.mark_failed (0, /*y*/0,  /*x*/77);
    CHECK(pt.pending_count() == 0u && pt.resident_count() == 0u);

    // The normal in-range path is preserved by the fix.
    pt.mark_pending(0, 1, 2);
    CHECK(pt.lookup(0, 1, 2).status == S::Pending);
    CHECK(pt.pending_count() == 1u);
    pt.mark_resident(0, 1, 2, /*slot*/7u);
    CHECK(pt.lookup(0, 1, 2).status == S::Resident);
    CHECK(pt.lookup(0, 1, 2).slot == 7u);
    CHECK(pt.pending_count() == 0u && pt.resident_count() == 1u);
}

}  // namespace

int main() {
    test_tilekey();
    test_status_name();
    test_hash_tile();
    test_geometry();
    test_lookup_marks();
    test_mark_out_of_range();

    if (g_fail == 0) {
        cardinal::log::infof("vttest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("vttest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
