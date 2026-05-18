#pragma once

// =============================================================================
// Cardinal — Virtual Texturing primitives.
//
// One Virtual Texture (VT) is a logical raster that can be enormous (up to
// 2^21 = 2M tiles per side, ~256K x 256K texels at 128px tiles) without
// allocating that storage up front. Only the *visible* tiles at the right
// LOD are streamed into a process-shared physical pool.
//
// Numbers worth memorising:
//   - Tile size           : 128 x 128 texels (TileEdge below)
//   - Border              : 2 px on each side, sampled into the next tile
//                           so anisotropic samples don't cross tile seams
//   - Bytes per tile      : 128*128*4 = 64 KiB (RGBA8 baseline; BC7 is 16 KiB)
//   - Max tiles per VT    : 2^21 per axis * 2^21 per axis * up to 22 mips
//   - Pool default        : 4096 physical slots ≈ 256 MiB at 64 KiB/tile
//   - TileKey size        : 64 bits — vt_id(16) | mip(6) | y(21) | x(21)
//
// TileKey is the universal currency: every cache entry, request, page-table
// pointer, and dedup hash is keyed by one. 64 bits keeps it lock-free-friendly
// (atomic load/store / CAS on 64-bit platforms).
// =============================================================================

#include <cardinal/core/types.hpp>

#include <functional>

namespace cardinal::vt {

// ---------------------------------------------------------------------------
// Compile-time tunables. Don't change these without auditing the bit packing
// in TileKey + the BorderInner / TileTexels math.
// ---------------------------------------------------------------------------
inline constexpr u32 kTileEdge      = 128;     // tile side length in texels
inline constexpr u32 kTileBorder    = 2;       // border per side (each tile)
inline constexpr u32 kTileTexels    = kTileEdge * kTileEdge;
inline constexpr u32 kTileBytesRGBA = kTileTexels * 4;   // 64 KiB

inline constexpr u32 kMaxMipLevels  = 22;      // 0..21 inclusive (matches 21 bit y/x)
inline constexpr u32 kMaxVtId       = (1u << 16) - 1;
inline constexpr u32 kMaxAxis       = (1u << 21) - 1;    // tiles per side per mip

using VtId = u16;

// ---------------------------------------------------------------------------
// TileKey — single u64 packing (vt_id, mip, y, x).
//
// Layout (MSB → LSB):
//   bits 63..48 : vt_id  (16)
//   bits 47..42 : mip    ( 6)
//   bits 41..21 : y      (21)
//   bits 20..00 : x      (21)
//
// Comparable, hashable, atomic-loadable. Sentinel kInvalidTileKey is all-ones.
// ---------------------------------------------------------------------------
struct TileKey {
    u64 raw{static_cast<u64>(-1)};

    constexpr TileKey() noexcept = default;

    constexpr TileKey(VtId vt, u32 mip, u32 y, u32 x) noexcept
        : raw((static_cast<u64>(vt)  << 48) |
              (static_cast<u64>(mip) << 42) |
              (static_cast<u64>(y)   << 21) |
              (static_cast<u64>(x))) {}

    constexpr VtId vt_id() const noexcept { return static_cast<VtId>((raw >> 48) & 0xFFFFu); }
    constexpr u32  mip()   const noexcept { return static_cast<u32>((raw >> 42) & 0x3Fu); }
    constexpr u32  y()     const noexcept { return static_cast<u32>((raw >> 21) & 0x1FFFFFu); }
    constexpr u32  x()     const noexcept { return static_cast<u32>( raw        & 0x1FFFFFu); }

    constexpr bool valid() const noexcept { return raw != static_cast<u64>(-1); }

    constexpr bool operator==(const TileKey& o) const noexcept { return raw == o.raw; }
    constexpr bool operator!=(const TileKey& o) const noexcept { return raw != o.raw; }
};

inline constexpr TileKey kInvalidTileKey{};

}  // namespace cardinal::vt

// std::hash specialisation so TileKey works in unordered containers.
namespace std {
template <> struct hash<cardinal::vt::TileKey> {
    size_t operator()(const cardinal::vt::TileKey& k) const noexcept {
        // Splittable hash — same constants as cardinal::core::hash uses.
        cardinal::u64 z = k.raw + 0x9e3779b97f4a7c15ull;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        z =  z ^ (z >> 31);
        return static_cast<size_t>(z);
    }
};
}  // namespace std

namespace cardinal::vt {

// ---------------------------------------------------------------------------
// PhysicalSlot — index into the shared physical pool. Sentinel = no-resident.
// ---------------------------------------------------------------------------
using PhysicalSlot = u32;
inline constexpr PhysicalSlot kSlotEmpty = static_cast<PhysicalSlot>(-1);

// ---------------------------------------------------------------------------
// Tile content hash — used for cross-VT dedup. SplitMix64 of the decoded
// tile bytes; collisions at 64 bits are vanishingly unlikely for billions
// of tiles, and a collision only causes a missed dedup (not corruption).
// ---------------------------------------------------------------------------
using TileHash = u64;

TileHash hash_tile(const u8* bytes, usize len) noexcept;

// ---------------------------------------------------------------------------
// Lookup result — what VirtualTexture::lookup() returns. Slot is meaningful
// only when status == Resident. Status surfaces every transition so a
// shader / debug overlay can render misses distinctly.
// ---------------------------------------------------------------------------
enum class TileStatus : u8 {
    NotResident = 0,    // never requested, or evicted
    Pending     = 1,    // request enqueued, decode in flight
    Resident    = 2,    // bytes are in the physical pool, slot is valid
    Failed      = 3,    // decoder reported an error (returned empty)
};

const char* tile_status_name(TileStatus s) noexcept;

struct TileLookup {
    TileStatus    status{TileStatus::NotResident};
    PhysicalSlot  slot{kSlotEmpty};
};

}  // namespace cardinal::vt
