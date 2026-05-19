#pragma once

// =============================================================================
// Cardinal — texture math helpers (Phase 6 baseline).
//
// Pure math for texture authoring + streaming. No GPU calls; the actual
// texture upload + sampling lives in the RHI when those primitives exist.
// What's in here is the bookkeeping that drives those calls correctly:
//
//   * Mip-chain math      — count, dimensions, byte size for any base size
//   * Anisotropic filter  — recommended max-aniso level per quality preset
//   * Block-compressed    — BC1/BC3/BC4/BC5/BC7 + ASTC byte-size for a level
//   * Streaming budget    — pick a target mip / quality given a VRAM budget
//
// Everything here is constexpr or noexcept-pure so it can run on any thread,
// at any phase of the frame, without locks. Optimisation flag of the day:
// each helper avoids divisions where possible — mip math is power-of-two
// stuff that compilers turn into shifts when you spell it carefully.
// =============================================================================

#include <cardinal/core/types.hpp>

namespace cardinal::render::tex {

// ---------------------------------------------------------------------------
// Format taxonomy — mirrors the subset we care about for tex math. The RHI
// has its own Format enum for runtime; this one is for offline / authoring
// math where we know what we're targeting.
// ---------------------------------------------------------------------------
enum class CompressedFormat : u32 {
    None = 0,
    BC1,        // RGB,  4 bpp,  4×4 block, 8  bytes/block
    BC3,        // RGBA, 8 bpp,  4×4 block, 16 bytes/block
    BC4,        // R,    4 bpp,  4×4 block, 8  bytes/block
    BC5,        // RG,   8 bpp,  4×4 block, 16 bytes/block
    BC6H,       // HDR,  8 bpp,  4×4 block, 16 bytes/block
    BC7,        // RGBA, 8 bpp,  4×4 block, 16 bytes/block (preferred)
    ASTC_4x4,   // 4×4   block, 8.00 bpp
    ASTC_6x6,   // 6×6   block, 3.56 bpp
    ASTC_8x8,   // 8×8   block, 2.00 bpp
};

// Bytes per 4×4 (or larger) block. For uncompressed, returns 0.
constexpr u32 block_bytes(CompressedFormat f) noexcept {
    switch (f) {
        case CompressedFormat::BC1:      return 8;
        case CompressedFormat::BC3:      return 16;
        case CompressedFormat::BC4:      return 8;
        case CompressedFormat::BC5:      return 16;
        case CompressedFormat::BC6H:     return 16;
        case CompressedFormat::BC7:      return 16;
        case CompressedFormat::ASTC_4x4: return 16;
        case CompressedFormat::ASTC_6x6: return 16;
        case CompressedFormat::ASTC_8x8: return 16;
        case CompressedFormat::None:     return 0;
    }
    return 0;
}

constexpr u32 block_dim_x(CompressedFormat f) noexcept {
    switch (f) {
        case CompressedFormat::ASTC_6x6: return 6;
        case CompressedFormat::ASTC_8x8: return 8;
        default:                          return 4;     // BC* and ASTC 4×4
    }
}
constexpr u32 block_dim_y(CompressedFormat f) noexcept {
    switch (f) {
        case CompressedFormat::ASTC_6x6: return 6;
        case CompressedFormat::ASTC_8x8: return 8;
        default:                          return 4;
    }
}

// ---------------------------------------------------------------------------
// Mip chain math — works in pixels, no allocation.
//
// mip_count_for_size(w, h)   — full chain down to 1×1
// mip_dim_at(d, level)        — d / 2^level, clamped to >=1 (matches the spec)
// mip_chain_bytes(...)        — total size of all mips for a 2D image
// ---------------------------------------------------------------------------

// Count of leading zero bits for the smaller dimension == log2(max). We
// compute it portably so the math stays constexpr.
constexpr u32 floor_log2_u32(u32 v) noexcept {
    if (v == 0) return 0;
    u32 r = 0;
    while (v > 1) { v >>= 1; ++r; }
    return r;
}

// Number of mips a 2D image of (w,h) supports — full chain.
constexpr u32 mip_count_for_size(u32 w, u32 h) noexcept {
    const u32 m = (w > h) ? w : h;
    return floor_log2_u32(m) + 1;
}

constexpr u32 mip_dim_at(u32 d, u32 level) noexcept {
    // A shift count >= the operand's bit width is undefined behaviour in
    // C++ (and a hard error in a constant expression). Any level that
    // big has long since collapsed the dimension to 0, so it clamps to
    // the documented >=1 minimum. Guarding here keeps callers (mip_bytes
    // / mip_chain_bytes / plan_for_budget) UB-free for arbitrary levels.
    if (level >= 32u) return 1u;
    const u32 v = d >> level;
    return v > 0u ? v : 1u;
}

// Bytes per mip level (uncompressed RGBA8 = 4bpp, BC7 = 1bpp etc.)
constexpr u64 mip_bytes(u32 w, u32 h, u32 level, CompressedFormat fmt,
                       u32 uncompressed_bytes_per_pixel = 4) noexcept {
    const u32 mw = mip_dim_at(w, level);
    const u32 mh = mip_dim_at(h, level);
    if (fmt == CompressedFormat::None) {
        return static_cast<u64>(mw) * mh * uncompressed_bytes_per_pixel;
    }
    const u32 bx = block_dim_x(fmt);
    const u32 by = block_dim_y(fmt);
    // Round up to a whole block — small mips share their last block.
    const u32 bw = (mw + bx - 1) / bx;
    const u32 bh = (mh + by - 1) / by;
    return static_cast<u64>(bw) * bh * block_bytes(fmt);
}

constexpr u64 mip_chain_bytes(u32 w, u32 h, CompressedFormat fmt,
                              u32 uncompressed_bytes_per_pixel = 4) noexcept {
    const u32 levels = mip_count_for_size(w, h);
    u64 total = 0;
    for (u32 m = 0; m < levels; ++m) {
        total += mip_bytes(w, h, m, fmt, uncompressed_bytes_per_pixel);
    }
    return total;
}

// ---------------------------------------------------------------------------
// Anisotropic filtering — quality presets map to a max-aniso level the
// sampler should request (1 = none, 16 = max-quality on every modern GPU).
//
// Anisotropy is FREE on modern desktop GPUs up to about 4x; higher values
// cost progressively more bandwidth in screen regions where the LOD is
// stretched along an axis. Worth exposing as a knob.
// ---------------------------------------------------------------------------
enum class AnisoQuality : u32 {
    Off       = 0,    // bilinear / trilinear only
    Low       = 1,    // 2x
    Medium    = 2,    // 4x   ← good default
    High      = 3,    // 8x
    Ultra     = 4,    // 16x
};

constexpr u32 aniso_max_for(AnisoQuality q) noexcept {
    switch (q) {
        case AnisoQuality::Off:    return 1;
        case AnisoQuality::Low:    return 2;
        case AnisoQuality::Medium: return 4;
        case AnisoQuality::High:   return 8;
        case AnisoQuality::Ultra:  return 16;
    }
    return 1;
}

// Max-LOD bias to apply when downgrading quality. Negative bias = sharper
// (use higher-detail mip), positive = softer (use lower-detail mip).
constexpr float lod_bias_for_quality(AnisoQuality q) noexcept {
    switch (q) {
        case AnisoQuality::Off:    return  1.0f;
        case AnisoQuality::Low:    return  0.5f;
        case AnisoQuality::Medium: return  0.0f;
        case AnisoQuality::High:   return -0.25f;
        case AnisoQuality::Ultra:  return -0.5f;
    }
    return 0.0f;
}

// ---------------------------------------------------------------------------
// Streaming budget — given a VRAM ceiling, pick the highest mip we can
// afford to keep resident across N textures of a given average size.
//
// Streaming math is otherwise a pile of magic numbers — keeping the formula
// in one place lets the editor surface a Knob without reaching for a tape
// measure.
// ---------------------------------------------------------------------------
struct StreamingPlan {
    u32 keep_mip_offset{0};   // 0 = keep full chain, 1 = drop mip 0, etc.
    u64 used_bytes{0};        // estimated VRAM consumption after the cut
};

// Iteratively drops the highest-resolution mip(s) until the chain fits.
constexpr StreamingPlan plan_for_budget(u32 w, u32 h, CompressedFormat fmt,
                                        u32 texture_count,
                                        u64 vram_budget_bytes,
                                        u32 uncompressed_bytes_per_pixel = 4) noexcept
{
    StreamingPlan p{};
    if (texture_count == 0) return p;

    // Start at full chain and drop top mips until we fit.
    u32 drop = 0;
    const u32 max_drop = mip_count_for_size(w, h);
    while (drop < max_drop) {
        u64 per_tex = 0;
        for (u32 m = drop; m < mip_count_for_size(w, h); ++m) {
            per_tex += mip_bytes(w, h, m, fmt, uncompressed_bytes_per_pixel);
        }
        const u64 total = per_tex * texture_count;
        if (total <= vram_budget_bytes) {
            p.keep_mip_offset = drop;
            p.used_bytes      = total;
            return p;
        }
        ++drop;
    }
    p.keep_mip_offset = max_drop;
    p.used_bytes      = 0;
    return p;
}

}  // namespace cardinal::render::tex
