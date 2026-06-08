#pragma once

// =============================================================================
// cardinal::core::compress — hand-rolled, zero-dep PNG decoder.
//
// Built on the inflate keystone (IDAT is a zlib stream). Decodes the
// non-interlaced 8/16-bit grayscale / RGB / GA / RGBA subset — which covers
// heightmaps (16-bit grayscale) + ordinary colour textures. Palette (colour
// type 3), interlaced (Adam7), and sub-byte depths (1/2/4) are rejected
// cleanly. Chunk CRCs are NOT validated (optional per the PNG spec) so no
// CRC32 dependency is pulled in. Every read is bounds-checked (untrusted input).
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/containers.hpp>

namespace cardinal::core::compress {

// Decode a PNG into raw, un-filtered samples. On success `out` holds
// height * width * channels * (bit_depth/8) bytes, row-major top-to-bottom;
// 16-bit samples are kept BIG-ENDIAN (network order, as stored in the PNG —
// the caller reconstructs via (hi<<8)|lo). channels: 1 (gray) / 2 (gray+alpha)
// / 3 (RGB) / 4 (RGBA). bit_depth: 8 or 16. Returns false on any error or
// unsupported variant (palette / interlaced / 1-2-4-bit).
bool decode_png(const cardinal::u8* in, cardinal::usize in_n,
                cardinal::vector<cardinal::u8>& out,
                cardinal::u32& width, cardinal::u32& height,
                cardinal::u32& channels, cardinal::u32& bit_depth) noexcept;

}  // namespace cardinal::core::compress
