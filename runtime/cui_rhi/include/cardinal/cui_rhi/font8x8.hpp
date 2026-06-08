#pragma once

// =============================================================================
// Cardinal UI — embedded 8x8 bitmap font (rendered as quads).
//
// A compact, dependency-free uppercase ASCII font so the native renderer can
// draw text without a font-atlas texture (the public RHI has no texture-upload
// path yet). Each glyph is 8 rows of 8 bits; bit 0 (0x01) is the LEFTMOST
// column. Lowercase maps to uppercase; unknown glyphs render as a space.
// A real proportional font / glyph atlas can replace this once texture upload
// lands.
// =============================================================================

#include <cardinal/core/types.hpp>

namespace cardinal::cui_rhi {

// Pointer to 8 bytes (rows top->bottom, bit0 = leftmost column).
const u8* glyph8x8(char c) noexcept;

}  // namespace cardinal::cui_rhi
