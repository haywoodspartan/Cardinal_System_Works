#pragma once

// =============================================================================
// cardinal::core::compress — hand-rolled, zero-dep DEFLATE inflate.
//
// The engine had NO reachable inflate (the PNG cooker stubs IDAT, and the only
// libdeflate is unlinkable NTC-SDK-internal source). This is a first-party
// tinfl-style decoder for RFC 1951 (raw DEFLATE) + RFC 1950 (zlib wrapper),
// std-only per the FOUNDATION RULE. It is the keystone that unblocks FBX
// (Encoding==1 compressed property arrays), USDZ (DEFLATE-compressed zip
// entries), and a real PNG decoder (IDAT) — all of which know their exact
// uncompressed size up front, which is the mode this decoder targets.
//
// Trust boundary: importers feed untrusted bytes here, so every read is
// bounds-checked — malformed input yields 0 (failure), never an OOB access.
// =============================================================================

#include <cardinal/core/types.hpp>

namespace cardinal::core::compress {

// Raw DEFLATE (RFC 1951). Decompress `in_n` bytes at `in` into the caller's
// buffer `out` of capacity `out_cap`. Returns the number of bytes written on
// success, or 0 on any error (bad stream, distance-too-far, output overflow).
// `out_cap` is typically the known exact uncompressed size.
cardinal::usize inflate_raw(const cardinal::u8* in, cardinal::usize in_n,
                            cardinal::u8* out, cardinal::usize out_cap) noexcept;

// zlib stream (RFC 1950): 2-byte CMF/FLG header + raw DEFLATE + 4-byte adler32
// trailer. Validates the header (CM==8, FCHECK, no preset dictionary); the
// adler32 trailer is not verified. Same return contract as inflate_raw.
cardinal::usize inflate_zlib(const cardinal::u8* in, cardinal::usize in_n,
                             cardinal::u8* out, cardinal::usize out_cap) noexcept;

}  // namespace cardinal::core::compress
