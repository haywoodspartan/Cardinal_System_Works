#pragma once

// =============================================================================
// Cardinal — VT tile decoder interface.
//
// One Decoder per VirtualTexture (or one shared, doesn't matter). The system
// hands the decoder a TileKey and expects either:
//   - a fresh kTileBytesRGBA buffer (in-band data; the system copies into the
//     cache), OR
//   - empty vector indicating decode failed.
//
// The decoder is invoked on a worker thread (`cardinal::async`), so it must
// be thread-safe. Stateful decoders (file handle, mmap'd page cache, etc.)
// should guard their state internally.
//
// We ship one built-in decoder — ProceduralDecoder — that synthesises a
// noise / checker pattern based on the TileKey. Useful for tests +
// demonstrating the system without any disk I/O.
// =============================================================================

#include <cardinal/vt/types.hpp>

#include <cardinal/core/containers.hpp>
#include <cardinal/core/utility.hpp>

namespace cardinal::vt {

class Decoder {
public:
    virtual ~Decoder() = default;

    // Returns kTileBytesRGBA bytes on success, empty vector on failure.
    virtual cardinal::vector<u8> decode(TileKey key) = 0;
};

// ---------------------------------------------------------------------------
// ProceduralDecoder — paints each tile with a hue derived from (x, y, mip)
// plus a checkerboard / noise base, so a sample app can verify the page
// table is plumbing tiles correctly. No I/O.
// ---------------------------------------------------------------------------
class ProceduralDecoder final : public Decoder {
public:
    explicit ProceduralDecoder(u32 seed = 1337u) noexcept : seed_(seed) {}
    cardinal::vector<u8> decode(TileKey key) override;
private:
    u32 seed_;
};

// ---------------------------------------------------------------------------
// FileDecoder — reads decoded tile bytes straight off disk. Path layout is
// driven by a callback so the integrator picks (cache-on-disk, packfile,
// network) without modifying the decoder.
//
// Returns empty if the resolved file is missing or short.
// ---------------------------------------------------------------------------
class FileDecoder final : public Decoder {
public:
    using PathResolver = cardinal::function<cardinal::string(TileKey)>;
    explicit FileDecoder(PathResolver resolver) : resolver_(cardinal::move(resolver)) {}
    cardinal::vector<u8> decode(TileKey key) override;
private:
    PathResolver resolver_;
};

}  // namespace cardinal::vt
