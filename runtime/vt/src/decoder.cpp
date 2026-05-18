#include <cardinal/vt/decoder.hpp>

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/cmath.hpp>
#include <cardinal/core/cstdio.hpp>

namespace cardinal::vt {

namespace {

// SplitMix32 — used for the procedural color hash.
inline u32 splitmix32(u32 x) noexcept {
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

// HSV → RGB, all in [0..1]. Returns {r, g, b} packed in u32 (alpha = 255).
inline u32 hsv_to_rgba(float h, float s, float v) noexcept {
    if (s <= 0.0f) {
        const u8 g = static_cast<u8>(v * 255.0f);
        return 0xFF000000u | (g << 16) | (g << 8) | g;
    }
    h = h - cardinal::floor(h);
    h *= 6.0f;
    const int i = static_cast<int>(h);
    const float f = h - i;
    const float p = v * (1.0f - s);
    const float q = v * (1.0f - s * f);
    const float t = v * (1.0f - s * (1.0f - f));
    float r = 0, g = 0, b = 0;
    switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
    }
    return 0xFF000000u
         | (static_cast<u8>(b * 255.0f) << 16)
         | (static_cast<u8>(g * 255.0f) << 8)
         |  static_cast<u8>(r * 255.0f);
}

}  // namespace

cardinal::vector<u8> ProceduralDecoder::decode(TileKey key) {
    cardinal::vector<u8> out(kTileBytesRGBA);
    if (!key.valid()) return out;

    // Per-tile hue based on (vt, mip, x, y).
    const u32 tile_seed = splitmix32(
        seed_ ^ static_cast<u32>(key.raw) ^ (static_cast<u32>(key.raw >> 32)));
    const float hue = static_cast<float>(tile_seed & 0xFFFFu) / 65535.0f;
    const float sat = 0.55f - static_cast<float>(key.mip()) * 0.04f;
    const float val = 0.95f;
    const u32   base_rgba = hsv_to_rgba(hue, cardinal::max(0.10f, sat), val);
    const u32   alt_rgba  = hsv_to_rgba(hue,
        cardinal::max(0.10f, sat * 0.5f), cardinal::min(1.0f, val + 0.05f));

    // Checker pattern + a subtle border so the user can see tile edges.
    auto* px = reinterpret_cast<u32*>(out.data());
    for (u32 y = 0; y < kTileEdge; ++y) {
        for (u32 x = 0; x < kTileEdge; ++x) {
            const bool border = (x < 2 || x >= kTileEdge - 2 ||
                                 y < 2 || y >= kTileEdge - 2);
            const bool cell   = ((x >> 4) ^ (y >> 4)) & 1;
            u32 c = cell ? alt_rgba : base_rgba;
            if (border) c = 0xFF202020u;
            px[y * kTileEdge + x] = c;
        }
    }
    return out;
}

cardinal::vector<u8> FileDecoder::decode(TileKey key) {
    if (!resolver_) return {};
    const cardinal::string path = resolver_(key);
    if (path.empty()) return {};

    FILE* f = cardinal::fopen(path.c_str(), "rb");
    if (f == nullptr) return {};

    cardinal::vector<u8> out(kTileBytesRGBA);
    const usize n = cardinal::fread(out.data(), 1, out.size(), f);
    cardinal::fclose(f);
    if (n != out.size()) {
        out.clear();   // short read or missing — surface as decode failure
    }
    return out;
}

}  // namespace cardinal::vt
