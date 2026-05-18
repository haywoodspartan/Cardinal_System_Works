#include <cardinal/vt/types.hpp>

namespace cardinal::vt {

const char* tile_status_name(TileStatus s) noexcept {
    switch (s) {
        case TileStatus::NotResident: return "NotResident";
        case TileStatus::Pending:     return "Pending";
        case TileStatus::Resident:    return "Resident";
        case TileStatus::Failed:      return "Failed";
    }
    return "?";
}

// SplitMix64 over the tile bytes — one round per 8 bytes. Faster than xxhash
// for small fixed buffers and the collision rate at 64 bits is fine for our
// use (a hash collision causes a redundant insert, not corruption — we still
// memcmp before deduping).
TileHash hash_tile(const u8* bytes, usize len) noexcept {
    if (bytes == nullptr || len == 0) return 0;

    u64 h = 0x9e3779b97f4a7c15ull ^ static_cast<u64>(len);
    const usize stride = sizeof(u64);
    const usize tail   = len % stride;
    const usize body   = len - tail;

    for (usize i = 0; i < body; i += stride) {
        u64 word = 0;
        // unaligned-safe load
        for (usize k = 0; k < stride; ++k) {
            word |= static_cast<u64>(bytes[i + k]) << (k * 8);
        }
        u64 z = (h ^ word) + 0x9e3779b97f4a7c15ull;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        h = z ^ (z >> 31);
    }
    if (tail > 0) {
        u64 word = 0;
        for (usize k = 0; k < tail; ++k) {
            word |= static_cast<u64>(bytes[body + k]) << (k * 8);
        }
        u64 z = (h ^ word) + 0x9e3779b97f4a7c15ull;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        h = z ^ (z >> 31);
    }
    return h;
}

}  // namespace cardinal::vt
