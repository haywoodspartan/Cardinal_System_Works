// =============================================================================
// cardinal::core::compress — PNG decoder (see png.hpp).
//
// Chunk walk (IHDR / IDAT* / IEND) -> concatenate IDAT -> inflate_zlib ->
// reverse the per-scanline filters (None/Sub/Up/Average/Paeth). Non-interlaced
// 8/16-bit gray/RGB/GA/RGBA only. CRCs skipped (optional per spec).
// =============================================================================

#include <cardinal/core/compress/png.hpp>
#include <cardinal/core/compress/inflate.hpp>

namespace cardinal::core::compress {

namespace {

inline cardinal::u32 rd_be32(const cardinal::u8* p) noexcept {
    return (static_cast<cardinal::u32>(p[0]) << 24) |
           (static_cast<cardinal::u32>(p[1]) << 16) |
           (static_cast<cardinal::u32>(p[2]) <<  8) |
            static_cast<cardinal::u32>(p[3]);
}

// PNG Paeth predictor (RFC 2083 §6.6): pick a/b/c nearest to the linear
// estimate p = a + b - c. a=left, b=above, c=above-left.
inline cardinal::u8 paeth(int a, int b, int c) noexcept {
    const int p  = a + b - c;
    const int pa = p > a ? p - a : a - p;
    const int pb = p > b ? p - b : b - p;
    const int pc = p > c ? p - c : c - p;
    if (pa <= pb && pa <= pc) return static_cast<cardinal::u8>(a);
    if (pb <= pc)             return static_cast<cardinal::u8>(b);
    return static_cast<cardinal::u8>(c);
}

}  // namespace

bool decode_png(const cardinal::u8* in, cardinal::usize in_n,
                cardinal::vector<cardinal::u8>& out,
                cardinal::u32& width, cardinal::u32& height,
                cardinal::u32& channels, cardinal::u32& bit_depth) noexcept {
    static const cardinal::u8 kSig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (in_n < 8) return false;
    for (int i = 0; i < 8; ++i) if (in[i] != kSig[i]) return false;

    cardinal::u32 w = 0, h = 0;
    cardinal::u8  depth = 0, color = 0, interlace = 0;
    bool seen_ihdr = false;
    cardinal::vector<cardinal::u8> idat;

    cardinal::usize p = 8;
    while (p + 8 <= in_n) {                          // length(4) + type(4)
        const cardinal::u32 len = rd_be32(in + p);
        const cardinal::u8* type = in + p + 4;
        const cardinal::usize data_off = p + 8;
        if (data_off + static_cast<cardinal::usize>(len) + 4 > in_n) return false;  // data + crc
        const cardinal::u8* data = in + data_off;

        if (type[0]=='I'&&type[1]=='H'&&type[2]=='D'&&type[3]=='R') {
            if (len != 13) return false;
            w = rd_be32(data); h = rd_be32(data + 4);
            depth = data[8]; color = data[9];
            if (data[10] != 0 || data[11] != 0) return false;  // compression/filter method
            interlace = data[12];
            seen_ihdr = true;
        } else if (type[0]=='I'&&type[1]=='D'&&type[2]=='A'&&type[3]=='T') {
            idat.insert(idat.end(), data, data + len);
        } else if (type[0]=='I'&&type[1]=='E'&&type[2]=='N'&&type[3]=='D') {
            break;
        }
        p = data_off + static_cast<cardinal::usize>(len) + 4;   // skip data + crc
    }

    if (!seen_ihdr || w == 0 || h == 0) return false;
    if (w > 65535u || h > 65535u) return false;     // practical bound (untrusted input)
    if (interlace != 0)            return false;     // Adam7 unsupported
    if (depth != 8 && depth != 16) return false;     // sub-byte unsupported

    cardinal::u32 ch;
    switch (color) {
        case 0: ch = 1; break;   // grayscale
        case 2: ch = 3; break;   // RGB
        case 4: ch = 2; break;   // gray + alpha
        case 6: ch = 4; break;   // RGBA
        default: return false;   // palette (3) unsupported
    }

    const cardinal::u32   bpp      = ch * (depth == 16 ? 2u : 1u);  // bytes per pixel
    const cardinal::usize stride   = static_cast<cardinal::usize>(w) * bpp;
    const cardinal::usize filtered = (stride + 1) * h;             // +1 filter byte per row
    if (idat.empty()) return false;

    cardinal::vector<cardinal::u8> raw(filtered);
    if (inflate_zlib(idat.data(), idat.size(), raw.data(), filtered) != filtered) return false;

    out.assign(stride * h, 0);
    for (cardinal::u32 y = 0; y < h; ++y) {
        const cardinal::u8  ft   = raw[(stride + 1) * y];
        const cardinal::u8* fr   = &raw[(stride + 1) * y + 1];        // filtered row
        cardinal::u8*       cur  = &out[stride * y];
        const cardinal::u8* prev = (y > 0) ? &out[stride * (y - 1)] : nullptr;
        for (cardinal::usize x = 0; x < stride; ++x) {
            const int a = (x >= bpp)            ? cur[x - bpp]  : 0;  // left (reconstructed)
            const int b = prev                  ? prev[x]       : 0;  // above
            const int c = (prev && x >= bpp)    ? prev[x - bpp] : 0;  // above-left
            int v = fr[x];
            switch (ft) {
                case 0: break;                       // None
                case 1: v += a;             break;   // Sub
                case 2: v += b;             break;   // Up
                case 3: v += (a + b) / 2;   break;   // Average
                case 4: v += paeth(a, b, c);break;   // Paeth
                default: return false;               // unknown filter type
            }
            cur[x] = static_cast<cardinal::u8>(v & 0xFF);
        }
    }

    width = w; height = h; channels = ch; bit_depth = depth;
    return true;
}

}  // namespace cardinal::core::compress
