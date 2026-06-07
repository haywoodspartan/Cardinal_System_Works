// =============================================================================
// Cardinal — heightmap decoder: image -> neutral HeightField for terrain.
//
// Same hand-rolled, zero-dep ethos as the OBJ / glTF / Megascans backends.
// Decodes a heightmap image to a CPU float grid (HeightField) that
// scene/terrain turns into a mesh — kept in cardinal::import (core+scene only)
// so the import lib links no rhi.
//
// NO-inflate formats land here (DEFLATE doesn't exist in the engine yet):
//   * RAW16  (.r16 / .raw) — headerless little-endian u16; the World Machine /
//     Gaea export. Square dimension inferred from the byte count, or supplied.
//   * BMP    (.bmp) — uncompressed BI_RGB, 8 / 24 / 32-bit (bottom-up or top).
//   * TGA    (.tga) — uncompressed true-color (type 2) / grayscale (type 3).
// 8-bit sources use the red / luminance channel. Heights normalise to [0,1]
// by the format's full range (u16/65535, u8/255) so absolute elevation is
// preserved; vertical scale is applied later at mesh build. PNG (needs
// DEFLATE) and EXR (float) land once cardinal::core::compress::inflate exists.
// =============================================================================

#include <cardinal/import/import.hpp>
#include <cardinal/core/diag/log.hpp>

#include <cardinal/core/std/algorithm.hpp>   // cardinal::min / max
#include <cardinal/core/std/cctype.hpp>      // cardinal::tolower
#include <cardinal/core/std/cmath.hpp>       // cardinal::sqrt
#include <cardinal/core/std/fstream.hpp>     // cardinal::ifstream / ios
#include <cardinal/core/std/sstream.hpp>     // cardinal::ostringstream

namespace cardinal::import {

namespace {

cardinal::string to_lower(cardinal::string s) {
    for (char& c : s) c = static_cast<char>(cardinal::tolower(
        static_cast<unsigned char>(c)));
    return s;
}
cardinal::string ext_of(const cardinal::string& p) {
    const auto dot = p.find_last_of('.');
    if (dot == cardinal::string::npos) return {};
    const auto sl = p.find_last_of("/\\");
    if (sl != cardinal::string::npos && dot < sl) return {};
    return to_lower(p.substr(dot));
}

cardinal::vector<u8> read_all(const cardinal::string& path) {
    cardinal::ifstream f(path, cardinal::ios::binary | cardinal::ios::ate);
    if (!f) return {};
    const cardinal::streamsize n = f.tellg();
    if (n <= 0) return {};                          // -1 = unseekable, 0 = empty
    f.seekg(0);
    cardinal::vector<u8> v(static_cast<cardinal::size_t>(n));
    f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

inline u16 rd_u16(const u8* p) { return static_cast<u16>(p[0]) |
                                        static_cast<u16>(static_cast<u16>(p[1]) << 8); }
inline u32 rd_u32(const u8* p) {
    return static_cast<u32>(p[0]) | (static_cast<u32>(p[1]) << 8) |
           (static_cast<u32>(p[2]) << 16) | (static_cast<u32>(p[3]) << 24);
}
inline i32 rd_i32(const u8* p) { return static_cast<i32>(rd_u32(p)); }

void finalize(HeightField& hf, float rmin, float rmax) {
    hf.min = rmin;
    hf.max = rmax;
    hf.ok  = true;
}

// ---- RAW16: headerless little-endian u16 -------------------------------
HeightField decode_raw16(const cardinal::vector<u8>& b, u32 raw_dim) {
    HeightField hf; hf.source_format = "raw16";
    const cardinal::size_t count = b.size() / 2;
    if (count == 0) { hf.diagnostics = "raw16: empty / not a u16 blob"; return hf; }
    u32 w = 0, h = 0;
    if (raw_dim > 0) {
        w = raw_dim;
        h = static_cast<u32>(count / raw_dim);
    } else {
        const u32 side = static_cast<u32>(
            cardinal::sqrt(static_cast<double>(count)) + 0.5);
        if (side == 0 ||
            static_cast<cardinal::size_t>(side) * side != count) {
            hf.diagnostics = "raw16: non-square byte count; pass raw_dim";
            return hf;
        }
        w = h = side;
    }
    const cardinal::size_t need = static_cast<cardinal::size_t>(w) * h;
    if (need == 0 || need > count) {
        hf.diagnostics = "raw16: dimension does not fit the data";
        return hf;
    }
    hf.width = w; hf.height = h;
    hf.heights.resize(need);
    float rmin = 1e30f, rmax = -1e30f;
    for (cardinal::size_t i = 0; i < need; ++i) {
        const float v = static_cast<float>(rd_u16(b.data() + i * 2));
        rmin = cardinal::min(rmin, v); rmax = cardinal::max(rmax, v);
        hf.heights[i] = v / 65535.0f;
    }
    finalize(hf, rmin, rmax);
    return hf;
}

// ---- BMP: uncompressed BI_RGB 8/24/32-bit ------------------------------
HeightField decode_bmp(const cardinal::vector<u8>& b) {
    HeightField hf; hf.source_format = "bmp";
    if (b.size() < 54 || b[0] != 'B' || b[1] != 'M') {
        hf.diagnostics = "bmp: missing BM header"; return hf;
    }
    const u32 dataOff     = rd_u32(&b[10]);
    const u32 hdrSize     = rd_u32(&b[14]);
    const i32 w           = rd_i32(&b[18]);
    const i32 hRaw        = rd_i32(&b[22]);
    const u16 bpp         = rd_u16(&b[28]);
    const u32 compression = rd_u32(&b[30]);
    if (compression != 0) { hf.diagnostics = "bmp: only uncompressed BI_RGB"; return hf; }
    if (w <= 0 || hRaw == 0) { hf.diagnostics = "bmp: bad dimensions"; return hf; }
    if (bpp != 8 && bpp != 24 && bpp != 32) { hf.diagnostics = "bmp: bpp must be 8/24/32"; return hf; }
    const bool topDown = (hRaw < 0);
    const u32  W = static_cast<u32>(w);
    const u32  H = static_cast<u32>(topDown ? -hRaw : hRaw);
    const u32  bytespp  = bpp / 8;
    const u32  rowBytes = ((W * bytespp + 3u) / 4u) * 4u;   // 4-byte aligned rows
    if (static_cast<cardinal::size_t>(dataOff) +
        static_cast<cardinal::size_t>(rowBytes) * H > b.size()) {
        hf.diagnostics = "bmp: truncated pixel data"; return hf;
    }
    // 8-bit BMP indexes a BGRA palette at 14 + hdrSize; read its R safely.
    auto pal_r = [&](u8 idx) -> float {
        const cardinal::size_t off =
            static_cast<cardinal::size_t>(14) + hdrSize +
            static_cast<cardinal::size_t>(idx) * 4 + 2;
        return (off < dataOff && off < b.size())
                   ? static_cast<float>(b[off]) : static_cast<float>(idx);
    };
    hf.width = W; hf.height = H;
    hf.heights.resize(static_cast<cardinal::size_t>(W) * H);
    float rmin = 1e30f, rmax = -1e30f;
    for (u32 y = 0; y < H; ++y) {
        const u32  srcRow = topDown ? y : (H - 1 - y);     // BMP is bottom-up
        const u8*  row = b.data() + dataOff +
                         static_cast<cardinal::size_t>(srcRow) * rowBytes;
        for (u32 x = 0; x < W; ++x) {
            float v;
            if (bpp == 8) v = pal_r(row[x]);
            else          v = static_cast<float>(row[static_cast<cardinal::size_t>(x) * bytespp + 2]); // R of BGR(A)
            rmin = cardinal::min(rmin, v); rmax = cardinal::max(rmax, v);
            hf.heights[static_cast<cardinal::size_t>(y) * W + x] = v / 255.0f;
        }
    }
    finalize(hf, rmin, rmax);
    return hf;
}

// ---- TGA: uncompressed true-color (2) / grayscale (3) ------------------
HeightField decode_tga(const cardinal::vector<u8>& b) {
    HeightField hf; hf.source_format = "tga";
    if (b.size() < 18) { hf.diagnostics = "tga: header too small"; return hf; }
    const u8  idLen    = b[0];
    const u8  cmapType = b[1];
    const u8  imgType  = b[2];
    const u16 W        = rd_u16(&b[12]);
    const u16 H        = rd_u16(&b[14]);
    const u8  depth    = b[16];
    const u8  desc     = b[17];
    if (cmapType != 0) { hf.diagnostics = "tga: colour-mapped not supported"; return hf; }
    if (imgType != 2 && imgType != 3) { hf.diagnostics = "tga: only uncompressed RGB(2)/gray(3)"; return hf; }
    if (W == 0 || H == 0) { hf.diagnostics = "tga: bad dimensions"; return hf; }
    if (imgType == 3 && depth != 8) { hf.diagnostics = "tga: grayscale must be 8-bit"; return hf; }
    if (imgType == 2 && depth != 24 && depth != 32) { hf.diagnostics = "tga: RGB must be 24/32-bit"; return hf; }
    const u32 bytespp = depth / 8;
    const cardinal::size_t dataOff = static_cast<cardinal::size_t>(18) + idLen;
    if (dataOff + static_cast<cardinal::size_t>(W) * H * bytespp > b.size()) {
        hf.diagnostics = "tga: truncated pixel data"; return hf;
    }
    const bool topDown = (desc & 0x20u) != 0;       // bit 5: top-left origin
    hf.width = W; hf.height = H;
    hf.heights.resize(static_cast<cardinal::size_t>(W) * H);
    float rmin = 1e30f, rmax = -1e30f;
    for (u32 y = 0; y < H; ++y) {
        const u32 srcRow = topDown ? y : (H - 1 - y);
        const u8* row = b.data() + dataOff +
                        static_cast<cardinal::size_t>(srcRow) * W * bytespp;
        for (u32 x = 0; x < W; ++x) {
            const u8* px = row + static_cast<cardinal::size_t>(x) * bytespp;
            const float v = (bytespp == 1) ? static_cast<float>(px[0])
                                           : static_cast<float>(px[2]);  // gray or R (BGR)
            rmin = cardinal::min(rmin, v); rmax = cardinal::max(rmax, v);
            hf.heights[static_cast<cardinal::size_t>(y) * W + x] = v / 255.0f;
        }
    }
    finalize(hf, rmin, rmax);
    return hf;
}

}  // namespace

HeightField decode_heightmap(const cardinal::string& path,
                             cardinal::string* error, u32 raw_dim) {
    const cardinal::string e = ext_of(path);
    const cardinal::vector<u8> bytes = read_all(path);
    HeightField hf;
    if (bytes.empty()) {
        hf.diagnostics = "heightmap: cannot open or empty '" + path + "'";
        if (error) *error = hf.diagnostics;
        return hf;
    }

    if      (e == ".r16" || e == ".raw") hf = decode_raw16(bytes, raw_dim);
    else if (e == ".bmp")                hf = decode_bmp(bytes);
    else if (e == ".tga")                hf = decode_tga(bytes);
    else if (e == ".png")
        hf.diagnostics = "heightmap: PNG needs DEFLATE (lands with cardinal::core::compress)";
    else if (e == ".exr")
        hf.diagnostics = "heightmap: EXR float heightmaps not yet supported";
    else {
        // Unknown extension: a headerless square u16 blob is the most likely
        // intent (RAW exports often have arbitrary extensions).
        hf = decode_raw16(bytes, raw_dim);
        if (!hf.ok)
            hf.diagnostics = "heightmap: unrecognised format '" + e + "'";
    }

    if (hf.ok) {
        cardinal::ostringstream d;
        d << "heightmap: " << hf.source_format << ' ' << hf.width << 'x' << hf.height
          << ", raw range [" << hf.min << ", " << hf.max << "]";
        hf.diagnostics = d.str();
        cardinal::log::infof("import", "%s — %s", path.c_str(), hf.diagnostics.c_str());
    } else if (error) {
        *error = hf.diagnostics;
    }
    return hf;
}

}  // namespace cardinal::import
