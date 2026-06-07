// =============================================================================
// cardinal::core::compress — DEFLATE inflate (RFC 1951) + zlib (RFC 1950).
//
// Clean-room tinfl-style decoder: a bit reader, canonical-Huffman decode
// (puff-style: counts-per-length + sorted symbols), the three block types
// (stored / fixed-Huffman / dynamic-Huffman), and the LZ77 length/distance
// back-reference copy. Bounds-checked on every read + every output write so
// untrusted importer input can never read or write out of range.
// =============================================================================

#include <cardinal/core/compress/inflate.hpp>

namespace cardinal::core::compress {

namespace {

using cardinal::u8;
using cardinal::u16;
using cardinal::i16;
using cardinal::u32;
using cardinal::usize;

struct State {
    const u8* in {nullptr};
    usize     in_n {0};
    usize     in_pos {0};
    u32       bitbuf {0};
    int       bitcnt {0};
    u8*       out {nullptr};
    usize     out_cap {0};
    usize     out_pos {0};
    bool      error {false};
};

// Read `need` bits (LSB-first), refilling the accumulator from the byte stream.
// need <= 16. Sets error (and returns partial/zero) on input underrun.
u32 getbits(State& s, int need) noexcept {
    while (s.bitcnt < need) {
        if (s.in_pos >= s.in_n) { s.error = true; break; }
        s.bitbuf |= static_cast<u32>(s.in[s.in_pos++]) << s.bitcnt;
        s.bitcnt += 8;
    }
    const u32 val = (need == 0) ? 0u : (s.bitbuf & ((1u << need) - 1u));
    s.bitbuf >>= need;
    s.bitcnt -= need;
    if (s.bitcnt < 0) s.bitcnt = 0;
    return val;
}

// Drop partial bits so the next read starts on a byte boundary (stored blocks).
void align_byte(State& s) noexcept {
    const int drop = s.bitcnt & 7;
    s.bitbuf >>= drop;
    s.bitcnt -= drop;
}

// Canonical Huffman table: count of codes per bit-length + symbols sorted by
// (length, symbol). Sized for the largest alphabet (lit/len = 288).
struct Huff {
    i16 counts[16];
    i16 symbols[288];
};

void hbuild(Huff& h, const u8* lengths, int n) noexcept {
    for (int i = 0; i < 16; ++i) h.counts[i] = 0;
    for (int i = 0; i < n; ++i) h.counts[lengths[i]]++;
    h.counts[0] = 0;                          // length-0 = symbol unused
    i16 offs[16];
    offs[0] = 0; offs[1] = 0;
    for (int len = 1; len < 15; ++len)
        offs[len + 1] = static_cast<i16>(offs[len] + h.counts[len]);
    for (int i = 0; i < n; ++i)
        if (lengths[i]) h.symbols[offs[lengths[i]]++] = static_cast<i16>(i);
}

// Decode one symbol by walking bit-by-bit down the canonical code space.
int hdecode(State& s, const Huff& h) noexcept {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len <= 15; ++len) {
        code |= static_cast<int>(getbits(s, 1));
        if (s.error) return -1;
        const int count = h.counts[len];
        if (code - first < count) return h.symbols[index + (code - first)];
        index += count;
        first  = (first + count) << 1;
        code <<= 1;
    }
    return -1;
}

// RFC 1951 length (257..285) + distance (0..29) base values + extra-bit counts.
const u16 LBASE[29] = {3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,
                       67,83,99,115,131,163,195,227,258};
const u8  LEXT[29]  = {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
const u16 DBASE[30] = {1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,
                       1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
const u8  DEXT[30]  = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};

bool stored(State& s) noexcept {
    align_byte(s);
    const u32 len  = getbits(s, 16);
    const u32 nlen = getbits(s, 16);
    if (s.error) return false;
    if ((len ^ 0xFFFFu) != nlen) return false;        // NLEN must be ~LEN
    for (u32 i = 0; i < len; ++i) {
        if (s.out_pos >= s.out_cap) return false;
        const u32 b = getbits(s, 8);
        if (s.error) return false;
        s.out[s.out_pos++] = static_cast<u8>(b);
    }
    return true;
}

bool codes(State& s, const Huff& hl, const Huff& hd) noexcept {
    for (;;) {
        const int sym = hdecode(s, hl);
        if (sym < 0) return false;
        if (sym == 256) return true;                  // end of block
        if (sym < 256) {
            if (s.out_pos >= s.out_cap) return false;
            s.out[s.out_pos++] = static_cast<u8>(sym);
            continue;
        }
        const int ls = sym - 257;
        if (ls >= 29) return false;
        const int len = LBASE[ls] + static_cast<int>(getbits(s, LEXT[ls]));
        const int ds  = hdecode(s, hd);
        if (ds < 0 || ds >= 30) return false;
        const usize dist = DBASE[ds] + static_cast<usize>(getbits(s, DEXT[ds]));
        if (s.error) return false;
        if (dist > s.out_pos) return false;           // distance too far back
        if (s.out_pos + static_cast<usize>(len) > s.out_cap) return false;
        const usize from = s.out_pos - dist;
        for (int i = 0; i < len; ++i)                 // byte-wise: handles overlap
            s.out[s.out_pos + i] = s.out[from + i];
        s.out_pos += static_cast<usize>(len);
    }
}

bool fixed(State& s) noexcept {
    u8 lit[288];
    for (int i = 0;   i < 144; ++i) lit[i] = 8;
    for (int i = 144; i < 256; ++i) lit[i] = 9;
    for (int i = 256; i < 280; ++i) lit[i] = 7;
    for (int i = 280; i < 288; ++i) lit[i] = 8;
    u8 dist[30];
    for (int i = 0; i < 30; ++i) dist[i] = 5;
    Huff hl, hd;
    hbuild(hl, lit, 288);
    hbuild(hd, dist, 30);
    return codes(s, hl, hd);
}

bool dynamic(State& s) noexcept {
    static const u8 ORDER[19] =
        {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
    const int hlit  = static_cast<int>(getbits(s, 5)) + 257;
    const int hdist = static_cast<int>(getbits(s, 5)) + 1;
    const int hclen = static_cast<int>(getbits(s, 4)) + 4;
    if (s.error || hlit > 286 || hdist > 30) return false;

    u8 cl[19] = {0};
    for (int i = 0; i < hclen; ++i) cl[ORDER[i]] = static_cast<u8>(getbits(s, 3));
    if (s.error) return false;
    Huff hc;
    hbuild(hc, cl, 19);

    u8 lengths[288 + 32] = {0};
    const int total = hlit + hdist;
    int n = 0;
    while (n < total) {
        const int sym = hdecode(s, hc);
        if (sym < 0) return false;
        if (sym < 16) {
            lengths[n++] = static_cast<u8>(sym);
        } else if (sym == 16) {                       // copy previous 3..6
            if (n == 0) return false;
            int rep = static_cast<int>(getbits(s, 2)) + 3;
            const u8 prev = lengths[n - 1];
            while (rep-- && n < total) lengths[n++] = prev;
        } else if (sym == 17) {                       // repeat zero 3..10
            int rep = static_cast<int>(getbits(s, 3)) + 3;
            while (rep-- && n < total) lengths[n++] = 0;
        } else if (sym == 18) {                       // repeat zero 11..138
            int rep = static_cast<int>(getbits(s, 7)) + 11;
            while (rep-- && n < total) lengths[n++] = 0;
        } else {
            return false;
        }
        if (s.error) return false;
    }
    if (lengths[256] == 0) return false;              // need an end-of-block code

    Huff hl, hd;
    hbuild(hl, lengths, hlit);
    hbuild(hd, lengths + hlit, hdist);
    return codes(s, hl, hd);
}

}  // namespace

usize inflate_raw(const u8* in, usize in_n, u8* out, usize out_cap) noexcept {
    if (!in || !out) return 0;
    State s;
    s.in = in; s.in_n = in_n; s.out = out; s.out_cap = out_cap;
    bool final_block = false;
    do {
        final_block = getbits(s, 1) != 0;
        const u32 type = getbits(s, 2);
        if (s.error) return 0;
        bool ok = false;
        switch (type) {
            case 0: ok = stored(s);  break;
            case 1: ok = fixed(s);   break;
            case 2: ok = dynamic(s); break;
            default: return 0;                        // type 3 is reserved
        }
        if (!ok || s.error) return 0;
    } while (!final_block);
    return s.out_pos;
}

usize inflate_zlib(const u8* in, usize in_n, u8* out, usize out_cap) noexcept {
    if (!in || in_n < 6) return 0;                    // 2 header + >=0 data + 4 adler
    const u8 cmf = in[0], flg = in[1];
    if ((cmf & 0x0F) != 8) return 0;                  // CM must be 8 (deflate)
    if (((static_cast<u32>(cmf) << 8) | flg) % 31u != 0) return 0;  // FCHECK
    if (flg & 0x20) return 0;                         // FDICT preset dict unsupported
    // The raw deflate stream ends at its BFINAL block; the trailing 4-byte
    // adler32 is left unread (not verified). Pass everything after the header.
    return inflate_raw(in + 2, in_n - 2, out, out_cap);
}

}  // namespace cardinal::core::compress
