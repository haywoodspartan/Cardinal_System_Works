// =============================================================================
// Cardinal — deterministic hashing-primitive regression suite.
//
// splitmix64 / fnv1a64 / fxhash64 underpin content IDs, serialised keys
// and hash-map keying. A silent drift in any constant, shift or step
// order changes EVERY derived id and breaks on-disk / save compat
// invisibly. This suite pins:
//
//   * splitmix64 — the canonical published SplitMix64 stream outputs for
//     seed 0 (0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F)
//     and splitmix64(0)==0, asserted at COMPILE TIME (constexpr path) so
//     drift fails the build itself, plus bijection / avalanche / purity.
//   * fnv1a64 — the published FNV-1a-64 digests ("", "a", "foobar"),
//     the constants, the void*/const char* overload agreement, the
//     streaming-chain identity, embedded-NUL handling, seed/length
//     sensitivity.
//   * fxhash64 — hand-derivable primitive values (K, rotate-by-5), the
//     8-byte-block == fxhash64(seed,word) identity (endian-agnostic via
//     memcpy round-trip), multi-block decomposition, empty==seed.
//
// Pure + header-only. The few byte-order-sensitive tail cases assume
// little-endian (the engine targets x64 / ARM64 LE). Exit 0 = all pass.
// =============================================================================

#include <cardinal/core/hash.hpp>
#include <cardinal/core/log.hpp>

namespace {

namespace h = cardinal::core;
using cardinal::u8;
using cardinal::u64;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("hashtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

int pc64(u64 v) { int n = 0; while (v) { n += static_cast<int>(v & 1ull); v >>= 1; } return n; }

// ----- COMPILE-TIME golden pins (constexpr path) --------------------
// A drift in any splitmix64 constant/shift fails the build right here.
static_assert(h::kFnvPrime64  == 1099511628211ull,            "FNV prime");
static_assert(h::kFnvOffset64 == 14695981039346656037ull,     "FNV offset");
static_assert(h::kFnvOffset64 == 0xCBF29CE484222325ull,       "FNV offset hex");
static_assert(h::splitmix64(0ull) == 0ull,                    "smx(0)=0");
// Input = the golden constant itself (no hand-derived state): the very
// first SplitMix64(seed 0) output. Outputs #2/#3 are pinned at runtime
// via the next() chain so the compiler — not this file — derives the
// intermediate stream state.
static_assert(h::splitmix64(0x9E3779B97F4A7C15ull) ==
              0xE220A8397B1DCDAFull,                           "SplitMix64 #1");

// ---- splitmix64: published stream + purity + bijection ------------
void test_splitmix64() {
    // Mirror the compile-time golden as runtime checks (so they count
    // and report). The canonical SplitMix64 PRNG seeded 0: state +=
    // 0x9E3779B97F4A7C15 then finalize.
    CHECK(h::splitmix64(0ull) == 0ull);
    u64 s = 0;
    auto next = [&]() { s += 0x9E3779B97F4A7C15ull; return h::splitmix64(s); };
    CHECK(next() == 0xE220A8397B1DCDAFull);   // canonical output #1
    CHECK(next() == 0x6E789E6AA1B965F4ull);   // #2
    CHECK(next() == 0x06C45D188009454Full);   // #3

    // Deterministic / pure.
    CHECK(h::splitmix64(12345ull) == h::splitmix64(12345ull));
    CHECK(h::splitmix64(0xDEADBEEFull) == h::splitmix64(0xDEADBEEFull));

    // Bijective over a sample (it is an invertible mix): 256 distinct
    // inputs → 256 distinct outputs.
    u64 outs[256];
    for (int i = 0; i < 256; ++i) outs[i] = h::splitmix64(static_cast<u64>(i));
    bool all_distinct = true;
    for (int i = 0; i < 256 && all_distinct; ++i)
        for (int j = i + 1; j < 256; ++j)
            if (outs[i] == outs[j]) { all_distinct = false; break; }
    CHECK(all_distinct);
    CHECK(outs[0] == 0ull);                   // splitmix64(0)==0 again

    // Avalanche: a 1-bit input change flips many output bits (good mix
    // lands near 32; a no-op/identity regression would be ~0 or ~64).
    const int d1 = pc64(h::splitmix64(0ull) ^ h::splitmix64(1ull));
    const u64 base = 0xA5A5A5A5A5A5A5A5ull;
    const int d2 = pc64(h::splitmix64(base) ^ h::splitmix64(base ^ (1ull << 20)));
    CHECK(d1 >= 8 && d1 <= 56);
    CHECK(d2 >= 8 && d2 <= 56);
}

// ---- fnv1a64: published digests + identities ----------------------
void test_fnv1a64() {
    CHECK(h::kFnvPrime64  == 1099511628211ull);
    CHECK(h::kFnvOffset64 == 14695981039346656037ull);
    CHECK(h::kFnvOffset64 == 0xCBF29CE484222325ull);

    // Empty input → the offset basis (both overloads).
    CHECK(h::fnv1a64("") == h::kFnvOffset64);
    CHECK(h::fnv1a64(static_cast<const void*>(""), 0) == h::kFnvOffset64);

    // Canonical published FNV-1a-64 digests.
    CHECK(h::fnv1a64("a")      == 0xAF63DC4C8601EC8Cull);
    CHECK(h::fnv1a64("foobar") == 0x85944171F73967E8ull);

    // void*/const char* overloads agree on identical bytes.
    CHECK(h::fnv1a64(static_cast<const void*>("foobar"), 6) ==
          0x85944171F73967E8ull);
    CHECK(h::fnv1a64("foobar") ==
          h::fnv1a64(static_cast<const void*>("foobar"), 6));
    CHECK(h::fnv1a64("a") == h::fnv1a64(static_cast<const void*>("a"), 1));

    // Streaming-chain identity: hashing AB == hash B seeded with hash A.
    const u64 hfoo = h::fnv1a64(static_cast<const void*>("foo"), 3);
    CHECK(h::fnv1a64(static_cast<const void*>("bar"), 3, hfoo) ==
          h::fnv1a64("foobar"));

    // Sensitivity: value / order / length / seed all matter.
    CHECK(h::fnv1a64("a")  != h::fnv1a64("b"));
    CHECK(h::fnv1a64("ab") != h::fnv1a64("ba"));
    CHECK(h::fnv1a64("a")  != h::fnv1a64("aa"));
    CHECK(h::fnv1a64(static_cast<const void*>("x"), 1, 1ull) !=
          h::fnv1a64(static_cast<const void*>("x"), 1, 2ull));

    // Embedded NUL: the void*/len form spans it; the C-string form stops.
    const char ab0[3] = { 'a', '\0', 'b' };
    CHECK(h::fnv1a64(static_cast<const void*>(ab0), 3) !=
          h::fnv1a64(static_cast<const void*>("a"), 1));
    CHECK(h::fnv1a64(ab0) == h::fnv1a64("a"));            // cstr stops at NUL
    CHECK(h::fnv1a64(ab0) == 0xAF63DC4C8601EC8Cull);
}

// ---- fxhash64: hand-derivable primitives + block identities -------
void test_fxhash64() {
    constexpr u64 K = 0x517CC1B727220A95ull;

    // Primitive: fxhash64(h,w) = rotl(h,5) ^ (w*K).
    CHECK(h::fxhash64(0ull, 0ull) == 0ull);
    CHECK(h::fxhash64(0ull, 1ull) == K);                 // rotl(0,5)=0 ^ K
    CHECK(h::fxhash64(1ull, 0ull) == 0x20ull);           // rotl(1,5)=32
    CHECK(h::fxhash64(1ull, 1ull) == (0x20ull ^ K));     // 0x517cc1b727220ab5
    CHECK(h::fxhash64(0x8000000000000000ull, 0ull) == 0x10ull);  // rotl wrap

    // Empty input → the seed, untouched.
    CHECK(h::fxhash64(static_cast<const void*>(""), 0, 0ull) == 0ull);
    CHECK(h::fxhash64(static_cast<const void*>(""), 0, 1234ull) == 1234ull);

    // One 8-byte block == fxhash64(seed, word). memcpy round-trips the
    // word, so this identity is endian-independent.
    u64 w = 0x1122334455667788ull;
    u64 s = 0xABCDEF0123456789ull;
    CHECK(h::fxhash64(&w, 8, s)   == h::fxhash64(s, w));
    CHECK(h::fxhash64(&w, 8, 0ull) == h::fxhash64(0ull, w));

    // Two blocks == nested application, left to right.
    u64 two[2] = { 0xDEADBEEF12345678ull, 0x0F0F0F0F0F0F0F0Full };
    CHECK(h::fxhash64(static_cast<const void*>(two), 16, s) ==
          h::fxhash64(h::fxhash64(s, two[0]), two[1]));

    // Deterministic; seed- and data-sensitive.
    CHECK(h::fxhash64(&w, 8, s) == h::fxhash64(&w, 8, s));
    CHECK(h::fxhash64(&w, 8, s) != h::fxhash64(&w, 8, s ^ 1ull));
    u64 w2 = w ^ (1ull << 40);
    CHECK(h::fxhash64(&w, 8, s) != h::fxhash64(&w2, 8, s));

    // Tail handling (little-endian: low bytes first).
    u8 oneb = static_cast<u8>(0xAB);
    CHECK(h::fxhash64(&oneb, 1, 0ull) == h::fxhash64(0ull, 0xABull));
    u8 b3[3] = { 1, 2, 3 };
    u8 b4[4] = { 1, 2, 3, 4 };
    CHECK(h::fxhash64(b3, 3, 0ull) != h::fxhash64(b4, 4, 0ull));
    CHECK(h::fxhash64(b3, 3, 0ull) == h::fxhash64(0ull, 0x030201ull));
}

}  // namespace

int main() {
    test_splitmix64();
    test_fnv1a64();
    test_fxhash64();

    if (g_fail == 0) {
        cardinal::log::infof("hashtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("hashtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
