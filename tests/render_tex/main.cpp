// =============================================================================
// Cardinal — deterministic texture-math regression suite.
//
// render::tex is header-only constexpr/noexcept-pure (no GPU). This
// suite pins:
//
//   * block_bytes / block_dim_{x,y} — the full CompressedFormat table
//     incl. the None=0 and out-of-range fallbacks;
//   * floor_log2_u32 / mip_count_for_size / mip_dim_at — exact values
//     and the >=1 clamp on over-shifted levels;
//   * mip_bytes — uncompressed (bpp-scaled) and BC/ASTC paths with the
//     round-UP-to-a-whole-block rule that makes tiny mips share a block;
//   * mip_chain_bytes — exact full-chain totals (the classic 4/3 sum for
//     uncompressed, and the block-rounded BC7 chain);
//   * aniso_max_for / lod_bias_for_quality — the 5 presets + fallbacks +
//     stable enum ordinals;
//   * plan_for_budget — drops top mips until it fits, returns the exact
//     residency; the exact-fit, forced-drop, can't-fit (-> max_drop,0)
//     and zero-texture edges.
//
// Every pure constexpr golden is ALSO a static_assert, so drift fails
// the build before the test even runs. Exit 0 = all pass.
// =============================================================================

#include <cardinal/render/tex.hpp>
#include <cardinal/core/log.hpp>

namespace {

namespace tx = cardinal::render::tex;
using tx::CompressedFormat;
using tx::AnisoQuality;
using cardinal::u32;
using cardinal::u64;

// ---- compile-time golden pins (drift fails the build) -------------
static_assert(tx::block_bytes(CompressedFormat::BC1) == 8u, "BC1=8B");
static_assert(tx::block_bytes(CompressedFormat::BC7) == 16u, "BC7=16B");
static_assert(tx::block_bytes(CompressedFormat::None) == 0u, "None=0B");
static_assert(tx::block_dim_x(CompressedFormat::ASTC_6x6) == 6u, "6x6");
static_assert(tx::block_dim_y(CompressedFormat::ASTC_8x8) == 8u, "8x8");
static_assert(tx::block_dim_x(CompressedFormat::BC5) == 4u, "BC5 4x4");
static_assert(tx::floor_log2_u32(0u)    == 0u, "log2 0");
static_assert(tx::floor_log2_u32(1u)    == 0u, "log2 1");
static_assert(tx::floor_log2_u32(1024u) == 10u, "log2 1024");
static_assert(tx::floor_log2_u32(1023u) == 9u,  "log2 1023");
static_assert(tx::mip_count_for_size(256u, 256u) == 9u, "256->9 mips");
static_assert(tx::mip_count_for_size(1u, 1u)     == 1u, "1x1->1 mip");
static_assert(tx::mip_count_for_size(1024u, 1u)  == 11u, "1024x1->11");
static_assert(tx::mip_dim_at(256u, 8u)  == 1u, "256 mip8 = 1");
static_assert(tx::mip_dim_at(256u, 31u) == 1u, "clamped >=1");
// Over-shift: level >= 32 (a u32 shift >= bit width). These would be a
// hard constant-expression error pre-fix; mip_dim_at now guards it and
// returns the documented >=1 minimum.
static_assert(tx::mip_dim_at(256u, 32u) == 1u, "mip32 over-shift -> 1");
static_assert(tx::mip_dim_at(256u, 99u) == 1u, "mip99 over-shift -> 1");
static_assert(tx::mip_dim_at(1u, 4000u) == 1u, "huge level -> 1");
static_assert(tx::mip_dim_at(640u, 2u)  == 160u, "640>>2");
static_assert(tx::mip_chain_bytes(256u,256u,CompressedFormat::None,4u)
              == 349524ull, "RGBA8 256 chain");
static_assert(tx::mip_chain_bytes(256u,256u,CompressedFormat::BC7,4u)
              == 87408ull, "BC7 256 chain");
static_assert(tx::aniso_max_for(AnisoQuality::Ultra)  == 16u, "ultra 16x");
static_assert(tx::aniso_max_for(AnisoQuality::Medium) == 4u,  "med 4x");
static_assert(tx::plan_for_budget(256u,256u,CompressedFormat::None,1u,
                                  1000000000ull,4u).keep_mip_offset == 0u,
              "huge budget keeps full chain");
static_assert(tx::plan_for_budget(256u,256u,CompressedFormat::None,0u,
                                  1000ull,4u).used_bytes == 0ull,
              "zero textures -> nothing");

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("textest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(double a, double b, double eps = 1e-6) {
    const double d = (a > b) ? (a - b) : (b - a);
    return d <= eps;
}
CompressedFormat cf(int v) {
    return static_cast<CompressedFormat>(static_cast<u32>(v));
}
AnisoQuality aq(int v) {
    return static_cast<AnisoQuality>(static_cast<u32>(v));
}

// ---- format tables ------------------------------------------------
void test_format_tables() {
    CHECK(tx::block_bytes(CompressedFormat::BC1)      == 8u);
    CHECK(tx::block_bytes(CompressedFormat::BC3)      == 16u);
    CHECK(tx::block_bytes(CompressedFormat::BC4)      == 8u);
    CHECK(tx::block_bytes(CompressedFormat::BC5)      == 16u);
    CHECK(tx::block_bytes(CompressedFormat::BC6H)     == 16u);
    CHECK(tx::block_bytes(CompressedFormat::BC7)      == 16u);
    CHECK(tx::block_bytes(CompressedFormat::ASTC_4x4) == 16u);
    CHECK(tx::block_bytes(CompressedFormat::ASTC_6x6) == 16u);
    CHECK(tx::block_bytes(CompressedFormat::ASTC_8x8) == 16u);
    CHECK(tx::block_bytes(CompressedFormat::None)     == 0u);
    CHECK(tx::block_bytes(cf(99))                     == 0u);

    // block dims: only ASTC 6x6 / 8x8 are non-4; everything else 4x4.
    CHECK(tx::block_dim_x(CompressedFormat::BC1) == 4u);
    CHECK(tx::block_dim_y(CompressedFormat::BC1) == 4u);
    CHECK(tx::block_dim_x(CompressedFormat::ASTC_4x4) == 4u);
    CHECK(tx::block_dim_x(CompressedFormat::ASTC_6x6) == 6u);
    CHECK(tx::block_dim_y(CompressedFormat::ASTC_6x6) == 6u);
    CHECK(tx::block_dim_x(CompressedFormat::ASTC_8x8) == 8u);
    CHECK(tx::block_dim_y(CompressedFormat::ASTC_8x8) == 8u);
    CHECK(tx::block_dim_x(CompressedFormat::None) == 4u);   // default arm
    CHECK(tx::block_dim_x(cf(99)) == 4u);
}

// ---- log2 / mip count / mip dim -----------------------------------
void test_mip_geometry() {
    CHECK(tx::floor_log2_u32(0u) == 0u);
    CHECK(tx::floor_log2_u32(1u) == 0u);
    CHECK(tx::floor_log2_u32(2u) == 1u);
    CHECK(tx::floor_log2_u32(3u) == 1u);
    CHECK(tx::floor_log2_u32(4u) == 2u);
    CHECK(tx::floor_log2_u32(7u) == 2u);
    CHECK(tx::floor_log2_u32(8u) == 3u);
    CHECK(tx::floor_log2_u32(256u) == 8u);

    CHECK(tx::mip_count_for_size(1u, 1u)     == 1u);
    CHECK(tx::mip_count_for_size(2u, 1u)     == 2u);
    CHECK(tx::mip_count_for_size(256u, 256u) == 9u);
    CHECK(tx::mip_count_for_size(256u, 128u) == 9u);   // uses max dim
    CHECK(tx::mip_count_for_size(640u, 480u) == 10u);  // log2(640)=9
    CHECK(tx::mip_count_for_size(0u, 0u)     == 1u);

    CHECK(tx::mip_dim_at(256u, 0u) == 256u);
    CHECK(tx::mip_dim_at(256u, 1u) == 128u);
    CHECK(tx::mip_dim_at(256u, 8u) == 1u);
    CHECK(tx::mip_dim_at(256u, 9u) == 1u);             // 256>>9=0 -> 1
    CHECK(tx::mip_dim_at(1u, 1u)   == 1u);
    CHECK(tx::mip_dim_at(640u, 1u) == 320u);
    // Over-shift guard at runtime too (level >= 32 == UB pre-fix).
    CHECK(tx::mip_dim_at(256u, 32u)   == 1u);
    CHECK(tx::mip_dim_at(256u, 99u)   == 1u);
    CHECK(tx::mip_dim_at(4096u, 64u)  == 1u);
}

// ---- mip_bytes: uncompressed + block, tiny-mip rounding ----------
void test_mip_bytes() {
    // uncompressed RGBA8 (bpp 4)
    CHECK(tx::mip_bytes(256u,256u,0u,CompressedFormat::None,4u) == 262144ull);
    CHECK(tx::mip_bytes(256u,256u,1u,CompressedFormat::None,4u) == 65536ull);
    CHECK(tx::mip_bytes(4u,4u,0u,CompressedFormat::None,4u)     == 64ull);
    // bpp parameter scales linearly.
    CHECK(tx::mip_bytes(16u,16u,0u,CompressedFormat::None,2u)   == 512ull);
    CHECK(tx::mip_bytes(16u,16u,0u,CompressedFormat::None,1u)   == 256ull);

    // BC1 (8 B / 4x4 block).
    CHECK(tx::mip_bytes(256u,256u,0u,CompressedFormat::BC1,4u) == 32768ull);
    // tiny mips round UP to one whole block (shared last block).
    CHECK(tx::mip_bytes(4u,4u,0u,CompressedFormat::BC1,4u) == 8ull);
    CHECK(tx::mip_bytes(2u,2u,0u,CompressedFormat::BC1,4u) == 8ull);
    CHECK(tx::mip_bytes(1u,1u,0u,CompressedFormat::BC1,4u) == 8ull);

    // BC7 (16 B / 4x4).
    CHECK(tx::mip_bytes(256u,256u,0u,CompressedFormat::BC7,4u) == 65536ull);
    // ASTC 8x8 (16 B / 8x8): 256/8 = 32 blocks each axis.
    CHECK(tx::mip_bytes(256u,256u,0u,CompressedFormat::ASTC_8x8,4u)
          == 16384ull);
    // ASTC 6x6: ceil(12/6)=2 blocks each axis -> 2*2*16 = 64.
    CHECK(tx::mip_bytes(12u,12u,0u,CompressedFormat::ASTC_6x6,4u) == 64ull);
    // ASTC 6x6 256: ceil(256/6)=43 -> 43*43*16.
    CHECK(tx::mip_bytes(256u,256u,0u,CompressedFormat::ASTC_6x6,4u)
          == 29584ull);
}

// ---- mip_chain_bytes: exact full-chain totals --------------------
void test_mip_chain() {
    CHECK(tx::mip_chain_bytes(1u,1u,CompressedFormat::None,4u)   == 4ull);
    CHECK(tx::mip_chain_bytes(2u,2u,CompressedFormat::None,4u)   == 20ull);
    CHECK(tx::mip_chain_bytes(4u,4u,CompressedFormat::None,4u)   == 84ull);
    // classic 4*(256^2+128^2+...+1) = 349524.
    CHECK(tx::mip_chain_bytes(256u,256u,CompressedFormat::None,4u)
          == 349524ull);
    // BC1 4x4 chain: 3 levels, each rounds to one 8-byte block.
    CHECK(tx::mip_chain_bytes(4u,4u,CompressedFormat::BC1,4u)    == 24ull);
    // BC7 256 chain (block-rounded tail).
    CHECK(tx::mip_chain_bytes(256u,256u,CompressedFormat::BC7,4u)
          == 87408ull);
}

// ---- aniso / LOD presets -----------------------------------------
void test_aniso_lod() {
    CHECK(tx::aniso_max_for(AnisoQuality::Off)    == 1u);
    CHECK(tx::aniso_max_for(AnisoQuality::Low)    == 2u);
    CHECK(tx::aniso_max_for(AnisoQuality::Medium) == 4u);
    CHECK(tx::aniso_max_for(AnisoQuality::High)   == 8u);
    CHECK(tx::aniso_max_for(AnisoQuality::Ultra)  == 16u);
    CHECK(tx::aniso_max_for(aq(99))               == 1u);   // fallback

    CHECK(ap(tx::lod_bias_for_quality(AnisoQuality::Off),     1.0));
    CHECK(ap(tx::lod_bias_for_quality(AnisoQuality::Low),     0.5));
    CHECK(ap(tx::lod_bias_for_quality(AnisoQuality::Medium),  0.0));
    CHECK(ap(tx::lod_bias_for_quality(AnisoQuality::High),   -0.25));
    CHECK(ap(tx::lod_bias_for_quality(AnisoQuality::Ultra),  -0.5));
    CHECK(ap(tx::lod_bias_for_quality(aq(99)),                0.0));

    CHECK(static_cast<u32>(AnisoQuality::Off)    == 0u);
    CHECK(static_cast<u32>(AnisoQuality::Ultra)  == 4u);
}

// ---- plan_for_budget: drop-until-fits + edges --------------------
void test_streaming() {
    const u64 full = tx::mip_chain_bytes(256u,256u,CompressedFormat::None,4u);
    CHECK(full == 349524ull);

    // huge budget -> keep full chain, used == full * count.
    {
        auto p = tx::plan_for_budget(256u,256u,CompressedFormat::None,
                                     1u, 1000000000ull, 4u);
        CHECK(p.keep_mip_offset == 0u && p.used_bytes == full);
    }
    // exact fit at drop 0 for 2 textures.
    {
        auto p = tx::plan_for_budget(256u,256u,CompressedFormat::None,
                                     2u, full * 2u, 4u);
        CHECK(p.keep_mip_offset == 0u && p.used_bytes == full * 2u);
    }
    // budget forces dropping mip 0 (level0 = 262144).
    {
        auto p = tx::plan_for_budget(256u,256u,CompressedFormat::None,
                                     1u, 100000ull, 4u);
        CHECK(p.keep_mip_offset == 1u);
        CHECK(p.used_bytes == full - 262144ull);       // 87380
    }
    // budget == smallest mip exactly -> keep only the last level.
    {
        auto p = tx::plan_for_budget(256u,256u,CompressedFormat::None,
                                     1u, 4ull, 4u);
        CHECK(p.keep_mip_offset == 8u && p.used_bytes == 4ull);
    }
    // can't fit even the 1x1 mip -> {max_drop, 0}.
    {
        auto p = tx::plan_for_budget(256u,256u,CompressedFormat::None,
                                     1u, 3ull, 4u);
        CHECK(p.keep_mip_offset == 9u && p.used_bytes == 0ull);
    }
    // zero textures -> default plan, regardless of budget.
    {
        auto p = tx::plan_for_budget(256u,256u,CompressedFormat::None,
                                     0u, 1000ull, 4u);
        CHECK(p.keep_mip_offset == 0u && p.used_bytes == 0ull);
    }
}

}  // namespace

int main() {
    test_format_tables();
    test_mip_geometry();
    test_mip_bytes();
    test_mip_chain();
    test_aniso_lod();
    test_streaming();

    if (g_fail == 0) {
        cardinal::log::infof("textest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("textest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
