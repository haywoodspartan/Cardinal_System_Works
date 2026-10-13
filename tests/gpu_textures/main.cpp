// =============================================================================
// Cardinal — RHI texture shapes on a real device (windowless).
//
// For each shape: create, upload a distinct byte pattern into EVERY mip of
// EVERY layer (3D: every mip slab), read each one back with
// Texture::download, compare byte-exact. Pins subresource addressing
// (mip + layer * mips on D3D12, per-subresource barriers on Vulkan), the
// 256-byte row pitch unpacking, cube face order and BC block packing.
// Sampling semantics (face selection, trilinear) are the compute-texture
// slice's job. SKIP (exit 0) without a device. Exit 0 = all pass.
// =============================================================================
#include <cardinal/rhi/rhi.hpp>
#include <cardinal/core/diag/log.hpp>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

namespace rhi = cardinal::rhi;
using cardinal::u8; using cardinal::u32; using cardinal::usize;

int g_checks = 0, g_fail = 0;
void check_impl(bool ok, const char* e, int l) {
    ++g_checks;
    if (!ok) { ++g_fail; cardinal::log::errorf("textest", "FAIL  L%d  %s", l, e); }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

std::vector<u8> pattern(usize n, u32 seed) {
    std::vector<u8> v(n);
    for (usize i = 0; i < n; ++i)
        v[i] = static_cast<u8>((seed * 31u + static_cast<u32>(i) * 7u + (static_cast<u32>(i) >> 5)) & 0xFFu);
    return v;
}

// Upload + read back every (layer, mip) of a texture built from `d`.
void roundtrip(rhi::Device& dev, const char* be, const char* what, const rhi::TextureDesc& d,
               u32 expect_mips, u32 expect_layers, u32 expect_depth) {
    auto t = dev.create_texture(d);
    CHECK(t != nullptr);
    if (!t) { cardinal::log::errorf("textest", "%s %s: create failed", be, what); return; }
    CHECK(t->mip_levels()   == expect_mips);
    CHECK(t->array_layers() == expect_layers);
    CHECK(t->depth()        == expect_depth);
    CHECK(t->dimension()    == d.dimension);
    CHECK(t->format()       == d.format);
    const u32 mips = t->mip_levels(), layers = t->array_layers();
    for (u32 layer = 0; layer < layers; ++layer)
        for (u32 mip = 0; mip < mips; ++mip) {
            const usize bytes = rhi::texture_mip_bytes(d.format, d.width, d.height, expect_depth, mip);
            CHECK(bytes > 0);
            const auto src = pattern(bytes, 1000u + layer * 16u + mip);
            t->upload(src.data(), src.size(), mip, layer);
        }
    u32 bad = 0;
    for (u32 layer = 0; layer < layers; ++layer)
        for (u32 mip = 0; mip < mips; ++mip) {
            const usize bytes = rhi::texture_mip_bytes(d.format, d.width, d.height, expect_depth, mip);
            const auto src = pattern(bytes, 1000u + layer * 16u + mip);
            std::vector<u8> got(bytes, 0xCDu);
            const bool ok = t->download(got.data(), got.size(), mip, layer);
            CHECK(ok);
            if (ok && std::memcmp(got.data(), src.data(), bytes) != 0) {
                ++bad;
                usize first = 0;
                while (first < bytes && got[first] == src[first]) ++first;
                cardinal::log::errorf("textest", "%s %s: layer %u mip %u differs at byte %zu of %zu (got %02X want %02X)",
                                      be, what, layer, mip, first, bytes, got[first], src[first]);
            }
        }
    CHECK(bad == 0);
    // A wrong-sized read / an out-of-range subresource is refused, not UB.
    {
        u8 tiny[4];
        CHECK(!t->download(tiny, sizeof tiny, 0, 0) || rhi::texture_mip_bytes(d.format, d.width, d.height, expect_depth, 0) <= 4);
        std::vector<u8> buf(rhi::texture_mip_bytes(d.format, d.width, d.height, expect_depth, 0));
        CHECK(!t->download(buf.data(), buf.size(), mips, 0));
        CHECK(!t->download(buf.data(), buf.size(), 0, layers));
    }
    cardinal::log::infof("textest", "%s %s: %u layer(s) x %u mip(s) round-trip ok", be, what, layers, mips);
}

void run_backend(rhi::Backend be, const char* name) {
    rhi::DeviceDesc dd{};
    dd.backend = be;
    auto dev = rhi::create_device(dd);
    if (!dev) { cardinal::log::infof("textest", "SKIP %s: no capable device", name); return; }

    const u32 S = static_cast<u32>(rhi::TextureUsage::Sampled);
    rhi::TextureDesc d{};
    d.usage = S;

    // 2D, full mip chain: 8x8 -> 4 mips.
    d = {}; d.usage = S; d.width = 8; d.height = 8; d.format = rhi::Format::R8G8B8A8_UNORM; d.mip_levels = 0;
    roundtrip(*dev, name, "2D rgba8 8x8 full chain", d, 4, 1, 1);
    // 2D sRGB, single mip, non-square.
    d = {}; d.usage = S; d.width = 6; d.height = 3; d.format = rhi::Format::R8G8B8A8_SRGB;
    roundtrip(*dev, name, "2D srgb 6x3", d, 1, 1, 1);
    // Cube, two mips: six faces.
    d = {}; d.usage = S; d.width = 4; d.height = 4; d.format = rhi::Format::R8G8B8A8_UNORM;
    d.dimension = rhi::TextureDimension::Cube; d.mip_levels = 2;
    roundtrip(*dev, name, "cube rgba8 4x4 x2 mips", d, 2, 6, 1);
    // 3D volume 4x4x4, two mips (mip 1 = 2x2x2 slab).
    d = {}; d.usage = S; d.width = 4; d.height = 4; d.format = rhi::Format::R8G8B8A8_UNORM;
    d.dimension = rhi::TextureDimension::Tex3D; d.depth = 4; d.mip_levels = 2;
    roundtrip(*dev, name, "3D rgba8 4x4x4 x2 mips", d, 2, 1, 4);
    // 2D array, three layers, full chain of a 4x2 (3 mips).
    d = {}; d.usage = S; d.width = 4; d.height = 2; d.format = rhi::Format::R8G8B8A8_UNORM;
    d.dimension = rhi::TextureDimension::Tex2DArray; d.array_layers = 3; d.mip_levels = 0;
    roundtrip(*dev, name, "2D array rgba8 4x2 x3 layers", d, 3, 3, 1);
    // Block compressed: BC1 8x8 (2x2 blocks of 8 B), BC7 8x8 with the 4x4 mip.
    d = {}; d.usage = S; d.width = 8; d.height = 8; d.format = rhi::Format::BC1_UNORM;
    roundtrip(*dev, name, "BC1 8x8", d, 1, 1, 1);
    d = {}; d.usage = S; d.width = 8; d.height = 8; d.format = rhi::Format::BC7_SRGB; d.mip_levels = 2;
    roundtrip(*dev, name, "BC7 srgb 8x8 x2 mips", d, 2, 1, 1);
    // BC5 cube (normal-map style) and BC6H (HDR environment style).
    d = {}; d.usage = S; d.width = 4; d.height = 4; d.format = rhi::Format::BC5_UNORM;
    d.dimension = rhi::TextureDimension::Cube;
    roundtrip(*dev, name, "BC5 cube 4x4", d, 1, 6, 1);
    d = {}; d.usage = S; d.width = 8; d.height = 4; d.format = rhi::Format::BC6H_UF16;
    roundtrip(*dev, name, "BC6H 8x4", d, 1, 1, 1);
    // HDR / single channel.
    d = {}; d.usage = S; d.width = 4; d.height = 4; d.format = rhi::Format::R16G16B16A16_Float;
    roundtrip(*dev, name, "rgba16f 4x4", d, 1, 1, 1);
    d = {}; d.usage = S; d.width = 4; d.height = 4; d.format = rhi::Format::R11G11B10_Float;
    roundtrip(*dev, name, "r11g11b10f 4x4", d, 1, 1, 1);
    d = {}; d.usage = S; d.width = 8; d.height = 8; d.format = rhi::Format::R8_UNORM; d.mip_levels = 0;
    roundtrip(*dev, name, "r8 8x8 full chain", d, 4, 1, 1);
    // mip_levels beyond the chain clamps, never over-allocates.
    d = {}; d.usage = S; d.width = 4; d.height = 4; d.format = rhi::Format::R8G8_UNORM; d.mip_levels = 9;
    roundtrip(*dev, name, "rg8 4x4 clamped mips", d, 3, 1, 1);
}

}  // namespace

int main() {
    // Host-side layout helpers (no device needed).
    CHECK(rhi::full_mip_count(8, 8) == 4);
    CHECK(rhi::full_mip_count(1, 1) == 1);
    CHECK(rhi::full_mip_count(4, 4, 16) == 5);
    CHECK(rhi::texture_mip_bytes(rhi::Format::R8G8B8A8_UNORM, 8, 8, 1, 0) == 256);
    CHECK(rhi::texture_mip_bytes(rhi::Format::R8G8B8A8_UNORM, 8, 8, 1, 3) == 4);
    CHECK(rhi::texture_mip_bytes(rhi::Format::BC1_UNORM, 8, 8, 1, 0) == 32);
    CHECK(rhi::texture_mip_bytes(rhi::Format::BC7_UNORM, 8, 8, 1, 1) == 16);
    CHECK(rhi::texture_mip_bytes(rhi::Format::BC7_UNORM, 8, 8, 1, 2) == 16);   // 2x2 still one block
    CHECK(rhi::texture_mip_bytes(rhi::Format::R16G16B16A16_Float, 4, 4, 4, 1) == 2 * 2 * 2 * 8);
    CHECK(rhi::format_is_block_compressed(rhi::Format::BC6H_UF16));
    CHECK(!rhi::format_is_block_compressed(rhi::Format::R8_UNORM));

    run_backend(rhi::Backend::Vulkan, "vulkan");
#if defined(_WIN32)
    run_backend(rhi::Backend::D3D12, "d3d12");
#endif
    if (g_fail != 0) {
        cardinal::log::errorf("textest", "%d / %d checks FAILED", g_fail, g_checks);
        return 1;
    }
    cardinal::log::infof("textest", "OK  %d checks passed", g_checks);
    return 0;
}
