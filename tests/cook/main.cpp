// =============================================================================
// Cardinal — deterministic cooker regression suite.
//
// cook is the middle of the asset pipeline: import → COOK → pack. The
// CookedAsset typed-blob header is exactly what pack/distribute reads
// to identify an asset, so a codec regression here silently mis-types
// or truncates every shipped asset. Zero test deps, fully deterministic
// + headless. Exit 0 = all pass.
//
// Locks: CookedAsset serialize/deserialize round-trip + the exact
// on-disk header + failure paths; asset_type_name / _for_extension;
// the MeshCooker import→cook delegation path (a temp OBJ driven through
// the real importer) plus the CARDMESH and box fallbacks, including
// build-determinism (same source ⇒ byte-identical cooked blob).
//
// Out of scope by design: cook_all/Driver (needs a project::Project
// tree) and the texture/shader/audio cookers (image decode /
// shader::Compiler) — they pair with the deferred pack distribute()
// integration arc.
// =============================================================================

#include <cardinal/cook/cook.hpp>
#include <cardinal/core/log.hpp>

#include <filesystem>
#include <fstream>
#include <vector>

namespace {

namespace ck = cardinal::cook;
using ck::AssetType;
using ck::CookedAsset;
using cardinal::u8;
using cardinal::u32;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("cooktest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

u32 rd_u32(const std::vector<u8>& b, cardinal::usize off) {
    return static_cast<u32>(b[off]) |
           (static_cast<u32>(b[off + 1]) << 8) |
           (static_cast<u32>(b[off + 2]) << 16) |
           (static_cast<u32>(b[off + 3]) << 24);
}

std::vector<u8> blob(u32 seed, cardinal::usize n) {
    std::vector<u8> v; v.reserve(n);
    u32 x = seed ? seed : 1u;
    for (cardinal::usize i = 0; i < n; ++i) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        v.push_back(static_cast<u8>(x & 0xFFu));
    }
    return v;
}

void write_text(const std::filesystem::path& p, const char* s) {
    std::ofstream f(p, std::ios::binary);
    f << s;
}

// ---- CookedAsset codec ----------------------------------------------
void test_codec() {
    CookedAsset a{};
    a.type           = AssetType::Texture;
    a.cooker_version = 7u;
    a.payload        = blob(0x1234u, 300);
    a.diagnostics    = "built-time only, not persisted";

    const std::vector<u8> s = a.serialize();
    // Exact on-disk header: "COOK" + u32 type + u32 size + u32 ver.
    CHECK(s.size() == sz(16) + a.payload.size());
    CHECK(s[0] == 'C' && s[1] == 'O' && s[2] == 'O' && s[3] == 'K');
    CHECK(rd_u32(s, 4)  == static_cast<u32>(AssetType::Texture));
    CHECK(rd_u32(s, 8)  == static_cast<u32>(a.payload.size()));
    CHECK(rd_u32(s, 12) == 7u);

    CookedAsset b{};
    CHECK(CookedAsset::deserialize(s, b));
    CHECK(b.type == AssetType::Texture);
    CHECK(b.cooker_version == 7u);
    CHECK(b.payload == a.payload);
    CHECK(b.diagnostics.empty());          // diagnostics is NOT persisted

    // Empty payload round-trips.
    CookedAsset e{}; e.type = AssetType::Audio; e.cooker_version = 1u;
    CookedAsset eo{};
    CHECK(CookedAsset::deserialize(e.serialize(), eo));
    CHECK(eo.type == AssetType::Audio);
    CHECK(eo.payload.empty());

    // Trailing extra bytes are tolerated (only `size` bytes consumed).
    std::vector<u8> padded = s;
    padded.push_back(0xAA); padded.push_back(0xBB);
    CookedAsset p{};
    CHECK(CookedAsset::deserialize(padded, p));
    CHECK(p.payload == a.payload);

    // Failure paths.
    CookedAsset junk{};
    CHECK(!CookedAsset::deserialize(std::vector<u8>(8, 0), junk));   // < 16
    std::vector<u8> bad = s; bad[1] = 'X';                           // magic
    CHECK(!CookedAsset::deserialize(bad, junk));
    std::vector<u8> trunc = s; trunc.resize(16 + 2);                 // size>actual
    CHECK(!CookedAsset::deserialize(trunc, junk));
}

// ---- type lookups ---------------------------------------------------
void test_type_lookups() {
    CHECK(std::string(ck::asset_type_name(AssetType::Unknown))  == "Unknown");
    CHECK(std::string(ck::asset_type_name(AssetType::Texture))  == "Texture");
    CHECK(std::string(ck::asset_type_name(AssetType::Mesh))     == "Mesh");
    CHECK(std::string(ck::asset_type_name(AssetType::Shader))   == "Shader");
    CHECK(std::string(ck::asset_type_name(AssetType::Audio))    == "Audio");
    CHECK(std::string(ck::asset_type_name(AssetType::Material)) == "Material");

    CHECK(ck::asset_type_for_extension(".png")      == AssetType::Texture);
    CHECK(ck::asset_type_for_extension(".TGA")      == AssetType::Texture);
    CHECK(ck::asset_type_for_extension(".obj")      == AssetType::Mesh);
    CHECK(ck::asset_type_for_extension(".GLB")      == AssetType::Mesh);
    CHECK(ck::asset_type_for_extension(".hlsl")     == AssetType::Shader);
    CHECK(ck::asset_type_for_extension(".wav")      == AssetType::Audio);
    CHECK(ck::asset_type_for_extension(".material") == AssetType::Material);
    CHECK(ck::asset_type_for_extension(".xyz")      == AssetType::Unknown);
    CHECK(ck::asset_type_for_extension("png")       == AssetType::Unknown); // no dot
}

// ---- MeshCooker: import→cook delegation + fallbacks + determinism ---
void test_mesh_cooker(const std::filesystem::path& dir) {
    auto cooker = ck::make_mesh_cooker();
    CHECK(cooker != nullptr);
    if (!cooker) return;
    CHECK(cooker->asset_type() == AssetType::Mesh);
    CHECK(cooker->cooker_version() == 2u);

    // (a) OBJ through the real importer (path-driven delegation).
    const auto objp = dir / "tri.obj";
    write_text(objp, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
    std::ifstream rf(objp, std::ios::binary);
    std::vector<u8> objbytes((std::istreambuf_iterator<char>(rf)),
                              std::istreambuf_iterator<char>());

    ck::CookContext ctx{};
    ctx.asset_path = objp.string();
    CookedAsset c1 = cooker->cook(objbytes, ctx);
    CHECK(c1.type == AssetType::Mesh);
    CHECK(c1.cooker_version == 2u);
    CHECK(!c1.payload.empty());

    // Determinism: same source ⇒ byte-identical cooked blob (the
    // contract the hash-based incremental cook depends on).
    CookedAsset c2 = cooker->cook(objbytes, ctx);
    CHECK(c1.serialize() == c2.serialize());

    // And it survives the codec round-trip with type intact.
    CookedAsset back{};
    CHECK(CookedAsset::deserialize(c1.serialize(), back));
    CHECK(back.type == AssetType::Mesh);
    CHECK(back.payload == c1.payload);

    // (b) CARDMESH text path (no recognised extension ⇒ src is used).
    ck::CookContext tctx{};
    tctx.asset_path = (dir / "blob.bin").string();   // detect → Unknown
    const char* cm =
        "CARDMESH 1\n"
        "v 0 0 0 0 1 0 1 1 1\n"
        "v 1 0 0 0 1 0 1 1 1\n"
        "v 0 1 0 0 1 0 1 1 1\n"
        "f 0 1 2\n";
    std::vector<u8> cmbytes(reinterpret_cast<const u8*>(cm),
                            reinterpret_cast<const u8*>(cm) +
                            std::char_traits<char>::length(cm));
    CookedAsset c3 = cooker->cook(cmbytes, tctx);
    CHECK(c3.type == AssetType::Mesh);
    CHECK(!c3.payload.empty());

    // (c) Box fallback — unknown path + empty src still yields a Mesh.
    ck::CookContext bctx{};
    bctx.asset_path = (dir / "nope.bin").string();
    CookedAsset c4 = cooker->cook(std::vector<u8>{}, bctx);
    CHECK(c4.type == AssetType::Mesh);
    CHECK(!c4.payload.empty());            // always emits geometry
}

}  // namespace

int main() {
    std::error_code ec;
    std::filesystem::path dir =
        std::filesystem::temp_directory_path(ec) / "cardinal_cook_test";
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        cardinal::log::infof("cooktest",
            "SKIP  no writable temp dir in this environment");
        return 0;
    }

    test_codec();
    test_type_lookups();
    test_mesh_cooker(dir);

    std::error_code rec;
    const auto removed = std::filesystem::remove_all(dir, rec);
    (void)removed;

    if (g_fail == 0) {
        cardinal::log::infof("cooktest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("cooktest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
