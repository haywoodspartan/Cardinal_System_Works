// =============================================================================
// Cardinal — end-to-end client-distribution integration suite.
//
// The capstone of the asset-pipeline arc. It composes the pieces the
// previous suites locked — CookedAsset::serialize() to synthesise a
// cooked project tree, and Archive to read the shipped result — to
// prove pack::distribute() turns a cooked project into a correct,
// loadable client bundle:
//
//   - every cooked/**.cooked → a pack entry keyed by its cooked-
//     relative path minus ".cooked" (generic '/'), typed from the
//     CookedAsset header
//   - non-.cooked files ignored; invalid .cooked kept verbatim but
//     typed Unknown (graceful, not a hard fail)
//   - save/** bundled under save/<rel> (nested paths preserved)
//   - extra_files staged next to the pack
//   - distribution.cardinal manifest the launcher validates
//   - idempotent re-runs; include_saves=false; missing-cooked failure
//
// Deterministic + headless (temp dirs only). Exit 0 = all pass.
// =============================================================================

#include <cardinal/pack/pack.hpp>
#include <cardinal/cook/cook.hpp>
#include <cardinal/core/log.hpp>

#include <filesystem>
#include <fstream>
#include <vector>

namespace {

namespace pk = cardinal::pack;
namespace ck = cardinal::cook;
namespace fs = std::filesystem;
using cardinal::u8;
using cardinal::u32;
using cardinal::u64;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("disttest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

std::vector<u8> blob(u32 seed, cardinal::usize n) {
    std::vector<u8> v; v.reserve(n);
    u32 x = seed ? seed : 1u;
    for (cardinal::usize i = 0; i < n; ++i) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        v.push_back(static_cast<u8>(x & 0xFFu));
    }
    return v;
}

void write_bytes(const fs::path& p, const std::vector<u8>& b) {
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(b.data()),
            static_cast<std::streamsize>(b.size()));
}

std::vector<u8> read_bytes(const fs::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<u8> v(static_cast<cardinal::usize>(n < 0 ? 0 : n));
    if (n > 0) f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

cardinal::string read_text(const fs::path& p) {
    const std::vector<u8> b = read_bytes(p);
    return cardinal::string(b.begin(), b.end());
}

std::vector<u8> cooked(ck::AssetType t, const std::vector<u8>& payload) {
    ck::CookedAsset ca{};
    ca.type = t;
    ca.cooker_version = 3u;
    ca.payload = payload;
    return ca.serialize();
}

}  // namespace

int main() {
    std::error_code ec;
    const fs::path base =
        fs::temp_directory_path(ec) / "cardinal_dist_test";
    fs::remove_all(base, ec);
    fs::create_directories(base, ec);
    if (ec) {
        cardinal::log::infof("disttest",
            "SKIP  no writable temp dir in this environment");
        return 0;
    }

    const fs::path root  = base / "project";
    const fs::path outd  = base / "out";
    const fs::path stage = base / "stage";

    // ---- synthesise a cooked project tree ---------------------------
    const std::vector<u8> pW = blob(0x11u, 200);   // texture payload
    const std::vector<u8> pT = blob(0x22u, 512);   // mesh payload
    const std::vector<u8> pS = blob(0x33u, 64);    // shader payload
    const std::vector<u8> woodC = cooked(ck::AssetType::Texture, pW);
    const std::vector<u8> teapC = cooked(ck::AssetType::Mesh,    pT);
    const std::vector<u8> litC  = cooked(ck::AssetType::Shader,  pS);
    const std::vector<u8> garbage = blob(0x44u, 99);   // NOT a CookedAsset

    write_bytes(root / "cooked" / "textures" / "wood.cooked",  woodC);
    write_bytes(root / "cooked" / "meshes"   / "teapot.cooked", teapC);
    write_bytes(root / "cooked" / "shaders"  / "lit.cooked",   litC);
    write_bytes(root / "cooked" / "raw"      / "garbage.cooked", garbage);
    write_bytes(root / "cooked" / "readme.txt",
                std::vector<u8>{ 'h','i' });          // must be IGNORED

    const std::vector<u8> saveL = blob(0x55u, 128);   // world save
    const std::vector<u8> saveQ = blob(0x66u, 48);    // nested sky save
    write_bytes(root / "save" / "level1.world",        saveL);
    write_bytes(root / "save" / "sub" / "quick.sky",   saveQ);

    const std::vector<u8> exeBytes = blob(0x77u, 256);
    write_bytes(stage / "game.exe", exeBytes);

    const u64 expect_bytes =
        static_cast<u64>(woodC.size() + teapC.size() + litC.size() +
                         garbage.size() + saveL.size() + saveQ.size());

    // ---- distribute -------------------------------------------------
    pk::DistOptions opt;
    opt.project_root   = root.string();
    opt.out_dir        = outd.string();
    opt.pack_name      = "client";
    opt.include_saves  = true;
    opt.extra_files    = { (stage / "game.exe").string() };
    opt.app_name       = "DistTestGame";
    opt.engine_version = "9.9.9";

    pk::DistResult r = pk::distribute(opt);
    CHECK(r.ok);
    CHECK(r.error.empty());
    CHECK(r.asset_count == 4u);          // 4 .cooked (readme.txt ignored)
    CHECK(r.save_count  == 2u);
    CHECK(r.extra_count == 1u);
    CHECK(r.total_bytes == expect_bytes);
    CHECK(!r.pack_path.empty());
    CHECK(!r.manifest_path.empty());
    CHECK(fs::exists(r.pack_path));
    CHECK(fs::exists(r.manifest_path));

    // ---- the shipped pack is correct + loadable ---------------------
    {
        cardinal::string err;
        auto a = pk::Archive::open(r.pack_path, &err);
        CHECK(a != nullptr);
        if (a) {
            CHECK(a->entry_count() == 6u);     // 4 assets + 2 saves

            // Keys are cooked-relative, '.cooked' stripped, '/'-joined.
            CHECK(a->contains("textures/wood"));
            CHECK(a->contains("meshes/teapot"));
            CHECK(a->contains("shaders/lit"));
            CHECK(a->contains("raw/garbage"));
            CHECK(a->contains("save/level1.world"));
            CHECK(a->contains("save/sub/quick.sky"));
            // The non-.cooked source never entered the pack.
            CHECK(!a->contains("readme.txt"));
            CHECK(!a->contains("readme"));

            // Types are read back from each CookedAsset header.
            const auto* ew = a->find("textures/wood");
            const auto* et = a->find("meshes/teapot");
            const auto* es = a->find("shaders/lit");
            const auto* eg = a->find("raw/garbage");
            const auto* ev = a->find("save/level1.world");
            CHECK(ew && ew->type == ck::AssetType::Texture);
            CHECK(et && et->type == ck::AssetType::Mesh);
            CHECK(es && es->type == ck::AssetType::Shader);
            CHECK(eg && eg->type == ck::AssetType::Unknown);  // invalid blob
            CHECK(ev && ev->type == ck::AssetType::Unknown);  // saves untyped

            // Payloads are the verbatim files; the cooked ones still
            // decode, proving the pipeline preserved them end to end.
            std::vector<u8> got;
            CHECK(a->load_blocking("textures/wood", got));
            CHECK(got == woodC);
            ck::CookedAsset deco;
            CHECK(ck::CookedAsset::deserialize(got, deco));
            CHECK(deco.type == ck::AssetType::Texture);
            CHECK(deco.payload == pW);

            CHECK(a->load_blocking("raw/garbage", got));
            CHECK(got == garbage);
            ck::CookedAsset none;
            CHECK(!ck::CookedAsset::deserialize(got, none));   // still junk

            CHECK(a->load_blocking("save/sub/quick.sky", got));
            CHECK(got == saveQ);
        }
    }

    // ---- manifest the launcher reads --------------------------------
    {
        const cardinal::string m = read_text(r.manifest_path);
        CHECK(m.find("app = \"DistTestGame\"")  != cardinal::string::npos);
        CHECK(m.find("engine = \"9.9.9\"")      != cardinal::string::npos);
        CHECK(m.find("pack = \"client.cpk\"")   != cardinal::string::npos);
        CHECK(m.find("assets = 4")              != cardinal::string::npos);
        CHECK(m.find("saves = 2")               != cardinal::string::npos);
        CHECK(m.find("extra = 1")               != cardinal::string::npos);
        CHECK(m.find("hash = 0x")               != cardinal::string::npos);
        CHECK(m.find("built = ")                != cardinal::string::npos);
    }

    // ---- extra runtime file staged next to the pack -----------------
    {
        const fs::path staged = outd / "game.exe";
        CHECK(fs::exists(staged));
        CHECK(read_bytes(staged) == exeBytes);
    }

    // ---- idempotent re-run ------------------------------------------
    {
        pk::DistResult r2 = pk::distribute(opt);
        CHECK(r2.ok);
        CHECK(r2.asset_count == 4u);
        CHECK(r2.save_count == 2u);
        auto a = pk::Archive::open(r2.pack_path);
        CHECK(a != nullptr);
        if (a) CHECK(a->entry_count() == 6u);
    }

    // ---- include_saves = false --------------------------------------
    {
        pk::DistOptions o2 = opt;
        o2.out_dir       = (base / "out_nosave").string();
        o2.include_saves = false;
        pk::DistResult r2 = pk::distribute(o2);
        CHECK(r2.ok);
        CHECK(r2.save_count == 0u);
        CHECK(r2.asset_count == 4u);
        auto a = pk::Archive::open(r2.pack_path);
        CHECK(a != nullptr);
        if (a) {
            CHECK(a->entry_count() == 4u);
            CHECK(!a->contains("save/level1.world"));
        }
    }

    // ---- failure: project without a cooked/ dir ---------------------
    {
        const fs::path empty_root = base / "empty_project";
        fs::create_directories(empty_root, ec);
        pk::DistOptions o3;
        o3.project_root = empty_root.string();
        o3.out_dir      = (base / "out_fail").string();
        pk::DistResult r3 = pk::distribute(o3);
        CHECK(!r3.ok);
        CHECK(!r3.error.empty());
        CHECK(r3.error.find("cooked") != cardinal::string::npos);
    }

    std::error_code rec;
    const auto removed = fs::remove_all(base, rec);
    (void)removed;

    if (g_fail == 0) {
        cardinal::log::infof("disttest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("disttest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
