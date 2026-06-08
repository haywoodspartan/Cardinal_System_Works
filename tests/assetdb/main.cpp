// =============================================================================
// Cardinal — Asset Database (cardinal::assetdb) regression suite.
//
// Builds a database from synthetic cook results + an in-memory material (so the
// dependency extraction has texture refs to follow), then exercises the query
// API (find / by_type / dependencies / dependents / missing) and the serialize
// + save/load round-trips. Headless, deterministic. Exit 0 = all pass.
// =============================================================================

#include <cardinal/assetdb/assetdb.hpp>
#include <cardinal/asset/asset.hpp>
#include <cardinal/cook/cook.hpp>
#include <cardinal/core/diag/log.hpp>

#include <filesystem>

namespace {

namespace adb = cardinal::assetdb;
namespace ck  = cardinal::cook;
namespace as  = cardinal::asset;

int g_checks = 0, g_fail = 0;
void check_impl(bool ok, const char* e, int l) {
    ++g_checks;
    if (!ok) { ++g_fail; cardinal::log::errorf("adbtest", "FAIL  L%d  %s", l, e); }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

ck::CookResult cooked(const char* src, ck::AssetType t, cardinal::u64 hash) {
    ck::CookResult r;
    r.status         = ck::CookResult::Status::Cooked;
    r.source_relpath = src;
    r.cooked_relpath = cardinal::string(src) + ".cooked";
    r.type           = t;
    r.source_hash    = hash;
    return r;
}

void test_build_and_query() {
    auto reg = as::Registry::create();
    as::MaterialAsset mat;
    mat.base_color_texture = "textures/metal_albedo.png";
    mat.normal_texture     = "textures/metal_normal.png";
    reg->register_material("materials/metal.mat", mat);

    cardinal::vector<ck::CookResult> results;
    results.push_back(cooked("materials/metal.mat",       ck::AssetType::Material, 111));
    results.push_back(cooked("textures/metal_albedo.png", ck::AssetType::Texture,  222));
    results.push_back(cooked("textures/metal_normal.png", ck::AssetType::Texture,  333));
    results.push_back(cooked("meshes/teapot.obj",         ck::AssetType::Mesh,     444));
    ck::CookResult bad;          // a failed cook must be skipped
    bad.status = ck::CookResult::Status::Failed;
    bad.source_relpath = "broken.png"; bad.cooked_relpath = "broken.png.cooked";
    bad.type = ck::AssetType::Texture;
    results.push_back(bad);

    adb::AssetDatabase db = adb::build_database(results, *reg);

    CHECK(db.size() == 4u);                                   // failed one skipped
    CHECK(db.find("materials/metal.mat") != nullptr);
    CHECK(db.find("broken.png") == nullptr);
    CHECK(db.type_of("meshes/teapot.obj") == ck::AssetType::Mesh);
    CHECK(db.by_type(ck::AssetType::Texture).size() == 2u);
    CHECK(db.find("materials/metal.mat")->id == adb::asset_id("materials/metal.mat"));
    CHECK(db.find_by_id(adb::asset_id("meshes/teapot.obj")) != nullptr);

    CHECK(db.dependencies("materials/metal.mat").size() == 2u);
    const auto dpt = db.dependents("textures/metal_albedo.png");
    CHECK(dpt.size() == 1u && dpt[0] == "materials/metal.mat");
    CHECK(db.missing_dependencies().empty());                 // both textures present
}

void test_missing_deps() {
    auto reg = as::Registry::create();
    as::MaterialAsset mat;
    mat.base_color_texture = "textures/ghost.png";           // referenced but NOT cooked
    reg->register_material("materials/m.mat", mat);

    cardinal::vector<ck::CookResult> results;
    results.push_back(cooked("materials/m.mat", ck::AssetType::Material, 1));

    auto db = adb::build_database(results, *reg);
    const auto miss = db.missing_dependencies();
    CHECK(miss.size() == 1u && miss[0] == "textures/ghost.png");
}

void test_serialize_roundtrip() {
    auto reg = as::Registry::create();
    as::MaterialAsset mat;
    mat.base_color_texture = "t/a.png";
    mat.normal_texture     = "t/n.png";
    reg->register_material("m/x.mat", mat);

    cardinal::vector<ck::CookResult> results;
    results.push_back(cooked("m/x.mat", ck::AssetType::Material, 7));
    results.push_back(cooked("t/a.png", ck::AssetType::Texture,  8));
    results.push_back(cooked("t/n.png", ck::AssetType::Texture,  9));
    auto db = adb::build_database(results, *reg);

    // serialize -> deserialize
    const cardinal::string text = db.serialize();
    adb::AssetDatabase db2;
    CHECK(adb::AssetDatabase::deserialize(text, db2));
    CHECK(db2.size() == db.size());
    CHECK(db2.find("m/x.mat") != nullptr);
    CHECK(db2.dependencies("m/x.mat").size() == 2u);
    CHECK(db2.find("t/a.png") != nullptr && db2.find("t/a.png")->source_hash == 8u);

    // save -> load (temp file)
    namespace fs = std::filesystem;
    const auto p = fs::temp_directory_path() / "cardinal_assetdb_test.db";
    CHECK(adb::save_database(db, p.string()));
    adb::AssetDatabase db3;
    CHECK(adb::load_database(p.string(), db3));
    CHECK(db3.size() == db.size());
    CHECK(db3.dependents("t/n.png").size() == 1u);
    std::error_code ec; fs::remove(p, ec);
}

}  // namespace

int main() {
    test_build_and_query();
    test_missing_deps();
    test_serialize_roundtrip();
    if (g_fail == 0) {
        cardinal::log::infof("adbtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("adbtest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
