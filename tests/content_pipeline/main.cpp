// =============================================================================
// Cardinal — deterministic AUTHORED-CONTENT pipeline integration suite.
//
// asset_pipeline pins codec→cook→pack→asset. This pins the link BEFORE
// the codec: a real on-disk source mesh through the importer + cooker
// and out the other end of the shipping chain —
//
//   tri.obj  --cook::make_mesh_cooker()->cook()-->  CookedAsset (Mesh)
//            --serialize--> bytes
//            --pack::Builder/.cpk + Archive::open-->  archive
//            --asset::Registry::mount_archive + load_mesh--> MeshAsset
//
// The seam invariants: the importer's geometry survives the whole
// import+cook+pack+load chain bit-exactly, the cook is deterministic
// (the contract the hash-based incremental cook depends on), the
// Registry path equals a direct codec decode of the cooked payload,
// and dispatch is on the CookedAsset.type (pack entry deliberately
// mislabeled). Pure CPU, deterministic, headless (skips with no temp
// dir). Exit 0 = pass.
// =============================================================================

#include <cardinal/asset/asset.hpp>
#include <cardinal/cook/cook.hpp>
#include <cardinal/pack/pack.hpp>
#include <cardinal/core/diag/log.hpp>

#include <filesystem>
#include <fstream>

namespace {

namespace as = cardinal::asset;
namespace ck = cardinal::cook;
namespace pk = cardinal::pack;
using cardinal::u8;
using cardinal::u32;
using cardinal::usize;
using Vec3 = cardinal::scene::Vec3;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("contenttest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool v3eq(const Vec3& a, const Vec3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

}  // namespace

int main() {
    std::error_code ec;
    std::filesystem::path dir =
        std::filesystem::temp_directory_path(ec) / "cardinal_content_test";
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        cardinal::log::infof("contenttest",
            "SKIP  no writable temp dir in this environment");
        return 0;
    }

    // ---- Author a deterministic single-triangle OBJ on disk. -------
    const auto objp = (dir / "tri.obj");
    {
        std::ofstream f(objp, std::ios::binary | std::ios::trunc);
        f << "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
    }
    std::ifstream in(objp, std::ios::binary | std::ios::ate);
    const std::streamsize n = in.tellg();
    in.seekg(0);
    cardinal::vector<u8> objbytes(static_cast<usize>(n));
    in.read(reinterpret_cast<char*>(objbytes.data()), n);
    in.close();
    CHECK(!objbytes.empty());

    // ---- Import + cook. --------------------------------------------
    auto cooker = ck::make_mesh_cooker();
    CHECK(cooker != nullptr);
    if (!cooker) return 1;
    CHECK(cooker->asset_type() == ck::AssetType::Mesh);

    ck::CookContext ctx{};
    ctx.asset_path = objp.string();                  // extension ⇒ OBJ importer
    ck::CookedAsset ca = cooker->cook(objbytes, ctx);
    CHECK(ca.type == ck::AssetType::Mesh);
    CHECK(!ca.payload.empty());

    // Deterministic cook (the incremental-cook hash contract).
    ck::CookedAsset ca2 = cooker->cook(objbytes, ctx);
    CHECK(ca.serialize() == ca2.serialize());

    // Direct codec decode of the cooked payload — the reference the
    // Registry path must match. The OBJ has 3 distinct verts + 1 face.
    as::MeshAsset direct;
    CHECK(as::codec::decode_mesh(ca.payload, direct));
    CHECK(direct.vertices.size() == 3u);
    CHECK(direct.indices.size()  == 3u);
    bool idx_ok = true;
    for (u32 i : direct.indices) if (i >= direct.vertices.size()) idx_ok = false;
    CHECK(idx_ok);
    // The 3 OBJ positions are all present (order is importer-defined;
    // assert as a set so we don't over-couple to dedup ordering).
    const Vec3 want[3] = { {0,0,0}, {1,0,0}, {0,1,0} };
    bool all_present = true;
    for (const auto& w : want) {
        int hits = 0;
        for (const auto& v : direct.vertices) if (v3eq(v.position, w)) ++hits;
        if (hits != 1) all_present = false;
    }
    CHECK(all_present);

    // ---- Pack it (pack entry type DELIBERATELY mislabeled) + ship
    // through Archive + Registry. ------------------------------------
    const auto cpk = (dir / "content.cpk").string();
    {
        pk::Builder b(cpk);
        CHECK(b.add("imported/tri", ck::AssetType::Unknown, ca.serialize()));
        cardinal::string err;
        CHECK(b.finalize(&err) && err.empty());
    }
    cardinal::string oerr;
    cardinal::shared_ptr<pk::Archive> arch = pk::Archive::open(cpk, &oerr);
    CHECK(arch != nullptr);
    if (!arch) { cardinal::log::errorf("contenttest", "open: %s", oerr.c_str());
                 return 1; }

    auto reg = as::Registry::create();
    reg->mount_archive(arch);
    auto m = reg->load_mesh("imported/tri");
    CHECK(m != nullptr);
    if (m) {
        // Registry path == direct codec decode of the cooked payload:
        // the import-origin geometry is transparent through pack+Registry.
        CHECK(m->vertices.size() == direct.vertices.size());
        CHECK(m->indices == direct.indices);
        bool same = (m->vertices.size() == direct.vertices.size());
        for (usize i = 0; same && i < direct.vertices.size(); ++i) {
            const auto& a = m->vertices[i];
            const auto& d = direct.vertices[i];
            if (!v3eq(a.position, d.position) ||
                !v3eq(a.normal,   d.normal)   ||
                !v3eq(a.color,    d.color)) same = false;
        }
        CHECK(same);
    }
    // Dispatch on CookedAsset.type (pack entry was Unknown): a wrong-
    // type / missing load returns nullptr.
    CHECK(reg->load_texture("imported/tri") == nullptr);
    CHECK(reg->load_mesh("imported/nope")   == nullptr);

    std::error_code rec;
    std::filesystem::remove_all(dir, rec);

    if (g_fail == 0) {
        cardinal::log::infof("contenttest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("contenttest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
