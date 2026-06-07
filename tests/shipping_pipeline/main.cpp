// =============================================================================
// Cardinal — deterministic FULL-SHIPPING-PIPELINE capstone suite.
//
// content_pipeline pinned source->cook->Builder->Registry for one
// in-memory blob. cook_incremental pinned the cook_all driver but
// stopped at the cooked/ tree. dist pinned pack::distribute() but from
// a HAND-SYNTHESISED cooked tree, read back via Archive only. Nothing
// spanned the entire REAL shipping chain with bytes the actual cooker
// produced:
//
//   assets/tri.obj
//     --cook::cook_all (real OBJ importer + MeshCooker)--> cooked/tri.obj.cooked
//     --pack::distribute (gathers the REAL cooked tree)--> dist/client.cpk
//     --pack::Archive::open--> archive
//     --asset::Registry::mount_archive + load_mesh--> MeshAsset
//
// Capstone invariants pinned: the importer's geometry survives the
// whole driver+distribute+pack+load chain bit-exactly; the cook type
// header round-trips through distribute() into the pack entry type the
// Registry dispatches on; load_raw unwraps the cook container to
// exactly the codec payload; and crucially DEV == SHIPPING — the SAME
// key resolves byte-identically whether the Registry is mounted on the
// real cooked/ directory (dev mode) or on the distributed .cpk
// (shipping). A regression anywhere in cook_all<->distribute<->Registry
// ships a broken client while every isolated suite stays green.
//
// Pure CPU, deterministic, headless (skips with no temp dir). Exit 0 = pass.
// =============================================================================

#include <cardinal/asset/asset.hpp>
#include <cardinal/cook/cook.hpp>
#include <cardinal/pack/pack.hpp>
#include <cardinal/project/project.hpp>
#include <cardinal/core/diag/log.hpp>

#include <filesystem>
#include <fstream>
#include <string>

namespace {

namespace as = cardinal::asset;
namespace ck = cardinal::cook;
namespace pk = cardinal::pack;
namespace pj = cardinal::project;
namespace fs = std::filesystem;
using cardinal::u8;
using cardinal::u32;
using cardinal::u64;
using cardinal::usize;
using Vec3 = cardinal::scene::Vec3;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("shiptest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

usize sz(int n) { return static_cast<usize>(n); }

bool v3eq(const Vec3& a, const Vec3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

cardinal::vector<u8> read_bytes(const fs::path& p) {
    cardinal::vector<u8> v;
    std::ifstream in(p, std::ios::binary | std::ios::ate);
    if (!in) return v;
    const std::streamsize n = in.tellg();
    if (n <= 0) return v;
    in.seekg(0);
    v.resize(static_cast<usize>(n));
    in.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

// Compare a loaded MeshAsset against the direct-codec reference: same
// vertex count, identical indices, and per-vertex exact (memcpy codec).
bool mesh_eq(const as::MeshAsset& m, const as::MeshAsset& ref) {
    if (m.vertices.size() != ref.vertices.size()) return false;
    if (m.indices != ref.indices) return false;
    for (usize i = 0; i < ref.vertices.size(); ++i) {
        const auto& a = m.vertices[i];
        const auto& r = ref.vertices[i];
        if (!v3eq(a.position, r.position) ||
            !v3eq(a.normal,   r.normal)   ||
            !v3eq(a.color,    r.color)) return false;
    }
    return true;
}

}  // namespace

int main() {
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec);
    if (ec) {
        cardinal::log::infof("shiptest",
            "SKIP  no writable temp dir in this environment");
        return 0;
    }
    const fs::path rootp = base / "cardinal_shipping_test";
    fs::remove_all(rootp, ec);
    const std::string root = rootp.string();

    // ---- A real project tree on disk. ------------------------------
    cardinal::string perr;
    auto proj = pj::Project::create_at(root, pj::ProjectInfo{}, &perr);
    CHECK(proj != nullptr);
    if (!proj) {
        cardinal::log::errorf("shiptest", "create_at: %s", perr.c_str());
        return 1;
    }
    const fs::path assets = fs::path(proj->dirs().assets);
    const fs::path cooked = fs::path(proj->dirs().cooked);

    // Deterministic single triangle — 3 distinct verts, 1 face.
    {
        std::ofstream f(assets / "tri.obj", std::ios::binary | std::ios::trunc);
        f << "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
    }

    // ---- Cook the project (real importer + MeshCooker). ------------
    ck::CookerRegistry reg;
    reg.register_builtin();
    CHECK(reg.find_for_extension(".obj") != nullptr);

    cardinal::vector<ck::CookResult> cres;
    ck::CookSummary cs = ck::cook_all(*proj, reg, cres, false);
    CHECK(cs.cooked_count == 1u);
    CHECK(cs.failed_count == 0u);
    const ck::CookResult* tri = nullptr;
    for (const auto& r : cres) if (r.source_relpath == "tri.obj") tri = &r;
    CHECK(tri != nullptr);
    if (tri) {
        CHECK(tri->status == ck::CookResult::Status::Cooked);
        CHECK(tri->type   == ck::AssetType::Mesh);
    }
    const fs::path cooked_blob = cooked / "tri.obj.cooked";
    CHECK(fs::exists(cooked_blob));

    // ---- Reference: direct codec decode of the real cooked payload.
    // This is what every shipped path must reproduce bit-for-bit.
    ck::CookedAsset cooked_ca;
    {
        const cardinal::vector<u8> wrapped = read_bytes(cooked_blob);
        CHECK(!wrapped.empty());
        CHECK(ck::CookedAsset::deserialize(wrapped, cooked_ca));
        CHECK(cooked_ca.type == ck::AssetType::Mesh);
    }
    as::MeshAsset ref;
    CHECK(as::codec::decode_mesh(cooked_ca.payload, ref));
    CHECK(ref.vertices.size() == 3u);
    CHECK(ref.indices.size()  == 3u);
    {
        const Vec3 want[3] = { {0,0,0}, {1,0,0}, {0,1,0} };
        bool all_present = true;
        for (const auto& w : want) {
            int hits = 0;
            for (const auto& v : ref.vertices) if (v3eq(v.position, w)) ++hits;
            if (hits != 1) all_present = false;
        }
        CHECK(all_present);
    }

    // ---- Distribute the cooked project into a shippable .cpk. ------
    pk::DistOptions opt;
    opt.project_root  = root;
    opt.out_dir       = (rootp / "dist").string();
    opt.pack_name     = "client";
    opt.include_saves = false;
    pk::DistResult dr = pk::distribute(opt);
    CHECK(dr.ok);
    CHECK(dr.error.empty());
    CHECK(dr.asset_count >= 1u);
    CHECK(dr.save_count  == 0u);
    CHECK(!dr.pack_path.empty()      && fs::exists(fs::path(dr.pack_path)));
    CHECK(!dr.manifest_path.empty()  && fs::exists(fs::path(dr.manifest_path)));

    // ---- Open the shipped pack. The cooked-relative key (sans
    // ".cooked") is exactly what the dev-mode directory mount resolves;
    // distribute typed the entry from the cooked header.
    cardinal::string oerr;
    cardinal::shared_ptr<pk::Archive> arch = pk::Archive::open(dr.pack_path, &oerr);
    CHECK(arch != nullptr);
    if (!arch) {
        cardinal::log::errorf("shiptest", "archive open: %s", oerr.c_str());
        return 1;
    }
    CHECK(arch->contains("tri.obj"));
    const pk::PackEntryDesc* ed = arch->find("tri.obj");
    CHECK(ed != nullptr);
    if (ed) CHECK(ed->type == ck::AssetType::Mesh);   // cook header round-trip

    // ---- SHIPPING path: Registry on the distributed archive. -------
    auto regArc = as::Registry::create();
    regArc->mount_archive(arch);
    auto mArc = regArc->load_mesh("tri.obj");
    CHECK(mArc != nullptr);
    if (mArc) CHECK(mesh_eq(*mArc, ref));              // bit-exact end to end

    // Dispatch is on the deserialized CookedAsset.type: load_texture on
    // a Mesh must refuse; a missing key is null.
    CHECK(regArc->load_texture("tri.obj")     == nullptr);
    CHECK(regArc->load_mesh("does/not/exist") == nullptr);

    // load_raw unwraps the cook container → exactly the codec payload
    // (cook header stripped through distribute + pack + Registry).
    cardinal::vector<u8> raw;
    CHECK(regArc->load_raw("tri.obj", raw));
    CHECK(raw == cooked_ca.payload);

    // ---- DEV path: Registry on the real cooked/ directory. The SAME
    // key must resolve to byte-identical geometry — dev and shipping
    // are key- AND byte-identical (the contract distribute() promises).
    auto regDir = as::Registry::create();
    regDir->mount_directory(proj->dirs().cooked);
    auto mDir = regDir->load_mesh("tri.obj");
    CHECK(mDir != nullptr);
    if (mDir) CHECK(mesh_eq(*mDir, ref));
    if (mArc && mDir) {
        CHECK(mDir->vertices.size() == mArc->vertices.size());
        CHECK(mDir->indices == mArc->indices);
        bool same = mDir->vertices.size() == mArc->vertices.size();
        for (usize i = 0; same && i < mArc->vertices.size(); ++i) {
            const auto& a = mArc->vertices[i];
            const auto& d = mDir->vertices[i];
            if (!v3eq(a.position, d.position) ||
                !v3eq(a.normal,   d.normal)   ||
                !v3eq(a.color,    d.color)) same = false;
        }
        CHECK(same);                                  // dev == shipping
    }
    cardinal::vector<u8> raw_dir;
    CHECK(regDir->load_raw("tri.obj", raw_dir));
    CHECK(raw_dir == raw);                            // identical unwrapped

    fs::remove_all(rootp, ec);

    if (g_fail == 0) {
        cardinal::log::infof("shiptest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("shiptest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
