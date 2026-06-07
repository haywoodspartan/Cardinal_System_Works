// =============================================================================
// Cardinal — deterministic asset-pipeline INTEGRATION suite.
//
// The per-module suites (asset_codec / cook / pack) each test one link
// in isolation. Nothing exercised the full shipping seam end to end:
//
//   MeshAsset --codec::encode--> payload
//             --cook::CookedAsset wrap + serialize--> cooked bytes
//             --pack::Builder.add + finalize--> .cpk on disk
//             --pack::Archive::open--> archive
//             --asset::Registry::mount_archive + load_mesh--> MeshAsset'
//
// This pins the cross-module contract: AssetType is shared cook↔pack↔
// asset, the archive key must survive verbatim to the Registry, the
// CookedAsset wrap/unwrap is transparent, and crucially the Registry
// dispatches on the DESERIALIZED CookedAsset.type — NOT the pack entry
// type (proved by setting them to disagree). Codec is a memcpy format,
// so the round-trip is bit-exact. Pure CPU, deterministic, headless
// (skips cleanly if the env has no writable temp dir). Exit 0 = pass.
// =============================================================================

#include <cardinal/asset/asset.hpp>
#include <cardinal/cook/cook.hpp>
#include <cardinal/pack/pack.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/std/utility.hpp>   // cardinal::move

#include <filesystem>

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
        cardinal::log::errorf("pipetest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool v3eq(const Vec3& a, const Vec3& b) {            // memcpy codec ⇒ exact
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

ck::CookedAsset wrap(ck::AssetType t, cardinal::vector<u8> payload) {
    ck::CookedAsset c;
    c.type    = t;
    c.payload = cardinal::move(payload);
    return c;
}

}  // namespace

int main() {
    std::error_code ec;
    std::filesystem::path dir =
        std::filesystem::temp_directory_path(ec) / "cardinal_pipeline_test";
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        cardinal::log::infof("pipetest",
            "SKIP  no writable temp dir in this environment");
        return 0;
    }
    const auto cpk = (dir / "shipped.cpk").string();

    // ---- Source assets — distinctive negatives/fractions so the
    // memcpy float codec is meaningfully bit-checked. ----------------
    as::MeshAsset src;
    {
        as::MeshVertex v0; v0.position={1.5f,-2.25f,3.0f};
        v0.normal={0,1,0};            v0.color={0.1f,0.2f,0.3f};
        as::MeshVertex v1; v1.position={-4.0f,5.5f,-6.75f};
        v1.normal={1,0,0};            v1.color={0.9f,0.8f,0.7f};
        as::MeshVertex v2; v2.position={0.125f,-0.0625f,7.5f};
        v2.normal={0,0,-1};           v2.color={0.4f,0.5f,0.6f};
        src.vertices = { v0, v1, v2 };
        src.indices  = { 0u,1u,2u, 2u,1u,0u };
    }
    as::TextureAsset stex;
    stex.width = 2; stex.height = 2; stex.channels = 4;
    stex.rgba = { 0,1,2,3,  250,251,252,253,  10,20,30,40,  99,98,97,96 };

    const cardinal::vector<u8> mesh_payload = as::codec::encode_mesh(src);
    const cardinal::vector<u8> tex_payload  = as::codec::encode_texture(stex);

    // ---- Build the shipped .cpk. DELIBERATELY mislabel the pack
    // entry types (Mesh entry tagged Unknown, Texture entry tagged
    // Mesh) — the Registry must ignore the pack entry type and use the
    // CookedAsset.type inside the bytes. ----------------------------
    {
        pk::Builder b(cpk);
        CHECK(b.add("meshes/probe",
                    ck::AssetType::Unknown,
                    wrap(ck::AssetType::Mesh, mesh_payload).serialize()));
        CHECK(b.add("tex/probe",
                    ck::AssetType::Mesh,
                    wrap(ck::AssetType::Texture, tex_payload).serialize()));
        cardinal::string err;
        CHECK(b.finalize(&err));
        CHECK(err.empty());
    }

    cardinal::string oerr;
    cardinal::shared_ptr<pk::Archive> arch = pk::Archive::open(cpk, &oerr);
    CHECK(arch != nullptr);
    if (!arch) {
        cardinal::log::errorf("pipetest", "archive open failed: %s",
                              oerr.c_str());
        return 1;
    }
    CHECK(arch->contains("meshes/probe") && arch->contains("tex/probe"));

    auto reg = as::Registry::create();
    CHECK(reg != nullptr);
    reg->mount_archive(arch);

    // ---- Mesh round-trips bit-exactly through the whole chain. -----
    auto m = reg->load_mesh("meshes/probe");
    CHECK(m != nullptr);
    if (m) {
        CHECK(m->vertices.size() == src.vertices.size());
        CHECK(m->indices == src.indices);
        bool verts_exact = (m->vertices.size() == src.vertices.size());
        for (usize i = 0; verts_exact && i < src.vertices.size(); ++i) {
            const auto& a = m->vertices[i];
            const auto& s = src.vertices[i];
            if (!v3eq(a.position, s.position) ||
                !v3eq(a.normal,   s.normal)   ||
                !v3eq(a.color,    s.color)) verts_exact = false;
        }
        CHECK(verts_exact);                          // bit-exact end to end
    }

    // Registry caches — second load is the SAME shared_ptr.
    auto m2 = reg->load_mesh("meshes/probe");
    CHECK(m2 == m);

    // Dispatch is on CookedAsset.type, NOT the (mislabeled) pack entry
    // type: the mesh entry's pack type was Unknown yet load_mesh works
    // (above); load_texture on it must fail (CookedAsset.type==Mesh).
    CHECK(reg->load_texture("meshes/probe") == nullptr);
    CHECK(reg->load_mesh("does/not/exist") == nullptr);

    // Texture round-trips; and its pack entry type was Mesh, yet
    // load_mesh must REFUSE it (CookedAsset.type==Texture governs).
    auto t = reg->load_texture("tex/probe");
    CHECK(t != nullptr);
    if (t) {
        CHECK(t->width == stex.width && t->height == stex.height &&
              t->channels == stex.channels);
        CHECK(t->rgba == stex.rgba);                 // bytewise exact
    }
    CHECK(reg->load_mesh("tex/probe") == nullptr);   // pack type ignored

    // load_raw unwraps the cook container transparently → exactly the
    // codec payload we encoded (cook header stripped end to end).
    cardinal::vector<u8> raw;
    CHECK(reg->load_raw("meshes/probe", raw));
    CHECK(raw == mesh_payload);

    std::error_code rec;
    std::filesystem::remove_all(dir, rec);

    if (g_fail == 0) {
        cardinal::log::infof("pipetest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("pipetest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
