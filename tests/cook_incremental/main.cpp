// =============================================================================
// Cardinal — deterministic INCREMENTAL-COOK driver suite.
//
// The cook suite tests Cookers in isolation and explicitly defers the
// driver: "Out of scope by design: cook_all/Driver (needs a
// project::Project tree)". Nothing exercised cook::cook_all + the
// hash manifest — the contract that decides, on every editor save and
// every shipping build, whether an asset is re-baked or reused. A
// regression here either re-cooks the whole project every time (slow)
// or ships STALE cooked bytes (silent corruption) while every unit
// suite stays green.
//
// This pins the driver END TO END through a real project::Project
// tree: first cook bakes + writes manifest; an unchanged second cook
// skips everything; editing one source re-cooks only that file;
// deleting a cooked output re-cooks it even on a manifest hash hit;
// force=true rebuilds unconditionally; the manifest round-trips on
// disk. Re-cooks of unchanged sources are byte-identical (driver-level
// determinism).
//
// HONEST CHARACTERIZATION — the implemented skip predicate is exactly
//   !force && manifest[rel] == fnv1a(source) && fs::exists(cooked)
// (cook.cpp). The header comment claims "source hash + cooker version
// match", but cooker_version is NOT persisted in CookManifest and is
// NOT consulted in the skip path: a cooker-version bump alone does NOT
// invalidate. This suite pins the REAL behavior; the doc/impl
// divergence is flagged separately, not papered over here.
//
// Pure CPU, deterministic, headless (skips with no temp dir). Exit 0 = pass.
// =============================================================================

#include <cardinal/cook/cook.hpp>
#include <cardinal/project/project.hpp>
#include <cardinal/core/log.hpp>

#include <filesystem>
#include <fstream>
#include <string>

namespace {

namespace ck = cardinal::cook;
namespace pj = cardinal::project;
namespace fs = std::filesystem;
using cardinal::u8;
using cardinal::u32;
using cardinal::u64;
using cardinal::usize;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("cookincr", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

usize sz(int n) { return static_cast<usize>(n); }

bool write_file(const fs::path& p, const std::string& text) {
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(text.data(), static_cast<std::streamsize>(text.size()));
    return true;
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

const ck::CookResult* find_r(const cardinal::vector<ck::CookResult>& v,
                             const char* rel) {
    for (const auto& r : v) if (r.source_relpath == rel) return &r;
    return nullptr;
}

bool is(const ck::CookResult* r, ck::CookResult::Status s) {
    return r != nullptr && r->status == s;
}

}  // namespace

int main() {
    using Status = ck::CookResult::Status;

    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec);
    if (ec) {
        cardinal::log::infof("cookincr",
            "SKIP  no writable temp dir in this environment");
        return 0;
    }
    const fs::path rootp = base / "cardinal_cook_incr";
    fs::remove_all(rootp, ec);
    const std::string root = rootp.string();

    cardinal::string perr;
    auto proj = pj::Project::create_at(root, pj::ProjectInfo{}, &perr);
    CHECK(proj != nullptr);
    if (!proj) {
        cardinal::log::errorf("cookincr", "create_at failed: %s",
                              perr.c_str());
        return 1;
    }
    const fs::path assets = fs::path(proj->dirs().assets);
    const fs::path cooked = fs::path(proj->dirs().cooked);
    const fs::path man    = cooked / "manifest.cook";

    // Deterministic sources. a/b are distinct triangles (distinct
    // bytes ⇒ distinct hashes ⇒ distinct cooked blobs). readme.txt has
    // no registered cooker — the no-cooker Skip branch.
    const std::string A1 = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
    const std::string B1 = "v 0 0 0\nv 2 0 0\nv 0 2 0\nf 1 2 3\n";
    const std::string A2 = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\n"
                           "f 1 2 3\nf 2 4 3\n";
    CHECK(write_file(assets / "a.obj",    A1));
    CHECK(write_file(assets / "b.obj",    B1));
    CHECK(write_file(assets / "readme.txt", "not an asset\n"));

    ck::CookerRegistry reg;
    reg.register_builtin();                       // no shader compiler needed
    CHECK(reg.find_for_extension(".obj") != nullptr);
    CHECK(reg.find_for_extension(".txt") == nullptr);

    // ---- Phase 1: first cook — both meshes baked, txt no-cooker. ----
    cardinal::vector<ck::CookResult> r1;
    ck::CookSummary s1 = ck::cook_all(*proj, reg, r1, false);
    CHECK(s1.cooked_count  == 2u);
    CHECK(s1.skipped_count == 1u);
    CHECK(s1.failed_count  == 0u);
    CHECK(s1.total_payload_bytes > 0u);
    CHECK(r1.size() == sz(3));

    const ck::CookResult* a1 = find_r(r1, "a.obj");
    const ck::CookResult* b1 = find_r(r1, "b.obj");
    const ck::CookResult* x1 = find_r(r1, "readme.txt");
    CHECK(is(a1, Status::Cooked));
    CHECK(is(b1, Status::Cooked));
    CHECK(is(x1, Status::Skipped));
    if (a1) {
        CHECK(a1->type == ck::AssetType::Mesh);
        CHECK(a1->source_hash != 0u);
        CHECK(a1->cooked_relpath == "a.obj.cooked");
        CHECK(a1->payload_bytes > 0u);
    }
    if (x1) {
        // No-cooker skip: type/hash untouched, error names the ext.
        CHECK(x1->type == ck::AssetType::Unknown);
        CHECK(x1->source_hash == 0u);
        CHECK(x1->error_message.find("no cooker") != cardinal::string::npos);
        CHECK(x1->cooked_relpath.empty());
    }
    CHECK(fs::exists(cooked / "a.obj.cooked"));
    CHECK(fs::exists(cooked / "b.obj.cooked"));
    CHECK(fs::exists(man));
    CHECK(!fs::exists(cooked / "readme.txt.cooked"));

    const cardinal::vector<u8> a_cook_1 = read_bytes(cooked / "a.obj.cooked");
    const cardinal::vector<u8> b_cook_1 = read_bytes(cooked / "b.obj.cooked");
    const u64 a_hash_1 = a1 ? a1->source_hash : 0u;
    const u64 b_hash_1 = b1 ? b1->source_hash : 0u;
    CHECK(!a_cook_1.empty());
    CHECK(!b_cook_1.empty());
    CHECK(a_cook_1 != b_cook_1);                   // distinct src ⇒ distinct
    CHECK(a_hash_1 != 0u && b_hash_1 != 0u && a_hash_1 != b_hash_1);

    // ---- Phase 2: unchanged second cook — everything skips. --------
    cardinal::vector<ck::CookResult> r2;
    ck::CookSummary s2 = ck::cook_all(*proj, reg, r2, false);
    CHECK(s2.cooked_count  == 0u);
    CHECK(s2.skipped_count == 3u);                 // a + b (manifest) + txt
    CHECK(s2.failed_count  == 0u);
    const ck::CookResult* a2 = find_r(r2, "a.obj");
    const ck::CookResult* b2 = find_r(r2, "b.obj");
    CHECK(is(a2, Status::Skipped));
    CHECK(is(b2, Status::Skipped));
    if (a2) {
        // Manifest-skip results still carry hash + cooked path (the
        // hash is recomputed before the skip check in cook.cpp).
        CHECK(a2->source_hash == a_hash_1);
        CHECK(a2->cooked_relpath == "a.obj.cooked");
    }
    CHECK(read_bytes(cooked / "a.obj.cooked") == a_cook_1);   // untouched
    CHECK(read_bytes(cooked / "b.obj.cooked") == b_cook_1);

    // ---- Phase 3: edit a.obj only — selective re-cook. -------------
    CHECK(write_file(assets / "a.obj", A2));
    cardinal::vector<ck::CookResult> r3;
    ck::CookSummary s3 = ck::cook_all(*proj, reg, r3, false);
    CHECK(s3.cooked_count  == 1u);
    CHECK(s3.skipped_count == 2u);                 // b (manifest) + txt
    const ck::CookResult* a3 = find_r(r3, "a.obj");
    const ck::CookResult* b3 = find_r(r3, "b.obj");
    CHECK(is(a3, Status::Cooked));
    CHECK(is(b3, Status::Skipped));
    if (a3) CHECK(a3->source_hash != a_hash_1);    // hash moved with bytes
    const cardinal::vector<u8> a_cook_3 = read_bytes(cooked / "a.obj.cooked");
    CHECK(!a_cook_3.empty());
    CHECK(a_cook_3 != a_cook_1);                    // re-baked
    CHECK(read_bytes(cooked / "b.obj.cooked") == b_cook_1);   // b untouched

    // ---- Phase 4: delete b's cooked output, keep manifest hash. ----
    // Manifest hash still matches b's (unchanged) source, but the
    // fs::exists(cooked) clause must force a re-cook anyway.
    CHECK(fs::remove(cooked / "b.obj.cooked", ec));
    cardinal::vector<ck::CookResult> r4;
    ck::CookSummary s4 = ck::cook_all(*proj, reg, r4, false);
    CHECK(s4.cooked_count  == 1u);
    CHECK(s4.skipped_count == 2u);                 // a (manifest) + txt
    CHECK(is(find_r(r4, "b.obj"), Status::Cooked)); // re-cooked: output gone
    CHECK(is(find_r(r4, "a.obj"), Status::Skipped));// unchanged + present
    CHECK(fs::exists(cooked / "b.obj.cooked"));
    // Deterministic: re-cook of the unchanged b source == original bytes.
    CHECK(read_bytes(cooked / "b.obj.cooked") == b_cook_1);

    // ---- Phase 5: force=true — unconditional rebuild. --------------
    cardinal::vector<ck::CookResult> r5;
    ck::CookSummary s5 = ck::cook_all(*proj, reg, r5, true);
    CHECK(s5.cooked_count  == 2u);                 // a + b regardless
    CHECK(s5.skipped_count == 1u);                 // txt still no cooker
    CHECK(s5.failed_count  == 0u);
    const ck::CookResult* a5 = find_r(r5, "a.obj");
    const ck::CookResult* b5 = find_r(r5, "b.obj");
    CHECK(is(a5, Status::Cooked));
    CHECK(is(b5, Status::Cooked));
    CHECK(read_bytes(cooked / "a.obj.cooked") == a_cook_3); // determinism
    CHECK(read_bytes(cooked / "b.obj.cooked") == b_cook_1);

    // ---- Live manifest reflects exactly the cooked set. ------------
    ck::CookManifest live;
    CHECK(live.load_from(man.string()));
    CHECK(live.hashes.size() == sz(2));            // a + b only; txt absent
    CHECK(live.hashes.find("a.obj") != live.hashes.end());
    CHECK(live.hashes.find("b.obj") != live.hashes.end());
    CHECK(live.hashes.find("readme.txt") == live.hashes.end());
    if (a5) CHECK(live.hashes["a.obj"] == a5->source_hash);
    if (b5) {
        CHECK(live.hashes["b.obj"] == b5->source_hash);
        CHECK(b5->source_hash == b_hash_1);        // b never changed
    }

    // ---- CookManifest on-disk format round-trips (incl. full
    // 64-bit / high-bit value, '/'-bearing key) + missing-file clear.
    {
        ck::CookManifest m;
        m.hashes["dir/sub/mesh.obj"] = 0xdeadbeefcafef00dull;
        m.hashes["tex.png"]          = 0x1ull;
        const std::string mp = (rootp / "rt.cook").string();
        CHECK(m.save_to(mp));
        ck::CookManifest m2;
        CHECK(m2.load_from(mp));
        CHECK(m2.hashes.size() == sz(2));
        CHECK(m2.hashes["dir/sub/mesh.obj"] == 0xdeadbeefcafef00dull);
        CHECK(m2.hashes["tex.png"]          == 0x1ull);
        // Missing file ⇒ false AND hashes cleared (load_from contract).
        const std::string nope = (rootp / "absent.cook").string();
        CHECK(!m2.load_from(nope));
        CHECK(m2.hashes.empty());
    }

    fs::remove_all(rootp, ec);

    if (g_fail == 0) {
        cardinal::log::infof("cookincr", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("cookincr", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
