// =============================================================================
// Cardinal — deterministic project-module regression suite.
//
// project::Project is the Studio create/open/save backbone: a
// `project.cardinal` text manifest + a fixed root→subdir layout. Pinned:
//   * template_kind name/description tables (+ invalid);
//   * ProjectInfo defaults (NOTE: header comments engine_version "default
//     0.1.0" but the member has no initializer and nothing sets it — the
//     actual default is "" ; this suite pins the IMPLEMENTED behaviour
//     and the mismatch is flagged separately);
//   * create_at → manifest + dir tree; the root→src/assets/cooked/pack/
//     shaders/shaders​cache/save derivation (literal "/"-joined, so
//     bit-exact regardless of platform);
//   * open() round-trips every ProjectInfo field — incl. quote-strip,
//     whitespace trim, FIRST-'=' split (so '=' survives inside values),
//     and bool parse (== "true"); missing manifest → null + error;
//   * save() rewrites; reopen reflects mutations;
//   * list_source/cooked_assets (sorted, .cooked filter);
//   * instantiate_template drops the scaffold files; non-empty target
//     rejected unless overwrite;
//   * RecentProjects most-recent-first + dedupe + cap-16 + text round-trip.
// Disk round-trips run through a unique std::filesystem temp dir the
// suite creates and removes. Exit 0 = all pass.
// =============================================================================

#include <cardinal/project/project.hpp>
#include <cardinal/core/log.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace pj = cardinal::project;
namespace fs = std::filesystem;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("projtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool streq(const char* a, const char* b) { return std::string(a) == b; }
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }
bool listed(const std::vector<std::string>& v, const char* needle) {
    for (const auto& s : v)
        if (s.find(needle) != std::string::npos) return true;
    return false;
}

// A fresh, empty temp directory path (removed if it already existed).
fs::path tmp_root(const char* tag) {
    fs::path p = fs::temp_directory_path() / (std::string("cardinal_proj_") + tag);
    std::error_code ec;
    fs::remove_all(p, ec);
    return p;
}
void rm(const fs::path& p) { std::error_code ec; fs::remove_all(p, ec); }
bool write_file(const fs::path& p, const std::string& text) {
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(text.data(), static_cast<std::streamsize>(text.size()));
    return true;
}

// ---- template kind tables -----------------------------------------
void test_template_kinds() {
    using K = pj::TemplateKind;
    CHECK(streq(pj::template_kind_name(K::Blank),       "Blank"));
    CHECK(streq(pj::template_kind_name(K::FirstPerson), "First Person"));
    CHECK(streq(pj::template_kind_name(K::TopDown),     "Top-Down"));
    CHECK(streq(pj::template_kind_name(K::Cinematic),   "Cinematic"));
    CHECK(streq(pj::template_kind_name(static_cast<K>(99u)), "?"));
    CHECK(std::string(pj::template_kind_description(K::Blank)).size() > 0);
    CHECK(std::string(pj::template_kind_description(K::Cinematic)).size() > 0);
    CHECK(streq(pj::template_kind_description(static_cast<K>(99u)), ""));
}

// ---- ProjectInfo defaults + manifest constants --------------------
void test_info_defaults() {
    pj::ProjectInfo i{};
    CHECK(i.name.empty());
    // Header comment claims "0.1.0" but there is no initializer and
    // nothing assigns it — actual default is empty. (Mismatch flagged.)
    CHECK(i.engine_version.empty());
    CHECK(i.author.empty() && i.description.empty());
    CHECK(i.default_pack_name == "main");
    CHECK(i.cook_on_save == true);
    CHECK(i.pack_on_cook == true);
    CHECK(streq(pj::kManifestFilename, "project.cardinal"));
    CHECK(streq(pj::kManifestMagic,    "# Cardinal project v1"));
}

// ---- create_at → manifest + dirs ; open() round-trip --------------
void test_create_open_roundtrip() {
    const fs::path rootp = tmp_root("rt");
    const std::string root = rootp.string();

    pj::ProjectInfo info;
    info.name           = "My Game";
    info.engine_version = "0.2.1";
    info.author         = "me";
    info.description    = "a desc with = sign";   // '=' must survive
    info.default_pack_name = "lvls";
    info.cook_on_save   = false;
    info.pack_on_cook   = true;

    std::string err;
    auto p = pj::Project::create_at(root, info, &err);
    CHECK(p != nullptr);
    if (p) {
        CHECK(p->info().name == "My Game");
        // Root→subdir derivation is literal "/"-joined string concat.
        const auto& d = p->dirs();
        CHECK(d.root         == root);
        CHECK(d.src          == root + "/src");
        CHECK(d.assets       == root + "/assets");
        CHECK(d.cooked       == root + "/cooked");
        CHECK(d.pack         == root + "/pack");
        CHECK(d.shaders      == root + "/shaders");
        CHECK(d.shader_cache == root + "/shaders/cache");
        CHECK(d.save         == root + "/save");
        std::error_code ec;
        CHECK(fs::exists(rootp / "project.cardinal", ec));
        CHECK(fs::exists(fs::path(d.src), ec));
        CHECK(fs::exists(fs::path(d.shader_cache), ec));
    }

    auto q = pj::Project::open(root, &err);
    CHECK(q != nullptr);
    if (q) {
        const auto& qi = q->info();
        CHECK(qi.name == "My Game");
        CHECK(qi.engine_version == "0.2.1");
        CHECK(qi.author == "me");
        CHECK(qi.description == "a desc with = sign");   // first-'=' split
        CHECK(qi.default_pack_name == "lvls");
        CHECK(qi.cook_on_save == false);                 // "false" → false
        CHECK(qi.pack_on_cook == true);                  // "true"  → true
        CHECK(q->dirs().src == root + "/src");           // derived from root
    }

    // Missing manifest → nullptr + error message.
    std::string err2;
    auto miss = pj::Project::open((rootp / "no_such_sub").string(), &err2);
    CHECK(miss == nullptr);
    CHECK(!err2.empty());

    rm(rootp);
}

// ---- save() rewrites; reopen reflects mutation --------------------
void test_save_mutate_reopen() {
    const fs::path rootp = tmp_root("save");
    const std::string root = rootp.string();
    pj::ProjectInfo info; info.name = "Orig"; info.cook_on_save = false;

    auto p = pj::Project::create_at(root, info, nullptr);
    CHECK(p != nullptr);
    if (p) {
        p->info().name = "Renamed";
        p->info().cook_on_save = true;
        p->info().author = "later";
        CHECK(p->save(nullptr));
    }
    auto q = pj::Project::open(root, nullptr);
    CHECK(q != nullptr);
    if (q) {
        CHECK(q->info().name == "Renamed");
        CHECK(q->info().cook_on_save == true);
        CHECK(q->info().author == "later");
    }
    rm(rootp);
}

// ---- list_source_assets / list_cooked_assets ----------------------
void test_list_assets() {
    const fs::path rootp = tmp_root("list");
    const std::string root = rootp.string();
    auto p = pj::Project::create_at(root, pj::ProjectInfo{}, nullptr);
    CHECK(p != nullptr);
    if (!p) { rm(rootp); return; }

    CHECK(p->list_source_assets().empty());              // assets/ empty
    CHECK(p->list_cooked_assets().empty());

    write_file(rootp / "assets" / "tex.png",            "x");
    write_file(rootp / "assets" / "sub" / "mesh.obj",   "y");
    write_file(rootp / "cooked" / "a.cooked",           "z");
    write_file(rootp / "cooked" / "note.txt",           "n");
    write_file(rootp / "cooked" / "sub" / "c.cooked",   "w");

    auto src = p->list_source_assets();
    CHECK(src.size() == sz(2));
    CHECK(listed(src, "tex.png") && listed(src, "mesh.obj"));
    bool src_sorted = true;
    for (cardinal::usize i = 1; i < src.size(); ++i)
        if (src[i - 1] > src[i]) src_sorted = false;
    CHECK(src_sorted);

    auto ck = p->list_cooked_assets();
    CHECK(ck.size() == sz(2));                            // .cooked only
    CHECK(listed(ck, "a.cooked") && listed(ck, "c.cooked"));
    CHECK(!listed(ck, "note.txt"));

    rm(rootp);
}

// ---- instantiate_template -----------------------------------------
void test_instantiate_template() {
    {
        const fs::path rootp = tmp_root("tmpl");
        pj::InstantiateOptions o;
        o.root = rootp.string();
        o.info.name = "Tmpl";
        o.kind = pj::TemplateKind::FirstPerson;
        std::string err;
        auto p = pj::instantiate_template(o, &err);
        CHECK(p != nullptr);
        if (p) {
            std::error_code ec;
            CHECK(fs::exists(rootp / "src"     / "main.cpp", ec));
            CHECK(fs::exists(rootp / "shaders" / "material.hlsl", ec));
            CHECK(fs::exists(rootp / "assets"  / "README.txt", ec));
            CHECK(fs::exists(rootp / "project.cardinal", ec));
            CHECK(p->info().name == "Tmpl");
            CHECK(p->dirs().root == rootp.string());
        }
        rm(rootp);
    }
    {   // Non-empty existing target, overwrite=false → reject.
        const fs::path rootp = tmp_root("tmpl_ne");
        write_file(rootp / "stray.txt", "occupied");
        pj::InstantiateOptions o;
        o.root = rootp.string();
        o.overwrite_existing = false;
        std::string err;
        auto p = pj::instantiate_template(o, &err);
        CHECK(p == nullptr);
        CHECK(!err.empty());
        rm(rootp);
    }
    {   // Empty existing dir is allowed.
        const fs::path rootp = tmp_root("tmpl_empty");
        std::error_code ec;
        fs::create_directories(rootp, ec);
        pj::InstantiateOptions o;
        o.root = rootp.string();
        o.overwrite_existing = false;
        auto p = pj::instantiate_template(o, nullptr);
        CHECK(p != nullptr);
        rm(rootp);
    }
}

// ---- RecentProjects: most-recent / dedupe / cap-16 / round-trip ---
void test_recent_projects() {
    const fs::path store = tmp_root("recent");      // a dir path; use a file
    const std::string sf = (store.string() + ".txt");
    { std::error_code ec; fs::remove(sf, ec); }

    {
        pj::RecentProjects r(sf);
        CHECK(r.entries().empty());
        r.add("/p/a");
        r.add("/p/b");
        r.add("/p/c");
        // Most-recent first.
        CHECK(r.entries().size() == sz(3));
        CHECK(r.entries()[0] == "/p/c");
        CHECK(r.entries()[1] == "/p/b");
        CHECK(r.entries()[2] == "/p/a");

        r.add("/p/a");                               // re-add → move to front
        CHECK(r.entries().size() == sz(3));
        CHECK(r.entries()[0] == "/p/a");
        CHECK(r.entries()[1] == "/p/c");
        CHECK(r.entries()[2] == "/p/b");

        r.remove("/p/c");
        CHECK(r.entries().size() == sz(2));
        CHECK(r.entries()[0] == "/p/a" && r.entries()[1] == "/p/b");
        r.remove("/p/unknown");                      // no-op
        CHECK(r.entries().size() == sz(2));

        r.save();
    }
    {   // load() round-trips order, one path per line.
        pj::RecentProjects r2(sf);
        r2.load();
        CHECK(r2.entries().size() == sz(2));
        CHECK(r2.entries()[0] == "/p/a" && r2.entries()[1] == "/p/b");
    }
    {   // Cap at 16 newest (oldest dropped).
        pj::RecentProjects r3(sf + ".cap");
        for (int i = 0; i < 20; ++i)
            r3.add("/p/" + std::to_string(i));
        CHECK(r3.entries().size() == sz(16));
        CHECK(r3.entries().front() == "/p/19");      // newest
        CHECK(r3.entries().back()  == "/p/4");       // 20 added, 0..3 dropped
    }
    {   // load() on a missing store → empty (no throw).
        pj::RecentProjects r4(sf + ".does_not_exist");
        r4.add("/tmp/x");
        r4.load();                                   // clears, file absent
        CHECK(r4.entries().empty());
    }
    {   // load() enforces the same cap as add() even when the store has
        // > kMaxRecent lines (older build / hand-edited / corrupt) —
        // keeps the first (most-recent-first) 16.
        const std::string big = sf + ".big";
        std::string blob;
        for (int i = 0; i < 25; ++i) blob += "/p/" + std::to_string(i) + "\n";
        CHECK(write_file(big, blob));
        pj::RecentProjects r5(big);
        r5.load();
        CHECK(r5.entries().size() == sz(16));
        CHECK(r5.entries().front() == "/p/0");       // first line kept
        CHECK(r5.entries().back()  == "/p/15");      // 16..24 dropped
        std::error_code ec2; fs::remove(big, ec2);
    }
    std::error_code ec;
    fs::remove(sf, ec);
    fs::remove(sf + ".cap", ec);
}

}  // namespace

int main() {
    test_template_kinds();
    test_info_defaults();
    test_create_open_roundtrip();
    test_save_mutate_reopen();
    test_list_assets();
    test_instantiate_template();
    test_recent_projects();

    if (g_fail == 0) {
        cardinal::log::infof("projtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("projtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
