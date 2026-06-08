// =============================================================================
// Cardinal — BuildCookRun pipeline (cardinal::buildtool) regression suite.
//
// Drives the orchestrator end-to-end HEADLESS: instantiate a project, run the
// staged pipeline (Build generate -> Cook -> Pack/Stage -> Archive) and assert
// the per-stage reports + the shippable bundle on disk. The Build stage only
// GENERATES files + composes the cmake command (run_compile=false) — actually
// compiling the engine is the user's toolchain step, out of scope for ctest.
// Disk round-trips run through a unique temp dir. Exit 0 = all pass.
// =============================================================================

#include <cardinal/buildtool/pipeline.hpp>
#include <cardinal/project/project.hpp>
#include <cardinal/core/diag/log.hpp>

#include <filesystem>
#include <fstream>
#include <string>

namespace {

namespace bt = cardinal::buildtool;
namespace pj = cardinal::project;
namespace fs = std::filesystem;

int g_checks = 0, g_fail = 0;
void check_impl(bool ok, const char* e, int l) {
    ++g_checks;
    if (!ok) { ++g_fail; cardinal::log::errorf("bttest", "FAIL  L%d  %s", l, e); }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool contains(const std::string& h, const char* n) { return h.find(n) != std::string::npos; }
fs::path tmp_root(const char* t) {
    fs::path p = fs::temp_directory_path() / (std::string("cardinal_bt_") + t);
    std::error_code ec; fs::remove_all(p, ec); return p;
}
void rm(const fs::path& p) { std::error_code ec; fs::remove_all(p, ec); }
bool write_file(const fs::path& p, const std::string& s) {
    std::error_code ec; fs::create_directories(p.parent_path(), ec);
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(s.data(), static_cast<std::streamsize>(s.size())); return true;
}

// ---- config / target tables ---------------------------------------
void test_tables() {
    CHECK(std::string(bt::build_config_name(bt::BuildConfig::Debug))       == "Debug");
    CHECK(std::string(bt::build_config_name(bt::BuildConfig::Development)) == "Development");
    CHECK(std::string(bt::build_config_name(bt::BuildConfig::Shipping))    == "Shipping");
    CHECK(std::string(bt::build_config_cmake(bt::BuildConfig::Debug))       == "Debug");
    CHECK(std::string(bt::build_config_cmake(bt::BuildConfig::Development)) == "RelWithDebInfo");
    CHECK(std::string(bt::build_config_cmake(bt::BuildConfig::Shipping))    == "Release");
    CHECK(std::string(bt::build_target_name(bt::BuildTarget::Editor)) == "Editor");
    CHECK(std::string(bt::build_target_name(bt::BuildTarget::Server)) == "Server");
}

// ---- full Build->Cook->Pack->Archive run --------------------------
void test_full_pipeline() {
    const fs::path rootp = tmp_root("pipe");
    pj::InstantiateOptions o;
    o.root             = rootp.string();
    o.info.name        = "Packaged Game";
    o.info.engine_root = "G:/Cardinal_System_Works";
    o.kind             = pj::TemplateKind::Blank;
    std::string err;
    auto p = pj::instantiate_template(o, &err);
    CHECK(p != nullptr);
    if (!p) { rm(rootp); return; }

    // A fake "executable" to stage next to the pack (proves extra_files).
    const auto exe = rootp / "GameStub.bin";
    write_file(exe, "stub-exe");

    bt::PipelineOptions opt;
    opt.config      = bt::BuildConfig::Shipping;     // -> CMAKE_BUILD_TYPE Release
    opt.target      = bt::BuildTarget::Game;
    opt.do_build    = true;   opt.run_compile = false;   // generate only
    opt.do_cook     = true;   opt.do_pack     = true;     opt.do_archive = true;
    opt.force_cook  = true;
    opt.extra_files.push_back(exe.string());

    auto res = bt::run_pipeline(*p, opt);
    CHECK(res.ok);
    CHECK(res.stages.size() == 4u);                       // build, cook, pack, archive

    // Build stage: cmake command composed for the Shipping (Release) config.
    CHECK(contains(res.build_command, "cmake"));
    CHECK(contains(res.build_command, "Release"));
    CHECK(contains(res.build_command, "CARDINAL_ENGINE_ROOT"));
    // Build files were (re)generated.
    std::error_code ec;
    CHECK(fs::exists(rootp / "CMakeLists.txt", ec));

    // Cook stage: empty assets/ -> ran cleanly, nothing failed.
    CHECK(res.cook.failed_count == 0u);

    // Pack stage: produced a shippable bundle (.cpk + manifest + staged exe).
    CHECK(res.dist.ok);
    CHECK(fs::exists(res.dist.pack_path, ec));            // <name>.cpk
    CHECK(fs::exists(res.dist.manifest_path, ec));        // distribution.cardinal
    CHECK(res.dist.extra_count >= 1u);
    CHECK(!res.artifact_dir.empty());
    CHECK(fs::exists(res.artifact_dir, ec));
    CHECK(fs::exists(fs::path(res.artifact_dir) / "GameStub.bin", ec));

    // Archive stage: copied the bundle out.
    bool archived = false;
    for (const auto& s : res.stages)
        if (s.name == "Archive" && s.status == bt::StageStatus::Ok) archived = true;
    CHECK(archived);

    rm(rootp);
}

// ---- stage selection + short-circuit ------------------------------
void test_stage_selection() {
    const fs::path rootp = tmp_root("sel");
    pj::InstantiateOptions o;
    o.root = rootp.string(); o.info.name = "Sel"; o.info.engine_root = "G:/Cardinal_System_Works";
    auto p = pj::instantiate_template(o, nullptr);
    CHECK(p != nullptr);
    if (p) {
        bt::PipelineOptions opt;
        opt.do_build = false; opt.do_cook = true; opt.do_pack = true; opt.do_archive = false;
        auto res = bt::run_pipeline(*p, opt);
        CHECK(res.ok);
        CHECK(res.stages.size() == 2u);                  // cook + pack only
        CHECK(res.build_command.empty());                // build skipped
        CHECK(res.dist.ok);
    }
    rm(rootp);
}

}  // namespace

int main() {
    test_tables();
    test_full_pipeline();
    test_stage_selection();
    if (g_fail == 0) {
        cardinal::log::infof("bttest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("bttest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
