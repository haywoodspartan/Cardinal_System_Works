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

// ---- process capture ----------------------------------------------
void test_run_capture() {
    // `echo` exists on both cmd.exe (_popen) and /bin/sh (popen).
    auto r = bt::run_capture("echo cardinal_capture_ok");
    CHECK(r.launched);
    CHECK(r.exit_code == 0);
    CHECK(contains(r.output, "cardinal_capture_ok"));
}

// ---- engine-source build orchestration ----------------------------
void test_build_engine() {
    // (a) A checkout WITH a top-level CMakeLists.txt validates + composes the
    //     configure/build commands (compose-only: pure, no spawn).
    const fs::path eng = tmp_root("engine");
    write_file(eng / "CMakeLists.txt", "cmake_minimum_required(VERSION 3.27)\n");
    bt::EngineBuildOptions o;
    o.engine_root = eng.string();
    o.config      = bt::BuildConfig::Development;   // -> RelWithDebInfo
    o.run_compile = false;
    auto rep = bt::build_engine(o);
    CHECK(rep.validated);
    CHECK(rep.ok);                                  // compose-only success
    CHECK(!rep.compiled);                           // did not spawn
    CHECK(contains(rep.configure_command, "cmake"));
    CHECK(contains(rep.configure_command, "RelWithDebInfo"));
    CHECK(contains(rep.configure_command, eng.generic_string().c_str()));   // engine root in -S
    CHECK(contains(rep.build_command, "--target"));
    CHECK(contains(rep.build_command, "cardinal_engine"));          // default target
    CHECK(!rep.build_dir.empty());

    // (b) A dir WITHOUT a CMakeLists.txt fails validation.
    const fs::path bad = tmp_root("engine_bad");
    std::error_code ec; fs::create_directories(bad, ec);
    bt::EngineBuildOptions o2; o2.engine_root = bad.string(); o2.run_compile = false;
    auto rep2 = bt::build_engine(o2);
    CHECK(!rep2.validated);
    CHECK(!rep2.ok);

    // (c) A custom target name flows into the build command.
    bt::EngineBuildOptions o3; o3.engine_root = eng.string(); o3.target = "cardinal_core";
    auto rep3 = bt::build_engine(o3);
    CHECK(contains(rep3.build_command, "cardinal_core"));

    rm(eng); rm(bad);
}

// ---- pipeline with the optional engine-build stage ----------------
void test_pipeline_build_engine() {
    // A temp "engine checkout" (has a top-level CMakeLists.txt) so validation
    // passes on any machine (CI-safe — no dependence on the dev checkout path).
    const fs::path eng = tmp_root("pipe_engine");
    write_file(eng / "CMakeLists.txt", "cmake_minimum_required(VERSION 3.27)\n");

    const fs::path rootp = tmp_root("pipe_be");
    pj::InstantiateOptions o;
    o.root = rootp.string(); o.info.name = "BE"; o.info.engine_root = eng.string();
    auto p = pj::instantiate_template(o, nullptr);
    CHECK(p != nullptr);
    if (p) {
        bt::PipelineOptions opt;
        opt.do_build_engine = true;
        opt.do_build = false; opt.do_cook = false; opt.do_pack = false; opt.do_archive = false;
        opt.run_compile = false;                       // compose-only
        auto res = bt::run_pipeline(*p, opt);
        CHECK(res.ok);
        CHECK(res.stages.size() == 1u);                // only BuildEngine ran
        bool be_ok = false;
        for (const auto& s : res.stages)
            if (s.name == "BuildEngine" && s.status == bt::StageStatus::Ok) be_ok = true;
        CHECK(be_ok);
    }
    rm(eng); rm(rootp);
}

}  // namespace

int main() {
    test_tables();
    test_full_pipeline();
    test_stage_selection();
    test_run_capture();
    test_build_engine();
    test_pipeline_build_engine();
    if (g_fail == 0) {
        cardinal::log::infof("bttest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("bttest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
