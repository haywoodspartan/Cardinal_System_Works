#pragma once

// =============================================================================
// Cardinal — BuildCookRun pipeline (the UE5-style project build orchestrator).
//
// One call takes a cardinal::project::Project from source to a shippable bundle
// by running the staged pipeline, composing the engine's existing tools:
//
//   Build   : project::generate_build_files() + a per-config cmake
//             configure+build command (optionally spawned).
//   Cook    : cook::cook_all()  — source assets -> cooked/<rel>.cooked.
//   Pack    : pack::distribute() — cooked/ -> a shippable bundle (a .cpk +
//             distribution.cardinal manifest + staged exe/DLLs) in out_dir.
//   Archive : copy the staged bundle to an archive dir.
//
// This is Cardinal's analog of Unreal's BuildCookRun (UAT). It lives ABOVE
// project / cook / pack — cook already depends on project, so the orchestrator
// can't live inside project without a dependency cycle.
//
// Verifiable headless: Cook + Pack + Archive run with no GPU/device. The Build
// stage GENERATES the build files + the exact cmake command by default; setting
// run_compile actually spawns the compiler (heavy + toolchain-env-specific).
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/cook/cook.hpp>     // cook::CookSummary
#include <cardinal/pack/pack.hpp>     // pack::DistResult

namespace cardinal::project { class Project; }
namespace cardinal::shader  { class Compiler; }

namespace cardinal::buildtool {

// Build configuration — maps to a CMAKE_BUILD_TYPE (Unreal: Debug / Development
// / Shipping).
enum class BuildConfig : cardinal::u32 { Debug, Development, Shipping };
// Build target — what kind of executable the build produces (informational here;
// affects naming + what gets staged).
enum class BuildTarget : cardinal::u32 { Editor, Game, Server };

const char* build_config_name (BuildConfig c) noexcept;   // "Debug"/"Development"/"Shipping"
const char* build_config_cmake(BuildConfig c) noexcept;   // CMAKE_BUILD_TYPE token
const char* build_target_name (BuildTarget t) noexcept;   // "Editor"/"Game"/"Server"

struct PipelineOptions {
    BuildConfig config {BuildConfig::Development};
    BuildTarget target {BuildTarget::Game};

    bool do_build   {true};    // generate build files + compose the build command
    bool run_compile{false};   // also SPAWN cmake configure+build (env-gated)
    bool do_cook    {true};
    bool do_pack    {true};    // stage/package via pack::distribute
    bool do_archive {false};
    bool force_cook {false};   // re-cook even up-to-date assets

    cardinal::string out_dir;      // staging dir (default: <root>/dist/<config>)
    cardinal::string archive_dir;  // default: <root>/dist/archive/<name>-<config>
    cardinal::vector<cardinal::string> extra_files;  // exe + DLLs staged beside the pack

    cardinal::shader::Compiler* shader_compiler {nullptr};  // for the shader cooker
};

enum class StageStatus : cardinal::u32 { Skipped, Ok, Failed };
const char* stage_status_name(StageStatus s) noexcept;

struct StageReport {
    cardinal::string name;
    StageStatus      status {StageStatus::Skipped};
    cardinal::string detail;
};

struct PipelineResult {
    bool                          ok {false};
    cardinal::vector<StageReport> stages;
    cook::CookSummary             cook {};
    pack::DistResult              dist {};
    cardinal::string              build_command;  // the composed cmake invocation
    cardinal::string              artifact_dir;   // the staged bundle directory
};

// Progress callback: fired once per stage with the stage name + a short message.
using ProgressFn = void (*)(const char* stage, const char* message, void* user);

// Run the selected stages in order (Build -> Cook -> Pack -> Archive),
// short-circuiting on the first failure. Returns the per-stage report.
PipelineResult run_pipeline(const project::Project& proj,
                            const PipelineOptions& opts,
                            ProgressFn progress = nullptr, void* user = nullptr);

// Composable individual stages (run_pipeline calls these).
StageReport stage_build  (const project::Project& proj, const PipelineOptions& opts,
                          cardinal::string& build_command_out);
StageReport stage_cook   (const project::Project& proj, const PipelineOptions& opts,
                          cook::CookSummary& out);
StageReport stage_pack   (const project::Project& proj, const PipelineOptions& opts,
                          pack::DistResult& out);
StageReport stage_archive(const project::Project& proj, const PipelineOptions& opts,
                          const cardinal::string& staged_dir);

}  // namespace cardinal::buildtool
