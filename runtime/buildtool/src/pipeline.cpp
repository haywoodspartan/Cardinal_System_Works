// =============================================================================
// Cardinal — BuildCookRun pipeline implementation (see pipeline.hpp).
// =============================================================================

#include <cardinal/buildtool/pipeline.hpp>

#include <cardinal/project/project.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/std/filesystem.hpp>   // cardinal::fs, error_code
#include <cardinal/core/std/cstdlib.hpp>      // cardinal::system
#include <cardinal/core/std/cstdio.hpp>       // cardinal::snprintf
#include <cardinal/core/std/utility.hpp>      // cardinal::move

namespace fs = cardinal::fs;

namespace cardinal::buildtool {

const char* build_config_name(BuildConfig c) noexcept {
    switch (c) {
        case BuildConfig::Debug:       return "Debug";
        case BuildConfig::Development: return "Development";
        case BuildConfig::Shipping:    return "Shipping";
    }
    return "Development";
}
const char* build_config_cmake(BuildConfig c) noexcept {
    switch (c) {
        case BuildConfig::Debug:       return "Debug";
        case BuildConfig::Development: return "RelWithDebInfo";
        case BuildConfig::Shipping:    return "Release";
    }
    return "RelWithDebInfo";
}
const char* build_target_name(BuildTarget t) noexcept {
    switch (t) {
        case BuildTarget::Editor: return "Editor";
        case BuildTarget::Game:   return "Game";
        case BuildTarget::Server: return "Server";
    }
    return "Game";
}
const char* stage_status_name(StageStatus s) noexcept {
    switch (s) {
        case StageStatus::Skipped: return "skipped";
        case StageStatus::Ok:      return "ok";
        case StageStatus::Failed:  return "FAILED";
    }
    return "?";
}

namespace {

cardinal::string fwd_slash(cardinal::string p) {
    for (char& c : p) if (c == '\\') c = '/';
    return p;
}

cardinal::string nonempty(const cardinal::string& s, const char* fallback) {
    return s.empty() ? cardinal::string(fallback) : s;
}

// The staged-bundle directory the Pack stage writes into.
cardinal::string resolve_out_dir(const project::Project& proj, const PipelineOptions& opts) {
    if (!opts.out_dir.empty()) return opts.out_dir;
    return proj.dirs().root + "/dist/" + build_config_name(opts.config);
}

cardinal::string resolve_archive_dir(const project::Project& proj, const PipelineOptions& opts) {
    if (!opts.archive_dir.empty()) return opts.archive_dir;
    return proj.dirs().root + "/dist/archive/" +
           nonempty(proj.info().name, "Game") + "-" + build_config_name(opts.config);
}

}  // namespace

// ---- Build ----------------------------------------------------------------
StageReport stage_build(const project::Project& proj, const PipelineOptions& opts,
                        cardinal::string& cmd_out) {
    StageReport r; r.name = "Build";

    // Always (re)generate the build files so the project is buildable + the
    // command below targets a real CMakeLists.
    cardinal::string err;
    if (!project::generate_build_files(proj, &err)) {
        r.status = StageStatus::Failed;
        r.detail = "generate_build_files failed: " + err;
        return r;
    }

    const cardinal::string root   = fwd_slash(proj.dirs().root);
    const cardinal::string engine = fwd_slash(proj.info().engine_root);
    const cardinal::string bdir   = root + "/build/" + build_config_name(opts.config);

    // The project manifest is untrusted + engine_root reaches std::system below.
    // Refuse paths with characters that could break out of the quoted cmake
    // command (quote / backtick / $ / newline) in cmd.exe or bash.
    auto path_unsafe = [](const cardinal::string& s) {
        for (char c : s)
            if (c == '"' || c == '`' || c == '$' || c == '\n' || c == '\r') return true;
        return false;
    };
    if (path_unsafe(root) || path_unsafe(engine)) {
        r.status = StageStatus::Failed;
        r.detail = "project/engine path contains unsafe shell characters";
        return r;
    }

    cmd_out =
        "cmake -G Ninja -B \"" + bdir + "\" -S \"" + root +
        "\" -DCARDINAL_ENGINE_ROOT=\"" + engine +
        "\" -DCMAKE_BUILD_TYPE=" + build_config_cmake(opts.config) +
        " && cmake --build \"" + bdir + "\"";

    if (opts.run_compile) {
        const int rc = cardinal::system(cmd_out.c_str());
        if (rc != 0) {
            r.status = StageStatus::Failed;
            r.detail = "compile failed (cmake exit != 0) — see console";
            return r;
        }
        r.status = StageStatus::Ok;
        r.detail = cardinal::string("compiled ") + build_config_name(opts.config);
        return r;
    }

    r.status = StageStatus::Ok;
    r.detail = "build files generated (run_compile=false); command ready";
    return r;
}

// ---- Cook -----------------------------------------------------------------
StageReport stage_cook(const project::Project& proj, const PipelineOptions& opts,
                       cook::CookSummary& out) {
    StageReport r; r.name = "Cook";

    cook::CookerRegistry reg;
    reg.register_builtin(opts.shader_compiler);

    cardinal::vector<cook::CookResult> results;
    out = cook::cook_all(proj, reg, results, opts.force_cook);

    char buf[160];
    cardinal::snprintf(buf, sizeof(buf), "cooked %u, skipped %u, failed %u",
                       out.cooked_count, out.skipped_count, out.failed_count);
    r.detail  = buf;
    r.status  = (out.failed_count > 0) ? StageStatus::Failed : StageStatus::Ok;
    return r;
}

// ---- Pack / Stage ---------------------------------------------------------
StageReport stage_pack(const project::Project& proj, const PipelineOptions& opts,
                       pack::DistResult& out) {
    StageReport r; r.name = "Pack";

    pack::DistOptions d;
    d.project_root   = proj.dirs().root;
    d.out_dir        = resolve_out_dir(proj, opts);
    d.pack_name      = nonempty(proj.info().default_pack_name, "client");
    d.include_saves  = true;
    d.extra_files    = opts.extra_files;
    d.app_name       = nonempty(proj.info().name, "Cardinal Game");
    d.engine_version = nonempty(proj.info().engine_version, "0.1.0");

    out = pack::distribute(d);
    if (!out.ok) {
        r.status = StageStatus::Failed;
        r.detail = "distribute failed: " + out.error;
        return r;
    }
    char buf[224];
    cardinal::snprintf(buf, sizeof(buf), "%u assets, %u saves, %u extra -> %s",
                       out.asset_count, out.save_count, out.extra_count,
                       out.pack_path.c_str());
    r.detail = buf;
    r.status = StageStatus::Ok;
    return r;
}

// ---- Archive --------------------------------------------------------------
StageReport stage_archive(const project::Project& proj, const PipelineOptions& opts,
                          const cardinal::string& staged_dir) {
    StageReport r; r.name = "Archive";

    const cardinal::string adir = resolve_archive_dir(proj, opts);
    cardinal::error_code ec;
    fs::create_directories(adir, ec);
    if (ec) { r.status = StageStatus::Failed; r.detail = "mkdir: " + ec.message(); return r; }
    fs::copy(staged_dir, adir,
             fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
    if (ec) { r.status = StageStatus::Failed; r.detail = "copy: " + ec.message(); return r; }

    r.status = StageStatus::Ok;
    r.detail = adir;
    return r;
}

// ---- Orchestrator ---------------------------------------------------------
PipelineResult run_pipeline(const project::Project& proj, const PipelineOptions& opts,
                            ProgressFn progress, void* user) {
    PipelineResult res;
    auto report = [&](StageReport s) {
        if (progress) progress(s.name.c_str(), s.detail.c_str(), user);
        cardinal::log::infof("buildtool", "[%s] %s — %s",
                             s.name.c_str(), stage_status_name(s.status), s.detail.c_str());
        const bool failed = (s.status == StageStatus::Failed);
        res.stages.push_back(cardinal::move(s));
        return !failed;   // false => short-circuit
    };

    if (opts.do_build) {
        if (!report(stage_build(proj, opts, res.build_command))) return res;
    }
    if (opts.do_cook) {
        if (!report(stage_cook(proj, opts, res.cook))) return res;
    }
    if (opts.do_pack) {
        if (!report(stage_pack(proj, opts, res.dist))) return res;
        res.artifact_dir = resolve_out_dir(proj, opts);
    }
    if (opts.do_archive) {
        cardinal::string staged = res.artifact_dir.empty()
            ? resolve_out_dir(proj, opts) : res.artifact_dir;
        if (!report(stage_archive(proj, opts, staged))) return res;
    }

    res.ok = true;
    cardinal::log::infof("buildtool", "pipeline OK (%s / %s)",
                         build_config_name(opts.config), build_target_name(opts.target));
    return res;
}

}  // namespace cardinal::buildtool
