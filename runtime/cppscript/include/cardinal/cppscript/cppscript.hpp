#pragma once

// =============================================================================
// Cardinal — C++ scripting engine.
//
// Compile-and-load arbitrary C++ source at runtime. Each script becomes a
// DLL (compiled by a real compiler in a child process) that the plugin
// runtime loads via cardinal::plugin::Registry. The script gets the same
// hot-load / hot-unload / SEH-isolation guarantees as a hand-built plugin
// — see runtime/plugin/include/cardinal/plugin/plugin.hpp — plus a file
// watcher that recompiles on save (Phase 2).
//
// "Subprocess" guarantees, in two layers:
//
//   1. COMPILE-TIME ISOLATION (real). The compiler (clang-cl or MSVC cl)
//      always runs as a child process. A cl.exe ICE / clang segfault
//      reports as a non-zero exit code; the editor stays alive.
//
//   2. RUNTIME ISOLATION (SEH-tier). The compiled DLL loads into the
//      editor process and runs through cardinal::plugin's SEH-protected
//      tick. Access violations / divide-by-zero / similar synchronous
//      hardware exceptions are caught and the script is auto-disabled.
//      Heap corruption / dual-thread races are NOT caught — for that you
//      need true process isolation (see cardinal::sandbox below).
//
//   3. SUBPROCESS RUNTIME (cardinal::sandbox). Optional companion module
//      that hosts the script DLL in a child process and marshals tick /
//      detach calls over a named pipe. Survives heap corruption + UB.
//      Higher latency per tick (IPC roundtrip) — opt-in per script.
//
// Front-end usage:
//
//   cppscript::Desc d;
//   auto cs = cppscript::Engine::create(d);
//   auto job = cs->compile_and_load("scripts/my_tonemap.cpp");
//   if (job.failed()) print(job.diagnostics);
//   ...
//   cs->tick();   // drains async compile completions; reloads on watch hit
//
// Each loaded script becomes visible in `plugin::Registry::enumerate()`
// alongside hand-built plugins.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/containers.hpp>

namespace cardinal::cppscript {

// Compile job lifecycle. Async work driven by Engine::tick().
enum class JobStatus : u32 {
    Pending      = 0,   // queued, worker not yet picked it up
    Compiling,          // compiler process running
    LoadPending,        // compile succeeded, plugin Registry::load pending
    Loaded,             // attached to the plugin Registry
    CompileFailed,      // non-zero exit from compiler — see diagnostics
    LoadFailed,         // DLL produced but the plugin Registry rejected it
};

// One handle per submitted job. The Engine owns the underlying record;
// JobHandle is a copyable id you keep around to query status / output.
struct JobHandle { u64 id{0}; explicit operator bool() const { return id != 0; } };

// Snapshot of a job's state — returned by Engine::query().
struct JobInfo {
    JobHandle    handle;
    cardinal::string  source_path;     // input .cpp
    cardinal::string  dll_path;        // output .dll (filled once compile starts)
    JobStatus    status{JobStatus::Pending};
    cardinal::string  diagnostics;     // compiler stdout+stderr; populated on completion
    i64          exit_code{0};    // compiler exit code; valid once status >= LoadPending
    f64          elapsed_seconds{0.0};
};

// Compiler the engine should drive. Auto = probe in this order:
//   1. Environment var CARDINAL_CPPSCRIPT_COMPILER
//   2. clang-cl on PATH
//   3. cl.exe via vswhere on Windows
//   4. clang++ on POSIX
enum class CompilerKind : u32 { Auto, ClangCl, ClPath, ClangPosix };

struct Desc {
    // Where to drop compiled .dll/.so files. Defaults to <exe>/cppscript_cache.
    cardinal::string cache_dir{};

    // Extra compile flags appended to the default set (the defaults
    // already include /std:c++23 /MD /EHsc /Zc:preprocessor /D_HAS_EXCEPTIONS=1
    // on MSVC; -std=c++23 -fPIC on POSIX).
    cardinal::vector<cardinal::string> extra_flags{};

    // Include search dirs added to the compile command (the engine's own
    // public-include tree should be added here so scripts can reach the
    // plugin headers).
    cardinal::vector<cardinal::string> include_dirs{};

    // Macro definitions added to the compile command.
    cardinal::vector<cardinal::string> defines{};

    // Compiler choice (Auto by default).
    CompilerKind compiler{CompilerKind::Auto};

    // Override path to the compiler executable. When non-empty, takes
    // precedence over `compiler`. Use this to pin a specific clang-cl /
    // cl.exe (e.g. an installed VS2022 path).
    cardinal::string compiler_override{};
};

// Optional callback fired once per terminal job state transition (Loaded /
// CompileFailed / LoadFailed). Useful for the Studio panel to refresh.
using JobObserver = cardinal::function<void(const JobInfo&)>;

class Engine {
public:
    static cardinal::unique_ptr<Engine> create(const Desc& desc);
    virtual ~Engine() = default;
    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    // Queue a compile + load. Returns immediately with a JobHandle; the
    // worker thread picks it up, runs the compiler, then schedules a load
    // through the plugin Registry on the next tick(). Use query(handle)
    // to poll status, or set_observer() to be notified.
    virtual JobHandle compile_and_load(const char* source_path) = 0;

    // Same, but the source is supplied as a literal C++ string. The
    // engine writes it to <cache_dir>/<name>.cpp first.
    virtual JobHandle compile_string(const char* source_code,
                                     const char* logical_name) = 0;

    // Unload a script by its source path or by the plugin's reported
    // name. Routes through plugin::Registry::unload.
    virtual bool unload(const char* name_or_path) = 0;

    // Force-reload (recompile + load). Idempotent if no source change.
    virtual JobHandle reload(const char* source_path) = 0;

    // File watcher — the engine polls `path` for mtime change every tick;
    // a change schedules a reload. Recursive on directories.
    virtual void watch  (const char* path) = 0;
    virtual void unwatch(const char* path) = 0;

    // Drain pending compile completions + schedule loads + service the
    // file watcher. Call once per editor frame.
    virtual void tick() = 0;

    // Observer for job state transitions. nullptr clears.
    virtual void set_observer(JobObserver cb) = 0;

    // Read-only snapshots.
    virtual JobInfo              query(JobHandle h) const = 0;
    virtual cardinal::vector<JobInfo> jobs() const = 0;
    // Path of the resolved compiler executable (for the `cppscript.compiler`
    // console command). Empty when no compiler was found.
    virtual cardinal::string          compiler_path() const = 0;

protected:
    Engine() = default;
};

}  // namespace cardinal::cppscript
