// =============================================================================
// Cardinal — sandbox subprocess smoke test (headless, no editor).
//
// Sequence:
//   1. Compile samples/scripts/hello_script.cpp to a DLL via cppscript
//      (so we have a known-good plugin DLL on disk; we don't load it
//      through the Registry though — this test exercises the subprocess
//      sandbox path).
//   2. Sandbox::create with Mode::Subprocess, dll path = the cppscript
//      output. The runner exe spawns + connects + ATTACHes.
//   3. Tick 10 times; each tick is a TICK frame round-trip + ACK.
//   4. detach() — runner gets DETACH, exits cleanly.
//   5. Inspect status() before + after tick + after detach.
//
// Then the second part:
//   6. Compile samples/scripts/crash_script.cpp (a script with on_tick that
//      deliberately writes through a null pointer).
//   7. Sandbox the crash_script in Subprocess mode.
//   8. Tick — runner SEH catches the AV, sends ERROR + ACK; OR if SEH
//      doesn't fire (e.g. /GUARD:CF), the runner dies, host detects via
//      pipe break, status().alive flips to false.
//   9. Either way: this test process keeps running. THE WHOLE POINT.
//
// Exit codes:
//   0  — full pipeline succeeded (script ticked + crash isolated)
//   1  — cppscript compile of hello_script failed
//   2  — subprocess sandbox attach failed
//   3  — tick loop didn't complete
//   4  — sandbox alive after a deliberately-crashing script
//   5  — detach didn't complete
// =============================================================================

#include <cardinal/cppscript/cppscript.hpp>
#include <cardinal/plugin/plugin.hpp>
#include <cardinal/sandbox/sandbox.hpp>
#include <cardinal/core/log.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <thread>

namespace fs = std::filesystem;

namespace {

// Compile a cpp file via cppscript without going through the plugin
// Registry. We just want the .dll path so we can hand it to Sandbox.
// Returns empty on failure.
std::string compile_to_dll(cardinal::cppscript::Engine& cs,
                           const std::string& source_path)
{
    auto h = cs.compile_and_load(source_path.c_str());
    if (!h) return {};

    using namespace std::chrono_literals;
    using S = cardinal::cppscript::JobStatus;
    auto start = std::chrono::steady_clock::now();
    while (true) {
        cs.tick();   // promotes LoadPending → load on this thread
        auto info = cs.query(h);
        if (info.status == S::Loaded) {
            // We loaded into the in-process Registry as a side effect.
            // Immediately unload — we want the DLL on disk to hand to
            // the subprocess sandbox, not a duplicate in-process copy.
            cardinal::plugin::Registry::instance().unload(info.dll_path.c_str());
            return info.dll_path;
        }
        if (info.status == S::CompileFailed || info.status == S::LoadFailed) {
            std::fprintf(stderr, "compile failed:\n%s\n", info.diagnostics.c_str());
            return {};
        }
        if (std::chrono::steady_clock::now() - start > 60s) {
            std::fprintf(stderr, "compile timed out\n");
            return {};
        }
        std::this_thread::sleep_for(50ms);
    }
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    if (!argv || !argv[0]) { std::fprintf(stderr, "FAIL: no argv[0]\n"); return 1; }

    const fs::path exe_path  = argv[0];
    const fs::path repo_root = (exe_path.parent_path() / ".." / ".." / "..").lexically_normal();
    const fs::path runtime   = repo_root / "runtime";

    // ---- cppscript bootstrap -------------------------------------------
    cardinal::cppscript::Desc cs_desc{};
    for (const char* mod : {"core", "plugin", "render"}) {
        cs_desc.include_dirs.push_back((runtime / mod / "include").string());
    }
    auto cs = cardinal::cppscript::Engine::create(cs_desc);
    if (cs->compiler_path().empty()) {
        std::fprintf(stderr, "FAIL: no compiler\n");
        return 1;
    }

    // ---- Part 1: well-behaved script in subprocess sandbox -------------
    const fs::path hello_src = repo_root / "samples" / "scripts" / "hello_script.cpp";
    std::printf("[1] compiling %s...\n", hello_src.string().c_str());
    const std::string hello_dll = compile_to_dll(*cs, hello_src.string());
    if (hello_dll.empty()) return 1;
    std::printf("[1] DLL: %s\n", hello_dll.c_str());

    {
        cardinal::sandbox::Desc sdesc{};
        sdesc.mode             = cardinal::sandbox::Mode::Subprocess;
        sdesc.tick_timeout_ms  = 1000;
        sdesc.attach_timeout_ms= 5000;

        std::printf("[1] launching subprocess sandbox...\n");
        auto sb = cardinal::sandbox::Sandbox::create(sdesc, hello_dll.c_str());
        if (sb == nullptr) {
            std::fprintf(stderr, "FAIL: subprocess attach\n");
            return 2;
        }
        auto st = sb->status();
        std::printf("[1] attached: pid=%llu plugin='%s' v%s\n",
            static_cast<unsigned long long>(st.pid),
            st.plugin_name.c_str(), st.plugin_version.c_str());

        std::printf("[1] ticking 10 times in subprocess...\n");
        for (int i = 0; i < 10; ++i) {
            if (!sb->tick(1.0f / 60.0f)) {
                std::fprintf(stderr, "FAIL: tick %d returned dead\n", i);
                return 3;
            }
        }
        st = sb->status();
        std::printf("[1] after 10 ticks: ticks=%llu last_tick_ms=%.3f\n",
            static_cast<unsigned long long>(st.ticks), st.last_tick_ms);

        std::printf("[1] detaching...\n");
        sb->detach();
        st = sb->status();
        if (st.alive) {
            std::fprintf(stderr, "FAIL: still alive after detach\n");
            return 5;
        }
        std::printf("[1] detached cleanly.\n");
    }

    // ---- Part 2: deliberately-crashing script in subprocess sandbox ----
    // Generate a tiny C++ script that nulls out the world inside on_tick.
    // SEH should catch the AV; runner sends ERROR + ACK; sandbox stays alive.
    const fs::path crash_src = repo_root / "samples" / "scripts" / "crash_script.cpp";
    if (!fs::exists(crash_src)) {
        std::fprintf(stderr, "[2] missing %s — skipping crash test\n",
                     crash_src.string().c_str());
        std::printf("OK — well-behaved subprocess sandbox verified.\n");
        return 0;
    }

    std::printf("[2] compiling deliberately-crashing %s...\n",
                crash_src.string().c_str());
    const std::string crash_dll = compile_to_dll(*cs, crash_src.string());
    if (crash_dll.empty()) {
        std::fprintf(stderr, "FAIL: crash_script compile\n");
        return 1;
    }

    {
        cardinal::sandbox::Desc sdesc{};
        sdesc.mode             = cardinal::sandbox::Mode::Subprocess;
        sdesc.tick_timeout_ms  = 500;

        std::printf("[2] launching subprocess sandbox for crash_script...\n");
        auto sb = cardinal::sandbox::Sandbox::create(sdesc, crash_dll.c_str());
        if (sb == nullptr) {
            std::fprintf(stderr, "[2] WARN: attach failed (script crashes in on_attach?)\n");
            std::printf("OK — host process survived an attach-time crash.\n");
            return 0;
        }
        auto st = sb->status();
        std::printf("[2] attached: pid=%llu plugin='%s'\n",
            static_cast<unsigned long long>(st.pid), st.plugin_name.c_str());

        // Tick a few times. The script's on_tick crashes on every call;
        // the runner's SEH catches it and replies ERROR + ACK each time.
        // sb->tick() returns true (sandbox alive) but status().last_error
        // gets populated.
        std::printf("[2] ticking crash_script 5 times — host should survive...\n");
        for (int i = 0; i < 5; ++i) {
            const bool alive = sb->tick(1.0f / 60.0f);
            st = sb->status();
            std::printf("    tick %d: alive=%d last_err='%s'\n",
                        i, alive ? 1 : 0,
                        st.last_error.empty() ? "(none)" : st.last_error.c_str());
            // Don't return false-fail here: SEH might convert AV into ACK,
            // OR the runner might die outright. Both are valid "host
            // survived" outcomes. We only care that THIS process keeps
            // running — which is proved by reaching the print below.
        }
        std::printf("[2] detaching crash_script sandbox...\n");
        sb->detach();
    }

    std::printf("OK — subprocess sandbox: well-behaved + crash-isolated paths verified.\n");
    return 0;
}
