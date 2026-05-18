// =============================================================================
// Cardinal — cppscript end-to-end smoke test (headless, no editor).
//
// Sequence:
//   1. Create cppscript::Engine; verify the auto-discovered compiler.
//   2. compile_and_load samples/scripts/hello_script.cpp
//   3. Tick the cppscript engine + plugin Registry until the job reports
//      Loaded (or until a 60s budget runs out).
//   4. Tick the plugin a few hundred times so on_tick fires.
//   5. Unload it; verify plugin Registry no longer lists it.
//
// Exit codes:
//   0  — full pipeline succeeded
//   1  — compiler not discovered
//   2  — compile_and_load returned no handle (bad path?)
//   3  — job reached CompileFailed
//   4  — job reached LoadFailed
//   5  — timeout waiting for Loaded
//   6  — unload didn't remove the plugin from the Registry
// =============================================================================
#include <cardinal/cppscript/cppscript.hpp>
#include <cardinal/plugin/plugin.hpp>
#include <cardinal/core/log.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <thread>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    (void)argc;
    using namespace std::chrono_literals;
    if (!argv || !argv[0]) { std::fprintf(stderr, "FAIL: no argv[0]\n"); return 1; }

    cardinal::cppscript::Desc d{};
    // The script #includes <cardinal/plugin/plugin.hpp> which itself drags
    // in <cardinal/core/types.hpp>; expose both runtime/include trees.
    const fs::path exe_path  = argv[0];
    const fs::path repo_root = (exe_path.parent_path() / ".." / ".." / "..").lexically_normal();
    const fs::path runtime   = repo_root / "runtime";
    for (const char* mod : {"core", "plugin", "render"}) {
        d.include_dirs.push_back((runtime / mod / "include").string());
    }

    auto cs = cardinal::cppscript::Engine::create(d);
    if (cs->compiler_path().empty()) {
        std::fprintf(stderr, "FAIL: no C++ compiler discovered\n");
        return 1;
    }
    std::printf("compiler: %s\n", cs->compiler_path().c_str());

    const fs::path script = repo_root / "samples" / "scripts" / "hello_script.cpp";
    std::printf("script:   %s\n", script.string().c_str());

    auto h = cs->compile_and_load(script.string().c_str());
    if (!h) { std::fprintf(stderr, "FAIL: compile_and_load returned 0\n"); return 2; }
    std::printf("queued job %llu — waiting for compile + load...\n",
                static_cast<unsigned long long>(h.id));

    // Wait up to 60s for the compile to finish + the main-thread tick to
    // promote LoadPending -> Loaded.
    auto start = std::chrono::steady_clock::now();
    cardinal::cppscript::JobInfo last_info{};
    while (true) {
        cs->tick();   // promotes LoadPending → load on this thread
        last_info = cs->query(h);
        if (last_info.status == cardinal::cppscript::JobStatus::Loaded ||
            last_info.status == cardinal::cppscript::JobStatus::CompileFailed ||
            last_info.status == cardinal::cppscript::JobStatus::LoadFailed) break;
        if (std::chrono::steady_clock::now() - start > 60s) break;
        std::this_thread::sleep_for(50ms);
    }

    using S = cardinal::cppscript::JobStatus;
    if (last_info.status == S::CompileFailed) {
        std::fprintf(stderr, "FAIL: compile (rc=%lld)\n%s\n",
            static_cast<long long>(last_info.exit_code),
            last_info.diagnostics.c_str());
        return 3;
    }
    if (last_info.status == S::LoadFailed) {
        std::fprintf(stderr, "FAIL: load (Registry::load returned false)\n");
        return 4;
    }
    if (last_info.status != S::Loaded) {
        std::fprintf(stderr, "FAIL: timeout (status=%d)\n", static_cast<int>(last_info.status));
        return 5;
    }
    std::printf("LOADED in %.2fs (rc=%lld)\n",
                last_info.elapsed_seconds, static_cast<long long>(last_info.exit_code));

    // Tick the plugin a few hundred times to see on_tick fire.
    for (int i = 0; i < 250; ++i) cardinal::plugin::Registry::instance().tick(1.0f / 60.0f);

    // Verify it shows in the registry.
    auto enumerated = cardinal::plugin::Registry::instance().enumerate();
    std::printf("registry has %zu plugin(s) loaded:\n", enumerated.size());
    for (const auto& e : enumerated) {
        std::printf("  - %s v%s   %s\n",
            e.name.c_str(), e.version.c_str(), e.path.c_str());
    }

    // Unload.
    std::printf("unloading hello_script...\n");
    bool ok = cs->unload("hello_script");
    if (!ok) ok = cs->unload(script.string().c_str());

    enumerated = cardinal::plugin::Registry::instance().enumerate();
    if (std::any_of(enumerated.begin(), enumerated.end(),
                    [](const auto& e){ return e.name == "hello_script"; })) {
        std::fprintf(stderr, "FAIL: hello_script still in registry after unload\n");
        return 6;
    }
    std::printf("OK — full compile/load/tick/unload pipeline verified.\n");
    return 0;
}
