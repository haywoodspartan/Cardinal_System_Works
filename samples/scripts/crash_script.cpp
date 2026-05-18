// =============================================================================
// Cardinal — DELIBERATELY-CRASHING C++ script.
//
// Used to prove that cardinal::sandbox's Subprocess mode actually isolates
// host crashes. on_tick writes through a null pointer; the runner's SEH
// __try / __except catches the access violation, reports it back as an
// ERROR frame, and keeps going. The host (Cardinal_System_Sandbox) sees the error
// in status().last_error but its own process keeps running.
//
// DO NOT load this with InProcess mode — the SEH protection there is the
// plugin Registry's per-tick wrapper which will mark the plugin disabled
// (still survivable, but not the same demonstration).
// =============================================================================

#include <cardinal/plugin/plugin.hpp>

namespace { const CardinalPluginHostApi* g_host = nullptr; }

extern "C" CARDINAL_PLUGIN_EXPORT
void cardinal_plugin_register(CardinalPluginInfo* out_info) {
    out_info->api_version = CARDINAL_PLUGIN_API_VERSION;
    out_info->name        = "crash_script";
    out_info->version     = "0.1";
    out_info->author      = "Cardinal";
    out_info->description = "Deliberately crashes in on_tick to prove subprocess isolation works.";

    out_info->on_attach = [](const CardinalPluginHostApi* host) {
        g_host = host;
        if (g_host && g_host->log_info) {
            g_host->log_info("crash_script", "attached — about to crash on first tick");
        }
    };
    out_info->on_tick = [](float /*dt*/) {
        // The /analyze and /WX checks at the engine level would normally
        // refuse to compile this — but cppscript's compile flags don't
        // include /analyze, so we get the AV at runtime as intended.
        volatile int* p = nullptr;
        *p = 0xDEAD;   // EXCEPTION_ACCESS_VIOLATION
    };
    out_info->on_detach = []() {
        if (g_host && g_host->log_info) g_host->log_info("crash_script", "detached");
    };
}
