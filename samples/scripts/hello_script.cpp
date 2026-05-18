// =============================================================================
// Cardinal — example C++ script.
//
// Compiled at runtime by `cardinal::cppscript::Engine` via the editor's
// child-process compiler invocation, then loaded as a plugin through
// cardinal::plugin::Registry. Prove-out script: registers a logging hook
// in on_attach + counts ticks. Lifecycle observable in the editor's Log
// panel and via `plugin.list` / `script.list` console commands.
//
// To try:
//   editor> script.compile <repo>/samples/scripts/hello_script.cpp
//   editor> script.list                 # status: Loaded
//   editor> plugin.list                 # appears as "hello_script v1"
//   editor> plugin.unload hello_script  # detach
// =============================================================================

#include <cardinal/plugin/plugin.hpp>

#include <atomic>

namespace {
const CardinalPluginHostApi* g_host = nullptr;
std::atomic<unsigned>        g_ticks{0};
}  // namespace

extern "C" CARDINAL_PLUGIN_EXPORT
void cardinal_plugin_register(CardinalPluginInfo* out_info) {
    out_info->api_version = CARDINAL_PLUGIN_API_VERSION;
    out_info->name        = "hello_script";
    out_info->version     = "1.0";
    out_info->author      = "Cardinal";
    out_info->description =
        "C++ scripting smoke test — compiled at runtime by cardinal::cppscript.";
    out_info->on_attach = [](const CardinalPluginHostApi* host) {
        g_host = host;
        if (g_host && g_host->log_info) {
            g_host->log_info("hello_script",
                "attached — compiled at runtime by cardinal::cppscript");
        }
    };
    out_info->on_tick = [](float /*dt*/) {
        // Heartbeat every ~120 ticks (~2 seconds at 60 FPS) so we don't
        // spam the log but still confirm the plugin is alive.
        const unsigned t = ++g_ticks;
        if (t % 120u == 0u && g_host && g_host->log_info) {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "tick %u (still alive)", t);
            g_host->log_info("hello_script", buf);
        }
    };
    out_info->on_detach = []() {
        if (g_host && g_host->log_info) {
            g_host->log_info("hello_script", "detached");
        }
        g_host = nullptr;
    };
}
