// =============================================================================
// Cardinal — sample plugin "hello".
//
// Builds as a DLL/SO that the host can drop into its `plugins/` folder.
// Demonstrates two things:
//
//   1) Lifecycle callbacks — attach / tick / detach with a once-per-second
//      log heartbeat, so the host's Log panel shows the plugin is alive.
//
//   2) Registering a custom *render algorithm* — drops a new "Hejl-Burgess
//      filmic" tonemap operator into the AlgoRegistry on attach. The host's
//      Render Pipeline panel picks it up automatically (no Studio recompile);
//      the user can select it from the Tonemap dropdown to apply it.
//
// The same pattern works for every algorithm category (Hash/RNG, Mip
// filter, Cluster culling, Sampling, Normal encoding, Tessellation
// factor) — drop in a CPU function, pair it with an HLSL function name,
// register, done.
// =============================================================================

#include <cardinal/plugin/plugin.hpp>

#include <algorithm>
#include <atomic>
#include <cstdio>

// AlgoIn/AlgoOut layout — must match cardinal::render::algo. We avoid
// including the engine header so the plugin doesn't pull in the engine's
// own (separate) copy of the AlgoRegistry static. The host sees the
// buffers through register_render_algo()'s opaque void* shape.
namespace algo_abi {
struct AlgoIn {
    float distance;
    float edge_pixels;
    cardinal::u32 seed;
    cardinal::u32 index;
    float color3[3];
    float unit3[3];
    float samples4[12];
};
struct AlgoOut {
    float color3[3];
    float unit3[3];
    float factor;
    int   flag;
};
}  // namespace algo_abi

namespace {
const CardinalPluginHostApi* g_host = nullptr;
std::atomic<float> g_elapsed{0.0f};
std::atomic<int>   g_ticks{0};

// ---- Custom render algorithm: Hejl-Burgess filmic ----------------------
// Jim Hejl + Richard Burgess-Dawson's optimised filmic curve. Cheap, bakes
// in sRGB-ish gamma so the result goes straight to the swap chain.
extern "C" void tonemap_hejl_burgess(const void* in_v, void* out_v) {
    const auto* in  = static_cast<const algo_abi::AlgoIn*>(in_v);
    auto*       out = static_cast<algo_abi::AlgoOut*>(out_v);
    auto saturate = [](float x){ return std::min(1.0f, std::max(0.0f, x)); };
    for (int i = 0; i < 3; ++i) {
        const float c = std::max(0.0f, in->color3[i] - 0.004f);
        out->color3[i] = saturate((c * (6.2f * c + 0.5f)) /
                                  (c * (6.2f * c + 1.7f) + 0.06f));
    }
}

void register_custom_algos() {
    if (g_host == nullptr || g_host->register_render_algo == nullptr) return;
    // category_id == 0 is Tonemap (matches CategoryId::Tonemap).
    g_host->register_render_algo(
        /*category_id*/  0,
        /*id*/           "hejl_burgess",
        /*label*/        "Hejl-Burgess (plugin)",
        /*tooltip*/      "Optimised filmic curve (Hejl + Burgess-Dawson). "
                         "Single rational + sRGB-ish gamma.",
        /*hlsl_function*/ "cardinal_tonemap_hejl_burgess",
        /*cpu_fn*/       &tonemap_hejl_burgess);
}

void on_attach(const CardinalPluginHostApi* host) {
    g_host = host;
    if (g_host) g_host->log_info("plugin/hello", "attached — hello, Cardinal!");
    register_custom_algos();
    if (g_host) g_host->log_info("plugin/hello",
        "registered tonemap algo 'Hejl-Burgess (plugin)' — "
        "select it from the Render Pipeline panel");
}

void on_tick(float dt) {
    const float prev = g_elapsed.fetch_add(dt);
    const int   t    = ++g_ticks;
    // Log a heartbeat once per second — keeps the host's log panel quiet.
    if (g_host && static_cast<int>(prev) != static_cast<int>(prev + dt) && t > 1) {
        char buf[64];
        std::snprintf(buf, sizeof(buf),
            "tick #%d (%.1fs)", t, static_cast<double>(prev + dt));
        g_host->log_info("plugin/hello", buf);
    }
}

void on_detach() {
    if (g_host) g_host->log_info("plugin/hello", "detached");
    g_host = nullptr;
}
}  // namespace

extern "C" CARDINAL_PLUGIN_EXPORT
void cardinal_plugin_register(CardinalPluginInfo* out) {
    if (out == nullptr) return;
    out->api_version = CARDINAL_PLUGIN_API_VERSION;
    out->name        = "hello";
    out->version     = "0.0.1";
    out->author      = "Cardinal";
    out->description = "Sample plugin: logs lifecycle messages every second.";
    out->on_attach   = &on_attach;
    out->on_tick     = &on_tick;
    out->on_detach   = &on_detach;
}
