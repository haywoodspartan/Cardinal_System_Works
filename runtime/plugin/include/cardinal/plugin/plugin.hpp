#pragma once

// =============================================================================
// Cardinal — runtime plugin system.
//
// Each plugin is a DLL/SO exposing a single C entry point named
// `cardinal_plugin_register`. The host (engine or editor) loads the binary
// with LoadLibrary / dlopen, resolves the symbol, and calls it with a
// versioned PluginHostApi struct.
//
// The plugin returns a PluginInfo describing itself + a few callbacks:
//   on_attach  — called once after registration (state, sub-systems, etc.)
//   on_tick    — called every frame the host is processing the engine
//   on_detach  — called once during shutdown / unload
//
// Plugins are pure observers in this baseline cut. Phase 4-D2 will add a
// PluginRegistry surface (register_panel, register_command) so plugins can
// contribute editor UI, asset importers, render passes, etc.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/containers.hpp>

extern "C" {

// Bumped every time PluginHostApi or PluginInfo grows. Plugins compiled
// against a different version are rejected with a logged message.
inline constexpr cardinal::u32 CARDINAL_PLUGIN_API_VERSION = 2;

// Render-algorithm registration through the plugin host API. The plugin
// can't call cardinal::render::algo::AlgoRegistry directly because each
// statically-linked TU has its own copy of the registry — the host owns
// the canonical one. Pass an opaque CPU function pointer + the same
// metadata fields as cardinal::render::algo::Algo.
//
// `category_id` matches cardinal::render::algo::CategoryId numerically:
//   0 = Tonemap, 1 = Hash/RNG, 2 = Mip filter, 3 = Cluster cull,
//   4 = Tess factor, 5 = Sampling, 6 = Normal encoding
//
// The CPU function uses the same opaque (in_buffer, out_buffer) shape as
// the engine's Algo::AlgoFn — see runtime/render/include/cardinal/render/algos.hpp.
typedef void (*CardinalAlgoCpuFn)(const void* in, void* out);

struct CardinalPluginHostApi {
    cardinal::u32 api_version;             // == CARDINAL_PLUGIN_API_VERSION
    void (*log_info) (const char* category, const char* message);
    void (*log_warn) (const char* category, const char* message);
    void (*log_error)(const char* category, const char* message);

    // Register a render algorithm. Returns true on success, false if the
    // category id is unknown or the (category,id) pair is already taken.
    bool (*register_render_algo)(cardinal::u32 category_id,
                                 const char*   id,
                                 const char*   label,
                                 const char*   tooltip,
                                 const char*   hlsl_function,
                                 CardinalAlgoCpuFn cpu_fn);
    // Future fields: register_panel, register_command, asset_importer, …
};

struct CardinalPluginInfo {
    cardinal::u32 api_version;             // plugin echoes the host's value
    const char*   name;
    const char*   version;
    const char*   author;
    const char*   description;

    // Callbacks. Any may be nullptr.
    void (*on_attach)(const CardinalPluginHostApi* host);
    void (*on_tick)  (float dt_seconds);
    void (*on_detach)();
};

// Plugins implement this and export it (CARDINAL_PLUGIN_EXPORT below).
typedef void (*CardinalPluginRegisterFn)(CardinalPluginInfo* out_info);

}  // extern "C"

#if defined(_WIN32)
    #define CARDINAL_PLUGIN_EXPORT __declspec(dllexport)
#else
    #define CARDINAL_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

namespace cardinal::plugin {

// Host-side registry — owns loaded DLLs, dispatches lifecycle events.
class Registry {
public:
    static Registry& instance();

    // Scan a directory for *.dll / *.so plugins and load each one. Returns
    // the count of successfully attached plugins. Safe to call multiple
    // times — already-loaded files are skipped.
    u32 load_directory(const char* dir);

    // Manual single-file load.
    bool load(const char* file_path);

    // Per-frame dispatch — calls on_tick on every loaded plugin. Crashes
    // inside a plugin's tick are localised via the same SEH trick used in
    // the compile worker, so a faulty plugin can't take the editor down.
    void tick(float dt_seconds);

    // Detach + unload everything. Called automatically at process exit.
    void shutdown();

    u32 loaded_count() const noexcept;

    // ---- Hot unload / reload --------------------------------------------
    //
    // Both lookup forms work:
    //   - by plugin's `name` field (case-sensitive)
    //   - by canonical file path
    //
    // unload() detaches + FreeLibrary's the DLL. Subsequent attempts to
    // load the same path succeed (the path is removed from the loaded set).
    // Returns true on success; false if no match.
    //
    // reload() = unload + load. Used by the cppscript hot-reload path
    // (file watcher → recompile → reload). The DLL must already be on
    // disk at the new path; if the load fails, the plugin is gone (we
    // don't keep an old version around to revert to — that's the caller's
    // responsibility via two-phase load).
    bool unload (const char* name_or_path);
    bool reload (const char* name_or_path);

    // Snapshot of every loaded plugin's metadata + status — used by the
    // Studio panel + the `plugin.list` console command.
    struct Info {
        cardinal::string name;
        cardinal::string version;
        cardinal::string author;
        cardinal::string description;
        cardinal::string path;
        bool        disabled{false};   // true if the plugin's tick crashed
    };
    cardinal::vector<Info> enumerate() const;

private:
    Registry() = default;
    Registry(const Registry&)            = delete;
    Registry& operator=(const Registry&) = delete;
    ~Registry();

    struct Loaded;
    cardinal::vector<cardinal::unique_ptr<Loaded>>* impl_{nullptr};   // PIMPL via header-only fwd
    void ensure_impl();
    static bool matches_loaded_(const Loaded& p, const cardinal::string& q);
};

}  // namespace cardinal::plugin
