// =============================================================================
// Cardinal — plugin registry implementation.
// =============================================================================
#include <cardinal/plugin/plugin.hpp>

#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/platform.hpp>
#include <cardinal/render/algos.hpp>
#include <cardinal/trace/stack.hpp>

#include <cardinal/core/std/containers.hpp>
#include <cardinal/core/std/filesystem.hpp>
#include <cardinal/core/std/utility.hpp>

#if CARDINAL_PLATFORM_WINDOWS
    #include <Windows.h>
    using LibHandle = HMODULE;
    static LibHandle lib_open (const char* p) { return LoadLibraryA(p); }
    static void*     lib_sym  (LibHandle h, const char* s) {
        return reinterpret_cast<void*>(GetProcAddress(h, s));
    }
    static void      lib_close(LibHandle h) { FreeLibrary(h); }
    static const char* lib_ext() { return ".dll"; }
#else
    #include <dlfcn.h>
    using LibHandle = void*;
    static LibHandle lib_open (const char* p) { return dlopen(p, RTLD_NOW | RTLD_LOCAL); }
    static void*     lib_sym  (LibHandle h, const char* s) { return dlsym(h, s); }
    static void      lib_close(LibHandle h) { dlclose(h); }
    static const char* lib_ext() { return ".so"; }
#endif

namespace cardinal::plugin {

struct Registry::Loaded {
    cardinal::string             path;
    LibHandle               lib{};
    CardinalPluginInfo      info{};
    // The plugin holds onto the host API pointer for its entire lifetime, so
    // we have to own a stable copy here rather than passing a stack local.
    CardinalPluginHostApi   host_api{};
    // Disabled when the plugin's tick raises an SEH exception — keeps the
    // editor up and lets the user inspect the captured crash in the Stack
    // Tracer panel.
    bool                    disabled{false};
};

namespace {

void log_info_thunk (const char* cat, const char* msg) { cardinal::log::infof (cat ? cat : "plugin", "%s", msg ? msg : ""); }
void log_warn_thunk (const char* cat, const char* msg) { cardinal::log::warnf (cat ? cat : "plugin", "%s", msg ? msg : ""); }
void log_error_thunk(const char* cat, const char* msg) { cardinal::log::errorf(cat ? cat : "plugin", "%s", msg ? msg : ""); }

// Plugins can't reach the host's AlgoRegistry directly because each
// statically-linked TU has its own static. This thunk wraps the registry
// behind a stable C ABI — the plugin's CPU function gets remarshalled
// here into the engine's strongly-typed AlgoIn/AlgoOut shape.
bool register_render_algo_thunk(cardinal::u32 category_id,
                                const char* id, const char* label,
                                const char* tooltip, const char* hlsl_function,
                                CardinalAlgoCpuFn cpu_fn)
{
    using namespace cardinal::render::algo;
    if (category_id >= static_cast<cardinal::u32>(CategoryId::Count_) ||
        id == nullptr || label == nullptr) return false;

    Algo a;
    a.category      = static_cast<CategoryId>(category_id);
    a.id            = id;
    a.label         = label;
    a.tooltip       = tooltip       ? tooltip       : "";
    a.hlsl_function = hlsl_function ? hlsl_function : "";
    a.is_user       = true;
    if (cpu_fn != nullptr) {
        // Forward the opaque buffers as the typed AlgoIn/AlgoOut. Both
        // sides have to agree on the pair's bit layout — the plugin
        // header documents the slot contract per category.
        a.cpu_fn = [cpu_fn](const AlgoIn& in, AlgoOut& out) {
            cpu_fn(static_cast<const void*>(&in), static_cast<void*>(&out));
        };
    }
    return AlgoRegistry::instance().register_algo(cardinal::move(a));
}

CardinalPluginHostApi make_host_api() {
    CardinalPluginHostApi api{};
    api.api_version          = CARDINAL_PLUGIN_API_VERSION;
    api.log_info             = &log_info_thunk;
    api.log_warn             = &log_warn_thunk;
    api.log_error            = &log_error_thunk;
    api.register_render_algo = &register_render_algo_thunk;
    return api;
}

}  // namespace

Registry& Registry::instance() {
    static Registry r;
    r.ensure_impl();
    return r;
}

void Registry::ensure_impl() {
    if (impl_ == nullptr) {
        impl_ = new cardinal::vector<cardinal::unique_ptr<Loaded>>();
    }
}

Registry::~Registry() {
    shutdown();
    delete impl_;
    impl_ = nullptr;
}

bool Registry::load(const char* file_path) {
    ensure_impl();
    if (file_path == nullptr) return false;

    // Skip already-loaded plugins by canonical path.
    cardinal::error_code ec;
    auto canon = cardinal::fs::weakly_canonical(file_path, ec).string();
    for (auto& p : *impl_) if (p->path == canon) return true;

    LibHandle h = lib_open(file_path);
    if (h == nullptr) {
        cardinal::log::errorf("plugin", "load failed: %s", file_path);
        return false;
    }
    auto fn = reinterpret_cast<CardinalPluginRegisterFn>(
        lib_sym(h, "cardinal_plugin_register"));
    if (fn == nullptr) {
        cardinal::log::warnf("plugin",
            "%s has no cardinal_plugin_register export — skipping", file_path);
        lib_close(h);
        return false;
    }

    CardinalPluginInfo info{};
    info.api_version = CARDINAL_PLUGIN_API_VERSION;
    fn(&info);

    if (info.api_version != CARDINAL_PLUGIN_API_VERSION) {
        cardinal::log::warnf("plugin",
            "%s reports API v%u but host is v%u — skipping",
            file_path, info.api_version, CARDINAL_PLUGIN_API_VERSION);
        lib_close(h);
        return false;
    }

    cardinal::log::infof("plugin",
        "loaded %s v%s by %s — %s",
        info.name        ? info.name        : "(unnamed)",
        info.version     ? info.version     : "?",
        info.author      ? info.author      : "?",
        info.description ? info.description : "");

    // Own the host API on the heap so the plugin's stored pointer stays
    // valid for the plugin's entire lifetime (the plugin will keep
    // dereferencing it during on_tick / on_detach).
    auto rec = cardinal::make_unique<Loaded>();
    rec->path     = canon;
    rec->lib      = h;
    rec->info     = info;
    rec->host_api = make_host_api();
    if (info.on_attach) info.on_attach(&rec->host_api);
    impl_->push_back(cardinal::move(rec));
    return true;
}

u32 Registry::load_directory(const char* dir) {
    ensure_impl();
    if (dir == nullptr) return 0;
    cardinal::error_code ec;
    if (!cardinal::fs::exists(dir, ec) ||
        !cardinal::fs::is_directory(dir, ec)) {
        cardinal::log::warnf("plugin", "directory missing: %s", dir);
        return 0;
    }
    u32 count = 0;
    for (auto& entry : cardinal::fs::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file(ec)) continue;
        const cardinal::string ext = entry.path().extension().string();
        if (ext != lib_ext()) continue;
        if (load(entry.path().string().c_str())) ++count;
    }
    cardinal::log::infof("plugin", "scanned %s — %u loaded", dir, count);
    return count;
}

#if CARDINAL_PLATFORM_WINDOWS
// SEH wrapper around the plugin's on_tick. Same trick as the compile worker:
// __try / __except can't coexist with C++ destructors in the same function,
// so the protected call lives in this no-RAII helper.
namespace {
DWORD invoke_tick_seh(void (*fn)(float), float dt) noexcept {
    DWORD code = 0;
    __try { fn(dt); return 0; }
    __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) { return code; }
}
}  // namespace
#endif

void Registry::tick(float dt) {
    ensure_impl();
    // Snapshot the tick entries before invoking any callback. A
    // plugin's on_tick may call back into the Registry (load/unload/
    // reload) — common via cppscript's hot-reload, where a watched
    // file change triggers Registry::unload + Registry::load. A
    // direct range-for over *impl_ would have the iterator dangle on
    // unload (vector erase) and potentially realloc on load. Same
    // iteration-callback UAF class as actor::World::broadcast
    // (b7f36e1), sim::run_group_ (1f10242), partition (309abdf).
    // Snapshot the function pointer + path (for re-find on SEH) so
    // disable-on-crash writes land on the current Loaded entry even
    // after callbacks shuffle the vector.
    struct TickEntry {
        const char*    name;
        cardinal::string path;
        void (*on_tick)(float);
    };
    cardinal::vector<TickEntry> snapshot;
    snapshot.reserve(impl_->size());
    for (auto& p : *impl_) {
        if (!p || p->disabled || p->info.on_tick == nullptr) continue;
        snapshot.push_back({ p->info.name, p->path, p->info.on_tick });
    }
    for (const auto& e : snapshot) {
#if CARDINAL_PLATFORM_WINDOWS
        const DWORD seh = invoke_tick_seh(e.on_tick, dt);
        if (seh != 0) {
            // Re-find the entry by path — the plugin may have
            // unloaded itself before crashing, in which case the
            // disable write lands on no entry (already gone).
            for (auto& p : *impl_) {
                if (p && p->path == e.path) { p->disabled = true; break; }
            }
            cardinal::log::errorf("plugin",
                "%s on_tick crashed (SEH 0x%08lx) — plugin disabled",
                e.name ? e.name : "(unnamed)", seh);
            // Capture a stack trace at the throw site for the panel.
            auto frames = trace::capture(0, 32);
            cardinal::log::errorf("plugin",
                "stack:\n%s", trace::format_full(frames).c_str());
        }
#else
        e.on_tick(dt);
#endif
    }
}

void Registry::shutdown() {
    if (impl_ == nullptr) return;
    // Snapshot the detach callbacks + lib handles before iterating.
    // A re-entrant on_detach that calls Registry::load/unload would
    // otherwise UAF the range-for (same class as Registry::tick
    // above). We're tearing down anyway so this loop is final —
    // any handlers added during a detach are still ignored by the
    // final clear() below.
    struct DetachEntry {
        void (*on_detach)();
        LibHandle lib;
    };
    cardinal::vector<DetachEntry> snapshot;
    snapshot.reserve(impl_->size());
    for (auto& p : *impl_) {
        if (p) snapshot.push_back({ p->info.on_detach, p->lib });
    }
    for (const auto& e : snapshot) {
        if (e.on_detach) e.on_detach();
        if (e.lib) lib_close(e.lib);
    }
    impl_->clear();
}

u32 Registry::loaded_count() const noexcept {
    return impl_ ? static_cast<u32>(impl_->size()) : 0u;
}

// Match by either the plugin's `name` field or its canonical file path.
// `Loaded` is a private nested type so this helper lives as a member rather
// than a free function (free functions in the same namespace would still
// fail access control on `Loaded`'s members).
bool Registry::matches_loaded_(const Loaded& p, const cardinal::string& q) {
    if (q.empty()) return false;
    if (p.info.name && q == p.info.name) return true;
    if (q == p.path) return true;
    // Tolerate user-typed paths that aren't canonicalised (e.g. relative).
    cardinal::error_code ec;
    auto canon = cardinal::fs::weakly_canonical(q, ec).string();
    return !ec && canon == p.path;
}

bool Registry::unload(const char* name_or_path) {
    ensure_impl();
    if (name_or_path == nullptr) return false;
    const cardinal::string q = name_or_path;
    for (auto it = impl_->begin(); it != impl_->end(); ++it) {
        auto& p = **it;
        if (!matches_loaded_(p, q)) continue;
        const cardinal::string nm = p.info.name ? p.info.name : "(unnamed)";
        // Detach is best-effort. We INTENTIONALLY don't SEH-wrap detach:
        // a plugin that crashes on detach is a programming bug we want to
        // see in a debugger, not silently swallow. If it becomes a problem
        // in shipping plugins we'll wrap then.
        if (p.info.on_detach) p.info.on_detach();
        if (p.lib) lib_close(p.lib);
        impl_->erase(it);
        cardinal::log::infof("plugin", "unloaded %s", nm.c_str());
        return true;
    }
    cardinal::log::warnf("plugin", "unload: no match for '%s'", name_or_path);
    return false;
}

bool Registry::reload(const char* name_or_path) {
    ensure_impl();
    if (name_or_path == nullptr) return false;
    const cardinal::string q = name_or_path;

    // Find the loaded record so we can recover its on-disk path BEFORE
    // unloading (since unload removes the entry from impl_).
    cardinal::string path_to_reload;
    for (auto& p : *impl_) {
        if (matches_loaded_(*p, q)) { path_to_reload = p->path; break; }
    }
    if (path_to_reload.empty()) {
        // Maybe the user passed a path of an unloaded plugin — try loading
        // it directly (treat reload as "load if missing, replace if present").
        return load(name_or_path);
    }

    if (!unload(name_or_path)) return false;
    return load(path_to_reload.c_str());
}

cardinal::vector<Registry::Info> Registry::enumerate() const {
    cardinal::vector<Info> out;
    if (impl_ == nullptr) return out;
    out.reserve(impl_->size());
    for (const auto& p : *impl_) {
        Info i;
        i.name        = p->info.name        ? p->info.name        : "";
        i.version     = p->info.version     ? p->info.version     : "";
        i.author      = p->info.author      ? p->info.author      : "";
        i.description = p->info.description ? p->info.description : "";
        i.path        = p->path;
        i.disabled    = p->disabled;
        out.push_back(cardinal::move(i));
    }
    return out;
}

}  // namespace cardinal::plugin
