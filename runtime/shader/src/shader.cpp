#include <cardinal/shader/shader.hpp>

#include <cardinal/core/file_watcher.hpp>
#include <cardinal/core/log.hpp>
#include <cardinal/rhi/rhi.hpp>

#include <cardinal/core/atomic.hpp>
#include <cardinal/core/chrono.hpp>
#include <cardinal/core/containers.hpp>
#include <cardinal/core/cstdio.hpp>
#include <cardinal/core/filesystem.hpp>
#include <cardinal/core/fstream.hpp>
#include <cardinal/core/thread.hpp>
#include <cardinal/core/utility.hpp>

namespace fs = cardinal::fs;

namespace cardinal::shader {

const char* stage_name(Stage s) noexcept {
    switch (s) {
        case Stage::Vertex:   return "vertex";
        case Stage::Fragment: return "fragment";
        case Stage::Compute:  return "compute";
    }
    return "?";
}

namespace {

u64 fnv1a64(const u8* data, usize n) noexcept {
    u64 h = 0xcbf29ce484222325ull;
    for (usize i = 0; i < n; ++i) { h ^= data[i]; h *= 0x100000001b3ull; }
    return h;
}

u64 hash_request(const CompileRequest& r) noexcept {
    u64 h = fnv1a64(reinterpret_cast<const u8*>(r.source_text.data()),
                    r.source_text.size());
    h = h ^ fnv1a64(reinterpret_cast<const u8*>(r.entry_point.data()),
                    r.entry_point.size());
    h = h ^ static_cast<u64>(r.stage);
    for (const auto& d : r.defines) {
        h = h ^ fnv1a64(reinterpret_cast<const u8*>(d.data()), d.size());
    }
    return h;
}

bool read_all(const cardinal::string& path, cardinal::vector<u8>& out) {
    cardinal::ifstream f(path, cardinal::ios::binary);
    if (!f) return false;
    f.seekg(0, cardinal::ios::end);
    const auto n = f.tellg();
    f.seekg(0, cardinal::ios::beg);
    out.resize(static_cast<usize>(n));
    f.read(reinterpret_cast<char*>(out.data()), n);
    return true;
}

bool write_all(const cardinal::string& path, const cardinal::vector<u8>& bytes) {
    cardinal::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    cardinal::ofstream f(path, cardinal::ios::binary | cardinal::ios::trunc);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<cardinal::streamsize>(bytes.size()));
    return true;
}

cardinal::rhi::ShaderStage to_rhi_stage(Stage s) noexcept {
    switch (s) {
        case Stage::Vertex:   return cardinal::rhi::ShaderStage::Vertex;
        case Stage::Fragment: return cardinal::rhi::ShaderStage::Fragment;
        case Stage::Compute:  return cardinal::rhi::ShaderStage::Compute;
    }
    return cardinal::rhi::ShaderStage::Vertex;
}

}  // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------
struct Compiler::Impl {
    cardinal::rhi::Device* device{nullptr};
    cardinal::string            cache_dir;
    cardinal::mutex             mtx;
    // In-memory bytecode cache keyed by hash.
    cardinal::unordered_map<u64, cardinal::vector<u8>> mem_cache;
    cardinal::atomic<u64>       compiles_total{0};
    cardinal::atomic<u64>       cache_hits    {0};
    cardinal::atomic<u64>       hot_reloads   {0};

    struct Watch {
        WatchHandle id;
        cardinal::string source_path;
        cardinal::string entry_point;
        Stage stage;
        cardinal::vector<cardinal::string> defines;
        OnChange on_change;
        u64 mtime_ns{0};
    };
    cardinal::unordered_map<WatchHandle, Watch> watches;
    WatchHandle next_id{1};

    // Event-driven mode — when active, file_watcher_ owns a worker thread
    // that pushes changed paths into dirty_paths_. tick() walks just the
    // dirty set (under the same mutex as watches) and recompiles
    // matching watches, then clears the set. The mtime poll is skipped.
    cardinal::unique_ptr<cardinal::core::FileWatcher> file_watcher;
    cardinal::unordered_set<cardinal::string>              dirty_paths;
    cardinal::string                                  watch_root;
};

cardinal::shared_ptr<Compiler> Compiler::create(cardinal::rhi::Device& device,
                                           cardinal::string cache_dir)
{
    auto c = cardinal::shared_ptr<Compiler>(new Compiler());
    if (!c->initialize_(device, cardinal::move(cache_dir))) return nullptr;
    return c;
}

Compiler::~Compiler() { delete impl_; }

bool Compiler::initialize_(cardinal::rhi::Device& device, cardinal::string cache_dir) {
    impl_ = new Impl{};
    impl_->device    = &device;
    impl_->cache_dir = cardinal::move(cache_dir);
    cardinal::error_code ec;
    fs::create_directories(impl_->cache_dir, ec);
    cardinal::log::infof("shader",
        "compiler online — cache=%s", impl_->cache_dir.c_str());
    return true;
}

CompileResult Compiler::compile(const CompileRequest& req) {
    using clock = cardinal::chrono::high_resolution_clock;
    const auto t0 = clock::now();

    CompileResult cr{};
    cr.cache_key = hash_request(req);

    // 1) In-memory cache hit?
    {
        cardinal::lock_guard<cardinal::mutex> lg(impl_->mtx);
        auto it = impl_->mem_cache.find(cr.cache_key);
        if (it != impl_->mem_cache.end()) {
            cr.ok = true;
            cr.bytecode = it->second;
            cr.served_from_cache = true;
            cr.compile_seconds = cardinal::chrono::duration<double>(clock::now() - t0).count();
            impl_->cache_hits.fetch_add(1, cardinal::memory_order_relaxed);
            return cr;
        }
    }

    // 2) On-disk cache hit?
    char hex[32];
    cardinal::snprintf(hex, sizeof(hex), "%016llx.bin",
                  static_cast<unsigned long long>(cr.cache_key));
    const cardinal::string cache_path = impl_->cache_dir + "/" + hex;
    {
        cardinal::vector<u8> bytes;
        if (read_all(cache_path, bytes) && !bytes.empty()) {
            cr.ok = true;
            cr.bytecode = cardinal::move(bytes);
            cr.served_from_cache = true;
            cr.compile_seconds = cardinal::chrono::duration<double>(clock::now() - t0).count();
            cardinal::lock_guard<cardinal::mutex> lg(impl_->mtx);
            impl_->mem_cache[cr.cache_key] = cr.bytecode;
            impl_->cache_hits.fetch_add(1, cardinal::memory_order_relaxed);
            return cr;
        }
    }

    // 3) Real compile via RHI.
    cardinal::string text = req.source_text;
    // Prepend defines as `#define X` lines so the compiler honours them
    // even if it doesn't take a defines list directly.
    if (!req.defines.empty()) {
        cardinal::string prefix;
        for (const auto& d : req.defines) {
            const auto eq = d.find('=');
            if (eq == cardinal::string::npos) {
                prefix += "#define " + d + "\n";
            } else {
                prefix += "#define " + d.substr(0, eq) + " " + d.substr(eq + 1) + "\n";
            }
        }
        text = prefix + text;
    }

    auto blob = impl_->device->compile_shader(to_rhi_stage(req.stage),
                                               text.c_str(),
                                               req.entry_point.c_str());
    if (!blob.ok()) {
        cr.ok = false;
        cr.diagnostics = "RHI compile_shader failed";
        impl_->compiles_total.fetch_add(1, cardinal::memory_order_relaxed);
        return cr;
    }
    cr.ok = true;
    cr.bytecode = cardinal::move(blob.bytes);
    cr.compile_seconds = cardinal::chrono::duration<double>(clock::now() - t0).count();
    impl_->compiles_total.fetch_add(1, cardinal::memory_order_relaxed);

    write_all(cache_path, cr.bytecode);
    {
        cardinal::lock_guard<cardinal::mutex> lg(impl_->mtx);
        impl_->mem_cache[cr.cache_key] = cr.bytecode;
    }
    return cr;
}

cardinal::vector<CompileResult> Compiler::compile_variants(
    const CompileRequest& base,
    const cardinal::vector<cardinal::vector<cardinal::string>>& variant_defines)
{
    cardinal::vector<CompileResult> out;
    out.reserve(variant_defines.size());
    for (const auto& vd : variant_defines) {
        CompileRequest r = base;
        for (const auto& d : vd) r.defines.push_back(d);
        out.push_back(compile(r));
    }
    return out;
}

Compiler::WatchHandle Compiler::watch(const cardinal::string& source_path,
                                      const cardinal::string& entry_point,
                                      Stage stage,
                                      const cardinal::vector<cardinal::string>& defines,
                                      OnChange on_change)
{
    cardinal::lock_guard<cardinal::mutex> lg(impl_->mtx);
    Impl::Watch w{};
    w.id          = impl_->next_id++;
    w.source_path = source_path;
    w.entry_point = entry_point;
    w.stage       = stage;
    w.defines     = defines;
    w.on_change   = cardinal::move(on_change);
    cardinal::error_code ec;
    auto t = fs::last_write_time(source_path, ec);
    w.mtime_ns = ec ? 0 :
        static_cast<u64>(cardinal::chrono::duration_cast<cardinal::chrono::nanoseconds>(
            t.time_since_epoch()).count());
    impl_->watches.emplace(w.id, cardinal::move(w));
    return impl_->watches.size() ? (impl_->next_id - 1) : 0;
}

void Compiler::unwatch(WatchHandle h) {
    cardinal::lock_guard<cardinal::mutex> lg(impl_->mtx);
    impl_->watches.erase(h);
}

void Compiler::tick() {
    cardinal::vector<Impl::Watch> changed;
    {
        cardinal::lock_guard<cardinal::mutex> lg(impl_->mtx);

        if (impl_->file_watcher != nullptr) {
            // Event-driven path: only walk paths the FileWatcher has
            // flagged as dirty since the last tick. No syscalls,
            // O(dirty) instead of O(watches).
            if (!impl_->dirty_paths.empty()) {
                for (auto& [id, w] : impl_->watches) {
                    // The watcher reports paths relative to its root.
                    // Match either as full path or as suffix (so the
                    // user can register watches with absolute or
                    // relative paths and both work).
                    for (const auto& dp : impl_->dirty_paths) {
                        if (w.source_path == dp ||
                            (w.source_path.size() >= dp.size() &&
                             w.source_path.compare(
                                w.source_path.size() - dp.size(),
                                dp.size(), dp) == 0))
                        {
                            changed.push_back(w);
                            break;
                        }
                    }
                }
                impl_->dirty_paths.clear();
            }
        } else {
            // Legacy poll path: stat() every watched file each tick.
            for (auto& [id, w] : impl_->watches) {
                cardinal::error_code ec;
                auto t = fs::last_write_time(w.source_path, ec);
                const u64 cur = ec ? 0 :
                    static_cast<u64>(cardinal::chrono::duration_cast<cardinal::chrono::nanoseconds>(
                        t.time_since_epoch()).count());
                if (cur != w.mtime_ns) {
                    w.mtime_ns = cur;
                    changed.push_back(w);
                }
            }
        }
    }
    for (auto& w : changed) {
        cardinal::vector<u8> bytes;
        if (!read_all(w.source_path, bytes)) continue;
        CompileRequest r{};
        r.source_path = w.source_path;
        r.source_text.assign(bytes.begin(), bytes.end());
        r.entry_point = w.entry_point;
        r.stage       = w.stage;
        r.defines     = w.defines;
        // Force re-compile by removing the cache entry so we don't return
        // the stale blob.
        const u64 key = hash_request(r);
        {
            cardinal::lock_guard<cardinal::mutex> lg(impl_->mtx);
            impl_->mem_cache.erase(key);
        }
        auto cr = compile(r);
        impl_->hot_reloads.fetch_add(1, cardinal::memory_order_relaxed);
        if (w.on_change) w.on_change(cr);
    }
}

bool Compiler::start_event_driven_watch(const cardinal::string& root) {
    // Stop any prior watcher first — we only own one root at a time.
    stop_event_driven_watch();

    // The FileWatcher callback runs on its own worker thread. All we
    // do here is push the path into dirty_paths under our mutex; tick()
    // (called from the main thread) drains it. No GPU resource creation
    // happens on the watcher thread.
    auto watcher = cardinal::core::FileWatcher::create(
        root, /*recursive*/ true,
        [this](const cardinal::core::FileEvent& e) {
            if (e.kind == cardinal::core::FileEventKind::Removed) return;
            cardinal::lock_guard<cardinal::mutex> lg(impl_->mtx);
            impl_->dirty_paths.insert(e.path);
        });

    if (watcher == nullptr) {
        cardinal::log::warnf("shader",
            "start_event_driven_watch(%s) failed — falling back to mtime poll",
            root.c_str());
        return false;
    }

    {
        cardinal::lock_guard<cardinal::mutex> lg(impl_->mtx);
        impl_->file_watcher = cardinal::move(watcher);
        impl_->watch_root   = root;
    }
    cardinal::log::infof("shader",
        "event-driven hot-reload active on %s", root.c_str());
    return true;
}

void Compiler::stop_event_driven_watch() {
    cardinal::unique_ptr<cardinal::core::FileWatcher> doomed;
    {
        cardinal::lock_guard<cardinal::mutex> lg(impl_->mtx);
        doomed = cardinal::move(impl_->file_watcher);
        impl_->dirty_paths.clear();
        impl_->watch_root.clear();
    }
    // doomed destructor joins its worker thread OUTSIDE our mutex so a
    // pending callback racing with stop doesn't deadlock on the mutex.
}

bool Compiler::event_driven_watch_active() const noexcept {
    cardinal::lock_guard<cardinal::mutex> lg(impl_->mtx);
    return impl_->file_watcher != nullptr;
}

Compiler::Stats Compiler::stats() const noexcept {
    Stats s{};
    cardinal::lock_guard<cardinal::mutex> lg(impl_->mtx);
    s.cached_blobs   = static_cast<u32>(impl_->mem_cache.size());
    for (const auto& [_, b] : impl_->mem_cache) s.cache_bytes += b.size();
    s.compiles_total = impl_->compiles_total.load(cardinal::memory_order_relaxed);
    s.cache_hits     = impl_->cache_hits.load(cardinal::memory_order_relaxed);
    s.watched_files  = static_cast<u32>(impl_->watches.size());
    s.hot_reloads    = impl_->hot_reloads.load(cardinal::memory_order_relaxed);
    return s;
}

}  // namespace cardinal::shader
