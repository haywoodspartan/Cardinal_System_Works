#pragma once

// =============================================================================
// Cardinal — Shader compiler facade.
//
// Wraps cardinal::rhi::Device::compile_shader with:
//   - On-disk binary cache keyed by hash(source + entry + stage + defines)
//   - Hot-reload via mtime polling on watched files
//   - Variant generation: pass a list of preprocessor defines, get back
//     N compiled blobs
//
// The Compiler does NOT own the rhi::Device — it borrows one passed at
// construction. Multiple Compilers may share a device.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/containers.hpp>

namespace cardinal::rhi { class Device; }

namespace cardinal::shader {

enum class Stage : u32 { Vertex = 0, Fragment = 1, Compute = 2 };
const char* stage_name(Stage s) noexcept;

struct CompileRequest {
    cardinal::string                          source_path;     // for diagnostics + watch
    cardinal::string                          source_text;     // the actual HLSL
    cardinal::string                          entry_point{"main"};
    Stage                                stage{Stage::Vertex};
    cardinal::vector<cardinal::string>             defines;         // "FOO=1" / "BAR"
};

struct CompileResult {
    bool                                 ok{false};
    cardinal::vector<u8>                      bytecode;        // SPIR-V or DXIL
    cardinal::string                          diagnostics;
    u64                                  cache_key{0};
    bool                                 served_from_cache{false};
    f64                                  compile_seconds{0.0};
};

// ---------------------------------------------------------------------------
// Compiler — process-level cache + facade.
// ---------------------------------------------------------------------------
class Compiler {
public:
    static cardinal::shared_ptr<Compiler> create(cardinal::rhi::Device& device,
                                            cardinal::string cache_dir);

    // Compile one shader. Hits the on-disk cache when possible.
    CompileResult compile(const CompileRequest& req);

    // Preprocessor variant batch. Each entry is a definitions vector;
    // common defines (passed in `req`) are merged with each entry. Returns
    // one CompileResult per entry, in order.
    cardinal::vector<CompileResult> compile_variants(
        const CompileRequest& base,
        const cardinal::vector<cardinal::vector<cardinal::string>>& variant_defines);

    // ---- Hot-reload watch ------------------------------------------
    // Watch a source file for mtime changes. When a change is observed
    // during tick(), the file is reloaded from disk and recompiled, then
    // `on_change` is invoked with the fresh CompileResult.
    using OnChange = cardinal::function<void(const CompileResult&)>;
    using WatchHandle = u32;
    WatchHandle watch(const cardinal::string& source_path,
                      const cardinal::string& entry_point,
                      Stage stage,
                      const cardinal::vector<cardinal::string>& defines,
                      OnChange on_change);
    void        unwatch(WatchHandle h);

    // Drives hot-reload polling; cheap (one stat() per watched file).
    // When event-driven watching is enabled (see below) the stat() is
    // skipped — tick() only walks the dirty set.
    void tick();

    // Switch from per-frame mtime polling to event-driven file-change
    // notifications. Creates a cardinal::core::FileWatcher on `root`
    // (recursive) that flips a per-watch dirty flag whenever any file
    // in the directory changes. tick() then only acts on flagged
    // watches — no syscalls per frame regardless of watch count.
    //
    // Returns false if the watcher couldn't be created (bad path, or
    // platform not yet supported). Safe to call multiple times — the
    // previous watcher is dropped first.
    bool start_event_driven_watch(const cardinal::string& root);
    void stop_event_driven_watch();
    bool event_driven_watch_active() const noexcept;

    struct Stats {
        u32 cached_blobs{0};
        u64 cache_bytes{0};
        u64 compiles_total{0};
        u64 cache_hits{0};
        u32 watched_files{0};
        u64 hot_reloads{0};
    };
    Stats stats() const noexcept;

private:
    Compiler() = default;
    bool initialize_(cardinal::rhi::Device& device, cardinal::string cache_dir);

    struct Impl;
    Impl* impl_{nullptr};
public:
    ~Compiler();
};

}  // namespace cardinal::shader
