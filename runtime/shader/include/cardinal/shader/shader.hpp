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
#include <cardinal/core/std/containers.hpp>

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

// When the engine's shaders are turned into cached bytecode:
//   OnDemand — each shader is compiled lazily the first time compile() is
//              called for it (fast startup, a one-time hitch on first use).
//   UpFront  — the host warms the whole known set via precompile() at boot
//              (no in-frame compile hitches; slower startup). The on-disk
//              cache means a warmed set survives across runs, so "up front"
//              after the first run is mostly cache-hit reload.
// The mode is a POLICY the Compiler records for diagnostics/UI; compile()
// behaves identically either way. The host enacts UpFront by calling
// precompile() (typically with every pass's hlsl_source()) during init.
enum class CompileMode : u32 { OnDemand = 0, UpFront = 1 };
const char* compile_mode_name(CompileMode m) noexcept;

// Summary of one precompile() batch.
struct PrecompileReport {
    u32 requested{0};    // requests submitted
    u32 compiled{0};     // freshly compiled (cache miss)
    u32 from_cache{0};   // served from the on-disk / in-memory cache
    u32 failed{0};       // failed to compile (see per-result diagnostics)
    f64 seconds{0.0};    // summed compile time across the batch
    bool ok() const noexcept { return failed == 0; }
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

    // ---- Compile mode (on-demand vs up-front) ----------------------
    // Record the host's compilation policy (default OnDemand). Pure state —
    // does not change compile()'s behavior; lets the Studio's shader panel
    // show + toggle the active mode, and lets the host decide whether to
    // warm the cache at boot.
    void        set_mode(CompileMode m) noexcept;
    CompileMode mode() const noexcept;

    // Up-front warm: compile every request now, filling the on-disk +
    // in-memory cache so later compile() calls for the same requests are
    // cache hits with no in-frame hitch. Idempotent + cheap to repeat (a
    // second call is all cache hits). Returns a per-batch report; never
    // throws (a failed entry is counted, not fatal). This is the explicit
    // "compile up front" action — independent of the mode flag, though a
    // host in UpFront mode calls it during init with every known shader.
    PrecompileReport precompile(const cardinal::vector<CompileRequest>& requests);

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
