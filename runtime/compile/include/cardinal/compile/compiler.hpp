#pragma once

// =============================================================================
// Cardinal — async shader / asset compilation service.
//
// Owns a background worker thread that processes compile jobs serially.
// The worker is wrapped in a structured-exception barrier on Windows so
// when a bad input crashes the compiler (DXC, etc.) the editor stays up;
// the failed job returns an error rather than taking the process down.
//
// Submitting a job returns a future-style handle that resolves when the
// worker finishes the job. Cancellation is best-effort — a cancelled job
// is dropped if it hasn't started, ignored if it's running.
//
// Goals (Phase 4-C):
//   ✓ Engine never blocks on a compile (UI stays at 60+ fps)
//   ✓ Bad shaders never kill the process — crash → error result
//   ✓ Single dependency-free C++23 wrapper around DXC + a worker thread
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/rhi/rhi.hpp>

#include <cardinal/core/std/future.hpp>   // also pulls core/types.hpp
                                      // (unique_ptr / string vocabulary)

namespace cardinal::compile {

// Result of a finished compile.
struct CompileResult {
    bool             ok{false};
    rhi::ShaderBlob  blob;            // populated on success
    cardinal::string      error;           // populated on failure
    bool             crashed{false};  // true when the worker caught SEH
    u64              elapsed_us{0};   // wall-clock time spent in DXC
};

struct CompileRequest {
    rhi::ShaderStage stage{rhi::ShaderStage::Vertex};
    cardinal::string      source;
    cardinal::string      entry_point;
    cardinal::string      filename;        // for error messages, optional
};

// Opaque future-style handle returned by submit(). The caller polls
// `is_ready()` from the main thread, then `take()` to consume the result.
class CompileFuture {
public:
    explicit CompileFuture(cardinal::future<CompileResult> fut);
    bool          is_ready() const noexcept;
    CompileResult take();          // blocks until ready
private:
    cardinal::future<CompileResult> fut_;
};

// Worker. One per process is plenty — DXC isn't multithread-safe inside a
// single instance, but its serial throughput is fine for an engine.
class Worker {
public:
    static cardinal::unique_ptr<Worker> create();
    virtual ~Worker() = default;

    // Submit a compile. Always returns a valid future, even if the worker
    // is shutting down (the future will resolve to an error result).
    virtual CompileFuture submit(const CompileRequest& req) = 0;

    // How many jobs are currently queued (informational — racy).
    virtual u32 pending() const noexcept = 0;

    // Drain + stop. Called by the destructor; explicit shutdown is useful
    // when the host wants to control teardown ordering.
    virtual void shutdown() = 0;

protected:
    Worker() = default;
};

}  // namespace cardinal::compile
