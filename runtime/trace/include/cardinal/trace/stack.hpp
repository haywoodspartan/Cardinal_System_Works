#pragma once

// =============================================================================
// Cardinal — runtime stack tracing.
//
// Captures the current call stack of the calling thread (or any thread, by
// HANDLE) and resolves frames to "module!symbol+offset (file:line)" via
// DbgHelp. Used by:
//   - the Stack Tracer panel for on-demand inspection
//   - the plugin SEH handler when a plugin crashes
//   - assertion / log paths that want context
//
// On Linux this currently returns an empty vector; backtrace(3) + addr2line
// integration lands when WSI does (Phase 6).
// =============================================================================

#include <cardinal/core/types.hpp>

#include <string>
#include <vector>

namespace cardinal::trace {

struct StackFrame {
    u64         address{0};
    std::string module;     // dll/exe basename
    std::string symbol;     // demangled function name
    std::string file;       // source file (when available)
    u32         line{0};    // source line (0 if unknown)
};

// Capture the current thread's stack, skipping `skip` top frames (so the
// helper itself doesn't show up). Returns up to `max_depth` frames.
std::vector<StackFrame> capture(u32 skip = 0, u32 max_depth = 64);

// Capture another thread's stack — only valid on Windows for now (uses
// SuspendThread + GetThreadContext). Pass a HANDLE-equivalent void* on
// Windows; nullptr fallback returns the current thread's stack.
std::vector<StackFrame> capture_thread(void* native_thread_handle,
                                       u32 max_depth = 64);

// One-line formatter — useful for log lines.
std::string format(const StackFrame& f);

// Multi-line formatter — useful for crash dumps and the Studio panel.
std::string format_full(const std::vector<StackFrame>& frames,
                        const char* indent = "  ");

}  // namespace cardinal::trace
