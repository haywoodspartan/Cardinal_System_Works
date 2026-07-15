#pragma once

// =============================================================================
// Cardinal — crash reporter.
//
// Installs a process-wide unhandled-exception filter (Win32 SEH) that, on
// any uncaught structured exception, writes:
//   1) A minidump (.dmp) capturing thread state, stacks, modules
//   2) A short text crash log with the exception code, faulting address,
//      and module/symbol if available
//   3) The line that fired plus the last few cardinal::log lines
//
// Files land in a `crashes/` directory next to the executable, named with a
// timestamp + PID:
//
//   crashes/2026-05-11_142307_PID12348.dmp
//   crashes/2026-05-11_142307_PID12348.log
//
// Usage at boot (before any work that could crash):
//
//     cardinal::crash::install();   // one call, no return value needed
//
// The filter is process-wide. It runs ONCE per process — after the first
// crash it returns EXCEPTION_EXECUTE_HANDLER and lets the OS terminate
// (we don't try to "recover"). If you want the OS-default Windows Error
// Reporting dialog after we've written our dump, set `also_default = true`
// in CrashConfig.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <string>

namespace cardinal::crash {

struct CrashConfig {
    // Where to write dumps + logs. Empty → "<exe-dir>/crashes".
    std::string dump_dir{};

    // After our dump is written, return EXCEPTION_CONTINUE_SEARCH so
    // Windows Error Reporting / a debugger gets a shot. When false
    // (default) we return EXCEPTION_EXECUTE_HANDLER so the process
    // terminates cleanly with our log + .dmp on disk.
    bool also_default{false};

    // MiniDumpWriteDump type. The default is a balance — captures
    // enough to debug a stack/heap walk without writing 4 GiB of full-
    // memory dump. Override to MiniDumpWithFullMemory if you're chasing
    // a heap-corruption bug.
    //   small     : MiniDumpNormal | WithThreadInfo | WithUnloadedModules
    //   default   : small + WithDataSegs + WithHandleData + WithIndirectlyReferencedMemory
    //   full      : default + WithFullMemory  (huge)
    enum class Detail : u8 { Small, Default, Full };
    Detail detail{Detail::Default};
};

// Install the unhandled-exception filter. Idempotent — repeated calls
// reuse the existing filter. Returns true on success; false on Linux
// (no-op stub) or when the platform doesn't expose a way to install.
bool install(const CrashConfig& cfg = {});

// Remove the filter. Useful in unit tests so a deliberate crash in one
// test doesn't terminate the harness.
void uninstall() noexcept;

// Manually write a dump + log right now (for "non-fatal but worth
// investigating" cases). Returns the path the dump was written to, or
// empty on failure.
std::string write_dump_now(const char* reason = "manual");

// Returns the last dump path the filter wrote (or write_dump_now), so
// the host can surface "your last crash dump is at X" in the UI.
std::string last_dump_path() noexcept;

// ---------------------------------------------------------------------------
// Hang watchdog — catches the failure crashes never report: the main loop
// FREEZING (Windows "stopped interacting and was closed", Event 1002, no
// dump, no stack — exactly the 2026-07-15 Studio incident). A background
// thread checks a heartbeat the main loop pokes every frame; if no poke
// lands for `stall_seconds`, it writes ONE full-thread minidump + log line
// (via write_dump_now, so install()'s dump_dir/detail apply) and keeps the
// process running — the frozen threads' stacks are in the dump.
//
//     cardinal::crash::install();
//     cardinal::crash::watchdog_start(8);
//     while (frame) { cardinal::crash::watchdog_poke(); ... }
//     cardinal::crash::watchdog_stop();
//
// Single-fire per start; poke/stop are safe from any thread.
// ---------------------------------------------------------------------------
bool watchdog_start(u32 stall_seconds = 8);
void watchdog_poke() noexcept;
void watchdog_stop() noexcept;
// True once the watchdog has fired since the last start (for tests/UI).
bool watchdog_fired() noexcept;

}  // namespace cardinal::crash
