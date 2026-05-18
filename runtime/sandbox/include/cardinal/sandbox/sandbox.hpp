#pragma once

// =============================================================================
// Cardinal — process-isolation sandbox for plugin / cppscript code.
//
// Two execution modes per loaded plugin:
//
//   InProcess  — the plugin's DLL loads into the editor's process. Crashes
//                are caught by cardinal::plugin::Registry's SEH __try /
//                __except wrapper around on_tick: synchronous hardware
//                exceptions (AVs, divide-by-zero, illegal instruction,
//                stack overflow) auto-disable the plugin without taking
//                the editor down. Heap corruption / data races / UB that
//                doesn't trip a hardware fault are NOT survivable in this
//                mode — the plugin shares the host's address space.
//
//   Subprocess — the plugin loads into a child process
//                (cardinal_sandbox_runner.exe), and the host communicates
//                with it over a per-sandbox named pipe. ANY crash inside
//                the plugin (heap corruption, abort, ExitProcess, infinite
//                loop, etc.) only kills the child; the host detects the
//                death + marks the sandbox dead + keeps running. Per-tick
//                timeout (default 250 ms) bounds runner stalls. Cost: one
//                IPC roundtrip per tick (~tens of microseconds local) and
//                a forked address space (no shared GPU resources).
//
// Wire protocol (Subprocess mode)
// -------------------------------
// Pipe name: \\.\pipe\cardinal_sandbox_<host_pid>_<sandbox_index>
// Frames are length-prefixed: every message is u32 opcode + u32 payload_len
// + bytes payload. Multi-byte values little-endian.
//
// Host → runner:
//   0x01 ATTACH   payload: u32 dll_path_len + bytes dll_path
//                 Runner LoadLibrarys the DLL + calls plugin's on_attach.
//                 Runner replies READY (0x80) on success, ERROR (0x83) on fail.
//   0x02 TICK     payload: f32 dt
//                 Runner calls plugin's on_tick(dt) inside SEH __try.
//                 Replies ACK (0x81) on success, ERROR on SEH catch.
//   0x03 DETACH   payload: empty
//                 Runner calls plugin's on_detach + exits cleanly with rc=0.
//
// Runner → host (out-of-band — host reads opportunistically between requests):
//   0x80 READY    payload: u32 name_len + bytes name + u32 ver_len + bytes ver
//                 Sent once after a successful ATTACH.
//   0x81 ACK      payload: empty. Sent after every TICK that survived SEH.
//   0x82 LOG      payload: u32 level (0 trace / 1 info / 2 warn / 3 error)
//                          + u32 cat_len + bytes cat
//                          + u32 msg_len + bytes msg
//                 Sent inline whenever the plugin calls log_info / warn / error
//                 through the host_api. Host re-emits via cardinal::log.
//   0x83 ERROR    payload: u32 msg_len + bytes msg
//                 Sent on ATTACH failure or on caught SEH inside TICK.
//                 After ERROR the runner stays alive (next TICK is retried)
//                 unless the SEH was unrecoverable, in which case the runner
//                 exits with rc != 0 and the host detects via process death.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <memory>
#include <string>

namespace cardinal::sandbox {

// Wire protocol opcodes — exposed publicly so the runner exe + the host
// implementation share one definition (link them via this header).
namespace proto {
    inline constexpr u32 OP_ATTACH = 0x01;
    inline constexpr u32 OP_TICK   = 0x02;
    inline constexpr u32 OP_DETACH = 0x03;
    // 0x04 ATTACH_VM  payload: u32 module_len + bytes (cardinal::vm bytecode)
    //   Runner verifies + loads the module with cardinal::vm (capability
    //   sandbox), binds a host-fn allowlist that bridges log over OP_LOG,
    //   resolves the entry fn "tick" (1 param = dt as f64-bits). Replies
    //   READY (synthetic name/version) on success, ERROR on verify/load
    //   failure. TICK then drives the VM's tick(); a Trap surfaces as an
    //   inline ERROR (then ACK, like the DLL path) — a VM bug only ever
    //   kills the child, and the capability model contains the script
    //   regardless of process isolation.
    inline constexpr u32 OP_ATTACH_VM = 0x04;

    inline constexpr u32 OP_READY  = 0x80;
    inline constexpr u32 OP_ACK    = 0x81;
    inline constexpr u32 OP_LOG    = 0x82;
    inline constexpr u32 OP_ERROR  = 0x83;

    inline constexpr u32 LOG_TRACE = 0;
    inline constexpr u32 LOG_INFO  = 1;
    inline constexpr u32 LOG_WARN  = 2;
    inline constexpr u32 LOG_ERROR = 3;
}

enum class Mode : u32 {
    InProcess  = 0,
    Subprocess = 1,
};

struct Desc {
    Mode mode{Mode::InProcess};
    // Optional override path to cardinal_sandbox_runner.exe. When empty,
    // the host searches <host-exe-dir>/cardinal_sandbox_runner.exe first,
    // then PATH.
    std::string runner_override{};
    // Tick timeout — if a sandbox doesn't ACK within this many ms, the
    // host kills the runner with TerminateProcess and marks the sandbox
    // dead. Default 250 ms; raise for genuinely heavy per-tick scripts.
    u32 tick_timeout_ms{250};
    // Per-attach timeout (LoadLibrary + on_attach can be slow on cold
    // disks). Default 5000 ms.
    u32 attach_timeout_ms{5000};
};

// Status snapshot for diagnostics — the editor's plugin / sandbox panels
// poll this to display health.
struct Status {
    bool        alive{false};
    Mode        mode{Mode::InProcess};
    u64         pid{0};            // Subprocess: child OS pid; InProcess: 0
    u64         ticks{0};
    f64         last_tick_ms{0.0}; // wall-clock time of the most recent tick
    std::string plugin_name;       // populated after a successful ATTACH
    std::string plugin_version;
    std::string last_error;        // last LOG-error or IPC failure message
};

class Sandbox {
public:
    // Returns nullptr on attach failure (compile errors / DLL missing /
    // runner spawn failed / pipe broken before READY).
    static std::unique_ptr<Sandbox> create(const Desc& desc, const char* dll_path);

    // Capability-sandbox variant: instead of a native DLL, ship a
    // cardinal::vm bytecode module to the child, which verifies + runs it
    // in the VM. Strictly stronger isolation than the DLL path (no syscalls
    // / bounded memory+compute / host-call allowlist) AND still process-
    // isolated. Always uses the child process (mode is forced to
    // Subprocess); returns nullptr if that's unavailable on this OS or the
    // module fails to verify/attach. `module_bytes` is copied.
    static std::unique_ptr<Sandbox> create_vm(const Desc& desc,
                                              const u8* module_bytes,
                                              usize len);

    Sandbox()                          = default;
    virtual ~Sandbox() = default;
    Sandbox(const Sandbox&)            = delete;
    Sandbox& operator=(const Sandbox&) = delete;

    // Forward a tick to the hosted plugin. Subprocess mode: blocks until
    // ACK or timeout. Returns false if the sandbox is dead (caller should
    // drop the pointer); true otherwise. Tick failures (SEH inside the
    // plugin's on_tick) bump status().last_error but keep the sandbox
    // alive — only outright runner crashes / pipe breaks return false.
    virtual bool tick(f32 dt) = 0;

    // Graceful shutdown — Subprocess: send DETACH, wait for clean runner
    // exit, force-kill after attach_timeout_ms. InProcess: unload via
    // plugin::Registry::unload.
    virtual void detach() = 0;

    virtual Status status() const noexcept = 0;
};

}  // namespace cardinal::sandbox
