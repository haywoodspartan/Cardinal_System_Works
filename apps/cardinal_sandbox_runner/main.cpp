// =============================================================================
// Cardinal — sandbox runner (Subprocess execution host for plugin DLLs).
//
// Standalone exe spawned by cardinal::sandbox when a plugin asks for
// process-isolated execution. All real work happens here:
//   1. argv:  cardinal_sandbox_runner.exe <pipe_name> <dll_path>
//   2. Open the named pipe (CreateFile in CLIENT mode).
//   3. Send READY (0x80) with our own pid + a sentinel — actually wait.
//      We don't send READY until ATTACH succeeds; the host blocks on it.
//   4. Loop: read a frame.
//        ATTACH → LoadLibrary the DLL, find cardinal_plugin_register,
//                 call it, then call on_attach inside SEH __try.
//                 On success: send READY with name + version.
//                 On failure: send ERROR + exit rc=1.
//        TICK   → call on_tick(dt) inside SEH __try. Always reply ACK
//                 (the host treats ERROR as "tick failed but runner is
//                 still alive"; an actual runner crash manifests as
//                 broken pipe / process exit).
//        DETACH → call on_detach + ExitProcess(0).
//
// Crash isolation:
//   - SEH __try / __except wraps every plugin call. On catch we send
//     ERROR + GetExceptionCode then continue (caller decides whether
//     to keep ticking).
//   - Process death (TerminateProcess from host on tick timeout, or any
//     CRT abort the SEH didn't catch) shows up host-side as a broken
//     pipe + WaitForSingleObject(process_handle) returning WAIT_OBJECT_0.
//
// Note on the host_api: the plugin we host calls log_info / log_warn /
// log_error / register_render_algo through pointers we pass at on_attach.
// log_*  forwards over the pipe via OP_LOG; register_render_algo is a
// no-op + logged warning in subprocess mode (CPU function pointers can't
// cross address spaces — see sandbox.hpp's protocol comment).
// =============================================================================

#include <cardinal/plugin/plugin.hpp>
#include <cardinal/sandbox/sandbox.hpp>
#include <cardinal/vm/vm.hpp>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

namespace proto = cardinal::sandbox::proto;
using cardinal::u32;
using cardinal::u64;
using cardinal::i64;
using cardinal::f32;

// ============================================================================
// Pipe IO helpers
// ============================================================================
namespace {

HANDLE g_pipe = INVALID_HANDLE_VALUE;
const CardinalPluginHostApi* g_owned_host_api_storage = nullptr;  // unused; we own a static copy

bool pipe_write_all(const void* p, u32 n) {
    const char* b = static_cast<const char*>(p);
    DWORD written = 0;
    while (n > 0) {
        if (!WriteFile(g_pipe, b, n, &written, nullptr) || written == 0) return false;
        b += written;
        n -= written;
    }
    return true;
}

bool pipe_read_all(void* p, u32 n) {
    char* b = static_cast<char*>(p);
    DWORD got = 0;
    while (n > 0) {
        if (!ReadFile(g_pipe, b, n, &got, nullptr) || got == 0) return false;
        b += got;
        n -= got;
    }
    return true;
}

bool send_frame(u32 opcode, const void* payload, u32 payload_len) {
    u32 hdr[2] = { opcode, payload_len };
    if (!pipe_write_all(hdr, sizeof(hdr))) return false;
    if (payload_len > 0 && !pipe_write_all(payload, payload_len)) return false;
    return true;
}

// Encode + send a length-prefixed string field. Used by READY / LOG / ERROR
// payloads.
void put_u32(std::vector<char>& buf, u32 v) {
    buf.insert(buf.end(),
               reinterpret_cast<const char*>(&v),
               reinterpret_cast<const char*>(&v) + 4);
}
void put_str(std::vector<char>& buf, const char* s) {
    const u32 n = s ? static_cast<u32>(std::strlen(s)) : 0u;
    put_u32(buf, n);
    if (n > 0) buf.insert(buf.end(), s, s + n);
}

void send_log(u32 level, const char* cat, const char* msg) {
    std::vector<char> p;
    put_u32(p, level);
    put_str(p, cat ? cat : "sandbox");
    put_str(p, msg ? msg : "");
    send_frame(proto::OP_LOG, p.data(), static_cast<u32>(p.size()));
}

void send_error(const char* msg) {
    std::vector<char> p;
    put_str(p, msg);
    send_frame(proto::OP_ERROR, p.data(), static_cast<u32>(p.size()));
}

// ============================================================================
// Host API exposed to the plugin (forwards everything over the pipe)
// ============================================================================
void host_log_info (const char* cat, const char* msg) { send_log(proto::LOG_INFO,  cat, msg); }
void host_log_warn (const char* cat, const char* msg) { send_log(proto::LOG_WARN,  cat, msg); }
void host_log_error(const char* cat, const char* msg) { send_log(proto::LOG_ERROR, cat, msg); }

bool host_register_render_algo(u32 /*category_id*/,
                               const char* id,
                               const char* /*label*/,
                               const char* /*tooltip*/,
                               const char* /*hlsl_function*/,
                               CardinalAlgoCpuFn /*cpu_fn*/)
{
    // Render algos can't cross process boundaries: the cpu_fn pointer is
    // valid only in our address space, and the host's renderer can't reach
    // into us per-frame without a much heavier IPC contract. Log + reject;
    // plugins that need to register algos should run InProcess.
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "register_render_algo('%s'): subprocess-mode plugins can't register "
        "render algos (CPU fn pointer crosses process boundary). Move to "
        "InProcess mode for this script.",
        id ? id : "?");
    send_log(proto::LOG_WARN, "sandbox", buf);
    return false;
}

CardinalPluginHostApi make_host_api() {
    CardinalPluginHostApi a{};
    a.api_version          = CARDINAL_PLUGIN_API_VERSION;
    a.log_info             = &host_log_info;
    a.log_warn             = &host_log_warn;
    a.log_error            = &host_log_error;
    a.register_render_algo = &host_register_render_algo;
    return a;
}

// ============================================================================
// SEH wrappers — must be in functions with no C++ destructors so the
// compiler accepts mixing __try / __except with anything stack-cleanup-y.
// ============================================================================
DWORD invoke_register_seh(CardinalPluginRegisterFn fn,
                          CardinalPluginInfo* info_out) noexcept
{
    DWORD code = 0;
    __try { fn(info_out); return 0; }
    __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) { return code; }
}

DWORD invoke_attach_seh(void (*fn)(const CardinalPluginHostApi*),
                        const CardinalPluginHostApi* api) noexcept
{
    DWORD code = 0;
    __try { fn(api); return 0; }
    __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) { return code; }
}

DWORD invoke_tick_seh(void (*fn)(float), float dt) noexcept {
    DWORD code = 0;
    __try { fn(dt); return 0; }
    __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) { return code; }
}

DWORD invoke_detach_seh(void (*fn)()) noexcept {
    DWORD code = 0;
    __try { fn(); return 0; }
    __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) { return code; }
}

}  // namespace

// ============================================================================
// Loaded plugin state
// ============================================================================
namespace {

struct Loaded {
    HMODULE                module{nullptr};
    CardinalPluginInfo     info{};
    CardinalPluginHostApi  host_api{};
};

Loaded g_loaded;

bool handle_attach(const std::string& dll_path) {
    g_loaded.module = LoadLibraryA(dll_path.c_str());
    if (g_loaded.module == nullptr) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "LoadLibrary failed (%lu): %s",
                      GetLastError(), dll_path.c_str());
        send_error(buf);
        return false;
    }
    auto reg_fn = reinterpret_cast<CardinalPluginRegisterFn>(
        GetProcAddress(g_loaded.module, "cardinal_plugin_register"));
    if (reg_fn == nullptr) {
        send_error("missing cardinal_plugin_register export");
        return false;
    }

    g_loaded.info.api_version = CARDINAL_PLUGIN_API_VERSION;
    DWORD seh = invoke_register_seh(reg_fn, &g_loaded.info);
    if (seh != 0) {
        char buf[64];
        std::snprintf(buf, sizeof(buf),
            "cardinal_plugin_register raised SEH 0x%08lx", seh);
        send_error(buf);
        return false;
    }
    if (g_loaded.info.api_version != CARDINAL_PLUGIN_API_VERSION) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "plugin reports API v%u, runner is v%u",
            g_loaded.info.api_version, CARDINAL_PLUGIN_API_VERSION);
        send_error(buf);
        return false;
    }
    g_loaded.host_api = make_host_api();
    if (g_loaded.info.on_attach) {
        seh = invoke_attach_seh(g_loaded.info.on_attach, &g_loaded.host_api);
        if (seh != 0) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "on_attach raised SEH 0x%08lx", seh);
            send_error(buf);
            return false;
        }
    }

    // READY payload: name + version (length-prefixed strings).
    std::vector<char> p;
    put_str(p, g_loaded.info.name    ? g_loaded.info.name    : "");
    put_str(p, g_loaded.info.version ? g_loaded.info.version : "");
    send_frame(proto::OP_READY, p.data(), static_cast<u32>(p.size()));
    return true;
}

// ============================================================================
// Capability-sandbox path: a cardinal::vm bytecode module instead of a DLL.
// The VM is the security boundary (no syscalls / bounded memory + compute /
// host-call allowlist); the child process is defence in depth.
// ============================================================================
namespace vm = cardinal::vm;

struct LoadedVm {
    vm::ModulePtr            module;
    std::unique_ptr<vm::VM>  vmi;
    cardinal::i64            tick_idx{-1};
    bool                     attached{false};
};
LoadedVm g_vm;

// The ENTIRE outside world available to sandboxed bytecode: one allowlisted
// host fn that logs an i64 (forwarded over the pipe as a normal LOG frame).
cardinal::i64 vm_host_log(vm::HostContext&, const cardinal::i64* a,
                          cardinal::u32 n) noexcept {
    char buf[64];
    const long long v = (n >= 1u) ? static_cast<long long>(a[0]) : 0LL;
    std::snprintf(buf, sizeof(buf), "%lld", v);
    send_log(proto::LOG_INFO, "cvm", buf);
    return 0;
}
const vm::HostFnDesc g_vm_host[] = {
    { "log", &vm_host_log, 1u },
};

bool handle_attach_vm(const std::vector<char>& payload) {
    if (payload.size() < 4u) { send_error("ATTACH_VM: short payload"); return false; }
    cardinal::u32 mlen = 0;
    std::memcpy(&mlen, payload.data(), 4);
    if (static_cast<cardinal::usize>(mlen) + 4u > payload.size()) {
        send_error("ATTACH_VM: truncated module"); return false;
    }
    const cardinal::u8* mbytes =
        reinterpret_cast<const cardinal::u8*>(payload.data()) + 4;

    vm::Limits lim{};
    std::string err;
    g_vm.module = vm::load(mbytes, mlen, lim, &err);
    if (!g_vm.module) {
        std::string m = "VM verify/load failed: " + err;
        send_error(m.c_str());
        return false;
    }
    const cardinal::i64 ti = vm::module_find_func(*g_vm.module, "tick");
    if (ti < 0) { send_error("VM module has no 'tick' export"); return false; }
    g_vm.tick_idx = ti;
    g_vm.vmi = vm::VM::create(lim);
    g_vm.vmi->set_host_fns(g_vm_host,
        static_cast<cardinal::u32>(sizeof(g_vm_host) / sizeof(g_vm_host[0])));
    g_vm.attached = true;

    std::vector<char> p;
    put_str(p, "cvm-module");
    put_str(p, "1");
    send_frame(proto::OP_READY, p.data(), static_cast<u32>(p.size()));
    return true;
}

void handle_tick_vm(f32 dt) {
    // dt reaches the script as the f64 bit-pattern in local 0 (the VM has
    // F* ops). Result cell is ignored.
    const cardinal::f64 d = static_cast<cardinal::f64>(dt);
    cardinal::u64 bits = 0;
    std::memcpy(&bits, &d, 8);
    cardinal::i64 arg = static_cast<cardinal::i64>(bits);
    cardinal::i64 ret = 0;
    const vm::Trap t = g_vm.vmi->call(*g_vm.module,
        static_cast<cardinal::u32>(g_vm.tick_idx), &arg, 1u, &ret);
    if (t != vm::Trap::Finished && t != vm::Trap::Halted) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "VM trap: %s", vm::trap_name(t));
        send_error(buf);   // non-fatal: ACK still follows (like the DLL path)
    }
    send_frame(proto::OP_ACK, nullptr, 0);
}

void handle_tick(f32 dt) {
    if (g_vm.attached) { handle_tick_vm(dt); return; }
    if (g_loaded.info.on_tick == nullptr) {
        send_frame(proto::OP_ACK, nullptr, 0);
        return;
    }
    DWORD seh = invoke_tick_seh(g_loaded.info.on_tick, dt);
    if (seh != 0) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "on_tick raised SEH 0x%08lx", seh);
        send_error(buf);
        // Still ACK so the host's tick loop doesn't time out — the SEH
        // surfaces via the inline ERROR frame that arrived just above.
        send_frame(proto::OP_ACK, nullptr, 0);
        return;
    }
    send_frame(proto::OP_ACK, nullptr, 0);
}

void handle_detach_and_exit() {
    if (g_loaded.info.on_detach) {
        (void)invoke_detach_seh(g_loaded.info.on_detach);
    }
    if (g_loaded.module) FreeLibrary(g_loaded.module);
    // Don't FlushFileBuffers — the host already issued the DETACH and will
    // close its pipe end, so a flush here can deadlock against a host that
    // moved on to WaitForSingleObject(process). Just close + exit.
    if (g_pipe != INVALID_HANDLE_VALUE) {
        CloseHandle(g_pipe);
    }
    ExitProcess(0);
}

}  // namespace

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: cardinal_sandbox_runner <pipe_name> <dll_path>\n");
        return 2;
    }
    const std::string pipe_name = argv[1];
    const std::string dll_path  = argv[2];

    // Connect to the host's pipe. The host CreateNamedPipe'd it before
    // spawning us; CreateFile on the same name connects + the host's
    // ConnectNamedPipe returns. Retry a few times if the pipe isn't ready
    // yet (race between CreateProcess return and CreateNamedPipe in the
    // host — ConnectNamedPipe should have been called before spawn, but
    // belt + braces with WaitNamedPipe).
    for (int attempt = 0; attempt < 50; ++attempt) {
        g_pipe = CreateFileA(pipe_name.c_str(),
            GENERIC_READ | GENERIC_WRITE, 0, nullptr,
            OPEN_EXISTING, 0, nullptr);
        if (g_pipe != INVALID_HANDLE_VALUE) break;
        if (GetLastError() != ERROR_PIPE_BUSY && attempt > 0) {
            // Pipe doesn't exist yet — wait briefly and retry.
            Sleep(10);
        } else {
            WaitNamedPipeA(pipe_name.c_str(), 200);
        }
    }
    if (g_pipe == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr,
            "runner: failed to connect to pipe '%s' (err=%lu)\n",
            pipe_name.c_str(), GetLastError());
        return 3;
    }
    // Disable Nagle / message-mode buffering tradeoff — we want byte stream
    // semantics (we framed it ourselves).
    DWORD mode = PIPE_READMODE_BYTE | PIPE_WAIT;
    SetNamedPipeHandleState(g_pipe, &mode, nullptr, nullptr);

    // Frame loop. Single-threaded — the host serializes its requests.
    bool attached = false;
    for (;;) {
        u32 hdr[2] = {0, 0};
        if (!pipe_read_all(hdr, sizeof(hdr))) {
            // Pipe broken — host probably went away. Cleanup + exit.
            if (g_loaded.info.on_detach) (void)invoke_detach_seh(g_loaded.info.on_detach);
            if (g_loaded.module) FreeLibrary(g_loaded.module);
            return 4;
        }
        const u32 opcode      = hdr[0];
        const u32 payload_len = hdr[1];

        std::vector<char> payload(payload_len);
        if (payload_len > 0 && !pipe_read_all(payload.data(), payload_len)) {
            return 5;
        }

        switch (opcode) {
            case proto::OP_ATTACH: {
                if (attached) { send_error("duplicate ATTACH"); break; }
                // Payload format: u32 path_len + bytes path. We could also
                // accept the dll_path from argv (which we already have);
                // the explicit ATTACH payload lets the host re-attach to
                // a different DLL on this same runner if that's ever needed.
                std::string path = dll_path;
                if (payload_len >= 4) {
                    u32 plen = 0;
                    std::memcpy(&plen, payload.data(), 4);
                    if (plen + 4u <= payload_len) {
                        path.assign(payload.data() + 4, plen);
                    }
                }
                attached = handle_attach(path);
                if (!attached) {
                    // ATTACH failed — host got our ERROR; bail with rc=1
                    // so the host's WaitForSingleObject(process) wakes up.
                    return 1;
                }
            } break;

            case proto::OP_ATTACH_VM: {
                if (attached) { send_error("duplicate ATTACH"); break; }
                attached = handle_attach_vm(payload);
                if (!attached) return 1;   // host's WaitForSingleObject wakes
            } break;

            case proto::OP_TICK: {
                f32 dt = 0.0f;
                if (payload_len >= sizeof(f32)) std::memcpy(&dt, payload.data(), sizeof(f32));
                handle_tick(dt);
            } break;

            case proto::OP_DETACH:
                handle_detach_and_exit();   // never returns
                break;

            default: {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "unknown opcode 0x%X", opcode);
                send_error(buf);
            } break;
        }
    }
}
