// =============================================================================
// Cardinal — sandbox implementation.
//
// InProcess mode: delegates to cardinal::plugin::Registry (SEH-tier).
// Subprocess mode: spawns cardinal_sandbox_runner.exe per sandbox + talks
//                  to it over a per-sandbox named pipe. Real process
//                  isolation — runner crashes don't take the host down.
//
// Wire protocol + opcode docs live in sandbox.hpp.
// =============================================================================

#include <cardinal/sandbox/sandbox.hpp>

#include <cardinal/core/diag/log.hpp>
#include <cardinal/plugin/plugin.hpp>

#include <cardinal/core/std/atomic.hpp>
#include <cardinal/core/std/chrono.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/core/std/cstdio.hpp>
#include <cardinal/core/std/cstring.hpp>
#include <cardinal/core/std/filesystem.hpp>
#include <cardinal/core/std/utility.hpp>

#if defined(_WIN32)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <Windows.h>
#endif

namespace cardinal::sandbox {

namespace fs = cardinal::fs;

// ============================================================================
// InProcessSandbox — delegates to plugin::Registry, no IPC.
// ============================================================================
namespace {

class InProcessSandbox final : public Sandbox {
public:
    InProcessSandbox(const Desc& d, cardinal::string dll) : desc_(d), dll_(cardinal::move(dll)) {}

    bool initialize() {
        return cardinal::plugin::Registry::instance().load(dll_.c_str());
    }

    bool tick(f32 /*dt*/) override {
        // The host's main loop already calls plugin::Registry::tick() once
        // per frame; we don't double-tick here. Just count + return.
        ++ticks_;
        return true;
    }

    void detach() override {
        cardinal::plugin::Registry::instance().unload(dll_.c_str());
    }

    Status status() const noexcept override {
        Status s;
        s.alive          = true;
        s.mode           = Mode::InProcess;
        s.ticks          = ticks_.load();
        s.last_tick_ms   = 0.0;
        return s;
    }

private:
    Desc        desc_;
    cardinal::string dll_;
    cardinal::atomic<u64> ticks_{0};
};

#if defined(_WIN32)

// ============================================================================
// SubprocessSandbox — Windows implementation.
//
// Pipe lifecycle:
//   1. CreateNamedPipeA before CreateProcess → server end is ours.
//   2. CreateProcessA the runner with pipe name + dll path on argv.
//   3. ConnectNamedPipe blocks (with timeout) until the runner CreateFile's it.
//   4. Send ATTACH; wait for READY (or process death).
//   5. Per tick: send TICK + dt; loop reading frames, draining LOG, until
//      ACK or timeout or process death.
//   6. detach(): send DETACH, wait for clean process exit, force-kill on
//      timeout.
// ============================================================================
class SubprocessSandbox final : public Sandbox {
public:
    SubprocessSandbox(const Desc& d, cardinal::string dll) : desc_(d), dll_(cardinal::move(dll)) {}
    SubprocessSandbox(const Desc& d, cardinal::vector<u8> mod)
        : desc_(d), vm_mode_(true), module_bytes_(cardinal::move(mod)) {}

    ~SubprocessSandbox() override { teardown(); }

    bool initialize() {
        // Locate the runner exe.
        runner_path_ = resolve_runner_path_();
        if (runner_path_.empty()) {
            cardinal::log::errorf("sandbox",
                "cardinal_sandbox_runner.exe not found (looked in <host-exe-dir>, PATH, runner_override)");
            return false;
        }

        // Unique pipe name per sandbox. Includes pid + atomic counter so
        // multiple sandboxes in one editor process don't collide.
        static cardinal::atomic<u32> s_idx{0};
        const u32 idx = ++s_idx;
        char namebuf[128];
        cardinal::snprintf(namebuf, sizeof(namebuf),
            "\\\\.\\pipe\\cardinal_sandbox_%lu_%u",
            GetCurrentProcessId(), idx);
        pipe_name_ = namebuf;

        // Server side: PIPE_ACCESS_DUPLEX (we read + write), TYPE_BYTE
        // (we framed it ourselves). Bound buffer sizes are heuristics —
        // 64 KB each direction should swallow a worst-case LOG burst.
        pipe_ = CreateNamedPipeA(pipe_name_.c_str(),
            PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
            1, 64 * 1024, 64 * 1024,
            0, nullptr);
        if (pipe_ == INVALID_HANDLE_VALUE) {
            cardinal::log::errorf("sandbox", "CreateNamedPipe failed (err=%lu)", GetLastError());
            return false;
        }

        // Spawn the runner. Pass pipe name + dll path on the command line.
        cardinal::string cmdline;
        cmdline.reserve(512);
        cmdline += "\"";  cmdline += runner_path_; cmdline += "\"";
        cmdline += " \""; cmdline += pipe_name_;   cmdline += "\"";
        cmdline += " \""; cmdline += dll_;         cmdline += "\"";

        STARTUPINFOA si{};
        si.cb = sizeof(si);
        // CREATE_NO_WINDOW so we don't get a console flash per sandbox.
        // We could DETACHED_PROCESS but no_window is friendlier (the
        // runner can still write to stderr if it needs to).
        PROCESS_INFORMATION pi{};
        BOOL ok = CreateProcessA(nullptr, cmdline.data(), nullptr, nullptr,
            FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
        if (!ok) {
            cardinal::log::errorf("sandbox",
                "CreateProcess(runner) failed (err=%lu): %s",
                GetLastError(), cmdline.c_str());
            CloseHandle(pipe_); pipe_ = INVALID_HANDLE_VALUE;
            return false;
        }
        process_ = pi.hProcess;
        pid_     = pi.dwProcessId;
        CloseHandle(pi.hThread);

        // Accept the runner's connection — overlapped + manual-reset event
        // so we can wait on (event, process) and time out.
        OVERLAPPED ov{};
        HANDLE evt = CreateEventA(nullptr, TRUE, FALSE, nullptr);
        ov.hEvent = evt;
        BOOL conn = ConnectNamedPipe(pipe_, &ov);
        DWORD err = conn ? 0 : GetLastError();
        if (!conn && err == ERROR_IO_PENDING) {
            HANDLE waits[2] = { evt, process_ };
            DWORD wr = WaitForMultipleObjects(2, waits, FALSE,
                                              desc_.attach_timeout_ms);
            if (wr == WAIT_OBJECT_0) {
                conn = TRUE;
            } else {
                CancelIoEx(pipe_, &ov);
                cardinal::log::errorf("sandbox",
                    "ConnectNamedPipe %s (wait=%lu, runner_pid=%lu)",
                    wr == WAIT_OBJECT_0 + 1 ? "runner died before connect" : "timed out",
                    wr, pid_);
                CloseHandle(evt);
                teardown();
                return false;
            }
        } else if (!conn && err != ERROR_PIPE_CONNECTED) {
            cardinal::log::errorf("sandbox",
                "ConnectNamedPipe failed (err=%lu)", err);
            CloseHandle(evt);
            teardown();
            return false;
        }
        CloseHandle(evt);

        // VM mode: ship the bytecode module (u32 len + bytes) and let the
        // runner verify + host it in cardinal::vm. DLL mode: ship the dll
        // path (the runner also has it from argv; belt + braces).
        cardinal::vector<char> p;
        u32 attach_op;
        if (vm_mode_) {
            const u32 mlen = static_cast<u32>(module_bytes_.size());
            p.insert(p.end(), reinterpret_cast<const char*>(&mlen),
                              reinterpret_cast<const char*>(&mlen) + 4);
            p.insert(p.end(),
                     reinterpret_cast<const char*>(module_bytes_.data()),
                     reinterpret_cast<const char*>(module_bytes_.data())
                         + module_bytes_.size());
            attach_op = proto::OP_ATTACH_VM;
        } else {
            const u32 plen = static_cast<u32>(dll_.size());
            p.insert(p.end(), reinterpret_cast<const char*>(&plen),
                              reinterpret_cast<const char*>(&plen) + 4);
            p.insert(p.end(), dll_.begin(), dll_.end());
            attach_op = proto::OP_ATTACH;
        }
        if (!send_frame_(attach_op, p.data(), static_cast<u32>(p.size()))) {
            cardinal::log::errorf("sandbox", "ATTACH send failed");
            teardown();
            return false;
        }

        // Drain frames until READY or runner death. Inline LOG / ERROR are
        // forwarded to the host's log so the user sees plugin output.
        u32  op = 0; cardinal::vector<char> payload;
        while (true) {
            if (!read_frame_with_timeout_(op, payload, desc_.attach_timeout_ms)) {
                last_error_ = "ATTACH timed out / runner died";
                teardown();
                return false;
            }
            if (op == proto::OP_READY) {
                parse_ready_(payload);
                cardinal::log::infof("sandbox",
                    "subprocess plugin '%s' v%s attached (pid=%lu)",
                    plugin_name_.c_str(), plugin_version_.c_str(), pid_);
                return true;
            }
            if (op == proto::OP_LOG)   { forward_log_(payload); continue; }
            if (op == proto::OP_ERROR) {
                last_error_ = parse_string_(payload);
                cardinal::log::errorf("sandbox", "ATTACH error: %s", last_error_.c_str());
                teardown();
                return false;
            }
            // Unknown opcode during attach — protocol mismatch. Bail.
            cardinal::log::errorf("sandbox",
                "unexpected opcode 0x%X during ATTACH", op);
            teardown();
            return false;
        }
    }

    bool tick(f32 dt) override {
        if (!alive_) return false;
        const auto t0 = cardinal::chrono::steady_clock::now();
        if (!send_frame_(proto::OP_TICK, &dt, sizeof(dt))) {
            last_error_ = "TICK send failed (broken pipe)";
            mark_dead_();
            return false;
        }
        u32 op = 0; cardinal::vector<char> payload;
        while (true) {
            if (!read_frame_with_timeout_(op, payload, desc_.tick_timeout_ms)) {
                last_error_ = "TICK timeout (no ACK within budget)";
                cardinal::log::warnf("sandbox",
                    "tick timeout for plugin '%s' — terminating runner",
                    plugin_name_.c_str());
                kill_runner_();
                return false;
            }
            if (op == proto::OP_ACK) {
                ++ticks_;
                last_tick_ms_ = cardinal::chrono::duration<f64, cardinal::milli>(
                    cardinal::chrono::steady_clock::now() - t0).count();
                return true;
            }
            if (op == proto::OP_LOG)   { forward_log_(payload); continue; }
            if (op == proto::OP_ERROR) {
                last_error_ = parse_string_(payload);
                cardinal::log::errorf("sandbox",
                    "tick error in '%s': %s", plugin_name_.c_str(), last_error_.c_str());
                continue;   // ERROR is non-fatal; ACK should still arrive
            }
            cardinal::log::errorf("sandbox",
                "unexpected opcode 0x%X during TICK", op);
            mark_dead_();
            return false;
        }
    }

    void detach() override {
        if (!alive_) { teardown(); return; }
        // Best-effort DETACH. Whether or not the runner cooperates, we
        // ultimately wait for the process to exit (or kill it).
        send_frame_(proto::OP_DETACH, nullptr, 0);
        if (process_ != nullptr) {
            DWORD w = WaitForSingleObject(process_, desc_.attach_timeout_ms);
            if (w != WAIT_OBJECT_0) {
                cardinal::log::warnf("sandbox",
                    "runner '%s' didn't exit after DETACH — killing",
                    plugin_name_.c_str());
                TerminateProcess(process_, 1);
                WaitForSingleObject(process_, 1000);
            }
        }
        teardown();
    }

    Status status() const noexcept override {
        Status s;
        s.alive          = alive_;
        s.mode           = Mode::Subprocess;
        s.pid            = pid_;
        s.ticks          = ticks_;
        s.last_tick_ms   = last_tick_ms_;
        s.plugin_name    = plugin_name_;
        s.plugin_version = plugin_version_;
        s.last_error     = last_error_;
        return s;
    }

private:
    // -------- helpers ----------

    static cardinal::string current_exe_dir_() {
        char buf[MAX_PATH] = {0};
        GetModuleFileNameA(nullptr, buf, MAX_PATH);
        return fs::path(buf).parent_path().string();
    }

    cardinal::string resolve_runner_path_() const {
        // 1. Explicit override.
        if (!desc_.runner_override.empty() && fs::exists(desc_.runner_override))
            return desc_.runner_override;
        // 2. Sibling of the host exe — what cmake produces.
        fs::path sibling = fs::path(current_exe_dir_()) / "cardinal_sandbox_runner.exe";
        if (fs::exists(sibling)) return sibling.string();
        // 3. PATH lookup.
        char buf[MAX_PATH] = {0};
        if (SearchPathA(nullptr, "cardinal_sandbox_runner.exe", nullptr,
                        MAX_PATH, buf, nullptr) > 0) {
            return cardinal::string(buf);
        }
        return {};
    }

    bool send_frame_(u32 opcode, const void* payload, u32 payload_len) {
        u32 hdr[2] = { opcode, payload_len };
        if (!write_all_(hdr, sizeof(hdr))) return false;
        if (payload_len > 0 && !write_all_(payload, payload_len)) return false;
        return true;
    }

    bool write_all_(const void* p, u32 n) {
        const char* b = static_cast<const char*>(p);
        DWORD written = 0;
        while (n > 0) {
            if (!WriteFile(pipe_, b, n, &written, nullptr) || written == 0) return false;
            b += written;
            n -= written;
        }
        return true;
    }

    // Blocking read with timeout against the pipe + the process handle.
    // Returns false if timed out OR runner died OR pipe broken.
    bool read_frame_with_timeout_(u32& opcode,
                                  cardinal::vector<char>& payload,
                                  u32 timeout_ms)
    {
        if (!read_with_timeout_(&opcode, sizeof(opcode), timeout_ms)) return false;
        u32 plen = 0;
        if (!read_with_timeout_(&plen, sizeof(plen), timeout_ms)) return false;
        // Sanity-cap untrusted payload size. plen comes off the pipe from
        // the runner process; the runner is OUR binary but the WHOLE
        // POINT of the subprocess sandbox is to contain a compromised /
        // crashed runner (a malicious plugin loaded by the runner could
        // corrupt its proto state and send arbitrary frames). Without
        // a cap, plen = 0xFFFFFFFF → payload.resize(4 GiB) → either
        // bad_alloc (uncaught, terminates the HOST) or a successful
        // 4 GiB allocation that pins the engine's working set for one
        // crashed sandbox. 64 MiB easily covers every legitimate
        // protocol payload — largest by far is OP_ATTACH_VM with the
        // module bytes, and even a generous bytecode module fits in
        // single-digit MB. Same hostile-data sanity-cap shape as
        // pack::Archive::open's entry_count cap (pack.cpp:237) and the
        // PNG IHDR dimension cap in cook (7f02c8b).
        constexpr u32 kMaxFrameBytes = 64u * 1024u * 1024u;
        if (plen > kMaxFrameBytes) {
            cardinal::log::errorf("sandbox",
                "runner sent oversized frame (op=0x%X, plen=%u) — rejecting",
                opcode, plen);
            return false;
        }
        payload.resize(plen);
        if (plen > 0 && !read_with_timeout_(payload.data(), plen, timeout_ms))
            return false;
        return true;
    }

    bool read_with_timeout_(void* p, u32 n, u32 timeout_ms) {
        char* b = static_cast<char*>(p);
        const auto deadline = cardinal::chrono::steady_clock::now()
                              + cardinal::chrono::milliseconds(timeout_ms);
        while (n > 0) {
            // Use overlapped read to make timeout work.
            OVERLAPPED ov{};
            HANDLE evt = CreateEventA(nullptr, TRUE, FALSE, nullptr);
            ov.hEvent = evt;
            DWORD got = 0;
            BOOL ok = ReadFile(pipe_, b, n, &got, &ov);
            DWORD err = ok ? 0 : GetLastError();
            if (!ok && err != ERROR_IO_PENDING) {
                CloseHandle(evt);
                return false;
            }
            // Wait on (read-completed, process-died, deadline).
            const auto now = cardinal::chrono::steady_clock::now();
            const auto remaining = cardinal::chrono::duration_cast<cardinal::chrono::milliseconds>(
                deadline - now).count();
            const DWORD wait_ms = remaining <= 0 ? 0u : static_cast<DWORD>(remaining);
            HANDLE waits[2] = { evt, process_ };
            const DWORD count = process_ ? 2u : 1u;
            DWORD wr = WaitForMultipleObjects(count, waits, FALSE, wait_ms);
            if (wr == WAIT_OBJECT_0) {
                if (!GetOverlappedResult(pipe_, &ov, &got, FALSE) || got == 0) {
                    CloseHandle(evt);
                    return false;
                }
                b += got;
                n -= got;
                CloseHandle(evt);
                continue;
            }
            // Timeout / process death — cancel + bail.
            CancelIoEx(pipe_, &ov);
            GetOverlappedResult(pipe_, &ov, &got, TRUE);
            CloseHandle(evt);
            return false;
        }
        return true;
    }

    static u32 read_u32_(const cardinal::vector<char>& buf, size_t& pos) {
        u32 v = 0;
        if (pos + 4 <= buf.size()) {
            cardinal::memcpy(&v, buf.data() + pos, 4);
            pos += 4;
        }
        return v;
    }

    static cardinal::string read_str_(const cardinal::vector<char>& buf, size_t& pos) {
        const u32 n = read_u32_(buf, pos);
        cardinal::string s;
        if (pos + n <= buf.size()) {
            s.assign(buf.data() + pos, n);
            pos += n;
        }
        return s;
    }

    void parse_ready_(const cardinal::vector<char>& payload) {
        size_t pos = 0;
        plugin_name_    = read_str_(payload, pos);
        plugin_version_ = read_str_(payload, pos);
    }

    static cardinal::string parse_string_(const cardinal::vector<char>& payload) {
        size_t pos = 0;
        return read_str_(payload, pos);
    }

    void forward_log_(const cardinal::vector<char>& payload) {
        size_t pos = 0;
        const u32 level = read_u32_(payload, pos);
        const cardinal::string cat = read_str_(payload, pos);
        const cardinal::string msg = read_str_(payload, pos);
        switch (level) {
            case proto::LOG_TRACE: cardinal::log::infof (cat.c_str(), "%s", msg.c_str()); break;
            case proto::LOG_INFO:  cardinal::log::infof (cat.c_str(), "%s", msg.c_str()); break;
            case proto::LOG_WARN:  cardinal::log::warnf (cat.c_str(), "%s", msg.c_str()); break;
            case proto::LOG_ERROR: cardinal::log::errorf(cat.c_str(), "%s", msg.c_str()); break;
            default:               cardinal::log::infof (cat.c_str(), "%s", msg.c_str()); break;
        }
    }

    void kill_runner_() {
        if (process_ != nullptr) {
            TerminateProcess(process_, 1);
            WaitForSingleObject(process_, 1000);
        }
        mark_dead_();
    }

    void mark_dead_() {
        alive_ = false;
    }

    void teardown() {
        if (pipe_ != INVALID_HANDLE_VALUE) {
            CloseHandle(pipe_);
            pipe_ = INVALID_HANDLE_VALUE;
        }
        if (process_ != nullptr) {
            // Ensure runner is gone — best-effort.
            DWORD wait = WaitForSingleObject(process_, 100);
            if (wait != WAIT_OBJECT_0) TerminateProcess(process_, 1);
            CloseHandle(process_);
            process_ = nullptr;
        }
        alive_ = false;
    }

    Desc            desc_;
    cardinal::string     dll_;
    bool            vm_mode_{false};
    cardinal::vector<u8> module_bytes_;
    cardinal::string     runner_path_;
    cardinal::string     pipe_name_;

    HANDLE      pipe_{INVALID_HANDLE_VALUE};
    HANDLE      process_{nullptr};
    u32         pid_{0};

    cardinal::atomic<bool> alive_{true};
    cardinal::string plugin_name_;
    cardinal::string plugin_version_;
    u64         ticks_{0};
    f64         last_tick_ms_{0.0};
    cardinal::string last_error_;
};

#endif  // _WIN32

}  // namespace

cardinal::unique_ptr<Sandbox> Sandbox::create(const Desc& desc, const char* dll_path) {
    if (dll_path == nullptr || *dll_path == '\0') return nullptr;
#if defined(_WIN32)
    if (desc.mode == Mode::Subprocess) {
        auto s = cardinal::make_unique<SubprocessSandbox>(desc, cardinal::string(dll_path));
        if (!s->initialize()) return nullptr;
        return s;
    }
#else
    if (desc.mode == Mode::Subprocess) {
        cardinal::log::warnf("sandbox",
            "Subprocess mode not yet implemented on this OS — falling back to InProcess");
    }
#endif
    auto s = cardinal::make_unique<InProcessSandbox>(desc, cardinal::string(dll_path));
    if (!s->initialize()) return nullptr;
    return s;
}

cardinal::unique_ptr<Sandbox> Sandbox::create_vm(const Desc& desc,
                                            const u8* module_bytes, usize len)
{
    if (module_bytes == nullptr || len == 0) return nullptr;
#if defined(_WIN32)
    // The capability sandbox is always process-isolated (that's the point).
    cardinal::vector<u8> mod(module_bytes, module_bytes + len);
    auto s = cardinal::make_unique<SubprocessSandbox>(desc, cardinal::move(mod));
    if (!s->initialize()) return nullptr;
    return s;
#else
    (void)desc;
    cardinal::log::errorf("sandbox",
        "create_vm: subprocess VM host not implemented on this OS");
    return nullptr;
#endif
}

}  // namespace cardinal::sandbox
