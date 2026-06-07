#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/platform.hpp>

#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <vector>

#if CARDINAL_PLATFORM_WINDOWS && defined(_DEBUG)
    #include <cstdlib>     // _set_abort_behavior
    #include <crtdbg.h>    // _CrtSetReportMode / _CrtSetReportFile
    #include <Windows.h>   // SetErrorMode + SEM_*
#endif

namespace cardinal::log {

namespace {

#if CARDINAL_PLATFORM_WINDOWS && defined(_DEBUG)
// Static-init abort()/assert() dialog silencer. Lives inside log.cpp (an
// always-linked TU — every cardinal::log consumer drags it in) so the
// static library linker can't drop it. Without this MSVC pops TWO modal
// dialogs on assert(false)/abort():
//   1. "Debug Assertion Failed" from _wassert() (the assert macro impl).
//   2. "Debug Error! abort() has been called" from abort() afterwards.
// Both are unwelcome for headless tests + CI. Redirect the CRT report
// streams to stderr + clear the abort() message-box bits + the Win32
// SEM_FAILCRITICALERRORS / SEM_NOGPFAULTERRORBOX flags.
struct AbortSilencer {
    AbortSilencer() noexcept {
        ::_set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
        const int kStreams[] = {_CRT_ASSERT, _CRT_ERROR, _CRT_WARN};
        for (int s : kStreams) {
            ::_CrtSetReportMode(s, _CRTDBG_MODE_FILE);
            ::_CrtSetReportFile(s, _CRTDBG_FILE_STDERR);
        }
        ::SetErrorMode(::GetErrorMode()
                       | SEM_FAILCRITICALERRORS
                       | SEM_NOGPFAULTERRORBOX);
    }
};
const AbortSilencer g_abort_silencer{};
#endif

// Built-in stderr sink — installed on first use. The default behaviour
// matches the engine's old fprintf-everywhere style so dropping this in
// preserves existing console output.
class StderrSink final : public Sink {
public:
    void on_message(Level lvl, const char* cat, const char* msg) override {
        std::fprintf(stderr, "[%s] [%s] %s\n",
                     level_name(lvl), cat ? cat : "", msg ? msg : "");
        std::fflush(stderr);
    }
};

struct GlobalLog {
    std::mutex          mutex;
    std::vector<Sink*>  sinks;
    Level               min_level{Level::Trace};
    StderrSink          stderr_sink;
    bool                stderr_installed{false};
};

GlobalLog& g() {
    static GlobalLog s;
    return s;
}

void ensure_default_sink_locked(GlobalLog& gl) {
    if (gl.stderr_installed) return;
    gl.sinks.push_back(&gl.stderr_sink);
    gl.stderr_installed = true;
}

}  // namespace

const char* level_name(Level l) noexcept {
    switch (l) {
        case Level::Trace: return "trace";
        case Level::Info:  return "info";
        case Level::Warn:  return "warn";
        case Level::Error: return "error";
    }
    return "?";
}

void add_sink(Sink* sink) {
    if (sink == nullptr) return;
    auto& gl = g();
    std::lock_guard lk(gl.mutex);
    ensure_default_sink_locked(gl);
    for (auto* s : gl.sinks) if (s == sink) return;
    gl.sinks.push_back(sink);
}

void remove_sink(Sink* sink) {
    if (sink == nullptr) return;
    auto& gl = g();
    std::lock_guard lk(gl.mutex);
    for (auto it = gl.sinks.begin(); it != gl.sinks.end(); ++it) {
        if (*it == sink) { gl.sinks.erase(it); return; }
    }
}

void set_min_level(Level l) {
    auto& gl = g();
    std::lock_guard lk(gl.mutex);
    gl.min_level = l;
}

void emitf(Level lvl, const char* category, const char* fmt, ...) {
    va_list a; va_start(a, fmt);
    vemitf(lvl, category, fmt, a);
    va_end(a);
}

void vemitf(Level lvl, const char* category, const char* fmt, va_list args) {
    auto& gl = g();
    {
        std::lock_guard lk(gl.mutex);
        if (static_cast<u32>(lvl) < static_cast<u32>(gl.min_level)) return;
        ensure_default_sink_locked(gl);
    }

    // Stack buffer first; spill to heap if it doesn't fit. Two passes are
    // fine because the second formats into the right-sized buffer with the
    // copied va_list — this matches std::vsnprintf semantics.
    char     stackbuf[1024];
    va_list  args_copy;
    va_copy (args_copy, args);
    int needed = std::vsnprintf(stackbuf, sizeof(stackbuf), fmt, args_copy);
    va_end(args_copy);

    const char* out = stackbuf;
    std::vector<char> heapbuf;
    if (needed >= static_cast<int>(sizeof(stackbuf))) {
        heapbuf.resize(static_cast<size_t>(needed) + 1);
        va_copy(args_copy, args);
        std::vsnprintf(heapbuf.data(), heapbuf.size(), fmt, args_copy);
        va_end(args_copy);
        out = heapbuf.data();
    } else if (needed < 0) {
        out = "<log format error>";
    }

    // Snapshot sinks under lock, then dispatch outside the lock so a sink
    // calling back into the log API doesn't deadlock.
    std::vector<Sink*> snapshot;
    {
        std::lock_guard lk(gl.mutex);
        snapshot = gl.sinks;
    }
    for (auto* s : snapshot) s->on_message(lvl, category, out);
}

}  // namespace cardinal::log
