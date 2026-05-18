// =============================================================================
// Cardinal — stack capture / symbolisation.
// =============================================================================
#include <cardinal/trace/stack.hpp>
#include <cardinal/core/log.hpp>
#include <cardinal/core/platform.hpp>

#include <cardinal/core/cstdio.hpp>    // cardinal::snprintf
#include <cardinal/core/cstring.hpp>   // cardinal::strncpy/strncat/strlen
#include <cardinal/core/thread.hpp>    // cardinal::mutex/lock_guard

#if CARDINAL_PLATFORM_WINDOWS

#include <Windows.h>
#include <DbgHelp.h>
#pragma comment(lib, "DbgHelp.lib")

namespace cardinal::trace {

namespace {

cardinal::mutex& dbghelp_mutex() {
    // DbgHelp is single-threaded; serialise every call.
    static cardinal::mutex m;
    return m;
}

bool ensure_sym_initialised() {
    static bool init = []() {
        SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_DEFERRED_LOADS |
                      SYMOPT_UNDNAME    | SYMOPT_NO_PROMPTS);
        if (!SymInitialize(GetCurrentProcess(), nullptr, TRUE)) {
            cardinal::log::warnf("trace", "SymInitialize failed (err=%lu)", GetLastError());
            return false;
        }
        return true;
    }();
    return init;
}

void resolve_one(u64 addr, StackFrame& out) {
    out.address = addr;
    if (!ensure_sym_initialised()) return;

    HANDLE proc = GetCurrentProcess();
    cardinal::lock_guard lk(dbghelp_mutex());

    // Symbol name + offset.
    char buf[sizeof(SYMBOL_INFO) + 256]{};
    auto* sym = reinterpret_cast<SYMBOL_INFO*>(buf);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen   = 255;
    DWORD64 displacement = 0;
    if (SymFromAddr(proc, addr, &displacement, sym)) {
        out.symbol = sym->Name;
        if (displacement) {
            char off[32];
            cardinal::snprintf(off, sizeof(off), "+0x%llx",
                static_cast<unsigned long long>(displacement));
            out.symbol += off;
        }
    } else {
        char hex[32];
        cardinal::snprintf(hex, sizeof(hex), "0x%llx", static_cast<unsigned long long>(addr));
        out.symbol = hex;
    }

    // File + line.
    IMAGEHLP_LINE64 line{}; line.SizeOfStruct = sizeof(line);
    DWORD line_disp = 0;
    if (SymGetLineFromAddr64(proc, addr, &line_disp, &line)) {
        out.file = line.FileName ? line.FileName : "";
        out.line = line.LineNumber;
    }

    // Module name (basename only).
    IMAGEHLP_MODULE64 mod{}; mod.SizeOfStruct = sizeof(mod);
    if (SymGetModuleInfo64(proc, addr, &mod)) {
        const char* name = mod.LoadedImageName;
        const char* slash = nullptr;
        for (const char* p = name; *p; ++p) if (*p == '\\' || *p == '/') slash = p;
        out.module = slash ? (slash + 1) : name;
    }
}

}  // namespace

cardinal::vector<StackFrame> capture(u32 skip, u32 max_depth) {
    cardinal::vector<StackFrame> frames;
    if (max_depth == 0) return frames;
    if (max_depth > 256) max_depth = 256;

    void* addrs[256];
    USHORT n = CaptureStackBackTrace(skip + 1 /* skip self */, max_depth, addrs, nullptr);
    frames.resize(n);
    for (USHORT i = 0; i < n; ++i) {
        resolve_one(reinterpret_cast<u64>(addrs[i]), frames[i]);
    }
    return frames;
}

cardinal::vector<StackFrame> capture_thread(void* native_thread_handle, u32 max_depth) {
    cardinal::vector<StackFrame> frames;
    if (max_depth > 256) max_depth = 256;
    if (native_thread_handle == nullptr) return capture(0, max_depth);

    HANDLE thread = static_cast<HANDLE>(native_thread_handle);
    HANDLE proc   = GetCurrentProcess();

    if (SuspendThread(thread) == (DWORD)-1) {
        cardinal::log::warnf("trace", "SuspendThread failed (err=%lu)", GetLastError());
        return frames;
    }

    CONTEXT ctx{}; ctx.ContextFlags = CONTEXT_FULL;
    if (!GetThreadContext(thread, &ctx)) {
        cardinal::log::warnf("trace", "GetThreadContext failed (err=%lu)", GetLastError());
        ResumeThread(thread);
        return frames;
    }

    if (!ensure_sym_initialised()) { ResumeThread(thread); return frames; }
    cardinal::lock_guard lk(dbghelp_mutex());

    STACKFRAME64 sf{};
    DWORD machine = 0;
#if defined(_M_X64)
    machine = IMAGE_FILE_MACHINE_AMD64;
    sf.AddrPC.Offset    = ctx.Rip;  sf.AddrPC.Mode    = AddrModeFlat;
    sf.AddrFrame.Offset = ctx.Rbp;  sf.AddrFrame.Mode = AddrModeFlat;
    sf.AddrStack.Offset = ctx.Rsp;  sf.AddrStack.Mode = AddrModeFlat;
#elif defined(_M_ARM64)
    machine = IMAGE_FILE_MACHINE_ARM64;
    sf.AddrPC.Offset    = ctx.Pc;   sf.AddrPC.Mode    = AddrModeFlat;
    sf.AddrFrame.Offset = ctx.Fp;   sf.AddrFrame.Mode = AddrModeFlat;
    sf.AddrStack.Offset = ctx.Sp;   sf.AddrStack.Mode = AddrModeFlat;
#endif
    while (frames.size() < max_depth) {
        if (!StackWalk64(machine, proc, thread, &sf, &ctx, nullptr,
                         SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) break;
        if (sf.AddrPC.Offset == 0) break;
        frames.emplace_back();
        resolve_one(sf.AddrPC.Offset, frames.back());
    }

    ResumeThread(thread);
    return frames;
}

cardinal::string format(const StackFrame& f) {
    char buf[1024];
    cardinal::snprintf(buf, sizeof(buf), "0x%016llx  %s!%s%s%s%s%u",
        static_cast<unsigned long long>(f.address),
        f.module.empty()  ? "?" : f.module.c_str(),
        f.symbol.empty()  ? "?" : f.symbol.c_str(),
        f.file.empty() ? "" : "  (",
        f.file.empty() ? "" : f.file.c_str(),
        f.file.empty() ? "" : ":",
        f.line);
    if (!f.file.empty()) cardinal::strncat(buf, ")", sizeof(buf) - cardinal::strlen(buf) - 1);
    return buf;
}

cardinal::string format_full(const cardinal::vector<StackFrame>& frames, const char* indent) {
    cardinal::string out;
    for (size_t i = 0; i < frames.size(); ++i) {
        char idx[16]; cardinal::snprintf(idx, sizeof(idx), "%2zu  ", i);
        if (indent) out += indent;
        out += idx;
        out += format(frames[i]);
        out += '\n';
    }
    return out;
}

}  // namespace cardinal::trace

#else  // POSIX — stub for now

namespace cardinal::trace {
cardinal::vector<StackFrame> capture(u32, u32)              { return {}; }
cardinal::vector<StackFrame> capture_thread(void*, u32)     { return {}; }
cardinal::string format(const StackFrame&)                  { return ""; }
cardinal::string format_full(const cardinal::vector<StackFrame>&, const char*) { return ""; }
}

#endif
