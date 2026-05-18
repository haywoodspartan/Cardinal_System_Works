// =============================================================================
// Cardinal — crash reporter implementation.
//
// Win32 SEH path uses MiniDumpWriteDump from dbghelp.dll — that's the
// canonical Microsoft way to capture process state at the moment of an
// unhandled exception. We dynamically resolve dbghelp so we don't force
// the link on hosts that don't want crash reporting.
//
// On Linux this is a stub today (signal handlers + breakpad/sentry would
// be the right path; not in scope for the first cut).
// =============================================================================
#include <cardinal/core/crash.hpp>

#include <cardinal/core/log.hpp>
#include <cardinal/core/platform.hpp>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>

#if CARDINAL_PLATFORM_WINDOWS

#ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <Windows.h>
#include <DbgHelp.h>     // MiniDumpWriteDump types — we resolve the function dynamically

// Function-pointer types so we don't link dbghelp at compile time.
using PFN_MiniDumpWriteDump = BOOL (WINAPI*)(
    HANDLE hProcess, DWORD ProcessId, HANDLE hFile,
    MINIDUMP_TYPE DumpType,
    PMINIDUMP_EXCEPTION_INFORMATION ExceptionParam,
    PMINIDUMP_USER_STREAM_INFORMATION UserStreamParam,
    PMINIDUMP_CALLBACK_INFORMATION CallbackParam);

#endif

namespace cardinal::crash {

namespace {

CrashConfig          g_cfg{};
std::atomic<bool>    g_installed{false};
std::atomic<bool>    g_in_handler{false};       // re-entrancy guard
std::mutex           g_path_mtx;
std::string          g_last_dump_path;

#if CARDINAL_PLATFORM_WINDOWS
LPTOP_LEVEL_EXCEPTION_FILTER g_prev_filter{nullptr};
HMODULE                      g_dbghelp{nullptr};
PFN_MiniDumpWriteDump        g_pfn_minidump{nullptr};

// Resolve dbghelp.dll lazily. Returns true if MiniDumpWriteDump is
// usable. Cheap to call repeatedly — caches the resolved pointer.
bool ensure_dbghelp() {
    if (g_pfn_minidump != nullptr) return true;
    g_dbghelp = LoadLibraryW(L"dbghelp.dll");
    if (g_dbghelp == nullptr) return false;
    g_pfn_minidump = reinterpret_cast<PFN_MiniDumpWriteDump>(
        GetProcAddress(g_dbghelp, "MiniDumpWriteDump"));
    return g_pfn_minidump != nullptr;
}

MINIDUMP_TYPE dump_type_for(CrashConfig::Detail d) {
    switch (d) {
        case CrashConfig::Detail::Small:
            return static_cast<MINIDUMP_TYPE>(
                MiniDumpNormal | MiniDumpWithThreadInfo |
                MiniDumpWithUnloadedModules);
        case CrashConfig::Detail::Full:
            return static_cast<MINIDUMP_TYPE>(
                MiniDumpWithFullMemory | MiniDumpWithHandleData |
                MiniDumpWithThreadInfo | MiniDumpWithUnloadedModules |
                MiniDumpWithFullMemoryInfo | MiniDumpWithProcessThreadData);
        case CrashConfig::Detail::Default:
        default:
            return static_cast<MINIDUMP_TYPE>(
                MiniDumpNormal | MiniDumpWithDataSegs |
                MiniDumpWithHandleData | MiniDumpWithThreadInfo |
                MiniDumpWithIndirectlyReferencedMemory |
                MiniDumpWithUnloadedModules);
    }
}

// Returns "<exe-dir>" with no trailing slash, or "." on failure.
std::string exe_dir() {
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, sizeof(buf));
    if (n == 0 || n >= sizeof(buf)) return ".";
    for (DWORD i = n; i > 0; --i) {
        if (buf[i - 1] == '\\' || buf[i - 1] == '/') {
            buf[i - 1] = '\0';
            return buf;
        }
    }
    return ".";
}

std::string make_stamp_filename(const char* tag) {
    SYSTEMTIME t; GetLocalTime(&t);
    char buf[128];
    std::snprintf(buf, sizeof(buf),
        "%04u-%02u-%02u_%02u%02u%02u_PID%lu%s",
        t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond,
        GetCurrentProcessId(), tag);
    return buf;
}

void ensure_dir(const std::string& path) {
    // CreateDirectoryA returns FALSE + ERROR_ALREADY_EXISTS if the dir
    // exists, which we treat as success. Doesn't recurse — the parent
    // is always <exe-dir> which already exists.
    CreateDirectoryA(path.c_str(), nullptr);
}

const char* exception_code_name(DWORD code) noexcept {
    switch (code) {
        case EXCEPTION_ACCESS_VIOLATION:        return "ACCESS_VIOLATION";
        case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:   return "ARRAY_BOUNDS_EXCEEDED";
        case EXCEPTION_DATATYPE_MISALIGNMENT:   return "DATATYPE_MISALIGNMENT";
        case EXCEPTION_FLT_DIVIDE_BY_ZERO:      return "FLT_DIVIDE_BY_ZERO";
        case EXCEPTION_FLT_INVALID_OPERATION:   return "FLT_INVALID_OPERATION";
        case EXCEPTION_FLT_OVERFLOW:            return "FLT_OVERFLOW";
        case EXCEPTION_FLT_STACK_CHECK:         return "FLT_STACK_CHECK";
        case EXCEPTION_FLT_UNDERFLOW:           return "FLT_UNDERFLOW";
        case EXCEPTION_ILLEGAL_INSTRUCTION:     return "ILLEGAL_INSTRUCTION";
        case EXCEPTION_IN_PAGE_ERROR:           return "IN_PAGE_ERROR";
        case EXCEPTION_INT_DIVIDE_BY_ZERO:      return "INT_DIVIDE_BY_ZERO";
        case EXCEPTION_INT_OVERFLOW:            return "INT_OVERFLOW";
        case EXCEPTION_PRIV_INSTRUCTION:        return "PRIV_INSTRUCTION";
        case EXCEPTION_STACK_OVERFLOW:          return "STACK_OVERFLOW";
        case EXCEPTION_BREAKPOINT:              return "BREAKPOINT";
        case EXCEPTION_SINGLE_STEP:             return "SINGLE_STEP";
        case EXCEPTION_GUARD_PAGE:              return "GUARD_PAGE";
        case 0xE06D7363u:                       return "MSVC_CPP_EXCEPTION"; // throw
        default:                                return "UNKNOWN";
    }
}

bool write_dump_internal(EXCEPTION_POINTERS* eptr, const char* reason) {
    if (!ensure_dbghelp()) return false;

    const std::string dir = g_cfg.dump_dir.empty() ? (exe_dir() + "\\crashes")
                                                   : g_cfg.dump_dir;
    ensure_dir(dir);
    const std::string stem = dir + "\\" + make_stamp_filename("");
    const std::string dump = stem + ".dmp";
    const std::string log  = stem + ".log";

    // ----- Write the .dmp -------------------------------------------------
    HANDLE fh = CreateFileA(dump.c_str(), GENERIC_WRITE, 0, nullptr,
                            CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    bool ok = false;
    if (fh != INVALID_HANDLE_VALUE) {
        MINIDUMP_EXCEPTION_INFORMATION mei{};
        mei.ThreadId          = GetCurrentThreadId();
        mei.ExceptionPointers = eptr;
        mei.ClientPointers    = FALSE;
        ok = g_pfn_minidump(GetCurrentProcess(), GetCurrentProcessId(),
                            fh, dump_type_for(g_cfg.detail),
                            eptr ? &mei : nullptr, nullptr, nullptr) != FALSE;
        CloseHandle(fh);
    }

    // ----- Write the .log (small, easy to read without WinDbg) -----------
    if (FILE* lp = std::fopen(log.c_str(), "w")) {
        SYSTEMTIME t; GetLocalTime(&t);
        std::fprintf(lp, "Cardinal crash report\n");
        std::fprintf(lp, "Time : %04u-%02u-%02u %02u:%02u:%02u\n",
            t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond);
        std::fprintf(lp, "PID  : %lu\n", GetCurrentProcessId());
        std::fprintf(lp, "TID  : %lu\n", GetCurrentThreadId());
        std::fprintf(lp, "Why  : %s\n", reason ? reason : "(none)");
        if (eptr && eptr->ExceptionRecord) {
            const auto* er = eptr->ExceptionRecord;
            std::fprintf(lp, "Code : 0x%08lX (%s)\n",
                static_cast<unsigned long>(er->ExceptionCode),
                exception_code_name(er->ExceptionCode));
            std::fprintf(lp, "Addr : %p\n", er->ExceptionAddress);
            if (er->ExceptionCode == EXCEPTION_ACCESS_VIOLATION &&
                er->NumberParameters >= 2) {
                std::fprintf(lp, "AVKind: %s\n",
                    er->ExceptionInformation[0] == 0 ? "READ"  :
                    er->ExceptionInformation[0] == 1 ? "WRITE" :
                    er->ExceptionInformation[0] == 8 ? "DEP"   : "?");
                std::fprintf(lp, "AVPtr : 0x%p\n",
                    reinterpret_cast<void*>(er->ExceptionInformation[1]));
            }
        }
        std::fprintf(lp, "Dump : %s\n", dump.c_str());
        std::fprintf(lp, "Open with: WinDbg / Visual Studio (File→Open→Dump).\n");
        std::fclose(lp);
    }

    if (ok) {
        std::lock_guard<std::mutex> lg(g_path_mtx);
        g_last_dump_path = dump;
    }
    return ok;
}

LONG WINAPI top_level_filter(EXCEPTION_POINTERS* eptr) {
    // Re-entrancy guard — if our own dump-writing crashes, don't loop.
    bool expected = false;
    if (!g_in_handler.compare_exchange_strong(expected, true)) {
        return EXCEPTION_EXECUTE_HANDLER;
    }
    write_dump_internal(eptr, "unhandled exception");
    // Don't try to do anything fancy from here — the process state is
    // partially undefined; just terminate (or hand off to default).
    return g_cfg.also_default ? EXCEPTION_CONTINUE_SEARCH
                              : EXCEPTION_EXECUTE_HANDLER;
}
#endif  // CARDINAL_PLATFORM_WINDOWS

}  // namespace

bool install(const CrashConfig& cfg) {
#if CARDINAL_PLATFORM_WINDOWS
    bool expected = false;
    if (!g_installed.compare_exchange_strong(expected, true)) return true;
    g_cfg = cfg;
    g_prev_filter = SetUnhandledExceptionFilter(&top_level_filter);
    cardinal::log::infof("crash",
        "crash reporter armed (dump_dir=%s detail=%s also_default=%s)",
        cfg.dump_dir.empty() ? "<exe>/crashes" : cfg.dump_dir.c_str(),
        cfg.detail == CrashConfig::Detail::Small   ? "small" :
        cfg.detail == CrashConfig::Detail::Full    ? "full"  : "default",
        cfg.also_default ? "yes" : "no");
    return true;
#else
    (void)cfg;
    return false;
#endif
}

void uninstall() noexcept {
#if CARDINAL_PLATFORM_WINDOWS
    bool expected = true;
    if (!g_installed.compare_exchange_strong(expected, false)) return;
    SetUnhandledExceptionFilter(g_prev_filter);
    g_prev_filter = nullptr;
#endif
}

std::string write_dump_now(const char* reason) {
#if CARDINAL_PLATFORM_WINDOWS
    write_dump_internal(nullptr, reason);
    std::lock_guard<std::mutex> lg(g_path_mtx);
    return g_last_dump_path;
#else
    (void)reason;
    return {};
#endif
}

std::string last_dump_path() noexcept {
    std::lock_guard<std::mutex> lg(g_path_mtx);
    return g_last_dump_path;
}

}  // namespace cardinal::crash
