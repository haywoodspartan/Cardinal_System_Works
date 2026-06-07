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
#include <cardinal/core/diag/crash.hpp>

#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/platform.hpp>

#include <atomic>
#include <csignal>      // signal / raise / SIGABRT — abort()/assert path
#include <cstdint>      // uintptr_t — _invalid_parameter_handler signature
#include <cstdio>
#include <cstdlib>      // _set_purecall_handler / _set_invalid_parameter_handler
#include <cstring>
#include <exception>    // std::set_terminate — uncaught / escaped exceptions
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

// Stack-symbolisation surface — also resolved dynamically out of dbghelp.dll
// so the .log carries a readable frame list (module!symbol+off [file:line])
// without anyone needing WinDbg to crack the .dmp.
using PFN_SymInitialize        = BOOL    (WINAPI*)(HANDLE, PCSTR, BOOL);
using PFN_SymSetOptions        = DWORD   (WINAPI*)(DWORD);
using PFN_SymFromAddr          = BOOL    (WINAPI*)(HANDLE, DWORD64, PDWORD64, PSYMBOL_INFO);
using PFN_SymGetLineFromAddr64 = BOOL    (WINAPI*)(HANDLE, DWORD64, PDWORD, PIMAGEHLP_LINE64);
using PFN_StackWalk64          = BOOL    (WINAPI*)(DWORD, HANDLE, HANDLE, LPSTACKFRAME64,
                                                   PVOID, PREAD_PROCESS_MEMORY_ROUTINE64,
                                                   PFUNCTION_TABLE_ACCESS_ROUTINE64,
                                                   PGET_MODULE_BASE_ROUTINE64,
                                                   PTRANSLATE_ADDRESS_ROUTINE64);
using PFN_SymFunctionTableAccess64 = PVOID   (WINAPI*)(HANDLE, DWORD64);
using PFN_SymGetModuleBase64       = DWORD64 (WINAPI*)(HANDLE, DWORD64);

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
// Previous non-SEH fatal handlers, restored by uninstall().
std::terminate_handler       g_prev_terminate{nullptr};
_purecall_handler            g_prev_purecall{nullptr};
_invalid_parameter_handler   g_prev_invparam{nullptr};
HMODULE                      g_dbghelp{nullptr};
PFN_MiniDumpWriteDump        g_pfn_minidump{nullptr};
PFN_SymInitialize            g_pfn_sym_init{nullptr};
PFN_SymSetOptions            g_pfn_sym_opts{nullptr};
PFN_SymFromAddr              g_pfn_sym_from_addr{nullptr};
PFN_SymGetLineFromAddr64     g_pfn_sym_line{nullptr};
PFN_StackWalk64              g_pfn_stack_walk{nullptr};
PFN_SymFunctionTableAccess64 g_pfn_fn_table{nullptr};
PFN_SymGetModuleBase64       g_pfn_mod_base{nullptr};

// Resolve dbghelp.dll lazily. Returns true if MiniDumpWriteDump is
// usable. Cheap to call repeatedly — caches the resolved pointer.
bool ensure_dbghelp() {
    if (g_pfn_minidump != nullptr) return true;
    g_dbghelp = LoadLibraryW(L"dbghelp.dll");
    if (g_dbghelp == nullptr) return false;
    g_pfn_minidump = reinterpret_cast<PFN_MiniDumpWriteDump>(
        GetProcAddress(g_dbghelp, "MiniDumpWriteDump"));

    // Best-effort symbolisation surface. None of these is required for the
    // .dmp itself — if any is missing we just emit a thinner stack.
    g_pfn_sym_init      = reinterpret_cast<PFN_SymInitialize>(
        GetProcAddress(g_dbghelp, "SymInitialize"));
    g_pfn_sym_opts      = reinterpret_cast<PFN_SymSetOptions>(
        GetProcAddress(g_dbghelp, "SymSetOptions"));
    g_pfn_sym_from_addr = reinterpret_cast<PFN_SymFromAddr>(
        GetProcAddress(g_dbghelp, "SymFromAddr"));
    g_pfn_sym_line      = reinterpret_cast<PFN_SymGetLineFromAddr64>(
        GetProcAddress(g_dbghelp, "SymGetLineFromAddr64"));
    g_pfn_stack_walk    = reinterpret_cast<PFN_StackWalk64>(
        GetProcAddress(g_dbghelp, "StackWalk64"));
    g_pfn_fn_table      = reinterpret_cast<PFN_SymFunctionTableAccess64>(
        GetProcAddress(g_dbghelp, "SymFunctionTableAccess64"));
    g_pfn_mod_base      = reinterpret_cast<PFN_SymGetModuleBase64>(
        GetProcAddress(g_dbghelp, "SymGetModuleBase64"));

    return g_pfn_minidump != nullptr;
}

// Walk + symbolise one thread's stack into `lp`. Best-effort: any missing
// dbghelp entry, or a frame we can't symbolise, degrades to a raw address.
// `ctx` is mutated by StackWalk64, so the caller must pass a private copy.
void write_stack_trace(FILE* lp, CONTEXT* ctx) {
    if (lp == nullptr || ctx == nullptr) return;
    if (g_pfn_stack_walk == nullptr || g_pfn_fn_table == nullptr ||
        g_pfn_mod_base   == nullptr) {
        std::fprintf(lp, "Stack: <dbghelp StackWalk64 unavailable>\n");
        return;
    }

    const HANDLE proc = GetCurrentProcess();
    const HANDLE thr  = GetCurrentThread();

    if (g_pfn_sym_init != nullptr) {
        if (g_pfn_sym_opts != nullptr) {
            // UNDNAME → demangled C++ names; LOAD_LINES → file:line records.
            g_pfn_sym_opts(SYMOPT_UNDNAME | SYMOPT_LOAD_LINES |
                           SYMOPT_DEFERRED_LOADS | SYMOPT_FAIL_CRITICAL_ERRORS);
        }
        // invade=TRUE → eagerly enumerate the process' modules so PDBs that
        // sit next to the .exe (our debug build) resolve without a path.
        g_pfn_sym_init(proc, nullptr, TRUE);
    }

    STACKFRAME64 sf{};
    sf.AddrPC.Offset    = ctx->Rip; sf.AddrPC.Mode    = AddrModeFlat;
    sf.AddrFrame.Offset = ctx->Rbp; sf.AddrFrame.Mode = AddrModeFlat;
    sf.AddrStack.Offset = ctx->Rsp; sf.AddrStack.Mode = AddrModeFlat;

    std::fprintf(lp, "Stack (faulting thread, most-recent first):\n");

    // SYMBOL_INFO + room for the (undecorated) name.
    alignas(SYMBOL_INFO) unsigned char sym_buf[sizeof(SYMBOL_INFO) + 512] = {};
    auto* sym = reinterpret_cast<SYMBOL_INFO*>(sym_buf);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen   = 511;

    for (int frame = 0; frame < 64; ++frame) {
        if (!g_pfn_stack_walk(IMAGE_FILE_MACHINE_AMD64, proc, thr, &sf, ctx,
                              nullptr, g_pfn_fn_table, g_pfn_mod_base, nullptr)) {
            break;
        }
        const DWORD64 pc = sf.AddrPC.Offset;
        if (pc == 0) break;

        char modname[64] = "?";
        if (g_pfn_mod_base != nullptr) {
            const DWORD64 base = g_pfn_mod_base(proc, pc);
            if (base != 0) {
                char full[MAX_PATH];
                if (GetModuleFileNameA(reinterpret_cast<HMODULE>(base),
                                       full, sizeof(full)) > 0) {
                    const char* leaf = full;
                    for (const char* p = full; *p; ++p)
                        if (*p == '\\' || *p == '/') leaf = p + 1;
                    std::snprintf(modname, sizeof(modname), "%s", leaf);
                }
            }
        }

        DWORD64 disp = 0;
        bool named = (g_pfn_sym_from_addr != nullptr) &&
                     g_pfn_sym_from_addr(proc, pc, &disp, sym) != FALSE;

        IMAGEHLP_LINE64 line{}; line.SizeOfStruct = sizeof(line);
        DWORD line_disp = 0;
        bool has_line = (g_pfn_sym_line != nullptr) &&
                        g_pfn_sym_line(proc, pc, &line_disp, &line) != FALSE;

        if (named && has_line) {
            std::fprintf(lp, "  #%02d %s!%s+0x%llX  [%s:%lu]  (0x%llX)\n",
                frame, modname, sym->Name,
                static_cast<unsigned long long>(disp),
                line.FileName, static_cast<unsigned long>(line.LineNumber),
                static_cast<unsigned long long>(pc));
        } else if (named) {
            std::fprintf(lp, "  #%02d %s!%s+0x%llX  (0x%llX)\n",
                frame, modname, sym->Name,
                static_cast<unsigned long long>(disp),
                static_cast<unsigned long long>(pc));
        } else {
            std::fprintf(lp, "  #%02d %s!0x%llX\n",
                frame, modname, static_cast<unsigned long long>(pc));
        }
    }
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
        std::fprintf(lp, "\n");

        // Symbolised stack. StackWalk64 mutates the CONTEXT, so always hand
        // it a private copy. Prefer the faulting thread's context from the
        // exception record; fall back to capturing the current frame for
        // the manual write_dump_now() path.
        CONTEXT walk_ctx{};
        if (eptr && eptr->ContextRecord) {
            walk_ctx = *eptr->ContextRecord;
        } else {
            RtlCaptureContext(&walk_ctx);
        }
        write_stack_trace(lp, &walk_ctx);
        std::fclose(lp);
    }

    if (ok) {
        std::lock_guard<std::mutex> lg(g_path_mtx);
        g_last_dump_path = dump;
    }
    return ok;
}

// Shared tail for the non-SEH fatal paths (abort()/assert, std::terminate,
// pure-virtual, CRT invalid-parameter). There is no EXCEPTION_POINTERS
// here, so write_dump_internal falls back to RtlCaptureContext — the
// symbolised stack still pinpoints the failing frame, exactly what was
// missing for the Vulkan-exit VMA assert (assert→abort→SIGABRT never
// reached the SEH filter).
void capture_fatal(const char* reason) {
    bool expected = false;
    if (!g_in_handler.compare_exchange_strong(expected, true)) return;
    write_dump_internal(nullptr, reason);
}

void on_terminate() {
    capture_fatal("std::terminate (uncaught / escaped exception)");
    if (g_prev_terminate) g_prev_terminate();
    _Exit(3);                       // never return from a terminate handler
}

void __cdecl on_signal_abort(int) {
    capture_fatal("abort() / assert (SIGABRT)");
    ::signal(SIGABRT, SIG_DFL);     // restore default, then die for real
    ::raise(SIGABRT);
}

void __cdecl on_purecall() {
    capture_fatal("pure virtual function call");
    _Exit(3);
}

void __cdecl on_invalid_parameter(const wchar_t*, const wchar_t*,
                                   const wchar_t*, unsigned int,
                                   uintptr_t) {
    capture_fatal("CRT invalid parameter");
    _Exit(3);                       // returning would continue in UB
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
    // Non-SEH fatal paths the unhandled-exception filter never sees:
    // assert()/abort() (SIGABRT), uncaught / noexcept-escaped C++
    // exceptions (std::terminate), pure-virtual calls, and CRT
    // invalid-parameter. All route into the same symbolised dump+log.
    g_prev_terminate = std::set_terminate(&on_terminate);
    g_prev_purecall  = _set_purecall_handler(&on_purecall);
    g_prev_invparam  = _set_invalid_parameter_handler(&on_invalid_parameter);
    ::signal(SIGABRT, &on_signal_abort);
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
    std::set_terminate(g_prev_terminate);
    _set_purecall_handler(g_prev_purecall);
    _set_invalid_parameter_handler(g_prev_invparam);
    ::signal(SIGABRT, SIG_DFL);
    g_prev_terminate = nullptr;
    g_prev_purecall  = nullptr;
    g_prev_invparam  = nullptr;
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
