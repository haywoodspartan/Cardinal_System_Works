// =============================================================================
// Cardinal Launcher — standalone WinMain GUI executable.
//
// Shows the backend-selection dialog (cardinal::launcher::show_startup_menu)
// in its own window, then spawns the Studio executable
// (Cardinal_System_Studio.exe by default — overridable via the
// CARDINAL_GAME_EXE env var or `-e <path>`) with `--backend=<name>` on the
// command line. The launcher exits as soon as the game has been started; it
// does not wait for the game to exit.
//
// Built as /SUBSYSTEM:WINDOWS so launching from the shell never opens a
// console window. Diagnostic output goes to a `cardinal_launcher.log` file
// next to the launcher binary.
// =============================================================================

#include <cardinal/launcher/startup.hpp>

// WIN32_LEAN_AND_MEAN + NOMINMAX come from cardinal::compile_flags.
#include <Windows.h>
#include <shellapi.h>

#include <cstdio>
#include <cstring>
#include <string>

namespace {

// Resolve the directory containing the launcher exe (no trailing slash kept).
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

// Tiny rolling log next to the launcher. Useful when the user reports
// "the launcher window flashed and disappeared" — we don't have stdout.
class FileLog {
public:
    explicit FileLog(const std::string& path) {
        f_ = std::fopen(path.c_str(), "a");
        if (f_) {
            SYSTEMTIME t; GetLocalTime(&t);
            std::fprintf(f_, "\n--- %04u-%02u-%02u %02u:%02u:%02u ---\n",
                t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond);
        }
    }
    ~FileLog() { if (f_) std::fclose(f_); }
    void writef(const char* fmt, ...) {
        if (!f_) return;
        va_list a; va_start(a, fmt);
        std::vfprintf(f_, fmt, a);
        std::fputc('\n', f_);
        std::fflush(f_);
        va_end(a);
    }
private:
    std::FILE* f_{nullptr};
};

// Pop a small message-box error so the user isn't left guessing when something
// breaks. Used for misconfiguration (game exe not found) and CreateProcess fail.
void show_error(const char* title, const char* msg) {
    MessageBoxA(nullptr, msg, title, MB_OK | MB_ICONERROR | MB_TOPMOST);
}

const char* backend_arg(cardinal::launcher::BackendChoice c) {
    switch (c) {
        case cardinal::launcher::BackendChoice::Vulkan: return "vulkan";
        case cardinal::launcher::BackendChoice::D3D12:  return "d3d12";
        case cardinal::launcher::BackendChoice::Auto:   return "auto";
        default:                                        return "auto";
    }
}

// Default Studio executable name. Build target lives at
// samples/03_studio/CMakeLists.txt and is named Cardinal_System_Studio,
// so the produced binary is `Cardinal_System_Studio.exe`. Keep this in
// sync if the target name ever changes again.
constexpr const char* kDefaultStudioExe = "Cardinal_System_Studio.exe";

// Find the Studio exe. Order:
//   1) `-e <path>` on the command line
//   2) %CARDINAL_GAME_EXE%
//   3) <launcher-dir>\Cardinal_System_Studio.exe
//
// If none of the above resolves to an existing file we still return the
// default path so the caller can show a meaningful "not found" error
// pointing at where we expected to find it.
std::string resolve_game_exe(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "-e") == 0) return argv[i + 1];
    }
    char env[MAX_PATH];
    DWORD n = GetEnvironmentVariableA("CARDINAL_GAME_EXE", env, sizeof(env));
    if (n > 0 && n < sizeof(env)) return env;

    return exe_dir() + "\\" + kDefaultStudioExe;
}

}  // namespace

// Entry point — /SUBSYSTEM:WINDOWS, no console.
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
    // Recover argc/argv from the OS so we can support -e <path> overrides.
    int      argc = 0;
    LPWSTR*  wargv = CommandLineToArgvW(GetCommandLineW(), &argc);
    char**   argv = static_cast<char**>(std::calloc(argc, sizeof(char*)));
    for (int i = 0; i < argc; ++i) {
        int sz = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        argv[i] = static_cast<char*>(std::malloc(sz));
        WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, argv[i], sz, nullptr, nullptr);
    }
    LocalFree(wargv);

    FileLog log(exe_dir() + "\\cardinal_launcher.log");
    log.writef("Launcher started — exe_dir=%s", exe_dir().c_str());

    // ---- Show the dialog ---------------------------------------------------
    cardinal::launcher::StartupOptions opts{};
    opts.app_title = "Cardinal";
    opts.subtitle  = "Choose graphics backend";
    const auto pick = cardinal::launcher::show_startup_menu(opts);
    if (pick == cardinal::launcher::BackendChoice::Cancel) {
        log.writef("User cancelled.");
        return 0;
    }
    log.writef("User picked: %s", cardinal::launcher::backend_name(pick));

    // ---- Spawn the game ----------------------------------------------------
    const std::string game = resolve_game_exe(argc, argv);
    if (GetFileAttributesA(game.c_str()) == INVALID_FILE_ATTRIBUTES) {
        char msg[1024];
        std::snprintf(msg, sizeof(msg),
            "Couldn't find the game executable.\n\nLooked for:\n  %s\n\n"
            "Pass `-e <path>` on the command line, or set the\n"
            "CARDINAL_GAME_EXE environment variable.",
            game.c_str());
        show_error("Cardinal Launcher", msg);
        log.writef("ERROR: game exe not found: %s", game.c_str());
        return 1;
    }

    // Build the command line: "<game>" --backend=<name>
    char cmdline[MAX_PATH + 64];
    std::snprintf(cmdline, sizeof(cmdline), "\"%s\" --backend=%s",
                  game.c_str(), backend_arg(pick));
    log.writef("Spawning: %s", cmdline);

    STARTUPINFOA        si{};  si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    BOOL ok = CreateProcessA(nullptr, cmdline, nullptr, nullptr, FALSE,
                             0, nullptr, exe_dir().c_str(), &si, &pi);
    if (!ok) {
        char msg[512];
        const DWORD err = GetLastError();
        std::snprintf(msg, sizeof(msg),
            "Failed to launch the game (Win32 error %lu).\n\nCommand:\n  %s",
            err, cmdline);
        show_error("Cardinal Launcher", msg);
        log.writef("ERROR: CreateProcess failed (err=%lu)", err);
        return 1;
    }

    // We don't wait for the game — the launcher exits immediately. Close the
    // handles so the OS reaps the child once it terminates.
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    log.writef("Game launched, launcher exiting.");

    for (int i = 0; i < argc; ++i) std::free(argv[i]);
    std::free(argv);
    return 0;
}
