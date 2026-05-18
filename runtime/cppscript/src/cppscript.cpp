// =============================================================================
// Cardinal — C++ scripting engine implementation.
//
// Pipeline per submitted job:
//   1. Resolve compiler executable (Desc::compiler / env / vswhere / PATH).
//   2. Worker thread picks the job, runs the compiler as a child process
//      via CreateProcess (Windows) / posix_spawn (Linux), capturing stdout
//      + stderr through anonymous pipes into job.diagnostics.
//   3. On non-zero exit -> JobStatus::CompileFailed, observer fires.
//   4. On success -> mark LoadPending, the next Engine::tick() picks it
//      off the main thread and calls plugin::Registry::load(dll_path).
//   5. plugin::Registry handles the DLL lifetime + on_attach / on_tick /
//      on_detach + SEH protection — same path as a hand-built plugin.
// =============================================================================

#include <cardinal/cppscript/cppscript.hpp>

#include <cardinal/core/log.hpp>
#include <cardinal/core/platform.hpp>
#include <cardinal/plugin/plugin.hpp>

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/atomic.hpp>
#include <cardinal/core/chrono.hpp>
#include <cardinal/core/cctype.hpp>
#include <cardinal/core/containers.hpp>
#include <cardinal/core/cstdio.hpp>
#include <cardinal/core/cstdlib.hpp>
#include <cardinal/core/cstring.hpp>
#include <cardinal/core/filesystem.hpp>
#include <cardinal/core/fstream.hpp>
#include <cardinal/core/thread.hpp>
#include <cardinal/core/utility.hpp>

#if CARDINAL_PLATFORM_WINDOWS
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <Windows.h>
#else
    #include <spawn.h>
    #include <sys/wait.h>
    #include <unistd.h>
    extern char** environ;
#endif

namespace cardinal::cppscript {

namespace fs = cardinal::fs;

// ----------------------------------------------------------------------------
// Compiler discovery — runs once at Engine::create. Probes a few candidates
// and stops at the first hit. Order matches Desc::CompilerKind::Auto's contract.
// ----------------------------------------------------------------------------
namespace {

bool file_exists(const cardinal::string& path) {
    cardinal::error_code ec;
    return !path.empty() && fs::exists(path, ec) && fs::is_regular_file(path, ec);
}

cardinal::string from_env(const char* name) {
    const char* v = cardinal::getenv(name);
    return v ? cardinal::string(v) : cardinal::string{};
}

#if CARDINAL_PLATFORM_WINDOWS
// `where <exe>` returns the first hit on PATH. We invoke it via popen because
// implementing the full PATH walk + .EXE/.CMD/.BAT extension search would
// duplicate cmd.exe's logic and is brittle on systems with PATHEXT tweaks.
cardinal::string which_via_where(const char* exe_basename) {
    cardinal::string cmd = "where ";
    cmd += exe_basename;
    cmd += " 2>NUL";
    FILE* f = _popen(cmd.c_str(), "r");
    if (f == nullptr) return {};
    cardinal::string out;
    char buf[1024];
    while (cardinal::fgets(buf, sizeof(buf), f) != nullptr) out += buf;
    _pclose(f);
    // Strip the first line (where prints multiple if there are multiple hits).
    auto nl = out.find_first_of("\r\n");
    if (nl != cardinal::string::npos) out.resize(nl);
    return out;
}

// vswhere.exe ships with every Visual Studio 2017+ install; locating it is
// reliable because Microsoft committed to its location.
cardinal::string vswhere_find_install_path() {
    const cardinal::string vswhere =
        R"(C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe)";
    if (!file_exists(vswhere)) return {};
    cardinal::string cmd = "\"" + vswhere +
        R"(" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>NUL)";
    FILE* f = _popen(cmd.c_str(), "r");
    if (f == nullptr) return {};
    char buf[1024]; cardinal::string install_path;
    while (cardinal::fgets(buf, sizeof(buf), f) != nullptr) install_path += buf;
    _pclose(f);
    while (!install_path.empty() &&
           (install_path.back() == '\n' || install_path.back() == '\r' ||
            install_path.back() == ' ')) install_path.pop_back();
    return install_path;
}

cardinal::string vswhere_find_cl() {
    const cardinal::string install_path = vswhere_find_install_path();
    if (install_path.empty()) return {};

    fs::path tools = fs::path(install_path) / "VC" / "Tools" / "MSVC";
    cardinal::error_code ec;
    if (!fs::exists(tools, ec)) return {};

    cardinal::string best_version;
    for (auto& e : fs::directory_iterator(tools, ec)) {
        if (!e.is_directory(ec)) continue;
        const cardinal::string v = e.path().filename().string();
        if (v > best_version) best_version = v;
    }
    if (best_version.empty()) return {};

    fs::path cl = tools / best_version / "bin" / "Hostx64" / "x64" / "cl.exe";
    return file_exists(cl.string()) ? cl.string() : cardinal::string{};
}

// Locate VsDevCmd.bat. Sourced before each cl.exe invocation so cl can find
// <cstddef>, <Windows.h>, kernel32.lib, etc. without us re-implementing the
// full INCLUDE/LIB scan. vswhere → installationPath → Common7\Tools\VsDevCmd.bat.
cardinal::string vswhere_find_vsdevcmd() {
    const cardinal::string install_path = vswhere_find_install_path();
    if (install_path.empty()) return {};
    fs::path bat = fs::path(install_path) / "Common7" / "Tools" / "VsDevCmd.bat";
    return file_exists(bat.string()) ? bat.string() : cardinal::string{};
}
#endif

cardinal::string discover_compiler(const Desc& d) {
    if (!d.compiler_override.empty() && file_exists(d.compiler_override)) {
        return d.compiler_override;
    }
    if (auto env = from_env("CARDINAL_CPPSCRIPT_COMPILER"); !env.empty()) {
        if (file_exists(env)) return env;
    }
#if CARDINAL_PLATFORM_WINDOWS
    if (d.compiler == CompilerKind::Auto || d.compiler == CompilerKind::ClangCl) {
        if (auto p = which_via_where("clang-cl.exe"); file_exists(p)) return p;
    }
    if (d.compiler == CompilerKind::Auto || d.compiler == CompilerKind::ClPath) {
        if (auto p = which_via_where("cl.exe"); file_exists(p)) return p;
        if (auto p = vswhere_find_cl();        file_exists(p)) return p;
    }
#else
    (void)d;
    // POSIX path: just probe PATH for clang++ then g++.
    auto which = [](const char* name) -> cardinal::string {
        cardinal::string cmd = cardinal::string("which ") + name + " 2>/dev/null";
        FILE* f = popen(cmd.c_str(), "r");
        if (!f) return {};
        char buf[512]; cardinal::string out;
        while (cardinal::fgets(buf, sizeof(buf), f)) out += buf;
        pclose(f);
        while (!out.empty() && (out.back()=='\n' || out.back()=='\r')) out.pop_back();
        return out;
    };
    if (auto p = which("clang++"); file_exists(p)) return p;
    if (auto p = which("g++");     file_exists(p)) return p;
#endif
    return {};
}

bool compiler_is_msvc_style(const cardinal::string& path) {
    // clang-cl AND cl.exe both accept MSVC-style flags ("/std:c++23"), so
    // we treat them the same for command-line construction.
    auto base = fs::path(path).filename().string();
    for (auto& c : base) c = static_cast<char>(cardinal::tolower(static_cast<unsigned char>(c)));
    return base == "cl.exe" || base == "clang-cl.exe";
}

// Run a child process; capture stdout+stderr together; block until exit.
// Returns the process exit code (-1 on spawn failure).
i64 run_subprocess_capturing(const cardinal::string& exe,
                             const cardinal::vector<cardinal::string>& argv,
                             cardinal::string& out_log)
{
#if CARDINAL_PLATFORM_WINDOWS
    // Build a properly-quoted command line. Standard CreateProcess quoting:
    // wrap each argv in double-quotes, escape embedded "s as \", escape
    // embedded backslashes that immediately precede a quote.
    auto quote = [](const cardinal::string& s) -> cardinal::string {
        if (!s.empty() && s.find_first_of(" \t\"") == cardinal::string::npos) return s;
        cardinal::string q = "\"";
        size_t bs = 0;
        for (char c : s) {
            if (c == '\\') { ++bs; q.push_back(c); continue; }
            if (c == '"') {
                q.append(bs + 1, '\\');   // escape the run of backslashes + the quote
                q.push_back('"');
                bs = 0;
                continue;
            }
            bs = 0;
            q.push_back(c);
        }
        q.append(bs, '\\');               // escape trailing run for the closing quote
        q.push_back('"');
        return q;
    };
    cardinal::string cmdline = quote(exe);
    for (const auto& a : argv) { cmdline.push_back(' '); cmdline += quote(a); }

    SECURITY_ATTRIBUTES sa{};
    sa.nLength              = sizeof(sa);
    sa.bInheritHandle       = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) {
        out_log = "[cppscript] CreatePipe failed\n";
        return -1;
    }
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb         = sizeof(si);
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdOutput = wr;
    si.hStdError  = wr;
    si.hStdInput  = nullptr;

    PROCESS_INFORMATION pi{};
    BOOL ok = CreateProcessA(nullptr,
        cmdline.data(), nullptr, nullptr,
        TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(wr);
    if (!ok) {
        CloseHandle(rd);
        out_log = "[cppscript] CreateProcessA failed: " + cardinal::to_string(GetLastError()) + "\n";
        return -1;
    }

    // Drain stdout+stderr in this thread. The child writes to `wr` (now
    // closed in the parent); reading `rd` returns EOF when the child exits.
    char buf[4096]; DWORD n = 0;
    while (ReadFile(rd, buf, sizeof(buf), &n, nullptr) && n > 0) {
        out_log.append(buf, n);
    }
    CloseHandle(rd);

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return static_cast<i64>(code);
#else
    int pipe_fd[2] = {-1, -1};
    if (pipe(pipe_fd) != 0) { out_log = "[cppscript] pipe() failed\n"; return -1; }

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addclose(&fa, pipe_fd[0]);
    posix_spawn_file_actions_adddup2 (&fa, pipe_fd[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2 (&fa, pipe_fd[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&fa, pipe_fd[1]);

    cardinal::vector<char*> argv_c;
    argv_c.reserve(argv.size() + 2);
    argv_c.push_back(const_cast<char*>(exe.c_str()));
    for (auto& a : argv) argv_c.push_back(const_cast<char*>(a.c_str()));
    argv_c.push_back(nullptr);

    pid_t pid = 0;
    int spawn_rc = posix_spawn(&pid, exe.c_str(), &fa, nullptr, argv_c.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    close(pipe_fd[1]);
    if (spawn_rc != 0) {
        close(pipe_fd[0]);
        out_log = "[cppscript] posix_spawn failed\n";
        return -1;
    }

    char buf[4096]; ssize_t n;
    while ((n = read(pipe_fd[0], buf, sizeof(buf))) > 0) out_log.append(buf, static_cast<size_t>(n));
    close(pipe_fd[0]);
    int wstatus = 0;
    waitpid(pid, &wstatus, 0);
    return WIFEXITED(wstatus) ? static_cast<i64>(WEXITSTATUS(wstatus)) : -1;
#endif
}

}  // namespace

// ----------------------------------------------------------------------------
// Engine implementation
// ----------------------------------------------------------------------------
class EngineImpl final : public Engine {
public:
    explicit EngineImpl(const Desc& d) : desc_(d) {}

    bool initialize() {
        compiler_ = discover_compiler(desc_);
        if (compiler_.empty()) {
            cardinal::log::warnf("cppscript",
                "no C++ compiler found — set CARDINAL_CPPSCRIPT_COMPILER, "
                "or install Visual Studio 2022 / clang-cl");
            // Engine still returns OK — the user can submit jobs which will
            // fail with a clear "no compiler" diagnostic. Better than a hard
            // boot failure when the editor doesn't always need scripting.
        } else {
            cardinal::log::infof("cppscript", "compiler: %s", compiler_.c_str());
        }

#if CARDINAL_PLATFORM_WINDOWS
        // MSVC-style compilers (cl.exe and clang-cl.exe) need INCLUDE / LIB
        // env vars set up by VsDevCmd.bat before they can find <cstddef> /
        // kernel32.lib / etc. We discover the .bat path once + invoke
        // through `cmd /c "VsDevCmd & cl ..."` per compile job. clang-cl
        // doesn't strictly need VsDevCmd if its own toolchain is on PATH,
        // but for the auto-discovered case it usually does.
        if (!compiler_.empty() && compiler_is_msvc_style(compiler_)) {
            vsdevcmd_path_ = vswhere_find_vsdevcmd();
            if (!vsdevcmd_path_.empty()) {
                cardinal::log::infof("cppscript",
                    "VsDevCmd: %s (sourced per compile)",
                    vsdevcmd_path_.c_str());
            } else {
                cardinal::log::warnf("cppscript",
                    "MSVC compiler resolved but VsDevCmd.bat not found — "
                    "compiles may fail with missing INCLUDE/LIB. Run from a "
                    "VS Dev Prompt as a workaround.");
            }
        }
#endif

        if (desc_.cache_dir.empty()) {
            // Default: <cwd>/cppscript_cache. exe-relative would be nicer
            // but cwd is what the launcher already sets to bin/.
            desc_.cache_dir = (fs::current_path() / "cppscript_cache").string();
        }
        cardinal::error_code ec;
        fs::create_directories(desc_.cache_dir, ec);

        worker_ = cardinal::thread([this]{ worker_main(); });
        return true;
    }

    ~EngineImpl() override {
        {
            cardinal::lock_guard lk(queue_mu_);
            shutting_down_ = true;
            queue_cv_.notify_all();
        }
        if (worker_.joinable()) worker_.join();
    }

    JobHandle compile_and_load(const char* source_path) override {
        if (source_path == nullptr || *source_path == '\0') return {};
        cardinal::error_code ec;
        cardinal::string canon = fs::weakly_canonical(source_path, ec).string();
        if (canon.empty()) canon = source_path;
        return enqueue_job_(canon, /*is_string=*/false, /*literal=*/{});
    }

    JobHandle compile_string(const char* source_code,
                             const char* logical_name) override
    {
        if (source_code == nullptr || logical_name == nullptr) return {};
        // Stage the literal source into the cache dir so the compiler can
        // see it. We use the logical name for the on-disk file too so
        // multiple compile_string calls with the same name overwrite each
        // other (intended for live-edit workflows).
        cardinal::string fname = cardinal::string(logical_name) + ".cpp";
        // Sanitise: strip path separators so the user can't escape the cache.
        for (char& c : fname) if (c == '/' || c == '\\') c = '_';
        fs::path src = fs::path(desc_.cache_dir) / fname;
        cardinal::ofstream f(src, cardinal::ios::binary | cardinal::ios::trunc);
        if (!f) return {};
        f << source_code;
        f.close();
        return enqueue_job_(src.string(), /*is_string=*/true, source_code);
    }

    bool unload(const char* name_or_path) override {
        if (name_or_path == nullptr) return false;
        // Look up which DLL we built for this source so the user can pass
        // either the .cpp source path or the plugin's name.
        {
            cardinal::lock_guard lk(records_mu_);
            for (auto& [id, r] : records_) {
                if (r.source_path == name_or_path) {
                    return cardinal::plugin::Registry::instance().unload(
                        r.dll_path.c_str());
                }
            }
        }
        // Fall through: maybe the user passed a plugin name directly.
        return cardinal::plugin::Registry::instance().unload(name_or_path);
    }

    JobHandle reload(const char* source_path) override {
        // Identical to compile_and_load — the plugin Registry's load() is
        // path-keyed and will skip an already-loaded entry, but we go via
        // unload + load to force a re-attach.
        if (source_path == nullptr) return {};
        // Find prior DLL output for this source so we can unload first.
        cardinal::string prior_dll;
        {
            cardinal::lock_guard lk(records_mu_);
            for (auto& [id, r] : records_) {
                if (r.source_path == source_path && !r.dll_path.empty()) {
                    prior_dll = r.dll_path;
                    break;
                }
            }
        }
        if (!prior_dll.empty()) {
            cardinal::plugin::Registry::instance().unload(prior_dll.c_str());
        }
        return compile_and_load(source_path);
    }

    void watch(const char* path) override {
        if (path == nullptr) return;
        cardinal::lock_guard lk(watch_mu_);
        WatchEntry e;
        e.path = path;
        cardinal::error_code ec;
        e.last_mtime = fs::last_write_time(path, ec);
        watches_.push_back(cardinal::move(e));
    }

    void unwatch(const char* path) override {
        if (path == nullptr) return;
        cardinal::lock_guard lk(watch_mu_);
        watches_.erase(cardinal::remove_if(watches_.begin(), watches_.end(),
            [&](const WatchEntry& e){ return e.path == path; }),
            watches_.end());
    }

    void tick() override {
        // 1. Promote LoadPending records → actual plugin load. This MUST
        //    run on the main thread because LoadLibrary + plugin on_attach
        //    aren't thread-safe relative to the engine.
        cardinal::vector<u64> to_load;
        {
            cardinal::lock_guard lk(records_mu_);
            for (auto& [id, r] : records_) {
                if (r.status == JobStatus::LoadPending) to_load.push_back(id);
            }
        }
        for (u64 id : to_load) attempt_load_main_thread_(id);

        // 2. Service the file watcher.
        service_watches_();
    }

    void set_observer(JobObserver cb) override {
        cardinal::lock_guard lk(records_mu_);
        observer_ = cardinal::move(cb);
    }

    JobInfo query(JobHandle h) const override {
        cardinal::lock_guard lk(records_mu_);
        auto it = records_.find(h.id);
        return (it != records_.end()) ? it->second.snapshot() : JobInfo{};
    }

    cardinal::vector<JobInfo> jobs() const override {
        cardinal::vector<JobInfo> out;
        cardinal::lock_guard lk(records_mu_);
        out.reserve(records_.size());
        for (auto& [id, r] : records_) out.push_back(r.snapshot());
        return out;
    }

    cardinal::string compiler_path() const override { return compiler_; }

private:
    struct Record {
        JobHandle    handle;
        cardinal::string  source_path;
        cardinal::string  dll_path;
        JobStatus    status{JobStatus::Pending};
        cardinal::string  diagnostics;
        i64          exit_code{0};
        cardinal::chrono::steady_clock::time_point t_start;
        f64          elapsed_seconds{0.0};

        JobInfo snapshot() const {
            JobInfo i;
            i.handle      = handle;
            i.source_path = source_path;
            i.dll_path    = dll_path;
            i.status      = status;
            i.diagnostics = diagnostics;
            i.exit_code   = exit_code;
            i.elapsed_seconds = elapsed_seconds;
            return i;
        }
    };

    JobHandle enqueue_job_(const cardinal::string& source_path,
                           bool /*is_string*/,
                           const cardinal::string& /*literal*/)
    {
        const u64 id = ++next_job_id_;
        Record r;
        r.handle.id    = id;
        r.source_path  = source_path;
        // Output DLL path: <cache>/<base>.<id>.dll. Embedding the job id
        // means hot-reload always loads a fresh DLL the loader hasn't seen.
        const cardinal::string base = fs::path(source_path).stem().string();
        r.dll_path     = (fs::path(desc_.cache_dir) /
                          (base + "." + cardinal::to_string(id) + ".dll")).string();
        r.t_start      = cardinal::chrono::steady_clock::now();
        {
            cardinal::lock_guard lk(records_mu_);
            records_.emplace(id, cardinal::move(r));
        }
        {
            cardinal::lock_guard lk(queue_mu_);
            queue_.push(id);
            queue_cv_.notify_one();
        }
        return JobHandle{id};
    }

    void worker_main() {
        for (;;) {
            u64 id = 0;
            {
                cardinal::unique_lock lk(queue_mu_);
                queue_cv_.wait(lk, [&]{ return shutting_down_ || !queue_.empty(); });
                if (shutting_down_ && queue_.empty()) return;
                id = queue_.front(); queue_.pop();
            }
            run_compile_job_(id);
        }
    }

    void run_compile_job_(u64 id) {
        // Snapshot what we need under the lock so we don't hold it across
        // the compiler subprocess (which can take seconds).
        cardinal::string source_path, dll_path;
        {
            cardinal::lock_guard lk(records_mu_);
            auto it = records_.find(id);
            if (it == records_.end()) return;
            it->second.status = JobStatus::Compiling;
            source_path       = it->second.source_path;
            dll_path          = it->second.dll_path;
        }
        if (compiler_.empty()) {
            finalize_(id, JobStatus::CompileFailed, -1,
                "[cppscript] no compiler available — set CARDINAL_CPPSCRIPT_COMPILER\n");
            return;
        }

        cardinal::vector<cardinal::string> args = build_compile_args_(source_path, dll_path);
        cardinal::string log;
        i64 rc = -1;

#if CARDINAL_PLATFORM_WINDOWS
        // MSVC compilers need INCLUDE/LIB/PATH env vars from VsDevCmd.bat.
        // Build a single cmd.exe command line of the form:
        //   cmd /c "call "VsDevCmd.bat" -arch=x64 -no_logo >NUL & "cl" <args> 2>&1"
        // and run via _popen. _popen handles the cmd.exe invocation for us
        // and returns a pipe with stdout+stderr (because of 2>&1 + cmd's
        // own pipe handling).
        if (compiler_is_msvc_style(compiler_) && !vsdevcmd_path_.empty()) {
            cardinal::string cmdline;
            cmdline.reserve(2048);
            // Outer quotes are critical — cmd /c "..." treats the contents
            // as a single command.
            cmdline += "\"\"";
            cmdline += vsdevcmd_path_;
            cmdline += "\" -arch=x64 -no_logo >NUL && \"";
            cmdline += compiler_;
            cmdline += "\"";
            for (const auto& a : args) {
                cmdline += " ";
                if (a.find_first_of(" \t\"") != cardinal::string::npos) {
                    cmdline += "\"";
                    for (char c : a) {
                        if (c == '"') cmdline += "\\\"";
                        else          cmdline.push_back(c);
                    }
                    cmdline += "\"";
                } else {
                    cmdline += a;
                }
            }
            cmdline += " 2>&1\"";
            // Final form: `"" ... 2>&1"` — the leading "" + trailing " is
            // cmd.exe's documented "strip outer quotes" idiom (cmd /? for
            // details). _popen's "r" mode captures stdout (and we 2>&1
            // stderr into stdout above).
            FILE* f = _popen(cmdline.c_str(), "r");
            if (f == nullptr) {
                log = "[cppscript] _popen failed for compile pipeline\n";
                rc = -1;
            } else {
                char buf[4096];
                while (cardinal::fgets(buf, sizeof(buf), f) != nullptr) log += buf;
                int term = _pclose(f);
                rc = static_cast<i64>(term);
            }
        } else
#endif
        {
            rc = run_subprocess_capturing(compiler_, args, log);
        }

        if (rc != 0 || !file_exists(dll_path)) {
            finalize_(id, JobStatus::CompileFailed, rc, cardinal::move(log));
            return;
        }
        finalize_(id, JobStatus::LoadPending, rc, cardinal::move(log));
    }

    cardinal::vector<cardinal::string> build_compile_args_(const cardinal::string& src,
                                                 const cardinal::string& dll)
    {
        cardinal::vector<cardinal::string> a;
        const bool msvc = compiler_is_msvc_style(compiler_);
        if (msvc) {
            a.push_back("/nologo");
            a.push_back("/std:c++latest");
            a.push_back("/EHsc");
            a.push_back("/MD");
            a.push_back("/Zc:__cplusplus");
            a.push_back("/Zc:preprocessor");
            a.push_back("/W3");
            a.push_back("/utf-8");
            a.push_back("/D_HAS_EXCEPTIONS=1");
            for (auto& d : desc_.defines)      a.push_back("/D" + d);
            for (auto& i : desc_.include_dirs) a.push_back("/I" + i);
            for (auto& f : desc_.extra_flags)  a.push_back(f);
            a.push_back(src);
            a.push_back("/LD");
            a.push_back("/Fe:" + dll);
            // Drop intermediate .obj / .exp / .lib next to the DLL so the
            // cache_dir doesn't accumulate clutter elsewhere.
            const cardinal::string stem = (fs::path(dll).parent_path() / fs::path(dll).stem()).string();
            a.push_back("/Fo" + stem + ".obj");
            a.push_back("/link");
            a.push_back("/IMPLIB:" + stem + ".lib");
        } else {
            // POSIX-style (clang++ / g++).
            a.push_back("-std=c++23");
            a.push_back("-fPIC");
            a.push_back("-shared");
            a.push_back("-O2");
            a.push_back("-Wall");
            for (auto& d : desc_.defines)      a.push_back("-D" + d);
            for (auto& i : desc_.include_dirs) a.push_back("-I" + i);
            for (auto& f : desc_.extra_flags)  a.push_back(f);
            a.push_back(src);
            a.push_back("-o");
            a.push_back(dll);
        }
        return a;
    }

    void finalize_(u64 id, JobStatus status, i64 rc, cardinal::string log) {
        JobInfo snap;
        JobObserver cb_copy;
        {
            cardinal::lock_guard lk(records_mu_);
            auto it = records_.find(id);
            if (it == records_.end()) return;
            it->second.status      = status;
            it->second.exit_code   = rc;
            it->second.diagnostics = cardinal::move(log);
            it->second.elapsed_seconds = cardinal::chrono::duration<f64>(
                cardinal::chrono::steady_clock::now() - it->second.t_start).count();
            snap   = it->second.snapshot();
            cb_copy = observer_;
        }
        // Observer callback fires off-lock so handlers can call back into
        // the engine without deadlock. Note: this fires on the WORKER
        // thread for terminal CompileFailed; on the MAIN thread for
        // Loaded / LoadFailed (those are decided by attempt_load_main_thread_).
        if (cb_copy && status == JobStatus::CompileFailed) cb_copy(snap);
    }

    void attempt_load_main_thread_(u64 id) {
        cardinal::string dll_path; cardinal::string source_path;
        {
            cardinal::lock_guard lk(records_mu_);
            auto it = records_.find(id);
            if (it == records_.end()) return;
            if (it->second.status != JobStatus::LoadPending) return;
            dll_path    = it->second.dll_path;
            source_path = it->second.source_path;
        }
        // If a previous version of this script is loaded under a DIFFERENT
        // dll_path (because we embed the job id in the filename), unload it
        // first so two copies don't run simultaneously.
        {
            cardinal::lock_guard lk(records_mu_);
            for (auto& [oid, r] : records_) {
                if (oid == id) continue;
                if (r.source_path == source_path &&
                    r.status == JobStatus::Loaded && !r.dll_path.empty())
                {
                    cardinal::plugin::Registry::instance().unload(r.dll_path.c_str());
                    r.status = JobStatus::CompileFailed;   // mark superseded
                }
            }
        }
        const bool ok = cardinal::plugin::Registry::instance().load(dll_path.c_str());
        JobObserver cb_copy; JobInfo snap;
        {
            cardinal::lock_guard lk(records_mu_);
            auto it = records_.find(id);
            if (it == records_.end()) return;
            it->second.status = ok ? JobStatus::Loaded : JobStatus::LoadFailed;
            snap   = it->second.snapshot();
            cb_copy = observer_;
        }
        if (cb_copy) cb_copy(snap);
    }

    void service_watches_() {
        cardinal::vector<cardinal::string> reload_paths;
        {
            cardinal::lock_guard lk(watch_mu_);
            for (auto& w : watches_) {
                cardinal::error_code ec;
                if (!fs::exists(w.path, ec)) continue;
                auto m = fs::last_write_time(w.path, ec);
                if (ec) continue;
                if (m != w.last_mtime) {
                    w.last_mtime = m;
                    reload_paths.push_back(w.path);
                }
            }
        }
        for (const auto& p : reload_paths) {
            cardinal::log::infof("cppscript", "watch: %s changed → reload", p.c_str());
            reload(p.c_str());
        }
    }

    Desc                                desc_;
    cardinal::string                         compiler_;
    // MSVC-only — the .bat script we source in each child process to set
    // up INCLUDE / LIB / PATH for cl.exe before it runs. Empty when the
    // compiler isn't MSVC-style or VsDevCmd.bat couldn't be located.
    cardinal::string                         vsdevcmd_path_;
    cardinal::thread                         worker_;
    cardinal::mutex                          queue_mu_;
    cardinal::condition_variable             queue_cv_;
    cardinal::queue<u64>                     queue_;
    bool                                shutting_down_{false};

    mutable cardinal::mutex                  records_mu_;
    cardinal::unordered_map<u64, Record>     records_;
    cardinal::atomic<u64>                    next_job_id_{0};
    JobObserver                         observer_;

    struct WatchEntry {
        cardinal::string                     path;
        fs::file_time_type              last_mtime{};
    };
    cardinal::mutex                          watch_mu_;
    cardinal::vector<WatchEntry>             watches_;
};

cardinal::unique_ptr<Engine> Engine::create(const Desc& desc) {
    auto e = cardinal::make_unique<EngineImpl>(desc);
    if (!e->initialize()) return nullptr;
    return e;
}

}  // namespace cardinal::cppscript
