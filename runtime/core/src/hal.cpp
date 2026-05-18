#include <cardinal/core/hal.hpp>
#include <cardinal/core/platform.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

#if CARDINAL_PLATFORM_WINDOWS
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <Windows.h>
    #include <Psapi.h>
    #include <intrin.h>
    #include <Lmcons.h>
    #pragma comment(lib, "psapi.lib")
#elif CARDINAL_PLATFORM_LINUX
    #include <dlfcn.h>
    #include <pthread.h>
    #include <pwd.h>
    #include <sys/utsname.h>
    #include <time.h>
    #include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace cardinal::hal {

// ---------------------------------------------------------------------------
// File
// ---------------------------------------------------------------------------
namespace {

class StdFile final : public File {
public:
    explicit StdFile(FILE* f) : f_(f) {}
    ~StdFile() override { if (f_) std::fclose(f_); }

    u64  size() const override {
        if (!f_) return 0;
        const auto cur = std::ftell(f_);
        std::fseek(f_, 0, SEEK_END);
        const u64 n = static_cast<u64>(std::ftell(f_));
        std::fseek(f_, cur, SEEK_SET);
        return n;
    }
    u64  position() const override { return f_ ? static_cast<u64>(std::ftell(f_)) : 0; }
    void seek(u64 offset) override { if (f_) std::fseek(f_, static_cast<long>(offset), SEEK_SET); }
    usize read (void* buffer, usize bytes) override {
        return f_ ? std::fread(buffer, 1, bytes, f_) : 0;
    }
    usize write(const void* buffer, usize bytes) override {
        return f_ ? std::fwrite(buffer, 1, bytes, f_) : 0;
    }
    void flush() override { if (f_) std::fflush(f_); }
    bool good() const override { return f_ != nullptr && !std::ferror(f_); }
private:
    FILE* f_;
};

}  // namespace

std::unique_ptr<File> File::open(const std::string& path, FileMode mode) {
    const char* m = "rb";
    switch (mode) {
        case FileMode::Read:      m = "rb";  break;
        case FileMode::Write:     m = "wb";  break;
        case FileMode::ReadWrite: m = "r+b"; break;
        case FileMode::Append:    m = "ab";  break;
    }
    FILE* f = std::fopen(path.c_str(), m);
    if (f == nullptr) return nullptr;
    return std::unique_ptr<File>(new StdFile(f));
}

bool read_all(const std::string& path, std::vector<u8>& out) {
    auto f = File::open(path, FileMode::Read);
    if (!f) return false;
    const u64 n = f->size();
    out.resize(static_cast<usize>(n));
    return f->read(out.data(), out.size()) == out.size();
}
bool write_all(const std::string& path, const void* bytes, usize n) {
    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    auto f = File::open(path, FileMode::Write);
    if (!f) return false;
    return f->write(bytes, n) == n;
}

bool exists(const std::string& path)       { std::error_code ec; return fs::exists(path, ec); }
bool is_directory(const std::string& path) { std::error_code ec; return fs::is_directory(path, ec); }
bool create_dirs(const std::string& path)  { std::error_code ec; return fs::create_directories(path, ec) || !ec; }
bool remove_path(const std::string& path)  { std::error_code ec; return fs::remove_all(path, ec) > 0; }
u64  file_size(const std::string& path)    { std::error_code ec; return static_cast<u64>(fs::file_size(path, ec)); }
u64  file_mtime_ns(const std::string& path) {
    std::error_code ec;
    const auto t = fs::last_write_time(path, ec);
    if (ec) return 0;
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        t.time_since_epoch()).count());
}
std::vector<std::string> list_dir(const std::string& path, bool recursive) {
    std::vector<std::string> r;
    std::error_code ec;
    if (!fs::exists(path, ec)) return r;
    if (recursive) {
        for (auto& e : fs::recursive_directory_iterator(path, ec)) {
            if (ec) break; if (!e.is_regular_file()) continue;
            r.push_back(e.path().string());
        }
    } else {
        for (auto& e : fs::directory_iterator(path, ec)) {
            if (ec) break; if (!e.is_regular_file()) continue;
            r.push_back(e.path().string());
        }
    }
    return r;
}

// ---------------------------------------------------------------------------
// Dynamic library
// ---------------------------------------------------------------------------
namespace {
class DylibImpl final : public DynamicLibrary {
public:
    DylibImpl(void* h, std::string p) : handle_(h), path_(std::move(p)) {}
    ~DylibImpl() override {
#if CARDINAL_PLATFORM_WINDOWS
        if (handle_) ::FreeLibrary(reinterpret_cast<HMODULE>(handle_));
#elif CARDINAL_PLATFORM_LINUX
        if (handle_) dlclose(handle_);
#endif
    }
    void* lookup(const std::string& sym) override {
#if CARDINAL_PLATFORM_WINDOWS
        return reinterpret_cast<void*>(::GetProcAddress(
            reinterpret_cast<HMODULE>(handle_), sym.c_str()));
#elif CARDINAL_PLATFORM_LINUX
        return dlsym(handle_, sym.c_str());
#else
        (void)sym; return nullptr;
#endif
    }
    const std::string& path() const override { return path_; }
private:
    void*       handle_;
    std::string path_;
};
}  // namespace

std::unique_ptr<DynamicLibrary> DynamicLibrary::load(const std::string& path,
                                                     std::string* error_out)
{
    void* h = nullptr;
#if CARDINAL_PLATFORM_WINDOWS
    h = reinterpret_cast<void*>(::LoadLibraryA(path.c_str()));
    if (!h) {
        if (error_out) {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "LoadLibraryA failed (%lu)",
                          static_cast<unsigned long>(::GetLastError()));
            *error_out = buf;
        }
        return nullptr;
    }
#elif CARDINAL_PLATFORM_LINUX
    h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        if (error_out) *error_out = dlerror();
        return nullptr;
    }
#else
    if (error_out) *error_out = "no dylib loader for this platform";
    return nullptr;
#endif
    return std::unique_ptr<DynamicLibrary>(new DylibImpl(h, path));
}

// ---------------------------------------------------------------------------
// OsInfo
// ---------------------------------------------------------------------------
namespace {

OsInfo build_os_info() {
    OsInfo o{};
#if CARDINAL_PLATFORM_WINDOWS
    o.name = "Windows";
    SYSTEM_INFO si{};
    ::GetSystemInfo(&si);
    o.page_size_bytes    = si.dwPageSize;
    o.logical_cpu_count  = si.dwNumberOfProcessors;
    DWORD_PTR proc_mask = 0, sys_mask = 0;
    ::GetProcessAffinityMask(::GetCurrentProcess(), &proc_mask, &sys_mask);
    // Approximation: count bits in proc_mask. Better: GetLogicalProcessorInformation.
    u32 phys = 0;
    for (DWORD_PTR m = proc_mask; m; m >>= 1) phys += static_cast<u32>(m & 1);
    o.physical_cpu_count = phys ? phys : o.logical_cpu_count;
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (::GlobalMemoryStatusEx(&ms)) o.total_ram_bytes = ms.ullTotalPhys;
    char user[UNLEN + 1]{};
    DWORD usz = sizeof(user);
    if (::GetUserNameA(user, &usz)) o.user_name = user;
    char host[256]{};
    DWORD hsz = sizeof(host);
    if (::GetComputerNameA(host, &hsz)) o.host_name = host;
    o.kernel = "NT";
    o.locale = "en_US";
#elif CARDINAL_PLATFORM_LINUX
    utsname u{};
    if (uname(&u) == 0) {
        o.name   = u.sysname;
        o.kernel = u.release;
        o.host_name = u.nodename;
    }
    o.page_size_bytes   = static_cast<u32>(sysconf(_SC_PAGESIZE));
    o.logical_cpu_count = static_cast<u32>(sysconf(_SC_NPROCESSORS_ONLN));
    o.physical_cpu_count= o.logical_cpu_count;   // can refine via /proc/cpuinfo
    if (auto* pw = getpwuid(geteuid())) o.user_name = pw->pw_name;
    o.locale = "en_US";
#endif
    return o;
}

}  // namespace

const OsInfo& os_info() {
    // Built once at first call, read by every subsystem that wants OS
    // info — page size, CPU counts, total RAM. Read-mostly.
    CARDINAL_READ_MOSTLY static const OsInfo info = build_os_info();
    return info;
}

// ---------------------------------------------------------------------------
// Mono clock + thread name
// ---------------------------------------------------------------------------
u64 mono_now_ns() noexcept {
#if CARDINAL_PLATFORM_WINDOWS
    static LARGE_INTEGER freq = []{ LARGE_INTEGER f; ::QueryPerformanceFrequency(&f); return f; }();
    LARGE_INTEGER c;
    ::QueryPerformanceCounter(&c);
    return static_cast<u64>((c.QuadPart * 1'000'000'000ull) / freq.QuadPart);
#elif CARDINAL_PLATFORM_LINUX
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<u64>(ts.tv_sec) * 1'000'000'000ull +
           static_cast<u64>(ts.tv_nsec);
#else
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
#endif
}

void set_current_thread_name(const char* name) noexcept {
    if (name == nullptr) return;
#if CARDINAL_PLATFORM_WINDOWS
    // SetThreadDescription wants a wide string; we stick to <256 chars so
    // the conversion is cheap.
    wchar_t wbuf[128]{};
    MultiByteToWideChar(CP_UTF8, 0, name, -1, wbuf, 127);
    using SetThreadDescriptionPtr = HRESULT (WINAPI*)(HANDLE, PCWSTR);
    static auto fn = []() {
        HMODULE k = ::GetModuleHandleW(L"kernel32.dll");
        return k ? reinterpret_cast<SetThreadDescriptionPtr>(
            ::GetProcAddress(k, "SetThreadDescription")) : nullptr;
    }();
    if (fn) fn(::GetCurrentThread(), wbuf);
#elif CARDINAL_PLATFORM_LINUX
    pthread_setname_np(pthread_self(), name);
#endif
}

// ---------------------------------------------------------------------------
// CPU features
// ---------------------------------------------------------------------------
namespace {

CpuFeatures detect_cpu_features() {
    CpuFeatures c{};
#if CARDINAL_PLATFORM_WINDOWS && (defined(_M_X64) || defined(_M_IX86))
    int regs[4]{};
    __cpuid(regs, 0);
    int max_id = regs[0];
    std::memcpy(c.vendor + 0, &regs[1], 4);
    std::memcpy(c.vendor + 4, &regs[3], 4);
    std::memcpy(c.vendor + 8, &regs[2], 4);
    c.vendor[12] = 0;

    if (max_id >= 1) {
        __cpuid(regs, 1);
        c.sse    = (regs[3] & (1 << 25)) != 0;
        c.sse2   = (regs[3] & (1 << 26)) != 0;
        c.sse3   = (regs[2] & (1 << 0))  != 0;
        c.ssse3  = (regs[2] & (1 << 9))  != 0;
        c.sse41  = (regs[2] & (1 << 19)) != 0;
        c.sse42  = (regs[2] & (1 << 20)) != 0;
        c.avx    = (regs[2] & (1 << 28)) != 0;
        c.fma3   = (regs[2] & (1 << 12)) != 0;
        c.aes    = (regs[2] & (1 << 25)) != 0;
        c.f16c   = (regs[2] & (1 << 29)) != 0;
        c.popcnt = (regs[2] & (1 << 23)) != 0;
    }
    if (max_id >= 7) {
        __cpuidex(regs, 7, 0);
        c.bmi1     = (regs[1] & (1 << 3))  != 0;
        c.avx2     = (regs[1] & (1 << 5))  != 0;
        c.bmi2     = (regs[1] & (1 << 8))  != 0;
        c.avx512f  = (regs[1] & (1 << 16)) != 0;
        c.avx512dq = (regs[1] & (1 << 17)) != 0;
        c.avx512bw = (regs[1] & (1 << 30)) != 0;
        c.avx512vl = (regs[1] & (1 << 31)) != 0;
    }
    __cpuid(regs, 0x80000000);
    const u32 max_ext = static_cast<u32>(regs[0]);
    if (max_ext >= 0x80000001u) {
        // Extended feature flags — SSE4A (AMD), LZCNT, FMA4 live here.
        // ECX bit layout per AMD/Intel manuals:
        //   bit 5  = LZCNT
        //   bit 6  = SSE4A           (AMD only)
        //   bit 11 = XOP             (AMD only — not exposed)
        //   bit 16 = FMA4            (AMD only — withdrawn from Zen 3+)
        __cpuid(regs, 0x80000001);
        c.lzcnt = (regs[2] & (1 << 5))  != 0;
        c.sse4a = (regs[2] & (1 << 6))  != 0;
        c.fma4  = (regs[2] & (1 << 16)) != 0;
    }
    if (max_ext >= 0x80000004u) {
        for (u32 i = 0; i < 3; ++i) {
            __cpuid(regs, 0x80000002 + i);
            std::memcpy(c.brand + i * 16, regs, 16);
        }
        c.brand[63] = 0;
    }
#endif
    return c;
}

}  // namespace

const CpuFeatures& cpu_features() {
    // Detected once via cpuid; queried by RHI + shader picker + SIMD
    // dispatchers on every fast-vs-slow code-path choice.
    CARDINAL_READ_MOSTLY static const CpuFeatures cf = detect_cpu_features();
    return cf;
}

// ---------------------------------------------------------------------------
// Process memory
// ---------------------------------------------------------------------------
ProcessMemSnapshot process_memory() {
    ProcessMemSnapshot s{};
#if CARDINAL_PLATFORM_WINDOWS
    PROCESS_MEMORY_COUNTERS_EX c{};
    c.cb = sizeof(c);
    if (::GetProcessMemoryInfo(::GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&c), sizeof(c))) {
        s.working_set_bytes      = static_cast<u64>(c.WorkingSetSize);
        s.peak_working_set_bytes = static_cast<u64>(c.PeakWorkingSetSize);
        s.private_bytes          = static_cast<u64>(c.PrivateUsage);
        s.virtual_bytes          = static_cast<u64>(c.PagefileUsage);
    }
#elif CARDINAL_PLATFORM_LINUX
    FILE* f = std::fopen("/proc/self/status", "r");
    if (f) {
        char line[256];
        while (std::fgets(line, sizeof(line), f)) {
            unsigned long long kb = 0;
            if (std::sscanf(line, "VmRSS: %llu", &kb) == 1)  s.working_set_bytes = kb * 1024;
            else if (std::sscanf(line, "VmHWM: %llu", &kb) == 1) s.peak_working_set_bytes = kb * 1024;
            else if (std::sscanf(line, "VmData: %llu", &kb) == 1) s.private_bytes = kb * 1024;
            else if (std::sscanf(line, "VmSize: %llu", &kb) == 1) s.virtual_bytes = kb * 1024;
        }
        std::fclose(f);
    }
#endif
    return s;
}

}  // namespace cardinal::hal
