#pragma once

// =============================================================================
// Cardinal — Hardware Abstraction Layer.
//
// Cross-platform primitives that don't fit elsewhere: file I/O, dynamic
// library loading, OS info, monotonic clocks, CPU feature detection,
// thread naming, working-set helpers.
//
// Goal: any subsystem that needs to talk to the OS goes through here, so
// porting to a new platform means re-implementing one well-defined API,
// not chasing #ifdefs across the entire codebase.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cardinal::hal {

// ---------------------------------------------------------------------------
// File I/O — synchronous baseline. cardinal::io adds the async dispatcher
// layer on top of this.
// ---------------------------------------------------------------------------
enum class FileMode : u32 { Read, Write, ReadWrite, Append };

class File {
public:
    static std::unique_ptr<File> open(const std::string& path, FileMode mode);
    virtual ~File() = default;
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    virtual u64  size() const = 0;
    virtual u64  position() const = 0;
    virtual void seek(u64 offset) = 0;
    // Returns bytes actually read; 0 indicates EOF or error.
    virtual usize read (void* buffer, usize bytes) = 0;
    virtual usize write(const void* buffer, usize bytes) = 0;
    virtual void  flush() = 0;
    virtual bool  good() const = 0;

protected:
    File() = default;
};

// Convenience.
bool read_all (const std::string& path, std::vector<u8>& out);
bool write_all(const std::string& path, const void* bytes, usize n);

// File-system queries.
bool        exists       (const std::string& path);
bool        is_directory (const std::string& path);
bool        create_dirs  (const std::string& path);
bool        remove_path  (const std::string& path);
u64         file_size    (const std::string& path);
u64         file_mtime_ns(const std::string& path);
std::vector<std::string> list_dir(const std::string& path, bool recursive = false);

// ---------------------------------------------------------------------------
// Dynamic library loading.
// ---------------------------------------------------------------------------
class DynamicLibrary {
public:
    static std::unique_ptr<DynamicLibrary> load(const std::string& path,
                                                 std::string* error_out = nullptr);
    virtual ~DynamicLibrary() = default;
    virtual void* lookup(const std::string& symbol) = 0;
    virtual const std::string& path() const = 0;

protected:
    DynamicLibrary() = default;
};

// ---------------------------------------------------------------------------
// OS info — strings stay valid for the process lifetime.
// ---------------------------------------------------------------------------
struct OsInfo {
    std::string name;        // "Windows 11", "Ubuntu 22.04"
    std::string kernel;      // version string
    std::string user_name;
    std::string host_name;
    std::string locale;
    u32         page_size_bytes{4096};
    u32         logical_cpu_count{1};
    u32         physical_cpu_count{1};
    u64         total_ram_bytes{0};
};

const OsInfo& os_info();

// ---------------------------------------------------------------------------
// Monotonic clock — nanosecond resolution. Wraps QPC on Windows + clock_gettime
// on Linux. Cheaper than chrono::steady_clock::now() in tight loops.
// ---------------------------------------------------------------------------
u64 mono_now_ns() noexcept;
inline f64 mono_now_seconds() noexcept { return static_cast<f64>(mono_now_ns()) * 1e-9; }

// ---------------------------------------------------------------------------
// Thread naming — surfaces in profilers (Visual Studio, Tracy, perf).
// ---------------------------------------------------------------------------
void set_current_thread_name(const char* name) noexcept;

// ---------------------------------------------------------------------------
// CPU features — superset of the cpuid bits the rest of the engine asks
// about. Detected once at first call.
// ---------------------------------------------------------------------------
struct CpuFeatures {
    bool sse{false}, sse2{false}, sse3{false}, ssse3{false}, sse41{false}, sse42{false};
    bool sse4a{false};            // AMD-specific extras (LZCNT-via-SSE4A, EXTRQ, INSERTQ)
    bool avx{false}, avx2{false};
    // AVX-512 family — 512-bit ops only when avx512f, masked elementwise
    // when avx512bw/dq adds the byte/word and 64-bit-int paths used by
    // some int16/int8 SIMD loops (textures, ECS bitmasks).
    bool avx512f{false}, avx512vl{false}, avx512bw{false}, avx512dq{false};
    bool fma3{false}, fma4{false};
    bool bmi1{false}, bmi2{false};
    bool popcnt{false}, lzcnt{false};
    bool aes{false};
    bool f16c{false};
    bool aarch64_neon{false}, aarch64_sve{false};
    char vendor[16]{};
    char brand[64]{};
};
const CpuFeatures& cpu_features();

// ---------------------------------------------------------------------------
// Process working set — used by the budget broker today via
// memory.cpp's separate path; HAL exposes a unified view too.
// ---------------------------------------------------------------------------
struct ProcessMemSnapshot {
    u64 working_set_bytes{0};
    u64 peak_working_set_bytes{0};
    u64 private_bytes{0};
    u64 virtual_bytes{0};
};
ProcessMemSnapshot process_memory();

}  // namespace cardinal::hal
