#include <cardinal/core/memory.hpp>
#include <cardinal/core/platform.hpp>

#if CARDINAL_PLATFORM_LINUX

#include <cstdio>
#include <cstring>
#include <unistd.h>

namespace cardinal::memory {

namespace {

// Parse a /proc file looking for a "Key: <number> kB" line.
// Returns 0 if not found.
u64 read_kb_field(const char* path, const char* key) noexcept {
    FILE* f = std::fopen(path, "r");
    if (f == nullptr) return 0;
    char line[256];
    const usize klen = std::strlen(key);
    u64 value_bytes = 0;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        if (std::strncmp(line, key, klen) == 0 && line[klen] == ':') {
            unsigned long long kb = 0;
            // "Key:    12345 kB"
            std::sscanf(line + klen + 1, "%llu", &kb);
            value_bytes = static_cast<u64>(kb) * 1024ull;
            break;
        }
    }
    std::fclose(f);
    return value_bytes;
}

}  // namespace

SystemSnapshot query_system() noexcept {
    SystemSnapshot s{};
    s.total_bytes     = read_kb_field("/proc/meminfo", "MemTotal");
    s.available_bytes = read_kb_field("/proc/meminfo", "MemAvailable");
    if (s.available_bytes == 0) {
        // Pre-3.14 fallback. MemFree underestimates but is always present.
        s.available_bytes = read_kb_field("/proc/meminfo", "MemFree");
    }
    if (s.total_bytes > 0) {
        const double avail = static_cast<double>(s.available_bytes);
        const double total = static_cast<double>(s.total_bytes);
        s.load_percent = 100.0 * (1.0 - (avail / total));
        if (s.load_percent < 0.0)   s.load_percent = 0.0;
        if (s.load_percent > 100.0) s.load_percent = 100.0;
    }
    return s;
}

ProcessSnapshot query_process() noexcept {
    ProcessSnapshot p{};
    p.working_set_bytes = read_kb_field("/proc/self/status", "VmRSS");
    p.peak_working_set_bytes = read_kb_field("/proc/self/status", "VmHWM");
    p.private_bytes     = read_kb_field("/proc/self/status", "VmData");
    return p;
}

}  // namespace cardinal::memory

#endif  // CARDINAL_PLATFORM_LINUX
