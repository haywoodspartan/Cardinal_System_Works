#include <cardinal/core/memory.hpp>
#include <cardinal/core/platform.hpp>

#if CARDINAL_PLATFORM_WINDOWS

#ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <Windows.h>
#include <Psapi.h>

#pragma comment(lib, "psapi.lib")

namespace cardinal::memory {

SystemSnapshot query_system() noexcept {
    SystemSnapshot s{};
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof(m);
    if (GlobalMemoryStatusEx(&m)) {
        s.total_bytes     = static_cast<u64>(m.ullTotalPhys);
        s.available_bytes = static_cast<u64>(m.ullAvailPhys);
        // dwMemoryLoad is the OS's own number — same calc, but rounded to %.
        // We compute it ourselves for higher resolution.
        if (s.total_bytes > 0) {
            const double avail = static_cast<double>(s.available_bytes);
            const double total = static_cast<double>(s.total_bytes);
            s.load_percent = 100.0 * (1.0 - (avail / total));
            if (s.load_percent < 0.0)   s.load_percent = 0.0;
            if (s.load_percent > 100.0) s.load_percent = 100.0;
        }
    }
    return s;
}

ProcessSnapshot query_process() noexcept {
    ProcessSnapshot p{};
    PROCESS_MEMORY_COUNTERS_EX c{};
    c.cb = sizeof(c);
    if (GetProcessMemoryInfo(GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&c), sizeof(c))) {
        p.working_set_bytes      = static_cast<u64>(c.WorkingSetSize);
        p.peak_working_set_bytes = static_cast<u64>(c.PeakWorkingSetSize);
        p.private_bytes          = static_cast<u64>(c.PrivateUsage);
    }
    return p;
}

}  // namespace cardinal::memory

#endif  // CARDINAL_PLATFORM_WINDOWS
