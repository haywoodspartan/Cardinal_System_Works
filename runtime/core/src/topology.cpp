#include <cardinal/core/topology.hpp>

#include <cardinal/core/platform.hpp>

#include <algorithm>
#include <cstring>
#include <limits>

#if CARDINAL_ARCH_X64
    #if CARDINAL_COMPILER_MSVC
        #include <intrin.h>
    #else
        #include <cpuid.h>
    #endif
#endif

namespace cardinal::topology {

namespace platform_detail {
// Implemented in topology_windows.cpp / topology_linux.cpp.
// Fills cores, NUMA, clusters, and (where available) hybrid core_class.
void query_topology(CpuInfo& info);
}  // namespace platform_detail

namespace {

#if CARDINAL_ARCH_X64
struct CpuidRegs {
    u32 eax, ebx, ecx, edx;
};

CpuidRegs cpuid_call(u32 leaf, u32 subleaf = 0) {
    CpuidRegs r{};
#if CARDINAL_COMPILER_MSVC
    int regs[4]{};
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    r.eax = static_cast<u32>(regs[0]);
    r.ebx = static_cast<u32>(regs[1]);
    r.ecx = static_cast<u32>(regs[2]);
    r.edx = static_cast<u32>(regs[3]);
#else
    __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
#endif
    return r;
}

Vendor read_vendor() {
    auto r = cpuid_call(0);
    char vendor[13]{};
    std::memcpy(&vendor[0], &r.ebx, 4);
    std::memcpy(&vendor[4], &r.edx, 4);
    std::memcpy(&vendor[8], &r.ecx, 4);
    if (std::memcmp(vendor, "GenuineIntel", 12) == 0) return Vendor::Intel;
    if (std::memcmp(vendor, "AuthenticAMD", 12) == 0) return Vendor::AMD;
    return Vendor::Unknown;
}

void read_brand(char out[64]) {
    auto max_ext = cpuid_call(0x80000000).eax;
    if (max_ext < 0x80000004) {
        std::memcpy(out, "Unknown", 8);
        return;
    }
    u32 buf[12]{};
    for (u32 i = 0; i < 3; ++i) {
        auto r = cpuid_call(0x80000002 + i);
        buf[i * 4 + 0] = r.eax;
        buf[i * 4 + 1] = r.ebx;
        buf[i * 4 + 2] = r.ecx;
        buf[i * 4 + 3] = r.edx;
    }
    std::memcpy(out, buf, 48);
    out[48] = '\0';
    auto len = std::strlen(out);
    while (len > 0 && (out[len - 1] == ' ' || out[len - 1] == '\0')) {
        out[--len] = '\0';
    }
}

void read_family_model(u32& family, u32& model, u32& stepping) {
    auto r = cpuid_call(1);
    u32 base_family = (r.eax >> 8) & 0xFu;
    u32 base_model  = (r.eax >> 4) & 0xFu;
    u32 ext_family  = (r.eax >> 20) & 0xFFu;
    u32 ext_model   = (r.eax >> 16) & 0xFu;
    family   = base_family + ((base_family == 0xFu) ? ext_family : 0u);
    model    = base_model | (((base_family == 0xFu) || (base_family == 0x6u)) ? (ext_model << 4) : 0u);
    stepping = r.eax & 0xFu;
}

bool detect_hybrid_intel() {
    // Leaf 7, sub 0, EDX bit 15 = Hybrid
    auto r = cpuid_call(7, 0);
    return (r.edx & (1u << 15)) != 0;
}
#endif  // CARDINAL_ARCH_X64

void apply_v_cache_heuristic(CpuInfo& info) {
    // AMD 3D V-Cache adds ~64MB on one CCD, leaving the other(s) at standard ~32MB.
    // Detection rules:
    //   1) Multi-cluster system with 2:1+ disparity → the larger L3 cluster has V-Cache.
    //   2) Single-CCD V-Cache (e.g. 7800X3D) → only one cluster, but L3 ≥ 64MiB.
    if (info.vendor != Vendor::AMD || info.clusters.empty()) return;

    u32 max_l3 = 0;
    u32 min_l3 = std::numeric_limits<u32>::max();
    for (const auto& c : info.clusters) {
        max_l3 = std::max(max_l3, c.l3_size_bytes);
        min_l3 = std::min(min_l3, c.l3_size_bytes);
    }

    constexpr u32 v_cache_threshold = 64u * 1024u * 1024u;  // 64 MiB
    bool dual_disparity = info.clusters.size() >= 2 && min_l3 > 0 && max_l3 >= 2u * min_l3;
    bool single_oversize = info.clusters.size() == 1 && max_l3 >= v_cache_threshold;

    if (dual_disparity || single_oversize) {
        for (auto& c : info.clusters) {
            c.has_v_cache = (c.l3_size_bytes == max_l3);
        }
    }
}

}  // namespace

CpuInfo detect() {
    CpuInfo info{};

#if CARDINAL_ARCH_X64
    info.vendor = read_vendor();
    read_brand(info.brand);
    read_family_model(info.family, info.model, info.stepping);
    if (info.vendor == Vendor::Intel) {
        info.is_hybrid = detect_hybrid_intel();
    }
#else
    std::memcpy(info.brand, "Unknown", 8);
#endif

    platform_detail::query_topology(info);
    apply_v_cache_heuristic(info);

    // For non-hybrid systems, every core is "performance" by default.
    if (!info.is_hybrid) {
        for (auto& lc : info.logical_cores) {
            if (lc.core_class == CoreClass::Unknown) lc.core_class = CoreClass::Performance;
        }
    }

    return info;
}

i32 v_cache_cluster_index(const CpuInfo& info) {
    for (usize i = 0; i < info.clusters.size(); ++i) {
        if (info.clusters[i].has_v_cache) return static_cast<i32>(i);
    }
    return -1;
}

}  // namespace cardinal::topology
