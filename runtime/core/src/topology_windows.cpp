#include <cardinal/core/platform.hpp>

#if CARDINAL_PLATFORM_WINDOWS

#include <cardinal/core/topology.hpp>

#include <Windows.h>

#include <cstddef>
#include <cstring>
#include <vector>

namespace cardinal::topology::platform_detail {

namespace {

// GLPIEx returns a packed buffer of variable-sized records — walk it manually.
template <typename Visitor>
void for_each_glpiex(LOGICAL_PROCESSOR_RELATIONSHIP rel, Visitor&& visitor) {
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(rel, nullptr, &bytes);
    if (bytes == 0) return;

    std::vector<std::byte> buffer(bytes);
    auto* head = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
    if (!GetLogicalProcessorInformationEx(rel, head, &bytes)) return;

    auto* p   = reinterpret_cast<std::byte*>(head);
    auto* end = p + bytes;
    while (p < end) {
        auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(p);
        visitor(*info);
        p += info->Size;
    }
}

void collect_logical_ids(const GROUP_AFFINITY& aff, std::vector<u32>& out) {
    KAFFINITY mask = aff.Mask;
    u32 base = static_cast<u32>(aff.Group) * 64u;  // up to 64 LPs per processor group
    while (mask != 0) {
        unsigned long bit = 0;
        _BitScanForward64(&bit, mask);
        out.push_back(base + static_cast<u32>(bit));
        mask &= mask - 1;
    }
}

}  // namespace

void query_topology(CpuInfo& info) {
    SYSTEM_INFO sys{};
    GetNativeSystemInfo(&sys);
    info.logical_core_count = static_cast<u32>(sys.dwNumberOfProcessors);
    info.logical_cores.assign(info.logical_core_count, LogicalCore{});
    for (u32 i = 0; i < info.logical_core_count; ++i) {
        info.logical_cores[i].os_id = i;
    }

    // Physical cores — each Core record maps a set of LPs to one physical core.
    u32 physical_index = 0;
    for_each_glpiex(RelationProcessorCore, [&](const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX& r) {
        std::vector<u32> ids;
        for (WORD i = 0; i < r.Processor.GroupCount; ++i) {
            collect_logical_ids(r.Processor.GroupMask[i], ids);
        }
        for (u32 lp : ids) {
            if (lp < info.logical_core_count) {
                info.logical_cores[lp].physical_core_id = physical_index;
                if (info.is_hybrid) {
                    // Intel hybrid: EfficiencyClass 0 = E-core, 1 = P-core.
                    info.logical_cores[lp].core_class =
                        (r.Processor.EfficiencyClass >= 1) ? CoreClass::Performance : CoreClass::Efficiency;
                }
            }
        }
        ++physical_index;
    });
    info.physical_core_count = physical_index;

    // NUMA nodes
    for_each_glpiex(RelationNumaNode, [&](const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX& r) {
        std::vector<u32> ids;
        collect_logical_ids(r.NumaNode.GroupMask, ids);
        for (u32 lp : ids) {
            if (lp < info.logical_core_count) {
                info.logical_cores[lp].numa_node = r.NumaNode.NodeNumber;
            }
        }
    });

    // L3 caches define our cluster (CCD/CCX) grouping.
    u32 cluster_index = 0;
    for_each_glpiex(RelationCache, [&](const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX& r) {
        if (r.Cache.Level != 3 || r.Cache.Type == CacheInstruction) return;

        Cluster c{};
        c.id = cluster_index;
        c.l3_size_bytes = r.Cache.CacheSize;

        if (r.Cache.GroupCount == 0) {
            // Pre-Win10 record format — single GroupMask in the union.
            collect_logical_ids(r.Cache.GroupMask, c.logical_cores);
        } else {
            for (WORD i = 0; i < r.Cache.GroupCount; ++i) {
                collect_logical_ids(r.Cache.GroupMasks[i], c.logical_cores);
            }
        }

        for (u32 lp : c.logical_cores) {
            if (lp < info.logical_core_count) {
                info.logical_cores[lp].cluster_id = cluster_index;
            }
        }
        info.clusters.push_back(std::move(c));
        ++cluster_index;
    });
}

}  // namespace cardinal::topology::platform_detail

#endif  // CARDINAL_PLATFORM_WINDOWS
