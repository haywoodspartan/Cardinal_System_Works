#pragma once

#include <cardinal/core/types.hpp>

#include <vector>

namespace cardinal::topology {

enum class Vendor : u32 {
    Unknown = 0,
    Intel,
    AMD,
};

enum class CoreClass : u8 {
    Unknown = 0,
    Performance,  // P-core (Intel hybrid) or standard core (AMD / non-hybrid)
    Efficiency,   // E-core (Intel hybrid)
};

struct LogicalCore {
    u32 os_id;             // OS-assigned logical-processor index
    u32 physical_core_id;  // Index into the de-duplicated physical-core list
    u32 cluster_id;        // CCD/CCX index — cores sharing one L3
    u32 numa_node;
    CoreClass core_class;
};

struct Cluster {
    u32 id;
    std::vector<u32> logical_cores;  // OS IDs of cores sharing this L3
    u32 l3_size_bytes;
    bool has_v_cache;                // AMD 3D V-Cache (heuristic — see implementation)
};

struct CpuInfo {
    Vendor vendor;
    char brand[64];
    u32 family;
    u32 model;
    u32 stepping;
    bool is_hybrid;        // Intel hybrid (P+E)
    u32 logical_core_count;
    u32 physical_core_count;
    std::vector<LogicalCore> logical_cores;
    std::vector<Cluster> clusters;
};

// Detect host CPU topology. One-shot init call — do not put in hot paths.
CpuInfo detect();

// Returns the cluster index flagged with 3D V-Cache, or -1 if none detected.
i32 v_cache_cluster_index(const CpuInfo& info);

}  // namespace cardinal::topology
