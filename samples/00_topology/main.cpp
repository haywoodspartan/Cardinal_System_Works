#include <cardinal/core/sync/topology.hpp>

#include <cstdio>

namespace ct = cardinal::topology;

namespace {

const char* vendor_name(ct::Vendor v) {
    switch (v) {
        case ct::Vendor::Intel:   return "Intel";
        case ct::Vendor::AMD:     return "AMD";
        case ct::Vendor::Unknown:
        default:                  return "Unknown";
    }
}

const char* class_glyph(ct::CoreClass c) {
    switch (c) {
        case ct::CoreClass::Performance: return "P";
        case ct::CoreClass::Efficiency:  return "E";
        case ct::CoreClass::Unknown:
        default:                         return "?";
    }
}

}  // namespace

int main() {
    auto info = ct::detect();

    std::printf("Vendor:          %s\n", vendor_name(info.vendor));
    std::printf("Brand:           %s\n", info.brand);
    std::printf("Family/Model:    %u/%u stepping %u\n",
                info.family, info.model, info.stepping);
    std::printf("Hybrid:          %s\n", info.is_hybrid ? "yes" : "no");
    std::printf("Logical cores:   %u\n", info.logical_core_count);
    std::printf("Physical cores:  %u\n", info.physical_core_count);
    std::printf("Clusters (L3):   %zu\n\n", info.clusters.size());

    for (const auto& c : info.clusters) {
        std::printf("  Cluster %u: %zu logical cores, L3 = %.1f MiB%s\n",
                    c.id,
                    c.logical_cores.size(),
                    static_cast<double>(c.l3_size_bytes) / (1024.0 * 1024.0),
                    c.has_v_cache ? "  [3D V-Cache]" : "");
        std::printf("    cores:");
        for (auto lp : c.logical_cores) std::printf(" %u", lp);
        std::printf("\n");
    }

    std::printf("\nPer-core class (P/E/?):\n");
    for (const auto& lc : info.logical_cores) {
        std::printf("  cpu%-3u  phys=%-3u  cluster=%-2u  numa=%-2u  class=%s\n",
                    lc.os_id, lc.physical_core_id, lc.cluster_id, lc.numa_node,
                    class_glyph(lc.core_class));
    }

    auto vc = ct::v_cache_cluster_index(info);
    if (vc >= 0) {
        std::printf("\n3D V-Cache cluster: %d  (pin perf-critical workers here)\n", vc);
    }

    return 0;
}
