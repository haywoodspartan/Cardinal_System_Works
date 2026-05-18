#include <cardinal/core/platform.hpp>

#if CARDINAL_PLATFORM_LINUX

#include <cardinal/core/topology.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace cardinal::topology::platform_detail {

namespace fs = std::filesystem;

namespace {

std::string read_trimmed(const fs::path& p) {
    std::ifstream f(p);
    if (!f) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    auto s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

bool is_digits(std::string_view s) {
    if (s.empty()) return false;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
    }
    return true;
}

// "0-3,5,7-9" → {0,1,2,3,5,7,8,9}
std::vector<u32> parse_cpulist(const std::string& s) {
    std::vector<u32> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        auto dash = tok.find('-');
        try {
            if (dash == std::string::npos) {
                out.push_back(static_cast<u32>(std::stoul(tok)));
            } else {
                u32 a = static_cast<u32>(std::stoul(tok.substr(0, dash)));
                u32 b = static_cast<u32>(std::stoul(tok.substr(dash + 1)));
                for (u32 i = a; i <= b; ++i) out.push_back(i);
            }
        } catch (...) {
            // ignore malformed entries
        }
    }
    return out;
}

// "32M" / "32768K" / "1024" → bytes
u32 parse_size_kmg(const std::string& s) {
    if (s.empty()) return 0;
    char unit = s.back();
    u64 mult = 1;
    std::string num = s;
    if (unit == 'K' || unit == 'k') { mult = 1024ULL;                       num.pop_back(); }
    else if (unit == 'M' || unit == 'm') { mult = 1024ULL * 1024;            num.pop_back(); }
    else if (unit == 'G' || unit == 'g') { mult = 1024ULL * 1024 * 1024;     num.pop_back(); }
    try {
        return static_cast<u32>(std::stoul(num) * mult);
    } catch (...) {
        return 0;
    }
}

u32 read_uint(const fs::path& p) {
    auto s = read_trimmed(p);
    if (s.empty()) return 0;
    try {
        return static_cast<u32>(std::stoul(s));
    } catch (...) {
        return 0;
    }
}

}  // namespace

void query_topology(CpuInfo& info) {
    const fs::path sys_cpu = "/sys/devices/system/cpu";
    if (!fs::exists(sys_cpu)) return;

    // 1) Enumerate logical CPUs (cpu0, cpu1, ...).
    std::vector<u32> cpu_ids;
    for (auto& entry : fs::directory_iterator(sys_cpu)) {
        auto name = entry.path().filename().string();
        if (name.size() <= 3 || !name.starts_with("cpu")) continue;
        std::string_view suffix(name.data() + 3, name.size() - 3);
        if (!is_digits(suffix)) continue;
        try {
            cpu_ids.push_back(static_cast<u32>(std::stoul(name.substr(3))));
        } catch (...) {
            // skip
        }
    }
    std::sort(cpu_ids.begin(), cpu_ids.end());

    info.logical_core_count = static_cast<u32>(cpu_ids.size());
    info.logical_cores.assign(info.logical_core_count, LogicalCore{});

    // 2) Per-core: package_id, core_id, NUMA node.
    std::set<std::pair<u32, u32>> phys_keys;  // unique (pkg, core) pairs
    struct CpuMeta { u32 pkg; u32 core; };
    std::vector<CpuMeta> meta(info.logical_core_count);

    for (usize idx = 0; idx < cpu_ids.size(); ++idx) {
        u32 cpu = cpu_ids[idx];
        auto base = sys_cpu / ("cpu" + std::to_string(cpu));
        auto& lc = info.logical_cores[idx];
        lc.os_id = cpu;

        meta[idx].pkg  = read_uint(base / "topology" / "physical_package_id");
        meta[idx].core = read_uint(base / "topology" / "core_id");
        phys_keys.emplace(meta[idx].pkg, meta[idx].core);

        // NUMA: cpu*/node* symlink names the node.
        if (fs::exists(base)) {
            for (auto& e : fs::directory_iterator(base)) {
                auto n = e.path().filename().string();
                if (n.size() <= 4 || !n.starts_with("node")) continue;
                std::string_view suffix(n.data() + 4, n.size() - 4);
                if (!is_digits(suffix)) continue;
                try {
                    lc.numa_node = static_cast<u32>(std::stoul(n.substr(4)));
                } catch (...) {
                    // skip
                }
                break;
            }
        }
    }

    std::vector<std::pair<u32, u32>> phys_vec(phys_keys.begin(), phys_keys.end());
    info.physical_core_count = static_cast<u32>(phys_vec.size());
    for (usize idx = 0; idx < info.logical_cores.size(); ++idx) {
        auto key = std::make_pair(meta[idx].pkg, meta[idx].core);
        auto it = std::find(phys_vec.begin(), phys_vec.end(), key);
        info.logical_cores[idx].physical_core_id =
            (it == phys_vec.end()) ? 0u : static_cast<u32>(std::distance(phys_vec.begin(), it));
    }

    // 3) L3 caches define clusters. Each shared_cpu_list is one cluster.
    struct ClusterTmp {
        std::string shared_list;
        u32 size_bytes;
        std::vector<u32> cores;
    };
    std::vector<ClusterTmp> tmp;
    for (u32 cpu : cpu_ids) {
        auto cache_dir = sys_cpu / ("cpu" + std::to_string(cpu)) / "cache";
        if (!fs::exists(cache_dir)) continue;
        for (auto& e : fs::directory_iterator(cache_dir)) {
            auto idx_name = e.path().filename().string();
            if (!idx_name.starts_with("index")) continue;
            if (read_trimmed(e.path() / "level") != "3") continue;
            if (read_trimmed(e.path() / "type")  == "Instruction") continue;

            auto shared = read_trimmed(e.path() / "shared_cpu_list");
            auto size_b = parse_size_kmg(read_trimmed(e.path() / "size"));
            auto found = std::find_if(tmp.begin(), tmp.end(),
                [&](const ClusterTmp& c) { return c.shared_list == shared; });
            if (found == tmp.end()) {
                tmp.push_back(ClusterTmp{shared, size_b, parse_cpulist(shared)});
            }
            break;
        }
    }

    info.clusters.reserve(tmp.size());
    for (usize i = 0; i < tmp.size(); ++i) {
        Cluster c{};
        c.id = static_cast<u32>(i);
        c.l3_size_bytes = tmp[i].size_bytes;
        c.logical_cores = tmp[i].cores;
        for (u32 lp : c.logical_cores) {
            for (auto& lc : info.logical_cores) {
                if (lc.os_id == lp) lc.cluster_id = c.id;
            }
        }
        info.clusters.push_back(std::move(c));
    }

    // 4) Hybrid (Intel P/E core) on Linux 5.13+:
    //    /sys/devices/cpu_core/cpus  → P-cores
    //    /sys/devices/cpu_atom/cpus  → E-cores
    auto sys_devices = sys_cpu.parent_path().parent_path() / "cpu_core" / "cpus";
    auto sys_atom    = sys_cpu.parent_path().parent_path() / "cpu_atom" / "cpus";
    if (fs::exists(sys_devices) && fs::exists(sys_atom)) {
        info.is_hybrid = true;
        auto p_cpus = parse_cpulist(read_trimmed(sys_devices));
        auto e_cpus = parse_cpulist(read_trimmed(sys_atom));
        for (auto& lc : info.logical_cores) {
            if (std::find(p_cpus.begin(), p_cpus.end(), lc.os_id) != p_cpus.end()) {
                lc.core_class = CoreClass::Performance;
            } else if (std::find(e_cpus.begin(), e_cpus.end(), lc.os_id) != e_cpus.end()) {
                lc.core_class = CoreClass::Efficiency;
            }
        }
    }
}

}  // namespace cardinal::topology::platform_detail

#endif  // CARDINAL_PLATFORM_LINUX
