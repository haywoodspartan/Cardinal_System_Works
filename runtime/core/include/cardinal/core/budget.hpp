#pragma once

// =============================================================================
// Cardinal — Budget Broker.
//
// Subsystems voluntarily honour a memory budget. The broker:
//
//   1. Owns a memory::Monitor (system RAM + this-process working set).
//   2. Optionally consumes a GPU budget reported by the RHI device.
//   3. Computes a Pressure tier per Domain (System, Gpu) on each tick().
//   4. On a tier *transition*, fans a callback to every registered Subsystem
//      so it can grow / shrink its caches.
//
// Subsystems also report their own usage back so the broker can show "world
// streamer is using 412 MB" in a panel without poking at internals. Reporting
// is opt-in — a subsystem that registers without ever calling report_used()
// is treated as "size unknown" by the panel but still receives tier callbacks.
//
// The broker is single-threaded by design — call tick() once per frame from
// the main thread. Subsystem callbacks fire on the same thread, so they can
// touch their own state without locks. If a subsystem needs to dispatch its
// reaction onto worker threads, that's the subsystem's call.
// =============================================================================

#include <cardinal/core/memory.hpp>
#include <cardinal/core/types.hpp>

#include <functional>
#include <string>
#include <vector>

namespace cardinal::budget {

// Backend-agnostic VRAM snapshot. Filled by a GpuQueryFn provided by the
// host (engine wires the rhi::Device call site, the broker stays linkage-
// independent of cardinal::rhi).
struct GpuMemorySample {
    u64 budget_bytes{0};
    u64 current_usage_bytes{0};
};

using GpuQueryFn = std::function<GpuMemorySample()>;

enum class Domain : u32 {
    System = 0,   // host RAM (process-relevant slice of system memory)
    Gpu    = 1,   // adapter local memory (VRAM)
};

const char* domain_name(Domain d) noexcept;

// One Subsystem per registered consumer. The callback fires on tier
// *transitions*; steady-state ticks do not invoke it. Callbacks see the
// new tier and the snapshot that produced it.
//
// `advisory_min_bytes` and `advisory_max_bytes` are hints the panel uses
// to draw a usage bar. They are not enforced by the broker — the
// subsystem owns its allocation policy.
struct Subsystem {
    std::string  name;
    Domain       domain{Domain::System};
    u64          advisory_min_bytes{0};
    u64          advisory_max_bytes{0};
    std::function<void(memory::Pressure new_tier)> on_pressure_change;
};

struct Snapshot {
    memory::SystemSnapshot   system{};
    memory::ProcessSnapshot  process{};

    // Filled by the broker if a GPU device is bound; else zero.
    u64                      gpu_budget_bytes{0};
    u64                      gpu_current_usage_bytes{0};

    memory::Pressure         system_pressure{memory::Pressure::Low};
    memory::Pressure         gpu_pressure   {memory::Pressure::Low};
};

struct SubsystemReport {
    u32                     id{0};
    std::string             name;
    Domain                  domain{Domain::System};
    u64                     used_bytes{0};
    u64                     advisory_min_bytes{0};
    u64                     advisory_max_bytes{0};
    memory::Pressure        last_seen_tier{memory::Pressure::Low};
};

class Broker {
public:
    Broker();
    ~Broker();

    Broker(const Broker&)            = delete;
    Broker& operator=(const Broker&) = delete;

    // Optional GPU binding. The host supplies a callable that knows how to
    // ask the active rhi::Device for a fresh VRAM sample. Pass an empty
    // function to clear it.
    void bind_gpu_query(GpuQueryFn query) noexcept;

    // Customise per-Domain thresholds. Defaults work for most workloads.
    void set_thresholds(Domain d, const memory::PressureThresholds& th) noexcept;

    // Returns an opaque id used to deregister.
    u32  register_subsystem(Subsystem sub);
    void deregister_subsystem(u32 id) noexcept;

    // A subsystem reports its current consumption. Cheap; just stores it.
    void report_used(u32 id, u64 bytes) noexcept;

    // Drive once per frame. Refreshes monitor, computes pressure, fires
    // tier-transition callbacks. `interval_ms` throttles the underlying
    // OS-counter sampling — tier callbacks still fire at frame rate.
    void tick(u32 sample_interval_ms = 250);

    // Force an immediate refresh + tier recompute (e.g. after a major load).
    void refresh();

    const Snapshot& last_snapshot() const noexcept;
    std::vector<SubsystemReport> subsystem_reports() const;

    // For tests / Studio panel to apply synthetic pressure without OS state.
    void debug_force_pressure(Domain d, memory::Pressure p);
    void debug_clear_force(Domain d) noexcept;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace cardinal::budget
