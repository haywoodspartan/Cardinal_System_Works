#include <cardinal/core/budget.hpp>

#include <cardinal/core/log.hpp>

#include <array>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace cardinal::budget {

const char* domain_name(Domain d) noexcept {
    switch (d) {
        case Domain::System: return "System";
        case Domain::Gpu:    return "GPU";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------
struct Broker::Impl {
    memory::Monitor                          monitor{};
    GpuQueryFn                               gpu_query{};

    std::array<memory::PressureThresholds, 2> thresholds{};

    // Optional pinning of pressure for tests. -1 == "use real value".
    std::array<int, 2>                       forced{-1, -1};

    std::unordered_map<u32, Subsystem>       subs;
    std::unordered_map<u32, u64>             used_bytes;
    std::unordered_map<u32, memory::Pressure> last_seen_tier;

    std::array<memory::Pressure, 2>          last_tier{
        memory::Pressure::Low, memory::Pressure::Low};

    Snapshot                                 last_snapshot{};

    std::atomic<u32>                         next_id{1};
    mutable std::mutex                       mtx;
};

Broker::Broker() : impl_(new Impl{}) {
    // Slightly stricter thresholds for VRAM — the GPU has no swap.
    impl_->thresholds[static_cast<u32>(Domain::Gpu)] =
        memory::PressureThresholds{55.0, 75.0, 90.0};
}

Broker::~Broker() {
    delete impl_;
}

void Broker::bind_gpu_query(GpuQueryFn query) noexcept {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    impl_->gpu_query = std::move(query);
}

void Broker::set_thresholds(Domain d, const memory::PressureThresholds& th) noexcept {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    impl_->thresholds[static_cast<u32>(d)] = th;
}

u32 Broker::register_subsystem(Subsystem sub) {
    const u32 id = impl_->next_id.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lg(impl_->mtx);
    impl_->subs.emplace(id, std::move(sub));
    impl_->used_bytes.emplace(id, 0u);
    impl_->last_seen_tier.emplace(id, memory::Pressure::Low);
    return id;
}

void Broker::deregister_subsystem(u32 id) noexcept {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    impl_->subs.erase(id);
    impl_->used_bytes.erase(id);
    impl_->last_seen_tier.erase(id);
}

void Broker::report_used(u32 id, u64 bytes) noexcept {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    auto it = impl_->used_bytes.find(id);
    if (it != impl_->used_bytes.end()) it->second = bytes;
}

void Broker::tick(u32 sample_interval_ms) {
    impl_->monitor.tick(sample_interval_ms);

    Snapshot snap{};
    snap.system  = impl_->monitor.last_system();
    snap.process = impl_->monitor.last_process();

    // System tier — react to OS-wide pressure since other apps can squeeze us.
    snap.system_pressure = memory::classify(
        snap.system.load_percent,
        impl_->thresholds[static_cast<u32>(Domain::System)]);

    // GPU tier (if a query is bound).
    GpuQueryFn query;
    {
        std::lock_guard<std::mutex> lg(impl_->mtx);
        query = impl_->gpu_query;
    }
    if (query) {
        const auto vs = query();
        snap.gpu_budget_bytes        = vs.budget_bytes;
        snap.gpu_current_usage_bytes = vs.current_usage_bytes;
        if (vs.budget_bytes > 0) {
            const double pct = 100.0 *
                (static_cast<double>(vs.current_usage_bytes) /
                 static_cast<double>(vs.budget_bytes));
            snap.gpu_pressure = memory::classify(
                pct, impl_->thresholds[static_cast<u32>(Domain::Gpu)]);
        }
    }

    // Apply forced overrides (tests / Studio panel).
    if (impl_->forced[0] >= 0) {
        snap.system_pressure = static_cast<memory::Pressure>(impl_->forced[0]);
    }
    if (impl_->forced[1] >= 0) {
        snap.gpu_pressure = static_cast<memory::Pressure>(impl_->forced[1]);
    }

    // Detect tier transitions & queue callbacks.
    std::vector<std::pair<std::function<void(memory::Pressure)>, memory::Pressure>> notify;
    {
        std::lock_guard<std::mutex> lg(impl_->mtx);
        const auto sys_old = impl_->last_tier[static_cast<u32>(Domain::System)];
        const auto gpu_old = impl_->last_tier[static_cast<u32>(Domain::Gpu)];
        const bool sys_changed = (sys_old != snap.system_pressure);
        const bool gpu_changed = (gpu_old != snap.gpu_pressure);

        impl_->last_tier[static_cast<u32>(Domain::System)] = snap.system_pressure;
        impl_->last_tier[static_cast<u32>(Domain::Gpu)]    = snap.gpu_pressure;
        impl_->last_snapshot = snap;

        if (sys_changed || gpu_changed) {
            for (auto& [id, sub] : impl_->subs) {
                const memory::Pressure new_tier =
                    (sub.domain == Domain::System) ? snap.system_pressure
                                                   : snap.gpu_pressure;
                const bool changed_for_me =
                    (sub.domain == Domain::System) ? sys_changed : gpu_changed;
                if (!changed_for_me) continue;

                impl_->last_seen_tier[id] = new_tier;
                if (sub.on_pressure_change) {
                    notify.emplace_back(sub.on_pressure_change, new_tier);
                }
            }
        }
    }

    // Fire outside the lock so callbacks may call back into the broker
    // (report_used, etc.) without deadlocking.
    for (auto& [cb, tier] : notify) {
        cb(tier);
    }

    if (!notify.empty()) {
        cardinal::log::infof("core/budget",
            "pressure transition - sys=%s gpu=%s (sys load=%.1f%%, gpu used=%llu/%llu)",
            memory::pressure_name(snap.system_pressure),
            memory::pressure_name(snap.gpu_pressure),
            snap.system.load_percent,
            static_cast<unsigned long long>(snap.gpu_current_usage_bytes),
            static_cast<unsigned long long>(snap.gpu_budget_bytes));
    }
}

void Broker::refresh() {
    impl_->monitor.refresh();
    tick(0);
}

const Snapshot& Broker::last_snapshot() const noexcept {
    return impl_->last_snapshot;
}

std::vector<SubsystemReport> Broker::subsystem_reports() const {
    std::vector<SubsystemReport> out;
    std::lock_guard<std::mutex> lg(impl_->mtx);
    out.reserve(impl_->subs.size());
    for (const auto& [id, sub] : impl_->subs) {
        SubsystemReport r{};
        r.id                 = id;
        r.name               = sub.name;
        r.domain             = sub.domain;
        r.advisory_min_bytes = sub.advisory_min_bytes;
        r.advisory_max_bytes = sub.advisory_max_bytes;
        auto u = impl_->used_bytes.find(id);
        r.used_bytes = (u != impl_->used_bytes.end()) ? u->second : 0;
        auto t = impl_->last_seen_tier.find(id);
        r.last_seen_tier = (t != impl_->last_seen_tier.end())
            ? t->second : memory::Pressure::Low;
        out.push_back(std::move(r));
    }
    return out;
}

void Broker::debug_force_pressure(Domain d, memory::Pressure p) {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    impl_->forced[static_cast<u32>(d)] = static_cast<int>(p);
}

void Broker::debug_clear_force(Domain d) noexcept {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    impl_->forced[static_cast<u32>(d)] = -1;
}

}  // namespace cardinal::budget
