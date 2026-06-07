#include <cardinal/core/budget/budget.hpp>

#include <cardinal/core/diag/log.hpp>

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

    // Snapshot all mutex-protected config under ONE lock acquire — the
    // previous code read impl_->thresholds (lines 98 / 115) and impl_->
    // forced (120-124) OUTSIDE impl_->mtx while set_thresholds /
    // debug_force_pressure / debug_clear_force wrote them UNDER it.
    // PressureThresholds is plain double pairs and forced is array<int,2>:
    // a torn read could surface as a wrong pressure tier for one frame
    // (forced[d] read mid-write between -1 and >=0 → classify ignored
    // OR applied wrong; thresholds read between low_pct and high_pct
    // updates → pressure crossing the wrong band). Same data-race-on-
    // shared-state class as budget::last_snapshot 033c642 / audio::
    // set_listener a317126 / io::cancelled_handles 82604f9. Folding
    // gpu_query into the same snapshot also avoids the prior second
    // lock acquire.
    memory::PressureThresholds sys_thresh, gpu_thresh;
    int forced_sys, forced_gpu;
    GpuQueryFn query;
    {
        std::lock_guard<std::mutex> lg(impl_->mtx);
        sys_thresh = impl_->thresholds[static_cast<u32>(Domain::System)];
        gpu_thresh = impl_->thresholds[static_cast<u32>(Domain::Gpu)];
        forced_sys = impl_->forced[0];
        forced_gpu = impl_->forced[1];
        query      = impl_->gpu_query;
    }

    // System tier — react to OS-wide pressure since other apps can squeeze us.
    snap.system_pressure = memory::classify(snap.system.load_percent, sys_thresh);

    // GPU tier (if a query is bound).
    if (query) {
        const auto vs = query();
        snap.gpu_budget_bytes        = vs.budget_bytes;
        snap.gpu_current_usage_bytes = vs.current_usage_bytes;
        if (vs.budget_bytes > 0) {
            const double pct = 100.0 *
                (static_cast<double>(vs.current_usage_bytes) /
                 static_cast<double>(vs.budget_bytes));
            snap.gpu_pressure = memory::classify(pct, gpu_thresh);
        }
    }

    // Apply forced overrides (tests / Studio panel).
    if (forced_sys >= 0) {
        snap.system_pressure = static_cast<memory::Pressure>(forced_sys);
    }
    if (forced_gpu >= 0) {
        snap.gpu_pressure = static_cast<memory::Pressure>(forced_gpu);
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

Snapshot Broker::last_snapshot() const noexcept {
    // impl_->last_snapshot is written under impl_->mtx by tick() (line
    // 138); returning a const reference let every concurrent reader
    // observe a half-updated Snapshot (~64 B across u64 / enum /
    // nested SystemSnapshot+ProcessSnapshot — field-level tearing on
    // load_percent / pressure mid-update). Take the mutex and return
    // a copy. Same shape as the io.cpp cancelled_handles race fix
    // (82604f9): a read of mutex-protected state was happening
    // outside the lock. The copy is trivial POD (no allocation, no
    // throw) — noexcept stays.
    std::lock_guard<std::mutex> lg(impl_->mtx);
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
