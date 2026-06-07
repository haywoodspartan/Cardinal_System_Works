#pragma once

// =============================================================================
// Cardinal — Memory AutoScaler helper.
//
// A common pattern atop cardinal::budget::Broker. Subsystems that "scale
// by tier" (render scale, mesh cache size, shadow map resolution,
// streaming radius, VT pool slots, ...) all share the same recipe:
//
//   - 4 values, one per Pressure tier (Low / Medium / High / Critical)
//   - Apply on tier-change callback
//   - Optional clamp / round
//
// AutoScaler<T> wraps that. Construct with a name + per-tier values + an
// apply callback. Register with a Broker. Done.
//
// Template parameter T is whatever the subsystem's "knob" is (float for
// render scale, u32 for slot counts, etc.). The apply lambda gets the
// new value AND the new tier (so subsystems can do tier-specific extra
// work — log, evict, pool resize).
// =============================================================================

#include <cardinal/core/budget/budget.hpp>
#include <cardinal/core/budget/memory.hpp>
#include <cardinal/core/types.hpp>

#include <functional>
#include <string>

namespace cardinal::budget {

template <class T>
class AutoScaler {
public:
    using Apply = std::function<void(const T& new_value, memory::Pressure tier)>;

    struct Config {
        std::string  name;
        Domain       domain{Domain::Gpu};
        T            value_low{};
        T            value_medium{};
        T            value_high{};
        T            value_critical{};
        u64          advisory_min_bytes{0};
        u64          advisory_max_bytes{0};
        Apply        apply;
    };

    AutoScaler(Broker& broker, Config cfg)
        : broker_(broker), cfg_(std::move(cfg)), id_(0)
    {
        Subsystem sub{};
        sub.name               = cfg_.name;
        sub.domain             = cfg_.domain;
        sub.advisory_min_bytes = cfg_.advisory_min_bytes;
        sub.advisory_max_bytes = cfg_.advisory_max_bytes;
        sub.on_pressure_change = [this](memory::Pressure p) {
            const T& v = pick_value_(p);
            current_   = v;
            current_tier_ = p;
            if (cfg_.apply) cfg_.apply(v, p);
        };
        id_ = broker_.register_subsystem(std::move(sub));
        // Apply Low at construction so the subsystem starts in a known
        // state without waiting for a transition.
        current_      = cfg_.value_low;
        current_tier_ = memory::Pressure::Low;
        if (cfg_.apply) cfg_.apply(current_, current_tier_);
    }

    ~AutoScaler() { broker_.deregister_subsystem(id_); }

    AutoScaler(const AutoScaler&) = delete;
    AutoScaler& operator=(const AutoScaler&) = delete;

    void report_used(u64 bytes) noexcept { broker_.report_used(id_, bytes); }
    const T& current() const noexcept { return current_; }
    memory::Pressure current_tier() const noexcept { return current_tier_; }
    u32 broker_id() const noexcept { return id_; }

private:
    const T& pick_value_(memory::Pressure p) const noexcept {
        switch (p) {
            case memory::Pressure::Low:      return cfg_.value_low;
            case memory::Pressure::Medium:   return cfg_.value_medium;
            case memory::Pressure::High:     return cfg_.value_high;
            case memory::Pressure::Critical: return cfg_.value_critical;
        }
        return cfg_.value_low;
    }

    Broker&          broker_;
    Config           cfg_;
    u32              id_;
    T                current_{};
    memory::Pressure current_tier_{memory::Pressure::Low};
};

}  // namespace cardinal::budget
