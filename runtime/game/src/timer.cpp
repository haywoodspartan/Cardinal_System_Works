// =============================================================================
// Cardinal — TimerSet. See timer.hpp.
// =============================================================================
#include <cardinal/game/timer.hpp>
#include <cardinal/core/std/cmath.hpp>
#include <cardinal/core/std/utility.hpp>

namespace cardinal::game {

namespace { constexpr float kDueEpsilon = 1.0e-6f; }

TimerSet::Handle TimerSet::after(float seconds, Fn fn) {
    Entry e;
    e.id        = next_id_++;
    e.remaining = (cardinal::isfinite(seconds) && seconds > 0.0f) ? seconds : 0.0f;
    e.period    = 0.0f;
    e.fn        = cardinal::move(fn);
    entries_.push_back(cardinal::move(e));
    return Handle{ entries_.back().id };
}

TimerSet::Handle TimerSet::every(float seconds, Fn fn) {
    constexpr float kMinPeriod = 1.0e-3f;
    Entry e;
    e.id        = next_id_++;
    e.period    = (cardinal::isfinite(seconds) && seconds > kMinPeriod) ? seconds : kMinPeriod;
    e.remaining = e.period;
    e.fn        = cardinal::move(fn);
    entries_.push_back(cardinal::move(e));
    return Handle{ entries_.back().id };
}

bool TimerSet::cancel(Handle h) noexcept {
    for (auto& e : entries_)
        if (e.id == h.id && e.alive) { e.alive = false; return true; }
    return false;
}

bool TimerSet::active(Handle h) const noexcept {
    for (const auto& e : entries_)
        if (e.id == h.id) return e.alive;
    return false;
}

float TimerSet::remaining(Handle h) const noexcept {
    for (const auto& e : entries_)
        if (e.id == h.id && e.alive) return e.remaining;
    return 0.0f;
}

usize TimerSet::active_count() const noexcept {
    usize n = 0;
    for (const auto& e : entries_) if (e.alive) ++n;
    return n;
}

u32 TimerSet::tick(float dt) {
    if (!cardinal::isfinite(dt) || dt < 0.0f) dt = 0.0f;
    u32 fired = 0;
    in_tick_ = true;
    // Walk by INDEX over the entries that existed at tick start: callbacks
    // may push new entries (which therefore fire from the NEXT tick on) and
    // the vector may reallocate.
    const usize n = entries_.size();
    for (usize i = 0; i < n; ++i) {
        if (!entries_[i].alive) continue;
        entries_[i].remaining -= dt;
        // Due when the remaining time is within float noise of zero (the
        // subtractions accumulate ~1e-7 error); a repeating timer fires at
        // most once per tick (no catch-up burst).
        if (entries_[i].remaining > kDueEpsilon) continue;
        if (entries_[i].period > 0.0f) {
            entries_[i].remaining += entries_[i].period;
            if (entries_[i].remaining < 0.0f) entries_[i].remaining = entries_[i].period;
        } else {
            entries_[i].alive = false;
        }
        if (entries_[i].fn) {
            Fn fn = entries_[i].fn;   // copy: the callback may cancel / mutate the set
            fn();
            ++fired;
        }
    }
    in_tick_ = false;
    // Compact finished / cancelled entries.
    usize keep = 0;
    for (usize i = 0; i < entries_.size(); ++i) {
        if (!entries_[i].alive) continue;
        if (keep != i) entries_[keep] = cardinal::move(entries_[i]);
        ++keep;
    }
    entries_.resize(keep);
    return fired;
}

}  // namespace cardinal::game
