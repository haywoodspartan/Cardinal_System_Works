#pragma once
// =============================================================================
// Cardinal — TimerSet: gameplay timers ("do X after N seconds / every N s").
//
// A small deterministic scheduler a GameActor (or the host) owns and ticks
// with the game dt. Callbacks may schedule or cancel timers from inside a
// callback: additions made during tick() are not fired until the NEXT tick
// (so a repeating timer can't run away within one frame) and cancellation
// is safe at any time — a cancelled timer never fires again, even if it is
// already due in the current tick.
//
//   TimerSet timers;
//   auto h = timers.after(2.0f, [] { open_door(); });
//   timers.every(0.5f, [] { spawn_wave(); });
//   ... per frame: timers.tick(dt);
//   timers.cancel(h);
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/containers.hpp>

namespace cardinal::game {

class TimerSet {
public:
    struct Handle {
        u32 id{0};
        bool valid() const noexcept { return id != 0; }
    };
    using Fn = cardinal::function<void()>;

    // One-shot after `seconds` (<= 0 fires on the next tick).
    Handle after(float seconds, Fn fn);
    // Repeating every `seconds` (clamped to a small positive minimum).
    Handle every(float seconds, Fn fn);
    // Stops a timer. Returns false if unknown / already finished.
    bool   cancel(Handle h) noexcept;
    bool   active(Handle h) const noexcept;

    // Advance all timers by `dt` (non-finite / negative dt is ignored).
    // Returns how many callbacks fired.
    u32    tick(float dt);

    usize  active_count() const noexcept;
    void   clear() noexcept { entries_.clear(); }

    // Seconds until `h` fires next (0 if unknown).
    float  remaining(Handle h) const noexcept;

private:
    struct Entry {
        u32   id{0};
        float remaining{0.0f};
        float period{0.0f};      // 0 = one-shot
        bool  alive{true};
        Fn    fn;
    };
    cardinal::vector<Entry> entries_;
    u32 next_id_{1};
    bool in_tick_{false};
};

}  // namespace cardinal::game
