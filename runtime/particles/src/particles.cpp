#include <cardinal/particles/particles.hpp>

#include <cardinal/core/algorithm.hpp>   // cardinal::clamp/remove_if
#include <cardinal/core/cmath.hpp>       // cardinal scalar math
#include <cardinal/core/utility.hpp>     // cardinal::move

namespace cardinal::particles {

namespace {

// Fast deterministic 32-bit RNG — splitmix-esque. Returns a uniform float in [0,1).
inline u32 splitmix32(u32& s) noexcept {
    s += 0x9E3779B9u;
    u32 z = s;
    z = (z ^ (z >> 16)) * 0x21f0aaadu;
    z = (z ^ (z >> 15)) * 0x735a2d97u;
    z =  z ^ (z >> 15);
    return z;
}
inline float rand01(u32& s) noexcept {
    return static_cast<float>(splitmix32(s)) * (1.0f / 4294967296.0f);
}
inline float rand_in(u32& s, float lo, float hi) noexcept {
    return lo + (hi - lo) * rand01(s);
}

// Lerp two RGBA u32 colors at t∈[0,1], component-wise.
inline u32 lerp_rgba(u32 a, u32 b, float t) noexcept {
    if (t <= 0.0f) return a;
    if (t >= 1.0f) return b;
    auto blend = [t](u8 ca, u8 cb) {
        return static_cast<u8>(ca + (cb - ca) * t);
    };
    const u8 ar = (a >> 0) & 0xFF, ag = (a >> 8) & 0xFF, ab = (a >> 16) & 0xFF, aa = (a >> 24) & 0xFF;
    const u8 br = (b >> 0) & 0xFF, bg = (b >> 8) & 0xFF, bb = (b >> 16) & 0xFF, ba = (b >> 24) & 0xFF;
    return (blend(ar, br) << 0) | (blend(ag, bg) << 8) |
           (blend(ab, bb) << 16) | (blend(aa, ba) << 24);
}

}  // namespace

// ---------------------------------------------------------------------------
// Emitter
// ---------------------------------------------------------------------------
Emitter::Emitter(EmitterDesc desc)
    : desc_(cardinal::move(desc)), rng_state_(desc_.rng_seed)
{
    live_.reserve(desc_.max_particles);
}

void Emitter::spawn_one_() noexcept {
    if (live_.size() >= desc_.max_particles) return;
    // Sanitize every spawn-time desc field. EmitterDesc is public with
    // no setters — a NaN/+Inf component reaches the particle's
    // PERSISTENT state and poisons every subsequent integrator tick.
    // Concretely, before this fix:
    //   * NaN desc_.origin   → p.position NaN at spawn → position
    //     stays NaN every later tick.
    //   * NaN desc_.gravity  → p.acceleration NaN → `p.velocity +=
    //     accel * dt` makes velocity NaN tick 1 → position NaN tick 2.
    //   * NaN desc_.velocity_min/max → rand_in yields NaN velocity →
    //     position NaN tick 1.
    //   * NaN desc_.size_min/max → NaN p.size → renderer reads NaN
    //     size (downstream UB / NaN vertex attribute).
    // The drag/lifetime reaper at 3704b48 only catches non-finite
    // lifetime; this commit catches the other four fields at the
    // SOURCE so the particle's persistent state is finite by
    // construction. Default each non-finite component to 0.0f — a
    // particle spawned at origin with zero velocity/acceleration/size
    // is degenerate but defined (no further poison).
    auto fz = [](float v) noexcept {
        return cardinal::isfinite(v) ? v : 0.0f;
    };
    Particle p{};
    p.position.x   = fz(desc_.origin.x);
    p.position.y   = fz(desc_.origin.y);
    p.position.z   = fz(desc_.origin.z);
    p.velocity.x   = rand_in(rng_state_, fz(desc_.velocity_min.x), fz(desc_.velocity_max.x));
    p.velocity.y   = rand_in(rng_state_, fz(desc_.velocity_min.y), fz(desc_.velocity_max.y));
    p.velocity.z   = rand_in(rng_state_, fz(desc_.velocity_min.z), fz(desc_.velocity_max.z));
    p.acceleration.x = fz(desc_.gravity.x);
    p.acceleration.y = fz(desc_.gravity.y);
    p.acceleration.z = fz(desc_.gravity.z);
    p.size         = rand_in(rng_state_, fz(desc_.size_min), fz(desc_.size_max));
    p.lifetime     = rand_in(rng_state_, fz(desc_.lifetime_min), fz(desc_.lifetime_max));
    p.age          = 0.0f;
    p.color_rgba   = desc_.start_rgba;
    live_.push_back(p);
    ++total_spawned_;
}

void Emitter::tick(float dt) noexcept {
    // Reject non-positive AND non-finite dt. A bare `dt <= 0.0f` lets NaN
    // through (NaN <= 0 is false) and +Inf through (Inf <= 0 is false):
    // spawn_accum_ then goes non-finite, static_cast<u32>(non-finite) is UB
    // (~2e9 "integer indefinite" → a multi-billion-iteration spawn loop),
    // the accumulator stays NaN forever (NaN - x = NaN) poisoning every
    // later valid tick, and age += dt makes `age >= lifetime` false →
    // immortal NaN particles. This noexcept tick must no-op on such input.
    if (!(dt > 0.0f) || !cardinal::isfinite(dt)) return;

    // 1) Spawn new particles using fractional accumulator (preserves rate
    //    independent of frame rate).
    if (desc_.emitting && desc_.rate_per_second > 0.0f) {
        spawn_accum_ += dt * desc_.rate_per_second;
        u32 to_spawn = static_cast<u32>(spawn_accum_);
        spawn_accum_ -= static_cast<float>(to_spawn);
        for (u32 i = 0; i < to_spawn; ++i) spawn_one_();
    }

    // 2) Integrate live particles. Iterate index-style so we can swap-pop
    //    on death without invalidating iterators.
    //
    // desc_.drag is a public field with no setter — the API exposes
    // EmitterDesc directly via desc(). A NaN drag (user bug or a
    // poisoned upstream parameter sweep) flowed into the damp clamp as
    // `cardinal::clamp(1.0f - NaN*dt, 0, 1)` = NaN (clamp is documented
    // NaN-passthrough), then `p.velocity.x *= NaN` poisons EVERY live
    // particle simultaneously every tick. Sanitize at the use site —
    // 0 drag (= no decay, particles keep their velocity) is a sensible
    // fallback that doesn't change long-term behaviour.
    const float drag_safe = cardinal::isfinite(desc_.drag) ? desc_.drag : 0.0f;
    const float damp = cardinal::clamp(1.0f - drag_safe * dt, 0.0f, 1.0f);
    usize i = 0;
    while (i < live_.size()) {
        Particle& p = live_[i];
        p.velocity.x += p.acceleration.x * dt;
        p.velocity.y += p.acceleration.y * dt;
        p.velocity.z += p.acceleration.z * dt;
        p.velocity.x *= damp;
        p.velocity.y *= damp;
        p.velocity.z *= damp;
        p.position.x += p.velocity.x * dt;
        p.position.y += p.velocity.y * dt;
        p.position.z += p.velocity.z * dt;
        p.age        += dt;
        const float t = (p.lifetime > 0.0f)
            ? cardinal::clamp(p.age / p.lifetime, 0.0f, 1.0f) : 1.0f;
        p.color_rgba = lerp_rgba(desc_.start_rgba, desc_.end_rgba, t);

        // Reap if expired OR non-finite-lifetime. NaN lifetime (spawned
        // from a NaN lifetime_min/max in the desc) makes `p.age >=
        // p.lifetime` unordered-false → IMMORTAL particle that leaks
        // memory indefinitely (live_ grows unbounded; the swap-pop is
        // never reached for it). Treat non-finite lifetime as expired.
        if (p.age >= p.lifetime || !cardinal::isfinite(p.lifetime)) {
            // swap-pop
            p = live_.back();
            live_.pop_back();
        } else {
            ++i;
        }
    }
}

void Emitter::clear() noexcept {
    live_.clear();
    spawn_accum_ = 0.0f;
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------
cardinal::shared_ptr<System> System::create() {
    return cardinal::shared_ptr<System>(new System());
}

Emitter* System::add(EmitterDesc desc) {
    emitters_.push_back(cardinal::make_unique<Emitter>(cardinal::move(desc)));
    return emitters_.back().get();
}

void System::remove(Emitter* e) {
    emitters_.erase(cardinal::remove_if(emitters_.begin(), emitters_.end(),
        [e](const cardinal::unique_ptr<Emitter>& u){ return u.get() == e; }),
        emitters_.end());
}

void System::clear() { emitters_.clear(); }

void System::tick(float dt) {
    for (auto& e : emitters_) e->tick(dt);
}

System::Stats System::stats() const noexcept {
    Stats s{};
    s.emitters_active = static_cast<u32>(emitters_.size());
    for (const auto& e : emitters_) {
        s.particles_alive += static_cast<u32>(e->live_count());
        s.particles_total_spawned += e->total_spawned();
    }
    return s;
}

}  // namespace cardinal::particles
