#pragma once

// =============================================================================
// Cardinal — CPU Particle system.
//
// A System owns N Emitters; each Emitter has an EmitterDesc (rate, lifetime,
// initial conditions) and a flat std::vector<Particle> pool. Per tick:
//
//   1. Spawn — emit ceil(rate * dt) new particles, picking start position
//              + velocity + size + color from the desc's RNG ranges
//   2. Update — integrate each live particle (vel += accel*dt, pos += vel*dt,
//               age += dt), apply optional drag + gravity
//   3. Cull   — remove particles whose age >= lifetime
//
// The renderer integration is a follow-up (billboard sprites in a vertex
// buffer); this module ships the simulator + a CPU snapshot accessor so
// the editor + sample can render with ImGui draw lists for now.
//
// Designed to scale: per-emitter particle pools never reallocate after the
// first lifetime cycle, so steady-state allocation is zero.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/scene/math.hpp>

#include <memory>
#include <string>
#include <vector>

namespace cardinal::particles {

// ---------------------------------------------------------------------------
// Particle — one simulated point. 64 bytes — sized so 16 fit per cache line.
// ---------------------------------------------------------------------------
struct Particle {
    cardinal::scene::Vec3 position    {0, 0, 0};
    cardinal::scene::Vec3 velocity    {0, 0, 0};
    cardinal::scene::Vec3 acceleration{0, 0, 0};
    float                 size        {0.10f};
    float                 age         {0.0f};
    float                 lifetime    {1.0f};
    u32                   color_rgba  {0xFFFFFFFFu};
    u32                   _pad        {0};
};

// ---------------------------------------------------------------------------
// EmitterDesc — authoring data. Ranges are sampled uniformly between
// _min/_max each spawn; set min == max for a fixed value.
// ---------------------------------------------------------------------------
struct EmitterDesc {
    std::string name;

    cardinal::scene::Vec3 origin{0, 0, 0};

    float rate_per_second{30.0f};
    float lifetime_min{1.0f};
    float lifetime_max{2.0f};
    float size_min{0.05f};
    float size_max{0.25f};

    cardinal::scene::Vec3 velocity_min{-0.5f,  1.0f, -0.5f};
    cardinal::scene::Vec3 velocity_max{ 0.5f,  3.0f,  0.5f};

    cardinal::scene::Vec3 gravity{0.0f, -9.81f, 0.0f};
    float                 drag   {0.05f};                // per-second linear damping

    // Color over life: linear lerp from start_rgba (age 0) to end_rgba (age == lifetime).
    u32 start_rgba{0xFFFFFFFFu};
    u32 end_rgba  {0x00FFFFFFu};   // fade out alpha by default

    u32  max_particles{1024};
    bool emitting{true};
    u32  rng_seed{1337};
};

// ---------------------------------------------------------------------------
// Emitter — runtime state for one simulated source.
// ---------------------------------------------------------------------------
class Emitter {
public:
    explicit Emitter(EmitterDesc desc);

    void tick(float dt) noexcept;
    void clear() noexcept;
    void set_emitting(bool e) noexcept { desc_.emitting = e; }

    EmitterDesc&       desc()       noexcept { return desc_; }
    const EmitterDesc& desc() const noexcept { return desc_; }

    const std::vector<Particle>& particles() const noexcept { return live_; }
    usize live_count() const noexcept { return live_.size(); }
    u64   total_spawned() const noexcept { return total_spawned_; }

private:
    void spawn_one_() noexcept;

    EmitterDesc            desc_;
    std::vector<Particle>  live_;
    float                  spawn_accum_{0.0f};
    u32                    rng_state_  {0};
    u64                    total_spawned_{0};
};

// ---------------------------------------------------------------------------
// System — registry of emitters. Tick all of them; iterate them for
// rendering/inspection.
// ---------------------------------------------------------------------------
class System {
public:
    static std::shared_ptr<System> create();

    Emitter* add(EmitterDesc desc);
    void     remove(Emitter* e);
    void     clear();

    void tick(float dt);

    const std::vector<std::unique_ptr<Emitter>>& emitters() const noexcept { return emitters_; }

    struct Stats {
        u32 emitters_active{0};
        u32 particles_alive{0};
        u64 particles_total_spawned{0};
    };
    Stats stats() const noexcept;

private:
    std::vector<std::unique_ptr<Emitter>> emitters_;
};

}  // namespace cardinal::particles
