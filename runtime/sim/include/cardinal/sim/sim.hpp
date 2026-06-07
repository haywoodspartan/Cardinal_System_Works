#pragma once

// =============================================================================
// Cardinal — Simulation orchestrator.
//
// Wraps an actor::World + tick groups + a clock. Hosts call tick(real_dt)
// once per frame and the SimWorld figures out:
//
//   - Whether to run a fixed-step physics tick (and how many sub-steps if
//     real_dt > fixed_dt — capped to avoid the spiral of death)
//   - Which TickGroups to run, in order, with FrameScopes for profiler
//   - Whether the sim is paused / single-stepping / time-dilated
//
// Tick groups (run in this order each frame):
//   1. PreUpdate        — input sampling, AI decisions
//   2. PrePhysics       — gather forces, apply impulses
//   3. Physics          — integrate rigid bodies (fixed step)
//   4. PostPhysics      — collision response, constraints
//   5. Update           — game logic / scripts
//   6. LateUpdate       — camera follow, animation root motion
//   7. Render           — pure read; renderer pulls from here
//
// Each group accepts a callback list. The World::tick(dt) is run inside
// the Update group automatically.
// =============================================================================

#include <cardinal/actor/world.hpp>
#include <cardinal/core/types.hpp>        // function/string/unique_ptr
#include <cardinal/core/std/containers.hpp>   // cardinal::vector

namespace cardinal::sim {

enum class TickGroup : u32 {
    PreUpdate   = 0,
    PrePhysics  = 1,
    Physics     = 2,
    PostPhysics = 3,
    Update      = 4,
    LateUpdate  = 5,
    Render      = 6,
    Count       = 7,
};

const char* tick_group_name(TickGroup g) noexcept;

struct SimDesc {
    float fixed_dt        {1.0f / 60.0f};   // physics step
    float max_real_dt     {0.10f};          // clamp to avoid huge integrations
    u32   max_substeps    {4};              // cap when real_dt is large
    bool  start_paused    {false};
    u32   determinism_seed{0};              // 0 = use real time as seed

    // ---- Parallel handler dispatch (opt-in) -----------------------------
    //
    // When true, handlers within a TickGroup are dispatched concurrently
    // across the JobSystem's worker pool (using async::parallel_invoke).
    // Off by default — opt in only when every handler in every group is
    // independent (no two handlers write the same component / sound voice
    // / actor list, no two handlers fight over the same mutex).
    //
    // Physics integration + actor::World::tick stay sequential regardless
    // — only user-installed handlers are fanned out. Inside one Tick (a
    // call to SimWorld::tick) the order between groups is unchanged.
    bool  parallel_handlers{false};
};

struct SimStats {
    u64    total_ticks{0};
    u64    physics_substeps_last{0};
    double sim_time_seconds{0.0};
    double real_time_seconds{0.0};
    float  time_scale{1.0f};
    bool   paused{false};
    bool   single_step_armed{false};
};

class SimWorld {
public:
    explicit SimWorld(const SimDesc& desc = {});
    ~SimWorld();
    SimWorld(const SimWorld&) = delete;
    SimWorld& operator=(const SimWorld&) = delete;

    cardinal::actor::World&       world()       noexcept { return *world_; }
    const cardinal::actor::World& world() const noexcept { return *world_; }

    // ---- Per-frame entry point --------------------------------------
    // `real_dt` is wall-clock seconds since last call. Returns the
    // simulation dt actually applied (may be 0 when paused).
    float tick(float real_dt);

    // ---- Pause / step / time scale ----------------------------------
    void  pause();
    void  resume();
    void  step_one_frame();             // single-step then auto-pause
    void  set_time_scale(float s);
    bool  paused() const noexcept;
    float time_scale() const noexcept;

    // ---- Tick groups -------------------------------------------------
    using TickFn = cardinal::function<void(float dt)>;
    using HandlerId = u32;
    HandlerId add_handler(TickGroup g, TickFn fn);
    void      remove_handler(HandlerId id);

    // ---- Stats / desc ------------------------------------------------
    SimStats  stats() const noexcept;
    const SimDesc& desc() const noexcept { return desc_; }

private:
    void run_group_(TickGroup g, float dt);
    void integrate_physics_(float dt);

    SimDesc                                desc_{};
    cardinal::unique_ptr<cardinal::actor::World> world_;
    struct Sub { HandlerId id; TickFn fn; };
    cardinal::vector<Sub>                       handlers_[static_cast<usize>(TickGroup::Count)];
    HandlerId                              next_handler_id_{1};

    bool                                   paused_{false};
    bool                                   single_step_{false};
    float                                  time_scale_{1.0f};
    float                                  physics_accum_{0.0f};
    SimStats                               stats_{};
};

}  // namespace cardinal::sim
