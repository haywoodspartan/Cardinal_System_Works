#pragma once

// =============================================================================
// Cardinal — Game lifecycle orchestrator.
//
// One Game owns:
//   - a sim::SimWorld (which owns the actor::World)
//   - a GameState (Stopped / Playing / Paused)
//   - the begin_play / end_play protocol
//
// State transitions:
//
//   Stopped  --start_play()-->  Playing
//                                  ^ pause_play()
//   Paused   <----------------- Playing
//                ^ resume_play()
//
//   Playing  --stop_play()---->  Stopped (also end_play on every GameActor)
//   Paused   --stop_play()---->  Stopped
//
// While Stopped: the SimWorld still ticks (so the editor renders + camera
// works), but Update / PrePhysics / Physics / PostPhysics handlers are
// suppressed. While Paused: only Render runs (existing SimWorld behaviour).
// While Playing: everything runs.
//
// Hosts can subscribe state-change callbacks for analytics / save points.
// =============================================================================

#include <cardinal/actor/world.hpp>
#include <cardinal/core/types.hpp>
#include <cardinal/core/std/utility.hpp>
#include <cardinal/game/game_actor.hpp>
#include <cardinal/sim/sim.hpp>

namespace cardinal::game {

enum class GameState : u32 { Stopped = 0, Playing = 1, Paused = 2 };
const char* game_state_name(GameState s) noexcept;

class Game {
public:
    explicit Game(cardinal::sim::SimWorld& sim);
    ~Game();
    Game(const Game&) = delete;
    Game& operator=(const Game&) = delete;

    cardinal::sim::SimWorld&     sim()        noexcept { return sim_; }
    cardinal::actor::World&      world()      noexcept { return sim_.world(); }
    GameState                    state() const noexcept { return state_; }

    // ---- State transitions -----------------------------------------
    void start_play();
    void stop_play();
    void pause_play();
    void resume_play();
    // True when the state is Playing OR Paused (i.e. we've started a
    // game session — convenience for editor enable/disable).
    bool game_active() const noexcept;

    // ---- Spawning by class -----------------------------------------
    //
    // spawn_class(class_name) creates a fresh actor::Actor wrapped around
    // a GameActor instance produced by the registered ClassDef. Returns
    // the host actor, or nullptr if the class isn't registered.
    cardinal::actor::Actor* spawn_class(const cardinal::string& class_name,
                                       const cardinal::string& actor_name = "");

    // ---- State-change subscription ---------------------------------
    using OnStateChange = cardinal::function<void(GameState old_, GameState now)>;
    void set_on_state_change(OnStateChange cb) { on_change_ = cardinal::move(cb); }

    // ---- Per-frame -------------------------------------------------
    // Drives the SimWorld with the right tick-group gating. Hosts call
    // this instead of sim.tick() once Game is in the loop.
    float tick(float real_dt);

    // Stats for the panel.
    u32 game_actor_count() const;
    u32 begin_play_pending() const noexcept { return begin_play_pending_; }

private:
    void apply_lifecycle_();      // catches up new GameActors that were spawned mid-Playing
    void broadcast_begin_play_();
    void broadcast_end_play_();

    cardinal::sim::SimWorld&  sim_;
    GameState                 state_{GameState::Stopped};
    OnStateChange             on_change_{};
    cardinal::sim::SimWorld::HandlerId pre_h_{0};
    cardinal::sim::SimWorld::HandlerId phys_h_{0};
    cardinal::sim::SimWorld::HandlerId post_h_{0};
    cardinal::sim::SimWorld::HandlerId update_h_{0};

    // Counter for "actors spawned that need begin_play but the next sim
    // tick hasn't fired yet". Surface in the panel for diagnostics.
    u32 begin_play_pending_{0};
};

}  // namespace cardinal::game
