#include <cardinal/game/game.hpp>

#include <cardinal/core/log.hpp>
#include <cardinal/core/utility.hpp>
#include <cardinal/game/reflection.hpp>

namespace cardinal::game {

const char* game_state_name(GameState s) noexcept {
    switch (s) {
        case GameState::Stopped: return "Stopped";
        case GameState::Playing: return "Playing";
        case GameState::Paused:  return "Paused";
    }
    return "?";
}

Game::Game(cardinal::sim::SimWorld& sim) : sim_(sim) {
    // Sim handlers — added once. They early-out based on game state so we
    // don't need to add/remove on every transition (which would race with
    // a tick already in progress).
    pre_h_   = sim_.add_handler(cardinal::sim::TickGroup::PreUpdate,  [this](float){
        if (state_ == GameState::Playing) apply_lifecycle_();
    });
    update_h_= sim_.add_handler(cardinal::sim::TickGroup::Update, [this](float dt){
        if (state_ != GameState::Playing) return;
        // Index-snapshot, NOT range-for: a GameActor::on_tick very
        // commonly spawn_class()-es (→ SimWorld::world().spawn() →
        // actors_.push_back) or add_component()s, reallocating the
        // vector mid-iteration. A range-for would then dangle into the
        // freed old buffer whose unique_ptrs were moved-from to null →
        // deterministic null-deref. Actor*/Component* stay valid across
        // realloc (heap-behind-unique_ptr); operator[] re-reads .data()
        // each step; the pre-spawn count defers fresh actors to next
        // frame (spawn_class already fired their begin_play immediately).
        const auto& acts = sim_.world().actors();
        for (cardinal::usize i = 0, n = acts.size(); i < n; ++i) {
            cardinal::actor::Actor* aptr = acts[i].get();
            if (!aptr->alive()) continue;
            const auto& comps = aptr->components();
            for (cardinal::usize j = 0, m = comps.size(); j < m; ++j) {
                if (auto* ga = dynamic_cast<GameActor*>(comps[j].get())) {
                    if (ga->playing()) ga->on_tick(dt);
                }
            }
        }
    });
}

Game::~Game() {
    if (state_ != GameState::Stopped) stop_play();
    sim_.remove_handler(pre_h_);
    sim_.remove_handler(update_h_);
}

bool Game::game_active() const noexcept {
    return state_ == GameState::Playing || state_ == GameState::Paused;
}

void Game::start_play() {
    if (state_ != GameState::Stopped) return;
    state_ = GameState::Playing;
    sim_.resume();
    broadcast_begin_play_();
    cardinal::log::infof("game", "start_play -> Playing (actors=%u)", game_actor_count());
    if (on_change_) on_change_(GameState::Stopped, state_);
}

void Game::stop_play() {
    if (state_ == GameState::Stopped) return;
    const GameState old = state_;
    state_ = GameState::Stopped;
    sim_.pause();
    broadcast_end_play_();
    cardinal::log::infof("game", "stop_play <- %s", game_state_name(old));
    if (on_change_) on_change_(old, state_);
}

void Game::pause_play() {
    if (state_ != GameState::Playing) return;
    state_ = GameState::Paused;
    sim_.pause();
    cardinal::log::infof("game", "pause_play -> Paused");
    if (on_change_) on_change_(GameState::Playing, state_);
}

void Game::resume_play() {
    if (state_ != GameState::Paused) return;
    state_ = GameState::Playing;
    sim_.resume();
    cardinal::log::infof("game", "resume_play -> Playing");
    if (on_change_) on_change_(GameState::Paused, state_);
}

cardinal::actor::Actor* Game::spawn_class(const cardinal::string& class_name,
                                         const cardinal::string& actor_name)
{
    const ClassDef* def = ClassRegistry::instance().find(class_name);
    if (def == nullptr || !def->create) {
        cardinal::log::warnf("game", "spawn_class: class '%s' not registered",
                             class_name.c_str());
        return nullptr;
    }
    cardinal::actor::Actor* a = sim_.world().spawn(
        actor_name.empty() ? class_name : actor_name);
    auto inst = def->create();
    inst->set_class_name(class_name);
    GameActor* raw = inst.get();
    a->add_component<GameActor>();   // placeholder to make get_component<GameActor>() work
    // Replace the placeholder with the real instance — we own components_
    // through the unique_ptr factory, so swap:
    // (Simpler: append the typed instance directly.)
    auto& comps = const_cast<cardinal::vector<cardinal::unique_ptr<cardinal::actor::Component>>&>(a->components());
    comps.pop_back();   // remove the placeholder we just appended
    comps.push_back(cardinal::move(inst));
    raw->on_attach(*a);
    if (state_ == GameState::Playing) {
        // Spawned mid-game: fire begin_play right away.
        raw->begin_play();
        raw->_set_playing(true);
    } else {
        ++begin_play_pending_;
    }
    return a;
}

float Game::tick(float real_dt) {
    return sim_.tick(real_dt);
}

void Game::apply_lifecycle_() {
    // Catch GameActors that were spawned while the game was already
    // Playing but haven't seen begin_play yet. Cheap O(N) sweep.
    if (begin_play_pending_ == 0) return;
    u32 fired = 0;
    // Index-snapshot (see update_h_): begin_play() can spawn_class().
    const auto& acts = sim_.world().actors();
    for (cardinal::usize i = 0, n = acts.size(); i < n; ++i) {
        cardinal::actor::Actor* aptr = acts[i].get();
        if (!aptr->alive()) continue;
        const auto& comps = aptr->components();
        for (cardinal::usize j = 0, m = comps.size(); j < m; ++j) {
            if (auto* ga = dynamic_cast<GameActor*>(comps[j].get())) {
                if (!ga->playing()) {
                    ga->begin_play();
                    ga->_set_playing(true);
                    ++fired;
                }
            }
        }
    }
    // The full sweep above visits EVERY alive actor in the world; any
    // remaining "pending" residue must be GameActors that were
    // destroyed (marked-dead or swept) before their deferred begin_
    // play could fire. The old `(fired >= pending) ? 0 : (pending -
    // fired)` arithmetic left that residue forever — apply_lifecycle_
    // then ran a no-op O(N) sweep EVERY PreUpdate frame from then on.
    // Trigger sequence (parked-supervised item #6 in feedback_
    // integration_coverage_map.md): pause_play (state=Paused) →
    // spawn_class (++pending) → world.destroy(actor) → resume_play
    // (does NOT broadcast_begin_play_, unlike start_play) → tick.
    // apply_lifecycle_ sees !alive() and skips, fired stays 0, the
    // old arithmetic left pending=1 forever. Reset to 0 unconditionally
    // after a full sweep — `++pending` re-arms us if new spawn events
    // arrive next frame.
    begin_play_pending_ = 0u;
}

void Game::broadcast_begin_play_() {
    // Index-snapshot (see update_h_): begin_play() can spawn_class();
    // a fresh actor spawned mid-broadcast already had begin_play fired
    // by spawn_class (state_==Playing), so not revisiting it here also
    // correctly avoids a double begin_play.
    const auto& acts = sim_.world().actors();
    for (cardinal::usize i = 0, n = acts.size(); i < n; ++i) {
        const auto& comps = acts[i]->components();
        for (cardinal::usize j = 0, m = comps.size(); j < m; ++j) {
            if (auto* ga = dynamic_cast<GameActor*>(comps[j].get())) {
                if (!ga->playing()) {
                    ga->begin_play();
                    ga->_set_playing(true);
                }
            }
        }
    }
    begin_play_pending_ = 0;
}

void Game::broadcast_end_play_() {
    // Index-snapshot (see update_h_): end_play() could spawn_class().
    const auto& acts = sim_.world().actors();
    for (cardinal::usize i = 0, n = acts.size(); i < n; ++i) {
        const auto& comps = acts[i]->components();
        for (cardinal::usize j = 0, m = comps.size(); j < m; ++j) {
            if (auto* ga = dynamic_cast<GameActor*>(comps[j].get())) {
                if (ga->playing()) {
                    ga->end_play();
                    ga->_set_playing(false);
                }
            }
        }
    }
}

u32 Game::game_actor_count() const {
    u32 n = 0;
    for (const auto& aptr : sim_.world().actors()) {
        for (const auto& c : aptr->components()) {
            if (dynamic_cast<const GameActor*>(c.get()) != nullptr) ++n;
        }
    }
    return n;
}

}  // namespace cardinal::game
