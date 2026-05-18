#include <cardinal/game/game.hpp>

#include <cardinal/core/log.hpp>
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
        for (auto& aptr : sim_.world().actors()) {
            if (!aptr->alive()) continue;
            for (const auto& c : aptr->components()) {
                if (auto* ga = dynamic_cast<GameActor*>(c.get())) {
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

cardinal::actor::Actor* Game::spawn_class(const std::string& class_name,
                                         const std::string& actor_name)
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
    auto& comps = const_cast<std::vector<std::unique_ptr<cardinal::actor::Component>>&>(a->components());
    comps.pop_back();   // remove the placeholder we just appended
    comps.push_back(std::move(inst));
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
    for (auto& aptr : sim_.world().actors()) {
        if (!aptr->alive()) continue;
        for (const auto& c : aptr->components()) {
            if (auto* ga = dynamic_cast<GameActor*>(c.get())) {
                if (!ga->playing()) {
                    ga->begin_play();
                    ga->_set_playing(true);
                    ++fired;
                }
            }
        }
    }
    if (fired > 0) {
        begin_play_pending_ = (fired >= begin_play_pending_)
            ? 0u : (begin_play_pending_ - fired);
    }
}

void Game::broadcast_begin_play_() {
    for (auto& aptr : sim_.world().actors()) {
        for (const auto& c : aptr->components()) {
            if (auto* ga = dynamic_cast<GameActor*>(c.get())) {
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
    for (auto& aptr : sim_.world().actors()) {
        for (const auto& c : aptr->components()) {
            if (auto* ga = dynamic_cast<GameActor*>(c.get())) {
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
