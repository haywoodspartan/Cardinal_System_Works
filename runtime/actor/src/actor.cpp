#include <cardinal/actor/actor.hpp>

namespace cardinal::actor {

Actor::Actor(ActorId id, cardinal::string name) noexcept
    : id_(id), name_(cardinal::move(name)) {}

Actor::~Actor() {
    for (auto& c : components_) c->on_detach(*this);
    components_.clear();
}

void Actor::tick(float dt) {
    if (!alive_) return;
    for (auto& c : components_) c->on_tick(*this, dt);
}

}  // namespace cardinal::actor
