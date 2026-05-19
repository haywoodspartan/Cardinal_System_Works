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
    // Index-snapshot, not range-for: a component's on_tick can add a
    // component to this same actor (add_component → components_.push_back
    // → realloc), which would dangle a range-for iterator into the freed
    // old buffer (moved-from-null unique_ptrs → null deref). operator[]
    // re-reads components_.data() each step; the pre-add count `n` defers
    // a just-added component's first on_tick to next frame.
    for (usize i = 0, n = components_.size(); i < n; ++i)
        components_[i]->on_tick(*this, dt);
}

}  // namespace cardinal::actor
