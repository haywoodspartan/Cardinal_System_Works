#include <cardinal/actor/world.hpp>

#include <cardinal/core/log.hpp>

#include <algorithm>

namespace cardinal::actor {

World::World() {
    actors_.reserve(64);
}
World::~World() = default;

Actor* World::spawn(std::string name) {
    auto a = std::make_unique<Actor>(next_id_++, std::move(name));
    Actor* raw = a.get();
    // Every actor gets a TransformComponent by default.
    raw->add_component<TransformComponent>();
    actors_.push_back(std::move(a));
    return raw;
}

Actor* World::spawn_blueprint(const std::string& blueprint_name) {
    auto it = blueprints_.find(blueprint_name);
    if (it == blueprints_.end()) {
        cardinal::log::warnf("actor/world",
            "spawn_blueprint('%s'): no such blueprint", blueprint_name.c_str());
        return nullptr;
    }
    Actor* a = spawn(blueprint_name);
    if (it->second.build) it->second.build(*a);
    return a;
}

void World::destroy(ActorId id) {
    for (auto& a : actors_) if (a->id() == id) { a->kill(); return; }
}

void World::sweep() {
    actors_.erase(
        std::remove_if(actors_.begin(), actors_.end(),
                       [](const std::unique_ptr<Actor>& a){ return !a->alive(); }),
        actors_.end());
}

Actor* World::find(ActorId id) {
    for (auto& a : actors_) if (a->id() == id) return a.get();
    return nullptr;
}
const Actor* World::find(ActorId id) const {
    return const_cast<World*>(this)->find(id);
}

Actor* World::find_by_name(const std::string& name) {
    for (auto& a : actors_) if (a->name() == name) return a.get();
    return nullptr;
}

std::vector<Actor*> World::find_by_tag(const std::string& tag) {
    std::vector<Actor*> r;
    for (auto& a : actors_) {
        if (auto* tc = a->get_component<TagComponent>()) {
            if (tc->has(tag)) r.push_back(a.get());
        }
    }
    return r;
}

usize World::actor_count() const noexcept { return actors_.size(); }

void World::tick(float dt) {
    // Live actors only — destroyed actors are swept later.
    for (auto& a : actors_) {
        if (!a->alive()) continue;
        a->tick(dt);
    }
}

void World::register_blueprint(Blueprint bp) {
    auto name = bp.name;
    blueprints_[name] = std::move(bp);
}
void World::unregister_blueprint(const std::string& name) {
    blueprints_.erase(name);
}
const Blueprint* World::find_blueprint(const std::string& name) const {
    auto it = blueprints_.find(name);
    return it == blueprints_.end() ? nullptr : &it->second;
}
std::vector<std::string> World::blueprint_names() const {
    std::vector<std::string> r;
    r.reserve(blueprints_.size());
    for (const auto& [n, _] : blueprints_) r.push_back(n);
    std::sort(r.begin(), r.end());
    return r;
}

World::HandlerId World::subscribe(const std::string& event, EventFn fn) {
    Sub s{ next_handler_id_++, std::move(fn) };
    const HandlerId id = s.id;
    subscribers_[event].push_back(std::move(s));
    return id;
}
void World::unsubscribe(HandlerId id) {
    for (auto& [_, list] : subscribers_) {
        list.erase(std::remove_if(list.begin(), list.end(),
            [id](const Sub& s){ return s.id == id; }), list.end());
    }
}
void World::broadcast(const std::string& event, const std::any& payload) {
    auto it = subscribers_.find(event);
    if (it == subscribers_.end()) return;
    for (auto& s : it->second) if (s.fn) s.fn(payload);
}

}  // namespace cardinal::actor
