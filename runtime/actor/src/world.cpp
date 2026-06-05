#include <cardinal/actor/world.hpp>

#include <cardinal/core/log.hpp>

#include <cardinal/core/algorithm.hpp>   // cardinal::sort/remove_if
// cardinal::make_unique/move/any/vector arrive via actor/world.hpp

namespace cardinal::actor {

World::World() {
    actors_.reserve(64);
}
World::~World() = default;

Actor* World::spawn_bare_(cardinal::string name) {
    auto a = cardinal::make_unique<Actor>(next_id_++, cardinal::move(name));
    Actor* raw = a.get();
    actors_.push_back(cardinal::move(a));
    return raw;
}

Actor* World::spawn(cardinal::string name) {
    Actor* raw = spawn_bare_(cardinal::move(name));
    // Every actor gets a TransformComponent by default.
    raw->add_component<TransformComponent>();
    return raw;
}

Actor* World::spawn_blueprint(const cardinal::string& blueprint_name) {
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
        cardinal::remove_if(actors_.begin(), actors_.end(),
                       [](const cardinal::unique_ptr<Actor>& a){ return !a->alive(); }),
        actors_.end());
}

Actor* World::find(ActorId id) {
    for (auto& a : actors_) if (a->id() == id) return a.get();
    return nullptr;
}
const Actor* World::find(ActorId id) const {
    return const_cast<World*>(this)->find(id);
}

Actor* World::find_by_name(const cardinal::string& name) {
    for (auto& a : actors_) if (a->name() == name) return a.get();
    return nullptr;
}

cardinal::vector<Actor*> World::find_by_tag(const cardinal::string& tag) {
    cardinal::vector<Actor*> r;
    for (auto& a : actors_) {
        if (auto* tc = a->get_component<TagComponent>()) {
            if (tc->has(tag)) r.push_back(a.get());
        }
    }
    return r;
}

usize World::actor_count() const noexcept { return actors_.size(); }

void World::tick(float dt) {
    // Live actors only — destroyed actors are swept later. Iterate by
    // INDEX with a snapshot count, NOT a range-for: a ticking actor
    // very commonly spawns new actors, and World::spawn does
    // actors_.push_back which can REALLOCATE this vector mid-iteration.
    // A range-for/iterator would then dangle into the freed old buffer
    // — whose unique_ptrs were moved-from to null by the realloc — and
    // the next `a->alive()` is a null deref (crash). operator[] re-reads
    // actors_.data() each step (realloc-safe); the pre-spawn count `n`
    // defers freshly-spawned actors to next frame (already the intended
    // begin_play-before-first-tick semantics).
    for (usize i = 0, n = actors_.size(); i < n; ++i) {
        Actor* a = actors_[i].get();
        if (!a->alive()) continue;
        a->tick(dt);
    }
}

void World::register_blueprint(Blueprint bp) {
    auto name = bp.name;
    blueprints_[name] = cardinal::move(bp);
}
void World::unregister_blueprint(const cardinal::string& name) {
    blueprints_.erase(name);
}
const Blueprint* World::find_blueprint(const cardinal::string& name) const {
    auto it = blueprints_.find(name);
    return it == blueprints_.end() ? nullptr : &it->second;
}
cardinal::vector<cardinal::string> World::blueprint_names() const {
    cardinal::vector<cardinal::string> r;
    r.reserve(blueprints_.size());
    for (const auto& [n, _] : blueprints_) r.push_back(n);
    cardinal::sort(r.begin(), r.end());
    return r;
}

// ---- Prefabs ----------------------------------------------------------
bool World::create_prefab(const cardinal::string& name, ActorId source) {
    Actor* src = find(source);
    if (src == nullptr) {
        cardinal::log::warnf("actor/world",
            "create_prefab('%s'): source actor %u not found",
            name.c_str(), source);
        return false;
    }
    // Detached prototype: id 0, never pushed to actors_, never ticked.
    auto proto = cardinal::make_unique<Actor>(0u, name);
    u32 skipped = 0;
    const u32 cloned = src->clone_components_into(*proto, &skipped);
    if (cloned == 0) {
        cardinal::log::warnf("actor/world",
            "create_prefab('%s'): source actor %u had no cloneable components "
            "(%u skipped) — prefab not created", name.c_str(), source, skipped);
        return false;
    }
    if (skipped > 0) {
        cardinal::log::infof("actor/world",
            "create_prefab('%s'): captured %u component(s), skipped %u "
            "without clone() override", name.c_str(), cloned, skipped);
    }
    prefabs_[name] = cardinal::move(proto);
    return true;
}

Actor* World::spawn_prefab(const cardinal::string& name,
                           const cardinal::string& instance_name) {
    auto it = prefabs_.find(name);
    if (it == prefabs_.end()) {
        cardinal::log::warnf("actor/world",
            "spawn_prefab('%s'): no such prefab", name.c_str());
        return nullptr;
    }
    cardinal::string inst = instance_name.empty()
        ? (name + " (instance)")
        : instance_name;
    // Bare actor — no auto-Transform; the prototype carries its own.
    Actor* a = spawn_bare_(cardinal::move(inst));
    it->second->clone_components_into(*a, nullptr);
    return a;
}

bool World::has_prefab(const cardinal::string& name) const {
    return prefabs_.find(name) != prefabs_.end();
}

void World::remove_prefab(const cardinal::string& name) {
    prefabs_.erase(name);
}

cardinal::vector<cardinal::string> World::prefab_names() const {
    cardinal::vector<cardinal::string> r;
    r.reserve(prefabs_.size());
    for (const auto& [n, _] : prefabs_) r.push_back(n);
    cardinal::sort(r.begin(), r.end());
    return r;
}

u32 World::prefab_component_count(const cardinal::string& name) const {
    auto it = prefabs_.find(name);
    if (it == prefabs_.end()) return 0;
    return static_cast<u32>(it->second->components().size());
}

World::HandlerId World::subscribe(const cardinal::string& event, EventFn fn) {
    Sub s{ next_handler_id_++, cardinal::move(fn) };
    const HandlerId id = s.id;
    subscribers_[event].push_back(cardinal::move(s));
    return id;
}
void World::unsubscribe(HandlerId id) {
    for (auto& [_, list] : subscribers_) {
        list.erase(cardinal::remove_if(list.begin(), list.end(),
            [id](const Sub& s){ return s.id == id; }), list.end());
    }
}
void World::broadcast(const cardinal::string& event, const cardinal::any& payload) {
    auto it = subscribers_.find(event);
    if (it == subscribers_.end()) return;
    // Snapshot the subscriber list (and decouple from the outer
    // unordered_map iterator) before invoking callbacks. A handler
    // that calls subscribe(...) into the SAME event would push_back
    // and potentially realloc the inner vector → range-for dangle;
    // a handler that calls subscribe to a DIFFERENT event could
    // rehash subscribers_ → `it` itself dangles → UAF on the next
    // iteration's `it->second` deref. Same range-for-over-mutating-
    // container UAF class as sim 1f10242, actor::World::tick
    // f3ed9c1, game 5057580, partition 309abdf. Contract: handlers
    // added or removed during a broadcast take effect on the NEXT
    // broadcast — matches the actor/game/sim spawn-during-tick
    // contract. Cost: one EventFn copy per registered handler for
    // this event (typically a handful), negligible vs the work each
    // handler does.
    auto snapshot = it->second;
    for (auto& s : snapshot) if (s.fn) s.fn(payload);
}

}  // namespace cardinal::actor
