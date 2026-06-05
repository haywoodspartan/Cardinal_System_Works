#include <cardinal/actor/world.hpp>

#include <cardinal/core/log.hpp>

#include <cardinal/core/algorithm.hpp>   // cardinal::sort/remove_if
#include <cardinal/core/cstdio.hpp>      // cardinal::snprintf (duplicate naming)
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

Actor* World::duplicate(ActorId id) {
    Actor* src = find(id);
    if (src == nullptr) {
        cardinal::log::warnf("actor/world", "duplicate(%u): no such actor", id);
        return nullptr;
    }
    // Unique name: "<base> (copy)", then "(copy 2)", "(copy 3)"… until free.
    // Strip an existing "(copy...)" suffix so duplicating a copy doesn't
    // accrete "(copy) (copy)".
    cardinal::string base = src->name();
    const auto paren = base.rfind(" (copy");
    if (paren != cardinal::string::npos && base.back() == ')') {
        base = base.substr(0, paren);
    }
    cardinal::string candidate = base + " (copy)";
    for (u32 n = 2; find_by_name(candidate) != nullptr; ++n) {
        char suffix[40];
        cardinal::snprintf(suffix, sizeof(suffix), " (copy %u)", n);
        candidate = base + suffix;
    }

    Actor* dst = spawn_bare_(candidate);   // no auto-Transform; clone carries it
    src->clone_components_into(*dst, nullptr);
    dst->set_enabled(src->enabled());      // carry the active/disabled state
    return dst;
}

// ---- Placement --------------------------------------------------------
namespace {
void place_at_(Actor& a, const cardinal::scene::Vec3& pos) {
    auto* tr = a.get_component<TransformComponent>();
    if (tr == nullptr) tr = a.add_component<TransformComponent>();
    tr->translation = pos;
}
}  // namespace

Actor* World::spawn_at(cardinal::string name, const cardinal::scene::Vec3& pos) {
    Actor* a = spawn(cardinal::move(name));
    place_at_(*a, pos);
    return a;
}

Actor* World::spawn_prefab_at(const cardinal::string& name,
                              const cardinal::scene::Vec3& pos,
                              const cardinal::string& instance_name) {
    Actor* a = spawn_prefab(name, instance_name);
    if (a == nullptr) return nullptr;
    place_at_(*a, pos);
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

// ---- Bulk operations --------------------------------------------------
u32 World::bulk_set_enabled(const cardinal::vector<ActorId>& ids, bool enabled) {
    u32 n = 0;
    for (ActorId id : ids) {
        if (Actor* a = find(id)) { a->set_enabled(enabled); ++n; }
    }
    return n;
}

u32 World::bulk_destroy(const cardinal::vector<ActorId>& ids) {
    u32 n = 0;
    for (ActorId id : ids) {
        if (Actor* a = find(id)) { if (a->alive()) { a->kill(); ++n; } }
    }
    return n;
}

u32 World::bulk_add_tag(const cardinal::vector<ActorId>& ids, const cardinal::string& tag) {
    if (tag.empty()) return 0;
    u32 n = 0;
    for (ActorId id : ids) {
        Actor* a = find(id);
        if (a == nullptr) continue;
        auto* tc = a->get_component<TagComponent>();
        if (tc == nullptr) tc = a->add_component<TagComponent>();
        tc->add(tag);            // add() dedupes
        ++n;
    }
    return n;
}

u32 World::bulk_remove_tag(const cardinal::vector<ActorId>& ids, const cardinal::string& tag) {
    if (tag.empty()) return 0;
    u32 n = 0;
    for (ActorId id : ids) {
        Actor* a = find(id);
        if (a == nullptr) continue;
        if (auto* tc = a->get_component<TagComponent>()) {
            if (tc->has(tag)) { tc->remove(tag); ++n; }
        }
    }
    return n;
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

namespace {
// ASCII lower — locale-independent, enough for actor names.
char ascii_lower(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}
// Case-insensitive substring test (needle already lowered when ci).
bool contains_ci(const cardinal::string& hay, const cardinal::string& needle_lc,
                 bool case_insensitive) {
    if (needle_lc.empty()) return true;
    if (hay.size() < needle_lc.size()) return false;
    const usize last = hay.size() - needle_lc.size();
    for (usize i = 0; i <= last; ++i) {
        bool ok = true;
        for (usize j = 0; j < needle_lc.size(); ++j) {
            const char h = case_insensitive ? ascii_lower(hay[i + j]) : hay[i + j];
            if (h != needle_lc[j]) { ok = false; break; }
        }
        if (ok) return true;
    }
    return false;
}
}  // namespace

cardinal::vector<Actor*> World::find_all_by_name(const cardinal::string& substr,
                                                 bool case_insensitive) {
    cardinal::string needle = substr;
    if (case_insensitive)
        for (auto& ch : needle) ch = ascii_lower(ch);
    cardinal::vector<Actor*> r;
    for (auto& a : actors_) {
        if (!a->alive()) continue;
        if (contains_ci(a->name(), needle, case_insensitive)) r.push_back(a.get());
    }
    return r;
}

cardinal::vector<Actor*> World::find_all_by_tag(const cardinal::string& tag) {
    cardinal::vector<Actor*> r;
    for (auto& a : actors_) {
        if (!a->alive()) continue;
        if (auto* tc = a->get_component<TagComponent>()) {
            if (tc->has(tag)) r.push_back(a.get());
        }
    }
    return r;
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
        if (!a->alive() || !a->enabled()) continue;   // disabled actors don't tick
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
    u32 cloned = 0, skipped = 0;
    // Filtered capture: clone every component EXCEPT a PrefabLink (a prefab
    // must not carry a link to another prefab — capturing an instance bakes
    // its config, not its lineage).
    for (const auto& c : src->components()) {
        if (cardinal::strcmp(c->type_name(), "PrefabLink") == 0) continue;
        auto copy = c->clone();
        if (!copy) { ++skipped; continue; }
        proto->adopt_component(cardinal::move(copy));
        ++cloned;
    }
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
    // Tag the instance with its prefab lineage for revert / apply.
    auto link = cardinal::make_unique<PrefabLinkComponent>();
    link->prefab_name = name;
    a->adopt_component(cardinal::move(link));
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

const Actor* World::prefab_prototype(const cardinal::string& name) const {
    auto it = prefabs_.find(name);
    return it == prefabs_.end() ? nullptr : it->second.get();
}

void World::add_prefab(cardinal::string name, cardinal::unique_ptr<Actor> prototype) {
    if (!prototype) return;
    prefabs_[cardinal::move(name)] = cardinal::move(prototype);
}

cardinal::string World::prefab_of(ActorId id) const {
    const Actor* a = find(id);
    if (a == nullptr) return {};
    if (const auto* link = a->get_component<PrefabLinkComponent>())
        return link->prefab_name;
    return {};
}

bool World::revert_to_prefab(ActorId id) {
    Actor* a = find(id);
    if (a == nullptr) return false;
    auto* link = a->get_component<PrefabLinkComponent>();
    if (link == nullptr) return false;
    const cardinal::string pname = link->prefab_name;
    auto it = prefabs_.find(pname);
    if (it == prefabs_.end()) {
        cardinal::log::warnf("actor/world",
            "revert_to_prefab(%u): linked prefab '%s' no longer exists",
            id, pname.c_str());
        return false;
    }
    // Wipe local edits, re-clone the prototype, restore the link. on_detach
    // fires for the discarded components via clear_components -> dtors.
    a->clear_components();
    it->second->clone_components_into(*a, nullptr);
    auto relink = cardinal::make_unique<PrefabLinkComponent>();
    relink->prefab_name = pname;
    a->adopt_component(cardinal::move(relink));
    return true;
}

bool World::apply_to_prefab(ActorId id) {
    const cardinal::string pname = prefab_of(id);
    if (pname.empty()) return false;
    if (prefabs_.find(pname) == prefabs_.end()) return false;
    // create_prefab re-captures the instance's CURRENT components into the
    // prototype (replacing it), excluding the PrefabLink. Every future
    // spawn + revert now inherits this instance's edits.
    return create_prefab(pname, id);
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
