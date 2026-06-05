#include <cardinal/actor/world.hpp>

#include <cardinal/core/log.hpp>

#include <cardinal/core/algorithm.hpp>   // cardinal::sort/remove_if
#include <cardinal/core/cstdio.hpp>      // cardinal::snprintf (duplicate naming)
#include <cardinal/core/cmath.hpp>       // cardinal::round (grid snap)
// cardinal::make_unique/move/any/vector arrive via actor/world.hpp

namespace cardinal::actor {

World::World() {
    actors_.reserve(64);
}
World::~World() = default;

Actor* World::spawn_bare_(cardinal::string name) {
    auto a = cardinal::make_unique<Actor>(next_id_++, cardinal::move(name));
    Actor* raw = a.get();
    by_id_[raw->id()] = raw;        // index for O(1) find (ids never reused)
    actors_.push_back(cardinal::move(a));
    ++revision_;   // creation is an authored-scene change (covers spawn /
                   // spawn_prefab / duplicate, which all route through here)
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

cardinal::vector<Actor*> World::stamp_copies_(
    Actor& src, const cardinal::vector<cardinal::scene::Vec3>& offsets) {
    cardinal::vector<Actor*> out;

    // Clamp to a sane maximum (a huge offset list would reserve + spawn
    // unbounded; the linear/grid generators bound their own input too).
    constexpr u32 kMaxArray = 4096u;   // e.g. a 64x64 grid
    cardinal::usize n = offsets.size();
    if (n > kMaxArray) n = kMaxArray;
    if (n == 0) return out;

    // Source position + state, captured BEFORE any spawn (the spawns below
    // push_back to actors_, which can reallocate and invalidate `src`).
    cardinal::scene::Vec3 base{0, 0, 0};
    if (auto* st = src.get_component<TransformComponent>()) base = st->translation;
    const bool src_enabled = src.enabled();
    // Snapshot the source's components ONCE into a detached prototype, then
    // stamp each copy from it — O(n) instead of re-cloning the (moving)
    // source per copy. Strip a trailing "(copy...)" so array names are clean.
    auto proto = cardinal::make_unique<Actor>(0u, src.name());
    src.clone_components_into(*proto, nullptr);
    cardinal::string nm = src.name();
    const auto paren = nm.rfind(" (copy");
    if (paren != cardinal::string::npos && !nm.empty() && nm.back() == ')')
        nm = nm.substr(0, paren);

    out.reserve(n);
    for (cardinal::usize i = 0; i < n; ++i) {
        char suffix[40];
        cardinal::snprintf(suffix, sizeof(suffix), " (copy %zu)", i + 1);
        Actor* dst = spawn_bare_(nm + suffix);   // no auto-Transform; clone carries it
        proto->clone_components_into(*dst, nullptr);
        dst->set_enabled(src_enabled);
        if (auto* t = dst->get_component<TransformComponent>()) {
            t->translation = { base.x + offsets[i].x,
                               base.y + offsets[i].y,
                               base.z + offsets[i].z };
        }
        out.push_back(dst);
    }
    return out;
}

cardinal::vector<Actor*> World::array_actor(ActorId src_id, u32 count,
                                            const cardinal::scene::Vec3& step) {
    Actor* src = find(src_id);
    if (src == nullptr || count == 0) return {};
    // Linear offsets: copy i (1-based) at step*i.
    constexpr u32 kMaxArray = 4096u;
    if (count > kMaxArray) count = kMaxArray;
    cardinal::vector<cardinal::scene::Vec3> offsets;
    offsets.reserve(count);
    for (u32 i = 1; i <= count; ++i) {
        const float fi = static_cast<float>(i);
        offsets.push_back({ step.x * fi, step.y * fi, step.z * fi });
    }
    return stamp_copies_(*src, offsets);
}

cardinal::vector<Actor*> World::array_grid(ActorId src_id, u32 nx, u32 ny, u32 nz,
                                           const cardinal::scene::Vec3& spacing) {
    Actor* src = find(src_id);
    if (src == nullptr) return {};
    if (nx == 0) nx = 1;
    if (ny == 0) ny = 1;
    if (nz == 0) nz = 1;
    // Bound the product up front so the triple loop can't generate a
    // gigantic offset list (stamp_copies_ clamps the spawn count too).
    constexpr u32 kMaxCells = 4097u;            // 4096 copies + the origin cell
    if (static_cast<cardinal::u64>(nx) * ny * nz > kMaxCells) {
        // Shrink the largest axis until the product fits (keeps a usable grid).
        while (static_cast<cardinal::u64>(nx) * ny * nz > kMaxCells) {
            if (nx >= ny && nx >= nz && nx > 1) --nx;
            else if (ny >= nz && ny > 1)        --ny;
            else if (nz > 1)                    --nz;
            else break;
        }
    }
    // Lattice offsets, skipping (0,0,0) — that cell is the source itself.
    cardinal::vector<cardinal::scene::Vec3> offsets;
    offsets.reserve(static_cast<cardinal::usize>(nx) * ny * nz);
    for (u32 i = 0; i < nx; ++i)
        for (u32 j = 0; j < ny; ++j)
            for (u32 k = 0; k < nz; ++k) {
                if (i == 0 && j == 0 && k == 0) continue;
                offsets.push_back({ spacing.x * static_cast<float>(i),
                                    spacing.y * static_cast<float>(j),
                                    spacing.z * static_cast<float>(k) });
            }
    return stamp_copies_(*src, offsets);
}

void World::destroy(ActorId id) {
    // Only bump the revision on a genuine alive->dead transition (mirrors
    // bulk_destroy) — destroying an already-dead-but-unswept actor is a
    // no-op and must not spawn a spurious undo checkpoint.
    for (auto& a : actors_)
        if (a->id() == id) { if (a->alive()) { a->kill(); ++revision_; } return; }
}

void World::sweep() {
    const usize before = actors_.size();
    // Drop the swept actors' ids from the index first (the unique_ptrs — and
    // thus the Actor objects the map points at — are about to be destroyed).
    for (const auto& a : actors_)
        if (!a->alive()) by_id_.erase(a->id());
    actors_.erase(
        cardinal::remove_if(actors_.begin(), actors_.end(),
                       [](const cardinal::unique_ptr<Actor>& a){ return !a->alive(); }),
        actors_.end());
    if (actors_.size() != before) ++revision_;   // only when something left
}

// ---- Bulk operations --------------------------------------------------
u32 World::bulk_set_enabled(const cardinal::vector<ActorId>& ids, bool enabled) {
    u32 n = 0;
    for (ActorId id : ids) {
        if (Actor* a = find(id)) { a->set_enabled(enabled); ++n; }
    }
    if (n) ++revision_;
    return n;
}

u32 World::bulk_destroy(const cardinal::vector<ActorId>& ids) {
    u32 n = 0;
    for (ActorId id : ids) {
        if (Actor* a = find(id)) { if (a->alive()) { a->kill(); ++n; } }
    }
    if (n) ++revision_;
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
    if (n) ++revision_;
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
    if (n) ++revision_;
    return n;
}

// ---- Alignment / distribution -----------------------------------------
namespace {
float& axis_ref(cardinal::scene::Vec3& v, World::Axis a) {
    switch (a) {
        case World::Axis::X: return v.x;
        case World::Axis::Y: return v.y;
        default:             return v.z;
    }
}
}  // namespace

u32 World::align_actors(const cardinal::vector<ActorId>& ids, Axis axis, AlignMode mode) {
    // Gather the transforms that exist.
    cardinal::vector<TransformComponent*> trs;
    trs.reserve(ids.size());
    for (ActorId id : ids) {
        if (Actor* a = find(id)) {
            if (auto* t = a->get_component<TransformComponent>()) trs.push_back(t);
        }
    }
    if (trs.empty()) return 0;

    float lo = axis_ref(trs[0]->translation, axis);
    float hi = lo;
    for (auto* t : trs) {
        const float v = axis_ref(t->translation, axis);
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    const float target = (mode == AlignMode::Min)    ? lo
                       : (mode == AlignMode::Max)    ? hi
                       : 0.5f * (lo + hi);            // Center

    u32 n = 0;
    for (auto* t : trs) { axis_ref(t->translation, axis) = target; ++n; }
    if (n) ++revision_;
    return n;
}

u32 World::distribute_actors(const cardinal::vector<ActorId>& ids, Axis axis) {
    cardinal::vector<TransformComponent*> trs;
    trs.reserve(ids.size());
    for (ActorId id : ids) {
        if (Actor* a = find(id)) {
            if (auto* t = a->get_component<TransformComponent>()) {
                // Skip non-finite axis values: a NaN breaks the sort
                // comparator's strict-weak-ordering, which is UB for
                // cardinal::sort (can corrupt the heap). A corrupt
                // transform simply isn't distributed.
                if (cardinal::isfinite(axis_ref(t->translation, axis)))
                    trs.push_back(t);
            }
        }
    }
    if (trs.size() < 3) return 0;   // endpoints + ≥1 interior needed

    // Sort by axis position so the extremes are the ends. All values are
    // finite here, so the predicate is a valid strict-weak-ordering.
    cardinal::sort(trs.begin(), trs.end(),
        [axis](TransformComponent* a, TransformComponent* b) {
            return axis_ref(a->translation, axis) < axis_ref(b->translation, axis);
        });
    const usize last = trs.size() - 1;
    const float lo = axis_ref(trs[0]->translation, axis);
    const float hi = axis_ref(trs[last]->translation, axis);

    u32 n = 0;
    for (usize i = 1; i < last; ++i) {
        const float f = static_cast<float>(i) / static_cast<float>(last);
        axis_ref(trs[i]->translation, axis) = lo + (hi - lo) * f;
        ++n;
    }
    if (n) ++revision_;
    return n;
}

float snap_to_grid(float v, float step) noexcept {
    if (step <= 0.0f) return v;
    return step * cardinal::round(v / step);
}

u32 World::snap_actors_to_grid(const cardinal::vector<ActorId>& ids, float step) {
    if (step <= 0.0f) return 0;
    u32 n = 0;
    for (ActorId id : ids) {
        Actor* a = find(id);
        if (a == nullptr) continue;
        if (auto* t = a->get_component<TransformComponent>()) {
            t->translation.x = snap_to_grid(t->translation.x, step);
            t->translation.y = snap_to_grid(t->translation.y, step);
            t->translation.z = snap_to_grid(t->translation.z, step);
            ++n;
        }
    }
    if (n) ++revision_;
    return n;
}

Actor* World::find(ActorId id) {
    auto it = by_id_.find(id);   // O(1) — was a linear scan over actors_
    return it == by_id_.end() ? nullptr : it->second;
}
const Actor* World::find(ActorId id) const {
    auto it = by_id_.find(id);
    return it == by_id_.end() ? nullptr : it->second;
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
    // Drop any PrefabLink that came from the prototype (create_prefab strips
    // it on capture, but a prototype installed via add_prefab — e.g. a
    // hand-edited prefab file — could carry one), then tag this instance
    // with exactly one fresh link to its source prefab.
    a->remove_component("PrefabLink");
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

bool World::rename_prefab(const cardinal::string& old_name,
                          const cardinal::string& new_name) {
    if (new_name.empty() || new_name == old_name) return false;
    auto it = prefabs_.find(old_name);
    if (it == prefabs_.end()) return false;          // nothing to rename
    if (prefabs_.find(new_name) != prefabs_.end()) return false;  // target taken
    auto proto = cardinal::move(it->second);
    proto->set_name(new_name);                       // keep prototype name in sync
    prefabs_.erase(it);
    prefabs_[new_name] = cardinal::move(proto);
    return true;
}

bool World::duplicate_prefab(const cardinal::string& name,
                             const cardinal::string& new_name) {
    auto it = prefabs_.find(name);
    if (it == prefabs_.end()) return false;          // no source

    cardinal::string target = new_name;
    if (target.empty()) {
        // Auto-unique "<name> (copy)" / "(copy N)" (mirrors actor duplicate).
        cardinal::string base = name;
        const auto paren = base.rfind(" (copy");
        if (paren != cardinal::string::npos && base.back() == ')')
            base = base.substr(0, paren);
        target = base + " (copy)";
        for (u32 n = 2; prefabs_.find(target) != prefabs_.end(); ++n) {
            char suffix[40];
            cardinal::snprintf(suffix, sizeof(suffix), " (copy %u)", n);
            target = base + suffix;
        }
    } else if (prefabs_.find(target) != prefabs_.end()) {
        return false;                                // explicit name already taken
    }

    // Clone the prototype's components into a fresh detached prototype.
    auto proto = cardinal::make_unique<Actor>(0u, target);
    it->second->clone_components_into(*proto, nullptr);
    prefabs_[target] = cardinal::move(proto);
    return true;
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

// ---- Group prefabs ----------------------------------------------------
bool World::create_group_prefab(const cardinal::string& name,
                                const cardinal::vector<ActorId>& ids) {
    // Gather the valid (alive) source actors in the given order.
    cardinal::vector<Actor*> srcs;
    srcs.reserve(ids.size());
    for (ActorId id : ids) {
        if (Actor* a = find(id)) { if (a->alive()) srcs.push_back(a); }
    }
    if (srcs.empty()) {
        cardinal::log::warnf("actor/world",
            "create_group_prefab('%s'): no valid actors in the set", name.c_str());
        return false;
    }

    // Anchor = the first actor's position; members store positions relative
    // to it so the cluster keeps its shape wherever it's stamped.
    cardinal::scene::Vec3 anchor{0, 0, 0};
    if (auto* t0 = srcs[0]->get_component<TransformComponent>()) anchor = t0->translation;

    GroupPrefab_ gp;
    gp.members.reserve(srcs.size());
    for (Actor* a : srcs) {
        GroupMember_ m;
        m.proto = cardinal::make_unique<Actor>(0u, a->name());
        // Clone components EXCEPT a PrefabLink (a group template stores
        // config, not instance lineage) — mirrors create_prefab.
        for (const auto& c : a->components()) {
            if (cardinal::strcmp(c->type_name(), "PrefabLink") == 0) continue;
            if (auto copy = c->clone()) m.proto->adopt_component(cardinal::move(copy));
        }
        cardinal::scene::Vec3 pos{0, 0, 0};
        if (auto* t = a->get_component<TransformComponent>()) pos = t->translation;
        m.rel = { pos.x - anchor.x, pos.y - anchor.y, pos.z - anchor.z };
        gp.members.push_back(cardinal::move(m));
    }
    group_prefabs_[name] = cardinal::move(gp);
    return true;
}

cardinal::vector<Actor*> World::spawn_group_prefab(const cardinal::string& name,
                                                   const cardinal::scene::Vec3& at) {
    cardinal::vector<Actor*> out;
    auto it = group_prefabs_.find(name);
    if (it == group_prefabs_.end()) {
        cardinal::log::warnf("actor/world",
            "spawn_group_prefab('%s'): no such group", name.c_str());
        return out;
    }
    out.reserve(it->second.members.size());
    for (const auto& m : it->second.members) {
        Actor* dst = spawn_bare_(m.proto->name());   // no auto-Transform
        m.proto->clone_components_into(*dst, nullptr);
        if (auto* t = dst->get_component<TransformComponent>()) {
            t->translation = { at.x + m.rel.x, at.y + m.rel.y, at.z + m.rel.z };
        }
        out.push_back(dst);
    }
    return out;
}

bool World::has_group_prefab(const cardinal::string& name) const {
    return group_prefabs_.find(name) != group_prefabs_.end();
}

void World::remove_group_prefab(const cardinal::string& name) {
    group_prefabs_.erase(name);
}

cardinal::vector<cardinal::string> World::group_prefab_names() const {
    cardinal::vector<cardinal::string> r;
    r.reserve(group_prefabs_.size());
    for (const auto& [n, _] : group_prefabs_) r.push_back(n);
    cardinal::sort(r.begin(), r.end());
    return r;
}

u32 World::group_prefab_member_count(const cardinal::string& name) const {
    auto it = group_prefabs_.find(name);
    return it == group_prefabs_.end() ? 0u
         : static_cast<u32>(it->second.members.size());
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
    // clear_components fires on_detach for each discarded component before
    // destroying it (matching destroy/remove semantics).
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
