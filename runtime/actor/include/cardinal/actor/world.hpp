#pragma once

// =============================================================================
// Cardinal — Actor World.
//
// One World owns a flat cardinal::vector<unique_ptr<Actor>>. The simulation's
// tick groups iterate over its actors. Actors are created via spawn()
// (which assigns an id) and removed via destroy() (which marks alive=false
// and the World sweeps in a later tick — defers iterator invalidation).
//
// Blueprints: a small registry of named templates. spawn_blueprint(name)
// instantiates a fresh Actor pre-populated with components.
//
// Event bus: subscribe(event_name, fn) and broadcast(event_name, payload).
// Cheap unordered_map<string, vector<callback>>; events fire synchronously.
// =============================================================================

#include <cardinal/actor/actor.hpp>
#include <cardinal/core/types.hpp>        // function/memory/string
#include <cardinal/core/any.hpp>          // cardinal::any
#include <cardinal/core/utility.hpp>      // cardinal::move
#include <cardinal/core/containers.hpp>   // unordered_map/vector

namespace cardinal::actor {

// ---------------------------------------------------------------------------
// Blueprint = "make me a new actor that looks like this".
//
// `spawn` is a factory the user provides; it fills components on the freshly
// created Actor. The World holds the function and invokes it for each call
// to spawn_blueprint(name).
// ---------------------------------------------------------------------------
struct Blueprint {
    cardinal::string                          name;
    cardinal::function<void(Actor& /*new*/)>  build;
};

class World {
public:
    World();
    ~World();
    World(const World&) = delete;
    World& operator=(const World&) = delete;

    // ---- Spawn / destroy ---------------------------------------------
    Actor*  spawn(cardinal::string name);
    Actor*  spawn_blueprint(const cardinal::string& blueprint_name);

    // Duplicate an existing actor — deep-clones EVERY component (incl. a
    // PrefabLink, so a duplicated prefab instance stays linked to the same
    // prefab, matching editor expectations) onto a fresh actor with a
    // unique "<name> (copy)" / "(copy N)" name. Runtime state resets via
    // each component's clone(). Returns nullptr if `id` is unknown. The
    // Ctrl-D primitive every editor needs.
    Actor*  duplicate(ActorId id);

    // Mark for destruction; the actual remove happens on the next sweep().
    void    destroy(ActorId id);
    void    sweep();   // physically remove dead actors

    // ---- Lookup / iteration ------------------------------------------
    Actor*       find(ActorId id);
    const Actor* find(ActorId id) const;
    Actor*       find_by_name(const cardinal::string& name);
    cardinal::vector<Actor*> find_by_tag(const cardinal::string& tag);

    // Substring name search over ALIVE actors (the Outliner search box +
    // any "find everything called X" query). Case-insensitive by default.
    // An empty `substr` returns every alive actor (the "no filter" case).
    cardinal::vector<Actor*> find_all_by_name(const cardinal::string& substr,
                                              bool case_insensitive = true);
    // Alive-filtered tag query (find_by_tag does NOT skip dead actors;
    // this one does — what the editor wants).
    cardinal::vector<Actor*> find_all_by_tag(const cardinal::string& tag);

    const cardinal::vector<cardinal::unique_ptr<Actor>>& actors() const noexcept { return actors_; }
    usize actor_count() const noexcept;

    // ---- Per-frame --------------------------------------------------
    // Call once per simulation tick. The World walks all live actors and
    // invokes their tick(dt). SimWorld typically wraps this with
    // FrameScopes for the profiler.
    void tick(float dt);

    // ---- Blueprints --------------------------------------------------
    void register_blueprint(Blueprint bp);
    void unregister_blueprint(const cardinal::string& name);
    const Blueprint* find_blueprint(const cardinal::string& name) const;
    cardinal::vector<cardinal::string> blueprint_names() const;

    // ---- Prefabs (data-driven, captured from live actors) ------------
    //
    // A Prefab is a frozen snapshot of an actor's components — the
    // game-creation workflow of "configure an actor in the editor, save
    // it as a reusable template, stamp out copies". Unlike a Blueprint
    // (a C++ build function), a Prefab is captured at runtime by deep-
    // cloning a live actor's components into a detached prototype Actor.
    //
    //   create_prefab("Crate", crate_id)  — snapshot crate_id's components
    //   spawn_prefab("Crate")             — new actor cloned from the snapshot
    //
    // Each spawn_prefab produces an independent instance: editing one
    // doesn't touch the others or the prototype. Components that don't
    // override Component::clone() are skipped during capture (logged).
    //
    // Returns true on capture success (source exists + ≥1 component cloned).
    bool create_prefab(const cardinal::string& name, ActorId source);
    // Stamp a new actor from the named prefab. Returns nullptr if no such
    // prefab. The new actor is named "<prefab> (instance)" unless
    // instance_name is given.
    Actor* spawn_prefab(const cardinal::string& name,
                        const cardinal::string& instance_name = "");
    bool   has_prefab(const cardinal::string& name) const;
    void   remove_prefab(const cardinal::string& name);
    cardinal::vector<cardinal::string> prefab_names() const;
    // Number of components captured in the named prefab (0 if absent) —
    // for the panel's "Crate (5 components)" display.
    u32    prefab_component_count(const cardinal::string& name) const;

    // Read access to a prefab's prototype Actor (its components are the
    // captured snapshot). Used by serial::save_prefabs to walk + emit each
    // component. Returns nullptr if no such prefab.
    const Actor* prefab_prototype(const cardinal::string& name) const;
    // Install a pre-built prototype directly (used by serial::load_prefabs
    // when reconstructing a prefab library from disk). Replaces any
    // existing prefab of the same name.
    void   add_prefab(cardinal::string name, cardinal::unique_ptr<Actor> prototype);

    // ---- Prefab instance linkage (Unity-style edit loop) -------------
    //
    // spawn_prefab tags each instance with a PrefabLinkComponent naming
    // its source prefab. These three operate on that link:
    //
    //   prefab_of(id)        — the prefab this actor instances ("" if none).
    //   revert_to_prefab(id) — discard the instance's local edits: clear
    //                          its components + re-clone from the prototype
    //                          (the link is preserved). Returns false if
    //                          the actor isn't a linked instance or the
    //                          prefab is gone.
    //   apply_to_prefab(id)  — push the instance's CURRENT components up
    //                          into the prototype so every future spawn
    //                          (and revert) inherits the edits. The link
    //                          component is excluded from the captured
    //                          prototype. Returns false on the same
    //                          conditions.
    cardinal::string prefab_of(ActorId id) const;
    bool             revert_to_prefab(ActorId id);
    bool             apply_to_prefab(ActorId id);

    // ---- Event bus ---------------------------------------------------
    using EventFn = cardinal::function<void(const cardinal::any& payload)>;
    using HandlerId = u32;
    HandlerId subscribe(const cardinal::string& event, EventFn fn);
    void      unsubscribe(HandlerId id);
    void      broadcast(const cardinal::string& event, const cardinal::any& payload = {});

private:
    // Spawn a bare actor (id assigned, pushed to actors_) WITHOUT the
    // default TransformComponent. spawn() layers the Transform on top;
    // spawn_prefab() skips it and clones the prefab's own components in.
    Actor* spawn_bare_(cardinal::string name);

    ActorId                                                  next_id_{1};
    cardinal::vector<cardinal::unique_ptr<Actor>>                      actors_;
    cardinal::unordered_map<cardinal::string, Blueprint>               blueprints_;
    // Prefab prototypes — detached actors (id 0, never in actors_/never
    // ticked) whose components are the captured snapshot. spawn_prefab
    // clones from these.
    cardinal::unordered_map<cardinal::string, cardinal::unique_ptr<Actor>> prefabs_;
    struct Sub { HandlerId id; EventFn fn; };
    cardinal::unordered_map<cardinal::string, cardinal::vector<Sub>>        subscribers_;
    HandlerId                                                next_handler_id_{1};
};

}  // namespace cardinal::actor
