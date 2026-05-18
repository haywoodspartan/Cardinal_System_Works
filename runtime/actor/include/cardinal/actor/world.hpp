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

    // Mark for destruction; the actual remove happens on the next sweep().
    void    destroy(ActorId id);
    void    sweep();   // physically remove dead actors

    // ---- Lookup / iteration ------------------------------------------
    Actor*       find(ActorId id);
    const Actor* find(ActorId id) const;
    Actor*       find_by_name(const cardinal::string& name);
    cardinal::vector<Actor*> find_by_tag(const cardinal::string& tag);

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

    // ---- Event bus ---------------------------------------------------
    using EventFn = cardinal::function<void(const cardinal::any& payload)>;
    using HandlerId = u32;
    HandlerId subscribe(const cardinal::string& event, EventFn fn);
    void      unsubscribe(HandlerId id);
    void      broadcast(const cardinal::string& event, const cardinal::any& payload = {});

private:
    ActorId                                                  next_id_{1};
    cardinal::vector<cardinal::unique_ptr<Actor>>                      actors_;
    cardinal::unordered_map<cardinal::string, Blueprint>               blueprints_;
    struct Sub { HandlerId id; EventFn fn; };
    cardinal::unordered_map<cardinal::string, cardinal::vector<Sub>>        subscribers_;
    HandlerId                                                next_handler_id_{1};
};

}  // namespace cardinal::actor
