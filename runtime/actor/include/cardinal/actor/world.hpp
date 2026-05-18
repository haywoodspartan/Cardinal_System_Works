#pragma once

// =============================================================================
// Cardinal — Actor World.
//
// One World owns a flat std::vector<unique_ptr<Actor>>. The simulation's
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
#include <cardinal/core/types.hpp>

#include <any>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cardinal::actor {

// ---------------------------------------------------------------------------
// Blueprint = "make me a new actor that looks like this".
//
// `spawn` is a factory the user provides; it fills components on the freshly
// created Actor. The World holds the function and invokes it for each call
// to spawn_blueprint(name).
// ---------------------------------------------------------------------------
struct Blueprint {
    std::string                          name;
    std::function<void(Actor& /*new*/)>  build;
};

class World {
public:
    World();
    ~World();
    World(const World&) = delete;
    World& operator=(const World&) = delete;

    // ---- Spawn / destroy ---------------------------------------------
    Actor*  spawn(std::string name);
    Actor*  spawn_blueprint(const std::string& blueprint_name);

    // Mark for destruction; the actual remove happens on the next sweep().
    void    destroy(ActorId id);
    void    sweep();   // physically remove dead actors

    // ---- Lookup / iteration ------------------------------------------
    Actor*       find(ActorId id);
    const Actor* find(ActorId id) const;
    Actor*       find_by_name(const std::string& name);
    std::vector<Actor*> find_by_tag(const std::string& tag);

    const std::vector<std::unique_ptr<Actor>>& actors() const noexcept { return actors_; }
    usize actor_count() const noexcept;

    // ---- Per-frame --------------------------------------------------
    // Call once per simulation tick. The World walks all live actors and
    // invokes their tick(dt). SimWorld typically wraps this with
    // FrameScopes for the profiler.
    void tick(float dt);

    // ---- Blueprints --------------------------------------------------
    void register_blueprint(Blueprint bp);
    void unregister_blueprint(const std::string& name);
    const Blueprint* find_blueprint(const std::string& name) const;
    std::vector<std::string> blueprint_names() const;

    // ---- Event bus ---------------------------------------------------
    using EventFn = std::function<void(const std::any& payload)>;
    using HandlerId = u32;
    HandlerId subscribe(const std::string& event, EventFn fn);
    void      unsubscribe(HandlerId id);
    void      broadcast(const std::string& event, const std::any& payload = {});

private:
    ActorId                                                  next_id_{1};
    std::vector<std::unique_ptr<Actor>>                      actors_;
    std::unordered_map<std::string, Blueprint>               blueprints_;
    struct Sub { HandlerId id; EventFn fn; };
    std::unordered_map<std::string, std::vector<Sub>>        subscribers_;
    HandlerId                                                next_handler_id_{1};
};

}  // namespace cardinal::actor
