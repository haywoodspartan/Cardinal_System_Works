#pragma once

// =============================================================================
// Cardinal — Actor.
//
// Actor = a named, identifiable container for Components. Sits one level
// above scene::Entity (which is just a transform + mesh handle). The
// ActorWorld owns Actors; queries and ticks go through it.
//
// Actors aren't meant to be deeply hierarchical — flat lists scale well to
// thousands. Parent/child links are advisory; transforms are not inherited
// by default (matches the existing Entity policy).
// =============================================================================

#include <cardinal/actor/component.hpp>
#include <cardinal/core/types.hpp>        // memory/string
#include <cardinal/core/utility.hpp>      // cardinal::move/forward
#include <cardinal/core/cstring.hpp>      // cardinal::strcmp
#include <cardinal/core/containers.hpp>   // unordered_map/vector

namespace cardinal::actor {

using ActorId = u32;
inline constexpr ActorId kInvalidActor = 0;

class World;

class Actor {
public:
    explicit Actor(ActorId id, cardinal::string name) noexcept;
    ~Actor();
    Actor(const Actor&) = delete;
    Actor& operator=(const Actor&) = delete;

    ActorId id()    const noexcept { return id_; }
    const cardinal::string& name() const noexcept { return name_; }
    void set_name(cardinal::string n) { name_ = cardinal::move(n); }

    bool alive()   const noexcept { return alive_; }
    void kill()    noexcept { alive_ = false; }

    // Enabled = "active in the scene". A disabled actor stays alive (still
    // listed, selectable, editable, saved) but is SKIPPED by the sim tick
    // groups — no Update tick, no physics integration. The authoring
    // primitive for toggling an actor's behaviour on/off without deleting
    // it (every editor's actor "active" checkbox). Default on.
    bool enabled() const noexcept { return enabled_; }
    void set_enabled(bool e) noexcept { enabled_ = e; }

    ActorId parent() const noexcept { return parent_; }
    void set_parent(ActorId p) noexcept { parent_ = p; }

    // ---- Component management ----------------------------------------
    //
    // add_component<T>() constructs T in place and returns a pointer.
    // Each Actor may hold one component of any given type (matched by
    // type_name()).
    template <class T, class... Args>
    T* add_component(Args&&... args) {
        auto up = cardinal::make_unique<T>(cardinal::forward<Args>(args)...);
        T* raw = up.get();
        components_.push_back(cardinal::move(up));
        components_.back()->on_attach(*this);
        return raw;
    }

    // Find by static type_name() — returns nullptr if absent.
    template <class T>
    T* get_component() noexcept {
        const char* tname = T{}.type_name();   // type_name is const noexcept
        for (auto& c : components_) {
            if (c->type_name() == tname || cardinal::strcmp(c->type_name(), tname) == 0) {
                return static_cast<T*>(c.get());
            }
        }
        return nullptr;
    }
    template <class T>
    const T* get_component() const noexcept {
        return const_cast<Actor*>(this)->get_component<T>();
    }

    // Whole-component access (panel rendering, serialisation).
    const cardinal::vector<cardinal::unique_ptr<Component>>& components() const noexcept {
        return components_;
    }

    // ---- Presence / removal ------------------------------------------
    //
    // has_component / remove_component round out add_component + the
    // get_component query. Matched by type_name() (same rule as
    // get_component). remove_component fires on_detach on the removed
    // component and erases the FIRST match; returns whether one was found.
    // The editor uses has_component to grey out "Add" for types already
    // present (add_component does NOT dedup) and remove_component for the
    // per-component delete button.
    bool has_component(const char* tname) const noexcept {
        for (const auto& c : components_)
            if (cardinal::strcmp(c->type_name(), tname) == 0) return true;
        return false;
    }
    template <class T>
    bool has_component() noexcept { return get_component<T>() != nullptr; }

    bool remove_component(const char* tname) {
        for (auto it = components_.begin(); it != components_.end(); ++it) {
            if (cardinal::strcmp((*it)->type_name(), tname) == 0) {
                (*it)->on_detach(*this);
                components_.erase(it);
                return true;
            }
        }
        return false;
    }
    template <class T>
    bool remove_component() { return remove_component(T{}.type_name()); }

    // ---- Prefab cloning ----------------------------------------------
    //
    // Deep-copy every component on this actor onto `dst`, invoking each
    // clone's on_attach(dst). Components whose clone() returns nullptr
    // (no override) are skipped — `skipped` (if non-null) receives the
    // count so the caller can surface a diagnostic. Returns the number of
    // components successfully cloned. Used by World::create_prefab (this →
    // prototype) and World::spawn_prefab (prototype → fresh instance).
    u32 clone_components_into(Actor& dst, u32* skipped = nullptr) const {
        u32 cloned = 0, miss = 0;
        for (const auto& c : components_) {
            auto copy = c->clone();
            if (!copy) { ++miss; continue; }
            copy->on_attach(dst);
            dst.components_.push_back(cardinal::move(copy));
            ++cloned;
        }
        if (skipped) *skipped = miss;
        return cloned;
    }

    // Remove all components (used before stamping a prefab over a bare
    // actor so the auto-Transform doesn't duplicate the prefab's own).
    void clear_components() noexcept { components_.clear(); }

    // Adopt an already-constructed component (e.g. from the deserializer's
    // make_component_by_name factory or a game-class ClassRegistry::create).
    // Fires on_attach + returns the raw pointer. The Actor takes ownership.
    Component* adopt_component(cardinal::unique_ptr<Component> c) {
        if (!c) return nullptr;
        Component* raw = c.get();
        raw->on_attach(*this);
        components_.push_back(cardinal::move(c));
        return raw;
    }

    void tick(float dt);

private:
    ActorId                                  id_{0};
    cardinal::string                              name_;
    ActorId                                  parent_{0};
    bool                                     alive_{true};
    bool                                     enabled_{true};
    cardinal::vector<cardinal::unique_ptr<Component>>  components_;
};

}  // namespace cardinal::actor
