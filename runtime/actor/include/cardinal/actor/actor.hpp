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

    void tick(float dt);

private:
    ActorId                                  id_{0};
    cardinal::string                              name_;
    ActorId                                  parent_{0};
    bool                                     alive_{true};
    cardinal::vector<cardinal::unique_ptr<Component>>  components_;
};

}  // namespace cardinal::actor
