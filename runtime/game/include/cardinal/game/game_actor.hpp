#pragma once

// =============================================================================
// Cardinal — GameActor base.
//
// A GameActor is a Component that lives on an actor::Actor. Its lifecycle
// goes:
//
//   construct                    — ctor, then on_attach (from actor::Component)
//   begin_play()                 — fired by Game when state -> Playing
//   on_tick(dt)                  — every Update tick while Playing
//   end_play()                   — fired by Game when state -> Stopped
//   destruct                     — on_detach + dtor
//
// The Game orchestrator drives begin_play / end_play. Pause does NOT call
// end_play — the actor stays in its current state but on_tick is gated.
// =============================================================================

#include <cardinal/actor/component.hpp>
#include <cardinal/core/types.hpp>
#include <cardinal/core/std/utility.hpp>

namespace cardinal::game {

class GameActor : public cardinal::actor::Component {
public:
    GameActor() = default;
    ~GameActor() override = default;

    // GameActor subclasses ALWAYS have type_name() == "GameActor" so the
    // Actor::get_component<GameActor>() helper finds them all uniformly.
    // The class identity lives in the ClassRegistry, not in C++ RTTI.
    const char* type_name() const noexcept override { return "GameActor"; }

    // The owning Actor — set by attach(). May be null pre-attach.
    cardinal::actor::Actor* owner() const noexcept { return owner_; }

    // Class name — set by Game::spawn_class so the inspector can find the
    // ClassDef and its property descriptors.
    const cardinal::string& class_name() const noexcept { return class_name_; }
    void set_class_name(cardinal::string n) { class_name_ = cardinal::move(n); }

    bool playing() const noexcept { return playing_; }

    // Lifecycle hooks — override in your subclass.
    virtual void begin_play() {}
    virtual void on_tick(float /*dt*/) {}
    virtual void end_play() {}

    // ---------- Component overrides ----------------------------------
    void on_attach(cardinal::actor::Actor& a) override { owner_ = &a; }
    void on_detach(cardinal::actor::Actor&) override   { owner_ = nullptr; }

    // Deep-copy for the Prefab system. Re-creates the correct subclass via
    // the ClassRegistry (keyed by class_name_), copies the class name, and
    // copies every reflected property value source -> clone by kind.
    // Runtime state (owner_, playing_) is NOT copied — a fresh instance is
    // unattached + not playing until the world + Game wire it up. Returns
    // nullptr if the class isn't registered (out-of-line: needs the
    // ClassRegistry, kept out of this header to avoid the reflection dep).
    cardinal::unique_ptr<cardinal::actor::Component> clone() const override;

    // Friend access for Game.
    void _set_playing(bool p) noexcept { playing_ = p; }

private:
    cardinal::actor::Actor* owner_   {nullptr};
    bool                    playing_ {false};
    cardinal::string             class_name_;
};

}  // namespace cardinal::game
