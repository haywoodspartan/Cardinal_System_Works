#pragma once
// =============================================================================
// Cardinal — Gameplay Kit: engine-provided GameActor classes.
//
// Reusable behaviours a level designer drops onto actors from the Studio
// Class Picker ("Kit/…" categories) and tunes in the inspector through the
// reflected properties. Every class is a plain GameActor: begin_play /
// on_tick / end_play, driven by the Game while Playing, no C++ needed.
//
//   Kit/Motion
//     RotatorActor        spin about an axis (local or world space)
//     OrbiterActor        revolve around a point, optionally facing it
//     LookAtActor         face the nearest actor carrying a tag (turn-rate limited)
//     WaypointMoverActor  move along up to four waypoints (loop / ping-pong / once)
//   Kit/Lifecycle
//     LifetimeActor       destroy the owner after N seconds
//     SpawnerActor        stamp a prefab near the owner every N seconds, capped
//
// All rotation goes through actor::xf (quaternion ops), never Euler adds.
// Hosts call register_gameplay_kit() once at startup: it pins this TU into
// the link (the class registrations are static initialisers in a static
// library) and returns how many kit classes are registered.
// =============================================================================

#include <cardinal/game/game_actor.hpp>
#include <cardinal/game/reflection.hpp>
#include <cardinal/core/types.hpp>

namespace cardinal::game::kit {

// Forces the kit's registrations to be linked; returns the kit class count.
u32 register_gameplay_kit();

// The registered kit class names, in registration order.
cardinal::vector<cardinal::string> kit_class_names();

}  // namespace cardinal::game::kit
