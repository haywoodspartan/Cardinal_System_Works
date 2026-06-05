#pragma once

// =============================================================================
// Cardinal — built-in starter prefabs.
//
// register_builtin_prefabs(world) installs a small library of ready-to-use
// prefab templates into a World's prefab table, so a creator can immediately
// stamp common objects (lights, a physics prop, a camera, gameplay markers)
// from the Studio Prefab panel without authoring them from scratch.
//
// Each is built as a detached prototype actor (Transform + the type's
// components) and registered via World::add_prefab. They live in the same
// table as designer-captured prefabs — spawn / delete / save all work
// identically. Re-running is idempotent in effect (add_prefab replaces a
// same-name entry), but the helper SKIPS any name a designer already
// defined so it never clobbers user content.
//
// The canonical names (stable — referenced by tests + the panel):
//   "Point Light", "Directional Light", "Spot Light",
//   "Physics Cube", "Camera", "Trigger Volume", "Player Start"
// =============================================================================

#include <cardinal/core/types.hpp>

namespace cardinal::actor {

class World;

// Install the starter prefab library into `world`. Existing prefabs of the
// same name are left untouched (designer content wins). Returns the number
// of prefabs newly registered.
u32 register_builtin_prefabs(World& world);

}  // namespace cardinal::actor
