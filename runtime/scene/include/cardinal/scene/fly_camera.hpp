#pragma once

// =============================================================================
// Cardinal — fly camera controller.
//
// Standalone helper that drives a `Camera` from keyboard + mouse input.
// The sample feeds it ImGui's input state each frame; the controller
// updates the camera's `position` + `target` so the existing view matrix
// path "just works".
//
// Default bindings:
//
//   W / S            : move forward / backward
//   A / D            : strafe left / right
//   Q / E            : descend / ascend (world-Y)
//   RMB held + drag  : look around (yaw + pitch)
//   Shift held       : sprint (speed * sprint_multiplier)
//   Mouse wheel      : nudge speed (clamped to [min, max])
//
// All input is gated on `accept_input` — the sample sets this to true only
// when the Viewport panel is hovered (so typing in the Console doesn't
// accidentally fly the camera).
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/scene/math.hpp>
#include <cardinal/scene/scene.hpp>

namespace cardinal::scene {

struct FlyInput {
    bool  accept_input{false};   // gate — usually true when Viewport hovered
    bool  forward{false};        // W
    bool  backward{false};       // S
    bool  left{false};           // A
    bool  right{false};          // D
    bool  up{false};             // E
    bool  down{false};           // Q
    bool  sprint{false};         // Shift
    bool  look{false};           // RMB held
    float mouse_dx{0.0f};        // px since last frame
    float mouse_dy{0.0f};
    float scroll{0.0f};          // wheel ticks since last frame
};

class FlyCamera {
public:
    // Tunables — the sample / editor can tweak these from a settings UI.
    float speed{4.0f};               // world units / second
    float speed_min{0.25f};
    float speed_max{200.0f};
    float sprint_multiplier{4.0f};
    float look_sensitivity{0.0035f}; // radians per pixel
    float scroll_speed_scale{1.20f}; // wheel notch multiplier

    // Pitch is clamped so the camera can't flip upside-down. Yaw is free.
    float min_pitch_rad{-1.553f};    // ~-89°
    float max_pitch_rad{ 1.553f};    // ~+89°

    // Derive yaw + pitch from the camera's current forward direction so
    // existing camera state is preserved when the sample first wires us up.
    void sync_from(const Camera& cam) noexcept;

    // Apply one frame's input to the camera. dt = elapsed seconds.
    void tick(Camera& cam, const FlyInput& in, float dt) noexcept;

    // Tear off the cached yaw/pitch (e.g. for a debug overlay).
    float yaw_rad()   const noexcept { return yaw_; }
    float pitch_rad() const noexcept { return pitch_; }

private:
    float yaw_{0.0f};
    float pitch_{0.0f};
    bool  initialised_{false};
};

}  // namespace cardinal::scene
