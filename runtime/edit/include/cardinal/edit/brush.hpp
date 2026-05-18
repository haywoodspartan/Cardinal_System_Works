#pragma once

// =============================================================================
// Cardinal — sculpt / paint brush primitives.
//
// A brush is a tool that maps a world-space stamp (point + radius + falloff)
// onto some target field (terrain heights, vertex colours, voxel density,
// texture pixels). The Brush struct is pure config — applying it is the
// caller's job because the target field is system-specific.
//
// Falloffs:
//   None     — uniform 1.0 inside radius, 0 outside (hard edge)
//   Linear   — 1 at center, 0 at radius
//   Smooth   — smoothstep cubic (no derivative discontinuity at center)
//   Gaussian — exp(-3 * r²) — wide-but-soft, never reaches zero
//
// Modes:
//   Add      — push field upward / colour added
//   Subtract — pull field downward / colour subtracted
//   Set      — clamp field toward target value
//   Smooth   — replace each sample with a neighbourhood average
//   Erase    — restore field toward its baseline value
// =============================================================================

#include <cardinal/core/types.hpp>   // cardinal::function

namespace cardinal::edit::brush {

enum class Falloff : u32 { None, Linear, Smooth, Gaussian };
enum class Mode    : u32 { Add, Subtract, Set, Smooth, Erase };

struct Brush {
    Falloff falloff{Falloff::Smooth};
    Mode    mode{Mode::Add};
    float   radius_world{1.0f};      // world-space radius
    float   strength{1.0f};          // multiplier on dt-scaled application
    float   target_value{0.0f};      // used by Mode::Set
    float   spacing{0.25f};          // re-stamp every (spacing * radius) units
};

// weight_at(distance_from_center, brush) -> 0..1 falloff weight.
//   distance < 0           → treated as 0
//   distance >= radius     → 0
float weight_at(float distance_world, const Brush& b) noexcept;

// Convenience: stamp a brush onto a uniform 2D height grid.
//
//   heights : column-major grid of size width*height (heights[y*width + x])
//   cell    : world units per grid cell (square cells assumed)
//   origin_x/y : world coordinates of grid sample (0,0)
//   stamp_x/y : world coordinates of brush center
//   dt       : seconds elapsed since previous stamp (limits per-stamp delta)
//
// Returns the AABB of touched cells (inclusive), useful for partial uploads
// to GPU. When no cells were touched, all four out_* are set to 0.
struct CellAabb { u32 x0{0}, y0{0}, x1{0}, y1{0}; bool any{false}; };

CellAabb stamp_height_grid(float* heights, u32 width, u32 height,
                           float cell, float origin_x, float origin_y,
                           float stamp_x, float stamp_y,
                           float dt, const Brush& b) noexcept;

// Generic stamp: walk every (i,j) sample inside the brush AABB and call
// `apply(idx, weight)`. The caller decides what `idx` means (vertex id,
// texel id, voxel id, etc.) and how to mix the weight into the field.
//
// Sample positions come from a callback so the same routine works for
// regular grids, irregular vertex sets, voxel tables, etc.
//
// `world_pos` is the brush center; `radius` is the brush's effective
// radius (b.radius_world); the callback decides whether each candidate
// is in-radius using its own coordinates.
using SampleEnumFn = cardinal::function<void(
    const cardinal::function<void(u32 idx, float dist_world)>& touch)>;

void stamp_generic(const Brush& b, const SampleEnumFn& enumerate,
                   const cardinal::function<void(u32 idx, float weight)>& apply);

}  // namespace cardinal::edit::brush
