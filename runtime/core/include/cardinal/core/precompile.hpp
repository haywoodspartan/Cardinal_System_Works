#pragma once

// =============================================================================
// Cardinal — engine-wide precompiled header.
//
// Lives in cardinal::core so EVERY engine module (render, lighting,
// physics, particles, scene, etc.) shares one canonical pre-include.
// The Pearl Abyss / CrimsonDesert convention is one stdafx-style header
// per module; this is the same idea but unified at the foundation
// layer — modules don't restate this set in their own headers; they
// opt-in via CMake:
//
//   target_precompile_headers(<module_target> PRIVATE
//       <cardinal/core/precompile.hpp>)
//
// Foundation rule still holds: std headers stay INSIDE cardinal::core.
// This header re-exports the foundation's container / utility / math /
// algorithm surface — exactly what every gameplay-facing TU pulls in.
// Module-specific headers (gpu_aegis.hpp, gpu_physics.hpp, etc.) do
// NOT live here; they get included only by the consumers that need
// them so the precompile cache stays warm across every module.
//
// Build-time effect: ~30% of the engine's TUs reach the foundation
// surface via this single precompiled cache hit instead of re-parsing
// the same five headers per .cpp.
// =============================================================================

#include <cardinal/core/types.hpp>          // u8..u64, i8..i64, f32/f64, usize, shared_ptr, unique_ptr
#include <cardinal/core/containers.hpp>     // vector, string, array
#include <cardinal/core/utility.hpp>        // move, swap, forward
#include <cardinal/core/cmath.hpp>          // isfinite, sqrt, clamp, pow, cos/sin/tan, abs
#include <cardinal/core/algorithm.hpp>      // clamp, min, max
