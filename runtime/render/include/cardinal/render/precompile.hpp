#pragma once

// =============================================================================
// Cardinal — render module precompiled header.
//
// Per-module pre-include (Pearl Abyss / CrimsonDesert convention — each
// module has its own stdafx.h covering its hot-path includes). CMake
// flags this as a precompiled header via target_precompile_headers; every
// runtime/render/src/*.cpp picks it up implicitly so the foundation
// types (cardinal::shared_ptr, vector, string, u32, math) compile once.
//
// Module rule: this header MUST stay sub-1KB-of-includes and include
// ONLY headers that the entire render module needs. Do NOT add module-
// specific headers here (gpu_aegis.hpp etc.) — those go directly into
// the consumers that use them so the precompile cache stays warm
// across the whole module.
// =============================================================================

#include <cardinal/core/types.hpp>          // u8..u64, i8..i64, f32/f64, usize
#include <cardinal/core/containers.hpp>     // vector, string, array
#include <cardinal/core/utility.hpp>        // move, swap, forward
#include <cardinal/core/cmath.hpp>          // isfinite, sqrt, clamp, pow, cos/sin/tan, abs
#include <cardinal/core/algorithm.hpp>      // clamp, min, max
