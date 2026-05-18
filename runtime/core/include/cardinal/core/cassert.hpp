#pragma once

// =============================================================================
// Cardinal core — <cassert> include site (FOUNDATION RULE).
//
// `assert` is a preprocessor macro and cannot be hoisted into the cardinal::
// namespace; the rule it satisfies is that the <cassert> *include* lives
// only in cardinal::core. Prefer the engine's richer diagnostics
// (cardinal::log + crash handler) for shipping checks; reach for plain
// `assert` only in throwaway/internal invariants.
// =============================================================================

#include <cardinal/core/types.hpp>   // MASTER typedef header (FOUNDATION RULE)

#include <cassert>
