#pragma once

// =============================================================================
// Cardinal core — <climits> include site (FOUNDATION RULE).
//
// INT_MAX / CHAR_BIT / LONG_MAX / … are preprocessor macros and cannot be
// hoisted into the cardinal:: namespace; the rule this satisfies is that the
// <climits> *include* lives only in cardinal::core. Prefer the C++ surface
// cardinal::numeric_limits<T> (core/limits.hpp) in new code — reach for the
// raw C limit macros only where a macro-constant is genuinely required.
// =============================================================================

#include <cardinal/core/types.hpp>   // MASTER typedef header (FOUNDATION RULE)

#include <climits>
