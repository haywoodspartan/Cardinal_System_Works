#pragma once

// =============================================================================
// Cardinal core — variadic args (<cstdarg>) include site (FOUNDATION RULE).
//
// va_start / va_arg / va_end / va_copy are preprocessor macros and cannot
// be hoisted into the cardinal:: namespace; the rule this satisfies is that
// the <cstdarg> *include* lives only in cardinal::core. The `va_list` TYPE
// is aliased. Prefer typed variadics / cardinal::log formatting over raw
// C varargs in new code.
// =============================================================================

#include <cardinal/core/types.hpp>   // MASTER typedef header (FOUNDATION RULE)

#include <cstdarg>

namespace cardinal {

using std::va_list;

}  // namespace cardinal
