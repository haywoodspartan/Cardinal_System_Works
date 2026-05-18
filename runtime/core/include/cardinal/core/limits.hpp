#pragma once

// =============================================================================
// Cardinal core — <limits> vocabulary (single sanctioned include site per the
// FOUNDATION RULE). Non-core code uses cardinal::numeric_limits<T>.
// =============================================================================

#include <limits>

namespace cardinal {

template <class T> using numeric_limits = std::numeric_limits<T>;
using std::float_round_style;

}  // namespace cardinal
