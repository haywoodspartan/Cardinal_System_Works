#pragma once

// =============================================================================
// Cardinal core — <any> vocabulary (single sanctioned include site per the
// FOUNDATION RULE). Type-erased payload used by event/blueprint plumbing.
// =============================================================================

#include <cardinal/core/types.hpp>   // MASTER typedef header (FOUNDATION RULE)

#include <any>

namespace cardinal {

using std::any;
using std::any_cast;
using std::make_any;
using std::bad_any_cast;

}  // namespace cardinal
