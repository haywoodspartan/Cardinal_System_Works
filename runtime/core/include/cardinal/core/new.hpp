#pragma once

// =============================================================================
// Cardinal core — <new> vocabulary (single sanctioned include site per the
// FOUNDATION RULE). Placement-new itself is a core-language operator (no
// alias possible / needed); this exposes the supporting library names.
// =============================================================================

#include <new>

namespace cardinal {

using std::nothrow;
using std::nothrow_t;
using std::bad_alloc;
using std::bad_array_new_length;
using std::align_val_t;
using std::launder;
using std::hardware_destructive_interference_size;
using std::hardware_constructive_interference_size;

}  // namespace cardinal
