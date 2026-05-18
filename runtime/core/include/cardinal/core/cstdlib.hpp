#pragma once

// =============================================================================
// Cardinal core — C stdlib (<cstdlib>) vocabulary (single sanctioned include
// site per the FOUNDATION RULE). Raw allocation should prefer the cardinal
// allocators in core/memory.hpp; this exposes process / env / parse helpers.
// (std::abs lives in core/cmath.hpp to keep one overload set.)
// =============================================================================

#include <cstdlib>

namespace cardinal {

using std::malloc;
using std::calloc;
using std::realloc;
using std::free;
using std::abort;
using std::exit;
using std::quick_exit;
using std::atexit;
using std::getenv;
using std::system;
using std::atoi;
using std::atol;
using std::atoll;
using std::atof;
using std::strtol;
using std::strtoul;
using std::strtoll;
using std::strtoull;
using std::strtof;
using std::strtod;
using std::qsort;
using std::bsearch;

}  // namespace cardinal
