#pragma once

// =============================================================================
// Cardinal core — C time (<ctime>) vocabulary (single sanctioned include
// site per the FOUNDATION RULE). The engine clock lives in core/time.hpp;
// this is the wall-clock / calendar surface used by build stamps etc.
// =============================================================================

#include <cardinal/core/types.hpp>   // MASTER typedef header (FOUNDATION RULE)

#include <ctime>

namespace cardinal {

using std::time;
using std::time_t;
using std::tm;
using std::clock;
using std::clock_t;
using std::difftime;
using std::mktime;
using std::localtime;
using std::gmtime;
using std::strftime;
using std::asctime;
using std::ctime;

}  // namespace cardinal
