#pragma once

// =============================================================================
// Cardinal core — <chrono> vocabulary (single sanctioned include site per
// the FOUNDATION RULE). core/time.hpp owns the engine clock; this exposes
// raw std::chrono for the call sites that measure wall durations directly.
// =============================================================================

#include <cardinal/core/types.hpp>   // MASTER typedef header (FOUNDATION RULE)

#include <chrono>

namespace cardinal {

namespace chrono = std::chrono;

using std::chrono::steady_clock;
using std::chrono::system_clock;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::time_point;
using std::chrono::time_point_cast;
using std::chrono::nanoseconds;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::minutes;
using std::chrono::hours;

}  // namespace cardinal
