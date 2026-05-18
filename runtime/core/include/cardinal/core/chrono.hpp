#pragma once

// =============================================================================
// Cardinal core — <chrono> vocabulary (single sanctioned include site per
// the FOUNDATION RULE). core/time.hpp owns the engine clock; this exposes
// raw std::chrono for the call sites that measure wall durations directly.
// =============================================================================

#include <cardinal/core/types.hpp>   // MASTER typedef header (FOUNDATION RULE)

#include <chrono>
#include <ratio>    // SI ratio typedefs are the period arg of chrono::duration
                     // — folded in here (chrono is their only real consumer),
                     // mirroring how <system_error> lives in filesystem/charconv

namespace cardinal {

namespace chrono = std::chrono;

// Compile-time ratio + the SI prefix specialisations from <ratio>. These are
// the `Period` template argument of cardinal::chrono::duration<Rep, Period>
// (e.g. duration<f32, cardinal::micro> for fractional microseconds).
using std::ratio;
using std::atto;   using std::femto;  using std::pico;   using std::nano;
using std::micro;  using std::milli;  using std::centi;  using std::deci;
using std::deca;   using std::hecto;  using std::kilo;   using std::mega;
using std::giga;   using std::tera;   using std::peta;   using std::exa;

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

// Duration user-defined literals (1s / 16ms / 250us / 5min ...). Exposed
// as a namespace alias so `using namespace cardinal::chrono_literals;`
// brings the std UDL operators into scope without a direct std include.
namespace chrono_literals = std::chrono_literals;

}  // namespace cardinal
