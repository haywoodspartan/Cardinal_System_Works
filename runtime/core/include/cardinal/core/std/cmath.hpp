#pragma once

// =============================================================================
// Cardinal core — scalar math (<cmath>) vocabulary (single sanctioned
// include site per the FOUNDATION RULE). cardinal::core/math.hpp keeps the
// vector/matrix types; this is the libm scalar surface. using-declarations
// import every overload so float/double both resolve.
// =============================================================================

#include <cardinal/core/types.hpp>   // MASTER typedef header (FOUNDATION RULE)

#include <cmath>

namespace cardinal {

using std::sqrt;   using std::cbrt;   using std::hypot;
using std::sin;    using std::cos;    using std::tan;
using std::asin;   using std::acos;   using std::atan;   using std::atan2;
using std::sinh;   using std::cosh;   using std::tanh;
using std::exp;    using std::exp2;   using std::expm1;
// NOTE: bare `log` is intentionally NOT re-exported — `cardinal::log` is the
// engine's logging namespace (core/log.hpp). Natural log is cardinal::ln().
using std::log2;   using std::log10;  using std::log1p;
using std::pow;
using std::floor;  using std::ceil;   using std::round;  using std::trunc;
using std::nearbyint; using std::rint; using std::lround; using std::llround;
using std::fabs;   using std::abs;
using std::fmod;   using std::remainder; using std::fma;
using std::copysign;
using std::fmin;   using std::fmax;   using std::fdim;
using std::modf;   using std::frexp;  using std::ldexp;  using std::scalbn;
using std::isnan;  using std::isinf;  using std::isfinite; using std::isnormal;
using std::signbit;
using std::erf;    using std::erfc;   using std::tgamma; using std::lgamma;

// Natural logarithm under a non-colliding name (see note above).
inline float       ln(float x)       noexcept { return std::log(x); }
inline double      ln(double x)      noexcept { return std::log(x); }
inline long double ln(long double x) noexcept { return std::log(x); }

}  // namespace cardinal
