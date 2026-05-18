#pragma once

// =============================================================================
// Cardinal core — <charconv> vocabulary (single sanctioned include site per
// the FOUNDATION RULE). Locale-independent, allocation-free number<->text.
// =============================================================================

#include <charconv>
#include <system_error>   // std::errc is the result-status enum of to/from_chars

namespace cardinal {

using std::errc;
using std::from_chars;
using std::to_chars;
using std::from_chars_result;
using std::to_chars_result;
using std::chars_format;

}  // namespace cardinal
