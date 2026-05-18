#pragma once

// =============================================================================
// Cardinal core — <string_view> vocabulary (single sanctioned include site
// per the FOUNDATION RULE). Companion to cardinal::string in core/types.hpp.
// =============================================================================

#include <string_view>

namespace cardinal {

using string_view  = std::string_view;
using wstring_view = std::wstring_view;
using std::string_view_literals::operator""sv;

}  // namespace cardinal
