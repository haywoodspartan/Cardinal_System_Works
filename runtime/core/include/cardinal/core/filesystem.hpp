#pragma once

// =============================================================================
// Cardinal core — <filesystem> vocabulary (single sanctioned include site
// per the FOUNDATION RULE). Non-core code uses cardinal::fs::path etc.;
// core/io.hpp remains the higher-level engine I/O surface.
// =============================================================================

#include <cardinal/core/types.hpp>   // MASTER typedef header (FOUNDATION RULE)

#include <filesystem>
#include <system_error>   // std::error_code — the fs no-throw overload arg

namespace cardinal {

namespace fs = std::filesystem;

using std::error_code;
using std::error_condition;
using std::error_category;
using std::system_error;
using std::system_category;
using std::generic_category;

}  // namespace cardinal
