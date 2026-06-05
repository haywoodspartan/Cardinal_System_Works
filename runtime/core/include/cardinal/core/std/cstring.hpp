#pragma once

// =============================================================================
// Cardinal core — raw byte ops (<cstring>) vocabulary (single sanctioned
// include site per the FOUNDATION RULE). Non-core code uses cardinal::memcpy
// etc. so the bytewise/SIMD policy stays restructurable from one place.
// =============================================================================

#include <cardinal/core/types.hpp>   // MASTER typedef header (FOUNDATION RULE)

#include <cstring>

namespace cardinal {

using std::memcpy;
using std::memmove;
using std::memset;
using std::memcmp;
using std::memchr;
using std::strlen;
using std::strcmp;
using std::strncmp;
using std::strcpy;
using std::strncpy;
using std::strcat;
using std::strncat;
using std::strchr;
using std::strstr;

}  // namespace cardinal
