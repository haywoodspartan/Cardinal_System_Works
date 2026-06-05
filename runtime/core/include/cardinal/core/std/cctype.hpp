#pragma once

// =============================================================================
// Cardinal core — character classification (<cctype>) vocabulary (single
// sanctioned include site per the FOUNDATION RULE). Always feed these an
// `unsigned char` (or EOF) — passing a plain signed char is UB.
// =============================================================================

#include <cardinal/core/types.hpp>   // MASTER typedef header (FOUNDATION RULE)

#include <cctype>

namespace cardinal {

using std::tolower;   using std::toupper;
using std::isspace;   using std::isblank;
using std::isdigit;   using std::isxdigit;
using std::isalpha;   using std::isalnum;
using std::isupper;   using std::islower;
using std::ispunct;   using std::iscntrl;
using std::isprint;   using std::isgraph;

}  // namespace cardinal
