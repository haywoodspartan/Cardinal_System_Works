#pragma once

// =============================================================================
// Cardinal core — C stdio (<cstdio>) vocabulary (single sanctioned include
// site per the FOUNDATION RULE). Mostly reached for snprintf in diagnostics.
// NOTE: stderr/stdout/stdin are C macros and cannot be namespaced — code
// that needs the streams should prefer cardinal::log; this header exists so
// the <cstdio> *include* still lives only in cardinal::core.
// =============================================================================

#include <cstdio>

namespace cardinal {

using std::snprintf;
using std::vsnprintf;
using std::printf;
using std::fprintf;
using std::sprintf;
using std::sscanf;
using std::FILE;
using std::fopen;
using std::fclose;
using std::fread;
using std::fwrite;
using std::fseek;
using std::ftell;
using std::fflush;
using std::fputs;
using std::fgets;
// (file remove/rename live in cardinal::fs — see core/filesystem.hpp — to
//  avoid a name clash with the <algorithm> std::remove under core/std.hpp)

}  // namespace cardinal
