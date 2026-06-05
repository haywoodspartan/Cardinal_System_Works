#pragma once

// =============================================================================
// Cardinal core — C stdio (<cstdio>) vocabulary (single sanctioned include
// site per the FOUNDATION RULE). Mostly reached for snprintf in diagnostics.
// NOTE: stderr/stdout/stdin are C macros and cannot be namespaced — code
// that needs the streams should prefer cardinal::log; this header exists so
// the <cstdio> *include* still lives only in cardinal::core.
// =============================================================================

#include <cardinal/core/platform.hpp> // CARDINAL_PLATFORM_WINDOWS
#include <cardinal/core/types.hpp>   // MASTER typedef header (FOUNDATION RULE)

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

// ---- 64-bit file offsets (cardinal::fseek64 / cardinal::ftell64) ---
// std::fseek/ftell use `long`, which is 32-bit on Windows LLP64 (and
// on 32-bit Linux without _FILE_OFFSET_BITS=64). Any file > 2 GiB
// gets silently truncated to the low 31 bits — mis-seek on the
// (writer) tail and a truncated size on read. The pack module (asset
// archives, multi-GiB common in production) used to cast u64 offsets
// to long and called fseek/ftell — fixed by the call-site migration
// to these 64-bit wrappers. Parked-supervised item #1 in
// feedback_integration_coverage_map.md.
inline int fseek64(FILE* f, i64 offset, int whence) noexcept {
#if CARDINAL_PLATFORM_WINDOWS
    return ::_fseeki64(f, static_cast<long long>(offset), whence);
#else
    // POSIX `fseeko` takes off_t; with _FILE_OFFSET_BITS=64 (the
    // cardinal build default on Linux) off_t is i64.
    return ::fseeko(f, static_cast<off_t>(offset), whence);
#endif
}
inline i64 ftell64(FILE* f) noexcept {
#if CARDINAL_PLATFORM_WINDOWS
    return static_cast<i64>(::_ftelli64(f));
#else
    return static_cast<i64>(::ftello(f));
#endif
}

}  // namespace cardinal
