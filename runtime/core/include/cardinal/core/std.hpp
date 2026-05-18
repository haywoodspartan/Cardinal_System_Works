#pragma once

// =============================================================================
// Cardinal core — the complete restructured std vocabulary, one include.
//
// FOUNDATION RULE umbrella: pulling <cardinal/core/std.hpp> gives non-core
// code the full cardinal:: alias surface (types + containers + algorithms +
// utility + traits + concurrency + libm/cstring/etc.) WITHOUT any direct std
// header. Granular headers remain available when a TU only needs a slice.
// =============================================================================

#include <cardinal/core/types.hpp>        // i8..u64, f32/f64, usize, string,
                                          // unique_ptr/shared_ptr, function
#include <cardinal/core/string_view.hpp>
#include <cardinal/core/containers.hpp>
#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/utility.hpp>
#include <cardinal/core/traits.hpp>
#include <cardinal/core/limits.hpp>
#include <cardinal/core/bit.hpp>
#include <cardinal/core/charconv.hpp>
#include <cardinal/core/cmath.hpp>
#include <cardinal/core/cstring.hpp>
#include <cardinal/core/cctype.hpp>
#include <cardinal/core/cstdio.hpp>
#include <cardinal/core/cstdarg.hpp>
#include <cardinal/core/cstdlib.hpp>
#include <cardinal/core/ctime.hpp>
#include <cardinal/core/cassert.hpp>
#include <cardinal/core/new.hpp>
#include <cardinal/core/atomic.hpp>
#include <cardinal/core/thread.hpp>
#include <cardinal/core/future.hpp>
#include <cardinal/core/chrono.hpp>
#include <cardinal/core/filesystem.hpp>
#include <cardinal/core/fstream.hpp>
#include <cardinal/core/sstream.hpp>
