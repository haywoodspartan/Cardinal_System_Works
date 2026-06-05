#pragma once

// =============================================================================
// Cardinal core — the complete std vocabulary, one include.
//
// FOUNDATION RULE umbrella: pulling <cardinal/core/std.hpp> gives non-core
// code the full cardinal:: alias surface (types + containers + algorithms +
// utility + traits + concurrency + libm/cstring/etc.) WITHOUT any direct std
// header. Granular headers remain available when a TU only needs a slice.
// =============================================================================

#include <cardinal/core/types.hpp>             // i8..u64, f32/f64, usize, string,
                                               // unique_ptr/shared_ptr, function
#include <cardinal/core/std/string_view.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/core/std/algorithm.hpp>
#include <cardinal/core/std/utility.hpp>
#include <cardinal/core/std/any.hpp>
#include <cardinal/core/std/traits.hpp>
#include <cardinal/core/std/limits.hpp>
#include <cardinal/core/std/bit.hpp>
#include <cardinal/core/std/charconv.hpp>
#include <cardinal/core/std/cmath.hpp>
#include <cardinal/core/std/cstring.hpp>
#include <cardinal/core/std/cctype.hpp>
#include <cardinal/core/std/cstdio.hpp>
#include <cardinal/core/std/cstdarg.hpp>
#include <cardinal/core/std/cstdlib.hpp>
#include <cardinal/core/std/ctime.hpp>
#include <cardinal/core/std/cassert.hpp>
#include <cardinal/core/std/climits.hpp>
#include <cardinal/core/std/new.hpp>
#include <cardinal/core/std/atomic.hpp>
#include <cardinal/core/std/thread.hpp>
#include <cardinal/core/std/future.hpp>
#include <cardinal/core/std/chrono.hpp>
#include <cardinal/core/std/filesystem.hpp>
#include <cardinal/core/std/fstream.hpp>
#include <cardinal/core/std/sstream.hpp>
