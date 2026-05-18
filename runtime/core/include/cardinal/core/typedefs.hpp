#pragma once

// =============================================================================
// Cardinal — common typedefs.
//
// Aliases for the std::vector<T> instantiations the engine reaches for
// every other file. Pulling them through one header makes call sites
// shorter AND gives us a single line to switch later (custom allocator,
// small-vector optimisation, etc.) without touching every consumer.
//
//   cardinal::ByteVec    instead of std::vector<u8>
//   cardinal::FloatVec   instead of std::vector<f32>
//   cardinal::Vec3Array  instead of std::vector<core::Vec3>   (when math.hpp pulled in)
//
// Living in cardinal:: (not cardinal::core::) so call sites stay terse —
// most code already says `cardinal::u32`.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace cardinal {

// ---- Primitive vectors ----------------------------------------------------
// Bytes (file IO, network packets, raw GPU upload staging).
using ByteVec    = std::vector<u8>;
// Signed/unsigned integer vectors at every common width.
using Int8Vec    = std::vector<i8>;
using UInt8Vec   = std::vector<u8>;       // alias of ByteVec for symmetry
using Int16Vec   = std::vector<i16>;
using UInt16Vec  = std::vector<u16>;
using Int32Vec   = std::vector<i32>;
using UInt32Vec  = std::vector<u32>;
using Int64Vec   = std::vector<i64>;
using UInt64Vec  = std::vector<u64>;
using SizeVec    = std::vector<usize>;
using SSizeVec   = std::vector<isize>;
// Floating point.
using FloatVec   = std::vector<f32>;
using DoubleVec  = std::vector<f64>;
// Booleans — note std::vector<bool> is the bit-packed specialisation; if
// you want a true bool array, use UInt8Vec instead.
using BoolVec    = std::vector<bool>;

// ---- String vectors -------------------------------------------------------
// Owning + non-owning string lists. The non-owning variant (StringViewVec)
// is useful for "list of names that live elsewhere" — argv, command
// dictionaries, log category filters.
using StringList     = std::vector<std::string>;
using StringVec      = std::vector<std::string>;          // alias for StringList
using StringViewVec  = std::vector<std::string_view>;

// ---- Composable container aliases ----------------------------------------
//
// Vec<T>     — std::vector<T>, dynamic, heap-allocated.
// Arr<T,N>   — std::array<T, N>, fixed-size, stack/inlined.
//
// Together they form a kit for naming ANY nested layout. Use these
// directly when you want clarity at the call site without typing the
// full std:: chain:
//
//   Vec<Mat4>             // dynamic palette of 4x4 matrices (skinning)
//   Arr<Mat4, 4>          // exactly four matrices (cascaded shadow maps)
//   Vec<Vec<u32>>         // jagged 2D index grid
//   Arr<Vec<Mat4>, 8>     // 8 fixed slots, each a dynamic matrix list
//
// The named combos below (MatVec, MatVecVec, AoVoAoM, ...) pre-bake the
// most common geometry-of-geometry layouts so call sites can type a
// single identifier instead of a nested template instantiation.
template <typename T>            using Vec = std::vector<T>;
template <typename T, usize N>   using Arr = std::array<T, N>;

}  // namespace cardinal

#if defined(CARDINAL_MATH_TYPES_VISIBLE)
namespace cardinal {

// ---- Leaf math aliases ---------------------------------------------------
// Re-exports of the canonical types so call sites don't need to qualify
// `cardinal::core::Mat4` everywhere.
using Vec3 = ::cardinal::core::Vec3;
using Vec4 = ::cardinal::core::Vec4;
using Mat3 = ::cardinal::core::Mat3;
using Mat4 = ::cardinal::core::Mat4;

// ---- 1-deep std::vector<T> ----------------------------------------------
using Vec3Array  = std::vector<Vec3>;
using Vec4Array  = std::vector<Vec4>;
using Mat3Array  = std::vector<Mat3>;
using Mat4Array  = std::vector<Mat4>;

// ---- Pre-baked nested combinations --------------------------------------
//
// Naming convention — one letter per layer, OUTERMOST first:
//
//      A = std::array<T, N>     (fixed-size, N appended to the typedef)
//      V = std::vector<T>       (dynamic)
//      M = matrix leaf          (Mat4 by default; suffix _M3 for Mat3)
//      P = position leaf        (Vec3)
//
// Examples:
//      MV       — V<M>             vector of matrices
//      MVV      — V<V<M>>          vector of vector of matrices
//      MAV<8>   — V<A<M, 8>>       vector of arrays-of-8-matrices
//      AoVoAoM<4,4> — A<V<A<M,4>>,4>  the literal "AoVoAoM" with sizes
//
// The two outer layers (V or A) are dynamic on type; the leaf is the
// matrix. Inner array sizes that need to be specified appear as
// template parameters on the alias rather than being baked in — that
// keeps the combinatorial explosion under control.

// --- 1-deep ----------------------------------------------------------------
using MatVec  = Vec<Mat4>;                              // V<M>
using Mat3Vec = Vec<Mat3>;                              // V<M3>
template <usize N> using MatArr  = Arr<Mat4, N>;        // A<M, N>
template <usize N> using Mat3Arr = Arr<Mat3, N>;        // A<M3, N>

// --- 2-deep — vector of containers of matrices ----------------------------
using MatVecVec = Vec<Vec<Mat4>>;                       // V<V<M>>
using Mat3VecVec = Vec<Vec<Mat3>>;
template <usize N> using MatArrVec = Vec<Arr<Mat4, N>>; // V<A<M, N>>     ← VoAoM
template <usize N> using MatVecArr = Arr<Vec<Mat4>, N>; // A<V<M>>, N>    ← AoVoM

// --- 3-deep — the canonical AoVoAoM and friends ---------------------------
//
// AoVoAoM<NOuter, NInner> spells out the user's literal request:
//   Array (outer, NOuter slots) of
//     Vector of
//       Array (inner, NInner slots) of
//         Matrix.
//
// Plus the four other 3-deep AVM permutations for completeness.
template <usize NOuter, usize NInner>
using AoVoAoM = Arr<Vec<Arr<Mat4, NInner>>, NOuter>;    // A<V<A<M,Nin>>,Nout>

template <usize NOuter, usize NInner>
using AoAoVoM = Arr<Arr<Vec<Mat4>, NInner>, NOuter>;    // A<A<V<M>,Nin>,Nout>

template <usize NInner>
using VoAoVoM = Vec<Arr<Vec<Mat4>, NInner>>;            // V<A<V<M>, Nin>>

template <usize NInner>
using VoVoAoM = Vec<Vec<Arr<Mat4, NInner>>>;            // V<V<A<M, Nin>>>

template <usize NOuter>
using AoVoVoM = Arr<Vec<Vec<Mat4>>, NOuter>;            // A<V<V<M>>, Nout>

using VoVoVoM = Vec<Vec<Vec<Mat4>>>;                    // V<V<V<M>>> — fully dynamic

// --- Position-leaf variants (when the leaf is Vec3 instead of Mat4) ------
//
// Same naming but with P (Position = Vec3) as the leaf. Useful for
// per-frame transformed-vertex caches, particle position buffers, etc.
template <usize NOuter, usize NInner>
using AoVoAoP = Arr<Vec<Arr<Vec3, NInner>>, NOuter>;
template <usize NOuter>
using AoVoVoP = Arr<Vec<Vec<Vec3>>, NOuter>;
using VoVoVoP = Vec<Vec<Vec<Vec3>>>;

// --- Mat3 variants for inertia / orientation tables ----------------------
template <usize NOuter, usize NInner>
using AoVoAoM3 = Arr<Vec<Arr<Mat3, NInner>>, NOuter>;
using VoVoVoM3 = Vec<Vec<Vec<Mat3>>>;

// --- Generic composable templates for "any combination thereof" ----------
//
// When the curated names above don't fit your shape, build it inline
// from these — every legal nesting is reachable:
//
//      using Skinning  = Vec<MatArr<32>>;          // V<A<Mat4,32>>
//      using LodCascades = Arr<Vec<MatArr<4>>, 8>; // A<V<A<Mat4,4>>,8>
//      using Anim      = Vec<Vec<Vec<Mat4>>>;      // VoVoVoM (dynamic)
//
// The two templates below are aliases of Vec / Arr with a name that
// reads better at deep call sites. They're PURE re-exports; the
// compile-time identity is the same.
template <typename T>           using AnyVec = Vec<T>;        // alias of Vec
template <typename T, usize N>  using AnyArr = Arr<T, N>;     // alias of Arr

}  // namespace cardinal
#endif
