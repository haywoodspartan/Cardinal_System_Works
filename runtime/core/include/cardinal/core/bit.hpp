#pragma once

// =============================================================================
// Cardinal core — <bit> vocabulary (single sanctioned include site per the
// FOUNDATION RULE). bit_cast is the blessed type-pun (replaces ad-hoc
// memcpy bit-casts); the rest is constexpr bit twiddling.
// =============================================================================

#include <cardinal/core/types.hpp>   // MASTER typedef header (FOUNDATION RULE)

#include <bit>

namespace cardinal {

using std::bit_cast;
using std::byteswap;
using std::has_single_bit;
using std::bit_ceil;
using std::bit_floor;
using std::bit_width;
using std::rotl;
using std::rotr;
using std::countl_zero;
using std::countl_one;
using std::countr_zero;
using std::countr_one;
using std::popcount;
using std::endian;

}  // namespace cardinal
