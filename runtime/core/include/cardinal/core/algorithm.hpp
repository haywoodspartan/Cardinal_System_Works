#pragma once

// =============================================================================
// Cardinal core — <algorithm> + <numeric> vocabulary (single sanctioned
// include site per the FOUNDATION RULE). Non-core code calls cardinal::sort,
// cardinal::clamp, cardinal::accumulate, … never std:: directly.
// using-declarations import every overload, so these stay fully generic.
// =============================================================================

#include <algorithm>
#include <numeric>

namespace cardinal {

using std::min;          using std::max;          using std::minmax;
using std::clamp;
using std::min_element;  using std::max_element;  using std::minmax_element;
using std::sort;         using std::stable_sort;  using std::partial_sort;
using std::nth_element;  using std::is_sorted;
using std::make_heap;    using std::push_heap;    using std::pop_heap;
using std::sort_heap;    using std::is_heap;
using std::find;         using std::find_if;      using std::find_if_not;
using std::count;        using std::count_if;
using std::for_each;
using std::copy;         using std::copy_n;       using std::copy_if;
using std::copy_backward;
using std::move_backward;
using std::fill;         using std::fill_n;
using std::transform;
using std::remove;       using std::remove_if;
using std::replace;      using std::replace_if;
using std::reverse;      using std::rotate;
using std::unique;
using std::lower_bound;  using std::upper_bound;
using std::binary_search; using std::equal_range;
using std::all_of;       using std::any_of;       using std::none_of;
using std::equal;        using std::mismatch;
// NOTE: bare `partition` / `stable_partition` are intentionally NOT
// re-exported — `cardinal::partition` is an engine module namespace
// (runtime/partition). Use std::partition directly inside cardinal::core
// only, or add a non-colliding alias if a consumer ever needs it.
using std::swap_ranges;
using std::merge;        using std::set_union;    using std::set_intersection;
using std::set_difference;
using std::distance;     using std::advance;      using std::next;  using std::prev;
using std::begin;        using std::end;
using std::iter_swap;

// <numeric>
using std::accumulate;
using std::inner_product;
using std::partial_sum;
using std::adjacent_difference;
using std::iota;
using std::reduce;
using std::gcd;          using std::lcm;
using std::midpoint;

}  // namespace cardinal
