#pragma once

// =============================================================================
// Cardinal core — <utility> / <tuple> / <optional> / <variant> vocabulary
// (single sanctioned include site per the FOUNDATION RULE).
// =============================================================================

#include <optional>
#include <tuple>
#include <utility>
#include <variant>

namespace cardinal {

// <utility>
using std::move;
using std::forward;
using std::swap;
using std::exchange;
using std::as_const;
using std::declval;
using std::pair;
using std::make_pair;
using std::to_underlying;
using std::in_place;
using std::in_place_type;
using std::in_place_index;

// <tuple>
using std::tuple;
using std::make_tuple;
using std::tie;
using std::forward_as_tuple;
using std::tuple_cat;
using std::tuple_size;
using std::tuple_size_v;
using std::tuple_element;
using std::tuple_element_t;
using std::apply;
using std::ignore;

// <optional>
using std::optional;
using std::nullopt;
using std::nullopt_t;
using std::make_optional;
using std::bad_optional_access;

// <variant>
using std::variant;
using std::monostate;
using std::variant_size;
using std::variant_size_v;
using std::variant_alternative;
using std::variant_alternative_t;
using std::holds_alternative;
using std::visit;
using std::bad_variant_access;
// NOTE: std::get / std::get_if are intentionally re-exported via
// cardinal::core/tuple-aware sites; pull them in here too for variant use.
using std::get;
using std::get_if;

}  // namespace cardinal
