#pragma once

// =============================================================================
// Cardinal core — <type_traits> vocabulary (single sanctioned include site
// per the FOUNDATION RULE). Curated re-export of the metaprogramming traits
// the engine actually reaches for.
// =============================================================================

#include <cardinal/core/types.hpp>   // MASTER typedef header (FOUNDATION RULE)

#include <type_traits>

namespace cardinal {

using std::enable_if;            using std::enable_if_t;
using std::conditional;          using std::conditional_t;
using std::is_same;              using std::is_same_v;
using std::is_base_of;           using std::is_base_of_v;
using std::is_convertible;       using std::is_convertible_v;
using std::is_constructible;     using std::is_constructible_v;
using std::is_default_constructible; using std::is_default_constructible_v;
using std::is_trivially_copyable;    using std::is_trivially_copyable_v;
using std::is_trivial;           using std::is_trivial_v;
using std::is_standard_layout;   using std::is_standard_layout_v;
using std::is_enum;              using std::is_enum_v;
using std::is_integral;          using std::is_integral_v;
using std::is_floating_point;    using std::is_floating_point_v;
using std::is_arithmetic;        using std::is_arithmetic_v;
using std::is_signed;            using std::is_signed_v;
using std::is_unsigned;          using std::is_unsigned_v;
using std::is_pointer;           using std::is_pointer_v;
using std::is_reference;         using std::is_reference_v;
using std::is_const;             using std::is_const_v;
using std::is_void;              using std::is_void_v;
using std::is_class;             using std::is_class_v;
using std::is_function;          using std::is_function_v;
using std::is_empty;             using std::is_empty_v;
using std::is_invocable;         using std::is_invocable_v;
using std::is_invocable_r;       using std::is_invocable_r_v;
using std::invoke_result;        using std::invoke_result_t;
using std::remove_cv;            using std::remove_cv_t;
using std::remove_const;         using std::remove_const_t;
using std::remove_reference;     using std::remove_reference_t;
using std::remove_cvref;         using std::remove_cvref_t;
using std::remove_pointer;       using std::remove_pointer_t;
using std::remove_extent;        using std::remove_extent_t;
using std::add_pointer;          using std::add_pointer_t;
using std::add_const;            using std::add_const_t;
using std::add_lvalue_reference; using std::add_lvalue_reference_t;
using std::decay;                using std::decay_t;
using std::common_type;          using std::common_type_t;
using std::underlying_type;      using std::underlying_type_t;
using std::make_signed;          using std::make_signed_t;
using std::make_unsigned;        using std::make_unsigned_t;
using std::alignment_of;         using std::alignment_of_v;
using std::void_t;
using std::true_type;            using std::false_type;
using std::bool_constant;
using std::integral_constant;
using std::type_identity;        using std::type_identity_t;

}  // namespace cardinal
