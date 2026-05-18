#pragma once

// platform.hpp ships the foundational compiler / arch / cache macros
// (CARDINAL_PLATFORM_*, CARDINAL_FORCEINLINE, CARDINAL_READ_MOSTLY,
// CARDINAL_CACHE_ALIGNED, CARDINAL_HOT, CARDINAL_PREFETCH, …). Pulling
// it in via types.hpp means every core consumer gets the macros for free
// without an explicit include — types.hpp is the universal core header.
#include <cardinal/core/platform.hpp>

// ┌─────────────────────────────────────────────────────────────────┐
// │ FOUNDATION RULE — read before adding any std header anywhere.    │
// │                                                                  │
// │ Base C++ headers are referenced HERE and nowhere else outside    │
// │ cardinal::core. The Foundation restructures them into the        │
// │ cardinal:: vocabulary below so ownership / allocator / ABI /     │
// │ small-buffer policy is changeable in ONE place.                  │
// │                                                                  │
// │ WORKFLOW (mandatory ordering): when a new feature needs a native │
// │ header not yet wrapped, the header is added to cardinal::core    │
// │ and structured into the cardinal:: namespace FIRST — before any  │
// │ dependent implementation or feature code is written. Foundation  │
// │ first, feature second. Non-core code never includes a std header │
// │ directly; it uses the cardinal:: alias.                          │
// └─────────────────────────────────────────────────────────────────┘
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace cardinal {

using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using f32 = float;
using f64 = double;

using usize = std::size_t;
using isize = std::ptrdiff_t;

// ---------------------------------------------------------------------------
// Foundation-owned core vocabulary. The rest of the engine uses these
// instead of <string>/<memory>/<functional> directly, keeping the
// owning-pointer / string-storage / callable ABI restructurable from
// cardinal::core alone. Alias templates (not typedefs) so they stay
// fully generic; make_unique/make_shared re-exported for call sites.
// ---------------------------------------------------------------------------
using string  = std::string;
using wstring = std::wstring;
using std::to_string;
using std::to_wstring;
using std::stoi;   using std::stol;   using std::stoll;
using std::stoul;  using std::stoull;
using std::stof;   using std::stod;   using std::stold;
using std::getline;                     // std::getline(istream, std::string&)

template <class T, class D = std::default_delete<T>>
using unique_ptr = std::unique_ptr<T, D>;          // D defaulted: existing
                                                   // single-arg uses unchanged
template <class T> using shared_ptr = std::shared_ptr<T>;
template <class T> using weak_ptr   = std::weak_ptr<T>;
using std::default_delete;
using std::make_shared;
using std::make_unique;

template <class Sig> using function = std::function<Sig>;
template <class T>   using hash     = std::hash<T>;   // std hash functor

// Standard comparison / function-object functors (also from <functional>).
using std::less;            using std::greater;
using std::less_equal;      using std::greater_equal;
using std::equal_to;        using std::not_equal_to;
using std::reference_wrapper;
using std::ref;             using std::cref;
using std::invoke;

static_assert(sizeof(f32) == 4, "f32 must be 4 bytes");
static_assert(sizeof(f64) == 8, "f64 must be 8 bytes");
static_assert(sizeof(void*) == 8, "Cardinal requires a 64-bit target");

// =============================================================================
// Engine-wide compile-time constants — mathematical, metallic ratios, physical.
//
// Modern C++ note: these are `inline constexpr` rather than `typedef`s.
// `typedef` introduces a type alias; for *named compile-time numeric values*
// `inline constexpr` is the equivalent — same zero-cost compile-time
// substitution, plus type checking + ODR-safety across translation units.
// Useful in any header without causing multiple-definition link errors.
//
// All values are f64 by default. The `constants32` sibling namespace
// pre-rounds to f32 for hot paths that don't need double precision.
// =============================================================================
namespace constants {

// ---- Mathematical (irrational) constants ----------------------------------
inline constexpr f64 pi          = 3.141592653589793238462643383279502884;
inline constexpr f64 two_pi      = 6.283185307179586476925286766559005768;
inline constexpr f64 half_pi     = 1.570796326794896619231321691639751442;
inline constexpr f64 quarter_pi  = 0.785398163397448309615660845819875721;
inline constexpr f64 inv_pi      = 0.318309886183790671537767526745028724;
inline constexpr f64 inv_two_pi  = 0.159154943091895335768883763372514362;
inline constexpr f64 e           = 2.718281828459045235360287471352662497;
inline constexpr f64 sqrt2       = 1.414213562373095048801688724209698079;
inline constexpr f64 sqrt3       = 1.732050807568877293527446341505872367;
inline constexpr f64 sqrt5       = 2.236067977499789696409173668731276235;
inline constexpr f64 inv_sqrt2   = 0.707106781186547524400844362104849039;
inline constexpr f64 inv_sqrt3   = 0.577350269189625764509148780501957455;
inline constexpr f64 ln2         = 0.693147180559945309417232121458176568;
inline constexpr f64 ln10        = 2.302585092994045684017991454684364208;
inline constexpr f64 deg_to_rad  = pi / 180.0;
inline constexpr f64 rad_to_deg  = 180.0 / pi;

// ---- Metallic ratios (solutions of x² = n·x + 1) --------------------------
// Each metallic mean φ_n satisfies φ_n² = n·φ_n + 1 → φ_n = (n + √(n²+4)) / 2.
// They appear in self-similar geometry, Fibonacci-like sequences (silver
// uses Pell numbers, bronze uses tribonacci-adjacent), recursive procedural
// generation, and pleasant aspect ratios for UI / level layout.
inline constexpr f64 golden  = 1.618033988749894848204586834365638118;  // (1 + √5) / 2
inline constexpr f64 silver  = 2.414213562373095048801688724209698079;  // 1 + √2
inline constexpr f64 bronze  = 3.302775637731994646559610633735247974;  // (3 + √13) / 2
inline constexpr f64 copper  = 4.236067977499789696409173668731276235;  // (4 + √20) / 2
inline constexpr f64 nickel  = 5.192582403567252418170220330001543275;  // (5 + √29) / 2

// Golden-ratio companions used in art / procgen. The angle (137.508°) is
// the magic phyllotaxis spacing for sunflower-style point distributions.
inline constexpr f64 inv_golden        = 0.618033988749894848204586834365638118;
inline constexpr f64 golden_angle_rad  = 2.399963229728653322231555506633613853;  // 2π · (1 - 1/φ)
inline constexpr f64 golden_angle_deg  = 137.50776405003785;

// ---- Physical constants (SI base units) -----------------------------------
// Engine code working in game units should rescale before consuming these.
// Constants follow the 2019 SI redefinition (exact values where applicable).
inline constexpr f64 light_speed_mps        = 299792458.0;          // c    (exact)
inline constexpr f64 gravitational_const    = 6.67430e-11;          // G    N·m²/kg²
inline constexpr f64 planck_const           = 6.62607015e-34;       // h    J·s   (exact)
inline constexpr f64 reduced_planck_const   = 1.054571817e-34;      // ℏ
inline constexpr f64 vacuum_permittivity    = 8.8541878128e-12;     // ε₀   F/m
inline constexpr f64 vacuum_permeability    = 1.25663706212e-6;     // μ₀ — "Mu" — H/m
inline constexpr f64 boltzmann_const        = 1.380649e-23;         // k_B  J/K   (exact)
inline constexpr f64 avogadro_const         = 6.02214076e23;        // N_A        (exact)
inline constexpr f64 elementary_charge_c    = 1.602176634e-19;      // e    C     (exact)
inline constexpr f64 electron_mass_kg       = 9.1093837015e-31;     // m_e
inline constexpr f64 proton_mass_kg         = 1.67262192369e-27;    // m_p
inline constexpr f64 stefan_boltzmann       = 5.670374419e-8;       // σ    W/(m²·K⁴)

// Earth-frame helpers — the engine's default gravity uses earth_gravity_mps2.
inline constexpr f64 standard_gravity_mps2  = 9.80665;              // g_n  (exact, ISO)
inline constexpr f64 earth_gravity_mps2     = 9.81;                 // engine convention
inline constexpr f64 earth_radius_m         = 6.371008e6;           // mean
inline constexpr f64 earth_mass_kg          = 5.9722e24;
inline constexpr f64 au_m                   = 1.495978707e11;       // astronomical unit
inline constexpr f64 light_year_m           = 9.4607304725808e15;
inline constexpr f64 parsec_m               = 3.0856775814913673e16;

}  // namespace constants

// f32 mirrors for hot-path code where the f64 value would force a costly
// downcast on every use. Identical names so callers can switch precision
// by changing the namespace alias.
namespace constants32 {
inline constexpr f32 pi             = static_cast<f32>(constants::pi);
inline constexpr f32 two_pi         = static_cast<f32>(constants::two_pi);
inline constexpr f32 half_pi        = static_cast<f32>(constants::half_pi);
inline constexpr f32 quarter_pi     = static_cast<f32>(constants::quarter_pi);
inline constexpr f32 inv_pi         = static_cast<f32>(constants::inv_pi);
inline constexpr f32 e              = static_cast<f32>(constants::e);
inline constexpr f32 sqrt2          = static_cast<f32>(constants::sqrt2);
inline constexpr f32 sqrt3          = static_cast<f32>(constants::sqrt3);
inline constexpr f32 sqrt5          = static_cast<f32>(constants::sqrt5);
inline constexpr f32 deg_to_rad     = static_cast<f32>(constants::deg_to_rad);
inline constexpr f32 rad_to_deg     = static_cast<f32>(constants::rad_to_deg);
inline constexpr f32 golden         = static_cast<f32>(constants::golden);
inline constexpr f32 silver         = static_cast<f32>(constants::silver);
inline constexpr f32 bronze         = static_cast<f32>(constants::bronze);
inline constexpr f32 copper         = static_cast<f32>(constants::copper);
inline constexpr f32 nickel         = static_cast<f32>(constants::nickel);
inline constexpr f32 inv_golden     = static_cast<f32>(constants::inv_golden);
inline constexpr f32 golden_angle_rad = static_cast<f32>(constants::golden_angle_rad);
inline constexpr f32 earth_gravity_mps2 = static_cast<f32>(constants::earth_gravity_mps2);
}  // namespace constants32

// Top-level "ratios" namespace for the most commonly reached-for values.
// `cardinal::ratios::golden` is shorter than `cardinal::constants::golden`
// when you're just iterating across phyllotaxis spacings.
namespace ratios {
inline constexpr f64 golden = constants::golden;
inline constexpr f64 silver = constants::silver;
inline constexpr f64 bronze = constants::bronze;
inline constexpr f64 copper = constants::copper;
inline constexpr f64 nickel = constants::nickel;
}  // namespace ratios

}  // namespace cardinal
