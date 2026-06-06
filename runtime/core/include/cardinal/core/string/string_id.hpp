#pragma once

// =============================================================================
// cardinal::core::StringId — a hashed string identifier.
//
// Wraps a 64-bit FNV-1a hash of a string so identity comparisons are a single
// integer compare instead of a strcmp. The engine compares string ids all over
// the place — command ids ("world.place_asset"), asset ids, event/channel
// names, material + parameter names — and those compares are on hot paths
// (palette dispatch, per-frame lookups). StringId turns them into u64 ==.
//
// Compile-time: a string LITERAL is hashed at compile time, so
//   if (cmd == "world.undo"_sid)               // both sides are constants
// compiles to an integer compare with the hash baked into the binary — no
// runtime hashing, no string storage.
//
// Runtime: construct from a cardinal::string / const char* when the text is
// only known at runtime. Same hash, so a runtime id matches a compile-time
// literal id of the same text.
//
// Collisions: 64-bit FNV over engine-scale id sets has negligible collision
// probability; StringId is for identity, not security. A future reverse-name
// registry (intern table) can land separately for debug display.
//
// FOUNDATION RULE: lives in cardinal::core (uses std + the core hash). Exposed
// as cardinal::StringId; the ""_sid literal is re-exported into cardinal too.
// =============================================================================

#include <cardinal/core/types.hpp>        // u64, usize, cardinal::string
#include <cardinal/core/std/hash.hpp>     // constexpr fnv1a64

namespace cardinal::core {

class StringId {
public:
    using value_type = cardinal::u64;

    // Default = the invalid / empty id (hash 0). Note an explicitly-empty
    // string hashes to the FNV offset basis, NOT 0, so "" != default.
    constexpr StringId() noexcept : hash_(0) {}

    // Wrap a precomputed hash (e.g. a deserialised id or the ""_sid literal).
    constexpr explicit StringId(value_type precomputed) noexcept : hash_(precomputed) {}

    // From a null-terminated string. constexpr for string literals (hashed at
    // compile time); also valid at runtime for any const char*.
    constexpr explicit StringId(const char* s) noexcept : hash_(fnv1a64(s)) {}

    // From a runtime cardinal::string (hashes the exact bytes, length-based).
    // NOTE: cast to const void* so this binds the (data, len) overload — the
    // (const char*, u64 seed) overload would otherwise win and treat size()
    // as the SEED, producing a different hash than the literal/c-string ctors.
    explicit StringId(const cardinal::string& s) noexcept
        : hash_(fnv1a64(static_cast<const void*>(s.data()), s.size())) {}

    [[nodiscard]] constexpr value_type value() const noexcept { return hash_; }
    [[nodiscard]] constexpr bool       valid() const noexcept { return hash_ != 0; }
    constexpr explicit operator value_type() const noexcept { return hash_; }

    friend constexpr bool operator==(StringId a, StringId b) noexcept { return a.hash_ == b.hash_; }
    friend constexpr bool operator!=(StringId a, StringId b) noexcept { return a.hash_ != b.hash_; }
    friend constexpr bool operator< (StringId a, StringId b) noexcept { return a.hash_ <  b.hash_; }
    friend constexpr bool operator> (StringId a, StringId b) noexcept { return a.hash_ >  b.hash_; }

private:
    value_type hash_;
};

// User-defined literal: "world.undo"_sid → a compile-time StringId. The hash
// matches StringId("world.undo") / StringId(cardinal::string{"world.undo"}).
inline constexpr StringId operator""_sid(const char* s, cardinal::usize /*len*/) noexcept {
    return StringId(fnv1a64(s));
}

}  // namespace cardinal::core

namespace cardinal {
using StringId = core::StringId;
using core::operator""_sid;   // so `using namespace cardinal;` + "x"_sid works
}  // namespace cardinal

// Hashable → usable as an unordered_map / unordered_set key. The id IS already
// a well-mixed hash, so identity-map it.
template <>
struct std::hash<cardinal::core::StringId> {
    cardinal::usize operator()(cardinal::core::StringId id) const noexcept {
        return static_cast<cardinal::usize>(id.value());
    }
};
