#pragma once

// =============================================================================
// cardinal::core::Flags<Enum> — a type-safe bit-set over a scoped enum.
//
// The engine has bitmask enums everywhere (rhi TextureUsage / BufferUsage,
// feature gates, dirty masks) and today manipulates them with raw casts:
//     if (d.usage & static_cast<u32>(TextureUsage::DepthRenderTarget)) ...
// which loses the enum type, silently mixes unrelated masks, and is easy to
// get wrong (& vs &&, forgetting the cast). Flags<E> keeps the enum type on
// the value while giving the full set of bit ops, all constexpr:
//     Flags<TextureUsage> u = TextureUsage::DepthRenderTarget | TextureUsage::Sampled;
//     if (u.has(TextureUsage::Sampled)) ...
//
// Opt an enum into enum-level `a | b` / `~a` (yielding Flags<E>) with
// CARDINAL_ENABLE_FLAGS(E) in the enum's namespace, just after its definition.
//
// FOUNDATION RULE: lives in cardinal::core (uses std type traits). Exposed as
// cardinal::Flags; CARDINAL_ENABLE_FLAGS is a global macro.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <type_traits>

namespace cardinal::core {

template <class E>
class Flags {
    static_assert(std::is_enum_v<E>, "Flags<E> requires an enum type");
    using U = std::underlying_type_t<E>;

public:
    using enum_type       = E;
    using underlying_type = U;

    constexpr Flags() noexcept : bits_(0) {}
    constexpr Flags(E flag) noexcept : bits_(static_cast<U>(flag)) {}   // implicit: single flag → set
    constexpr explicit Flags(U raw) noexcept : bits_(raw) {}

    [[nodiscard]] constexpr U    value() const noexcept { return bits_; }
    [[nodiscard]] constexpr bool any()   const noexcept { return bits_ != U(0); }
    [[nodiscard]] constexpr bool none()  const noexcept { return bits_ == U(0); }
    constexpr explicit operator bool()   const noexcept { return bits_ != U(0); }

    // has(flag): every bit of `flag` is set (and `flag` is non-zero, so a
    // "None = 0" enumerator never reads as present).
    [[nodiscard]] constexpr bool has(E flag) const noexcept {
        const U f = static_cast<U>(flag);
        return f != U(0) && (bits_ & f) == f;
    }
    [[nodiscard]] constexpr bool has_any(Flags m) const noexcept { return (bits_ & m.bits_) != U(0); }
    [[nodiscard]] constexpr bool has_all(Flags m) const noexcept { return (bits_ & m.bits_) == m.bits_; }

    constexpr Flags& set(E flag)    noexcept { bits_ |= static_cast<U>(flag);  return *this; }
    constexpr Flags& clear(E flag)  noexcept { bits_ &= static_cast<U>(~static_cast<U>(flag)); return *this; }
    constexpr Flags& toggle(E flag) noexcept { bits_ ^= static_cast<U>(flag);  return *this; }
    constexpr void   clear_all()    noexcept { bits_ = U(0); }

    constexpr Flags operator|(Flags o) const noexcept { return Flags(static_cast<U>(bits_ | o.bits_)); }
    constexpr Flags operator&(Flags o) const noexcept { return Flags(static_cast<U>(bits_ & o.bits_)); }
    constexpr Flags operator^(Flags o) const noexcept { return Flags(static_cast<U>(bits_ ^ o.bits_)); }
    constexpr Flags operator~()        const noexcept { return Flags(static_cast<U>(~bits_)); }
    constexpr Flags& operator|=(Flags o) noexcept { bits_ |= o.bits_; return *this; }
    constexpr Flags& operator&=(Flags o) noexcept { bits_ &= o.bits_; return *this; }
    constexpr Flags& operator^=(Flags o) noexcept { bits_ ^= o.bits_; return *this; }

    friend constexpr bool operator==(Flags a, Flags b) noexcept { return a.bits_ == b.bits_; }
    friend constexpr bool operator!=(Flags a, Flags b) noexcept { return a.bits_ != b.bits_; }

private:
    U bits_;
};

}  // namespace cardinal::core

namespace cardinal {
template <class E>
using Flags = core::Flags<E>;
}  // namespace cardinal

// Enable ergonomic enum-level combination for a scoped enum: after this,
// `E::A | E::B` and `~E::A` yield a cardinal::Flags<E>. Invoke at NAMESPACE
// scope in the same namespace as E (so the operators are found by ADL).
#define CARDINAL_ENABLE_FLAGS(E)                                               \
    [[maybe_unused]] inline constexpr ::cardinal::core::Flags<E>               \
    operator|(E a, E b) noexcept {                                            \
        return ::cardinal::core::Flags<E>(a) | ::cardinal::core::Flags<E>(b); \
    }                                                                         \
    [[maybe_unused]] inline constexpr ::cardinal::core::Flags<E>               \
    operator~(E a) noexcept { return ~::cardinal::core::Flags<E>(a); }
