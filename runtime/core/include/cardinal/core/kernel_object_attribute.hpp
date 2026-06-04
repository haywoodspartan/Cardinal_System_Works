#pragma once

// =============================================================================
// Cardinal Core — Windows KernelObjectAttribute system.
//
// RAII / builder-pattern wrapper around Windows OBJECT_ATTRIBUTES — the
// structure every NT API call (NtCreateFile / NtCreateEvent / NtOpenKey
// / NtCreateMutant / etc.) takes to describe how a kernel object is
// created or opened. The raw NT API takes ~6 fields wired through the
// InitializeObjectAttributes macro; this wrapper:
//
//   * Owns its UNICODE_STRING name + wchar_t buffer (no caller lifetime
//     traps — the buffer lives as long as the attribute does).
//   * Exposes a fluent builder for the OBJ_* flag set (inherit, kernel
//     handle, case-insensitive, permanent, exclusive, openif, etc.)
//     via the typed KernelObjectAttributeFlag enum class.
//   * Converts to the raw OBJECT_ATTRIBUTES* via native() for NT API
//     calls without exposing the Windows-private layout to non-Windows
//     translation units.
//   * Compiles to a single zero-sized empty type on non-Windows so
//     downstream code can `#include` it unconditionally without
//     `#ifdef _WIN32` ladders at every call site.
//
// Use case: Cardinal subsystems that need explicit kernel-object
// control — handle inheritance for child processes, named events /
// mutexes shared across security boundaries, registry access with
// strict case sensitivity, or any low-level scenario the high-level
// CreateFile / CreateEvent wrappers don't expose. The class lives in
// cardinal::core so any module (engine, plugin host, sandbox) can use
// it without taking a Windows-specific dependency.
//
// Foundation rule: std headers stay inside cardinal::core. <Windows.h>
// + <winternl.h> are NOT std headers; cardinal::core is the right
// home for the Windows kernel surface.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/platform.hpp>
#include <cardinal/core/containers.hpp>

namespace cardinal::core {

// ---------------------------------------------------------------------------
// KernelObjectAttributeFlag — typed enum-class mirror of the OBJ_*
// constants. Defined unconditionally (visible on every platform) so
// downstream code can compose flag sets in cross-platform headers.
// Values match Windows OBJ_* exactly so a static_cast<u32> matches
// what NT expects in OBJECT_ATTRIBUTES::Attributes.
// ---------------------------------------------------------------------------
enum class KernelObjectAttributeFlag : u32 {
    None                        = 0x00000000,
    Inherit                     = 0x00000002,  // OBJ_INHERIT — handle inherits to child process
    Permanent                   = 0x00000010,  // OBJ_PERMANENT — object survives last handle close
    Exclusive                   = 0x00000020,  // OBJ_EXCLUSIVE — no other process may open
    CaseInsensitive             = 0x00000040,  // OBJ_CASE_INSENSITIVE — name lookup ignores case
    OpenIf                      = 0x00000080,  // OBJ_OPENIF — open if exists, else create
    OpenLink                    = 0x00000100,  // OBJ_OPENLINK — open the symlink itself, not the target
    KernelHandle                = 0x00000200,  // OBJ_KERNEL_HANDLE — kernel-mode handle
    ForceAccessCheck            = 0x00000400,  // OBJ_FORCE_ACCESS_CHECK — apply ACL check even in kernel
    IgnoreImpersonatedDevicemap = 0x00000800,  // OBJ_IGNORE_IMPERSONATED_DEVICEMAP
    DontReparse                 = 0x00001000,  // OBJ_DONT_REPARSE — don't follow reparse points
};

// Bitwise operators on the flag enum so callers can compose with
// the natural `a | b` syntax instead of static_cast gymnastics.
inline constexpr KernelObjectAttributeFlag operator|(KernelObjectAttributeFlag a,
                                                     KernelObjectAttributeFlag b) noexcept {
    return static_cast<KernelObjectAttributeFlag>(
        static_cast<u32>(a) | static_cast<u32>(b));
}
inline constexpr KernelObjectAttributeFlag operator&(KernelObjectAttributeFlag a,
                                                     KernelObjectAttributeFlag b) noexcept {
    return static_cast<KernelObjectAttributeFlag>(
        static_cast<u32>(a) & static_cast<u32>(b));
}
inline constexpr KernelObjectAttributeFlag operator~(KernelObjectAttributeFlag a) noexcept {
    return static_cast<KernelObjectAttributeFlag>(~static_cast<u32>(a));
}
inline constexpr KernelObjectAttributeFlag& operator|=(KernelObjectAttributeFlag& a,
                                                       KernelObjectAttributeFlag b) noexcept {
    a = a | b; return a;
}
inline constexpr KernelObjectAttributeFlag& operator&=(KernelObjectAttributeFlag& a,
                                                       KernelObjectAttributeFlag b) noexcept {
    a = a & b; return a;
}
inline constexpr bool has(KernelObjectAttributeFlag set,
                          KernelObjectAttributeFlag flag) noexcept {
    return (static_cast<u32>(set) & static_cast<u32>(flag)) != 0u;
}

// ---------------------------------------------------------------------------
// KernelObjectAttribute — RAII wrapper around OBJECT_ATTRIBUTES.
//
// Use:
//   auto attrs = KernelObjectAttribute()
//       .name(L"\\Sessions\\1\\BaseNamedObjects\\Cardinal_GraphFence")
//       .add_flags(KernelObjectAttributeFlag::CaseInsensitive
//                | KernelObjectAttributeFlag::OpenIf);
//   NtCreateEvent(&handle, EVENT_ALL_ACCESS, attrs.native(),
//                 NotificationEvent, FALSE);
//
// The native() pointer is valid as long as the KernelObjectAttribute is
// alive. Calling name(...) reallocates the backing wide-character
// buffer, which preserves the UNICODE_STRING that OBJECT_ATTRIBUTES
// points to (the wrapper rewrites the UNICODE_STRING after every name
// change). It is safe to call native() repeatedly.
//
// On non-Windows, the class is an empty type with no-op builders;
// native() returns nullptr. Downstream cross-platform code can hold a
// KernelObjectAttribute member unconditionally and call its builders
// without #ifdef ladders — the NT API call itself is the only thing
// that needs to be guarded.
// ---------------------------------------------------------------------------
class KernelObjectAttribute {
public:
    KernelObjectAttribute() noexcept;
    ~KernelObjectAttribute();

    KernelObjectAttribute(const KernelObjectAttribute&)            = delete;
    KernelObjectAttribute& operator=(const KernelObjectAttribute&) = delete;
    KernelObjectAttribute(KernelObjectAttribute&& other) noexcept;
    KernelObjectAttribute& operator=(KernelObjectAttribute&& other) noexcept;

    // ---- Fluent builders ------------------------------------------------
    // Each returns *this for chaining. set_/with_ naming follows the
    // engine's existing builder convention.

    // Set the object name. Copies the string into an internal buffer so
    // the caller's lifetime doesn't have to outlive this object.
    KernelObjectAttribute& name(const wchar_t* utf16_name) noexcept;
    // Clear the name (returns to "no named object" — the default).
    KernelObjectAttribute& clear_name() noexcept;

    // Set the root-directory handle. Subsequent name lookups are
    // relative to this handle (NT API convention). Default = nullptr
    // (absolute lookup). The handle is borrowed — caller owns lifetime.
    KernelObjectAttribute& root_directory(void* handle) noexcept;

    // Set the security descriptor pointer. Borrowed — caller owns.
    KernelObjectAttribute& security_descriptor(void* sd) noexcept;

    // Replace / OR-in the flag set.
    KernelObjectAttribute& flags(KernelObjectAttributeFlag f) noexcept;
    KernelObjectAttribute& add_flags(KernelObjectAttributeFlag f) noexcept;
    KernelObjectAttribute& clear_flags(KernelObjectAttributeFlag f) noexcept;

    // Convenience single-flag shortcuts.
    KernelObjectAttribute& inherit() noexcept          { return add_flags(KernelObjectAttributeFlag::Inherit); }
    KernelObjectAttribute& permanent() noexcept        { return add_flags(KernelObjectAttributeFlag::Permanent); }
    KernelObjectAttribute& exclusive() noexcept        { return add_flags(KernelObjectAttributeFlag::Exclusive); }
    KernelObjectAttribute& case_insensitive() noexcept { return add_flags(KernelObjectAttributeFlag::CaseInsensitive); }
    KernelObjectAttribute& open_if() noexcept          { return add_flags(KernelObjectAttributeFlag::OpenIf); }
    KernelObjectAttribute& kernel_handle() noexcept    { return add_flags(KernelObjectAttributeFlag::KernelHandle); }
    KernelObjectAttribute& dont_reparse() noexcept     { return add_flags(KernelObjectAttributeFlag::DontReparse); }

    // ---- Read-only accessors --------------------------------------------
    KernelObjectAttributeFlag attribute_flags() const noexcept { return flags_; }
    bool has_flag(KernelObjectAttributeFlag f) const noexcept {
        return cardinal::core::has(flags_, f);
    }
    void* root_directory_handle() const noexcept { return root_directory_; }
    void* security_descriptor_ptr() const noexcept { return security_descriptor_; }
    const wchar_t* name_cstr() const noexcept;
    u32  name_length_chars() const noexcept;   // wide-char count, not bytes

    // ---- Native conversion ---------------------------------------------
    // Returns OBJECT_ATTRIBUTES* on Windows (typed as void* on the
    // header to avoid forcing every consumer to include <winternl.h>).
    // Cast the result to POBJECT_ATTRIBUTES at the NT API call site.
    // Returns nullptr on non-Windows.
    void*       native()       noexcept;
    const void* native() const noexcept;

    // ---- Diagnostics surface -------------------------------------------
    // True iff this object holds a non-empty name string.
    bool        has_name() const noexcept;

private:
    void reseat_unicode_string_() noexcept;
    void release_() noexcept;

    KernelObjectAttributeFlag flags_              {KernelObjectAttributeFlag::None};
    void*                     root_directory_     {nullptr};
    void*                     security_descriptor_{nullptr};
    cardinal::vector<wchar_t> name_buffer_;          // owns the UTF-16 name (incl. trailing 0)

    // Opaque storage for the native OBJECT_ATTRIBUTES + UNICODE_STRING.
    // Sized to fit the Windows structures (validated by static_assert
    // in the .cpp); no Windows headers leak through this .hpp.
    // 96 bytes is comfortably above the actual size:
    //   OBJECT_ATTRIBUTES = 48 B (x64) — 6 pointer/u32 fields aligned
    //   UNICODE_STRING    = 16 B (x64) — 2 USHORT + 1 PWSTR
    // We carry both in a single buffer; the impl lays them out side-
    // by-side and points OBJECT_ATTRIBUTES::ObjectName at the
    // UNICODE_STRING that sits behind it in the same buffer.
    alignas(void*) cardinal::u8 native_storage_[96] {};
};

}  // namespace cardinal::core
