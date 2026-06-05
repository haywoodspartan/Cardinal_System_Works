#include <cardinal/core/kernel_object_attribute.hpp>
#include <cardinal/core/utility.hpp>

#if CARDINAL_PLATFORM_WINDOWS

#include <Windows.h>
#include <winternl.h>   // OBJECT_ATTRIBUTES + UNICODE_STRING

namespace cardinal::core {

namespace {

// Side-by-side layout inside native_storage_:
//   bytes [0  .. sizeof(OBJECT_ATTRIBUTES)) — OBJECT_ATTRIBUTES
//   bytes [N  .. N + sizeof(UNICODE_STRING)) — UNICODE_STRING (N rounded to alignof(UNICODE_STRING))
constexpr cardinal::usize kObjectAttributesSize = sizeof(OBJECT_ATTRIBUTES);
constexpr cardinal::usize kUnicodeStringSize    = sizeof(UNICODE_STRING);
constexpr cardinal::usize kUnicodeStringAlign   = alignof(UNICODE_STRING);

inline constexpr cardinal::usize align_up(cardinal::usize v, cardinal::usize a) noexcept {
    return (v + a - 1u) & ~(a - 1u);
}

inline OBJECT_ATTRIBUTES* obj_attrs(void* storage) noexcept {
    return static_cast<OBJECT_ATTRIBUTES*>(storage);
}
inline const OBJECT_ATTRIBUTES* obj_attrs(const void* storage) noexcept {
    return static_cast<const OBJECT_ATTRIBUTES*>(storage);
}

inline UNICODE_STRING* unicode_string(void* storage) noexcept {
    auto* bytes = static_cast<cardinal::u8*>(storage);
    const cardinal::usize off = align_up(kObjectAttributesSize, kUnicodeStringAlign);
    return reinterpret_cast<UNICODE_STRING*>(bytes + off);
}
inline const UNICODE_STRING* unicode_string(const void* storage) noexcept {
    auto* bytes = static_cast<const cardinal::u8*>(storage);
    const cardinal::usize off = align_up(kObjectAttributesSize, kUnicodeStringAlign);
    return reinterpret_cast<const UNICODE_STRING*>(bytes + off);
}

cardinal::usize wcslen_safe(const wchar_t* s) noexcept {
    if (s == nullptr) return 0;
    cardinal::usize n = 0;
    while (s[n] != L'\0') ++n;
    return n;
}

}  // namespace

// Compile-time sanity: the opaque storage buffer must hold both structures
// side-by-side. 96 bytes covers x64 (OA=48 + alignment-padding + US=16).
static_assert(96u >= align_up(kObjectAttributesSize, kUnicodeStringAlign) + kUnicodeStringSize,
              "KernelObjectAttribute::native_storage_ too small");

KernelObjectAttribute::KernelObjectAttribute() noexcept {
    // Zero-initialise OBJECT_ATTRIBUTES via InitializeObjectAttributes-style fill.
    auto* oa = obj_attrs(native_storage_);
    auto* us = unicode_string(native_storage_);
    oa->Length                   = static_cast<ULONG>(sizeof(OBJECT_ATTRIBUTES));
    oa->RootDirectory            = nullptr;
    oa->ObjectName               = us;   // points at our embedded UNICODE_STRING
    oa->Attributes               = 0;
    oa->SecurityDescriptor       = nullptr;
    oa->SecurityQualityOfService = nullptr;
    us->Length        = 0;
    us->MaximumLength = 0;
    us->Buffer        = nullptr;
}

KernelObjectAttribute::~KernelObjectAttribute() {
    release_();
}

KernelObjectAttribute::KernelObjectAttribute(KernelObjectAttribute&& other) noexcept
    : flags_(other.flags_)
    , root_directory_(other.root_directory_)
    , security_descriptor_(other.security_descriptor_)
    , name_buffer_(cardinal::move(other.name_buffer_))
{
    // Bit-copy the OBJECT_ATTRIBUTES + UNICODE_STRING block from `other`,
    // then re-point the OA's ObjectName at OUR embedded UNICODE_STRING so
    // the move target owns a self-contained structure.
    for (cardinal::usize i = 0; i < sizeof(native_storage_); ++i) {
        native_storage_[i] = other.native_storage_[i];
    }
    reseat_unicode_string_();

    // Zero out `other`'s OA so its destructor doesn't try to invalidate
    // a still-live UNICODE_STRING buffer (we now own it).
    auto* oa_other = obj_attrs(other.native_storage_);
    auto* us_other = unicode_string(other.native_storage_);
    oa_other->RootDirectory      = nullptr;
    oa_other->Attributes         = 0;
    oa_other->SecurityDescriptor = nullptr;
    us_other->Length        = 0;
    us_other->MaximumLength = 0;
    us_other->Buffer        = nullptr;
    other.flags_              = KernelObjectAttributeFlag::None;
    other.root_directory_     = nullptr;
    other.security_descriptor_= nullptr;
}

KernelObjectAttribute& KernelObjectAttribute::operator=(KernelObjectAttribute&& other) noexcept {
    if (this == &other) return *this;
    release_();
    flags_               = other.flags_;
    root_directory_      = other.root_directory_;
    security_descriptor_ = other.security_descriptor_;
    name_buffer_         = cardinal::move(other.name_buffer_);
    for (cardinal::usize i = 0; i < sizeof(native_storage_); ++i) {
        native_storage_[i] = other.native_storage_[i];
    }
    reseat_unicode_string_();
    auto* oa_other = obj_attrs(other.native_storage_);
    auto* us_other = unicode_string(other.native_storage_);
    oa_other->RootDirectory      = nullptr;
    oa_other->Attributes         = 0;
    oa_other->SecurityDescriptor = nullptr;
    us_other->Length        = 0;
    us_other->MaximumLength = 0;
    us_other->Buffer        = nullptr;
    other.flags_              = KernelObjectAttributeFlag::None;
    other.root_directory_     = nullptr;
    other.security_descriptor_= nullptr;
    return *this;
}

void KernelObjectAttribute::release_() noexcept {
    auto* oa = obj_attrs(native_storage_);
    auto* us = unicode_string(native_storage_);
    oa->Length                   = static_cast<ULONG>(sizeof(OBJECT_ATTRIBUTES));
    oa->RootDirectory            = nullptr;
    oa->ObjectName               = us;
    oa->Attributes               = 0;
    oa->SecurityDescriptor       = nullptr;
    oa->SecurityQualityOfService = nullptr;
    us->Length        = 0;
    us->MaximumLength = 0;
    us->Buffer        = nullptr;
    name_buffer_.clear();
}

// Re-point the embedded UNICODE_STRING at our owned name_buffer_ and
// re-point OBJECT_ATTRIBUTES::ObjectName at the embedded UNICODE_STRING.
// Called after every name change + after move so the native struct
// always references our internal storage (never the source object's).
void KernelObjectAttribute::reseat_unicode_string_() noexcept {
    auto* oa = obj_attrs(native_storage_);
    auto* us = unicode_string(native_storage_);
    oa->ObjectName = us;
    if (name_buffer_.empty()) {
        us->Length        = 0;
        us->MaximumLength = 0;
        us->Buffer        = nullptr;
    } else {
        // name_buffer_ contains the wide string + trailing null. The
        // UNICODE_STRING's Length is in BYTES and excludes the null;
        // MaximumLength includes the null terminator.
        const cardinal::usize chars_with_null = name_buffer_.size();
        const cardinal::usize chars_no_null   = (chars_with_null > 0u) ? (chars_with_null - 1u) : 0u;
        us->Length        = static_cast<USHORT>(chars_no_null * sizeof(wchar_t));
        us->MaximumLength = static_cast<USHORT>(chars_with_null * sizeof(wchar_t));
        us->Buffer        = name_buffer_.data();
    }
}

KernelObjectAttribute& KernelObjectAttribute::name(const wchar_t* utf16_name) noexcept {
    if (utf16_name == nullptr) {
        return clear_name();
    }
    const cardinal::usize len = wcslen_safe(utf16_name);
    name_buffer_.clear();
    name_buffer_.reserve(len + 1u);
    for (cardinal::usize i = 0; i < len; ++i) name_buffer_.push_back(utf16_name[i]);
    name_buffer_.push_back(L'\0');
    reseat_unicode_string_();
    return *this;
}

KernelObjectAttribute& KernelObjectAttribute::clear_name() noexcept {
    name_buffer_.clear();
    reseat_unicode_string_();
    return *this;
}

KernelObjectAttribute& KernelObjectAttribute::root_directory(void* handle) noexcept {
    root_directory_ = handle;
    obj_attrs(native_storage_)->RootDirectory = handle;
    return *this;
}

KernelObjectAttribute& KernelObjectAttribute::security_descriptor(void* sd) noexcept {
    security_descriptor_ = sd;
    obj_attrs(native_storage_)->SecurityDescriptor = sd;
    return *this;
}

KernelObjectAttribute& KernelObjectAttribute::flags(KernelObjectAttributeFlag f) noexcept {
    flags_ = f;
    obj_attrs(native_storage_)->Attributes = static_cast<ULONG>(f);
    return *this;
}

KernelObjectAttribute& KernelObjectAttribute::add_flags(KernelObjectAttributeFlag f) noexcept {
    flags_ |= f;
    obj_attrs(native_storage_)->Attributes = static_cast<ULONG>(flags_);
    return *this;
}

KernelObjectAttribute& KernelObjectAttribute::clear_flags(KernelObjectAttributeFlag f) noexcept {
    flags_ &= ~f;
    obj_attrs(native_storage_)->Attributes = static_cast<ULONG>(flags_);
    return *this;
}

const wchar_t* KernelObjectAttribute::name_cstr() const noexcept {
    return name_buffer_.empty() ? nullptr : name_buffer_.data();
}

u32 KernelObjectAttribute::name_length_chars() const noexcept {
    if (name_buffer_.empty()) return 0;
    return static_cast<u32>(name_buffer_.size() - 1u);   // excl. null
}

void* KernelObjectAttribute::native() noexcept {
    return native_storage_;
}
const void* KernelObjectAttribute::native() const noexcept {
    return native_storage_;
}

bool KernelObjectAttribute::has_name() const noexcept {
    return !name_buffer_.empty();
}

}  // namespace cardinal::core

#else  // CARDINAL_PLATFORM_WINDOWS

// Non-Windows fallback — every method is a no-op so cross-platform code
// can hold a KernelObjectAttribute member without #ifdef ladders. native()
// returns nullptr; the caller must guard the actual NT API call with
// a platform check.

namespace cardinal::core {

KernelObjectAttribute::KernelObjectAttribute() noexcept = default;
KernelObjectAttribute::~KernelObjectAttribute() = default;

KernelObjectAttribute::KernelObjectAttribute(KernelObjectAttribute&& other) noexcept
    : flags_(other.flags_)
    , root_directory_(other.root_directory_)
    , security_descriptor_(other.security_descriptor_)
    , name_buffer_(cardinal::move(other.name_buffer_))
{
    other.flags_              = KernelObjectAttributeFlag::None;
    other.root_directory_     = nullptr;
    other.security_descriptor_= nullptr;
}

KernelObjectAttribute& KernelObjectAttribute::operator=(KernelObjectAttribute&& other) noexcept {
    if (this == &other) return *this;
    flags_               = other.flags_;
    root_directory_      = other.root_directory_;
    security_descriptor_ = other.security_descriptor_;
    name_buffer_         = cardinal::move(other.name_buffer_);
    other.flags_              = KernelObjectAttributeFlag::None;
    other.root_directory_     = nullptr;
    other.security_descriptor_= nullptr;
    return *this;
}

KernelObjectAttribute& KernelObjectAttribute::name(const wchar_t* utf16_name) noexcept {
    name_buffer_.clear();
    if (utf16_name == nullptr) return *this;
    cardinal::usize n = 0;
    while (utf16_name[n] != L'\0') ++n;
    name_buffer_.reserve(n + 1u);
    for (cardinal::usize i = 0; i < n; ++i) name_buffer_.push_back(utf16_name[i]);
    name_buffer_.push_back(L'\0');
    return *this;
}
KernelObjectAttribute& KernelObjectAttribute::clear_name() noexcept {
    name_buffer_.clear(); return *this;
}
KernelObjectAttribute& KernelObjectAttribute::root_directory(void* h) noexcept {
    root_directory_ = h; return *this;
}
KernelObjectAttribute& KernelObjectAttribute::security_descriptor(void* sd) noexcept {
    security_descriptor_ = sd; return *this;
}
KernelObjectAttribute& KernelObjectAttribute::flags(KernelObjectAttributeFlag f) noexcept {
    flags_ = f; return *this;
}
KernelObjectAttribute& KernelObjectAttribute::add_flags(KernelObjectAttributeFlag f) noexcept {
    flags_ |= f; return *this;
}
KernelObjectAttribute& KernelObjectAttribute::clear_flags(KernelObjectAttributeFlag f) noexcept {
    flags_ &= ~f; return *this;
}

const wchar_t* KernelObjectAttribute::name_cstr() const noexcept {
    return name_buffer_.empty() ? nullptr : name_buffer_.data();
}
u32 KernelObjectAttribute::name_length_chars() const noexcept {
    return name_buffer_.empty() ? 0u : static_cast<u32>(name_buffer_.size() - 1u);
}
void*       KernelObjectAttribute::native()       noexcept { return nullptr; }
const void* KernelObjectAttribute::native() const noexcept { return nullptr; }
bool        KernelObjectAttribute::has_name() const noexcept { return !name_buffer_.empty(); }

// Helper-fn stubs only referenced on the Windows path.
void KernelObjectAttribute::reseat_unicode_string_() noexcept {}
void KernelObjectAttribute::release_() noexcept { name_buffer_.clear(); }

}  // namespace cardinal::core

#endif  // CARDINAL_PLATFORM_WINDOWS
