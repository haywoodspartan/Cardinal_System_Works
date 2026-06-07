// =============================================================================
// Cardinal — KernelObjectAttribute regression suite.
//
// Exercises the builder-pattern wrapper around Windows OBJECT_ATTRIBUTES:
//   * default construction is a valid empty OA
//   * name() copies the wide string and the UNICODE_STRING reflects it
//   * flag composition via add_flags / clear_flags / convenience shortcuts
//   * root_directory / security_descriptor pass-through
//   * native() returns a non-null pointer on Windows + nullptr off-Windows
//   * move semantics: target owns its own UNICODE_STRING (not pointing back
//     into the source object's storage)
//   * The OA's ObjectName field actually points at this object's embedded
//     UNICODE_STRING (so NT API calls reach the right buffer)
//
// Exit 0 = all pass.
// =============================================================================

#include <cardinal/core/os/kernel_object_attribute.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/std/utility.hpp>

#if CARDINAL_PLATFORM_WINDOWS
#include <Windows.h>
#include <winternl.h>
#endif

namespace {

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("koa", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

using cardinal::core::KernelObjectAttribute;
using cardinal::core::KernelObjectAttributeFlag;

// ---------------------------------------------------------------------------
void test_default_state() {
    KernelObjectAttribute a;
    CHECK(a.attribute_flags() == KernelObjectAttributeFlag::None);
    CHECK(a.root_directory_handle() == nullptr);
    CHECK(a.security_descriptor_ptr() == nullptr);
    CHECK(a.name_cstr() == nullptr);
    CHECK(a.name_length_chars() == 0u);
    CHECK(!a.has_name());
#if CARDINAL_PLATFORM_WINDOWS
    CHECK(a.native() != nullptr);   // OA is always present on Windows
#else
    CHECK(a.native() == nullptr);   // no-op stub off Windows
#endif
}

// ---------------------------------------------------------------------------
void test_flag_bitops() {
    auto a = KernelObjectAttributeFlag::CaseInsensitive | KernelObjectAttributeFlag::OpenIf;
    CHECK(cardinal::core::has(a, KernelObjectAttributeFlag::CaseInsensitive));
    CHECK(cardinal::core::has(a, KernelObjectAttributeFlag::OpenIf));
    CHECK(!cardinal::core::has(a, KernelObjectAttributeFlag::Inherit));

    auto b = a & KernelObjectAttributeFlag::OpenIf;
    CHECK(b == KernelObjectAttributeFlag::OpenIf);

    auto c = ~KernelObjectAttributeFlag::None;
    CHECK(cardinal::core::has(c, KernelObjectAttributeFlag::Inherit));
}

// ---------------------------------------------------------------------------
void test_builder_flags() {
    KernelObjectAttribute a;
    a.case_insensitive().inherit().open_if();
    CHECK(a.has_flag(KernelObjectAttributeFlag::CaseInsensitive));
    CHECK(a.has_flag(KernelObjectAttributeFlag::Inherit));
    CHECK(a.has_flag(KernelObjectAttributeFlag::OpenIf));
    CHECK(!a.has_flag(KernelObjectAttributeFlag::Permanent));

    a.clear_flags(KernelObjectAttributeFlag::Inherit);
    CHECK(!a.has_flag(KernelObjectAttributeFlag::Inherit));
    CHECK(a.has_flag(KernelObjectAttributeFlag::CaseInsensitive));

    a.flags(KernelObjectAttributeFlag::KernelHandle);     // replace
    CHECK(a.has_flag(KernelObjectAttributeFlag::KernelHandle));
    CHECK(!a.has_flag(KernelObjectAttributeFlag::CaseInsensitive));
}

// ---------------------------------------------------------------------------
void test_name_copy() {
    KernelObjectAttribute a;
    a.name(L"Cardinal_TestEvent");
    CHECK(a.has_name());
    CHECK(a.name_length_chars() == 18u);   // "Cardinal_TestEvent"
    const wchar_t* p = a.name_cstr();
    CHECK(p != nullptr);
    CHECK(p[0] == L'C');
    CHECK(p[17] == L't');
    CHECK(p[18] == L'\0');                 // null-terminated

    a.clear_name();
    CHECK(!a.has_name());
    CHECK(a.name_cstr() == nullptr);
    CHECK(a.name_length_chars() == 0u);
}

// ---------------------------------------------------------------------------
void test_root_and_security() {
    KernelObjectAttribute a;
    int dummy_handle = 0;
    int dummy_sd     = 0;
    a.root_directory(&dummy_handle);
    a.security_descriptor(&dummy_sd);
    CHECK(a.root_directory_handle() == &dummy_handle);
    CHECK(a.security_descriptor_ptr() == &dummy_sd);
}

// ---------------------------------------------------------------------------
#if CARDINAL_PLATFORM_WINDOWS
// On Windows, verify the OBJECT_ATTRIBUTES that native() points at is
// actually internally consistent: ObjectName points at the embedded
// UNICODE_STRING; UNICODE_STRING.Buffer points at our name buffer;
// Attributes byte-equals our flag set.
void test_native_layout_windows() {
    KernelObjectAttribute a;
    a.name(L"Cardinal_NamedEvent")
     .case_insensitive()
     .open_if()
     .kernel_handle();

    auto* oa = static_cast<POBJECT_ATTRIBUTES>(a.native());
    CHECK(oa != nullptr);
    CHECK(oa->Length == sizeof(OBJECT_ATTRIBUTES));
    CHECK(oa->RootDirectory == nullptr);
    CHECK(oa->SecurityDescriptor == nullptr);
    CHECK(oa->ObjectName != nullptr);

    const ULONG expected_flags =
        OBJ_CASE_INSENSITIVE | OBJ_OPENIF | OBJ_KERNEL_HANDLE;
    CHECK(oa->Attributes == expected_flags);

    const UNICODE_STRING* us = oa->ObjectName;
    CHECK(us->Length        == 19u * sizeof(wchar_t));  // "Cardinal_NamedEvent"
    CHECK(us->MaximumLength == 20u * sizeof(wchar_t));  // incl. trailing 0
    CHECK(us->Buffer != nullptr);
    CHECK(us->Buffer[0]  == L'C');
    CHECK(us->Buffer[18] == L't');
    CHECK(us->Buffer[19] == L'\0');
}

void test_name_rebind_windows() {
    KernelObjectAttribute a;
    a.name(L"Short");
    auto* oa1 = static_cast<POBJECT_ATTRIBUTES>(a.native());
    CHECK(oa1->ObjectName->Length == 5u * sizeof(wchar_t));

    // Re-name to longer string — internal buffer realloc must be
    // followed by UNICODE_STRING reseat, so the OA still sees the
    // new buffer at the right size.
    a.name(L"MuchLongerObjectName");
    auto* oa2 = static_cast<POBJECT_ATTRIBUTES>(a.native());
    CHECK(oa2 == oa1);                                  // same OA storage
    CHECK(oa2->ObjectName->Length == 20u * sizeof(wchar_t));
    CHECK(oa2->ObjectName->Buffer != nullptr);
    CHECK(oa2->ObjectName->Buffer[0]  == L'M');
    CHECK(oa2->ObjectName->Buffer[19] == L'e');

    // Clear the name — UNICODE_STRING should go back to empty/null.
    a.clear_name();
    auto* oa3 = static_cast<POBJECT_ATTRIBUTES>(a.native());
    CHECK(oa3->ObjectName != nullptr);                  // OA still points at embedded US
    CHECK(oa3->ObjectName->Length == 0u);
    CHECK(oa3->ObjectName->Buffer == nullptr);
}
#endif  // CARDINAL_PLATFORM_WINDOWS

// ---------------------------------------------------------------------------
void test_move_semantics() {
    KernelObjectAttribute src;
    src.name(L"MovedName").exclusive().add_flags(KernelObjectAttributeFlag::Inherit);

    KernelObjectAttribute dst = cardinal::move(src);
    CHECK(dst.has_name());
    CHECK(dst.name_length_chars() == 9u);
    CHECK(dst.has_flag(KernelObjectAttributeFlag::Exclusive));
    CHECK(dst.has_flag(KernelObjectAttributeFlag::Inherit));

    // Source state after move: empty.
    CHECK(!src.has_name());
    CHECK(src.attribute_flags() == KernelObjectAttributeFlag::None);

#if CARDINAL_PLATFORM_WINDOWS
    // The moved-to object's OA::ObjectName must point at its OWN embedded
    // UNICODE_STRING — not the source's. If it pointed at the source, a
    // subsequent destructor of `src` would invalidate `dst`'s name.
    auto* dst_oa = static_cast<POBJECT_ATTRIBUTES>(dst.native());
    auto* src_oa = static_cast<POBJECT_ATTRIBUTES>(src.native());
    CHECK(dst_oa->ObjectName != src_oa->ObjectName);
    CHECK(dst_oa->ObjectName != nullptr);
    CHECK(dst_oa->ObjectName->Buffer != nullptr);
    CHECK(dst_oa->ObjectName->Buffer[0] == L'M');
#endif

    // Move-assign to a populated target.
    KernelObjectAttribute target;
    target.name(L"OldName").permanent();
    target = cardinal::move(dst);
    CHECK(target.has_name());
    CHECK(target.name_length_chars() == 9u);
    CHECK(target.has_flag(KernelObjectAttributeFlag::Exclusive));
    CHECK(!target.has_flag(KernelObjectAttributeFlag::Permanent));  // overwritten
}

}  // namespace

int main() {
    cardinal::log::infof("koa", "KernelObjectAttribute regression suite");

    test_default_state();
    test_flag_bitops();
    test_builder_flags();
    test_name_copy();
    test_root_and_security();
#if CARDINAL_PLATFORM_WINDOWS
    test_native_layout_windows();
    test_name_rebind_windows();
#endif
    test_move_semantics();

    cardinal::log::infof("koa", "checks=%d  failures=%d", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
