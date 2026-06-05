#pragma once

// =============================================================================
// Cardinal core — StringA<N> / StringW<N> — modern C++20 port of
// the fixed-string surface's fixed-capacity stack strings.
//
// Design: stack-allocated N+1 buffers (room for NUL), no heap, never throws.
// Truncates silently on overflow (matches — matches strcpy_s with
// _TRUNCATE). Used by code paths for short identifiers / labels /
// log fragments where heap traffic is unwanted.
//
// Modernisation notes:
//   * Replaced manual va_list+vsnprintf with constexpr-friendly snprintf via
//     <cstdio>. va_list path still available via format(...) for runtime
//     format strings.
//   * char↔wchar conversion via MultiByteToWideChar on Windows; on Linux
//     uses std::mbstowcs (UTF-8 locale assumed — matches engine convention).
//   * Operator overloads `operator=`/`operator+=` keep a familiar string-class feel.
//   * Conversion operators to `const char*` / `const wchar_t*` so the string
//     drops straight into printf/wprintf / NT API calls.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/platform.hpp>

#include <cstdio>      // snprintf, vsnprintf
#include <cstdarg>     // va_list
#include <cstring>     // memcpy, strlen
#include <cwchar>      // wcslen, swprintf, vswprintf

#if CARDINAL_PLATFORM_WINDOWS
#include <Windows.h>   // WideCharToMultiByte / MultiByteToWideChar
#endif

namespace cardinal::core {

// ---------------------------------------------------------------------------
// detail — narrow ↔ wide conversion helpers, codepage UTF-8 on Win.
// ---------------------------------------------------------------------------
namespace detail {

inline usize wide_to_narrow(const wchar_t* src, char* dst, usize dst_cap) noexcept {
    if (src == nullptr || dst == nullptr || dst_cap == 0) return 0;
#if CARDINAL_PLATFORM_WINDOWS
    const int n = ::WideCharToMultiByte(CP_UTF8, 0, src, -1, dst, static_cast<int>(dst_cap), nullptr, nullptr);
    if (n <= 0) { dst[0] = '\0'; return 0; }
    return static_cast<usize>(n - 1);
#else
    const std::size_t n = std::wcstombs(dst, src, dst_cap - 1);
    if (n == static_cast<std::size_t>(-1)) { dst[0] = '\0'; return 0; }
    dst[n] = '\0';
    return n;
#endif
}

inline usize narrow_to_wide(const char* src, wchar_t* dst, usize dst_cap) noexcept {
    if (src == nullptr || dst == nullptr || dst_cap == 0) return 0;
#if CARDINAL_PLATFORM_WINDOWS
    const int n = ::MultiByteToWideChar(CP_UTF8, 0, src, -1, dst, static_cast<int>(dst_cap));
    if (n <= 0) { dst[0] = L'\0'; return 0; }
    return static_cast<usize>(n - 1);
#else
    const std::size_t n = std::mbstowcs(dst, src, dst_cap - 1);
    if (n == static_cast<std::size_t>(-1)) { dst[0] = L'\0'; return 0; }
    dst[n] = L'\0';
    return n;
#endif
}

}  // namespace detail

// ---------------------------------------------------------------------------
// StringA<N> — fixed-capacity ASCII/UTF-8 stack string (capacity excl. NUL).
// ---------------------------------------------------------------------------
template <u32 Capacity>
class StringA {
public:
    static constexpr u32 kCapacity = Capacity;

    StringA() noexcept                          { buffer_[0] = '\0'; }
    explicit StringA(const char* msg) noexcept  { set(msg); }
    explicit StringA(const wchar_t* msg) noexcept { set(msg); }

    void set(const char* msg) noexcept {
        if (msg == nullptr) { buffer_[0] = '\0'; return; }
        usize n = 0;
        while (msg[n] != '\0' && n < Capacity) { buffer_[n] = msg[n]; ++n; }
        buffer_[n] = '\0';
    }
    void set(const char* msg, u32 length) noexcept {
        const u32 n = (length < Capacity) ? length : Capacity;
        for (u32 i = 0; i < n; ++i) buffer_[i] = msg[i];
        buffer_[n] = '\0';
    }
    void set(const wchar_t* msg) noexcept {
        detail::wide_to_narrow(msg, buffer_, Capacity + 1u);
    }

    void append(const char* msg) noexcept {
        if (msg == nullptr) return;
        usize n = length();
        while (msg[0] != '\0' && n < Capacity) { buffer_[n++] = *msg++; }
        buffer_[n] = '\0';
    }

    // Printf-like formatter. Truncates silently on overflow.
    void format(const char* fmt, ...) noexcept {
        std::va_list args;
        va_start(args, fmt);
        ::std::vsnprintf(buffer_, Capacity + 1u, fmt, args);
        va_end(args);
        buffer_[Capacity] = '\0';
    }

    StringA& operator=(const char* msg) noexcept    { set(msg); return *this; }
    StringA& operator=(const wchar_t* msg) noexcept { set(msg); return *this; }
    StringA& operator+=(const char* msg) noexcept   { append(msg); return *this; }

    [[nodiscard]] char&       operator[](u32 i)       noexcept { return buffer_[i]; }
    [[nodiscard]] const char& operator[](u32 i) const noexcept { return buffer_[i]; }

    void replace(char from, char to) noexcept {
        for (auto p = buffer_; *p != '\0'; ++p) { if (*p == from) *p = to; }
    }

    void reset() noexcept { buffer_[0] = '\0'; }
    [[nodiscard]] bool         is_null()  const noexcept { return buffer_[0] == '\0'; }
    [[nodiscard]] const char*  c_str()    const noexcept { return buffer_; }
    [[nodiscard]] const char*  get_message() const noexcept { return buffer_; }
    [[nodiscard]] char*        buffer()         noexcept { return buffer_; }
    [[nodiscard]] u32          capacity() const noexcept { return Capacity; }
    [[nodiscard]] usize        length()   const noexcept {
        usize n = 0; while (n <= Capacity && buffer_[n] != '\0') ++n; return n;
    }

    operator const char*() const noexcept { return buffer_; }

private:
    char buffer_[Capacity + 1u];
};

// ---------------------------------------------------------------------------
// StringW<N> — fixed-capacity UTF-16 stack string (capacity excl. NUL).
// ---------------------------------------------------------------------------
template <u32 Capacity>
class StringW {
public:
    static constexpr u32 kCapacity = Capacity;

    StringW() noexcept                          { buffer_[0] = L'\0'; }
    explicit StringW(const char* msg) noexcept  { set(msg); }
    explicit StringW(const wchar_t* msg) noexcept { set(msg); }

    void set(const wchar_t* msg) noexcept {
        if (msg == nullptr) { buffer_[0] = L'\0'; return; }
        usize n = 0;
        while (msg[n] != L'\0' && n < Capacity) { buffer_[n] = msg[n]; ++n; }
        buffer_[n] = L'\0';
    }
    void set(const wchar_t* msg, u32 length) noexcept {
        const u32 n = (length < Capacity) ? length : Capacity;
        for (u32 i = 0; i < n; ++i) buffer_[i] = msg[i];
        buffer_[n] = L'\0';
    }
    void set(const char* msg) noexcept {
        detail::narrow_to_wide(msg, buffer_, Capacity + 1u);
    }

    void append(const wchar_t* msg) noexcept {
        if (msg == nullptr) return;
        usize n = length();
        while (msg[0] != L'\0' && n < Capacity) { buffer_[n++] = *msg++; }
        buffer_[n] = L'\0';
    }

    void format(const wchar_t* fmt, ...) noexcept {
        std::va_list args;
        va_start(args, fmt);
#if CARDINAL_PLATFORM_WINDOWS
        ::_vsnwprintf_s(buffer_, Capacity + 1u, _TRUNCATE, fmt, args);
#else
        ::vswprintf(buffer_, Capacity + 1u, fmt, args);
#endif
        va_end(args);
        buffer_[Capacity] = L'\0';
    }

    StringW& operator=(const char* msg) noexcept    { set(msg); return *this; }
    StringW& operator=(const wchar_t* msg) noexcept { set(msg); return *this; }
    StringW& operator+=(const wchar_t* msg) noexcept{ append(msg); return *this; }

    [[nodiscard]] wchar_t&       operator[](u32 i)       noexcept { return buffer_[i]; }
    [[nodiscard]] const wchar_t& operator[](u32 i) const noexcept { return buffer_[i]; }

    void replace(wchar_t from, wchar_t to) noexcept {
        for (auto p = buffer_; *p != L'\0'; ++p) { if (*p == from) *p = to; }
    }

    void reset() noexcept { buffer_[0] = L'\0'; }
    [[nodiscard]] bool             is_null()  const noexcept { return buffer_[0] == L'\0'; }
    [[nodiscard]] const wchar_t*   c_str()    const noexcept { return buffer_; }
    [[nodiscard]] const wchar_t*   get_message() const noexcept { return buffer_; }
    [[nodiscard]] wchar_t*         buffer()         noexcept { return buffer_; }
    [[nodiscard]] u32              capacity() const noexcept { return Capacity; }
    [[nodiscard]] usize            length()   const noexcept {
        usize n = 0; while (n <= Capacity && buffer_[n] != L'\0') ++n; return n;
    }

    operator const wchar_t*() const noexcept { return buffer_; }

private:
    wchar_t buffer_[Capacity + 1u];
};

}  // namespace cardinal::core
