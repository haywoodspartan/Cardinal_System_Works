#pragma once

// =============================================================================
// cardinal::core::StringBuilder — efficient incremental string assembly.
//
// Building strings with `a + b + to_string(n) + ...` allocates a temporary
// per `+` and reallocates as it grows. StringBuilder appends in place into one
// amortized-growth buffer with TYPED appenders (no manual snprintf at each
// call site) and a fluent interface — for log lines, serialised text, path
// joining, generated source/shader strings.
//
// Backed by cardinal::string, so it inherits small-string optimisation (short
// results never touch the heap) and exponential growth.
//
// FOUNDATION RULE: lives in cardinal::core (uses std + the core string/cstdio).
// Exposed as cardinal::StringBuilder.
// =============================================================================

#include <cardinal/core/types.hpp>     // cardinal::string, i64/u64, usize
#include <cardinal/core/cstdio.hpp>    // cardinal::snprintf
#include <cardinal/core/utility.hpp>   // cardinal::move

namespace cardinal::core {

class StringBuilder {
public:
    StringBuilder() = default;
    explicit StringBuilder(cardinal::usize reserve_bytes) { buf_.reserve(reserve_bytes); }

    // ---- raw / text appenders ----------------------------------------
    StringBuilder& append(const char* s) { if (s) buf_.append(s); return *this; }
    StringBuilder& append(const char* s, cardinal::usize n) { buf_.append(s, n); return *this; }
    StringBuilder& append(const cardinal::string& s) { buf_.append(s); return *this; }
    StringBuilder& append(char c) { buf_.push_back(c); return *this; }
    StringBuilder& append(bool b) { buf_.append(b ? "true" : "false"); return *this; }
    StringBuilder& append_repeat(char c, cardinal::usize count) { buf_.append(count, c); return *this; }
    StringBuilder& append_line(const char* s = "") { append(s); buf_.push_back('\n'); return *this; }

    // ---- numeric appenders -------------------------------------------
    StringBuilder& append_int(cardinal::i64 v)  { return fmt_("%lld",  static_cast<long long>(v)); }
    StringBuilder& append_uint(cardinal::u64 v) { return fmt_("%llu",  static_cast<unsigned long long>(v)); }
    StringBuilder& append_hex(cardinal::u64 v, bool prefix = true) {
        return fmt_(prefix ? "0x%llx" : "%llx", static_cast<unsigned long long>(v));
    }
    StringBuilder& append_float(double v, int precision = 6) {
        char b[64];
        int n = cardinal::snprintf(b, sizeof(b), "%.*g", precision, v);
        if (n > 0) buf_.append(b, clamp_len_(n, sizeof(b)));
        return *this;
    }

    // ---- fluent operator<< -------------------------------------------
    StringBuilder& operator<<(const char* s)            { return append(s); }
    StringBuilder& operator<<(const cardinal::string& s){ return append(s); }
    StringBuilder& operator<<(char c)                   { return append(c); }
    StringBuilder& operator<<(bool b)                   { return append(b); }
    StringBuilder& operator<<(cardinal::i32 v)          { return append_int(v); }
    StringBuilder& operator<<(cardinal::i64 v)          { return append_int(v); }
    StringBuilder& operator<<(cardinal::u32 v)          { return append_uint(v); }
    StringBuilder& operator<<(cardinal::u64 v)          { return append_uint(v); }
    StringBuilder& operator<<(double v)                 { return append_float(v); }
    StringBuilder& operator<<(float v)                  { return append_float(static_cast<double>(v)); }

    // ---- access ------------------------------------------------------
    void reserve(cardinal::usize n)       { buf_.reserve(n); }
    StringBuilder& clear() noexcept       { buf_.clear(); return *this; }
    [[nodiscard]] bool empty() const noexcept { return buf_.empty(); }
    [[nodiscard]] cardinal::usize size() const noexcept { return buf_.size(); }
    [[nodiscard]] const cardinal::string& str() const noexcept { return buf_; }
    [[nodiscard]] const char* c_str() const noexcept { return buf_.c_str(); }

    // Move the built string out, leaving the builder empty + reusable.
    [[nodiscard]] cardinal::string take() {
        cardinal::string out = cardinal::move(buf_);
        buf_.clear();
        return out;
    }

private:
    cardinal::string buf_;

    static cardinal::usize clamp_len_(int n, cardinal::usize cap) noexcept {
        const cardinal::usize un = static_cast<cardinal::usize>(n);
        return un < cap ? un : (cap - 1);   // guard snprintf truncation
    }
    template <class T>
    StringBuilder& fmt_(const char* spec, T v) {
        char b[24];
        int n = cardinal::snprintf(b, sizeof(b), spec, v);
        if (n > 0) buf_.append(b, clamp_len_(n, sizeof(b)));
        return *this;
    }
};

}  // namespace cardinal::core

namespace cardinal {
using StringBuilder = core::StringBuilder;
}  // namespace cardinal
