#pragma once

// =============================================================================
// Cardinal core — SehCallback / SehManager — modern C++20 port of
// Pearl Abyss PaSeh.h.
//
// Wraps Windows SEH (SetUnhandledExceptionFilter + MiniDumpWriteDump) into
// the PA call-vocabulary. Coexists with cardinal::core::crash (which already
// owns the engine's unhandled-exception filter); SehManager is the
// alternate API surface PA-style code uses.
//
// Compiles to a stub on non-Windows (set_handler returns ENOSYS-equivalent;
// dump_mini does nothing). Cross-platform callers can hold a SehManager
// reference and call .set_handler() unconditionally — the call is a no-op
// off Win.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/platform.hpp>

namespace cardinal::core {

// ---------------------------------------------------------------------------
// SehCallback — virtual base class. Subclass + register with SehManager to
// receive (OnLoggingStart -> OnLoggingEnd) + OnMinidumpWrote notifications
// when an unhandled SEH exception fires.
// ---------------------------------------------------------------------------
class SehCallback {
public:
    SehCallback() noexcept          = default;
    virtual ~SehCallback() noexcept = default;

    SehCallback(const SehCallback&)            = delete;
    SehCallback& operator=(const SehCallback&) = delete;

    virtual void on_logging_start(const char* message) noexcept;
    virtual void on_logging_end() noexcept;
    virtual void on_minidump_wrote(const wchar_t* path, i32 err_no) noexcept;
};

// ---------------------------------------------------------------------------
// SehManager — singleton SEH dispatch hub. set_handler installs a
// SetUnhandledExceptionFilter; dump_mini writes a minidump on demand.
// ---------------------------------------------------------------------------
class SehManager {
public:
    [[nodiscard]] static SehManager& instance() noexcept;

    SehManager(const SehManager&)            = delete;
    SehManager& operator=(const SehManager&) = delete;

    // Install the SEH filter. dump_path is the prefix for emitted .dmp files
    // (the filter appends a YYYYMMDDHHMMSS timestamp + ".dmp"). user_message
    // is embedded in the minidump's comment section. log_callstack controls
    // whether the filter spends time symbolicating the stack before writing.
    [[nodiscard]] i32 set_handler(const wchar_t* dump_path,
                                  const char*    user_message,
                                  bool           log_callstack,
                                  SehCallback*   callback) noexcept;

    // Restore the previous SEH filter.
    void dont_catch_exception() noexcept;

    [[nodiscard]] bool is_set() const noexcept;

    // Set the "user context" that goes into every minidump: who was logged
    // in, which transaction was running, character name + id. Useful for
    // post-mortem triage on server crashes.
    void set_user_information(const wchar_t* user_name,
                              i64            user_no,
                              const wchar_t* tr_description,
                              const wchar_t* character_name,
                              i64            character_no) noexcept;

    // Write a minidump on demand (no exception context). Returns 0 on
    // success; OS error code on failure.
    [[nodiscard]] i32 dump_mini(bool dump_exception_information = false) noexcept;

    // Override the next dump's path (also used by dump_mini).
    void                 set_dump_file_name(const wchar_t* path) noexcept;
    [[nodiscard]] const wchar_t* dump_file_name() const noexcept;

private:
    SehManager() noexcept;
    ~SehManager() noexcept;

    struct Impl;
    Impl* impl_;
};

}  // namespace cardinal::core
