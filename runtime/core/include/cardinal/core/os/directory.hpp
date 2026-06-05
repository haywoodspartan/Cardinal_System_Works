#pragma once

// =============================================================================
// Cardinal core — Directory — modern C++20 port of Pearl Abyss
// PaDirectory.h.
//
// Wraps std::filesystem::directory_iterator into the begin()/next()/end()/
// get() iteration shape PA's code expects. Static helpers (Make / Move /
// Copy / Remove / Rename / ChangeCurrentDirectory) are thin wrappers around
// the std::filesystem equivalents — they propagate the error_code as the
// returned i32 (Windows error-style: 0 = success).
//
// Wildcard support: PA uses Win32 FindFirstFile wildcards (`*`, `?`) in the
// begin() path. We honour that by splitting the input into a parent directory
// + filename pattern and filtering directory_iterator results with a small
// glob matcher. Simpler than the regex route and matches the FFI semantics.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/platform.hpp>

#include <filesystem>
#include <string>      // std::wstring — small allocation per Directory instance

namespace cardinal::core {

// ---------------------------------------------------------------------------
// Entry — minimal stand-in for WIN32_FIND_DATAW (the fields PA call sites
// actually touch). Returned by Directory::get().
// ---------------------------------------------------------------------------
struct DirectoryEntry {
    std::wstring file_name;        // leaf name only (no path)
    u64          size_bytes;       // 0 if directory
    bool         is_directory;
};

class Directory {
public:
    Directory() noexcept;
    ~Directory() noexcept = default;

    Directory(const Directory&)            = delete;
    Directory& operator=(const Directory&) = delete;

    // Begin enumeration. `path` may include a wildcard filename component
    // (e.g. L"./logs/*.txt"). Returns 0 on success; ERROR_FILE_NOT_FOUND
    // (2) if no entries match.
    [[nodiscard]] i32  begin(const wchar_t* path) noexcept;
    // Advance to the next entry. Returns 0; ERROR_NO_MORE_FILES (18) at end.
    [[nodiscard]] i32  next() noexcept;
    void               end() noexcept;
    [[nodiscard]] bool is_begin() const noexcept { return is_begin_; }

    [[nodiscard]] const DirectoryEntry& get() const noexcept { return current_; }
    [[nodiscard]] bool                  is_directory() const noexcept { return current_.is_directory; }

    // ---- Static helpers -----------------------------------------------
    [[nodiscard]] static i32 change_current_directory(const wchar_t* path = nullptr) noexcept;
    [[nodiscard]] static i32 make(const wchar_t* path) noexcept;
    [[nodiscard]] static i32 move(const wchar_t* from, const wchar_t* to,
                                  const wchar_t* skip_name = nullptr, bool do_make = true) noexcept;
    [[nodiscard]] static i32 copy(const wchar_t* from, const wchar_t* to,
                                  const wchar_t* skip_name = nullptr, bool do_make = true) noexcept;
    [[nodiscard]] static i32 remove(const wchar_t* from, const wchar_t* skip_name = nullptr) noexcept;
    [[nodiscard]] static i32 rename(const wchar_t* old_name, const wchar_t* new_name) noexcept;

private:
    [[nodiscard]] i32 fill_current_() noexcept;
    [[nodiscard]] bool matches_pattern_(const std::wstring& name) const noexcept;

    std::filesystem::directory_iterator iter_;
    std::filesystem::directory_iterator end_;
    std::wstring                        pattern_;          // filename glob (empty = match all)
    DirectoryEntry                      current_;
    bool                                is_begin_;
};

}  // namespace cardinal::core
