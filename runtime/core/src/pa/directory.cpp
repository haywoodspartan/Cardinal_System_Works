#include <cardinal/core/pa/directory.hpp>

#include <filesystem>
#include <system_error>

namespace fs = std::filesystem;

namespace cardinal::core {

namespace {

// Tiny glob matcher: supports '*' and '?' (Win32 FindFirstFile semantics).
// Case-insensitive on Windows.
bool glob_match(const wchar_t* pat, const wchar_t* str) noexcept {
    while (*pat != L'\0') {
        if (*pat == L'*') {
            ++pat;
            if (*pat == L'\0') return true;
            while (*str != L'\0') {
                if (glob_match(pat, str)) return true;
                ++str;
            }
            return false;
        }
        if (*str == L'\0') return false;
        const wchar_t p = *pat;
        const wchar_t s = *str;
        const bool eq = (p == L'?') ||
                        (p == s) ||
#if CARDINAL_PLATFORM_WINDOWS
                        (p >= L'A' && p <= L'Z' && (p + 0x20) == s) ||
                        (p >= L'a' && p <= L'z' && (p - 0x20) == s);
#else
                        false;
#endif
        if (!eq) return false;
        ++pat; ++str;
    }
    return *str == L'\0';
}

// Split L"./logs/*.txt" into (L"./logs", L"*.txt").
void split_path_pattern(const wchar_t* in, fs::path& out_dir, std::wstring& out_pat) {
    fs::path p(in);
    const std::wstring leaf = p.filename().wstring();
    const bool has_glob = leaf.find(L'*') != std::wstring::npos
                       || leaf.find(L'?') != std::wstring::npos;
    if (has_glob) {
        out_dir = p.parent_path();
        if (out_dir.empty()) out_dir = L".";
        out_pat = leaf;
    } else if (fs::is_directory(p)) {
        out_dir = p;
        out_pat = L"";
    } else {
        out_dir = p.parent_path();
        if (out_dir.empty()) out_dir = L".";
        out_pat = leaf;
    }
}

}  // namespace

Directory::Directory() noexcept : is_begin_(false) {
    current_.size_bytes = 0;
    current_.is_directory = false;
}

i32 Directory::begin(const wchar_t* path) noexcept {
    is_begin_ = false;
    current_.file_name.clear();
    current_.size_bytes   = 0;
    current_.is_directory = false;

    fs::path dir;
    split_path_pattern(path, dir, pattern_);

    std::error_code ec;
    iter_ = fs::directory_iterator(dir, ec);
    end_  = fs::directory_iterator();
    if (ec) return ec.value();

    while (iter_ != end_) {
        const std::wstring leaf = iter_->path().filename().wstring();
        if (matches_pattern_(leaf)) {
            const i32 r = fill_current_();
            if (r == 0) { is_begin_ = true; return 0; }
        }
        iter_.increment(ec);
        if (ec) return ec.value();
    }
    return 2;  // ERROR_FILE_NOT_FOUND
}

i32 Directory::next() noexcept {
    if (iter_ == end_) return 18;  // ERROR_NO_MORE_FILES
    std::error_code ec;
    iter_.increment(ec);
    if (ec) return ec.value();
    while (iter_ != end_) {
        const std::wstring leaf = iter_->path().filename().wstring();
        if (matches_pattern_(leaf)) {
            return fill_current_();
        }
        iter_.increment(ec);
        if (ec) return ec.value();
    }
    return 18;
}

void Directory::end() noexcept {
    iter_ = fs::directory_iterator();
    is_begin_ = false;
}

i32 Directory::fill_current_() noexcept {
    if (iter_ == end_) return 18;
    const auto& p = iter_->path();
    current_.file_name = p.filename().wstring();
    std::error_code ec;
    current_.is_directory = fs::is_directory(p, ec);
    if (ec) return ec.value();
    current_.size_bytes = current_.is_directory ? 0u : static_cast<u64>(fs::file_size(p, ec));
    return 0;
}

bool Directory::matches_pattern_(const std::wstring& name) const noexcept {
    if (pattern_.empty()) return true;
    return glob_match(pattern_.c_str(), name.c_str());
}

// ---- Static helpers ------------------------------------------------------

i32 Directory::change_current_directory(const wchar_t* path) noexcept {
    std::error_code ec;
    if (path == nullptr) {
        // No-op — caller wanted "restore to launch directory". We don't
        // remember the launch directory; return 0 (no error).
        return 0;
    }
    fs::current_path(fs::path(path), ec);
    return ec.value();
}

i32 Directory::make(const wchar_t* path) noexcept {
    std::error_code ec;
    fs::create_directories(fs::path(path), ec);
    return ec.value();
}

i32 Directory::remove(const wchar_t* from, const wchar_t* /*skip_name*/) noexcept {
    std::error_code ec;
    fs::remove_all(fs::path(from), ec);
    return ec.value();
}

i32 Directory::rename(const wchar_t* old_name, const wchar_t* new_name) noexcept {
    std::error_code ec;
    fs::rename(fs::path(old_name), fs::path(new_name), ec);
    return ec.value();
}

namespace {

i32 move_or_copy(const wchar_t* from, const wchar_t* to,
                 const wchar_t* skip_name, bool do_make, bool do_move) noexcept {
    fs::path src_dir;
    std::wstring pat;
    split_path_pattern(from, src_dir, pat);

    std::error_code ec;
    if (do_make) {
        fs::create_directories(fs::path(to), ec);
        if (ec) return ec.value();
    }

    fs::directory_iterator end;
    auto it = fs::directory_iterator(src_dir, ec);
    if (ec) return ec.value();

    auto skip_match = [skip_name](const std::wstring& leaf) {
        if (skip_name == nullptr || skip_name[0] == L'\0') return false;
        return leaf.find(skip_name) != std::wstring::npos;
    };
    auto pat_match = [&pat](const std::wstring& leaf) {
        if (pat.empty()) return true;
        return glob_match(pat.c_str(), leaf.c_str());
    };

    for (; it != end; it.increment(ec)) {
        if (ec) return ec.value();
        const auto leaf = it->path().filename().wstring();
        if (!pat_match(leaf) || skip_match(leaf)) continue;
        const fs::path dst = fs::path(to) / leaf;
        if (do_move) {
            fs::rename(it->path(), dst, ec);
        } else {
            fs::copy(it->path(), dst,
                     fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
        }
        if (ec) return ec.value();
    }
    return 0;
}

}  // namespace

i32 Directory::move(const wchar_t* from, const wchar_t* to,
                    const wchar_t* skip_name, bool do_make) noexcept {
    return move_or_copy(from, to, skip_name, do_make, /*move=*/true);
}

i32 Directory::copy(const wchar_t* from, const wchar_t* to,
                    const wchar_t* skip_name, bool do_make) noexcept {
    return move_or_copy(from, to, skip_name, do_make, /*move=*/false);
}

}  // namespace cardinal::core
