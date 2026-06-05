#include <cardinal/core/os/file.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <system_error>

#if CARDINAL_PLATFORM_WINDOWS
#include <Windows.h>
#endif

namespace cardinal::core {

namespace {
inline FILE*       as_file(void* h)       noexcept { return static_cast<FILE*>(h); }
inline const FILE* as_file(const void* h) noexcept { return static_cast<const FILE*>(h); }

inline int to_whence(SeekFrom w) noexcept {
    switch (w) {
        case SeekFrom::Begin:   return SEEK_SET;
        case SeekFrom::Current: return SEEK_CUR;
        case SeekFrom::End:     return SEEK_END;
    }
    return SEEK_SET;
}
}  // namespace

File::File() noexcept : handle_(nullptr) {}
File::~File() noexcept { close_impl(); }

bool  File::is_opened() const noexcept { return handle_ != nullptr; }
void* File::handle()    const noexcept { return handle_; }

i32 File::open_impl(const wchar_t* path, bool read, bool append, bool writable, bool write_utf16_bom) noexcept {
    if (path == nullptr) return EINVAL;
    if (handle_ != nullptr) close_impl();

    // Build the fopen mode string.
    const char* mode = "rb";
    if (read) {
        mode = writable ? "rb+" : "rb";
    } else if (append) {
        mode = write_utf16_bom ? "ab+" : "ab";
    } else {
        mode = "wb";
    }

#if CARDINAL_PLATFORM_WINDOWS
    // Win uses wide-mode fopen; convert mode to wide.
    wchar_t wmode[8] = {};
    for (u32 i = 0; mode[i] != '\0' && i < 7u; ++i) wmode[i] = static_cast<wchar_t>(mode[i]);
    FILE* f = nullptr;
    if (::_wfopen_s(&f, path, wmode) != 0 || f == nullptr) {
        return static_cast<i32>(::GetLastError());
    }
    handle_ = f;
#else
    // Non-Windows: convert path to narrow (mbstowcs reverse via wcstombs).
    char narrow[4096] = {};
    std::wcstombs(narrow, path, sizeof(narrow) - 1);
    FILE* f = std::fopen(narrow, mode);
    if (f == nullptr) return errno;
    handle_ = f;
#endif

    if (!read && write_utf16_bom && !append) {
        // UTF-16 LE BOM: 0xFF 0xFE
        const unsigned char bom[2] = { 0xFF, 0xFE };
        std::fwrite(bom, 1, 2, as_file(handle_));
    }
    return 0;
}

void File::close_impl() noexcept {
    if (handle_ != nullptr) {
        std::fclose(as_file(handle_));
        handle_ = nullptr;
    }
}

i32 File::write_impl(const void* data, u32 size) noexcept {
    if (handle_ == nullptr) return EBADF;
    const std::size_t n = std::fwrite(data, 1, size, as_file(handle_));
    return (n == size) ? 0 : ferror(as_file(handle_));
}

i32 File::flush_impl() noexcept {
    if (handle_ == nullptr) return EBADF;
    return std::fflush(as_file(handle_));
}

i32 File::seek_impl(i32 offset, SeekFrom whence) const noexcept {
    if (handle_ == nullptr) return EBADF;
    return std::fseek(as_file(handle_), offset, to_whence(whence));
}

// ---- Static metadata -----------------------------------------------------

i32 File::get_size(const wchar_t* path, u64& out_bytes) noexcept {
    out_bytes = 0;
    std::error_code ec;
    const auto sz = std::filesystem::file_size(std::filesystem::path(path), ec);
    if (ec) return ec.value();
    out_bytes = static_cast<u64>(sz);
    return 0;
}

i32 File::get_my_size(u64& out_bytes) noexcept {
#if CARDINAL_PLATFORM_WINDOWS
    wchar_t path[MAX_PATH + 1] = {};
    if (::GetModuleFileNameW(nullptr, path, MAX_PATH) == 0) {
        return static_cast<i32>(::GetLastError());
    }
    return get_size(path, out_bytes);
#else
    out_bytes = 0;
    return ENOSYS;
#endif
}

#if CARDINAL_PLATFORM_WINDOWS
i32 File::get_time(const wchar_t* path, u64* c_ft, u64* a_ft, u64* w_ft) noexcept {
    WIN32_FILE_ATTRIBUTE_DATA fad{};
    if (!::GetFileAttributesExW(path, GetFileExInfoStandard, &fad)) {
        return static_cast<i32>(::GetLastError());
    }
    auto to_u64 = [](const FILETIME& ft) -> u64 {
        return (static_cast<u64>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
    };
    if (c_ft) *c_ft = to_u64(fad.ftCreationTime);
    if (a_ft) *a_ft = to_u64(fad.ftLastAccessTime);
    if (w_ft) *w_ft = to_u64(fad.ftLastWriteTime);
    return 0;
}

i32 File::get_my_time(u64* c_ft, u64* a_ft, u64* w_ft) noexcept {
    wchar_t path[MAX_PATH + 1] = {};
    if (::GetModuleFileNameW(nullptr, path, MAX_PATH) == 0) {
        return static_cast<i32>(::GetLastError());
    }
    return get_time(path, c_ft, a_ft, w_ft);
}

i32 File::get_version(const wchar_t* path, u32& v1, u32& v2, u32& v3, u32& v4) noexcept {
    v1 = v2 = v3 = v4 = 0;
    DWORD dummy = 0;
    const DWORD size = ::GetFileVersionInfoSizeW(path, &dummy);
    if (size == 0) return static_cast<i32>(::GetLastError());

    HLOCAL block = ::LocalAlloc(LMEM_FIXED, size);
    if (block == nullptr) return ERROR_OUTOFMEMORY;
    i32 result = 0;
    if (!::GetFileVersionInfoW(path, 0, size, block)) {
        result = static_cast<i32>(::GetLastError());
    } else {
        VS_FIXEDFILEINFO* fi = nullptr;
        UINT fi_len = 0;
        if (::VerQueryValueW(block, L"\\", reinterpret_cast<LPVOID*>(&fi), &fi_len) && fi != nullptr) {
            v1 = HIWORD(fi->dwFileVersionMS);
            v2 = LOWORD(fi->dwFileVersionMS);
            v3 = HIWORD(fi->dwFileVersionLS);
            v4 = LOWORD(fi->dwFileVersionLS);
        } else {
            result = static_cast<i32>(::GetLastError());
        }
    }
    ::LocalFree(block);
    return result;
}

i32 File::get_my_version(u32& v1, u32& v2, u32& v3, u32& v4) noexcept {
    wchar_t path[MAX_PATH + 1] = {};
    if (::GetModuleFileNameW(nullptr, path, MAX_PATH) == 0) {
        return static_cast<i32>(::GetLastError());
    }
    return get_version(path, v1, v2, v3, v4);
}
#endif  // CARDINAL_PLATFORM_WINDOWS

// ---- SyncReadFile --------------------------------------------------------

i32 SyncReadFile::open(const wchar_t* path, bool writable) noexcept {
    return open_impl(path, /*read=*/true, /*append=*/false, writable, /*bom=*/false);
}
void SyncReadFile::close() noexcept { close_impl(); }

i32 SyncReadFile::read(void* buffer, u32& inout_size) const noexcept {
    if (handle_ == nullptr) return EBADF;
    const std::size_t n = std::fread(buffer, 1, inout_size, as_file(handle_));
    inout_size = static_cast<u32>(n);
    if (n < inout_size && std::ferror(as_file(handle_))) {
        return std::ferror(as_file(handle_));
    }
    return 0;  // n == 0 with feof() set is EOF, still OK
}

i32 SyncReadFile::seek(i32 offset, SeekFrom whence) const noexcept { return seek_impl(offset, whence); }

// ---- SyncWriteFile -------------------------------------------------------

i32 SyncWriteFile::open(const wchar_t* path, bool append, bool write_utf16_bom) noexcept {
    return open_impl(path, /*read=*/false, append, /*writable=*/true, write_utf16_bom);
}
void SyncWriteFile::close() noexcept { close_impl(); }

i32 SyncWriteFile::write(const void* data, u32 size) noexcept { return write_impl(data, size); }

i32 SyncWriteFile::write(const char* utf8) noexcept {
    if (utf8 == nullptr) return EINVAL;
    return write_impl(utf8, static_cast<u32>(std::strlen(utf8)));
}

i32 SyncWriteFile::write(const wchar_t* utf16) noexcept {
    if (utf16 == nullptr) return EINVAL;
    u32 n = 0;
    while (utf16[n] != L'\0') ++n;
    return write_impl(utf16, static_cast<u32>(n * sizeof(wchar_t)));
}

i32 SyncWriteFile::flush() noexcept                          { return flush_impl(); }
i32 SyncWriteFile::seek(i32 offset, SeekFrom whence) const noexcept { return seek_impl(offset, whence); }

}  // namespace cardinal::core
