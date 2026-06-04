#pragma once

// =============================================================================
// Cardinal core — pa::File / pa::SyncReadFile / pa::SyncWriteFile — modern
// C++20 port of Pearl Abyss PaFile.h.
//
// Surface keeps the open/close/Read/Write/Seek vocabulary; under the hood
// uses std::FILE* (so it works the same on Win + Linux + macOS without
// pulling Windows.h into the public header) plus optional UTF-16 BOM
// writing for the legacy logging path.
//
// Static helpers: GetSize / GetTime / GetVersion (Win-only Version Resource
// scan; non-Windows: returns ENOSYS-equivalent).
//
// NOTE: pa::AsyncWriteFile (PA's overlapped-IO write path) is intentionally
// deferred — Cardinal already has cardinal::core::io for priority async I/O
// and there's no Pa-shaped consumer yet. Add when a real call site needs it.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/platform.hpp>

#include <cstdio>      // FILE*, fopen, fseek, fread, fwrite
#include <cstring>     // memcpy

namespace cardinal::core::pa {

// Pa-style seek whence values — mirror FILE_BEGIN / FILE_CURRENT / FILE_END
// so PA call sites that pass `FILE_BEGIN` keep working.
enum class SeekFrom : u32 {
    Begin   = 0,
    Current = 1,
    End     = 2,
};

// ---------------------------------------------------------------------------
// File — common base. Owns a FILE* handle, exposes is_opened + handle().
// ---------------------------------------------------------------------------
class File {
public:
    File() noexcept;
    ~File() noexcept;
    File(const File&)            = delete;
    File& operator=(const File&) = delete;

    [[nodiscard]] bool        is_opened() const noexcept;
    [[nodiscard]] void*       handle()    const noexcept;   // FILE* (opaque)

    // ---- Static metadata helpers ----------------------------------------
    [[nodiscard]] static i32  get_size(const wchar_t* path, u64& out_bytes) noexcept;
    [[nodiscard]] static i32  get_my_size(u64& out_bytes) noexcept;

#if CARDINAL_PLATFORM_WINDOWS
    // Returns FILETIME triple; on non-Windows these come back zero-filled.
    [[nodiscard]] static i32  get_time(const wchar_t* path,
                                       u64* out_create_ft, u64* out_access_ft, u64* out_write_ft) noexcept;
    [[nodiscard]] static i32  get_my_time(u64* out_create_ft, u64* out_access_ft, u64* out_write_ft) noexcept;

    // Win32 Version Resource scan — returns 0 on success, ERROR_RESOURCE_TYPE_NOT_FOUND (1813)
    // or ERROR_NOT_FOUND (1168) when the binary has no Version Resource.
    [[nodiscard]] static i32  get_version(const wchar_t* path,
                                          u32& v1, u32& v2, u32& v3, u32& v4) noexcept;
    [[nodiscard]] static i32  get_my_version(u32& v1, u32& v2, u32& v3, u32& v4) noexcept;
#endif

protected:
    [[nodiscard]] i32  open_impl(const wchar_t* path, bool read, bool append,
                                 bool writable, bool write_utf16_bom) noexcept;
    void               close_impl() noexcept;
    [[nodiscard]] i32  write_impl(const void* data, u32 size) noexcept;
    [[nodiscard]] i32  flush_impl() noexcept;
    [[nodiscard]] i32  seek_impl(i32 offset, SeekFrom whence) const noexcept;

    void* handle_;   // FILE* (kept void* so we don't leak <cstdio> through every consumer)
};

// ---------------------------------------------------------------------------
// SyncReadFile — synchronous binary reader.
// ---------------------------------------------------------------------------
class SyncReadFile : public File {
public:
    [[nodiscard]] i32 open(const wchar_t* path, bool writable = true) noexcept;
    void              close() noexcept;
    [[nodiscard]] i32 read(void* buffer, u32& inout_size) const noexcept;
    [[nodiscard]] i32 seek(i32 offset, SeekFrom whence) const noexcept;
};

// ---------------------------------------------------------------------------
// SyncWriteFile — synchronous binary / text writer with optional UTF-16 BOM.
// ---------------------------------------------------------------------------
class SyncWriteFile : public File {
public:
    [[nodiscard]] i32 open(const wchar_t* path, bool append = false, bool write_utf16_bom = false) noexcept;
    void              close() noexcept;
    [[nodiscard]] i32 write(const void* data, u32 size) noexcept;
    [[nodiscard]] i32 write(const char* utf8) noexcept;
    [[nodiscard]] i32 write(const wchar_t* utf16) noexcept;   // raw UTF-16 LE bytes
    [[nodiscard]] i32 flush() noexcept;
    [[nodiscard]] i32 seek(i32 offset, SeekFrom whence) const noexcept;
};

}  // namespace cardinal::core::pa
