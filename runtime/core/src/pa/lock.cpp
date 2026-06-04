// ProcessLock — cross-process named mutex. Win path uses CreateMutexW;
// non-Windows reduces to an in-process mutex pointer (named-IPC stub).

#include <cardinal/core/pa/lock.hpp>

#if CARDINAL_PLATFORM_WINDOWS
#include <Windows.h>
#endif

namespace cardinal::core {

#if CARDINAL_PLATFORM_WINDOWS

ProcessLock::ProcessLock(const wchar_t* name, bool do_create) noexcept
    : handle_(nullptr), name_(name), do_create_(do_create) {}

ProcessLock::~ProcessLock() noexcept { close(); }

i32 ProcessLock::open() noexcept {
    if (handle_ != nullptr) return 0;
    if (do_create_) {
        handle_ = ::CreateMutexW(nullptr, FALSE, name_);
        if (handle_ == nullptr) return static_cast<i32>(::GetLastError());
        // ERROR_ALREADY_EXISTS (183) is informational, not a failure
        return static_cast<i32>(::GetLastError());
    }
    handle_ = ::OpenMutexW(SYNCHRONIZE, FALSE, name_);
    if (handle_ == nullptr) return static_cast<i32>(::GetLastError());
    return 0;
}

void ProcessLock::close() noexcept {
    if (handle_ != nullptr) {
        ::CloseHandle(static_cast<HANDLE>(handle_));
        handle_ = nullptr;
    }
}

bool ProcessLock::is_opened() const noexcept { return handle_ != nullptr; }

void ProcessLock::lock_shared() const noexcept {
    if (handle_ != nullptr) ::WaitForSingleObject(static_cast<HANDLE>(handle_), INFINITE);
}
void ProcessLock::lock_exclusive() const noexcept { lock_shared(); }

void ProcessLock::unlock_shared() const noexcept {
    if (handle_ != nullptr) ::ReleaseMutex(static_cast<HANDLE>(handle_));
}
void ProcessLock::unlock_exclusive() const noexcept { unlock_shared(); }

bool ProcessLock::try_lock_shared() const noexcept {
    if (handle_ == nullptr) return false;
    return ::WaitForSingleObject(static_cast<HANDLE>(handle_), 0) == WAIT_OBJECT_0;
}
bool ProcessLock::try_lock_exclusive() const noexcept { return try_lock_shared(); }

#else  // CARDINAL_PLATFORM_WINDOWS

// Non-Windows stub. Real Linux impl would use sem_open(O_CREAT|O_EXCL).
ProcessLock::ProcessLock(const wchar_t* name, bool do_create) noexcept
    : handle_(nullptr), name_(name), do_create_(do_create) {}
ProcessLock::~ProcessLock() noexcept { close(); }
i32  ProcessLock::open()  noexcept { handle_ = reinterpret_cast<void*>(uintptr_t{1}); return 0; }
void ProcessLock::close() noexcept { handle_ = nullptr; }
bool ProcessLock::is_opened() const noexcept { return handle_ != nullptr; }
void ProcessLock::lock_shared()    const noexcept {}
void ProcessLock::lock_exclusive() const noexcept {}
void ProcessLock::unlock_shared()    const noexcept {}
void ProcessLock::unlock_exclusive() const noexcept {}
bool ProcessLock::try_lock_shared()    const noexcept { return true; }
bool ProcessLock::try_lock_exclusive() const noexcept { return true; }

#endif  // CARDINAL_PLATFORM_WINDOWS

}  // namespace cardinal::core
