#include <cardinal/core/os/seh.hpp>

#include <cstring>

#if CARDINAL_PLATFORM_WINDOWS
#include <Windows.h>
#include <DbgHelp.h>     // MiniDumpWriteDump
// dbghelp.lib is loaded lazily via LoadLibrary in the impl — no link
// dependency here, matches the existing cardinal::core::crash strategy.
#endif

namespace cardinal::core {

// ---------------------------------------------------------------------------
// SehCallback default impls — no-op.
// ---------------------------------------------------------------------------
void SehCallback::on_logging_start(const char* /*message*/) noexcept {}
void SehCallback::on_logging_end() noexcept {}
void SehCallback::on_minidump_wrote(const wchar_t* /*path*/, i32 /*err*/) noexcept {}

// ---------------------------------------------------------------------------
// SehManager::Impl — Windows-only state.
// ---------------------------------------------------------------------------
struct SehManager::Impl {
    bool          is_set;
    bool          log_callstack;
    char          user_message[256 + 1];
    wchar_t       dump_path[260 + 1];
    wchar_t       dump_path_last[260 + 1];
    wchar_t       extern_dump_path[260 + 1];
    SehCallback*  callback;

    // User context fields — written into every minidump's user-stream.
    wchar_t       user_name[64];
    i64           user_no;
    wchar_t       tr_description[128];
    wchar_t       character_name[64];
    i64           character_no;

#if CARDINAL_PLATFORM_WINDOWS
    LPTOP_LEVEL_EXCEPTION_FILTER previous_filter;
#endif

    Impl() noexcept
        : is_set(false), log_callstack(false), callback(nullptr)
        , user_no(0), character_no(0)
#if CARDINAL_PLATFORM_WINDOWS
        , previous_filter(nullptr)
#endif
    {
        user_message[0]      = '\0';
        dump_path[0]         = L'\0';
        dump_path_last[0]    = L'\0';
        extern_dump_path[0]  = L'\0';
        user_name[0]         = L'\0';
        tr_description[0]    = L'\0';
        character_name[0]    = L'\0';
    }
};

// Singleton storage. Heap-allocated to dodge static-init-order races with
// other singletons that use SehManager during construction.
SehManager& SehManager::instance() noexcept {
    static SehManager mgr;
    return mgr;
}

SehManager::SehManager() noexcept : impl_(new Impl) {}
SehManager::~SehManager() noexcept { dont_catch_exception(); delete impl_; }

bool SehManager::is_set() const noexcept { return impl_->is_set; }

void SehManager::set_user_information(const wchar_t* user_name, i64 user_no,
                                      const wchar_t* tr_description,
                                      const wchar_t* character_name, i64 character_no) noexcept {
    auto copy_w = [](wchar_t* dst, usize cap, const wchar_t* src) {
        if (src == nullptr) { dst[0] = L'\0'; return; }
        usize i = 0;
        while (src[i] != L'\0' && i + 1 < cap) { dst[i] = src[i]; ++i; }
        dst[i] = L'\0';
    };
    copy_w(impl_->user_name,       sizeof(impl_->user_name) / sizeof(wchar_t),       user_name);
    copy_w(impl_->tr_description,  sizeof(impl_->tr_description) / sizeof(wchar_t),  tr_description);
    copy_w(impl_->character_name,  sizeof(impl_->character_name) / sizeof(wchar_t),  character_name);
    impl_->user_no      = user_no;
    impl_->character_no = character_no;
}

void SehManager::set_dump_file_name(const wchar_t* path) noexcept {
    if (path == nullptr) { impl_->extern_dump_path[0] = L'\0'; return; }
    usize i = 0;
    constexpr usize kCap = sizeof(impl_->extern_dump_path) / sizeof(wchar_t);
    while (path[i] != L'\0' && i + 1 < kCap) { impl_->extern_dump_path[i] = path[i]; ++i; }
    impl_->extern_dump_path[i] = L'\0';
}

const wchar_t* SehManager::dump_file_name() const noexcept {
    return impl_->dump_path_last;
}

#if CARDINAL_PLATFORM_WINDOWS

namespace {

// SEH top-level filter. Writes a minidump if registered, then chains.
LONG WINAPI seh_top_level_filter(EXCEPTION_POINTERS* info) noexcept {
    auto& mgr = SehManager::instance();
    (void)info;
    (void)mgr.dump_mini(/*dump_exception_information=*/true);
    return EXCEPTION_CONTINUE_SEARCH;
}

// Build a timestamped dump path next to the configured prefix.
void build_timestamped_path(const wchar_t* prefix, wchar_t* out, usize out_cap) noexcept {
    SYSTEMTIME st{};
    ::GetLocalTime(&st);
    ::swprintf_s(out, out_cap, L"%s%04u%02u%02u%02u%02u%02u.dmp",
                 (prefix && prefix[0] != L'\0') ? prefix : L"dump_",
                 st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
}

}  // namespace

i32 SehManager::set_handler(const wchar_t* dump_path,
                            const char*    user_message,
                            bool           log_callstack,
                            SehCallback*   callback) noexcept {
    if (impl_->is_set) return ERROR_ALREADY_EXISTS;

    if (dump_path != nullptr) {
        usize i = 0;
        constexpr usize kCap = sizeof(impl_->dump_path) / sizeof(wchar_t);
        while (dump_path[i] != L'\0' && i + 1 < kCap) { impl_->dump_path[i] = dump_path[i]; ++i; }
        impl_->dump_path[i] = L'\0';
    } else {
        impl_->dump_path[0] = L'\0';
    }
    if (user_message != nullptr) {
        usize i = 0;
        while (user_message[i] != '\0' && i + 1 < sizeof(impl_->user_message)) {
            impl_->user_message[i] = user_message[i]; ++i;
        }
        impl_->user_message[i] = '\0';
    } else {
        impl_->user_message[0] = '\0';
    }
    impl_->log_callstack    = log_callstack;
    impl_->callback         = callback;
    impl_->previous_filter  = ::SetUnhandledExceptionFilter(&seh_top_level_filter);
    impl_->is_set           = true;
    return 0;
}

void SehManager::dont_catch_exception() noexcept {
    if (!impl_->is_set) return;
    ::SetUnhandledExceptionFilter(impl_->previous_filter);
    impl_->previous_filter = nullptr;
    impl_->is_set          = false;
}

i32 SehManager::dump_mini(bool /*dump_exception_information*/) noexcept {
    // Determine output path.
    wchar_t path[MAX_PATH + 1] = {};
    if (impl_->extern_dump_path[0] != L'\0') {
        usize i = 0;
        while (impl_->extern_dump_path[i] != L'\0' && i + 1 < MAX_PATH) {
            path[i] = impl_->extern_dump_path[i]; ++i;
        }
        path[i] = L'\0';
    } else {
        build_timestamped_path(impl_->dump_path, path, MAX_PATH + 1);
    }

    HANDLE file = ::CreateFileW(path, GENERIC_WRITE, 0, nullptr,
                                CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        const i32 err = static_cast<i32>(::GetLastError());
        if (impl_->callback != nullptr) impl_->callback->on_minidump_wrote(path, err);
        return err;
    }

    // Lazy-load MiniDumpWriteDump from dbghelp.dll — keeps the link clean.
    using PFN_MDWD = BOOL (WINAPI*)(HANDLE, DWORD, HANDLE, MINIDUMP_TYPE,
                                    PMINIDUMP_EXCEPTION_INFORMATION,
                                    PMINIDUMP_USER_STREAM_INFORMATION,
                                    PMINIDUMP_CALLBACK_INFORMATION);
    HMODULE dbg = ::LoadLibraryW(L"dbghelp.dll");
    PFN_MDWD mini_dump = dbg ? reinterpret_cast<PFN_MDWD>(
        ::GetProcAddress(dbg, "MiniDumpWriteDump")) : nullptr;
    i32 result = 0;
    if (mini_dump != nullptr) {
        const BOOL ok = mini_dump(::GetCurrentProcess(), ::GetCurrentProcessId(), file,
                                  MiniDumpNormal, nullptr, nullptr, nullptr);
        if (!ok) result = static_cast<i32>(::GetLastError());
    } else {
        result = ERROR_PROC_NOT_FOUND;
    }
    if (dbg) ::FreeLibrary(dbg);
    ::CloseHandle(file);

    // Record for dump_file_name().
    {
        usize i = 0;
        constexpr usize kCap = sizeof(impl_->dump_path_last) / sizeof(wchar_t);
        while (path[i] != L'\0' && i + 1 < kCap) { impl_->dump_path_last[i] = path[i]; ++i; }
        impl_->dump_path_last[i] = L'\0';
    }
    if (impl_->callback != nullptr) impl_->callback->on_minidump_wrote(path, result);
    return result;
}

#else  // CARDINAL_PLATFORM_WINDOWS

i32  SehManager::set_handler(const wchar_t*, const char*, bool, SehCallback*) noexcept { return ENOSYS; }
void SehManager::dont_catch_exception() noexcept {}
i32  SehManager::dump_mini(bool) noexcept { return ENOSYS; }

#endif  // CARDINAL_PLATFORM_WINDOWS

}  // namespace cardinal::core
