// =============================================================================
// Cardinal — file watcher implementation.
//
// Windows: one worker thread per FileWatcher running an OVERLAPPED
// ReadDirectoryChangesW loop. Events come in batches (Win32 packs them
// into one buffer); we walk the batch, debounce inside the coalesce
// window, then dispatch.
//
// Linux: stub returning nullptr until WSI / inotify integration lands.
// =============================================================================
#include <cardinal/core/file_watcher.hpp>

#include <cardinal/core/log.hpp>
#include <cardinal/core/platform.hpp>

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if CARDINAL_PLATFORM_WINDOWS
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <Windows.h>
#endif

namespace cardinal::core {

const char* file_event_name(FileEventKind k) noexcept {
    switch (k) {
        case FileEventKind::Added:    return "added";
        case FileEventKind::Modified: return "modified";
        case FileEventKind::Removed:  return "removed";
        case FileEventKind::Renamed:  return "renamed";
    }
    return "?";
}

namespace {

#if CARDINAL_PLATFORM_WINDOWS

std::string wide_to_utf8(const wchar_t* w, DWORD wlen) {
    if (wlen == 0) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w, static_cast<int>(wlen),
                                nullptr, 0, nullptr, nullptr);
    if (n <= 0) return {};
    std::string out(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, static_cast<int>(wlen),
                        out.data(), n, nullptr, nullptr);
    return out;
}

class WindowsWatcher final : public FileWatcher {
public:
    bool initialize(const std::string& root, bool recursive, FileEventCallback cb) {
        root_      = root;
        recursive_ = recursive;
        cb_        = std::move(cb);

        dir_ = CreateFileA(root.c_str(),
                           FILE_LIST_DIRECTORY,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, OPEN_EXISTING,
                           FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED,
                           nullptr);
        if (dir_ == INVALID_HANDLE_VALUE) {
            cardinal::log::warnf("file_watcher",
                "CreateFile(%s) failed (err=%lu)", root.c_str(),
                GetLastError());
            return false;
        }

        stop_event_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (stop_event_ == nullptr) {
            CloseHandle(dir_); dir_ = INVALID_HANDLE_VALUE;
            return false;
        }
        overlapped_ = {};
        overlapped_.hEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (overlapped_.hEvent == nullptr) {
            CloseHandle(stop_event_); stop_event_ = nullptr;
            CloseHandle(dir_); dir_ = INVALID_HANDLE_VALUE;
            return false;
        }

        thread_ = std::thread([this] { worker_(); });
        cardinal::log::infof("file_watcher",
            "watching %s (recursive=%s)", root.c_str(),
            recursive ? "yes" : "no");
        return true;
    }

    ~WindowsWatcher() override {
        if (stop_event_ != nullptr) SetEvent(stop_event_);
        if (thread_.joinable()) thread_.join();
        if (overlapped_.hEvent) CloseHandle(overlapped_.hEvent);
        if (stop_event_)        CloseHandle(stop_event_);
        if (dir_ != INVALID_HANDLE_VALUE) CloseHandle(dir_);
    }

    void set_coalesce_ms(u32 ms) override {
        coalesce_ms_.store(ms, std::memory_order_relaxed);
    }

private:
    void worker_() {
        constexpr DWORD kBufBytes = 32u * 1024u;
        std::vector<u8> buf(kBufBytes);

        constexpr DWORD kFilter =
            FILE_NOTIFY_CHANGE_FILE_NAME  |
            FILE_NOTIFY_CHANGE_DIR_NAME   |
            FILE_NOTIFY_CHANGE_SIZE       |
            FILE_NOTIFY_CHANGE_LAST_WRITE |
            FILE_NOTIFY_CHANGE_CREATION;

        std::string pending_rename_old;   // tracks OLD_NAME → NEW_NAME pairs

        while (true) {
            DWORD bytes_returned = 0;
            ResetEvent(overlapped_.hEvent);
            if (!ReadDirectoryChangesW(
                    dir_, buf.data(), kBufBytes, recursive_ ? TRUE : FALSE,
                    kFilter, &bytes_returned, &overlapped_, nullptr))
            {
                // Directory removed / handle invalidated. Bail.
                break;
            }

            HANDLE waits[2] = { overlapped_.hEvent, stop_event_ };
            DWORD wait = WaitForMultipleObjects(2, waits, FALSE, INFINITE);
            if (wait == WAIT_OBJECT_0 + 1) break;          // stop requested
            if (wait != WAIT_OBJECT_0)     break;          // error

            if (!GetOverlappedResult(dir_, &overlapped_, &bytes_returned, FALSE)) {
                continue;
            }
            if (bytes_returned == 0) {
                // Buffer overflow — too many events. Best we can do is log
                // and continue; downstream will lose granular events for
                // this batch but the next ReadDirectoryChangesW will
                // resume normal flow.
                cardinal::log::warnf("file_watcher",
                    "buffer overflow watching %s — events dropped this batch",
                    root_.c_str());
                continue;
            }

            // Walk the variable-sized FILE_NOTIFY_INFORMATION records.
            const u8* p   = buf.data();
            const u8* end = p + bytes_returned;
            while (p < end) {
                const auto* info = reinterpret_cast<const FILE_NOTIFY_INFORMATION*>(p);
                const std::string fname = wide_to_utf8(
                    info->FileName,
                    info->FileNameLength / sizeof(wchar_t));

                FileEventKind kind = FileEventKind::Modified;
                std::string old_path;
                switch (info->Action) {
                    case FILE_ACTION_ADDED:           kind = FileEventKind::Added;    break;
                    case FILE_ACTION_REMOVED:         kind = FileEventKind::Removed;  break;
                    case FILE_ACTION_MODIFIED:        kind = FileEventKind::Modified; break;
                    case FILE_ACTION_RENAMED_OLD_NAME:
                        // Buffer the old name; emit the Renamed event when
                        // we see the matching NEW_NAME entry (always next
                        // in the batch per Win32 semantics).
                        pending_rename_old = fname;
                        goto next;
                    case FILE_ACTION_RENAMED_NEW_NAME:
                        kind     = FileEventKind::Renamed;
                        old_path = pending_rename_old;
                        pending_rename_old.clear();
                        break;
                    default: goto next;
                }

                dispatch_with_coalesce_(kind, fname, old_path);
              next:
                if (info->NextEntryOffset == 0) break;
                p += info->NextEntryOffset;
            }
        }
    }

    void dispatch_with_coalesce_(FileEventKind kind, const std::string& path,
                                 const std::string& old_path) {
        const auto now    = std::chrono::steady_clock::now();
        const u32  coalesce = coalesce_ms_.load(std::memory_order_relaxed);

        // Coalesce by (path) — keep last-write timestamp; if within
        // window, suppress. For Removed/Renamed we always emit.
        if (kind == FileEventKind::Modified || kind == FileEventKind::Added) {
            std::lock_guard<std::mutex> lg(coalesce_mtx_);
            auto it = last_emit_.find(path);
            if (it != last_emit_.end()) {
                const auto elapsed_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - it->second).count();
                if (elapsed_ms < static_cast<i64>(coalesce)) return;
            }
            last_emit_[path] = now;
        }

        FileEvent e{};
        e.kind     = kind;
        e.path     = path;
        e.old_path = old_path;
        if (cb_) cb_(e);
    }

    HANDLE             dir_{INVALID_HANDLE_VALUE};
    HANDLE             stop_event_{nullptr};
    OVERLAPPED         overlapped_{};
    std::thread        thread_;
    std::string        root_;
    bool               recursive_{false};
    FileEventCallback  cb_;
    std::atomic<u32>   coalesce_ms_{50};
    std::mutex         coalesce_mtx_;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> last_emit_;
};
#endif  // CARDINAL_PLATFORM_WINDOWS

}  // namespace

std::unique_ptr<FileWatcher> FileWatcher::create(
    const std::string& root_path,
    bool               recursive,
    FileEventCallback  cb)
{
#if CARDINAL_PLATFORM_WINDOWS
    auto w = std::make_unique<WindowsWatcher>();
    if (!w->initialize(root_path, recursive, std::move(cb))) return nullptr;
    return w;
#else
    (void)root_path; (void)recursive; (void)cb;
    // TODO(linux-fs-watch): inotify-backed implementation when WSI lands.
    return nullptr;
#endif
}

}  // namespace cardinal::core
