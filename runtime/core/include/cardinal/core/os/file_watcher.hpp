#pragma once

// =============================================================================
// Cardinal — directory file watcher.
//
// Event-driven replacement for per-frame "scan directory mtimes" polling.
// On Windows uses ReadDirectoryChangesW with a worker thread; on Linux
// will use inotify when WSI lands. The API is platform-agnostic.
//
// Usage:
//
//     auto watch = cardinal::core::FileWatcher::create(
//         "shaders/",
//         /*recursive*/ true,
//         [](const FileEvent& e) {
//             cardinal::log::infof("hotreload",
//                 "%s %s", file_event_name(e.kind), e.path.c_str());
//         });
//
// Callbacks fire on the watcher's worker thread. Marshal to the main
// thread yourself if needed (push into a thread-safe queue).
// =============================================================================

#include <cardinal/core/types.hpp>

#include <functional>
#include <memory>
#include <string>

namespace cardinal::core {

enum class FileEventKind : u8 {
    Added    = 0,
    Modified = 1,
    Removed  = 2,
    Renamed  = 3,
};

const char* file_event_name(FileEventKind k) noexcept;

struct FileEvent {
    FileEventKind kind;
    // Absolute (or relative-to-watched-root) path of the affected file.
    std::string   path;
    // Set only when kind == Renamed — the previous name.
    std::string   old_path;
};

using FileEventCallback = std::function<void(const FileEvent&)>;

class FileWatcher {
public:
    // root_path: directory to watch (must exist and be a directory).
    // recursive: include subdirectories.
    // cb: invoked from the watcher's worker thread per event. Don't
    //     block — push the event into a queue for main-thread handling.
    // Returns nullptr on Linux today, or on Windows if the path is bad.
    static std::unique_ptr<FileWatcher> create(
        const std::string& root_path,
        bool               recursive,
        FileEventCallback  cb);

    virtual ~FileWatcher() = default;
    FileWatcher(const FileWatcher&)            = delete;
    FileWatcher& operator=(const FileWatcher&) = delete;

    // Coalesce-window in milliseconds. Editors like VS Code save by
    // writing N times in quick succession (truncate → write); within
    // this window only the LAST event for a given path is forwarded.
    // Default 50ms — small enough to feel instant, large enough to
    // dedupe multi-write saves.
    virtual void set_coalesce_ms(u32 ms) = 0;

protected:
    FileWatcher() = default;
};

}  // namespace cardinal::core
