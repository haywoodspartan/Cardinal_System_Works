#pragma once

// =============================================================================
// Cardinal core — Thread / ThreadManager — modern C++20 port of
// the worker thread surface.
//
// Design:
//   * Thread is a "base class with virtual Run()" — the pattern many
//     long-running engine subsystems are built on. Subclass overrides Run(), calls
//     Start(stackSize), Stop() to cooperatively cancel.
//   * Under the hood we own a cardinal::jthread (std::jthread): cooperative
//     cancellation via stop_token, automatic join on destruction.
//   * SetAffinity / SetPriority forward to the existing cardinal::affinity
//     pin_current_thread + Win32 SetThreadPriority. Linux gets the pthread
//     equivalent (pthread_setaffinity_np / pthread_setschedparam).
//   * SetThreadName uses SetThreadDescription on Win10+ (visible to the
//     debugger + ETW); falls back to RaiseException trick on older. On
//     Linux uses pthread_setname_np.
//   * ThreadManager singleton tracks registered threads — useful for
//     debugging "show me every Thread group=foo".
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/platform.hpp>
#include <cardinal/core/thread.hpp>      // cardinal::jthread, stop_token
#include <cardinal/core/atomic.hpp>      // cardinal::atomic
#include <cardinal/core/containers.hpp>  // cardinal::vector
#include <cardinal/core/sync/lock.hpp>     // ThreadLock

#include <string>      // std::wstring for the thread name
#include <unordered_set>

namespace cardinal::core {

using TrGroupId = i32;
using ThreadId  = u64;

// ---------------------------------------------------------------------------
// Thread — virtual Run() base class with start/stop lifecycle.
// ---------------------------------------------------------------------------
class Thread {
public:
    explicit Thread(const wchar_t* name, TrGroupId group_id = 0, bool do_register_to_manager = true) noexcept;
    virtual ~Thread() noexcept;

    Thread(const Thread&)            = delete;
    Thread& operator=(const Thread&) = delete;

    // Lifecycle — open() initialises the object (Run() not started yet);
    // close() tears it down.
    [[nodiscard]] virtual i32  open(void* parameter = nullptr) noexcept;
    virtual void               close() noexcept;

    [[nodiscard]] bool is_started() const noexcept;
    [[nodiscard]] bool is_alive()   const noexcept;

    // Block the caller for up to wait_ms milliseconds for the thread to end.
    // Returns 0 if the thread ended; WAIT_TIMEOUT (258) on timeout.
    [[nodiscard]] i32  wait(u32 wait_ms = 0xFFFFFFFFu) const noexcept;

    // Spawn the OS thread. Subclasses override Run() — that's the worker fn.
    [[nodiscard]] i32  start(u32 stack_size) noexcept;
    // Request cooperative shutdown. Default implementation sets stop_token;
    // Run() should poll stop_requested() periodically.
    [[nodiscard]] virtual i32  stop() noexcept;

    // Priority / affinity.
    [[nodiscard]] i32  set_priority(i32 priority = 0) const noexcept;
    [[nodiscard]] i32  set_affinity(u64 mask)         const noexcept;

    // Sets the calling thread's name (debugger-visible).
    static void  set_thread_name(const wchar_t* name) noexcept;

    [[nodiscard]] void*          thread_handle() const noexcept;
    [[nodiscard]] ThreadId       thread_id()     const noexcept { return thread_id_; }
    [[nodiscard]] i32            exit_code()     const noexcept { return exit_code_; }
    [[nodiscard]] const wchar_t* name()          const noexcept { return name_.c_str(); }
    [[nodiscard]] TrGroupId      group_id()      const noexcept { return group_id_; }

    [[nodiscard]] static TrGroupId current_group_id() noexcept;

protected:
    // Override in subclass. Return value lands in exit_code_.
    [[nodiscard]] virtual i32 run() noexcept = 0;

    // Stop-token surface for cooperative cancellation inside run().
    [[nodiscard]] bool stop_requested() const noexcept;

private:
    void run_stub_() noexcept;

    std::wstring     name_;
    TrGroupId        group_id_;
    bool             do_register_to_manager_;
    cardinal::jthread thread_;
    cardinal::atomic<ThreadId> thread_id_;
    cardinal::atomic<i32>      exit_code_;
    cardinal::atomic<bool>     started_;
};

// ---------------------------------------------------------------------------
// ThreadManager — singleton registry of live Thread instances.
// ---------------------------------------------------------------------------
class ThreadManager {
public:
    [[nodiscard]] static ThreadManager& instance() noexcept;

    [[nodiscard]] i32  open() noexcept;
    void               close() noexcept;
    [[nodiscard]] bool is_opened() const noexcept;

    // Returns the subset of registered threads whose group_id matches the
    // filter (group_id_filter == -1 = all groups).
    [[nodiscard]] cardinal::vector<Thread*> get_list(TrGroupId group_id_filter = -1) const noexcept;

private:
    friend class Thread;
    ThreadManager() noexcept;
    ~ThreadManager() noexcept;
    void register_thread_(Thread* t) noexcept;
    void unregister_thread_(Thread* t) noexcept;

    mutable ThreadLock                    lock_;
    std::unordered_set<Thread*>           object_list_;
    bool                                  is_opened_;
};

}  // namespace cardinal::core
