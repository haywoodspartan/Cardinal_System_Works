#include <cardinal/core/sync/worker_thread.hpp>
#include <cardinal/core/sync/affinity.hpp>

#include <chrono>

#if CARDINAL_PLATFORM_WINDOWS
#include <Windows.h>
#include <processthreadsapi.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

namespace cardinal::core {

// ---------------------------------------------------------------------------
// ThreadManager singleton.
// ---------------------------------------------------------------------------

ThreadManager& ThreadManager::instance() noexcept {
    static ThreadManager mgr;
    return mgr;
}

ThreadManager::ThreadManager() noexcept : is_opened_(false) {}
ThreadManager::~ThreadManager() noexcept {}

i32  ThreadManager::open() noexcept { is_opened_ = true; return 0; }
void ThreadManager::close() noexcept { is_opened_ = false; object_list_.clear(); }
bool ThreadManager::is_opened() const noexcept { return is_opened_; }

void ThreadManager::register_thread_(Thread* t) noexcept {
    ExclusiveLockGuard<ThreadLock> g(&lock_);
    object_list_.insert(t);
}
void ThreadManager::unregister_thread_(Thread* t) noexcept {
    ExclusiveLockGuard<ThreadLock> g(&lock_);
    object_list_.erase(t);
}

cardinal::vector<Thread*> ThreadManager::get_list(TrGroupId group_id_filter) const noexcept {
    SharedLockGuard<ThreadLock> g(&lock_);
    cardinal::vector<Thread*> out;
    out.reserve(object_list_.size());
    for (Thread* t : object_list_) {
        if (group_id_filter == -1 || t->group_id() == group_id_filter) {
            out.push_back(t);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Thread implementation.
// ---------------------------------------------------------------------------

Thread::Thread(const wchar_t* name, TrGroupId group_id, bool do_register_to_manager) noexcept
    : name_(name ? name : L"unnamed")
    , group_id_(group_id)
    , do_register_to_manager_(do_register_to_manager)
    , thread_id_(0)
    , exit_code_(0)
    , started_(false)
{
    if (do_register_to_manager_) {
        ThreadManager::instance().register_thread_(this);
    }
}

Thread::~Thread() noexcept {
    if (thread_.joinable()) {
        thread_.request_stop();
        thread_.join();
    }
    if (do_register_to_manager_) {
        ThreadManager::instance().unregister_thread_(this);
    }
}

i32  Thread::open(void* /*parameter*/) noexcept { return 0; }
void Thread::close() noexcept                   {}

bool Thread::is_started() const noexcept { return started_.load(); }
bool Thread::is_alive()   const noexcept { return thread_.joinable() && started_.load(); }

i32 Thread::wait(u32 wait_ms) const noexcept {
    // std::jthread doesn't expose a timed-join; we model it with a poll.
    if (!thread_.joinable()) return 0;
#if CARDINAL_PLATFORM_WINDOWS
    HANDLE h = const_cast<Thread*>(this)->thread_handle() != nullptr
                   ? static_cast<HANDLE>(const_cast<Thread*>(this)->thread_handle())
                   : nullptr;
    if (h != nullptr) {
        const DWORD r = ::WaitForSingleObject(h, wait_ms);
        return (r == WAIT_OBJECT_0) ? 0 : 258;   // WAIT_TIMEOUT
    }
#endif
    // Fallback: spin-poll for at most wait_ms milliseconds.
    using clock = std::chrono::steady_clock;
    const auto deadline = clock::now() + std::chrono::milliseconds(wait_ms);
    while (clock::now() < deadline) {
        if (!started_.load()) return 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return started_.load() ? 258 : 0;
}

i32 Thread::start(u32 /*stack_size*/) noexcept {
    // std::jthread does not expose a stack-size knob portably; on Win we
    // could use _beginthreadex, but we accept the default. The argument is
    // kept for source compat.
    if (thread_.joinable()) return EALREADY;
    started_.store(true);
    thread_ = cardinal::jthread([this](cardinal::stop_token /*tok*/){
        run_stub_();
    });
    return 0;
}

void Thread::run_stub_() noexcept {
#if CARDINAL_PLATFORM_WINDOWS
    thread_id_.store(static_cast<ThreadId>(::GetCurrentThreadId()));
#else
    thread_id_.store(static_cast<ThreadId>(reinterpret_cast<std::uintptr_t>(pthread_self())));
#endif
    set_thread_name(name_.c_str());
    const i32 rc = run();
    exit_code_.store(rc);
    started_.store(false);
}

i32 Thread::stop() noexcept {
    if (!thread_.joinable()) return ESRCH;
    thread_.request_stop();
    return 0;
}

bool Thread::stop_requested() const noexcept {
    return thread_.get_stop_token().stop_requested();
}

void* Thread::thread_handle() const noexcept {
    // std::jthread::native_handle() is non-const; cast away const on our
    // own member to call it. The native handle is process-stable, so the
    // const interface remains semantically correct.
    auto& mutable_thread = const_cast<cardinal::jthread&>(thread_);
    if (!mutable_thread.joinable()) return nullptr;
    return reinterpret_cast<void*>(mutable_thread.native_handle());
}

i32 Thread::set_priority(i32 priority) const noexcept {
#if CARDINAL_PLATFORM_WINDOWS
    HANDLE h = static_cast<HANDLE>(const_cast<Thread*>(this)->thread_handle());
    if (h == nullptr) return EINVAL;
    return ::SetThreadPriority(h, priority) ? 0 : static_cast<i32>(::GetLastError());
#else
    sched_param sp{};
    sp.sched_priority = priority;
    return pthread_setschedparam(thread_.native_handle(), SCHED_OTHER, &sp);
#endif
}

i32 Thread::set_affinity(u64 mask) const noexcept {
#if CARDINAL_PLATFORM_WINDOWS
    HANDLE h = static_cast<HANDLE>(const_cast<Thread*>(this)->thread_handle());
    if (h == nullptr) return EINVAL;
    DWORD_PTR rc = ::SetThreadAffinityMask(h, static_cast<DWORD_PTR>(mask));
    return rc != 0 ? 0 : static_cast<i32>(::GetLastError());
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    for (u32 i = 0; i < 64; ++i) if (mask & (u64{1} << i)) CPU_SET(i, &set);
    return pthread_setaffinity_np(thread_.native_handle(), sizeof(set), &set);
#endif
}

void Thread::set_thread_name(const wchar_t* name) noexcept {
    if (name == nullptr) return;
#if CARDINAL_PLATFORM_WINDOWS
    // SetThreadDescription is the modern way (Win10 1607+).
    ::SetThreadDescription(::GetCurrentThread(), name);
#else
    // POSIX pthread_setname_np takes narrow + 16-char limit.
    char narrow[16] = {};
    for (u32 i = 0; i < 15u && name[i] != L'\0'; ++i) {
        narrow[i] = static_cast<char>(name[i] & 0x7Fu);
    }
    pthread_setname_np(pthread_self(), narrow);
#endif
}

TrGroupId Thread::current_group_id() noexcept {
    // No TLS lookup yet — would require a thread_local registry indexed by
    // OS thread id. Returns 0 for now; the "unknown group" default.
    return 0;
}

}  // namespace cardinal::core
