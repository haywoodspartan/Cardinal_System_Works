#pragma once

// =============================================================================
// Cardinal core — SyncQueue / NonDuplicableUnorderedSyncQueue /
// WaitableQueue / PriorityQueue / StaticCircularQueue — modern
// C++20 port of the queue surface.
//
// Surface 1:1 with the original templates (so callers port
// mechanically) but the synchronisation primitive is now any type that
// satisfies the cardinal::core::ThreadLock concept (shared/exclusive
// + try variants) — defaults to ThreadLock.
//
// Key design choices:
//   * Backing container is std::deque by default  — std::vector
//     can be plugged via the third template arg if random access matters.
//   * WaitableQueue uses cardinal::condition_variable_any + the same lock
//     concept (replaces the legacy CreateSemaphore + WaitForSingleObjectEx loop);
//     supports timed wait with chrono::milliseconds.
//   * StaticCircularQueue is the lock-free, fixed-capacity ring (used a lot
//     by frame-rate-sensitive logging paths); no allocations after construction.
//   * Lightweight return-code convention: u32 result; 0 = ok, 4306 = empty
//     (matches Win32 ERROR_EMPTY) — downstream code already understands this.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/containers.hpp>     // cardinal::vector / cardinal::deque
#include <cardinal/core/sync/lock.hpp>

#include <deque>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <chrono>

namespace cardinal::core {

// Result codes — match Win32 sym numbers the engine already uses.
inline constexpr i32 kQueueOk    = 0;
inline constexpr i32 kQueueEmpty = 4306;   // ERROR_EMPTY

// ---------------------------------------------------------------------------
// SyncQueue<T> — bounded FIFO with shared/exclusive locking.
// ---------------------------------------------------------------------------
template <class T, class TLock = ThreadLock, class TList = std::deque<T>>
class SyncQueue {
public:
    using value_type = typename TList::value_type;
    using size_type  = typename TList::size_type;

    SyncQueue() noexcept : is_opened_(false) {}
    ~SyncQueue() noexcept = default;
    SyncQueue(const SyncQueue&)            = delete;
    SyncQueue& operator=(const SyncQueue&) = delete;

    [[nodiscard]] i32  open()       noexcept { is_opened_ = true; return 0; }
    void               close()      noexcept { list_.clear(); is_opened_ = false; }
    [[nodiscard]] bool is_opened()  const noexcept { return is_opened_; }

    void lock_shared()    const noexcept { lock_.lock_shared(); }
    void lock_exclusive() const noexcept { lock_.lock_exclusive(); }
    void unlock_shared()    const noexcept { lock_.unlock_shared(); }
    void unlock_exclusive() const noexcept { lock_.unlock_exclusive(); }

    void push(const value_type& v) noexcept {
        ExclusiveLockGuard<TLock> g(&lock_);
        list_.push_back(v);
    }
    [[nodiscard]] i32 pop_front(value_type& out) noexcept {
        ExclusiveLockGuard<TLock> g(&lock_);
        if (list_.empty()) return kQueueEmpty;
        out = list_.front();
        list_.pop_front();
        return kQueueOk;
    }
    [[nodiscard]] i32 pop_back(value_type& out) noexcept {
        ExclusiveLockGuard<TLock> g(&lock_);
        if (list_.empty()) return kQueueEmpty;
        out = list_.back();
        list_.pop_back();
        return kQueueOk;
    }
    void erase(const value_type& v) noexcept {
        ExclusiveLockGuard<TLock> g(&lock_);
        auto it = std::find(list_.begin(), list_.end(), v);
        if (it != list_.end()) list_.erase(it);
    }
    [[nodiscard]] bool is_exist(const value_type& v) const noexcept {
        SharedLockGuard<TLock> g(&lock_);
        return std::find(list_.begin(), list_.end(), v) != list_.end();
    }
    [[nodiscard]] size_type size() const noexcept {
        SharedLockGuard<TLock> g(&lock_);
        return list_.size();
    }

private:
    mutable TLock lock_;
    TList         list_;
    bool          is_opened_;
};

// ---------------------------------------------------------------------------
// NonDuplicableUnorderedSyncQueue<T> — set-backed dedup queue. push/pop don't
// preserve order; the value type must be hashable.
// ---------------------------------------------------------------------------
template <class T, class TLock = ThreadLock, class TSet = std::unordered_set<T>>
class NonDuplicableUnorderedSyncQueue {
public:
    using value_type = typename TSet::value_type;
    using size_type  = typename TSet::size_type;

    NonDuplicableUnorderedSyncQueue() noexcept : is_opened_(false) {}
    ~NonDuplicableUnorderedSyncQueue() noexcept = default;
    NonDuplicableUnorderedSyncQueue(const NonDuplicableUnorderedSyncQueue&)            = delete;
    NonDuplicableUnorderedSyncQueue& operator=(const NonDuplicableUnorderedSyncQueue&) = delete;

    [[nodiscard]] i32  open()       noexcept { is_opened_ = true; return 0; }
    void               close()      noexcept { set_.clear(); is_opened_ = false; }
    [[nodiscard]] bool is_opened()  const noexcept { return is_opened_; }

    void push(const value_type& v) noexcept {
        ExclusiveLockGuard<TLock> g(&lock_);
        set_.insert(v);
    }
    [[nodiscard]] i32 pop_front(value_type& out) noexcept {
        ExclusiveLockGuard<TLock> g(&lock_);
        if (set_.empty()) return kQueueEmpty;
        auto it = set_.begin();
        out = *it;
        set_.erase(it);
        return kQueueOk;
    }
    void erase(const value_type& v) noexcept {
        ExclusiveLockGuard<TLock> g(&lock_);
        set_.erase(v);
    }
    [[nodiscard]] bool is_exist(const value_type& v) const noexcept {
        SharedLockGuard<TLock> g(&lock_);
        return set_.find(v) != set_.end();
    }
    [[nodiscard]] size_type size() const noexcept {
        SharedLockGuard<TLock> g(&lock_);
        return set_.size();
    }

private:
    mutable TLock lock_;
    TSet          set_;
    bool          is_opened_;
};

// ---------------------------------------------------------------------------
// WaitableQueue<T> — blocking FIFO. pop_front blocks up to N ms for an item.
// Backed by std::condition_variable_any (works with any BasicLockable) so it
// composes with shared_mutex without forcing the caller into unique_lock.
// ---------------------------------------------------------------------------
template <class T, class TList = std::deque<T>>
class WaitableQueue {
public:
    using value_type = typename TList::value_type;
    using size_type  = typename TList::size_type;

    WaitableQueue() noexcept : is_opened_(false) {}
    ~WaitableQueue() noexcept = default;
    WaitableQueue(const WaitableQueue&)            = delete;
    WaitableQueue& operator=(const WaitableQueue&) = delete;

    [[nodiscard]] i32 open()       noexcept { is_opened_ = true; return 0; }
    void              close()      noexcept {
        { std::lock_guard<std::mutex> g(mutex_); list_.clear(); is_opened_ = false; }
        cv_.notify_all();
    }
    [[nodiscard]] bool is_opened() const noexcept { return is_opened_; }

    void push(const value_type& v) noexcept {
        { std::lock_guard<std::mutex> g(mutex_); list_.push_back(v); }
        cv_.notify_one();
    }

    [[nodiscard]] i32 pop_front(u32 wait_ms, value_type& out) noexcept {
        std::unique_lock<std::mutex> g(mutex_);
        if (!cv_.wait_for(g, std::chrono::milliseconds(wait_ms),
                          [this] { return !list_.empty() || !is_opened_; })) {
            return kQueueEmpty;     // timeout
        }
        if (list_.empty()) return kQueueEmpty;
        out = list_.front();
        list_.pop_front();
        return kQueueOk;
    }

    [[nodiscard]] size_type size() const noexcept {
        std::lock_guard<std::mutex> g(mutex_);
        return list_.size();
    }
    [[nodiscard]] bool is_exist(const value_type& v) const noexcept {
        std::lock_guard<std::mutex> g(mutex_);
        return std::find(list_.begin(), list_.end(), v) != list_.end();
    }

private:
    mutable std::mutex          mutex_;
    std::condition_variable     cv_;
    TList                       list_;
    bool                        is_opened_;
};

// ---------------------------------------------------------------------------
// PriorityQueue<T> — std::priority_queue adapter exposing begin/end + isExist.
// Not thread-safe .
// ---------------------------------------------------------------------------
template <class T, class Container = std::vector<T>, class Compare = std::less<T>>
class PriorityQueue : public std::priority_queue<T, Container, Compare> {
    using base = std::priority_queue<T, Container, Compare>;
public:
    using value_type      = typename base::value_type;
    using iterator        = typename Container::iterator;
    using const_iterator  = typename Container::const_iterator;

    [[nodiscard]] iterator       begin()       noexcept { return this->c.begin(); }
    [[nodiscard]] iterator       end()         noexcept { return this->c.end(); }
    [[nodiscard]] const_iterator begin() const noexcept { return this->c.begin(); }
    [[nodiscard]] const_iterator end()   const noexcept { return this->c.end(); }

    void clear() noexcept { this->c.clear(); }
    [[nodiscard]] bool is_exist(const value_type& v) const noexcept {
        return std::find(this->c.begin(), this->c.end(), v) != this->c.end();
    }
};

// ---------------------------------------------------------------------------
// StaticCircularQueue<T, N> — fixed-capacity overwriting ring. Lock-free
// (single-threaded), N+1 slots so empty != full discriminates.
// ---------------------------------------------------------------------------
template <class T, u32 Capacity>
class StaticCircularQueue {
public:
    using value_type = T;
    using Functor    = std::function<void(value_type& v)>;

    StaticCircularQueue() noexcept : push_index_(0), pop_index_(0) {
        for (auto& v : list_) v = value_type{};
    }
    StaticCircularQueue(const value_type* buffer, u32 push_index, u32 pop_index) noexcept
        : push_index_(push_index), pop_index_(pop_index) {
        for (u32 i = 0; i < Capacity + 1u; ++i) list_[i] = buffer[i];
    }

    void clear() noexcept { push_index_ = pop_index_ = 0; }
    [[nodiscard]] bool empty() const noexcept { return push_index_ == pop_index_; }
    [[nodiscard]] bool full()  const noexcept { return ((push_index_ + 1u) % (Capacity + 1u)) == pop_index_; }

    [[nodiscard]] u32 begin() const noexcept { return pop_index_; }
    [[nodiscard]] u32 end()   const noexcept { return push_index_; }
    [[nodiscard]] u32 rbegin() const noexcept {
        u32 p = push_index_;
        if (p == 0) p = Capacity + 1u;
        return p - 1u;
    }

    [[nodiscard]] value_type        get(u32 idx) const noexcept { return list_[idx]; }
    [[nodiscard]] value_type&       get(u32 idx)       noexcept { return list_[idx]; }

    void push(const value_type& v) noexcept {
        list_[push_index_] = v;
        push_index_ = (push_index_ + 1u) % (Capacity + 1u);
        if (push_index_ == pop_index_) {
            pop_index_ = (pop_index_ + 1u) % (Capacity + 1u);   // overwrite oldest
        }
    }
    [[nodiscard]] value_type pop() noexcept {
        const value_type v = list_[pop_index_];
        pop_index_ = (pop_index_ + 1u) % (Capacity + 1u);
        return v;
    }
    [[nodiscard]] value_type front() noexcept { return list_[pop_index_]; }

    [[nodiscard]] u32 capacity() const noexcept { return Capacity + 1u; }

    void for_each(const Functor& f) noexcept {
        for (u32 i = begin(); i != end(); i = (i + 1u) % (Capacity + 1u)) f(get(i));
    }

    [[nodiscard]] const value_type* raw_buffer() const noexcept { return list_; }
    [[nodiscard]] u32               raw_push_index() const noexcept { return push_index_; }
    [[nodiscard]] u32               raw_pop_index()  const noexcept { return pop_index_; }

private:
    u32        push_index_;
    u32        pop_index_;
    value_type list_[Capacity + 1u];
};

}  // namespace cardinal::core
