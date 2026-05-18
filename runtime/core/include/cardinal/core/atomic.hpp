#pragma once

// =============================================================================
// Cardinal core — <atomic> vocabulary (single sanctioned include site per
// the FOUNDATION RULE). Keeps the lock-free policy / memory-order defaults
// restructurable from one place.
// =============================================================================

#include <atomic>

namespace cardinal {

template <class T> using atomic = std::atomic<T>;
using std::atomic_flag;
using std::memory_order;
using std::memory_order_relaxed;
using std::memory_order_consume;
using std::memory_order_acquire;
using std::memory_order_release;
using std::memory_order_acq_rel;
using std::memory_order_seq_cst;
using std::atomic_thread_fence;
using std::atomic_signal_fence;
using std::atomic_init;

using atomic_bool = std::atomic<bool>;
using atomic_i32  = std::atomic<std::int32_t>;
using atomic_u32  = std::atomic<std::uint32_t>;
using atomic_i64  = std::atomic<std::int64_t>;
using atomic_u64  = std::atomic<std::uint64_t>;
using atomic_usize = std::atomic<std::size_t>;

}  // namespace cardinal
