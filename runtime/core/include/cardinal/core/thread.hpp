#pragma once

// =============================================================================
// Cardinal core — threading vocabulary (<thread>/<mutex>/<shared_mutex>/
// <condition_variable>) — single sanctioned include site per the FOUNDATION
// RULE. The engine's job system is the preferred concurrency surface; these
// primitives back it and the few places that need raw OS threads.
// =============================================================================

#include <condition_variable>
#include <mutex>
#include <shared_mutex>
#include <thread>

namespace cardinal {

using std::thread;
using std::jthread;
using std::stop_token;
using std::stop_source;
namespace this_thread = std::this_thread;

using std::mutex;
using std::recursive_mutex;
using std::timed_mutex;
using std::shared_mutex;
using std::lock_guard;
using std::unique_lock;
using std::shared_lock;
using std::scoped_lock;
using std::defer_lock;
using std::adopt_lock;
using std::try_to_lock;
using std::call_once;
using std::once_flag;

using std::condition_variable;
using std::condition_variable_any;
using std::cv_status;

}  // namespace cardinal
