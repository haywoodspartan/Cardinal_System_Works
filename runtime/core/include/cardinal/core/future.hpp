#pragma once

// =============================================================================
// Cardinal core — <future> vocabulary (single sanctioned include site per
// the FOUNDATION RULE). The job system is the preferred async surface;
// std::future/promise back the few APIs that hand out a waitable result.
// =============================================================================

#include <future>

namespace cardinal {

template <class T> using future        = std::future<T>;
template <class T> using shared_future = std::shared_future<T>;
template <class T> using promise       = std::promise<T>;
template <class T> using packaged_task = std::packaged_task<T>;
// NOTE: bare `async` is intentionally NOT re-exported — `cardinal::async`
// is an engine module namespace (core/async.hpp). Same collision class as
// cardinal::log / cardinal::partition. Use std::async only inside
// cardinal::core, or add a non-colliding alias if ever needed.
using std::launch;
using std::future_status;
using std::future_error;

}  // namespace cardinal
