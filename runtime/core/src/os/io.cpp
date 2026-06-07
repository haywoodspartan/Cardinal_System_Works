#include <cardinal/core/os/io.hpp>

#include <cardinal/core/sync/async.hpp>
#include <cardinal/core/os/hal.hpp>
#include <cardinal/core/diag/log.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <unordered_set>

namespace cardinal::io {

const char* priority_name(Priority p) noexcept {
    switch (p) {
        case Priority::Background: return "Background";
        case Priority::Low:        return "Low";
        case Priority::Normal:     return "Normal";
        case Priority::High:       return "High";
        case Priority::Critical:   return "Critical";
    }
    return "?";
}

namespace {

constexpr u32 kPriorityCount = 5;

struct Pending {
    RequestHandle handle{0};
    Request       req;
    bool          cancelled{false};
};

}  // namespace

struct Dispatcher::Impl {
    DispatcherDesc                                    desc{};
    std::mutex                                        mtx;
    std::deque<Pending>                               queues[kPriorityCount];
    u32                                               in_flight[kPriorityCount]{};
    std::atomic<u32>                                  in_flight_total{0};
    std::atomic<u32>                                  in_flight_total_pretty{0};
    std::atomic<u64>                                  bytes_in_flight{0};
    std::atomic<u64>                                  bytes_completed{0};
    std::atomic<u64>                                  requests_seen{0};
    std::atomic<u64>                                  requests_completed{0};
    std::atomic<u64>                                  requests_failed{0};
    RequestHandle                                     next_handle{1};
    std::atomic<f64>                                  last_tick_ms{0.0};
    // Completed requests waiting to fire their on_done on tick().
    struct Done {
        RequestHandle handle;
        Request       req;
        std::vector<u8> bytes;
        bool          ok;
    };
    std::vector<Done>                                 done_queue;
    // Cancelled handles.
    std::unordered_set<RequestHandle>                 cancelled_handles;
};

std::shared_ptr<Dispatcher> Dispatcher::create(const DispatcherDesc& desc) {
    auto d = std::shared_ptr<Dispatcher>(new Dispatcher());
    if (!d->initialize_(desc)) return nullptr;
    return d;
}

Dispatcher::~Dispatcher() { delete impl_; }

bool Dispatcher::initialize_(const DispatcherDesc& desc) {
    impl_ = new Impl{};
    impl_->desc = desc;
    cardinal::log::infof("io",
        "IO dispatcher online — caps: B=%u L=%u N=%u H=%u C=%u total=%u",
        desc.max_concurrent_background, desc.max_concurrent_low,
        desc.max_concurrent_normal, desc.max_concurrent_high,
        desc.max_concurrent_critical, desc.max_concurrent_total);
    return true;
}

RequestHandle Dispatcher::submit(Request req) {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    Pending p{};
    p.handle = impl_->next_handle++;
    p.req    = std::move(req);
    const auto pri_idx = static_cast<u32>(p.req.priority);
    impl_->queues[pri_idx].push_back(std::move(p));
    impl_->requests_seen.fetch_add(1, std::memory_order_relaxed);
    return impl_->queues[pri_idx].back().handle;
}

std::shared_future<std::vector<u8>>
Dispatcher::submit_and_future(Request req)
{
    auto promise = std::make_shared<std::promise<std::vector<u8>>>();
    auto future  = promise->get_future().share();
    auto orig_cb = std::move(req.on_done);
    req.on_done = [promise, orig_cb](const std::vector<u8>& bytes) {
        if (orig_cb) orig_cb(bytes);
        promise->set_value(bytes);
    };
    submit(std::move(req));
    return future;
}

u32 Dispatcher::cancel(RequestHandle h) {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    impl_->cancelled_handles.insert(h);
    u32 n = 0;
    for (auto& q : impl_->queues) {
        for (auto& p : q) {
            if (p.handle == h && !p.cancelled) { p.cancelled = true; ++n; }
        }
    }
    return n;
}

u32 Dispatcher::cancel_tag(u64 tag) {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    u32 n = 0;
    for (auto& q : impl_->queues) {
        for (auto& p : q) {
            if (!p.cancelled && p.req.tag == tag) {
                p.cancelled = true;
                impl_->cancelled_handles.insert(p.handle);
                ++n;
            }
        }
    }
    return n;
}

namespace {

void run_request(Dispatcher::Impl* impl, Pending p) {
    if (p.cancelled) {
        impl->in_flight_total.fetch_sub(1, std::memory_order_relaxed);
        impl->bytes_in_flight.fetch_sub(p.req.size, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lg(impl->mtx);
        --impl->in_flight[static_cast<u32>(p.req.priority)];
        return;
    }
    std::vector<u8> bytes;
    bool ok = false;
    auto f = cardinal::hal::File::open(p.req.path, cardinal::hal::FileMode::Read);
    if (f) {
        const u64 file_size = f->size();
        u64 sz = p.req.size;
        if (sz == 0 || p.req.offset + sz > file_size) {
            sz = (p.req.offset >= file_size) ? 0 : (file_size - p.req.offset);
        }
        if (sz > 0) {
            f->seek(p.req.offset);
            bytes.resize(static_cast<usize>(sz));
            const usize got = f->read(bytes.data(), bytes.size());
            ok = (got == bytes.size());
            if (!ok) bytes.clear();
        } else ok = true;
    }
    {
        std::lock_guard<std::mutex> lg(impl->mtx);
        Dispatcher::Impl::Done d{};
        d.handle = p.handle;
        d.req    = std::move(p.req);
        d.bytes  = std::move(bytes);
        d.ok     = ok;
        impl->done_queue.push_back(std::move(d));
        --impl->in_flight[static_cast<u32>(d.req.priority)];
    }
    impl->in_flight_total.fetch_sub(1, std::memory_order_relaxed);
    impl->bytes_in_flight.fetch_sub(p.req.size, std::memory_order_relaxed);
}

}  // namespace

void Dispatcher::tick() {
    using clock = std::chrono::high_resolution_clock;
    const auto t0 = clock::now();

    // 1) Move ready jobs to async pool. Walk priorities high → low.
    {
        std::lock_guard<std::mutex> lg(impl_->mtx);
        const u32 caps[5] = {
            impl_->desc.max_concurrent_background,
            impl_->desc.max_concurrent_low,
            impl_->desc.max_concurrent_normal,
            impl_->desc.max_concurrent_high,
            impl_->desc.max_concurrent_critical,
        };
        // Iterate Critical → Background.
        for (i32 pi = 4; pi >= 0; --pi) {
            const u32 cap = caps[pi];
            auto& q = impl_->queues[pi];
            while (!q.empty() && impl_->in_flight[pi] < cap &&
                   (impl_->desc.max_concurrent_total == 0 ||
                    impl_->in_flight_total.load(std::memory_order_relaxed) <
                        impl_->desc.max_concurrent_total))
            {
                Pending p = std::move(q.front());
                q.pop_front();
                if (p.cancelled) continue;
                ++impl_->in_flight[pi];
                impl_->in_flight_total.fetch_add(1, std::memory_order_relaxed);
                impl_->bytes_in_flight.fetch_add(p.req.size, std::memory_order_relaxed);
                Impl* impl_ptr = impl_;
                cardinal::async::async([impl_ptr, p = std::move(p)]() mutable {
                    run_request(impl_ptr, std::move(p));
                    return 0;
                });
            }
        }
    }

    // 2) Drain completions and fire callbacks (outside the lock).
    // The cancellation lookup MUST happen under impl_->mtx — Dispatcher::
    // cancel and cancel_tag insert into impl_->cancelled_handles under the
    // lock from arbitrary caller threads. Reading the unordered_set's
    // bucket structure via count() concurrently with insert() is a data
    // race (std::unordered_set is not thread-safe), which is UB on the
    // container. Pre-resolve `cancelled` for each drained item while
    // we hold the lock; the user on_done callback then fires off-lock
    // (same lock-release-before-user-callback discipline as cppscript
    // 92ba1b5 and audio ff1acdf — handler may call back into
    // Dispatcher::cancel / submit / stats, all of which take mtx).
    struct DrainedItem {
        Impl::Done d;
        bool       cancelled;
    };
    std::vector<DrainedItem> drained;
    {
        std::lock_guard<std::mutex> lg(impl_->mtx);
        drained.reserve(impl_->done_queue.size());
        for (auto& d : impl_->done_queue) {
            DrainedItem item;
            item.cancelled = impl_->cancelled_handles.count(d.handle) > 0;
            item.d         = std::move(d);
            drained.push_back(std::move(item));
        }
        impl_->done_queue.clear();
        // Compact the cancelled set under the SAME lock — reading its
        // size() outside was the other half of the race. Doing it here
        // also avoids re-locking later.
        if (impl_->cancelled_handles.size() > 512) {
            impl_->cancelled_handles.clear();
        }
    }
    for (auto& it : drained) {
        if (!it.cancelled) {
            if (it.d.ok) {
                impl_->requests_completed.fetch_add(1, std::memory_order_relaxed);
                impl_->bytes_completed.fetch_add(it.d.bytes.size(), std::memory_order_relaxed);
            } else {
                impl_->requests_failed.fetch_add(1, std::memory_order_relaxed);
            }
            if (it.d.req.on_done) it.d.req.on_done(it.d.bytes);
        }
    }

    const auto t1 = clock::now();
    impl_->last_tick_ms.store(std::chrono::duration<f64, std::milli>(t1 - t0).count(),
                              std::memory_order_relaxed);
}

DispatcherStats Dispatcher::stats() const noexcept {
    DispatcherStats s{};
    std::lock_guard<std::mutex> lg(impl_->mtx);
    for (u32 i = 0; i < kPriorityCount; ++i) {
        s.queued[i]    = static_cast<u32>(impl_->queues[i].size());
        s.in_flight[i] = impl_->in_flight[i];
        s.queued_total += s.queued[i];
        s.in_flight_total += s.in_flight[i];
    }
    s.requests_seen      = impl_->requests_seen.load(std::memory_order_relaxed);
    s.requests_completed = impl_->requests_completed.load(std::memory_order_relaxed);
    s.requests_failed    = impl_->requests_failed.load(std::memory_order_relaxed);
    s.bytes_in_flight    = impl_->bytes_in_flight.load(std::memory_order_relaxed);
    s.bytes_completed    = impl_->bytes_completed.load(std::memory_order_relaxed);
    s.last_tick_ms       = impl_->last_tick_ms.load(std::memory_order_relaxed);
    return s;
}

}  // namespace cardinal::io
