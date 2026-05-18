#pragma once

// =============================================================================
// Cardinal — function-event timeline (lock-free SPSC ring buffer).
//
// A single global ring of {tid, name, kind, t_ns, depth} events that drives
// the Function Tracer panel. Producers:
//   - cardinal::script — Lua hook pushes Call/Return events for every Lua
//     function on the active state
//   - native code — CARDINAL_TRACE_SCOPE("name") RAII helper pushes
//     Call/Return events around any C++ scope
//
// The ring is a simple lock-free single-writer-multi-reader design — pushes
// from multiple threads serialise on a relaxed atomic counter; readers (the
// Studio panel) snapshot a copy under that counter. Lossy by design: when
// the ring is full, oldest events fall off.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <atomic>
#include <chrono>
#include <string>
#include <vector>

namespace cardinal::trace {

enum class EventKind : u32 {
    Call   = 0,
    Return = 1,
    Marker = 2,    // a one-shot point event (no matching return)
};

struct Event {
    u64        t_ns{0};        // ns since process start
    u32        thread_id{0};
    u32        depth{0};       // call depth in producer's bookkeeping
    EventKind  kind{EventKind::Marker};
    char       name[64]{};     // function or scope name (truncated if longer)
    char       category[16]{}; // "lua", "engine", "ui", etc.
};

class Timeline {
public:
    static Timeline& instance();

    void push(const char* name, const char* category,
              EventKind kind, u32 depth = 0);

    // Snapshot current contents into out. Lossy if writers added more than
    // capacity since the last snapshot.
    void snapshot(std::vector<Event>& out) const;

    // Clear the ring (also resets the writer index). Useful when the user
    // hits "Restart capture" in the panel.
    void clear();

    u32  capacity() const noexcept { return kCap; }

private:
    Timeline();
    Timeline(const Timeline&)            = delete;
    Timeline& operator=(const Timeline&) = delete;

    static constexpr u32 kCap = 16384;
    std::vector<Event>      ring_;
    std::atomic<u64>        write_index_{0};
    std::chrono::steady_clock::time_point t0_;
};

// RAII scope tracer for native code: emits Call on construction, Return on
// destruction. The name pointer is stored truncated (no allocation in
// hot path), so prefer string literals or interned strings.
class ScopeTracer {
public:
    ScopeTracer(const char* name, const char* category = "engine") noexcept
        : name_(name), category_(category) {
        Timeline::instance().push(name_, category_, EventKind::Call);
    }
    ~ScopeTracer() noexcept {
        Timeline::instance().push(name_, category_, EventKind::Return);
    }
    ScopeTracer(const ScopeTracer&)            = delete;
    ScopeTracer& operator=(const ScopeTracer&) = delete;
private:
    const char* name_{""};
    const char* category_{"engine"};
};

}  // namespace cardinal::trace

// __LINE__ must expand to its value BEFORE the paste — operands of ## are
// not macro-expanded, so one level of indirection is required to get a
// unique variable name per source line (else two-per-scope = C2374).
#define CARDINAL_TRACE_DETAIL_CAT2(a, b) a##b
#define CARDINAL_TRACE_DETAIL_CAT(a, b)  CARDINAL_TRACE_DETAIL_CAT2(a, b)

#define CARDINAL_TRACE_SCOPE(name)         ::cardinal::trace::ScopeTracer CARDINAL_TRACE_DETAIL_CAT(_ct_scope_, __LINE__)((name))
#define CARDINAL_TRACE_SCOPE_C(name, cat)  ::cardinal::trace::ScopeTracer CARDINAL_TRACE_DETAIL_CAT(_ct_scope_, __LINE__)((name), (cat))
