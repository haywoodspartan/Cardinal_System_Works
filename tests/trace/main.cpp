// =============================================================================
// Cardinal — deterministic trace-timeline ring-buffer regression suite.
//
// Timeline is the lock-free SPSC ring behind the Function Tracer panel
// and the replay record. A fixed kCap ring + a monotonic write index;
// snapshot() returns the most-recent min(written, kCap) events,
// oldest-first, via (written - count) % kCap modular indexing. Lossy by
// design: on overflow the OLDEST events fall off. A regression silently
// corrupts that record — mis-ordered events, evicted-too-early, or a
// stale snapshot — and is invisible until someone studies a bad trace.
//
// This suite is the sole writer (single-threaded, headless) so the
// push / snapshot / clear / wrap-eviction math + strncpy name/category
// truncation are fully deterministic and exactly pinned. t_ns (steady
// clock) and thread_id (OS) are non-deterministic VALUES — asserted
// only as monotonic / self-consistent, never absolutely. Exit 0 = all
// pass.
// =============================================================================

#include <cardinal/trace/timeline.hpp>
#include <cardinal/core/diag/log.hpp>

#include <string>
#include <vector>

namespace {

namespace tr = cardinal::trace;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("tracetest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

cardinal::usize sz(cardinal::u64 n) { return static_cast<cardinal::usize>(n); }

tr::Timeline& tl() { return tr::Timeline::instance(); }

// Push a Marker carrying an identity in `depth` so snapshots are exactly
// verifiable after wrap/eviction.
void push_id(cardinal::u32 id, const char* name = "ev",
             const char* cat = "t") {
    tl().push(name, cat, tr::EventKind::Marker, id);
}

// ---- singleton identity + fixed capacity ---------------------------
void test_singleton_capacity() {
    CHECK(&tr::Timeline::instance() == &tr::Timeline::instance());
    CHECK(tl().capacity() == 16384u);
    CHECK(tl().capacity() > 0u);
}

// ---- clear() empties; snapshot resizes a dirty out-vector ----------
void test_clear_empty() {
    tl().clear();
    std::vector<tr::Event> out;
    out.resize(9);                 // pre-dirty: snapshot must shrink it
    tl().snapshot(out);
    CHECK(out.empty());

    // Repeated clear is idempotent; still empty.
    tl().clear();
    tl().snapshot(out);
    CHECK(out.empty());
}

// ---- push → snapshot field round-trip (order, kind, depth, names) --
void test_push_roundtrip() {
    tl().clear();
    tl().push("alpha", "lua",    tr::EventKind::Call,   3u);
    tl().push("beta",  "engine", tr::EventKind::Return, 7u);
    tl().push("gamma", "ui",     tr::EventKind::Marker, 0u);

    std::vector<tr::Event> out;
    tl().snapshot(out);
    CHECK(out.size() == sz(3));

    CHECK(std::string(out[0].name) == "alpha");
    CHECK(std::string(out[0].category) == "lua");
    CHECK(out[0].kind == tr::EventKind::Call);
    CHECK(out[0].depth == 3u);

    CHECK(std::string(out[1].name) == "beta");
    CHECK(out[1].kind == tr::EventKind::Return);
    CHECK(out[1].depth == 7u);

    CHECK(std::string(out[2].name) == "gamma");
    CHECK(out[2].category == std::string("ui"));
    CHECK(out[2].kind == tr::EventKind::Marker);
    CHECK(out[2].depth == 0u);

    // nullptr name/category → zero-initialised empty strings (no crash).
    tl().clear();
    tl().push(nullptr, nullptr, tr::EventKind::Marker, 1u);
    tl().snapshot(out);
    CHECK(out.size() == sz(1));
    CHECK(std::string(out[0].name).empty());
    CHECK(std::string(out[0].category).empty());
    CHECK(out[0].depth == 1u);
}

// ---- strncpy truncation: name[64] → 63 chars, category[16] → 15 ----
void test_truncation() {
    std::vector<tr::Event> out;

    // Short strings copied verbatim.
    tl().clear();
    tl().push("hi", "x", tr::EventKind::Marker);
    tl().snapshot(out);
    CHECK(std::string(out[0].name) == "hi");
    CHECK(std::string(out[0].category) == "x");

    // Name: exactly 63 fits whole; 64 and 70 truncate to 63.
    const std::string n63(63, 'A');
    const std::string n64(64, 'B');
    const std::string n70(70, 'C');
    const std::string c15(15, 'd');
    const std::string c16(16, 'e');
    const std::string c20(20, 'f');

    tl().clear();
    tl().push(n63.c_str(), c15.c_str(), tr::EventKind::Marker);
    tl().push(n64.c_str(), c16.c_str(), tr::EventKind::Marker);
    tl().push(n70.c_str(), c20.c_str(), tr::EventKind::Marker);
    tl().snapshot(out);
    CHECK(out.size() == sz(3));

    CHECK(std::string(out[0].name) == n63);            // 63 → unchanged
    CHECK(std::string(out[0].category) == c15);        // 15 → unchanged
    CHECK(std::string(out[0].name).size() == sz(63));

    CHECK(std::string(out[1].name).size() == sz(63));  // 64 → 63
    CHECK(std::string(out[1].name) == std::string(63, 'B'));
    CHECK(std::string(out[1].category).size() == sz(15));   // 16 → 15
    CHECK(std::string(out[1].category) == std::string(15, 'e'));

    CHECK(std::string(out[2].name).size() == sz(63));  // 70 → 63
    CHECK(std::string(out[2].name) == std::string(63, 'C'));
    CHECK(std::string(out[2].category).size() == sz(15));   // 20 → 15
}

// ---- t_ns monotonic + thread_id self-consistent + snapshot const ---
void test_meta_and_idempotent() {
    tl().clear();
    for (cardinal::u32 i = 0; i < 64u; ++i) push_id(i);

    std::vector<tr::Event> a, b;
    tl().snapshot(a);
    tl().snapshot(b);                       // const: must not mutate state
    CHECK(a.size() == sz(64) && b.size() == sz(64));

    bool ts_mono = true, tid_same = true, depth_seq = true, idem = true;
    for (cardinal::usize i = 0; i < a.size(); ++i) {
        if (a[i].thread_id != a[0].thread_id) tid_same = false;
        if (a[i].depth != static_cast<cardinal::u32>(i)) depth_seq = false;
        if (a[i].t_ns != b[i].t_ns || a[i].depth != b[i].depth) idem = false;
        if (i > 0 && a[i].t_ns < a[i - 1].t_ns) ts_mono = false;
    }
    CHECK(ts_mono);      // steady_clock never goes backward
    CHECK(tid_same);     // all from this one thread
    CHECK(depth_seq);    // preserved push order
    CHECK(idem);         // two snapshots identical (snapshot is const)
}

// ---- wrap-around + oldest-eviction: the headline ring math ---------
void test_wrap_eviction() {
    const cardinal::u32 cap = tl().capacity();   // 16384
    std::vector<tr::Event> out;

    // (a) Exactly cap pushes → no eviction yet: depths 0..cap-1.
    tl().clear();
    for (cardinal::u32 i = 0; i < cap; ++i) push_id(i);
    tl().snapshot(out);
    CHECK(out.size() == sz(cap));
    CHECK(out.front().depth == 0u);
    CHECK(out.back().depth == cap - 1u);

    // (b) One more → oldest (depth 0) falls off; window shifts by 1.
    push_id(cap);
    tl().snapshot(out);
    CHECK(out.size() == sz(cap));               // still saturated at cap
    CHECK(out.front().depth == 1u);             // event 0 evicted
    CHECK(out.back().depth == cap);

    // (c) 99 more (total cap+100) → oldest 100 gone; contiguous window
    //     [100 .. cap+99], strictly +1 across the whole snapshot.
    for (cardinal::u32 i = cap + 1u; i < cap + 100u; ++i) push_id(i);
    tl().snapshot(out);
    CHECK(out.size() == sz(cap));
    CHECK(out.front().depth == 100u);
    CHECK(out.back().depth == cap + 99u);
    CHECK(out[sz(cap) / 2].depth == 100u + cap / 2u);   // sampled middle

    bool contiguous = true;
    for (cardinal::usize i = 1; i < out.size(); ++i)
        if (out[i].depth != out[i - 1].depth + 1u) contiguous = false;
    CHECK(contiguous);   // exact order preserved across the wrap seam
}

// ---- clear() mid-stream resets the write index cleanly ------------
void test_clear_midstream() {
    std::vector<tr::Event> out;

    // Overflow the ring, then clear: the stale ring contents must NOT
    // resurface — a fresh short burst reads back exactly itself.
    tl().clear();
    const cardinal::u32 cap = tl().capacity();
    for (cardinal::u32 i = 0; i < cap + 50u; ++i) push_id(i, "stale");
    tl().clear();
    tl().snapshot(out);
    CHECK(out.empty());

    for (cardinal::u32 i = 0; i < 5u; ++i) push_id(900u + i, "fresh");
    tl().snapshot(out);
    CHECK(out.size() == sz(5));
    bool fresh_ok = true;
    for (cardinal::usize i = 0; i < out.size(); ++i) {
        if (out[i].depth != 900u + static_cast<cardinal::u32>(i)) fresh_ok = false;
        if (std::string(out[i].name) != "fresh") fresh_ok = false;
    }
    CHECK(fresh_ok);     // no stale event leaked through after clear
}

// ---- ScopeTracer RAII: Call on ctor, Return on dtor (nested) ------
void test_scope_tracer() {
    std::vector<tr::Event> out;

    tl().clear();
    {
        tr::ScopeTracer s("outer", "ui");
        {
            tr::ScopeTracer inner("inner", "engine");
        }                                  // inner Return here
    }                                      // outer Return here
    tl().snapshot(out);

    CHECK(out.size() == sz(4));
    CHECK(out[0].kind == tr::EventKind::Call   &&
          std::string(out[0].name) == "outer");
    CHECK(std::string(out[0].category) == "ui");
    CHECK(out[1].kind == tr::EventKind::Call   &&
          std::string(out[1].name) == "inner");
    CHECK(out[2].kind == tr::EventKind::Return &&
          std::string(out[2].name) == "inner");   // inner closes first
    CHECK(out[3].kind == tr::EventKind::Return &&
          std::string(out[3].name) == "outer");
    CHECK(std::string(out[3].category) == "ui");   // dtor re-uses ctor category

    // Default category is "engine" when omitted.
    tl().clear();
    { tr::ScopeTracer s("solo"); }
    tl().snapshot(out);
    CHECK(out.size() == sz(2));
    CHECK(std::string(out[0].category) == "engine");
    CHECK(out[0].kind == tr::EventKind::Call);
    CHECK(out[1].kind == tr::EventKind::Return);
}

// ---- macro: two CARDINAL_TRACE_SCOPE in one scope = unique names ---
// Regression guard for the __LINE__ token-paste fix. Pre-fix both uses
// expanded to the same identifier (_ct_scope___LINE__) → C2374 hard
// error; this TU would not compile at all. Two distinct ScopeTracers in
// one block ⇒ 4 events; both Returns fire at scope exit in LIFO order
// ("b" constructed last, destroyed first).
void test_macro_unique_names() {
    std::vector<tr::Event> out;

    tl().clear();
    {
        CARDINAL_TRACE_SCOPE("a");
        CARDINAL_TRACE_SCOPE_C("b", "ui");
    }                                      // b Return, then a Return
    tl().snapshot(out);

    CHECK(out.size() == sz(4));
    CHECK(out[0].kind == tr::EventKind::Call   &&
          std::string(out[0].name) == "a");
    CHECK(std::string(out[0].category) == "engine");   // default category
    CHECK(out[1].kind == tr::EventKind::Call   &&
          std::string(out[1].name) == "b");
    CHECK(std::string(out[1].category) == "ui");
    CHECK(out[2].kind == tr::EventKind::Return &&
          std::string(out[2].name) == "b");            // LIFO: b closes first
    CHECK(out[3].kind == tr::EventKind::Return &&
          std::string(out[3].name) == "a");
}

}  // namespace

int main() {
    test_singleton_capacity();
    test_clear_empty();
    test_push_roundtrip();
    test_truncation();
    test_meta_and_idempotent();
    test_wrap_eviction();
    test_clear_midstream();
    test_scope_tracer();
    test_macro_unique_names();

    if (g_fail == 0) {
        cardinal::log::infof("tracetest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("tracetest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
