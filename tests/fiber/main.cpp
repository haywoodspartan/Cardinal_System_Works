// =============================================================================
// Cardinal — cardinal::fiber regression suite (the context switch itself).
//
// Drives the primitive without the job system so a fault names its stage:
//   1. allocate_stack: guard page below the usable region
//   2. a fresh context's first entry + switch back
//   3. ping-pong switches (and a ns/switch figure)
//   4. nested contexts (A -> B -> A -> main)
//   5. deep recursion on a fiber stack (> 4 KB frames = stack probes)
//   6. a C++ exception thrown and caught on a fiber stack
//   7. floating-point state (XMM6..15 / MXCSR) across a switch
// Exit 0 = all pass. SKIP (exit 0) when CARDINAL_FIBER_ASM is 0.
// =============================================================================
#include <cardinal/core/sync/fiber.hpp>
#include <cardinal/core/diag/log.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

namespace fb = cardinal::fiber;
using cardinal::u8; using cardinal::u32; using cardinal::usize;

int g_checks = 0, g_fail = 0;
void check_impl(bool ok, const char* e, int l) {
    ++g_checks;
    if (!ok) { ++g_fail; cardinal::log::errorf("fibertest", "FAIL  L%d  %s", l, e); }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

void stage(const char* what) {
    std::printf("[fibertest] stage: %s\n", what);
    std::fflush(stdout);
}

#if CARDINAL_FIBER_ASM

// ---- shared state for the contexts -----------------------------------------
fb::Context g_main;      // the thread's own stack
fb::Context g_a, g_b;
int         g_trace[16];
int         g_trace_n = 0;
void trace(int v) { if (g_trace_n < 16) g_trace[g_trace_n++] = v; }

void entry_simple(void* arg) noexcept {
    trace(*static_cast<int*>(arg));          // 1: we are in
    fb::switch_context(g_a, g_main);         // back to main
    trace(3);                                // resumed a second time
    fb::switch_context(g_a, g_main);
    std::abort();                            // never resumed again
}

void entry_pingpong(void* arg) noexcept {
    auto* rounds = static_cast<int*>(arg);
    for (;;) {
        ++*rounds;
        fb::switch_context(g_a, g_main);
    }
}

void entry_b(void*) noexcept {
    trace(20);
    fb::switch_context(g_b, g_a);            // back into A
    std::abort();
}
void entry_a_nested(void*) noexcept {
    trace(10);
    fb::switch_context(g_a, g_b);            // run B
    trace(11);                               // B switched back to us
    fb::switch_context(g_a, g_main);
    std::abort();
}

int deep(int depth) noexcept {
    volatile char pad[8192];
    pad[0] = static_cast<char>(depth);
    pad[8191] = 1;
    if (depth == 0) return static_cast<int>(pad[0]) + static_cast<int>(pad[8191]);
    return deep(depth - 1) + 1;
}
void entry_deep(void* arg) noexcept {
    *static_cast<int*>(arg) = deep(48);      // ~400 KB of a 1 MiB stack
    fb::switch_context(g_a, g_main);
    std::abort();
}

void entry_throw(void* arg) noexcept {
    int caught = 0;
    try { throw 42; } catch (int v) { caught = v; }
    *static_cast<int*>(arg) = caught;
    fb::switch_context(g_a, g_main);
    std::abort();
}

void entry_fp(void* arg) noexcept {
    // Churn the FP registers on this side so the main side's values must
    // come back from the saved frame, not from luck.
    double acc = 0.0;
    for (int i = 1; i <= 4096; ++i) acc += 1.0 / static_cast<double>(i);
    *static_cast<double*>(arg) = acc;
    fb::switch_context(g_a, g_main);
    std::abort();
}

int run() {
    stage("allocate_stack");
    fb::Stack sa = fb::allocate_stack(1024 * 1024);
    fb::Stack sb = fb::allocate_stack(256 * 1024);
    CHECK(sa.base != nullptr && sb.base != nullptr);
    CHECK(sa.size >= 1024 * 1024 && sb.size >= 256 * 1024);
    CHECK(sa.alloc != nullptr && static_cast<u8*>(sa.base) > static_cast<u8*>(sa.alloc));
    CHECK(sa.alloc_size > sa.size);
    if (g_fail) return 1;

    stage("first entry + switch back");
    int one = 1;
    fb::make_context(g_a, sa, &entry_simple, &one);
    g_trace_n = 0;
    fb::switch_context(g_main, g_a);
    trace(2);
    fb::switch_context(g_main, g_a);
    trace(4);
    CHECK(g_trace_n == 4 && g_trace[0] == 1 && g_trace[1] == 2 && g_trace[2] == 3 && g_trace[3] == 4);
    if (g_fail) return 1;

    stage("ping-pong");
    int rounds = 0;
    fb::make_context(g_a, sa, &entry_pingpong, &rounds);
    constexpr int kRounds = 200000;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kRounds; ++i) fb::switch_context(g_main, g_a);
    const auto t1 = std::chrono::steady_clock::now();
    CHECK(rounds == kRounds);
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / (2.0 * kRounds);
    cardinal::log::infof("fibertest", "switch_context: %.1f ns per switch (%d round trips)", ns, kRounds);

    stage("nested contexts");
    g_trace_n = 0;
    fb::make_context(g_a, sa, &entry_a_nested, nullptr);
    fb::make_context(g_b, sb, &entry_b, nullptr);
    fb::switch_context(g_main, g_a);
    CHECK(g_trace_n == 3 && g_trace[0] == 10 && g_trace[1] == 20 && g_trace[2] == 11);
    if (g_fail) return 1;

    stage("deep recursion on a fiber stack");
    int deep_out = 0;
    fb::make_context(g_a, sa, &entry_deep, &deep_out);
    fb::switch_context(g_main, g_a);
    CHECK(deep_out == 49);

    stage("exception caught on a fiber stack");
    int caught = 0;
    fb::make_context(g_a, sa, &entry_throw, &caught);
    fb::switch_context(g_main, g_a);
    CHECK(caught == 42);

    stage("floating-point state across a switch");
    double keep[10];
    for (int k = 0; k < 10; ++k) keep[k] = 1.0 + 0.125 * static_cast<double>(k);
    double other = 0.0;
    fb::make_context(g_a, sa, &entry_fp, &other);
    double live = 0.0;
    for (int k = 0; k < 10; ++k) live += keep[k] * keep[k];   // likely parked in xmm regs
    fb::switch_context(g_main, g_a);
    double again = 0.0;
    for (int k = 0; k < 10; ++k) again += keep[k] * keep[k];
    CHECK(live == again);
    CHECK(other > 8.0 && other < 9.0);   // H(4096) ≈ 8.90

    stage("free_stack");
    fb::free_stack(sa);
    fb::free_stack(sb);
    CHECK(sa.base == nullptr && sb.base == nullptr);
    return 0;
}

#endif  // CARDINAL_FIBER_ASM

}  // namespace

int main() {
#if CARDINAL_FIBER_ASM
    if (run() != 0 || g_fail != 0) {
        cardinal::log::errorf("fibertest", "%d / %d checks FAILED", g_fail, g_checks);
        return 1;
    }
    cardinal::log::infof("fibertest", "OK  %d checks passed", g_checks);
    return 0;
#else
    cardinal::log::infof("fibertest", "SKIP: CARDINAL_FIBER_ASM is 0 on this target");
    return 0;
#endif
}
