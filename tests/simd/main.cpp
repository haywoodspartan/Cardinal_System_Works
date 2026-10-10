// =============================================================================
// Cardinal — SIMD math dispatch suite.
//
// Part 1 (pass/fail): for EVERY ISA tier this CPU supports, force the dispatch
// table onto it (simd::force_backend) and check
//   * the reductions (dot / sum / min / max) against a double-precision
//     reference, on three input families:
//       - integer-valued data: every partial sum is exactly representable, so
//         the result is independent of accumulation order and must be EXACT
//         across tiers (this is what catches a dropped tail, a chain that was
//         never folded in, or a duplicated block);
//       - random data in [-1, 1): the float error bound;
//       - degenerate sizes: 0, 1, and every width-boundary ±1;
//   * the element-wise ops (add / sub / mul / scale) against a lane-exact
//     scalar computation — same single rounding, so bit-exact across tiers;
//   * that the forced tier is what the dispatcher reports.
//
// Part 2 (informational): time the multi-chain reductions against the
// original single-chain loops (naive_*.cpp, verbatim, same arch flags) in
// the L1-resident regime (where latency dominates) and the streaming regime
// (where bandwidth does). Logged, never asserted — a slow box is not a bug.
//
// Tiers the host lacks are skipped with a log line. Exit 0 = all pass.
// =============================================================================
#include <cardinal/core/math/simd_math.hpp>
#include <cardinal/core/os/hal.hpp>
#include <cardinal/core/diag/log.hpp>

#include "naive.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace {

namespace sd = cardinal::core::simd;
using cardinal::f32; using cardinal::u32; using cardinal::u64; using cardinal::usize;

int g_checks = 0, g_fail = 0;
void check_impl(bool ok, const char* e, int l) {
    ++g_checks;
    if (!ok) { ++g_fail; cardinal::log::errorf("simdtest", "FAIL  L%d  %s", l, e); }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

// Deterministic generator (splitmix64) — same data on every host / run.
struct Rng {
    u64 s;
    explicit Rng(u64 seed) : s(seed) {}
    u64 next() {
        u64 z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    f32 unit() {   // [-1, 1)
        return static_cast<f32>(static_cast<double>(next() >> 11) *
                                (2.0 / 9007199254740992.0)) - 1.0f;
    }
};

const sd::Backend kTiers[] = {
    sd::Backend::Scalar, sd::Backend::SSE42, sd::Backend::AVX,
    sd::Backend::AVX2,   sd::Backend::AVX512,
};

// Every width boundary ±1 for 4 / 8 / 16 / 32 / 64 / 128-float blocks, plus a
// few odd lengths. The multi-chain loops step 128 (or 64 / 32) at a time and
// then fall through 16 / 8 / 4-wide remainders and a tail, so each boundary
// is a different code path.
const usize kSizes[] = {
    0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
    127, 128, 129, 255, 256, 257, 383, 384, 385, 1000, 4096 + 5, 65536 + 13,
};

// ---- Part 1a: reductions on integer-valued data — exact across tiers ------
void test_reductions_exact_integer(sd::Backend tier) {
    for (usize n : kSizes) {
        std::vector<f32> a(n), b(n);
        for (usize i = 0; i < n; ++i) {
            a[i] = static_cast<f32>(static_cast<int>(i % 7) - 3);   // -3..3
            b[i] = static_cast<f32>(static_cast<int>(i % 5) - 2);   // -2..2
        }
        double dot_ref = 0.0, sum_ref = 0.0;
        f32 min_ref = std::numeric_limits<f32>::infinity();
        f32 max_ref = -std::numeric_limits<f32>::infinity();
        for (usize i = 0; i < n; ++i) {
            dot_ref += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            sum_ref += static_cast<double>(a[i]);
            if (a[i] < min_ref) min_ref = a[i];
            if (a[i] > max_ref) max_ref = a[i];
        }
        // |partial sums| < 2^24 everywhere -> every float add is exact.
        CHECK(sd::dot_f32(a.data(), b.data(), n) == static_cast<f32>(dot_ref));
        CHECK(sd::sum_f32(a.data(), n)           == static_cast<f32>(sum_ref));
        CHECK(sd::min_f32(a.data(), n)           == min_ref);
        CHECK(sd::max_f32(a.data(), n)           == max_ref);
        (void)tier;
    }
    // All-ones: dot == sum == n exactly, up to 2^24.
    {
        const usize n = 1u << 20;
        std::vector<f32> ones(n, 1.0f);
        CHECK(sd::dot_f32(ones.data(), ones.data(), n) == static_cast<f32>(n));
        CHECK(sd::sum_f32(ones.data(), n)              == static_cast<f32>(n));
    }
}

// ---- Part 1b: reductions on random data — float error bound ----------------
void test_reductions_random(sd::Backend tier) {
    Rng rng(0xC0FFEEu + static_cast<u64>(tier));
    for (usize n : kSizes) {
        std::vector<f32> a(n), b(n);
        for (usize i = 0; i < n; ++i) { a[i] = rng.unit(); b[i] = rng.unit(); }
        double dot_ref = 0.0, sum_ref = 0.0, dot_abs = 0.0, sum_abs = 0.0;
        for (usize i = 0; i < n; ++i) {
            const double p = static_cast<double>(a[i]) * static_cast<double>(b[i]);
            dot_ref += p;             dot_abs += std::fabs(p);
            sum_ref += a[i];          sum_abs += std::fabs(static_cast<double>(a[i]));
        }
        // Worst-case sequential float accumulation error is n·eps·Σ|terms|
        // (the scalar tier IS sequential); the multi-chain tiers are far
        // inside this bound, which is the point of pairwise folding.
        const double eps = static_cast<double>(n) * 6.0e-8 + 1.0e-7;
        const double d = static_cast<double>(sd::dot_f32(a.data(), b.data(), n));
        const double s = static_cast<double>(sd::sum_f32(a.data(), n));
        CHECK(std::fabs(d - dot_ref) <= eps * dot_abs + 1.0e-6);
        CHECK(std::fabs(s - sum_ref) <= eps * sum_abs + 1.0e-6);
        f32 mn = std::numeric_limits<f32>::infinity(), mx = -mn;
        for (usize i = 0; i < n; ++i) { if (a[i] < mn) mn = a[i]; if (a[i] > mx) mx = a[i]; }
        CHECK(sd::min_f32(a.data(), n) == mn);
        CHECK(sd::max_f32(a.data(), n) == mx);
    }
}

// ---- Part 1c: element-wise ops are lane-exact across tiers -----------------
void test_elementwise_exact(sd::Backend tier) {
    Rng rng(0xBEEFu + static_cast<u64>(tier));
    for (usize n : kSizes) {
        std::vector<f32> a(n), b(n), out(n, 0.0f);
        for (usize i = 0; i < n; ++i) { a[i] = rng.unit() * 8.0f; b[i] = rng.unit() * 8.0f; }
        bool ok = true;
        sd::vec_add_f32(out.data(), a.data(), b.data(), n);
        for (usize i = 0; i < n; ++i) ok = ok && (out[i] == a[i] + b[i]);
        sd::vec_sub_f32(out.data(), a.data(), b.data(), n);
        for (usize i = 0; i < n; ++i) ok = ok && (out[i] == a[i] - b[i]);
        sd::vec_mul_f32(out.data(), a.data(), b.data(), n);
        for (usize i = 0; i < n; ++i) ok = ok && (out[i] == a[i] * b[i]);
        sd::vec_scale_f32(out.data(), a.data(), 1.5f, n);
        for (usize i = 0; i < n; ++i) ok = ok && (out[i] == a[i] * 1.5f);
        CHECK(ok);
    }
}

void test_dispatcher_reports(sd::Backend tier) {
    CHECK(sd::active_backend() == tier);
    CHECK(sd::backend_for(sd::OpId::Dot_f32) == tier);
    CHECK(sd::backend_for(sd::OpId::Sum_f32) == tier);
    // AVX borrows mat4_mul_array from SSE4.2 — the only mixed-tier op.
    if (tier == sd::Backend::AVX) CHECK(sd::backend_for(sd::OpId::Mat4Mul_Array) == sd::Backend::SSE42);
    else                          CHECK(sd::backend_for(sd::OpId::Mat4Mul_Array) == tier);
}

// ---- Part 2: micro-benchmark, old single-chain vs new multi-chain ----------
using DotFn = f32 (*)(const f32*, const f32*, usize);
using SumFn = f32 (*)(const f32*, usize);

double time_ns_per_elem(DotFn dot, SumFn sum, const f32* a, const f32* b, usize n, int reps) {
    volatile f32 sink = 0.0f;
    // Warm-up pulls the arrays into cache for the L1 regime.
    for (int r = 0; r < 4; ++r) { sink = sink + (dot ? dot(a, b, n) : sum(a, n)); }
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) { sink = sink + (dot ? dot(a, b, n) : sum(a, n)); }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return ns / (static_cast<double>(reps) * static_cast<double>(n));
}

void bench_tier(const char* name, sd::Backend tier, DotFn old_dot, SumFn old_sum) {
    if (!sd::force_backend(tier)) return;
    // Three regimes. Two arrays of 4096 floats = 32 KiB total — inside a
    // 48 KB L1D, where the loop is latency-bound and the chains pay off in
    // full. 256K floats = 2 MiB total — L2/L3, where L2 fill bandwidth caps
    // both versions at roughly the same rate. 4M floats = 32 MiB — DRAM
    // streaming, where only memory-level parallelism differs.
    const usize sizes [3] = { 4096, usize(1) << 18, usize(1) << 22 };
    const int   reps  [3] = { 8000, 200, 16 };
    const char* regime[3] = { "L1-resident", "L2/L3", "streaming" };
    for (int k = 0; k < 3; ++k) {
        const usize n = sizes[k];
        std::vector<f32> a(n), b(n);
        for (usize i = 0; i < n; ++i) {
            a[i] = 1.0f + static_cast<f32>(i % 13) * 0.01f;
            b[i] = 0.5f + static_cast<f32>(i % 7)  * 0.02f;
        }
        const double old_d = time_ns_per_elem(old_dot, nullptr, a.data(), b.data(), n, reps[k]);
        const double new_d = time_ns_per_elem(sd::dot_f32, nullptr, a.data(), b.data(), n, reps[k]);
        const double old_s = time_ns_per_elem(nullptr, old_sum, a.data(), nullptr, n, reps[k]);
        const double new_s = time_ns_per_elem(nullptr, sd::sum_f32, a.data(), nullptr, n, reps[k]);
        cardinal::log::infof("simdtest",
            "bench %-7s %-11s n=%-8zu dot %.4f -> %.4f ns/elem (%.2fx)   sum %.4f -> %.4f ns/elem (%.2fx)",
            name, regime[k], static_cast<size_t>(n),
            old_d, new_d, old_d / (new_d > 0.0 ? new_d : 1.0e-9),
            old_s, new_s, old_s / (new_s > 0.0 ? new_s : 1.0e-9));
    }
}

}  // namespace

int main() {
    sd::init();
    const sd::Backend best = sd::best_backend();
    cardinal::log::infof("simdtest", "best tier on this CPU: %s", sd::backend_name(best));

    // A tier above the CPU's best must be refused and leave the binding alone.
    if (best != sd::Backend::AVX512) {
        CHECK(!sd::force_backend(sd::Backend::AVX512));
        CHECK(sd::active_backend() == best);
    }

    for (sd::Backend tier : kTiers) {
        if (!sd::force_backend(tier)) {
            cardinal::log::infof("simdtest", "SKIP %s: not supported on this CPU", sd::backend_name(tier));
            continue;
        }
        const int before = g_fail;
        test_dispatcher_reports(tier);
        test_reductions_exact_integer(tier);
        test_reductions_random(tier);
        test_elementwise_exact(tier);
        cardinal::log::infof("simdtest", "%-9s %s", sd::backend_name(tier),
                             g_fail == before ? "ok" : "FAILED");
    }

    const auto& cf = cardinal::hal::cpu_features();
    if (cf.avx2 && cf.fma3) bench_tier("AVX2",    sd::Backend::AVX2,   &naive_avx2::dot_f32,   &naive_avx2::sum_f32);
    if (cf.avx512f)         bench_tier("AVX-512", sd::Backend::AVX512, &naive_avx512::dot_f32, &naive_avx512::sum_f32);

    CHECK(sd::force_backend(best));
    CHECK(sd::active_backend() == best);

    if (g_fail == 0) {
        cardinal::log::infof("simdtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("simdtest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
