// =============================================================================
// Cardinal — async asset-STREAMING integration suite (pack↔async↔jobs).
//
// The pack suite locks Builder/Archive synchronously and explicitly
// DEFERS load_async ("needs the JobSystem + thread nondeterminism").
// The async suite pins the Future then()-vs-resolve lost-wakeup
// contract that load_async DEPENDS on, but never drives the real
// pack::Archive::load_async path. Nothing exercised the actual
// shipping streaming seam:
//
//   pk::Archive::load_async(key)
//     --cardinal::async::async--> JobSystem worker reads the .cpk slice
//     --Future.then → promise--> cardinal::shared_future<vector<u8>>
//
// A regression here streams the WRONG bytes for a key, cross-talks
// between concurrent loads, leaks the in-flight counter, or loses the
// continuation (asset never resolves → load hang) — none of which any
// isolated suite catches.
//
// Schedule-INDEPENDENT invariants only (same robust approach as the
// jobs/async suites): N concurrent loads each resolve to EXACTLY their
// own entry's bytes regardless of completion order; the in-flight
// counter is balanced back to 0; bytes_streamed reconciles exactly;
// the no-pool path resolves inline; missing keys resolve empty (no
// hang). 256 concurrent loads also stress the work-stealing deque
// backpressure fixed in ac77f18 under realistic streaming fan-out.
//
// Deterministic + headless (temp dir only). Exit 0 = all pass.
// =============================================================================

#include <cardinal/pack/pack.hpp>
#include <cardinal/core/sync/async.hpp>
#include <cardinal/core/sync/jobs.hpp>
#include <cardinal/core/sync/topology.hpp>
#include <cardinal/core/diag/log.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace pk = cardinal::pack;
namespace ca = cardinal::async;
namespace fs = std::filesystem;
using cardinal::u8;
using cardinal::u32;
using cardinal::u64;
using cardinal::usize;
using AssetType = cardinal::cook::AssetType;
using SF = cardinal::shared_future<cardinal::vector<u8>>;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("packstream", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

// Deterministic xorshift blob — distinct content AND distinct length
// per seed so a cross-talked / truncated stream is provably caught.
std::vector<u8> blob(u32 seed, usize n) {
    std::vector<u8> v;
    v.reserve(n);
    u32 x = seed ? seed : 1u;
    for (usize i = 0; i < n; ++i) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        v.push_back(static_cast<u8>(x & 0xFFu));
    }
    return v;
}

}  // namespace

int main() {
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec);
    if (ec) {
        cardinal::log::infof("packstream",
            "SKIP  no writable temp dir in this environment");
        return 0;
    }
    const fs::path dir = base / "cardinal_pack_stream_test";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    if (ec) {
        cardinal::log::infof("packstream", "SKIP  temp dir uncreatable");
        return 0;
    }
    const std::string cpk = (dir / "stream.cpk").string();

    // ---- Build a real .cpk: N entries, each a unique (content,length)
    // so a future returning another entry's bytes is impossible to miss.
    constexpr int N = 256;
    std::vector<std::string>     keys(static_cast<usize>(N));
    std::vector<std::vector<u8>> payloads(static_cast<usize>(N));
    u64 total_payload = 0;
    for (int i = 0; i < N; ++i) {
        keys[static_cast<usize>(i)] = "asset/" + std::to_string(i);
        const usize len = 48u + static_cast<usize>(i % 200);   // 48..247
        payloads[static_cast<usize>(i)] =
            blob(static_cast<u32>(i) * 2654435761u + 1u, len);
        total_payload += len;
    }
    {
        pk::Builder b(cpk);
        for (int i = 0; i < N; ++i)
            CHECK(b.add(keys[static_cast<usize>(i)], AssetType::Unknown,
                        payloads[static_cast<usize>(i)]));
        cardinal::string err;
        CHECK(b.finalize(&err));
        CHECK(err.empty());
    }
    cardinal::string oerr;
    cardinal::shared_ptr<pk::Archive> arch = pk::Archive::open(cpk, &oerr);
    CHECK(arch != nullptr);
    if (!arch) {
        cardinal::log::errorf("packstream", "open: %s", oerr.c_str());
        return 1;
    }
    CHECK(arch->entry_count() == static_cast<u32>(N));

    // ---- Phase A: NO pool bound → load_async resolves INLINE,
    // deterministically. The real streaming entry point must still work
    // (and missing keys resolve empty, never hang) with no JobSystem.
    CHECK(ca::pool() == nullptr);
    {
        SF f0 = arch->load_async(keys[0]);
        SF f9 = arch->load_async(keys[9]);
        CHECK(f0.get() == payloads[0]);
        CHECK(f9.get() == payloads[9]);
        SF fmiss = arch->load_async("does/not/exist");
        CHECK(fmiss.get().empty());                  // empty, no hang
    }

    // ---- Phase B: pooled — fire ALL N concurrently, then collect.
    // Issuing every load before getting any maximises real overlap on
    // the JobSystem (and the deque backpressure path).
    cardinal::JobSystem js;
    js.start(cardinal::derive_worker_plan_physical_only(
        cardinal::topology::detect()));
    ca::set_pool(&js);
    CHECK(ca::pool() == &js);

    const u64 streamed_before = arch->stats().bytes_streamed;
    {
        std::vector<SF> futs;
        futs.reserve(static_cast<usize>(N));
        for (int i = 0; i < N; ++i)
            futs.push_back(arch->load_async(keys[static_cast<usize>(i)]));

        // Collect in REVERSE order — out-of-order retrieval is the
        // hard stop for any shared-state cross-talk in the per-call
        // closure / Future→promise adapter.
        bool all_exact = true;
        for (int i = N - 1; i >= 0; --i) {
            const cardinal::vector<u8> v = futs[static_cast<usize>(i)].get();
            if (v != payloads[static_cast<usize>(i)]) all_exact = false;
        }
        CHECK(all_exact);                            // exactly its OWN bytes
    }
    {
        const pk::Archive::Stats st = arch->stats();
        CHECK(st.streams_in_flight == 0u);           // counter balanced
        CHECK(st.bytes_streamed - streamed_before == total_payload);
    }

    // ---- Phase C: determinism — a second concurrent batch yields the
    // same bytes and streams exactly total_payload again (each load is
    // an independent re-read of the file).
    {
        std::vector<SF> futs;
        futs.reserve(static_cast<usize>(N));
        for (int i = 0; i < N; ++i)
            futs.push_back(arch->load_async(keys[static_cast<usize>(i)]));
        bool all_exact = true;
        for (int i = 0; i < N; ++i)
            if (futs[static_cast<usize>(i)].get() != payloads[static_cast<usize>(i)])
                all_exact = false;
        CHECK(all_exact);
        const pk::Archive::Stats st = arch->stats();
        CHECK(st.streams_in_flight == 0u);
        CHECK(st.bytes_streamed - streamed_before == 2u * total_payload);
    }

    // ---- Phase D: pooled missing key resolves empty and does NOT
    // perturb bytes_streamed (size==0 path returns before the counter).
    {
        const u64 b4 = arch->stats().bytes_streamed;
        SF fmiss = arch->load_async("nope/missing");
        CHECK(fmiss.get().empty());
        CHECK(arch->stats().bytes_streamed == b4);
        CHECK(arch->stats().streams_in_flight == 0u);
    }

    ca::set_pool(nullptr);
    CHECK(ca::pool() == nullptr);
    js.shutdown();

    // Post-shutdown, post-unbind: still resolves inline & correct.
    {
        SF f = arch->load_async(keys[123]);
        CHECK(f.get() == payloads[123]);
    }

    fs::remove_all(dir, ec);

    if (g_fail == 0) {
        cardinal::log::infof("packstream", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("packstream", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
