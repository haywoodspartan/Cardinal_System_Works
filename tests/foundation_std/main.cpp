// =============================================================================
// Cardinal — Phase-0 Foundation-vocabulary regression suite.
//
// Includes ONLY the cardinal umbrella (no direct std header) and exercises
// every alias category, proving <cardinal/core/std.hpp> delivers the full
// restructured vocabulary with no internal ambiguity. Deterministic.
// Exit 0 = all pass.
// =============================================================================

#include <cardinal/core/std.hpp>
#include <cardinal/core/diag/log.hpp>

namespace {

using cardinal::u32;
using cardinal::i64;
using cardinal::usize;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("fstd", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

// ---- types (core/types.hpp via the umbrella) ----------------------
void test_types() {
    cardinal::string s = "cardinal";
    CHECK(s.size() == 8u);
    cardinal::unique_ptr<int> p = cardinal::make_unique<int>(42);
    CHECK(p && *p == 42);
    cardinal::function<int(int)> f = [](int x) { return x * 2; };
    CHECK(f(21) == 42);
    cardinal::string_view sv = "abcdef";
    CHECK(sv.substr(2, 3) == "cde");
}

// ---- containers ---------------------------------------------------
void test_containers() {
    cardinal::vector<int> v{5, 3, 1, 4, 2};
    cardinal::sort(v.begin(), v.end());
    CHECK(v.front() == 1 && v.back() == 5);

    cardinal::array<int, 4> a{10, 20, 30, 40};
    CHECK(cardinal::get<2>(a) == 30);

    cardinal::span<int> sp(v);
    CHECK(sp.size() == 5u && sp[0] == 1);

    cardinal::unordered_map<cardinal::string, int> m;
    m["x"] = 7; m["y"] = 9;
    CHECK(m.size() == 2u && m.at("y") == 9);

    cardinal::map<int, int> om; om[3] = 1; om[1] = 1; om[2] = 1;
    CHECK(om.begin()->first == 1);                 // ordered

    cardinal::unordered_set<int> us{1, 2, 2, 3};
    CHECK(us.size() == 3u);
}

// ---- algorithm / numeric ------------------------------------------
void test_algorithm() {
    cardinal::vector<int> v{4, 8, 15, 16, 23, 42};
    CHECK(cardinal::accumulate(v.begin(), v.end(), 0) == 108);
    CHECK(*cardinal::max_element(v.begin(), v.end()) == 42);
    CHECK(*cardinal::min_element(v.begin(), v.end()) == 4);
    CHECK(cardinal::clamp(99, 0, 50) == 50);
    CHECK(cardinal::min(3, 7) == 3 && cardinal::max(3, 7) == 7);
    CHECK(cardinal::count_if(v.begin(), v.end(),
                             [](int x){ return x > 15; }) == 3);
    cardinal::vector<int> d(3);
    cardinal::iota(d.begin(), d.end(), 10);
    CHECK(d[0] == 10 && d[2] == 12);
    CHECK(cardinal::gcd(12, 18) == 6);
}

// ---- utility / tuple / optional / variant -------------------------
void test_utility() {
    auto pr = cardinal::make_pair(1, cardinal::string("a"));
    CHECK(pr.first == 1 && pr.second == "a");

    auto tp = cardinal::make_tuple(1, 2.5, 'z');
    CHECK(cardinal::get<0>(tp) == 1 && cardinal::get<2>(tp) == 'z');

    int x = 1, y = 2;
    cardinal::swap(x, y);
    CHECK(x == 2 && y == 1);
    CHECK(cardinal::exchange(x, 9) == 2 && x == 9);

    cardinal::optional<int> o;
    CHECK(!o.has_value());
    o = 5;
    CHECK(o.has_value() && *o == 5);

    cardinal::variant<int, cardinal::string> var = cardinal::string("hi");
    CHECK(cardinal::holds_alternative<cardinal::string>(var));
    CHECK(cardinal::get<cardinal::string>(var) == "hi");
}

// ---- traits -------------------------------------------------------
void test_traits() {
    CHECK((cardinal::is_same_v<cardinal::decay_t<const int&>, int>));
    CHECK(cardinal::is_integral_v<i64>);
    CHECK(!cardinal::is_floating_point_v<i64>);
    CHECK((cardinal::is_same_v<cardinal::conditional_t<true, int, char>, int>));
    CHECK(cardinal::is_trivially_copyable_v<u32>);
}

// ---- limits / cmath / bit -----------------------------------------
void test_numeric() {
    CHECK(cardinal::numeric_limits<cardinal::u8>::max() == 255u);
    CHECK(cardinal::numeric_limits<i64>::min() < 0);

    CHECK(cardinal::sqrt(144.0) == 12.0);
    CHECK(cardinal::floor(3.9) == 3.0);
    CHECK(cardinal::abs(-7) == 7);
    CHECK(cardinal::fabs(-2.5) == 2.5);
    CHECK(cardinal::isnan(cardinal::sqrt(-1.0)) ||
          cardinal::sqrt(-1.0) != cardinal::sqrt(-1.0));

    CHECK(cardinal::popcount(0xFFu) == 8);
    CHECK(cardinal::has_single_bit(64u));
    const u32 bits = cardinal::bit_cast<u32>(1.0f);
    CHECK(bits == 0x3F800000u);
}

// ---- cstring / charconv -------------------------------------------
void test_cstr_charconv() {
    char buf[8] = {};
    cardinal::memset(buf, 0, sizeof(buf));
    cardinal::memcpy(buf, "abc", 3);
    CHECK(cardinal::strlen(buf) == 3u);
    CHECK(cardinal::memcmp(buf, "abc", 3) == 0);

    char num[16];
    auto r = cardinal::to_chars(num, num + sizeof(num), 12345);
    CHECK(r.ec == cardinal::errc{});
    *r.ptr = '\0';
    CHECK(cardinal::strcmp(num, "12345") == 0);
    int parsed = 0;
    cardinal::from_chars(num, r.ptr, parsed);
    CHECK(parsed == 12345);
}

// ---- atomic / thread ----------------------------------------------
void test_concurrency() {
    cardinal::atomic<int> a{0};
    a.fetch_add(5, cardinal::memory_order_relaxed);
    CHECK(a.load() == 5);

    cardinal::mutex mtx;
    int guarded = 0;
    {
        cardinal::lock_guard<cardinal::mutex> lk(mtx);
        guarded = 1;
    }
    CHECK(guarded == 1);

    int from_thread = 0;
    cardinal::thread t([&]{ from_thread = 99; });
    t.join();
    CHECK(from_thread == 99);
}

// ---- cardinal::fseek64 / cardinal::ftell64 (64-bit file offsets) ---
// std::fseek / std::ftell use `long`, which is 32-bit on Windows LLP64
// and on 32-bit Linux without _FILE_OFFSET_BITS=64. Offsets above 2 GiB
// silently truncate (parked-supervised item #1 in feedback_integration_
// coverage_map.md — pack archives, the actual >2GiB consumer, now use
// the 64-bit wrappers). This test pins the API contract on a small
// file: the return type is i64, SEEK_END/SET/CUR all work, and round-
// tripping a position gives back exact bytes.
void test_cstdio_offsets_64bit() {
    namespace fs = cardinal::fs;
    auto tmp = fs::temp_directory_path() / "cardinal_fseek64_test.bin";
    {
        cardinal::FILE* fw = cardinal::fopen(tmp.string().c_str(), "wb");
        CHECK(fw != nullptr);
        if (!fw) return;
        // Write a tiny 16-byte signature so we can probe positions.
        const cardinal::u8 buf[16] = { 'C','A','R','D', 0,1,2,3,4,5,6,7,8,9,10,11 };
        cardinal::fwrite(buf, 1, 16, fw);
        cardinal::fclose(fw);
    }
    cardinal::FILE* f = cardinal::fopen(tmp.string().c_str(), "rb");
    CHECK(f != nullptr);
    if (!f) return;

    // SEEK_END then ftell64 = file size. Type is i64, NOT long.
    CHECK(cardinal::fseek64(f, 0, SEEK_END) == 0);
    cardinal::i64 sz = cardinal::ftell64(f);
    CHECK(sz == 16);
    // Trait: ftell64 must return i64 — if someone changes the return to
    // long, this static_assert breaks the build.
    static_assert(cardinal::is_same_v<decltype(cardinal::ftell64(f)),
                                      cardinal::i64>,
                  "cardinal::ftell64 must return i64");

    // SEEK_SET to a precise position, read 4 bytes, verify.
    CHECK(cardinal::fseek64(f, 4, SEEK_SET) == 0);
    CHECK(cardinal::ftell64(f) == 4);
    cardinal::u8 read[4] = {};
    cardinal::fread(read, 1, 4, f);
    CHECK(read[0] == 0 && read[1] == 1 && read[2] == 2 && read[3] == 3);

    // SEEK_CUR relative from position 8 (post-read = 8).
    CHECK(cardinal::ftell64(f) == 8);
    CHECK(cardinal::fseek64(f, 4, SEEK_CUR) == 0);
    CHECK(cardinal::ftell64(f) == 12);

    // Negative SEEK_CUR — only meaningful on the 64-bit signed offset
    // type. (std::fseek's `long` accepts negatives too, but the cast
    // chain in pack.cpp's offset arithmetic was unsigned u64 → long
    // which would have lost sign on >2 GiB values too. fseek64
    // accepts i64 directly so the contract is clean.)
    CHECK(cardinal::fseek64(f, -8, SEEK_CUR) == 0);
    CHECK(cardinal::ftell64(f) == 4);

    cardinal::fclose(f);
    std::error_code ec;
    fs::remove(tmp, ec);
}

// ---- chrono / filesystem ------------------------------------------
void test_chrono_fs() {
    const auto t0 = cardinal::chrono::steady_clock::now();
    const auto t1 = t0 + cardinal::chrono::milliseconds(5);
    const auto ms = cardinal::chrono::duration_cast<
        cardinal::chrono::milliseconds>(t1 - t0).count();
    CHECK(ms == 5);

    cardinal::fs::path p = "a/b/c.txt";
    CHECK(p.extension() == ".txt");
    CHECK(p.filename() == "c.txt");
    CHECK(p.parent_path() == cardinal::fs::path("a/b"));
}

}  // namespace

int main() {
    test_types();
    test_containers();
    test_algorithm();
    test_utility();
    test_traits();
    test_numeric();
    test_cstr_charconv();
    test_concurrency();
    test_cstdio_offsets_64bit();
    test_chrono_fs();

    if (g_fail == 0) {
        cardinal::log::infof("fstd", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("fstd", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
