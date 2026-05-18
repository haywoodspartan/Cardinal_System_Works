// =============================================================================
// Cardinal — deterministic CPK archive regression suite.
//
// The .cpk is the client-distribution container: a regression here
// ships corrupt or unreadable packs to players. Zero test deps (same
// harness as the other suites). Fully deterministic + headless —
// Builder/Archive are plain file I/O into a temp dir.
//
// Locks: Builder → Archive byte-exact round-trip (incl. all-256-byte
// payload, an embedded "CPK1" magic, empty + ~100 KB blobs), the
// on-disk FNV-1a64 hash, the contiguous header→payload→index layout,
// the last-wins duplicate-key rule, and the failure paths (missing
// file / too small / bad magic / missing key). Exit 0 = all pass.
//
// Out of scope by design: load_async() (needs the JobSystem + thread
// nondeterminism) and distribute() (needs a cooked project tree —
// belongs in a cook+pack integration arc).
// =============================================================================

#include <cardinal/pack/pack.hpp>
#include <cardinal/cook/cook.hpp>
#include <cardinal/core/log.hpp>

#include <filesystem>
#include <fstream>
#include <vector>

namespace {

namespace pk = cardinal::pack;
using cardinal::cook::AssetType;
using cardinal::u8;
using cardinal::u32;
using cardinal::u64;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("packtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

// The exact hash the archive stores per entry — recomputed here so a
// codec change can't quietly invalidate every shipped pack's integrity
// field without this test screaming.
u64 fnv1a64(const std::vector<u8>& d) {
    u64 h = 0xcbf29ce484222325ull;
    for (u8 b : d) { h ^= b; h *= 0x100000001b3ull; }
    return h;
}

// Deterministic pseudo-random blob (xorshift) — reproducible bytes.
std::vector<u8> blob(u32 seed, cardinal::usize n) {
    std::vector<u8> v;
    v.reserve(n);
    u32 x = seed ? seed : 1u;
    for (cardinal::usize i = 0; i < n; ++i) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        v.push_back(static_cast<u8>(x & 0xFFu));
    }
    return v;
}

void write_raw(const std::filesystem::path& p,
               const std::vector<u8>& bytes) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
}

}  // namespace

int main() {
    std::error_code ec;
    std::filesystem::path dir =
        std::filesystem::temp_directory_path(ec) / "cardinal_pack_test";
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        cardinal::log::infof("packtest",
            "SKIP  no writable temp dir in this environment");
        return 0;
    }

    // ---- payloads: adversarial on purpose ---------------------------
    std::vector<u8> all256(256);
    for (int i = 0; i < 256; ++i) all256[static_cast<cardinal::usize>(i)] =
        static_cast<u8>(i);
    // Payload whose CONTENT is the archive magic — offsets are explicit,
    // so the reader must not be fooled by data that looks like a header.
    const std::vector<u8> magicish = { 'C','P','K','1', 0,0,0,1, 0,0,0,0 };
    const std::vector<u8> empty;                 // size 0
    const std::vector<u8> big = blob(0xABCDu, 100000);

    const auto cpk = (dir / "main.cpk").string();

    // ---- build ------------------------------------------------------
    {
        pk::Builder b(cpk);
        CHECK(b.add("tex/all256",   AssetType::Texture, all256));
        CHECK(b.add("raw/magicish", AssetType::Unknown, magicish));
        CHECK(b.add("empty/zero",   AssetType::Unknown, empty));
        CHECK(b.add("big/blob",     AssetType::Mesh,    big));
        CHECK(b.entry_count() == 4u);
        cardinal::string err;
        CHECK(b.finalize(&err));
        CHECK(err.empty());
    }

    // ---- round-trip + integrity + layout ----------------------------
    {
        cardinal::string err;
        auto a = pk::Archive::open(cpk, &err);
        CHECK(a != nullptr);
        if (a) {
            CHECK(err.empty());
            CHECK(a->entry_count() == 4u);

            const pk::PackEntryDesc* e0 = a->find("tex/all256");
            const pk::PackEntryDesc* e1 = a->find("raw/magicish");
            const pk::PackEntryDesc* e2 = a->find("empty/zero");
            const pk::PackEntryDesc* e3 = a->find("big/blob");
            CHECK(e0 && e1 && e2 && e3);
            CHECK(a->contains("big/blob"));
            CHECK(!a->contains("not/here"));

            if (e0 && e1 && e2 && e3) {
                // Types survive.
                CHECK(e0->type == AssetType::Texture);
                CHECK(e1->type == AssetType::Unknown);
                CHECK(e3->type == AssetType::Mesh);
                // Sizes survive.
                CHECK(e0->size == sz(256));
                CHECK(e2->size == 0u);
                CHECK(e3->size == static_cast<u64>(big.size()));
                // On-disk integrity hash matches a fresh recompute.
                CHECK(e0->hash == fnv1a64(all256));
                CHECK(e1->hash == fnv1a64(magicish));
                CHECK(e3->hash == fnv1a64(big));
                // Layout: first payload sits right after the 32 B
                // header, the rest are contiguous in add() order.
                CHECK(e0->offset == 32u);
                CHECK(e1->offset == e0->offset + e0->size);
                CHECK(e2->offset == e1->offset + e1->size);
                CHECK(e3->offset == e2->offset + e2->size);
            }

            // Byte-exact payload fidelity.
            std::vector<u8> got;
            CHECK(a->load_blocking("tex/all256", got));
            CHECK(got == all256);
            CHECK(a->load_blocking("raw/magicish", got));
            CHECK(got == magicish);
            CHECK(a->load_blocking("big/blob", got));
            CHECK(got == big);
            CHECK(a->load_blocking("empty/zero", got));
            CHECK(got.empty());

            // Stats reflect the index.
            const auto st = a->stats();
            CHECK(st.entry_count == 4u);
            CHECK(st.file_size > 0u);
            CHECK(st.total_payload ==
                  static_cast<u64>(all256.size() + magicish.size() +
                                   empty.size() + big.size()));

            // Missing key → clean failure.
            std::vector<u8> none;
            cardinal::string lerr;
            CHECK(!a->load_blocking("missing", none, &lerr));
            CHECK(!lerr.empty());
        }
    }

    // ---- last-wins duplicate key ------------------------------------
    {
        const auto dpath = (dir / "dup.cpk").string();
        const std::vector<u8> x = { 1, 2, 3 };
        const std::vector<u8> y = { 9, 9, 9, 9, 9 };
        {
            pk::Builder b(dpath);
            CHECK(b.add("dup", AssetType::Mesh,    x));
            CHECK(b.add("dup", AssetType::Texture, y));
            CHECK(b.entry_count() == 1u);        // slot replaced
            CHECK(b.finalize());
        }
        auto a = pk::Archive::open(dpath);
        CHECK(a != nullptr);
        if (a) {
            CHECK(a->entry_count() == 1u);
            const auto* e = a->find("dup");
            CHECK(e != nullptr);
            if (e) {
                CHECK(e->type == AssetType::Texture);   // the later add
                CHECK(e->size == sz(5));
            }
            std::vector<u8> got;
            CHECK(a->load_blocking("dup", got));
            CHECK(got == y);                            // last wins
        }
    }

    // ---- failure paths ----------------------------------------------
    {
        cardinal::string err;
        CHECK(pk::Archive::open((dir / "no_such.cpk").string(), &err)
              == nullptr);
        CHECK(!err.empty());

        // Too small for the 32-byte header.
        err.clear();
        const auto tiny = (dir / "tiny.cpk").string();
        write_raw(tiny, std::vector<u8>{ 1, 2, 3, 4, 5 });
        CHECK(pk::Archive::open(tiny, &err) == nullptr);
        CHECK(!err.empty());

        // Big enough, wrong magic.
        err.clear();
        const auto badmagic = (dir / "bad.cpk").string();
        write_raw(badmagic, std::vector<u8>(40, 0xEE));
        CHECK(pk::Archive::open(badmagic, &err) == nullptr);
        CHECK(!err.empty());
    }

    std::error_code rec;
    const auto removed = std::filesystem::remove_all(dir, rec);
    (void)removed;

    if (g_fail == 0) {
        cardinal::log::infof("packtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("packtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
