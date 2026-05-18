// =============================================================================
// Cardinal — deterministic ChunkCoord / ChunkCoordHash / ChunkSet suite.
//
// The spatial-hash keystone: scene visible-chunk culling and the
// world-partition streamer both key containers on ChunkCoord via
// ChunkCoordHash. A hash or equality regression silently mis-buckets
// chunks — visible objects vanish (cull filter) or chunks never
// stream/evict — and stays invisible until it ships. The header notes
// it fixed an old u64-packing bug where same-(x,z)/different-y chunks
// COLLIDED; this suite pins that contract hard. Pure, header-only,
// deterministic. Exit 0 = all pass.
// =============================================================================

#include <cardinal/core/spatial.hpp>
#include <cardinal/core/log.hpp>

#include <type_traits>
#include <unordered_set>
#include <vector>

namespace {

namespace co = cardinal::core;
using co::ChunkCoord;
using co::ChunkCoordHash;
using co::ChunkSet;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("spatialtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

// ---- ChunkCoord equality (incl. the documented same-xz/diff-y bug) --
void test_chunkcoord() {
    const ChunkCoord d;
    CHECK(d.x == 0 && d.y == 0 && d.z == 0);            // default origin
    const ChunkCoord a{ 3, -7, 12 };
    CHECK(a.x == 3 && a.y == -7 && a.z == 12);

    CHECK(a == ChunkCoord(3, -7, 12));
    CHECK(a != ChunkCoord(3, -7, 13));                  // z matters
    CHECK(a != ChunkCoord(4, -7, 12));                  // x matters
    // THE regression: same x and z, different y → must be UNEQUAL.
    CHECK(a != ChunkCoord(3, -6, 12));
    CHECK(ChunkCoord(10, 0, 20) != ChunkCoord(10, 1, 20));
    CHECK(!(ChunkCoord(10, 0, 20) == ChunkCoord(10, 1, 20)));

    // constexpr-usable (compile-time contract).
    static_assert(ChunkCoord(1, 2, 3) == ChunkCoord(1, 2, 3),
                  "ChunkCoord::operator== must be constexpr");
    static_assert(ChunkCoord(1, 2, 3) != ChunkCoord(1, 9, 3),
                  "same-xz/diff-y must differ at compile time too");

    // Negatives compare correctly (two's-complement round-trip).
    CHECK(ChunkCoord(-1, -1, -1) == ChunkCoord(-1, -1, -1));
    CHECK(ChunkCoord(-1, 0, 0)   != ChunkCoord(1, 0, 0));
}

// ---- ChunkCoordHash: determinism + axis sensitivity ----------------
void test_hash() {
    ChunkCoordHash H;

    // Deterministic + consistent with equality (a==b ⇒ hash==hash).
    CHECK(H(ChunkCoord(7, 8, 9)) == H(ChunkCoord(7, 8, 9)));
    const ChunkCoord p{ -42, 17, 99 };
    const ChunkCoord q{ -42, 17, 99 };
    CHECK(p == q && H(p) == H(q));

    // Sensitive to EACH axis independently — a hash that drops an axis
    // would collapse these. The y case is the historical collision.
    const ChunkCoord base{ 10, 0, 20 };
    CHECK(H(base) != H(ChunkCoord(11, 0, 20)));          // x
    CHECK(H(base) != H(ChunkCoord(10, 1, 20)));          // y  (the bug)
    CHECK(H(base) != H(ChunkCoord(10, 0, 21)));          // z
    CHECK(H(ChunkCoord(10, 0, 20)) != H(ChunkCoord(10, 2, 20)));
    CHECK(H(ChunkCoord(10, 1, 20)) != H(ChunkCoord(10, 2, 20)));

    // Negatives hash distinctly from their positive mirror.
    CHECK(H(ChunkCoord(-5, 0, 0)) != H(ChunkCoord(5, 0, 0)));
    CHECK(H(ChunkCoord(0, -5, 0)) != H(ChunkCoord(0, 5, 0)));
    CHECK(H(ChunkCoord(0, 0, -5)) != H(ChunkCoord(0, 0, 5)));

    // Broad grid: 9³ = 729 distinct coords should map to (essentially)
    // 729 distinct hashes — a regression that ignores an axis collapses
    // this hard. Allow a 1% slack for legitimate 64-bit collisions.
    std::vector<cardinal::usize> hs;
    for (int x = -4; x <= 4; ++x)
        for (int y = -4; y <= 4; ++y)
            for (int z = -4; z <= 4; ++z)
                hs.push_back(H(ChunkCoord(x, y, z)));
    cardinal::usize uniq = 0;
    for (cardinal::usize i = 0; i < hs.size(); ++i) {
        bool dup = false;
        for (cardinal::usize j = 0; j < i; ++j)
            if (hs[j] == hs[i]) { dup = true; break; }
        if (!dup) ++uniq;
    }
    CHECK(hs.size() == sz(729));
    CHECK(uniq >= sz(722));                              // ≥ 99% unique
}

// ---- ChunkSet: the container the engine actually keys on -----------
void test_chunkset() {
    ChunkSet s;
    s.insert(ChunkCoord(1, 2, 3));
    s.insert(ChunkCoord(1, 2, 3));                       // dup → no-op
    CHECK(s.size() == sz(1));
    CHECK(s.count(ChunkCoord(1, 2, 3)) == sz(1));
    CHECK(s.count(ChunkCoord(1, 2, 4)) == sz(0));
    CHECK(s.erase(ChunkCoord(1, 2, 3)) == sz(1));
    CHECK(s.empty());

    // THE contract: a column of chunks sharing (x,z) but differing in y
    // must all be retained as DISTINCT keys (old u64 packing collapsed
    // them → world streamer would evict the wrong floor / cull culls a
    // whole vertical column).
    ChunkSet col;
    for (int y = -8; y <= 8; ++y) col.insert(ChunkCoord(5, y, 9));
    CHECK(col.size() == sz(17));
    for (int y = -8; y <= 8; ++y)
        CHECK(col.count(ChunkCoord(5, y, 9)) == sz(1));
    CHECK(col.count(ChunkCoord(5, 9, 9)) == sz(0));      // just outside
    CHECK(col.count(ChunkCoord(6, 0, 9)) == sz(0));      // diff x

    // Negative coords as keys round-trip.
    ChunkSet n;
    n.insert(ChunkCoord(-3, -4, -5));
    CHECK(n.count(ChunkCoord(-3, -4, -5)) == sz(1));
    CHECK(n.count(ChunkCoord(3, 4, 5)) == sz(0));

    // Deterministic membership: populate the same grid twice and via
    // two independent sets — identical contents + size both ways.
    auto build = [](ChunkSet& out) {
        for (int x = -3; x <= 3; ++x)
            for (int y = -2; y <= 2; ++y)
                for (int z = -3; z <= 3; ++z)
                    out.insert(ChunkCoord(x, y, z));
    };
    ChunkSet g1, g2;
    build(g1);
    build(g2);
    const cardinal::usize expect = sz(7 * 5 * 7);        // 245
    CHECK(g1.size() == expect);
    CHECK(g2.size() == expect);
    bool same = true;
    for (int x = -3; x <= 3; ++x)
        for (int y = -2; y <= 2; ++y)
            for (int z = -3; z <= 3; ++z) {
                const ChunkCoord c{ x, y, z };
                if (g1.count(c) != sz(1) || g2.count(c) != sz(1))
                    same = false;
            }
    CHECK(same);
    // A coord just outside the built range is absent in both.
    CHECK(g1.count(ChunkCoord(4, 0, 0)) == sz(0));
    CHECK(g2.count(ChunkCoord(0, 3, 0)) == sz(0));
}

// ---- Single-source aliasing invariant ------------------------------
void test_aliases() {
    // core::AABB / core::Ray must BE the geom types (the header's whole
    // point — zero-copy cross-module type-punning). Compile-time.
    static_assert(std::is_same_v<co::AABB, co::geom::AABB>,
                  "core::AABB must alias geom::AABB");
    static_assert(std::is_same_v<co::Ray, co::geom::Ray>,
                  "core::Ray must alias geom::Ray");
    CHECK(true);   // recorded so the suite count reflects this group
}

}  // namespace

int main() {
    test_chunkcoord();
    test_hash();
    test_chunkset();
    test_aliases();

    if (g_fail == 0) {
        cardinal::log::infof("spatialtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("spatialtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
