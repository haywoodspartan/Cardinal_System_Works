// =============================================================================
// Cardinal — deterministic scene chunk-visibility regression suite.
//
// assign_chunks() bins every entity into a world chunk via FLOOR
// division (correct for negative coords; naive truncation-toward-zero
// would mis-bin the entire negative half-world) and reports how many
// entities changed cell. is_entity_visible() then gates the renderer's
// streaming cull on the visible-chunk set AND the per-entity visible
// flag. A regression here silently culls visible objects (popping) or
// fails to cull off-region ones (wasted draws) — invisible until it
// ships. The cull-predicate path is pure CPU (no Mesh / rhi::Device),
// fully headless + deterministic; builds on the already-locked
// ChunkCoord/ChunkSet. Exit 0 = all pass.
// =============================================================================

#include <cardinal/scene/scene.hpp>
#include <cardinal/core/diag/log.hpp>

#include <limits>

namespace {

namespace sc = cardinal::scene;
using cardinal::i32;
using cardinal::u32;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("culltest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

// Add an entity at a world position; return its id (re-fetch by id —
// add_entity can reallocate entities_, so never hold the Entity&).
u32 add_at(sc::Scene& s, const char* name, float x, float y, float z) {
    const u32 id = s.add_entity(name).id;
    sc::Entity* e = s.find_by_id(id);
    e->transform.translation = cardinal::scene::Vec3{ x, y, z };
    return id;
}
bool chunk_is(sc::Scene& s, u32 id, i32 cx, i32 cy, i32 cz) {
    sc::Entity* e = s.find_by_id(id);
    return e && e->chunk_x == cx && e->chunk_y == cy && e->chunk_z == cz;
}

// ---- assign_chunks: floor binning + change count + edge cases -------
void test_assign_chunks() {
    sc::Scene s;
    // Chosen so the expected chunk is hand-verifiable at size 10:
    //   floor(0/10)=0  floor(25/10)=2  floor(-1/10)=-1
    //   floor(-25/10)=-3  floor(-11/10)=-2  floor(10/10)=1
    const u32 e1 = add_at(s, "E1",   0.0f,   0.0f,   0.0f);
    const u32 e2 = add_at(s, "E2",  25.0f,  -1.0f, -25.0f);
    const u32 e3 = add_at(s, "E3", -11.0f,  10.0f,   9.0f);

    // First pass: E1 already at default (0,0,0) → NOT counted; E2,E3 move.
    CHECK(s.assign_chunks(10.0f) == sz(2));
    CHECK(chunk_is(s, e1,  0,  0,  0));
    CHECK(chunk_is(s, e2,  2, -1, -3));     // floor(-25/10) = -3, not -2
    CHECK(chunk_is(s, e3, -2,  1,  0));     // floor(-11/10) = -2, not -1

    // Idempotent: same positions → zero changes.
    CHECK(s.assign_chunks(10.0f) == sz(0));

    // Move one entity → exactly one change; only it updates.
    s.find_by_id(e1)->transform.translation = cardinal::scene::Vec3{10.0f,0,0};
    CHECK(s.assign_chunks(10.0f) == sz(1));
    CHECK(chunk_is(s, e1, 1, 0, 0));
    CHECK(chunk_is(s, e2, 2, -1, -3));      // unchanged

    // Boundary exactness: pos == k*size → chunk k (not k-1).
    const u32 b = add_at(s, "B", 30.0f, 0.0f, 0.0f);   // 30/10 == 3.0
    CHECK(s.assign_chunks(10.0f) >= sz(1));
    CHECK(chunk_is(s, b, 3, 0, 0));
    s.find_by_id(b)->transform.translation =
        cardinal::scene::Vec3{ 29.999f, 0.0f, 0.0f };   // just under 3*10
    s.assign_chunks(10.0f);
    CHECK(chunk_is(s, b, 2, 0, 0));

    // chunk_size <= 1e-3 → no-op (returns 0, chunks untouched).
    const i32 px = s.find_by_id(e2)->chunk_x;
    CHECK(s.assign_chunks(0.0f) == sz(0));
    CHECK(s.assign_chunks(0.0005f) == sz(0));
    CHECK(s.find_by_id(e2)->chunk_x == px);

    // Non-finite translation is skipped (chunk stays, not counted).
    sc::Scene n;
    const u32 g  = add_at(n, "G", 100.0f, 0.0f, 0.0f);
    const u32 bad = add_at(n, "BAD",
        std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f);
    const cardinal::usize ch = n.assign_chunks(10.0f);
    CHECK(chunk_is(n, g, 10, 0, 0));        // finite one binned
    CHECK(chunk_is(n, bad, 0, 0, 0));       // NaN one left at default
    CHECK(ch == sz(1));                     // only the finite change counted
}

// ---- visible-chunk filter + is_entity_visible ----------------------
void test_visibility() {
    sc::Scene s;
    const u32 e1 = add_at(s, "E1",   0.0f, 0.0f,   0.0f);  // chunk (0,0,0)
    const u32 e2 = add_at(s, "E2",  25.0f, -1.0f, -25.0f); // (2,-1,-3)
    const u32 e3 = add_at(s, "E3", -11.0f, 10.0f,  9.0f);  // (-2,1,0)
    s.assign_chunks(10.0f);

    // No filter set → every *visible* entity passes; no chunk is "in".
    CHECK(!s.has_visible_chunk_filter());
    CHECK(s.visible_chunk_count() == sz(0));
    CHECK(!s.is_chunk_visible(0, 0, 0));
    CHECK(s.is_entity_visible(*s.find_by_id(e1)));
    CHECK(s.is_entity_visible(*s.find_by_id(e2)));
    // visible flag still gates even with no chunk filter.
    s.find_by_id(e1)->visible = false;
    CHECK(!s.is_entity_visible(*s.find_by_id(e1)));
    s.find_by_id(e1)->visible = true;

    // Mark E2's + E3's chunks visible; E1's stays culled.
    s.mark_chunk_visible(2, -1, -3);
    s.mark_chunk_visible(2, -1, -3);          // dedup → still 1
    CHECK(s.visible_chunk_count() == sz(1));
    s.mark_chunk_visible(-2, 1, 0);           // negative coords
    CHECK(s.visible_chunk_count() == sz(2));
    CHECK(s.has_visible_chunk_filter());
    CHECK(s.visible_chunks().size() == sz(2));
    CHECK(s.is_chunk_visible(2, -1, -3));
    CHECK(s.is_chunk_visible(-2, 1, 0));
    CHECK(!s.is_chunk_visible(0, 0, 0));      // E1's chunk not marked

    CHECK(s.is_entity_visible(*s.find_by_id(e2)));     // chunk in set
    CHECK(s.is_entity_visible(*s.find_by_id(e3)));
    CHECK(!s.is_entity_visible(*s.find_by_id(e1)));    // chunk NOT in set

    // visible flag overrides an in-set chunk.
    s.find_by_id(e2)->visible = false;
    CHECK(!s.is_entity_visible(*s.find_by_id(e2)));

    // Clearing the filter restores "all visible entities pass".
    s.clear_visible_chunks();
    CHECK(!s.has_visible_chunk_filter());
    CHECK(s.visible_chunk_count() == sz(0));
    CHECK(!s.is_chunk_visible(2, -1, -3));             // gone after clear
    CHECK(s.is_entity_visible(*s.find_by_id(e1)));     // no filter → passes
    CHECK(!s.is_entity_visible(*s.find_by_id(e2)));    // still flag=false
    s.find_by_id(e2)->visible = true;
    CHECK(s.is_entity_visible(*s.find_by_id(e2)));
}

}  // namespace

int main() {
    test_assign_chunks();
    test_visibility();

    if (g_fail == 0) {
        cardinal::log::infof("culltest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("culltest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
