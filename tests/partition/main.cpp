// =============================================================================
// Cardinal — deterministic world-partition regression suite.
//
// WorldPartition::tick is the cell-streaming brain: union every active
// viewer's needs, apply load/unload HYSTERESIS (a not-loaded cell uses
// load_radius; a loaded one uses the larger unload_radius, so cells don't
// churn at the seam), keep Always cells resident, and enforce a soft
// resident cap by evicting the lowest-priority non-Always cells. A
// regression silently breaks streaming: holes (never load), VRAM blowup
// (never evict), or boundary thrash. The decision path is pure + sync
// (no threads in tick). Frustum/Vision coupling is owned by the geom
// suite — only the no-viewer and distance-triggered DistanceOrVision
// paths (no frustum semantics needed) are pinned here. Point-AABBs make
// every closest distance an exact perfect square. Exit 0 = all pass.
// =============================================================================

#include <cardinal/partition/partition.hpp>
#include <cardinal/core/log.hpp>

#include <string>
#include <vector>

namespace {

namespace pn   = cardinal::partition;
namespace geom = cardinal::core::geom;
using Vec3     = cardinal::scene::Vec3;
using cardinal::u32;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("parttest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-3f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }
bool streq(const char* a, const char* b) {
    return std::string(a) == b;
}
bool has(const std::vector<pn::CellId>& v, pn::CellId id) {
    for (auto x : v) if (x == id) return true;
    return false;
}

// Coordinate params are double so plain integer literals at call sites
// (point_box(0,0,0)) widen without C4244; we narrow once, explicitly.
inline float f(double v) { return static_cast<float>(v); }

// A point-AABB at (cx,cy,cz): closest_dist from p == |p - (cx,cy,cz)|.
geom::AABB point_box(double cx, double cy, double cz) {
    geom::AABB b;
    b.min = Vec3{f(cx), f(cy), f(cz)};
    b.max = Vec3{f(cx), f(cy), f(cz)};
    return b;
}
pn::CellDesc dist_cell(double load_r, double unload_r, double cx = 0.0,
                       double cy = 0.0, double cz = 0.0, u32 prio = 0u) {
    pn::CellDesc d;
    d.name = "c";
    d.bounds = point_box(cx, cy, cz);
    d.mode = pn::StreamMode::Distance;
    d.load_radius = f(load_r);
    d.unload_radius = f(unload_r);
    d.priority = prio;
    return d;
}
pn::Viewer viewer_at(double x, double y, double z, bool active = true) {
    pn::Viewer v;
    v.position = Vec3{f(x), f(y), f(z)};
    v.active = active;
    return v;
}

struct Rec {
    std::vector<pn::CellId> loaded;
    std::vector<pn::CellId> unloaded;
};
void wire(pn::WorldPartition& W, Rec& r) {
    W.set_on_load  ([&r](pn::CellId id, const pn::CellDesc&) { r.loaded.push_back(id); });
    W.set_on_unload([&r](pn::CellId id, const pn::CellDesc&) { r.unloaded.push_back(id); });
}

// ---- enum name tables ---------------------------------------------
void test_names() {
    using SM = pn::StreamMode;
    using CS = pn::CellState;
    CHECK(streq(pn::stream_mode_name(SM::Always),           "Always"));
    CHECK(streq(pn::stream_mode_name(SM::Distance),         "Distance"));
    CHECK(streq(pn::stream_mode_name(SM::Vision),           "Vision"));
    CHECK(streq(pn::stream_mode_name(SM::DistanceOrVision), "Dist+Vision"));
    CHECK(streq(pn::stream_mode_name(static_cast<SM>(99u)), "?"));
    CHECK(streq(pn::cell_state_name(CS::Unloaded),  "Unloaded"));
    CHECK(streq(pn::cell_state_name(CS::Loading),   "Loading"));
    CHECK(streq(pn::cell_state_name(CS::Loaded),    "Loaded"));
    CHECK(streq(pn::cell_state_name(CS::Unloading), "Unloading"));
    CHECK(streq(pn::cell_state_name(static_cast<CS>(99u)), "?"));
    CHECK(pn::kInvalidCellId == 0u);
}

// ---- add_cell id allocation + radius normalisation ----------------
void test_add_cell() {
    pn::WorldPartitionDesc desc;            // cap 32, default load 64 / unload 96
    auto wp = pn::WorldPartition::create(desc);
    auto& W = *wp;

    // Monotonic ids from 1.
    pn::CellId a = W.add_cell(dist_cell(10.0f, 20.0f));
    pn::CellId b = W.add_cell(dist_cell(10.0f, 20.0f));
    CHECK(a == 1u && b == 2u);
    CHECK(W.cell_count() == sz(2));
    CHECK(W.describe(a) != nullptr);
    CHECK(W.describe(9999u) == nullptr);
    CHECK(W.state(a) == pn::CellState::Unloaded);
    CHECK(W.state(9999u) == pn::CellState::Unloaded);

    // load_radius <= 0 → falls back to the partition default.
    pn::CellDesc z = dist_cell(0.0f, 0.0f);
    pn::CellId zc = W.add_cell(z);
    const pn::CellDesc* zd = W.describe(zc);
    CHECK(zd != nullptr);
    CHECK(ap(zd->load_radius, 64.0f));               // default_load_radius
    // unload_radius <= load_radius → load_radius * 1.5.
    CHECK(ap(zd->unload_radius, 96.0f));             // 64 * 1.5

    // Explicit unload <= load is also bumped to load*1.5.
    pn::CellId c = W.add_cell(dist_cell(40.0f, 30.0f));
    const pn::CellDesc* cd = W.describe(c);
    CHECK(cd != nullptr && ap(cd->load_radius, 40.0f));
    CHECK(ap(cd->unload_radius, 60.0f));             // 40 * 1.5

    // Explicit unload > load is preserved.
    pn::CellId d = W.add_cell(dist_cell(40.0f, 100.0f));
    CHECK(ap(W.describe(d)->unload_radius, 100.0f));
}

// ---- remove_cell / clear_cells fire on_unload (NOT the counter) ---
void test_remove_clear() {
    auto wp = pn::WorldPartition::create();
    auto& W = *wp; Rec r; wire(W, r);

    pn::CellId a = W.add_cell(dist_cell(10.0f, 20.0f));
    pn::CellId b = W.add_cell(dist_cell(10.0f, 20.0f));
    W.force_load(a);                                 // a Loaded, total_loaded 1
    CHECK(W.stats().cells_loaded_total == static_cast<cardinal::u64>(1));

    CHECK(W.remove_cell(9999u) == false);            // unknown
    CHECK(W.remove_cell(a) == true);                 // Loaded → fires on_unload
    CHECK(has(r.unloaded, a));
    CHECK(W.cell_count() == sz(1));
    // remove_cell fires the callback but does NOT bump the counter.
    CHECK(W.stats().cells_unloaded_total == static_cast<cardinal::u64>(0));

    W.force_load(b);
    Rec r2; wire(W, r2);
    W.clear_cells();                                 // fires on_unload for b
    CHECK(has(r2.unloaded, b));
    CHECK(W.cell_count() == sz(0));
    CHECK(W.stats().cells_unloaded_total == static_cast<cardinal::u64>(0));
}

// ---- viewers: monotonic ids, count, remove/update -----------------
void test_viewers() {
    auto wp = pn::WorldPartition::create();
    auto& W = *wp;
    u32 v1 = W.add_viewer(viewer_at(0,0,0));
    u32 v2 = W.add_viewer(viewer_at(1,0,0));
    CHECK(v1 == 1u && v2 == 2u);
    CHECK(W.viewer_count() == sz(2));
    W.remove_viewer(v1);
    CHECK(W.viewer_count() == sz(1));
    W.update_viewer(v2, viewer_at(5,0,0));           // no crash / count stable
    CHECK(W.viewer_count() == sz(1));
    W.remove_viewer(9999u);                          // unknown → no-op
    CHECK(W.viewer_count() == sz(1));
}

// ---- distance streaming with load/unload hysteresis ---------------
void test_distance_hysteresis() {
    auto wp = pn::WorldPartition::create();
    auto& W = *wp; Rec r; wire(W, r);

    pn::CellId c = W.add_cell(dist_cell(10.0f, 20.0f));   // point box @ origin
    u32 v = W.add_viewer(viewer_at(5,0,0));               // dist 5

    // 5 <= load_radius 10 → loads.
    W.tick();
    CHECK(W.state(c) == pn::CellState::Loaded);
    CHECK(r.loaded.size() == sz(1) && r.loaded[0] == c);
    CHECK(W.stats().cells_loaded_total == static_cast<cardinal::u64>(1));

    // Already Loaded, so the (larger) unload_radius gates it: 10 <= 20.
    W.update_viewer(v, viewer_at(10,0,0));
    W.tick();
    CHECK(W.state(c) == pn::CellState::Loaded);

    // Hysteresis: between load(10) and unload(20) a LOADED cell stays.
    W.update_viewer(v, viewer_at(15,0,0));
    W.tick();
    CHECK(W.state(c) == pn::CellState::Loaded);            // 15 <= unload 20
    CHECK(r.unloaded.empty());

    // Past unload_radius → unloads.
    W.update_viewer(v, viewer_at(25,0,0));
    W.tick();
    CHECK(W.state(c) == pn::CellState::Unloaded);
    CHECK(r.unloaded.size() == sz(1) && r.unloaded[0] == c);
    CHECK(W.stats().cells_unloaded_total == static_cast<cardinal::u64>(1));

    // Hysteresis the other way: within unload but outside load → stays
    // Unloaded (must re-enter load_radius to come back).
    W.update_viewer(v, viewer_at(15,0,0));
    W.tick();
    CHECK(W.state(c) == pn::CellState::Unloaded);          // 15 > load 10

    // Back inside load_radius → reloads.
    W.update_viewer(v, viewer_at(8,0,0));
    W.tick();
    CHECK(W.state(c) == pn::CellState::Loaded);
    CHECK(W.stats().cells_loaded_total == static_cast<cardinal::u64>(2));
}

// ---- Always / no-viewers / inactive / vision-no-viewer / DV -------
void test_modes_and_gating() {
    {   // Always: loads with ZERO viewers and never distance-unloads.
        auto wp = pn::WorldPartition::create(); auto& W = *wp;
        pn::CellDesc d; d.name = "sky"; d.mode = pn::StreamMode::Always;
        d.bounds = point_box(0,0,0);
        pn::CellId c = W.add_cell(d);
        W.tick();
        CHECK(W.state(c) == pn::CellState::Loaded);
        W.tick();
        CHECK(W.state(c) == pn::CellState::Loaded);        // persists
    }
    {   // Distance with no viewers → never loads.
        auto wp = pn::WorldPartition::create(); auto& W = *wp;
        pn::CellId c = W.add_cell(dist_cell(50.0f, 80.0f));
        W.tick();
        CHECK(W.state(c) == pn::CellState::Unloaded);
    }
    {   // Inactive viewer is skipped; activating it loads the cell.
        auto wp = pn::WorldPartition::create(); auto& W = *wp;
        pn::CellId c = W.add_cell(dist_cell(50.0f, 80.0f));
        u32 v = W.add_viewer(viewer_at(5,0,0,/*active=*/false));
        W.tick();
        CHECK(W.state(c) == pn::CellState::Unloaded);       // gated off
        W.update_viewer(v, viewer_at(5,0,0,/*active=*/true));
        W.tick();
        CHECK(W.state(c) == pn::CellState::Loaded);
    }
    {   // Vision mode with no viewers → never loads (frustum not consulted).
        auto wp = pn::WorldPartition::create(); auto& W = *wp;
        pn::CellDesc d; d.name = "v"; d.mode = pn::StreamMode::Vision;
        d.bounds = point_box(0,0,0);
        pn::CellId c = W.add_cell(d);
        W.tick();
        CHECK(W.state(c) == pn::CellState::Unloaded);
    }
    {   // DistanceOrVision triggered purely by distance (frustum-agnostic).
        auto wp = pn::WorldPartition::create(); auto& W = *wp;
        pn::CellDesc d; d.name = "dv";
        d.mode = pn::StreamMode::DistanceOrVision;
        d.bounds = point_box(0,0,0);
        d.load_radius = 10.0f; d.unload_radius = 20.0f;
        pn::CellId c = W.add_cell(d);
        W.add_viewer(viewer_at(5,0,0));
        W.tick();
        CHECK(W.state(c) == pn::CellState::Loaded);          // via distance
    }
}

// ---- multi-viewer union -------------------------------------------
void test_multi_viewer_union() {
    auto wp = pn::WorldPartition::create();
    auto& W = *wp;
    pn::CellId c = W.add_cell(dist_cell(10.0f, 20.0f));
    // NB: `near` / `far` are legacy Win32 macros — never name locals that.
    u32 v_far  = W.add_viewer(viewer_at(50,0,0));   // out of range
    u32 v_near = W.add_viewer(viewer_at(5,0,0));    // in range
    W.tick();
    CHECK(W.state(c) == pn::CellState::Loaded);   // any viewer in range loads

    // Drop the near viewer → only the far one remains → unloads (50>20).
    W.remove_viewer(v_near);
    W.update_viewer(v_far, viewer_at(50,0,0));
    W.tick();
    CHECK(W.state(c) == pn::CellState::Unloaded);
}

// ---- soft cap + lowest-priority eviction + Always immunity --------
void test_cap_eviction() {
    {
        pn::WorldPartitionDesc desc; desc.max_resident_cells = 2;
        auto wp = pn::WorldPartition::create(desc);
        auto& W = *wp; Rec r; wire(W, r);

        // 3 cells all in range of one viewer; distinct priorities.
        pn::CellId A = W.add_cell(dist_cell(1000.f, 2000.f, 0,0,0, /*prio*/10u));
        pn::CellId B = W.add_cell(dist_cell(1000.f, 2000.f, 0,0,0, /*prio*/ 5u));
        pn::CellId C = W.add_cell(dist_cell(1000.f, 2000.f, 0,0,0, /*prio*/ 1u));
        W.add_viewer(viewer_at(0,0,0));

        // Tick 1: cap can't pre-empt a fresh load (nothing evictable yet)
        // → all three transiently resident.
        W.tick();
        CHECK(W.stats().loaded == 3u);

        // Tick 2: over cap, evict the LOWEST priority (C).
        W.tick();
        CHECK(W.stats().loaded == 2u);
        CHECK(W.state(C) == pn::CellState::Unloaded);     // prio 1 — evicted
        CHECK(W.state(A) == pn::CellState::Loaded);        // prio 10 — kept
        CHECK(W.state(B) == pn::CellState::Loaded);        // prio 5  — kept
        CHECK(W.stats().cells_unloaded_total == static_cast<cardinal::u64>(1));
        CHECK(has(r.unloaded, C) && !has(r.unloaded, A));
    }
    {   // Always cells are immune to cap eviction.
        pn::WorldPartitionDesc desc; desc.max_resident_cells = 1;
        auto wp = pn::WorldPartition::create(desc);
        auto& W = *wp;
        pn::CellDesc always; always.name = "sky";
        always.mode = pn::StreamMode::Always; always.bounds = point_box(0,0,0);
        pn::CellId X = W.add_cell(always);
        pn::CellId Y = W.add_cell(dist_cell(1000.f, 2000.f));
        W.add_viewer(viewer_at(0,0,0));
        W.tick();                                          // both resident
        CHECK(W.stats().loaded == 2u);
        W.tick();                                          // over cap → evict
        CHECK(W.state(X) == pn::CellState::Loaded);         // Always survives
        CHECK(W.state(Y) == pn::CellState::Unloaded);       // the only evictable
        CHECK(W.stats().loaded == 1u);
    }
}

// ---- force_load / force_unload no-op rules ------------------------
void test_force() {
    auto wp = pn::WorldPartition::create();
    auto& W = *wp; Rec r; wire(W, r);
    pn::CellId c = W.add_cell(dist_cell(10.0f, 20.0f));

    W.force_load(c);
    CHECK(W.state(c) == pn::CellState::Loaded);
    CHECK(has(r.loaded, c));
    CHECK(W.stats().cells_loaded_total == static_cast<cardinal::u64>(1));

    // Already loaded → no-op (no duplicate callback / counter bump).
    W.force_load(c);
    CHECK(W.stats().cells_loaded_total == static_cast<cardinal::u64>(1));
    CHECK(r.loaded.size() == sz(1));
    W.force_load(9999u);                              // unknown → no-op
    CHECK(W.stats().cells_loaded_total == static_cast<cardinal::u64>(1));

    W.force_unload(c);
    CHECK(W.state(c) == pn::CellState::Unloaded);
    CHECK(has(r.unloaded, c));
    CHECK(W.stats().cells_unloaded_total == static_cast<cardinal::u64>(1));
    // Not loaded → no-op.
    W.force_unload(c);
    CHECK(W.stats().cells_unloaded_total == static_cast<cardinal::u64>(1));
    W.force_unload(9999u);                            // unknown → no-op
    CHECK(W.stats().cells_unloaded_total == static_cast<cardinal::u64>(1));
}

// ---- stats / loaded_cells / describe_cells ------------------------
void test_stats_describe() {
    auto wp = pn::WorldPartition::create();
    auto& W = *wp;
    pn::CellId c1 = W.add_cell(dist_cell(10.0f, 20.0f,    0,0,0));
    pn::CellId c2 = W.add_cell(dist_cell(10.0f, 20.0f,  100,0,0));
    pn::CellId c3 = W.add_cell(dist_cell(10.0f, 20.0f, -100,0,0));

    // No viewers → describe distance is f32 "infinity" sentinel.
    {
        auto rows0 = W.describe_cells();
        CHECK(rows0.size() == sz(3));
        CHECK(rows0[0].closest_viewer_distance > 1.0e30f);
    }

    W.add_viewer(viewer_at(0,0,0));
    W.force_load(c1);

    pn::WorldPartitionStats s = W.stats();
    CHECK(s.cell_count == 3u);
    CHECK(s.loaded == 1u);
    CHECK(s.loading == 0u && s.unloading == 0u);       // tick has no in-betweens
    CHECK(s.cells_loaded_total == static_cast<cardinal::u64>(1));

    auto lc = W.loaded_cells();
    CHECK(lc.size() == sz(1) && lc[0] == c1);

    auto rows = W.describe_cells();
    CHECK(rows.size() == sz(3));
    // describe_cells is sorted by id ascending.
    CHECK(rows[0].id == c1 && rows[1].id == c2 && rows[2].id == c3);
    CHECK(rows[0].state == pn::CellState::Loaded);
    CHECK(rows[1].state == pn::CellState::Unloaded);
    CHECK(rows[0].desc != nullptr);
    // Exact closest distances (viewer at origin, point boxes).
    CHECK(ap(rows[0].closest_viewer_distance, 0.0f));
    CHECK(ap(rows[1].closest_viewer_distance, 100.0f));
    CHECK(ap(rows[2].closest_viewer_distance, 100.0f));
}

}  // namespace

int main() {
    test_names();
    test_add_cell();
    test_remove_clear();
    test_viewers();
    test_distance_hysteresis();
    test_modes_and_gating();
    test_multi_viewer_union();
    test_cap_eviction();
    test_force();
    test_stats_describe();

    if (g_fail == 0) {
        cardinal::log::infof("parttest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("parttest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
