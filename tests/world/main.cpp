// =============================================================================
// Cardinal — deterministic world grid + chunk-streamer regression suite.
//
// WorldGrid: pure config + math. ctor clamps (chunk_size<=0→1, extents
// <=0→1, render dist<0→0); setters clamp differently (chunk_size→max
// 1e-3, extents→max 1). chunk_of = floor(w * 1/chunk_size) — correct for
// negatives. chunk_world_min/max/center, half-open in_world_bounds
// (x∈[-ex,ex)), and the cylindrical compute_visible_set (dx²+dz²≤rxz²
// AND |dy|≤ry, world-bounds filtered, out cleared).
//
// WorldStreamer: diffs the visible set frame-over-frame, load-before-
// unload, with a camera-chunk no-move short-circuit (returns false, no
// callbacks), first-tick always recomputes, invalidate() forces a
// recompute, evict_all() unloads everything + re-arms first_tick.
//
// A regression silently breaks open-world streaming. Pure, single-
// threaded, fully deterministic. chunk_of is asserted only at MID-chunk
// coords so the 1/size float error can't flip a floor at a boundary.
// Exit 0 = all pass.
// =============================================================================

#include <cardinal/world/world.hpp>
#include <cardinal/core/diag/log.hpp>

#include <vector>

namespace {

namespace cw = cardinal::world;
using CC   = cw::ChunkCoord;
using Vec3 = cw::Vec3;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("worldtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-4f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }
bool cc_is(const CC& c, int x, int y, int z) {
    return c.x == x && c.y == y && c.z == z;
}
// double coords so plain integer literals widen without C4244.
bool vis(const Vec3& v, double x, double y, double z) {
    return ap(v.x, static_cast<float>(x)) &&
           ap(v.y, static_cast<float>(y)) &&
           ap(v.z, static_cast<float>(z));
}
bool has(const std::vector<CC>& v, int x, int y, int z) {
    for (const auto& c : v) if (cc_is(c, x, y, z)) return true;
    return false;
}

// ---- WorldGrid config + clamps ------------------------------------
void test_grid_config() {
    {   // ctor clamps: chunk_size<=0→1, extents<=0→1, render dist<0→0.
        cw::WorldGridDesc d;
        d.chunk_size = 0.0f; d.extent_x = 0; d.extent_y = -3;
        d.extent_z = -1; d.render_distance_xz = -2; d.render_distance_y = -5;
        cw::WorldGrid g(d);
        CHECK(ap(g.chunk_size(), 1.0f));
        CHECK(g.extent_x() == 1 && g.extent_y() == 1 && g.extent_z() == 1);
        CHECK(g.render_distance_xz() == 0 && g.render_distance_y() == 0);
    }
    {   // setters clamp: chunk_size→max(1e-3), extents→max(1), rd→max(0).
        cw::WorldGrid g;
        g.set_chunk_size(0.0f);   CHECK(ap(g.chunk_size(), 1.0e-3f, 1e-6f));
        g.set_chunk_size(-2.0f);  CHECK(ap(g.chunk_size(), 1.0e-3f, 1e-6f));
        g.set_chunk_size(5.0f);   CHECK(ap(g.chunk_size(), 5.0f));
        g.set_render_distance_xz(-1); CHECK(g.render_distance_xz() == 0);
        g.set_render_distance_xz(3);  CHECK(g.render_distance_xz() == 3);
        g.set_render_distance_y(-4);  CHECK(g.render_distance_y() == 0);
        g.set_render_distance_y(2);   CHECK(g.render_distance_y() == 2);
        g.set_extent_x(0);   CHECK(g.extent_x() == 1);
        g.set_extent_x(10);  CHECK(g.extent_x() == 10);
        g.set_extent_uniform(7);
        CHECK(g.extent_x()==7 && g.extent_y()==7 && g.extent_z()==7);
    }
    {   // footprint math.
        cw::WorldGridDesc d;
        d.chunk_size = 2.0f; d.extent_x = 3; d.extent_y = 4; d.extent_z = 5;
        cw::WorldGrid g(d);
        CHECK(ap(g.world_size_x_units(), 12.0f));         // 2*(3*2)
        CHECK(ap(g.world_size_y_units(), 16.0f));
        CHECK(ap(g.world_size_z_units(), 20.0f));
        CHECK(g.total_chunk_count() == static_cast<cardinal::i64>(480));
    }
}

// ---- chunk_of MUST NOT invoke UB on non-finite inputs -------------
// Same float→i32 UB-cast pattern as mesh_ops 439d2eb / tex_ops 3bc8360.
// Two ingresses: (a) NaN chunk_size flowing through the constructor's
// NaN-blind `chunk_size <= 0.0f` guard → inv = 1/NaN = NaN → floor(wx
// * NaN) = NaN → static_cast<i32>(NaN) UB. (b) NaN wx/wy/wz from a
// poisoned actor position → same UB even with chunk_size finite.
void test_grid_nonfinite() {
    volatile float z = 0.0f;
    const float qnan = z / z;
    const float inf  = 1.0f / z;

    {   // (a) Constructor must reject NaN chunk_size and fall back
        // to the documented 1.0 default.
        cw::WorldGridDesc d; d.chunk_size = qnan;
        cw::WorldGrid g(d);
        CHECK(ap(g.chunk_size(), 1.0f));
        // chunk_of must produce a defined ChunkCoord (no UB cast).
        auto c = g.chunk_of(0.0f, 0.0f, 0.0f);
        CHECK(c.x == 0 && c.y == 0 && c.z == 0);
    }
    {   // Same for +Inf / -Inf chunk_size.
        cw::WorldGridDesc d; d.chunk_size = inf;
        cw::WorldGrid g(d);
        CHECK(ap(g.chunk_size(), 1.0f));
    }
    {   // Same for -Inf — the original `<= 0.0f` ordered compare WOULD
        // catch -Inf (-Inf < 0 is true), but the isfinite guard now
        // routes it through the same fallback for uniform behaviour.
        cw::WorldGridDesc d; d.chunk_size = -inf;
        cw::WorldGrid g(d);
        CHECK(ap(g.chunk_size(), 1.0f));
    }
    {   // (b) NaN world coords with finite chunk_size — per-component
        // sanitize in chunk_of maps NaN → 0 BEFORE the floor/cast.
        cw::WorldGridDesc d; d.chunk_size = 10.0f;
        d.extent_x = 100; d.extent_y = 100; d.extent_z = 100;
        cw::WorldGrid g(d);
        // NaN x → 0/10=0; finite y/z stay. Result must be defined.
        auto c = g.chunk_of(qnan, 25.0f, 35.0f);
        CHECK(c.x == 0 && c.y == 2 && c.z == 3);
        // All three NaN → origin.
        auto o = g.chunk_of(qnan, qnan, qnan);
        CHECK(o.x == 0 && o.y == 0 && o.z == 0);
        // ±Inf x → 0 (same sanitization). The other components stay.
        auto p = g.chunk_of( inf, 25.0f, 35.0f);
        CHECK(p.x == 0 && p.y == 2 && p.z == 3);
        auto q = g.chunk_of(-inf, 25.0f, 35.0f);
        CHECK(q.x == 0 && q.y == 2 && q.z == 3);
    }
}

// ---- chunk_of (floor, neg-correct) + world min/max/center ---------
void test_chunk_math() {
    cw::WorldGridDesc d; d.chunk_size = 10.0f;
    d.extent_x = 100000; d.extent_y = 100000; d.extent_z = 100000;
    cw::WorldGrid g(d);

    CHECK(cc_is(g.chunk_of(0.0f,  0.0f,  0.0f),  0,  0,  0));
    CHECK(cc_is(g.chunk_of(5.0f,  5.0f,  5.0f),  0,  0,  0));
    CHECK(cc_is(g.chunk_of(15.0f, 25.0f, 35.0f), 1,  2,  3));
    CHECK(cc_is(g.chunk_of(-5.0f, -5.0f, -5.0f), -1, -1, -1));   // floor < 0
    CHECK(cc_is(g.chunk_of(-15.0f,-25.0f,-35.0f),-2, -3, -4));
    CHECK(cc_is(g.chunk_of(95.0f, 5.0f,  5.0f),  9,  0,  0));
    CHECK(cc_is(g.chunk_of(-95.0f,5.0f,  5.0f), -10, 0,  0));

    CHECK(vis(g.chunk_world_min(CC{0,0,0}),    0,  0,  0));
    CHECK(vis(g.chunk_world_max(CC{0,0,0}),   10, 10, 10));
    CHECK(vis(g.chunk_world_center(CC{0,0,0}), 5,  5,  5));
    CHECK(vis(g.chunk_world_min(CC{2,-1,3}),  20,-10, 30));
    CHECK(vis(g.chunk_world_max(CC{2,-1,3}),  30,  0, 40));
    CHECK(vis(g.chunk_world_center(CC{2,-1,3}),25, -5, 35));
    CHECK(vis(g.chunk_world_min(CC{-1,-2,-3}),-10,-20,-30));
    CHECK(vis(g.chunk_world_center(CC{-1,-2,-3}),-5,-15,-25));
}

// ---- in_world_bounds: half-open [-extent, +extent) ----------------
void test_in_bounds() {
    cw::WorldGridDesc d;
    d.extent_x = 4; d.extent_y = 2; d.extent_z = 4;
    cw::WorldGrid g(d);
    CHECK(g.in_world_bounds(CC{0,0,0}));
    CHECK(g.in_world_bounds(CC{3,1,3}));
    CHECK(g.in_world_bounds(CC{-4,-2,-4}));               // lower edge IN
    CHECK(!g.in_world_bounds(CC{4,0,0}));                 // upper edge OUT
    CHECK(!g.in_world_bounds(CC{-5,0,0}));
    CHECK(!g.in_world_bounds(CC{0,2,0}));                 // y upper OUT
    CHECK(!g.in_world_bounds(CC{0,-3,0}));
    CHECK(!g.in_world_bounds(CC{0,0,4}));
    CHECK(!g.in_world_bounds(CC{0,0,-5}));
}

// ---- compute_visible_set: cylinder + bounds filter ----------------
void test_visible_set() {
    std::vector<CC> out;
    {   // radius 0 → just the camera chunk.
        cw::WorldGridDesc d; d.render_distance_xz = 0; d.render_distance_y = 0;
        cw::WorldGrid g(d);
        g.compute_visible_set(CC{5,2,3}, out);
        CHECK(out.size() == sz(1) && has(out, 5, 2, 3));
    }
    {   // rxz 1, ry 0 → 5-cell disc (corners excluded: 1²+1²>1²).
        cw::WorldGridDesc d; d.render_distance_xz = 1; d.render_distance_y = 0;
        cw::WorldGrid g(d);
        g.compute_visible_set(CC{0,0,0}, out);
        CHECK(out.size() == sz(5));
        CHECK(has(out,0,0,0) && has(out,1,0,0) && has(out,-1,0,0));
        CHECK(has(out,0,0,1) && has(out,0,0,-1));
        CHECK(!has(out,1,0,1) && !has(out,-1,0,-1));
    }
    {   // rxz 2, ry 1 → 13-cell disc × 3 dy layers = 39.
        cw::WorldGridDesc d; d.render_distance_xz = 2; d.render_distance_y = 1;
        cw::WorldGrid g(d);
        g.compute_visible_set(CC{0,0,0}, out);
        CHECK(out.size() == sz(39));
        CHECK(has(out,0,0,0) && has(out,2,0,0) && has(out,0,0,2));
        CHECK(has(out,0,1,0) && has(out,0,-1,0));
        CHECK(!has(out,0,2,0));                            // |dy|>ry
        CHECK(!has(out,2,0,1) && !has(out,2,0,2));         // dx²+dz²>4
    }
    {   // World-bounds filter: extent_x 1 ⇒ x∈{-1,0} only.
        cw::WorldGridDesc d;
        d.render_distance_xz = 2; d.render_distance_y = 0;
        d.extent_x = 1; d.extent_y = 100; d.extent_z = 100;
        cw::WorldGrid g(d);
        g.compute_visible_set(CC{0,0,0}, out);
        CHECK(out.size() == sz(8));
        CHECK(has(out,0,0,0) && has(out,-1,0,0));
        CHECK(!has(out,1,0,0) && !has(out,2,0,0));         // x≥extent_x=1
    }
}

// ---- WorldStreamer: diff / no-move / invalidate / evict -----------
void test_streamer() {
    cw::WorldGridDesc d;
    d.chunk_size = 10.0f;
    d.extent_x = 100000; d.extent_y = 100000; d.extent_z = 100000;
    d.render_distance_xz = 1; d.render_distance_y = 0;
    cw::WorldGrid g(d);
    cw::WorldStreamer s(g);
    std::vector<CC> loaded, unloaded;
    s.set_on_load  ([&](CC c){ loaded.push_back(c); });
    s.set_on_unload([&](CC c){ unloaded.push_back(c); });

    // First tick → chunk (0,0,0); loads the whole 5-cell disc.
    CHECK(s.tick(5.0f, 5.0f, 5.0f) == true);
    CHECK(cc_is(s.camera_chunk(), 0, 0, 0));
    CHECK(s.active_count() == sz(5));
    CHECK(loaded.size() == sz(5));
    CHECK(has(loaded,0,0,0) && has(loaded,1,0,0) && has(loaded,-1,0,0));
    CHECK(has(loaded,0,0,1) && has(loaded,0,0,-1));
    CHECK(unloaded.empty());
    CHECK(s.is_loaded(CC{0,0,0}) && !s.is_loaded(CC{9,9,9}));

    // Same chunk next frame → short-circuit (false, no callbacks).
    loaded.clear(); unloaded.clear();
    CHECK(s.tick(6.0f, 4.0f, 5.0f) == false);
    CHECK(loaded.empty() && unloaded.empty());
    CHECK(s.active_count() == sz(5));

    // Move to chunk (1,0,0) → diff: 3 load, 3 unload.
    loaded.clear(); unloaded.clear();
    CHECK(s.tick(15.0f, 5.0f, 5.0f) == true);
    CHECK(cc_is(s.camera_chunk(), 1, 0, 0));
    CHECK(s.active_count() == sz(5));
    CHECK(loaded.size() == sz(3));
    CHECK(has(loaded,2,0,0) && has(loaded,1,0,1) && has(loaded,1,0,-1));
    CHECK(unloaded.size() == sz(3));
    CHECK(has(unloaded,-1,0,0) && has(unloaded,0,0,1) && has(unloaded,0,0,-1));
    CHECK(s.is_loaded(CC{2,0,0}) && !s.is_loaded(CC{-1,0,0}));

    // invalidate() forces recompute even at the same chunk — but the
    // set is identical, so no spurious load/unload.
    loaded.clear(); unloaded.clear();
    s.invalidate();
    CHECK(s.tick(15.0f, 5.0f, 5.0f) == true);
    CHECK(loaded.empty() && unloaded.empty());
    CHECK(s.active_count() == sz(5));

    // evict_all() unloads everything and re-arms first_tick.
    loaded.clear(); unloaded.clear();
    s.evict_all();
    CHECK(unloaded.size() == sz(5));
    CHECK(s.active_count() == sz(0));
    CHECK(!s.is_loaded(CC{1,0,0}));
    loaded.clear(); unloaded.clear();
    CHECK(s.tick(15.0f, 5.0f, 5.0f) == true);              // first_tick again
    CHECK(loaded.size() == sz(5));
    CHECK(s.active_count() == sz(5));
}

// ---- streamer reacts to a render-distance change via invalidate ---
void test_streamer_radius_change() {
    cw::WorldGridDesc d;
    d.chunk_size = 10.0f;
    d.extent_x = 100000; d.extent_y = 100000; d.extent_z = 100000;
    d.render_distance_xz = 0; d.render_distance_y = 0;
    cw::WorldGrid g(d);
    cw::WorldStreamer s(g);
    std::vector<CC> loaded, unloaded;
    s.set_on_load  ([&](CC c){ loaded.push_back(c); });
    s.set_on_unload([&](CC c){ unloaded.push_back(c); });

    CHECK(s.tick(5.0f, 5.0f, 5.0f) == true);
    CHECK(loaded.size() == sz(1) && has(loaded,0,0,0));
    CHECK(s.active_count() == sz(1));

    g.set_render_distance_xz(1);
    s.invalidate();
    loaded.clear(); unloaded.clear();
    CHECK(s.tick(5.0f, 5.0f, 5.0f) == true);               // same chunk, dirty
    CHECK(loaded.size() == sz(4));                          // ring grows by 4
    CHECK(has(loaded,1,0,0) && has(loaded,-1,0,0));
    CHECK(has(loaded,0,0,1) && has(loaded,0,0,-1));
    CHECK(!has(loaded,0,0,0));                              // already active
    CHECK(unloaded.empty());
    CHECK(s.active_count() == sz(5));
}

}  // namespace

int main() {
    test_grid_config();
    test_grid_nonfinite();
    test_chunk_math();
    test_in_bounds();
    test_visible_set();
    test_streamer();
    test_streamer_radius_change();

    if (g_fail == 0) {
        cardinal::log::infof("worldtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("worldtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
