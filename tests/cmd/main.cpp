// =============================================================================
// Cardinal — cardinal::cmd unified command-bus tests (headless).
// =============================================================================
#include <cardinal/cmd/command.hpp>
#include <cardinal/scene/math.hpp>
#include <cardinal/actor/world.hpp>
#include <cardinal/core/log.hpp>
#include <cardinal/core/utility.hpp>   // cardinal::move

namespace cc = cardinal::cmd;
namespace sc = cardinal::scene;
using Mat4 = cardinal::core::Mat4;
using Vec3 = cardinal::core::Vec3;
using Vec4 = cardinal::core::Vec4;

static int g_checks = 0;
static int g_fail   = 0;
static void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) { ++g_fail; cardinal::log::errorf("cmdtest", "FAIL  L%d: %s", line, expr); }
}
#define CHECK(x) check_impl(static_cast<bool>(x), #x, __LINE__)

static bool ap(float a, float b, float eps = 1e-3f) {
    const float d = a - b;
    return (d < 0 ? -d : d) <= eps;
}

int main() {
    // ---- registry basics -------------------------------------------------
    cc::CommandRegistry reg;
    CHECK(reg.size() == 0u);
    CHECK(reg.find("nope") == nullptr);

    cc::Command noop;
    noop.id = "test.noop"; noop.label = "Noop";
    noop.run = [](cc::CommandContext&) { return cc::CommandResult{ true, {} }; };
    reg.add(cardinal::move(noop));
    CHECK(reg.size() == 1u);
    CHECK(reg.has("test.noop"));
    CHECK(reg.find("test.noop") != nullptr);

    cc::CommandContext ctx{};
    CHECK(reg.dispatch("test.noop", ctx).ok);
    CHECK(!reg.dispatch("does.not.exist", ctx).ok);   // unknown -> graceful, no throw

    // ---- builtins + sorted ids ------------------------------------------
    cc::register_builtin_commands(reg);
    CHECK(reg.has("world.place_asset"));
    auto ids = reg.ids();
    CHECK(ids.size() >= 2u);
    for (cardinal::usize i = 1; i < ids.size(); ++i) CHECK(ids[i - 1] < ids[i]);  // sorted

    // ---- screen_to_ground: invalid vp rejected --------------------------
    cc::ViewProj bad{};
    Vec3 h{};
    CHECK(!cc::screen_to_ground(bad, 0.0f, 0.0f, 0.0f, false, 0.0f, &h));

    // Camera looking straight down from (0,10,0). Centre NDC -> ground origin.
    cc::ViewProj vp;
    vp.view  = Mat4::look_at(Vec3{ 0, 10, 0 }, Vec3{ 0, 0, 0 }, Vec3{ 0, 0, -1 });
    vp.proj  = Mat4::perspective(sc::kPi / 3.0f, 16.0f / 9.0f, 0.1f, 100.0f);
    vp.valid = true;
    CHECK(cc::screen_to_ground(vp, 0.0f, 0.0f, 0.0f, false, 0.0f, &h));
    CHECK(ap(h.x, 0.0f, 1e-2f) && ap(h.z, 0.0f, 1e-2f) && ap(h.y, 0.0f, 1e-3f));

    // ---- THE FIX, pinned: project a known ground point with vp, then
    // unproject it back through the command helper with the SAME vp -> the hit
    // recovers the point (the captured-matrix path). A MISMATCHED projection
    // (the original offset bug) misses by a clear margin.
    const Vec3  P{ 2.0f, 0.0f, -3.0f };
    const Vec4  clip = vp.proj * (vp.view * Vec4{ P.x, P.y, P.z, 1.0f });
    const float nx = clip.x / clip.w;
    const float ny = clip.y / clip.w;
    CHECK(cc::screen_to_ground(vp, nx, ny, 0.0f, false, 0.0f, &h));
    CHECK(ap(h.x, 2.0f, 1e-2f) && ap(h.z, -3.0f, 1e-2f));         // matches -> on cursor

    cc::ViewProj wrong = vp;
    wrong.proj = Mat4::perspective(sc::kPi / 3.0f, 1.0f, 0.1f, 100.0f);  // wrong aspect
    Vec3 hw{};
    cc::screen_to_ground(wrong, nx, ny, 0.0f, false, 0.0f, &hw);
    CHECK(!(ap(hw.x, 2.0f, 1e-2f) && ap(hw.z, -3.0f, 1e-2f)));    // offset (the bug)

    // ---- snap ------------------------------------------------------------
    // Project a non-grid point, snap to 1.0 -> rounds to nearest unit.
    const Vec3  Q{ 2.4f, 0.0f, -3.6f };
    const Vec4  qc = vp.proj * (vp.view * Vec4{ Q.x, Q.y, Q.z, 1.0f });
    CHECK(cc::screen_to_ground(vp, qc.x / qc.w, qc.y / qc.w, 0.0f, true, 1.0f, &h));
    CHECK(ap(h.x, 2.0f, 1e-2f) && ap(h.z, -4.0f, 1e-2f));         // snapped

    // ---- world.place_asset dispatch fails gracefully without context ----
    cc::CommandContext pctx{};
    pctx.viewport = vp; pctx.active_asset_id = "x";
    CHECK(!reg.dispatch("world.place_asset", pctx).ok);           // no placement bound
    pctx.active_asset_id.clear();
    CHECK(!reg.dispatch("world.place_asset", pctx).ok);           // no asset

    // ---- actor / layout commands act on a real World, HEADLESS ----------
    namespace ac = cardinal::actor;
    ac::World w;
    auto* a = w.spawn("A"); a->get_component<ac::TransformComponent>()->translation = { 0, 1, 0 };
    auto* b = w.spawn("B"); b->get_component<ac::TransformComponent>()->translation = { 4, 5, 0 };
    auto* c = w.spawn("C"); c->get_component<ac::TransformComponent>()->translation = { 10, 9, 0 };
    cc::CommandContext wx{};
    wx.world = &w;
    wx.selection = { a->id(), b->id(), c->id() };

    // builtins include the new commands.
    CHECK(reg.has("actor.duplicate") && reg.has("actor.delete"));
    CHECK(reg.has("layout.align") && reg.has("layout.distribute") && reg.has("layout.snap"));

    // duplicate the 3 selected -> 3 new actors.
    const cardinal::usize before = w.actor_count();
    CHECK(reg.dispatch("actor.duplicate", wx).ok);
    CHECK(wx.result_count == 3u);
    CHECK(w.actor_count() == before + 3u);

    // align the original 3 to their Y-center (min 1, max 9 -> center 5).
    wx.axis = 1; wx.align_mode = 1;   // Y, Center
    CHECK(reg.dispatch("layout.align", wx).ok);
    CHECK(ap(a->get_component<ac::TransformComponent>()->translation.y, 5.0f, 1e-3f));
    CHECK(ap(c->get_component<ac::TransformComponent>()->translation.y, 5.0f, 1e-3f));

    // distribute along X (endpoints fixed at 0 and 10, middle -> 5).
    wx.axis = 0;
    CHECK(reg.dispatch("layout.distribute", wx).ok);
    CHECK(ap(b->get_component<ac::TransformComponent>()->translation.x, 5.0f, 1e-3f));

    // snap to a grid of 1.0 (already integer here -> unchanged, count == 3).
    wx.grid_step = 1.0f;
    CHECK(reg.dispatch("layout.snap", wx).ok);
    CHECK(wx.result_count == 3u);

    // delete the selection (deferred kill).
    CHECK(reg.dispatch("actor.delete", wx).ok);
    CHECK(wx.result_count == 3u);
    CHECK(!a->alive() && !b->alive() && !c->alive());

    // Graceful failures: no world, empty selection.
    cc::CommandContext empty{};
    CHECK(!reg.dispatch("actor.duplicate", empty).ok);   // no world
    cc::CommandContext nosel{}; nosel.world = &w;
    CHECK(!reg.dispatch("actor.duplicate", nosel).ok);   // empty selection

    // ---- world.validate / world.autofix --------------------------------
    CHECK(reg.has("world.validate") && reg.has("world.autofix"));
    ac::World vw;
    vw.spawn("Dup");
    vw.spawn("Dup");                       // duplicate name -> a validation issue
    cc::CommandContext vx{}; vx.world = &vw;
    CHECK(reg.dispatch("world.validate", vx).ok);
    CHECK(vx.result_count >= 1u);          // at least the shared-name issue
    CHECK(reg.dispatch("world.autofix", vx).ok);    // runs (result_count = #fixes)
    CHECK(!reg.dispatch("world.validate", empty).ok);   // no world -> graceful

    if (g_fail == 0) cardinal::log::infof("cmdtest", "OK  %d checks passed", g_checks);
    else             cardinal::log::errorf("cmdtest", "%d/%d checks FAILED", g_fail, g_checks);
    return g_fail == 0 ? 0 : 1;
}
