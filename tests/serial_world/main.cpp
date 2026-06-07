// =============================================================================
// Cardinal — deterministic world save/load round-trip suite.
//
// "Actors are the primary game state" (serial.hpp) — so this is the
// highest-stakes save surface: a regression here silently eats player
// progress. Zero test deps (same harness as the other suites).
//
// It defines + registers a reflected test GameActor exercising EVERY
// PropertyKind (float/int/bool/vec3/color/string), round-trips a World
// through save_world → load_world into a *fresh* Game, and asserts the
// actor set + transforms + every property survive exactly. It is also
// the regression lock for the string-quote-accretion fix in serial.cpp
// (pre-fix, s_label came back as `"value"` and grew a quote layer every
// cycle).
//
// Headless + deterministic: SimWorld/Game/World are pure CPU (no
// rhi::Device). Fixture in a temp dir. Exit 0 = all pass.
// =============================================================================

#include <cardinal/serial/serial.hpp>
#include <cardinal/sim/sim.hpp>
#include <cardinal/game/game.hpp>
#include <cardinal/game/game_actor.hpp>
#include <cardinal/game/reflection.hpp>
#include <cardinal/actor/world.hpp>
#include <cardinal/actor/actor.hpp>
#include <cardinal/actor/component.hpp>
#include <cardinal/core/diag/log.hpp>

#include <filesystem>
#include <vector>

// ---- reflected test actor: one field per PropertyKind ---------------
class SerialTestActor : public cardinal::game::GameActor {
public:
    float                 f_rpm   {0.0f};
    int                   i_count {0};
    bool                  b_flag  {false};
    cardinal::scene::Vec3 v_axis  {0.0f, 0.0f, 0.0f};
    cardinal::scene::Vec3 c_tint  {0.0f, 0.0f, 0.0f};
    std::string           s_label;
};

CARDINAL_REGISTER_GAME_CLASS(SerialTestActor, "Test/Serial",
    PROP_FLOAT(f_rpm, 0.0f, 1000.0f, "rpm")
    PROP_INT(i_count, -100, 100, "count")
    PROP_BOOL(b_flag, "flag")
    PROP_VEC3(v_axis, "axis")
    PROP_COLOR(c_tint, "tint")
    PROP_STRING(s_label, "label"))

namespace {

namespace ser = cardinal::serial;
using cardinal::actor::Actor;
using cardinal::actor::TransformComponent;
using cardinal::game::GameActor;

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

bool approx(float a, float b, float eps) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= eps;
}

const SerialTestActor* as_test(const Actor& a) {
    const GameActor* g = a.get_component<GameActor>();
    if (g == nullptr) return nullptr;
    if (g->class_name() != "SerialTestActor") return nullptr;
    return static_cast<const SerialTestActor*>(g);
}

}  // namespace

int main() {
    std::error_code ec;
    std::filesystem::path dir =
        std::filesystem::temp_directory_path(ec) / "cardinal_world_test";
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        cardinal::log::infof("worldtest",
            "SKIP  no writable temp dir in this environment");
        return 0;
    }
    const auto path = (dir / "save.world").string();

    // ---- build a source world ---------------------------------------
    {
        cardinal::sim::SimWorld sim;
        cardinal::game::Game game(sim);

        Actor* alpha = game.spawn_class("SerialTestActor", "Alpha");
        Actor* beta  = game.spawn_class("SerialTestActor", "Beta");
        Actor* plain = game.world().spawn("Plain");   // transform-only
        CHECK(alpha != nullptr);
        CHECK(beta  != nullptr);
        CHECK(plain != nullptr);

        if (alpha) {
            auto* a = static_cast<SerialTestActor*>(
                alpha->get_component<GameActor>());
            a->f_rpm   = 42.5f;
            a->i_count = -7;
            a->b_flag  = true;
            a->v_axis  = {1.0f, 2.0f, 3.0f};
            a->c_tint  = {0.5f, 0.25f, 0.75f};
            a->s_label = "hello world";            // spaces + quote-fix
            auto* tr = alpha->get_component<TransformComponent>();
            tr->translation    = {1.0f, 2.0f, 3.0f};
            tr->rotation_euler = {0.1f, 0.2f, 0.3f};
            tr->scale          = {2.0f, 2.0f, 2.0f};
        }
        if (beta) {
            auto* b = static_cast<SerialTestActor*>(
                beta->get_component<GameActor>());
            b->f_rpm   = 99.0f;
            b->i_count = 13;
            b->b_flag  = false;
            b->v_axis  = {-1.0f, 0.0f, 4.5f};
            b->c_tint  = {0.1f, 0.2f, 0.3f};
            b->s_label = "beta=\"q\"";             // tricky inner chars
            auto* tr = beta->get_component<TransformComponent>();
            tr->translation = {-5.0f, 0.5f, 9.0f};
        }
        if (plain) {
            auto* tr = plain->get_component<TransformComponent>();
            tr->translation = {7.0f, 8.0f, 9.0f};
        }

        cardinal::string err;
        ser::SaveStats ss = ser::save_world(game.world(), path, &err);
        CHECK(err.empty());
        CHECK(ss.actors_written == 3u);
        CHECK(ss.properties_written == 12u);   // 2 actors x 6 props
        CHECK(ss.bytes_written > 0u);
    }

    // ---- load into a fresh world, with a junk actor to be replaced --
    {
        cardinal::sim::SimWorld sim;
        cardinal::game::Game game(sim);
        game.world().spawn("JunkShouldBeReplaced");

        cardinal::string err;
        ser::LoadStats ls = ser::load_world(game, path,
                                            /*replace_existing=*/true, &err);
        CHECK(err.empty());
        CHECK(ls.actors_spawned == 3u);
        CHECK(ls.properties_applied == 12u);
        CHECK(ls.errors == 0u);
        CHECK(ls.actors_skipped == 0u);

        // Count alive actors + locate ours. Classed actors round-trip
        // their saved name: load_world now passes the parsed actor name
        // through game.spawn_class(), so a GameActor-backed actor comes
        // back named "Alpha"/"Beta" (NOT "SerialTestActor"). Identity is
        // verified by name *and* class. The anonymous transform-only
        // actor keeps its name too. (Regression lock: pre-fix, classed
        // actors lost their name and came back named after their class.)
        int alive = 0, classed = 0, plain = 0;
        const Actor* plain_actor = nullptr;
        for (const auto& ap : game.world().actors()) {
            if (!ap->alive()) continue;
            ++alive;
            if (as_test(*ap) != nullptr) {
                ++classed;
            } else if (ap->name() == "Plain") {
                ++plain;
                plain_actor = ap.get();
            }
        }
        CHECK(alive == 3);
        CHECK(classed == 2);
        CHECK(plain == 1);

        // Locate the classed actors BY NAME — this is the corrected
        // contract. find_by_name doesn't filter dead actors, so assert
        // alive() explicitly; load_world destroyed+swept the placeholder
        // before respawn, so the only "Alpha"/"Beta" are the live ones.
        const Actor* alpha_actor = game.world().find_by_name("Alpha");
        const Actor* beta_actor  = game.world().find_by_name("Beta");
        CHECK(alpha_actor != nullptr);
        CHECK(beta_actor  != nullptr);
        if (alpha_actor) CHECK(alpha_actor->alive());
        if (beta_actor)  CHECK(beta_actor->alive());
        const SerialTestActor* a =
            alpha_actor ? as_test(*alpha_actor) : nullptr;
        const SerialTestActor* b =
            beta_actor  ? as_test(*beta_actor)  : nullptr;
        CHECK(a != nullptr);   // named "Alpha" AND is a SerialTestActor
        CHECK(b != nullptr);   // named "Beta"  AND is a SerialTestActor
        if (a) CHECK(approx(a->f_rpm, 42.5f, 1e-3f));
        if (b) CHECK(approx(b->f_rpm, 99.0f, 1e-3f));

        // Alpha: every property kind survived, incl. the string fix.
        CHECK(a != nullptr);
        if (a) {
            CHECK(approx(a->f_rpm, 42.5f, 1e-4f));
            CHECK(a->i_count == -7);
            CHECK(a->b_flag == true);
            CHECK(approx(a->v_axis.x, 1.0f, 1e-4f));
            CHECK(approx(a->v_axis.y, 2.0f, 1e-4f));
            CHECK(approx(a->v_axis.z, 3.0f, 1e-4f));
            CHECK(approx(a->c_tint.x, 0.5f, 1e-4f));
            CHECK(approx(a->c_tint.y, 0.25f, 1e-4f));
            CHECK(approx(a->c_tint.z, 0.75f, 1e-4f));
            CHECK(a->s_label == "hello world");   // NOT "\"hello world\""
        }
        // Beta: distinct values, and a label with inner quotes/'='.
        CHECK(b != nullptr);
        if (b) {
            CHECK(b->i_count == 13);
            CHECK(b->b_flag == false);
            CHECK(approx(b->v_axis.z, 4.5f, 1e-4f));
            CHECK(b->s_label == "beta=\"q\"");
        }
        // Transforms round-trip (emit %.6f → ~1e-5). Alpha is already
        // located by name above — read its transform directly.
        if (alpha_actor) {
            const auto* tr = alpha_actor->get_component<TransformComponent>();
            CHECK(tr != nullptr);
            if (tr) {
                CHECK(approx(tr->translation.x, 1.0f, 1e-4f));
                CHECK(approx(tr->translation.y, 2.0f, 1e-4f));
                CHECK(approx(tr->translation.z, 3.0f, 1e-4f));
                CHECK(approx(tr->rotation_euler.x, 0.1f, 1e-4f));
                CHECK(approx(tr->rotation_euler.z, 0.3f, 1e-4f));
                CHECK(approx(tr->scale.x, 2.0f, 1e-4f));
            }
        }
        // Anonymous transform-only actor kept its name + transform.
        CHECK(plain_actor != nullptr);
        if (plain_actor) {
            const auto* tr =
                plain_actor->get_component<TransformComponent>();
            CHECK(tr != nullptr);
            if (tr) {
                CHECK(approx(tr->translation.x, 7.0f, 1e-4f));
                CHECK(approx(tr->translation.z, 9.0f, 1e-4f));
            }
        }
        // replace_existing wiped the pre-existing junk actor.
        CHECK(game.world().find_by_name("JunkShouldBeReplaced") == nullptr);
    }

    // ---- failure path -----------------------------------------------
    {
        cardinal::sim::SimWorld sim;
        cardinal::game::Game game(sim);
        cardinal::string err;
        ser::LoadStats ls = ser::load_world(
            game, (dir / "nope.world").string(), false, &err);
        CHECK(!err.empty());
        CHECK(ls.actors_spawned == 0u);
    }

    std::error_code rec;
    const auto removed = std::filesystem::remove_all(dir, rec);
    (void)removed;

    if (g_fail == 0) {
        cardinal::log::infof("worldtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("worldtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
