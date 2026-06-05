// =============================================================================
// Cardinal — deterministic serialization regression suite.
//
// A save-format regression is the worst kind: it compiles, runs, ships,
// and silently eats user saves. This locks the observable contract with
// ZERO test deps (same harness as the net/import suites).
//
// Scope is what's cleanly + deterministically testable headless:
//   • text:: format primitives — exact-string emit + parse round-trips
//     and the parser's edge cases (the foundation every block is built
//     from).
//   • Sky save/load — a full mutate → save → load → assert-equal
//     round-trip. Sky is a self-contained CPU object; Scene/Mesh need
//     an rhi::Device and World needs an actor ClassRegistry, so those
//     round-trips are deliberately out of scope here (separate arcs).
//
// Fixtures live in a temp dir, written + removed by the test. Exit 0 =
// all pass.
// =============================================================================

#include <cardinal/serial/serial.hpp>
#include <cardinal/sky/sky.hpp>
#include <cardinal/actor/world.hpp>
#include <cardinal/game/game.hpp>
#include <cardinal/sim/sim.hpp>
#include <cardinal/core/log.hpp>

#include <filesystem>
#include <vector>

namespace {

namespace ser = cardinal::serial;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("sertest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool approx(float a, float b, float eps) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= eps;
}
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

// ---- text:: format primitives ---------------------------------------
void test_text_helpers() {
    // emit_kv → exact quoted form; parse_kv strips indent + quotes.
    {
        cardinal::string o;
        ser::text::emit_kv(o, "class", "SpinnerActor");
        CHECK(o == "  class = \"SpinnerActor\"\n");
        cardinal::string k, v;
        CHECK(ser::text::parse_kv(o, &k, &v));
        CHECK(k == "class");
        CHECK(v == "SpinnerActor");          // quotes + \n + indent gone
    }
    // emit_kf → fixed %.6f; round-trips through parse_kv as the literal.
    {
        cardinal::string o;
        ser::text::emit_kf(o, "rpm", 30.0f);
        CHECK(o == "  rpm = 30.000000\n");
        cardinal::string k, v;
        CHECK(ser::text::parse_kv(o, &k, &v));
        CHECK(k == "rpm");
        CHECK(v == "30.000000");
    }
    // emit_kv3 → parenthesised triple; value keeps its parens (no quote
    // strip — only matched leading/trailing quotes are removed).
    {
        cardinal::string o;
        ser::text::emit_kv3(o, "axis", 0.0f, 1.0f, 0.0f);
        CHECK(o == "  axis = (0.000000, 1.000000, 0.000000)\n");
        cardinal::string k, v;
        CHECK(ser::text::parse_kv(o, &k, &v));
        CHECK(k == "axis");
        CHECK(v == "(0.000000, 1.000000, 0.000000)");
    }
    // No '=' ⇒ not a kv line.
    {
        cardinal::string k, v;
        CHECK(!ser::text::parse_kv("just a bare line", &k, &v));
        CHECK(!ser::text::parse_kv("# a comment", &k, &v));
    }
    // Quote strip only when BOTH ends are quotes.
    {
        cardinal::string k, v;
        CHECK(ser::text::parse_kv("x = \"abc", &k, &v));
        CHECK(v == "\"abc");                 // unmatched ⇒ left intact
    }
    // Whitespace + CR trimming on both key and value.
    {
        cardinal::string k, v;
        CHECK(ser::text::parse_kv("  spin\t =  \t42 \r", &k, &v));
        CHECK(k == "spin");
        CHECK(v == "42");
    }
    // '=' inside the value: split on the FIRST '=' only.
    {
        cardinal::string k, v;
        CHECK(ser::text::parse_kv("uri = a=b=c", &k, &v));
        CHECK(k == "uri");
        CHECK(v == "a=b=c");
    }
}

// ---- Sky save / load full round-trip --------------------------------
bool key_eq(const cardinal::sky::SkyKey& a,
            const cardinal::sky::SkyKey& b, float e) {
    return approx(a.hour, b.hour, e) &&
           approx(a.zenith.x, b.zenith.x, e) &&
           approx(a.zenith.y, b.zenith.y, e) &&
           approx(a.zenith.z, b.zenith.z, e) &&
           approx(a.horizon.x, b.horizon.x, e) &&
           approx(a.horizon.y, b.horizon.y, e) &&
           approx(a.horizon.z, b.horizon.z, e) &&
           approx(a.sun_color.x, b.sun_color.x, e) &&
           approx(a.sun_color.y, b.sun_color.y, e) &&
           approx(a.sun_color.z, b.sun_color.z, e) &&
           approx(a.sun_intensity, b.sun_intensity, e) &&
           approx(a.ambient.x, b.ambient.x, e) &&
           approx(a.ambient.y, b.ambient.y, e) &&
           approx(a.ambient.z, b.ambient.z, e);
}

void test_sky_roundtrip(const std::filesystem::path& dir) {
    cardinal::sky::Sky a;                 // ctor seeds default phase keys
    a.set_hour(7.25f);
    a.set_time_scale(300.0f);
    a.set_frozen(true);

    CHECK(!a.keys().empty());             // defaults present
    // Mutate an existing key + add a brand-new one (exercises variable
    // key count + the post-load sort).
    a.keys()[0].sun_intensity = 0.4321f;
    a.keys()[0].zenith = cardinal::scene::Vec3{0.11f, 0.22f, 0.33f};
    cardinal::sky::SkyKey extra{};
    extra.hour = 3.3f;
    extra.sun_color = cardinal::scene::Vec3{0.9f, 0.4f, 0.2f};
    extra.sun_intensity = 1.75f;
    a.keys().push_back(extra);
    a.sort_keys();                        // canonical order for compare

    // Snapshot the expectation.
    const float exp_hour  = a.hour();
    const float exp_scale = a.time_scale();
    const bool  exp_froz  = a.frozen();
    const std::vector<cardinal::sky::SkyKey> exp_keys = a.keys();

    const auto path = (dir / "world.sky").string();
    cardinal::string err;
    CHECK(ser::save_sky(a, path, &err));
    CHECK(err.empty());

    cardinal::sky::Sky b;                 // different default state
    CHECK(ser::load_sky(b, path, &err));
    CHECK(err.empty());

    // Scalars: hour/time_scale at %.6f, frozen exact.
    CHECK(approx(b.hour(), exp_hour, 1e-4f));
    CHECK(approx(b.time_scale(), exp_scale, 1e-3f));
    CHECK(b.frozen() == exp_froz);

    // Keys: same count, same values (saved at %.4f ⇒ 1e-3 tolerance),
    // same order (load_sky sort_keys()'d; expectation was sorted too).
    CHECK(b.keys().size() == exp_keys.size());
    if (b.keys().size() == exp_keys.size()) {
        bool all = true;
        for (cardinal::usize i = 0; i < exp_keys.size(); ++i)
            if (!key_eq(b.keys()[i], exp_keys[i], 1e-3f)) all = false;
        CHECK(all);
        // Spot-check the specifically-mutated + added values survived.
        bool saw_extra = false, saw_mut = false;
        for (const auto& k : b.keys()) {
            if (approx(k.hour, 3.3f, 1e-3f) &&
                approx(k.sun_intensity, 1.75f, 1e-3f)) saw_extra = true;
            if (approx(k.sun_intensity, 0.4321f, 1e-3f) &&
                approx(k.zenith.x, 0.11f, 1e-3f)) saw_mut = true;
        }
        CHECK(saw_extra);
        CHECK(saw_mut);
    }
}

void test_sky_failures(const std::filesystem::path& dir) {
    cardinal::sky::Sky b;
    cardinal::string err;
    const auto missing = (dir / "nope_does_not_exist.sky").string();
    CHECK(!ser::load_sky(b, missing, &err));
    CHECK(!err.empty());
}

}  // namespace

// ---- prefab library save / load round-trip (built-in components) ----
// No ClassRegistry needed — built-in components are pure CPU, so this is
// fully deterministic. GameActor prefab round-trip is covered by the
// game suite (which owns the registry); here we lock the multi-component
// file format + the make_component_by_name / deserialize_field path.
void test_prefab_roundtrip(const std::filesystem::path& dir) {
    namespace ac = cardinal::actor;
    const cardinal::string path = (dir / "lib.cardinalprefab").string();

    // Source world: one richly-configured actor captured as "Crate".
    {
        ac::World w;
        ac::Actor* a = w.spawn("Crate");                  // auto-Transform
        auto* t = a->get_component<ac::TransformComponent>();
        t->translation = { 4.0f, 2.0f, -6.0f };
        t->scale       = { 3.0f, 3.0f, 3.0f };
        auto* m = a->add_component<ac::MeshComponent>();
        m->asset_id = "props/crate";
        m->tint     = { 0.7f, 0.5f, 0.3f };
        m->visible  = false;
        auto* l = a->add_component<ac::LightComponent>();
        l->kind = ac::LightKind::Point;
        l->intensity = 5.5f; l->range = 18.0f;
        auto* tg = a->add_component<ac::TagComponent>();
        tg->add("breakable"); tg->add("loot drop");      // space in a tag
        CHECK(w.create_prefab("Crate", a->id()));
        CHECK(w.prefab_component_count("Crate") == sz(4));

        const auto ss = ser::save_prefabs(w, path);
        CHECK(ss.prefabs_written == 1u);
        CHECK(ss.components_written == 4u);
        CHECK(ss.bytes_written > 0u);
    }

    // Fresh world: load the library back + verify a stamped instance.
    {
        ac::World w2;
        const auto ls = ser::load_prefabs(w2, path);
        CHECK(ls.prefabs_loaded == 1u);
        CHECK(ls.components_loaded == 4u);
        CHECK(ls.components_skipped == 0u);
        CHECK(w2.has_prefab("Crate"));
        CHECK(w2.prefab_component_count("Crate") == sz(4));

        ac::Actor* inst = w2.spawn_prefab("Crate");
        CHECK(inst != nullptr);
        if (inst != nullptr) {
            auto* t = inst->get_component<ac::TransformComponent>();
            CHECK(t != nullptr);
            CHECK(approx(t->translation.x, 4.0f, 1e-3f));
            CHECK(approx(t->translation.z, -6.0f, 1e-3f));
            CHECK(approx(t->scale.y, 3.0f, 1e-3f));
            auto* m = inst->get_component<ac::MeshComponent>();
            CHECK(m != nullptr && m->asset_id == "props/crate");
            CHECK(approx(m->tint.x, 0.7f, 1e-3f));
            CHECK(m->visible == false);
            auto* l = inst->get_component<ac::LightComponent>();
            CHECK(l != nullptr && l->kind == ac::LightKind::Point);
            CHECK(approx(l->intensity, 5.5f, 1e-3f) && approx(l->range, 18.0f, 1e-3f));
            auto* tg = inst->get_component<ac::TagComponent>();
            CHECK(tg != nullptr && tg->has("breakable") && tg->has("loot drop"));
        }
    }

    // Missing-file load fails cleanly (error_out set, zero stats).
    {
        ac::World w3;
        cardinal::string err;
        const auto ls = ser::load_prefabs(w3, (dir / "nope.cardinalprefab").string(),
                                          false, &err);
        CHECK(ls.prefabs_loaded == 0u);
        CHECK(!err.empty());
    }
}

// ---- full-component world save / load (level fidelity) ------------
// save_world now persists the COMPLETE component set, not just Transform +
// GameActor props. Verify a plain actor (no game class, so no registry
// dependency) carrying Mesh + Light + Tag round-trips all three through a
// save -> fresh-world load.
void test_world_full_components(const std::filesystem::path& dir) {
    namespace ac = cardinal::actor;
    const cardinal::string path = (dir / "level.cardinalworld").string();

    // Save: one richly-componented anonymous actor.
    {
        cardinal::sim::SimWorld sw{cardinal::sim::SimDesc{}};
        cardinal::game::Game game{sw};
        ac::Actor* a = game.world().spawn("Crate");        // class="" (no GameActor)
        a->get_component<ac::TransformComponent>()->translation = { 2.0f, 3.0f, 4.0f };
        auto* m = a->add_component<ac::MeshComponent>();
        m->asset_id = "props/crate"; m->tint = { 0.9f, 0.8f, 0.7f }; m->visible = false;
        auto* l = a->add_component<ac::LightComponent>();
        l->kind = ac::LightKind::Spot; l->intensity = 6.0f; l->range = 25.0f;
        auto* tg = a->add_component<ac::TagComponent>();
        tg->add("trigger"); tg->add("zone a");
        a->set_enabled(false);                             // disabled — must persist

        const auto ss = ser::save_world(game.world(), path);
        CHECK(ss.actors_written >= 1u);
        CHECK(ss.bytes_written > 0u);
    }

    // Load into a fresh world + verify every component survived.
    {
        cardinal::sim::SimWorld sw{cardinal::sim::SimDesc{}};
        cardinal::game::Game game{sw};
        const auto ls = ser::load_world(game, path, /*replace_existing=*/true);
        CHECK(ls.actors_spawned >= 1u);

        ac::Actor* a = game.world().find_by_name("Crate");
        CHECK(a != nullptr);
        if (a != nullptr) {
            auto* t = a->get_component<ac::TransformComponent>();
            CHECK(t != nullptr && approx(t->translation.y, 3.0f, 1e-3f));
            auto* m = a->get_component<ac::MeshComponent>();
            CHECK(m != nullptr);
            if (m) {
                CHECK(m->asset_id == "props/crate");
                CHECK(approx(m->tint.x, 0.9f, 1e-3f));
                CHECK(m->visible == false);
            }
            auto* l = a->get_component<ac::LightComponent>();
            CHECK(l != nullptr);
            if (l) {
                CHECK(l->kind == ac::LightKind::Spot);
                CHECK(approx(l->intensity, 6.0f, 1e-3f) && approx(l->range, 25.0f, 1e-3f));
            }
            auto* tg = a->get_component<ac::TagComponent>();
            CHECK(tg != nullptr);
            if (tg) { CHECK(tg->has("trigger") && tg->has("zone a")); }
            CHECK(!a->enabled());                  // disabled state round-tripped
        }
    }
}

// ---- play-mode snapshot / restore (the PIE contract) --------------
// game_bar snapshots the world on Play (serialize_world -> string) and
// restores it on Stop (deserialize_world replace). Verify that contract
// in-memory: snapshot, mutate the world as playtesting would (move actors,
// spawn new ones, disable some), then restore -> the authored scene back.
void test_play_snapshot_restore(const std::filesystem::path& /*dir*/) {
    namespace ac = cardinal::actor;

    cardinal::sim::SimWorld sw{cardinal::sim::SimDesc{}};
    cardinal::game::Game game{sw};

    // Authored scene: two actors with set positions + a tag.
    ac::Actor* hero = game.world().spawn("Hero");
    hero->get_component<ac::TransformComponent>()->translation = { 1.0f, 0.0f, 2.0f };
    hero->add_component<ac::TagComponent>()->add("player");
    ac::Actor* prop = game.world().spawn("Prop");
    prop->get_component<ac::TransformComponent>()->translation = { -4.0f, 0.0f, 0.0f };
    const cardinal::usize authored_count = game.world().actor_count();
    CHECK(authored_count == sz(2));

    // --- Play: snapshot to an in-memory string (no disk) ---
    const cardinal::string snap = ser::serialize_world(game.world());
    CHECK(!snap.empty());

    // --- Playtest mutations: move the hero, spawn a projectile, disable prop ---
    hero->get_component<ac::TransformComponent>()->translation = { 50.0f, 9.0f, 50.0f };
    game.world().spawn("Projectile(runtime)");
    prop->set_enabled(false);
    CHECK(game.world().actor_count() == sz(3));    // a runtime actor appeared

    // --- Stop: restore from the snapshot string ---
    const auto ls = ser::deserialize_world(game, snap, /*replace_existing=*/true);
    CHECK(ls.actors_spawned >= 2u);
    game.world().sweep();

    // The authored scene is back: 2 actors, hero at its authored pos, the
    // runtime projectile gone, prop re-enabled.
    CHECK(game.world().actor_count() == authored_count);
    ac::Actor* rhero = game.world().find_by_name("Hero");
    CHECK(rhero != nullptr);
    if (rhero) {
        auto* t = rhero->get_component<ac::TransformComponent>();
        CHECK(approx(t->translation.x, 1.0f, 1e-3f) && approx(t->translation.z, 2.0f, 1e-3f));
        CHECK(rhero->get_component<ac::TagComponent>() != nullptr);
        CHECK(rhero->get_component<ac::TagComponent>()->has("player"));
    }
    CHECK(game.world().find_by_name("Projectile(runtime)") == nullptr);  // runtime gone
    ac::Actor* rprop = game.world().find_by_name("Prop");
    CHECK(rprop != nullptr && rprop->enabled());                          // re-enabled
}

int main() {
    std::error_code ec;
    std::filesystem::path dir =
        std::filesystem::temp_directory_path(ec) / "cardinal_serial_test";
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        cardinal::log::infof("sertest",
            "SKIP  no writable temp dir in this environment");
        return 0;
    }

    test_text_helpers();
    test_sky_roundtrip(dir);
    test_sky_failures(dir);
    test_prefab_roundtrip(dir);
    test_world_full_components(dir);
    test_play_snapshot_restore(dir);

    const auto removed = std::filesystem::remove_all(dir, ec);
    (void)removed;

    if (g_fail == 0) {
        cardinal::log::infof("sertest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("sertest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
