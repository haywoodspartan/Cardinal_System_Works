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
