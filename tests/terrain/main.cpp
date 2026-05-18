// =============================================================================
// Cardinal — deterministic terrain regression suite.
//
// Pins the pure terrain surface (mesh-gen needs rhi::Device → omitted):
//
//   terrain_color_profile  — closed-form height+slope tint ramp, pinned
//                            EXACTLY: t-clamp, span guard, cliff blend,
//                            threshold clamp (no div-by-zero), n.y clamp.
//   terrain_height(_profile)— seeded value-noise fbm. Values aren't hand-
//                            predictable, so the CONTRACT is pinned:
//                            determinism (replay/stream consistency),
//                            height_scale + base_height linearity,
//                            disabled-layer == absent, Add weight ratio,
//                            octave/frequency clamp equalities, and
//                            global finiteness (no NaN/Inf blowup).
//   noise_op/shape_name    — every enumerator + invalid → "?".
//   TerrainProfileLibrary  — set/replace/find/remove/names, the text
//                            save↔load round-trip, and the idempotency
//                            of register_default_terrain_profiles.
//
// Library is a process-wide singleton; tests are baseline-relative and
// clean up after themselves. Pure CPU, headless, deterministic.
// Exit 0 = all pass.
// =============================================================================

#include <cardinal/scene/terrain.hpp>
#include <cardinal/core/log.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace sc = cardinal::scene;
using Vec3   = sc::Vec3;
using sc::NoiseOp;
using sc::NoiseShape;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("terrtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-4f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
bool apv(const Vec3& v, float x, float y, float z, float e = 1e-4f) {
    return ap(v.x, x, e) && ap(v.y, y, e) && ap(v.z, z, e);
}
bool fin(float v) { return (v == v) && (v < 3.0e38f) && (v > -3.0e38f); }
bool streq(const char* a, const char* b) { return std::string(a) == b; }
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

sc::TerrainProfile color_profile() {
    sc::TerrainProfile P;
    P.tint_low  = Vec3{0.0f, 0.0f, 0.0f};
    P.tint_high = Vec3{1.0f, 1.0f, 1.0f};
    P.tint_cliff= Vec3{0.5f, 0.25f, 0.75f};
    P.tint_height_low  = 0.0f;
    P.tint_height_high = 10.0f;
    P.cliff_blend_threshold = 0.5f;
    return P;
}

// ---- terrain_color_profile: exact closed form ---------------------
void test_color_profile() {
    const sc::TerrainProfile P = color_profile();
    const Vec3 up   {0.0f, 1.0f, 0.0f};
    const Vec3 vert {0.0f, 0.0f, 0.0f};   // n.y = 0 → full slope

    // Height ramp on flat ground (no cliff): lerp(tint_low,tint_high,t).
    CHECK(apv(sc::terrain_color_profile(P, 0.0f,  up), 0.0f, 0.0f, 0.0f));
    CHECK(apv(sc::terrain_color_profile(P, 5.0f,  up), 0.5f, 0.5f, 0.5f));
    CHECK(apv(sc::terrain_color_profile(P, 10.0f, up), 1.0f, 1.0f, 1.0f));
    CHECK(apv(sc::terrain_color_profile(P, 25.0f, up), 1.0f, 1.0f, 1.0f)); // clamp hi
    CHECK(apv(sc::terrain_color_profile(P, -7.0f, up), 0.0f, 0.0f, 0.0f)); // clamp lo

    // Slope blend: vertical face at mid-height ⇒ exactly tint_cliff.
    CHECK(apv(sc::terrain_color_profile(P, 5.0f, vert), 0.5f, 0.25f, 0.75f));
    // Partial slope: n.y=0.25 ⇒ slope 0.75 ⇒ cliff 0.5 of the way.
    CHECK(apv(sc::terrain_color_profile(P, 5.0f, Vec3{0.0f,0.25f,0.0f}),
              0.5f, 0.375f, 0.625f));
    // Slope below threshold ⇒ no cliff, pure base.
    CHECK(apv(sc::terrain_color_profile(P, 5.0f, Vec3{0.0f,0.9f,0.0f}),
              0.5f, 0.5f, 0.5f));
    // normal.y is clamped to [0,1].
    CHECK(apv(sc::terrain_color_profile(P, 5.0f, Vec3{0.0f,2.0f,0.0f}),
              0.5f, 0.5f, 0.5f));                       // n.y>1 → slope 0
    CHECK(apv(sc::terrain_color_profile(P, 5.0f, Vec3{0.0f,-1.0f,0.0f}),
              0.5f, 0.25f, 0.75f));                     // n.y<0 → slope 1

    // Span guard: tint_height_low == high ⇒ span = 1e-3, t saturates,
    // no div-by-zero.
    {
        sc::TerrainProfile S = P;
        S.tint_height_low = 5.0f; S.tint_height_high = 5.0f;
        const Vec3 lo = sc::terrain_color_profile(S, 4.0f, up);
        const Vec3 hi = sc::terrain_color_profile(S, 6.0f, up);
        CHECK(apv(lo, 0.0f, 0.0f, 0.0f));               // below → t 0
        CHECK(apv(hi, 1.0f, 1.0f, 1.0f));               // above → t 1
        CHECK(fin(lo.x) && fin(hi.x));
    }
    // cliff_blend_threshold clamped to <1 ⇒ vertical face still resolves
    // (no div-by-zero) to tint_cliff even at threshold 1.0.
    {
        sc::TerrainProfile S = P;
        S.cliff_blend_threshold = 1.0f;
        const Vec3 c = sc::terrain_color_profile(S, 5.0f, vert);
        CHECK(apv(c, 0.5f, 0.25f, 0.75f));
        CHECK(fin(c.x) && fin(c.y) && fin(c.z));
    }
}

// ---- noise op / shape name tables ---------------------------------
void test_names() {
    CHECK(streq(sc::noise_op_name(NoiseOp::Add),      "Add"));
    CHECK(streq(sc::noise_op_name(NoiseOp::Multiply), "Multiply"));
    CHECK(streq(sc::noise_op_name(NoiseOp::Max),      "Max"));
    CHECK(streq(sc::noise_op_name(NoiseOp::Min),      "Min"));
    CHECK(streq(sc::noise_op_name(static_cast<NoiseOp>(99u)), "?"));
    CHECK(streq(sc::noise_shape_name(NoiseShape::Smooth), "Smooth"));
    CHECK(streq(sc::noise_shape_name(NoiseShape::Ridge),  "Ridge"));
    CHECK(streq(sc::noise_shape_name(NoiseShape::Billow), "Billow"));
    CHECK(streq(sc::noise_shape_name(static_cast<NoiseShape>(99u)), "?"));
}

// helper: one deterministic Add layer.
sc::NoiseLayer add_layer(cardinal::u32 seed = 1337u, float weight = 1.0f) {
    sc::NoiseLayer L;
    L.id = "L"; L.enabled = true; L.op = NoiseOp::Add;
    L.shape = NoiseShape::Smooth; L.weight = weight;
    L.frequency = 0.05f; L.octaves = 4; L.lacunarity = 2.0f;
    L.gain = 0.5f; L.seed = seed;
    return L;
}

// ---- terrain_height_profile: composition contract -----------------
void test_height_profile() {
    // Empty layers ⇒ exactly base_height * height_scale.
    {
        sc::TerrainProfile P; P.base_height = 2.0f; P.height_scale = 3.0f;
        CHECK(ap(sc::terrain_height_profile(P, 11.0f, -4.0f), 6.0f));
        P.base_height = -1.5f; P.height_scale = 4.0f;
        CHECK(ap(sc::terrain_height_profile(P, 0.0f, 0.0f), -6.0f));
        P.base_height = 9.0f; P.height_scale = 0.0f;
        CHECK(ap(sc::terrain_height_profile(P, 5.0f, 5.0f), 0.0f));
    }
    // A disabled layer contributes nothing ⇒ same as base*scale.
    {
        sc::TerrainProfile P; P.base_height = 2.0f; P.height_scale = 1.0f;
        sc::NoiseLayer L = add_layer(); L.enabled = false;
        P.layers.push_back(L);
        CHECK(ap(sc::terrain_height_profile(P, 7.0f, 13.0f), 2.0f));
    }
    // Determinism: identical inputs ⇒ identical output (replay-safe).
    {
        sc::TerrainProfile P; P.height_scale = 5.0f;
        P.layers.push_back(add_layer());
        const float a = sc::terrain_height_profile(P, 12.5f, -3.25f);
        const float b = sc::terrain_height_profile(P, 12.5f, -3.25f);
        CHECK(a == b);
        CHECK(fin(a));
    }
    // height_scale is a pure post-multiply ⇒ output is linear in it.
    {
        sc::TerrainProfile P1; P1.base_height = 0.0f; P1.height_scale = 1.0f;
        P1.layers.push_back(add_layer());
        sc::TerrainProfile P7 = P1; P7.height_scale = 7.0f;
        sc::TerrainProfile P0 = P1; P0.height_scale = 0.0f;
        const float h1 = sc::terrain_height_profile(P1, 8.0f, 8.0f);
        const float h7 = sc::terrain_height_profile(P7, 8.0f, 8.0f);
        const float h0 = sc::terrain_height_profile(P0, 8.0f, 8.0f);
        CHECK(ap(h7, 7.0f * h1, 1e-3f));
        CHECK(ap(h0, 0.0f));
    }
    // base_height shift adds exactly height_scale per unit (layer cancels).
    {
        sc::TerrainProfile A; A.base_height = 0.0f; A.height_scale = 2.0f;
        A.layers.push_back(add_layer());
        sc::TerrainProfile B = A; B.base_height = 1.0f;
        sc::TerrainProfile C = A; C.base_height = -3.0f;
        const float ra = sc::terrain_height_profile(A, 4.0f, 9.0f);
        const float rb = sc::terrain_height_profile(B, 4.0f, 9.0f);
        const float rc = sc::terrain_height_profile(C, 4.0f, 9.0f);
        CHECK(ap(rb - ra,  2.0f, 1e-3f));    // +1 base * scale 2
        CHECK(ap(rc - ra, -6.0f, 1e-3f));    // -3 base * scale 2
    }
    // Add op scales linearly with the layer weight (v_raw is fixed).
    {
        sc::TerrainProfile P1; P1.base_height = 0.0f; P1.height_scale = 1.0f;
        P1.layers.push_back(add_layer(1337u, 1.0f));
        sc::TerrainProfile P2; P2.base_height = 0.0f; P2.height_scale = 1.0f;
        P2.layers.push_back(add_layer(1337u, 2.0f));
        sc::TerrainProfile P0 = P1;
        P0.layers.clear(); P0.layers.push_back(add_layer(1337u, 0.0f));
        const float r1 = sc::terrain_height_profile(P1, 6.0f, -6.0f);
        const float r2 = sc::terrain_height_profile(P2, 6.0f, -6.0f);
        const float r0 = sc::terrain_height_profile(P0, 6.0f, -6.0f);
        CHECK(ap(r2, 2.0f * r1, 1e-3f));
        CHECK(ap(r0, 0.0f));
    }
}

// ---- terrain_height (legacy): clamps + determinism ----------------
void test_legacy_height() {
    sc::TerrainParams p;                       // defaults
    // Determinism.
    CHECK(sc::terrain_height(p, 5.0f, 7.0f) == sc::terrain_height(p, 5.0f, 7.0f));

    // octaves is clamped to [1,8] ⇒ out-of-range == the clamped value.
    {
        sc::TerrainParams a = p; a.octaves = 0;
        sc::TerrainParams b = p; b.octaves = 1;
        CHECK(sc::terrain_height(a, 3.0f, 4.0f) ==
              sc::terrain_height(b, 3.0f, 4.0f));
        sc::TerrainParams c = p; c.octaves = 100;
        sc::TerrainParams d = p; d.octaves = 8;
        CHECK(sc::terrain_height(c, 3.0f, 4.0f) ==
              sc::terrain_height(d, 3.0f, 4.0f));
        sc::TerrainParams e = p; e.octaves = -9;
        CHECK(sc::terrain_height(e, 3.0f, 4.0f) ==
              sc::terrain_height(b, 3.0f, 4.0f));      // -9 → 1
    }
    // frequency 0 ⇒ noise sampled at a fixed point ⇒ height is
    // position-independent.
    {
        sc::TerrainParams f = p; f.frequency = 0.0f;
        const float h0 = sc::terrain_height(f, 0.0f, 0.0f);
        CHECK(ap(sc::terrain_height(f, 100.0f, -50.0f), h0, 1e-6f));
        CHECK(ap(sc::terrain_height(f, -999.0f, 333.0f), h0, 1e-6f));
    }
    // height_scale is a pure post-multiply.
    {
        sc::TerrainParams s1 = p; s1.height_scale = 1.0f;
        sc::TerrainParams s5 = p; s5.height_scale = 5.0f;
        const float a = sc::terrain_height(s1, 9.0f, 2.0f);
        const float b = sc::terrain_height(s5, 9.0f, 2.0f);
        CHECK(ap(b, 5.0f * a, 1e-3f));
    }
    // Seed actually changes the field (sample several points).
    {
        sc::TerrainParams a = p; a.seed = 1u;
        sc::TerrainParams b = p; b.seed = 2u;
        int differ = 0;
        for (int i = 0; i < 9; ++i) {
            const float wx = static_cast<float>(i) * 7.0f + 1.0f;
            const float wz = static_cast<float>(i) * 3.0f - 2.0f;
            if (!ap(sc::terrain_height(a, wx, wz),
                    sc::terrain_height(b, wx, wz), 1e-4f)) ++differ;
        }
        CHECK(differ > 0);
    }
}

// ---- finiteness across the shipped presets ------------------------
void test_finiteness() {
    sc::register_default_terrain_profiles();
    auto& lib = sc::TerrainProfileLibrary::instance();
    const char* presets[] = { "Plains", "Rolling Hills", "Alpine Mountains",
                              "Volcanic Wasteland", "Coastal Dunes", "Canyon" };
    bool all_finite = true, all_det = true;
    for (const char* nm : presets) {
        const sc::TerrainProfile* P = lib.find(nm);
        CHECK(P != nullptr);
        if (!P) continue;
        for (int gz = -2; gz <= 2; ++gz) {
            for (int gx = -2; gx <= 2; ++gx) {
                const float wx = static_cast<float>(gx) * 30.0f;
                const float wz = static_cast<float>(gz) * 30.0f;
                const float h  = sc::terrain_height_profile(*P, wx, wz);
                if (!fin(h)) all_finite = false;
                if (sc::terrain_height_profile(*P, wx, wz) != h)
                    all_det = false;
            }
        }
    }
    CHECK(all_finite);
    CHECK(all_det);

    // Legacy default params are finite too.
    sc::TerrainParams p;
    bool legacy_fin = true;
    for (int i = -3; i <= 3; ++i)
        if (!fin(sc::terrain_height(p, static_cast<float>(i) * 25.0f,
                                       static_cast<float>(i) * 11.0f)))
            legacy_fin = false;
    CHECK(legacy_fin);
}

// ---- TerrainProfileLibrary: registry semantics --------------------
void test_library() {
    auto& lib = sc::TerrainProfileLibrary::instance();
    const cardinal::usize base = lib.all().size();

    CHECK(lib.find("__ct_A") == nullptr);
    CHECK(lib.find(nullptr) == nullptr);
    CHECK(lib.remove(nullptr) == false);
    CHECK(lib.remove("__ct_absent") == false);

    sc::TerrainProfile a; a.name = "__ct_A"; a.base_height = 1.0f;
    lib.set(a);
    CHECK(lib.all().size() == base + sz(1));
    const sc::TerrainProfile* fa = lib.find("__ct_A");
    CHECK(fa != nullptr && ap(fa->base_height, 1.0f));

    // set() with an existing name REPLACES in place (no size growth).
    a.base_height = 9.0f;
    lib.set(a);
    CHECK(lib.all().size() == base + sz(1));
    fa = lib.find("__ct_A");
    CHECK(fa != nullptr && ap(fa->base_height, 9.0f));

    sc::TerrainProfile b; b.name = "__ct_B";
    lib.set(b);
    CHECK(lib.all().size() == base + sz(2));
    // names() lists every entry (insertion order; ours appended at end).
    auto nm = lib.names();
    bool saw_a = false, saw_b = false;
    for (const auto& s : nm) { if (s == "__ct_A") saw_a = true;
                               if (s == "__ct_B") saw_b = true; }
    CHECK(saw_a && saw_b && nm.size() == lib.all().size());

    CHECK(lib.remove("__ct_A") == true);
    CHECK(lib.find("__ct_A") == nullptr);
    CHECK(lib.all().size() == base + sz(1));
    CHECK(lib.remove("__ct_A") == false);        // already gone
    CHECK(lib.remove("__ct_B") == true);
    CHECK(lib.all().size() == base);             // restored to baseline
}

// ---- register_default_terrain_profiles idempotency ----------------
void test_register_idempotent() {
    auto& lib = sc::TerrainProfileLibrary::instance();
    sc::register_default_terrain_profiles();
    const cardinal::usize n1 = lib.all().size();
    CHECK(n1 >= sz(6));
    CHECK(lib.find("Plains") != nullptr);
    CHECK(lib.find("Canyon") != nullptr);

    // A second call must be a no-op (no duplicates).
    sc::register_default_terrain_profiles();
    CHECK(lib.all().size() == n1);

    // Idempotent = "skip if present": a user edit to a preset survives
    // a re-register (add_if_missing only adds missing names).
    sc::TerrainProfile P = *lib.find("Plains");
    P.base_height = 42.0f;
    lib.set(P);
    sc::register_default_terrain_profiles();
    const sc::TerrainProfile* after = lib.find("Plains");
    CHECK(after != nullptr && ap(after->base_height, 42.0f));
    CHECK(lib.all().size() == n1);
}

// ---- text save ↔ load round-trip ----------------------------------
void test_save_load() {
    namespace fs = std::filesystem;
    auto& lib = sc::TerrainProfileLibrary::instance();

    sc::TerrainProfile P;
    P.name = "__ct rt prof";                 // spaces ⇒ quoted round-trip
    P.description = "round trip desc";
    P.base_height = 2.0f;
    P.height_scale = 24.0f;
    P.tint_low  = Vec3{0.25f, 0.5f,  0.75f};
    P.tint_high = Vec3{0.125f,0.625f,1.0f};
    P.tint_cliff= Vec3{0.5f,  0.5f,  0.5f};
    P.tint_height_low  = -2.0f;
    P.tint_height_high = 18.0f;
    P.cliff_blend_threshold = 0.5f;
    {
        sc::NoiseLayer L; L.id = "L0"; L.enabled = true;
        L.op = NoiseOp::Add; L.shape = NoiseShape::Smooth;
        L.weight = 0.5f; L.frequency = 0.03125f; L.octaves = 4;
        L.lacunarity = 2.0f; L.gain = 0.5f; L.seed = 12345u;
        L.offset_x = 3.5f; L.offset_z = -7.25f;
        L.warp_amount = 12.0f; L.warp_frequency = 0.0625f;
        P.layers.push_back(L);
    }
    {
        sc::NoiseLayer L; L.id = "L1"; L.enabled = false;
        L.op = NoiseOp::Multiply; L.shape = NoiseShape::Ridge;
        L.weight = 0.75f; L.frequency = 0.015625f; L.octaves = 3;
        L.lacunarity = 2.5f; L.gain = 0.625f; L.seed = 999u;
        L.offset_x = 0.0f; L.offset_z = 0.0f;
        L.warp_amount = 0.0f; L.warp_frequency = 0.0625f;
        P.layers.push_back(L);
    }

    fs::path tmp  = fs::temp_directory_path() / "cardinal_terrain_rt.profiles";
    fs::path miss = fs::temp_directory_path() / "cardinal_terrain_absent.zzz";
    if (fs::exists(tmp))  fs::remove(tmp);
    if (fs::exists(miss)) fs::remove(miss);
    const std::string tmps  = tmp.string();
    const std::string misss = miss.string();

    // Bad-path guards.
    CHECK(lib.save_to_file(nullptr) == false);
    CHECK(lib.save_to_file("") == false);
    CHECK(lib.load_from_file(nullptr) == false);
    CHECK(lib.load_from_file(misss.c_str()) == false);   // no such file

    lib.set(P);
    CHECK(lib.save_to_file(tmps.c_str()) == true);
    CHECK(fs::exists(tmp));

    // Drop it, then reload from disk and compare every field.
    CHECK(lib.remove("__ct rt prof") == true);
    CHECK(lib.find("__ct rt prof") == nullptr);
    CHECK(lib.load_from_file(tmps.c_str()) == true);

    const sc::TerrainProfile* R = lib.find("__ct rt prof");
    CHECK(R != nullptr);
    if (R) {
        CHECK(R->name == "__ct rt prof");          // quoted name survived
        CHECK(R->description == "round trip desc");
        CHECK(ap(R->base_height, 2.0f));
        CHECK(ap(R->height_scale, 24.0f));
        CHECK(apv(R->tint_low,  0.25f, 0.5f,  0.75f));
        CHECK(apv(R->tint_high, 0.125f,0.625f,1.0f));
        CHECK(apv(R->tint_cliff,0.5f,  0.5f,  0.5f));
        CHECK(ap(R->tint_height_low,  -2.0f));
        CHECK(ap(R->tint_height_high, 18.0f));
        CHECK(ap(R->cliff_blend_threshold, 0.5f));
        CHECK(R->layers.size() == sz(2));
        if (R->layers.size() == sz(2)) {
            const auto& a = R->layers[0];
            CHECK(a.id == "L0" && a.enabled == true);
            CHECK(a.op == NoiseOp::Add && a.shape == NoiseShape::Smooth);
            CHECK(ap(a.weight, 0.5f) && ap(a.frequency, 0.03125f));
            CHECK(a.octaves == 4 && ap(a.lacunarity, 2.0f));
            CHECK(ap(a.gain, 0.5f) && a.seed == 12345u);
            CHECK(ap(a.offset_x, 3.5f) && ap(a.offset_z, -7.25f));
            CHECK(ap(a.warp_amount, 12.0f) && ap(a.warp_frequency, 0.0625f));
            const auto& b = R->layers[1];
            CHECK(b.id == "L1" && b.enabled == false);
            CHECK(b.op == NoiseOp::Multiply && b.shape == NoiseShape::Ridge);
            CHECK(ap(b.weight, 0.75f) && ap(b.frequency, 0.015625f));
            CHECK(b.octaves == 3 && ap(b.lacunarity, 2.5f));
            CHECK(ap(b.gain, 0.625f) && b.seed == 999u);
        }
    }

    // Clean up: library back to baseline, temp file gone.
    lib.remove("__ct rt prof");
    if (fs::exists(tmp)) fs::remove(tmp);
    CHECK(lib.find("__ct rt prof") == nullptr);
}

}  // namespace

int main() {
    // Library-touching tests first (pristine singleton), then pure math.
    test_save_load();
    test_library();
    test_register_idempotent();
    test_finiteness();
    test_color_profile();
    test_names();
    test_height_profile();
    test_legacy_height();

    if (g_fail == 0) {
        cardinal::log::infof("terrtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("terrtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
