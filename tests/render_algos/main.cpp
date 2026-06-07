// =============================================================================
// Cardinal — deterministic render-algorithm-registry regression suite.
//
// render::algo::AlgoRegistry is the catalogue every pipeline queries to
// build Knob enums + wire shader variants; its cpu_fn reference impls
// are the CPU-side contract the HLSL twins must match. Pinned:
//   * category_name/description (8 + Count_→"?");
//   * the curated catalogue — exact per-category counts, INSERTION order,
//     and is_default lookup (NOT always index 0 — ClusterCull's default
//     "frustum_cone" is registered 3rd);
//   * find (id / nullptr / bad-category), default_for, register_algo
//     dedupe + out-of-range reject;
//   * labels_for / default_index_for / hlsl_for_choice / id_for_choice
//     incl. negative + overflow + empty-list indices;
//   * the closed-form cpu_fn outputs (linear/reinhard/aces tonemap,
//     box/tent/kaiser/lanczos/catmull mip, distance/edge/phong tess,
//     halton/bayer/sobol sampling, octa/stereo normal-encode) + hash
//     determinism/range/seed-sensitivity + fp32 identity.
//
// The registry is a process singleton; the catalogue counts are asserted
// BEFORE any test registers a custom algo. Pure CPU, deterministic.
// Exit 0 = all pass.
// =============================================================================

#include <cardinal/render/algos.hpp>
#include <cardinal/core/diag/log.hpp>

#include <string>
#include <vector>

namespace {

namespace ra = cardinal::render::algo;
using CID = ra::CategoryId;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("ralgotest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-4f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }
bool streq(const char* a, const char* b) { return std::string(a) == b; }

ra::AlgoRegistry& reg() { return ra::AlgoRegistry::instance(); }

ra::AlgoOut run(CID c, const char* id, const ra::AlgoIn& in, int line) {
    const ra::Algo* a = reg().find(c, id);
    ::check_impl(a != nullptr && static_cast<bool>(a->cpu_fn), id, line);
    ra::AlgoOut o{};
    if (a && a->cpu_fn) a->cpu_fn(in, o);
    return o;
}
#define RUN(C, ID, IN) run((C), (ID), (IN), __LINE__)

bool unit01(float v) { return v >= 0.0f && v < 1.0f + 1e-6f; }

// ---- category name / description ----------------------------------
void test_categories() {
    CHECK(streq(ra::category_name(CID::Tonemap),       "Tonemap"));
    CHECK(streq(ra::category_name(CID::HashRng),       "Hash / RNG"));
    CHECK(streq(ra::category_name(CID::MipFilter),     "Mip Filter"));
    CHECK(streq(ra::category_name(CID::ClusterCull),   "Cluster Culling"));
    CHECK(streq(ra::category_name(CID::TessFactor),    "Tessellation Factor"));
    CHECK(streq(ra::category_name(CID::Sampling),      "Sampling Pattern"));
    CHECK(streq(ra::category_name(CID::NormalEncode),  "Normal Encoding"));
    CHECK(streq(ra::category_name(CID::PrecisionMath), "Precision Math"));
    CHECK(streq(ra::category_name(CID::Count_),        "?"));
    CHECK(std::string(ra::category_description(CID::Tonemap)).size() > 10);
    CHECK(std::string(ra::category_description(CID::PrecisionMath)).size() > 10);
    CHECK(streq(ra::category_description(CID::Count_), "?"));
}

// ---- curated catalogue: counts / order / default / find -----------
void test_catalogue() {
    CHECK(reg().list(CID::Tonemap).size()       == sz(7));
    CHECK(reg().list(CID::HashRng).size()       == sz(4));
    CHECK(reg().list(CID::MipFilter).size()     == sz(5));
    CHECK(reg().list(CID::ClusterCull).size()   == sz(4));
    CHECK(reg().list(CID::TessFactor).size()    == sz(3));
    CHECK(reg().list(CID::Sampling).size()      == sz(5));
    CHECK(reg().list(CID::NormalEncode).size()  == sz(3));
    CHECK(reg().list(CID::PrecisionMath).size() == sz(6));
    CHECK(reg().list(CID::Count_).empty());

    const auto& tm = reg().list(CID::Tonemap);
    CHECK(tm[0].id == "linear" && tm[0].is_default);
    CHECK(tm[4].id == "aces_approx" && tm[6].id == "lottes");
    CHECK(tm[0].hlsl_function == "cardinal_tonemap_linear");
    CHECK(!tm[0].is_user);

    // ClusterCull default is registered THIRD — list[0] is NOT default.
    const auto& cc = reg().list(CID::ClusterCull);
    CHECK(cc[0].id == "none" && !cc[0].is_default);
    CHECK(cc[2].id == "frustum_cone" && cc[2].is_default);
    const ra::Algo* ccd = reg().default_for(CID::ClusterCull);
    CHECK(ccd != nullptr && ccd->id == "frustum_cone");
    const ra::Algo* tmd = reg().default_for(CID::Tonemap);
    CHECK(tmd != nullptr && tmd->id == "linear");
    CHECK(reg().default_for(CID::Count_) == nullptr);

    const ra::Algo* af = reg().find(CID::Tonemap, "aces_full");
    CHECK(af != nullptr);
    CHECK(af->label == "ACES (RRT/ODT fit)");
    CHECK(af->hlsl_function == "cardinal_tonemap_aces_full");
    CHECK(reg().find(CID::Tonemap, "nope") == nullptr);
    CHECK(reg().find(CID::Tonemap, nullptr) == nullptr);
    CHECK(reg().find(CID::Count_, "linear") == nullptr);

    // fp32 has no GPU twin (empty hlsl) and is the precision default.
    const ra::Algo* fp32 = reg().find(CID::PrecisionMath, "fp32");
    CHECK(fp32 != nullptr && fp32->hlsl_function.empty() && fp32->is_default);
}

// ---- picker helpers -----------------------------------------------
void test_helpers() {
    auto labels = ra::labels_for(CID::Tonemap);
    CHECK(labels.size() == sz(7));
    CHECK(labels[0] == "Linear (clamp)" && labels[4] == "ACES (approximate)");
    CHECK(ra::default_index_for(CID::Tonemap)      == 0);
    CHECK(ra::default_index_for(CID::ClusterCull)  == 2);   // 3rd entry
    CHECK(ra::default_index_for(CID::NormalEncode) == 0);
    CHECK(streq(ra::hlsl_for_choice(CID::Tonemap, 4),
                "cardinal_tonemap_aces_approx"));
    CHECK(streq(ra::hlsl_for_choice(CID::Tonemap, -1), ""));
    CHECK(streq(ra::hlsl_for_choice(CID::Tonemap, 99), ""));
    CHECK(streq(ra::id_for_choice(CID::Tonemap, 6), "lottes"));
    CHECK(streq(ra::id_for_choice(CID::Tonemap, -1), ""));
    CHECK(streq(ra::id_for_choice(CID::Count_, 0), ""));     // empty list
}

// ---- tonemap cpu_fn -----------------------------------------------
void test_tonemap() {
    ra::AlgoIn in{};
    in.color3[0] = 2.0f; in.color3[1] = 0.5f; in.color3[2] = -1.0f;
    ra::AlgoOut o = RUN(CID::Tonemap, "linear", in);
    CHECK(ap(o.color3[0], 1.0f) && ap(o.color3[1], 0.5f) && ap(o.color3[2], 0.0f));

    in.color3[0] = 1.0f; in.color3[1] = 0.0f; in.color3[2] = 3.0f;
    o = RUN(CID::Tonemap, "reinhard", in);                  // x/(x+1)
    CHECK(ap(o.color3[0], 0.5f) && ap(o.color3[1], 0.0f) && ap(o.color3[2], 0.75f));

    in.color3[0] = 0.0f; in.color3[1] = 4.0f; in.color3[2] = 0.0f;
    o = RUN(CID::Tonemap, "reinhard_ext", in);              // wp=4 → x=4 maps to 1
    CHECK(ap(o.color3[0], 0.0f) && ap(o.color3[1], 1.0f));

    in.color3[0] = 0.0f; in.color3[1] = 100.0f; in.color3[2] = 0.0f;
    o = RUN(CID::Tonemap, "aces_approx", in);
    CHECK(ap(o.color3[0], 0.0f) && ap(o.color3[1], 1.0f, 1e-2f));   // saturates

    o = RUN(CID::Tonemap, "aces_full", in);
    CHECK(ap(o.color3[0], 0.0f) && ap(o.color3[1], 1.0f, 1e-2f));
}

// ---- mip filters (exact weighted sums) ----------------------------
void test_mip() {
    ra::AlgoIn in{};
    // 4 RGB samples; R = {4,8,12,16}, G/B = 0.
    const float s[12] = { 4,0,0, 8,0,0, 12,0,0, 16,0,0 };
    for (int i = 0; i < 12; ++i) in.samples4[i] = s[i];

    ra::AlgoOut o = RUN(CID::MipFilter, "box", in);
    CHECK(ap(o.color3[0], 10.0f) && ap(o.color3[1], 0.0f));   // 0.25*40
    o = RUN(CID::MipFilter, "tent", in);
    CHECK(ap(o.color3[0], 8.8f, 1e-3f));                       // .4*4+.2*(8+12+16)
    o = RUN(CID::MipFilter, "kaiser", in);
    CHECK(ap(o.color3[0], 8.2f, 1e-3f));                       // .45*4+.2*8+.2*12+.15*16
    o = RUN(CID::MipFilter, "lanczos2", in);
    CHECK(ap(o.color3[0], 7.0f, 1e-3f));                       // .55*4+.2*8+.2*12+.05*16
    o = RUN(CID::MipFilter, "catmull_rom", in);
    CHECK(ap(o.color3[0], 7.3f, 1e-3f));                       // .5*4+.225*8+.225*12+.05*16
}

// ---- tessellation factor (clamped formulas) -----------------------
void test_tess() {
    ra::AlgoIn in{};
    in.distance = 1.0f;   CHECK(ap(RUN(CID::TessFactor,"distance",in).factor, 16.0f));
    in.distance = 16.0f;  CHECK(ap(RUN(CID::TessFactor,"distance",in).factor, 1.0f));
    in.distance = 0.25f;  CHECK(ap(RUN(CID::TessFactor,"distance",in).factor, 64.0f)); // clamp hi
    in.distance = 32.0f;  CHECK(ap(RUN(CID::TessFactor,"distance",in).factor, 1.0f));  // clamp lo
    in.distance = 0.0f;   CHECK(ap(RUN(CID::TessFactor,"distance",in).factor, 64.0f)); // d→0.001

    in.edge_pixels = 160.0f; CHECK(ap(RUN(CID::TessFactor,"edge",in).factor, 10.0f));
    in.edge_pixels = 8.0f;   CHECK(ap(RUN(CID::TessFactor,"edge",in).factor, 1.0f));
    in.edge_pixels = 4000.0f;CHECK(ap(RUN(CID::TessFactor,"edge",in).factor, 64.0f));

    in.distance = 1.0f;   CHECK(ap(RUN(CID::TessFactor,"phong",in).factor, 24.0f));
    in.distance = 24.0f;  CHECK(ap(RUN(CID::TessFactor,"phong",in).factor, 1.0f));
}

// ---- sampling patterns --------------------------------------------
void test_sampling() {
    ra::AlgoIn in{};
    in.index = 0;
    ra::AlgoOut o = RUN(CID::Sampling, "halton", in);
    CHECK(ap(o.unit3[0], 0.5f) && ap(o.unit3[1], 1.0f/3.0f) && ap(o.unit3[2], 0.0f));
    in.index = 1;
    o = RUN(CID::Sampling, "halton", in);
    CHECK(ap(o.unit3[0], 0.25f) && ap(o.unit3[1], 2.0f/3.0f));

    in.index = 0;  o = RUN(CID::Sampling, "bayer", in);
    CHECK(ap(o.unit3[0], 0.0f) && ap(o.unit3[1], 0.0f));
    in.index = 1;  o = RUN(CID::Sampling, "bayer", in);
    CHECK(ap(o.unit3[0], 0.5f));                              // m[1]=8 → 8/16
    in.index = 4;  o = RUN(CID::Sampling, "bayer", in);
    CHECK(ap(o.unit3[0], 0.75f));                             // m[4]=12 → 12/16
    in.index = 16; o = RUN(CID::Sampling, "bayer", in);
    CHECK(ap(o.unit3[0], 0.0f));                              // (16&15)=0

    in.index = 0;  o = RUN(CID::Sampling, "hammersley", in);
    CHECK(ap(o.unit3[0], 0.0f) && ap(o.unit3[1], 0.0f));
    in.index = 512; o = RUN(CID::Sampling, "hammersley", in);
    CHECK(ap(o.unit3[0], 0.5f));                              // 512/1024

    in.index = 0;  o = RUN(CID::Sampling, "sobol", in);
    CHECK(ap(o.unit3[0], 0.5f, 1e-6f) && ap(o.unit3[1], 0.25f, 1e-6f));
    // Large index: the bit-loop reaches b ~= 31, so the dim_offset=1
    // dimension used to underflow `31 - b - dim_offset` and shift by
    // ~0xFFFFFFFF (UB). Must now stay finite + in [0,1).
    in.index = 0xFFFFFFF0u; o = RUN(CID::Sampling, "sobol", in);
    CHECK(unit01(o.unit3[0]) && unit01(o.unit3[1]));
    in.index = 0x7FFFFFFFu; o = RUN(CID::Sampling, "sobol", in);
    CHECK(unit01(o.unit3[0]) && unit01(o.unit3[1]));

    in.seed = 12345u;
    ra::AlgoOut b1 = RUN(CID::Sampling, "blue_noise", in);
    ra::AlgoOut b2 = RUN(CID::Sampling, "blue_noise", in);
    CHECK(ap(b1.unit3[0], b2.unit3[0]) && unit01(b1.unit3[0]));   // deterministic
}

// ---- hash / rng: determinism + range + seed-sensitivity -----------
void test_hash() {
    const char* ids[4] = { "wang", "pcg", "hash13", "ign" };
    for (const char* id : ids) {
        ra::AlgoIn a{}; a.seed = 42u;
        ra::AlgoIn b{}; b.seed = 42u;
        ra::AlgoIn c{}; c.seed = 43u;
        ra::AlgoOut oa = RUN(CID::HashRng, id, a);
        ra::AlgoOut ob = RUN(CID::HashRng, id, b);
        ra::AlgoOut oc = RUN(CID::HashRng, id, c);
        CHECK(ap(oa.unit3[0], ob.unit3[0]) &&
              ap(oa.unit3[1], ob.unit3[1]) &&
              ap(oa.unit3[2], ob.unit3[2]));               // deterministic
        CHECK(unit01(oa.unit3[0]) && unit01(oa.unit3[1]) && unit01(oa.unit3[2]));
        // Different seed ⇒ different output (vanishingly unlikely to tie).
        CHECK(!ap(oa.unit3[0], oc.unit3[0], 1e-7f) ||
              !ap(oa.unit3[1], oc.unit3[1], 1e-7f) ||
              !ap(oa.unit3[2], oc.unit3[2], 1e-7f));
    }
}

// ---- normal encoding ----------------------------------------------
void test_normal() {
    ra::AlgoIn in{};
    in.unit3[0]=0; in.unit3[1]=0; in.unit3[2]=1;             // +Z
    ra::AlgoOut o = RUN(CID::NormalEncode, "octahedral", in);
    CHECK(ap(o.color3[0], 0.5f) && ap(o.color3[1], 0.5f) && ap(o.color3[2], 0.0f));
    in.unit3[0]=1; in.unit3[1]=0; in.unit3[2]=0;             // +X
    o = RUN(CID::NormalEncode, "octahedral", in);
    CHECK(ap(o.color3[0], 1.0f) && ap(o.color3[1], 0.5f));
    in.unit3[0]=0; in.unit3[1]=0; in.unit3[2]=-1;            // -Z (folded)
    o = RUN(CID::NormalEncode, "octahedral", in);
    CHECK(ap(o.color3[0], 1.0f) && ap(o.color3[1], 1.0f));

    in.unit3[0]=0; in.unit3[1]=0; in.unit3[2]=1;
    o = RUN(CID::NormalEncode, "stereographic", in);         // n/(z+1)
    CHECK(ap(o.color3[0], 0.0f) && ap(o.color3[1], 0.0f));
    in.unit3[0]=1; in.unit3[1]=0; in.unit3[2]=0;
    o = RUN(CID::NormalEncode, "stereographic", in);         // 1/(0+1)
    CHECK(ap(o.color3[0], 1.0f) && ap(o.color3[1], 0.0f));
    in.unit3[0]=0; in.unit3[1]=0; in.unit3[2]=-1;
    o = RUN(CID::NormalEncode, "stereographic", in);         // denom 0 → 0
    CHECK(ap(o.color3[0], 0.0f) && ap(o.color3[1], 0.0f));
}

// ---- precision math: fp32 identity + 0/1 round-trip ---------------
void test_precision() {
    ra::AlgoIn in{};
    in.color3[0] = 0.123f; in.color3[1] = -2.5f; in.color3[2] = 7.0f;
    ra::AlgoOut o = RUN(CID::PrecisionMath, "fp32", in);
    CHECK(o.color3[0] == 0.123f && o.color3[1] == -2.5f && o.color3[2] == 7.0f);

    const char* fmts[5] = { "fp16","fp8_e4m3","fp8_e5m2","fp4_e2m1","fp4_e3m0" };
    for (const char* f : fmts) {
        ra::AlgoIn q{}; q.color3[0]=1.0f; q.color3[1]=0.0f; q.color3[2]=1.0f;
        ra::AlgoOut a = RUN(CID::PrecisionMath, f, q);
        // 0 and 1 are representable in every listed format.
        CHECK(ap(a.color3[0], 1.0f, 1e-6f) && ap(a.color3[1], 0.0f, 1e-6f));
        ra::AlgoIn d{}; d.color3[0]=0.7f;
        ra::AlgoOut x = RUN(CID::PrecisionMath, f, d);
        ra::AlgoOut y = RUN(CID::PrecisionMath, f, d);
        CHECK(x.color3[0] == y.color3[0]);                   // deterministic
    }
}

// ---- register_algo: dedupe + out-of-range (run LAST) --------------
void test_register() {
    const cardinal::usize base = reg().list(CID::MipFilter).size();   // 5

    ra::Algo dup;
    dup.category = CID::MipFilter; dup.id = "box"; dup.label = "Dup";
    dup.cpu_fn = [](const ra::AlgoIn&, ra::AlgoOut&) {};
    CHECK(reg().register_algo(std::move(dup)) == false);              // id taken
    CHECK(reg().list(CID::MipFilter).size() == base);

    ra::Algo na;
    na.category = CID::MipFilter; na.id = "test.custom_mip";
    na.label = "Custom"; na.hlsl_function = "x";
    na.cpu_fn = [](const ra::AlgoIn&, ra::AlgoOut& o) { o.factor = 9.0f; };
    CHECK(reg().register_algo(std::move(na)) == true);
    CHECK(reg().list(CID::MipFilter).size() == base + sz(1));
    const ra::Algo* got = reg().find(CID::MipFilter, "test.custom_mip");
    CHECK(got != nullptr && got->label == "Custom");
    CHECK(reg().list(CID::MipFilter).back().id == "test.custom_mip"); // appended

    ra::Algo oob;
    oob.category = CID::Count_; oob.id = "x";
    CHECK(reg().register_algo(std::move(oob)) == false);
    ra::Algo oob2;
    oob2.category = static_cast<CID>(99u); oob2.id = "y";
    CHECK(reg().register_algo(std::move(oob2)) == false);
    CHECK(reg().list(CID::MipFilter).size() == base + sz(1));         // unchanged
}

}  // namespace

int main() {
    test_categories();
    test_catalogue();         // exact counts — BEFORE any custom register
    test_helpers();
    test_tonemap();
    test_mip();
    test_tess();
    test_sampling();
    test_hash();
    test_normal();
    test_precision();
    test_register();          // mutates the singleton — must run LAST

    if (g_fail == 0) {
        cardinal::log::infof("ralgotest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("ralgotest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
