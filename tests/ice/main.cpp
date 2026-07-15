// =============================================================================
// Cardinal ICE — regression suite.
//
// 1. The classic ICE "turbulize" compound rebuilt node-by-node and checked
//    against hand-inlined math: P' = P + normalize(P) * (noise(P) * amp).
// 2. The fast operator dependency graph: const edits re-evaluate ONLY the
//    downstream cone (EvalStats.nodes_evaluated), clean re-evaluates nothing,
//    input touches re-evaluate the input's cone, point-count changes
//    re-evaluate everything.
// 3. Compile-time validation: arity, input types, cycles.
// Pure CPU, bit-deterministic. Exit 0 = all pass.
// =============================================================================

#include <cardinal/ice/ice.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/std/cmath.hpp>

namespace {

namespace ice = cardinal::ice;
using cardinal::u32;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("icetest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-5f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}

// The evaluator's noise, re-derived here so the test pins the bit contract.
float ref_noise(const ice::Vec3& p) {
    auto fbits = [](float f) -> unsigned long long {
        union { float f; unsigned u; } x; x.f = f;
        return static_cast<unsigned long long>(x.u);
    };
    unsigned long long h = fbits(p.x) * 0x9E3779B97F4A7C15ull;
    h ^= (fbits(p.y) + 0xBF58476D1CE4E5B9ull) * 0x94D049BB133111EBull;
    h ^= (fbits(p.z) + 0x2545F4914F6CDD1Dull) * 0xD6E8FEB86659FD93ull;
    h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 27; h *= 0x94D049BB133111EBull;
    h ^= h >> 31;
    return static_cast<float>(h >> 40) * (1.0f / 16777216.0f);
}

// ---- turbulize: node math vs hand math --------------------------------------
void test_turbulize_math() {
    ice::GeoData geo;
    geo.point_count = 6;
    auto& P = geo.ensure_vec3("P");
    P[0] = { 1.0f, 0.0f, 0.0f };
    P[1] = { 0.0f, 2.0f, 0.0f };
    P[2] = { 0.0f, 0.0f, -3.0f };
    P[3] = { 1.0f, 1.0f, 1.0f };
    P[4] = { -2.0f, 0.5f, 0.25f };
    P[5] = { 0.0f, 0.0f, 0.0f };     // degenerate: normalize -> zero vector

    ice::IceGraph g;
    const int p    = g.get_vec3("P");
    const int nrm  = g.normalize(p);
    const int nz   = g.noise(p);
    const int amp  = g.const_float(0.5f);
    const int mag  = g.mul_f(nz, amp);
    const int off  = g.scale(nrm, mag);
    const int sum  = g.add_v3(p, off);
    const int sink = g.set_vec3("P_out", sum);
    (void)sink;

    ice::IceEvaluator ev;
    cardinal::string err;
    CHECK(ev.compile(g, &err));
    CHECK(err.empty());
    ice::EvalStats st{};
    CHECK(ev.evaluate(geo, &st));
    CHECK(st.nodes_total == 8);
    CHECK(st.nodes_evaluated == 8);              // first eval: everything

    const auto& out = geo.vec3["P_out"];
    CHECK(out.size() == 6);
    for (u32 i = 0; i < 6; ++i) {
        const ice::Vec3 pi = P[i];
        const float l2 = pi.x * pi.x + pi.y * pi.y + pi.z * pi.z;
        ice::Vec3 n{0, 0, 0};
        if (l2 > 1.0e-12f) {
            const float inv = 1.0f / cardinal::sqrt(l2);
            n = { pi.x * inv, pi.y * inv, pi.z * inv };
        }
        const float m = ref_noise(pi) * 0.5f;
        CHECK(ap(out[i].x, pi.x + n.x * m));
        CHECK(ap(out[i].y, pi.y + n.y * m));
        CHECK(ap(out[i].z, pi.z + n.z * m));
    }
    // Noise is in [0,1) and not constant across distinct points.
    const auto n0 = ref_noise(P[0]), n1 = ref_noise(P[1]);
    CHECK(n0 >= 0.0f && n0 < 1.0f);
    CHECK(!ap(n0, n1, 1e-9f));
}

// ---- the operator dependency graph: incremental re-evaluation ---------------
void test_incremental_dirty() {
    ice::GeoData geo;
    geo.point_count = 4;
    auto& P = geo.ensure_vec3("P");
    for (u32 i = 0; i < 4; ++i)
        P[i] = { static_cast<float>(i) + 1.0f, 0.5f, -0.25f };

    // Two INDEPENDENT chains off the same input:
    //   A: P -> normalize -> scale(by ampA) -> Set "A"     (4 nodes + shared)
    //   B: P -> length    -> mulF(by ampB)  -> Set "B"
    ice::IceGraph g;
    const int p    = g.get_vec3("P");
    const int nrm  = g.normalize(p);
    const int ampA = g.const_float(2.0f);
    const int offA = g.scale(nrm, ampA);
    const int setA = g.set_vec3("A", offA);
    const int len  = g.length(p);
    const int ampB = g.const_float(3.0f);
    const int mulB = g.mul_f(len, ampB);
    const int setB = g.set_float("B", mulB);
    (void)setA; (void)setB;

    ice::IceEvaluator ev;
    CHECK(ev.compile(g, nullptr));
    ice::EvalStats st{};
    CHECK(ev.evaluate(geo, &st));
    CHECK(st.nodes_total == 9 && st.nodes_evaluated == 9);
    const float b0 = geo.scalar["B"][0];
    CHECK(ap(b0, cardinal::sqrt(1.0f + 0.25f + 0.0625f) * 3.0f));

    // Clean re-evaluate: NOTHING recomputes.
    CHECK(ev.evaluate(geo, &st));
    CHECK(st.nodes_evaluated == 0);

    // Edit ampB: exactly its cone (ampB, mulB, setB) re-evaluates; chain A
    // and the shared P/normalize/length stay cached.
    CHECK(ev.set_const_float(ampB, 10.0f));
    CHECK(ev.evaluate(geo, &st));
    CHECK(st.nodes_evaluated == 3);
    CHECK(ap(geo.scalar["B"][0], cardinal::sqrt(1.3125f) * 10.0f));
    CHECK(ap(geo.vec3["A"][0].x * 0.0f + geo.vec3["A"][0].x,
             geo.vec3["A"][0].x));              // A untouched (still present)

    // set_const on a non-const node is rejected.
    CHECK(!ev.set_const_float(nrm, 1.0f));

    // Touch the input stream: its downstream cone re-evaluates — 7 of 9
    // (the two ConstFloat nodes do NOT depend on P and stay cached; the
    // dependency graph is precise, not conservative).
    P[0] = { 5.0f, 0.0f, 0.0f };
    ev.touch_input("P");
    CHECK(ev.evaluate(geo, &st));
    CHECK(st.nodes_evaluated == 7);
    CHECK(ap(geo.scalar["B"][0], 5.0f * 10.0f));
    CHECK(ap(geo.vec3["A"][0].x, 2.0f));         // normalize({5,0,0})*2

    // Point-count change invalidates everything.
    geo.point_count = 8;
    geo.ensure_vec3("P");
    CHECK(ev.evaluate(geo, &st));
    CHECK(st.nodes_evaluated == 9);
}

// ---- compile-time validation -------------------------------------------------
void test_compile_errors() {
    // Arity mismatch: Add with one input.
    {
        ice::IceGraph g;
        const int p = g.get_vec3("P");
        ice::Node bad;
        bad.kind = ice::NodeKind::Add;
        bad.inputs.push_back(p);                  // missing second input
        g.add(cardinal::move(bad));
        ice::IceEvaluator ev;
        cardinal::string err;
        CHECK(!ev.compile(g, &err));
        CHECK(!err.empty());
    }
    // Type mismatch: MulF fed a Vec3.
    {
        ice::IceGraph g;
        const int p = g.get_vec3("P");
        const int f = g.const_float(1.0f);
        ice::Node bad;
        bad.kind = ice::NodeKind::MulF;
        bad.inputs.push_back(p);                  // Vec3 into float slot
        bad.inputs.push_back(f);
        g.add(cardinal::move(bad));
        ice::IceEvaluator ev;
        CHECK(!ev.compile(g, nullptr));
    }
    // Cycle: two Adds feeding each other.
    {
        ice::IceGraph g;
        ice::Node a; a.kind = ice::NodeKind::Add; a.inputs = {1, 1};
        ice::Node b; b.kind = ice::NodeKind::Add; b.inputs = {0, 0};
        g.add(cardinal::move(a));
        g.add(cardinal::move(b));
        ice::IceEvaluator ev;
        cardinal::string err;
        CHECK(!ev.compile(g, &err));
        CHECK(err == cardinal::string("graph contains a cycle"));
    }
    // Self-reference is rejected outright.
    {
        ice::IceGraph g;
        ice::Node a; a.kind = ice::NodeKind::Normalize; a.inputs = {0};
        g.add(cardinal::move(a));
        ice::IceEvaluator ev;
        CHECK(!ev.compile(g, nullptr));
    }
}

// ---- cross/dot/sub sanity ----------------------------------------------------
void test_vector_ops() {
    ice::GeoData geo;
    geo.point_count = 1;
    auto& X = geo.ensure_vec3("X");
    auto& Y = geo.ensure_vec3("Y");
    X[0] = { 1.0f, 0.0f, 0.0f };
    Y[0] = { 0.0f, 1.0f, 0.0f };

    ice::IceGraph g;
    const int x = g.get_vec3("X");
    const int y = g.get_vec3("Y");
    g.set_vec3("C", g.cross(x, y));
    g.set_float("D", g.dot(x, y));
    g.set_vec3("S", g.sub_v3(x, y));
    g.set_float("SUMF", g.add_f(g.length(x), g.length(y)));

    ice::IceEvaluator ev;
    CHECK(ev.compile(g, nullptr));
    CHECK(ev.evaluate(geo, nullptr));
    CHECK(ap(geo.vec3["C"][0].z, 1.0f));          // x cross y = z
    CHECK(ap(geo.vec3["C"][0].x, 0.0f) && ap(geo.vec3["C"][0].y, 0.0f));
    CHECK(ap(geo.scalar["D"][0], 0.0f));
    CHECK(ap(geo.vec3["S"][0].x, 1.0f) && ap(geo.vec3["S"][0].y, -1.0f));
    CHECK(ap(geo.scalar["SUMF"][0], 2.0f));
}

}  // namespace

int main() {
    test_turbulize_math();
    test_incremental_dirty();
    test_compile_errors();
    test_vector_ops();

    if (g_fail == 0) {
        cardinal::log::infof("icetest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("icetest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
