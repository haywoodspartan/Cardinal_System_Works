// =============================================================================
// Cardinal Studio — SIMD diagnostic panel implementation.
//
// Renders the dispatch table + a tiny micro-benchmark. The bench
// allocates fixed-size scratch buffers ONCE on first run and re-uses
// them across re-bench clicks (no per-frame churn — this is a
// diagnostic tool, not a hot path).
// =============================================================================
#include <cardinal/ui/simd_panel.hpp>

#include <cardinal/core/hal.hpp>
#include <cardinal/core/simd_math.hpp>

#include <cardinal/ui/imgui.hpp>

#include <cardinal/core/chrono.hpp>
#include <cardinal/core/cmath.hpp>
#include <cardinal/core/cstdio.hpp>
#include <cardinal/core/limits.hpp>
#include <cardinal/core/containers.hpp>

namespace cardinal::ui {

namespace {

// Op-id labels mirroring cardinal::core::simd::OpId — kept in sync
// manually because the enum doesn't carry strings (overhead in the
// hot path; we only need them for diagnostics).
struct OpRow {
    cardinal::core::simd::OpId id;
    const char*                name;
    const char*                shape;       // "axpy(N)", "dot(N)", ...
};

constexpr OpRow kOps[] = {
    {cardinal::core::simd::OpId::VecAdd_f32,                 "vec_add_f32",                 "out[i]=a[i]+b[i]"},
    {cardinal::core::simd::OpId::VecSub_f32,                 "vec_sub_f32",                 "out[i]=a[i]-b[i]"},
    {cardinal::core::simd::OpId::VecMul_f32,                 "vec_mul_f32",                 "out[i]=a[i]*b[i]"},
    {cardinal::core::simd::OpId::VecScale_f32,               "vec_scale_f32",               "out[i]=a[i]*s"},
    {cardinal::core::simd::OpId::VecAxpy_f32,                "vec_axpy_f32",                "y[i]+=a*x[i]"},
    {cardinal::core::simd::OpId::Dot_f32,                    "dot_f32",                     "Σ a[i]*b[i]"},
    {cardinal::core::simd::OpId::Sum_f32,                    "sum_f32",                     "Σ a[i]"},
    {cardinal::core::simd::OpId::Min_f32,                    "min_f32",                     "min(a[i])"},
    {cardinal::core::simd::OpId::Max_f32,                    "max_f32",                     "max(a[i])"},
    {cardinal::core::simd::OpId::TransformPointsMat4,        "transform_points_mat4",       "M·{x,y,z,1}"},
    {cardinal::core::simd::OpId::Vec3Cross_Array,            "vec3_cross_array",            "a×b"},
    {cardinal::core::simd::OpId::Vec3Length_Array,           "vec3_length_array",           "‖v‖"},
    {cardinal::core::simd::OpId::Vec3Normalize_ArrayInplace, "vec3_normalize_array_inplace","v/‖v‖"},
    {cardinal::core::simd::OpId::Mat4Mul_Array,              "mat4_mul_array",              "A·B (4×4)"},
    {cardinal::core::simd::OpId::FrustumCullSpheres_f32,     "frustum_cull_spheres",        "6-plane reject"},
    {cardinal::core::simd::OpId::TransformAabbArray,         "transform_aabb_array",        "M·AABB (Arvo)"},
    {cardinal::core::simd::OpId::FrustumCullAabbs_f32,       "frustum_cull_aabbs",          "6-plane p-corner"},
    {cardinal::core::simd::OpId::RayAabbIntersect_f32,       "ray_aabb_intersect_array",    "1 ray vs N AABBs"},
    {cardinal::core::simd::OpId::MinIndex_f32,               "min_index_f32",               "argmin + value"},
    {cardinal::core::simd::OpId::MaxIndex_f32,               "max_index_f32",               "argmax + value"},
    {cardinal::core::simd::OpId::MortonEncode3d_u32,         "morton_encode_3d_array",      "Z-curve (x,y,z)→u64"},
};

struct BenchResult {
    bool   ran{false};
    double per_call_us{0.0};
    double throughput_gbps{0.0};   // effective memory bandwidth
};

struct BenchState {
    static constexpr int kFloatN  = 256 * 1024;     // 1 MiB of f32
    static constexpr int kPointN  = 64  * 1024;     // 768 KiB of Vec3
    static constexpr int kMatN    = 1024;           // 64 KiB of Mat4
    static constexpr int kSphereN = 64  * 1024;     // 64K spheres = 1MiB SoA
    static constexpr int kAabbN   = 64  * 1024;     // 64K AABBs = 1.5MiB SoA in
    static constexpr int kRepeats = 32;             // averaged

    cardinal::vector<float> a, b, y;
    cardinal::vector<float> pts_in, pts_out;
    cardinal::vector<float> mat_a, mat_b, mat_out;
    // Cull bench buffers — SoA: parallel cx/cy/cz/r arrays + output bits.
    cardinal::vector<float> sph_cx, sph_cy, sph_cz, sph_r;
    cardinal::vector<unsigned char> sph_bits;
    float              cull_planes[24]{};            // 6 outward planes
    // AABB bench buffers — 6 SoA arrays in + 6 out, plus bitmask for cull.
    cardinal::vector<float> abb_in_min_x,  abb_in_min_y,  abb_in_min_z;
    cardinal::vector<float> abb_in_max_x,  abb_in_max_y,  abb_in_max_z;
    cardinal::vector<float> abb_out_min_x, abb_out_min_y, abb_out_min_z;
    cardinal::vector<float> abb_out_max_x, abb_out_max_y, abb_out_max_z;
    cardinal::vector<unsigned char> abb_bits;
    cardinal::vector<float> ray_t;        // ray-vs-AABB output buffer
    // Morton bench buffers — 64K cells of (x,y,z) u32 → u64 codes.
    static constexpr int       kMortonN = 64 * 1024;
    cardinal::vector<unsigned int>  mort_x, mort_y, mort_z;
    cardinal::vector<unsigned long long> mort_out;
    BenchResult        results[static_cast<size_t>(cardinal::core::simd::OpId::Count)];
    bool               buffers_ready{false};
};

void prep_bench_buffers(BenchState& s) {
    if (s.buffers_ready) return;
    s.a.assign(s.kFloatN, 1.0f);
    s.b.assign(s.kFloatN, 2.0f);
    s.y.assign(s.kFloatN, 0.0f);
    s.pts_in .assign(s.kPointN * 3, 1.0f);
    s.pts_out.assign(s.kPointN * 3, 0.0f);
    s.mat_a  .assign(s.kMatN * 16, 0.0f);
    s.mat_b  .assign(s.kMatN * 16, 0.0f);
    s.mat_out.assign(s.kMatN * 16, 0.0f);
    // Identity-like contents so the result isn't a fp-junk firehose.
    for (int i = 0; i < s.kMatN; ++i) {
        for (int k = 0; k < 4; ++k) { s.mat_a[i*16 + k*4 + k] = 1.0f; s.mat_b[i*16 + k*4 + k] = 1.0f; }
    }
    // Cull bench: 64K spheres at slightly varying centers, all small
    // radius. The "frustum" is a unit cube around the origin (planes
    // pointing outward) — so most spheres fail the test, exercising
    // the per-plane AND-then-reject fast path.
    s.sph_cx.assign(s.kSphereN, 0.0f);
    s.sph_cy.assign(s.kSphereN, 0.0f);
    s.sph_cz.assign(s.kSphereN, 0.0f);
    s.sph_r .assign(s.kSphereN, 0.5f);
    for (int i = 0; i < s.kSphereN; ++i) {
        // Spread on a ring so visibility is mixed, not uniformly
        // accept-or-reject (which would let the predictor fold
        // everything into a constant).
        const float t = static_cast<float>(i) * 0.0001f;
        s.sph_cx[i] = cardinal::sin(t) * 5.0f;
        s.sph_cy[i] = static_cast<float>(i & 7) * 0.5f - 2.0f;
        s.sph_cz[i] = cardinal::cos(t) * 5.0f;
    }
    s.sph_bits.assign((s.kSphereN + 7) / 8, 0);
    // 6 outward-facing axis-aligned planes for a unit-cube frustum
    // around origin. Plane equation: n·p + d >= 0 inside.
    auto set_plane = [&](int p, float nx, float ny, float nz, float d) {
        s.cull_planes[p*4 + 0] = nx;
        s.cull_planes[p*4 + 1] = ny;
        s.cull_planes[p*4 + 2] = nz;
        s.cull_planes[p*4 + 3] = d;
    };
    set_plane(0, +1, 0, 0, 1);   // -x bound
    set_plane(1, -1, 0, 0, 1);   // +x bound
    set_plane(2, 0, +1, 0, 1);
    set_plane(3, 0, -1, 0, 1);
    set_plane(4, 0, 0, +1, 1);
    set_plane(5, 0, 0, -1, 1);
    // AABB bench: 64K small unit AABBs scattered on a ring (same shape
    // as the sphere bench so the cull workloads are comparable).
    s.abb_in_min_x.assign(s.kAabbN, 0.0f);
    s.abb_in_min_y.assign(s.kAabbN, 0.0f);
    s.abb_in_min_z.assign(s.kAabbN, 0.0f);
    s.abb_in_max_x.assign(s.kAabbN, 0.0f);
    s.abb_in_max_y.assign(s.kAabbN, 0.0f);
    s.abb_in_max_z.assign(s.kAabbN, 0.0f);
    s.abb_out_min_x.assign(s.kAabbN, 0.0f);
    s.abb_out_min_y.assign(s.kAabbN, 0.0f);
    s.abb_out_min_z.assign(s.kAabbN, 0.0f);
    s.abb_out_max_x.assign(s.kAabbN, 0.0f);
    s.abb_out_max_y.assign(s.kAabbN, 0.0f);
    s.abb_out_max_z.assign(s.kAabbN, 0.0f);
    for (int i = 0; i < s.kAabbN; ++i) {
        const float t = static_cast<float>(i) * 0.0001f;
        const float cx = cardinal::sin(t) * 5.0f;
        const float cy = static_cast<float>(i & 7) * 0.5f - 2.0f;
        const float cz = cardinal::cos(t) * 5.0f;
        s.abb_in_min_x[i] = cx - 0.4f; s.abb_in_max_x[i] = cx + 0.4f;
        s.abb_in_min_y[i] = cy - 0.4f; s.abb_in_max_y[i] = cy + 0.4f;
        s.abb_in_min_z[i] = cz - 0.4f; s.abb_in_max_z[i] = cz + 0.4f;
    }
    s.abb_bits.assign((s.kAabbN + 7) / 8, 0);
    s.ray_t.assign(s.kAabbN, 0.0f);
    // Morton input: spread cells over a 2K^3 grid so the bit-spread
    // chain actually has work to do (a constant input would compile
    // away to a trivial broadcast).
    s.mort_x.resize(s.kMortonN);
    s.mort_y.resize(s.kMortonN);
    s.mort_z.resize(s.kMortonN);
    for (int i = 0; i < s.kMortonN; ++i) {
        s.mort_x[i] = static_cast<unsigned>(i * 13u % 2048u);
        s.mort_y[i] = static_cast<unsigned>(i * 7u  % 2048u);
        s.mort_z[i] = static_cast<unsigned>(i * 19u % 2048u);
    }
    s.mort_out.assign(s.kMortonN, 0ull);
    s.buffers_ready = true;
}

// One-shot timer wrapper. Returns per-call microseconds.
template <class Fn>
double time_op(int repeats, Fn&& fn) {
    using clk = cardinal::chrono::high_resolution_clock;
    const auto t0 = clk::now();
    for (int r = 0; r < repeats; ++r) fn();
    const auto t1 = clk::now();
    return cardinal::chrono::duration<double, cardinal::micro>(t1 - t0).count() / repeats;
}

void run_bench(BenchState& s) {
    prep_bench_buffers(s);
    namespace simd = cardinal::core::simd;
    using OpId = simd::OpId;

    auto bytes_for = [&](OpId op) -> double {
        // Read+write bandwidth model — close enough for relative
        // comparisons. dot/sum/min/max are read-only over `a`.
        const double F = static_cast<double>(s.kFloatN) * sizeof(float);
        const double P = static_cast<double>(s.kPointN) * 3.0 * sizeof(float);
        const double M = static_cast<double>(s.kMatN)   * 16.0 * sizeof(float);
        switch (op) {
            case OpId::VecAdd_f32:
            case OpId::VecSub_f32:
            case OpId::VecMul_f32:                       return 3.0 * F;
            case OpId::VecScale_f32:                     return 2.0 * F;
            case OpId::VecAxpy_f32:                      return 3.0 * F; // r-x, r-y, w-y
            case OpId::Dot_f32:                          return 2.0 * F;
            case OpId::Sum_f32:
            case OpId::Min_f32:
            case OpId::Max_f32:                          return 1.0 * F;
            case OpId::TransformPointsMat4:              return 2.0 * P;
            case OpId::Vec3Cross_Array:                  return 3.0 * P;
            case OpId::Vec3Length_Array:                 return 1.0 * P + s.kPointN * sizeof(float);
            case OpId::Vec3Normalize_ArrayInplace:       return 2.0 * P;
            case OpId::Mat4Mul_Array:                    return 3.0 * M;
            case OpId::FrustumCullSpheres_f32: {
                // 4 SoA float arrays read + bitmask write (1 bit / sphere).
                const double S = static_cast<double>(s.kSphereN) * sizeof(float);
                const double bits = static_cast<double>(s.kSphereN) / 8.0;
                return 4.0 * S + bits;
            }
            case OpId::TransformAabbArray: {
                // 6 SoA in + 6 SoA out, all f32.
                const double A = static_cast<double>(s.kAabbN) * sizeof(float);
                return 12.0 * A;
            }
            case OpId::FrustumCullAabbs_f32: {
                // 6 SoA float arrays read + bitmask write (1 bit / AABB).
                const double A = static_cast<double>(s.kAabbN) * sizeof(float);
                const double bits = static_cast<double>(s.kAabbN) / 8.0;
                return 6.0 * A + bits;
            }
            case OpId::RayAabbIntersect_f32: {
                // 6 SoA float arrays read + 1 float-per-AABB t output write.
                const double A = static_cast<double>(s.kAabbN) * sizeof(float);
                return 7.0 * A;
            }
            case OpId::MinIndex_f32:
            case OpId::MaxIndex_f32: {
                // One pass over the float-array buffer; output is two
                // scalars (value + index) — negligible vs the read.
                return 1.0 * static_cast<double>(s.kFloatN) * sizeof(float);
            }
            case OpId::MortonEncode3d_u32: {
                // 3 SoA u32 arrays read + 1 u64 array written.
                const double N = static_cast<double>(s.kMortonN);
                return 3.0 * N * sizeof(unsigned) + N * sizeof(unsigned long long);
            }
            case OpId::Count:                            return 0.0;
        }
        return 0.0;
    };

    auto run = [&](OpId op, auto&& body) {
        BenchResult& r = s.results[static_cast<size_t>(op)];
        r.per_call_us     = time_op(s.kRepeats, body);
        r.throughput_gbps = (bytes_for(op) / 1e9) / (r.per_call_us * 1e-6);
        r.ran             = true;
    };

    using namespace cardinal::core::simd;
    cardinal::core::simd::Mat4Compact M{};
    for (int k = 0; k < 4; ++k) M.m[k*4 + k] = 1.0f;

    run(OpId::VecAdd_f32,           [&]{ vec_add_f32  (s.y.data(), s.a.data(), s.b.data(), s.kFloatN); });
    run(OpId::VecSub_f32,           [&]{ vec_sub_f32  (s.y.data(), s.a.data(), s.b.data(), s.kFloatN); });
    run(OpId::VecMul_f32,           [&]{ vec_mul_f32  (s.y.data(), s.a.data(), s.b.data(), s.kFloatN); });
    run(OpId::VecScale_f32,         [&]{ vec_scale_f32(s.y.data(), s.a.data(), 1.5f, s.kFloatN); });
    run(OpId::VecAxpy_f32,          [&]{ vec_axpy_f32 (s.y.data(), 0.5f, s.a.data(), s.kFloatN); });
    run(OpId::Dot_f32,              [&]{ volatile float v = dot_f32(s.a.data(), s.b.data(), s.kFloatN); (void)v; });
    run(OpId::Sum_f32,              [&]{ volatile float v = sum_f32(s.a.data(), s.kFloatN); (void)v; });
    run(OpId::Min_f32,              [&]{ volatile float v = min_f32(s.a.data(), s.kFloatN); (void)v; });
    run(OpId::Max_f32,              [&]{ volatile float v = max_f32(s.a.data(), s.kFloatN); (void)v; });
    run(OpId::TransformPointsMat4,  [&]{ transform_points_mat4(s.pts_out.data(), s.pts_in.data(), M, s.kPointN); });
    run(OpId::Vec3Cross_Array,      [&]{ vec3_cross_array (s.pts_out.data(), s.pts_in.data(), s.pts_in.data(), s.kPointN); });
    run(OpId::Vec3Length_Array,     [&]{ vec3_length_array(s.y.data(), s.pts_in.data(), s.kPointN); });
    run(OpId::Vec3Normalize_ArrayInplace, [&]{ vec3_normalize_array_inplace(s.pts_out.data(), s.kPointN); });
    run(OpId::Mat4Mul_Array,        [&]{ mat4_mul_array(s.mat_out.data(), s.mat_a.data(), s.mat_b.data(), s.kMatN); });
    run(OpId::FrustumCullSpheres_f32, [&]{
        frustum_cull_spheres(s.sph_bits.data(), s.cull_planes,
            s.sph_cx.data(), s.sph_cy.data(), s.sph_cz.data(), s.sph_r.data(),
            static_cast<usize>(s.kSphereN));
    });
    run(OpId::TransformAabbArray, [&]{
        transform_aabb_array(
            s.abb_out_min_x.data(), s.abb_out_min_y.data(), s.abb_out_min_z.data(),
            s.abb_out_max_x.data(), s.abb_out_max_y.data(), s.abb_out_max_z.data(),
            s.abb_in_min_x .data(), s.abb_in_min_y .data(), s.abb_in_min_z .data(),
            s.abb_in_max_x .data(), s.abb_in_max_y .data(), s.abb_in_max_z .data(),
            M, static_cast<usize>(s.kAabbN));
    });
    run(OpId::FrustumCullAabbs_f32, [&]{
        frustum_cull_aabbs(s.abb_bits.data(), s.cull_planes,
            s.abb_in_min_x.data(), s.abb_in_min_y.data(), s.abb_in_min_z.data(),
            s.abb_in_max_x.data(), s.abb_in_max_y.data(), s.abb_in_max_z.data(),
            static_cast<usize>(s.kAabbN));
    });
    run(OpId::RayAabbIntersect_f32, [&]{
        // Bench ray: shoot from (0,0,-10) along +z. Most AABBs in the
        // scattered ring miss; some hit. Pre-computed inv_dir keeps
        // the kernel inputs realistic (callers typically share the
        // same ray across many AABBs).
        const float ox = 0.0f, oy = 0.0f, oz = -10.0f;
        const float inv_dx = cardinal::numeric_limits<float>::infinity();   // dx = 0
        const float inv_dy = cardinal::numeric_limits<float>::infinity();   // dy = 0
        const float inv_dz = 1.0f;                                     // dz = 1
        ray_aabb_intersect_array(s.ray_t.data(),
            ox, oy, oz, inv_dx, inv_dy, inv_dz,
            s.abb_in_min_x.data(), s.abb_in_min_y.data(), s.abb_in_min_z.data(),
            s.abb_in_max_x.data(), s.abb_in_max_y.data(), s.abb_in_max_z.data(),
            static_cast<usize>(s.kAabbN));
    });
    run(OpId::MinIndex_f32, [&]{
        // Reduce over the kFloatN buffer (1 MiB of f32). Realistic
        // workload for downstream "argmin over a results array" calls.
        volatile float v = 0.0f; volatile cardinal::u32 i = 0;
        min_index_f32(const_cast<float*>(&v), const_cast<cardinal::u32*>(&i),
                      s.a.data(), static_cast<usize>(s.kFloatN));
    });
    run(OpId::MaxIndex_f32, [&]{
        volatile float v = 0.0f; volatile cardinal::u32 i = 0;
        max_index_f32(const_cast<float*>(&v), const_cast<cardinal::u32*>(&i),
                      s.a.data(), static_cast<usize>(s.kFloatN));
    });
    run(OpId::MortonEncode3d_u32, [&]{
        morton_encode_3d_array(
            reinterpret_cast<cardinal::u64*>(s.mort_out.data()),
            reinterpret_cast<const cardinal::u32*>(s.mort_x.data()),
            reinterpret_cast<const cardinal::u32*>(s.mort_y.data()),
            reinterpret_cast<const cardinal::u32*>(s.mort_z.data()),
            static_cast<usize>(s.kMortonN));
    });
}

}  // namespace

void draw_simd_panel(bool* p_open) {
    if (!ImGui::Begin("SIMD Math", p_open, ImGuiWindowFlags_AlwaysAutoResize))
    { ImGui::End(); return; }

    namespace simd = cardinal::core::simd;
    const auto& cf = cardinal::hal::cpu_features();

    // ----- Header: CPU + active backend ------------------------------------
    ImGui::TextDisabled("Vendor");      ImGui::SameLine(120); ImGui::Text("%s", cf.vendor[0] ? cf.vendor : "(unknown)");
    ImGui::TextDisabled("Brand");       ImGui::SameLine(120); ImGui::Text("%s", cf.brand[0]  ? cf.brand  : "(unknown)");
    ImGui::TextDisabled("Active");      ImGui::SameLine(120);
    ImGui::TextColored(ImVec4(0.4f, 0.85f, 0.45f, 1.0f), "%s",
        simd::backend_name(simd::active_backend()));

    ImGui::Separator();

    // ----- Feature flags grid (CPUID bits) ---------------------------------
    auto flag = [&](const char* label, bool yes) {
        ImGui::TextColored(yes ? ImVec4(0.4f, 0.85f, 0.45f, 1.0f)
                               : ImVec4(0.55f, 0.55f, 0.55f, 1.0f),
                           "%s", label);
    };
    ImGui::TextDisabled("CPUID feature bits:");
    flag("SSE2",      cf.sse2);    ImGui::SameLine();
    flag("SSE3",      cf.sse3);    ImGui::SameLine();
    flag("SSSE3",     cf.ssse3);   ImGui::SameLine();
    flag("SSE4.1",    cf.sse41);   ImGui::SameLine();
    flag("SSE4.2",    cf.sse42);   ImGui::SameLine();
    flag("SSE4A",     cf.sse4a);
    flag("AVX",       cf.avx);     ImGui::SameLine();
    flag("AVX2",      cf.avx2);    ImGui::SameLine();
    flag("FMA3",      cf.fma3);    ImGui::SameLine();
    flag("FMA4",      cf.fma4);    ImGui::SameLine();
    flag("F16C",      cf.f16c);    ImGui::SameLine();
    flag("BMI1",      cf.bmi1);    ImGui::SameLine();
    flag("BMI2",      cf.bmi2);
    flag("AVX-512F",  cf.avx512f);  ImGui::SameLine();
    flag("AVX-512VL", cf.avx512vl); ImGui::SameLine();
    flag("AVX-512BW", cf.avx512bw); ImGui::SameLine();
    flag("AVX-512DQ", cf.avx512dq); ImGui::SameLine();
    flag("LZCNT",     cf.lzcnt);    ImGui::SameLine();
    flag("POPCNT",    cf.popcnt);

    ImGui::Separator();

    // ----- Per-op binding table -------------------------------------------
    static BenchState s_bench{};
    if (ImGui::BeginTable("##simd_ops", 4,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
            ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("Op",          ImGuiTableColumnFlags_WidthFixed, 200.0f);
        ImGui::TableSetupColumn("Shape",       ImGuiTableColumnFlags_WidthFixed, 140.0f);
        ImGui::TableSetupColumn("Backend",     ImGuiTableColumnFlags_WidthFixed, 120.0f);
        ImGui::TableSetupColumn("Bench (µs / GB·s)", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        const auto active = simd::active_backend();
        for (const auto& row : kOps) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("%s", row.name);
            ImGui::TableSetColumnIndex(1); ImGui::TextDisabled("%s", row.shape);
            ImGui::TableSetColumnIndex(2);
            const auto op_be = simd::backend_for(row.id);
            const bool fallback = (op_be != active);
            if (fallback) {
                // Per-op fallback path — paint amber so the asymmetry
                // is visible (e.g. mat4_mul_array on AVX-only CPUs).
                ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.30f, 1.0f),
                    "%s ↓", simd::backend_name(op_be));
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "This op fell back from %s to %s — the higher tier\n"
                        "doesn't ship a kernel for it. The dispatcher's\n"
                        "tier-walk landed on the next-lower implementation.",
                        simd::backend_name(active), simd::backend_name(op_be));
                }
            } else {
                ImGui::TextColored(ImVec4(0.4f, 0.85f, 0.45f, 1.0f),
                    "%s", simd::backend_name(op_be));
            }
            ImGui::TableSetColumnIndex(3);
            const auto& r = s_bench.results[static_cast<size_t>(row.id)];
            if (r.ran) {
                ImGui::Text("%6.2f µs   %5.1f GB/s", r.per_call_us, r.throughput_gbps);
            } else {
                ImGui::TextDisabled("(not benchmarked)");
            }
        }
        ImGui::EndTable();
    }

    ImGui::Separator();
    if (ImGui::Button("Run benchmark")) {
        run_bench(s_bench);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("256K floats / 64K verts / 1024 mats × 32 reps avg");

    ImGui::End();
}

}  // namespace cardinal::ui
