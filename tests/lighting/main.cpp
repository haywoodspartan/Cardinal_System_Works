// =============================================================================
// Cardinal — RTX path-tracer regression suite.
//
// The lighting module ships two CPU references:
//   * lighting.cpp  — geometry intersection + Cook-Torrance BRDF + RNG
//   * raytracer.cpp — recursive path tracer with NEE + RR termination
//
// A regression in either silently changes every shaded pixel, so this
// suite pins the BRDF math on canonical inputs, scene intersection on
// known sphere / plane layouts, direct shading against every LightKind
// (directional / point / spot) with occlusion, full-image rendering
// determinism (same seed → same bytes), Knob-driven editor surface,
// and the NaN-defensive invariant: corrupt ray / material / light
// produces FINITE output (no framebuffer poison).
//
// Pure CPU + headless. Exit 0 = all pass.
// =============================================================================

#include <cardinal/lighting/lighting.hpp>
#include <cardinal/lighting/raytracer.hpp>
#include <cardinal/lighting/bake.hpp>
#include <cardinal/lighting/gpu_lighting.hpp>
#include <cardinal/lighting/gpu_radiance_cache.hpp>
#include <cardinal/render/graph.hpp>
#include <cardinal/core/log.hpp>
#include <cardinal/core/utility.hpp>

#include <limits>

namespace {

namespace lx = cardinal::lighting;
namespace sc = cardinal::scene;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("rtxtest", "FAIL L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-3f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
bool fin(float v) { return v == v && (v - v) == 0.0f; }
bool finv(const sc::Vec3& v) { return fin(v.x) && fin(v.y) && fin(v.z); }

// Three-element scene: a red sphere above a grey floor.
lx::Scene make_basic_scene(sc::LightSet& lights) {
    lx::Scene s;
    s.materials.push_back(lx::Material{sc::Vec3{0.8f, 0.2f, 0.2f}, sc::Vec3{0, 0, 0}, 0.0f, 0.6f});  // red diffuse
    s.materials.push_back(lx::Material{sc::Vec3{0.6f, 0.6f, 0.6f}, sc::Vec3{0, 0, 0}, 0.0f, 0.8f});  // grey floor
    s.spheres.push_back(lx::Sphere{sc::Vec3{0, 1, 0}, 0.5f, 0u});
    s.planes.push_back (lx::Plane {sc::Vec3{0, 0, 0}, sc::Vec3{0, 1, 0}, 1u});
    s.lights = &lights;
    return s;
}

// ----------------------------------------------------------------------------
// trace_closest — sphere + plane geometry.
// ----------------------------------------------------------------------------
void test_trace_closest_sphere_hit() {
    sc::LightSet lights;
    lx::Scene s = make_basic_scene(lights);
    sc::Ray ray{sc::Vec3{0, 1, -5}, sc::Vec3{0, 0, 1}};   // looking down +Z
    lx::RayHit h;
    CHECK(lx::trace_closest(ray, s, h));
    CHECK(h.material_idx == 0u);                          // red sphere
    CHECK(ap(h.t, 4.5f, 1e-3f));                          // entry at z = -0.5
    CHECK(ap(h.position.z, -0.5f, 1e-3f));
    CHECK(ap(sc::length(h.normal), 1.0f, 1e-3f));
    CHECK(h.normal.z < 0.0f);                             // outward toward camera
}

void test_trace_closest_plane_hit() {
    sc::LightSet lights;
    lx::Scene s = make_basic_scene(lights);
    sc::Ray ray{sc::Vec3{5, 2, 0}, sc::Vec3{0, -1, 0}};   // straight down, miss sphere
    lx::RayHit h;
    CHECK(lx::trace_closest(ray, s, h));
    CHECK(h.material_idx == 1u);                          // floor
    CHECK(ap(h.t, 2.0f, 1e-3f));
    CHECK(ap(h.position.y, 0.0f, 1e-3f));
    CHECK(h.normal.y > 0.0f);                             // up
}

void test_trace_closest_miss() {
    sc::LightSet lights;
    lx::Scene s = make_basic_scene(lights);
    sc::Ray ray{sc::Vec3{0, 10, 0}, sc::Vec3{0, 1, 0}};   // straight up, hits nothing
    lx::RayHit h;
    CHECK(!lx::trace_closest(ray, s, h));
}

void test_trace_closest_closest_wins() {
    // Two spheres on the ray's path — the nearer must be reported.
    sc::LightSet lights;
    lx::Scene s;
    s.materials.push_back(lx::Material{sc::Vec3{1, 0, 0}, sc::Vec3{0, 0, 0}, 0.0f, 0.5f});
    s.materials.push_back(lx::Material{sc::Vec3{0, 1, 0}, sc::Vec3{0, 0, 0}, 0.0f, 0.5f});
    s.spheres.push_back(lx::Sphere{sc::Vec3{0, 0,  5}, 1.0f, 0u});   // near
    s.spheres.push_back(lx::Sphere{sc::Vec3{0, 0, 10}, 1.0f, 1u});   // far
    s.lights = &lights;
    sc::Ray ray{sc::Vec3{0, 0, 0}, sc::Vec3{0, 0, 1}};
    lx::RayHit h;
    CHECK(lx::trace_closest(ray, s, h));
    CHECK(h.material_idx == 0u);                          // near red
    CHECK(ap(h.t, 4.0f, 1e-3f));
}

void test_trace_closest_nan_ray_is_miss() {
    // NaN ray origin or direction must report "miss" — never a crash, never
    // a poisoned RayHit.
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    sc::LightSet lights;
    lx::Scene s = make_basic_scene(lights);
    {
        sc::Ray ray{sc::Vec3{qnan, qnan, qnan}, sc::Vec3{0, 0, 1}};
        lx::RayHit h;
        CHECK(!lx::trace_closest(ray, s, h));
    }
    {
        sc::Ray ray{sc::Vec3{0, 0, 0}, sc::Vec3{qnan, qnan, qnan}};
        lx::RayHit h;
        CHECK(!lx::trace_closest(ray, s, h));
    }
}

// ----------------------------------------------------------------------------
// visible — occlusion.
// ----------------------------------------------------------------------------
void test_visible_unobstructed() {
    sc::LightSet lights;
    lx::Scene s = make_basic_scene(lights);
    // Floor point (0,0,3) → sky (0,1,0). Sphere at (0,1,0) is to the side.
    CHECK(lx::visible(sc::Vec3{5, 0, 0}, sc::Vec3{0, 1, 0}, 100.0f, s));
}

void test_visible_blocked() {
    sc::LightSet lights;
    lx::Scene s = make_basic_scene(lights);
    // Floor point directly under the sphere, ray straight up → blocked.
    CHECK(!lx::visible(sc::Vec3{0, 0, 0}, sc::Vec3{0, 1, 0}, 100.0f, s));
}

void test_visible_nan_returns_false_safely() {
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    sc::LightSet lights;
    lx::Scene s = make_basic_scene(lights);
    // NaN origin / direction must NOT crash. Conservative answer is fine.
    CHECK(!lx::visible(sc::Vec3{qnan, qnan, qnan}, sc::Vec3{0, 1, 0}, 100.0f, s));
}

// ----------------------------------------------------------------------------
// BRDF — Cook-Torrance correctness on canonical inputs.
// ----------------------------------------------------------------------------
void test_brdf_directly_lit_diffuse() {
    // Perfect diffuse (metallic=0, roughness=1): straight-down light, normal up,
    // camera straight up. Diffuse term = base/π; specular at high roughness ≈ 0.
    lx::Material m;
    m.base_color = {1, 1, 1};
    m.metallic   = 0.0f;
    m.roughness  = 1.0f;
    const sc::Vec3 n{0, 1, 0};
    const sc::Vec3 v{0, 1, 0};
    const sc::Vec3 l{0, 1, 0};
    const sc::Vec3 f = lx::brdf(n, v, l, m);
    CHECK(ap(f.x, 1.0f / sc::kPi, 0.01f));
    CHECK(ap(f.y, 1.0f / sc::kPi, 0.01f));
    CHECK(ap(f.z, 1.0f / sc::kPi, 0.01f));
}

void test_brdf_below_horizon_is_zero() {
    // Light below the surface: BRDF must clamp to zero (NdotL <= 0).
    lx::Material m;
    const sc::Vec3 n{0, 1, 0};
    const sc::Vec3 v{0, 1, 0};
    const sc::Vec3 l{0, -1, 0};
    const sc::Vec3 f = lx::brdf(n, v, l, m);
    CHECK(f.x == 0.0f); CHECK(f.y == 0.0f); CHECK(f.z == 0.0f);
}

void test_brdf_metal_specular_at_mirror() {
    // Metal at mirror config (v == l reflected about n): non-zero specular.
    lx::Material m;
    m.base_color = {1.0f, 0.84f, 0.0f};   // gold-ish
    m.metallic   = 1.0f;
    m.roughness  = 0.05f;
    const sc::Vec3 n{0, 1, 0};
    const sc::Vec3 v{0, 1, 0};
    const sc::Vec3 l{0, 1, 0};            // both straight up — h = n exactly
    const sc::Vec3 f = lx::brdf(n, v, l, m);
    CHECK(finv(f));
    CHECK(f.x > 0.0f);
    CHECK(f.y > 0.0f);
    // Diffuse mix is zero for full-metal → x/y tint follows base_color.
    CHECK(f.x > f.z);                     // red > blue (gold)
}

void test_brdf_nan_inputs_safe() {
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    lx::Material m;
    {
        const sc::Vec3 f = lx::brdf(sc::Vec3{qnan, qnan, qnan}, sc::Vec3{0, 1, 0},
                                    sc::Vec3{0, 1, 0}, m);
        CHECK(finv(f));
        CHECK(f.x == 0.0f);
    }
    {
        m.base_color = {qnan, qnan, qnan};
        m.roughness  = qnan;
        m.metallic   = qnan;
        const sc::Vec3 f = lx::brdf(sc::Vec3{0, 1, 0}, sc::Vec3{0, 1, 0},
                                    sc::Vec3{0, 1, 0}, m);
        CHECK(finv(f));
    }
}

// ----------------------------------------------------------------------------
// shade_direct — every LightKind under occlusion.
// ----------------------------------------------------------------------------
void test_shade_direct_directional_no_occlusion() {
    sc::LightSet lights;
    lights.set_ambient(sc::Vec3{0, 0, 0});                // isolate the directional
    sc::Light L;
    L.kind = sc::LightKind::Directional;
    L.direction = sc::Vec3{0, -1, 0};                     // pointing straight down
    L.color = sc::Vec3{1, 1, 1};
    L.intensity = 1.0f;
    lights.add(L);

    lx::Scene s;
    s.materials.push_back(lx::Material{sc::Vec3{1, 1, 1}, sc::Vec3{0, 0, 0}, 0.0f, 1.0f});
    s.planes.push_back(lx::Plane{sc::Vec3{0, 0, 0}, sc::Vec3{0, 1, 0}, 0u});
    s.lights = &lights;

    auto tracer = lx::PathTracer::create();
    lx::RayHit hit;
    hit.position = sc::Vec3{5, 0, 5};
    hit.normal   = sc::Vec3{0, 1, 0};
    hit.material_idx = 0u;
    hit.t = 1.0f;
    const sc::Vec3 v{0, 1, 0};
    const sc::Vec3 r = tracer->shade_direct(hit, v, s);
    CHECK(finv(r));
    CHECK(r.x > 0.0f);
    CHECK(ap(r.x, r.y, 1e-4f));
    CHECK(ap(r.x, r.z, 1e-4f));            // white light, white surface
}

void test_shade_direct_point_attenuation() {
    sc::LightSet lights;
    lights.set_ambient(sc::Vec3{0, 0, 0});
    sc::Light L;
    L.kind = sc::LightKind::Point;
    L.position = sc::Vec3{0, 2, 0};
    L.color = sc::Vec3{1, 1, 1};
    L.intensity = 1.0f;
    L.range = 10.0f;
    lights.add(L);

    lx::Scene s;
    s.materials.push_back(lx::Material{sc::Vec3{1, 1, 1}, sc::Vec3{0, 0, 0}, 0.0f, 1.0f});
    s.planes.push_back(lx::Plane{sc::Vec3{0, 0, 0}, sc::Vec3{0, 1, 0}, 0u});
    s.lights = &lights;

    auto tracer = lx::PathTracer::create();
    lx::RayHit h_near, h_far;
    h_near.position = sc::Vec3{0.1f, 0, 0};
    h_near.normal   = sc::Vec3{0, 1, 0};
    h_near.material_idx = 0u;
    h_far.position  = sc::Vec3{5, 0, 0};
    h_far.normal    = sc::Vec3{0, 1, 0};
    h_far.material_idx = 0u;
    const sc::Vec3 v{0, 1, 0};
    const sc::Vec3 rn = tracer->shade_direct(h_near, v, s);
    const sc::Vec3 rf = tracer->shade_direct(h_far,  v, s);
    CHECK(finv(rn)); CHECK(finv(rf));
    CHECK(rn.x > rf.x);                    // closer = brighter
}

void test_shade_direct_spot_cone_falloff() {
    // Spot pointing straight down with a tight cone. Center is lit; far off-axis is dark.
    sc::LightSet lights;
    lights.set_ambient(sc::Vec3{0, 0, 0});
    sc::Light L;
    L.kind = sc::LightKind::Spot;
    L.position = sc::Vec3{0, 5, 0};
    L.direction = sc::Vec3{0, -1, 0};
    L.color = sc::Vec3{1, 1, 1};
    L.intensity = 5.0f;
    L.range = 100.0f;
    L.spot_inner_cos = 0.99f;
    L.spot_outer_cos = 0.95f;
    lights.add(L);

    lx::Scene s;
    s.materials.push_back(lx::Material{sc::Vec3{1, 1, 1}, sc::Vec3{0, 0, 0}, 0.0f, 1.0f});
    s.planes.push_back(lx::Plane{sc::Vec3{0, 0, 0}, sc::Vec3{0, 1, 0}, 0u});
    s.lights = &lights;

    auto tracer = lx::PathTracer::create();
    lx::RayHit h_center, h_outside;
    h_center.position = sc::Vec3{0, 0, 0};
    h_center.normal   = sc::Vec3{0, 1, 0};
    h_center.material_idx = 0u;
    h_outside.position = sc::Vec3{20, 0, 0};
    h_outside.normal   = sc::Vec3{0, 1, 0};
    h_outside.material_idx = 0u;
    const sc::Vec3 v{0, 1, 0};
    const sc::Vec3 rc = tracer->shade_direct(h_center, v, s);
    const sc::Vec3 ro = tracer->shade_direct(h_outside, v, s);
    CHECK(finv(rc)); CHECK(finv(ro));
    CHECK(rc.x > 0.0f);
    CHECK(ro.x == 0.0f);                   // outside cone — dark
}

void test_shade_direct_occlusion() {
    // A blocker between the floor and a point light → floor stays dark.
    sc::LightSet lights;
    lights.set_ambient(sc::Vec3{0, 0, 0});
    sc::Light L;
    L.kind = sc::LightKind::Point;
    L.position = sc::Vec3{0, 5, 0};
    L.color = sc::Vec3{1, 1, 1};
    L.intensity = 5.0f;
    L.range = 20.0f;
    lights.add(L);

    lx::Scene s;
    s.materials.push_back(lx::Material{sc::Vec3{1, 1, 1}, sc::Vec3{0, 0, 0}, 0.0f, 1.0f});
    s.materials.push_back(lx::Material{sc::Vec3{1, 0, 0}, sc::Vec3{0, 0, 0}, 0.0f, 1.0f});
    s.planes.push_back(lx::Plane{sc::Vec3{0, 0, 0}, sc::Vec3{0, 1, 0}, 0u});
    s.spheres.push_back(lx::Sphere{sc::Vec3{0, 2, 0}, 1.0f, 1u});   // blocker
    s.lights = &lights;

    auto tracer = lx::PathTracer::create();
    lx::RayHit h_shadow, h_lit;
    h_shadow.position = sc::Vec3{0, 0, 0};
    h_shadow.normal   = sc::Vec3{0, 1, 0};
    h_shadow.material_idx = 0u;
    h_lit.position = sc::Vec3{5, 0, 0};                   // off to the side, unobscured
    h_lit.normal   = sc::Vec3{0, 1, 0};
    h_lit.material_idx = 0u;
    const sc::Vec3 v{0, 1, 0};
    const sc::Vec3 rs = tracer->shade_direct(h_shadow, v, s);
    const sc::Vec3 rl = tracer->shade_direct(h_lit, v, s);
    CHECK(finv(rs)); CHECK(finv(rl));
    CHECK(rs.x == 0.0f);                   // in shadow
    CHECK(rl.x > 0.0f);                    // lit
}

void test_shade_direct_emissive_adds() {
    // Emissive material contributes even with no lights.
    sc::LightSet lights;
    lights.set_ambient(sc::Vec3{0, 0, 0});
    lx::Scene s;
    s.materials.push_back(lx::Material{sc::Vec3{0, 0, 0}, sc::Vec3{2, 3, 4}, 0.0f, 1.0f});
    s.planes.push_back(lx::Plane{sc::Vec3{0, 0, 0}, sc::Vec3{0, 1, 0}, 0u});
    s.lights = &lights;
    auto tracer = lx::PathTracer::create();
    lx::RayHit hit;
    hit.position = sc::Vec3{0, 0, 0};
    hit.normal   = sc::Vec3{0, 1, 0};
    hit.material_idx = 0u;
    const sc::Vec3 v{0, 1, 0};
    const sc::Vec3 r = tracer->shade_direct(hit, v, s);
    CHECK(ap(r.x, 2.0f, 1e-3f));
    CHECK(ap(r.y, 3.0f, 1e-3f));
    CHECK(ap(r.z, 4.0f, 1e-3f));
}

// ----------------------------------------------------------------------------
// trace_radiance — recursive path trace.
// ----------------------------------------------------------------------------
void test_trace_radiance_miss_returns_sky() {
    sc::LightSet lights;
    lx::Scene s;
    s.lights = &lights;
    lx::PathTracerConfig cfg;
    cfg.sky_color = sc::Vec3{0.5f, 0.25f, 0.125f};
    auto tracer = lx::PathTracer::create(cfg);
    lx::Rng rng(42);
    sc::Ray ray{sc::Vec3{0, 0, 0}, sc::Vec3{0, 1, 0}};
    const sc::Vec3 r = tracer->trace_radiance(ray, s, rng);
    CHECK(finv(r));
    CHECK(ap(r.x, 0.5f,   1e-3f));
    CHECK(ap(r.y, 0.25f,  1e-3f));
    CHECK(ap(r.z, 0.125f, 1e-3f));
    CHECK(tracer->stats().misses == 1u);
}

void test_trace_radiance_direct_only_is_finite() {
    // max_bounces=0 → direct lighting only; finite + nonzero on a lit floor.
    sc::LightSet lights;
    lights.set_ambient(sc::Vec3{0.1f, 0.1f, 0.1f});
    sc::Light L; L.kind = sc::LightKind::Directional;
    L.direction = sc::Vec3{0, -1, 0}; L.color = sc::Vec3{1, 1, 1}; L.intensity = 1.0f;
    lights.add(L);

    lx::Scene s = make_basic_scene(lights);
    lx::PathTracerConfig cfg;
    cfg.max_bounces = 0;
    cfg.samples_per_pixel = 1;
    cfg.enable_indirect = false;
    auto tracer = lx::PathTracer::create(cfg);
    lx::Rng rng(7);
    sc::Ray ray{sc::Vec3{0, 1, -5}, sc::Vec3{0, 0, 1}};   // hit the sphere
    const sc::Vec3 r = tracer->trace_radiance(ray, s, rng);
    CHECK(finv(r));
    CHECK(r.x > 0.0f);
}

void test_trace_radiance_indirect_brighter_than_direct_in_closed_box() {
    // With multiple bounces, an emissive box wall should add to the floor's
    // radiance vs a direct-only render.
    sc::LightSet lights;
    lights.set_ambient(sc::Vec3{0, 0, 0});                // ambient = 0 to isolate effect
    sc::Light L; L.kind = sc::LightKind::Directional;
    L.direction = sc::Vec3{0, -1, 0}; L.color = sc::Vec3{1, 1, 1}; L.intensity = 1.0f;
    lights.add(L);

    lx::Scene s;
    // White diffuse floor.
    s.materials.push_back(lx::Material{sc::Vec3{0.9f, 0.9f, 0.9f}, sc::Vec3{0, 0, 0}, 0.0f, 1.0f});
    s.planes.push_back(lx::Plane{sc::Vec3{0, 0, 0}, sc::Vec3{0, 1, 0}, 0u});
    s.lights = &lights;

    auto tracer_direct = lx::PathTracer::create(lx::PathTracerConfig{
        /*max_bounces*/ 0, /*spp*/ 32, /*rr_depth*/ 2, /*indirect*/ false, /*nee*/ true,
        /*seed*/ 1u, sc::Vec3{0, 0, 0},
    });
    auto tracer_indirect = lx::PathTracer::create(lx::PathTracerConfig{
        /*max_bounces*/ 2, /*spp*/ 32, /*rr_depth*/ 2, /*indirect*/ true,  /*nee*/ true,
        /*seed*/ 1u, sc::Vec3{0.4f, 0.4f, 0.4f},                              // bright sky
    });

    lx::Rng rng_a(1), rng_b(1);
    sc::Vec3 acc_a{0, 0, 0}, acc_b{0, 0, 0};
    for (int i = 0; i < 64; ++i) {
        sc::Ray ray{sc::Vec3{5, 0.5f, 5}, sc::Vec3{0, -0.3f, 0}};
        ray.direction = sc::normalize(sc::Vec3{0, -1, 0});
        acc_a = lx::add(acc_a, tracer_direct  ->trace_radiance(ray, s, rng_a));
        acc_b = lx::add(acc_b, tracer_indirect->trace_radiance(ray, s, rng_b));
    }
    const float inv = 1.0f / 64.0f;
    const sc::Vec3 mean_a{acc_a.x * inv, acc_a.y * inv, acc_a.z * inv};
    const sc::Vec3 mean_b{acc_b.x * inv, acc_b.y * inv, acc_b.z * inv};
    CHECK(finv(mean_a)); CHECK(finv(mean_b));
    CHECK(mean_b.x > mean_a.x);                            // indirect adds light from the bright sky
}

void test_trace_radiance_nan_ray_yields_finite() {
    // A NaN ray must produce a finite radiance value (zero is fine).
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    sc::LightSet lights;
    lx::Scene s = make_basic_scene(lights);
    auto tracer = lx::PathTracer::create();
    lx::Rng rng(13);
    sc::Ray bad{sc::Vec3{qnan, qnan, qnan}, sc::Vec3{0, 0, 1}};
    const sc::Vec3 r = tracer->trace_radiance(bad, s, rng);
    CHECK(finv(r));
}

// ----------------------------------------------------------------------------
// trace_image — full pinhole, deterministic seed.
// ----------------------------------------------------------------------------
void test_image_determinism() {
    sc::LightSet lights;
    sc::Light L; L.kind = sc::LightKind::Directional;
    L.direction = sc::Vec3{0, -1, 0}; L.color = sc::Vec3{1, 1, 1}; L.intensity = 1.0f;
    lights.add(L);
    lx::Scene s = make_basic_scene(lights);

    constexpr cardinal::u32 W = 8, H = 6;
    cardinal::vector<float> rad_a(W * H * 3), rad_b(W * H * 3);
    cardinal::vector<cardinal::u8> rgba_a(W * H * 4), rgba_b(W * H * 4);
    lx::PathTracerConfig cfg;
    cfg.max_bounces = 1;
    cfg.samples_per_pixel = 2;
    cfg.rng_seed = 0xCAFEu;
    auto t_a = lx::PathTracer::create(cfg);
    auto t_b = lx::PathTracer::create(cfg);
    t_a->trace_image(W, H, sc::Vec3{0, 1, -5},
                     sc::Vec3{0, 0, 1}, sc::Vec3{1, 0, 0}, sc::Vec3{0, 1, 0},
                     45.0f * sc::kDegToRad, s, rad_a.data(), rgba_a.data());
    t_b->trace_image(W, H, sc::Vec3{0, 1, -5},
                     sc::Vec3{0, 0, 1}, sc::Vec3{1, 0, 0}, sc::Vec3{0, 1, 0},
                     45.0f * sc::kDegToRad, s, rad_b.data(), rgba_b.data());
    for (cardinal::usize i = 0; i < rad_a.size(); ++i) CHECK(rad_a[i] == rad_b[i]);
    for (cardinal::usize i = 0; i < rgba_a.size(); ++i) CHECK(rgba_a[i] == rgba_b[i]);
    CHECK(t_a->stats().rays_primary == W * H * cfg.samples_per_pixel);
}

void test_image_all_finite_with_nan_material() {
    // Inject NaN into a material; the whole image must remain finite.
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    sc::LightSet lights;
    sc::Light L; L.kind = sc::LightKind::Directional;
    L.direction = sc::Vec3{0, -1, 0}; L.color = sc::Vec3{1, 1, 1}; L.intensity = 1.0f;
    lights.add(L);
    lx::Scene s = make_basic_scene(lights);
    s.materials[0].base_color = sc::Vec3{qnan, qnan, qnan};
    s.materials[0].roughness  = qnan;
    s.materials[0].metallic   = qnan;

    constexpr cardinal::u32 W = 4, H = 4;
    cardinal::vector<float> rad(W * H * 3);
    cardinal::vector<cardinal::u8> rgba(W * H * 4);
    auto t = lx::PathTracer::create();
    t->trace_image(W, H, sc::Vec3{0, 1, -5},
                   sc::Vec3{0, 0, 1}, sc::Vec3{1, 0, 0}, sc::Vec3{0, 1, 0},
                   45.0f * sc::kDegToRad, s, rad.data(), rgba.data());
    for (float v : rad) CHECK(fin(v));
}

// ----------------------------------------------------------------------------
// Knob editor surface.
// ----------------------------------------------------------------------------
void test_knob_surface() {
    auto tracer = lx::PathTracer::create();
    const auto& ks = tracer->knobs();
    CHECK(ks.size() >= 6u);
    bool saw_bounces = false, saw_spp = false, saw_sky_r = false, saw_nee = false;
    for (const auto& k : ks) {
        if (k.id == "rtx.max_bounces") { saw_bounces = true; CHECK(k.kind == lx::KnobKind::Int); CHECK(k.i_max <= 16); }
        if (k.id == "rtx.spp")         { saw_spp     = true; CHECK(k.kind == lx::KnobKind::Int); CHECK(k.i >= 1); }
        if (k.id == "rtx.sky.r")       { saw_sky_r   = true; CHECK(k.kind == lx::KnobKind::Float); }
        if (k.id == "rtx.nee")         { saw_nee     = true; CHECK(k.kind == lx::KnobKind::Bool); }
        CHECK(!k.id.empty());
        CHECK(!k.label.empty());
    }
    CHECK(saw_bounces); CHECK(saw_spp); CHECK(saw_sky_r); CHECK(saw_nee);
}

void test_knob_clamping_in_config() {
    auto tracer = lx::PathTracer::create();
    // Out-of-range writes round-trip through clamp.
    for (auto& k : tracer->knobs()) {
        if (k.id == "rtx.max_bounces") k.i = 999;
        if (k.id == "rtx.spp")         k.i = -10;
    }
    auto c = tracer->config();
    CHECK(c.max_bounces <= 16u);
    CHECK(c.samples_per_pixel >= 1u);
}

// ----------------------------------------------------------------------------
// RNG — determinism + distribution sanity.
// ----------------------------------------------------------------------------
void test_rng_determinism() {
    lx::Rng a(42), b(42), c(43);
    for (int i = 0; i < 1000; ++i) {
        const cardinal::u32 va = a.next_u32();
        const cardinal::u32 vb = b.next_u32();
        CHECK(va == vb);
    }
    CHECK(a.next_u32() != c.next_u32());          // different seed → different stream (overwhelmingly likely)
}

void test_rng_cosine_hemisphere_above_horizon() {
    // Every cosine-weighted sample must lie in the hemisphere (NdotL >= 0).
    lx::Rng rng(1);
    const sc::Vec3 n{0, 1, 0};
    for (int i = 0; i < 500; ++i) {
        const sc::Vec3 d = rng.next_cosine_hemisphere(n);
        CHECK(finv(d));
        CHECK(sc::dot(d, n) >= -1e-4f);
        CHECK(ap(sc::length(d), 1.0f, 1e-3f));
    }
}

void test_rng_unit_sphere_is_unit() {
    lx::Rng rng(2);
    for (int i = 0; i < 500; ++i) {
        const sc::Vec3 d = rng.next_unit_sphere();
        CHECK(finv(d));
        CHECK(ap(sc::length(d), 1.0f, 1e-3f));
    }
}

// ----------------------------------------------------------------------------
// BakedLightmap — sampler bilinear + clamp + NaN UV.
// ----------------------------------------------------------------------------
void test_lightmap_sampler_uniform() {
    // A uniformly-coloured atlas samples to that exact colour everywhere.
    lx::BakedLightmap lm;
    lm.resize(8, 8);
    for (cardinal::usize i = 0; i < lm.data.size(); i += 3) {
        lm.data[i + 0] = 0.5f;
        lm.data[i + 1] = 0.25f;
        lm.data[i + 2] = 0.75f;
    }
    const sc::Vec3 c = lx::sample_lightmap(lm, 0.5f, 0.5f);
    CHECK(ap(c.x, 0.5f,  1e-4f));
    CHECK(ap(c.y, 0.25f, 1e-4f));
    CHECK(ap(c.z, 0.75f, 1e-4f));
}

void test_lightmap_sampler_bilinear() {
    // Two-colour atlas: left half = (0,0,0), right half = (1,1,1).
    // Midpoint of the gradient is ~0.5 in each channel.
    lx::BakedLightmap lm;
    lm.resize(4, 1);
    auto set = [&](cardinal::u32 x, float r, float g, float b) {
        lm.data[x * 3 + 0] = r;
        lm.data[x * 3 + 1] = g;
        lm.data[x * 3 + 2] = b;
    };
    set(0, 0, 0, 0); set(1, 0, 0, 0);
    set(2, 1, 1, 1); set(3, 1, 1, 1);
    const sc::Vec3 left  = lx::sample_lightmap(lm, 0.0f, 0.0f);
    const sc::Vec3 right = lx::sample_lightmap(lm, 1.0f, 0.0f);
    const sc::Vec3 mid   = lx::sample_lightmap(lm, 0.5f, 0.0f);
    CHECK(ap(left.x,  0.0f, 1e-3f));
    CHECK(ap(right.x, 1.0f, 1e-3f));
    CHECK(mid.x > 0.0f && mid.x < 1.0f);     // strictly between (bilinear blend)
}

void test_lightmap_sampler_nan_uv_safe() {
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    lx::BakedLightmap lm;
    lm.resize(4, 4);
    for (cardinal::usize i = 0; i < lm.data.size(); ++i) lm.data[i] = 0.3f;
    const sc::Vec3 c = lx::sample_lightmap(lm, qnan, qnan);
    CHECK(finv(c));
    CHECK(ap(c.x, 0.3f, 1e-4f));
}

void test_lightmap_sampler_oob_clamps() {
    lx::BakedLightmap lm;
    lm.resize(4, 4);
    for (cardinal::usize i = 0; i < lm.data.size(); ++i) lm.data[i] = 0.7f;
    const sc::Vec3 a = lx::sample_lightmap(lm, -10.0f, -10.0f);
    const sc::Vec3 b = lx::sample_lightmap(lm,  10.0f,  10.0f);
    CHECK(finv(a)); CHECK(finv(b));
    CHECK(ap(a.x, 0.7f, 1e-4f));
    CHECK(ap(b.x, 0.7f, 1e-4f));
}

void test_lightmap_sampler_empty_safe() {
    lx::BakedLightmap lm;            // empty
    const sc::Vec3 c = lx::sample_lightmap(lm, 0.5f, 0.5f);
    CHECK(c.x == 0.0f); CHECK(c.y == 0.0f); CHECK(c.z == 0.0f);
}

// ----------------------------------------------------------------------------
// LightmapBaker — end-to-end bake on a known scene.
// ----------------------------------------------------------------------------
void test_lightmap_bake_directional_floor() {
    // Open floor under a bright sky → every cosine-hemisphere ray from a
    // floor texel goes up, misses, and returns sky radiance. With a
    // constant sky every texel must read the same value (within MC
    // variance). This is a sky-only bake — directional lights have zero
    // angular extent, so NEE is the right primitive for them and BRDF-
    // hemisphere integration alone can't see one; this test isolates the
    // hemisphere integral.
    sc::LightSet lights;
    lights.set_ambient(sc::Vec3{0, 0, 0});
    lx::Scene s;
    s.materials.push_back(lx::Material{sc::Vec3{0.8f, 0.8f, 0.8f}, sc::Vec3{0, 0, 0}, 0.0f, 1.0f});
    s.planes.push_back(lx::Plane{sc::Vec3{0, 0, 0}, sc::Vec3{0, 1, 0}, 0u});
    s.lights = &lights;

    cardinal::vector<lx::LightmapTexel> texels;
    constexpr cardinal::u32 N = 4;
    for (cardinal::u32 v = 0; v < N; ++v) {
        for (cardinal::u32 u = 0; u < N; ++u) {
            lx::LightmapTexel t;
            t.world_pos = sc::Vec3{
                static_cast<float>(u) - static_cast<float>(N) * 0.5f,
                0.1f,
                static_cast<float>(v) - static_cast<float>(N) * 0.5f,
            };
            t.world_normal = sc::Vec3{0, 1, 0};
            t.atlas_u = u;
            t.atlas_v = v;
            texels.push_back(t);
        }
    }
    lx::PathTracerConfig pcfg;
    pcfg.max_bounces       = 0;
    pcfg.samples_per_pixel = 1;
    pcfg.enable_indirect   = false;
    pcfg.sky_color         = sc::Vec3{1.0f, 1.0f, 1.0f};
    auto tracer = lx::PathTracer::create(pcfg);

    lx::BakedLightmap lm;
    lx::LightmapBakeConfig bcfg;
    bcfg.samples_per_texel = 16;
    bcfg.rng_seed = 1234u;
    auto baker = lx::LightmapBaker::create(tracer);
    baker->bake(s, texels, N, N, lm, bcfg);

    CHECK(lm.width == N);
    CHECK(lm.height == N);
    CHECK(baker->stats().texels == N * N);
    CHECK(baker->stats().rays == static_cast<cardinal::u64>(N * N * bcfg.samples_per_texel));

    // Every texel reads ~sky radiance (1.0) — the hemisphere is fully
    // open above each floor texel.
    float min_v = 1e30f, max_v = 0.0f;
    for (cardinal::usize i = 0; i < lm.data.size(); i += 3) {
        const float r = lm.data[i + 0];
        CHECK(fin(r)); CHECK(r >= 0.0f);
        if (r < min_v) min_v = r;
        if (r > max_v) max_v = r;
    }
    CHECK(min_v > 0.5f);    // sky=1, MC variance keeps it well above 0.5
    CHECK(max_v < 1.5f);    // and below 1.5
}

void test_lightmap_bake_determinism() {
    sc::LightSet lights;
    sc::Light L; L.kind = sc::LightKind::Directional;
    L.direction = sc::Vec3{0, -1, 0}; L.color = sc::Vec3{1, 1, 1}; L.intensity = 1.0f;
    lights.add(L);
    lx::Scene s = make_basic_scene(lights);

    cardinal::vector<lx::LightmapTexel> texels;
    for (cardinal::u32 u = 0; u < 4; ++u) {
        lx::LightmapTexel t;
        t.world_pos = sc::Vec3{static_cast<float>(u) * 0.5f, 0.0f, 0.0f};
        t.world_normal = sc::Vec3{0, 1, 0};
        t.atlas_u = u; t.atlas_v = 0;
        texels.push_back(t);
    }
    lx::PathTracerConfig pcfg;
    pcfg.max_bounces = 1;
    auto t_a = lx::PathTracer::create(pcfg);
    auto t_b = lx::PathTracer::create(pcfg);
    auto b_a = lx::LightmapBaker::create(t_a);
    auto b_b = lx::LightmapBaker::create(t_b);

    lx::BakedLightmap lm_a, lm_b;
    lx::LightmapBakeConfig bcfg;
    bcfg.samples_per_texel = 4;
    bcfg.rng_seed = 0x77u;
    b_a->bake(s, texels, 4, 1, lm_a, bcfg);
    b_b->bake(s, texels, 4, 1, lm_b, bcfg);
    for (cardinal::usize i = 0; i < lm_a.data.size(); ++i) CHECK(lm_a.data[i] == lm_b.data[i]);
}

void test_lightmap_bake_nan_texel_safe() {
    // A NaN-positioned texel mustn't crash, mustn't poison its slot.
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    sc::LightSet lights;
    sc::Light L; L.kind = sc::LightKind::Directional;
    L.direction = sc::Vec3{0, -1, 0}; L.color = sc::Vec3{1, 1, 1}; L.intensity = 1.0f;
    lights.add(L);
    lx::Scene s = make_basic_scene(lights);

    cardinal::vector<lx::LightmapTexel> texels;
    {
        lx::LightmapTexel t;
        t.world_pos = sc::Vec3{qnan, qnan, qnan};
        t.world_normal = sc::Vec3{qnan, qnan, qnan};
        t.atlas_u = 0; t.atlas_v = 0;
        texels.push_back(t);
    }
    auto tracer = lx::PathTracer::create();
    auto baker = lx::LightmapBaker::create(tracer);
    lx::BakedLightmap lm;
    lx::LightmapBakeConfig bcfg;
    bcfg.samples_per_texel = 2;
    baker->bake(s, texels, 2, 2, lm, bcfg);
    for (float v : lm.data) CHECK(fin(v));
}

void test_lightmap_bake_skips_oob_texels() {
    sc::LightSet lights;
    lx::Scene s = make_basic_scene(lights);
    cardinal::vector<lx::LightmapTexel> texels;
    {   // out of bounds (atlas only 2x2)
        lx::LightmapTexel t;
        t.world_pos = sc::Vec3{0, 0, 0};
        t.world_normal = sc::Vec3{0, 1, 0};
        t.atlas_u = 99; t.atlas_v = 99;
        texels.push_back(t);
    }
    auto tracer = lx::PathTracer::create();
    auto baker = lx::LightmapBaker::create(tracer);
    lx::BakedLightmap lm;
    baker->bake(s, texels, 2, 2, lm, {});
    CHECK(baker->stats().texels == 0u);
    // Atlas is still resized + cleared to 0 → all finite zero.
    for (float v : lm.data) CHECK(fin(v));
}

// ----------------------------------------------------------------------------
// IrradianceProbeVolume — sampler + bake.
// ----------------------------------------------------------------------------
void test_probe_sampler_axis_weights() {
    // A 1×1×1 volume with known per-axis values. Sampling with normal +Y
    // returns the +Y face; with normal -Y returns the -Y face.
    lx::IrradianceProbeVolume vol;
    vol.origin  = sc::Vec3{0, 0, 0};
    vol.spacing = sc::Vec3{1, 1, 1};
    vol.resize(1, 1, 1);
    // axis order: +X, -X, +Y, -Y, +Z, -Z
    auto setf = [&](int axis, float r, float g, float b) {
        vol.data[axis * 3 + 0] = r;
        vol.data[axis * 3 + 1] = g;
        vol.data[axis * 3 + 2] = b;
    };
    setf(0, 1, 0, 0);   // +X red
    setf(1, 0, 1, 0);   // -X green
    setf(2, 0, 0, 1);   // +Y blue
    setf(3, 1, 1, 0);   // -Y yellow
    setf(4, 1, 0, 1);   // +Z magenta
    setf(5, 0, 1, 1);   // -Z cyan

    const sc::Vec3 py = lx::sample_probe_volume(vol, sc::Vec3{0, 0, 0}, sc::Vec3{0,  1, 0});
    const sc::Vec3 ny = lx::sample_probe_volume(vol, sc::Vec3{0, 0, 0}, sc::Vec3{0, -1, 0});
    const sc::Vec3 px = lx::sample_probe_volume(vol, sc::Vec3{0, 0, 0}, sc::Vec3{ 1, 0, 0});
    CHECK(ap(py.z, 1.0f, 1e-3f));   // +Y blue
    CHECK(ap(py.x, 0.0f, 1e-3f));
    CHECK(ap(ny.x, 1.0f, 1e-3f));   // -Y yellow → R=1
    CHECK(ap(ny.y, 1.0f, 1e-3f));   // -Y yellow → G=1
    CHECK(ap(px.x, 1.0f, 1e-3f));   // +X red
}

void test_probe_sampler_trilinear() {
    // 2×1×1 volume: x=0 probe is (1,0,0), x=1 probe is (0,0,1). Sampling
    // midway should give roughly (0.5, 0, 0.5) on the +X axis weight.
    lx::IrradianceProbeVolume vol;
    vol.origin = sc::Vec3{0, 0, 0};
    vol.spacing = sc::Vec3{1, 1, 1};
    vol.resize(2, 1, 1);
    // Probe 0 (x=0): +X face = red
    vol.data[0 * 6 * 3 + 0 * 3 + 0] = 1.0f;
    // Probe 1 (x=1): +X face = blue
    vol.data[1 * 6 * 3 + 0 * 3 + 2] = 1.0f;

    const sc::Vec3 lo  = lx::sample_probe_volume(vol, sc::Vec3{0.0f, 0, 0}, sc::Vec3{1, 0, 0});
    const sc::Vec3 hi  = lx::sample_probe_volume(vol, sc::Vec3{1.0f, 0, 0}, sc::Vec3{1, 0, 0});
    const sc::Vec3 mid = lx::sample_probe_volume(vol, sc::Vec3{0.5f, 0, 0}, sc::Vec3{1, 0, 0});
    CHECK(ap(lo.x, 1.0f, 1e-3f));
    CHECK(ap(hi.z, 1.0f, 1e-3f));
    CHECK(ap(mid.x, 0.5f, 1e-3f));
    CHECK(ap(mid.z, 0.5f, 1e-3f));
}

void test_probe_sampler_oob_clamps() {
    lx::IrradianceProbeVolume vol;
    vol.origin  = sc::Vec3{0, 0, 0};
    vol.spacing = sc::Vec3{1, 1, 1};
    vol.resize(2, 2, 2);
    vol.data[0 * 3 + 0] = 0.42f;     // +X of probe (0,0,0)
    const sc::Vec3 inside  = lx::sample_probe_volume(vol, sc::Vec3{0, 0, 0},   sc::Vec3{1, 0, 0});
    const sc::Vec3 outside = lx::sample_probe_volume(vol, sc::Vec3{-99, -99, -99}, sc::Vec3{1, 0, 0});
    CHECK(ap(inside.x, 0.42f, 1e-3f));
    CHECK(ap(outside.x, 0.42f, 1e-3f));   // clamped to boundary probe (0,0,0)
}

void test_probe_sampler_nan_pos_safe() {
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    lx::IrradianceProbeVolume vol;
    vol.origin = sc::Vec3{0, 0, 0};
    vol.spacing = sc::Vec3{1, 1, 1};
    vol.resize(2, 2, 2);
    for (cardinal::usize i = 0; i < vol.data.size(); ++i) vol.data[i] = 0.25f;
    const sc::Vec3 c = lx::sample_probe_volume(vol, sc::Vec3{qnan, qnan, qnan},
                                                    sc::Vec3{qnan, qnan, qnan});
    CHECK(finv(c));
}

void test_probe_sampler_empty_safe() {
    lx::IrradianceProbeVolume vol;
    const sc::Vec3 c = lx::sample_probe_volume(vol, sc::Vec3{0, 0, 0}, sc::Vec3{0, 1, 0});
    CHECK(c.x == 0.0f); CHECK(c.y == 0.0f); CHECK(c.z == 0.0f);
}

void test_probe_volume_bake_end_to_end() {
    // Open scene under a bright sky → every probe + axis facing up sees
    // sky radiance. Floor under the probes occludes the -Y axis.
    sc::LightSet lights;
    lights.set_ambient(sc::Vec3{0, 0, 0});
    lx::Scene s;
    s.materials.push_back(lx::Material{sc::Vec3{0.5f, 0.5f, 0.5f}, sc::Vec3{0, 0, 0}, 0.0f, 1.0f});
    s.planes.push_back(lx::Plane{sc::Vec3{0, 0, 0}, sc::Vec3{0, 1, 0}, 0u});
    s.lights = &lights;

    lx::PathTracerConfig pcfg;
    pcfg.max_bounces = 0;
    pcfg.enable_indirect = false;
    pcfg.sky_color = sc::Vec3{1.0f, 1.0f, 1.0f};
    auto tracer = lx::PathTracer::create(pcfg);

    lx::IrradianceProbeVolume vol;
    vol.origin  = sc::Vec3{-1, 1, -1};
    vol.spacing = sc::Vec3{1, 1, 1};
    vol.resize(2, 2, 2);
    auto baker = lx::ProbeVolumeBaker::create(tracer);
    lx::ProbeBakeConfig cfg;
    cfg.samples_per_axis = 16;
    cfg.rng_seed = 999u;
    baker->bake(s, vol, cfg);

    CHECK(baker->stats().probes == vol.probe_count());
    CHECK(baker->stats().rays == static_cast<cardinal::u64>(vol.probe_count() * 6 * cfg.samples_per_axis));

    // For the (0,0,0) probe at world (-1, 1, -1): +Y must see the bright
    // sky. -Y faces the floor — the directional sky lookup integrates to
    // ~0 (floor absorbs / doesn't emit in this direct-only config).
    const cardinal::usize base = 0;            // probe (0,0,0)
    const float plus_y_r  = vol.data[base + 2 * 3 + 0];
    const float minus_y_r = vol.data[base + 3 * 3 + 0];
    CHECK(fin(plus_y_r));
    CHECK(fin(minus_y_r));
    CHECK(plus_y_r > 0.0f);
    CHECK(plus_y_r > minus_y_r);             // sky brighter than floor
}

// ============================================================================
// GPU-pass coverage: ForwardShadePass + PathTracePass (under CpuBackend).
// ============================================================================
namespace rg = cardinal::render::graph;
namespace glx = cardinal::lighting::gpu;

struct InitBlob { rg::ResourceHandle h; const void* src; cardinal::usize bytes; };
void rec_init_blob(rg::ExecutionContext& ec, void* uctx) noexcept {
    auto* c = static_cast<InitBlob*>(uctx);
    void* dst = ec.map_buffer_write(c->h);
    if (!dst) return;
    auto* d = static_cast<cardinal::u8*>(dst);
    auto* s = static_cast<const cardinal::u8*>(c->src);
    for (cardinal::usize i = 0; i < c->bytes; ++i) d[i] = s[i];
}
void add_init_pass(rg::Graph& g, const char* name, rg::ResourceHandle h, InitBlob* blob) {
    rg::PassDesc pd; pd.name = name; pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{h, rg::AccessMode::Write, 0});
    pd.record = rec_init_blob; pd.user_ctx = blob;
    g.add_pass(cardinal::move(pd));
}

void test_forward_shade_pass_directional() {
    // Single white directional light, white diffuse fragments facing up,
    // black ambient. Fragments are positioned below the camera (at origin)
    // so view_dir = normalize(-pos) = +Y (aligned with the normal, NdotV > 0).
    constexpr cardinal::u32 N = 4;
    cardinal::vector<float> wp(N * 3u, 0.0f);
    // Place fragments at (0, -1, 0) so view_dir from origin = +Y.
    for (cardinal::u32 i = 0; i < N; ++i) {
        wp[1 * N + i] = -1.0f;
    }
    cardinal::vector<float> wn(N * 3u, 0.0f);
    // Normals: all +Y
    for (cardinal::u32 i = 0; i < N; ++i) {
        wn[1 * N + i] = 1.0f;
    }
    cardinal::vector<float> mt(N * 5u, 0.0f);
    // base color = (1, 1, 1), metallic = 0, roughness = 1
    for (cardinal::u32 i = 0; i < N; ++i) {
        mt[0 * N + i] = 1.0f;
        mt[1 * N + i] = 1.0f;
        mt[2 * N + i] = 1.0f;
        mt[3 * N + i] = 0.0f;
        mt[4 * N + i] = 1.0f;
    }
    glx::PackedLight light{};
    light.kind = 0u;
    light.direction[0] = 0; light.direction[1] = -1; light.direction[2] = 0;
    light.color[0] = 1; light.color[1] = 1; light.color[2] = 1;
    light.intensity = 1.0f;
    light.range = 100.0f;
    light.inner_cos = 0.95f;
    light.outer_cos = 0.85f;
    cardinal::vector<float> ambient = {0.0f, 0.0f, 0.0f};

    auto g = rg::Graph::create();
    auto h_wp = g->declare_buffer(rg::BufferDesc{"wp", wp.size() * sizeof(float), 0, true});
    auto h_wn = g->declare_buffer(rg::BufferDesc{"wn", wn.size() * sizeof(float), 0, true});
    auto h_mt = g->declare_buffer(rg::BufferDesc{"mt", mt.size() * sizeof(float), 0, true});
    auto h_lt = g->declare_buffer(rg::BufferDesc{"lt", sizeof(glx::PackedLight), 0, true});
    auto h_am = g->declare_buffer(rg::BufferDesc{"am", ambient.size() * sizeof(float), 0, true});
    InitBlob b_wp{h_wp, wp.data(), wp.size() * sizeof(float)};
    InitBlob b_wn{h_wn, wn.data(), wn.size() * sizeof(float)};
    InitBlob b_mt{h_mt, mt.data(), mt.size() * sizeof(float)};
    InitBlob b_lt{h_lt, &light,     sizeof(glx::PackedLight)};
    InitBlob b_am{h_am, ambient.data(), ambient.size() * sizeof(float)};
    add_init_pass(*g, "iwp", h_wp, &b_wp);
    add_init_pass(*g, "iwn", h_wn, &b_wn);
    add_init_pass(*g, "imt", h_mt, &b_mt);
    add_init_pass(*g, "ilt", h_lt, &b_lt);
    add_init_pass(*g, "iam", h_am, &b_am);

    auto st = glx::ForwardShadePass::add_to_graph(*g, h_wp, h_wn, h_mt, h_lt, h_am, N, 1);
    CHECK(g->compile());
    auto backend = rg::CpuBackend::create();
    backend->execute(*g);

    auto out = backend->buffer_contents(st->out_shaded);
    CHECK(out.size() == N * 3u * sizeof(float));
    const float* of = reinterpret_cast<const float*>(out.data());
    const float* outr = of + 0 * N;
    const float* outg = of + 1 * N;
    const float* outb = of + 2 * N;
    for (cardinal::u32 i = 0; i < N; ++i) {
        CHECK(outr[i] > 0.0f);
        CHECK(outg[i] > 0.0f);
        CHECK(outb[i] > 0.0f);
        // White light, white fragment → r ~ g ~ b
        CHECK(ap(outr[i], outg[i], 1e-3f));
        CHECK(ap(outr[i], outb[i], 1e-3f));
    }
    CHECK(st->fragments_lit == N);
}

void test_forward_shade_pass_back_facing_dark() {
    // Single fragment with normal pointing AWAY from a directional light
    // should receive ambient only.
    constexpr cardinal::u32 N = 1;
    cardinal::vector<float> wp(N * 3u, 0.0f);
    // Place fragment at (0, +1, 0) so view_dir = -Y points along the (now -Y) normal.
    wp[1 * N + 0] = 1.0f;
    cardinal::vector<float> wn(N * 3u, 0.0f);
    // Normal pointing -Y (away from a downward-shining +Y light)
    wn[1 * N + 0] = -1.0f;
    cardinal::vector<float> mt = {0.5f, 0.5f, 0.5f, 0.0f, 1.0f};

    glx::PackedLight light{};
    light.kind = 0u;
    light.direction[0] = 0; light.direction[1] = -1; light.direction[2] = 0;
    light.color[0] = 1; light.color[1] = 1; light.color[2] = 1;
    light.intensity = 1.0f;
    cardinal::vector<float> ambient = {0.1f, 0.1f, 0.1f};

    auto g = rg::Graph::create();
    auto h_wp = g->declare_buffer(rg::BufferDesc{"wp", wp.size() * sizeof(float), 0, true});
    auto h_wn = g->declare_buffer(rg::BufferDesc{"wn", wn.size() * sizeof(float), 0, true});
    auto h_mt = g->declare_buffer(rg::BufferDesc{"mt", mt.size() * sizeof(float), 0, true});
    auto h_lt = g->declare_buffer(rg::BufferDesc{"lt", sizeof(glx::PackedLight), 0, true});
    auto h_am = g->declare_buffer(rg::BufferDesc{"am", ambient.size() * sizeof(float), 0, true});
    InitBlob b_wp{h_wp, wp.data(), wp.size() * sizeof(float)};
    InitBlob b_wn{h_wn, wn.data(), wn.size() * sizeof(float)};
    InitBlob b_mt{h_mt, mt.data(), mt.size() * sizeof(float)};
    InitBlob b_lt{h_lt, &light,     sizeof(glx::PackedLight)};
    InitBlob b_am{h_am, ambient.data(), ambient.size() * sizeof(float)};
    add_init_pass(*g, "iwp", h_wp, &b_wp);
    add_init_pass(*g, "iwn", h_wn, &b_wn);
    add_init_pass(*g, "imt", h_mt, &b_mt);
    add_init_pass(*g, "ilt", h_lt, &b_lt);
    add_init_pass(*g, "iam", h_am, &b_am);

    auto st = glx::ForwardShadePass::add_to_graph(*g, h_wp, h_wn, h_mt, h_lt, h_am, N, 1);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    auto out = b->buffer_contents(st->out_shaded);
    const float* of = reinterpret_cast<const float*>(out.data());
    // Only ambient * base_color contributes → 0.1 * 0.5 = 0.05
    CHECK(ap(of[0], 0.05f, 1e-3f));   // r
    CHECK(st->fragments_lit == 0u);   // back-facing → no light pass NdotL > 0
}

void test_forward_shade_pass_nan_safe() {
    // NaN inputs anywhere must produce finite output.
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    constexpr cardinal::u32 N = 2;
    cardinal::vector<float> wp(N * 3u, qnan);
    cardinal::vector<float> wn(N * 3u, qnan);
    cardinal::vector<float> mt(N * 5u, qnan);
    glx::PackedLight light{};
    light.kind = 0u;
    light.direction[0] = qnan; light.direction[1] = qnan; light.direction[2] = qnan;
    light.color[0] = qnan; light.color[1] = qnan; light.color[2] = qnan;
    light.intensity = qnan;
    cardinal::vector<float> ambient = {qnan, qnan, qnan};

    auto g = rg::Graph::create();
    auto h_wp = g->declare_buffer(rg::BufferDesc{"wp", wp.size() * sizeof(float), 0, true});
    auto h_wn = g->declare_buffer(rg::BufferDesc{"wn", wn.size() * sizeof(float), 0, true});
    auto h_mt = g->declare_buffer(rg::BufferDesc{"mt", mt.size() * sizeof(float), 0, true});
    auto h_lt = g->declare_buffer(rg::BufferDesc{"lt", sizeof(glx::PackedLight), 0, true});
    auto h_am = g->declare_buffer(rg::BufferDesc{"am", ambient.size() * sizeof(float), 0, true});
    InitBlob b_wp{h_wp, wp.data(), wp.size() * sizeof(float)};
    InitBlob b_wn{h_wn, wn.data(), wn.size() * sizeof(float)};
    InitBlob b_mt{h_mt, mt.data(), mt.size() * sizeof(float)};
    InitBlob b_lt{h_lt, &light,     sizeof(glx::PackedLight)};
    InitBlob b_am{h_am, ambient.data(), ambient.size() * sizeof(float)};
    add_init_pass(*g, "iwp", h_wp, &b_wp);
    add_init_pass(*g, "iwn", h_wn, &b_wn);
    add_init_pass(*g, "imt", h_mt, &b_mt);
    add_init_pass(*g, "ilt", h_lt, &b_lt);
    add_init_pass(*g, "iam", h_am, &b_am);

    auto st = glx::ForwardShadePass::add_to_graph(*g, h_wp, h_wn, h_mt, h_lt, h_am, N, 1);
    CHECK(g->compile());
    rg::CpuBackend::create()->execute(*g);
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    auto out = b->buffer_contents(st->out_shaded);
    const float* of = reinterpret_cast<const float*>(out.data());
    for (cardinal::usize i = 0; i < N * 3; ++i) {
        CHECK(of[i] == of[i]);
        CHECK((of[i] - of[i]) == 0.0f);
    }
}

void test_forward_shade_pass_hlsl_nonempty() {
    const char* hlsl = glx::ForwardShadePass::hlsl_source();
    CHECK(hlsl != nullptr);
    CHECK(hlsl[0] != '\0');
}

// ----------------------------------------------------------------------------
// PathTracePass — wraps the CPU PathTracer under a graph node.
// ----------------------------------------------------------------------------
void test_path_trace_pass_renders_finite_image() {
    sc::LightSet lights;
    sc::Light L; L.kind = sc::LightKind::Directional;
    L.direction = sc::Vec3{0, -1, 0}; L.color = sc::Vec3{1, 1, 1}; L.intensity = 1.0f;
    lights.add(L);
    lx::Scene s = make_basic_scene(lights);

    lx::PathTracerConfig pcfg;
    pcfg.samples_per_pixel = 1;
    pcfg.max_bounces       = 1;
    auto tracer = lx::PathTracer::create(pcfg);

    constexpr cardinal::u32 W = 4, H = 4;
    auto g = rg::Graph::create();
    auto st = glx::PathTracePass::add_to_graph(*g, tracer, s, W, H);
    st->camera_pos = sc::Vec3{0, 1, -5};
    st->forward    = sc::Vec3{0, 0, 1};
    st->right      = sc::Vec3{1, 0, 0};
    st->up         = sc::Vec3{0, 1, 0};
    st->vfov_rad   = 45.0f * sc::kDegToRad;

    CHECK(g->compile());
    auto backend = rg::CpuBackend::create();
    backend->execute(*g);
    auto rad = backend->buffer_contents(st->out_radiance);
    auto rgba= backend->buffer_contents(st->out_rgba);
    CHECK(rad.size()  == W * H * 3 * sizeof(float));
    CHECK(rgba.size() == W * H * 4);
    const float* rf = reinterpret_cast<const float*>(rad.data());
    for (cardinal::usize i = 0; i < W * H * 3; ++i) CHECK(fin(rf[i]));
    // Every alpha must be 255 (the tracer always writes opaque).
    for (cardinal::usize i = 0; i < W * H; ++i) CHECK(rgba[i * 4 + 3] == 255u);
    // Rays counter > 0.
    CHECK(st->rays_total > 0u);
}

void test_path_trace_pass_determinism() {
    sc::LightSet lights;
    sc::Light L; L.kind = sc::LightKind::Directional;
    L.direction = sc::Vec3{0, -1, 0}; L.color = sc::Vec3{1, 1, 1}; L.intensity = 1.0f;
    lights.add(L);
    lx::Scene s = make_basic_scene(lights);

    lx::PathTracerConfig pcfg;
    pcfg.samples_per_pixel = 1;
    pcfg.max_bounces       = 0;
    pcfg.rng_seed          = 0xC0FFEE;
    auto ta = lx::PathTracer::create(pcfg);
    auto tb = lx::PathTracer::create(pcfg);

    constexpr cardinal::u32 W = 4, H = 4;
    auto ga = rg::Graph::create();
    auto gb = rg::Graph::create();
    auto sta = glx::PathTracePass::add_to_graph(*ga, ta, s, W, H);
    auto stb = glx::PathTracePass::add_to_graph(*gb, tb, s, W, H);
    for (auto* st : {sta.get(), stb.get()}) {
        st->camera_pos = sc::Vec3{0, 1, -5};
        st->forward = sc::Vec3{0, 0, 1};
        st->right   = sc::Vec3{1, 0, 0};
        st->up      = sc::Vec3{0, 1, 0};
        st->vfov_rad = 45.0f * sc::kDegToRad;
    }
    CHECK(ga->compile());
    CHECK(gb->compile());
    auto ba = rg::CpuBackend::create();
    auto bb = rg::CpuBackend::create();
    ba->execute(*ga);
    bb->execute(*gb);
    auto ra = ba->buffer_contents(sta->out_radiance);
    auto rb = bb->buffer_contents(stb->out_radiance);
    CHECK(ra.size() == rb.size());
    for (cardinal::usize i = 0; i < ra.size(); ++i) CHECK(ra[i] == rb[i]);
}

void test_path_trace_pass_hlsl_nonempty() {
    const char* hlsl = glx::PathTracePass::hlsl_source();
    CHECK(hlsl != nullptr);
    CHECK(hlsl[0] != '\0');
}

// ----------------------------------------------------------------------------
// RadianceCachePass — wraps ProbeVolumeBaker as a graph pass
// (AEGIS Block 7 — Radiance Cache, sibling to ReSTIR DI / GI)
// ----------------------------------------------------------------------------
void test_radiance_cache_pass_bakes_probe_volume() {
    sc::LightSet lights;
    lights.set_ambient(sc::Vec3{0.1f, 0.1f, 0.1f});
    lx::Scene s;
    s.materials.push_back(lx::Material{sc::Vec3{0.5f, 0.5f, 0.5f}, sc::Vec3{0, 0, 0}, 0.0f, 1.0f});
    s.planes.push_back(lx::Plane{sc::Vec3{0, 0, 0}, sc::Vec3{0, 1, 0}, 0u});
    s.lights = &lights;

    lx::PathTracerConfig pcfg;
    pcfg.max_bounces = 0;
    pcfg.enable_indirect = false;
    pcfg.sky_color = sc::Vec3{1.0f, 1.0f, 1.0f};
    auto tracer = lx::PathTracer::create(pcfg);

    constexpr cardinal::u32 D = 2;
    auto g = rg::Graph::create();
    auto st = glx::RadianceCachePass::add_to_graph(
        *g, tracer, s,
        sc::Vec3{-1, 1, -1}, sc::Vec3{1, 1, 1}, D, D, D, 4);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);

    // Probe count + ray count match the bake config.
    CHECK(st->probes_baked == D * D * D);
    CHECK(st->rays_cast == st->probes_baked * 6 * 4);

    // Output buffer layout matches IrradianceProbeVolume::data
    // (probe_count * 6 axes * 3 floats).
    auto out = b->buffer_contents(st->out_probe_data);
    CHECK(out.size() == D * D * D * 6 * 3 * sizeof(float));
    const float* of = reinterpret_cast<const float*>(out.data());
    // With sky lit (white) and the floor below the volume, the +Y
    // hemisphere of every probe should be > 0 (sky scatter); the -Y
    // hemisphere should be lower (floor below). Probe at (0, 0, 0) at
    // world (-1, 1, -1): axis order is +X, -X, +Y, -Y, +Z, -Z.
    const cardinal::usize probe0_off = 0;
    const float plus_y  = of[probe0_off + 2 * 3 + 0];  // +Y red
    const float minus_y = of[probe0_off + 3 * 3 + 0];  // -Y red
    // Every probe value must be finite.
    for (cardinal::usize i = 0; i < D * D * D * 6 * 3; ++i) {
        CHECK(of[i] == of[i]);
    }
    CHECK(plus_y >= 0.0f);
    CHECK(minus_y >= 0.0f);
}

void test_radiance_cache_pass_zero_dims_safe() {
    sc::LightSet lights;
    lx::Scene s;
    s.lights = &lights;
    auto tracer = lx::PathTracer::create();
    auto g = rg::Graph::create();
    auto st = glx::RadianceCachePass::add_to_graph(
        *g, tracer, s,
        sc::Vec3{0, 0, 0}, sc::Vec3{1, 1, 1}, 0, 0, 0, 4);
    CHECK(g->compile());
    auto b = rg::CpuBackend::create();
    b->execute(*g);
    CHECK(st->probes_baked == 0u);
    CHECK(st->rays_cast == 0u);
}

void test_radiance_cache_pass_hlsl_nonempty() {
    CHECK(glx::RadianceCachePass::hlsl_source() != nullptr);
    CHECK(glx::RadianceCachePass::hlsl_source()[0] != '\0');
}

void test_probe_volume_bake_determinism() {
    sc::LightSet lights;
    sc::Light L; L.kind = sc::LightKind::Directional;
    L.direction = sc::Vec3{0, -1, 0}; L.color = sc::Vec3{1, 1, 1}; L.intensity = 1.0f;
    lights.add(L);
    lx::Scene s = make_basic_scene(lights);

    auto t_a = lx::PathTracer::create();
    auto t_b = lx::PathTracer::create();
    auto b_a = lx::ProbeVolumeBaker::create(t_a);
    auto b_b = lx::ProbeVolumeBaker::create(t_b);

    lx::IrradianceProbeVolume v_a, v_b;
    v_a.origin = v_b.origin = sc::Vec3{0, 1, 0};
    v_a.spacing = v_b.spacing = sc::Vec3{1, 1, 1};
    v_a.resize(2, 2, 2); v_b.resize(2, 2, 2);
    lx::ProbeBakeConfig cfg;
    cfg.samples_per_axis = 4;
    cfg.rng_seed = 0xDEADBEEFu;
    b_a->bake(s, v_a, cfg);
    b_b->bake(s, v_b, cfg);
    for (cardinal::usize i = 0; i < v_a.data.size(); ++i) CHECK(v_a.data[i] == v_b.data[i]);
}

}  // namespace

int main() {
    test_trace_closest_sphere_hit();
    test_trace_closest_plane_hit();
    test_trace_closest_miss();
    test_trace_closest_closest_wins();
    test_trace_closest_nan_ray_is_miss();

    test_visible_unobstructed();
    test_visible_blocked();
    test_visible_nan_returns_false_safely();

    test_brdf_directly_lit_diffuse();
    test_brdf_below_horizon_is_zero();
    test_brdf_metal_specular_at_mirror();
    test_brdf_nan_inputs_safe();

    test_shade_direct_directional_no_occlusion();
    test_shade_direct_point_attenuation();
    test_shade_direct_spot_cone_falloff();
    test_shade_direct_occlusion();
    test_shade_direct_emissive_adds();

    test_trace_radiance_miss_returns_sky();
    test_trace_radiance_direct_only_is_finite();
    test_trace_radiance_indirect_brighter_than_direct_in_closed_box();
    test_trace_radiance_nan_ray_yields_finite();

    test_image_determinism();
    test_image_all_finite_with_nan_material();

    test_knob_surface();
    test_knob_clamping_in_config();

    test_rng_determinism();
    test_rng_cosine_hemisphere_above_horizon();
    test_rng_unit_sphere_is_unit();

    // ---- baked lighting (lightmap + probe volume)
    test_lightmap_sampler_uniform();
    test_lightmap_sampler_bilinear();
    test_lightmap_sampler_nan_uv_safe();
    test_lightmap_sampler_oob_clamps();
    test_lightmap_sampler_empty_safe();
    test_lightmap_bake_directional_floor();
    test_lightmap_bake_determinism();
    test_lightmap_bake_nan_texel_safe();
    test_lightmap_bake_skips_oob_texels();
    test_probe_sampler_axis_weights();
    test_probe_sampler_trilinear();
    test_probe_sampler_oob_clamps();
    test_probe_sampler_nan_pos_safe();
    test_probe_sampler_empty_safe();
    test_probe_volume_bake_end_to_end();
    test_probe_volume_bake_determinism();

    // ---- GPU passes (graph-hosted, CpuBackend-executed)
    test_forward_shade_pass_directional();
    test_forward_shade_pass_back_facing_dark();
    test_forward_shade_pass_nan_safe();
    test_forward_shade_pass_hlsl_nonempty();
    test_path_trace_pass_renders_finite_image();
    test_path_trace_pass_determinism();
    test_path_trace_pass_hlsl_nonempty();
    test_radiance_cache_pass_bakes_probe_volume();
    test_radiance_cache_pass_zero_dims_safe();
    test_radiance_cache_pass_hlsl_nonempty();

    if (g_fail == 0) {
        cardinal::log::infof("rtxtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("rtxtest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
