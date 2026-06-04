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
#include <cardinal/core/log.hpp>

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

    if (g_fail == 0) {
        cardinal::log::infof("rtxtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("rtxtest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
