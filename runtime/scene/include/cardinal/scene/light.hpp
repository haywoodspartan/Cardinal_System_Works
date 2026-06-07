#pragma once

// =============================================================================
// Cardinal — Scene lighting representation.
//
// Renderer-facing lights — kept here in cardinal::scene so the renderer
// (which only knows about scene types) can integrate them without pulling
// in cardinal::actor or cardinal::game.
//
// Hosts that use actors mirror their LightComponents into a LightSet each
// frame; hosts using raw scene::Entity can drive a LightSet directly.
//
// LightSet is a flat vector — culling is the renderer's job. The default
// renderer iterates the whole list per fragment which is fine for the
// editor's typical scene density (≤ a few dozen lights).
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/scene/math.hpp>

namespace cardinal::scene {

enum class LightKind : u32 { Directional = 0, Point = 1, Spot = 2 };

struct Light {
    LightKind kind{LightKind::Directional};
    Vec3      position {0, 0, 0};        // world-space
    Vec3      direction{0, -1, 0};       // unit vector — points AT the scene
    Vec3      color    {1, 1, 1};
    float     intensity{1.0f};
    float     range    {30.0f};          // point/spot fall-off radius
    float     spot_inner_cos{0.95f};
    float     spot_outer_cos{0.85f};
    bool      cast_shadows{false};       // not yet honoured — reserved
};

class LightSet {
public:
    void  clear() noexcept            { lights_.clear(); ambient_ = {0.05f, 0.05f, 0.06f}; }
    void  add(const Light& l)         { lights_.push_back(l); }
    void  set_ambient(const Vec3& a)  { ambient_ = a; }

    const cardinal::vector<Light>& lights() const noexcept { return lights_; }
    // Non-const overload — hosts that want to animate per-frame state
    // (e.g. drive a directional from a sky time-of-day system) can
    // mutate lights_[i] in place rather than clear()+add() each tick.
    cardinal::vector<Light>&       lights()       noexcept { return lights_; }
    const Vec3&               ambient() const noexcept { return ambient_; }
    usize                     count()   const noexcept { return lights_.size(); }

    // Render reads this color when no skybox is bound.
    Vec3& clear_color() noexcept { return clear_color_; }
    const Vec3& clear_color() const noexcept { return clear_color_; }

    // HDR exposure pre-scale applied before the ACES tonemap in the
    // forward PS. 1.0 = neutral. Hosts tune this (a bright many-light
    // scene wants <1 to avoid blowing out; a dim one wants >1). Lives
    // here because LightSet is already the per-frame render-params
    // conduit the host owns + the renderer reads (ambient, clear).
    void  set_exposure(float e) noexcept { exposure_ = e; }
    float exposure() const noexcept      { return exposure_; }

    // Directional shadow-map toggle. When false the renderer skips the
    // depth pass + the PCF shadow test entirely (everything fully lit).
    // Default on. Same host-owned / renderer-read conduit rationale as
    // exposure above.
    void  set_shadows_enabled(bool e) noexcept { shadows_enabled_ = e; }
    bool  shadows_enabled() const noexcept     { return shadows_enabled_; }

    // Compute the surface color for a fragment — Lambertian + ambient.
    // Used by the CPU forward renderer; same math is the basis for the
    // shader path when it lands.
    Vec3 shade(const Vec3& world_pos, const Vec3& world_normal,
               const Vec3& base_color) const noexcept;

private:
    cardinal::vector<Light> lights_;
    Vec3               ambient_     {0.05f, 0.05f, 0.06f};
    Vec3               clear_color_ {0.07f, 0.08f, 0.10f};
    float              exposure_    {1.0f};
    bool               shadows_enabled_{true};
};

}  // namespace cardinal::scene
