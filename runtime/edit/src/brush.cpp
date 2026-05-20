#include <cardinal/edit/brush.hpp>

#include <cardinal/core/algorithm.hpp>   // cardinal::min/max
#include <cardinal/core/cmath.hpp>       // cardinal::exp/floor/ceil

namespace cardinal::edit::brush {

float weight_at(float distance_world, const Brush& b) noexcept {
    // `distance_world < 0.0f` was NaN-blind (NaN<0 unordered-false),
    // so a NaN distance flowed past the clamp to `t = NaN / radius =
    // NaN`, then the Gaussian `cardinal::exp(-3*NaN*NaN) = NaN` →
    // NaN weight → applied to heightmap = NaN poison. Treat non-
    // finite as "outside the radius" (weight 0). Note: clamping to
    // 0 instead would be WRONG — that's the maximum-weight position
    // and would falsely apply the brush at full strength.
    if (!cardinal::isfinite(distance_world)) return 0.0f;
    if (distance_world < 0.0f) distance_world = 0.0f;
    if (b.radius_world <= 0.0f || distance_world >= b.radius_world) return 0.0f;
    const float t = distance_world / b.radius_world;   // 0..1
    switch (b.falloff) {
        case Falloff::None:     return 1.0f;
        case Falloff::Linear:   return 1.0f - t;
        case Falloff::Smooth: {
            const float s = 1.0f - t;
            return s * s * (3.0f - 2.0f * s);  // smoothstep
        }
        case Falloff::Gaussian: return cardinal::exp(-3.0f * t * t);
    }
    return 0.0f;
}

CellAabb stamp_height_grid(float* heights, u32 width, u32 height,
                           float cell, float origin_x, float origin_y,
                           float stamp_x, float stamp_y,
                           float dt, const Brush& b) noexcept
{
    CellAabb r{};
    // Same float→int UB-cast pattern as world (dd37414), mesh_ops
    // (439d2eb), tex_ops (3bc8360), scene + physics (ad55e96). The
    // `cell <= 0.0f` and `radius <= 0.0f` ordered guards were NaN-
    // blind; a NaN cell or radius (or stamp_x/y / origin_x/y, the
    // upstream callers being editor brush input) flowed into the
    // gx/gy_min/max casts:
    //   static_cast<int>(floor((NaN ...) / NaN_cell)) → UB
    // Reject any non-finite scalar up-front (returns the empty
    // CellAabb — same contract as cell<=0 / radius<=0).
    if (heights == nullptr || width == 0 || height == 0) return r;
    if (!cardinal::isfinite(cell)     || cell     <= 0.0f) return r;
    const float radius = b.radius_world;
    if (!cardinal::isfinite(radius)   || radius   <= 0.0f) return r;
    if (!cardinal::isfinite(stamp_x)  || !cardinal::isfinite(stamp_y))  return r;
    if (!cardinal::isfinite(origin_x) || !cardinal::isfinite(origin_y)) return r;

    // World-space brush AABB → grid coords.
    const int gx_min = static_cast<int>(cardinal::floor((stamp_x - radius - origin_x) / cell));
    const int gy_min = static_cast<int>(cardinal::floor((stamp_y - radius - origin_y) / cell));
    const int gx_max = static_cast<int>(cardinal::ceil ((stamp_x + radius - origin_x) / cell));
    const int gy_max = static_cast<int>(cardinal::ceil ((stamp_y + radius - origin_y) / cell));

    const int cx_lo = cardinal::max(0, gx_min);
    const int cy_lo = cardinal::max(0, gy_min);
    const int cx_hi = cardinal::min(static_cast<int>(width)  - 1, gx_max);
    const int cy_hi = cardinal::min(static_cast<int>(height) - 1, gy_max);
    if (cx_lo > cx_hi || cy_lo > cy_hi) return r;

    const float dt_strength = b.strength * cardinal::max(0.0f, dt);

    for (int gy = cy_lo; gy <= cy_hi; ++gy) {
        const float wy = origin_y + static_cast<float>(gy) * cell;
        const float dy = wy - stamp_y;
        for (int gx = cx_lo; gx <= cx_hi; ++gx) {
            const float wx = origin_x + static_cast<float>(gx) * cell;
            const float dx = wx - stamp_x;
            const float d  = cardinal::sqrt(dx*dx + dy*dy);
            const float w  = weight_at(d, b);
            if (w <= 0.0f) continue;

            float& h = heights[static_cast<usize>(gy) * width + static_cast<usize>(gx)];
            switch (b.mode) {
                case Mode::Add:      h += w * dt_strength;            break;
                case Mode::Subtract: h -= w * dt_strength;            break;
                case Mode::Set:      h += (b.target_value - h) * w * cardinal::min(1.0f, dt_strength); break;
                case Mode::Smooth: {
                    // Average with 4-neighbours — boundary clamps to self.
                    const u32 ix = static_cast<u32>(gx);
                    const u32 iy = static_cast<u32>(gy);
                    const float n = heights[(iy>0?iy-1:iy)*width + ix];
                    const float s = heights[(iy<height-1?iy+1:iy)*width + ix];
                    const float e = heights[iy*width + (ix<width-1?ix+1:ix)];
                    const float we_= heights[iy*width + (ix>0?ix-1:ix)];
                    const float avg = (n + s + e + we_) * 0.25f;
                    h += (avg - h) * w * cardinal::min(1.0f, dt_strength);
                } break;
                case Mode::Erase:    h += (b.target_value - h) * w * 0.5f * dt_strength; break;
            }
            r.any = true;
            if (gx < static_cast<int>(r.x0) || !r.any) r.x0 = static_cast<u32>(gx);
            if (gy < static_cast<int>(r.y0) || !r.any) r.y0 = static_cast<u32>(gy);
            if (gx > static_cast<int>(r.x1)) r.x1 = static_cast<u32>(gx);
            if (gy > static_cast<int>(r.y1)) r.y1 = static_cast<u32>(gy);
        }
    }
    if (r.any) {
        r.x0 = static_cast<u32>(cx_lo);
        r.y0 = static_cast<u32>(cy_lo);
        r.x1 = static_cast<u32>(cx_hi);
        r.y1 = static_cast<u32>(cy_hi);
    }
    return r;
}

void stamp_generic(const Brush& b, const SampleEnumFn& enumerate,
                   const cardinal::function<void(u32, float)>& apply)
{
    if (!enumerate || !apply) return;
    enumerate([&](u32 idx, float dist_world) {
        const float w = weight_at(dist_world, b);
        if (w > 0.0f) apply(idx, w * b.strength);
    });
}

}  // namespace cardinal::edit::brush
