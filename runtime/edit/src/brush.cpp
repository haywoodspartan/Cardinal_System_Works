#include <cardinal/edit/brush.hpp>

#include <algorithm>
#include <cmath>

namespace cardinal::edit::brush {

float weight_at(float distance_world, const Brush& b) noexcept {
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
        case Falloff::Gaussian: return std::exp(-3.0f * t * t);
    }
    return 0.0f;
}

CellAabb stamp_height_grid(float* heights, u32 width, u32 height,
                           float cell, float origin_x, float origin_y,
                           float stamp_x, float stamp_y,
                           float dt, const Brush& b) noexcept
{
    CellAabb r{};
    if (heights == nullptr || width == 0 || height == 0 || cell <= 0.0f) return r;
    const float radius = b.radius_world;
    if (radius <= 0.0f) return r;

    // World-space brush AABB → grid coords.
    const int gx_min = static_cast<int>(std::floor((stamp_x - radius - origin_x) / cell));
    const int gy_min = static_cast<int>(std::floor((stamp_y - radius - origin_y) / cell));
    const int gx_max = static_cast<int>(std::ceil ((stamp_x + radius - origin_x) / cell));
    const int gy_max = static_cast<int>(std::ceil ((stamp_y + radius - origin_y) / cell));

    const int cx_lo = std::max(0, gx_min);
    const int cy_lo = std::max(0, gy_min);
    const int cx_hi = std::min(static_cast<int>(width)  - 1, gx_max);
    const int cy_hi = std::min(static_cast<int>(height) - 1, gy_max);
    if (cx_lo > cx_hi || cy_lo > cy_hi) return r;

    const float dt_strength = b.strength * std::max(0.0f, dt);

    for (int gy = cy_lo; gy <= cy_hi; ++gy) {
        const float wy = origin_y + static_cast<float>(gy) * cell;
        const float dy = wy - stamp_y;
        for (int gx = cx_lo; gx <= cx_hi; ++gx) {
            const float wx = origin_x + static_cast<float>(gx) * cell;
            const float dx = wx - stamp_x;
            const float d  = std::sqrt(dx*dx + dy*dy);
            const float w  = weight_at(d, b);
            if (w <= 0.0f) continue;

            float& h = heights[static_cast<usize>(gy) * width + static_cast<usize>(gx)];
            switch (b.mode) {
                case Mode::Add:      h += w * dt_strength;            break;
                case Mode::Subtract: h -= w * dt_strength;            break;
                case Mode::Set:      h += (b.target_value - h) * w * std::min(1.0f, dt_strength); break;
                case Mode::Smooth: {
                    // Average with 4-neighbours — boundary clamps to self.
                    const u32 ix = static_cast<u32>(gx);
                    const u32 iy = static_cast<u32>(gy);
                    const float n = heights[(iy>0?iy-1:iy)*width + ix];
                    const float s = heights[(iy<height-1?iy+1:iy)*width + ix];
                    const float e = heights[iy*width + (ix<width-1?ix+1:ix)];
                    const float we_= heights[iy*width + (ix>0?ix-1:ix)];
                    const float avg = (n + s + e + we_) * 0.25f;
                    h += (avg - h) * w * std::min(1.0f, dt_strength);
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
                   const std::function<void(u32, float)>& apply)
{
    if (!enumerate || !apply) return;
    enumerate([&](u32 idx, float dist_world) {
        const float w = weight_at(dist_world, b);
        if (w > 0.0f) apply(idx, w * b.strength);
    });
}

}  // namespace cardinal::edit::brush
