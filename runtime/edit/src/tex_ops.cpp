#include <cardinal/edit/tex_ops.hpp>

#include <cardinal/core/algorithm.hpp>   // cardinal::min/max/clamp
#include <cardinal/core/cmath.hpp>       // cardinal::floor/pow
#include <cardinal/core/cstring.hpp>     // cardinal::strlen

namespace cardinal::edit::tex_ops {

namespace {

inline u8 fclamp_u8(float f) noexcept {
    // Reject NaN/±Inf BEFORE the cast. The ordered compares below are
    // NaN-blind (NaN < 0 and NaN > 255 are both unordered-false), so
    // a NaN f would fall through to `static_cast<u8>(NaN + 0.5f)` —
    // UNDEFINED BEHAVIOR for the float→int cast on a non-finite
    // source. Multiple callers can produce NaN: noise_fractal's
    // `sum / max(1e-6, norm)` is NaN if persistence is NaN, levels'
    // `pow(NaN, ...)` cascades NaN, and any direct call site with a
    // NaN-poisoned f would hit the same UB. Returning 0 on non-finite
    // matches the "out-of-range clamp" semantic the function intends.
    // Parked-supervised item #4 in feedback_integration_coverage_map.md;
    // also touches the asset/cook pipeline via tex_ops cooker paths
    // (tests/content_pipeline exercises this indirectly).
    if (!cardinal::isfinite(f)) return 0;
    if (f < 0.0f) return 0;
    if (f > 255.0f) return 255;
    return static_cast<u8>(f + 0.5f);
}

// Cheap integer hash → u32. Used by noise generators.
inline u32 hash32(u32 x, u32 y, u32 seed) noexcept {
    u32 h = x * 0x27d4eb2du + y * 0x9e3779b1u + seed * 0x85ebca6bu;
    h ^= h >> 15;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}
inline float hash_f(u32 x, u32 y, u32 seed) noexcept {
    return static_cast<float>(hash32(x, y, seed)) / 4294967295.0f;
}

float smoothstep(float t) noexcept { return t * t * (3.0f - 2.0f * t); }

float value_noise_2d(float x, float y, u32 seed) noexcept {
    const i32 xi = static_cast<i32>(cardinal::floor(x));
    const i32 yi = static_cast<i32>(cardinal::floor(y));
    const float xf = x - xi;
    const float yf = y - yi;
    const float a = hash_f(static_cast<u32>(xi)  , static_cast<u32>(yi)  , seed);
    const float b = hash_f(static_cast<u32>(xi+1), static_cast<u32>(yi)  , seed);
    const float c = hash_f(static_cast<u32>(xi)  , static_cast<u32>(yi+1), seed);
    const float d = hash_f(static_cast<u32>(xi+1), static_cast<u32>(yi+1), seed);
    const float u = smoothstep(xf);
    const float v = smoothstep(yf);
    return (1.0f - v) * ((1.0f - u) * a + u * b)
         +         v  * ((1.0f - u) * c + u * d);
}

}  // namespace

cardinal::vector<u8> solid(u32 w, u32 h, Color c) {
    cardinal::vector<u8> out(static_cast<usize>(w) * h * 4);
    for (usize i = 0; i < out.size(); i += 4) {
        out[i+0] = c.r; out[i+1] = c.g; out[i+2] = c.b; out[i+3] = c.a;
    }
    return out;
}

cardinal::vector<u8> checker(u32 w, u32 h, u32 cells_x, u32 cells_y, Color a, Color b) {
    if (cells_x == 0) cells_x = 1;
    if (cells_y == 0) cells_y = 1;
    cardinal::vector<u8> out(static_cast<usize>(w) * h * 4);
    for (u32 y = 0; y < h; ++y) {
        const u32 cy = (y * cells_y) / h;
        for (u32 x = 0; x < w; ++x) {
            const u32 cx = (x * cells_x) / w;
            const Color c = ((cx + cy) & 1u) ? b : a;
            put_px(out, w, x, y, c);
        }
    }
    return out;
}

cardinal::vector<u8> gradient_linear(u32 w, u32 h, Color a, Color b, Axis axis) {
    cardinal::vector<u8> out(static_cast<usize>(w) * h * 4);
    for (u32 y = 0; y < h; ++y) {
        for (u32 x = 0; x < w; ++x) {
            float t = 0.0f;
            switch (axis) {
                case Axis::X:        t = w > 1 ? static_cast<float>(x) / (w-1) : 0.0f; break;
                case Axis::Y:        t = h > 1 ? static_cast<float>(y) / (h-1) : 0.0f; break;
                case Axis::Diagonal: t = ((float)x + y) / cardinal::max(1u, (w + h - 2u)); break;
            }
            const u8 r = fclamp_u8((1.0f-t) * a.r + t * b.r);
            const u8 g = fclamp_u8((1.0f-t) * a.g + t * b.g);
            const u8 bl= fclamp_u8((1.0f-t) * a.b + t * b.b);
            const u8 al= fclamp_u8((1.0f-t) * a.a + t * b.a);
            put_px(out, w, x, y, Color{r,g,bl,al});
        }
    }
    return out;
}

cardinal::vector<u8> noise_value(u32 w, u32 h, u32 seed, float scale, float contrast) {
    cardinal::vector<u8> out(static_cast<usize>(w) * h * 4);
    if (scale <= 0.0f) scale = 1.0f;
    for (u32 y = 0; y < h; ++y) {
        for (u32 x = 0; x < w; ++x) {
            const float fx = static_cast<float>(x) / w * scale;
            const float fy = static_cast<float>(y) / h * scale;
            float n = value_noise_2d(fx, fy, seed);
            // cardinal::clamp is NaN-passthrough; a NaN contrast →
            // `0.5 + (n-0.5)*NaN = NaN`, clamp returns NaN, then
            // `static_cast<u8>(NaN * 255.0f)` is UB on the float→int
            // cast. Sanitize n to 0 on non-finite BEFORE the cast so
            // existing-byte-identity callers (tests pin contrast==0 →
            // 127 via truncation, NOT rounding — see test_noise_value
            // L177-180) keep the same semantic on finite inputs.
            n = cardinal::clamp(0.5f + (n - 0.5f) * contrast, 0.0f, 1.0f);
            if (!cardinal::isfinite(n)) n = 0.0f;
            const u8 v = static_cast<u8>(n * 255.0f);
            put_px(out, w, x, y, Color{v,v,v,255});
        }
    }
    return out;
}

cardinal::vector<u8> noise_fractal(u32 w, u32 h, u32 seed, float base_scale,
                              u32 octaves, float persistence)
{
    if (octaves == 0) octaves = 1;
    cardinal::vector<u8> out(static_cast<usize>(w) * h * 4);
    if (base_scale <= 0.0f) base_scale = 1.0f;
    for (u32 y = 0; y < h; ++y) {
        for (u32 x = 0; x < w; ++x) {
            float amp = 1.0f, sum = 0.0f, norm = 0.0f, freq = base_scale;
            for (u32 o = 0; o < octaves; ++o) {
                const float fx = static_cast<float>(x) / w * freq;
                const float fy = static_cast<float>(y) / h * freq;
                sum  += amp * value_noise_2d(fx, fy, seed + o * 1013u);
                norm += amp;
                amp  *= persistence;
                freq *= 2.0f;
            }
            const float n = sum / cardinal::max(1e-6f, norm);
            const u8 v = fclamp_u8(n * 255.0f);
            put_px(out, w, x, y, Color{v,v,v,255});
        }
    }
    return out;
}

cardinal::vector<u8> voronoi_cells(u32 w, u32 h, u32 seed, u32 site_count) {
    if (site_count == 0) site_count = 1;
    cardinal::vector<float> sx(site_count), sy(site_count);
    cardinal::vector<Color> sc(site_count);
    for (u32 i = 0; i < site_count; ++i) {
        sx[i] = hash_f(i, 1, seed) * w;
        sy[i] = hash_f(i, 2, seed) * h;
        sc[i] = Color{
            static_cast<u8>(hash_f(i, 3, seed) * 255.0f),
            static_cast<u8>(hash_f(i, 4, seed) * 255.0f),
            static_cast<u8>(hash_f(i, 5, seed) * 255.0f), 255 };
    }
    cardinal::vector<u8> out(static_cast<usize>(w) * h * 4);
    for (u32 y = 0; y < h; ++y) {
        for (u32 x = 0; x < w; ++x) {
            float best = 1e30f;
            u32   idx  = 0;
            for (u32 i = 0; i < site_count; ++i) {
                const float dx = sx[i] - x;
                const float dy = sy[i] - y;
                const float d  = dx*dx + dy*dy;
                if (d < best) { best = d; idx = i; }
            }
            put_px(out, w, x, y, sc[idx]);
        }
    }
    return out;
}

void to_grayscale(cardinal::vector<u8>& img, u32 w, u32 h) {
    for (u32 y = 0; y < h; ++y) for (u32 x = 0; x < w; ++x) {
        const Color c = get_px(img, w, x, y);
        const u8 v = static_cast<u8>(0.299f*c.r + 0.587f*c.g + 0.114f*c.b);
        put_px(img, w, x, y, Color{v,v,v,c.a});
    }
}

void invert(cardinal::vector<u8>& img, u32 w, u32 h) {
    for (u32 y = 0; y < h; ++y) for (u32 x = 0; x < w; ++x) {
        const Color c = get_px(img, w, x, y);
        put_px(img, w, x, y, Color{
            static_cast<u8>(255 - c.r),
            static_cast<u8>(255 - c.g),
            static_cast<u8>(255 - c.b), c.a});
    }
}

void levels(cardinal::vector<u8>& img, u32 w, u32 h,
            float black_pt, float white_pt, float gamma)
{
    if (white_pt <= black_pt) white_pt = black_pt + 1e-3f;
    if (gamma <= 0.0f) gamma = 1.0f;
    for (u32 y = 0; y < h; ++y) for (u32 x = 0; x < w; ++x) {
        const Color c = get_px(img, w, x, y);
        auto remap = [&](u8 v) {
            float f = static_cast<float>(v) / 255.0f;
            f = cardinal::clamp((f - black_pt) / (white_pt - black_pt), 0.0f, 1.0f);
            f = cardinal::pow(f, 1.0f / gamma);
            return fclamp_u8(f * 255.0f);
        };
        put_px(img, w, x, y, Color{remap(c.r), remap(c.g), remap(c.b), c.a});
    }
}

void channel_swap(cardinal::vector<u8>& img, u32 w, u32 h, const char* layout) {
    if (layout == nullptr || cardinal::strlen(layout) < 4) return;
    auto pick = [](char ch, const Color& c) -> u8 {
        switch (ch) {
            case 'r': return c.r;
            case 'g': return c.g;
            case 'b': return c.b;
            case 'a': return c.a;
        }
        return 0;
    };
    for (u32 y = 0; y < h; ++y) for (u32 x = 0; x < w; ++x) {
        const Color c = get_px(img, w, x, y);
        Color out;
        out.r = pick(layout[0], c);
        out.g = pick(layout[1], c);
        out.b = pick(layout[2], c);
        out.a = pick(layout[3], c);
        put_px(img, w, x, y, out);
    }
}

void compose_alpha(cardinal::vector<u8>& dst, const cardinal::vector<u8>& src,
                   u32 w, u32 h)
{
    for (u32 y = 0; y < h; ++y) for (u32 x = 0; x < w; ++x) {
        const Color s = get_px(src, w, x, y);
        const Color d = get_px(dst, w, x, y);
        const float a = static_cast<float>(s.a) / 255.0f;
        const float ia = 1.0f - a;
        Color o{
            static_cast<u8>(s.r * a + d.r * ia),
            static_cast<u8>(s.g * a + d.g * ia),
            static_cast<u8>(s.b * a + d.b * ia),
            static_cast<u8>(cardinal::min(255.0f, s.a + d.a * ia)),
        };
        put_px(dst, w, x, y, o);
    }
}

}  // namespace cardinal::edit::tex_ops
