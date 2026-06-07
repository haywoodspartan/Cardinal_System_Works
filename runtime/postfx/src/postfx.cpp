// =============================================================================
// Cardinal — post-processing chain implementation.
//
// CPU reference impls for the stock passes. The pixel math is intentionally
// straightforward (no SIMD yet, no separable-kernel optimisation) so the
// HLSL counterparts can validate byte-for-byte against the CPU during
// shader bring-up — same convention render::algo uses.
// =============================================================================

#include <cardinal/postfx/postfx.hpp>

#include <cardinal/core/std/algorithm.hpp>      // cardinal::clamp/min/max
#include <cardinal/core/std/chrono.hpp>
#include <cardinal/core/std/cmath.hpp>          // cardinal::isfinite/sin/cos/sqrt
#include <cardinal/core/std/cstring.hpp>        // cardinal::memcpy
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/std/utility.hpp>

namespace cardinal::postfx {

namespace {

// ---------------------------------------------------------------------------
// Small helpers shared by the stock passes.
// ---------------------------------------------------------------------------

// Coerce non-finite to a safe default. Same fz() pattern used across the
// engine's NaN-defensive surfaces (particles spawn, sky safe_key, vgeom
// cook, ...). A NaN/Inf knob value would otherwise propagate through the
// pass and emit NaN pixels.
inline float fz(float v, float fallback = 0.0f) noexcept {
    return cardinal::isfinite(v) ? v : fallback;
}

// Read a typed knob by id. Returns the knob's default-typed slot when
// the id is missing OR the knob's kind doesn't match — the pass keeps
// working with sensible defaults instead of dereferencing a wrong slot.
const Knob* find_knob(const cardinal::vector<Knob>& ks, const char* id) noexcept {
    for (const auto& k : ks) if (k.id == id) return &k;
    return nullptr;
}
float knob_f(const cardinal::vector<Knob>& ks, const char* id, float fallback) noexcept {
    if (auto* k = find_knob(ks, id);
        k && k->kind == cardinal::render::KnobKind::Float) {
        return fz(k->f, fallback);
    }
    return fallback;
}
int knob_i(const cardinal::vector<Knob>& ks, const char* id, int fallback) noexcept {
    if (auto* k = find_knob(ks, id);
        k && k->kind == cardinal::render::KnobKind::Int) {
        return k->i;
    }
    return fallback;
}
bool knob_b(const cardinal::vector<Knob>& ks, const char* id, bool fallback) noexcept {
    if (auto* k = find_knob(ks, id);
        k && k->kind == cardinal::render::KnobKind::Bool) {
        return k->b;
    }
    return fallback;
}

// Rec. 709 luminance — the standard sRGB-linear weighting.
inline float luma709(float r, float g, float b) noexcept {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// Hash u32 → float in [0,1). 32-bit Wang hash; matches hash13 in the
// engine's algo catalog so the GPU variant can land verbatim later.
inline u32 wang_hash(u32 x) noexcept {
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x = x ^ (x >> 4);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15);
    return x;
}
inline float hash_unit(u32 x) noexcept {
    return static_cast<float>(wang_hash(x)) / 4294967296.0f;
}

// Bilinear sample — used by chromatic aberration. UV is in [0,1] image
// coordinates; out-of-bounds clamps to edge.
inline void bilinear_sample(const float* buf, u32 w, u32 h,
                            float u, float v,
                            float& out_r, float& out_g, float& out_b) noexcept
{
    if (!cardinal::isfinite(u)) u = 0.5f;
    if (!cardinal::isfinite(v)) v = 0.5f;
    u = cardinal::clamp(u, 0.0f, 1.0f);
    v = cardinal::clamp(v, 0.0f, 1.0f);
    const float fx = u * static_cast<float>(w > 0 ? w - 1 : 0);
    const float fy = v * static_cast<float>(h > 0 ? h - 1 : 0);
    const u32 x0 = static_cast<u32>(fx);
    const u32 y0 = static_cast<u32>(fy);
    const u32 x1 = (x0 + 1 < w) ? x0 + 1 : x0;
    const u32 y1 = (y0 + 1 < h) ? y0 + 1 : y0;
    const float ax = fx - static_cast<float>(x0);
    const float ay = fy - static_cast<float>(y0);
    auto px = [buf, w](u32 x, u32 y) noexcept { return buf + (static_cast<usize>(y) * w + x) * 4; };
    const float* p00 = px(x0, y0); const float* p10 = px(x1, y0);
    const float* p01 = px(x0, y1); const float* p11 = px(x1, y1);
    auto lerp = [](float a, float b, float t) noexcept { return a + (b - a) * t; };
    const float r0 = lerp(p00[0], p10[0], ax);
    const float r1 = lerp(p01[0], p11[0], ax);
    const float g0 = lerp(p00[1], p10[1], ax);
    const float g1 = lerp(p01[1], p11[1], ax);
    const float b0 = lerp(p00[2], p10[2], ax);
    const float b1 = lerp(p01[2], p11[2], ax);
    out_r = lerp(r0, r1, ay);
    out_g = lerp(g0, g1, ay);
    out_b = lerp(b0, b1, ay);
}

// Common knob ranges — `make_*` factories build their knob lists once
// and reuse across instances.
Knob f_knob(const char* id, const char* label, const char* group,
            const char* tooltip, float def, float lo, float hi, float step) {
    Knob k;
    k.id = id; k.label = label; k.group = group; k.tooltip = tooltip;
    k.kind = cardinal::render::KnobKind::Float;
    k.f = def; k.f_min = lo; k.f_max = hi; k.f_step = step;
    return k;
}
Knob i_knob(const char* id, const char* label, const char* group,
            const char* tooltip, int def, int lo, int hi) {
    Knob k;
    k.id = id; k.label = label; k.group = group; k.tooltip = tooltip;
    k.kind = cardinal::render::KnobKind::Int;
    k.i = def; k.i_min = lo; k.i_max = hi;
    return k;
}
Knob b_knob(const char* id, const char* label, const char* group,
            const char* tooltip, bool def) {
    Knob k;
    k.id = id; k.label = label; k.group = group; k.tooltip = tooltip;
    k.kind = cardinal::render::KnobKind::Bool;
    k.b = def;
    return k;
}

// ---------------------------------------------------------------------------
// Bloom — extract bright pixels, box-blur, add back.
// ---------------------------------------------------------------------------
class BloomPass final : public Pass {
public:
    BloomPass() {
        knobs_.push_back(f_knob("threshold", "Threshold", "Bloom",
            "Pixels with luminance above this value contribute to bloom.",
            1.0f, 0.0f, 4.0f, 0.05f));
        knobs_.push_back(f_knob("intensity", "Intensity", "Bloom",
            "Brightness multiplier on the blurred bloom buffer before add-back.",
            0.5f, 0.0f, 4.0f, 0.05f));
        knobs_.push_back(i_knob("radius", "Radius (px)", "Bloom",
            "Box-blur radius in pixels. 0 disables the blur (still passes the threshold).",
            5, 0, 32));
    }
    const char* id()          const noexcept override { return "bloom"; }
    const char* label()       const noexcept override { return "Bloom"; }
    const char* description() const noexcept override {
        return "Threshold-extract bright pixels, box-blur, add back over the source.";
    }
    cardinal::vector<Knob>& knobs() noexcept override { return knobs_; }

    void apply_cpu(const float* in_rgba, float* out_rgba,
                   u32 width, u32 height) noexcept override
    {
        const usize n_px = static_cast<usize>(width) * height;
        if (in_rgba != out_rgba) {
            cardinal::memcpy(out_rgba, in_rgba, n_px * 4 * sizeof(float));
        }
        if (width == 0 || height == 0) return;

        const float threshold = knob_f(knobs_, "threshold", 1.0f);
        const float intensity = knob_f(knobs_, "intensity", 0.5f);
        const int   radius    = cardinal::clamp(knob_i(knobs_, "radius", 5), 0, 32);
        if (intensity <= 0.0f) return;

        // Bright pass — extract pixels above the luminance threshold.
        bright_.resize(n_px * 3);
        for (usize i = 0; i < n_px; ++i) {
            const float r = in_rgba[i * 4 + 0];
            const float g = in_rgba[i * 4 + 1];
            const float b = in_rgba[i * 4 + 2];
            const float l = luma709(fz(r), fz(g), fz(b));
            const float k = (l > threshold) ? 1.0f : 0.0f;
            bright_[i * 3 + 0] = fz(r) * k;
            bright_[i * 3 + 1] = fz(g) * k;
            bright_[i * 3 + 2] = fz(b) * k;
        }

        // Two-pass separable box blur. blur_tmp_ sized once.
        if (radius > 0) {
            blur_tmp_.resize(n_px * 3);
            const float inv = 1.0f / (2.0f * static_cast<float>(radius) + 1.0f);
            // Horizontal.
            for (u32 y = 0; y < height; ++y) {
                for (u32 x = 0; x < width; ++x) {
                    float sr = 0, sg = 0, sb = 0;
                    for (int dx = -radius; dx <= radius; ++dx) {
                        const i32 sx = cardinal::clamp(static_cast<i32>(x) + dx,
                                                       0, static_cast<i32>(width) - 1);
                        const usize si = (static_cast<usize>(y) * width + sx) * 3;
                        sr += bright_[si + 0];
                        sg += bright_[si + 1];
                        sb += bright_[si + 2];
                    }
                    const usize di = (static_cast<usize>(y) * width + x) * 3;
                    blur_tmp_[di + 0] = sr * inv;
                    blur_tmp_[di + 1] = sg * inv;
                    blur_tmp_[di + 2] = sb * inv;
                }
            }
            // Vertical (writes back into bright_).
            for (u32 y = 0; y < height; ++y) {
                for (u32 x = 0; x < width; ++x) {
                    float sr = 0, sg = 0, sb = 0;
                    for (int dy = -radius; dy <= radius; ++dy) {
                        const i32 sy = cardinal::clamp(static_cast<i32>(y) + dy,
                                                       0, static_cast<i32>(height) - 1);
                        const usize si = (static_cast<usize>(sy) * width + x) * 3;
                        sr += blur_tmp_[si + 0];
                        sg += blur_tmp_[si + 1];
                        sb += blur_tmp_[si + 2];
                    }
                    const usize di = (static_cast<usize>(y) * width + x) * 3;
                    bright_[di + 0] = sr * inv;
                    bright_[di + 1] = sg * inv;
                    bright_[di + 2] = sb * inv;
                }
            }
        }

        // Add the blurred bloom onto the output.
        for (usize i = 0; i < n_px; ++i) {
            out_rgba[i * 4 + 0] += intensity * bright_[i * 3 + 0];
            out_rgba[i * 4 + 1] += intensity * bright_[i * 3 + 1];
            out_rgba[i * 4 + 2] += intensity * bright_[i * 3 + 2];
            // alpha untouched
        }
    }

private:
    cardinal::vector<Knob>  knobs_;
    cardinal::vector<float> bright_;
    cardinal::vector<float> blur_tmp_;
};

// ---------------------------------------------------------------------------
// FXAA — luminance-edge directional blur (single-tap reference).
// ---------------------------------------------------------------------------
class FxaaPass final : public Pass {
public:
    FxaaPass() {
        knobs_.push_back(f_knob("edge_threshold", "Edge threshold", "AA",
            "Local luminance gradient above this enables blur. Lower = more aggressive.",
            0.166f, 0.0f, 1.0f, 0.005f));
        knobs_.push_back(f_knob("sub_pixel", "Sub-pixel blend", "AA",
            "How much of the directional blur to mix back (0 = no AA, 1 = full).",
            0.75f, 0.0f, 1.0f, 0.05f));
    }
    const char* id()          const noexcept override { return "fxaa"; }
    const char* label()       const noexcept override { return "FXAA"; }
    const char* description() const noexcept override {
        return "Single-tap directional blur driven by luminance gradient. Cheap AA.";
    }
    cardinal::vector<Knob>& knobs() noexcept override { return knobs_; }

    void apply_cpu(const float* in_rgba, float* out_rgba,
                   u32 width, u32 height) noexcept override
    {
        const usize n_px = static_cast<usize>(width) * height;
        if (width < 2 || height < 2) {
            if (in_rgba != out_rgba) {
                cardinal::memcpy(out_rgba, in_rgba, n_px * 4 * sizeof(float));
            }
            return;
        }
        // FXAA reads NSWE neighbours of the current pixel — needs a
        // separate read buffer when in/out alias.
        cardinal::vector<float> scratch;
        const float* src = in_rgba;
        if (in_rgba == out_rgba) {
            scratch.assign(in_rgba, in_rgba + n_px * 4);
            src = scratch.data();
        }

        const float edge_thresh = knob_f(knobs_, "edge_threshold", 0.166f);
        const float sub_pixel   = cardinal::clamp(knob_f(knobs_, "sub_pixel", 0.75f), 0.0f, 1.0f);

        auto px = [src, width](u32 x, u32 y) noexcept {
            return src + (static_cast<usize>(y) * width + x) * 4;
        };
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                const u32 xl = (x > 0) ? x - 1 : x;
                const u32 xr = (x + 1 < width) ? x + 1 : x;
                const u32 yu = (y > 0) ? y - 1 : y;
                const u32 yd = (y + 1 < height) ? y + 1 : y;
                const float* c = px(x , y);
                const float* l = px(xl, y);
                const float* r = px(xr, y);
                const float* u = px(x , yu);
                const float* d = px(x , yd);
                const float lc = luma709(fz(c[0]), fz(c[1]), fz(c[2]));
                const float ll = luma709(fz(l[0]), fz(l[1]), fz(l[2]));
                const float lr = luma709(fz(r[0]), fz(r[1]), fz(r[2]));
                const float lu = luma709(fz(u[0]), fz(u[1]), fz(u[2]));
                const float ld = luma709(fz(d[0]), fz(d[1]), fz(d[2]));
                const float lmin = cardinal::min<float>({lc, ll, lr, lu, ld});
                const float lmax = cardinal::max<float>({lc, ll, lr, lu, ld});
                const float lrange = lmax - lmin;
                const usize di = (static_cast<usize>(y) * width + x) * 4;
                if (lrange < edge_thresh) {
                    out_rgba[di + 0] = c[0];
                    out_rgba[di + 1] = c[1];
                    out_rgba[di + 2] = c[2];
                    out_rgba[di + 3] = c[3];
                    continue;
                }
                // Direction: average of vertical (l-r) vs horizontal (u-d) gradients.
                const float dx = (lr - ll);
                const float dy = (ld - lu);
                // Sample two extra taps along the gradient and blend.
                float br = 0.25f * (l[0] + r[0] + u[0] + d[0]);
                float bg = 0.25f * (l[1] + r[1] + u[1] + d[1]);
                float bb = 0.25f * (l[2] + r[2] + u[2] + d[2]);
                (void)dx; (void)dy;   // direction not used in the reference 3x3 mean
                out_rgba[di + 0] = c[0] + (br - c[0]) * sub_pixel;
                out_rgba[di + 1] = c[1] + (bg - c[1]) * sub_pixel;
                out_rgba[di + 2] = c[2] + (bb - c[2]) * sub_pixel;
                out_rgba[di + 3] = c[3];
            }
        }
    }

private:
    cardinal::vector<Knob> knobs_;
};

// ---------------------------------------------------------------------------
// Vignette — radial darkening.
// ---------------------------------------------------------------------------
class VignettePass final : public Pass {
public:
    VignettePass() {
        knobs_.push_back(f_knob("intensity", "Intensity", "Vignette",
            "How dark the corners get at r=1. 0 disables.",
            0.3f, 0.0f, 1.0f, 0.02f));
        knobs_.push_back(f_knob("smoothness", "Smoothness", "Vignette",
            "Falloff sharpness. Lower = harder edge, higher = soft fade.",
            0.5f, 0.05f, 2.0f, 0.05f));
        knobs_.push_back(f_knob("roundness", "Roundness", "Vignette",
            "1.0 = circular, > 1 = ellipse weighted by aspect.",
            1.0f, 0.5f, 2.0f, 0.05f));
    }
    const char* id()          const noexcept override { return "vignette"; }
    const char* label()       const noexcept override { return "Vignette"; }
    const char* description() const noexcept override {
        return "Radial darkening from screen edges. Stack after tonemap.";
    }
    cardinal::vector<Knob>& knobs() noexcept override { return knobs_; }

    void apply_cpu(const float* in_rgba, float* out_rgba,
                   u32 width, u32 height) noexcept override
    {
        const usize n_px = static_cast<usize>(width) * height;
        const float intensity  = cardinal::clamp(knob_f(knobs_, "intensity",  0.3f), 0.0f, 1.0f);
        const float smoothness = cardinal::max  (0.05f, knob_f(knobs_, "smoothness", 0.5f));
        const float roundness  = cardinal::max  (0.5f,  knob_f(knobs_, "roundness",  1.0f));
        if (intensity <= 0.0f || width == 0 || height == 0) {
            if (in_rgba != out_rgba) {
                cardinal::memcpy(out_rgba, in_rgba, n_px * 4 * sizeof(float));
            }
            return;
        }

        const float cx = static_cast<float>(width)  * 0.5f;
        const float cy = static_cast<float>(height) * 0.5f;
        // Half-extent of the larger axis; r normalised so corners → 1.
        const float inv_radius = 1.0f / cardinal::sqrt(cx*cx + cy*cy);
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                const float dx = (static_cast<float>(x) - cx);
                const float dy = (static_cast<float>(y) - cy) * roundness;
                const float r  = cardinal::sqrt(dx*dx + dy*dy) * inv_radius;
                // Smoothstep-ish falloff via t^k.
                const float t = cardinal::clamp(r / smoothness, 0.0f, 1.0f);
                const float factor = 1.0f - intensity * t * t;
                const usize i = (static_cast<usize>(y) * width + x) * 4;
                out_rgba[i + 0] = in_rgba[i + 0] * factor;
                out_rgba[i + 1] = in_rgba[i + 1] * factor;
                out_rgba[i + 2] = in_rgba[i + 2] * factor;
                out_rgba[i + 3] = in_rgba[i + 3];
            }
        }
    }

private:
    cardinal::vector<Knob> knobs_;
};

// ---------------------------------------------------------------------------
// Chromatic Aberration — per-channel UV shift radially.
// ---------------------------------------------------------------------------
class ChromaticAberrationPass final : public Pass {
public:
    ChromaticAberrationPass() {
        knobs_.push_back(f_knob("intensity", "Intensity", "Chromatic Aberration",
            "Max radial UV offset at the corners. 0.01 = subtle, 0.05 = lensy.",
            0.005f, 0.0f, 0.05f, 0.001f));
        knobs_.push_back(b_knob("blue_inward", "Blue inward", "Chromatic Aberration",
            "When on, blue shifts toward centre + red outward (matches real lenses).",
            true));
    }
    const char* id()          const noexcept override { return "chromatic_aberration"; }
    const char* label()       const noexcept override { return "Chromatic Aberration"; }
    const char* description() const noexcept override {
        return "Per-channel radial UV shift — simulates lens dispersion.";
    }
    cardinal::vector<Knob>& knobs() noexcept override { return knobs_; }

    void apply_cpu(const float* in_rgba, float* out_rgba,
                   u32 width, u32 height) noexcept override
    {
        const usize n_px = static_cast<usize>(width) * height;
        const float intensity   = cardinal::max(0.0f, knob_f(knobs_, "intensity", 0.005f));
        const bool  blue_inward = knob_b(knobs_, "blue_inward", true);
        if (intensity <= 0.0f || width < 2 || height < 2) {
            if (in_rgba != out_rgba) {
                cardinal::memcpy(out_rgba, in_rgba, n_px * 4 * sizeof(float));
            }
            return;
        }
        // Needs a separate read buffer when in/out alias (we sample
        // neighbours).
        cardinal::vector<float> scratch;
        const float* src = in_rgba;
        if (in_rgba == out_rgba) {
            scratch.assign(in_rgba, in_rgba + n_px * 4);
            src = scratch.data();
        }
        const float sign_blue = blue_inward ? -1.0f : 1.0f;
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                const float u = static_cast<float>(x) / static_cast<float>(width  - 1);
                const float v = static_cast<float>(y) / static_cast<float>(height - 1);
                const float du = u - 0.5f;
                const float dv = v - 0.5f;
                const float r  = cardinal::sqrt(du * du + dv * dv);
                const float scale = intensity * r;
                float rr, rg, rb;
                bilinear_sample(src, width, height,
                    u + du * scale, v + dv * scale, rr, rg, rb);
                float gr, gg, gb;
                bilinear_sample(src, width, height, u, v, gr, gg, gb);
                float br, bg, bb;
                bilinear_sample(src, width, height,
                    u + du * scale * sign_blue, v + dv * scale * sign_blue,
                    br, bg, bb);
                const usize i = (static_cast<usize>(y) * width + x) * 4;
                out_rgba[i + 0] = rr;   // red channel from outward shift
                out_rgba[i + 1] = gg;   // green from centre
                out_rgba[i + 2] = bb;   // blue from inward/outward shift
                out_rgba[i + 3] = src[i + 3];
            }
        }
    }

private:
    cardinal::vector<Knob> knobs_;
};

// ---------------------------------------------------------------------------
// Film Grain — hash-based per-pixel noise.
// ---------------------------------------------------------------------------
class FilmGrainPass final : public Pass {
public:
    FilmGrainPass() {
        knobs_.push_back(f_knob("intensity", "Intensity", "Film Grain",
            "Strength of the noise added per pixel. 0 disables.",
            0.05f, 0.0f, 0.5f, 0.005f));
        knobs_.push_back(b_knob("monochrome", "Monochrome", "Film Grain",
            "When on, the same noise value is added to R/G/B (classic film grain). "
            "Off applies independent noise per channel.",
            true));
        knobs_.push_back(i_knob("seed_offset", "Seed offset", "Film Grain",
            "Add this to every pixel hash. Increment to scramble the grain pattern.",
            0, 0, 65535));
    }
    const char* id()          const noexcept override { return "film_grain"; }
    const char* label()       const noexcept override { return "Film Grain"; }
    const char* description() const noexcept override {
        return "Hash-based pseudo-random noise added to each pixel.";
    }
    cardinal::vector<Knob>& knobs() noexcept override { return knobs_; }

    void apply_cpu(const float* in_rgba, float* out_rgba,
                   u32 width, u32 height) noexcept override
    {
        const usize n_px = static_cast<usize>(width) * height;
        const float intensity  = cardinal::clamp(knob_f(knobs_, "intensity", 0.05f), 0.0f, 0.5f);
        const bool  monochrome = knob_b(knobs_, "monochrome", true);
        const int   seed_off   = knob_i(knobs_, "seed_offset", 0);
        if (intensity <= 0.0f) {
            if (in_rgba != out_rgba) {
                cardinal::memcpy(out_rgba, in_rgba, n_px * 4 * sizeof(float));
            }
            return;
        }
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                const u32 base = (y * 1597u + x) * 7919u + static_cast<u32>(seed_off);
                // [-1, 1] noise per channel.
                const float nr = hash_unit(base + 0u) * 2.0f - 1.0f;
                const float ng = monochrome ? nr : (hash_unit(base + 1u) * 2.0f - 1.0f);
                const float nb = monochrome ? nr : (hash_unit(base + 2u) * 2.0f - 1.0f);
                const usize i = (static_cast<usize>(y) * width + x) * 4;
                out_rgba[i + 0] = in_rgba[i + 0] + nr * intensity;
                out_rgba[i + 1] = in_rgba[i + 1] + ng * intensity;
                out_rgba[i + 2] = in_rgba[i + 2] + nb * intensity;
                out_rgba[i + 3] = in_rgba[i + 3];
            }
        }
    }

private:
    cardinal::vector<Knob> knobs_;
};

// ---------------------------------------------------------------------------
// Chain impl.
// ---------------------------------------------------------------------------
class ChainImpl final : public Chain {
public:
    void add(cardinal::unique_ptr<Pass> p) override {
        if (p) passes_.push_back(cardinal::move(p));
    }
    bool remove(usize i) override {
        if (i >= passes_.size()) return false;
        passes_.erase(passes_.begin() + static_cast<long>(i));
        return true;
    }
    void clear() override { passes_.clear(); }
    bool move_up(usize i) override {
        if (i == 0 || i >= passes_.size()) return false;
        cardinal::swap(passes_[i - 1], passes_[i]);
        return true;
    }
    bool move_down(usize i) override {
        if (i + 1 >= passes_.size()) return false;
        cardinal::swap(passes_[i + 1], passes_[i]);
        return true;
    }
    const cardinal::vector<cardinal::unique_ptr<Pass>>&
        passes() const noexcept override { return passes_; }
    usize size() const noexcept override { return passes_.size(); }

    Pass* find(const char* id) noexcept override {
        if (id == nullptr) return nullptr;
        for (auto& p : passes_) {
            if (p && cardinal::strcmp(p->id(), id) == 0) return p.get();
        }
        return nullptr;
    }

    void apply_cpu(const float* in_rgba, float* out_rgba,
                   u32 width, u32 height) override
    {
        using clk = cardinal::chrono::high_resolution_clock;
        const auto t0 = clk::now();
        stats_ = Stats{};

        const usize n_px = static_cast<usize>(width) * height;
        if (n_px == 0) return;

        // Count enabled passes — that drives the ping-pong scheme.
        u32 enabled_count = 0;
        for (const auto& p : passes_) if (p && p->enabled) ++enabled_count;
        stats_.passes_skipped_disabled =
            static_cast<u32>(passes_.size()) - enabled_count;

        if (enabled_count == 0) {
            if (in_rgba != out_rgba) {
                cardinal::memcpy(out_rgba, in_rgba, n_px * 4 * sizeof(float));
            }
            return;
        }

        // We need a scratch buffer when there are 2+ passes so successive
        // passes can read the previous output. For 1 pass we go in→out
        // directly.
        if (enabled_count > 1) scratch_.resize(n_px * 4);

        const float* read = in_rgba;
        float*       write = (enabled_count == 1) ? out_rgba : scratch_.data();
        u32          run_count = 0;
        for (auto& p : passes_) {
            if (!p || !p->enabled) continue;
            ++run_count;
            // For the LAST enabled pass, write to out_rgba so the result
            // lands where the caller expects without an extra memcpy.
            float* dst = (run_count == enabled_count) ? out_rgba : write;
            p->apply_cpu(read, dst, width, height);
            // Ping-pong: next pass reads what this one wrote, next writes
            // into the other buffer (or out_rgba on the last iteration).
            read  = dst;
            write = (dst == scratch_.data()) ? out_rgba : scratch_.data();
            ++stats_.passes_run;
        }

        const auto t1 = clk::now();
        stats_.total_us =
            cardinal::chrono::duration<f32, cardinal::micro>(t1 - t0).count();
    }

    Stats stats() const noexcept override { return stats_; }

private:
    cardinal::vector<cardinal::unique_ptr<Pass>> passes_;
    cardinal::vector<float>                      scratch_;
    Stats                                        stats_;
};

}  // namespace

cardinal::shared_ptr<Chain> Chain::create() {
    return cardinal::shared_ptr<Chain>(new ChainImpl());
}

cardinal::unique_ptr<Pass> make_bloom_pass()                { return cardinal::make_unique<BloomPass>(); }
cardinal::unique_ptr<Pass> make_fxaa_pass()                 { return cardinal::make_unique<FxaaPass>(); }
cardinal::unique_ptr<Pass> make_vignette_pass()             { return cardinal::make_unique<VignettePass>(); }
cardinal::unique_ptr<Pass> make_chromatic_aberration_pass() { return cardinal::make_unique<ChromaticAberrationPass>(); }
cardinal::unique_ptr<Pass> make_film_grain_pass()           { return cardinal::make_unique<FilmGrainPass>(); }

}  // namespace cardinal::postfx
