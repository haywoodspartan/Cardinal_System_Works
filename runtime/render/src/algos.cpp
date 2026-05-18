// =============================================================================
// Cardinal — runtime-selectable math algorithms.
//
// Curated set seeded at engine init; plugins / users can register more.
//
// CPU reference impls in this file match their HLSL counterparts in
// shaders/include/cardinal_*.hlsli byte-for-byte where the math allows.
// (Float precision varies between SSE and SM 6.5; the tests assert
// agreement to 1e-5.)
// =============================================================================
#include <cardinal/render/algos.hpp>

#include <cardinal/core/log.hpp>
#include <cardinal/render/precision.hpp>

#include <array>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace cardinal::render::algo {

const char* category_name(CategoryId id) noexcept {
    switch (id) {
        case CategoryId::Tonemap:       return "Tonemap";
        case CategoryId::HashRng:       return "Hash / RNG";
        case CategoryId::MipFilter:     return "Mip Filter";
        case CategoryId::ClusterCull:   return "Cluster Culling";
        case CategoryId::TessFactor:    return "Tessellation Factor";
        case CategoryId::Sampling:      return "Sampling Pattern";
        case CategoryId::NormalEncode:  return "Normal Encoding";
        case CategoryId::PrecisionMath: return "Precision Math";
        case CategoryId::Count_:        return "?";
    }
    return "?";
}

const char* category_description(CategoryId id) noexcept {
    switch (id) {
        case CategoryId::Tonemap:
            return "Maps HDR scene radiance into the [0,1] display range.";
        case CategoryId::HashRng:
            return "Pseudo-random / hash function for shader noise + dithering.";
        case CategoryId::MipFilter:
            return "Downsample kernel used when generating mip levels.";
        case CategoryId::ClusterCull:
            return "Visibility test applied to each meshlet before drawing.";
        case CategoryId::TessFactor:
            return "Hull-shader policy that picks the per-edge tess factor.";
        case CategoryId::Sampling:
            return "Low-discrepancy sequence for AA / GI / ambient.";
        case CategoryId::NormalEncode:
            return "Compression scheme for storing surface normals in 32 bits.";
        case CategoryId::PrecisionMath:
            return "FP8 / FP4 packed-math precision used by compute kernels and "
                   "bandwidth-bound intermediates (probes, neural materials).";
        case CategoryId::Count_: return "?";
    }
    return "?";
}

namespace {

// Common helper math ---------------------------------------------------------
inline float saturate(float x) noexcept { return std::min(1.0f, std::max(0.0f, x)); }
inline void  saturate3(float* c)  noexcept { for (int i=0;i<3;++i) c[i] = saturate(c[i]); }

// Tonemap operators (linear in / display out, normalised so 1.0 input → ~1.0 output).
void tonemap_linear (const AlgoIn& in, AlgoOut& out) {
    for (int i = 0; i < 3; ++i) out.color3[i] = saturate(in.color3[i]);
}
void tonemap_reinhard(const AlgoIn& in, AlgoOut& out) {
    for (int i = 0; i < 3; ++i) {
        const float x = in.color3[i];
        out.color3[i] = saturate(x / (x + 1.0f));
    }
}
void tonemap_reinhard_ext(const AlgoIn& in, AlgoOut& out) {
    constexpr float white_point = 4.0f;
    for (int i = 0; i < 3; ++i) {
        const float x = in.color3[i];
        out.color3[i] = saturate((x * (1.0f + x / (white_point * white_point))) / (x + 1.0f));
    }
}
// John Hable / Uncharted 2 filmic.
void tonemap_filmic(const AlgoIn& in, AlgoOut& out) {
    auto f = [](float x) {
        constexpr float A = 0.15f, B = 0.50f, C = 0.10f;
        constexpr float D = 0.20f, E = 0.02f, F = 0.30f;
        return ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E / F;
    };
    constexpr float W = 11.2f;
    const float white = f(W);
    for (int i = 0; i < 3; ++i) out.color3[i] = saturate(f(in.color3[i] * 2.0f) / white);
}
// Krzysztof Narkowicz approximation — single rational, very cheap.
void tonemap_aces_approx(const AlgoIn& in, AlgoOut& out) {
    constexpr float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    for (int i = 0; i < 3; ++i) {
        const float x = in.color3[i];
        out.color3[i] = saturate((x*(a*x + b)) / (x*(c*x + d) + e));
    }
}
// Stephen Hill's 3x3 RRT/ODT-fitted curve. Applied per-channel for the
// reference impl; production uses a colour-matrix variant.
void tonemap_aces_full(const AlgoIn& in, AlgoOut& out) {
    auto fit = [](float v) {
        const float a = v * (v + 0.0245786f) - 0.000090537f;
        const float b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
        return a / b;
    };
    for (int i = 0; i < 3; ++i) out.color3[i] = saturate(fit(in.color3[i]));
}
// Timothy Lottes 2016 — perceptual roll-off with adjustable shoulder.
void tonemap_lottes(const AlgoIn& in, AlgoOut& out) {
    constexpr float a = 1.6f, d = 0.977f, hdr_max = 8.0f, mid_in = 0.18f, mid_out = 0.267f;
    const float ad   = std::pow(hdr_max, a*d) - std::pow(mid_in, a*d);
    const float bb   = -std::pow(mid_in, a) + (std::pow(hdr_max, a*d) * mid_out -
                                               std::pow(hdr_max, a) * mid_out * std::pow(mid_in, a*d) /
                                               (std::pow(mid_in, a) * mid_out)) /
                                              (std::pow(hdr_max, a*d) - std::pow(mid_in, a*d));
    const float c    = (std::pow(hdr_max, a*d) * mid_out - std::pow(hdr_max, a) * mid_out *
                        std::pow(mid_in, a*d) / (std::pow(mid_in, a) * mid_out)) / ad;
    for (int i = 0; i < 3; ++i) {
        const float x = std::max(in.color3[i], 0.0f);
        const float pa  = std::pow(x, a);
        out.color3[i] = saturate(pa / (std::pow(pa, d) * bb + c));
    }
}

// Hash / RNG — return a unit3 (deterministic given seed).
void hash_wang(const AlgoIn& in, AlgoOut& out) {
    u32 v = in.seed;
    v = (v ^ 61u) ^ (v >> 16);
    v *= 9u;
    v ^= (v >> 4);
    v *= 0x27d4eb2du;
    v ^= (v >> 15);
    const float f = static_cast<float>(v & 0xFFFFFFu) / 16777216.0f;
    out.unit3[0] = f;
    out.unit3[1] = std::fmod(f * 1.6180339887f, 1.0f);
    out.unit3[2] = std::fmod(f * 2.5034518f, 1.0f);
}
void hash_pcg(const AlgoIn& in, AlgoOut& out) {
    // Melissa O'Neill's PCG single-step
    u32 state = in.seed * 747796405u + 2891336453u;
    u32 word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    u32 v     = (word >> 22u) ^ word;
    const float f = static_cast<float>(v & 0xFFFFFFu) / 16777216.0f;
    out.unit3[0] = f;
    out.unit3[1] = std::fmod(f * 1.6180339887f, 1.0f);
    out.unit3[2] = std::fmod(f * 2.5034518f, 1.0f);
}
void hash_hash13(const AlgoIn& in, AlgoOut& out) {
    // 3D Hash13 — float xyz seed (we fake from u32) → 1 float
    const float x = static_cast<float>((in.seed >> 0)  & 0xFF);
    const float y = static_cast<float>((in.seed >> 8)  & 0xFF);
    const float z = static_cast<float>((in.seed >> 16) & 0xFF);
    auto frac = [](float v) { return v - std::floor(v); };
    auto fract = [&](float v) {
        return frac(v * 0.1031f);
    };
    auto h = [&](float a, float b, float c) {
        float p = fract(a + b * 17.0f + c * 113.0f);
        return frac(p * (p + 33.33f) * 100.0f);
    };
    const float f = h(x, y, z);
    out.unit3[0] = f;
    out.unit3[1] = h(y, z, x);
    out.unit3[2] = h(z, x, y);
}
void hash_ign(const AlgoIn& in, AlgoOut& out) {
    // Jorge Jimenez's interleaved-gradient noise — designed for sampling
    // patterns at specific (px,py) coordinates. Here we treat (seed&0xFFFF,
    // seed>>16) as the (x,y) screen coordinates.
    const float x = static_cast<float>(in.seed & 0xFFFFu);
    const float y = static_cast<float>(in.seed >> 16);
    const float f = std::fmod(52.9829189f * std::fmod(0.06711056f*x + 0.00583715f*y, 1.0f), 1.0f);
    out.unit3[0] = f; out.unit3[1] = f; out.unit3[2] = f;
}

// Mip filter kernels — input is a 2x2 footprint (4 RGB samples).
void mip_box(const AlgoIn& in, AlgoOut& out) {
    for (int c = 0; c < 3; ++c) {
        out.color3[c] = 0.25f * (in.samples4[c+0] + in.samples4[c+3] +
                                 in.samples4[c+6] + in.samples4[c+9]);
    }
}
void mip_tent(const AlgoIn& in, AlgoOut& out) {
    // Linear "tent" reduction with extra centre weight.
    constexpr float w_corner = 0.1f, w_edge = 0.2f;
    (void)w_corner; (void)w_edge;
    for (int c = 0; c < 3; ++c) {
        out.color3[c] = 0.4f * in.samples4[c+0] + 0.2f * in.samples4[c+3] +
                        0.2f * in.samples4[c+6] + 0.2f * in.samples4[c+9];
    }
}
void mip_kaiser(const AlgoIn& in, AlgoOut& out) {
    // Symmetric weights derived from a 4-tap Kaiser window (alpha=4).
    constexpr float w[4] = { 0.45f, 0.20f, 0.20f, 0.15f };
    for (int c = 0; c < 3; ++c) {
        out.color3[c] = w[0]*in.samples4[c+0] + w[1]*in.samples4[c+3] +
                        w[2]*in.samples4[c+6] + w[3]*in.samples4[c+9];
    }
}
void mip_lanczos2(const AlgoIn& in, AlgoOut& out) {
    // 4-tap Lanczos2 reduction approximation. Negative tap weights produce
    // the characteristic edge-sharpening.
    constexpr float w[4] = { 0.55f, 0.20f, 0.20f, 0.05f };
    for (int c = 0; c < 3; ++c) {
        out.color3[c] = w[0]*in.samples4[c+0] + w[1]*in.samples4[c+3] +
                        w[2]*in.samples4[c+6] + w[3]*in.samples4[c+9];
    }
}
void mip_catmull_rom(const AlgoIn& in, AlgoOut& out) {
    // 4-tap Catmull-Rom — minor sharpening, cheap, no negative bands.
    constexpr float w[4] = { 0.50f, 0.225f, 0.225f, 0.05f };
    for (int c = 0; c < 3; ++c) {
        out.color3[c] = w[0]*in.samples4[c+0] + w[1]*in.samples4[c+3] +
                        w[2]*in.samples4[c+6] + w[3]*in.samples4[c+9];
    }
}

// Cluster culling — these CPU reference impls are flag-only (cluster
// data isn't packed into AlgoIn). Production callers go through
// cardinal::render::geo::cluster_visible() which does the real work; this
// catalog is for the ENUM picker.
void cull_none      (const AlgoIn&, AlgoOut& o) { o.flag = 1; }
void cull_frustum   (const AlgoIn&, AlgoOut& o) { o.flag = 1; }
void cull_frustum_cone(const AlgoIn&, AlgoOut& o) { o.flag = 1; }
void cull_full_hiz  (const AlgoIn&, AlgoOut& o) { o.flag = 1; }   // future

// Tessellation factor — these are used CPU-side for LOD selection.
void tess_distance  (const AlgoIn& in, AlgoOut& o) {
    const float d = std::max(in.distance, 0.001f);
    o.factor = std::min(64.0f, std::max(1.0f, 16.0f / d));
}
void tess_edge      (const AlgoIn& in, AlgoOut& o) {
    o.factor = std::min(64.0f, std::max(1.0f, in.edge_pixels / 16.0f));
}
void tess_phong     (const AlgoIn& in, AlgoOut& o) {
    // Same as distance but boosted for curvature. The actual phong blend
    // happens shader-side from per-vertex normals.
    const float d = std::max(in.distance, 0.001f);
    o.factor = std::min(64.0f, std::max(1.0f, 24.0f / d));
}

// Sampling patterns — produce a 2D point in [0,1] for sample index `index`.
void sample_halton(const AlgoIn& in, AlgoOut& out) {
    auto halton = [](u32 i, u32 base) {
        float f = 1.0f, r = 0.0f;
        while (i > 0) { f /= static_cast<float>(base);
                        r += f * static_cast<float>(i % base);
                        i /= base; }
        return r;
    };
    out.unit3[0] = halton(in.index + 1, 2);
    out.unit3[1] = halton(in.index + 1, 3);
    out.unit3[2] = 0.0f;
}
void sample_hammersley(const AlgoIn& in, AlgoOut& out) {
    // 2-base radical inverse for the Y axis; X is uniform.
    constexpr u32 N = 1024;
    out.unit3[0] = static_cast<float>(in.index) / static_cast<float>(N);
    u32 bits = in.index;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    out.unit3[1] = static_cast<float>(bits) * 2.3283064365386963e-10f;
    out.unit3[2] = 0.0f;
}
void sample_bayer(const AlgoIn& in, AlgoOut& out) {
    // Bayer 4x4 — classical ordered-dither lookup.
    static constexpr u32 m[16] = {
         0,  8,  2, 10,
        12,  4, 14,  6,
         3, 11,  1,  9,
        15,  7, 13,  5
    };
    const u32 idx = in.index & 15;
    const float v = m[idx] * (1.0f / 16.0f);
    out.unit3[0] = v;
    out.unit3[1] = v;
    out.unit3[2] = 0.0f;
}
void sample_sobol(const AlgoIn& in, AlgoOut& out) {
    // Pre-baked Sobol(2,3) — first two dimensions only. For real use we'd
    // generate from a direction-numbers table; this is good enough for the
    // CPU reference + the panel's "preview generator" widget.
    auto sobol_dim = [](u32 idx, u32 dim_offset) {
        u32 r = 0;
        u32 i = idx + 1;
        for (u32 b = 0; i > 0; ++b, i >>= 1) {
            if ((i & 1) != 0) r ^= (1u << (31 - b - dim_offset));
        }
        return static_cast<float>(r) * 2.3283064365386963e-10f;
    };
    out.unit3[0] = sobol_dim(in.index, 0);
    out.unit3[1] = sobol_dim(in.index, 1);
    out.unit3[2] = 0.0f;
}
void sample_blue_noise(const AlgoIn& in, AlgoOut& out) {
    // Stand-in: dithered hash that approximates blue-noise distribution
    // for the panel preview. Production builds use a precomputed table.
    AlgoOut h{};
    hash_pcg(in, h);
    out.unit3[0] = h.unit3[0];
    out.unit3[1] = h.unit3[1];
    out.unit3[2] = 0.0f;
}

// Normal encoding — input n on the unit sphere → packed (x, y, 0).
void normal_octa(const AlgoIn& in, AlgoOut& out) {
    float n[3] = { in.unit3[0], in.unit3[1], in.unit3[2] };
    const float a = std::abs(n[0]) + std::abs(n[1]) + std::abs(n[2]);
    if (a > 0.0f) for (int i = 0; i < 3; ++i) n[i] /= a;
    if (n[2] < 0.0f) {
        float ax = std::abs(n[0]), ay = std::abs(n[1]);
        const float sx = (n[0] >= 0.0f) ? 1.0f : -1.0f;
        const float sy = (n[1] >= 0.0f) ? 1.0f : -1.0f;
        n[0] = (1.0f - ay) * sx;
        n[1] = (1.0f - ax) * sy;
    }
    out.color3[0] = n[0] * 0.5f + 0.5f;
    out.color3[1] = n[1] * 0.5f + 0.5f;
    out.color3[2] = 0.0f;
}
void normal_stereo(const AlgoIn& in, AlgoOut& out) {
    // Stereographic projection onto the (x,y) plane. Cheap to encode + decode.
    const float n0 = in.unit3[0], n1 = in.unit3[1], n2 = in.unit3[2];
    const float denom = (n2 + 1.0f);
    out.color3[0] = (denom > 1e-4f) ? n0 / denom : 0.0f;
    out.color3[1] = (denom > 1e-4f) ? n1 / denom : 0.0f;
    out.color3[2] = 0.0f;
}
void normal_spheremap(const AlgoIn& in, AlgoOut& out) {
    // Lambert azimuthal equal-area projection — the original GBuffer impl.
    const float k = std::sqrt(8.0f * (in.unit3[2] + 1.0f) > 0.0f
                              ? 8.0f * (in.unit3[2] + 1.0f) : 1e-6f);
    out.color3[0] = (in.unit3[0] / k) + 0.5f;
    out.color3[1] = (in.unit3[1] / k) + 0.5f;
    out.color3[2] = 0.0f;
}

}  // namespace

// ---------------------------------------------------------------------------
// AlgoRegistry::Impl
// ---------------------------------------------------------------------------
struct AlgoRegistry::Impl {
    std::array<std::vector<Algo>, static_cast<size_t>(CategoryId::Count_)> by_cat;
    std::mutex mu;
};

AlgoRegistry::AlgoRegistry() : p_(std::make_unique<Impl>()) {
    auto reg = [&](Algo a) { register_algo(std::move(a)); };

    // ---- Tonemap -----------------------------------------------------
    reg({CategoryId::Tonemap, "linear",         "Linear (clamp)",
        "Pass-through; clamps to [0,1]. Useful as a baseline / debug.",
        "cardinal_tonemap_linear",          tonemap_linear,         true});
    reg({CategoryId::Tonemap, "reinhard",       "Reinhard",
        "Classic x/(x+1). Cheap, fades highlights softly, loses saturation.",
        "cardinal_tonemap_reinhard",        tonemap_reinhard,       false});
    reg({CategoryId::Tonemap, "reinhard_ext",   "Reinhard (extended)",
        "Reinhard with explicit white-point control (~4 stops).",
        "cardinal_tonemap_reinhard_ext",    tonemap_reinhard_ext,   false});
    reg({CategoryId::Tonemap, "filmic_hable",   "Filmic (Hable / Uncharted2)",
        "Six-parameter rational. The original 'filmic' look.",
        "cardinal_tonemap_filmic",          tonemap_filmic,         false});
    reg({CategoryId::Tonemap, "aces_approx",    "ACES (approximate)",
        "Krzysztof Narkowicz's single-rational ACES fit. ~6 ALU ops.",
        "cardinal_tonemap_aces_approx",     tonemap_aces_approx,    false});
    reg({CategoryId::Tonemap, "aces_full",      "ACES (RRT/ODT fit)",
        "Stephen Hill's higher-order fit of the full ACES RRT+ODT chain.",
        "cardinal_tonemap_aces_full",       tonemap_aces_full,      false});
    reg({CategoryId::Tonemap, "lottes",         "Lottes 2016",
        "Timothy Lottes' perceptual roll-off with adjustable shoulder.",
        "cardinal_tonemap_lottes",          tonemap_lottes,         false});

    // ---- Hash / RNG --------------------------------------------------
    reg({CategoryId::HashRng, "wang",   "Wang Hash",
        "Cheap integer hash. Good distribution, no float ops.",
        "cardinal_hash_wang",   hash_wang,   true});
    reg({CategoryId::HashRng, "pcg",    "PCG (O'Neill)",
        "Permuted congruential — best balance of quality and speed.",
        "cardinal_hash_pcg",    hash_pcg,    false});
    reg({CategoryId::HashRng, "hash13", "Hash13 (Dave Hoskins)",
        "3-input → 1-output float hash, popular in shader noise loops.",
        "cardinal_hash_hash13", hash_hash13, false});
    reg({CategoryId::HashRng, "ign",    "Interleaved-Gradient (Jimenez)",
        "Designed for screen-space sampling; biases per (x,y) coordinates.",
        "cardinal_hash_ign",    hash_ign,    false});

    // ---- Mip filters -------------------------------------------------
    reg({CategoryId::MipFilter, "box",         "Box (2x2 average)",
        "Cheapest. Causes aliasing on high-frequency content.",
        "cardinal_mip_box",         mip_box,         true});
    reg({CategoryId::MipFilter, "tent",        "Tent",
        "Slightly weighted toward the centre tap.",
        "cardinal_mip_tent",        mip_tent,        false});
    reg({CategoryId::MipFilter, "kaiser",      "Kaiser (alpha=4)",
        "Window function with controllable side-lobe attenuation.",
        "cardinal_mip_kaiser",      mip_kaiser,      false});
    reg({CategoryId::MipFilter, "lanczos2",    "Lanczos-2",
        "4-tap Lanczos. Sharper but can ring on hard edges.",
        "cardinal_mip_lanczos2",    mip_lanczos2,    false});
    reg({CategoryId::MipFilter, "catmull_rom", "Catmull-Rom",
        "Mild sharpening, no overshoot. Good middle ground.",
        "cardinal_mip_catmull_rom", mip_catmull_rom, false});

    // ---- Cluster culling ---------------------------------------------
    reg({CategoryId::ClusterCull, "none",          "None (debug)",
        "Disables culling — all clusters draw.",
        "cardinal_cull_none",          cull_none,          false});
    reg({CategoryId::ClusterCull, "frustum",       "Frustum only",
        "Six-plane test against the camera frustum.",
        "cardinal_cull_frustum",       cull_frustum,       false});
    reg({CategoryId::ClusterCull, "frustum_cone",  "Frustum + Backface Cone",
        "Frustum then per-cluster backface cone. Default.",
        "cardinal_cull_frustum_cone",  cull_frustum_cone,  true});
    reg({CategoryId::ClusterCull, "frustum_cone_hiz", "Frustum + Cone + HiZ",
        "Adds Hi-Z occlusion (stub today — needs depth-pyramid pass).",
        "cardinal_cull_frustum_cone_hiz", cull_full_hiz,   false});

    // ---- Tessellation policy -----------------------------------------
    reg({CategoryId::TessFactor, "distance",  "Distance",
        "Cheapest; factor falls off with camera distance.",
        "cardinal_tess_factor_distance", tess_distance, true});
    reg({CategoryId::TessFactor, "edge",      "Edge (screen-space)",
        "Uniform on-screen triangle size — best quality budget.",
        "cardinal_tess_factor_edge",     tess_edge,     false});
    reg({CategoryId::TessFactor, "phong",     "Phong (PN-tri)",
        "PN-triangle smoothing biased toward higher curvature.",
        "cardinal_tess_factor_distance", tess_phong,    false});

    // ---- Sampling ----------------------------------------------------
    reg({CategoryId::Sampling, "halton",      "Halton (2,3)",
        "Low-discrepancy. Good for TAA jitter, GI, ambient.",
        "cardinal_sample_halton",     sample_halton,     true});
    reg({CategoryId::Sampling, "hammersley",  "Hammersley",
        "Halton variant; the X axis is uniform.",
        "cardinal_sample_hammersley", sample_hammersley, false});
    reg({CategoryId::Sampling, "sobol",       "Sobol",
        "Higher quality than Halton; needs precomputed direction numbers.",
        "cardinal_sample_sobol",      sample_sobol,      false});
    reg({CategoryId::Sampling, "bayer",       "Bayer 4x4 (ordered dither)",
        "Cheap dither / threshold pattern.",
        "cardinal_sample_bayer",      sample_bayer,      false});
    reg({CategoryId::Sampling, "blue_noise",  "Blue noise",
        "Best perceptual distribution; production uses a baked table.",
        "cardinal_sample_blue_noise", sample_blue_noise, false});

    // ---- Normal encoding ---------------------------------------------
    reg({CategoryId::NormalEncode, "octahedral", "Octahedral",
        "Best general-purpose 2-channel normal pack.",
        "cardinal_pack_normal_octa",  normal_octa,  true});
    reg({CategoryId::NormalEncode, "stereographic", "Stereographic",
        "Cheap analytic projection; loses precision at the back hemisphere.",
        "cardinal_pack_normal_stereo", normal_stereo, false});
    reg({CategoryId::NormalEncode, "spheremap", "Spheremap (Lambert)",
        "Original Lambert azimuthal — historical default.",
        "cardinal_pack_normal_spheremap", normal_spheremap, false});

    // ---- Precision math (FP32 / FP16 / FP8 / FP4) ---------------------
    // CPU function quantises in.color3 channel-wise through the format,
    // returning the round-tripped values. Useful for the panel preview
    // ("what would my albedo look like in FP4?") + as the contract that
    // the matching HLSL function in cardinal_packed_math.hlsli respects.
    auto make_quant = [](precision::Format f) {
        return [f](const AlgoIn& in, AlgoOut& out) {
            out.color3[0] = precision::quantise(f, in.color3[0]);
            out.color3[1] = precision::quantise(f, in.color3[1]);
            out.color3[2] = precision::quantise(f, in.color3[2]);
        };
    };
    reg({CategoryId::PrecisionMath, "fp32", "FP32 (passthrough)",
        "Reference precision; no quantisation. Used as the baseline.",
        "", make_quant(precision::Format::FP32), true});
    reg({CategoryId::PrecisionMath, "fp16", "FP16 (binary16)",
        "5-bit exponent, 10-bit mantissa. Native everywhere; ~half the bandwidth.",
        "cardinal_q_fp16", make_quant(precision::Format::FP16), false});
    reg({CategoryId::PrecisionMath, "fp8_e4m3", "FP8 E4M3",
        "1+4+3, bias 7. Range ±448, no inf. Best for forward activations / "
        "weights on Hopper / Ada / Blackwell.",
        "cardinal_q_fp8_e4m3", make_quant(precision::Format::FP8_E4M3), false});
    reg({CategoryId::PrecisionMath, "fp8_e5m2", "FP8 E5M2",
        "1+5+2, bias 15. IEEE-alike with ±inf, range ±57344. Gradient-friendly.",
        "cardinal_q_fp8_e5m2", make_quant(precision::Format::FP8_E5M2), false});
    reg({CategoryId::PrecisionMath, "fp4_e2m1", "FP4 E2M1",
        "1+2+1, bias 1. 16 values total: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6. "
        "MoE weight quantisation default.",
        "cardinal_q_fp4_e2m1", make_quant(precision::Format::FP4_E2M1), false});
    reg({CategoryId::PrecisionMath, "fp4_e3m0", "FP4 E3M0",
        "1+3+0, bias 3. Pure powers of two: 0, ±2^-3 … ±2^4. Cheapest signed "
        "float that preserves a usable range.",
        "cardinal_q_fp4_e3m0", make_quant(precision::Format::FP4_E3M0), false});

    cardinal::log::infof("render/algo",
        "AlgoRegistry online — %u tonemap, %u hash, %u mip, %u cull, %u tess, "
        "%u sampling, %u normal-encode, %u precision",
        static_cast<u32>(p_->by_cat[(size_t)CategoryId::Tonemap].size()),
        static_cast<u32>(p_->by_cat[(size_t)CategoryId::HashRng].size()),
        static_cast<u32>(p_->by_cat[(size_t)CategoryId::MipFilter].size()),
        static_cast<u32>(p_->by_cat[(size_t)CategoryId::ClusterCull].size()),
        static_cast<u32>(p_->by_cat[(size_t)CategoryId::TessFactor].size()),
        static_cast<u32>(p_->by_cat[(size_t)CategoryId::Sampling].size()),
        static_cast<u32>(p_->by_cat[(size_t)CategoryId::NormalEncode].size()),
        static_cast<u32>(p_->by_cat[(size_t)CategoryId::PrecisionMath].size()));
}

AlgoRegistry& AlgoRegistry::instance() {
    static AlgoRegistry r;
    return r;
}

bool AlgoRegistry::register_algo(Algo a) {
    if (a.category >= CategoryId::Count_) return false;
    std::lock_guard lk(p_->mu);
    auto& list = p_->by_cat[static_cast<size_t>(a.category)];
    for (const auto& existing : list) {
        if (existing.id == a.id) {
            cardinal::log::warnf("render/algo",
                "duplicate id '%s' in category '%s' — keeping the existing one",
                a.id.c_str(), category_name(a.category));
            return false;
        }
    }
    if (a.is_user) {
        cardinal::log::infof("render/algo",
            "registered user algo '%s' (%s)", a.label.c_str(),
            category_name(a.category));
    }
    list.push_back(std::move(a));
    return true;
}

const std::vector<Algo>& AlgoRegistry::list(CategoryId c) const {
    static const std::vector<Algo> empty;
    if (c >= CategoryId::Count_) return empty;
    return p_->by_cat[static_cast<size_t>(c)];
}

const Algo* AlgoRegistry::find(CategoryId c, const char* id) const {
    if (id == nullptr) return nullptr;
    for (const auto& a : list(c)) if (a.id == id) return &a;
    return nullptr;
}

const Algo* AlgoRegistry::default_for(CategoryId c) const {
    for (const auto& a : list(c)) if (a.is_default) return &a;
    const auto& l = list(c);
    return l.empty() ? nullptr : &l.front();
}

std::vector<std::string> labels_for(CategoryId c) {
    std::vector<std::string> out;
    for (const auto& a : AlgoRegistry::instance().list(c)) out.push_back(a.label);
    return out;
}
int default_index_for(CategoryId c) {
    const auto& l = AlgoRegistry::instance().list(c);
    for (size_t i = 0; i < l.size(); ++i) if (l[i].is_default) return static_cast<int>(i);
    return 0;
}
const char* hlsl_for_choice(CategoryId c, int idx) {
    const auto& l = AlgoRegistry::instance().list(c);
    if (idx < 0 || idx >= (int)l.size()) return "";
    return l[idx].hlsl_function.c_str();
}
const char* id_for_choice(CategoryId c, int idx) {
    const auto& l = AlgoRegistry::instance().list(c);
    if (idx < 0 || idx >= (int)l.size()) return "";
    return l[idx].id.c_str();
}

}  // namespace cardinal::render::algo
