// =============================================================================
// Cardinal — terrain generator (heightmap noise + chunked grid + profiles).
// =============================================================================
#include <cardinal/scene/terrain.hpp>

#include <cardinal/core/log.hpp>
#include <cardinal/scene/scene.hpp>

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/cctype.hpp>
#include <cardinal/core/cmath.hpp>
#include <cardinal/core/cstdio.hpp>
#include <cardinal/core/cstdlib.hpp>
#include <cardinal/core/cstring.hpp>
#include <cardinal/core/fstream.hpp>
#include <cardinal/core/sstream.hpp>
#include <cardinal/core/utility.hpp>

namespace cardinal::scene {

// ---------------------------------------------------------------------------
// Hash + value-noise primitives. Cheap, deterministic, and SIMD-friendly
// when we get there. Range matches Perlin's: ~[-1, +1].
// ---------------------------------------------------------------------------
namespace {

inline u32 hash_u32(u32 x, u32 seed) noexcept {
    x ^= seed;
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

inline float hash_to_unit(int ix, int iz, u32 seed) noexcept {
    const u32 h = hash_u32(static_cast<u32>(ix) * 0x27d4eb2du
                         ^ static_cast<u32>(iz) * 0x165667b1u, seed);
    // Normalise to [-1, +1].
    return (static_cast<float>(h & 0xffffff) / 8388607.5f) - 1.0f;
}

inline float fade_smoothstep(float t) noexcept {
    // 6t^5 − 15t^4 + 10t^3 — Perlin's quintic fade. Slightly nicer than
    // the cubic smoothstep at chunk seams.
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

inline float value_noise_2d(float x, float z, u32 seed) noexcept {
    const float fx = cardinal::floor(x), fz = cardinal::floor(z);
    const int   ix = static_cast<int>(fx), iz = static_cast<int>(fz);
    const float tx = x - fx, tz = z - fz;
    const float ux = fade_smoothstep(tx), uz = fade_smoothstep(tz);

    const float v00 = hash_to_unit(ix,     iz,     seed);
    const float v10 = hash_to_unit(ix + 1, iz,     seed);
    const float v01 = hash_to_unit(ix,     iz + 1, seed);
    const float v11 = hash_to_unit(ix + 1, iz + 1, seed);

    const float a = v00 + (v10 - v00) * ux;
    const float b = v01 + (v11 - v01) * ux;
    return a + (b - a) * uz;
}

inline float fbm(float x, float z, int octaves, float lacunarity, float gain,
                 u32 seed) noexcept
{
    float amp  = 1.0f;
    float freq = 1.0f;
    float sum  = 0.0f;
    float norm = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        sum  += amp * value_noise_2d(x * freq, z * freq, seed + static_cast<u32>(i) * 131u);
        norm += amp;
        amp  *= gain;
        freq *= lacunarity;
    }
    return (norm > 0.0f) ? (sum / norm) : 0.0f;
}

// fbm with a per-octave shape remap (Smooth / Ridge / Billow). Used by
// the profile path's NoiseLayer evaluator.
inline float fbm_shaped(float x, float z, int octaves, float lacunarity,
                        float gain, u32 seed, NoiseShape shape) noexcept
{
    float amp  = 1.0f;
    float freq = 1.0f;
    float sum  = 0.0f;
    float norm = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        float v = value_noise_2d(x * freq, z * freq,
                                 seed + static_cast<u32>(i) * 131u);
        switch (shape) {
            case NoiseShape::Smooth:                     break;
            case NoiseShape::Ridge:  v = 1.0f - cardinal::fabs(v); break;
            case NoiseShape::Billow: v = cardinal::fabs(v);        break;
        }
        sum  += amp * v;
        norm += amp;
        amp  *= gain;
        freq *= lacunarity;
    }
    return (norm > 0.0f) ? (sum / norm) : 0.0f;
}

inline Vec3 lerp_vec3(const Vec3& a, const Vec3& b, float t) noexcept {
    return Vec3{ a.x + (b.x - a.x) * t,
                 a.y + (b.y - a.y) * t,
                 a.z + (b.z - a.z) * t };
}

}  // namespace

// ---------------------------------------------------------------------------
// Legacy single-noise path
// ---------------------------------------------------------------------------
float terrain_height(const TerrainParams& p, float wx, float wz) noexcept {
    const float n = fbm(wx * p.frequency, wz * p.frequency,
                        cardinal::clamp(p.octaves, 1, 8),
                        p.lacunarity, p.gain, p.seed);
    return n * p.height_scale;
}

// ---------------------------------------------------------------------------
// Profile-system enums + library
// ---------------------------------------------------------------------------
const char* noise_op_name(NoiseOp o) noexcept {
    switch (o) {
        case NoiseOp::Add:      return "Add";
        case NoiseOp::Multiply: return "Multiply";
        case NoiseOp::Max:      return "Max";
        case NoiseOp::Min:      return "Min";
    }
    return "?";
}
const char* noise_shape_name(NoiseShape s) noexcept {
    switch (s) {
        case NoiseShape::Smooth: return "Smooth";
        case NoiseShape::Ridge:  return "Ridge";
        case NoiseShape::Billow: return "Billow";
    }
    return "?";
}

// Per-layer evaluator. Domain-warps coords first, then runs shaped fbm.
static float layer_value(const NoiseLayer& L, float wx, float wz) noexcept {
    float sx = wx + L.offset_x;
    float sz = wz + L.offset_z;
    if (L.warp_amount > 0.0f && L.warp_frequency > 0.0f) {
        const float ax = value_noise_2d(sx * L.warp_frequency,
                                        sz * L.warp_frequency,
                                        L.seed + 7919u);
        const float az = value_noise_2d((sx + 100.0f) * L.warp_frequency,
                                        (sz - 100.0f) * L.warp_frequency,
                                        L.seed + 17389u);
        sx += ax * L.warp_amount;
        sz += az * L.warp_amount;
    }
    return fbm_shaped(sx * L.frequency, sz * L.frequency,
                      cardinal::clamp(L.octaves, 1, 8),
                      L.lacunarity, L.gain, L.seed, L.shape);
}

float terrain_height_profile(const TerrainProfile& P, float wx, float wz) noexcept {
    float h = P.base_height;
    for (const auto& L : P.layers) {
        if (!L.enabled) continue;
        const float v_raw = layer_value(L, wx, wz);
        const float v     = v_raw * L.weight;
        switch (L.op) {
            case NoiseOp::Add:      h += v;                              break;
            case NoiseOp::Multiply:
                // Treat layer as a [0..1] mask (remapped from [-1..+1]),
                // then lerp `h * mask` against `h` by L.weight so weight=0
                // is no-op and weight=1 fully applies the mask.
                {
                    const float mask = 0.5f + 0.5f * v_raw;
                    h = h * (1.0f - L.weight) + (h * mask) * L.weight;
                }
                break;
            case NoiseOp::Max:      h = cardinal::max(h, v);                  break;
            case NoiseOp::Min:      h = cardinal::min(h, v);                  break;
        }
    }
    return h * P.height_scale;
}

Vec3 terrain_color_profile(const TerrainProfile& P, float height,
                           const Vec3& normal) noexcept
{
    const float span = cardinal::max(1e-3f, P.tint_height_high - P.tint_height_low);
    float t = (height - P.tint_height_low) / span;
    t = cardinal::clamp(t, 0.0f, 1.0f);
    Vec3 base = lerp_vec3(P.tint_low, P.tint_high, t);

    const float slope = 1.0f - cardinal::clamp(normal.y, 0.0f, 1.0f);
    const float thr   = cardinal::clamp(P.cliff_blend_threshold, 0.0f, 0.9999f);
    const float cliff = cardinal::clamp((slope - thr) / (1.0f - thr), 0.0f, 1.0f);
    return lerp_vec3(base, P.tint_cliff, cliff);
}

struct TerrainProfileLibrary::Impl {
    cardinal::vector<TerrainProfile> entries;
};

TerrainProfileLibrary::TerrainProfileLibrary() : p_(cardinal::make_unique<Impl>()) {}

TerrainProfileLibrary& TerrainProfileLibrary::instance() {
    static TerrainProfileLibrary g;
    return g;
}

void TerrainProfileLibrary::set(TerrainProfile prof) {
    for (auto& e : p_->entries) {
        if (e.name == prof.name) { e = cardinal::move(prof); return; }
    }
    p_->entries.push_back(cardinal::move(prof));
}

bool TerrainProfileLibrary::remove(const char* name) {
    if (name == nullptr) return false;
    for (auto it = p_->entries.begin(); it != p_->entries.end(); ++it) {
        if (it->name == name) { p_->entries.erase(it); return true; }
    }
    return false;
}

const TerrainProfile* TerrainProfileLibrary::find(const char* name) const noexcept {
    if (name == nullptr) return nullptr;
    for (const auto& e : p_->entries) if (e.name == name) return &e;
    return nullptr;
}

const cardinal::vector<TerrainProfile>& TerrainProfileLibrary::all() const noexcept {
    return p_->entries;
}

cardinal::vector<cardinal::string> TerrainProfileLibrary::names() const {
    cardinal::vector<cardinal::string> out;
    out.reserve(p_->entries.size());
    for (const auto& e : p_->entries) out.push_back(e.name);
    return out;
}

// ---------------------------------------------------------------------------
// Persistence — tiny human-friendly text format.
// Lines are tokenised on whitespace; `profile "<name>"` opens a profile,
// `layer "<id>"` opens a layer (within the current profile), and the next
// `profile`, `layer`, or EOF closes the previous block. Everything else is
// a `<key> <values...>` setting that mutates whichever block is open.
// Quoted strings may contain spaces; comments start with '#'.
// ---------------------------------------------------------------------------
namespace {

inline cardinal::string trim(cardinal::string s) {
    while (!s.empty() && cardinal::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && cardinal::isspace(static_cast<unsigned char>(s.back())))  s.pop_back();
    return s;
}

// Split a line into tokens; honour double-quoted strings as one token.
inline cardinal::vector<cardinal::string> tokenise(const cardinal::string& line) {
    cardinal::vector<cardinal::string> out;
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && cardinal::isspace(static_cast<unsigned char>(line[i]))) ++i;
        if (i == line.size()) break;
        if (line[i] == '#') break;       // comment
        if (line[i] == '"') {
            size_t j = ++i;
            cardinal::string acc;
            while (j < line.size() && line[j] != '"') acc += line[j++];
            out.push_back(acc);
            i = (j < line.size()) ? j + 1 : j;
        } else {
            size_t j = i;
            while (j < line.size() && !cardinal::isspace(static_cast<unsigned char>(line[j]))) ++j;
            out.emplace_back(line.substr(i, j - i));
            i = j;
        }
    }
    return out;
}

inline NoiseOp parse_op(const cardinal::string& s) {
    if (s == "add"      || s == "Add")      return NoiseOp::Add;
    if (s == "multiply" || s == "Multiply") return NoiseOp::Multiply;
    if (s == "max"      || s == "Max")      return NoiseOp::Max;
    if (s == "min"      || s == "Min")      return NoiseOp::Min;
    return NoiseOp::Add;
}
inline NoiseShape parse_shape(const cardinal::string& s) {
    if (s == "smooth" || s == "Smooth") return NoiseShape::Smooth;
    if (s == "ridge"  || s == "Ridge")  return NoiseShape::Ridge;
    if (s == "billow" || s == "Billow") return NoiseShape::Billow;
    return NoiseShape::Smooth;
}

// cardinal::stof / cardinal::stoul ARE std::stof / std::stoul and THROW
// (std::invalid_argument on a non-numeric token, std::out_of_range on
// overflow). load_from_file parses a hand-editable text format the user
// can typo and live-reload, so a malformed value must degrade to the
// field's default and parsing must continue — exactly like the parser
// already ignores unknown keys — never propagate an exception out of a
// "load a file" call (which would abort the whole library load, or crash
// the editor on a typo'd live-reload).
inline float parse_f(const cardinal::string& s, float def) noexcept {
    try { return cardinal::stof(s); } catch (...) { return def; }
}
inline u32 parse_u32(const cardinal::string& s, u32 def) noexcept {
    try { return static_cast<u32>(cardinal::stoul(s)); }
    catch (...) { return def; }
}

}  // namespace

bool TerrainProfileLibrary::save_to_file(const char* path) const {
    if (path == nullptr || *path == '\0') return false;
    cardinal::ofstream f(path, cardinal::ios::trunc);
    if (!f) {
        cardinal::log::warnf("scene/terrain", "save_to_file: cannot open '%s'", path);
        return false;
    }
    f << "# Cardinal — terrain profile library\n";
    f << "# " << p_->entries.size() << " profile(s)\n\n";
    for (const auto& P : p_->entries) {
        f << "profile \"" << P.name << "\"\n";
        if (!P.description.empty())
            f << "  description \"" << P.description << "\"\n";
        f << "  base_height "       << P.base_height       << "\n";
        f << "  height_scale "      << P.height_scale      << "\n";
        f << "  tint_low "          << P.tint_low.x  << " " << P.tint_low.y  << " " << P.tint_low.z  << "\n";
        f << "  tint_high "         << P.tint_high.x << " " << P.tint_high.y << " " << P.tint_high.z << "\n";
        f << "  tint_cliff "        << P.tint_cliff.x<< " " << P.tint_cliff.y<< " " << P.tint_cliff.z<< "\n";
        f << "  tint_height_low "   << P.tint_height_low   << "\n";
        f << "  tint_height_high "  << P.tint_height_high  << "\n";
        f << "  cliff_blend "       << P.cliff_blend_threshold << "\n";
        for (const auto& L : P.layers) {
            f << "  layer \"" << L.id << "\"\n";
            f << "    enabled "        << (L.enabled ? "1" : "0") << "\n";
            f << "    op "             << noise_op_name(L.op)     << "\n";
            f << "    shape "          << noise_shape_name(L.shape)<< "\n";
            f << "    weight "         << L.weight                << "\n";
            f << "    frequency "      << L.frequency             << "\n";
            f << "    octaves "        << L.octaves               << "\n";
            f << "    lacunarity "     << L.lacunarity            << "\n";
            f << "    gain "           << L.gain                  << "\n";
            f << "    seed "           << L.seed                  << "\n";
            f << "    offset_x "       << L.offset_x              << "\n";
            f << "    offset_z "       << L.offset_z              << "\n";
            f << "    warp_amount "    << L.warp_amount           << "\n";
            f << "    warp_frequency " << L.warp_frequency        << "\n";
        }
        f << "\n";
    }
    return true;
}

bool TerrainProfileLibrary::load_from_file(const char* path) {
    if (path == nullptr || *path == '\0') return false;
    cardinal::ifstream f(path);
    if (!f) {
        cardinal::log::warnf("scene/terrain", "load_from_file: cannot open '%s'", path);
        return false;
    }
    cardinal::string line;
    TerrainProfile cur;
    NoiseLayer     curlayer;
    bool           in_profile = false;
    bool           in_layer   = false;
    int            loaded     = 0;

    auto commit_layer = [&]() {
        if (in_layer) { cur.layers.push_back(curlayer); in_layer = false; }
    };
    auto commit_profile = [&]() {
        if (in_profile) {
            commit_layer();
            this->set(cur);
            ++loaded;
            cur = {};
            in_profile = false;
        }
    };

    while (cardinal::getline(f, line)) {
        auto tok = tokenise(line);
        if (tok.empty()) continue;
        const cardinal::string& k = tok[0];

        if (k == "profile") {
            commit_profile();
            in_profile = true;
            cur = {};
            cur.name = (tok.size() > 1) ? tok[1] : "Unnamed";
        } else if (!in_profile) {
            // ignore stray settings outside any profile
            continue;
        } else if (k == "layer") {
            commit_layer();
            in_layer = true;
            curlayer = {};
            curlayer.id = (tok.size() > 1) ? tok[1] : "layer";
        } else if (in_layer) {
            // Layer-scope settings.
            if      (k == "enabled"        && tok.size() > 1) curlayer.enabled        = cardinal::atoi(tok[1].c_str()) != 0;
            else if (k == "op"             && tok.size() > 1) curlayer.op             = parse_op(tok[1]);
            else if (k == "shape"          && tok.size() > 1) curlayer.shape          = parse_shape(tok[1]);
            else if (k == "weight"         && tok.size() > 1) curlayer.weight         = parse_f(tok[1], curlayer.weight);
            else if (k == "frequency"      && tok.size() > 1) curlayer.frequency      = parse_f(tok[1], curlayer.frequency);
            else if (k == "octaves"        && tok.size() > 1) curlayer.octaves        = cardinal::atoi(tok[1].c_str());
            else if (k == "lacunarity"     && tok.size() > 1) curlayer.lacunarity     = parse_f(tok[1], curlayer.lacunarity);
            else if (k == "gain"           && tok.size() > 1) curlayer.gain           = parse_f(tok[1], curlayer.gain);
            else if (k == "seed"           && tok.size() > 1) curlayer.seed           = parse_u32(tok[1], curlayer.seed);
            else if (k == "offset_x"       && tok.size() > 1) curlayer.offset_x       = parse_f(tok[1], curlayer.offset_x);
            else if (k == "offset_z"       && tok.size() > 1) curlayer.offset_z       = parse_f(tok[1], curlayer.offset_z);
            else if (k == "warp_amount"    && tok.size() > 1) curlayer.warp_amount    = parse_f(tok[1], curlayer.warp_amount);
            else if (k == "warp_frequency" && tok.size() > 1) curlayer.warp_frequency = parse_f(tok[1], curlayer.warp_frequency);
        } else {
            // Profile-scope settings.
            if      (k == "description"      && tok.size() > 1) cur.description           = tok[1];
            else if (k == "base_height"      && tok.size() > 1) cur.base_height           = parse_f(tok[1], cur.base_height);
            else if (k == "height_scale"     && tok.size() > 1) cur.height_scale          = parse_f(tok[1], cur.height_scale);
            else if (k == "tint_low"         && tok.size() > 3) cur.tint_low              = { parse_f(tok[1], cur.tint_low.x),  parse_f(tok[2], cur.tint_low.y),  parse_f(tok[3], cur.tint_low.z)  };
            else if (k == "tint_high"        && tok.size() > 3) cur.tint_high             = { parse_f(tok[1], cur.tint_high.x), parse_f(tok[2], cur.tint_high.y), parse_f(tok[3], cur.tint_high.z) };
            else if (k == "tint_cliff"       && tok.size() > 3) cur.tint_cliff            = { parse_f(tok[1], cur.tint_cliff.x),parse_f(tok[2], cur.tint_cliff.y),parse_f(tok[3], cur.tint_cliff.z)};
            else if (k == "tint_height_low"  && tok.size() > 1) cur.tint_height_low       = parse_f(tok[1], cur.tint_height_low);
            else if (k == "tint_height_high" && tok.size() > 1) cur.tint_height_high      = parse_f(tok[1], cur.tint_height_high);
            else if (k == "cliff_blend"      && tok.size() > 1) cur.cliff_blend_threshold = parse_f(tok[1], cur.cliff_blend_threshold);
        }
    }
    commit_profile();
    cardinal::log::infof("scene/terrain",
        "load_from_file: '%s' — %d profile(s) loaded (library now %zu)",
        path, loaded, p_->entries.size());
    return loaded > 0;
}

// ---------------------------------------------------------------------------
// Engine-shipped curated presets. Each is one chunk of "what does a
// believable terrain of type X actually look like" tuned by hand. Users
// edit these in the modal and save derivatives via `save_to_file`.
// ---------------------------------------------------------------------------
void register_default_terrain_profiles() {
    auto& lib = TerrainProfileLibrary::instance();

    auto add_if_missing = [&](TerrainProfile p) {
        if (lib.find(p.name.c_str()) == nullptr) lib.set(cardinal::move(p));
    };

    // --- Plains -------------------------------------------------------------
    {
        TerrainProfile P{};
        P.name = "Plains";
        P.description = "Mostly flat, gentle undulations. Good for testing scale.";
        P.height_scale = 3.0f;
        P.tint_low  = { 0.50f, 0.65f, 0.32f };
        P.tint_high = { 0.62f, 0.70f, 0.42f };
        P.tint_height_low  = -1.0f;
        P.tint_height_high =  3.0f;
        P.cliff_blend_threshold = 0.7f;
        NoiseLayer L{};
        L.id = "ground"; L.frequency = 0.025f; L.octaves = 3; L.weight = 1.0f;
        L.gain = 0.45f; L.seed = 9001u;
        P.layers.push_back(L);
        add_if_missing(cardinal::move(P));
    }

    // --- Rolling Hills ------------------------------------------------------
    {
        TerrainProfile P{};
        P.name = "Rolling Hills";
        P.description = "Two-octave countryside — large hills modulated by smaller bumps.";
        P.height_scale = 6.0f;
        P.tint_low  = { 0.34f, 0.55f, 0.25f };
        P.tint_high = { 0.55f, 0.62f, 0.40f };
        P.tint_height_low  = -1.0f;
        P.tint_height_high =  6.0f;
        P.cliff_blend_threshold = 0.55f;
        {
            NoiseLayer L{}; L.id = "hills";
            L.frequency = 0.020f; L.octaves = 4; L.weight = 1.0f;
            L.gain = 0.5f;  L.seed = 1101u;
            P.layers.push_back(L);
        }
        {
            NoiseLayer L{}; L.id = "bumps";
            L.frequency = 0.10f;  L.octaves = 3; L.weight = 0.25f;
            L.gain = 0.5f;  L.seed = 1102u;
            P.layers.push_back(L);
        }
        add_if_missing(cardinal::move(P));
    }

    // --- Alpine Mountains ---------------------------------------------------
    {
        TerrainProfile P{};
        P.name = "Alpine Mountains";
        P.description = "Continental shelf + ridge layer + boulder noise; snowy peaks.";
        P.height_scale = 28.0f;
        P.tint_low  = { 0.32f, 0.50f, 0.28f };
        P.tint_high = { 0.95f, 0.96f, 0.99f };
        P.tint_cliff= { 0.50f, 0.46f, 0.42f };
        P.tint_height_low  =  2.0f;
        P.tint_height_high = 22.0f;
        P.cliff_blend_threshold = 0.55f;
        {
            NoiseLayer L{}; L.id = "continental";
            L.frequency = 0.010f; L.octaves = 5; L.weight = 1.0f;
            L.lacunarity = 2.0f; L.gain = 0.55f; L.seed = 2101u;
            L.warp_amount = 30.0f; L.warp_frequency = 0.005f;
            P.layers.push_back(L);
        }
        {
            NoiseLayer L{}; L.id = "ridges";
            L.shape = NoiseShape::Ridge;
            L.frequency = 0.035f; L.octaves = 6; L.weight = 0.7f;
            L.lacunarity = 2.1f; L.gain = 0.5f;  L.seed = 2102u;
            P.layers.push_back(L);
        }
        {
            NoiseLayer L{}; L.id = "boulders";
            L.frequency = 0.20f;  L.octaves = 3; L.weight = 0.15f;
            L.gain = 0.5f;  L.seed = 2103u;
            P.layers.push_back(L);
        }
        add_if_missing(cardinal::move(P));
    }

    // --- Volcanic Wasteland -------------------------------------------------
    {
        TerrainProfile P{};
        P.name = "Volcanic Wasteland";
        P.description = "Harsh ridges + sharp spires; grey ash with darker rifts.";
        P.height_scale = 18.0f;
        P.tint_low  = { 0.18f, 0.16f, 0.16f };
        P.tint_high = { 0.55f, 0.50f, 0.45f };
        P.tint_cliff= { 0.20f, 0.18f, 0.18f };
        P.tint_height_low  = -2.0f;
        P.tint_height_high = 12.0f;
        P.cliff_blend_threshold = 0.45f;
        {
            NoiseLayer L{}; L.id = "ash";
            L.shape = NoiseShape::Ridge;
            L.frequency = 0.05f;  L.octaves = 6; L.weight = 1.0f;
            L.gain = 0.55f; L.seed = 3101u;
            P.layers.push_back(L);
        }
        {
            NoiseLayer L{}; L.id = "spires";
            L.shape = NoiseShape::Ridge;
            L.frequency = 0.18f;  L.octaves = 4; L.weight = 0.4f;
            L.gain = 0.6f;  L.seed = 3102u;
            P.layers.push_back(L);
        }
        add_if_missing(cardinal::move(P));
    }

    // --- Coastal Dunes ------------------------------------------------------
    {
        TerrainProfile P{};
        P.name = "Coastal Dunes";
        P.description = "Soft billowing dunes — pure sand tint with subtle ridges.";
        P.height_scale = 4.0f;
        P.tint_low  = { 0.85f, 0.78f, 0.55f };
        P.tint_high = { 0.95f, 0.88f, 0.62f };
        P.tint_cliff= { 0.75f, 0.66f, 0.45f };
        P.tint_height_low  = -1.0f;
        P.tint_height_high =  4.0f;
        P.cliff_blend_threshold = 0.65f;
        {
            NoiseLayer L{}; L.id = "dunes";
            L.shape = NoiseShape::Billow;
            L.frequency = 0.030f; L.octaves = 4; L.weight = 1.0f;
            L.gain = 0.55f; L.seed = 4101u;
            L.warp_amount = 8.0f; L.warp_frequency = 0.015f;
            P.layers.push_back(L);
        }
        {
            NoiseLayer L{}; L.id = "ripples";
            L.frequency = 0.20f;  L.octaves = 2; L.weight = 0.10f;
            L.gain = 0.5f;  L.seed = 4102u;
            P.layers.push_back(L);
        }
        add_if_missing(cardinal::move(P));
    }

    // --- Canyon -------------------------------------------------------------
    {
        TerrainProfile P{};
        P.name = "Canyon";
        P.description = "Plateau base with carved valleys (ridge-noise multiply mask).";
        P.height_scale = 22.0f;
        P.base_height  = 0.6f;
        P.tint_low  = { 0.55f, 0.32f, 0.22f };
        P.tint_high = { 0.78f, 0.55f, 0.36f };
        P.tint_cliff= { 0.42f, 0.26f, 0.18f };
        P.tint_height_low  =  0.0f;
        P.tint_height_high = 16.0f;
        P.cliff_blend_threshold = 0.5f;
        {
            NoiseLayer L{}; L.id = "plateau";
            L.frequency = 0.02f;  L.octaves = 3; L.weight = 1.0f;
            L.gain = 0.5f;  L.seed = 5101u;
            P.layers.push_back(L);
        }
        {
            NoiseLayer L{}; L.id = "river_carve";
            L.op = NoiseOp::Multiply;
            L.shape = NoiseShape::Ridge;
            L.frequency = 0.012f; L.octaves = 3; L.weight = 0.85f;
            L.gain = 0.55f; L.seed = 5102u;
            P.layers.push_back(L);
        }
        add_if_missing(cardinal::move(P));
    }

    cardinal::log::infof("scene/terrain",
        "Terrain profiles online — %zu in library",
        TerrainProfileLibrary::instance().all().size());
}

// ---------------------------------------------------------------------------
// Chunk mesh — a regular (resolution-1) x (resolution-1) grid of quads,
// each split into two triangles. Vertex colour reflects local slope so
// flat ground reads green and cliffs read grey-brown.
// ---------------------------------------------------------------------------
cardinal::shared_ptr<Mesh> generate_terrain_chunk(rhi::Device& dev,
                                             const TerrainParams& p,
                                             int chunk_x, int chunk_z)
{
    const u32 N = cardinal::max<u32>(2, p.resolution);
    const float step = p.chunk_size / static_cast<float>(N - 1);

    // World-space origin of THIS chunk's grid (its (0,0) vertex).
    // Each chunk is axis-aligned and abuts its neighbours exactly; the
    // shared edge samples the same heightmap, so seams stay invisible.
    const float ox = static_cast<float>(chunk_x) * p.chunk_size - p.chunk_size * 0.5f;
    const float oz = static_cast<float>(chunk_z) * p.chunk_size - p.chunk_size * 0.5f;

    // Pre-sample the height grid so each vertex computes its normal from
    // central differences — much faster than re-evaluating fbm 4× per vert.
    const u32 stride = N + 2;             // 1-cell halo on each side
    cardinal::vector<float> H(stride * stride);
    for (u32 j = 0; j < stride; ++j) {
        for (u32 i = 0; i < stride; ++i) {
            const float wx = ox + (static_cast<float>(i) - 1.0f) * step;
            const float wz = oz + (static_cast<float>(j) - 1.0f) * step;
            H[j * stride + i] = terrain_height(p, wx, wz);
        }
    }
    auto h_at = [&](u32 i, u32 j) -> float { return H[(j + 1) * stride + (i + 1)]; };

    // Build vertex grid first.
    struct V { Vec3 pos; Vec3 nrm; Vec3 col; };
    cardinal::vector<V> grid(N * N);
    for (u32 j = 0; j < N; ++j) {
        for (u32 i = 0; i < N; ++i) {
            const float x = ox + static_cast<float>(i) * step;
            const float z = oz + static_cast<float>(j) * step;
            const float y = h_at(i, j);

            // Central-difference normal. dY/dX, dY/dZ from neighbours.
            const float hl = (i > 0)         ? h_at(i - 1, j) : H[(j + 1) * stride + 0];
            const float hr = (i + 1 < N)     ? h_at(i + 1, j) : H[(j + 1) * stride + (N + 1)];
            const float hd = (j > 0)         ? h_at(i, j - 1) : H[0 * stride + (i + 1)];
            const float hu = (j + 1 < N)     ? h_at(i, j + 1) : H[(N + 1) * stride + (i + 1)];
            Vec3 n = normalize(Vec3{ (hl - hr), 2.0f * step, (hd - hu) });

            // Slope-shaded vertex colour: flat → green, steep → grey-brown.
            const float slope = 1.0f - cardinal::clamp(n.y, 0.0f, 1.0f);
            Vec3 grass{ 0.30f, 0.55f, 0.25f };
            Vec3 cliff{ 0.55f, 0.50f, 0.45f };
            Vec3 col{ grass.x * (1.0f - slope) + cliff.x * slope,
                      grass.y * (1.0f - slope) + cliff.y * slope,
                      grass.z * (1.0f - slope) + cliff.z * slope };

            grid[j * N + i] = V{ Vec3{ x, y, z }, n, col };
        }
    }

    // Tessellate to triangle list — keeps the existing forward renderer
    // happy (it walks `cpu_vertices()` directly without an index buffer).
    cardinal::vector<Vertex> verts;
    verts.reserve(static_cast<size_t>(N - 1) * (N - 1) * 6);
    for (u32 j = 0; j < N - 1; ++j) {
        for (u32 i = 0; i < N - 1; ++i) {
            const V& a = grid[(j    ) * N + (i    )];
            const V& b = grid[(j    ) * N + (i + 1)];
            const V& c = grid[(j + 1) * N + (i + 1)];
            const V& d = grid[(j + 1) * N + (i    )];
            verts.push_back({ a.pos, a.nrm, a.col });
            verts.push_back({ b.pos, b.nrm, b.col });
            verts.push_back({ c.pos, c.nrm, c.col });
            verts.push_back({ a.pos, a.nrm, a.col });
            verts.push_back({ c.pos, c.nrm, c.col });
            verts.push_back({ d.pos, d.nrm, d.col });
        }
    }
    auto m = Mesh::from_vertices(dev, verts.data(), static_cast<u32>(verts.size()));
    if (m) {
        char nm[64];
        cardinal::snprintf(nm, sizeof(nm), "TerrainChunk[%d,%d]", chunk_x, chunk_z);
        m->set_name(nm);
    }
    return m;
}

// ---------------------------------------------------------------------------
// Profile-driven chunk mesh. Same tessellation strategy as the legacy path
// but uses terrain_height_profile + terrain_color_profile for height + tint.
// ---------------------------------------------------------------------------
cardinal::shared_ptr<Mesh> generate_terrain_chunk_profile(
    rhi::Device& dev, const TerrainProfile& P,
    float chunk_size, u32 resolution, int chunk_x, int chunk_z)
{
    const u32   N    = cardinal::max<u32>(2, resolution);
    const float step = chunk_size / static_cast<float>(N - 1);
    const float ox   = static_cast<float>(chunk_x) * chunk_size - chunk_size * 0.5f;
    const float oz   = static_cast<float>(chunk_z) * chunk_size - chunk_size * 0.5f;

    const u32 stride = N + 2;
    cardinal::vector<float> H(stride * stride);
    for (u32 j = 0; j < stride; ++j) {
        for (u32 i = 0; i < stride; ++i) {
            const float wx = ox + (static_cast<float>(i) - 1.0f) * step;
            const float wz = oz + (static_cast<float>(j) - 1.0f) * step;
            H[j * stride + i] = terrain_height_profile(P, wx, wz);
        }
    }
    auto h_at = [&](u32 i, u32 j) -> float { return H[(j + 1) * stride + (i + 1)]; };

    struct V { Vec3 pos; Vec3 nrm; Vec3 col; };
    cardinal::vector<V> grid(N * N);
    for (u32 j = 0; j < N; ++j) {
        for (u32 i = 0; i < N; ++i) {
            const float x = ox + static_cast<float>(i) * step;
            const float z = oz + static_cast<float>(j) * step;
            const float y = h_at(i, j);

            const float hl = (i > 0)     ? h_at(i - 1, j) : H[(j + 1) * stride + 0];
            const float hr = (i + 1 < N) ? h_at(i + 1, j) : H[(j + 1) * stride + (N + 1)];
            const float hd = (j > 0)     ? h_at(i, j - 1) : H[0 * stride + (i + 1)];
            const float hu = (j + 1 < N) ? h_at(i, j + 1) : H[(N + 1) * stride + (i + 1)];
            Vec3 n = normalize(Vec3{ (hl - hr), 2.0f * step, (hd - hu) });

            Vec3 col = terrain_color_profile(P, y, n);
            grid[j * N + i] = V{ Vec3{ x, y, z }, n, col };
        }
    }

    cardinal::vector<Vertex> verts;
    verts.reserve(static_cast<size_t>(N - 1) * (N - 1) * 6);
    for (u32 j = 0; j < N - 1; ++j) {
        for (u32 i = 0; i < N - 1; ++i) {
            const V& a = grid[(j    ) * N + (i    )];
            const V& b = grid[(j    ) * N + (i + 1)];
            const V& c = grid[(j + 1) * N + (i + 1)];
            const V& d = grid[(j + 1) * N + (i    )];
            verts.push_back({ a.pos, a.nrm, a.col });
            verts.push_back({ b.pos, b.nrm, b.col });
            verts.push_back({ c.pos, c.nrm, c.col });
            verts.push_back({ a.pos, a.nrm, a.col });
            verts.push_back({ c.pos, c.nrm, c.col });
            verts.push_back({ d.pos, d.nrm, d.col });
        }
    }
    auto m = Mesh::from_vertices(dev, verts.data(), static_cast<u32>(verts.size()));
    if (m) {
        char nm[64];
        cardinal::snprintf(nm, sizeof(nm), "TerrainChunk[%s,%d,%d]",
                      P.name.empty() ? "?" : P.name.c_str(), chunk_x, chunk_z);
        m->set_name(nm);
    }
    return m;
}

cardinal::vector<u32> spawn_terrain_grid(Scene& scene, rhi::Device& dev,
                                    const TerrainGridDesc& desc)
{
    cardinal::vector<u32> ids;
    const int cx = cardinal::max(1, desc.chunks_x);
    const int cz = cardinal::max(1, desc.chunks_z);
    ids.reserve(static_cast<size_t>(cx) * cz);

    for (int j = 0; j < cz; ++j) {
        for (int i = 0; i < cx; ++i) {
            const int gx = i - cx / 2;
            const int gz = j - cz / 2;
            auto mesh = generate_terrain_chunk(dev, desc.params, gx, gz);
            if (mesh == nullptr) continue;

            char nm[64];
            cardinal::snprintf(nm, sizeof(nm), "Terrain[%d,%d]", gx, gz);
            auto& e = scene.add_entity(nm);
            e.mesh = mesh;
            e.transform.translation = desc.origin;
            e.tint = desc.tint;
            ids.push_back(e.id);
        }
    }
    return ids;
}

cardinal::vector<u32> spawn_terrain_grid_profile(Scene& scene, rhi::Device& dev,
                                            const TerrainGridProfileDesc& desc)
{
    cardinal::vector<u32> ids;
    const int cx = cardinal::max(1, desc.chunks_x);
    const int cz = cardinal::max(1, desc.chunks_z);
    ids.reserve(static_cast<size_t>(cx) * cz);

    for (int j = 0; j < cz; ++j) {
        for (int i = 0; i < cx; ++i) {
            const int gx = i - cx / 2;
            const int gz = j - cz / 2;
            auto mesh = generate_terrain_chunk_profile(
                dev, desc.profile, desc.chunk_size, desc.resolution, gx, gz);
            if (mesh == nullptr) continue;

            char nm[80];
            cardinal::snprintf(nm, sizeof(nm), "Terrain[%s,%d,%d]",
                          desc.profile.name.empty() ? "?" : desc.profile.name.c_str(),
                          gx, gz);
            auto& e = scene.add_entity(nm);
            e.mesh = mesh;
            e.transform.translation = desc.origin;
            e.tint = { 1.0f, 1.0f, 1.0f };  // mesh already carries profile tint
            ids.push_back(e.id);
        }
    }
    return ids;
}

}  // namespace cardinal::scene
