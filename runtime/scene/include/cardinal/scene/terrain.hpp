#pragma once

// =============================================================================
// Cardinal — Terrain (heightmap) generation + profile system.
//
// Two layers of API:
//
//   1. Single-noise legacy path (TerrainParams + generate_terrain_chunk +
//      spawn_terrain_grid) — kept for back-compat and tiny smoke tests.
//
//   2. Multi-layer profile path — what every world-builder tool actually
//      ships. A `TerrainProfile` is a named recipe combining N noise
//      layers (Add / Multiply / Max / Min × Smooth / Ridge / Billow,
//      with optional domain-warp), a height-and-slope-based tint ramp,
//      and a global height scale. Profiles live in a process-wide
//      `TerrainProfileLibrary` that the editor populates from disk +
//      engine-shipped presets, and the user can edit / save / reload
//      live without restarting.
//
// Both paths share the same noise primitives so a single-layer profile
// produces identical terrain to the equivalent TerrainParams.
//
// On-disk format is a tiny human-editable text format:
//
//     profile "Alpine Mountains"
//       description "Continental shelf with rocky ridges and snowy peaks"
//       base_height 0.0
//       height_scale 24.0
//       tint_low 0.30 0.55 0.25
//       tint_high 0.95 0.95 0.97
//       tint_cliff 0.55 0.50 0.45
//       tint_height_low 0.0
//       tint_height_high 18.0
//       cliff_blend 0.55
//       layer "continental"
//         op add
//         shape smooth
//         weight 1.0
//         frequency 0.012
//         octaves 5
//         ...
//       layer "ridges"
//         op add
//         shape ridge
//         weight 0.7
//         ...
//     profile "Rolling Hills"
//       ...
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/containers.hpp>
#include <cardinal/scene/math.hpp>

namespace cardinal::rhi { class Device; }

namespace cardinal::scene {

class  Mesh;
class  Scene;

// =============================================================================
// Legacy single-noise path
// =============================================================================
struct TerrainParams {
    u32   resolution{64};         // verts per side per chunk (>= 2)
    float chunk_size{32.0f};      // world units per chunk side (X = Z)
    float height_scale{6.0f};     // peak amplitude (world units)
    int   octaves{4};             // fbm octave count (1..8)
    float frequency{0.06f};       // base spatial frequency (1/world units)
    float lacunarity{2.0f};       // octave frequency multiplier
    float gain{0.5f};             // octave amplitude multiplier
    u32   seed{1337u};            // hash seed
};

float terrain_height(const TerrainParams& p, float wx, float wz) noexcept;
cardinal::shared_ptr<Mesh> generate_terrain_chunk(rhi::Device& dev,
                                             const TerrainParams& p,
                                             int chunk_x, int chunk_z);

struct TerrainGridDesc {
    TerrainParams params;
    int           chunks_x{4};
    int           chunks_z{4};
    Vec3          origin{0, 0, 0};
    Vec3          tint{1.0f, 1.0f, 1.0f};
};
cardinal::vector<u32> spawn_terrain_grid(Scene& scene,
                                    rhi::Device& dev,
                                    const TerrainGridDesc& desc);

// =============================================================================
// World-engine profile system
// =============================================================================

// How a layer combines with the running height.
enum class NoiseOp : u32 {
    Add = 0,        // h += w * v
    Multiply,       // h *= (0.5 + 0.5 * v) * w + (1 - w)   (mask)
    Max,            // h = max(h, w * v)
    Min,            // h = min(h, w * v)
};
const char* noise_op_name(NoiseOp) noexcept;

// Per-octave nonlinearity. Determines the "look" of the layer.
//   Smooth  — classic value/perlin-style noise, range ~[-1, +1]
//   Ridge   — 1 - |v|, sharp peaks (mountain ranges)
//   Billow  — |v|, puffy hills (sand dunes / cumulus clouds)
enum class NoiseShape : u32 {
    Smooth = 0,
    Ridge,
    Billow,
};
const char* noise_shape_name(NoiseShape) noexcept;

// One layer in a profile. Layers are evaluated in declaration order.
struct NoiseLayer {
    cardinal::string id;                 // "base", "ridges", "river_mask"
    bool        enabled{true};
    NoiseOp     op{NoiseOp::Add};
    NoiseShape  shape{NoiseShape::Smooth};
    float       weight{1.0f};
    float       frequency{0.05f};   // base spatial frequency (1/world units)
    int         octaves{4};         // fbm octaves (1..8)
    float       lacunarity{2.0f};
    float       gain{0.5f};
    u32         seed{1337u};
    float       offset_x{0.0f};     // sample at (wx + offset, wz + offset)
    float       offset_z{0.0f};
    float       warp_amount{0.0f};  // domain-warp strength in world units
    float       warp_frequency{0.05f};
};

// One named, reusable terrain recipe.
struct TerrainProfile {
    cardinal::string             name;
    cardinal::string             description;
    float                   base_height{0.0f};        // DC offset before layers
    float                   height_scale{8.0f};       // multiplier on combined height
    cardinal::vector<NoiseLayer> layers;

    // Tint ramp — Vertex colour is computed from height and surface slope:
    //   tint_low  → grass/valley colour at tint_height_low
    //   tint_high → snow/peak colour at tint_height_high
    //   tint_cliff → blended in proportional to (1 - dot(normal, up))
    Vec3                    tint_low { 0.30f, 0.55f, 0.25f };
    Vec3                    tint_high{ 0.95f, 0.95f, 0.97f };
    Vec3                    tint_cliff{ 0.55f, 0.50f, 0.45f };
    float                   tint_height_low{0.0f};
    float                   tint_height_high{16.0f};
    float                   cliff_blend_threshold{0.5f};   // 1 - dot(n, up)
};

// ---------------------------------------------------------------------------
// TerrainProfileLibrary — process-wide named-profile store. The editor
// drives this through the Studio's terrain modal; plugins / scripts can
// register profiles too.
// ---------------------------------------------------------------------------
class TerrainProfileLibrary {
public:
    static TerrainProfileLibrary& instance();

    // Insert or replace by `name`. The name on the profile struct wins.
    void                                set(TerrainProfile p);
    bool                                remove(const char* name);
    const TerrainProfile*               find(const char* name) const noexcept;
    const cardinal::vector<TerrainProfile>&  all() const noexcept;
    cardinal::vector<cardinal::string>            names() const;

    // Persistence — append-style (load merges into existing library; save
    // writes the entire current set).
    bool   save_to_file (const char* path) const;
    bool   load_from_file(const char* path);

private:
    TerrainProfileLibrary();
    TerrainProfileLibrary(const TerrainProfileLibrary&)            = delete;
    TerrainProfileLibrary& operator=(const TerrainProfileLibrary&) = delete;

    struct Impl;
    cardinal::unique_ptr<Impl> p_;
};

// Register the engine-shipped curated profile presets:
//   "Plains", "Rolling Hills", "Alpine Mountains",
//   "Volcanic Wasteland", "Coastal Dunes", "Canyon".
// Idempotent — calling it again is a no-op for any profile that's already
// in the library under the same name.
void register_default_terrain_profiles();

// Sample the profile's combined height at world XZ.
float terrain_height_profile(const TerrainProfile& p,
                             float wx, float wz) noexcept;

// Profile-driven slope+height vertex tint.
Vec3  terrain_color_profile(const TerrainProfile& p,
                            float height, const Vec3& normal) noexcept;

// Build one chunk tile mesh from a profile.
cardinal::shared_ptr<Mesh> generate_terrain_chunk_profile(
    rhi::Device&           dev,
    const TerrainProfile&  profile,
    float                  chunk_size,
    u32                    resolution,
    int                    chunk_x,
    int                    chunk_z);

// Spawn an NxN profile-driven chunk grid into a scene.
struct TerrainGridProfileDesc {
    TerrainProfile profile;
    int            chunks_x{4};
    int            chunks_z{4};
    u32            resolution{64};
    float          chunk_size{32.0f};
    Vec3           origin{0, 0, 0};
};
cardinal::vector<u32> spawn_terrain_grid_profile(Scene& scene,
                                            rhi::Device& dev,
                                            const TerrainGridProfileDesc& desc);

}  // namespace cardinal::scene
