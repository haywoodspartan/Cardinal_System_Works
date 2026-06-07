#pragma once

// =============================================================================
// Cardinal — DCC asset importer.
//
// Brings geometry + materials in from the content tools artists actually
// use. None of those ship a readable native format, so — like every
// production engine — we import via the interchange formats they ALL
// export:
//
//   Maya / Autodesk ............ FBX, OBJ, glTF
//   Blender .................... glTF, OBJ, FBX
//   Daz 3D ..................... OBJ, FBX
//   Unreal Engine (out) ........ glTF, FBX, OBJ
//   …everything else ........... OBJ / glTF
//
// So one robust OBJ path + one robust glTF 2.0 path already covers every
// listed tool. OBJ (this file's backend) is hand-rolled + zero-dep and
// lands first; glTF 2.0 (.gltf/.glb — the modern PBR-carrying universal
// path) is the next backend and slots into the same ImportScene. FBX
// (Autodesk's binary interchange) follows behind the same interface.
//
// The importer emits a neutral, source-faithful ImportScene (multi-mesh,
// materials, node hierarchy, UVs) so nothing is lost; a thin converter
// projects it onto the engine's runtime asset structs for immediate use.
// =============================================================================

#include <cardinal/core/types.hpp>        // foundation: string / u32 / …
#include <cardinal/core/containers.hpp>   // cardinal::vector
#include <cardinal/scene/math.hpp>

// NOTE: cardinal::import is deliberately core+scene only (NOT asset).
// asset depends on cook, so an import→asset link would form an
// import→asset→cook cycle and forbid cook from delegating to the
// importer. The ImportScene → engine asset-struct converters live in
// the header-only bridge <cardinal/import/to_asset.hpp> instead, which
// callers that already link cardinal::asset include explicitly.

namespace cardinal::import {

struct Vec2 { float u{0.0f}, v{0.0f}; };

enum class Format : u32 { Unknown = 0, Obj, Gltf, Glb, Fbx };

const char* format_name(Format f) noexcept;
// Extension-based detection (".obj" → Obj, ".glb" → Glb, …).
Format      detect_format(const cardinal::string& path) noexcept;

// ---------------------------------------------------------------------------
// Neutral imported scene — a faithful copy of what the DCC exported.
// Geometry is de-indexed into compact (unique-vertex + index) form.
// ---------------------------------------------------------------------------
struct ImportMesh {
    cardinal::string                        name;
    cardinal::vector<cardinal::scene::Vec3> positions;
    cardinal::vector<cardinal::scene::Vec3> normals;     // empty ⇒ flat-shade later
    cardinal::vector<cardinal::scene::Vec3> colors;      // empty ⇒ white
    cardinal::vector<Vec2>                  uvs;          // empty ⇒ none
    cardinal::vector<u32>                   indices;      // triangle list
    int                                material{-1}; // into ImportScene.materials
};

// Metallic-roughness PBR (maps 1:1 onto asset::MaterialAsset).
struct ImportMaterial {
    cardinal::string           name;
    cardinal::scene::Vec3 base_color{0.8f, 0.8f, 0.8f};
    float                 metallic {0.0f};
    float                 roughness{0.5f};
    cardinal::scene::Vec3 emission {0.0f, 0.0f, 0.0f};
    float                 emission_strength{0.0f};
    cardinal::string           base_color_texture;   // albedo (file path/uri, relative)

    // ---- Full PBR texture-map set ------------------------------------------
    // Every field is a relative file path/uri (empty ⇒ that map is absent).
    // glTF packs metallic+roughness into ONE texture (G=roughness, B=metallic),
    // while DCC tools (OBJ map_Pm/map_Pr) and Megascans ship SEPARATE maps —
    // both are representable: metallic_roughness_texture (+ mr_packed) for the
    // packed case, metallic_texture/roughness_texture for the separate case.
    cardinal::string metallic_roughness_texture;     // packed ORM/RMA (see mr_packed)
    cardinal::string roughness_texture;              // separate (map_Pr / Megascans)
    cardinal::string metallic_texture;               // separate (map_Pm / Megascans)
    cardinal::string normal_texture;
    float            normal_scale{1.0f};             // glTF normalTexture.scale
    cardinal::string occlusion_texture;              // AO / cavity
    float            occlusion_strength{1.0f};       // glTF occlusionTexture.strength
    cardinal::string emissive_texture;
    cardinal::string height_texture;                 // displacement / height / bump
    float            height_scale{1.0f};
    cardinal::string specular_texture;
    cardinal::string opacity_texture;                // alpha mask
    bool             mr_packed{false};        // true ⇒ sample metallic_roughness_texture
    bool             invert_roughness{false}; // true ⇒ source is a gloss map (1 − roughness)
    // Recognised map types with no dedicated slot (translucency, fuzz,
    // curvature, …): preserved as resolved paths, never silently dropped.
    cardinal::vector<cardinal::string> unmapped_textures;
};

struct ImportNode {
    cardinal::string           name;
    cardinal::scene::Mat4 transform{cardinal::scene::Mat4::identity()};
    cardinal::vector<int>      meshes;     // indices into ImportScene.meshes
    cardinal::vector<int>      children;   // indices into ImportScene.nodes
};

struct ImportScene {
    cardinal::vector<ImportMesh>     meshes;
    cardinal::vector<ImportMaterial> materials;
    cardinal::vector<ImportNode>     nodes;
    cardinal::vector<int>            roots;       // root node indices
    cardinal::string                 source_format;
    cardinal::string                 diagnostics; // human-readable summary
    bool                        ok{false};

    // Totals (filled by the importer for quick UI/logging).
    u32 total_vertices() const noexcept;
    u32 total_triangles() const noexcept;
};

// ---------------------------------------------------------------------------
// Entry points. import_file autodetects by extension; the per-format
// functions can be called directly. On failure ok=false and (if given)
// *error carries the reason; diagnostics always describes what happened.
// ---------------------------------------------------------------------------
ImportScene import_file(const cardinal::string& path, cardinal::string* error = nullptr);
ImportScene import_obj (const cardinal::string& path, cardinal::string* error = nullptr);
ImportScene import_gltf(const cardinal::string& path, cardinal::string* error = nullptr);

// ---------------------------------------------------------------------------
// Megascans / Quixel Bridge — a scanned-PBR asset is a metadata JSON plus a
// set of texture maps (and, for 3D assets, mesh LODs). Deliberately NOT in
// the extension-dispatch enum: a manifest is a .json that could be anything,
// so it is imported explicitly. `path` may be the manifest .json itself OR
// the asset directory (the manifest is then located by sniffing). Surface
// assets yield a material-only scene; 3d assets recurse through import_file
// for the mesh LOD (OBJ/glTF; FBX-only LODs import material-complete with
// geometry pending the FBX backend). Tolerant to Bridge version drift:
// maps/components synonyms, uri/path/file aliases, and a filename-scan
// fallback when the JSON omits or stales a map.
// ---------------------------------------------------------------------------
ImportScene import_megascans(const cardinal::string& path, cardinal::string* error = nullptr);
// Cheap sniff: true when `dir_or_json` is (or contains) a parseable manifest
// whose root object carries any of maps / components / meshes / (id + type).
bool        is_megascans(const cardinal::string& dir_or_json);

// ---------------------------------------------------------------------------
// Heightmap import — for terrain placement. A heightmap is an image whose
// pixel intensity is elevation; this decodes it to a neutral CPU height grid
// that scene/terrain turns into a mesh (kept out of cardinal::import so the
// import lib links no rhi). Heights are normalised to [0,1]; `min`/`max` carry
// the raw observed range for UI. Row-major: heights[y*width + x] (matches the
// edit/brush stamp_height_grid layout, so an imported field interops with the
// sculpt brushes for free).
// ---------------------------------------------------------------------------
struct HeightField {
    cardinal::vector<float> heights;          // size = width*height, row-major, [0,1]
    u32                     width {0};
    u32                     height{0};
    float                   min {0.0f};       // raw min before normalisation
    float                   max {0.0f};       // raw max before normalisation
    bool                    ok  {false};
    cardinal::string        source_format;    // "raw16" / "bmp" / "tga" / …
    cardinal::string        diagnostics;
};

// Decode a heightmap image to a HeightField. Format is detected by extension
// then content. NO-inflate formats land first: headerless RAW16 (.r16/.raw,
// 16-bit LE, square dimension inferred from file size), uncompressed BMP
// (8/24/32-bit), and uncompressed TGA (type 2 RGB / type 3 grayscale). 8-bit
// sources use the red/luminance channel. PNG (needs DEFLATE) and EXR (float)
// land once cardinal::core::compress::inflate exists. For a non-square RAW16,
// pass the side length via `raw_dim` (0 ⇒ infer a square from the byte count).
HeightField decode_heightmap(const cardinal::string& path,
                             cardinal::string* error = nullptr,
                             u32 raw_dim = 0);

// ImportScene → engine runtime asset structs: see the header-only
// bridge <cardinal/import/to_asset.hpp> (kept out of this TU so the
// import library stays asset-free / cycle-free, see note up top).

}  // namespace cardinal::import
