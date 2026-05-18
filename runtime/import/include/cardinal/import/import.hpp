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

#include <cardinal/core/types.hpp>   // foundation: string / u32 / …
#include <cardinal/scene/math.hpp>

#include <vector>

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
    std::vector<cardinal::scene::Vec3> positions;
    std::vector<cardinal::scene::Vec3> normals;     // empty ⇒ flat-shade later
    std::vector<cardinal::scene::Vec3> colors;      // empty ⇒ white
    std::vector<Vec2>                  uvs;          // empty ⇒ none
    std::vector<u32>                   indices;      // triangle list
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
    cardinal::string           base_color_texture;   // file path/uri, relative
};

struct ImportNode {
    cardinal::string           name;
    cardinal::scene::Mat4 transform{cardinal::scene::Mat4::identity()};
    std::vector<int>      meshes;     // indices into ImportScene.meshes
    std::vector<int>      children;   // indices into ImportScene.nodes
};

struct ImportScene {
    std::vector<ImportMesh>     meshes;
    std::vector<ImportMaterial> materials;
    std::vector<ImportNode>     nodes;
    std::vector<int>            roots;       // root node indices
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

// ImportScene → engine runtime asset structs: see the header-only
// bridge <cardinal/import/to_asset.hpp> (kept out of this TU so the
// import library stays asset-free / cycle-free, see note up top).

}  // namespace cardinal::import
