#pragma once

// =============================================================================
// Cardinal — ImportScene → engine runtime asset-struct bridge.
//
// Header-only on purpose. cardinal::import is core+scene only so cook
// can delegate to it without an import→asset→cook link cycle (see the
// note in import.hpp). Callers that already link cardinal::asset and
// want engine structs include THIS header — the conversion compiles
// into their TU, no extra link edge on the importer.
//
//   #include <cardinal/import/to_asset.hpp>
//   auto mesh = cardinal::import::to_asset_mesh(scene.meshes[0]);
//
// One ImportMesh → one asset::MeshAsset (UVs dropped — the engine
// vertex has no UV slot yet; positions/normals/colors carried).
// Materials map 1:1 onto the metallic-roughness asset::MaterialAsset.
// =============================================================================

#include <cardinal/import/import.hpp>   // foundation vocab via core/types
#include <cardinal/asset/asset.hpp>

namespace cardinal::import {

inline cardinal::asset::MeshAsset to_asset_mesh(const ImportMesh& m) {
    cardinal::asset::MeshAsset out;
    out.vertices.resize(m.positions.size());
    for (cardinal::usize i = 0; i < m.positions.size(); ++i) {
        auto& v = out.vertices[i];
        v.position = m.positions[i];
        v.normal   = (i < m.normals.size())
                       ? m.normals[i] : cardinal::scene::Vec3{0, 1, 0};
        v.color    = (i < m.colors.size())
                       ? m.colors[i]  : cardinal::scene::Vec3{1, 1, 1};
    }
    out.indices = m.indices;
    return out;
}

inline cardinal::asset::MaterialAsset to_asset_material(const ImportMaterial& m) {
    cardinal::asset::MaterialAsset out;
    out.base_color         = m.base_color;
    out.metallic           = m.metallic;
    out.roughness          = m.roughness;
    out.emission           = m.emission;
    out.emission_strength  = m.emission_strength;
    out.base_color_texture = m.base_color_texture;
    return out;
}

}  // namespace cardinal::import
