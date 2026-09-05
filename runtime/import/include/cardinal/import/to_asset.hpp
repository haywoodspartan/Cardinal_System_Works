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
// One ImportMesh → one asset::MeshAsset (positions / normals / colors /
// UVs / skin influences carried). The (scene, mesh) overload also builds
// the mesh's skeleton from its ImportSkin so a skinned import round-trips
// through the codec whole. Materials map 1:1 onto asset::MaterialAsset.
// =============================================================================

#include <cardinal/import/import.hpp>   // foundation vocab via core/types
#include <cardinal/core/std/utility.hpp>  // cardinal::move
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
        v.texcoord = (i < m.uvs.size())
                       ? cardinal::scene::Vec2{m.uvs[i].u, m.uvs[i].v}
                       : cardinal::scene::Vec2{0.0f, 0.0f};
    }
    out.indices = m.indices;
    // Skin influences ride along only when they cover every vertex (the
    // importer already remapped joints parent-first + normalised weights).
    if (!m.influences.empty() && m.influences.size() == m.positions.size())
        out.influences = m.influences;
    return out;
}

// Skeleton for ImportScene.skins[skin_index] — names from the joint nodes,
// parents / binds copied flat. Empty when the index is out of range or the
// skin fails the parent-first invariant.
inline cardinal::scene::skin::Skeleton to_asset_skeleton(const ImportScene& s, int skin_index) {
    cardinal::scene::skin::Skeleton out;
    if (skin_index < 0 || static_cast<cardinal::usize>(skin_index) >= s.skins.size()) return out;
    const ImportSkin& sk = s.skins[static_cast<cardinal::usize>(skin_index)];
    const cardinal::usize n = sk.joint_nodes.size();
    if (sk.joint_parents.size() != n || sk.local_bind.size() != n || sk.inverse_bind.size() != n)
        return out;
    out.joints.reserve(n);
    for (cardinal::usize i = 0; i < n; ++i) {
        cardinal::scene::skin::Joint j;
        const int node = sk.joint_nodes[i];
        j.name = (node >= 0 && static_cast<cardinal::usize>(node) < s.nodes.size())
               ? s.nodes[static_cast<cardinal::usize>(node)].name : cardinal::string("joint");
        j.parent       = sk.joint_parents[i];
        j.local_bind   = sk.local_bind[i];
        j.inverse_bind = sk.inverse_bind[i];
        out.joints.push_back(cardinal::move(j));
    }
    if (!out.validate()) out.joints.clear();
    return out;
}

// Mesh + its skeleton in one step. A skinned mesh whose skin can't be
// resolved degrades to rigid (influences dropped) rather than failing.
inline cardinal::asset::MeshAsset to_asset_mesh(const ImportScene& s, const ImportMesh& m) {
    cardinal::asset::MeshAsset out = to_asset_mesh(m);
    if (!out.influences.empty()) {
        out.skeleton = to_asset_skeleton(s, m.skin);
        if (out.skeleton.joints.empty()) out.influences.clear();
    }
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
    // Full PBR map set — field names mirror ImportMaterial 1:1.
    out.metallic_roughness_texture = m.metallic_roughness_texture;
    out.roughness_texture          = m.roughness_texture;
    out.metallic_texture           = m.metallic_texture;
    out.normal_texture             = m.normal_texture;
    out.normal_scale               = m.normal_scale;
    out.occlusion_texture          = m.occlusion_texture;
    out.occlusion_strength         = m.occlusion_strength;
    out.emissive_texture           = m.emissive_texture;
    out.height_texture             = m.height_texture;
    out.height_scale               = m.height_scale;
    out.specular_texture           = m.specular_texture;
    out.opacity_texture            = m.opacity_texture;
    out.mr_packed                  = m.mr_packed;
    out.invert_roughness           = m.invert_roughness;
    // (m.unmapped_textures has no MaterialAsset counterpart — intentionally
    //  dropped at the bridge; those maps have no engine slot to feed.)
    return out;
}

}  // namespace cardinal::import
