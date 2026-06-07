#pragma once

// =============================================================================
// Cardinal — Runtime asset registry.
//
// What game code talks to. Loads cooked + packed assets on demand,
// caches them, hands out typed handles (MeshAsset / TextureAsset /
// MaterialAsset / ShaderAsset).
//
// Backing store can be either a directory of `.cooked` files (development
// mode) or one or more pack::Archive (shipping mode). Both expose the same
// `find(key) -> bytes` interface; the Registry doesn't care.
// =============================================================================

#include <cardinal/core/types.hpp>        // function/memory/string
#include <cardinal/core/containers.hpp>   // unordered_map/vector
#include <cardinal/cook/cook.hpp>
#include <cardinal/scene/math.hpp>

namespace cardinal::pack { class Archive; }

namespace cardinal::asset {

// ---------------------------------------------------------------------------
// Typed asset structures. These are *runtime* views over the cooked bytes
// (the cook module produces them in known formats, see cook.cpp).
// ---------------------------------------------------------------------------
struct TextureAsset {
    u32             width{0};
    u32             height{0};
    u32             channels{4};
    cardinal::vector<u8> rgba;          // size = width*height*channels
};

struct MeshVertex {
    cardinal::scene::Vec3 position{0,0,0};
    cardinal::scene::Vec3 normal  {0,1,0};
    cardinal::scene::Vec3 color   {1,1,1};
};

struct MeshAsset {
    cardinal::vector<MeshVertex> vertices;
    cardinal::vector<u32>        indices;     // triangle list
};

struct ShaderAsset {
    u32              stage{0};            // shader::Stage as u32
    cardinal::string      entry_point;
    cardinal::vector<u8>  bytecode;            // SPIR-V or DXIL
};

// PBR-ish material properties. Renderer's "Solid" mode multiplies these
// into the per-fragment shading.
struct MaterialAsset {
    cardinal::scene::Vec3 base_color {0.8f, 0.8f, 0.8f};
    float                 metallic   {0.0f};
    float                 roughness  {0.5f};
    cardinal::scene::Vec3 emission   {0.0f, 0.0f, 0.0f};
    float                 emission_strength{0.0f};
    cardinal::string           base_color_texture;     // albedo asset key (optional)

    // ---- Full PBR texture-map set (asset keys; empty ⇒ that map absent) ----
    // Mirrors import::ImportMaterial 1:1 so to_asset_material is a flat copy.
    // Stored now (so imports persist losslessly); the renderer samples them in
    // a later GPU phase. metallic_roughness_texture is the glTF ORM-packed
    // form (G=roughness, B=metallic) gated by mr_packed; roughness_texture /
    // metallic_texture are the separate (OBJ / Megascans) form.
    cardinal::string metallic_roughness_texture;
    cardinal::string roughness_texture;
    cardinal::string metallic_texture;
    cardinal::string normal_texture;
    float            normal_scale{1.0f};
    cardinal::string occlusion_texture;
    float            occlusion_strength{1.0f};
    cardinal::string emissive_texture;
    cardinal::string height_texture;
    float            height_scale{1.0f};
    cardinal::string specular_texture;
    cardinal::string opacity_texture;
    bool             mr_packed{false};
    bool             invert_roughness{false};
};

// ---------------------------------------------------------------------------
// Registry — what game code calls.
// ---------------------------------------------------------------------------
class Registry {
public:
    static cardinal::shared_ptr<Registry> create();
    ~Registry();

    // ----- Backing-store mounts -------------------------------------
    void mount_directory(const cardinal::string& cooked_dir);
    void mount_archive  (cardinal::shared_ptr<cardinal::pack::Archive> archive);
    void clear_mounts() noexcept;

    // ----- Existence query ------------------------------------------
    bool                       contains(const cardinal::string& key) const;
    cook::AssetType            type_of(const cardinal::string& key) const;
    cardinal::vector<cardinal::string>   keys() const;

    // ----- Typed loaders -------------------------------------------
    // Each returns a shared_ptr cached by key. If the asset is missing or
    // the cooked blob doesn't decode as the expected type, returns null.
    cardinal::shared_ptr<TextureAsset>  load_texture (const cardinal::string& key);
    cardinal::shared_ptr<MeshAsset>     load_mesh    (const cardinal::string& key);
    cardinal::shared_ptr<ShaderAsset>   load_shader  (const cardinal::string& key);
    cardinal::shared_ptr<MaterialAsset> load_material(const cardinal::string& key);

    // Generic raw-bytes loader (still indirects through cooked-blob unwrap).
    bool load_raw(const cardinal::string& key, cardinal::vector<u8>& out_bytes);

    // ----- Material registration / overrides ------------------------
    // Materials are tiny, so we keep them in-memory (no disk round-trip
    // unless explicitly cooked). register_material lets game/editor code
    // create runtime materials that look the same as cooked ones.
    void register_material(const cardinal::string& key, MaterialAsset m);

    struct Stats {
        u32 mount_dirs{0};
        u32 mount_archives{0};
        u32 cached_textures{0};
        u32 cached_meshes{0};
        u32 cached_shaders{0};
        u32 cached_materials{0};
        u64 bytes_resident{0};
    };
    Stats stats() const noexcept;

private:
    Registry() = default;
    bool fetch_raw_(const cardinal::string& key, cardinal::vector<u8>& out_bytes,
                    cook::AssetType* out_type);

    cardinal::vector<cardinal::string>                                    dirs_;
    cardinal::vector<cardinal::shared_ptr<cardinal::pack::Archive>>       archives_;
    cardinal::unordered_map<cardinal::string, cardinal::shared_ptr<TextureAsset>>  tex_cache_;
    cardinal::unordered_map<cardinal::string, cardinal::shared_ptr<MeshAsset>>     mesh_cache_;
    cardinal::unordered_map<cardinal::string, cardinal::shared_ptr<ShaderAsset>>   shader_cache_;
    cardinal::unordered_map<cardinal::string, cardinal::shared_ptr<MaterialAsset>> mat_cache_;
};

// ---------------------------------------------------------------------------
// Helpers — typed serialise/deserialise of the cook payloads.
// ---------------------------------------------------------------------------
namespace codec {
    cardinal::vector<u8>  encode_texture (const TextureAsset& t);
    bool             decode_texture (const cardinal::vector<u8>& bytes, TextureAsset& out);
    cardinal::vector<u8>  encode_mesh    (const MeshAsset& m);
    bool             decode_mesh    (const cardinal::vector<u8>& bytes, MeshAsset& out);
    cardinal::vector<u8>  encode_shader  (const ShaderAsset& s);
    bool             decode_shader  (const cardinal::vector<u8>& bytes, ShaderAsset& out);
    cardinal::vector<u8>  encode_material(const MaterialAsset& m);
    bool             decode_material(const cardinal::vector<u8>& bytes, MaterialAsset& out);
}

}  // namespace cardinal::asset
