#include <cardinal/asset/asset.hpp>

#include <cardinal/core/log.hpp>
#include <cardinal/pack/pack.hpp>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace cardinal::asset {

// ---------------------------------------------------------------------------
// Codec — small typed serialisers that travel inside cook::CookedAsset.payload
// ---------------------------------------------------------------------------
namespace codec {

namespace {

void wr_u32(std::vector<u8>& o, u32 v) {
    o.push_back(static_cast<u8>(v));
    o.push_back(static_cast<u8>(v >> 8));
    o.push_back(static_cast<u8>(v >> 16));
    o.push_back(static_cast<u8>(v >> 24));
}
void wr_f(std::vector<u8>& o, float v) {
    u32 bits;
    std::memcpy(&bits, &v, sizeof(bits));
    wr_u32(o, bits);
}
void wr_bytes(std::vector<u8>& o, const u8* p, usize n) {
    o.insert(o.end(), p, p + n);
}
void wr_str(std::vector<u8>& o, const std::string& s) {
    wr_u32(o, static_cast<u32>(s.size()));
    wr_bytes(o, reinterpret_cast<const u8*>(s.data()), s.size());
}

u32 rd_u32(const u8* p) {
    return static_cast<u32>(p[0]) | (static_cast<u32>(p[1]) << 8) |
           (static_cast<u32>(p[2]) << 16) | (static_cast<u32>(p[3]) << 24);
}
float rd_f(const u8* p) {
    u32 bits = rd_u32(p);
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

}  // namespace

std::vector<u8> encode_texture(const TextureAsset& t) {
    std::vector<u8> o;
    wr_u32(o, t.width);
    wr_u32(o, t.height);
    wr_u32(o, t.channels);
    wr_u32(o, static_cast<u32>(t.rgba.size()));
    wr_bytes(o, t.rgba.data(), t.rgba.size());
    return o;
}

bool decode_texture(const std::vector<u8>& bytes, TextureAsset& out) {
    if (bytes.size() < 16) return false;
    out.width    = rd_u32(bytes.data() + 0);
    out.height   = rd_u32(bytes.data() + 4);
    out.channels = rd_u32(bytes.data() + 8);
    const u32 n  = rd_u32(bytes.data() + 12);
    if (16 + n > bytes.size()) return false;
    out.rgba.assign(bytes.begin() + 16, bytes.begin() + 16 + n);
    return true;
}

std::vector<u8> encode_mesh(const MeshAsset& m) {
    std::vector<u8> o;
    wr_u32(o, static_cast<u32>(m.vertices.size()));
    for (const auto& v : m.vertices) {
        wr_f(o, v.position.x); wr_f(o, v.position.y); wr_f(o, v.position.z);
        wr_f(o, v.normal.x);   wr_f(o, v.normal.y);   wr_f(o, v.normal.z);
        wr_f(o, v.color.x);    wr_f(o, v.color.y);    wr_f(o, v.color.z);
    }
    wr_u32(o, static_cast<u32>(m.indices.size()));
    for (u32 i : m.indices) wr_u32(o, i);
    return o;
}

bool decode_mesh(const std::vector<u8>& bytes, MeshAsset& out) {
    if (bytes.size() < 4) return false;
    const u32 vc = rd_u32(bytes.data());
    const usize per_vert = 9 * 4;
    if (4 + vc * per_vert + 4 > bytes.size()) return false;
    out.vertices.resize(vc);
    usize off = 4;
    for (u32 i = 0; i < vc; ++i) {
        MeshVertex& v = out.vertices[i];
        v.position.x = rd_f(bytes.data() + off + 0);
        v.position.y = rd_f(bytes.data() + off + 4);
        v.position.z = rd_f(bytes.data() + off + 8);
        v.normal.x   = rd_f(bytes.data() + off + 12);
        v.normal.y   = rd_f(bytes.data() + off + 16);
        v.normal.z   = rd_f(bytes.data() + off + 20);
        v.color.x    = rd_f(bytes.data() + off + 24);
        v.color.y    = rd_f(bytes.data() + off + 28);
        v.color.z    = rd_f(bytes.data() + off + 32);
        off += per_vert;
    }
    const u32 ic = rd_u32(bytes.data() + off);
    off += 4;
    if (off + ic * 4 > bytes.size()) return false;
    out.indices.resize(ic);
    for (u32 i = 0; i < ic; ++i) out.indices[i] = rd_u32(bytes.data() + off + i * 4);
    return true;
}

std::vector<u8> encode_shader(const ShaderAsset& s) {
    std::vector<u8> o;
    wr_u32(o, s.stage);
    wr_str(o, s.entry_point);
    wr_u32(o, static_cast<u32>(s.bytecode.size()));
    wr_bytes(o, s.bytecode.data(), s.bytecode.size());
    return o;
}

bool decode_shader(const std::vector<u8>& bytes, ShaderAsset& out) {
    if (bytes.size() < 8) return false;
    out.stage = rd_u32(bytes.data());
    const u32 nl = rd_u32(bytes.data() + 4);
    if (8 + nl > bytes.size()) return false;
    out.entry_point.assign(reinterpret_cast<const char*>(bytes.data() + 8), nl);
    usize off = 8 + nl;
    if (off + 4 > bytes.size()) return false;
    const u32 bl = rd_u32(bytes.data() + off);
    off += 4;
    if (off + bl > bytes.size()) return false;
    out.bytecode.assign(bytes.begin() + off, bytes.begin() + off + bl);
    return true;
}

std::vector<u8> encode_material(const MaterialAsset& m) {
    std::vector<u8> o;
    wr_f(o, m.base_color.x); wr_f(o, m.base_color.y); wr_f(o, m.base_color.z);
    wr_f(o, m.metallic);
    wr_f(o, m.roughness);
    wr_f(o, m.emission.x);   wr_f(o, m.emission.y);   wr_f(o, m.emission.z);
    wr_f(o, m.emission_strength);
    wr_str(o, m.base_color_texture);
    return o;
}

bool decode_material(const std::vector<u8>& bytes, MaterialAsset& out) {
    if (bytes.size() < 36) return false;
    out.base_color.x       = rd_f(bytes.data() + 0);
    out.base_color.y       = rd_f(bytes.data() + 4);
    out.base_color.z       = rd_f(bytes.data() + 8);
    out.metallic           = rd_f(bytes.data() + 12);
    out.roughness          = rd_f(bytes.data() + 16);
    out.emission.x         = rd_f(bytes.data() + 20);
    out.emission.y         = rd_f(bytes.data() + 24);
    out.emission.z         = rd_f(bytes.data() + 28);
    out.emission_strength  = rd_f(bytes.data() + 32);
    if (bytes.size() < 40) { out.base_color_texture.clear(); return true; }
    const u32 nl = rd_u32(bytes.data() + 36);
    if (40 + nl > bytes.size()) return false;
    out.base_color_texture.assign(reinterpret_cast<const char*>(bytes.data() + 40), nl);
    return true;
}

}  // namespace codec

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------
std::shared_ptr<Registry> Registry::create() {
    return std::shared_ptr<Registry>(new Registry());
}
Registry::~Registry() = default;

void Registry::mount_directory(const std::string& dir) { dirs_.push_back(dir); }
void Registry::mount_archive  (std::shared_ptr<cardinal::pack::Archive> a) {
    if (a) archives_.push_back(std::move(a));
}
void Registry::clear_mounts() noexcept { dirs_.clear(); archives_.clear(); }

bool Registry::contains(const std::string& key) const {
    for (const auto& a : archives_) if (a && a->contains(key)) return true;
    for (const auto& d : dirs_) {
        if (fs::exists(d + "/" + key + ".cooked")) return true;
    }
    return false;
}

cook::AssetType Registry::type_of(const std::string& key) const {
    for (const auto& a : archives_) {
        if (auto* e = a ? a->find(key) : nullptr) return e->type;
    }
    return cook::AssetType::Unknown;
}

std::vector<std::string> Registry::keys() const {
    std::vector<std::string> r;
    for (const auto& a : archives_) {
        if (!a) continue;
        for (const auto& e : a->entries()) r.push_back(e.key);
    }
    for (const auto& d : dirs_) {
        std::error_code ec;
        if (!fs::exists(d, ec)) continue;
        for (auto& it : fs::recursive_directory_iterator(d, ec)) {
            if (ec) break;
            if (!it.is_regular_file()) continue;
            if (it.path().extension() != ".cooked") continue;
            const std::string rel = fs::relative(it.path(), d).string();
            // Strip ".cooked" suffix.
            r.push_back(rel.substr(0, rel.size() - 7));
        }
    }
    std::sort(r.begin(), r.end());
    r.erase(std::unique(r.begin(), r.end()), r.end());
    return r;
}

bool Registry::fetch_raw_(const std::string& key, std::vector<u8>& out_bytes,
                          cook::AssetType* out_type)
{
    // Archives win over directories (shipping mode > development mode).
    for (const auto& a : archives_) {
        if (a && a->contains(key)) {
            std::vector<u8> wrapped;
            if (!a->load_blocking(key, wrapped)) return false;
            cook::CookedAsset cooked;
            if (!cook::CookedAsset::deserialize(wrapped, cooked)) return false;
            if (out_type) *out_type = cooked.type;
            out_bytes = std::move(cooked.payload);
            return true;
        }
    }
    for (const auto& d : dirs_) {
        const std::string p = d + "/" + key + ".cooked";
        std::ifstream f(p, std::ios::binary);
        if (!f) continue;
        f.seekg(0, std::ios::end);
        const auto n = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<u8> wrapped(static_cast<usize>(n));
        f.read(reinterpret_cast<char*>(wrapped.data()), n);
        cook::CookedAsset cooked;
        if (!cook::CookedAsset::deserialize(wrapped, cooked)) continue;
        if (out_type) *out_type = cooked.type;
        out_bytes = std::move(cooked.payload);
        return true;
    }
    return false;
}

bool Registry::load_raw(const std::string& key, std::vector<u8>& out) {
    return fetch_raw_(key, out, nullptr);
}

std::shared_ptr<TextureAsset> Registry::load_texture(const std::string& key) {
    auto it = tex_cache_.find(key);
    if (it != tex_cache_.end()) return it->second;
    std::vector<u8> bytes;
    cook::AssetType type;
    if (!fetch_raw_(key, bytes, &type)) return nullptr;
    if (type != cook::AssetType::Texture) return nullptr;
    auto a = std::make_shared<TextureAsset>();
    if (!codec::decode_texture(bytes, *a)) return nullptr;
    tex_cache_[key] = a;
    return a;
}

std::shared_ptr<MeshAsset> Registry::load_mesh(const std::string& key) {
    auto it = mesh_cache_.find(key);
    if (it != mesh_cache_.end()) return it->second;
    std::vector<u8> bytes;
    cook::AssetType type;
    if (!fetch_raw_(key, bytes, &type)) return nullptr;
    if (type != cook::AssetType::Mesh) return nullptr;
    auto a = std::make_shared<MeshAsset>();
    if (!codec::decode_mesh(bytes, *a)) return nullptr;
    mesh_cache_[key] = a;
    return a;
}

std::shared_ptr<ShaderAsset> Registry::load_shader(const std::string& key) {
    auto it = shader_cache_.find(key);
    if (it != shader_cache_.end()) return it->second;
    std::vector<u8> bytes;
    cook::AssetType type;
    if (!fetch_raw_(key, bytes, &type)) return nullptr;
    if (type != cook::AssetType::Shader) return nullptr;
    auto a = std::make_shared<ShaderAsset>();
    if (!codec::decode_shader(bytes, *a)) return nullptr;
    shader_cache_[key] = a;
    return a;
}

std::shared_ptr<MaterialAsset> Registry::load_material(const std::string& key) {
    auto it = mat_cache_.find(key);
    if (it != mat_cache_.end()) return it->second;
    std::vector<u8> bytes;
    cook::AssetType type;
    if (!fetch_raw_(key, bytes, &type)) return nullptr;
    if (type != cook::AssetType::Material) return nullptr;
    auto a = std::make_shared<MaterialAsset>();
    if (!codec::decode_material(bytes, *a)) return nullptr;
    mat_cache_[key] = a;
    return a;
}

void Registry::register_material(const std::string& key, MaterialAsset m) {
    mat_cache_[key] = std::make_shared<MaterialAsset>(std::move(m));
}

Registry::Stats Registry::stats() const noexcept {
    Stats s{};
    s.mount_dirs       = static_cast<u32>(dirs_.size());
    s.mount_archives   = static_cast<u32>(archives_.size());
    s.cached_textures  = static_cast<u32>(tex_cache_.size());
    s.cached_meshes    = static_cast<u32>(mesh_cache_.size());
    s.cached_shaders   = static_cast<u32>(shader_cache_.size());
    s.cached_materials = static_cast<u32>(mat_cache_.size());
    for (const auto& [_, t] : tex_cache_)    s.bytes_resident += t->rgba.size();
    for (const auto& [_, m] : mesh_cache_)   s.bytes_resident += m->vertices.size() * sizeof(MeshVertex) + m->indices.size() * 4;
    for (const auto& [_, sh] : shader_cache_) s.bytes_resident += sh->bytecode.size();
    return s;
}

}  // namespace cardinal::asset
