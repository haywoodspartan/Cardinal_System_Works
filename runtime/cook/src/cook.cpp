#include <cardinal/cook/cook.hpp>

#include <cardinal/core/log.hpp>
#include <cardinal/import/import.hpp>   // robust OBJ/glTF DCC ingest
#include <cardinal/edit/mesh_ops.hpp>
#include <cardinal/edit/tex_ops.hpp>
#include <cardinal/project/project.hpp>
#include <cardinal/scene/math.hpp>
#include <cardinal/scene/scene.hpp>           // for scene::Vertex (used by mesh_ops)
#include <cardinal/shader/shader.hpp>

#include <cardinal/core/algorithm.hpp>    // cardinal::transform
#include <cardinal/core/bit.hpp>          // cardinal::bit_cast
#include <cardinal/core/cctype.hpp>       // cardinal::tolower
#include <cardinal/core/cstdio.hpp>       // cardinal::sscanf
#include <cardinal/core/cstdlib.hpp>      // cardinal::strtoull
#include <cardinal/core/cstring.hpp>      // cardinal::memcmp
#include <cardinal/core/filesystem.hpp>   // cardinal::fs
#include <cardinal/core/fstream.hpp>      // cardinal::ifstream/ofstream
#include <cardinal/core/utility.hpp>      // cardinal::move
#include <cardinal/core/containers.hpp>   // cardinal::vector

namespace fs = cardinal::fs;

namespace cardinal::cook {

const char* asset_type_name(AssetType t) noexcept {
    switch (t) {
        case AssetType::Unknown:  return "Unknown";
        case AssetType::Texture:  return "Texture";
        case AssetType::Mesh:     return "Mesh";
        case AssetType::Shader:   return "Shader";
        case AssetType::Audio:    return "Audio";
        case AssetType::Material: return "Material";
    }
    return "?";
}

AssetType asset_type_for_extension(const cardinal::string& ext) noexcept {
    auto lower = ext;
    cardinal::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c){ return static_cast<char>(cardinal::tolower(c)); });
    if (lower == ".png" || lower == ".jpg" || lower == ".jpeg" || lower == ".tga" || lower == ".bmp")
        return AssetType::Texture;
    if (lower == ".obj" || lower == ".cardmesh" || lower == ".gltf" || lower == ".glb")
        return AssetType::Mesh;
    if (lower == ".hlsl" || lower == ".glsl" || lower == ".wgsl")
        return AssetType::Shader;
    if (lower == ".wav" || lower == ".ogg" || lower == ".flac")
        return AssetType::Audio;
    if (lower == ".material" || lower == ".mat")
        return AssetType::Material;
    return AssetType::Unknown;
}

// ---------------------------------------------------------------------------
// CookedAsset header
// ---------------------------------------------------------------------------
cardinal::vector<u8> CookedAsset::serialize() const {
    cardinal::vector<u8> out;
    out.reserve(kCookHeaderBytes + payload.size());
    out.insert(out.end(), kCookMagic, kCookMagic + 4);
    auto append_u32 = [&](u32 v) {
        out.push_back(static_cast<u8>(v >> 0));
        out.push_back(static_cast<u8>(v >> 8));
        out.push_back(static_cast<u8>(v >> 16));
        out.push_back(static_cast<u8>(v >> 24));
    };
    append_u32(static_cast<u32>(type));
    append_u32(static_cast<u32>(payload.size()));
    append_u32(cooker_version);
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

bool CookedAsset::deserialize(const cardinal::vector<u8>& bytes, CookedAsset& out) {
    if (bytes.size() < kCookHeaderBytes) return false;
    if (cardinal::memcmp(bytes.data(), kCookMagic, 4) != 0) return false;
    auto rd_u32 = [&](usize off) {
        return static_cast<u32>(bytes[off]) |
               (static_cast<u32>(bytes[off + 1]) << 8) |
               (static_cast<u32>(bytes[off + 2]) << 16) |
               (static_cast<u32>(bytes[off + 3]) << 24);
    };
    out.type           = static_cast<AssetType>(rd_u32(4));
    const u32 size     = rd_u32(8);
    out.cooker_version = rd_u32(12);
    // `size` is attacker-controlled; `kCookHeaderBytes + size` in u32
    // wraps for size near UINT32_MAX, slips past this guard, and the
    // payload.assign below walks far out of bounds. 64-bit arithmetic.
    if (static_cast<usize>(kCookHeaderBytes) + size > bytes.size()) return false;
    out.payload.assign(bytes.begin() + kCookHeaderBytes,
                       bytes.begin() + kCookHeaderBytes + size);
    return true;
}

// ---------------------------------------------------------------------------
// Built-in cookers
// ---------------------------------------------------------------------------
namespace {

// ---------------------------------------------------------------------------
// Inline codec — same byte format the runtime asset registry decodes. Kept
// here so cook is independent of the asset library (asset depends on cook,
// not the other way around).
// ---------------------------------------------------------------------------
struct TextureBlob {
    u32 width{0}, height{0}, channels{4};
    cardinal::vector<u8> rgba;
};
struct MeshVert {
    cardinal::scene::Vec3 position, normal, color;
};
struct MeshBlob {
    cardinal::vector<MeshVert> vertices;
    cardinal::vector<u32>      indices;
};
struct ShaderBlob {
    u32 stage{0};
    cardinal::string entry_point;
    cardinal::vector<u8> bytecode;
};

void wr_u32(cardinal::vector<u8>& o, u32 v) {
    o.push_back(static_cast<u8>(v));
    o.push_back(static_cast<u8>(v >> 8));
    o.push_back(static_cast<u8>(v >> 16));
    o.push_back(static_cast<u8>(v >> 24));
}
void wr_f(cardinal::vector<u8>& o, float v) {
    wr_u32(o, cardinal::bit_cast<u32>(v));
}
void wr_str(cardinal::vector<u8>& o, const cardinal::string& s) {
    wr_u32(o, static_cast<u32>(s.size()));
    o.insert(o.end(), s.begin(), s.end());
}

cardinal::vector<u8> encode_texture_blob(const TextureBlob& t) {
    cardinal::vector<u8> o;
    wr_u32(o, t.width); wr_u32(o, t.height); wr_u32(o, t.channels);
    wr_u32(o, static_cast<u32>(t.rgba.size()));
    o.insert(o.end(), t.rgba.begin(), t.rgba.end());
    return o;
}
cardinal::vector<u8> encode_mesh_blob(const MeshBlob& m) {
    cardinal::vector<u8> o;
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
cardinal::vector<u8> encode_shader_blob(const ShaderBlob& s) {
    cardinal::vector<u8> o;
    wr_u32(o, s.stage);
    wr_str(o, s.entry_point);
    wr_u32(o, static_cast<u32>(s.bytecode.size()));
    o.insert(o.end(), s.bytecode.begin(), s.bytecode.end());
    return o;
}

// Detect a few image formats by magic. For source files we don't fully decode
// (no libpng); instead we accept already-RGBA8 files OR fall back to a 4x4
// solid-magenta texture so the pipeline always produces something.
bool decode_png_magic(const cardinal::vector<u8>& bytes, u32& w, u32& h,
                     cardinal::vector<u8>& rgba)
{
    static const u8 sig[8] = {0x89,'P','N','G',0x0D,0x0A,0x1A,0x0A};
    if (bytes.size() < 24) return false;
    if (cardinal::memcmp(bytes.data(), sig, 8) != 0) return false;
    // PNG IHDR is at offset 16 (after sig + 4-byte length + 4-byte 'IHDR'):
    // width(4) height(4) bit_depth(1) color_type(1) ...
    auto rd_be_u32 = [&](usize off) {
        return (static_cast<u32>(bytes[off]) << 24) |
               (static_cast<u32>(bytes[off + 1]) << 16) |
               (static_cast<u32>(bytes[off + 2]) << 8)  |
                static_cast<u32>(bytes[off + 3]);
    };
    w = rd_be_u32(16);
    h = rd_be_u32(20);
    // Without zlib we can't decode the IDAT — emit a placeholder RGBA8 of
    // the correct size so the asset has the right dimensions.
    rgba.assign(static_cast<usize>(w) * h * 4, 0xC0);
    for (usize i = 3; i < rgba.size(); i += 4) rgba[i] = 0xFF;
    return true;
}

class TextureCooker final : public Cooker {
public:
    AssetType   asset_type()    const noexcept override { return AssetType::Texture; }
    u32         cooker_version() const noexcept override { return 2; }
    const char* name()          const noexcept override { return "Texture"; }
    CookedAsset cook(const cardinal::vector<u8>& src, const CookContext& ctx) override
    {
        CookedAsset c{};
        c.type = AssetType::Texture;
        c.cooker_version = cooker_version();

        TextureBlob t{};
        if (decode_png_magic(src, t.width, t.height, t.rgba)) {
            c.diagnostics = "decoded PNG (size + placeholder pixels)";
        } else {
            t.width = t.height = 4;
            t.channels = 4;
            t.rgba.assign(64, 0);
            for (usize i = 0; i < 16; ++i) {
                t.rgba[i * 4 + 0] = (i & 1) ? 0xFF : 0x40;
                t.rgba[i * 4 + 1] = 0x00;
                t.rgba[i * 4 + 2] = (i & 1) ? 0xFF : 0x40;
                t.rgba[i * 4 + 3] = 0xFF;
            }
            c.diagnostics = "no decoder for source format - emitted 4x4 magenta";
        }
        (void)ctx;

        c.payload = encode_texture_blob(t);
        return c;
    }
};

class MeshCooker final : public Cooker {
public:
    AssetType   asset_type()    const noexcept override { return AssetType::Mesh; }
    u32         cooker_version() const noexcept override { return 2; }
    const char* name()          const noexcept override { return "Mesh"; }
    CookedAsset cook(const cardinal::vector<u8>& src, const CookContext& ctx) override
    {
        CookedAsset c{};
        c.type = AssetType::Mesh;
        c.cooker_version = cooker_version();

        MeshBlob m{};
        bool parsed = false;

        // Robust DCC path: .obj / .gltf / .glb go through the
        // cardinal::import backend (Maya/Blender/Daz/Autodesk/UE export
        // these). import is core+scene only, so cook→import adds no
        // dependency cycle. All imported meshes are merged into one
        // cooked blob (index-offset concatenation).
        const auto ifmt = cardinal::import::detect_format(ctx.asset_path);
        if (ifmt == cardinal::import::Format::Obj  ||
            ifmt == cardinal::import::Format::Gltf ||
            ifmt == cardinal::import::Format::Glb) {
            cardinal::string ierr;
            cardinal::import::ImportScene is =
                cardinal::import::import_file(ctx.asset_path, &ierr);
            if (is.ok) {
                for (const auto& im : is.meshes) {
                    const u32 base = static_cast<u32>(m.vertices.size());
                    for (usize i = 0; i < im.positions.size(); ++i) {
                        MeshVert mv{};
                        mv.position = im.positions[i];
                        mv.normal   = (i < im.normals.size())
                            ? im.normals[i]
                            : cardinal::scene::Vec3{0, 1, 0};
                        mv.color    = (i < im.colors.size())
                            ? im.colors[i]
                            : cardinal::scene::Vec3{1, 1, 1};
                        m.vertices.push_back(mv);
                    }
                    for (u32 idx : im.indices)
                        m.indices.push_back(base + idx);
                }
                parsed = !m.vertices.empty();
                c.diagnostics = "imported " + is.source_format + " — " +
                                is.diagnostics;
            } else {
                c.diagnostics = "import failed: " + ierr;
            }
        }

        if (!parsed && !src.empty()) {
            cardinal::string text(src.begin(), src.end());
            if (text.compare(0, 11, "CARDMESH 1\n") == 0) {
                cardinal::vector<MeshVert> verts;
                cardinal::vector<u32> idx;
                usize p = 11;
                while (p < text.size()) {
                    const auto nl = text.find('\n', p);
                    const auto line = text.substr(p, (nl == cardinal::string::npos) ? text.size() - p : nl - p);
                    p = (nl == cardinal::string::npos) ? text.size() : nl + 1;
                    if (line.empty()) continue;
                    if (line[0] == 'v') {
                        MeshVert v{};
                        cardinal::sscanf(line.c_str() + 1, " %f %f %f %f %f %f %f %f %f",
                                    &v.position.x, &v.position.y, &v.position.z,
                                    &v.normal.x, &v.normal.y, &v.normal.z,
                                    &v.color.x, &v.color.y, &v.color.z);
                        verts.push_back(v);
                    } else if (line[0] == 'f') {
                        u32 a, b, ci;
                        if (cardinal::sscanf(line.c_str() + 1, " %u %u %u", &a, &b, &ci) == 3) {
                            idx.push_back(a); idx.push_back(b); idx.push_back(ci);
                        }
                    }
                }
                m.vertices = cardinal::move(verts);
                m.indices  = cardinal::move(idx);
                parsed = !m.vertices.empty();
            }
        }
        if (!parsed) {
            auto verts = cardinal::edit::mesh_ops::make_box(1.0f);
            m.vertices.reserve(verts.size());
            for (const auto& v : verts) {
                MeshVert mv{};
                mv.position = v.position;
                mv.normal   = v.normal;
                mv.color    = v.color;
                m.vertices.push_back(mv);
            }
            m.indices.resize(m.vertices.size());
            for (u32 i = 0; i < m.vertices.size(); ++i) m.indices[i] = i;
            c.diagnostics = "no parseable .cardmesh - emitted 1u cube";
        } else {
            c.diagnostics = "parsed .cardmesh";
        }

        c.payload = encode_mesh_blob(m);
        return c;
    }
};

class ShaderCooker final : public Cooker {
public:
    AssetType   asset_type()    const noexcept override { return AssetType::Shader; }
    u32         cooker_version() const noexcept override { return 2; }
    const char* name()          const noexcept override { return "Shader"; }
    CookedAsset cook(const cardinal::vector<u8>& src, const CookContext& ctx) override
    {
        CookedAsset c{};
        c.type = AssetType::Shader;
        c.cooker_version = cooker_version();
        ShaderBlob s{};
        s.entry_point = "VSMain";   // host can override post-cook
        s.stage = 0;
        // If a shader::Compiler is bound, we compile through it; otherwise
        // we just store the source bytes so the runtime can recompile.
        if (ctx.shader_compiler != nullptr) {
            cardinal::shader::CompileRequest r{};
            r.source_path = ctx.asset_path;
            r.source_text.assign(src.begin(), src.end());
            r.entry_point = "VSMain";
            r.stage = cardinal::shader::Stage::Vertex;
            auto cr = ctx.shader_compiler->compile(r);
            if (cr.ok) {
                s.bytecode = cardinal::move(cr.bytecode);
                c.diagnostics = "compiled VSMain (" +
                    cardinal::to_string(s.bytecode.size()) + " bytes" +
                    (cr.served_from_cache ? ", cached" : "") + ")";
            } else {
                c.diagnostics = "compile failed: " + cr.diagnostics;
                s.bytecode = cardinal::vector<u8>(src.begin(), src.end());
            }
        } else {
            s.bytecode.assign(src.begin(), src.end());
            c.diagnostics = "stored source (no compiler bound)";
        }
        c.payload = encode_shader_blob(s);
        return c;
    }
};

class AudioCooker final : public Cooker {
public:
    AssetType   asset_type()    const noexcept override { return AssetType::Audio; }
    u32         cooker_version() const noexcept override { return 1; }
    const char* name()          const noexcept override { return "Audio"; }
    CookedAsset cook(const cardinal::vector<u8>& src, const CookContext&) override
    {
        // Pass-through for now. Real cooker would resample, encode to OGG
        // for streaming + WAV for SFX, etc.
        CookedAsset c{};
        c.type = AssetType::Audio;
        c.cooker_version = cooker_version();
        c.payload = src;
        c.diagnostics = "passthrough";
        return c;
    }
};

}  // namespace

cardinal::unique_ptr<Cooker> make_texture_cooker() { return cardinal::make_unique<TextureCooker>(); }
cardinal::unique_ptr<Cooker> make_mesh_cooker()    { return cardinal::make_unique<MeshCooker>(); }
cardinal::unique_ptr<Cooker> make_shader_cooker()  { return cardinal::make_unique<ShaderCooker>(); }
cardinal::unique_ptr<Cooker> make_audio_cooker()   { return cardinal::make_unique<AudioCooker>(); }

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------
void CookerRegistry::register_cooker(const cardinal::string& ext,
                                     cardinal::shared_ptr<Cooker> cooker)
{
    by_ext_[ext] = cardinal::move(cooker);
}

Cooker* CookerRegistry::find_for_extension(const cardinal::string& ext) const {
    auto lower = ext;
    cardinal::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c){ return static_cast<char>(cardinal::tolower(c)); });
    auto it = by_ext_.find(lower);
    return (it == by_ext_.end()) ? nullptr : it->second.get();
}

void CookerRegistry::register_builtin(cardinal::shader::Compiler*) {
    auto tex = cardinal::shared_ptr<Cooker>(make_texture_cooker());
    auto msh = cardinal::shared_ptr<Cooker>(make_mesh_cooker());
    auto sh  = cardinal::shared_ptr<Cooker>(make_shader_cooker());
    auto au  = cardinal::shared_ptr<Cooker>(make_audio_cooker());

    register_cooker(".png",      tex);
    register_cooker(".jpg",      tex);
    register_cooker(".jpeg",     tex);
    register_cooker(".tga",      tex);
    register_cooker(".bmp",      tex);
    register_cooker(".obj",      msh);
    register_cooker(".cardmesh", msh);
    register_cooker(".gltf",     msh);
    register_cooker(".glb",      msh);
    register_cooker(".hlsl",     sh);
    register_cooker(".glsl",     sh);
    register_cooker(".wav",      au);
    register_cooker(".ogg",      au);
    register_cooker(".flac",     au);
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------
bool CookManifest::load_from(const cardinal::string& path) {
    hashes.clear();
    cardinal::ifstream f(path);
    if (!f) return false;
    cardinal::string line;
    while (cardinal::getline(f, line)) {
        const auto sp = line.find(' ');
        if (sp == cardinal::string::npos) continue;
        const cardinal::string p = line.substr(0, sp);
        const u64 h = cardinal::strtoull(line.c_str() + sp + 1, nullptr, 16);
        hashes[p] = h;
    }
    return true;
}

bool CookManifest::save_to(const cardinal::string& path) const {
    cardinal::ofstream f(path, cardinal::ios::binary | cardinal::ios::trunc);
    if (!f) return false;
    char buf[64];
    for (const auto& [p, h] : hashes) {
        cardinal::snprintf(buf, sizeof(buf), " %016llx\n", static_cast<unsigned long long>(h));
        f << p << buf;
    }
    return true;
}

namespace {

u64 hash_bytes(const cardinal::vector<u8>& bytes) noexcept {
    u64 h = 0xcbf29ce484222325ull;        // FNV-1a 64
    for (u8 b : bytes) { h ^= b; h *= 0x100000001b3ull; }
    return h;
}

bool read_all(const cardinal::string& path, cardinal::vector<u8>& out) {
    cardinal::ifstream f(path, cardinal::ios::binary);
    if (!f) return false;
    f.seekg(0, cardinal::ios::end);
    const cardinal::streamsize n = f.tellg();
    f.seekg(0, cardinal::ios::beg);
    out.resize(static_cast<usize>(n));
    f.read(reinterpret_cast<char*>(out.data()), n);
    return true;
}

bool write_all(const cardinal::string& path, const cardinal::vector<u8>& bytes) {
    cardinal::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    cardinal::ofstream f(path, cardinal::ios::binary | cardinal::ios::trunc);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<cardinal::streamsize>(bytes.size()));
    return true;
}

}  // namespace

CookSummary cook_all(const cardinal::project::Project& proj,
                     const CookerRegistry& registry,
                     cardinal::vector<CookResult>& results,
                     bool force,
                     CookContext template_ctx)
{
    CookSummary s{};
    results.clear();
    const auto sources = proj.list_source_assets();

    CookManifest manifest;
    const cardinal::string manifest_path = proj.dirs().cooked + "/manifest.cook";
    manifest.load_from(manifest_path);

    for (const auto& rel : sources) {
        CookResult r{};
        r.source_relpath = rel;
        const cardinal::string abs = proj.dirs().assets + "/" + rel;
        const cardinal::string ext = fs::path(rel).extension().string();
        Cooker* cooker = registry.find_for_extension(ext);
        if (cooker == nullptr) {
            r.status = CookResult::Status::Skipped;
            r.error_message = "no cooker for extension " + ext;
            results.push_back(cardinal::move(r));
            ++s.skipped_count;
            continue;
        }
        r.type = cooker->asset_type();

        cardinal::vector<u8> src_bytes;
        if (!read_all(abs, src_bytes)) {
            r.status = CookResult::Status::Failed;
            r.error_message = "could not read " + abs;
            results.push_back(cardinal::move(r));
            ++s.failed_count;
            continue;
        }
        const u64 hash = hash_bytes(src_bytes);
        r.source_hash = hash;
        const cardinal::string cooked_rel = rel + ".cooked";
        const cardinal::string cooked_abs = proj.dirs().cooked + "/" + cooked_rel;
        r.cooked_relpath = cooked_rel;

        if (!force) {
            auto it = manifest.hashes.find(rel);
            if (it != manifest.hashes.end() && it->second == hash &&
                fs::exists(cooked_abs))
            {
                r.status = CookResult::Status::Skipped;
                results.push_back(cardinal::move(r));
                ++s.skipped_count;
                continue;
            }
        }

        CookContext ctx = template_ctx;
        ctx.project_root = proj.dirs().root;
        ctx.asset_path   = abs;
        CookedAsset cooked = cooker->cook(src_bytes, ctx);
        r.diagnostics  = cooked.diagnostics;
        r.payload_bytes = static_cast<u32>(cooked.payload.size());

        const cardinal::vector<u8> blob = cooked.serialize();
        if (!write_all(cooked_abs, blob)) {
            r.status = CookResult::Status::Failed;
            r.error_message = "could not write " + cooked_abs;
            results.push_back(cardinal::move(r));
            ++s.failed_count;
            continue;
        }
        manifest.hashes[rel] = hash;
        r.status = CookResult::Status::Cooked;
        results.push_back(cardinal::move(r));
        ++s.cooked_count;
        s.total_payload_bytes += blob.size();
    }
    manifest.save_to(manifest_path);
    cardinal::log::infof("cook",
        "cook_all: %u cooked, %u skipped, %u failed (%llu bytes total)",
        s.cooked_count, s.skipped_count, s.failed_count,
        static_cast<unsigned long long>(s.total_payload_bytes));
    return s;
}

}  // namespace cardinal::cook
