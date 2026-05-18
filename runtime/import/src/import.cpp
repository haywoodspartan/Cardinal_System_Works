// =============================================================================
// Cardinal — DCC importer: OBJ/MTL backend + framework + asset convert.
//
// Wavefront OBJ is hand-rolled (zero-dep) and handles what the listed
// DCC tools actually emit: v/vn/vt, optional per-vertex colour, o/g
// grouping, usemtl, mtllib, negative + */vt//vn index forms, n-gon
// triangulation. Companion MTL → metallic-roughness PBR (Kd/Ke/Ns plus
// the Pr/Pm PBR extensions Blender/Substance write). glTF 2.0 is the
// next backend (stub here, dispatched the same way).
// =============================================================================

#include <cardinal/import/import.hpp>
#include <cardinal/core/log.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace cardinal::import {

using cardinal::scene::Vec3;

const char* format_name(Format f) noexcept {
    switch (f) {
        case Format::Obj:  return "obj";
        case Format::Gltf: return "gltf";
        case Format::Glb:  return "glb";
        case Format::Fbx:  return "fbx";
        default:           return "unknown";
    }
}

namespace {

std::string to_lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(
        static_cast<unsigned char>(c)));
    return s;
}
std::string ext_of(const std::string& path) {
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return {};
    return to_lower(path.substr(dot));
}
std::string dir_of(const std::string& path) {
    const auto s = path.find_last_of("/\\");
    return (s == std::string::npos) ? std::string{} : path.substr(0, s + 1);
}

}  // namespace

Format detect_format(const std::string& path) noexcept {
    const std::string e = ext_of(path);
    if (e == ".obj")  return Format::Obj;
    if (e == ".gltf") return Format::Gltf;
    if (e == ".glb")  return Format::Glb;
    if (e == ".fbx")  return Format::Fbx;
    return Format::Unknown;
}

u32 ImportScene::total_vertices() const noexcept {
    u32 n = 0;
    for (const auto& m : meshes) n += static_cast<u32>(m.positions.size());
    return n;
}
u32 ImportScene::total_triangles() const noexcept {
    u32 n = 0;
    for (const auto& m : meshes) n += static_cast<u32>(m.indices.size() / 3);
    return n;
}

// ---------------------------------------------------------------------------
// MTL — metallic-roughness PBR. Kd→base_color, Ke→emission, Ns→roughness
// (specular-exponent → α), with the Pr (roughness) / Pm (metallic) PBR
// extensions overriding when present. map_Kd → base_color_texture.
// ---------------------------------------------------------------------------
namespace {

void parse_mtl(const std::string& mtl_path,
               std::vector<ImportMaterial>& out,
               std::unordered_map<std::string, int>& by_name) {
    std::ifstream f(mtl_path, std::ios::binary);
    if (!f) return;
    ImportMaterial* cur = nullptr;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ls(line);
        std::string tok;
        if (!(ls >> tok) || tok.empty() || tok[0] == '#') continue;
        if (tok == "newmtl") {
            std::string nm; ls >> nm;
            ImportMaterial m; m.name = nm;
            by_name[nm] = static_cast<int>(out.size());
            out.push_back(m);
            cur = &out.back();
        } else if (cur == nullptr) {
            continue;
        } else if (tok == "Kd") {
            ls >> cur->base_color.x >> cur->base_color.y >> cur->base_color.z;
        } else if (tok == "Ke") {
            ls >> cur->emission.x >> cur->emission.y >> cur->emission.z;
            const float m = std::max({cur->emission.x, cur->emission.y,
                                      cur->emission.z});
            cur->emission_strength = (m > 0.0f) ? 1.0f : 0.0f;
        } else if (tok == "Ns") {
            float ns = 0.0f; ls >> ns;
            // Blinn-Phong exponent → GGX-ish roughness. Clamp off 0/1.
            const float r = std::sqrt(2.0f / (std::max(ns, 0.0f) + 2.0f));
            cur->roughness = std::min(1.0f, std::max(0.03f, r));
        } else if (tok == "Pr") {            // PBR ext: roughness
            ls >> cur->roughness;
        } else if (tok == "Pm") {            // PBR ext: metallic
            ls >> cur->metallic;
        } else if (tok == "map_Kd") {
            // Last whitespace token is the path (skip -options).
            std::string last, t;
            while (ls >> t) last = t;
            cur->base_color_texture = last;
        }
    }
}

// One face-vertex of an OBJ `f` token: 1-based v / vt / vn (0 = absent).
struct FaceRef { long v{0}, vt{0}, vn{0}; };

FaceRef parse_face_ref(const std::string& s) {
    FaceRef r;
    // forms: v   v/vt   v//vn   v/vt/vn
    long* slots[3] = { &r.v, &r.vt, &r.vn };
    int slot = 0;
    std::size_t i = 0;
    while (i <= s.size() && slot < 3) {
        std::size_t j = s.find('/', i);
        const std::string part = (j == std::string::npos)
            ? s.substr(i) : s.substr(i, j - i);
        if (!part.empty()) *slots[slot] = std::strtol(part.c_str(), nullptr, 10);
        ++slot;
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return r;
}

// Resolve a possibly-negative/1-based OBJ index to 0-based, or -1.
long resolve_idx(long idx, std::size_t count) {
    if (idx > 0)  return idx - 1;
    if (idx < 0)  return static_cast<long>(count) + idx;   // relative
    return -1;                                             // absent
}

// Per-output-mesh de-dup of (v,vt,vn) triples → compact vertex+index.
struct MeshBuilder {
    ImportMesh mesh;
    std::unordered_map<u64, u32> dedup;

    u32 emit(const FaceRef& fr,
             const std::vector<Vec3>& P, const std::vector<Vec3>& N,
             const std::vector<Vec3>& C, const std::vector<Vec2>& T) {
        const long vi = resolve_idx(fr.v,  P.size());
        const long ti = resolve_idx(fr.vt, T.size());
        const long ni = resolve_idx(fr.vn, N.size());
        const u64 key = (static_cast<u64>(vi + 1) * 0x9E3779B1u)
                      ^ (static_cast<u64>(ti + 1) << 21)
                      ^ (static_cast<u64>(ni + 1) << 42);
        auto it = dedup.find(key);
        if (it != dedup.end()) return it->second;
        const u32 out = static_cast<u32>(mesh.positions.size());
        mesh.positions.push_back(
            (vi >= 0 && vi < static_cast<long>(P.size())) ? P[vi] : Vec3{});
        if (ni >= 0 && ni < static_cast<long>(N.size()))
            mesh.normals.push_back(N[ni]);
        if (!C.empty())
            mesh.colors.push_back(
                (vi >= 0 && vi < static_cast<long>(C.size()))
                    ? C[vi] : Vec3{1, 1, 1});
        if (ti >= 0 && ti < static_cast<long>(T.size()))
            mesh.uvs.push_back(T[ti]);
        dedup.emplace(key, out);
        return out;
    }
};

}  // namespace

ImportScene import_obj(const std::string& path, std::string* error) {
    ImportScene scene;
    scene.source_format = "obj";

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        if (error) *error = "cannot open '" + path + "'";
        scene.diagnostics = "OBJ: open failed";
        return scene;
    }
    const std::string base = dir_of(path);

    std::vector<Vec3> P, N, C;     // C parallel to P (per-vertex colour)
    std::vector<Vec2> T;
    bool have_colors = false;

    std::unordered_map<std::string, int> mtl_by_name;
    auto flush = [&](MeshBuilder& mb) {
        if (!mb.mesh.positions.empty() && !mb.mesh.indices.empty())
            scene.meshes.push_back(std::move(mb.mesh));
        mb = MeshBuilder{};
    };
    MeshBuilder mb;
    int  cur_mat   = -1;
    bool mesh_open = false;
    std::string cur_name = "mesh";

    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ls(line);
        std::string tok;
        if (!(ls >> tok) || tok.empty() || tok[0] == '#') continue;

        if (tok == "v") {
            Vec3 p; ls >> p.x >> p.y >> p.z;
            P.push_back(p);
            Vec3 c{1, 1, 1};
            if (ls >> c.x >> c.y >> c.z) have_colors = true;
            C.push_back(c);
        } else if (tok == "vn") {
            Vec3 n; ls >> n.x >> n.y >> n.z; N.push_back(n);
        } else if (tok == "vt") {
            Vec2 t; ls >> t.u >> t.v; T.push_back(t);
        } else if (tok == "o" || tok == "g") {
            if (mesh_open) flush(mb);
            ls >> cur_name;
            mb.mesh.name     = cur_name;
            mb.mesh.material = cur_mat;
            mesh_open = true;
        } else if (tok == "mtllib") {
            std::string mp; ls >> mp;
            parse_mtl(base + mp, scene.materials, mtl_by_name);
        } else if (tok == "usemtl") {
            std::string mn; ls >> mn;
            auto it = mtl_by_name.find(mn);
            cur_mat = (it != mtl_by_name.end()) ? it->second : -1;
            // Material change mid-group → split (one material per mesh).
            if (mesh_open && !mb.mesh.indices.empty()) {
                flush(mb);
                mb.mesh.name = cur_name;
                mesh_open = true;
            }
            mb.mesh.material = cur_mat;
        } else if (tok == "f") {
            if (!mesh_open) { mb.mesh.name = cur_name;
                              mb.mesh.material = cur_mat; mesh_open = true; }
            std::vector<FaceRef> fr;
            std::string vtok;
            while (ls >> vtok) fr.push_back(parse_face_ref(vtok));
            // Fan-triangulate the polygon.
            for (std::size_t i = 2; i < fr.size(); ++i) {
                mb.mesh.indices.push_back(mb.emit(fr[0],     P, N, C, T));
                mb.mesh.indices.push_back(mb.emit(fr[i - 1], P, N, C, T));
                mb.mesh.indices.push_back(mb.emit(fr[i],     P, N, C, T));
            }
        }
    }
    if (mesh_open) flush(mb);

    if (!have_colors)
        for (auto& m : scene.meshes) m.colors.clear();

    if (scene.meshes.empty()) {
        if (error) *error = "OBJ contained no triangulated geometry";
        scene.diagnostics = "OBJ: parsed but produced 0 meshes";
        return scene;
    }
    // Node per mesh (flat scene — OBJ has no hierarchy).
    for (std::size_t i = 0; i < scene.meshes.size(); ++i) {
        ImportNode nd;
        nd.name = scene.meshes[i].name;
        nd.meshes.push_back(static_cast<int>(i));
        scene.roots.push_back(static_cast<int>(scene.nodes.size()));
        scene.nodes.push_back(std::move(nd));
    }
    scene.ok = true;
    std::ostringstream d;
    d << "OBJ: " << scene.meshes.size() << " mesh(es), "
      << scene.materials.size() << " material(s), "
      << scene.total_vertices() << " verts, "
      << scene.total_triangles() << " tris";
    scene.diagnostics = d.str();
    cardinal::log::infof("import", "%s — %s", path.c_str(),
                         scene.diagnostics.c_str());
    return scene;
}

// import_gltf() is implemented in import_gltf.cpp (glTF 2.0 / GLB).

ImportScene import_file(const std::string& path, std::string* error) {
    switch (detect_format(path)) {
        case Format::Obj:  return import_obj(path, error);
        case Format::Gltf:
        case Format::Glb:  return import_gltf(path, error);
        case Format::Fbx: {
            ImportScene s;
            s.source_format = "fbx";
            s.diagnostics   = "FBX backend not implemented yet";
            if (error) *error = s.diagnostics;
            return s;
        }
        default: {
            ImportScene s;
            s.diagnostics = "unrecognised extension: " + path;
            if (error) *error = s.diagnostics;
            return s;
        }
    }
}

// ImportScene → engine asset-struct conversion moved to the header-only
// bridge <cardinal/import/to_asset.hpp> so this library links neither
// cardinal::asset nor (transitively) cook — letting cook delegate to
// the importer without a dependency cycle.

}  // namespace cardinal::import
