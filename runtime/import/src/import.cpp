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
#include <cardinal/core/diag/log.hpp>

#include <cardinal/core/std/algorithm.hpp>    // cardinal::min/max/sort
#include <cardinal/core/std/cctype.hpp>       // cardinal::tolower
#include <cardinal/core/std/cmath.hpp>        // cardinal::sqrt
#include <cardinal/core/std/cstdlib.hpp>      // cardinal::strtol
#include <cardinal/core/std/cstring.hpp>      // cardinal raw byte ops
#include <cardinal/core/std/fstream.hpp>      // cardinal::ifstream/ios/getline
#include <cardinal/core/std/sstream.hpp>      // cardinal::istringstream/ostringstream
#include <cardinal/core/std/utility.hpp>      // cardinal::move
#include <cardinal/core/std/containers.hpp>   // cardinal::unordered_map/vector

namespace cardinal::import {

using cardinal::scene::Vec3;

const char* format_name(Format f) noexcept {
    switch (f) {
        case Format::Obj:  return "obj";
        case Format::Gltf: return "gltf";
        case Format::Glb:  return "glb";
        case Format::Fbx:  return "fbx";
        case Format::Usda: return "usda";
        case Format::Usdz: return "usdz";
        case Format::Usdc: return "usdc";
        default:           return "unknown";
    }
}

namespace {

cardinal::string to_lower(cardinal::string s) {
    for (char& c : s) c = static_cast<char>(cardinal::tolower(
        static_cast<unsigned char>(c)));
    return s;
}
cardinal::string ext_of(const cardinal::string& path) {
    const auto dot = path.find_last_of('.');
    if (dot == cardinal::string::npos) return {};
    return to_lower(path.substr(dot));
}
cardinal::string dir_of(const cardinal::string& path) {
    const auto s = path.find_last_of("/\\");
    return (s == cardinal::string::npos) ? cardinal::string{} : path.substr(0, s + 1);
}

}  // namespace

Format detect_format(const cardinal::string& path) noexcept {
    const cardinal::string e = ext_of(path);
    if (e == ".obj")  return Format::Obj;
    if (e == ".gltf") return Format::Gltf;
    if (e == ".glb")  return Format::Glb;
    if (e == ".fbx")  return Format::Fbx;
    if (e == ".usda" || e == ".usd") return Format::Usda;  // .usd: ASCII or crate; sniffed in import_usd
    if (e == ".usdz") return Format::Usdz;
    if (e == ".usdc") return Format::Usdc;
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

// Last whitespace token on an MTL map line is the path; earlier tokens are
// `-bm`/`-o`/… options we skip (same rule for every map_*).
cardinal::string mtl_last_token(cardinal::istringstream& ls) {
    cardinal::string last, t;
    while (ls >> t) last = t;
    return last;
}

void parse_mtl(const cardinal::string& mtl_path,
               cardinal::vector<ImportMaterial>& out,
               cardinal::unordered_map<cardinal::string, int>& by_name) {
    cardinal::ifstream f(mtl_path, cardinal::ios::binary);
    if (!f) return;
    ImportMaterial* cur = nullptr;
    cardinal::string line;
    while (cardinal::getline(f, line)) {
        cardinal::istringstream ls(line);
        cardinal::string tok;
        if (!(ls >> tok) || tok.empty() || tok[0] == '#') continue;
        if (tok == "newmtl") {
            cardinal::string nm; ls >> nm;
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
            const float m = cardinal::max({cur->emission.x, cur->emission.y,
                                      cur->emission.z});
            cur->emission_strength = (m > 0.0f) ? 1.0f : 0.0f;
        } else if (tok == "Ns") {
            float ns = 0.0f; ls >> ns;
            // Blinn-Phong exponent → GGX-ish roughness. Clamp off 0/1.
            const float r = cardinal::sqrt(2.0f / (cardinal::max(ns, 0.0f) + 2.0f));
            cur->roughness = cardinal::min(1.0f, cardinal::max(0.03f, r));
        } else if (tok == "Pr") {            // PBR ext: roughness
            ls >> cur->roughness;
        } else if (tok == "Pm") {            // PBR ext: metallic
            ls >> cur->metallic;
        } else if (tok == "map_Kd" || tok == "map_kd") {
            cur->base_color_texture = mtl_last_token(ls);
        } else if (tok == "map_Pr" || tok == "map_pr") {        // PBR roughness
            cur->roughness_texture = mtl_last_token(ls);
        } else if (tok == "map_Pm" || tok == "map_pm") {        // PBR metallic
            cur->metallic_texture = mtl_last_token(ls);
        } else if (tok == "map_Bump" || tok == "map_bump" ||
                   tok == "bump"     || tok == "norm") {        // normal/bump
            cur->normal_texture = mtl_last_token(ls);
        } else if (tok == "map_Ke" || tok == "map_ke") {        // emissive
            cur->emissive_texture = mtl_last_token(ls);
        } else if (tok == "disp" || tok == "map_disp") {        // displacement
            cur->height_texture = mtl_last_token(ls);
        } else if (tok == "map_d") {                            // alpha/opacity
            cur->opacity_texture = mtl_last_token(ls);
        } else if (tok == "map_Ka" || tok == "map_ka") {        // ambient/AO-ish
            cur->occlusion_texture = mtl_last_token(ls);
        }
    }
}

// One face-vertex of an OBJ `f` token: 1-based v / vt / vn (0 = absent).
struct FaceRef { long v{0}, vt{0}, vn{0}; };

FaceRef parse_face_ref(const cardinal::string& s) {
    FaceRef r;
    // forms: v   v/vt   v//vn   v/vt/vn
    long* slots[3] = { &r.v, &r.vt, &r.vn };
    int slot = 0;
    cardinal::size_t i = 0;
    while (i <= s.size() && slot < 3) {
        cardinal::size_t j = s.find('/', i);
        const cardinal::string part = (j == cardinal::string::npos)
            ? s.substr(i) : s.substr(i, j - i);
        if (!part.empty()) *slots[slot] = cardinal::strtol(part.c_str(), nullptr, 10);
        ++slot;
        if (j == cardinal::string::npos) break;
        i = j + 1;
    }
    return r;
}

// Resolve a possibly-negative/1-based OBJ index to 0-based, or -1.
long resolve_idx(long idx, cardinal::size_t count) {
    if (idx > 0)  return idx - 1;
    if (idx < 0)  return static_cast<long>(count) + idx;   // relative
    return -1;                                             // absent
}

// Per-output-mesh de-dup of (v,vt,vn) triples → compact vertex+index.
struct MeshBuilder {
    ImportMesh mesh;
    cardinal::unordered_map<u64, u32> dedup;
    bool any_normal{false};
    bool any_uv{false};

    u32 emit(const FaceRef& fr,
             const cardinal::vector<Vec3>& P, const cardinal::vector<Vec3>& N,
             const cardinal::vector<Vec3>& C, const cardinal::vector<Vec2>& T) {
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
        // normals/uvs MUST stay strictly parallel to positions — one slot
        // per emitted vertex. The old per-vertex CONDITIONAL push desynced
        // the SoA: an OBJ that mixes `v//vn` and bare-`v` face-vertices (or
        // mixed `v/vt`) left normals[i] pointing at a DIFFERENT vertex than
        // positions[i], and the array shorter than positions, silently
        // corrupting every normal/uv after the first gap. Default-fill the
        // gap here; finalize() drops the whole array if the mesh carried no
        // real normal/uv at all (keeps the documented "empty ⇒ flat-shade /
        // no UVs" contract).
        const bool n_ok = (ni >= 0 && ni < static_cast<long>(N.size()));
        mesh.normals.push_back(n_ok ? N[ni] : Vec3{0, 1, 0});
        any_normal = any_normal || n_ok;
        if (!C.empty())
            mesh.colors.push_back(
                (vi >= 0 && vi < static_cast<long>(C.size()))
                    ? C[vi] : Vec3{1, 1, 1});
        const bool t_ok = (ti >= 0 && ti < static_cast<long>(T.size()));
        mesh.uvs.push_back(t_ok ? T[ti] : Vec2{0.0f, 0.0f});
        any_uv = any_uv || t_ok;
        dedup.emplace(key, out);
        return out;
    }

    // Restore the "empty ⇒ absent" contract: if NO face-vertex carried a
    // real normal / uv, drop the all-default parallel array so downstream
    // still flat-shades / treats UVs as none.
    void finalize() {
        if (!any_normal) mesh.normals.clear();
        if (!any_uv)     mesh.uvs.clear();
    }
};

}  // namespace

ImportScene import_obj(const cardinal::string& path, cardinal::string* error) {
    ImportScene scene;
    scene.source_format = "obj";

    cardinal::ifstream f(path, cardinal::ios::binary);
    if (!f) {
        if (error) *error = "cannot open '" + path + "'";
        scene.diagnostics = "OBJ: open failed";
        return scene;
    }
    const cardinal::string base = dir_of(path);

    cardinal::vector<Vec3> P, N, C;     // C parallel to P (per-vertex colour)
    cardinal::vector<Vec2> T;
    bool have_colors = false;

    cardinal::unordered_map<cardinal::string, int> mtl_by_name;
    auto flush = [&](MeshBuilder& mb) {
        mb.finalize();
        if (!mb.mesh.positions.empty() && !mb.mesh.indices.empty())
            scene.meshes.push_back(cardinal::move(mb.mesh));
        mb = MeshBuilder{};
    };
    MeshBuilder mb;
    int  cur_mat   = -1;
    bool mesh_open = false;
    cardinal::string cur_name = "mesh";

    cardinal::string line;
    while (cardinal::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        cardinal::istringstream ls(line);
        cardinal::string tok;
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
            cardinal::string mp; ls >> mp;
            parse_mtl(base + mp, scene.materials, mtl_by_name);
        } else if (tok == "usemtl") {
            cardinal::string mn; ls >> mn;
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
            cardinal::vector<FaceRef> fr;
            cardinal::string vtok;
            while (ls >> vtok) fr.push_back(parse_face_ref(vtok));
            // Fan-triangulate the polygon.
            for (cardinal::size_t i = 2; i < fr.size(); ++i) {
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
    for (cardinal::size_t i = 0; i < scene.meshes.size(); ++i) {
        ImportNode nd;
        nd.name = scene.meshes[i].name;
        nd.meshes.push_back(static_cast<int>(i));
        scene.roots.push_back(static_cast<int>(scene.nodes.size()));
        scene.nodes.push_back(cardinal::move(nd));
    }
    scene.ok = true;
    cardinal::ostringstream d;
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

ImportScene import_file(const cardinal::string& path, cardinal::string* error) {
    switch (detect_format(path)) {
        case Format::Obj:  return import_obj(path, error);
        case Format::Gltf:
        case Format::Glb:  return import_gltf(path, error);
        case Format::Fbx:  return import_fbx(path, error);
        case Format::Usda:
        case Format::Usdz:
        case Format::Usdc: return import_usd(path, error);
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
