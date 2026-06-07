// =============================================================================
// Cardinal — deterministic asset-import regression suite.
//
// The OBJ/MTL + glTF backends are hand-rolled, parse-heavy, and shipped
// without a single test — precisely the code that regresses silently
// (an index-base off-by-one, a dropped attribute, an MTL field swap).
// This locks the observable contract of import_file / import_obj /
// import_gltf with ZERO test deps (same ethos as the net suite).
//
// Fixtures are written to a temp dir at runtime, so the test is fully
// self-contained and deterministic — no fixture files to ship, no
// working-directory path fragility, no sockets, no timing. The harness
// is the same ~20-line CHECK used elsewhere. Exit 0 = all pass.
// =============================================================================

#include <cardinal/import/import.hpp>
#include <cardinal/core/diag/log.hpp>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {

namespace imp = cardinal::import;
using imp::Format;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("imptest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool approx(float a, float b, float eps) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= eps;
}
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

void write_file(const std::filesystem::path& p, const cardinal::string& s) {
    std::ofstream f(p, std::ios::binary);
    f.write(s.data(), static_cast<std::streamsize>(s.size()));
}

// Minimal standard base64 (for the glTF embedded-buffer data-URI).
cardinal::string b64(const std::vector<unsigned char>& in) {
    static const char* T =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    cardinal::string out;
    cardinal::usize i = 0;
    for (; i + 2 < in.size(); i += 3) {
        const unsigned v = (static_cast<unsigned>(in[i]) << 16) |
                           (static_cast<unsigned>(in[i + 1]) << 8) |
                            static_cast<unsigned>(in[i + 2]);
        out += T[(v >> 18) & 63]; out += T[(v >> 12) & 63];
        out += T[(v >> 6) & 63];  out += T[v & 63];
    }
    const cardinal::usize rem = in.size() - i;
    if (rem == 1) {
        const unsigned v = static_cast<unsigned>(in[i]) << 16;
        out += T[(v >> 18) & 63]; out += T[(v >> 12) & 63];
        out += "==";
    } else if (rem == 2) {
        const unsigned v = (static_cast<unsigned>(in[i]) << 16) |
                           (static_cast<unsigned>(in[i + 1]) << 8);
        out += T[(v >> 18) & 63]; out += T[(v >> 12) & 63];
        out += T[(v >> 6) & 63];  out += "=";
    }
    return out;
}

cardinal::string str_of(const std::filesystem::path& p) {
    return p.string();
}

// import_megascans resolves map uris to asset-dir-prefixed (absolute) paths,
// so fixtures assert on the trailing filename rather than the bare uri.
bool endswith(const cardinal::string& s, const char* suffix) {
    const cardinal::string t = suffix;
    return s.size() >= t.size() &&
           s.compare(s.size() - t.size(), t.size(), t) == 0;
}

// A fresh per-asset subdirectory so the Megascans filename-scan fallback only
// ever sees its OWN fixture files (tests stay isolated + deterministic).
std::filesystem::path subdir(const std::filesystem::path& base, const char* name) {
    std::error_code ec;
    const auto d = base / name;
    std::filesystem::create_directories(d, ec);
    return d;
}

// Write raw bytes (for synthetic binary heightmap fixtures).
void write_bytes(const std::filesystem::path& p,
                 const std::vector<unsigned char>& bytes) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
}

// ---- OBJ: triangle via import_file (also covers extension routing) --
void test_obj_triangle(const std::filesystem::path& dir) {
    const auto p = dir / "triangle.obj";
    write_file(p,
        "# tri\n"
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "vn 0 0 1\n"
        "vt 0 0\nvt 1 0\nvt 0 1\n"
        "f 1/1/1 2/2/1 3/3/1\n");

    cardinal::string err;
    imp::ImportScene s = imp::import_file(str_of(p), &err);
    CHECK(s.ok);
    CHECK(s.source_format == "obj");
    CHECK(s.meshes.size() == sz(1));
    if (!s.meshes.empty()) {
        const imp::ImportMesh& m = s.meshes[0];
        CHECK(m.positions.size() == sz(3));
        CHECK(m.indices.size() == sz(3));
        CHECK(m.normals.size() == sz(3));
        CHECK(m.uvs.size() == sz(3));
        CHECK(m.colors.empty());          // no per-vertex colour given
        CHECK(m.material == -1);          // no mtl
    }
    CHECK(s.total_vertices() == 3u);
    CHECK(s.total_triangles() == 1u);
    CHECK(s.nodes.size() == sz(1));
    CHECK(s.roots.size() == sz(1));
}

// ---- OBJ: n-gon fan triangulation -----------------------------------
void test_obj_quad(const std::filesystem::path& dir) {
    const auto p = dir / "quad.obj";
    write_file(p,
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "f 1 2 3 4\n");

    imp::ImportScene s = imp::import_obj(str_of(p));
    CHECK(s.ok);
    CHECK(s.meshes.size() == sz(1));
    if (!s.meshes.empty()) {
        CHECK(s.meshes[0].positions.size() == sz(4));   // 4 unique verts
        CHECK(s.meshes[0].indices.size() == sz(6));      // 2 fan triangles
    }
    CHECK(s.total_triangles() == 2u);
    CHECK(s.total_vertices() == 4u);
}

// ---- OBJ: negative (relative) indices resolve correctly -------------
void test_obj_negative(const std::filesystem::path& dir) {
    const auto p = dir / "neg.obj";
    write_file(p,
        "v 2 0 0\nv 0 2 0\nv 0 0 2\n"
        "f -3 -2 -1\n");

    imp::ImportScene s = imp::import_obj(str_of(p));
    CHECK(s.ok);
    CHECK(s.meshes.size() == sz(1));
    if (!s.meshes.empty() && s.meshes[0].positions.size() == sz(3)) {
        const auto& P = s.meshes[0].positions;
        CHECK(approx(P[0].x, 2.0f, 1e-6f) && approx(P[0].y, 0.0f, 1e-6f) &&
              approx(P[0].z, 0.0f, 1e-6f));
        CHECK(approx(P[1].x, 0.0f, 1e-6f) && approx(P[1].y, 2.0f, 1e-6f));
        CHECK(approx(P[2].z, 2.0f, 1e-6f));
        CHECK(s.meshes[0].indices.size() == sz(3));
    }
}

// ---- OBJ: mixed per-vertex normal/uv presence stays parallel --------
// Regression: MeshBuilder pushed positions every emit but normals/uvs
// only when THAT face-vertex carried a vn/vt. A face mixing `v//vn`,
// `v/vt` and bare `v` thus desynced the SoA — normals[i] no longer
// matched positions[i] and the array was short — silently corrupting
// every normal/uv after the first gap. Contract: normals/uvs are
// either empty (no attribute anywhere) or exactly parallel to
// positions, gaps default-filled (+Y normal / (0,0) uv).
void test_obj_mixed_normals(const std::filesystem::path& dir) {
    const auto p = dir / "mixed.obj";
    write_file(p,
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "vn 1 0 0\nvn 0 0 1\n"
        "vt 0.25 0.75\n"
        "f 1//1 2/1 3//2\n");          // fv0: vn only, fv1: vt only, fv2: vn only

    imp::ImportScene s = imp::import_obj(str_of(p));
    CHECK(s.ok);
    CHECK(s.meshes.size() == sz(1));
    if (!s.meshes.empty()) {
        const imp::ImportMesh& m = s.meshes[0];
        CHECK(m.positions.size() == sz(3));
        CHECK(m.indices.size()   == sz(3));
        // The core invariant: parallel to positions, not short/scrambled.
        CHECK(m.normals.size() == m.positions.size());
        CHECK(m.uvs.size()     == m.positions.size());
        if (m.normals.size() == sz(3)) {
            CHECK(approx(m.normals[0].x, 1.0f, 1e-6f) &&   // fv0 → vn 1 (+X)
                  approx(m.normals[0].y, 0.0f, 1e-6f) &&
                  approx(m.normals[0].z, 0.0f, 1e-6f));
            CHECK(approx(m.normals[1].x, 0.0f, 1e-6f) &&   // fv1 gap → +Y dflt
                  approx(m.normals[1].y, 1.0f, 1e-6f) &&
                  approx(m.normals[1].z, 0.0f, 1e-6f));
            CHECK(approx(m.normals[2].x, 0.0f, 1e-6f) &&   // fv2 → vn 2 (+Z)
                  approx(m.normals[2].y, 0.0f, 1e-6f) &&
                  approx(m.normals[2].z, 1.0f, 1e-6f));
        }
        if (m.uvs.size() == sz(3)) {
            CHECK(approx(m.uvs[0].u, 0.0f, 1e-6f) &&        // fv0 gap → (0,0)
                  approx(m.uvs[0].v, 0.0f, 1e-6f));
            CHECK(approx(m.uvs[1].u, 0.25f, 1e-6f) &&       // fv1 → vt 1
                  approx(m.uvs[1].v, 0.75f, 1e-6f));
            CHECK(approx(m.uvs[2].u, 0.0f, 1e-6f) &&        // fv2 gap → (0,0)
                  approx(m.uvs[2].v, 0.0f, 1e-6f));
        }
    }
    CHECK(s.total_vertices()  == 3u);
    CHECK(s.total_triangles() == 1u);
}

// ---- OBJ + MTL: metallic-roughness PBR binding ----------------------
void test_obj_mtl(const std::filesystem::path& dir) {
    write_file(dir / "mat.mtl",
        "newmtl Red\n"
        "Kd 0.9 0.1 0.05\n"
        "Ke 0 0 0\n"
        "Pr 0.25\n"
        "Pm 0.80\n"
        "map_Kd albedo.png\n"
        "map_Pr rough.png\n"
        "map_Pm metal.png\n"
        "map_Bump -bm 0.5 normal.png\n"     // -bm option skipped, path is last
        "map_Ke emis.png\n"
        "disp height.png\n");
    const auto p = dir / "matobj.obj";
    write_file(p,
        "mtllib mat.mtl\n"
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "usemtl Red\n"
        "f 1 2 3\n");

    imp::ImportScene s = imp::import_file(str_of(p));
    CHECK(s.ok);
    CHECK(s.materials.size() == sz(1));
    if (!s.materials.empty()) {
        const imp::ImportMaterial& mt = s.materials[0];
        CHECK(mt.name == "Red");
        CHECK(approx(mt.base_color.x, 0.9f, 1e-3f));
        CHECK(approx(mt.base_color.y, 0.1f, 1e-3f));
        CHECK(approx(mt.base_color.z, 0.05f, 1e-3f));
        CHECK(approx(mt.roughness, 0.25f, 1e-3f));    // Pr overrides Ns
        CHECK(approx(mt.metallic, 0.80f, 1e-3f));     // Pm ext
        CHECK(approx(mt.emission_strength, 0.0f, 1e-6f));
        // Full PBR map set parsed from MTL (separate maps, not packed).
        CHECK(mt.base_color_texture == "albedo.png");
        CHECK(mt.roughness_texture  == "rough.png");
        CHECK(mt.metallic_texture   == "metal.png");
        CHECK(mt.normal_texture     == "normal.png");
        CHECK(mt.emissive_texture   == "emis.png");
        CHECK(mt.height_texture     == "height.png");
        CHECK(!mt.mr_packed);
    }
    if (!s.meshes.empty()) CHECK(s.meshes[0].material == 0);
}

// ---- glTF 2.0: full PBR texture set (MR-packed + normal/occl/emissive) ---
void test_gltf_pbr_maps(const std::filesystem::path& dir) {
    const float pos[9] = { 0,0,0, 1,0,0, 0,1,0 };
    std::vector<unsigned char> bytes(sizeof(pos));
    std::memcpy(bytes.data(), pos, sizeof(pos));
    const cardinal::string uri =
        "data:application/octet-stream;base64," + b64(bytes);

    const cardinal::string json =
        cardinal::string("{\"asset\":{\"version\":\"2.0\"},")
        + "\"images\":[{\"uri\":\"alb.png\"},{\"uri\":\"mr.png\"},"
          "{\"uri\":\"nrm.png\"},{\"uri\":\"ao.png\"},{\"uri\":\"emi.png\"}],"
          "\"textures\":[{\"source\":0},{\"source\":1},{\"source\":2},"
          "{\"source\":3},{\"source\":4}],"
          "\"materials\":[{\"name\":\"PBR\",\"pbrMetallicRoughness\":{"
          "\"baseColorTexture\":{\"index\":0},"
          "\"metallicRoughnessTexture\":{\"index\":1}},"
          "\"normalTexture\":{\"index\":2,\"scale\":0.7},"
          "\"occlusionTexture\":{\"index\":3,\"strength\":0.4},"
          "\"emissiveTexture\":{\"index\":4},\"emissiveFactor\":[1,1,1]}],"
          "\"buffers\":[{\"byteLength\":36,\"uri\":\"" + uri + "\"}],"
          "\"bufferViews\":[{\"buffer\":0,\"byteOffset\":0,\"byteLength\":36}],"
          "\"accessors\":[{\"bufferView\":0,\"byteOffset\":0,"
          "\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"}],"
          "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0},"
          "\"material\":0}]}],"
          "\"nodes\":[{\"mesh\":0}],"
          "\"scenes\":[{\"nodes\":[0]}],\"scene\":0}";

    const auto p = dir / "pbr.gltf";
    write_file(p, json);
    cardinal::string err;
    imp::ImportScene s = imp::import_file(str_of(p), &err);
    CHECK(s.ok);
    CHECK(s.materials.size() == sz(1));
    if (!s.materials.empty()) {
        const imp::ImportMaterial& mt = s.materials[0];
        CHECK(mt.name == "PBR");
        CHECK(mt.base_color_texture == "alb.png");
        CHECK(mt.metallic_roughness_texture == "mr.png");
        CHECK(mt.mr_packed);                          // glTF packs MR
        CHECK(mt.normal_texture == "nrm.png");
        CHECK(approx(mt.normal_scale, 0.7f, 1e-5f));
        CHECK(mt.occlusion_texture == "ao.png");
        CHECK(approx(mt.occlusion_strength, 0.4f, 1e-5f));
        CHECK(mt.emissive_texture == "emi.png");
    }
}

// ---- OBJ: per-vertex colour is preserved ----------------------------
void test_obj_vertex_color(const std::filesystem::path& dir) {
    const auto p = dir / "colored.obj";
    write_file(p,
        "v 0 0 0 1 0 0\n"
        "v 1 0 0 0 1 0\n"
        "v 0 1 0 0 0 1\n"
        "f 1 2 3\n");

    imp::ImportScene s = imp::import_obj(str_of(p));
    CHECK(s.ok);
    if (!s.meshes.empty()) {
        const imp::ImportMesh& m = s.meshes[0];
        CHECK(m.colors.size() == sz(3));
        if (m.colors.size() == sz(3)) {
            CHECK(approx(m.colors[0].x, 1.0f, 1e-3f));
            CHECK(approx(m.colors[1].y, 1.0f, 1e-3f));
            CHECK(approx(m.colors[2].z, 1.0f, 1e-3f));
        }
    }
}

// ---- glTF 2.0: minimal embedded-buffer triangle, end to end ---------
void test_gltf_triangle(const std::filesystem::path& dir) {
    const float pos[9] = { 0.0f, 0.0f, 0.0f,
                           1.0f, 0.0f, 0.0f,
                           0.0f, 1.0f, 0.0f };
    std::vector<unsigned char> bytes(sizeof(pos));
    std::memcpy(bytes.data(), pos, sizeof(pos));
    const cardinal::string uri =
        "data:application/octet-stream;base64," + b64(bytes);

    cardinal::string json =
        "{\"asset\":{\"version\":\"2.0\"},"
        "\"buffers\":[{\"byteLength\":36,\"uri\":\"" + uri + "\"}],"
        "\"bufferViews\":[{\"buffer\":0,\"byteOffset\":0,"
        "\"byteLength\":36}],"
        "\"accessors\":[{\"bufferView\":0,\"byteOffset\":0,"
        "\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"}],"
        "\"meshes\":[{\"primitives\":[{\"attributes\":"
        "{\"POSITION\":0}}]}],"
        "\"nodes\":[{\"mesh\":0}],"
        "\"scenes\":[{\"nodes\":[0]}],\"scene\":0}";

    const auto p = dir / "tri.gltf";
    write_file(p, json);

    cardinal::string err;
    imp::ImportScene s = imp::import_file(str_of(p), &err);
    CHECK(s.ok);
    CHECK(s.source_format == "gltf");
    CHECK(s.meshes.size() == sz(1));
    if (!s.meshes.empty()) {
        CHECK(s.meshes[0].positions.size() == sz(3));
        CHECK(s.meshes[0].indices.size() == sz(3));     // auto 0,1,2
        const auto& P = s.meshes[0].positions;
        CHECK(approx(P[1].x, 1.0f, 1e-5f));
        CHECK(approx(P[2].y, 1.0f, 1e-5f));
    }
    CHECK(s.total_triangles() == 1u);
}

// ---- detect_format / format_name ------------------------------------
void test_format_detect() {
    CHECK(imp::detect_format("a/b/Model.obj")  == Format::Obj);
    CHECK(imp::detect_format("UP.OBJ")          == Format::Obj);  // lc'd
    CHECK(imp::detect_format("scene.gltf")      == Format::Gltf);
    CHECK(imp::detect_format("scene.glb")       == Format::Glb);
    CHECK(imp::detect_format("rig.fbx")         == Format::Fbx);
    CHECK(imp::detect_format("tex.png")         == Format::Unknown);
    CHECK(imp::detect_format("noext")           == Format::Unknown);
    CHECK(std::strcmp(imp::format_name(Format::Obj),  "obj")  == 0);
    CHECK(std::strcmp(imp::format_name(Format::Gltf), "gltf") == 0);
    CHECK(std::strcmp(imp::format_name(Format::Glb),  "glb")  == 0);
    CHECK(std::strcmp(imp::format_name(Format::Fbx),  "fbx")  == 0);
    CHECK(std::strcmp(imp::format_name(Format::Unknown),
                      "unknown") == 0);
}

// ---- failure paths: graceful, not crashy ----------------------------
void test_failures(const std::filesystem::path& dir) {
    cardinal::string err;

    imp::ImportScene a =
        imp::import_file(str_of(dir / "does_not_exist.obj"), &err);
    CHECK(!a.ok);
    CHECK(!err.empty());
    CHECK(!a.diagnostics.empty());

    err.clear();
    imp::ImportScene b =
        imp::import_file(str_of(dir / "weird.xyz"), &err);
    CHECK(!b.ok);                       // unknown extension
    CHECK(!err.empty());

    err.clear();
    imp::ImportScene c =
        imp::import_file(str_of(dir / "rig.fbx"), &err);
    CHECK(!c.ok);                       // FBX not implemented yet
    CHECK(c.source_format == "fbx");
    CHECK(!err.empty());

    // Malformed glTF JSON must fail cleanly (no crash, ok=false).
    const auto bad = dir / "bad.gltf";
    write_file(bad, "{ this is definitely not json ]");
    err.clear();
    imp::ImportScene d = imp::import_file(str_of(bad), &err);
    CHECK(!d.ok);
    CHECK(!err.empty());

    // Hostile glTF accessor counts must not OOM / crash: a negative
    // count casts to ~SIZE_MAX on resize(), a huge one over-allocates.
    // The decoded buffer is only 36 B, so read_accessor must reject and
    // yield an empty/bounded mesh — never terminate the process.
    {
        const float pos[9] = { 0,0,0, 1,0,0, 0,1,0 };
        std::vector<unsigned char> bb(sizeof(pos));
        std::memcpy(bb.data(), pos, sizeof(pos));
        const cardinal::string uri =
            "data:application/octet-stream;base64," + b64(bb);
        auto gltf_with_count = [&](const char* cnt) {
            return cardinal::string(
                "{\"asset\":{\"version\":\"2.0\"},"
                "\"buffers\":[{\"byteLength\":36,\"uri\":\"") + uri + "\"}],"
                "\"bufferViews\":[{\"buffer\":0,\"byteOffset\":0,"
                "\"byteLength\":36}],"
                "\"accessors\":[{\"bufferView\":0,\"byteOffset\":0,"
                "\"componentType\":5126,\"count\":" + cnt + ",\"type\":\"VEC3\"}],"
                "\"meshes\":[{\"primitives\":[{\"attributes\":"
                "{\"POSITION\":0}}]}],"
                "\"nodes\":[{\"mesh\":0}],"
                "\"scenes\":[{\"nodes\":[0]}],\"scene\":0}";
        };
        for (const char* badc : { "-1", "999999999", "2147483647" }) {
            const auto gp = dir / "hostile.gltf";
            write_file(gp, gltf_with_count(badc));
            cardinal::string ge;
            imp::ImportScene hs = imp::import_file(str_of(gp), &ge); // must return
            if (!hs.meshes.empty())
                CHECK(hs.meshes[0].positions.empty());
            CHECK(hs.total_vertices() == 0u);                        // bounded
        }
    }

    // Out-of-range glTF indices must not let downstream positions[idx]
    // read OOB: the corrupt triangle is dropped, the valid one kept, and
    // no emitted index ever exceeds the vertex count.
    {
        const float pos[9] = { 0,0,0, 1,0,0, 0,1,0 };
        const unsigned short idx[6] = { 0,1,2,  0,1,99 };   // 2nd tri bogus
        std::vector<unsigned char> bb(sizeof(pos) + sizeof(idx));
        std::memcpy(bb.data(), pos, sizeof(pos));
        std::memcpy(bb.data() + sizeof(pos), idx, sizeof(idx));
        const cardinal::string uri =
            "data:application/octet-stream;base64," + b64(bb);
        const cardinal::string json = cardinal::string(
            "{\"asset\":{\"version\":\"2.0\"},"
            "\"buffers\":[{\"byteLength\":48,\"uri\":\"") + uri + "\"}],"
            "\"bufferViews\":[{\"buffer\":0,\"byteOffset\":0,\"byteLength\":36},"
            "{\"buffer\":0,\"byteOffset\":36,\"byteLength\":12}],"
            "\"accessors\":[{\"bufferView\":0,\"byteOffset\":0,"
            "\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"},"
            "{\"bufferView\":1,\"byteOffset\":0,\"componentType\":5123,"
            "\"count\":6,\"type\":\"SCALAR\"}],"
            "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0},"
            "\"indices\":1}]}],"
            "\"nodes\":[{\"mesh\":0}],"
            "\"scenes\":[{\"nodes\":[0]}],\"scene\":0}";
        const auto gp = dir / "oob_idx.gltf";
        write_file(gp, json);
        cardinal::string ge;
        imp::ImportScene hs = imp::import_file(str_of(gp), &ge);  // must return
        bool all_in_range = true;
        for (const auto& m : hs.meshes) {
            CHECK(m.indices.size() % 3u == 0u);
            const cardinal::usize vc = m.positions.size();
            for (auto ix : m.indices)
                if (static_cast<cardinal::usize>(ix) >= vc) all_in_range = false;
        }
        CHECK(all_in_range);                  // never OOB downstream
        CHECK(hs.total_triangles() == 1u);    // bogus tri dropped, good kept
    }
}

// ---- Megascans: surface asset (material-only, full PBR map set) -----
void test_megascans_surface(const std::filesystem::path& base) {
    const auto d = subdir(base, "ms_surface");
    write_file(d / "rock_2K_Albedo.jpg",        "x");
    write_file(d / "rock_2K_Normal.jpg",        "x");
    write_file(d / "rock_2K_Roughness.jpg",     "x");
    write_file(d / "rock_2K_AO.jpg",            "x");
    write_file(d / "rock_2K_Displacement.exr",  "x");
    write_file(d / "rock_2K_Metalness.jpg",     "x");
    const auto man = d / "rock_surface_ab12.json";
    write_file(man,
        "{\"id\":\"ab12\",\"name\":\"rock_surface\",\"type\":\"surface\",\"maps\":["
        "{\"type\":\"albedo\",\"uri\":\"rock_2K_Albedo.jpg\"},"
        "{\"type\":\"normal\",\"uri\":\"rock_2K_Normal.jpg\"},"
        "{\"type\":\"roughness\",\"uri\":\"rock_2K_Roughness.jpg\"},"
        "{\"type\":\"ao\",\"uri\":\"rock_2K_AO.jpg\"},"
        "{\"type\":\"displacement\",\"uri\":\"rock_2K_Displacement.exr\"},"
        "{\"type\":\"metalness\",\"uri\":\"rock_2K_Metalness.jpg\"}]}");

    cardinal::string err;
    imp::ImportScene s = imp::import_megascans(str_of(man), &err);
    CHECK(s.ok);
    CHECK(s.source_format == "megascans");
    CHECK(s.meshes.empty());                  // surface ⇒ material only
    CHECK(s.materials.size() == sz(1));
    if (!s.materials.empty()) {
        const imp::ImportMaterial& m = s.materials[0];
        CHECK(endswith(m.base_color_texture, "rock_2K_Albedo.jpg"));
        CHECK(endswith(m.normal_texture,     "rock_2K_Normal.jpg"));
        CHECK(endswith(m.roughness_texture,  "rock_2K_Roughness.jpg"));
        CHECK(endswith(m.occlusion_texture,  "rock_2K_AO.jpg"));
        CHECK(endswith(m.height_texture,     "rock_2K_Displacement.exr"));
        CHECK(endswith(m.metallic_texture,   "rock_2K_Metalness.jpg"));
        CHECK(!m.invert_roughness);
        CHECK(!m.mr_packed);
    }
    // sniff: both the manifest path and its containing dir register.
    CHECK(imp::is_megascans(str_of(man)));
    CHECK(imp::is_megascans(str_of(d)));
}

// ---- Megascans: 3D asset (mesh LOD recursion through import_file) ----
void test_megascans_3d(const std::filesystem::path& base) {
    const auto d = subdir(base, "ms_3d");
    write_file(d / "mesh_LOD0.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
    write_file(d / "mesh_LOD1.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
    write_file(d / "rock_2K_Albedo.jpg", "x");
    write_file(d / "rock_2K_Normal.jpg", "x");
    const auto man = d / "rock_3d_cd34.json";
    write_file(man,
        "{\"id\":\"cd34\",\"name\":\"rock_3d\",\"type\":\"3d\",\"maps\":["
        "{\"type\":\"albedo\",\"uri\":\"rock_2K_Albedo.jpg\"},"
        "{\"type\":\"normal\",\"uri\":\"rock_2K_Normal.jpg\"}],"
        "\"meshes\":[{\"name\":\"Var1\",\"uris\":["
        "{\"uri\":\"mesh_LOD0.obj\",\"lod\":0},"
        "{\"uri\":\"mesh_LOD1.obj\",\"lod\":1}]}]}");

    cardinal::string err;
    imp::ImportScene s = imp::import_megascans(str_of(man), &err);
    CHECK(s.ok);
    CHECK(s.meshes.size() >= sz(1));
    CHECK(s.total_vertices()  == 3u);         // LOD0 geometry survived recursion
    CHECK(s.total_triangles() == 1u);         // only the primary LOD imported
    CHECK(s.nodes.size() >= sz(1));
    if (!s.meshes.empty()) CHECK(s.meshes[0].material == 0);
    if (!s.materials.empty()) {
        CHECK(endswith(s.materials[0].base_color_texture, "rock_2K_Albedo.jpg"));
        CHECK(endswith(s.materials[0].normal_texture,     "rock_2K_Normal.jpg"));
    }
}

// ---- Megascans: gloss map → roughness slot with invert flag ---------
void test_megascans_gloss(const std::filesystem::path& base) {
    const auto d = subdir(base, "ms_gloss");
    write_file(d / "x_2K_Albedo.jpg", "x");
    write_file(d / "x_2K_Gloss.jpg",  "x");
    const auto man = d / "x.json";
    write_file(man,
        "{\"type\":\"surface\",\"maps\":["
        "{\"type\":\"albedo\",\"uri\":\"x_2K_Albedo.jpg\"},"
        "{\"type\":\"gloss\",\"uri\":\"x_2K_Gloss.jpg\"}]}");

    imp::ImportScene s = imp::import_megascans(str_of(man));
    CHECK(s.ok);
    if (!s.materials.empty()) {
        CHECK(endswith(s.materials[0].roughness_texture, "x_2K_Gloss.jpg"));
        CHECK(s.materials[0].invert_roughness);     // gloss ⇒ 1 − roughness
    }
}

// ---- Megascans: tolerance — "components" synonym + "path" alias -----
void test_megascans_components(const std::filesystem::path& base) {
    const auto d = subdir(base, "ms_components");
    write_file(d / "c_2K_Albedo.jpg",    "x");
    write_file(d / "c_2K_Normal.jpg",    "x");
    write_file(d / "c_2K_Roughness.jpg", "x");
    const auto man = d / "c.json";
    write_file(man,
        "{\"type\":\"surface\",\"components\":["
        "{\"type\":\"albedo\",\"uri\":\"c_2K_Albedo.jpg\"},"
        "{\"type\":\"normal\",\"path\":\"c_2K_Normal.jpg\"},"   // "path" alias
        "{\"type\":\"roughness\",\"file\":\"c_2K_Roughness.jpg\"}]}");  // "file" alias

    imp::ImportScene s = imp::import_megascans(str_of(man));
    CHECK(s.ok);
    if (!s.materials.empty()) {
        CHECK(endswith(s.materials[0].base_color_texture, "c_2K_Albedo.jpg"));
        CHECK(endswith(s.materials[0].normal_texture,     "c_2K_Normal.jpg"));
        CHECK(endswith(s.materials[0].roughness_texture,  "c_2K_Roughness.jpg"));
    }
}

// ---- Megascans: filename-scan fallback (no maps array in JSON) -------
void test_megascans_fallback(const std::filesystem::path& base) {
    const auto d = subdir(base, "ms_fallback");
    write_file(d / "name_2K_Albedo.jpg",    "x");
    write_file(d / "name_2K_Normal.jpg",    "x");
    write_file(d / "name_2K_Roughness.jpg", "x");
    write_file(d / "name_Preview.png",      "x");   // unknown token ⇒ ignored
    const auto man = d / "name.json";
    write_file(man, "{\"id\":\"zz\",\"type\":\"surface\"}");   // NO maps array

    cardinal::string err;
    imp::ImportScene s = imp::import_megascans(str_of(man), &err);
    CHECK(s.ok);
    if (!s.materials.empty()) {
        const imp::ImportMaterial& m = s.materials[0];
        CHECK(endswith(m.base_color_texture, "name_2K_Albedo.jpg"));
        CHECK(endswith(m.normal_texture,     "name_2K_Normal.jpg"));
        CHECK(endswith(m.roughness_texture,  "name_2K_Roughness.jpg"));
        CHECK(m.opacity_texture.empty());     // preview not hoovered up
        CHECK(m.metallic_texture.empty());    // no metalness file present
    }
}

// ---- Megascans: failure paths stay graceful -------------------------
void test_megascans_failures(const std::filesystem::path& base) {
    cardinal::string err;
    // Missing manifest dir.
    imp::ImportScene a =
        imp::import_megascans(str_of(base / "ms_does_not_exist"), &err);
    CHECK(!a.ok);
    CHECK(!err.empty());
    // A dir with no manifest at all.
    const auto empty_dir = subdir(base, "ms_empty");
    err.clear();
    imp::ImportScene b = imp::import_megascans(str_of(empty_dir), &err);
    CHECK(!b.ok);
    CHECK(!imp::is_megascans(str_of(empty_dir)));
}

// ---- Heightmap: RAW16 headerless u16 ramp ---------------------------
void test_heightmap_raw16(const std::filesystem::path& dir) {
    // 4x4 ramp: value[i] = i*4369, so value[15] = 65535 and height[i] = i/15.
    std::vector<unsigned char> raw;
    for (int i = 0; i < 16; ++i) {
        const unsigned v = static_cast<unsigned>(i * 4369);
        raw.push_back(static_cast<unsigned char>(v & 0xFF));
        raw.push_back(static_cast<unsigned char>((v >> 8) & 0xFF));
    }
    const auto p = dir / "ramp.r16";
    write_bytes(p, raw);

    cardinal::string err;
    imp::HeightField hf = imp::decode_heightmap(str_of(p), &err);
    CHECK(hf.ok);
    CHECK(hf.source_format == "raw16");
    CHECK(hf.width == 4u && hf.height == 4u);
    CHECK(hf.heights.size() == sz(16));
    CHECK(approx(hf.min, 0.0f, 1e-3f));
    CHECK(approx(hf.max, 65535.0f, 1.0f));
    if (hf.heights.size() == sz(16)) {
        CHECK(approx(hf.heights[0], 0.0f, 1e-4f));
        CHECK(approx(hf.heights[15], 1.0f, 1e-4f));
        CHECK(approx(hf.heights[1], 1.0f / 15.0f, 1e-4f));
    }
    // Non-square byte count without raw_dim must fail cleanly.
    std::vector<unsigned char> odd(12, 0);          // 6 u16 -> not a square
    const auto q = dir / "odd.raw";
    write_bytes(q, odd);
    cardinal::string e2;
    imp::HeightField bad = imp::decode_heightmap(str_of(q), &e2);
    CHECK(!bad.ok);
    CHECK(!e2.empty());
    // …but with an explicit width it decodes as 3 wide x 2 tall.
    imp::HeightField okdim = imp::decode_heightmap(str_of(q), nullptr, 3);
    CHECK(okdim.ok && okdim.width == 3u && okdim.height == 2u);
}

// ---- Heightmap: uncompressed 24-bit BMP (bottom-up) -----------------
void test_heightmap_bmp(const std::filesystem::path& dir) {
    std::vector<unsigned char> b;
    auto pb    = [&](unsigned v){ b.push_back(static_cast<unsigned char>(v & 0xFF)); };
    auto u16le = [&](unsigned v){ pb(v); pb(v >> 8); };
    auto u32le = [&](unsigned v){ pb(v); pb(v >> 8); pb(v >> 16); pb(v >> 24); };
    b.push_back('B'); b.push_back('M');
    u32le(70);            // fileSize
    u32le(0);             // reserved
    u32le(54);            // pixel-data offset
    u32le(40);            // BITMAPINFOHEADER size
    u32le(2);             // width
    u32le(2);             // height (positive => bottom-up)
    u16le(1);             // planes
    u16le(24);            // bpp
    u32le(0);             // BI_RGB
    u32le(0); u32le(0); u32le(0); u32le(0); u32le(0);   // imgsize/ppm/clrs
    auto bgr = [&](unsigned r){ b.push_back(0); b.push_back(0);
                                b.push_back(static_cast<unsigned char>(r)); };
    // stored row 0 = bottom (=> output y=1): R 10, 20 ; + 2 pad
    bgr(10); bgr(20); b.push_back(0); b.push_back(0);
    // stored row 1 = top    (=> output y=0): R 30, 40 ; + 2 pad
    bgr(30); bgr(40); b.push_back(0); b.push_back(0);
    const auto p = dir / "hm.bmp";
    write_bytes(p, b);

    cardinal::string err;
    imp::HeightField hf = imp::decode_heightmap(str_of(p), &err);
    CHECK(hf.ok);
    CHECK(hf.source_format == "bmp");
    CHECK(hf.width == 2u && hf.height == 2u);
    if (hf.heights.size() == sz(4)) {
        CHECK(approx(hf.heights[0], 30.0f/255.0f, 1e-4f));   // top-left
        CHECK(approx(hf.heights[1], 40.0f/255.0f, 1e-4f));
        CHECK(approx(hf.heights[3], 20.0f/255.0f, 1e-4f));   // bottom-right
    }
    CHECK(approx(hf.min, 10.0f, 1e-3f));
    CHECK(approx(hf.max, 40.0f, 1e-3f));
}

// ---- Heightmap: uncompressed 8-bit grayscale TGA (bottom-up) --------
void test_heightmap_tga(const std::filesystem::path& dir) {
    std::vector<unsigned char> t(18, 0);
    t[2]  = 3;            // image type: uncompressed grayscale
    t[12] = 2; t[13] = 0; // width  = 2
    t[14] = 2; t[15] = 0; // height = 2
    t[16] = 8;            // 8 bpp
    t[17] = 0;            // descriptor: bottom-up origin
    // stored row 0 = bottom (=> output y=1): 50, 60
    t.push_back(50); t.push_back(60);
    // stored row 1 = top    (=> output y=0): 70, 80
    t.push_back(70); t.push_back(80);
    const auto p = dir / "hm.tga";
    write_bytes(p, t);

    cardinal::string err;
    imp::HeightField hf = imp::decode_heightmap(str_of(p), &err);
    CHECK(hf.ok);
    CHECK(hf.source_format == "tga");
    CHECK(hf.width == 2u && hf.height == 2u);
    if (hf.heights.size() == sz(4)) {
        CHECK(approx(hf.heights[0], 70.0f/255.0f, 1e-4f));   // top-left
        CHECK(approx(hf.heights[2], 50.0f/255.0f, 1e-4f));   // bottom-left
    }
    CHECK(approx(hf.min, 50.0f, 1e-3f));
    CHECK(approx(hf.max, 80.0f, 1e-3f));
}

// ---- FBX: synthetic binary-FBX fixtures (builder + tests) -----------
// A minimal binary-FBX writer that computes the absolute EndOffset each node
// record needs (FBX 7.4, u32 offsets), so the parser walks it for real.
struct FbxSpec {
    std::string name;
    std::vector<unsigned char> props;   // pre-encoded property bytes
    unsigned numProps = 0;
    std::vector<FbxSpec> children;
};
void fbx_u32(std::vector<unsigned char>& o, unsigned v) {
    o.push_back(static_cast<unsigned char>(v & 0xFFu));
    o.push_back(static_cast<unsigned char>((v >> 8) & 0xFFu));
    o.push_back(static_cast<unsigned char>((v >> 16) & 0xFFu));
    o.push_back(static_cast<unsigned char>((v >> 24) & 0xFFu));
}
std::vector<unsigned char> fbx_darr(const std::vector<double>& vals) {
    std::vector<unsigned char> p; p.push_back('d');
    fbx_u32(p, static_cast<unsigned>(vals.size())); fbx_u32(p, 0);
    fbx_u32(p, static_cast<unsigned>(vals.size() * 8));
    for (double d : vals) { unsigned char b[8]; std::memcpy(b, &d, 8); p.insert(p.end(), b, b + 8); }
    return p;
}
std::vector<unsigned char> fbx_darr_z(unsigned arrLen, const unsigned char* zl, unsigned zn) {
    std::vector<unsigned char> p; p.push_back('d');
    fbx_u32(p, arrLen); fbx_u32(p, 1); fbx_u32(p, zn);   // encoding 1 = zlib
    p.insert(p.end(), zl, zl + zn);
    return p;
}
std::vector<unsigned char> fbx_iarr(const std::vector<int>& vals) {
    std::vector<unsigned char> p; p.push_back('i');
    fbx_u32(p, static_cast<unsigned>(vals.size())); fbx_u32(p, 0);
    fbx_u32(p, static_cast<unsigned>(vals.size() * 4));
    for (int v : vals) { unsigned char b[4]; std::memcpy(b, &v, 4); p.insert(p.end(), b, b + 4); }
    return p;
}
std::vector<unsigned char> fbx_str(const std::string& s) {
    std::vector<unsigned char> p; p.push_back('S');
    fbx_u32(p, static_cast<unsigned>(s.size()));
    for (char c : s) p.push_back(static_cast<unsigned char>(c));
    return p;
}
std::vector<unsigned char> fbx_build(const FbxSpec& s, unsigned start) {
    const unsigned hdr = 13;     // 3*u32 + u8 (FBX 7.4)
    const unsigned childStart =
        start + hdr + static_cast<unsigned>(s.name.size()) + static_cast<unsigned>(s.props.size());
    std::vector<unsigned char> kids;
    if (!s.children.empty()) {
        unsigned off = childStart;
        for (const auto& c : s.children) {
            auto cb = fbx_build(c, off);
            off += static_cast<unsigned>(cb.size());
            kids.insert(kids.end(), cb.begin(), cb.end());
        }
        for (int k = 0; k < 13; ++k) kids.push_back(0);   // null terminator
    }
    const unsigned endOff = childStart + static_cast<unsigned>(kids.size());
    std::vector<unsigned char> out;
    fbx_u32(out, endOff); fbx_u32(out, s.numProps);
    fbx_u32(out, static_cast<unsigned>(s.props.size()));
    out.push_back(static_cast<unsigned char>(s.name.size()));
    for (char c : s.name) out.push_back(static_cast<unsigned char>(c));
    out.insert(out.end(), s.props.begin(), s.props.end());
    out.insert(out.end(), kids.begin(), kids.end());
    return out;
}
std::vector<unsigned char> fbx_file(const FbxSpec& topNode) {
    std::vector<unsigned char> f;
    const char* M = "Kaydara FBX Binary  ";
    for (int i = 0; i < 20; ++i) f.push_back(static_cast<unsigned char>(M[i]));
    f.push_back(0x00); f.push_back(0x1A); f.push_back(0x00);
    fbx_u32(f, 7400);                                     // version -> 7.4
    auto node = fbx_build(topNode, 27);                  // header is 27 bytes
    f.insert(f.end(), node.begin(), node.end());
    for (int k = 0; k < 13; ++k) f.push_back(0);          // top-level null record
    return f;
}

void test_fbx_triangle(const std::filesystem::path& dir) {
    FbxSpec verts; verts.name = "Vertices"; verts.numProps = 1;
    verts.props = fbx_darr({0,0,0, 1,0,0, 0,1,0});
    FbxSpec poly; poly.name = "PolygonVertexIndex"; poly.numProps = 1;
    poly.props = fbx_iarr({0, 1, ~2});                   // last index ~2 = -3
    FbxSpec nrm; nrm.name = "Normals"; nrm.numProps = 1;
    nrm.props = fbx_darr({0,0,1, 0,0,1, 0,0,1});
    FbxSpec mit; mit.name = "MappingInformationType"; mit.numProps = 1;
    mit.props = fbx_str("ByVertice");
    FbxSpec rit; rit.name = "ReferenceInformationType"; rit.numProps = 1;
    rit.props = fbx_str("Direct");
    FbxSpec nl; nl.name = "LayerElementNormal"; nl.children = { nrm, mit, rit };
    FbxSpec geo; geo.name = "Geometry"; geo.children = { verts, poly, nl };

    const auto p = dir / "tri.fbx";
    write_bytes(p, fbx_file(geo));
    cardinal::string err;
    imp::ImportScene s = imp::import_file(str_of(p), &err);
    CHECK(s.ok);
    CHECK(s.source_format == "fbx");
    CHECK(s.meshes.size() == sz(1));
    CHECK(s.total_vertices() == 3u);
    CHECK(s.total_triangles() == 1u);
    if (!s.meshes.empty() && s.meshes[0].positions.size() == sz(3)) {
        CHECK(approx(s.meshes[0].positions[1].x, 1.0f, 1e-5f));
        CHECK(approx(s.meshes[0].positions[2].y, 1.0f, 1e-5f));
        CHECK(s.meshes[0].normals.size() == sz(3));      // ByVertice/Direct
        if (s.meshes[0].normals.size() == sz(3))
            CHECK(approx(s.meshes[0].normals[0].z, 1.0f, 1e-5f));
    }
}

void test_fbx_compressed(const std::filesystem::path& dir) {
    // Vertices via a zlib-compressed 'd' array (9 doubles {0,0,0,2,0,0,0,3,0})
    // — exercises the FBX -> core::compress::inflate_zlib path end-to-end.
    static const unsigned char ZV[] = {
        0x78,0x9c,0x63,0x60,0xc0,0x0b,0x1c,0xf0,0x4b,0x73,0xc0,0xe5,0x01,0x0d,0x18,0x00,0x89 };
    FbxSpec verts; verts.name = "Vertices"; verts.numProps = 1;
    verts.props = fbx_darr_z(9, ZV, static_cast<unsigned>(sizeof(ZV)));
    FbxSpec poly; poly.name = "PolygonVertexIndex"; poly.numProps = 1;
    poly.props = fbx_iarr({0, 1, ~2});
    FbxSpec geo; geo.name = "Geometry"; geo.children = { verts, poly };

    const auto p = dir / "compressed.fbx";
    write_bytes(p, fbx_file(geo));
    cardinal::string err;
    imp::ImportScene s = imp::import_file(str_of(p), &err);
    CHECK(s.ok);
    CHECK(s.total_vertices() == 3u);
    CHECK(s.total_triangles() == 1u);
    if (!s.meshes.empty() && s.meshes[0].positions.size() == sz(3)) {
        CHECK(approx(s.meshes[0].positions[1].x, 2.0f, 1e-5f));   // decompressed
        CHECK(approx(s.meshes[0].positions[2].y, 3.0f, 1e-5f));
    }
}

void test_fbx_failures(const std::filesystem::path& dir) {
    // ASCII FBX (no binary magic) must fail cleanly, not crash.
    const auto p = dir / "ascii.fbx";
    write_file(p, "; FBX 7.4.0 project file\nObjects:  {\n}\n");
    cardinal::string err;
    imp::ImportScene s = imp::import_file(str_of(p), &err);
    CHECK(!s.ok);
    CHECK(s.source_format == "fbx");
    CHECK(!err.empty());
}

// ---- USDA: a def Mesh quad (points / counts / indices / normals / st) ----
void test_usda(const std::filesystem::path& dir) {
    const auto p = dir / "quad.usda";
    write_file(p,
        "#usda 1.0\n"
        "def Xform \"root\" {\n"
        "  def Mesh \"quad\" {\n"
        "    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]\n"
        "    int[] faceVertexCounts = [4]\n"
        "    int[] faceVertexIndices = [0, 1, 3, 2]\n"
        "    normal3f[] normals = [(0,0,1), (0,0,1), (0,0,1), (0,0,1)]\n"
        "    texCoord2f[] primvars:st = [(0,0), (1,0), (1,1), (0,1)]\n"
        "  }\n"
        "}\n");

    cardinal::string err;
    imp::ImportScene s = imp::import_file(str_of(p), &err);
    CHECK(s.ok);
    CHECK(s.source_format == "usda");
    CHECK(s.meshes.size() == sz(1));
    CHECK(s.total_vertices() == 4u);
    CHECK(s.total_triangles() == 2u);          // quad -> 2 fan triangles
    if (!s.meshes.empty()) {
        const imp::ImportMesh& m = s.meshes[0];
        CHECK(m.positions.size() == sz(4));
        CHECK(m.normals.size() == sz(4));      // per-vertex
        CHECK(m.uvs.size() == sz(4));
        if (m.positions.size() == sz(4)) {
            CHECK(approx(m.positions[1].x, 1.0f, 1e-5f));
            CHECK(approx(m.positions[3].y, 1.0f, 1e-5f));
        }
        if (m.normals.size() == sz(4)) CHECK(approx(m.normals[0].z, 1.0f, 1e-5f));
        if (m.uvs.size() == sz(4))     CHECK(approx(m.uvs[2].u, 1.0f, 1e-5f));
    }
}

// ---- USD: USDC binary crate is detected + rejected cleanly ----------
void test_usdc_reject(const std::filesystem::path& dir) {
    const auto p = dir / "crate.usd";          // .usd that is actually a crate
    write_bytes(p, std::vector<unsigned char>{'P','X','R','-','U','S','D','C',0,0,0,0});
    cardinal::string err;
    imp::ImportScene s = imp::import_file(str_of(p), &err);
    CHECK(!s.ok);
    CHECK(!err.empty());
}

// ---- USDZ: a STORED zip wrapping a USDA root layer ------------------
void test_usdz(const std::filesystem::path& dir) {
    const std::string usda =
        "#usda 1.0\n"
        "def Mesh \"m\" {\n"
        "  point3f[] points = [(0,0,0),(1,0,0),(0,1,0)]\n"
        "  int[] faceVertexCounts = [3]\n"
        "  int[] faceVertexIndices = [0,1,2]\n"
        "}\n";
    const std::string name = "root.usda";
    std::vector<unsigned char> z;
    auto u16 = [&](unsigned v){ z.push_back(static_cast<unsigned char>(v & 0xFFu));
                                z.push_back(static_cast<unsigned char>((v >> 8) & 0xFFu)); };
    auto u32 = [&](unsigned v){ for (int k = 0; k < 4; ++k)
                                z.push_back(static_cast<unsigned char>((v >> (8*k)) & 0xFFu)); };
    const unsigned dataLen = static_cast<unsigned>(usda.size());
    const unsigned nameLen = static_cast<unsigned>(name.size());
    // Local file header @ offset 0 (STORED, method 0).
    u32(0x04034b50); u16(20); u16(0); u16(0); u16(0); u16(0);
    u32(0); u32(dataLen); u32(dataLen); u16(nameLen); u16(0);
    for (char c : name) z.push_back(static_cast<unsigned char>(c));
    for (char c : usda) z.push_back(static_cast<unsigned char>(c));
    const unsigned cdOff = static_cast<unsigned>(z.size());
    // Central directory record (local header offset = 0).
    u32(0x02014b50); u16(20); u16(20); u16(0); u16(0); u16(0); u16(0);
    u32(0); u32(dataLen); u32(dataLen); u16(nameLen); u16(0); u16(0);
    u16(0); u16(0); u32(0); u32(0);
    for (char c : name) z.push_back(static_cast<unsigned char>(c));
    const unsigned cdSize = static_cast<unsigned>(z.size()) - cdOff;
    // End of central directory.
    u32(0x06054b50); u16(0); u16(0); u16(1); u16(1); u32(cdSize); u32(cdOff); u16(0);

    const auto p = dir / "model.usdz";
    write_bytes(p, z);
    cardinal::string err;
    imp::ImportScene s = imp::import_file(str_of(p), &err);
    CHECK(s.ok);
    CHECK(s.source_format == "usdz");
    CHECK(s.total_vertices() == 3u);
    CHECK(s.total_triangles() == 1u);
}

// ---- Heightmap: graceful failure on a not-yet-supported format ------
void test_heightmap_unsupported(const std::filesystem::path& dir) {
    // A .png is recognised but needs DEFLATE (not yet) — must fail cleanly.
    const auto p = dir / "fake.png";
    write_bytes(p, std::vector<unsigned char>{0x89,'P','N','G',0,0,0,0});
    cardinal::string err;
    imp::HeightField hf = imp::decode_heightmap(str_of(p), &err);
    CHECK(!hf.ok);
    CHECK(!err.empty());
}

}  // namespace

int main() {
    std::error_code ec;
    std::filesystem::path dir =
        std::filesystem::temp_directory_path(ec) / "cardinal_import_test";
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        cardinal::log::infof("imptest",
            "SKIP  no writable temp dir in this environment");
        return 0;
    }

    test_obj_triangle(dir);
    test_obj_quad(dir);
    test_obj_negative(dir);
    test_obj_mixed_normals(dir);
    test_obj_mtl(dir);
    test_obj_vertex_color(dir);
    test_gltf_triangle(dir);
    test_gltf_pbr_maps(dir);
    test_megascans_surface(dir);
    test_megascans_3d(dir);
    test_megascans_gloss(dir);
    test_megascans_components(dir);
    test_megascans_fallback(dir);
    test_megascans_failures(dir);
    test_heightmap_raw16(dir);
    test_heightmap_bmp(dir);
    test_heightmap_tga(dir);
    test_heightmap_unsupported(dir);
    test_fbx_triangle(dir);
    test_fbx_compressed(dir);
    test_fbx_failures(dir);
    test_usda(dir);
    test_usdc_reject(dir);
    test_usdz(dir);
    test_format_detect();
    test_failures(dir);

    const auto removed = std::filesystem::remove_all(dir, ec);
    (void)removed;

    if (g_fail == 0) {
        cardinal::log::infof("imptest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("imptest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
