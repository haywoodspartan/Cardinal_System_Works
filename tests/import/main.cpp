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
#include <cardinal/core/log.hpp>

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

// ---- OBJ + MTL: metallic-roughness PBR binding ----------------------
void test_obj_mtl(const std::filesystem::path& dir) {
    write_file(dir / "mat.mtl",
        "newmtl Red\n"
        "Kd 0.9 0.1 0.05\n"
        "Ke 0 0 0\n"
        "Pr 0.25\n"
        "Pm 0.80\n");
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
    }
    if (!s.meshes.empty()) CHECK(s.meshes[0].material == 0);
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
    test_obj_mtl(dir);
    test_obj_vertex_color(dir);
    test_gltf_triangle(dir);
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
