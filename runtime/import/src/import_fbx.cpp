// =============================================================================
// Cardinal — DCC importer: binary FBX backend (Autodesk FBX 7.4 / 7.5).
//
// Hand-rolled + zero-dep, same ethos as OBJ / glTF / Megascans. FBX is the
// native interchange of Maya, 3ds Max and Cinema 4D (and a Blender export),
// so this one backend opens those tools' richest export path.
//
// Format: a "Kaydara FBX Binary  \0" magic + version, then a tree of node
// records (EndOffset/NumProperties/PropertyListLen/NameLen/Name/properties/
// nested-records, the nested list ended by an all-zero record). Offsets are
// u32 for version < 7500 (FBX 7.4) and u64 for >= 7500 (FBX 7.5). Property
// arrays (f/d/l/i/b) may be zlib-DEFLATE compressed (Encoding==1) — decoded
// via cardinal::core::compress::inflate_zlib (the exact uncompressed size is
// ArrayLength*elemsize, so no growth). The importer is the cook/renderer
// trust boundary: every read is bounds-checked; malformed input fails clean.
//
// Scope (mirrors the glTF backend's deliberate first-pass scope): mesh
// geometry — Vertices, the negative-terminated n-gon PolygonVertexIndex
// (fan-triangulated), and per-vertex (ByVertice/Direct) normals. DEFERRED:
// per-polygon-vertex normal/UV de-dup, materials/Connections, skinning,
// blendshapes, animation, and the legacy ASCII FBX variant (sniffed +
// rejected cleanly). Materials are left unset (material = -1) for now.
// =============================================================================

#include <cardinal/import/import.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/compress/inflate.hpp>

#include <cardinal/core/std/cstring.hpp>      // cardinal::memcpy / memcmp
#include <cardinal/core/std/fstream.hpp>      // cardinal::ifstream / ios
#include <cardinal/core/std/sstream.hpp>      // cardinal::ostringstream
#include <cardinal/core/std/utility.hpp>      // cardinal::move
#include <cardinal/core/std/containers.hpp>   // cardinal::vector

namespace cardinal::import {

using cardinal::scene::Vec3;

namespace {

cardinal::vector<u8> read_file(const cardinal::string& path) {
    cardinal::ifstream f(path, cardinal::ios::binary | cardinal::ios::ate);
    if (!f) return {};
    const cardinal::streamsize n = f.tellg();
    if (n <= 0) return {};                            // -1 unseekable / 0 empty
    f.seekg(0);
    cardinal::vector<u8> v(static_cast<cardinal::size_t>(n));
    f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

// Bounds-checked little-endian reader over the whole file buffer.
struct Rd {
    const u8* p {nullptr};
    cardinal::usize n {0};
    cardinal::usize pos {0};
    bool err {false};

    bool avail(cardinal::usize k) noexcept {
        if (pos + k > n) { err = true; return false; }
        return true;
    }
    u8  u8_()  noexcept { return avail(1) ? p[pos++] : (err = true, u8{0}); }
    u32 u32_() noexcept {
        if (!avail(4)) return 0;
        const u32 v = static_cast<u32>(p[pos]) | (static_cast<u32>(p[pos+1]) << 8) |
                      (static_cast<u32>(p[pos+2]) << 16) | (static_cast<u32>(p[pos+3]) << 24);
        pos += 4; return v;
    }
    u64 u64_() noexcept {
        if (!avail(8)) return 0;
        u64 v = 0;
        for (int k = 0; k < 8; ++k) v |= static_cast<u64>(p[pos + k]) << (8 * k);
        pos += 8; return v;
    }
    u64 off(int W) noexcept { return W == 8 ? u64_() : static_cast<u64>(u32_()); }
    i16 i16_() noexcept { if (!avail(2)) return 0; i16 v; cardinal::memcpy(&v, p+pos, 2); pos+=2; return v; }
    float  f32_() noexcept { if (!avail(4)) return 0; float  v; cardinal::memcpy(&v, p+pos, 4); pos+=4; return v; }
    double f64_() noexcept { if (!avail(8)) return 0; double v; cardinal::memcpy(&v, p+pos, 8); pos+=8; return v; }
};

// One decoded property. Arrays/scalars land in `d` (float-y) or `i` (int-y);
// strings in `s`. (We only ever read d/i arrays + strings downstream.)
struct FbxProp {
    char                       type {0};
    cardinal::vector<double>   d;
    cardinal::vector<long long> i;
    cardinal::string           s;
};

struct FbxNode {
    cardinal::string         name;
    cardinal::vector<FbxProp> props;
    cardinal::vector<FbxNode> children;
    const FbxNode* child(const char* nm) const noexcept {
        for (const auto& c : children) if (c.name == nm) return &c;
        return nullptr;
    }
};

constexpr u32 kArrayCap = 1u << 28;       // sanity cap on array element count

void read_prop(Rd& r, cardinal::vector<FbxProp>& out) {
    FbxProp pr;
    pr.type = static_cast<char>(r.u8_());
    switch (pr.type) {
        case 'Y': pr.i.push_back(r.i16_()); break;
        case 'C': pr.i.push_back(r.u8_() ? 1 : 0); break;
        case 'I': pr.i.push_back(static_cast<long long>(static_cast<i32>(r.u32_()))); break;
        case 'L': pr.i.push_back(static_cast<long long>(r.u64_())); break;
        case 'F': pr.d.push_back(r.f32_()); break;
        case 'D': pr.d.push_back(r.f64_()); break;
        case 'S': case 'R': {
            const u32 len = r.u32_();
            if (r.avail(len)) {
                if (pr.type == 'S')
                    pr.s.assign(reinterpret_cast<const char*>(r.p + r.pos), len);
                r.pos += len;
            }
            break;
        }
        case 'f': case 'd': case 'l': case 'i': case 'b': {
            const u32 arrLen  = r.u32_();
            const u32 enc     = r.u32_();
            const u32 compLen = r.u32_();
            const int elem = (pr.type == 'd' || pr.type == 'l') ? 8
                           : (pr.type == 'b') ? 1 : 4;
            if (arrLen > kArrayCap) { r.err = true; break; }
            const cardinal::usize rawSize = static_cast<cardinal::usize>(arrLen) * elem;
            cardinal::vector<u8> raw;
            if (enc == 1) {                           // zlib-DEFLATE compressed
                if (!r.avail(compLen)) break;
                raw.resize(rawSize);
                const cardinal::usize got = cardinal::core::compress::inflate_zlib(
                    r.p + r.pos, compLen, raw.data(), rawSize);
                r.pos += compLen;
                if (got != rawSize) { r.err = true; break; }
            } else {                                  // raw
                if (!r.avail(rawSize)) break;
                raw.assign(r.p + r.pos, r.p + r.pos + rawSize);
                r.pos += rawSize;
            }
            const u8* q = raw.data();
            for (u32 k = 0; k < arrLen; ++k) {
                if (pr.type == 'f') { float  v; cardinal::memcpy(&v, q + k*4, 4); pr.d.push_back(v); }
                else if (pr.type == 'd') { double v; cardinal::memcpy(&v, q + k*8, 8); pr.d.push_back(v); }
                else if (pr.type == 'i') { i32 v; cardinal::memcpy(&v, q + k*4, 4); pr.i.push_back(v); }
                else if (pr.type == 'l') { i64 v; cardinal::memcpy(&v, q + k*8, 8); pr.i.push_back(v); }
                else { pr.i.push_back(q[k] ? 1 : 0); }   // 'b'
            }
            break;
        }
        default: r.err = true; break;                 // unknown property type
    }
    out.push_back(cardinal::move(pr));
}

// Parse one node record. Returns false on a null record (end of a child list)
// or on error. On success `out` holds the node + its recursively-parsed kids.
bool parse_node(Rd& r, int W, FbxNode& out, int depth) {
    if (depth > 64) { r.err = true; return false; }
    const u64 endOff   = r.off(W);
    const u64 numProps = r.off(W);
    const u64 propLen  = r.off(W);
    const u8  nameLen  = r.u8_();
    if (r.err) return false;
    if (endOff == 0 && numProps == 0 && propLen == 0 && nameLen == 0)
        return false;                                 // null terminator record
    if (endOff > r.n || endOff < r.pos) { r.err = true; return false; }
    if (r.avail(nameLen)) {
        out.name.assign(reinterpret_cast<const char*>(r.p + r.pos), nameLen);
        r.pos += nameLen;
    }
    for (u64 k = 0; k < numProps && !r.err; ++k) read_prop(r, out.props);
    if (r.err) return false;
    while (r.pos < endOff && !r.err) {
        FbxNode child;
        if (!parse_node(r, W, child, depth + 1)) break;
        out.children.push_back(cardinal::move(child));
    }
    if (!r.err && r.pos < endOff) r.pos = static_cast<cardinal::usize>(endOff);  // skip any tail
    return !r.err;
}

// Collect mesh geometry from every "Geometry" node in the tree.
void collect_geometry(const FbxNode& node, ImportScene& scene) {
    if (node.name == "Geometry") {
        const FbxNode* verts = node.child("Vertices");
        const FbxNode* poly  = node.child("PolygonVertexIndex");
        if (verts && !verts->props.empty() && poly && !poly->props.empty()) {
            const cardinal::vector<double>&    V = verts->props[0].d;
            const cardinal::vector<long long>& P = poly->props[0].i;

            ImportMesh m;
            m.name = "fbx_mesh";
            for (cardinal::usize k = 0; k + 3 <= V.size(); k += 3)
                m.positions.push_back(Vec3{ static_cast<float>(V[k]),
                                            static_cast<float>(V[k + 1]),
                                            static_cast<float>(V[k + 2]) });
            const u32 vcount = static_cast<u32>(m.positions.size());

            // Negative-terminated n-gons → fan triangles (drop OOB indices).
            cardinal::vector<u32> face;
            for (cardinal::usize k = 0; k < P.size(); ++k) {
                const long long raw_i = P[k];
                const bool last = raw_i < 0;
                const long long real = last ? ~raw_i : raw_i;   // FBX: last = ~index
                if (real >= 0 && static_cast<u64>(real) < vcount)
                    face.push_back(static_cast<u32>(real));
                else
                    face.push_back(0xFFFFFFFFu);                 // mark invalid
                if (last) {
                    for (cardinal::usize t = 2; t < face.size(); ++t) {
                        const u32 a = face[0], b = face[t - 1], c = face[t];
                        if (a != 0xFFFFFFFFu && b != 0xFFFFFFFFu && c != 0xFFFFFFFFu) {
                            m.indices.push_back(a);
                            m.indices.push_back(b);
                            m.indices.push_back(c);
                        }
                    }
                    face.clear();
                }
            }

            // Per-vertex (ByVertice/Direct) normals only — parallel to positions.
            if (const FbxNode* nl = node.child("LayerElementNormal")) {
                const FbxNode* nrm = nl->child("Normals");
                const FbxNode* mit = nl->child("MappingInformationType");
                const FbxNode* rit = nl->child("ReferenceInformationType");
                const cardinal::string map = (mit && !mit->props.empty()) ? mit->props[0].s : cardinal::string{};
                const cardinal::string ref = (rit && !rit->props.empty()) ? rit->props[0].s : cardinal::string{};
                if (nrm && !nrm->props.empty() &&
                    (map == "ByVertice" || map == "ByVertex") &&
                    (ref == "Direct" || ref.empty())) {
                    const cardinal::vector<double>& N = nrm->props[0].d;
                    if (N.size() / 3 == m.positions.size())
                        for (cardinal::usize k = 0; k + 3 <= N.size(); k += 3)
                            m.normals.push_back(Vec3{ static_cast<float>(N[k]),
                                                      static_cast<float>(N[k + 1]),
                                                      static_cast<float>(N[k + 2]) });
                }
            }

            if (!m.positions.empty() && !m.indices.empty())
                scene.meshes.push_back(cardinal::move(m));
        }
    }
    for (const auto& c : node.children) collect_geometry(c, scene);
}

}  // namespace

ImportScene import_fbx(const cardinal::string& path, cardinal::string* error) {
    ImportScene scene;
    scene.source_format = "fbx";

    const cardinal::vector<u8> file = read_file(path);
    if (file.empty()) {
        if (error) *error = "cannot open '" + path + "'";
        scene.diagnostics = "FBX: open failed";
        return scene;
    }
    // Magic: "Kaydara FBX Binary  " (20) + 0x00 + 0x1A 0x00 + u32 version.
    if (file.size() < 27 ||
        cardinal::memcmp(file.data(), "Kaydara FBX Binary  ", 20) != 0 ||
        file[21] != 0x1A) {
        if (error) *error = "not a binary FBX (ASCII FBX is not supported — "
                            "re-export as binary FBX)";
        scene.diagnostics = "FBX: bad/unknown header";
        return scene;
    }
    u32 version = 0;
    cardinal::memcpy(&version, file.data() + 23, 4);
    const int W = (version >= 7500) ? 8 : 4;          // FBX 7.5 uses u64 offsets

    Rd r;
    r.p = file.data(); r.n = file.size(); r.pos = 27;
    FbxNode root;
    while (r.pos < file.size() && !r.err) {
        FbxNode node;
        if (!parse_node(r, W, node, 0)) break;        // null record / error ends the list
        root.children.push_back(cardinal::move(node));
        if (file.size() - r.pos < 16) break;          // remaining is the footer
    }

    collect_geometry(root, scene);

    if (scene.meshes.empty()) {
        if (error) *error = "FBX: no mesh geometry found "
                            "(version " + cardinal::to_string(version) + ")";
        scene.diagnostics = "FBX: parsed but produced 0 meshes";
        return scene;
    }
    for (cardinal::usize i = 0; i < scene.meshes.size(); ++i) {
        ImportNode nd;
        nd.name = scene.meshes[i].name;
        nd.meshes.push_back(static_cast<int>(i));
        scene.roots.push_back(static_cast<int>(scene.nodes.size()));
        scene.nodes.push_back(cardinal::move(nd));
    }
    scene.ok = true;
    cardinal::ostringstream d;
    d << "FBX " << (version / 1000) << '.' << ((version / 100) % 10) << ": "
      << scene.meshes.size() << " mesh(es), " << scene.total_vertices()
      << " verts, " << scene.total_triangles() << " tris";
    scene.diagnostics = d.str();
    cardinal::log::infof("import", "%s — %s", path.c_str(), scene.diagnostics.c_str());
    return scene;
}

}  // namespace cardinal::import
