// =============================================================================
// Cardinal — DCC importer: glTF 2.0 backend (.gltf + .glb).
//
// glTF is THE modern interchange — Blender, Maya, Daz, Autodesk and
// Unreal all export it, and it carries metallic-roughness PBR natively,
// so this one backend is the high-fidelity path for every listed tool.
//
// Zero-dep (no tinygltf / cgltf), same ethos as the hand-rolled OBJ /
// physics / audio / net: a compact recursive JSON reader + the glTF
// graph walk (buffers → bufferViews → accessors → mesh primitives,
// pbrMetallicRoughness materials, node TRS hierarchy). Handles both
// containers: text .gltf (external / base64 data-URI buffers) and
// binary .glb (12-byte header + JSON chunk + BIN chunk).
//
// Scope: triangle primitives with POSITION / NORMAL / TEXCOORD_0 /
// COLOR_0 + indices (u8/u16/u32), the metallic-roughness material
// model, and the node graph. Skinning / animation / morph targets /
// KHR extensions are out of scope here (geometry + materials is what
// the asset pipeline consumes today).
// =============================================================================

#include <cardinal/import/import.hpp>
#include <cardinal/core/log.hpp>
#include <cardinal/core/math.hpp>

#include <cardinal/core/algorithm.hpp>    // cardinal::min/max
#include <cardinal/core/cctype.hpp>       // cardinal::isspace/isdigit
#include <cardinal/core/cmath.hpp>        // cardinal scalar math
#include <cardinal/core/cstdlib.hpp>      // cardinal::strtod/strtol
#include <cardinal/core/cstring.hpp>      // cardinal::memcpy
#include <cardinal/core/fstream.hpp>      // cardinal::ifstream/ios
#include <cardinal/core/sstream.hpp>      // cardinal::stringstream
#include <cardinal/core/utility.hpp>      // cardinal::move
#include <cardinal/core/containers.hpp>   // cardinal::unordered_map/vector

#include "json.hpp"                        // cardinal::import::detail::JVal/JParser

namespace cardinal::import {

using cardinal::scene::Vec3;
using cardinal::scene::Vec4;
using cardinal::scene::Mat4;

namespace {

// JSON reader (JVal / JParser) is shared with the Megascans backend; it lives
// in the src-private json.hpp under cardinal::import::detail. Pull it into this
// TU's anonymous namespace so every existing call site stays byte-identical.
using detail::JVal;
using detail::JParser;

// ---------------------------------------------------------------------------
// base64 (for "data:...;base64," buffer URIs).
// ---------------------------------------------------------------------------
cardinal::vector<u8> b64_decode(const cardinal::string& in) {
    static const auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    cardinal::vector<u8> out;
    // `acc` was `int` (signed) — the inner `acc = (acc << 6) | v` shifts
    // accumulated input bits left by 6 per character. After 6 input
    // chars the shifted value is 6*6 = 36 bits wide, exceeding int's
    // 31-bit value range → signed-integer overflow is UNDEFINED
    // BEHAVIOR per [expr.shift]. MSVC happens to wrap (matching the
    // intent — every consumer only reads the low byte via `& 0xFF`),
    // so the bug is benign on the supported toolchain, but UBSan
    // flags it and other compilers may reorder around the UB. Using
    // unsigned makes the shift well-defined mod 2^32; the low-byte
    // extraction is bit-identical to the prior MSVC output, so this
    // is a pure UB elimination with zero behavioural change.
    // Parked-supervised item #2 in feedback_integration_coverage_map.md;
    // closeable inline (no UBSan harness needed) — matches the net
    // wire-encoder pattern in 2fbc346 where enc_s16/enc_u16 had a
    // related UB cast also fixed without UBSan instrumentation.
    int bits = 0;
    unsigned acc = 0;
    for (char c : in) {
        if (c == '=') break;
        int v = val(c);
        if (v < 0) continue;
        acc = (acc << 6) | static_cast<unsigned>(v);
        bits += 6;
        if (bits >= 8) { bits -= 8; out.push_back(static_cast<u8>((acc>>bits)&0xFFu)); }
    }
    return out;
}

cardinal::string dir_of(const cardinal::string& path) {
    const auto s = path.find_last_of("/\\");
    return (s == cardinal::string::npos) ? cardinal::string{} : path.substr(0, s + 1);
}

cardinal::vector<u8> read_file(const cardinal::string& path) {
    cardinal::ifstream f(path, cardinal::ios::binary | cardinal::ios::ate);
    if (!f) return {};
    const cardinal::streamsize n = f.tellg();
    // tellg → -1 on non-seekable stream. The prior `if (n > 0) read`
    // skipped the read but the VECTOR was already constructed at
    // static_cast<size_t>(-1) = SIZE_MAX → bad_alloc on construction.
    // Same 6a1640d guard pattern as cook/shader read_all.
    if (n < 0) return {};
    f.seekg(0);
    cardinal::vector<u8> v(static_cast<cardinal::size_t>(n));
    if (n > 0) f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

// glTF accessor componentType codes.
constexpr int CT_BYTE=5120, CT_UBYTE=5121, CT_SHORT=5122,
              CT_USHORT=5123, CT_UINT=5125, CT_FLOAT=5126;

int comp_count(const cardinal::string& type) {
    if (type == "SCALAR") return 1;
    if (type == "VEC2")   return 2;
    if (type == "VEC3")   return 3;
    if (type == "VEC4")   return 4;
    if (type == "MAT4")   return 16;
    return 1;
}
int comp_size(int ct) {
    switch (ct) {
        case CT_BYTE: case CT_UBYTE:  return 1;
        case CT_SHORT: case CT_USHORT:return 2;
        case CT_UINT: case CT_FLOAT:  return 4;
        default:                      return 4;
    }
}

// Read one normalized/float component as float.
float read_comp(const u8* p, int ct, bool normalized) {
    switch (ct) {
        case CT_FLOAT: { float f; cardinal::memcpy(&f, p, 4); return f; }
        case CT_UBYTE: { u8  u=*p;  return normalized ? u/255.0f   : float(u); }
        case CT_BYTE:  { i8 s; cardinal::memcpy(&s,p,1);
                         return normalized ? cardinal::max(s/127.0f,-1.0f) : float(s); }
        case CT_USHORT:{ u16 u; cardinal::memcpy(&u,p,2);
                         return normalized ? u/65535.0f : float(u); }
        case CT_SHORT: { i16 s; cardinal::memcpy(&s,p,2);
                         return normalized ? cardinal::max(s/32767.0f,-1.0f):float(s);}
        case CT_UINT:  { u32 u; cardinal::memcpy(&u,p,4); return float(u); }
        default:       return 0.0f;
    }
}

// Decoded buffers + the parsed doc, threaded through the readers.
struct GltfCtx {
    JVal                          doc;
    cardinal::vector<cardinal::vector<u8>>  buffers;   // by buffer index
};

// accessor index → flat float array (count * ncomp), row per element.
cardinal::vector<float> read_accessor(const GltfCtx& g, int acc_idx, int& ncomp) {
    cardinal::vector<float> out;
    const JVal* accs = g.doc.find("accessors");
    const JVal* bvs  = g.doc.find("bufferViews");
    if (!accs || acc_idx < 0 || acc_idx >= (int)accs->arr.size()) return out;
    const JVal& a = accs->arr[acc_idx];
    const int ct  = a.find("componentType") ? a.find("componentType")->inum() : CT_FLOAT;
    const int cnt = a.find("count") ? a.find("count")->inum(0) : 0;
    const cardinal::string type = a.find("type") ? a.find("type")->str() : "SCALAR";
    ncomp = comp_count(type);
    const bool norm = a.find("normalized") && a.find("normalized")->b;
    const int  a_off = a.find("byteOffset") ? a.find("byteOffset")->inum(0) : 0;
    const int  bv_i  = a.find("bufferView") ? a.find("bufferView")->inum(-1) : -1;
    if (!bvs || bv_i < 0 || bv_i >= (int)bvs->arr.size()) return out;
    const JVal& bv = bvs->arr[bv_i];
    const int buf_i = bv.find("buffer") ? bv.find("buffer")->inum(-1) : -1;
    const int bv_off= bv.find("byteOffset") ? bv.find("byteOffset")->inum(0) : 0;
    const int stride= bv.find("byteStride") ? bv.find("byteStride")->inum(0) : 0;
    if (buf_i < 0 || buf_i >= (int)g.buffers.size()) return out;
    const cardinal::vector<u8>& buf = g.buffers[buf_i];
    const int csz   = comp_size(ct);
    const int estep = (stride > 0) ? stride : (csz * ncomp);
    // `cnt` comes straight from untrusted JSON. A negative value casts to
    // ~SIZE_MAX (instant bad_alloc on resize); a huge value OOMs. An
    // accessor can't hold more elements than the decoded buffer has bytes
    // (each element consumes >= 1 byte), so bound cnt by the real buffer.
    // Also rejects an unknown `type` (ncomp == 0) before the size math.
    if (cnt < 0 || ncomp <= 0 ||
        static_cast<cardinal::size_t>(cnt) > buf.size()) {
        return out;                       // hostile / impossible accessor
    }
    out.resize(static_cast<cardinal::size_t>(cnt) *
               static_cast<cardinal::size_t>(ncomp));
    for (int i = 0; i < cnt; ++i) {
        const cardinal::size_t base = static_cast<cardinal::size_t>(bv_off) + a_off
                               + static_cast<cardinal::size_t>(i) * estep;
        for (int c = 0; c < ncomp; ++c) {
            const cardinal::size_t off = base + static_cast<cardinal::size_t>(c) * csz;
            out[static_cast<cardinal::size_t>(i) *
                    static_cast<cardinal::size_t>(ncomp) +
                static_cast<cardinal::size_t>(c)]
                = (off + csz <= buf.size())
                ? read_comp(buf.data() + off, ct, norm) : 0.0f;
        }
    }
    return out;
}

Mat4 node_matrix(const JVal& nd) {
    if (const JVal* m = nd.find("matrix")) {
        if (m->is_arr() && m->arr.size() == 16) {
            Mat4 r;  // glTF matrix is column-major → same as Mat4 m[col][row]
            for (int c = 0; c < 4; ++c)
                for (int row = 0; row < 4; ++row)
                    r.m[c][row] = static_cast<float>(m->arr[c*4+row].num());
            return r;
        }
    }
    Vec3 t{0,0,0}, s{1,1,1};
    cardinal::core::Quat q{0,0,0,1};
    if (const JVal* a = nd.find("translation"); a && a->arr.size()==3)
        t = { (float)a->arr[0].num(), (float)a->arr[1].num(), (float)a->arr[2].num() };
    if (const JVal* a = nd.find("scale"); a && a->arr.size()==3)
        s = { (float)a->arr[0].num(), (float)a->arr[1].num(), (float)a->arr[2].num() };
    if (const JVal* a = nd.find("rotation"); a && a->arr.size()==4)
        q = { (float)a->arr[0].num(), (float)a->arr[1].num(),
              (float)a->arr[2].num(), (float)a->arr[3].num() };
    const cardinal::core::Mat3 r3 = cardinal::core::quat_to_mat3(q);
    Mat4 R = Mat4::identity();          // Mat3 row-major → Mat4 col-major
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            R.m[col][row] = r3.m[row][col];
    return Mat4::translation(t) * R * Mat4::scaling(s);
}

}  // namespace

ImportScene import_gltf(const cardinal::string& path, cardinal::string* error) {
    ImportScene scene;
    const bool glb = (detect_format(path) == Format::Glb);
    scene.source_format = glb ? "glb" : "gltf";

    cardinal::vector<u8> file = read_file(path);
    if (file.empty()) {
        if (error) *error = "cannot open '" + path + "'";
        scene.diagnostics = "glTF: open failed";
        return scene;
    }

    GltfCtx g;
    cardinal::vector<u8> glb_bin;            // GLB embedded BIN chunk
    cardinal::string json_text;

    if (glb) {
        if (file.size() < 12 || cardinal::memcmp(file.data(), "glTF", 4) != 0) {
            if (error) *error = "bad GLB header";
            scene.diagnostics = "glTF: invalid .glb";
            return scene;
        }
        cardinal::size_t off = 12;
        while (off + 8 <= file.size()) {
            u32 clen, ctype;
            cardinal::memcpy(&clen,  file.data()+off,   4);
            cardinal::memcpy(&ctype, file.data()+off+4, 4);
            off += 8;
            if (off + clen > file.size()) break;
            if (ctype == 0x4E4F534A)        // 'JSON'
                json_text.assign(reinterpret_cast<char*>(file.data()+off), clen);
            else if (ctype == 0x004E4942)   // 'BIN\0'
                glb_bin.assign(file.begin()+off, file.begin()+off+clen);
            off += clen + ((4 - (clen & 3)) & 3);   // 4-byte aligned
        }
    } else {
        json_text.assign(reinterpret_cast<char*>(file.data()), file.size());
    }

    JParser jp{ json_text.data(), json_text.data() + json_text.size() };
    g.doc = jp.parse();
    if (!jp.ok || !g.doc.is_obj()) {
        if (error) *error = "glTF JSON parse error";
        scene.diagnostics = "glTF: malformed JSON";
        return scene;
    }

    // Resolve buffers (data-URI / external file / GLB BIN).
    const cardinal::string base = dir_of(path);
    if (const JVal* bufs = g.doc.find("buffers")) {
        for (const JVal& b : bufs->arr) {
            const JVal* uri = b.find("uri");
            if (!uri) { g.buffers.push_back(glb_bin); continue; }
            const cardinal::string u = uri->str();
            const auto comma = u.find(',');
            if (u.rfind("data:", 0) == 0 && comma != cardinal::string::npos)
                g.buffers.push_back(b64_decode(u.substr(comma + 1)));
            else
                g.buffers.push_back(read_file(base + u));
        }
    }

    // Materials → metallic-roughness PBR.
    auto tex_uri = [&](int tex_idx) -> cardinal::string {
        const JVal* texs = g.doc.find("textures");
        const JVal* imgs = g.doc.find("images");
        if (!texs || !imgs || tex_idx < 0 ||
            tex_idx >= (int)texs->arr.size()) return {};
        const int src = texs->arr[tex_idx].find("source")
                      ? texs->arr[tex_idx].find("source")->inum(-1) : -1;
        if (src < 0 || src >= (int)imgs->arr.size()) return {};
        const JVal* iu = imgs->arr[src].find("uri");
        return iu ? iu->str() : cardinal::string{};
    };
    // glTF texture-reference object → image uri ({ "index", "texCoord", … }).
    auto tex_ref_uri = [&](const JVal* ref) -> cardinal::string {
        if (!ref) return {};
        return tex_uri(ref->find("index") ? ref->find("index")->inum(-1) : -1);
    };
    if (const JVal* mats = g.doc.find("materials")) {
        for (const JVal& m : mats->arr) {
            ImportMaterial im;
            im.name = m.find("name") ? m.find("name")->str() : "material";
            if (const JVal* pbr = m.find("pbrMetallicRoughness")) {
                if (const JVal* bc = pbr->find("baseColorFactor");
                    bc && bc->arr.size() >= 3)
                    im.base_color = { (float)bc->arr[0].num(),
                                      (float)bc->arr[1].num(),
                                      (float)bc->arr[2].num() };
                if (const JVal* mf = pbr->find("metallicFactor"))
                    im.metallic = (float)mf->num(1.0);
                if (const JVal* rf = pbr->find("roughnessFactor"))
                    im.roughness = (float)rf->num(1.0);
                if (const JVal* bt = pbr->find("baseColorTexture"))
                    im.base_color_texture = tex_ref_uri(bt);
                // metallicRoughnessTexture is ORM-packed (G=roughness, B=metal)
                // — recorded as a single ref + mr_packed, NOT split at import.
                if (const JVal* mrt = pbr->find("metallicRoughnessTexture")) {
                    im.metallic_roughness_texture = tex_ref_uri(mrt);
                    if (!im.metallic_roughness_texture.empty()) im.mr_packed = true;
                }
            }
            if (const JVal* nt = m.find("normalTexture")) {
                im.normal_texture = tex_ref_uri(nt);
                if (const JVal* sc = nt->find("scale"))
                    im.normal_scale = static_cast<float>(sc->num(1.0));
            }
            if (const JVal* ot = m.find("occlusionTexture")) {
                im.occlusion_texture = tex_ref_uri(ot);
                if (const JVal* st = ot->find("strength"))
                    im.occlusion_strength = static_cast<float>(st->num(1.0));
            }
            if (const JVal* et = m.find("emissiveTexture"))
                im.emissive_texture = tex_ref_uri(et);
            if (const JVal* ef = m.find("emissiveFactor");
                ef && ef->arr.size() >= 3) {
                im.emission = { (float)ef->arr[0].num(),
                                (float)ef->arr[1].num(),
                                (float)ef->arr[2].num() };
                const float mx = cardinal::max({im.emission.x, im.emission.y,
                                           im.emission.z});
                im.emission_strength = (mx > 0.0f) ? 1.0f : 0.0f;
            }
            scene.materials.push_back(cardinal::move(im));
        }
    }

    // Meshes → one ImportMesh per triangle primitive.
    // gltf-mesh index → list of resulting ImportScene.meshes indices,
    // so nodes can map their `mesh` to the produced meshes.
    cardinal::vector<cardinal::vector<int>> mesh_prims;
    if (const JVal* meshes = g.doc.find("meshes")) {
        for (const JVal& mesh : meshes->arr) {
            cardinal::vector<int> produced;
            const cardinal::string mname =
                mesh.find("name") ? mesh.find("name")->str() : "mesh";
            const JVal* prims = mesh.find("primitives");
            if (prims) for (const JVal& pr : prims->arr) {
                const int mode = pr.find("mode") ? pr.find("mode")->inum(4) : 4;
                if (mode != 4) continue;          // triangles only
                const JVal* attr = pr.find("attributes");
                if (!attr) continue;
                ImportMesh im;
                im.name     = mname;
                im.material = pr.find("material")
                            ? pr.find("material")->inum(-1) : -1;
                int nc = 0;
                if (const JVal* a = attr->find("POSITION")) {
                    auto v = read_accessor(g, a->inum(-1), nc);
                    for (cardinal::size_t i = 0; i + 2 < v.size(); i += 3)
                        im.positions.push_back({v[i], v[i+1], v[i+2]});
                }
                if (const JVal* a = attr->find("NORMAL")) {
                    auto v = read_accessor(g, a->inum(-1), nc);
                    for (cardinal::size_t i = 0; i + 2 < v.size(); i += 3)
                        im.normals.push_back({v[i], v[i+1], v[i+2]});
                }
                if (const JVal* a = attr->find("TEXCOORD_0")) {
                    auto v = read_accessor(g, a->inum(-1), nc);
                    for (cardinal::size_t i = 0; i + 1 < v.size(); i += 2)
                        im.uvs.push_back({v[i], v[i+1]});
                }
                if (const JVal* a = attr->find("COLOR_0")) {
                    auto v = read_accessor(g, a->inum(-1), nc);
                    if (nc >= 3)
                        for (cardinal::size_t i = 0; i + 2 < v.size(); i += nc)
                            im.colors.push_back({v[i], v[i+1], v[i+2]});
                }
                if (const JVal* idx = pr.find("indices")) {
                    auto v = read_accessor(g, idx->inum(-1), nc);
                    const u32 vcount = static_cast<u32>(im.positions.size());
                    // Indices are untrusted. Reject NaN / negative / >=2^32
                    // (the latter is also static_cast<u32>(float) UB) and
                    // anything past the vertex array, keeping only whole
                    // in-range triangles so no downstream positions[idx]
                    // reads OOB — the importer is the cooker/renderer
                    // trust boundary.
                    auto ridx = [vcount](float f) -> long {
                        if (!(f >= 0.0f) || f >= 4.294967296e9f) return -1;
                        const u32 u = static_cast<u32>(f + 0.5f);
                        return (u < vcount) ? static_cast<long>(u) : -1;
                    };
                    im.indices.reserve(v.size());
                    for (cardinal::size_t t = 0; t + 2 < v.size(); t += 3) {
                        const long a = ridx(v[t]);
                        const long b = ridx(v[t + 1]);
                        const long c = ridx(v[t + 2]);
                        if (a >= 0 && b >= 0 && c >= 0) {
                            im.indices.push_back(static_cast<u32>(a));
                            im.indices.push_back(static_cast<u32>(b));
                            im.indices.push_back(static_cast<u32>(c));
                        }
                    }
                } else {
                    for (u32 i = 0; i < im.positions.size(); ++i)
                        im.indices.push_back(i);
                }
                if (!im.positions.empty() && !im.indices.empty()) {
                    produced.push_back(static_cast<int>(scene.meshes.size()));
                    scene.meshes.push_back(cardinal::move(im));
                }
            }
            mesh_prims.push_back(cardinal::move(produced));
        }
    }

    // Node graph (TRS / matrix → Mat4; mesh refs remapped to produced).
    if (const JVal* nodes = g.doc.find("nodes")) {
        scene.nodes.reserve(nodes->arr.size());
        for (const JVal& nd : nodes->arr) {
            ImportNode in;
            in.name = nd.find("name") ? nd.find("name")->str() : "node";
            in.transform = node_matrix(nd);
            if (const JVal* mi = nd.find("mesh")) {
                const int gm = mi->inum(-1);
                if (gm >= 0 && gm < (int)mesh_prims.size())
                    for (int p : mesh_prims[gm]) in.meshes.push_back(p);
            }
            if (const JVal* ch = nd.find("children"))
                for (const JVal& c : ch->arr) in.children.push_back(c.inum(-1));
            scene.nodes.push_back(cardinal::move(in));
        }
    }
    if (const JVal* scs = g.doc.find("scenes")) {
        const int si = g.doc.find("scene") ? g.doc.find("scene")->inum(0) : 0;
        if (si >= 0 && si < (int)scs->arr.size())
            if (const JVal* rn = scs->arr[si].find("nodes"))
                for (const JVal& r : rn->arr) scene.roots.push_back(r.inum(-1));
    }

    if (scene.meshes.empty()) {
        if (error) *error = "glTF had no triangle geometry";
        scene.diagnostics = "glTF: parsed but produced 0 meshes";
        return scene;
    }
    scene.ok = true;
    cardinal::ostringstream d;
    d << "glTF: " << scene.meshes.size() << " mesh(es), "
      << scene.materials.size() << " material(s), "
      << scene.nodes.size() << " node(s), "
      << scene.total_vertices() << " verts, "
      << scene.total_triangles() << " tris";
    scene.diagnostics = d.str();
    cardinal::log::infof("import", "%s — %s", path.c_str(),
                         scene.diagnostics.c_str());
    return scene;
}

}  // namespace cardinal::import
