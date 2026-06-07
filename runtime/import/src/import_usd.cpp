// =============================================================================
// Cardinal — DCC importer: Pixar USD backend (USDA ASCII; USDZ + USDC routed).
//
// Hand-rolled + zero-dep, same ethos as the other backends. USD is the modern
// interchange for Maya / 3ds Max / Cinema 4D / Houdini / Omniverse. Three
// containers share this entry point (dispatched by extension + content sniff):
//   * USDA — the ASCII text layer (.usda, or a text .usd). Parsed here: every
//     `def Mesh` prim's points / faceVertexCounts / faceVertexIndices /
//     normals / primvars:st, polygons fan-triangulated like OBJ.
//   * USDZ — a zip wrapping a root layer (.usdz). Extracted + parsed in the
//     next phase (cardinal::core::compress::zip); for now a clean diagnostic.
//   * USDC — the binary "crate" layer (.usdc, or a crate .usd). A versioned
//     TOC of LZ4/integer-compressed sections; out of scope — detected via the
//     "PXR-USDC" magic and rejected cleanly (re-export as .usda / .usdz).
// =============================================================================

#include <cardinal/import/import.hpp>
#include <cardinal/core/log.hpp>
#include <cardinal/core/compress/inflate.hpp>   // USDZ deflate entries

#include <cardinal/core/cctype.hpp>      // cardinal::tolower
#include <cardinal/core/cstdlib.hpp>     // cardinal::strtod / strtol
#include <cardinal/core/cstring.hpp>     // cardinal::memcmp
#include <cardinal/core/fstream.hpp>     // cardinal::ifstream / ios
#include <cardinal/core/sstream.hpp>     // cardinal::ostringstream
#include <cardinal/core/utility.hpp>     // cardinal::move

namespace cardinal::import {

using cardinal::scene::Vec3;

namespace {

constexpr auto npos = cardinal::string::npos;

cardinal::string to_lower(cardinal::string s) {
    for (char& c : s) c = static_cast<char>(cardinal::tolower(
        static_cast<unsigned char>(c)));
    return s;
}
cardinal::string ext_of(const cardinal::string& p) {
    const auto dot = p.find_last_of('.');
    if (dot == npos) return {};
    const auto sl = p.find_last_of("/\\");
    if (sl != npos && dot < sl) return {};
    return to_lower(p.substr(dot));
}

cardinal::string read_text(const cardinal::string& path) {
    cardinal::ifstream f(path, cardinal::ios::binary);
    if (!f) return {};
    cardinal::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Pull all numbers out of an array literal body (commas / parens / brackets
// are separators, so "[(0,0,1),(1,0,0)]" -> 0 0 1 1 0 0).
cardinal::vector<float> parse_floats(const cardinal::string& s) {
    cardinal::vector<float> out;
    const char* p = s.c_str();
    const char* e = p + s.size();
    while (p < e) {
        while (p < e && !(*p == '-' || *p == '+' || *p == '.' ||
                          (*p >= '0' && *p <= '9'))) ++p;
        if (p >= e) break;
        char* endp = nullptr;
        const double v = cardinal::strtod(p, &endp);
        if (endp == p) { ++p; continue; }
        out.push_back(static_cast<float>(v));
        p = endp;
    }
    return out;
}
cardinal::vector<int> parse_ints(const cardinal::string& s) {
    cardinal::vector<int> out;
    const char* p = s.c_str();
    const char* e = p + s.size();
    while (p < e) {
        while (p < e && !(*p == '-' || (*p >= '0' && *p <= '9'))) ++p;
        if (p >= e) break;
        char* endp = nullptr;
        const long v = cardinal::strtol(p, &endp, 10);
        if (endp == p) { ++p; continue; }
        out.push_back(static_cast<int>(v));
        p = endp;
    }
    return out;
}

// The bracketed body of `attr`'s value within [from, to): finds the attribute
// name, the following '=', then the matching [...] (bracket-depth aware).
cardinal::string attr_array(const cardinal::string& t, cardinal::usize from,
                            cardinal::usize to, const char* attr) {
    cardinal::usize p = t.find(attr, from);
    if (p == npos || p >= to) return {};
    cardinal::usize eq = t.find('=', p);
    if (eq == npos || eq >= to) return {};
    cardinal::usize lb = t.find('[', eq);
    if (lb == npos || lb >= to) return {};
    int depth = 0;
    cardinal::usize i = lb;
    for (; i < to; ++i) {
        if (t[i] == '[') ++depth;
        else if (t[i] == ']') { if (--depth == 0) break; }
    }
    if (i >= to) return {};
    return t.substr(lb + 1, i - lb - 1);
}

// Find the matching '}' for the '{' at/after `open`; returns npos if none.
cardinal::usize match_brace(const cardinal::string& t, cardinal::usize open) {
    cardinal::usize lb = t.find('{', open);
    if (lb == npos) return npos;
    int depth = 0;
    for (cardinal::usize i = lb; i < t.size(); ++i) {
        if (t[i] == '{') ++depth;
        else if (t[i] == '}') { if (--depth == 0) return i; }
    }
    return npos;
}

bool endswith(const cardinal::string& s, const char* suf) {
    const cardinal::string t = suf;
    return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
}

// Extract the USDZ root layer from a zip (STORED or DEFLATE entries). USDZ
// mandates a STORED+aligned root, but deflated entries are handled too via the
// inflate keystone. Walks the End-Of-Central-Directory -> central directory ->
// the first .usd/.usda/.usdc entry's local header -> its data. Bounds-checked.
bool usdz_extract_root(const cardinal::string& z, cardinal::string& outName,
                       cardinal::string& outData, cardinal::string& err) {
    const u8* b = reinterpret_cast<const u8*>(z.data());
    const cardinal::usize n = z.size();
    if (n < 22) { err = "USDZ: file too small for a zip"; return false; }
    auto rd16 = [&](cardinal::usize o) -> u32 {
        return static_cast<u32>(b[o]) | (static_cast<u32>(b[o + 1]) << 8);
    };
    auto rd32 = [&](cardinal::usize o) -> u32 {
        return static_cast<u32>(b[o]) | (static_cast<u32>(b[o + 1]) << 8) |
               (static_cast<u32>(b[o + 2]) << 16) | (static_cast<u32>(b[o + 3]) << 24);
    };
    // Find the End-Of-Central-Directory record (sig 0x06054b50), scanning back.
    cardinal::usize eocd = npos;
    for (cardinal::usize i = n - 22 + 1; i-- > 0; ) {
        if (i + 4 <= n && rd32(i) == 0x06054b50u) { eocd = i; break; }
    }
    if (eocd == npos || eocd + 20 > n) { err = "USDZ: no zip EOCD record"; return false; }
    const u32 cdCount = rd16(eocd + 10);
    const u32 cdOff   = rd32(eocd + 16);

    cardinal::usize p = cdOff;
    for (u32 i = 0; i < cdCount; ++i) {
        if (p + 46 > n || rd32(p) != 0x02014b50u) break;       // central-dir record
        const u32 method     = rd16(p + 10);
        const u32 compSize   = rd32(p + 20);
        const u32 uncompSize = rd32(p + 24);
        const u32 nameLen    = rd16(p + 28);
        const u32 extraLen   = rd16(p + 30);
        const u32 cmtLen     = rd16(p + 32);
        const u32 lho        = rd32(p + 42);                   // local header offset
        if (p + 46 + nameLen > n) break;
        cardinal::string name(reinterpret_cast<const char*>(b + p + 46), nameLen);
        p += 46 + nameLen + extraLen + cmtLen;

        const cardinal::string ln = to_lower(name);
        if (!(endswith(ln, ".usda") || endswith(ln, ".usd") || endswith(ln, ".usdc")))
            continue;                                          // first layer wins
        if (static_cast<cardinal::usize>(lho) + 30 > n || rd32(lho) != 0x04034b50u) {
            err = "USDZ: bad local file header"; return false;
        }
        const u32 lnl = rd16(lho + 26), lel = rd16(lho + 28);
        const cardinal::usize dataOff =
            static_cast<cardinal::usize>(lho) + 30 + lnl + lel;
        if (dataOff + compSize > n) { err = "USDZ: truncated entry data"; return false; }
        outName = name;
        if (method == 0) {                                     // STORED
            outData.assign(reinterpret_cast<const char*>(b + dataOff), compSize);
            return true;
        }
        if (method == 8) {                                     // DEFLATE
            outData.resize(uncompSize);
            const cardinal::usize got = cardinal::core::compress::inflate_raw(
                b + dataOff, compSize,
                reinterpret_cast<u8*>(outData.data()), uncompSize);
            if (got != uncompSize) { err = "USDZ: entry inflate failed"; return false; }
            return true;
        }
        err = "USDZ: unsupported zip compression method"; return false;
    }
    err = "USDZ: no .usd/.usda/.usdc root layer in archive";
    return false;
}

void parse_usda(const cardinal::string& text, ImportScene& scene) {
    cardinal::usize search = 0;
    while (true) {
        const cardinal::usize d = text.find("def Mesh", search);
        if (d == npos) break;
        const cardinal::usize open  = text.find('{', d);
        const cardinal::usize close = match_brace(text, d);
        if (open == npos || close == npos) break;
        search = close + 1;

        const cardinal::vector<float> pts  = parse_floats(attr_array(text, open, close, "points"));
        const cardinal::vector<int>   fvc  = parse_ints  (attr_array(text, open, close, "faceVertexCounts"));
        const cardinal::vector<int>   fvi  = parse_ints  (attr_array(text, open, close, "faceVertexIndices"));
        const cardinal::vector<float> nrm  = parse_floats(attr_array(text, open, close, "normals"));
        const cardinal::vector<float> st   = parse_floats(attr_array(text, open, close, "primvars:st"));

        if (pts.size() < 3 || fvi.empty()) continue;

        ImportMesh m;
        m.name = "usd_mesh";
        for (cardinal::usize k = 0; k + 3 <= pts.size(); k += 3)
            m.positions.push_back(Vec3{ pts[k], pts[k + 1], pts[k + 2] });
        const u32 vcount = static_cast<u32>(m.positions.size());

        // Fan-triangulate each polygon (faceVertexCounts gives polygon sizes).
        cardinal::usize cursor = 0;
        auto emit_tri = [&](int a, int b, int c) {
            if (a >= 0 && b >= 0 && c >= 0 &&
                static_cast<u32>(a) < vcount && static_cast<u32>(b) < vcount &&
                static_cast<u32>(c) < vcount) {
                m.indices.push_back(static_cast<u32>(a));
                m.indices.push_back(static_cast<u32>(b));
                m.indices.push_back(static_cast<u32>(c));
            }
        };
        if (!fvc.empty()) {
            for (int c : fvc) {
                if (c < 3) { cursor += (c > 0 ? static_cast<cardinal::usize>(c) : 0); continue; }
                for (int t = 2; t < c; ++t) {
                    if (cursor + static_cast<cardinal::usize>(t) >= fvi.size()) break;
                    emit_tri(fvi[cursor], fvi[cursor + t - 1], fvi[cursor + t]);
                }
                cursor += static_cast<cardinal::usize>(c);
            }
        } else if (fvi.size() % 3 == 0) {            // no counts: assume triangles
            for (cardinal::usize k = 0; k + 3 <= fvi.size(); k += 3)
                emit_tri(fvi[k], fvi[k + 1], fvi[k + 2]);
        }

        // Per-vertex normals / UVs only (faceVarying de-dup deferred).
        if (nrm.size() / 3 == m.positions.size())
            for (cardinal::usize k = 0; k + 3 <= nrm.size(); k += 3)
                m.normals.push_back(Vec3{ nrm[k], nrm[k + 1], nrm[k + 2] });
        if (st.size() / 2 == m.positions.size())
            for (cardinal::usize k = 0; k + 2 <= st.size(); k += 2)
                m.uvs.push_back(Vec2{ st[k], st[k + 1] });

        if (!m.positions.empty() && !m.indices.empty())
            scene.meshes.push_back(cardinal::move(m));
    }
}

}  // namespace

ImportScene import_usd(const cardinal::string& path, cardinal::string* error) {
    ImportScene scene;
    scene.source_format = "usd";
    const cardinal::string e = ext_of(path);

    // Obtain the USDA text directly (.usda/.usd) or by extracting the root
    // layer from a USDZ zip (STORED or DEFLATE).
    cardinal::string text;
    cardinal::string usdz_name;
    bool from_usdz = false;
    if (e == ".usdz") {
        const cardinal::string zip = read_text(path);
        if (zip.empty()) {
            if (error) *error = "cannot open '" + path + "'";
            scene.diagnostics = "USDZ: open failed";
            return scene;
        }
        cardinal::string xerr;
        if (!usdz_extract_root(zip, usdz_name, text, xerr)) {
            if (error) *error = xerr;
            scene.diagnostics = xerr;
            return scene;
        }
        from_usdz = true;
    } else {
        text = read_text(path);
        if (text.empty()) {
            if (error) *error = "cannot open '" + path + "'";
            scene.diagnostics = "USD: open failed";
            return scene;
        }
    }
    // USDC binary crate — reject cleanly (direct .usdc, .usdc inside a USDZ,
    // or detected by the crate magic).
    if (e == ".usdc" ||
        (from_usdz && endswith(to_lower(usdz_name), ".usdc")) ||
        (text.size() >= 8 && cardinal::memcmp(text.data(), "PXR-USDC", 8) == 0)) {
        if (error) *error = "USDC binary crate not supported — "
                            "re-export as .usda or a stored .usdz";
        scene.diagnostics = "USDC: binary crate unsupported";
        return scene;
    }

    parse_usda(text, scene);

    if (scene.meshes.empty()) {
        if (error) *error = "USD: no def Mesh geometry found";
        scene.diagnostics = "USDA: parsed but produced 0 meshes";
        return scene;
    }
    for (cardinal::usize i = 0; i < scene.meshes.size(); ++i) {
        ImportNode nd;
        nd.name = scene.meshes[i].name;
        nd.meshes.push_back(static_cast<int>(i));
        scene.roots.push_back(static_cast<int>(scene.nodes.size()));
        scene.nodes.push_back(cardinal::move(nd));
    }
    scene.source_format = from_usdz ? "usdz" : "usda";
    scene.ok = true;
    cardinal::ostringstream d;
    d << (from_usdz ? "USDZ" : "USDA") << ": " << scene.meshes.size() << " mesh(es), "
      << scene.total_vertices() << " verts, " << scene.total_triangles() << " tris";
    scene.diagnostics = d.str();
    cardinal::log::infof("import", "%s — %s", path.c_str(), scene.diagnostics.c_str());
    return scene;
}

}  // namespace cardinal::import
