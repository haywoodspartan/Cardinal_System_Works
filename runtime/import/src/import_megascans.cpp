// =============================================================================
// Cardinal — DCC importer: Megascans / Quixel Bridge backend.
//
// A Megascans asset is NOT a single file — it is a metadata JSON manifest plus
// a set of texture maps (and, for 3D assets, mesh LODs) sitting in a folder.
// This backend reads the manifest, routes every scanned PBR map onto the
// neutral ImportMaterial slots, and (for 3D assets) recurses through the
// existing OBJ/glTF backends to load the mesh — so it reuses the whole import
// pipeline rather than duplicating geometry parsing.
//
// Tolerance is the whole point: real Bridge exports drift across versions, so
//   • the map array is read from EITHER "maps" OR "components";
//   • a map's path is read from uri / path / file / relativePath / location;
//   • map TYPE strings are normalised (lowercased, spaces/_/- stripped) and
//     routed through a synonym table (albedo→base_color, gloss→roughness with
//     invert, orm/arm/rma→packed metallic-roughness, …);
//   • unknown map types are PRESERVED in unmapped_textures, never dropped;
//   • when the JSON omits a map array, or names a file that is not on disk, a
//     filename-scan fallback infers maps from <name>_<RES>_<MapType>.<ext>.
//
// FBX mesh LODs (which import_file cannot load yet) degrade gracefully: the
// material imports fully, geometry comes back empty with a diagnostics note.
//
// cardinal::import stays core+scene only — directory scanning goes through
// cardinal::fs (the single sanctioned <filesystem> site), not raw std.
// =============================================================================

#include <cardinal/import/import.hpp>
#include "json.hpp"                        // cardinal::import::detail::JVal/JParser

#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/std/filesystem.hpp>   // cardinal::fs / cardinal::error_code
#include <cardinal/core/std/cctype.hpp>       // cardinal::tolower
#include <cardinal/core/std/fstream.hpp>      // cardinal::ifstream/ios
#include <cardinal/core/std/sstream.hpp>      // cardinal::ostringstream
#include <cardinal/core/std/utility.hpp>      // cardinal::move
#include <cardinal/core/std/containers.hpp>   // cardinal::vector

namespace cardinal::import {

namespace fs = cardinal::fs;
using detail::JVal;
using detail::JParser;

namespace {

constexpr auto npos = cardinal::string::npos;

cardinal::string to_lower(cardinal::string s) {
    for (char& c : s) c = static_cast<char>(cardinal::tolower(
        static_cast<unsigned char>(c)));
    return s;
}

// Lowercase + strip spaces/underscores/hyphens — canonicalises a map-type
// string ("Normal_DirectX" → "normaldirectx", "Ambient Occlusion" → "ao"…).
cardinal::string norm_type(const cardinal::string& in) {
    cardinal::string out;
    out.reserve(in.size());
    for (char c : in) {
        if (c == ' ' || c == '_' || c == '-') continue;
        out += static_cast<char>(cardinal::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

// Lowercased extension incl. the dot ("foo/bar.JPG" → ".jpg"); empty if none
// (or if the only dot sits in a directory component).
cardinal::string ext_of(const cardinal::string& p) {
    const auto dot = p.find_last_of('.');
    if (dot == npos) return {};
    const auto sl = p.find_last_of("/\\");
    if (sl != npos && dot < sl) return {};
    return to_lower(p.substr(dot));
}

cardinal::string dir_of(const cardinal::string& p) {
    const auto s = p.find_last_of("/\\");
    return (s == npos) ? cardinal::string{} : p.substr(0, s + 1);
}

cardinal::string file_stem(const cardinal::string& p) {
    cardinal::string fn = p;
    const auto sl = fn.find_last_of("/\\");
    if (sl != npos) fn = fn.substr(sl + 1);
    const auto dot = fn.find_last_of('.');
    return (dot == npos) ? fn : fn.substr(0, dot);
}

bool is_image_ext(const cardinal::string& e) {
    return e == ".jpg" || e == ".jpeg" || e == ".png" || e == ".exr" ||
           e == ".tga" || e == ".tif"  || e == ".tiff" || e == ".bmp";
}

// Read a whole file as text (manifest JSON). Empty on failure/empty file.
cardinal::string read_text(const cardinal::string& path) {
    cardinal::ifstream f(path, cardinal::ios::binary);
    if (!f) return {};
    cardinal::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// First non-empty of a map entry's path aliases.
cardinal::string entry_uri(const JVal& e) {
    const char* keys[] = { "uri", "path", "file", "relativePath", "location" };
    for (const char* k : keys)
        if (const JVal* v = e.find(k)) {
            cardinal::string s = v->str();
            if (!s.empty()) return s;
        }
    return {};
}

bool root_is_manifest(const JVal& d) {
    return d.is_obj() && (d.find("maps") || d.find("components") ||
                          d.find("meshes") || (d.find("id") && d.find("type")));
}

// Route a (normalised type, resolved uri) onto the right ImportMaterial slot.
// set-if-empty so earlier/JSON maps win over later/scan maps. Returns true iff
// it newly filled a recognised slot. allow_unmapped controls whether an
// unrecognised type is preserved (JSON pass) or ignored (filename scan, where
// stray files like previews must not be hoovered up).
bool route_map(ImportMaterial& m, const cardinal::string& nt,
               const cardinal::string& uri, bool allow_unmapped) {
    auto set = [](cardinal::string& slot, const cardinal::string& v) -> bool {
        if (slot.empty()) { slot = v; return true; }
        return false;
    };
    if (nt == "albedo" || nt == "diffuse" || nt == "basecolor" ||
        nt == "color"  || nt == "albedotransmission")
        return set(m.base_color_texture, uri);
    if (nt == "normal" || nt == "normalbump" || nt == "normalgl" ||
        nt == "normalopengl" || nt == "normaldirectx" || nt == "normaldx")
        return set(m.normal_texture, uri);
    if (nt == "bump") {  // bump → normal if free, else height
        if (m.normal_texture.empty()) return set(m.normal_texture, uri);
        return set(m.height_texture, uri);
    }
    if (nt == "roughness")
        return set(m.roughness_texture, uri);
    if (nt == "gloss" || nt == "glossiness") {
        const bool s = set(m.roughness_texture, uri);
        if (s) m.invert_roughness = true;       // gloss = 1 − roughness
        return s;
    }
    if (nt == "metalness" || nt == "metallic" || nt == "metal")
        return set(m.metallic_texture, uri);
    if (nt == "ao" || nt == "ambientocclusion" || nt == "occlusion" || nt == "cavity")
        return set(m.occlusion_texture, uri);
    if (nt == "specular")
        return set(m.specular_texture, uri);
    if (nt == "displacement" || nt == "height")
        return set(m.height_texture, uri);
    if (nt == "emissive" || nt == "emission" || nt == "emit")
        return set(m.emissive_texture, uri);
    if (nt == "opacity" || nt == "mask" || nt == "alpha")
        return set(m.opacity_texture, uri);
    if (nt == "orm" || nt == "arm" || nt == "rma" ||
        nt == "occlusionroughnessmetallic") {
        const bool s = set(m.metallic_roughness_texture, uri);
        if (s) m.mr_packed = true;
        return s;
    }
    if (allow_unmapped) m.unmapped_textures.push_back(uri);
    return false;
}

// Count of distinct resolved maps on a material (for diagnostics).
int slot_count(const ImportMaterial& m) {
    int n = 0;
    const cardinal::string* slots[] = {
        &m.base_color_texture, &m.metallic_roughness_texture, &m.roughness_texture,
        &m.metallic_texture, &m.normal_texture, &m.occlusion_texture,
        &m.emissive_texture, &m.height_texture, &m.specular_texture,
        &m.opacity_texture };
    for (const cardinal::string* s : slots) if (!s->empty()) ++n;
    return n + static_cast<int>(m.unmapped_textures.size());
}

// Parse `jsonPath` and report whether its root looks like a Bridge manifest.
bool sniff_manifest(const cardinal::string& jsonPath) {
    const cardinal::string txt = read_text(jsonPath);
    if (txt.empty()) return false;
    JParser jp{ txt.data(), txt.data() + txt.size() };
    const JVal d = jp.parse();
    return jp.ok && root_is_manifest(d);
}

}  // namespace

// ---------------------------------------------------------------------------
bool is_megascans(const cardinal::string& dir_or_json) {
    if (ext_of(dir_or_json) == ".json") return sniff_manifest(dir_or_json);
    cardinal::error_code ec;
    const fs::path dp(dir_or_json);
    if (!fs::is_directory(dp, ec)) return false;
    for (const auto& entry : fs::directory_iterator(dp, ec)) {
        if (ec) break;
        if (to_lower(entry.path().extension().string()) == ".json" &&
            sniff_manifest(entry.path().string()))
            return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
ImportScene import_megascans(const cardinal::string& path, cardinal::string* error) {
    ImportScene scene;
    scene.source_format = "megascans";
    cardinal::error_code ec;

    // --- STEP 1: locate the manifest + asset directory ----------------------
    cardinal::string manifest, asset_dir;
    if (ext_of(path) == ".json") {
        manifest  = path;
        asset_dir = dir_of(path);
    } else {
        const fs::path dp(path);
        if (fs::is_directory(dp, ec)) {
            asset_dir = path;
            if (!asset_dir.empty() && asset_dir.back() != '/' &&
                asset_dir.back() != '\\')
                asset_dir += '/';
            for (const auto& entry : fs::directory_iterator(dp, ec)) {
                if (ec) break;
                if (to_lower(entry.path().extension().string()) != ".json") continue;
                const cardinal::string j = entry.path().string();
                if (sniff_manifest(j)) { manifest = j; break; }
            }
        }
    }
    if (manifest.empty()) {
        if (error) *error = "megascans: no manifest JSON found at '" + path + "'";
        scene.diagnostics = "megascans: manifest not found";
        return scene;
    }
    if (asset_dir.empty()) asset_dir = dir_of(manifest);

    // --- STEP 2: parse the manifest ----------------------------------------
    const cardinal::string txt = read_text(manifest);
    JParser jp{ txt.data(), txt.data() + txt.size() };
    const JVal doc = jp.parse();
    if (txt.empty() || !jp.ok || !doc.is_obj()) {
        if (error) *error = "megascans: manifest parse failed for '" + manifest + "'";
        scene.diagnostics = "megascans: manifest parse failed";
        return scene;
    }

    const cardinal::string id   = doc.find("id")   ? doc.find("id")->str()   : cardinal::string{};
    cardinal::string       name = doc.find("name") ? doc.find("name")->str() : cardinal::string{};
    if (name.empty()) name = file_stem(manifest);
    const cardinal::string type = to_lower(doc.find("type") ? doc.find("type")->str()
                                                            : cardinal::string{});

    ImportMaterial mat;
    mat.name = name.empty() ? cardinal::string("megascans") : name;

    // --- STEP 3: map resolution from the JSON (maps OR components) ----------
    int maps_from_json = 0;
    const JVal* maps = doc.find("maps");
    if (!maps) maps = doc.find("components");
    const bool maps_present = (maps && maps->is_arr() && !maps->arr.empty());
    if (maps_present) {
        for (const JVal& e : maps->arr) {
            if (!e.is_obj()) continue;
            const cardinal::string raw =
                e.find("type") ? e.find("type")->str()
                               : (e.find("name") ? e.find("name")->str() : cardinal::string{});
            const cardinal::string uri = entry_uri(e);
            if (uri.empty()) continue;
            route_map(mat, norm_type(raw), asset_dir + uri, /*allow_unmapped=*/true);
            ++maps_from_json;
        }
    }

    // --- STEP 4: mesh / LOD resolution (3D assets) -------------------------
    cardinal::string mesh_note;
    if (const JVal* meshes = doc.find("meshes"); meshes && meshes->is_arr()) {
        for (const JVal& me : meshes->arr) {
            // Candidate (uri, lod) list: uris[] / lodList[] / lods[] / flat uri.
            const JVal* uris = me.find("uris");
            if (!uris) uris = me.find("lodList");
            if (!uris) uris = me.find("lods");
            struct Cand { cardinal::string uri; int lod; };
            cardinal::vector<Cand> cands;
            if (uris && uris->is_arr()) {
                for (const JVal& u : uris->arr) {
                    if (u.t == JVal::T::Str) {
                        cands.push_back({ u.str(), 0 });
                    } else if (u.is_obj()) {
                        const cardinal::string uu = entry_uri(u);
                        const int lod = u.find("lod") ? u.find("lod")->inum(0) : 0;
                        if (!uu.empty()) cands.push_back({ uu, lod });
                    }
                }
            }
            if (cands.empty()) {
                const cardinal::string uu = entry_uri(me);
                if (!uu.empty()) cands.push_back({ uu, 0 });
            }
            if (cands.empty()) continue;

            // Pick: prefer a loadable ext (obj/gltf/glb), then the lowest LOD.
            auto loadable = [](const cardinal::string& u) {
                const cardinal::string e = ext_of(u);
                return e == ".obj" || e == ".gltf" || e == ".glb";
            };
            int best = 0;
            for (int i = 1; i < static_cast<int>(cands.size()); ++i) {
                const bool bi = loadable(cands[i].uri), bb = loadable(cands[best].uri);
                if (bi != bb) { if (bi) best = i; }
                else if (cands[i].lod < cands[best].lod) best = i;
            }

            const cardinal::string primary = asset_dir + cands[best].uri;
            if (loadable(cands[best].uri)) {
                cardinal::string suberr;
                ImportScene sub = import_file(primary, &suberr);
                if (sub.ok && !sub.meshes.empty()) {
                    for (auto& sm : sub.meshes) {
                        sm.material = 0;                    // bind to the one material
                        const int mesh_idx = static_cast<int>(scene.meshes.size());
                        ImportNode nd;
                        nd.name = sm.name;
                        nd.meshes.push_back(mesh_idx);
                        scene.roots.push_back(static_cast<int>(scene.nodes.size()));
                        scene.nodes.push_back(cardinal::move(nd));
                        scene.meshes.push_back(cardinal::move(sm));
                    }
                } else if (mesh_note.empty()) {
                    mesh_note = " [mesh LOD import failed: " + suberr + "]";
                }
            } else if (mesh_note.empty()) {
                mesh_note = " [FBX/unknown LOD pending backend: " + cands[best].uri + "]";
            }
        }
    }

    // --- STEP 5: filename-scan fallback ------------------------------------
    // Trigger when the JSON had no usable map array, or albedo is still
    // unresolved, or the resolved albedo file is absent on disk. Fills ONLY
    // empty slots; ignores unrecognised files (previews/thumbnails).
    int maps_from_scan = 0;
    const bool albedo_missing =
        mat.base_color_texture.empty() ||
        !fs::exists(fs::path(mat.base_color_texture), ec);
    if (!maps_present || albedo_missing) {
        const fs::path dp(asset_dir.empty() ? cardinal::string(".") : asset_dir);
        if (fs::is_directory(dp, ec)) {
            for (const auto& entry : fs::directory_iterator(dp, ec)) {
                if (ec) break;
                if (!entry.is_regular_file(ec)) continue;
                if (!is_image_ext(to_lower(entry.path().extension().string()))) continue;
                const cardinal::string full = entry.path().string();
                const auto stem = file_stem(full);
                const auto us = stem.find_last_of('_');
                const cardinal::string tok =
                    norm_type((us == npos) ? stem : stem.substr(us + 1));
                if (route_map(mat, tok, full, /*allow_unmapped=*/false))
                    ++maps_from_scan;
            }
        }
    }

    // --- STEP 6: emit -------------------------------------------------------
    const int total_maps = slot_count(mat);
    const bool produced_geometry = !scene.meshes.empty();
    scene.ok = (total_maps > 0) || produced_geometry;
    scene.materials.push_back(cardinal::move(mat));

    cardinal::ostringstream d;
    d << "megascans: " << (type.empty() ? "asset" : type) << " '" << name << "', "
      << total_maps << " map(s) (" << maps_from_json << " json + "
      << maps_from_scan << " scan), " << scene.meshes.size() << " mesh(es)";
    if (!mesh_note.empty()) d << mesh_note;
    scene.diagnostics = d.str();

    if (!scene.ok && error)
        *error = "megascans: nothing imported (no maps and no geometry) from '"
               + manifest + "'";

    cardinal::log::infof("import", "%s — %s", manifest.c_str(),
                         scene.diagnostics.c_str());
    return scene;
}

}  // namespace cardinal::import
