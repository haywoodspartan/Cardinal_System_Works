// =============================================================================
// Cardinal — Asset Database implementation (see assetdb.hpp).
// =============================================================================
#include <cardinal/assetdb/assetdb.hpp>

#include <cardinal/asset/asset.hpp>          // Registry, MaterialAsset
#include <cardinal/core/std/algorithm.hpp>   // cardinal::sort
#include <cardinal/core/std/utility.hpp>     // cardinal::move
#include <cardinal/core/std/fstream.hpp>     // ifstream/ofstream
#include <cardinal/core/std/filesystem.hpp>

namespace cardinal::assetdb {

namespace {

cardinal::vector<cardinal::string> split(const cardinal::string& s, char d) {
    cardinal::vector<cardinal::string> out;
    cardinal::string cur;
    for (char c : s) {
        if (c == d) { out.push_back(cur); cur.clear(); }
        else        { cur += c; }
    }
    out.push_back(cur);
    return out;
}

// No-throw decimal parse (our own serialized format; tolerant of junk).
u64 parse_u64(const cardinal::string& s) noexcept {
    u64 v = 0;
    for (char c : s) { if (c < '0' || c > '9') break; v = v * 10 + static_cast<u64>(c - '0'); }
    return v;
}

void add_dep_unique(cardinal::vector<cardinal::string>& deps, const cardinal::string& k) {
    if (k.empty()) return;
    for (const auto& d : deps) if (d == k) return;
    deps.push_back(k);
}

}  // namespace

u64 asset_id(const cardinal::string& key) noexcept {
    u64 h = 1469598103934665603ull;            // FNV-1a 64 offset basis
    for (unsigned char c : key) { h ^= c; h *= 1099511628211ull; }
    return h;
}

void AssetDatabase::add(AssetRecord rec) {
    auto it = by_key_.find(rec.key);
    if (it != by_key_.end()) { records_[it->second] = cardinal::move(rec); return; }
    by_key_.emplace(rec.key, records_.size());
    records_.push_back(cardinal::move(rec));
}

void AssetDatabase::clear() noexcept { records_.clear(); by_key_.clear(); }

const AssetRecord* AssetDatabase::find(const cardinal::string& key) const noexcept {
    auto it = by_key_.find(key);
    return it == by_key_.end() ? nullptr : &records_[it->second];
}

const AssetRecord* AssetDatabase::find_by_id(u64 id) const noexcept {
    for (const auto& r : records_) if (r.id == id) return &r;
    return nullptr;
}

cook::AssetType AssetDatabase::type_of(const cardinal::string& key) const noexcept {
    const AssetRecord* r = find(key);
    return r ? r->type : cook::AssetType::Unknown;
}

cardinal::vector<cardinal::string> AssetDatabase::all_keys() const {
    cardinal::vector<cardinal::string> out;
    out.reserve(records_.size());
    for (const auto& r : records_) out.push_back(r.key);
    cardinal::sort(out.begin(), out.end());
    return out;
}

cardinal::vector<const AssetRecord*> AssetDatabase::by_type(cook::AssetType t) const {
    cardinal::vector<const AssetRecord*> out;
    for (const auto& r : records_) if (r.type == t) out.push_back(&r);
    return out;
}

cardinal::vector<cardinal::string> AssetDatabase::dependencies(const cardinal::string& key) const {
    const AssetRecord* r = find(key);
    return r ? r->deps : cardinal::vector<cardinal::string>{};
}

cardinal::vector<cardinal::string> AssetDatabase::dependents(const cardinal::string& key) const {
    cardinal::vector<cardinal::string> out;
    for (const auto& r : records_)
        for (const auto& d : r.deps)
            if (d == key) { out.push_back(r.key); break; }
    return out;
}

cardinal::vector<cardinal::string> AssetDatabase::missing_dependencies() const {
    cardinal::vector<cardinal::string> out;
    for (const auto& r : records_)
        for (const auto& d : r.deps)
            if (find(d) == nullptr) add_dep_unique(out, d);
    return out;
}

cardinal::string AssetDatabase::serialize() const {
    cardinal::string out = "# Cardinal assetdb v1\n";
    for (const auto& r : records_) {
        out += "asset\t" + r.key + "\t" + cardinal::to_string(r.id) + "\t"
             + cardinal::to_string(static_cast<u32>(r.type)) + "\t"
             + cardinal::to_string(r.source_hash) + "\t" + r.source + "\t";
        for (cardinal::usize i = 0; i < r.deps.size(); ++i) {
            if (i != 0) out += ",";
            out += r.deps[i];
        }
        out += "\n";
    }
    return out;
}

bool AssetDatabase::deserialize(const cardinal::string& text, AssetDatabase& out) {
    out.clear();
    for (const auto& ln : split(text, '\n')) {
        if (ln.empty() || ln[0] == '#') continue;
        const auto f = split(ln, '\t');
        if (f.size() < 6 || f[0] != "asset") continue;
        AssetRecord r;
        r.key         = f[1];
        r.id          = parse_u64(f[2]);
        r.type        = static_cast<cook::AssetType>(static_cast<u32>(parse_u64(f[3])));
        r.source_hash = parse_u64(f[4]);
        r.source      = f[5];
        if (f.size() >= 7 && !f[6].empty())
            for (const auto& d : split(f[6], ',')) add_dep_unique(r.deps, d);
        out.add(cardinal::move(r));
    }
    return true;
}

AssetDatabase build_database(const cardinal::vector<cook::CookResult>& results,
                             cardinal::asset::Registry& registry) {
    AssetDatabase db;
    for (const auto& res : results) {
        if (res.status == cook::CookResult::Status::Failed) continue;

        // Key = cooked relpath minus the ".cooked" suffix (== source relpath).
        cardinal::string key = res.cooked_relpath;
        if (key.size() > 7 && key.substr(key.size() - 7) == ".cooked")
            key = key.substr(0, key.size() - 7);
        else
            key = res.source_relpath;

        AssetRecord rec;
        rec.key         = key;
        rec.id          = asset_id(key);
        rec.type        = res.type;
        rec.source      = res.source_relpath;
        rec.source_hash = res.source_hash;

        // Materials reference textures — record those as dependencies.
        if (res.type == cook::AssetType::Material) {
            if (auto mat = registry.load_material(key)) {
                add_dep_unique(rec.deps, mat->base_color_texture);
                add_dep_unique(rec.deps, mat->metallic_roughness_texture);
                add_dep_unique(rec.deps, mat->roughness_texture);
                add_dep_unique(rec.deps, mat->metallic_texture);
                add_dep_unique(rec.deps, mat->normal_texture);
                add_dep_unique(rec.deps, mat->occlusion_texture);
                add_dep_unique(rec.deps, mat->emissive_texture);
                add_dep_unique(rec.deps, mat->height_texture);
                add_dep_unique(rec.deps, mat->specular_texture);
                add_dep_unique(rec.deps, mat->opacity_texture);
            }
        }

        db.add(cardinal::move(rec));
    }
    return db;
}

bool save_database(const AssetDatabase& db, const cardinal::string& path) {
    cardinal::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    cardinal::ofstream f(path, cardinal::ios::binary | cardinal::ios::trunc);
    if (!f) return false;
    const cardinal::string s = db.serialize();
    f.write(s.data(), static_cast<cardinal::streamsize>(s.size()));
    return static_cast<bool>(f);
}

bool load_database(const cardinal::string& path, AssetDatabase& out) {
    cardinal::ifstream f(path, cardinal::ios::binary);
    if (!f) return false;
    f.seekg(0, cardinal::ios::end);
    const cardinal::streamoff sz = f.tellg();
    if (sz < 0) return false;
    f.seekg(0, cardinal::ios::beg);
    cardinal::string s;
    s.resize(static_cast<cardinal::usize>(sz));
    if (sz > 0) f.read(&s[0], static_cast<cardinal::streamsize>(sz));
    return AssetDatabase::deserialize(s, out);
}

}  // namespace cardinal::assetdb
