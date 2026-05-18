#include <cardinal/project/project.hpp>

#include <cardinal/core/log.hpp>

#include <cardinal/core/algorithm.hpp>    // cardinal::sort/remove
#include <cardinal/core/cstdio.hpp>       // cardinal::snprintf
#include <cardinal/core/filesystem.hpp>   // cardinal::fs, error_code
#include <cardinal/core/fstream.hpp>      // ifstream/ofstream/ios/getline
#include <cardinal/core/sstream.hpp>      // cardinal::stringstream
#include <cardinal/core/utility.hpp>      // cardinal::move
#include <cardinal/core/containers.hpp>   // cardinal::vector

namespace fs = cardinal::fs;

namespace cardinal::project {

namespace {

void fill_dirs(const cardinal::string& root, ProjectDirs& d) {
    d.root         = root;
    d.src          = root + "/src";
    d.assets       = root + "/assets";
    d.cooked       = root + "/cooked";
    d.pack         = root + "/pack";
    d.shaders      = root + "/shaders";
    d.shader_cache = root + "/shaders/cache";
    d.save         = root + "/save";
}

bool make_dirs(const ProjectDirs& d, cardinal::string* err) {
    cardinal::error_code ec;
    fs::create_directories(d.root,         ec); if (ec) { if (err) *err = ec.message(); return false; }
    fs::create_directories(d.src,          ec);
    fs::create_directories(d.assets,       ec);
    fs::create_directories(d.cooked,       ec);
    fs::create_directories(d.pack,         ec);
    fs::create_directories(d.shaders,      ec);
    fs::create_directories(d.shader_cache, ec);
    fs::create_directories(d.save,         ec);
    return true;
}

cardinal::string trim(cardinal::string s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
    while (!s.empty() && (s.back()  == ' ' || s.back()  == '\t' || s.back() == '\r' || s.back() == '\n')) s.pop_back();
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') s = s.substr(1, s.size() - 2);
    return s;
}

}  // namespace

const char* template_kind_name(TemplateKind k) noexcept {
    switch (k) {
        case TemplateKind::Blank:       return "Blank";
        case TemplateKind::FirstPerson: return "First Person";
        case TemplateKind::TopDown:     return "Top-Down";
        case TemplateKind::Cinematic:   return "Cinematic";
    }
    return "?";
}

const char* template_kind_description(TemplateKind k) noexcept {
    switch (k) {
        case TemplateKind::Blank:
            return "Empty project — one main.cpp + assets/ scaffold + sample HLSL.";
        case TemplateKind::FirstPerson:
            return "FPS skeleton — WASD + mouse look, capsule character, jump.";
        case TemplateKind::TopDown:
            return "Top-down — orbit camera, click-to-move via cardinal::nav.";
        case TemplateKind::Cinematic:
            return "Camera-only — pre-built sequencer, no input. For trailers.";
    }
    return "";
}

// ---------------------------------------------------------------------------
// Project create / open / save
// ---------------------------------------------------------------------------
cardinal::shared_ptr<Project> Project::create_at(const cardinal::string& root,
                                            const ProjectInfo& info,
                                            cardinal::string* err)
{
    auto p = cardinal::shared_ptr<Project>(new Project());
    p->info_ = info;
    fill_dirs(root, p->dirs_);
    if (!make_dirs(p->dirs_, err)) return nullptr;
    if (!p->save(err)) return nullptr;
    cardinal::log::infof("project", "created project '%s' at %s",
                         info.name.c_str(), root.c_str());
    return p;
}

cardinal::shared_ptr<Project> Project::open(const cardinal::string& root, cardinal::string* err) {
    const cardinal::string path = root + "/" + kManifestFilename;
    cardinal::ifstream f(path);
    if (!f) {
        if (err) *err = "no project manifest at " + path;
        return nullptr;
    }
    auto p = cardinal::shared_ptr<Project>(new Project());
    fill_dirs(root, p->dirs_);
    cardinal::string line;
    while (cardinal::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line.front() == '#') continue;
        const auto eq = line.find('=');
        if (eq == cardinal::string::npos) continue;
        const cardinal::string key = trim(line.substr(0, eq));
        const cardinal::string val = trim(line.substr(eq + 1));
        if (key == "name")            p->info_.name = val;
        else if (key == "engine")     p->info_.engine_version = val;
        else if (key == "author")     p->info_.author = val;
        else if (key == "description")p->info_.description = val;
        else if (key == "default_pack_name") p->info_.default_pack_name = val;
        else if (key == "cook_on_save") p->info_.cook_on_save = (val == "true");
        else if (key == "pack_on_cook") p->info_.pack_on_cook = (val == "true");
    }
    cardinal::log::infof("project", "opened project '%s'", p->info_.name.c_str());
    return p;
}

bool Project::save(cardinal::string* err) const {
    const cardinal::string path = dirs_.root + "/" + kManifestFilename;
    cardinal::ofstream f(path, cardinal::ios::binary | cardinal::ios::trunc);
    if (!f) {
        if (err) *err = "could not open " + path + " for write";
        return false;
    }
    f << kManifestMagic << "\n\n";
    f << "name        = \"" << info_.name           << "\"\n";
    f << "engine      = \"" << info_.engine_version << "\"\n";
    f << "author      = \"" << info_.author         << "\"\n";
    f << "description = \"" << info_.description    << "\"\n";
    f << "default_pack_name = \"" << info_.default_pack_name << "\"\n";
    f << "cook_on_save = " << (info_.cook_on_save ? "true" : "false") << "\n";
    f << "pack_on_cook = " << (info_.pack_on_cook ? "true" : "false") << "\n";
    return true;
}

cardinal::vector<cardinal::string> Project::list_source_assets() const {
    cardinal::vector<cardinal::string> out;
    cardinal::error_code ec;
    if (!fs::exists(dirs_.assets, ec)) return out;
    for (auto& e : fs::recursive_directory_iterator(dirs_.assets, ec)) {
        if (ec) break;
        if (!e.is_regular_file()) continue;
        out.push_back(fs::relative(e.path(), dirs_.assets).string());
    }
    cardinal::sort(out.begin(), out.end());
    return out;
}

cardinal::vector<cardinal::string> Project::list_cooked_assets() const {
    cardinal::vector<cardinal::string> out;
    cardinal::error_code ec;
    if (!fs::exists(dirs_.cooked, ec)) return out;
    for (auto& e : fs::recursive_directory_iterator(dirs_.cooked, ec)) {
        if (ec) break;
        if (!e.is_regular_file()) continue;
        if (e.path().extension() != ".cooked") continue;
        out.push_back(fs::relative(e.path(), dirs_.cooked).string());
    }
    cardinal::sort(out.begin(), out.end());
    return out;
}

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------
namespace {

const char* kTemplateMainBlank = R"~~~(// Generated by Cardinal — blank project template.
//
// This file is the entry point your game's C++ runtime is built from.
// cardinal::cppscript can compile + load it at runtime, or your build
// system can compile it into a DLL the engine loads at boot.

#include <cardinal/game/game_actor.hpp>
#include <cardinal/game/reflection.hpp>

class HelloActor final : public cardinal::game::GameActor {
public:
    float        speed{1.0f};
    bool         visible{true};
    cardinal::scene::Vec3 axis{0.0f, 1.0f, 0.0f};

    void begin_play() override {}
    void on_tick(float /*dt*/) override {}
};

CARDINAL_REGISTER_GAME_CLASS(HelloActor, "Game/Hello",
    PROP_FLOAT(speed,   0.0f, 100.0f, "Movement speed (units/s)")
    PROP_BOOL (visible,             "Render this actor")
    PROP_VEC3 (axis,                "Reference axis"))
)~~~";

const char* kTemplateShaderHLSL = R"~~~(// Generated material.hlsl — runs through the cook + shader::Compiler.
struct VSIn  { float3 pos : POSITION; float3 col : COLOR; };
struct VSOut { float4 pos : SV_POSITION; float3 col : COLOR; };

VSOut VSMain(VSIn v) {
    VSOut o;
    o.pos = float4(v.pos, 1.0);
    o.col = v.col;
    return o;
}
float4 PSMain(VSOut i) : SV_TARGET {
    return float4(i.col, 1.0);
}
)~~~";

}  // namespace

cardinal::shared_ptr<Project> instantiate_template(const InstantiateOptions& opts,
                                              cardinal::string* err)
{
    cardinal::error_code ec;
    if (fs::exists(opts.root) && !opts.overwrite_existing) {
        // Allow when the directory is empty.
        bool empty = true;
        auto it = fs::directory_iterator(opts.root, ec);
        if (!ec && it != fs::directory_iterator()) empty = false;
        if (!empty) {
            if (err) *err = "directory already exists and is non-empty: " + opts.root;
            return nullptr;
        }
    }

    auto p = Project::create_at(opts.root, opts.info, err);
    if (p == nullptr) return nullptr;

    // Drop template files. Switch on `kind` here for richer scaffolds.
    auto write_text = [&](const cardinal::string& path, const cardinal::string& text) {
        cardinal::ofstream f(path, cardinal::ios::binary | cardinal::ios::trunc);
        if (!f) return false;
        f.write(text.data(), static_cast<cardinal::streamsize>(text.size()));
        return true;
    };

    write_text(p->dirs().src     + "/main.cpp",       kTemplateMainBlank);
    write_text(p->dirs().shaders + "/material.hlsl",  kTemplateShaderHLSL);
    write_text(p->dirs().assets  + "/README.txt",
        "Drop your source assets here (.png .obj .hlsl .wav) — they'll be cooked\n"
        "into ../cooked/ + packed into ../pack/ when you hit Cook & Pack.\n");

    cardinal::log::infof("project",
        "instantiated template '%s' at %s",
        template_kind_name(opts.kind), opts.root.c_str());
    return p;
}

// ---------------------------------------------------------------------------
// RecentProjects
// ---------------------------------------------------------------------------
RecentProjects::RecentProjects(cardinal::string store_path)
    : store_(cardinal::move(store_path)) {}

void RecentProjects::add(const cardinal::string& root) {
    entries_.erase(cardinal::remove(entries_.begin(), entries_.end(), root), entries_.end());
    entries_.insert(entries_.begin(), root);
    if (entries_.size() > 16) entries_.resize(16);
}
void RecentProjects::remove(const cardinal::string& root) {
    entries_.erase(cardinal::remove(entries_.begin(), entries_.end(), root), entries_.end());
}
void RecentProjects::load() {
    entries_.clear();
    cardinal::ifstream f(store_);
    if (!f) return;
    cardinal::string line;
    while (cardinal::getline(f, line)) if (!line.empty()) entries_.push_back(line);
}
void RecentProjects::save() const {
    cardinal::ofstream f(store_, cardinal::ios::binary | cardinal::ios::trunc);
    for (const auto& e : entries_) f << e << "\n";
}

}  // namespace cardinal::project
