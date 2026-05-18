#pragma once

// =============================================================================
// Cardinal — Project descriptor + scaffolding.
//
// One Project = a directory tree on disk with a `project.cardinal` manifest
// at the root. Studio creates / opens / saves projects through this module.
//
//   project.cardinal      — TOML-ish text manifest (name, engine version, dirs)
//   src/                  — game-specific C++ (built into a DLL by cppscript)
//   assets/               — source assets (.png, .obj, .hlsl, .wav)
//   cooked/               — build output: per-asset binaries (cook.cpp)
//   pack/                 — build output: streamable .cpk pack files
//   shaders/cache/        — shader::Cache
//   save/                 — runtime save files (serial::save_world)
//
// Templates ship as in-memory string literals — instantiate to a target
// directory and they expand into a working starter project.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cardinal::project {

inline constexpr const char* kManifestFilename = "project.cardinal";
inline constexpr const char* kManifestMagic    = "# Cardinal project v1";

// ---------------------------------------------------------------------------
// Project — the in-memory representation of a project root.
// ---------------------------------------------------------------------------
struct ProjectDirs {
    std::string root;          // absolute path
    std::string src;           // root + "/src"
    std::string assets;        // root + "/assets"
    std::string cooked;        // root + "/cooked"
    std::string pack;          // root + "/pack"
    std::string shaders;       // root + "/shaders"
    std::string shader_cache;  // root + "/shaders/cache"
    std::string save;          // root + "/save"
};

struct ProjectInfo {
    std::string name;          // human-readable
    std::string engine_version;// default "0.1.0"
    std::string author;
    std::string description;

    // Build settings.
    std::string default_pack_name {"main"};   // produces "pack/main.cpk"
    bool        cook_on_save      {true};
    bool        pack_on_cook      {true};
};

class Project {
public:
    static std::shared_ptr<Project> create_at(const std::string& root,
                                              const ProjectInfo& info,
                                              std::string* error_out = nullptr);
    static std::shared_ptr<Project> open(const std::string& root,
                                         std::string* error_out = nullptr);

    bool save(std::string* error_out = nullptr) const;

    const ProjectInfo& info() const noexcept { return info_; }
    ProjectInfo&       info()       noexcept { return info_; }
    const ProjectDirs& dirs() const noexcept { return dirs_; }

    // Walk an asset directory; returns relative paths (root-relative).
    std::vector<std::string> list_source_assets() const;
    std::vector<std::string> list_cooked_assets() const;

private:
    Project() = default;
    ProjectInfo info_{};
    ProjectDirs dirs_{};
};

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------
enum class TemplateKind : u32 { Blank, FirstPerson, TopDown, Cinematic };

const char* template_kind_name(TemplateKind k) noexcept;
const char* template_kind_description(TemplateKind k) noexcept;

struct InstantiateOptions {
    std::string  root;             // absolute target dir (must not exist or empty)
    ProjectInfo  info;
    TemplateKind kind{TemplateKind::Blank};
    bool         overwrite_existing{false};
};

// Returns the new Project on success. Creates the directory tree, drops a
// template `main.cpp`, a sample `material.hlsl`, and an empty `assets/`.
std::shared_ptr<Project> instantiate_template(const InstantiateOptions& opts,
                                              std::string* error_out = nullptr);

// ---------------------------------------------------------------------------
// Recent projects — small registry stored in the user's home so Studio can
// surface a "recent" list. Backed by a plain text file with one path per line.
// ---------------------------------------------------------------------------
class RecentProjects {
public:
    explicit RecentProjects(std::string store_path);

    void add(const std::string& root);
    void remove(const std::string& root);
    void load();
    void save() const;

    const std::vector<std::string>& entries() const noexcept { return entries_; }

private:
    std::string              store_;
    std::vector<std::string> entries_;
};

}  // namespace cardinal::project
