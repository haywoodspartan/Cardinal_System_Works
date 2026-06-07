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

#include <cardinal/core/types.hpp>        // function/memory/string
#include <cardinal/core/std/containers.hpp>   // cardinal::vector

namespace cardinal::project {

inline constexpr const char* kManifestFilename = "project.cardinal";
inline constexpr const char* kManifestMagic    = "# Cardinal project v1";

// ---------------------------------------------------------------------------
// Project — the in-memory representation of a project root.
// ---------------------------------------------------------------------------
struct ProjectDirs {
    cardinal::string root;          // absolute path
    cardinal::string src;           // root + "/src"
    cardinal::string assets;        // root + "/assets"
    cardinal::string cooked;        // root + "/cooked"
    cardinal::string pack;          // root + "/pack"
    cardinal::string shaders;       // root + "/shaders"
    cardinal::string shader_cache;  // root + "/shaders/cache"
    cardinal::string save;          // root + "/save"
};

struct ProjectInfo {
    cardinal::string name;          // human-readable
    cardinal::string engine_version;// default "0.1.0"
    cardinal::string author;
    cardinal::string description;

    // Build settings.
    cardinal::string default_pack_name {"main"};   // produces "pack/main.cpk"
    bool        cook_on_save      {true};
    bool        pack_on_cook      {true};

    // Runtime entry point: the world snapshot the engine/Studio loads when
    // the project boots. Root-relative; a fresh project ships a valid
    // default at this path (serial "# Cardinal save v1" format) so the
    // project is runnable the moment it is created.
    cardinal::string startup_world {"save/main.cardinalworld"};
};

class Project {
public:
    static cardinal::shared_ptr<Project> create_at(const cardinal::string& root,
                                              const ProjectInfo& info,
                                              cardinal::string* error_out = nullptr);
    static cardinal::shared_ptr<Project> open(const cardinal::string& root,
                                         cardinal::string* error_out = nullptr);

    bool save(cardinal::string* error_out = nullptr) const;

    const ProjectInfo& info() const noexcept { return info_; }
    ProjectInfo&       info()       noexcept { return info_; }
    const ProjectDirs& dirs() const noexcept { return dirs_; }

    // Walk an asset directory; returns relative paths (root-relative).
    cardinal::vector<cardinal::string> list_source_assets() const;
    cardinal::vector<cardinal::string> list_cooked_assets() const;

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
    cardinal::string  root;             // absolute target dir (must not exist or empty)
    ProjectInfo  info;
    TemplateKind kind{TemplateKind::Blank};
    bool         overwrite_existing{false};
};

// Returns the new Project on success. Creates the directory tree, drops a
// template `main.cpp`, a sample `material.hlsl`, and an empty `assets/`.
cardinal::shared_ptr<Project> instantiate_template(const InstantiateOptions& opts,
                                              cardinal::string* error_out = nullptr);

// ---------------------------------------------------------------------------
// Recent projects — small registry stored in the user's home so Studio can
// surface a "recent" list. Backed by a plain text file with one path per line.
// ---------------------------------------------------------------------------
class RecentProjects {
public:
    explicit RecentProjects(cardinal::string store_path);

    void add(const cardinal::string& root);
    void remove(const cardinal::string& root);
    void load();
    void save() const;

    const cardinal::vector<cardinal::string>& entries() const noexcept { return entries_; }

private:
    cardinal::string              store_;
    cardinal::vector<cardinal::string> entries_;
};

}  // namespace cardinal::project
