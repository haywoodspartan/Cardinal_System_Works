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

    // Path to the Cardinal engine source root this project builds against.
    // generate_build_files() bakes it into the project's CMakeLists as the
    // default CARDINAL_ENGINE_ROOT (the build add_subdirectory()s the engine,
    // since there is no installed SDK yet). Persisted so the build files can be
    // regenerated later without re-specifying it.
    cardinal::string engine_root;

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

// Returns the new Project on success. Creates the directory tree, writes the
// per-template starter game code (src/game.cpp), a sample material.hlsl, a
// runnable default world, READMEs — and the full build system (CMakeLists.txt,
// build.bat / build.sh, the src/main.cpp game-runner entry, .gitignore) via
// generate_build_files(). The result is a project that both opens in Studio and
// builds into a standalone executable.
cardinal::shared_ptr<Project> instantiate_template(const InstantiateOptions& opts,
                                              cardinal::string* error_out = nullptr);

// ---------------------------------------------------------------------------
// Build system
// ---------------------------------------------------------------------------
// (Re)generate the per-project build files for an existing project:
//   CMakeLists.txt   — add_subdirectory()s the Cardinal engine (from
//                      proj.info().engine_root, overridable via the
//                      CARDINAL_ENGINE_ROOT cache var) and links cardinal::engine.
//   src/main.cpp     — the game-runner entry: opens the project, loads the
//                      startup world, and ticks the Game (headless by default).
//   build.bat / .sh  — configure + build convenience wrappers.
//   .gitignore       — ignores build/ cooked/ pack/ shader cache.
// Safe to call repeatedly; overwrites only the generated files (never src/game.cpp
// or assets). Returns false (with error_out) on a write failure.
bool generate_build_files(const Project& proj, cardinal::string* error_out = nullptr);

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
