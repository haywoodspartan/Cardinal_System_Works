#include <cardinal/project/project.hpp>

#include <cardinal/core/diag/log.hpp>

#include <cardinal/core/std/algorithm.hpp>    // cardinal::sort/remove
#include <cardinal/core/std/cstdio.hpp>       // cardinal::snprintf
#include <cardinal/core/std/filesystem.hpp>   // cardinal::fs, error_code
#include <cardinal/core/std/fstream.hpp>      // ifstream/ofstream/ios/getline
#include <cardinal/core/std/sstream.hpp>      // cardinal::stringstream
#include <cardinal/core/std/utility.hpp>      // cardinal::move
#include <cardinal/core/std/containers.hpp>   // cardinal::vector

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

// Generic text writer (truncating, binary so line endings are exact).
bool write_text(const cardinal::string& path, const cardinal::string& text, cardinal::string* err) {
    cardinal::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);   // ensure parent dirs exist
    cardinal::ofstream f(path, cardinal::ios::binary | cardinal::ios::trunc);
    if (!f) { if (err) *err = "could not write " + path; return false; }
    f.write(text.data(), static_cast<cardinal::streamsize>(text.size()));
    return true;
}

// Reject manifest-supplied relative paths that escape the project root (path
// traversal) or are absolute — the manifest is treated as untrusted data.
bool is_safe_relpath(const cardinal::string& p) {
    if (p.empty()) return false;
    if (p.find("..") != cardinal::string::npos) return false;        // no traversal
    if (p.front() == '/' || p.front() == '\\') return false;         // no unix-absolute
    if (p.size() >= 2 && p[1] == ':') return false;                  // no drive-absolute
    return true;
}

cardinal::string replace_all(cardinal::string s, const cardinal::string& from, const cardinal::string& to) {
    if (from.empty()) return s;
    cardinal::string::size_type p = 0;
    while ((p = s.find(from, p)) != cardinal::string::npos) { s.replace(p, from.size(), to); p += to.size(); }
    return s;
}

// Turn a human project name into a valid CMake target / C++ identifier.
cardinal::string sanitize_target(const cardinal::string& name) {
    cardinal::string t;
    for (char c : name) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '_';
        if (ok)                      t += c;
        else if (c == ' ' || c == '-') t += '_';
    }
    if (t.empty())                       t = "CardinalGame";
    if (t[0] >= '0' && t[0] <= '9')      t = "_" + t;
    return t;
}

// Forward-slash a path so it is CMake-safe (backslashes are escapes in CMake).
cardinal::string cmake_path(cardinal::string p) {
    for (char& c : p) if (c == '\\') c = '/';
    return p;
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
        else if (key == "startup_world") {
            if (is_safe_relpath(val)) p->info_.startup_world = val;
            else cardinal::log::warnf("project",
                "ignoring unsafe startup_world '%s' (path traversal/absolute) — keeping default",
                val.c_str());
        }
        else if (key == "engine_root")  p->info_.engine_root = val;
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
    f << "startup_world = \"" << info_.startup_world << "\"\n";
    f << "engine_root = \"" << info_.engine_root << "\"\n";
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

// ===========================================================================
// Templates — per-kind starter game code + a runnable world.
// ===========================================================================
namespace {

// --- src/main.cpp : the standalone game-runner entry (kind-agnostic) -------
const char* kEntryMain = R"~~~(// Generated by Cardinal — standalone game-runner entry point.
//
// Boots the gameplay simulation, opens this project, loads its startup world,
// and ticks the Game. This default runner is HEADLESS (no window / GPU) so it
// runs gameplay deterministically — ideal for CI, tests and dedicated servers.
// For a rendered game, replace the loop below with a windowed one (see the
// engine's Studio / samples for Window + RHI + render setup), or open the
// project in Cardinal Studio and press Play.

#include <cardinal/project/project.hpp>
#include <cardinal/sim/sim.hpp>
#include <cardinal/game/game.hpp>
#include <cardinal/serial/serial.hpp>
#include <cardinal/core/diag/log.hpp>

int main(int argc, char** argv) {
    const cardinal::string root = (argc > 1) ? cardinal::string(argv[1])
                                             : cardinal::string(".");

    cardinal::string err;
    auto proj = cardinal::project::Project::open(root, &err);
    if (!proj) {
        cardinal::log::errorf("game", "cannot open project at '%s': %s",
                              root.c_str(), err.c_str());
        return 1;
    }

    cardinal::sim::SimWorld sim;
    cardinal::game::Game    game(sim);

    const cardinal::string world_path = proj->dirs().root + "/" + proj->info().startup_world;
    auto stats = cardinal::serial::load_world(game, world_path, /*replace_existing*/ true, &err);
    cardinal::log::infof("game", "loaded '%s': %u actors, %u props (%u skipped)",
                         world_path.c_str(), stats.actors_spawned,
                         stats.properties_applied, stats.actors_skipped);

    game.start_play();

    // Headless fixed-step loop: 5 seconds at 120 Hz. Swap the bound for a real
    // game loop (while !quit) once you add windowing / rendering.
    const float dt = 1.0f / 120.0f;
    for (int frame = 0; frame < 600; ++frame)
        game.tick(dt);

    game.stop_play();
    cardinal::log::infof("game", "'%s' exited cleanly", proj->info().name.c_str());
    return 0;
}
)~~~";

// --- src/game.cpp : per-template game classes ------------------------------
const char* kGameBlank = R"~~~(// Generated by Cardinal — Blank template game module.
//
// Define your gameplay by subclassing cardinal::game::GameActor and registering
// the class with CARDINAL_REGISTER_GAME_CLASS so the engine can spawn it from a
// world file by name + drive its reflected properties from the editor.

#include <cardinal/game/game_actor.hpp>
#include <cardinal/game/reflection.hpp>
#include <cardinal/core/diag/log.hpp>

// A minimal actor that spins about its axis every tick. Replace with your logic.
class SpinnerActor final : public cardinal::game::GameActor {
public:
    float        speed{45.0f};                  // degrees / second
    bool         enabled{true};
    cardinal::scene::Vec3 axis{0.0f, 1.0f, 0.0f};

    void begin_play() override {
        angle_ = 0.0f;
        cardinal::log::infof("game", "SpinnerActor begin_play (speed=%.1f deg/s)", speed);
    }
    void on_tick(float dt) override {
        if (!enabled) return;
        angle_ += speed * dt;
        if (angle_ >= 360.0f) angle_ -= 360.0f;
    }

private:
    float angle_{0.0f};
};

CARDINAL_REGISTER_GAME_CLASS(SpinnerActor, "Game/Spinner",
    PROP_FLOAT(speed,   0.0f, 720.0f, "Spin speed (deg/s)")
    PROP_BOOL (enabled,               "Tick this actor")
    PROP_VEC3 (axis,                  "Spin axis"))
)~~~";

const char* kGameFirstPerson = R"~~~(// Generated by Cardinal — First-Person template game module.

#include <cardinal/game/game_actor.hpp>
#include <cardinal/game/reflection.hpp>
#include <cardinal/core/diag/log.hpp>

// A first-person character. Holds movement tuning + integrates a simple
// velocity each tick; wire real input in on_tick (see cardinal::input).
class FirstPersonController final : public cardinal::game::GameActor {
public:
    float move_speed{6.0f};            // m/s
    float mouse_sensitivity{0.12f};
    float jump_height{1.2f};           // m
    bool  sprint_enabled{true};

    void begin_play() override {
        velocity_ = cardinal::scene::Vec3{0.0f, 0.0f, 0.0f};
        cardinal::log::infof("game", "FirstPersonController ready (speed=%.1f m/s)", move_speed);
    }
    void on_tick(float dt) override {
        // Placeholder integration — replace with input-driven movement.
        velocity_.y -= 9.81f * dt;     // gravity
    }

private:
    cardinal::scene::Vec3 velocity_{0.0f, 0.0f, 0.0f};
};

CARDINAL_REGISTER_GAME_CLASS(FirstPersonController, "Game/FirstPerson",
    PROP_FLOAT(move_speed,        0.0f, 50.0f, "Walk speed (m/s)")
    PROP_FLOAT(mouse_sensitivity, 0.01f, 2.0f, "Look sensitivity")
    PROP_FLOAT(jump_height,       0.0f, 5.0f,  "Jump apex (m)")
    PROP_BOOL (sprint_enabled,                 "Allow sprinting"))
)~~~";

const char* kGameTopDown = R"~~~(// Generated by Cardinal — Top-Down template game module.

#include <cardinal/game/game_actor.hpp>
#include <cardinal/game/reflection.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/std/cmath.hpp>

// A top-down pawn that walks toward a target point (click-to-move: set `target`
// from your input / nav code). Stops within accept_radius.
class TopDownPawn final : public cardinal::game::GameActor {
public:
    float move_speed{4.0f};
    float accept_radius{0.2f};
    cardinal::scene::Vec3 target{0.0f, 0.0f, 0.0f};

    void begin_play() override {
        position_ = cardinal::scene::Vec3{0.0f, 0.0f, 0.0f};
        cardinal::log::infof("game", "TopDownPawn ready (speed=%.1f)", move_speed);
    }
    void on_tick(float dt) override {
        const float dx = target.x - position_.x;
        const float dz = target.z - position_.z;
        const float dist = cardinal::sqrt(dx * dx + dz * dz);
        if (dist <= accept_radius || dist <= 0.0f) return;
        const float step = move_speed * dt;
        position_.x += dx / dist * step;
        position_.z += dz / dist * step;
    }

private:
    cardinal::scene::Vec3 position_{0.0f, 0.0f, 0.0f};
};

CARDINAL_REGISTER_GAME_CLASS(TopDownPawn, "Game/TopDown",
    PROP_FLOAT(move_speed,    0.0f, 50.0f, "Move speed (m/s)")
    PROP_FLOAT(accept_radius, 0.0f, 5.0f,  "Stop distance (m)")
    PROP_VEC3 (target,                     "Move-to target"))
)~~~";

const char* kGameCinematic = R"~~~(// Generated by Cardinal — Cinematic template game module.

#include <cardinal/game/game_actor.hpp>
#include <cardinal/game/reflection.hpp>
#include <cardinal/core/diag/log.hpp>

// A camera director that advances a normalized timeline 0..1 over `duration`
// seconds, looping if `loop`. Drive your camera / sequencer from time().
class CameraDirector final : public cardinal::game::GameActor {
public:
    float duration{10.0f};             // seconds
    bool  loop{true};

    void begin_play() override {
        time_ = 0.0f;
        cardinal::log::infof("game", "CameraDirector rolling (%.1fs, loop=%d)", duration, (int)loop);
    }
    void on_tick(float dt) override {
        if (duration <= 0.0f) return;
        time_ += dt / duration;
        if (time_ >= 1.0f) time_ = loop ? (time_ - 1.0f) : 1.0f;
    }
    float time() const { return time_; }

private:
    float time_{0.0f};
};

CARDINAL_REGISTER_GAME_CLASS(CameraDirector, "Game/Cinematic",
    PROP_FLOAT(duration, 0.1f, 600.0f, "Shot duration (s)")
    PROP_BOOL (loop,                   "Loop the shot"))
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

// --- build system templates (@TARGET@ / @ENGINE@ substituted) --------------
const char* kCMakeListsTmpl = R"~~~(# Generated by Cardinal — project build.
#
# There is no installed Cardinal SDK yet, so this builds the engine from source
# as a subdirectory and links your game against cardinal::engine. Point this at
# your engine checkout with -DCARDINAL_ENGINE_ROOT=/path/to/Cardinal_System_Works
# (or edit the default below).
cmake_minimum_required(VERSION 3.27)
project(@TARGET@ CXX)

if(NOT DEFINED CARDINAL_ENGINE_ROOT)
    set(CARDINAL_ENGINE_ROOT "@ENGINE@" CACHE PATH "Cardinal engine source root")
endif()
if(NOT EXISTS "${CARDINAL_ENGINE_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR
        "CARDINAL_ENGINE_ROOT does not point at a Cardinal checkout: ${CARDINAL_ENGINE_ROOT}\n"
        "Pass -DCARDINAL_ENGINE_ROOT=/path/to/Cardinal_System_Works")
endif()

# The engine's vcpkg manifest lives at its root; make the toolchain resolve the
# engine's dependencies when we build it as a subdirectory.
if(NOT DEFINED VCPKG_MANIFEST_DIR)
    set(VCPKG_MANIFEST_DIR "${CARDINAL_ENGINE_ROOT}" CACHE PATH "vcpkg manifest dir")
endif()

add_subdirectory("${CARDINAL_ENGINE_ROOT}" "${CMAKE_BINARY_DIR}/cardinal_engine")

file(GLOB GAME_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(@TARGET@ ${GAME_SOURCES})
target_link_libraries(@TARGET@ PRIVATE cardinal::engine)
if(TARGET cardinal::compile_flags)
    target_link_libraries(@TARGET@ PRIVATE cardinal::compile_flags)
endif()

# Run with the project root as argv[1] (where project.cardinal lives).
set_target_properties(@TARGET@ PROPERTIES
    VS_DEBUGGER_COMMAND_ARGUMENTS "\"${CMAKE_CURRENT_SOURCE_DIR}\"")
)~~~";

const char* kBuildBatTmpl = R"~~~(@echo off
REM Generated by Cardinal — configure + build this project.
REM Needs the same toolchain env as the engine: VsDevCmd (cl on PATH), cmake,
REM ninja, and a vcpkg toolchain. Override the engine path via CARDINAL_ENGINE_ROOT.
setlocal
set "ENGINE_ROOT=@ENGINE@"
if not "%CARDINAL_ENGINE_ROOT%"=="" set "ENGINE_ROOT=%CARDINAL_ENGINE_ROOT%"
echo [build] engine: %ENGINE_ROOT%
cmake -G Ninja -B build -S "%~dp0." -DCARDINAL_ENGINE_ROOT="%ENGINE_ROOT%" -DCMAKE_BUILD_TYPE=RelWithDebInfo
if errorlevel 1 exit /b 1
cmake --build build
)~~~";

const char* kBuildShTmpl = R"~~~(#!/usr/bin/env bash
# Generated by Cardinal — configure + build this project.
# Needs cmake, ninja and a vcpkg toolchain. Override the engine path via
# the CARDINAL_ENGINE_ROOT environment variable.
set -e
ENGINE_ROOT="${CARDINAL_ENGINE_ROOT:-@ENGINE@}"
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "[build] engine: ${ENGINE_ROOT}"
cmake -G Ninja -B build -S "${HERE}" -DCARDINAL_ENGINE_ROOT="${ENGINE_ROOT}" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
)~~~";

const char* kGitignore = R"~~~(# Generated by Cardinal.
/build/
/cooked/
/pack/
/shaders/cache/
)~~~";

const char* kReadmeTmpl = R"~~~(# @NAME@

@BLURB@

Created with the **Cardinal** engine.

## Layout
- `src/game.cpp` — your game classes (`GameActor` subclasses, registered with
  `CARDINAL_REGISTER_GAME_CLASS`). Edit this to build your game.
- `src/main.cpp` — the game-runner entry: opens the project, loads the startup
  world, and ticks the `Game`. Headless by default.
- `save/main.cardinalworld` — the startup world (edit in Studio, or by hand).
- `shaders/` — HLSL. `assets/` — source assets (cooked into `cooked/`, packed
  into `pack/`).
- `project.cardinal` — the project manifest.

## Build
```
build.bat            # Windows   (./build.sh on Linux)
```
This builds the Cardinal engine from source (`CARDINAL_ENGINE_ROOT`) and links
your game into `build/@TARGET@`.

## Run
```
build/@TARGET@ .     # pass the project root (this folder) as the first argument
```
Or open this folder in **Cardinal Studio** and press **Play**.
)~~~";

// The world snapshot — Ground + PlayerStart are anonymous (class="") so they
// always load; "Hero" references the template's registered game class with
// matching reflected properties.
cardinal::string build_world(const char* hero_class, const cardinal::string& hero_props) {
    cardinal::string s = "# Cardinal save v1\n";
    s += "actor \"Ground\" {\n  class = \"\"\n"
         "  position = (0.000, 0.000, 0.000)\n  rotation = (0.000, 0.000, 0.000)\n"
         "  scale = (50.000, 1.000, 50.000)\n}\n\n";
    s += "actor \"PlayerStart\" {\n  class = \"\"\n"
         "  position = (0.000, 1.000, 6.000)\n  rotation = (0.000, 0.000, 0.000)\n"
         "  scale = (1.000, 1.000, 1.000)\n}\n\n";
    s += "actor \"Hero\" {\n  class = \"";
    s += hero_class;
    s += "\"\n  position = (0.000, 1.500, 0.000)\n  rotation = (0.000, 0.000, 0.000)\n"
         "  scale = (1.000, 1.000, 1.000)\n";
    s += hero_props;
    s += "}\n";
    return s;
}

struct TemplateContent {
    const char*      game_cpp;     // src/game.cpp
    const char*      hero_class;   // world Hero block class name
    cardinal::string hero_props;   // world Hero block prop lines
    const char*      blurb;        // README description
};

TemplateContent content_for(TemplateKind k) {
    switch (k) {
        case TemplateKind::FirstPerson:
            return { kGameFirstPerson, "FirstPersonController",
                "  prop float move_speed = 6.000000\n"
                "  prop float mouse_sensitivity = 0.120000\n"
                "  prop float jump_height = 1.200000\n"
                "  prop bool sprint_enabled = true\n",
                "A first-person starter: a `FirstPersonController` with movement tuning, "
                "ready for you to wire input + collision." };
        case TemplateKind::TopDown:
            return { kGameTopDown, "TopDownPawn",
                "  prop float move_speed = 4.000000\n"
                "  prop float accept_radius = 0.200000\n"
                "  prop vec3 target = (0.000000, 0.000000, 0.000000)\n",
                "A top-down starter: a `TopDownPawn` that walks toward a target point "
                "(click-to-move). Hook up your input + navmesh." };
        case TemplateKind::Cinematic:
            return { kGameCinematic, "CameraDirector",
                "  prop float duration = 10.000000\n"
                "  prop bool loop = true\n",
                "A cinematic starter: a `CameraDirector` that advances a normalized "
                "timeline for camera / sequencer work. No player input." };
        case TemplateKind::Blank:
        default:
            return { kGameBlank, "SpinnerActor",
                "  prop float speed = 45.000000\n"
                "  prop bool enabled = true\n"
                "  prop vec3 axis = (0.000000, 1.000000, 0.000000)\n",
                "A blank starter: a single `SpinnerActor` you can replace with your "
                "own gameplay classes." };
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Build system
// ---------------------------------------------------------------------------
bool generate_build_files(const Project& proj, cardinal::string* err) {
    const auto& d    = proj.dirs();
    const auto& info = proj.info();
    const cardinal::string target = sanitize_target(info.name);
    const cardinal::string engine = info.engine_root.empty()
        ? cardinal::string("CHANGE_ME/path/to/Cardinal_System_Works")
        : cmake_path(info.engine_root);

    auto sub = [&](const char* tmpl) {
        return replace_all(replace_all(cardinal::string(tmpl), "@TARGET@", target),
                           "@ENGINE@", engine);
    };

    if (!write_text(d.root + "/CMakeLists.txt", sub(kCMakeListsTmpl), err)) return false;
    if (!write_text(d.src  + "/main.cpp",       kEntryMain,           err)) return false;
    if (!write_text(d.root + "/build.bat",      sub(kBuildBatTmpl),   err)) return false;
    if (!write_text(d.root + "/build.sh",       sub(kBuildShTmpl),    err)) return false;
    if (!write_text(d.root + "/.gitignore",     kGitignore,           err)) return false;
    cardinal::log::infof("project", "generated build files for '%s' (target %s)",
                         info.name.c_str(), target.c_str());
    return true;
}

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

    const TemplateContent tc = content_for(opts.kind);

    // Per-template game code + shader + assets note.
    write_text(p->dirs().src     + "/game.cpp",       tc.game_cpp,        nullptr);
    write_text(p->dirs().shaders + "/material.hlsl",  kTemplateShaderHLSL, nullptr);
    write_text(p->dirs().assets  + "/README.txt",
        "Drop your source assets here (.png .obj .hlsl .wav) — they'll be cooked\n"
        "into ../cooked/ + packed into ../pack/ when you hit Cook & Pack.\n", nullptr);

    // The runnable default world (ProjectInfo::startup_world), referencing the
    // template's registered game class.
    write_text(p->dirs().root + "/" + p->info().startup_world,
               build_world(tc.hero_class, tc.hero_props), nullptr);

    // Top-level README, parameterized for this template.
    {
        const cardinal::string readme = replace_all(replace_all(replace_all(
            cardinal::string(kReadmeTmpl),
            "@NAME@",  p->info().name.empty() ? cardinal::string("Cardinal Game") : p->info().name),
            "@BLURB@", cardinal::string(tc.blurb)),
            "@TARGET@", sanitize_target(p->info().name));
        write_text(p->dirs().root + "/README.md", readme, nullptr);
    }

    // The full build system (CMakeLists, src/main.cpp entry, build scripts).
    if (!generate_build_files(*p, err)) return nullptr;

    cardinal::log::infof("project",
        "instantiated template '%s' at %s",
        template_kind_name(opts.kind), opts.root.c_str());
    return p;
}

// ---------------------------------------------------------------------------
// RecentProjects
// ---------------------------------------------------------------------------
// Single source of truth for the "keep N most-recent" invariant — used
// by BOTH add() and load() so a store file with more entries (older
// build, hand-edited, corrupt) can't load past the cap.
inline constexpr usize kMaxRecent = 16;

RecentProjects::RecentProjects(cardinal::string store_path)
    : store_(cardinal::move(store_path)) {}

void RecentProjects::add(const cardinal::string& root) {
    entries_.erase(cardinal::remove(entries_.begin(), entries_.end(), root), entries_.end());
    entries_.insert(entries_.begin(), root);
    if (entries_.size() > kMaxRecent) entries_.resize(kMaxRecent);
}
void RecentProjects::remove(const cardinal::string& root) {
    entries_.erase(cardinal::remove(entries_.begin(), entries_.end(), root), entries_.end());
}
void RecentProjects::load() {
    entries_.clear();
    cardinal::ifstream f(store_);
    if (!f) return;
    cardinal::string line;
    while (cardinal::getline(f, line))
        if (!line.empty() && entries_.size() < kMaxRecent)
            entries_.push_back(line);
}
void RecentProjects::save() const {
    cardinal::ofstream f(store_, cardinal::ios::binary | cardinal::ios::trunc);
    for (const auto& e : entries_) f << e << "\n";
}

}  // namespace cardinal::project
