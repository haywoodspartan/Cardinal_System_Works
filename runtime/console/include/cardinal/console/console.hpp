#pragma once

// =============================================================================
// Cardinal — engine console (UE5-style CVars + CCommands).
//
// Anything in the engine or editor that's worth tweaking at runtime — render
// pipeline knobs, viewport count, world streaming radii, time scale, gravity,
// FPS caps, scene-graph queries — is registered here. The Studio's Console
// panel and any external tooling (debug HUD, network admin shell, scripting
// console) all dispatch through one Registry.
//
// Two object kinds:
//
//   CVar       — a typed, named variable. Lookup by name. Reads + writes go
//                through user-supplied get/set lambdas so the cvar can wrap
//                live engine state (no copy-back drift). Bool / int / float /
//                string flavours; range hints + a help string for the UI.
//
//   CCommand   — a named callable. Args come in as a single argv vector; the
//                command parses what it needs and writes one or more lines
//                back via the supplied output sink.
//
// The console is process-wide (singleton). Modules register on engine boot or
// inside their own initialise() — typically via the helper macros at the end
// of this file. Registration order doesn't matter; lookup is by name.
//
// Example registration:
//
//   CARDINAL_CVAR_BOOL  ("r.vsync",     "Enable vertical sync",
//                         [&] { return sw->vsync(); },
//                         [&](bool v) { sw->set_vsync(v); });
//
//   CARDINAL_CCOMMAND   ("scene.list",  "Print every entity's id + name",
//                         [&](const auto& argv, auto& out) {
//                             for (auto& e : scene.entities())
//                                 out("%u  %s", e.id, e.name.c_str());
//                         });
//
// Lookup (find_cvar / find_command / unregister) is case-insensitive over
// the WHOLE name — "R.VSync" matches "r.vsync" — so users don't have to
// remember capitalisation conventions. Note the asymmetry: lookup is
// case-insensitive, but the all_*() listings sort case-sensitively.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <functional>
#include <string>
#include <vector>

namespace cardinal::console {

// ----------------------------------------------------------------------------
// Output sink — a callable the command writes lines to. The Studio console
// stuffs each line into its scrollback; debug HUDs and scripting front-ends
// can route the same lines into their own log surfaces.
// ----------------------------------------------------------------------------
class Output {
public:
    using Fn = std::function<void(const std::string&)>;
    explicit Output(Fn f) : fn_(std::move(f)) {}

    // printf-style write — convenient for the common case of formatted lines.
    // The console panel scrollback and the Lua-bound write both come through
    // this entry, so bots and humans see identical output.
    void operator()(const char* fmt, ...);
    void line(const std::string& s) { if (fn_) fn_(s); }

private:
    Fn fn_;
};

// ----------------------------------------------------------------------------
// CVar — a typed, named runtime variable.
//
// The "type" is the storage kind that the get/set lambdas exchange. Range
// hints are advisory: the UI uses them for slider bounds, and the cvar
// command (`set <name> <value>`) clamps writes to the range when set.
// ----------------------------------------------------------------------------
enum class CVarType : u32 { Bool, Int, Float, String };

struct CVar {
    std::string name;        // dotted-namespace convention: "r.vsync", "world.render_dist_xz"
    std::string help;        // one-line description shown in the UI
    CVarType    type{CVarType::Bool};

    // Range hints — only relevant for Int / Float, ignored otherwise.
    f64 min{0.0};
    f64 max{1.0};

    // Type-erased getters / setters bind to live engine state. Each cvar
    // populates exactly one (get_*, set_*) pair matching `type`. The other
    // fields stay null and are never invoked.
    std::function<bool()>        get_bool;
    std::function<void(bool)>    set_bool;
    std::function<i64()>         get_int;
    std::function<void(i64)>     set_int;
    std::function<f64()>         get_float;
    std::function<void(f64)>     set_float;
    std::function<std::string()> get_string;
    std::function<void(const std::string&)> set_string;
};

// ----------------------------------------------------------------------------
// CCommand — a named callable. The first argv element is the command name
// itself (matching the registered name); the rest are the arguments split on
// whitespace (with simple "double-quoted strings" support).
// ----------------------------------------------------------------------------
struct CCommand {
    std::string name;        // dotted: "scene.list", "world.evict_all"
    std::string help;        // shown by `help <name>` and in autocomplete tooltips

    using Fn = std::function<void(const std::vector<std::string>& argv,
                                  Output& out)>;
    Fn fn;
};

// ----------------------------------------------------------------------------
// Registry — process-wide singleton. Plugins / engine subsystems register on
// init; UIs query at runtime.
// ----------------------------------------------------------------------------
class Registry {
public:
    static Registry& instance() noexcept;

    // ----- Registration -----
    // register_cvar / register_command return false if `name` is empty or
    // already used (the existing entry wins; the new one is dropped). The
    // helper macros below use these.
    bool register_cvar   (CVar    cv);
    bool register_command(CCommand cc);
    void unregister(const std::string& name);
    void clear();

    // ----- Lookup -----
    const CVar*     find_cvar   (const std::string& name) const noexcept;
    const CCommand* find_command(const std::string& name) const noexcept;

    // Snapshot accessors for the UI — lists are sorted alphabetically so the
    // autocomplete popup is stable frame-to-frame.
    std::vector<const CVar*>     all_cvars()    const;
    std::vector<const CCommand*> all_commands() const;

    // Prefix autocomplete — returns up to `max_results` names (cvars + commands)
    // whose lowercased name starts with `prefix` (also lowercased).
    std::vector<std::string> complete(const std::string& prefix,
                                      usize max_results = 16) const;

    // ----- Execution -----
    //
    // Parses one input line into argv. Recognised forms:
    //
    //   <cvar>                 -> print the cvar's current value
    //   <cvar> <value>         -> set the cvar (clamped to its range)
    //   <command> [args ...]   -> run the command with argv = [name, args...]
    //   help                   -> list categories
    //   help <name>            -> show help for one entry
    //
    // Returns true if `input` resolved to a registered cvar / command. Returns
    // false (without writing to `out`) when nothing matched, so the caller
    // can fall through to a scripting engine.
    bool execute(const std::string& input, Output& out);

private:
    Registry() = default;
    ~Registry() = default;
    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;

    std::vector<CVar>     cvars_;
    std::vector<CCommand> commands_;
};

// ----------------------------------------------------------------------------
// Convenience macros — wire a cvar / command into the Registry with a
// single call, including builder boilerplate. Use inside engine init code.
// Lambdas can capture by reference; the registry stores the std::function.
//
// Trailing args use `__VA_ARGS__` so call-site lambdas with multi-element
// capture lists like `[a, b, c](...) { ... }` don't get split by the
// preprocessor on the commas inside the brackets. The setter (or whole
// callable, for CCOMMAND) is the variadic tail; if both getter and setter
// of a CVar need multi-element captures, hoist one to a named lambda and
// pass that name through the macro.
// ----------------------------------------------------------------------------
#define CARDINAL_CVAR_BOOL(NAME, HELP, GET, ...)                               \
    do {                                                                       \
        cardinal::console::CVar _cv;                                           \
        _cv.name = (NAME); _cv.help = (HELP);                                  \
        _cv.type = cardinal::console::CVarType::Bool;                          \
        _cv.get_bool = (GET); _cv.set_bool = __VA_ARGS__;                      \
        cardinal::console::Registry::instance().register_cvar(std::move(_cv)); \
    } while (0)

#define CARDINAL_CVAR_INT(NAME, HELP, MIN, MAX, GET, ...)                      \
    do {                                                                       \
        cardinal::console::CVar _cv;                                           \
        _cv.name = (NAME); _cv.help = (HELP);                                  \
        _cv.type = cardinal::console::CVarType::Int;                           \
        _cv.min = static_cast<cardinal::f64>(MIN);                             \
        _cv.max = static_cast<cardinal::f64>(MAX);                             \
        _cv.get_int = (GET); _cv.set_int = __VA_ARGS__;                        \
        cardinal::console::Registry::instance().register_cvar(std::move(_cv)); \
    } while (0)

#define CARDINAL_CVAR_FLOAT(NAME, HELP, MIN, MAX, GET, ...)                    \
    do {                                                                       \
        cardinal::console::CVar _cv;                                           \
        _cv.name = (NAME); _cv.help = (HELP);                                  \
        _cv.type = cardinal::console::CVarType::Float;                         \
        _cv.min = static_cast<cardinal::f64>(MIN);                             \
        _cv.max = static_cast<cardinal::f64>(MAX);                             \
        _cv.get_float = (GET); _cv.set_float = __VA_ARGS__;                    \
        cardinal::console::Registry::instance().register_cvar(std::move(_cv)); \
    } while (0)

#define CARDINAL_CVAR_STRING(NAME, HELP, GET, ...)                             \
    do {                                                                       \
        cardinal::console::CVar _cv;                                           \
        _cv.name = (NAME); _cv.help = (HELP);                                  \
        _cv.type = cardinal::console::CVarType::String;                        \
        _cv.get_string = (GET); _cv.set_string = __VA_ARGS__;                  \
        cardinal::console::Registry::instance().register_cvar(std::move(_cv)); \
    } while (0)

#define CARDINAL_CCOMMAND(NAME, HELP, ...)                                     \
    do {                                                                       \
        cardinal::console::CCommand _cc;                                       \
        _cc.name = (NAME); _cc.help = (HELP);                                  \
        _cc.fn   = __VA_ARGS__;                                                \
        cardinal::console::Registry::instance().register_command(std::move(_cc)); \
    } while (0)

}  // namespace cardinal::console
