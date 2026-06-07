#pragma once

// =============================================================================
// Cardinal — embedded scripting engine (Lua 5.4 baseline).
//
// Owns a single lua_State plus the engine-side bindings exposed under the
// global `cardinal` table (logging, navigation, future scene API). The
// public surface is intentionally tiny — one C++ header, no sol2 / no
// LuaBridge — so build times stay reasonable. As bindings grow we'll
// fold them into a `bind_*.cpp` per subsystem rather than reaching for a
// generator framework.
//
// Phase 4-E goals:
//   ✓ One Engine per process, owning a single Lua state
//   ✓ run_string()  — execute a chunk; capture compile + runtime errors
//   ✓ repl_eval()   — evaluate a single expression and return printed value
//   ✓ Built-in `cardinal.log_info / log_warn / log_error` bindings so
//     plugins / level scripts can surface diagnostics through the same bus
//     the engine uses.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/core/std/utility.hpp>

namespace cardinal::script {

// Debugger state machine.
enum class DebugState : u32 {
    Idle = 0,        // running freely
    AtBreakpoint,    // paused at a line breakpoint
    Stepping,        // single-step in progress
};

enum class StepMode : u32 {
    Continue = 0,
    Over,            // step over (next line in current frame)
    In,              // step in (any next line, including descended into)
    Out,             // step out (next line in caller frame)
};

struct DebugVar {
    cardinal::string name;
    cardinal::string type;     // lua type tostring (number / string / table / etc.)
    cardinal::string value;    // tostring() of the value (truncated)
};

struct DebugFrame {
    cardinal::string source;        // chunk name / file
    u32         current_line{0};
    cardinal::string what;          // 'main', 'Lua', 'C', etc.
    cardinal::string name;          // function name (may be empty for anonymous)
};

class Engine {
public:
    static cardinal::unique_ptr<Engine> create();
    virtual ~Engine() = default;

    // Compile + execute a chunk of Lua code. Returns an empty string on
    // success, or a multi-line error string (compile or runtime) on failure.
    virtual cardinal::string run_string(const char* code, const char* chunk_name = "=chunk") = 0;

    // REPL-style evaluation. The line is first tried as `return <line>` so
    // the Lua expression's value is captured and tostring()-ed back to the
    // caller; if that doesn't compile we fall back to running the line as a
    // statement.
    virtual cardinal::string repl_eval(const char* line) = 0;

    // Loads + executes a file from disk. Returns the same kind of message
    // string as run_string().
    virtual cardinal::string run_file(const char* path) = 0;

    // ------------------------------------------------------------------
    // Debugger surface
    // ------------------------------------------------------------------
    // Toggle the debug + trace hook on/off. When enabled, every line the
    // VM executes gives the engine a chance to:
    //   - check breakpoints                   → AtBreakpoint
    //   - service step requests               → Stepping
    //   - record Call/Return into the trace timeline
    // Off by default — turn on from the Studio panel when needed.
    virtual void set_debug_enabled(bool on) = 0;
    virtual bool debug_enabled() const noexcept = 0;

    // Breakpoint set: pairs of (chunk_name_or_file, line_number).
    virtual void add_breakpoint   (const char* source, u32 line) = 0;
    virtual void remove_breakpoint(const char* source, u32 line) = 0;
    virtual void clear_breakpoints() = 0;
    virtual cardinal::vector<cardinal::pair<cardinal::string, u32>> breakpoints() const = 0;

    // Pause state. Returns true when the VM is currently parked at a
    // breakpoint; the Studio panel polls this so it can refresh the
    // locals/upvalues/stack views when the state changes.
    virtual DebugState state() const noexcept = 0;

    // While paused, request a step + resume. While idle, no-op.
    virtual void resume(StepMode mode = StepMode::Continue) = 0;

    // While paused, snapshot the active stack + locals/upvalues for the
    // top frame. Returns empty vectors when not paused.
    virtual cardinal::vector<DebugFrame> stack() const = 0;
    virtual cardinal::vector<DebugVar>   locals() const = 0;
    virtual cardinal::vector<DebugVar>   upvalues() const = 0;

protected:
    Engine() = default;
};

}  // namespace cardinal::script
