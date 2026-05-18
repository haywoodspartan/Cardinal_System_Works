#pragma once

// =============================================================================
// Studio — C++ scripting & sandbox panel.
//
// Exposes the cardinal::cppscript runtime in the editor:
//   - Compiler discovery state (path, vendor, "no compiler" warning)
//   - Per-job table: id, source path, status, exit code, elapsed, last
//     diagnostic snippet
//   - Compile / unload / reload / watch buttons
//   - Drag-drop a .cpp into the panel = queue a compile_and_load
//
// State is owned by the host (one cardinal::cppscript::Engine + the input
// path buffer); the panel just renders it. Stateless from a class-member
// standpoint apart from the input string the user is typing.
//
// Sandbox status (cardinal::sandbox) is rendered alongside each loaded
// script when a sandbox is bound to it. The host is responsible for
// keeping the (script -> sandbox) map; the panel reads it via the
// SandboxLookup callback.
// =============================================================================

#include <cardinal/core/types.hpp>

namespace cardinal::cppscript { class Engine; struct JobInfo; }
namespace cardinal::sandbox    { class Sandbox; struct Status; }

namespace cardinal::ui::panels::cppscript_panel {

struct State {
    // Path the user is typing into the "compile" input field. Persists
    // across frames so they can edit + retry without losing what they had.
    char input_path[1024]{};
    // Selected job id — drives the lower "diagnostics" pane.
    cardinal::u64 selected_job_id{0};
    // When true, the next compile_and_load goes through Subprocess sandbox
    // mode (via the host's bridge — the panel itself doesn't construct
    // sandboxes, that's the host's job).
    bool prefer_subprocess{false};
};

// Optional: ask the host whether a script (by source path) is currently
// hosted inside a Sandbox, and if so what its Status looks like. nullptr
// means the host doesn't track sandboxes; the panel hides those columns.
using SandboxLookup =
    cardinal::function<bool(const cardinal::string& source_path,
                       cardinal::sandbox::Status* out_status)>;

void draw(cardinal::cppscript::Engine* engine,
          const char*                  title,
          bool*                        p_open,
          State&                       state,
          SandboxLookup                lookup_sandbox = nullptr);

}  // namespace cardinal::ui::panels::cppscript_panel
