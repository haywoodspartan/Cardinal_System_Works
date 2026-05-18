#pragma once

// =============================================================================
// Studio — console panel (free-function helper).
//
// Lifted out of studio.cpp so console UX changes (autocomplete, history,
// REPL hooks) edit in isolation. The studio owns one ConsolePanelState
// across frames and forwards its `draw_console_panel` virtual call here;
// the eval callback is supplied per-call so the host can route input
// through any combination of CVars, scripted shells, etc.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <cardinal/core/containers.hpp>

namespace cardinal::ui::panels {

struct ConsolePanelState {
    cardinal::vector<cardinal::string> log;
    char                     input[1024]{};
    bool                     scroll_to_bottom{false};
    // Set true by suggestion-strip clicks so the next frame re-focuses
    // the input box (the click stole focus).
    bool                     request_focus{false};
    // Set true by suggestion double-clicks so the input is auto-submitted
    // on the next frame the input is queried.
    bool                     pending_submit{false};
};

// One callable per submitted line. Returns the formatted result that gets
// appended to the panel's scrollback (empty string = no extra line). The
// host typically chains: cardinal::console::Registry::execute → script.
using ConsoleEvalFn = cardinal::function<cardinal::string(const char* /* line */)>;

void draw(const ConsoleEvalFn& eval,
          const char*          title,
          bool*                p_open,
          ConsolePanelState&   state);

}  // namespace cardinal::ui::panels
