// =============================================================================
// Cardinal — deterministic editor mode-state-machine regression suite.
//
// edit::EditorState has no hidden state, so every contract is pinnable:
//
//   * editor_mode_name / _glyph  — exact strings for all 8 modes plus
//     the out-of-range "?" / "[?]" fallbacks;
//   * editor_mode_tooltip        — non-empty for valid modes, "" for
//     out-of-range, first line is "<Name> Mode (<hotkey>)" with the
//     stable Q/W/E/R/T/Y/U/I keybinding;
//   * default state              — Select, empty status, gizmo active,
//     brush inactive;
//   * set_mode                   — fires the change callback ONCE with
//     (prev, now) only on a real transition; a same-mode set is a no-op
//     (no callback, no change); callback is a single slot (last setter
//     wins); an empty std::function disables it; no callback set is safe;
//   * brush_active / gizmo_active — full 8-mode truth table (Mesh /
//     Landscape / Measure activate NEITHER);
//   * status_text                — set/get round-trip + overwrite.
//
// Pure, deterministic. Exit 0 = all pass.
// =============================================================================

#include <cardinal/edit/editor_mode.hpp>
#include <cardinal/core/log.hpp>

#include <string>
#include <vector>

namespace {

namespace ed = cardinal::edit;
using ed::EditorMode;
using ed::EditorState;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("edmtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool streq(const char* a, const char* b) { return std::string(a) == b; }
bool has(const char* hay, const char* needle) {
    return std::string(hay).find(needle) != std::string::npos;
}
bool starts_with(const char* hay, const char* pre) {
    return std::string(hay).rfind(pre, 0) == 0;
}
EditorMode em(int v) { return static_cast<EditorMode>(static_cast<cardinal::u32>(v)); }

// ---- name / glyph tables + out-of-range fallback ------------------
void test_name_glyph() {
    CHECK(streq(ed::editor_mode_name(EditorMode::Select),    "Select"));
    CHECK(streq(ed::editor_mode_name(EditorMode::Place),     "Place"));
    CHECK(streq(ed::editor_mode_name(EditorMode::Sculpt),    "Sculpt"));
    CHECK(streq(ed::editor_mode_name(EditorMode::Paint),     "Paint"));
    CHECK(streq(ed::editor_mode_name(EditorMode::Foliage),   "Foliage"));
    CHECK(streq(ed::editor_mode_name(EditorMode::Mesh),      "Mesh"));
    CHECK(streq(ed::editor_mode_name(EditorMode::Landscape), "Landscape"));
    CHECK(streq(ed::editor_mode_name(EditorMode::Measure),   "Measure"));
    CHECK(streq(ed::editor_mode_name(em(99)),                "?"));

    CHECK(streq(ed::editor_mode_glyph(EditorMode::Select),    "[+]"));
    CHECK(streq(ed::editor_mode_glyph(EditorMode::Place),     "[P]"));
    CHECK(streq(ed::editor_mode_glyph(EditorMode::Sculpt),    "[S]"));
    CHECK(streq(ed::editor_mode_glyph(EditorMode::Paint),     "[B]"));
    CHECK(streq(ed::editor_mode_glyph(EditorMode::Foliage),   "[F]"));
    CHECK(streq(ed::editor_mode_glyph(EditorMode::Mesh),      "[M]"));
    CHECK(streq(ed::editor_mode_glyph(EditorMode::Landscape), "[L]"));
    CHECK(streq(ed::editor_mode_glyph(EditorMode::Measure),   "[=]"));
    CHECK(streq(ed::editor_mode_glyph(em(99)),                "[?]"));
}

// ---- tooltips: structure + stable hotkey binding ------------------
void test_tooltip() {
    struct Row { EditorMode m; const char* name; const char* key; };
    const Row rows[] = {
        { EditorMode::Select,    "Select",    "(Q)" },
        { EditorMode::Place,     "Place",     "(W)" },
        { EditorMode::Sculpt,    "Sculpt",    "(E)" },
        { EditorMode::Paint,     "Paint",     "(R)" },
        { EditorMode::Foliage,   "Foliage",   "(T)" },
        { EditorMode::Mesh,      "Mesh",      "(Y)" },
        { EditorMode::Landscape, "Landscape", "(U)" },
        { EditorMode::Measure,   "Measure",   "(I)" },
    };
    for (const Row& r : rows) {
        const char* tip = ed::editor_mode_tooltip(r.m);
        CHECK(std::string(tip).size() > 0);
        CHECK(starts_with(tip, r.name));        // "<Name> Mode (..."
        CHECK(has(tip, "Mode ("));
        CHECK(has(tip, r.key));                 // stable keybinding
        CHECK(has(tip, "\n"));                  // 2-line layout
    }
    CHECK(streq(ed::editor_mode_tooltip(em(99)), ""));   // out-of-range
}

// ---- default-constructed state ------------------------------------
void test_default_state() {
    EditorState st;
    CHECK(st.mode() == EditorMode::Select);
    CHECK(st.status_text().empty());
    CHECK(st.gizmo_active());                   // Select -> gizmo
    CHECK(!st.brush_active());
}

// ---- set_mode + the change-callback firing rule -------------------
void test_set_mode_callback() {
    EditorState st;
    std::vector<EditorMode> olds, news;
    st.on_mode_change([&](EditorMode o, EditorMode n) {
        olds.push_back(o);
        news.push_back(n);
    });

    st.set_mode(EditorMode::Sculpt);            // Select -> Sculpt
    CHECK(st.mode() == EditorMode::Sculpt);
    CHECK(olds.size() == 1u);
    CHECK(olds[0] == EditorMode::Select && news[0] == EditorMode::Sculpt);

    st.set_mode(EditorMode::Sculpt);            // same mode -> NO fire
    CHECK(olds.size() == 1u);
    CHECK(st.mode() == EditorMode::Sculpt);

    st.set_mode(EditorMode::Paint);             // Sculpt -> Paint
    CHECK(olds.size() == 2u);
    CHECK(olds[1] == EditorMode::Sculpt && news[1] == EditorMode::Paint);

    // last-setter-wins: arming a second callback replaces the first.
    int first_hits = 0, second_hits = 0;
    st.on_mode_change([&](EditorMode, EditorMode) { ++first_hits; });
    st.on_mode_change([&](EditorMode, EditorMode) { ++second_hits; });
    st.set_mode(EditorMode::Place);
    CHECK(first_hits == 0 && second_hits == 1);
    CHECK(olds.size() == 2u);                   // original cb not called

    // an empty std::function disables the callback entirely.
    st.on_mode_change(EditorState::OnModeChange{});
    st.set_mode(EditorMode::Measure);
    CHECK(second_hits == 1);                     // unchanged
    CHECK(st.mode() == EditorMode::Measure);     // mode still transitions

    // no callback ever set is safe + still transitions.
    EditorState fresh;
    fresh.set_mode(EditorMode::Mesh);
    CHECK(fresh.mode() == EditorMode::Mesh);

    // setting back to the default also fires (it's a real transition).
    EditorState s2;
    int hits = 0;
    s2.on_mode_change([&](EditorMode, EditorMode) { ++hits; });
    s2.set_mode(EditorMode::Select);            // already Select -> no-op
    CHECK(hits == 0);
    s2.set_mode(EditorMode::Place);
    s2.set_mode(EditorMode::Select);            // Place -> Select fires
    CHECK(hits == 2);
}

// ---- brush_active / gizmo_active full truth table -----------------
void test_active_flags() {
    struct Row { EditorMode m; bool gizmo; bool brush; };
    const Row rows[] = {
        { EditorMode::Select,    true,  false },
        { EditorMode::Place,     true,  false },
        { EditorMode::Sculpt,    false, true  },
        { EditorMode::Paint,     false, true  },
        { EditorMode::Foliage,   false, true  },
        { EditorMode::Mesh,      false, false },
        { EditorMode::Landscape, false, false },
        { EditorMode::Measure,   false, false },
    };
    for (const Row& r : rows) {
        EditorState st;
        st.set_mode(r.m);
        CHECK(st.gizmo_active() == r.gizmo);
        CHECK(st.brush_active() == r.brush);
        // mutually exclusive: never both active at once.
        CHECK(!(st.gizmo_active() && st.brush_active()));
    }
}

// ---- status text round-trip ---------------------------------------
void test_status_text() {
    EditorState st;
    CHECK(st.status_text().empty());
    st.set_status_text("Picked 3 entities");
    CHECK(st.status_text() == "Picked 3 entities");
    st.set_status_text("Sculpting (radius 4.0)");
    CHECK(st.status_text() == "Sculpting (radius 4.0)");
    st.set_status_text("");
    CHECK(st.status_text().empty());
    // status is independent of mode transitions.
    st.set_status_text("keep me");
    st.set_mode(EditorMode::Foliage);
    CHECK(st.status_text() == "keep me");
}

}  // namespace

int main() {
    test_name_glyph();
    test_tooltip();
    test_default_state();
    test_set_mode_callback();
    test_active_flags();
    test_status_text();

    if (g_fail == 0) {
        cardinal::log::infof("edmtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("edmtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
