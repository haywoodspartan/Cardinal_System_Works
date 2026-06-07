// =============================================================================
// Cardinal — deterministic undo/redo-stack regression suite.
//
// edit::UndoStack is the editor's reversibility backbone. Commands here
// mutate an external recorder (an int + an op-log) so the suite pins the
// exact apply/revert ORDER and resulting value — not just sizes:
//
//   * push_and_apply runs apply ONCE; push_executed runs it NEVER
//     (the host already did it) — both bookkeep identically;
//   * undo reverts history[cursor-1] then decrements; redo applies
//     history[cursor] then increments; bools are false at the ends with
//     NO side effect;
//   * pushing mid-undo TRUNCATES the redo tail (lost forever);
//   * the capacity cap drops the OLDEST and re-bases the cursor
//     (incl. the cursor≤drop → 0 path, and capacity 0);
//   * null apply/revert still move the cursor + return true (no crash);
//   * next_undo/redo_label + clear + boundary returns.
//
// Pure, deterministic. Exit 0 = all pass.
// =============================================================================

#include <cardinal/edit/undo.hpp>
#include <cardinal/core/diag/log.hpp>

#include <string>
#include <vector>

namespace {

namespace ed = cardinal::edit;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("undotest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }
bool streq(const char* a, const char* b) { return std::string(a) == b; }

struct Rec {
    int                       v{0};
    std::vector<std::string>  log;
};

// Command Ci: apply → v=i, log "ai"; revert → v=i-1, log "ri".
ed::Command cmd(Rec& r, int id) {
    ed::Command c;
    c.label  = "C" + std::to_string(id);
    c.apply  = [&r, id] { r.v = id;     r.log.push_back("a" + std::to_string(id)); };
    c.revert = [&r, id] { r.v = id - 1; r.log.push_back("r" + std::to_string(id)); };
    return c;
}
bool log_is(const std::vector<std::string>& g, std::vector<std::string> exp) {
    if (g.size() != exp.size()) return false;
    for (cardinal::usize i = 0; i < g.size(); ++i)
        if (g[i] != exp[i]) return false;
    return true;
}

// ---- push_and_apply → full undo/redo cycle ------------------------
void test_basic_cycle() {
    Rec r;
    ed::UndoStack st;
    st.push_and_apply(cmd(r, 1));
    CHECK(r.v == 1 && st.size() == sz(1));
    CHECK(st.can_undo() && !st.can_redo());
    CHECK(streq(st.next_undo_label(), "C1"));
    CHECK(streq(st.next_redo_label(), ""));

    st.push_and_apply(cmd(r, 2));
    st.push_and_apply(cmd(r, 3));
    CHECK(r.v == 3 && st.size() == sz(3));
    CHECK(log_is(r.log, {"a1","a2","a3"}));

    CHECK(st.undo());                                 // revert C3
    CHECK(r.v == 2);
    CHECK(st.can_undo() && st.can_redo());
    CHECK(streq(st.next_undo_label(), "C2"));
    CHECK(streq(st.next_redo_label(), "C3"));
    CHECK(st.undo() && r.v == 1);                     // revert C2
    CHECK(st.undo() && r.v == 0);                     // revert C1
    CHECK(!st.can_undo() && st.can_redo());
    CHECK(streq(st.next_undo_label(), ""));
    CHECK(streq(st.next_redo_label(), "C1"));

    CHECK(!st.undo());                                // nothing to undo
    CHECK(r.v == 0);
    CHECK(log_is(r.log, {"a1","a2","a3","r3","r2","r1"}));   // undo didn't log

    CHECK(st.redo() && r.v == 1);                     // apply C1
    CHECK(st.redo() && r.v == 2);                     // apply C2
    CHECK(st.redo() && r.v == 3);                     // apply C3
    CHECK(!st.can_redo());
    CHECK(!st.redo() && r.v == 3);                    // nothing to redo
    CHECK(log_is(r.log,
        {"a1","a2","a3","r3","r2","r1","a1","a2","a3"}));
}

// ---- pushing mid-undo truncates the redo tail ---------------------
void test_redo_truncate() {
    Rec r;
    ed::UndoStack st;
    st.push_and_apply(cmd(r, 1));
    st.push_and_apply(cmd(r, 2));
    st.push_and_apply(cmd(r, 3));
    CHECK(st.undo() && st.undo());                    // cursor → 1, v == 1
    CHECK(r.v == 1 && st.can_redo() && st.size() == sz(3));

    st.push_and_apply(cmd(r, 4));                     // truncates C2,C3
    CHECK(r.v == 4 && st.size() == sz(2));            // [C1, C4]
    CHECK(!st.can_redo());                            // redo half gone
    CHECK(!st.redo());
    CHECK(streq(st.next_undo_label(), "C4"));

    CHECK(st.undo() && r.v == 3);                     // revert C4 (→ id-1)
    CHECK(streq(st.next_undo_label(), "C1"));
    CHECK(st.can_redo());
    CHECK(st.redo() && r.v == 4);                     // re-apply C4
    CHECK(!st.can_redo());
}

// ---- capacity cap drops the oldest, re-bases the cursor -----------
void test_capacity() {
    {
        Rec r;
        ed::UndoStack st(2);                          // hold at most 2
        st.push_and_apply(cmd(r, 1));
        st.push_and_apply(cmd(r, 2));
        st.push_and_apply(cmd(r, 3));                 // drops C1
        CHECK(st.size() == sz(2));
        CHECK(r.v == 3);
        CHECK(st.undo() && r.v == 2);                 // revert C3
        CHECK(st.undo() && r.v == 1);                 // revert C2
        CHECK(!st.undo());                            // C1 unrecoverable
        CHECK(r.v == 1);

        // cursor is 0 here; a new push wipes the (now-stale) tail.
        st.push_and_apply(cmd(r, 4));
        CHECK(r.v == 4 && st.size() == sz(1));
        CHECK(st.undo() && r.v == 3);
        CHECK(st.redo() && r.v == 4);
    }
    {   // capacity 0 → nothing is ever retained (cursor≤drop → 0 path);
        //  apply still runs, but the stack stays empty + non-undoable.
        Rec r;
        ed::UndoStack st(0);
        st.push_and_apply(cmd(r, 1));
        CHECK(r.v == 1);                              // apply ran
        CHECK(st.size() == sz(0));
        CHECK(!st.can_undo() && !st.can_redo());
        CHECK(!st.undo() && !st.redo());
        st.push_and_apply(cmd(r, 2));
        CHECK(r.v == 2 && st.size() == sz(0));
    }
}

// ---- push_executed defers apply; push_and_apply runs it once ------
void test_executed_vs_apply() {
    {
        Rec r;
        ed::UndoStack st;
        ed::Command c;
        c.label  = "M";
        c.apply  = [&r] { r.log.push_back("APPLIED"); };
        c.revert = [&r] { r.log.push_back("REVERTED"); };
        st.push_executed(std::move(c));               // host already applied
        CHECK(st.size() == sz(1));
        CHECK(log_is(r.log, {}));                     // apply NOT called
        CHECK(st.undo());
        CHECK(log_is(r.log, {"REVERTED"}));           // undo calls revert
        CHECK(st.redo());
        CHECK(log_is(r.log, {"REVERTED","APPLIED"})); // redo calls apply
    }
    {
        Rec r;
        ed::UndoStack st;
        ed::Command c;
        c.label  = "N";
        c.apply  = [&r] { r.log.push_back("AP"); };
        st.push_and_apply(std::move(c));
        CHECK(log_is(r.log, {"AP"}));                 // applied exactly once
        CHECK(st.size() == sz(1) && !st.can_redo());
    }
}

// ---- null apply/revert: cursor still moves, returns true ----------
void test_null_fns() {
    ed::UndoStack st;
    ed::Command c;                                    // apply/revert empty
    c.label = "NoOp";
    st.push_executed(std::move(c));
    CHECK(st.size() == sz(1) && st.can_undo());
    CHECK(st.undo());                                 // revert null → no-op
    CHECK(!st.can_undo() && st.can_redo());
    CHECK(st.redo());                                 // apply null → no-op
    CHECK(!st.can_redo() && st.can_undo());

    ed::Command c2;                                   // null apply
    c2.label = "NoOp2";
    ed::UndoStack st2;
    st2.push_and_apply(std::move(c2));                // must not crash
    CHECK(st2.size() == sz(1));
}

// ---- labels / clear / empty-stack boundaries ----------------------
void test_labels_clear_bounds() {
    ed::UndoStack st;
    CHECK(streq(st.next_undo_label(), ""));
    CHECK(streq(st.next_redo_label(), ""));
    CHECK(!st.undo() && !st.redo());
    CHECK(st.size() == sz(0) && !st.can_undo() && !st.can_redo());

    Rec r;
    st.push_and_apply(cmd(r, 1));
    CHECK(streq(st.next_undo_label(), "C1"));
    CHECK(streq(st.next_redo_label(), ""));
    st.undo();
    CHECK(streq(st.next_undo_label(), ""));
    CHECK(streq(st.next_redo_label(), "C1"));

    st.push_and_apply(cmd(r, 2));                     // truncates C1's redo
    CHECK(st.size() == sz(1));
    CHECK(streq(st.next_undo_label(), "C2"));
    CHECK(streq(st.next_redo_label(), ""));

    st.clear();
    CHECK(st.size() == sz(0));
    CHECK(!st.can_undo() && !st.can_redo());
    CHECK(streq(st.next_undo_label(), ""));
    CHECK(streq(st.next_redo_label(), ""));
}

}  // namespace

int main() {
    test_basic_cycle();
    test_redo_truncate();
    test_capacity();
    test_executed_vs_apply();
    test_null_fns();
    test_labels_clear_bounds();

    if (g_fail == 0) {
        cardinal::log::infof("undotest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("undotest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
