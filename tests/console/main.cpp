// =============================================================================
// Cardinal — deterministic engine-console regression suite.
//
// console::Registry is the single dispatch point for every runtime knob
// (render flags, streaming radii, time scale, FPS caps). A regression
// silently breaks every cvar/command — wrong parse, missing clamp (a
// range violation reaches live engine state), case drift, mis-tokenised
// quoted args, or a swallowed command. This pins the pure parser +
// registry:
//
//   * register dedup + cross-kind collision + empty/nullptr rejects;
//   * lookup is WHOLE-STRING case-insensitive (the header prose says
//     "LHS of the first dot" — the impl's ieq is whole-string; this
//     suite pins the IMPLEMENTED behaviour);
//   * all_*() listings sorted case-SENSITIVELY (lookup CI, sort CS);
//   * complete(): lowercased prefix, registration-order collection then
//     sort/unique, max_results cap;
//   * tokenizer: "quoted strings" survive as one arg, whitespace splits;
//   * cvar read/print + set with int & float range CLAMP (+ warning),
//     strtoll base-0 / trailing-garbage, parse-error keeps old value;
//   * command dispatch passes argv[0] = the user's raw token;
//   * help / ? / list built-ins; Output printf truncation at 1023.
//
// The Registry is a process-wide singleton (cleared per test). Pure,
// headless, fully deterministic. Exit 0 = all pass.
// =============================================================================

#include <cardinal/console/console.hpp>
#include <cardinal/core/log.hpp>
#include <cardinal/core/fstream.hpp>

#include <cstdio>     // std::remove (temp-file cleanup)
#include <string>
#include <vector>

namespace {

namespace cc = cardinal::console;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("contest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }
bool has(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
}
bool ap_d(double a, double b, double e = 1e-9) {
    const double d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}

cc::Registry& reg() { return cc::Registry::instance(); }

// Register a bool cvar wired to a backing variable.
void cv_bool(const char* name, const char* help, bool* slot,
             bool with_setter = true) {
    cc::CVar c;
    c.name = name; c.help = help; c.type = cc::CVarType::Bool;
    c.get_bool = [slot] { return *slot; };
    if (with_setter) c.set_bool = [slot](bool v) { *slot = v; };
    reg().register_cvar(std::move(c));
}
void cv_int(const char* name, const char* help, cardinal::i64 lo,
            cardinal::i64 hi, cardinal::i64* slot) {
    cc::CVar c;
    c.name = name; c.help = help; c.type = cc::CVarType::Int;
    c.min = static_cast<cardinal::f64>(lo);
    c.max = static_cast<cardinal::f64>(hi);
    c.get_int = [slot] { return *slot; };
    c.set_int = [slot](cardinal::i64 v) { *slot = v; };
    reg().register_cvar(std::move(c));
}
void cv_float(const char* name, const char* help, double lo, double hi,
              double* slot) {
    cc::CVar c;
    c.name = name; c.help = help; c.type = cc::CVarType::Float;
    c.min = lo; c.max = hi;
    c.get_float = [slot] { return *slot; };
    c.set_float = [slot](cardinal::f64 v) { *slot = v; };
    reg().register_cvar(std::move(c));
}
void cv_string(const char* name, const char* help, std::string* slot) {
    cc::CVar c;
    c.name = name; c.help = help; c.type = cc::CVarType::String;
    c.get_string = [slot] { return *slot; };
    c.set_string = [slot](const std::string& v) { *slot = v; };
    reg().register_cvar(std::move(c));
}

// ---- registration: dedup, cross-kind collision, unregister --------
void test_register() {
    reg().clear();
    bool a = false, b = false;
    cv_bool("r.vsync", "vsync", &a);
    CHECK(reg().find_cvar("r.vsync") != nullptr);

    // Duplicate (exact + case-insensitive) is rejected, first wins.
    cc::CVar dup; dup.name = "r.vsync"; dup.type = cc::CVarType::Bool;
    dup.get_bool = [&b] { return b; };
    CHECK(reg().register_cvar(dup) == false);
    cc::CVar dupci; dupci.name = "R.VSYNC"; dupci.type = cc::CVarType::Bool;
    dupci.get_bool = [&b] { return b; };
    CHECK(reg().register_cvar(dupci) == false);          // whole-string CI

    // Cross-kind name collision both directions.
    cc::CCommand col; col.name = "r.vsync"; col.help = "x";
    col.fn = [](const std::vector<std::string>&, cc::Output&) {};
    CHECK(reg().register_command(col) == false);

    cc::CCommand good; good.name = "c.do"; good.help = "h";
    good.fn = [](const std::vector<std::string>&, cc::Output&) {};
    CHECK(reg().register_command(good) == true);
    cc::CVar colcv; colcv.name = "c.do"; colcv.type = cc::CVarType::Bool;
    colcv.get_bool = [&b] { return b; };
    CHECK(reg().register_cvar(colcv) == false);          // command exists

    // Empty name / null fn rejected.
    cc::CVar noname; noname.type = cc::CVarType::Bool;
    CHECK(reg().register_cvar(noname) == false);
    cc::CCommand nofn; nofn.name = "c.nofn";
    CHECK(reg().register_command(nofn) == false);        // fn == nullptr

    // Lookup is whole-string case-insensitive.
    CHECK(reg().find_cvar("R.VsYnC") != nullptr);
    CHECK(reg().find_command("C.DO") != nullptr);
    CHECK(reg().find_cvar("missing") == nullptr);

    // unregister is case-insensitive; frees the name for reuse.
    reg().unregister("R.VSYNC");
    CHECK(reg().find_cvar("r.vsync") == nullptr);
    cv_bool("r.vsync", "again", &a);
    CHECK(reg().find_cvar("r.vsync") != nullptr);
    reg().unregister("c.do");
    CHECK(reg().find_command("c.do") == nullptr);

    reg().clear();
    CHECK(reg().all_cvars().empty() && reg().all_commands().empty());
}

// ---- listings sorted case-SENSITIVELY ------------------------------
void test_listing_sorted() {
    reg().clear();
    bool s = false;
    cv_bool("b.z", "h", &s);
    cv_bool("A.a", "h", &s);
    cv_bool("a.b", "h", &s);
    auto cvs = reg().all_cvars();
    CHECK(cvs.size() == sz(3));
    CHECK(cvs[0]->name == "A.a");      // 'A'(0x41) < 'a'(0x61) — case-sensitive
    CHECK(cvs[1]->name == "a.b");
    CHECK(cvs[2]->name == "b.z");

    cc::CCommand c1; c1.name = "cmd.b"; c1.help = "h";
    c1.fn = [](const std::vector<std::string>&, cc::Output&) {};
    cc::CCommand c2; c2.name = "cmd.a"; c2.help = "h";
    c2.fn = [](const std::vector<std::string>&, cc::Output&) {};
    reg().register_command(std::move(c1));
    reg().register_command(std::move(c2));
    auto cmds = reg().all_commands();
    CHECK(cmds.size() == sz(2));
    CHECK(cmds[0]->name == "cmd.a" && cmds[1]->name == "cmd.b");
}

// ---- prefix autocomplete ------------------------------------------
void test_complete() {
    reg().clear();
    bool s = false;
    cv_bool("r.vsync",   "h", &s);     // registration order matters
    cv_bool("r.fov",     "h", &s);     //  for the max_results cap
    cv_bool("render.dist","h", &s);
    cc::CCommand rc; rc.name = "run.it"; rc.help = "h";
    rc.fn = [](const std::vector<std::string>&, cc::Output&) {};
    reg().register_command(std::move(rc));

    auto all = reg().complete("");
    CHECK(all.size() == sz(4));
    CHECK(all[0] == "r.fov");           // sorted case-sensitively
    CHECK(all[1] == "r.vsync");
    CHECK(all[2] == "render.dist");
    CHECK(all[3] == "run.it");

    auto rdot = reg().complete("r.");
    CHECK(rdot.size() == sz(2));
    CHECK(rdot[0] == "r.fov" && rdot[1] == "r.vsync");

    auto ci = reg().complete("RE");     // prefix lowercased before match
    CHECK(ci.size() == sz(1) && ci[0] == "render.dist");

    CHECK(reg().complete("zzz").empty());

    auto cap = reg().complete("r", 2);  // cap during collection, then sort
    CHECK(cap.size() == sz(2));
    CHECK(cap[0] == "r.fov" && cap[1] == "r.vsync");
}

// ---- cvar read / print formatting ---------------------------------
void test_cvar_read() {
    reg().clear();
    bool   bb = true;
    cardinal::i64 ii = 7;
    double ff = 1.5;
    std::string ss = "hi";
    cv_bool ("t.b", "HB", &bb);
    cv_int  ("t.i", "HI", 0, 10, &ii);
    cv_float("t.f", "HF", 0.0, 1.0, &ff);
    cv_string("t.s", "HS", &ss);

    std::vector<std::string> L;
    cc::Output out([&L](const std::string& s) { L.push_back(s); });

    L.clear(); CHECK(reg().execute("t.b", out));
    CHECK(L.size() == sz(1) && has(L[0], "t.b = true") && has(L[0], "(bool)"));
    L.clear(); CHECK(reg().execute("T.B", out));          // CI resolves
    CHECK(has(L[0], "t.b = true"));
    L.clear(); reg().execute("t.i", out);
    CHECK(has(L[0], "t.i = 7") && has(L[0], "(int [0..10])"));
    L.clear(); reg().execute("t.f", out);
    CHECK(has(L[0], "t.f = 1.5") && has(L[0], "(float [0..1])"));
    L.clear(); reg().execute("t.s", out);
    CHECK(has(L[0], "t.s = \"hi\"") && has(L[0], "(string)"));
}

// ---- cvar set + range clamp + parse errors ------------------------
void test_cvar_set() {
    reg().clear();
    bool bb = false;
    cardinal::i64 ii = 5;
    double ff = 0.5, gg = 0.0;
    std::string ss;
    bool ro = true;
    cv_bool ("s.b", "H", &bb);
    cv_int  ("s.i", "H", 0, 10, &ii);
    cv_float("s.f", "H", 0.0, 1.0, &ff);
    cv_float("s.g", "H", 0.0, 0.0, &gg);          // degenerate range
    cv_string("s.s", "H", &ss);
    cv_bool ("s.ro", "H", &ro, /*with_setter=*/false);

    std::vector<std::string> L;
    cc::Output out([&L](const std::string& s) { L.push_back(s); });

    // Bool: accepted forms, case-insensitive; bad value keeps old + errors.
    CHECK(reg().execute("s.b on", out));  CHECK(bb == true);
    reg().execute("s.b 0", out);          CHECK(bb == false);
    reg().execute("s.b YES", out);        CHECK(bb == true);
    L.clear();
    CHECK(reg().execute("s.b nope", out) == true);   // matched a cvar
    CHECK(bb == true);                               // unchanged
    CHECK(has(L[0], "error") && has(L[0], "expects bool"));

    // Int: in-range, clamp high/low, base-0 hex, trailing garbage, bad.
    reg().execute("s.i 8", out);          CHECK(ii == 8);
    L.clear(); reg().execute("s.i 999", out);
    CHECK(ii == 10);
    CHECK(has(L[0], "warning: clamping 999 to [0..10]"));
    CHECK(has(L[1], "s.i = 10"));
    reg().execute("s.i -3", out);         CHECK(ii == 0);     // clamp low
    reg().execute("s.i 0x1F", out);       CHECK(ii == 10);    // 31 → clamp
    reg().execute("s.i 12abc", out);      CHECK(ii == 10);    // strtoll: 12
    L.clear();
    reg().execute("s.i abc", out);        CHECK(ii == 10);    // unchanged
    CHECK(has(L[0], "expects integer"));

    // Float: clamp, degenerate range = no clamp, bad value.
    reg().execute("s.f 0.25", out);       CHECK(ap_d(ff, 0.25));
    L.clear(); reg().execute("s.f 2.5", out);
    CHECK(ap_d(ff, 1.0));
    CHECK(has(L[0], "warning: clamping"));
    reg().execute("s.f -1", out);         CHECK(ap_d(ff, 0.0));
    L.clear(); reg().execute("s.g 999", out);
    CHECK(ap_d(gg, 999.0));                           // min==max → no clamp
    bool warned = false;
    for (auto& l : L) if (has(l, "warning")) warned = true;
    CHECK(!warned);
    L.clear(); reg().execute("s.f abc", out);
    CHECK(has(L[0], "expects number"));

    // String: re-joined with single spaces; quotes keep internal spaces.
    reg().execute("s.s hello world", out);   CHECK(ss == "hello world");
    reg().execute("s.s a    b", out);        CHECK(ss == "a b");   // collapse
    reg().execute("s.s \"x y z\"", out);     CHECK(ss == "x y z"); // quoted
    L.clear(); reg().execute("s.s plain", out);
    CHECK(has(L[0], "s.s = \"plain\""));

    // Read-only cvar: write parses but is a no-op (no setter bound).
    L.clear();
    CHECK(reg().execute("s.ro off", out) == true);
    CHECK(ro == true);                               // unchanged
    CHECK(has(L[0], "s.ro = true"));
}

// ---- cvar set: non-finite (nan/inf) rejected, not stored ----------
// Regression: parse_float used strtod, which per the C standard parses
// "nan"/"inf"/"infinity". A NaN then DEFEATS apply_cvar's [min,max]
// clamp (v<min and v>max are both false for NaN — unordered), so
// `set r.scale nan` wrote NaN straight into live engine state — the
// exact "missing clamp / range violation reaches live state" failure
// this suite exists to guard. Non-finite input must be rejected like
// any other unparseable value: old value kept + an error printed.
void test_cvar_nonfinite() {
    reg().clear();
    double ff = 0.5;
    cv_float("s.f", "H", 0.0, 1.0, &ff);

    std::vector<std::string> L;
    cc::Output out([&L](const std::string& s) { L.push_back(s); });

    // The legit finite path is unaffected.
    reg().execute("s.f 0.25", out);              CHECK(ap_d(ff, 0.25));

    // NaN forms: rejected, value UNCHANGED + still finite, error shown.
    for (const char* bad : { "s.f nan", "s.f NAN", "s.f -nan",
                             "s.f nan(1)" }) {
        L.clear();
        const double before = ff;
        CHECK(reg().execute(bad, out) == true);  // matched the cvar
        CHECK(ff == ff);                         // finite, not NaN
        CHECK(ap_d(ff, before));                 // unchanged
        CHECK(!L.empty() && has(L[0], "expects number"));
    }
    // Inf forms: also rejected (not silently clamped to max).
    for (const char* bad : { "s.f inf", "s.f -inf", "s.f infinity" }) {
        L.clear();
        const double before = ff;
        reg().execute(bad, out);
        CHECK(ap_d(ff, before));                 // unchanged
        CHECK(!L.empty() && has(L[0], "expects number"));
    }
    // A normal finite value still applies and clamps as before.
    reg().execute("s.f 2.5", out);               CHECK(ap_d(ff, 1.0));
    reg().execute("s.f 0.75", out);              CHECK(ap_d(ff, 0.75));
}

// ---- command dispatch ---------------------------------------------
void test_command() {
    reg().clear();
    std::vector<std::string> got;
    cc::CCommand c; c.name = "c.echo"; c.help = "echoes";
    c.fn = [&got](const std::vector<std::string>& av, cc::Output& o) {
        got = av;
        o("argc=%zu", av.size());
    };
    reg().register_command(std::move(c));

    cc::CCommand m; m.name = "c.multi"; m.help = "multi";
    m.fn = [](const std::vector<std::string>&, cc::Output& o) {
        o.line("L1");
        o("L%d", 2);
    };
    reg().register_command(std::move(m));

    std::vector<std::string> L;
    cc::Output out([&L](const std::string& s) { L.push_back(s); });

    L.clear(); got.clear();
    CHECK(reg().execute("c.echo a b", out) == true);
    CHECK(got.size() == sz(3));
    CHECK(got[0] == "c.echo" && got[1] == "a" && got[2] == "b");
    CHECK(L.size() == sz(1) && L[0] == "argc=3");

    L.clear(); got.clear();
    reg().execute("c.echo", out);
    CHECK(got.size() == sz(1) && got[0] == "c.echo");
    CHECK(L[0] == "argc=1");

    // Quoted arg → one argv element with the space preserved.
    got.clear();
    reg().execute("c.echo \"x y\" z", out);
    CHECK(got.size() == sz(3));
    CHECK(got[1] == "x y" && got[2] == "z");

    // find_command is CI, but argv[0] is the user's RAW token.
    got.clear();
    reg().execute("C.ECHO q", out);
    CHECK(got.size() == sz(2) && got[0] == "C.ECHO" && got[1] == "q");

    L.clear();
    reg().execute("c.multi", out);
    CHECK(L.size() == sz(2) && L[0] == "L1" && L[1] == "L2");
}

// ---- built-ins: help / ? / list -----------------------------------
void test_builtins() {
    reg().clear();
    bool v = false;
    cv_bool("b.v", "the vee", &v);
    cc::CCommand c; c.name = "b.c"; c.help = "the cee";
    c.fn = [](const std::vector<std::string>&, cc::Output&) {};
    reg().register_command(std::move(c));

    std::vector<std::string> L;
    cc::Output out([&L](const std::string& s) { L.push_back(s); });

    L.clear(); CHECK(reg().execute("help", out) == true);
    CHECK(L.size() == sz(2));
    CHECK(has(L[0], "1 cvars") && has(L[0], "1 commands"));
    CHECK(has(L[1], "Built-ins"));
    L.clear(); reg().execute("?", out);
    CHECK(L.size() == sz(2) && has(L[0], "1 cvars"));
    L.clear(); reg().execute("HELP", out);                // ieq built-in
    CHECK(has(L[0], "1 cvars"));

    L.clear(); CHECK(reg().execute("help b.v", out));
    CHECK(has(L[0], "b.v = ") && has(L[0], "(bool)"));
    L.clear(); CHECK(reg().execute("help B.C", out));     // CI command
    CHECK(has(L[0], "b.c") && has(L[0], "the cee"));
    L.clear(); CHECK(reg().execute("help nope", out));
    CHECK(has(L[0], "not found"));

    L.clear(); reg().execute("list cvars", out);
    bool saw_v = false, saw_c = false;
    for (auto& l : L) { if (has(l, "b.v")) saw_v = true;
                        if (has(l, "b.c")) saw_c = true; }
    CHECK(saw_v && !saw_c);
    L.clear(); reg().execute("list commands", out);
    saw_v = saw_c = false;
    for (auto& l : L) { if (has(l, "b.v")) saw_v = true;
                        if (has(l, "b.c")) saw_c = true; }
    CHECK(!saw_v && saw_c);
    L.clear(); reg().execute("list", out);
    saw_v = saw_c = false;
    for (auto& l : L) { if (has(l, "b.v")) saw_v = true;
                        if (has(l, "b.c")) saw_c = true; }
    CHECK(saw_v && saw_c);
    L.clear();
    CHECK(reg().execute("list bogus", out) == true);
    CHECK(L.empty());                                 // neither selected
}

// ---- unknown / empty inputs ---------------------------------------
void test_unknown_empty() {
    reg().clear();
    std::vector<std::string> L;
    cc::Output out([&L](const std::string& s) { L.push_back(s); });

    L.clear();
    CHECK(reg().execute("nope.thing", out) == false);  // unmatched
    CHECK(L.empty());                                  // no output written
    CHECK(reg().execute("", out) == false);
    CHECK(reg().execute("    ", out) == false);
    CHECK(reg().execute("\t  \t", out) == false);
    CHECK(reg().execute("??", out) == false);          // only "?" is built-in
    CHECK(L.empty());

    bool s = false;
    cv_bool("x.y", "h", &s);
    CHECK(reg().execute("definitely.not", out) == false);  // non-empty reg
    CHECK(L.empty());
}

// ---- Output: printf truncation + null-fmt + empty sink ------------
void test_output() {
    std::vector<std::string> L;
    cc::Output out([&L](const std::string& s) { L.push_back(s); });

    out("plain");
    CHECK(L.back() == "plain");
    out("%d-%s", 42, "z");
    CHECK(L.back() == "42-z");

    const std::string big(3000, 'x');
    out("%s", big.c_str());
    CHECK(L.back().size() == sz(1023));               // buf[1024] − NUL

    const cardinal::usize before = L.size();
    out(nullptr);                                     // null fmt → no-op
    CHECK(L.size() == before);

    out.line("direct");
    CHECK(L.back() == "direct");

    // An Output with no sink must be a safe no-op (no crash).
    cc::Output none(nullptr);
    none("anything %d", 1);
    none.line("x");
    CHECK(true);
}

}  // namespace

// Settings persistence: save every cvar to a file, wipe the live state, then
// load it back — the round-trip that backs the Studio settings file. Also
// pins exec_file's comment/blank skipping (the UE5-style config-is-a-script
// model).
void test_persistence() {
    reg().clear();

    bool        vb = false;
    cardinal::i64 vi = 0;
    double      vf = 0.0;
    std::string vs;

    cc::CVar cb; cb.name = "t.flag"; cb.type = cc::CVarType::Bool;
    cb.get_bool = [&] { return vb; }; cb.set_bool = [&](bool b) { vb = b; };
    reg().register_cvar(std::move(cb));

    cc::CVar ci; ci.name = "t.count"; ci.type = cc::CVarType::Int; ci.min = 0; ci.max = 1000;
    ci.get_int = [&] { return vi; }; ci.set_int = [&](cardinal::i64 v) { vi = v; };
    reg().register_cvar(std::move(ci));

    cc::CVar cf; cf.name = "t.scale"; cf.type = cc::CVarType::Float; cf.min = 0.0; cf.max = 10.0;
    cf.get_float = [&] { return vf; }; cf.set_float = [&](double v) { vf = v; };
    reg().register_cvar(std::move(cf));

    cc::CVar cs; cs.name = "t.name"; cs.type = cc::CVarType::String;
    cs.get_string = [&] { return cardinal::string(vs); };
    cs.set_string = [&](const cardinal::string& s) { vs = s; };
    reg().register_cvar(std::move(cs));

    // Non-default values, incl. a multi-word string (exercises set_string rejoin).
    vb = true; vi = 742; vf = 3.5; vs = "hello world cfg";

    const cardinal::string path = "./.cardinal_settings_test.cfg";
    CHECK(reg().save_cvars(path) == 4);

    // Wipe live state, then reload from the file.
    vb = false; vi = 0; vf = 0.0; vs.clear();
    CHECK(reg().load_cvars(path) == 4);

    CHECK(vb == true);
    CHECK(vi == 742);
    CHECK(ap_d(vf, 3.5));
    CHECK(vs == "hello world cfg");                 // multi-word string round-tripped

    // exec_file skips '#'/';' comments + blank lines; only real lines resolve.
    {
        cardinal::ofstream w("./.cardinal_cfg2.cfg");
        w << "# comment\n\n; also a comment\nt.count 99\n";
    }
    cc::Output sink{[](const cardinal::string&) {}};
    vi = 0;
    CHECK(reg().exec_file("./.cardinal_cfg2.cfg", sink) == 1);
    CHECK(vi == 99);

    std::remove(path.c_str());
    std::remove("./.cardinal_cfg2.cfg");
    reg().clear();
}

int main() {
    test_register();
    test_listing_sorted();
    test_complete();
    test_cvar_read();
    test_cvar_set();
    test_cvar_nonfinite();
    test_command();
    test_builtins();
    test_unknown_empty();
    test_output();
    test_persistence();

    if (g_fail == 0) {
        cardinal::log::infof("contest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("contest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
