// =============================================================================
// Cardinal — Blueprint graph <-> source round-trip suite.
//
// The interop guarantee: a graph generates canonical source, and that source
// parses back to an equivalent graph (graph -> text -> graph is structure-
// preserving), AND a hand-written source with NESTED expressions flattens into
// discrete nodes whose regenerated text is the canonical fixed point
// (text -> graph -> text is idempotent). This is what lets "edit the source" and
// "edit the blocks" stay two views of one model. Pure CPU, headless. Exit 0 = pass.
// =============================================================================

#include <cardinal/blueprint/blueprint.hpp>
#include <cardinal/vm/vm.hpp>
#include <cardinal/core/diag/log.hpp>

#include <cstring>   // std::memcpy (f64<->i64 cell reinterpret)

namespace {

namespace bp = cardinal::blueprint;

cardinal::i64 f2i(cardinal::f64 v) { cardinal::i64 r; std::memcpy(&r, &v, 8); return r; }
cardinal::f64 i2f(cardinal::i64 v) { cardinal::f64 r; std::memcpy(&r, &v, 8); return r; }

// Host fn for the host-call test: records its (f64) argument, returns it.
cardinal::f64 g_host_seen = 0.0;
cardinal::i64 host_emit(cardinal::vm::HostContext&, const cardinal::i64* args,
                        cardinal::u32 n) noexcept {
    if (n >= 1) g_host_seen = i2f(args[0]);
    return n >= 1 ? args[0] : 0;
}

int g_checks = 0, g_fail = 0;
void check_impl(bool ok, const char* e, int l) {
    ++g_checks;
    if (!ok) { ++g_fail; cardinal::log::errorf("bptest", "FAIL  L%d  %s", l, e); }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool args_equal(const cardinal::vector<bp::Arg>& a, const cardinal::vector<bp::Arg>& b) {
    if (a.size() != b.size()) return false;
    for (cardinal::usize i = 0; i < a.size(); ++i) {
        if (a[i].is_literal != b[i].is_literal) return false;
        if (a[i].is_literal) { if (a[i].literal != b[i].literal) return false; }
        else                 { if (a[i].var != b[i].var) return false; }
    }
    return true;
}

bool graphs_equal(const bp::Graph& a, const bp::Graph& b) {
    if (a.funcs.size() != b.funcs.size()) return false;
    for (cardinal::usize fi = 0; fi < a.funcs.size(); ++fi) {
        const bp::Function& fa = a.funcs[fi];
        const bp::Function& fb = b.funcs[fi];
        if (fa.name != fb.name) return false;
        if (fa.params.size() != fb.params.size()) return false;
        for (cardinal::usize p = 0; p < fa.params.size(); ++p)
            if (fa.params[p].name != fb.params[p].name ||
                fa.params[p].type != fb.params[p].type) return false;
        if (fa.nodes.size() != fb.nodes.size()) return false;
        for (cardinal::usize n = 0; n < fa.nodes.size(); ++n) {
            const bp::Node& na = fa.nodes[n];
            const bp::Node& nb = fb.nodes[n];
            if (na.kind != nb.kind || na.out != nb.out || na.fn != nb.fn) return false;
            if (na.kind == bp::NodeKind::Literal && na.literal != nb.literal) return false;
            if (!args_equal(na.args, nb.args)) return false;
        }
    }
    return true;
}

// Build a canonical graph by hand, generate source, parse it back, assert the
// graph survives the round-trip (graph -> text -> graph).
void test_graph_to_source_roundtrip() {
    bp::Graph g;
    bp::Function f;
    f.name = "tick";
    f.params.push_back({ "dt", "float" });
    // let speed = 5
    { bp::Node n; n.kind = bp::NodeKind::Literal; n.out = "speed"; n.literal = 5.0;
      f.nodes.push_back(n); }
    // let v = mul(speed, dt)
    { bp::Node n; n.kind = bp::NodeKind::Call; n.out = "v"; n.fn = "mul";
      n.args.push_back(bp::Arg::ref("speed")); n.args.push_back(bp::Arg::ref("dt"));
      f.nodes.push_back(n); }
    // move(v)   -- exec node, no output
    { bp::Node n; n.kind = bp::NodeKind::Call; n.fn = "move";
      n.args.push_back(bp::Arg::ref("v")); f.nodes.push_back(n); }
    // return v
    { bp::Node n; n.kind = bp::NodeKind::Return; n.args.push_back(bp::Arg::ref("v"));
      f.nodes.push_back(n); }
    g.funcs.push_back(cardinal::move(f));

    const cardinal::string src = bp::generate_source(g);
    bp::Graph g2;
    cardinal::string err;
    CHECK(bp::parse_source(src, g2, &err));
    if (!err.empty()) cardinal::log::errorf("bptest", "parse err: %s", err.c_str());
    CHECK(graphs_equal(g, g2));

    // Spot-check the emitted text is the readable canonical form.
    CHECK(src.find("func tick(dt: float) {") != cardinal::string::npos);
    CHECK(src.find("let v = mul(speed, dt)") != cardinal::string::npos);
    CHECK(src.find("move(v)") != cardinal::string::npos);
    CHECK(src.find("return v") != cardinal::string::npos);
}

// Hand-written source with NESTED calls flattens to discrete nodes, and the
// regenerated text is the canonical fixed point (idempotent canonicalize).
void test_source_nesting_flattens() {
    const cardinal::string src =
        "func f(a, b, c) {\n"
        "  let x = add(mul(a, b), c)\n"
        "  return x\n"
        "}\n";
    bp::Graph g;
    cardinal::string err;
    CHECK(bp::parse_source(src, g, &err));
    CHECK(g.funcs.size() == 1u);

    // mul -> temp node, add -> x, return -> 3 nodes total.
    const bp::Function& f = g.funcs[0];
    CHECK(f.nodes.size() == 3u);
    CHECK(f.nodes[0].fn == cardinal::string("mul"));      // _t0 = mul(a, b)
    CHECK(f.nodes[0].out == cardinal::string("_t0"));
    CHECK(f.nodes[1].fn == cardinal::string("add") && f.nodes[1].out == cardinal::string("x"));
    CHECK(f.nodes[1].args.size() == 2u);
    CHECK(!f.nodes[1].args[0].is_literal &&
          f.nodes[1].args[0].var == cardinal::string("_t0"));   // wired to the temp
    CHECK(f.nodes[2].kind == bp::NodeKind::Return);

    // Canonicalize is idempotent: canon(src) == canon(canon(src)).
    cardinal::string c1, c2;
    CHECK(bp::canonicalize(src, c1, &err));
    CHECK(bp::canonicalize(c1, c2, &err));
    CHECK(c1 == c2);
    CHECK(c1.find("let _t0 = mul(a, b)") != cardinal::string::npos);
    CHECK(c1.find("let x = add(_t0, c)") != cardinal::string::npos);
}

// Literals (int + float) round-trip exactly; bare-var alias becomes a copy node.
void test_literals_and_alias() {
    const cardinal::string src =
        "func g() {\n"
        "  let n = 42\n"
        "  let r = 2.5\n"
        "  let m = n\n"
        "  spawn(n, r, m)\n"
        "}\n";
    bp::Graph g;
    cardinal::string err;
    CHECK(bp::parse_source(src, g, &err));
    const bp::Function& f = g.funcs[0];
    CHECK(f.nodes.size() == 4u);
    CHECK(f.nodes[0].kind == bp::NodeKind::Literal && f.nodes[0].literal == 42.0);
    CHECK(f.nodes[1].kind == bp::NodeKind::Literal && f.nodes[1].literal == 2.5);
    CHECK(f.nodes[2].fn == cardinal::string("copy"));   // let m = n -> reroute
    CHECK(f.nodes[3].out.empty() && f.nodes[3].fn == cardinal::string("spawn"));

    cardinal::string regen = bp::generate_source(g);
    CHECK(regen.find("let n = 42") != cardinal::string::npos);
    CHECK(regen.find("let r = 2.5") != cardinal::string::npos);
}

// Syntax errors are reported with a line number, not a crash.
void test_parse_errors() {
    bp::Graph g; cardinal::string err;
    CHECK(!bp::parse_source("func {", g, &err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!bp::parse_source("func f( {\n}\n", g, &err));   // bad param list
    CHECK(!err.empty());
    err.clear();
    CHECK(!bp::parse_source("func f() {\n let = 5\n}\n", g, &err));   // missing name
    CHECK(!err.empty());
}

// Multiple functions in one source.
void test_multi_function() {
    const cardinal::string src =
        "func a() {\n  ping()\n}\n"
        "func b(x) {\n  return x\n}\n";
    bp::Graph g; cardinal::string err;
    CHECK(bp::parse_source(src, g, &err));
    CHECK(g.funcs.size() == 2u);
    CHECK(g.funcs[0].name == cardinal::string("a"));
    CHECK(g.funcs[1].name == cardinal::string("b"));
    bp::Graph g2;
    CHECK(bp::parse_source(bp::generate_source(g), g2, &err));
    CHECK(graphs_equal(g, g2));
}

// Document: live bidirectional sync, echo guard, error-keeps-graph.
void test_document_live_sync() {
    // Start from a graph (block-built): source is generated canonical text.
    bp::Graph g;
    { bp::Function f; f.name = "run"; f.params.push_back({ "dt", "" });
      bp::Node a; a.kind = bp::NodeKind::Call; a.out = "v"; a.fn = "scale";
      a.args.push_back(bp::Arg::ref("dt")); a.args.push_back(bp::Arg::lit(2));
      f.nodes.push_back(a);
      g.funcs.push_back(cardinal::move(f)); }
    bp::Document doc(cardinal::move(g));
    CHECK(doc.ok());
    CHECK(doc.source().find("let v = scale(dt, 2)") != cardinal::string::npos);

    // Echoing our own generated text back is a no-op (no feedback loop).
    CHECK(!doc.edit_source(doc.source()));
    CHECK(doc.ok());

    // A genuine SOURCE edit re-derives the graph (add a node + nesting).
    const cardinal::string edited =
        "func run(dt) {\n"
        "  let v = scale(dt, 2)\n"
        "  move(add(v, 1))\n"
        "}\n";
    CHECK(doc.edit_source(edited));
    CHECK(doc.ok());
    CHECK(doc.graph().funcs[0].nodes.size() == 3u);   // scale, _t0=add, move
    // Buffer was normalized to canonical form (nesting flattened).
    CHECK(doc.source().find("let _t0 = add(v, 1)") != cardinal::string::npos);
    CHECK(doc.source().find("move(_t0)") != cardinal::string::npos);
    // ...and re-echoing the normalized buffer is again a no-op.
    CHECK(!doc.edit_source(doc.source()));

    // A BLOCK edit (graph mutated by the host) regenerates source.
    bp::Graph g2 = doc.graph();
    { bp::Node r; r.kind = bp::NodeKind::Return; r.args.push_back(bp::Arg::ref("v"));
      g2.funcs[0].nodes.push_back(r); }
    doc.commit_graph(cardinal::move(g2));
    CHECK(doc.source().find("return v") != cardinal::string::npos);

    // A broken source edit keeps the graph + reports the error + holds the text.
    const cardinal::usize nodes_before = doc.graph().funcs[0].nodes.size();
    CHECK(!doc.edit_source("func run(dt) {\n  let = oops\n}\n"));
    CHECK(!doc.ok());
    CHECK(!doc.error().empty());
    CHECK(doc.graph().funcs[0].nodes.size() == nodes_before);   // graph intact
    CHECK(doc.source().find("oops") != cardinal::string::npos); // user text kept

    // Recovering with valid source clears the error.
    CHECK(doc.edit_source("func run(dt) {\n  ping()\n}\n"));
    CHECK(doc.ok());
}

// Execution: compile a graph to VM bytecode + run it, asserting the result.
void test_compile_and_run_arith() {
    // func calc(a, b) { let s = add(a, b)  let p = mul(s, 2)  return p }
    bp::Graph g;
    bp::Function f; f.name = "calc";
    f.params.push_back({ "a", "" }); f.params.push_back({ "b", "" });
    { bp::Node n; n.kind = bp::NodeKind::Call; n.out = "s"; n.fn = "add";
      n.args.push_back(bp::Arg::ref("a")); n.args.push_back(bp::Arg::ref("b"));
      f.nodes.push_back(n); }
    { bp::Node n; n.kind = bp::NodeKind::Call; n.out = "p"; n.fn = "mul";
      n.args.push_back(bp::Arg::ref("s")); n.args.push_back(bp::Arg::lit(2));
      f.nodes.push_back(n); }
    { bp::Node n; n.kind = bp::NodeKind::Return; n.args.push_back(bp::Arg::ref("p"));
      f.nodes.push_back(n); }
    g.funcs.push_back(cardinal::move(f));

    cardinal::vector<cardinal::u8> bytes;
    cardinal::string err;
    CHECK(bp::compile(g, {}, bytes, &err));
    if (!err.empty()) cardinal::log::errorf("bptest", "compile err: %s", err.c_str());

    cardinal::vm::Limits lim{};
    auto mod = cardinal::vm::load(bytes.data(), bytes.size(), lim, &err);
    CHECK(mod != nullptr);
    if (mod == nullptr) { cardinal::log::errorf("bptest", "load err: %s", err.c_str()); return; }

    auto vm = cardinal::vm::VM::create(lim);
    const cardinal::i64 fi = cardinal::vm::module_find_func(*mod, "calc");
    CHECK(fi >= 0);
    cardinal::i64 args[2] = { f2i(3.0), f2i(4.0) };
    cardinal::i64 ret = 0;
    const auto trap = vm->call(*mod, static_cast<cardinal::u32>(fi), args, 2, &ret);
    CHECK(trap == cardinal::vm::Trap::Finished || trap == cardinal::vm::Trap::Halted);
    CHECK(i2f(ret) == 14.0);   // (3 + 4) * 2
}

// Execution: a host call (exec node) actually invokes the bound host fn.
void test_compile_hostcall() {
    // func act(x) { emit(mul(x, 10)) }
    bp::Graph g;
    bp::Function f; f.name = "act"; f.params.push_back({ "x", "" });
    { bp::Node n; n.kind = bp::NodeKind::Call; n.out = "y"; n.fn = "mul";
      n.args.push_back(bp::Arg::ref("x")); n.args.push_back(bp::Arg::lit(10));
      f.nodes.push_back(n); }
    { bp::Node n; n.kind = bp::NodeKind::Call; n.fn = "emit";   // exec, no output
      n.args.push_back(bp::Arg::ref("y")); f.nodes.push_back(n); }
    g.funcs.push_back(cardinal::move(f));

    cardinal::vector<bp::HostBinding> hosts;
    hosts.push_back({ "emit", 0u, 1u });
    cardinal::vector<cardinal::u8> bytes;
    cardinal::string err;
    CHECK(bp::compile(g, hosts, bytes, &err));

    cardinal::vm::Limits lim{};
    auto mod = cardinal::vm::load(bytes.data(), bytes.size(), lim, &err);
    CHECK(mod != nullptr);
    if (mod == nullptr) return;
    auto vm = cardinal::vm::VM::create(lim);
    cardinal::vm::HostFnDesc desc{ "emit", host_emit, 1u };
    vm->set_host_fns(&desc, 1);
    const cardinal::i64 fi = cardinal::vm::module_find_func(*mod, "act");
    CHECK(fi >= 0);
    g_host_seen = 0.0;
    cardinal::i64 args[1] = { f2i(5.0) };
    cardinal::i64 ret = 0;
    const auto trap = vm->call(*mod, static_cast<cardinal::u32>(fi), args, 1, &ret);
    CHECK(trap == cardinal::vm::Trap::Finished || trap == cardinal::vm::Trap::Halted);
    CHECK(g_host_seen == 50.0);   // emit saw mul(5, 10)
}

// Compile errors: unknown op (no binding) + undefined variable.
void test_compile_errors() {
    cardinal::vector<cardinal::u8> bytes;
    cardinal::string err;
    {   // unknown operation with no host binding
        bp::Graph g; bp::Function f; f.name = "u";
        bp::Node n; n.kind = bp::NodeKind::Call; n.fn = "nope";
        f.nodes.push_back(n); g.funcs.push_back(cardinal::move(f));
        CHECK(!bp::compile(g, {}, bytes, &err));
        CHECK(!err.empty());
    }
    {   // reference to an undefined variable
        err.clear();
        bp::Graph g; bp::Function f; f.name = "u";
        bp::Node n; n.kind = bp::NodeKind::Return; n.args.push_back(bp::Arg::ref("ghost"));
        f.nodes.push_back(n); g.funcs.push_back(cardinal::move(f));
        CHECK(!bp::compile(g, {}, bytes, &err));
        CHECK(!err.empty());
    }
}

// End-to-end: hand-written SOURCE -> parse -> compile -> run.
void test_source_to_execution() {
    const cardinal::string src =
        "func calc(a, b) {\n"
        "  return add(mul(a, b), 1)\n"   // nested -> flattened, then compiled
        "}\n";
    bp::Graph g; cardinal::string err;
    CHECK(bp::parse_source(src, g, &err));
    cardinal::vector<cardinal::u8> bytes;
    CHECK(bp::compile(g, {}, bytes, &err));
    cardinal::vm::Limits lim{};
    auto mod = cardinal::vm::load(bytes.data(), bytes.size(), lim, &err);
    CHECK(mod != nullptr);
    if (mod == nullptr) return;
    auto vm = cardinal::vm::VM::create(lim);
    const cardinal::i64 fi = cardinal::vm::module_find_func(*mod, "calc");
    cardinal::i64 args[2] = { f2i(6.0), f2i(7.0) };
    cardinal::i64 ret = 0;
    vm->call(*mod, static_cast<cardinal::u32>(fi), args, 2, &ret);
    CHECK(i2f(ret) == 43.0);   // 6*7 + 1
}

}  // namespace

int main() {
    test_graph_to_source_roundtrip();
    test_source_nesting_flattens();
    test_literals_and_alias();
    test_parse_errors();
    test_multi_function();
    test_document_live_sync();
    test_compile_and_run_arith();
    test_compile_hostcall();
    test_compile_errors();
    test_source_to_execution();
    if (g_fail == 0) {
        cardinal::log::infof("bptest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("bptest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
