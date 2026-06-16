#pragma once

// =============================================================================
// Cardinal Blueprint — interoperable visual scripting.
//
// A blueprint is a node GRAPH of logic blocks. This module is the round-trip
// engine that makes the graph and its SOURCE TEXT two views of one thing:
//
//     graph  --generate_source-->  text   (codegen: one op per node, flat)
//     text   --parse_source----->  graph  (parser: flattens nesting to nodes)
//
// so building blocks emits editable source in real time, and editing that
// source re-derives the graph. We own BOTH directions (a small purpose-built
// DSL) precisely so the round-trip is lossless — Lua's parser is one-way and
// the VM is binary, neither of which round-trips to human-tweakable text.
//
// The DSL is deliberately tiny + canonical (three-address form): every node is
// `let <out> = <fn>(<args>)`, args are literals or variable refs (= wires).
// A graph thus maps 1:1 to readable lines; the parser flattens any nested
// expression a human types (add(mul(a,b),c)) into intermediate nodes, so the
// blueprint stays a flat node graph (UE-style) regardless of how source is
// written.
//
//   Grammar (recursive descent):
//     program   := func+
//     func      := 'func' IDENT '(' params? ')' '{' statement* '}'
//     params    := param (',' param)*        param := IDENT (':' IDENT)?
//     statement := 'let' IDENT '=' expr | expr | 'return' expr?
//     expr      := NUMBER | IDENT | IDENT '(' (expr (',' expr)*)? ')'
//
// Execution (compile DSL -> VM bytecode via cardinal::vm::ModuleBuilder) is a
// later slice; this module is the graph<->source interop foundation and is pure
// CPU + headless-testable.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/core/std/utility.hpp>   // cardinal::move

namespace cardinal::blueprint {

// A typed function parameter (type optional; "" = untyped/inferred).
struct Param {
    cardinal::string name;
    cardinal::string type;
};

// One input edge of a node: an inline numeric literal, or a reference to a
// variable — another node's named output or a function parameter (= a wire).
struct Arg {
    bool             is_literal{false};
    f64              literal{0.0};
    cardinal::string var;        // valid when !is_literal

    static Arg lit(f64 v)                  { return Arg{true,  v,   {}}; }
    static Arg ref(cardinal::string name)  { return Arg{false, 0.0, cardinal::move(name)}; }
};

enum class NodeKind : u32 {
    Call,      // out = fn(args)   (out may be empty -> a bare action/exec node)
    Literal,   // out = literal    (a constant value block)
    Return,    // return arg0 (or void if args empty)
};

// One logic block. id is stable within a Function (assigned on build/parse).
struct Node {
    u32                   id{0};
    NodeKind              kind{NodeKind::Call};
    cardinal::string      out;        // named output ("" = none / exec-only)
    cardinal::string      fn;         // Call: function/operation name
    f64                   literal{0.0};   // Literal: the constant
    cardinal::vector<Arg> args;       // input wires (Call / Return)
};

// A graph = a function: signature + an ordered list of nodes (exec order).
struct Function {
    cardinal::string        name;
    cardinal::vector<Param> params;
    cardinal::vector<Node>  nodes;
};

struct Graph {
    cardinal::vector<Function> funcs;
};

// ---- Round-trip engine ------------------------------------------------------

// Graph -> canonical source text. Deterministic, flat three-address form:
// `func name(a, b) {`  /  `  let v = fn(a, 2)`  /  `  return v`  /  `}`.
cardinal::string generate_source(const Graph& g);

// Source text -> graph. Nested call expressions are flattened into intermediate
// "_t<N>" nodes so hand-written nesting becomes discrete blueprint nodes.
// Returns false and fills `err` (if non-null) with "line N: message" on a
// syntax error; `out` is left in a partial state on failure.
bool parse_source(const cardinal::string& src, Graph& out, cardinal::string* err = nullptr);

// Convenience: canonicalize source (parse then regenerate). Idempotent — the
// fixed point is the form generate_source emits. Useful to normalize a
// hand-edited buffer before diffing against the live graph.
bool canonicalize(const cardinal::string& src, cardinal::string& out,
                  cardinal::string* err = nullptr);

// =============================================================================
// Execution — compile a graph to Cardinal VM bytecode (cardinal::vm).
//
// The current DSL is straight-line (no branches yet), so this is a linear
// compile: function params + every let-bound output become VM locals; numeric
// arithmetic add/sub/mul/div compile to the VM's f64 ops, `copy` to a
// load/store, and ANY other call compiles to a HOST CALL resolved through
// `hosts` (operation name -> a registered vm host-fn index + arity). Numbers
// are f64 (VM cells carry the bits); a function's return value is the f64 bits
// of its returned expression (0 for a void return / fall-through).
//
// Returns false and fills `err` on an unknown operation, a host-arity mismatch,
// or a reference to an undefined variable. The emitted bytes load() into a
// vm::Module that vm::VM::call() runs (see tests/blueprint for the round trip).
// =============================================================================
struct HostBinding {
    cardinal::string name;          // operation name as it appears in the graph
    u32              host_index{0}; // index into the VM's host-fn table
    u32              arity{0};      // exact arg count (verified)
};

bool compile(const Graph& g, const cardinal::vector<HostBinding>& hosts,
             cardinal::vector<u8>& out_bytes, cardinal::string* err = nullptr);

// =============================================================================
// Document — live, bidirectional graph<->source binding.
//
// Keeps a graph and its source buffer in sync in real time: the editor mutates
// ONE side and the Document re-derives the other. It caches the text it last
// generated so it can tell its OWN writes from genuine user edits — without
// that, regenerating source after a block edit would read back as a "source
// edit" and loop forever. The graph is the model; source is the editable view.
//
//   block edit:   doc.commit_graph(g)      -> source() becomes canonical text
//   source edit:  doc.edit_source(text)    -> graph() re-derived (if it parses)
//
// On a source syntax error the graph is left intact, error() holds "line N: ...",
// and source() retains the user's in-progress text so they can fix it.
// =============================================================================
class Document {
public:
    Document() = default;
    explicit Document(Graph g) { commit_graph(cardinal::move(g)); }

    const Graph&            graph()  const noexcept { return graph_; }
    const cardinal::string& source() const noexcept { return source_; }
    const cardinal::string& error()  const noexcept { return error_; }
    bool                    ok()     const noexcept { return error_.empty(); }

    // The blocks/graph were edited: adopt `g` and regenerate canonical source.
    void commit_graph(Graph g) {
        graph_  = cardinal::move(g);
        source_ = generate_source(graph_);
        gen_    = source_;
        error_.clear();
    }

    // The source buffer was edited to `text`. If `text` is exactly what we last
    // generated, it's our own echo -> no-op (returns false). Otherwise parse it:
    //   success -> adopt the new graph, re-canonicalize source(), clear error,
    //              return true (the graph changed);
    //   failure -> keep the graph, set error(), hold `text` in source() so the
    //              user can keep editing, return false.
    bool edit_source(const cardinal::string& text) {
        if (text == gen_) { source_ = text; return false; }   // our own write
        Graph g;
        cardinal::string err;
        if (!parse_source(text, g, &err)) {
            source_ = text;                  // keep the in-progress (broken) text
            error_  = cardinal::move(err);
            return false;
        }
        graph_  = cardinal::move(g);
        source_ = generate_source(graph_);   // normalize to canonical form
        gen_    = source_;
        error_.clear();
        return true;
    }

private:
    Graph            graph_;
    cardinal::string source_;
    cardinal::string gen_;      // last text generate_source produced (echo guard)
    cardinal::string error_;
};

}  // namespace cardinal::blueprint
