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

}  // namespace cardinal::blueprint
