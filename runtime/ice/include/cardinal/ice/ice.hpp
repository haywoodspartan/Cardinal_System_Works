#pragma once

// =============================================================================
// Cardinal ICE — the modernized Softimage "Interactive Creative Environment".
//
// XSI's ICE was a node graph evaluated over per-point geometry attribute
// streams, sitting on the operator stack's dependency graph. Cardinal's
// modernization keeps the three ideas and rebuilds them engine-native:
//
//   1. ATTRIBUTE STREAMS (GeoData) — named per-point streams (Vec3 / float),
//      "P" by convention for positions. Nodes read and write whole streams.
//   2. NODE GRAPH (IceGraph) — Get/Const/math/Set nodes wired by explicit
//      input indices. Deterministic, data-parallel per point.
//   3. FAST OPERATOR DEPENDENCY GRAPH (IceEvaluator) — the graph COMPILES
//      once into a flat, topologically-ordered op program ("dynamic graph
//      compilation": graph -> linear program at runtime, no interpreter
//      recursion). Every op carries a dirty bit + a cached output stream;
//      edits propagate dirt DOWNSTREAM only, and evaluate() recomputes just
//      the dirty subgraph — the XSI operator-stack trick that made huge
//      stacks interactive, measurable here via EvalStats.nodes_evaluated.
//
// Pure CPU + headless-deterministic (tests/ice). The Studio node editor and
// a blueprint-VM lowering ride on top of this later.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/core/std/utility.hpp>   // cardinal::move

namespace cardinal::ice {

// Minimal vector type — ICE is engine-core; it must not depend on scene.
struct Vec3 {
    float x{0.0f}, y{0.0f}, z{0.0f};
};

// -----------------------------------------------------------------------------
// GeoData — named per-point attribute streams.
// -----------------------------------------------------------------------------
struct GeoData {
    u32 point_count{0};

    cardinal::unordered_map<cardinal::string, cardinal::vector<Vec3>>  vec3;
    cardinal::unordered_map<cardinal::string, cardinal::vector<float>> scalar;

    cardinal::vector<Vec3>& ensure_vec3(const cardinal::string& name) {
        auto& v = vec3[name];
        v.resize(point_count);
        return v;
    }
    cardinal::vector<float>& ensure_scalar(const cardinal::string& name) {
        auto& v = scalar[name];
        v.resize(point_count);
        return v;
    }
};

// -----------------------------------------------------------------------------
// Nodes
// -----------------------------------------------------------------------------
enum class NodeKind : u8 {
    // Sources
    GetVec3,      // reads GeoData.vec3[attr]
    GetFloat,     // reads GeoData.scalar[attr]
    ConstVec3,
    ConstFloat,
    Noise,        // deterministic per-point hash noise of a Vec3 input -> float
                  // in [0,1) (splitmix-style; bit-stable across platforms)
    // Vec3 math
    Add,          // v3 + v3
    Sub,          // v3 - v3
    Scale,        // v3 * f
    Cross,        // v3 x v3
    Normalize,    // v3 -> unit v3 (zero-safe: {0,0,0})
    // Float math
    Dot,          // v3 . v3 -> f
    Length,       // |v3| -> f
    MulF,         // f * f
    AddF,         // f + f
    // Sinks
    SetVec3,      // writes GeoData.vec3[attr]
    SetFloat,     // writes GeoData.scalar[attr]
};

// Whether a node's OUTPUT is a Vec3 stream or a float stream.
bool node_outputs_vec3(NodeKind k) noexcept;

struct Node {
    NodeKind              kind{NodeKind::ConstFloat};
    cardinal::string      attr;          // Get*/Set* stream name
    float                 fval{0.0f};    // ConstFloat
    Vec3                  vval{};        // ConstVec3
    cardinal::vector<int> inputs;        // upstream node indices
};

// -----------------------------------------------------------------------------
// IceGraph — build-side container. Nodes reference inputs by index; sinks
// (SetVec3/SetFloat) are the roots the evaluator pulls from.
// -----------------------------------------------------------------------------
class IceGraph {
public:
    int add(Node n) {
        nodes_.push_back(cardinal::move(n));
        return static_cast<int>(nodes_.size()) - 1;
    }
    // Convenience builders (return the node index).
    int get_vec3(cardinal::string attr);
    int get_float(cardinal::string attr);
    int const_vec3(Vec3 v);
    int const_float(float f);
    int noise(int v3_in);
    int add_v3(int a, int b);
    int sub_v3(int a, int b);
    int scale(int v3_in, int f_in);
    int cross(int a, int b);
    int normalize(int v3_in);
    int dot(int a, int b);
    int length(int v3_in);
    int mul_f(int a, int b);
    int add_f(int a, int b);
    int set_vec3(cardinal::string attr, int v3_in);
    int set_float(cardinal::string attr, int f_in);

    const cardinal::vector<Node>& nodes() const noexcept { return nodes_; }
    cardinal::vector<Node>&       nodes() noexcept       { return nodes_; }

private:
    cardinal::vector<Node> nodes_;
};

// -----------------------------------------------------------------------------
// IceEvaluator — compile once, evaluate incrementally.
// -----------------------------------------------------------------------------
struct EvalStats {
    u32 nodes_total{0};
    u32 nodes_evaluated{0};   // dirty subset actually recomputed this call
};

class IceEvaluator {
public:
    // Flatten + topo-order the graph into the op program. Returns false on a
    // malformed graph (bad input index, arity mismatch, type mismatch,
    // cycle). Compiling marks everything dirty.
    bool compile(const IceGraph& g, cardinal::string* error = nullptr);

    // Mark the graph edit surface dirty:
    //   input stream changed        -> touch_input("P")
    //   a Const node's value edited -> set_const_float / set_const_vec3
    // Dirt propagates to every downstream op; clean ops keep their caches.
    void touch_input(const cardinal::string& attr);
    bool set_const_float(int node, float v);
    bool set_const_vec3(int node, Vec3 v);

    // Evaluate over `geo`: reads Get* streams, writes Set* streams. Only
    // dirty ops recompute; first call after compile() evaluates everything.
    // Changing geo.point_count implicitly re-evaluates all (caches resize).
    bool evaluate(GeoData& geo, EvalStats* stats = nullptr);

    u32 op_count() const noexcept { return static_cast<u32>(ops_.size()); }

private:
    struct Op {
        NodeKind              kind{NodeKind::ConstFloat};
        cardinal::string      attr;
        float                 fval{0.0f};
        Vec3                  vval{};
        cardinal::vector<int> inputs;      // indices into ops_ (topo order)
        bool                  dirty{true};
        // Output cache (one of the two, by node_outputs_vec3(kind)).
        cardinal::vector<Vec3>  out_v;
        cardinal::vector<float> out_f;
    };

    void mark_downstream_dirty_(int op);

    cardinal::vector<Op>          ops_;          // topo order
    cardinal::vector<cardinal::vector<int>> users_;   // op -> downstream ops
    cardinal::vector<int>         node_to_op_;   // original node idx -> op idx
    u32                           cached_points_{0};
};

}  // namespace cardinal::ice
