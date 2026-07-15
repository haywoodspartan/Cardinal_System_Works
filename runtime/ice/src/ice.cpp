// =============================================================================
// Cardinal ICE — implementation.
//
// compile(): Kahn topo sort over the node graph -> flat op program with a
// users_ (downstream) adjacency for dirty propagation. evaluate(): walk the
// program in order, recomputing ONLY dirty ops; each op's output stream is
// cached, so an edit re-evaluates exactly its downstream cone (EvalStats
// proves it). Per-point loops are simple SoA sweeps — data-parallel by
// construction (a jobs/parallel_for split is a later optimisation).
// =============================================================================
#include <cardinal/ice/ice.hpp>

#include <cardinal/core/std/cmath.hpp>     // cardinal::sqrt
#include <cardinal/core/std/utility.hpp>   // cardinal::move

namespace cardinal::ice {

bool node_outputs_vec3(NodeKind k) noexcept {
    switch (k) {
        case NodeKind::GetVec3:
        case NodeKind::ConstVec3:
        case NodeKind::Add:
        case NodeKind::Sub:
        case NodeKind::Scale:
        case NodeKind::Cross:
        case NodeKind::Normalize:
        case NodeKind::SetVec3:     // pass-through of its input
            return true;
        default:
            return false;
    }
}

namespace {

inline u32 arity_of(NodeKind k) noexcept {
    switch (k) {
        case NodeKind::GetVec3:
        case NodeKind::GetFloat:
        case NodeKind::ConstVec3:
        case NodeKind::ConstFloat:  return 0;
        case NodeKind::Noise:
        case NodeKind::Normalize:
        case NodeKind::Length:
        case NodeKind::SetVec3:
        case NodeKind::SetFloat:    return 1;
        default:                    return 2;   // Add/Sub/Scale/Cross/Dot/MulF/AddF
    }
}

// Expected input types: true = Vec3. Scale is (v3, f); everything else is
// homogeneous in its class.
inline bool input_is_vec3(NodeKind k, u32 slot) noexcept {
    switch (k) {
        case NodeKind::Scale:     return slot == 0;
        case NodeKind::Noise:
        case NodeKind::Normalize:
        case NodeKind::Length:
        case NodeKind::Add:
        case NodeKind::Sub:
        case NodeKind::Cross:
        case NodeKind::Dot:
        case NodeKind::SetVec3:   return true;
        default:                  return false;  // MulF/AddF/SetFloat
    }
}

// Deterministic per-point noise: splitmix-style avalanche of the Vec3 bits
// -> [0,1). Bit-stable across platforms/compilers (pure integer math).
inline float hash_noise(const Vec3& p) noexcept {
    auto fbits = [](float f) noexcept -> u64 {
        union { float f; u32 u; } x; x.f = f;
        return static_cast<u64>(x.u);
    };
    u64 h = fbits(p.x) * 0x9E3779B97F4A7C15ull;
    h ^= (fbits(p.y) + 0xBF58476D1CE4E5B9ull) * 0x94D049BB133111EBull;
    h ^= (fbits(p.z) + 0x2545F4914F6CDD1Dull) * 0xD6E8FEB86659FD93ull;
    h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 27; h *= 0x94D049BB133111EBull;
    h ^= h >> 31;
    // Top 24 bits -> [0,1) with exact float representation.
    return static_cast<float>(h >> 40) * (1.0f / 16777216.0f);
}

}  // namespace

// -----------------------------------------------------------------------------
// IceGraph builders
// -----------------------------------------------------------------------------
int IceGraph::get_vec3(cardinal::string attr) {
    Node n; n.kind = NodeKind::GetVec3; n.attr = cardinal::move(attr);
    return add(cardinal::move(n));
}
int IceGraph::get_float(cardinal::string attr) {
    Node n; n.kind = NodeKind::GetFloat; n.attr = cardinal::move(attr);
    return add(cardinal::move(n));
}
int IceGraph::const_vec3(Vec3 v) {
    Node n; n.kind = NodeKind::ConstVec3; n.vval = v;
    return add(cardinal::move(n));
}
int IceGraph::const_float(float f) {
    Node n; n.kind = NodeKind::ConstFloat; n.fval = f;
    return add(cardinal::move(n));
}
int IceGraph::noise(int v3_in) {
    Node n; n.kind = NodeKind::Noise; n.inputs.push_back(v3_in);
    return add(cardinal::move(n));
}
int IceGraph::add_v3(int a, int b) {
    Node n; n.kind = NodeKind::Add; n.inputs.push_back(a); n.inputs.push_back(b);
    return add(cardinal::move(n));
}
int IceGraph::sub_v3(int a, int b) {
    Node n; n.kind = NodeKind::Sub; n.inputs.push_back(a); n.inputs.push_back(b);
    return add(cardinal::move(n));
}
int IceGraph::scale(int v3_in, int f_in) {
    Node n; n.kind = NodeKind::Scale; n.inputs.push_back(v3_in); n.inputs.push_back(f_in);
    return add(cardinal::move(n));
}
int IceGraph::cross(int a, int b) {
    Node n; n.kind = NodeKind::Cross; n.inputs.push_back(a); n.inputs.push_back(b);
    return add(cardinal::move(n));
}
int IceGraph::normalize(int v3_in) {
    Node n; n.kind = NodeKind::Normalize; n.inputs.push_back(v3_in);
    return add(cardinal::move(n));
}
int IceGraph::dot(int a, int b) {
    Node n; n.kind = NodeKind::Dot; n.inputs.push_back(a); n.inputs.push_back(b);
    return add(cardinal::move(n));
}
int IceGraph::length(int v3_in) {
    Node n; n.kind = NodeKind::Length; n.inputs.push_back(v3_in);
    return add(cardinal::move(n));
}
int IceGraph::mul_f(int a, int b) {
    Node n; n.kind = NodeKind::MulF; n.inputs.push_back(a); n.inputs.push_back(b);
    return add(cardinal::move(n));
}
int IceGraph::add_f(int a, int b) {
    Node n; n.kind = NodeKind::AddF; n.inputs.push_back(a); n.inputs.push_back(b);
    return add(cardinal::move(n));
}
int IceGraph::set_vec3(cardinal::string attr, int v3_in) {
    Node n; n.kind = NodeKind::SetVec3; n.attr = cardinal::move(attr);
    n.inputs.push_back(v3_in);
    return add(cardinal::move(n));
}
int IceGraph::set_float(cardinal::string attr, int f_in) {
    Node n; n.kind = NodeKind::SetFloat; n.attr = cardinal::move(attr);
    n.inputs.push_back(f_in);
    return add(cardinal::move(n));
}

// -----------------------------------------------------------------------------
// IceEvaluator
// -----------------------------------------------------------------------------
bool IceEvaluator::compile(const IceGraph& g, cardinal::string* error) {
    ops_.clear();
    users_.clear();
    node_to_op_.clear();
    cached_points_ = 0;

    const auto& nodes = g.nodes();
    const int   n     = static_cast<int>(nodes.size());
    auto fail = [&](const char* msg) {
        if (error) *error = msg;
        ops_.clear(); users_.clear(); node_to_op_.clear();
        return false;
    };

    // Validate arity / indices / input types.
    for (int i = 0; i < n; ++i) {
        const Node& nd = nodes[static_cast<usize>(i)];
        if (nd.inputs.size() != arity_of(nd.kind))
            return fail("node arity mismatch");
        for (u32 s = 0; s < nd.inputs.size(); ++s) {
            const int in = nd.inputs[s];
            if (in < 0 || in >= n || in == i)
                return fail("node input index out of range");
            if (node_outputs_vec3(nodes[static_cast<usize>(in)].kind) !=
                input_is_vec3(nd.kind, s))
                return fail("node input type mismatch");
        }
    }

    // Kahn topo sort.
    cardinal::vector<u32> indeg(static_cast<usize>(n), 0u);
    for (int i = 0; i < n; ++i)
        for (int in : nodes[static_cast<usize>(i)].inputs)
            (void)in, ++indeg[static_cast<usize>(i)];
    // indeg = number of inputs (each input contributes one edge into i).
    cardinal::vector<cardinal::vector<int>> down(static_cast<usize>(n));
    for (int i = 0; i < n; ++i)
        for (int in : nodes[static_cast<usize>(i)].inputs)
            down[static_cast<usize>(in)].push_back(i);

    cardinal::vector<int> queue;
    for (int i = 0; i < n; ++i)
        if (indeg[static_cast<usize>(i)] == 0) queue.push_back(i);

    cardinal::vector<int> order;
    for (usize qi = 0; qi < queue.size(); ++qi) {
        const int cur = queue[qi];
        order.push_back(cur);
        for (int d : down[static_cast<usize>(cur)])
            if (--indeg[static_cast<usize>(d)] == 0) queue.push_back(d);
    }
    if (static_cast<int>(order.size()) != n)
        return fail("graph contains a cycle");

    // Emit the flat program in topo order.
    node_to_op_.assign(static_cast<usize>(n), -1);
    ops_.reserve(static_cast<usize>(n));
    for (int oi = 0; oi < n; ++oi) {
        const int   src = order[static_cast<usize>(oi)];
        const Node& nd  = nodes[static_cast<usize>(src)];
        Op op;
        op.kind = nd.kind;
        op.attr = nd.attr;
        op.fval = nd.fval;
        op.vval = nd.vval;
        for (int in : nd.inputs)
            op.inputs.push_back(node_to_op_[static_cast<usize>(in)]);
        op.dirty = true;
        node_to_op_[static_cast<usize>(src)] = oi;
        ops_.push_back(cardinal::move(op));
    }
    users_.assign(ops_.size(), {});
    for (usize i = 0; i < ops_.size(); ++i)
        for (int in : ops_[i].inputs)
            users_[static_cast<usize>(in)].push_back(static_cast<int>(i));
    return true;
}

void IceEvaluator::mark_downstream_dirty_(int op) {
    if (op < 0 || op >= static_cast<int>(ops_.size())) return;
    if (ops_[static_cast<usize>(op)].dirty) return;   // cone already dirty
    ops_[static_cast<usize>(op)].dirty = true;
    for (int u : users_[static_cast<usize>(op)])
        mark_downstream_dirty_(u);
}

void IceEvaluator::touch_input(const cardinal::string& attr) {
    for (usize i = 0; i < ops_.size(); ++i) {
        Op& op = ops_[i];
        if ((op.kind == NodeKind::GetVec3 || op.kind == NodeKind::GetFloat) &&
            op.attr == attr) {
            op.dirty = false;                       // force full re-mark below
            mark_downstream_dirty_(static_cast<int>(i));
        }
    }
}

bool IceEvaluator::set_const_float(int node, float v) {
    if (node < 0 || node >= static_cast<int>(node_to_op_.size())) return false;
    const int oi = node_to_op_[static_cast<usize>(node)];
    if (oi < 0) return false;
    Op& op = ops_[static_cast<usize>(oi)];
    if (op.kind != NodeKind::ConstFloat) return false;
    op.fval  = v;
    op.dirty = false;
    mark_downstream_dirty_(oi);
    return true;
}

bool IceEvaluator::set_const_vec3(int node, Vec3 v) {
    if (node < 0 || node >= static_cast<int>(node_to_op_.size())) return false;
    const int oi = node_to_op_[static_cast<usize>(node)];
    if (oi < 0) return false;
    Op& op = ops_[static_cast<usize>(oi)];
    if (op.kind != NodeKind::ConstVec3) return false;
    op.vval  = v;
    op.dirty = false;
    mark_downstream_dirty_(oi);
    return true;
}

bool IceEvaluator::evaluate(GeoData& geo, EvalStats* stats) {
    const u32 N = geo.point_count;
    if (N != cached_points_) {                       // resize invalidates all
        cached_points_ = N;
        for (auto& op : ops_) op.dirty = true;
    }

    u32 evaluated = 0;
    for (usize oi = 0; oi < ops_.size(); ++oi) {
        Op& op = ops_[oi];
        if (!op.dirty) continue;
        ++evaluated;
        const usize count = static_cast<usize>(N);
        auto& OV = op.out_v;
        auto& OF = op.out_f;
        auto in_v = [&](u32 s) -> const cardinal::vector<Vec3>& {
            return ops_[static_cast<usize>(op.inputs[s])].out_v;
        };
        auto in_f = [&](u32 s) -> const cardinal::vector<float>& {
            return ops_[static_cast<usize>(op.inputs[s])].out_f;
        };

        switch (op.kind) {
        case NodeKind::GetVec3: {
            OV = geo.ensure_vec3(op.attr);            // copy-in snapshot
            break;
        }
        case NodeKind::GetFloat: {
            OF = geo.ensure_scalar(op.attr);
            break;
        }
        case NodeKind::ConstVec3: {
            OV.assign(count, op.vval);
            break;
        }
        case NodeKind::ConstFloat: {
            OF.assign(count, op.fval);
            break;
        }
        case NodeKind::Noise: {
            const auto& a = in_v(0);
            OF.resize(count);
            for (usize i = 0; i < count; ++i) OF[i] = hash_noise(a[i]);
            break;
        }
        case NodeKind::Add: {
            const auto& a = in_v(0); const auto& b = in_v(1);
            OV.resize(count);
            for (usize i = 0; i < count; ++i)
                OV[i] = { a[i].x + b[i].x, a[i].y + b[i].y, a[i].z + b[i].z };
            break;
        }
        case NodeKind::Sub: {
            const auto& a = in_v(0); const auto& b = in_v(1);
            OV.resize(count);
            for (usize i = 0; i < count; ++i)
                OV[i] = { a[i].x - b[i].x, a[i].y - b[i].y, a[i].z - b[i].z };
            break;
        }
        case NodeKind::Scale: {
            const auto& a = in_v(0); const auto& f = in_f(1);
            OV.resize(count);
            for (usize i = 0; i < count; ++i)
                OV[i] = { a[i].x * f[i], a[i].y * f[i], a[i].z * f[i] };
            break;
        }
        case NodeKind::Cross: {
            const auto& a = in_v(0); const auto& b = in_v(1);
            OV.resize(count);
            for (usize i = 0; i < count; ++i)
                OV[i] = { a[i].y * b[i].z - a[i].z * b[i].y,
                          a[i].z * b[i].x - a[i].x * b[i].z,
                          a[i].x * b[i].y - a[i].y * b[i].x };
            break;
        }
        case NodeKind::Normalize: {
            const auto& a = in_v(0);
            OV.resize(count);
            for (usize i = 0; i < count; ++i) {
                const float l2 = a[i].x * a[i].x + a[i].y * a[i].y + a[i].z * a[i].z;
                if (l2 > 1.0e-12f) {
                    const float inv = 1.0f / cardinal::sqrt(l2);
                    OV[i] = { a[i].x * inv, a[i].y * inv, a[i].z * inv };
                } else {
                    OV[i] = { 0.0f, 0.0f, 0.0f };
                }
            }
            break;
        }
        case NodeKind::Dot: {
            const auto& a = in_v(0); const auto& b = in_v(1);
            OF.resize(count);
            for (usize i = 0; i < count; ++i)
                OF[i] = a[i].x * b[i].x + a[i].y * b[i].y + a[i].z * b[i].z;
            break;
        }
        case NodeKind::Length: {
            const auto& a = in_v(0);
            OF.resize(count);
            for (usize i = 0; i < count; ++i)
                OF[i] = cardinal::sqrt(a[i].x * a[i].x + a[i].y * a[i].y +
                                       a[i].z * a[i].z);
            break;
        }
        case NodeKind::MulF: {
            const auto& a = in_f(0); const auto& b = in_f(1);
            OF.resize(count);
            for (usize i = 0; i < count; ++i) OF[i] = a[i] * b[i];
            break;
        }
        case NodeKind::AddF: {
            const auto& a = in_f(0); const auto& b = in_f(1);
            OF.resize(count);
            for (usize i = 0; i < count; ++i) OF[i] = a[i] + b[i];
            break;
        }
        case NodeKind::SetVec3: {
            const auto& a = in_v(0);
            auto& dst = geo.ensure_vec3(op.attr);
            for (usize i = 0; i < count; ++i) dst[i] = a[i];
            OV = a;                                   // pass-through cache
            break;
        }
        case NodeKind::SetFloat: {
            const auto& a = in_f(0);
            auto& dst = geo.ensure_scalar(op.attr);
            for (usize i = 0; i < count; ++i) dst[i] = a[i];
            OF = a;
            break;
        }
        }
        op.dirty = false;
    }

    if (stats) {
        stats->nodes_total     = static_cast<u32>(ops_.size());
        stats->nodes_evaluated = evaluated;
    }
    return true;
}

}  // namespace cardinal::ice
