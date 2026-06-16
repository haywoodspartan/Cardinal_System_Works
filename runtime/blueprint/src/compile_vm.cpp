// =============================================================================
// Cardinal Blueprint — graph -> Cardinal VM bytecode (execution backend).
//
// Straight-line compile (the DSL has no branches yet). Kept in its own TU so
// the blueprint<->source core has no VM dependency; only this file pulls in
// cardinal::vm via the stable ModuleBuilder seam.
// =============================================================================
#include <cardinal/blueprint/blueprint.hpp>

#include <cardinal/vm/vm.hpp>
#include <cardinal/core/std/cstdio.hpp>    // snprintf (error formatting)

namespace cardinal::blueprint {

namespace {

// add/sub/mul/div -> the VM's f64 arithmetic ops (numbers are f64 here).
bool arith_op(const cardinal::string& fn, cardinal::vm::Op& out) {
    if (fn == "add") { out = cardinal::vm::Op::FAdd; return true; }
    if (fn == "sub") { out = cardinal::vm::Op::FSub; return true; }
    if (fn == "mul") { out = cardinal::vm::Op::FMul; return true; }
    if (fn == "div") { out = cardinal::vm::Op::FDiv; return true; }
    return false;
}

// name -> local slot for one function (params + let-bound outputs). Functions
// are small, so a linear vector beats pulling in a hash map.
struct SlotMap {
    cardinal::vector<cardinal::string> names;
    bool find(const cardinal::string& n, u16& out) const {
        for (usize i = 0; i < names.size(); ++i)
            if (names[i] == n) { out = static_cast<u16>(i); return true; }
        return false;
    }
    u16 add(const cardinal::string& n) {
        u16 s; if (find(n, s)) return s;
        names.push_back(n);
        return static_cast<u16>(names.size() - 1);
    }
};

const HostBinding* find_host(const cardinal::vector<HostBinding>& hosts,
                             const cardinal::string& name) {
    for (const auto& h : hosts) if (h.name == name) return &h;
    return nullptr;
}

bool fail(cardinal::string* err, const char* fnname, const char* msg) {
    if (err != nullptr) {
        char buf[200];
        cardinal::snprintf(buf, sizeof(buf), "func '%s': %s", fnname, msg);
        *err = buf;
    }
    return false;
}

bool compile_func(const Function& f,
                  const cardinal::vector<HostBinding>& hosts,
                  cardinal::vm::ModuleBuilder& mb,
                  cardinal::string* err) {
    using cardinal::vm::Op;

    // --- slot allocation: params first, then each let-bound output ----------
    SlotMap slots;
    for (const auto& p : f.params) slots.add(p.name);
    const u32 num_params = static_cast<u32>(slots.names.size());
    for (const Node& n : f.nodes)
        if (!n.out.empty()) slots.add(n.out);
    const u32 num_locals = static_cast<u32>(slots.names.size());

    mb.begin_func(f.name.c_str(), num_params, num_locals);

    // Emit one input (literal -> push; var-ref -> load its slot).
    auto emit_arg = [&](const Arg& a) -> bool {
        if (a.is_literal) { mb.push_f64(a.literal); return true; }
        u16 s;
        if (!slots.find(a.var, s)) return false;   // undefined variable
        mb.load_local(s);
        return true;
    };

    bool ended_with_return = false;
    for (const Node& n : f.nodes) {
        ended_with_return = false;
        switch (n.kind) {
        case NodeKind::Literal: {
            mb.push_f64(n.literal);
            u16 s; slots.find(n.out, s);
            mb.store_local(s);
            break;
        }
        case NodeKind::Return: {
            if (!n.args.empty()) { if (!emit_arg(n.args[0])) return fail(err, f.name.c_str(), "undefined variable in return"); }
            else                 mb.push_f64(0.0);
            mb.op(Op::Ret);
            ended_with_return = true;
            break;
        }
        case NodeKind::Call: {
            Op aop;
            if (n.fn == "copy") {
                if (n.args.size() != 1) return fail(err, f.name.c_str(), "'copy' needs 1 argument");
                if (!emit_arg(n.args[0])) return fail(err, f.name.c_str(), "undefined variable");
            } else if (arith_op(n.fn, aop)) {
                if (n.args.size() != 2) return fail(err, f.name.c_str(), "arithmetic needs 2 arguments");
                if (!emit_arg(n.args[0]) || !emit_arg(n.args[1]))
                    return fail(err, f.name.c_str(), "undefined variable");
                mb.op(aop);
            } else {
                const HostBinding* h = find_host(hosts, n.fn);
                if (h == nullptr) return fail(err, f.name.c_str(), "unknown operation (no host binding)");
                if (h->arity != n.args.size()) return fail(err, f.name.c_str(), "host call arity mismatch");
                for (const Arg& a : n.args)
                    if (!emit_arg(a)) return fail(err, f.name.c_str(), "undefined variable");
                mb.hostcall(h->host_index, static_cast<u8>(h->arity));
            }
            // Store the produced value, or drop it for an exec-only node.
            if (!n.out.empty()) { u16 s; slots.find(n.out, s); mb.store_local(s); }
            else                  mb.op(Op::Drop);
            break;
        }
        }
    }
    if (!ended_with_return) { mb.push_f64(0.0); mb.op(Op::Ret); }   // guarantee a return
    mb.end_func();
    return true;
}

}  // namespace

bool compile(const Graph& g, const cardinal::vector<HostBinding>& hosts,
             cardinal::vector<u8>& out_bytes, cardinal::string* err) {
    if (g.funcs.empty()) {
        if (err) *err = "empty graph (no functions)";
        return false;
    }
    cardinal::vm::ModuleBuilder mb;
    for (const Function& f : g.funcs)
        if (!compile_func(f, hosts, mb, err)) return false;
    out_bytes = mb.finish();
    return true;
}

}  // namespace cardinal::blueprint
