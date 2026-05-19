// =============================================================================
// Cardinal — sandboxed bytecode VM (stack machine) implementation.
//
// load() verifies a module up front (offsets, opcodes, jump targets, local /
// call indices, and a fixed-point operand-stack height pass) so the
// interpreter can run a verified module without the verifier's invariants
// being violatable from inside the bytecode. Runtime guards remain anyway
// (defence in depth + the args path). Everything is deterministic.
// =============================================================================
#include <cardinal/vm/vm.hpp>

#include <cardinal/core/bit.hpp>       // cardinal::bit_cast (type-puns)
#include <cardinal/core/utility.hpp>   // cardinal::move
// cardinal::vector / cardinal::string / cardinal::unique_ptr arrive via
// vm.hpp (core/containers.hpp + core/types.hpp).

namespace cardinal::vm {

namespace {

inline constexpr i64 kI64Min = (-9223372036854775807LL - 1);
inline constexpr i64 kI64Max =   9223372036854775807LL;

// ---- little-endian readers / bit-casts ------------------------------------
inline u16 rd_u16(const u8* p) noexcept {
    return static_cast<u16>(static_cast<u16>(p[0]) | (static_cast<u16>(p[1]) << 8));
}
inline u32 rd_u32(const u8* p) noexcept {
    return  static_cast<u32>(p[0])        | (static_cast<u32>(p[1]) << 8)
         | (static_cast<u32>(p[2]) << 16) | (static_cast<u32>(p[3]) << 24);
}
inline u64 rd_u64(const u8* p) noexcept {
    return static_cast<u64>(rd_u32(p)) | (static_cast<u64>(rd_u32(p + 4)) << 32);
}
inline i32 rd_i32(const u8* p) noexcept {
    return static_cast<i32>(rd_u32(p));
}
constexpr f64 bits_to_f64(u64 b) noexcept { return cardinal::bit_cast<f64>(b); }
constexpr u64 f64_to_bits(f64 d) noexcept { return cardinal::bit_cast<u64>(d); }

// Immediate byte count for an opcode (0 if operand-less). 0xFF => bad opcode.
u32 imm_size(Op o) noexcept {
    switch (o) {
        case Op::PushI64: case Op::PushF64:                 return 8;
        case Op::LoadLocal: case Op::StoreLocal:            return 2;
        case Op::Jmp: case Op::Jz: case Op::Jnz:            return 4;
        case Op::Call:                                      return 4;
        case Op::HostCall:                                  return 5;  // u32+u8
        case Op::Nop: case Op::Halt: case Op::Drop: case Op::Dup:
        case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Mod:
        case Op::Neg: case Op::And: case Op::Or: case Op::Xor:
        case Op::Shl: case Op::Shr: case Op::Not:
        case Op::Eq: case Op::Ne: case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge:
        case Op::FAdd: case Op::FSub: case Op::FMul: case Op::FDiv:
        case Op::I2F: case Op::F2I:
        case Op::Ret:
        case Op::MLoad8: case Op::MLoad16: case Op::MLoad32: case Op::MLoad64:
        case Op::MStore8: case Op::MStore16: case Op::MStore32: case Op::MStore64:
        case Op::MemSize:                                   return 0;
    }
    return 0xFFFFFFFFu;
}
bool is_known_op(u8 b) noexcept { return imm_size(static_cast<Op>(b)) != 0xFFFFFFFFu; }

}  // namespace

struct Module {
    struct Func {
        u32 num_params{0};
        u32 num_locals{0};
        u32 code_off{0};      // into `code`
        u32 code_len{0};
        cardinal::string name;
    };
    u32                mem_pages{0};
    cardinal::vector<u8>    code;
    cardinal::vector<Func>  funcs;
};

void ModuleDeleter::operator()(Module* m) const noexcept { delete m; }

const char* trap_name(Trap t) noexcept {
    switch (t) {
        case Trap::None:              return "None";
        case Trap::BadModule:         return "BadModule";
        case Trap::BadOpcode:         return "BadOpcode";
        case Trap::StackOverflow:     return "StackOverflow";
        case Trap::StackUnderflow:    return "StackUnderflow";
        case Trap::CallDepth:         return "CallDepth";
        case Trap::OutOfBoundsMemory: return "OutOfBoundsMemory";
        case Trap::DivByZero:         return "DivByZero";
        case Trap::BadCall:           return "BadCall";
        case Trap::BudgetExhausted:   return "BudgetExhausted";
        case Trap::HostError:         return "HostError";
        case Trap::Halted:            return "Halted";
        case Trap::Finished:          return "Finished";
    }
    return "?";
}

u32 module_func_count(const Module& m) noexcept {
    return static_cast<u32>(m.funcs.size());
}
i64 module_find_func(const Module& m, const char* name) noexcept {
    if (name == nullptr) return -1;
    for (usize i = 0; i < m.funcs.size(); ++i)
        if (m.funcs[i].name == name) return static_cast<i64>(i);
    return -1;
}

// ===========================================================================
// Verifier + loader
// ===========================================================================
namespace {

bool fail(cardinal::string* e, const char* msg) { if (e) *e = msg; return false; }

// Per-function structural + stack-height verification.
bool verify_func(const Module& m, const Module::Func& f,
                 const Limits& lim, cardinal::string* err)
{
    const u8* C  = m.code.data() + f.code_off;
    const u32 N  = f.code_len;

    // Pass 1: linear decode -> instruction-start map + per-op index checks.
    cardinal::vector<u8> is_start(N, 0u);
    for (u32 ip = 0; ip < N; ) {
        const u8 ob = C[ip];
        if (!is_known_op(ob)) return fail(err, "bad opcode");
        const Op  o  = static_cast<Op>(ob);
        const u32 is = imm_size(o);
        if (ip + 1u + is > N) return fail(err, "truncated immediate");
        is_start[ip] = 1u;

        if (o == Op::LoadLocal || o == Op::StoreLocal) {
            if (rd_u16(C + ip + 1) >= f.num_locals)
                return fail(err, "local index out of range");
        } else if (o == Op::Call) {
            if (rd_u32(C + ip + 1) >= m.funcs.size())
                return fail(err, "call index out of range");
        } else if (o == Op::HostCall) {
            // index bound is VM-side (runtime); arity must be sane here.
            if (C[ip + 5] > 32u) return fail(err, "hostcall arity too large");
        }
        ip += 1u + is;
    }

    // Pass 2: jump targets must land on an instruction start (or == N = end).
    for (u32 ip = 0; ip < N; ) {
        const Op  o  = static_cast<Op>(C[ip]);
        const u32 is = imm_size(o);
        if (o == Op::Jmp || o == Op::Jz || o == Op::Jnz) {
            const i32 rel = rd_i32(C + ip + 1);
            const i64 tgt = static_cast<i64>(ip) + 1 + 4 + rel;
            if (tgt < 0 || tgt > static_cast<i64>(N))
                return fail(err, "jump target out of range");
            if (tgt != static_cast<i64>(N) && is_start[static_cast<u32>(tgt)] == 0u)
                return fail(err, "jump into mid-instruction");
        }
        ip += 1u + is;
    }

    // Pass 3: operand-stack height fixed point. height[ip] is the stack
    // depth on entry to the instruction at ip; -1 = not yet seen.
    cardinal::vector<i32> h(N + 1u, -1);
    cardinal::vector<u32> work;
    h[0] = 0;
    work.push_back(0u);

    auto want = [&](i32 cur, i32 need, cardinal::string* e2) -> bool {
        if (cur < need) return fail(e2, "operand stack underflow");
        return true;
    };
    auto set_succ = [&](u32 at, i32 hgt, cardinal::string* e2) -> bool {
        if (hgt < 0 || hgt > static_cast<i32>(lim.max_stack))
            return fail(e2, "operand stack overflow");
        if (at == N) return true;                 // function-end sink
        if (h[at] < 0) { h[at] = hgt; work.push_back(at); return true; }
        if (h[at] != hgt) return fail(e2, "stack height merge mismatch");
        return true;
    };

    while (!work.empty()) {
        const u32 ip = work.back(); work.pop_back();
        const Op  o  = static_cast<Op>(C[ip]);
        const u32 is = imm_size(o);
        const u32 nx = ip + 1u + is;            // fall-through ip
        i32       cur = h[ip];

        switch (o) {
            case Op::Nop: case Op::I2F: case Op::F2I: case Op::Neg: case Op::Not:
                if (o != Op::Nop && !want(cur, 1, err)) return false;
                if (!set_succ(nx, cur, err)) return false; break;

            case Op::Halt:
                break;                          // terminator, result optional

            case Op::Drop: case Op::StoreLocal:
                if (!want(cur, 1, err)) return false;
                if (!set_succ(nx, cur - 1, err)) return false; break;

            case Op::Dup:
                if (!want(cur, 1, err)) return false;
                if (!set_succ(nx, cur + 1, err)) return false; break;

            case Op::PushI64: case Op::PushF64: case Op::LoadLocal:
            case Op::MemSize:
                if (!set_succ(nx, cur + 1, err)) return false; break;

            case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Mod:
            case Op::And: case Op::Or: case Op::Xor: case Op::Shl: case Op::Shr:
            case Op::Eq: case Op::Ne: case Op::Lt: case Op::Le: case Op::Gt:
            case Op::Ge: case Op::FAdd: case Op::FSub: case Op::FMul: case Op::FDiv:
                if (!want(cur, 2, err)) return false;
                if (!set_succ(nx, cur - 1, err)) return false; break;

            case Op::MLoad8: case Op::MLoad16: case Op::MLoad32: case Op::MLoad64:
                if (!want(cur, 1, err)) return false;
                if (!set_succ(nx, cur, err)) return false; break;

            case Op::MStore8: case Op::MStore16: case Op::MStore32: case Op::MStore64:
                if (!want(cur, 2, err)) return false;
                if (!set_succ(nx, cur - 2, err)) return false; break;

            case Op::Jmp: {
                const i32 rel = rd_i32(C + ip + 1);
                const u32 tgt = static_cast<u32>(static_cast<i64>(ip) + 5 + rel);
                if (!set_succ(tgt, cur, err)) return false;
            } break;

            case Op::Jz: case Op::Jnz: {
                if (!want(cur, 1, err)) return false;
                const i32 rel = rd_i32(C + ip + 1);
                const u32 tgt = static_cast<u32>(static_cast<i64>(ip) + 5 + rel);
                if (!set_succ(tgt, cur - 1, err)) return false;
                if (!set_succ(nx,  cur - 1, err)) return false;
            } break;

            case Op::Call: {
                const u32 fi = rd_u32(C + ip + 1);
                const i32 np = static_cast<i32>(m.funcs[fi].num_params);
                if (!want(cur, np, err)) return false;
                if (!set_succ(nx, cur - np + 1, err)) return false;
            } break;

            case Op::HostCall: {
                const i32 ar = static_cast<i32>(C[ip + 5]);
                if (!want(cur, ar, err)) return false;
                if (!set_succ(nx, cur - ar + 1, err)) return false;
            } break;

            case Op::Ret:
                break;                          // terminator
        }
    }
    return true;
}

}  // namespace

ModulePtr load(const u8* bytes, usize len,
               const Limits& limits, cardinal::string* out_err)
{
    auto bad = [&](const char* m) -> ModulePtr {
        if (out_err) *out_err = m; return ModulePtr{};
    };
    if (bytes == nullptr || len < 20u)        return bad("module too small");
    if (rd_u32(bytes + 0) != kMagic)          return bad("bad magic");
    if (rd_u32(bytes + 4) != kVersion)        return bad("bad version");
    // bytes+8 = flags (reserved, ignored)
    const u32 mem_pages = rd_u32(bytes + 12);
    const u32 nfunc     = rd_u32(bytes + 16);
    if (mem_pages > limits.max_mem_pages)     return bad("mem_pages exceeds limit");
    if (nfunc == 0u || nfunc > kMaxFuncs)     return bad("bad function count");

    const usize table_off = 20u;
    const usize table_sz  = static_cast<usize>(nfunc) * 16u;
    if (len < table_off + table_sz)           return bad("truncated func table");

    ModulePtr mod(new Module());
    mod->mem_pages = mem_pages;
    mod->funcs.resize(nfunc);

    // First gather descriptors + compute total body size.
    usize body = 0;
    for (u32 i = 0; i < nfunc; ++i) {
        const u8* d = bytes + table_off + static_cast<usize>(i) * 16u;
        Module::Func& f = mod->funcs[i];
        f.num_params = rd_u32(d + 0);
        f.num_locals = rd_u32(d + 4);
        f.code_len   = rd_u32(d + 8);
        const u32 name_len = rd_u32(d + 12);
        if (f.num_locals < f.num_params) return bad("num_locals < num_params");
        if (f.num_locals > 65535u)       return bad("too many locals");
        if (f.code_len == 0u)            return bad("empty function");
        body += static_cast<usize>(name_len) + f.code_len;
        // stash name_len temporarily in code_off (re-used below)
        f.code_off = name_len;
    }
    if (len != table_off + table_sz + body)  return bad("size mismatch");

    // Second pass: slice names + code into the flat blob.
    mod->code.reserve(body);
    usize cur = table_off + table_sz;
    for (u32 i = 0; i < nfunc; ++i) {
        Module::Func& f = mod->funcs[i];
        const u32 name_len = f.code_off;       // stashed
        if (name_len > 0u) {
            f.name.assign(reinterpret_cast<const char*>(bytes + cur), name_len);
            cur += name_len;
        }
        f.code_off = static_cast<u32>(mod->code.size());
        mod->code.insert(mod->code.end(), bytes + cur, bytes + cur + f.code_len);
        cur += f.code_len;
    }

    // Verify every function.
    for (const Module::Func& f : mod->funcs) {
        if (!verify_func(*mod, f, limits, out_err)) return nullptr;
    }
    if (out_err) out_err->clear();
    return mod;
}

// ===========================================================================
// Interpreter
// ===========================================================================
namespace {

class VMImpl final : public VM {
public:
    explicit VMImpl(const Limits& lim) : lim_(lim) {
        u32 pages = 0;
        // memory sized lazily on first call() to the module's request,
        // clamped to the limit; start empty.
        (void)pages;
        ostack_.reserve(lim_.max_stack + 8u);
    }

    void set_host_fns(const HostFnDesc* fns, u32 count) noexcept override {
        host_ = fns; host_n_ = (fns ? count : 0u);
    }
    void set_user(void* user) noexcept override { ctx_.user = user; }

    u8*   memory()            noexcept override { return mem_.empty() ? nullptr : mem_.data(); }
    usize memory_size() const noexcept override { return mem_.size(); }
    u64   instructions_executed() const noexcept override { return executed_; }

    Trap call(const Module& m, u32 func_index,
              const i64* args, u32 nargs, i64* out_ret) noexcept override;

private:
    struct Frame {
        const Module::Func* fn{nullptr};
        u32                 ip{0};
        u32                 locals_base{0};   // into locals_
        u32                 ostack_base{0};   // into ostack_
    };

    Limits                 lim_;
    const HostFnDesc*      host_{nullptr};
    u32                    host_n_{0};
    HostContext            ctx_{};
    cardinal::vector<u8>        mem_;
    cardinal::vector<u64>       ostack_;
    cardinal::vector<u64>       locals_;
    u64                    executed_{0};

    bool push(u64 v) noexcept {
        if (ostack_.size() >= lim_.max_stack) return false;
        ostack_.push_back(v); return true;
    }
};

inline i64  s(u64 b) noexcept { return static_cast<i64>(b); }
inline u64  u(i64 b) noexcept { return static_cast<u64>(b); }

Trap VMImpl::call(const Module& m, u32 func_index,
                  const i64* args, u32 nargs, i64* out_ret) noexcept
{
    executed_ = 0;
    if (func_index >= m.funcs.size()) return Trap::BadCall;

    // (Re)size linear memory to the module's request, clamped to the limit.
    {
        u32 pages = m.mem_pages;
        if (pages > lim_.max_mem_pages) pages = lim_.max_mem_pages;
        const usize want_sz = static_cast<usize>(pages) * kPageSize;
        mem_.assign(want_sz, 0u);
    }
    ostack_.clear();
    locals_.clear();
    ctx_.vm    = this;
    ctx_.error = false;

    cardinal::vector<Frame> frames;
    frames.reserve(lim_.max_call_depth);

    auto enter = [&](u32 fi, u32 argc) -> bool {
        if (frames.size() >= lim_.max_call_depth) return false;
        const Module::Func& fn = m.funcs[fi];
        Frame fr;
        fr.fn          = &fn;
        fr.ip          = 0;
        fr.locals_base = static_cast<u32>(locals_.size());
        fr.ostack_base = static_cast<u32>(ostack_.size());
        locals_.resize(locals_.size() + fn.num_locals, 0u);
        // Move argc values from the caller's operand stack into locals
        // 0..argc-1 (argc == callee num_params for verified CALL; for the
        // top-level call() we copy from `args`).
        (void)argc;
        frames.push_back(fr);
        return true;
    };

    // Top-level frame: copy caller args into locals.
    {
        const Module::Func& fn = m.funcs[func_index];
        u32 ac = nargs;
        if (ac > fn.num_locals) ac = fn.num_locals;
        if (!enter(func_index, 0u)) return Trap::CallDepth;
        Frame& top = frames.back();
        for (u32 i = 0; i < ac; ++i)
            locals_[top.locals_base + i] = u(args ? args[i] : 0);
    }

    u64 budget = lim_.instruction_budget;
    i64 result = 0;

    auto popv = [&](u64& out) -> bool {
        Frame& fr = frames.back();
        if (ostack_.size() <= fr.ostack_base) return false;
        out = ostack_.back(); ostack_.pop_back(); return true;
    };

    while (!frames.empty()) {
        Frame&              fr = frames.back();
        const Module::Func& fn = *fr.fn;
        const u8*           C  = m.code.data() + fn.code_off;

        // Fell off the end => implicit return (result = TOS if any).
        if (fr.ip >= fn.code_len) {
            if (ostack_.size() > fr.ostack_base) result = s(ostack_.back());
            // unwind frame
            ostack_.resize(fr.ostack_base);
            locals_.resize(fr.locals_base);
            frames.pop_back();
            if (frames.empty()) { if (out_ret) *out_ret = result; return Trap::Finished; }
            if (!push(u(result))) return Trap::StackOverflow;
            continue;
        }

        if (budget == 0u) return Trap::BudgetExhausted;
        --budget; ++executed_;

        const Op  o  = static_cast<Op>(C[fr.ip]);
        const u32 is = imm_size(o);
        const u8* IM = C + fr.ip + 1;
        u32 next     = fr.ip + 1u + is;

        u64 a = 0, b = 0;
        switch (o) {
            case Op::Nop: break;
            case Op::Halt: {
                if (ostack_.size() > fr.ostack_base) result = s(ostack_.back());
                if (out_ret) *out_ret = result;
                return Trap::Halted;
            }
            case Op::Drop: if (!popv(a)) return Trap::StackUnderflow; break;
            case Op::Dup:
                if (ostack_.size() <= fr.ostack_base) return Trap::StackUnderflow;
                if (!push(ostack_.back())) return Trap::StackOverflow; break;

            case Op::PushI64: if (!push(rd_u64(IM))) return Trap::StackOverflow; break;
            case Op::PushF64: if (!push(rd_u64(IM))) return Trap::StackOverflow; break;
            case Op::LoadLocal: {
                const u16 sl = rd_u16(IM);
                if (!push(locals_[fr.locals_base + sl])) return Trap::StackOverflow;
            } break;
            case Op::StoreLocal: {
                if (!popv(a)) return Trap::StackUnderflow;
                const u16 sl = rd_u16(IM);
                locals_[fr.locals_base + sl] = a;
            } break;

            case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Mod:
            case Op::And: case Op::Or:  case Op::Xor: case Op::Shl: case Op::Shr:
            case Op::Eq:  case Op::Ne:  case Op::Lt:  case Op::Le:  case Op::Gt:
            case Op::Ge: {
                if (!popv(b) || !popv(a)) return Trap::StackUnderflow;
                const i64 x = s(a), y = s(b);
                i64 r = 0;
                switch (o) {
                    case Op::Add: r = s(a + b); break;
                    case Op::Sub: r = s(a - b); break;
                    case Op::Mul: r = s(a * b); break;
                    case Op::Div:
                        if (y == 0) return Trap::DivByZero;
                        r = (x == kI64Min && y == -1) ? kI64Min : (x / y); break;
                    case Op::Mod:
                        if (y == 0) return Trap::DivByZero;
                        r = (x == kI64Min && y == -1) ? 0 : (x % y); break;
                    case Op::And: r = s(a & b); break;
                    case Op::Or:  r = s(a | b); break;
                    case Op::Xor: r = s(a ^ b); break;
                    case Op::Shl: r = s(a << (b & 63u)); break;
                    case Op::Shr: r = s(a >> (b & 63u)); break;
                    case Op::Eq:  r = (x == y); break;
                    case Op::Ne:  r = (x != y); break;
                    case Op::Lt:  r = (x <  y); break;
                    case Op::Le:  r = (x <= y); break;
                    case Op::Gt:  r = (x >  y); break;
                    case Op::Ge:  r = (x >= y); break;
                    default: break;
                }
                if (!push(u(r))) return Trap::StackOverflow;
            } break;

            case Op::Neg:
                if (!popv(a)) return Trap::StackUnderflow;
                if (!push(u(s(0) - s(a)))) return Trap::StackOverflow; break;
            case Op::Not:
                if (!popv(a)) return Trap::StackUnderflow;
                if (!push(~a)) return Trap::StackOverflow; break;

            case Op::FAdd: case Op::FSub: case Op::FMul: case Op::FDiv: {
                if (!popv(b) || !popv(a)) return Trap::StackUnderflow;
                const f64 x = bits_to_f64(a), y = bits_to_f64(b);
                f64 r = 0.0;
                switch (o) {
                    case Op::FAdd: r = x + y; break;
                    case Op::FSub: r = x - y; break;
                    case Op::FMul: r = x * y; break;
                    case Op::FDiv: r = x / y; break;   // IEEE: /0 -> inf/nan
                    default: break;
                }
                if (!push(f64_to_bits(r))) return Trap::StackOverflow;
            } break;
            case Op::I2F:
                if (!popv(a)) return Trap::StackUnderflow;
                if (!push(f64_to_bits(static_cast<f64>(s(a))))) return Trap::StackOverflow;
                break;
            case Op::F2I: {
                if (!popv(a)) return Trap::StackUnderflow;
                const f64 d = bits_to_f64(a);
                i64 r;
                if (d != d)                           r = 0;                  // NaN
                else if (d >=  9223372036854775808.0) r = kI64Max;
                else if (d <  -9223372036854775808.0) r = kI64Min;
                else                                  r = static_cast<i64>(d);// trunc
                if (!push(u(r))) return Trap::StackOverflow;
            } break;

            case Op::Jmp: {
                const i32 rel = rd_i32(IM);
                next = static_cast<u32>(static_cast<i64>(fr.ip) + 5 + rel);
            } break;
            case Op::Jz: case Op::Jnz: {
                if (!popv(a)) return Trap::StackUnderflow;
                const bool take = (o == Op::Jz) ? (a == 0u) : (a != 0u);
                if (take) {
                    const i32 rel = rd_i32(IM);
                    next = static_cast<u32>(static_cast<i64>(fr.ip) + 5 + rel);
                }
            } break;

            case Op::Call: {
                const u32 fi = rd_u32(IM);
                if (fi >= m.funcs.size()) return Trap::BadCall;
                const Module::Func& callee = m.funcs[fi];
                const u32 np = callee.num_params;
                if (ostack_.size() < fr.ostack_base + np) return Trap::StackUnderflow;
                fr.ip = next;                         // resume here after RET
                if (frames.size() >= lim_.max_call_depth) return Trap::CallDepth;
                // pop np args off caller stack, seed callee locals
                Frame nf;
                nf.fn          = &callee;
                nf.ip          = 0;
                nf.locals_base = static_cast<u32>(locals_.size());
                locals_.resize(locals_.size() + callee.num_locals, 0u);
                for (u32 k = 0; k < np; ++k)
                    locals_[nf.locals_base + (np - 1u - k)] = ostack_.back(), ostack_.pop_back();
                nf.ostack_base = static_cast<u32>(ostack_.size());
                frames.push_back(nf);
                continue;                              // do NOT advance caller ip here
            }
            case Op::Ret: {
                if (ostack_.size() > fr.ostack_base) result = s(ostack_.back());
                ostack_.resize(fr.ostack_base);
                locals_.resize(fr.locals_base);
                frames.pop_back();
                if (frames.empty()) { if (out_ret) *out_ret = result; return Trap::Finished; }
                if (!push(u(result))) return Trap::StackOverflow;
                continue;
            }

            case Op::MLoad8: case Op::MLoad16: case Op::MLoad32: case Op::MLoad64: {
                if (!popv(a)) return Trap::StackUnderflow;
                const usize addr = static_cast<usize>(a);
                u32 n = (o == Op::MLoad8) ? 1u : (o == Op::MLoad16) ? 2u
                       : (o == Op::MLoad32) ? 4u : 8u;
                if (addr + n > mem_.size()) return Trap::OutOfBoundsMemory;
                u64 v = 0;
                for (u32 k = 0; k < n; ++k)
                    v |= static_cast<u64>(mem_[addr + k]) << (8u * k);
                if (!push(v)) return Trap::StackOverflow;
            } break;
            case Op::MStore8: case Op::MStore16: case Op::MStore32: case Op::MStore64: {
                if (!popv(b) || !popv(a)) return Trap::StackUnderflow;  // b=val a=addr
                const usize addr = static_cast<usize>(a);
                u32 n = (o == Op::MStore8) ? 1u : (o == Op::MStore16) ? 2u
                       : (o == Op::MStore32) ? 4u : 8u;
                if (addr + n > mem_.size()) return Trap::OutOfBoundsMemory;
                for (u32 k = 0; k < n; ++k)
                    mem_[addr + k] = static_cast<u8>((b >> (8u * k)) & 0xFFu);
            } break;

            case Op::HostCall: {
                const u32 hi = rd_u32(IM);
                const u8  ar = IM[4];
                if (hi >= host_n_ || host_ == nullptr) return Trap::BadCall;
                const HostFnDesc& hd = host_[hi];
                if (hd.fn == nullptr || hd.arity != ar) return Trap::BadCall;
                if (ostack_.size() < fr.ostack_base + ar) return Trap::StackUnderflow;
                i64 abuf[32];
                for (u32 k = 0; k < ar; ++k)
                    abuf[ar - 1u - k] = s(ostack_.back()), ostack_.pop_back();
                const i64 rv = hd.fn(ctx_, abuf, ar);
                if (ctx_.error) return Trap::HostError;
                if (!push(u(rv))) return Trap::StackOverflow;
            } break;

            case Op::MemSize:
                if (!push(static_cast<u64>(mem_.size()))) return Trap::StackOverflow;
                break;
        }
        frames.back().ip = next;
    }
    if (out_ret) *out_ret = result;
    return Trap::Finished;
}

}  // namespace

cardinal::unique_ptr<VM> VM::create(const Limits& limits) {
    return cardinal::make_unique<VMImpl>(limits);
}

// ===========================================================================
// ModuleBuilder
// ===========================================================================
struct ModuleBuilder::Impl {
    struct Fixup { u32 pos{0}; u32 label{0}; };       // i32-slot pos + label id
    struct Fn {
        cardinal::string         name;
        u32                 num_params{0};
        u32                 num_locals{0};
        cardinal::vector<u8>     code;
        cardinal::vector<u32>    labels;                   // label id -> offset / ~0u
        cardinal::vector<Fixup>  fixups;
    };
    u32             mem_pages{1};
    cardinal::vector<Fn> fns;
    i64             cur{-1};

    Fn& f() { return fns[static_cast<usize>(cur)]; }
    void w8 (u8 v)  { f().code.push_back(v); }
    void w16(u16 v) { w8(static_cast<u8>(v & 0xFF)); w8(static_cast<u8>(v >> 8)); }
    void w32(u32 v) { for (int i=0;i<4;++i) w8(static_cast<u8>((v >> (8*i)) & 0xFF)); }
    void w64(u64 v) { for (int i=0;i<8;++i) w8(static_cast<u8>((v >> (8*i)) & 0xFF)); }
};

ModuleBuilder::ModuleBuilder()  : p_(cardinal::make_unique<Impl>()) {}
ModuleBuilder::~ModuleBuilder() = default;

void ModuleBuilder::set_mem_pages(u32 pages) noexcept { p_->mem_pages = pages; }

u32 ModuleBuilder::begin_func(const char* name, u32 num_params, u32 num_locals) noexcept {
    Impl::Fn fn;
    fn.name        = (name ? name : "");
    fn.num_params  = num_params;
    fn.num_locals  = (num_locals < num_params) ? num_params : num_locals;
    p_->fns.push_back(cardinal::move(fn));
    p_->cur = static_cast<i64>(p_->fns.size() - 1);
    return static_cast<u32>(p_->cur);
}
void ModuleBuilder::end_func() noexcept { p_->cur = -1; }

void ModuleBuilder::op(Op o) noexcept { p_->w8(static_cast<u8>(o)); }
void ModuleBuilder::push_i64(i64 v) noexcept { p_->w8(static_cast<u8>(Op::PushI64)); p_->w64(static_cast<u64>(v)); }
void ModuleBuilder::push_f64(f64 v) noexcept { p_->w8(static_cast<u8>(Op::PushF64)); p_->w64(f64_to_bits(v)); }
void ModuleBuilder::load_local(u16 s)  noexcept { p_->w8(static_cast<u8>(Op::LoadLocal));  p_->w16(s); }
void ModuleBuilder::store_local(u16 s) noexcept { p_->w8(static_cast<u8>(Op::StoreLocal)); p_->w16(s); }
void ModuleBuilder::call(u32 fi) noexcept { p_->w8(static_cast<u8>(Op::Call)); p_->w32(fi); }
void ModuleBuilder::hostcall(u32 hi, u8 ar) noexcept {
    p_->w8(static_cast<u8>(Op::HostCall)); p_->w32(hi); p_->w8(ar);
}

u32 ModuleBuilder::make_label() noexcept {
    p_->f().labels.push_back(0xFFFFFFFFu);
    return static_cast<u32>(p_->f().labels.size() - 1);
}
void ModuleBuilder::place_label(u32 label) noexcept {
    p_->f().labels[label] = static_cast<u32>(p_->f().code.size());
}
void ModuleBuilder::jmp(u32 label) noexcept {
    p_->w8(static_cast<u8>(Op::Jmp));
    p_->f().fixups.push_back(Impl::Fixup{ static_cast<u32>(p_->f().code.size()), label });
    p_->w32(0);
}
void ModuleBuilder::jz(u32 label) noexcept {
    p_->w8(static_cast<u8>(Op::Jz));
    p_->f().fixups.push_back(Impl::Fixup{ static_cast<u32>(p_->f().code.size()), label });
    p_->w32(0);
}
void ModuleBuilder::jnz(u32 label) noexcept {
    p_->w8(static_cast<u8>(Op::Jnz));
    p_->f().fixups.push_back(Impl::Fixup{ static_cast<u32>(p_->f().code.size()), label });
    p_->w32(0);
}

cardinal::vector<u8> ModuleBuilder::finish() {
    // Resolve fixups: rel = target - (slot_pos + 4).
    for (Impl::Fn& fn : p_->fns) {
        for (auto& fx : fn.fixups) {
            const u32 slot = fx.pos;
            const u32 lbl  = fx.label;
            const u32 tgt  = fn.labels[lbl];
            const i32 rel  = static_cast<i32>(static_cast<i64>(tgt)
                                              - (static_cast<i64>(slot) + 4));
            const u32 r = static_cast<u32>(rel);
            for (int i = 0; i < 4; ++i)
                fn.code[slot + static_cast<u32>(i)] =
                    static_cast<u8>((r >> (8 * i)) & 0xFF);
        }
    }
    cardinal::vector<u8> out;
    auto p32 = [&](u32 v){ for (int i=0;i<4;++i) out.push_back(static_cast<u8>((v>>(8*i))&0xFF)); };
    p32(kMagic); p32(kVersion); p32(0u);
    p32(p_->mem_pages);
    p32(static_cast<u32>(p_->fns.size()));
    for (Impl::Fn& fn : p_->fns) {
        p32(fn.num_params);
        p32(fn.num_locals);
        p32(static_cast<u32>(fn.code.size()));
        p32(static_cast<u32>(fn.name.size()));
    }
    for (Impl::Fn& fn : p_->fns) {
        out.insert(out.end(), fn.name.begin(), fn.name.end());
        out.insert(out.end(), fn.code.begin(), fn.code.end());
    }
    return out;
}

}  // namespace cardinal::vm
