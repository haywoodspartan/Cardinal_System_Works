// =============================================================================
// Cardinal — deterministic sandboxed-bytecode-VM regression suite.
//
// Exercises the full ModuleBuilder -> load (verify) -> VM::call contract.
// The verifier is a security boundary, so its REJECTIONS are pinned as
// hard as its acceptances. Everything is deterministic: identical result
// AND identical instruction count across VM instances. Exit 0 = all pass.
// =============================================================================

#include <cardinal/vm/vm.hpp>
#include <cardinal/core/log.hpp>

#include <memory>
#include <string>
#include <vector>

namespace {

namespace vm = cardinal::vm;
using vm::Op;
using vm::Trap;
using vm::Limits;
using cardinal::u8;
using cardinal::u32;
using cardinal::u64;
using cardinal::i32;
using cardinal::i64;
using cardinal::usize;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("vmtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

// Build + load + call a single named function. Returns the trap; result in ret.
Trap run(vm::ModuleBuilder& mb, const char* fn, std::vector<i64> args,
         i64* ret, const Limits& lim = Limits{}, u64* instrs = nullptr) {
    std::vector<u8> bytes = mb.finish();
    std::string err;
    auto m = vm::load(bytes.data(), bytes.size(), lim, &err);
    if (!m) return Trap::BadModule;
    auto v = vm::VM::create(lim);
    i64 fi = vm::module_find_func(*m, fn);
    if (fi < 0) return Trap::BadCall;
    Trap t = v->call(*m, static_cast<u32>(fi),
                      args.empty() ? nullptr : args.data(),
                      static_cast<u32>(args.size()), ret);
    if (instrs) *instrs = v->instructions_executed();
    return t;
}

// Is `bytes` accepted by the verifier?
bool loads_ok(const std::vector<u8>& bytes) {
    std::string err;
    return vm::load(bytes.data(), bytes.size(), Limits{}, &err) != nullptr;
}

// Minimal single-function container around raw code bytes (matches load()).
std::vector<u8> raw_mod(u32 mem_pages, u32 nparams, u32 nlocals,
                        const std::string& name, const std::vector<u8>& code) {
    std::vector<u8> b;
    auto p32 = [&](u32 v) {
        for (int i = 0; i < 4; ++i) b.push_back(static_cast<u8>((v >> (8 * i)) & 0xFFu));
    };
    p32(vm::kMagic); p32(vm::kVersion); p32(0u);
    p32(mem_pages);  p32(1u);
    p32(nparams); p32(nlocals);
    p32(static_cast<u32>(code.size()));
    p32(static_cast<u32>(name.size()));
    b.insert(b.end(), name.begin(), name.end());
    b.insert(b.end(), code.begin(), code.end());
    return b;
}

// ---- arithmetic / locals ------------------------------------------------
void test_basic() {
    vm::ModuleBuilder mb;
    mb.begin_func("add", 0, 0);
    mb.push_i64(2); mb.push_i64(3); mb.op(Op::Add); mb.op(Op::Ret);
    mb.end_func();
    i64 r = -1; u64 n1 = 0;
    CHECK(run(mb, "add", {}, &r, Limits{}, &n1) == Trap::Finished);
    CHECK(r == 5);
    CHECK(n1 > 0);

    vm::ModuleBuilder sb;
    sb.begin_func("sub", 2, 2);
    sb.load_local(0); sb.load_local(1); sb.op(Op::Sub); sb.op(Op::Ret);
    sb.end_func();
    CHECK(run(sb, "sub", {10, 3}, &r) == Trap::Finished);
    CHECK(r == 7);
    CHECK(run(sb, "sub", {3, 10}, &r) == Trap::Finished && r == -7);
}

// ---- loop: sum 1..n -----------------------------------------------------
void test_loop() {
    vm::ModuleBuilder mb;
    mb.begin_func("sumto", 1, 3);              // L0=n L1=i L2=acc
    mb.push_i64(1); mb.store_local(1);
    mb.push_i64(0); mb.store_local(2);
    u32 loop = mb.make_label();
    u32 end  = mb.make_label();
    mb.place_label(loop);
    mb.load_local(1); mb.load_local(0); mb.op(Op::Gt);   // i > n ?
    mb.jnz(end);
    mb.load_local(2); mb.load_local(1); mb.op(Op::Add); mb.store_local(2);
    mb.load_local(1); mb.push_i64(1);   mb.op(Op::Add); mb.store_local(1);
    mb.jmp(loop);
    mb.place_label(end);
    mb.load_local(2); mb.op(Op::Ret);
    mb.end_func();
    i64 r = -1;
    CHECK(run(mb, "sumto", {100}, &r) == Trap::Finished);
    CHECK(r == 5050);
    CHECK(run(mb, "sumto", {1}, &r) == Trap::Finished && r == 1);
    CHECK(run(mb, "sumto", {0}, &r) == Trap::Finished && r == 0);  // loop never runs
}

// ---- recursion: factorial + fib + determinism ---------------------------
void test_recursion() {
    vm::ModuleBuilder mb;
    u32 fi = mb.begin_func("fact", 1, 1);
    mb.load_local(0); mb.push_i64(1); mb.op(Op::Le);
    u32 rec = mb.make_label();
    mb.jz(rec);
    mb.push_i64(1); mb.op(Op::Ret);
    mb.place_label(rec);
    mb.load_local(0);
    mb.load_local(0); mb.push_i64(1); mb.op(Op::Sub);
    mb.call(fi);
    mb.op(Op::Mul);
    mb.op(Op::Ret);
    mb.end_func();

    i64 r = -1; u64 n1 = 0, n2 = 0;
    CHECK(run(mb, "fact", {5},  &r) == Trap::Finished && r == 120);
    CHECK(run(mb, "fact", {10}, &r, Limits{}, &n1) == Trap::Finished && r == 3628800);

    // determinism: same module + args -> identical result AND instr count.
    vm::ModuleBuilder mb2;
    u32 fi2 = mb2.begin_func("fact", 1, 1);
    mb2.load_local(0); mb2.push_i64(1); mb2.op(Op::Le);
    u32 rec2 = mb2.make_label();
    mb2.jz(rec2);
    mb2.push_i64(1); mb2.op(Op::Ret);
    mb2.place_label(rec2);
    mb2.load_local(0);
    mb2.load_local(0); mb2.push_i64(1); mb2.op(Op::Sub);
    mb2.call(fi2);
    mb2.op(Op::Mul); mb2.op(Op::Ret);
    mb2.end_func();
    i64 r2 = -1;
    CHECK(run(mb2, "fact", {10}, &r2, Limits{}, &n2) == Trap::Finished);
    CHECK(r2 == 3628800);
    CHECK(n1 == n2 && n1 > 0);

    vm::ModuleBuilder fb;
    u32 ff = fb.begin_func("fib", 1, 1);
    fb.load_local(0); fb.push_i64(2); fb.op(Op::Lt);
    u32 fr = fb.make_label();
    fb.jz(fr);
    fb.load_local(0); fb.op(Op::Ret);
    fb.place_label(fr);
    fb.load_local(0); fb.push_i64(1); fb.op(Op::Sub); fb.call(ff);
    fb.load_local(0); fb.push_i64(2); fb.op(Op::Sub); fb.call(ff);
    fb.op(Op::Add); fb.op(Op::Ret);
    fb.end_func();
    CHECK(run(fb, "fib", {10}, &r) == Trap::Finished && r == 55);
    CHECK(run(fb, "fib", {15}, &r) == Trap::Finished && r == 610);
}

// ---- linear memory + bounds --------------------------------------------
void test_memory() {
    {
        vm::ModuleBuilder mb;
        mb.set_mem_pages(1);
        mb.begin_func("mem", 0, 0);
        mb.push_i64(16);                          // addr
        mb.push_i64(0x1122334455667788LL);        // val
        mb.op(Op::MStore64);
        mb.push_i64(16); mb.op(Op::MLoad64);
        mb.op(Op::Ret);
        mb.end_func();
        i64 r = 0;
        CHECK(run(mb, "mem", {}, &r) == Trap::Finished);
        CHECK(r == 0x1122334455667788LL);
    }
    {
        vm::ModuleBuilder mb;
        mb.set_mem_pages(1);
        mb.begin_func("ms", 0, 0);
        mb.op(Op::MemSize); mb.op(Op::Ret);
        mb.end_func();
        i64 r = 0;
        CHECK(run(mb, "ms", {}, &r) == Trap::Finished);
        CHECK(r == 65536);
    }
    {   // out-of-bounds load -> clean trap, no crash
        vm::ModuleBuilder mb;
        mb.set_mem_pages(1);
        mb.begin_func("oob", 0, 0);
        mb.push_i64(100000); mb.op(Op::MLoad8); mb.op(Op::Ret);
        mb.end_func();
        i64 r = 0;
        CHECK(run(mb, "oob", {}, &r) == Trap::OutOfBoundsMemory);
    }
    {   // out-of-bounds store -> trap
        vm::ModuleBuilder mb;
        mb.set_mem_pages(1);
        mb.begin_func("oobs", 0, 0);
        mb.push_i64(70000); mb.push_i64(1); mb.op(Op::MStore32); mb.op(Op::Ret);
        mb.end_func();
        i64 r = 0;
        CHECK(run(mb, "oobs", {}, &r) == Trap::OutOfBoundsMemory);
    }
}

// ---- float ops ----------------------------------------------------------
void test_float() {
    vm::ModuleBuilder mb;
    mb.begin_func("f", 0, 0);
    mb.push_f64(3.5); mb.push_f64(1.25); mb.op(Op::FAdd);  // 4.75
    mb.op(Op::F2I);                                        // -> 4 (trunc)
    mb.op(Op::Ret);
    mb.end_func();
    i64 r = 0;
    CHECK(run(mb, "f", {}, &r) == Trap::Finished && r == 4);

    vm::ModuleBuilder mb2;
    mb2.begin_func("g", 0, 0);
    mb2.push_i64(7); mb2.op(Op::I2F);
    mb2.push_f64(0.5); mb2.op(Op::FMul);                   // 3.5
    mb2.op(Op::F2I); mb2.op(Op::Ret);                      // -> 3
    mb2.end_func();
    CHECK(run(mb2, "g", {}, &r) == Trap::Finished && r == 3);

    // FDiv by zero is IEEE (inf), F2I clamps deterministically — NOT a trap.
    vm::ModuleBuilder mb3;
    mb3.begin_func("inf", 0, 0);
    mb3.push_f64(1.0); mb3.push_f64(0.0); mb3.op(Op::FDiv);
    mb3.op(Op::F2I); mb3.op(Op::Ret);
    mb3.end_func();
    CHECK(run(mb3, "inf", {}, &r) == Trap::Finished);
    CHECK(r == 9223372036854775807LL);                     // +inf -> kI64Max
}

// ---- integer traps ------------------------------------------------------
void test_int_traps() {
    {
        vm::ModuleBuilder mb;
        mb.begin_func("d0", 0, 0);
        mb.push_i64(5); mb.push_i64(0); mb.op(Op::Div); mb.op(Op::Ret);
        mb.end_func();
        i64 r = 0;
        CHECK(run(mb, "d0", {}, &r) == Trap::DivByZero);
    }
    {
        vm::ModuleBuilder mb;
        mb.begin_func("m0", 0, 0);
        mb.push_i64(5); mb.push_i64(0); mb.op(Op::Mod); mb.op(Op::Ret);
        mb.end_func();
        i64 r = 0;
        CHECK(run(mb, "m0", {}, &r) == Trap::DivByZero);
    }
    {   // kI64Min / -1 must NOT trap/UB — defined as wrap to kI64Min.
        vm::ModuleBuilder mb;
        mb.begin_func("ov", 0, 0);
        mb.push_i64(-9223372036854775807LL - 1);
        mb.push_i64(-1);
        mb.op(Op::Div); mb.op(Op::Ret);
        mb.end_func();
        i64 r = 0;
        CHECK(run(mb, "ov", {}, &r) == Trap::Finished);
        CHECK(r == (-9223372036854775807LL - 1));
    }
}

// ---- budget / call-depth / halt / fall-off ------------------------------
void test_limits_and_terminators() {
    {   // infinite loop bounded by the instruction budget
        vm::ModuleBuilder mb;
        mb.begin_func("spin", 0, 0);
        u32 l = mb.make_label();
        mb.place_label(l);
        mb.jmp(l);
        mb.end_func();
        Limits lim; lim.instruction_budget = 500;
        i64 r = 0; u64 n = 0;
        CHECK(run(mb, "spin", {}, &r, lim, &n) == Trap::BudgetExhausted);
        CHECK(n == 500);                                   // exact, deterministic
    }
    {   // unbounded recursion bounded by the call-depth cap
        vm::ModuleBuilder mb;
        u32 fi = mb.begin_func("rec", 0, 0);
        mb.call(fi); mb.op(Op::Ret);
        mb.end_func();
        Limits lim; lim.max_call_depth = 32;
        i64 r = 0;
        CHECK(run(mb, "rec", {}, &r, lim) == Trap::CallDepth);
    }
    {   // Halt returns top-of-stack and reports Halted (a SUCCESS trap)
        vm::ModuleBuilder mb;
        mb.begin_func("h", 0, 0);
        mb.push_i64(7); mb.op(Op::Halt);
        mb.push_i64(999); mb.op(Op::Ret);                  // unreachable
        mb.end_func();
        i64 r = 0;
        CHECK(run(mb, "h", {}, &r) == Trap::Halted && r == 7);
    }
    {   // falling off the end == implicit return of TOS
        vm::ModuleBuilder mb;
        mb.begin_func("fall", 0, 0);
        mb.push_i64(9);                                    // no Ret/Halt
        mb.end_func();
        i64 r = 0;
        CHECK(run(mb, "fall", {}, &r) == Trap::Finished && r == 9);
    }
}

// ---- host-call ABI ------------------------------------------------------
i64 host_add(vm::HostContext&, const i64* a, u32 n) noexcept {
    return (n == 2) ? (a[0] + a[1]) : -1;
}
i64 host_err(vm::HostContext& c, const i64*, u32) noexcept {
    c.error = true; return 0;
}
i64 host_bump(vm::HostContext& c, const i64*, u32) noexcept {
    if (c.user) ++*static_cast<int*>(c.user);
    return 0;
}

void test_hostcall() {
    vm::HostFnDesc descs[] = {
        { "add",  host_add,  2 },
        { "err",  host_err,  0 },
        { "bump", host_bump, 0 },
    };
    {   // normal host call
        vm::ModuleBuilder mb;
        mb.begin_func("useh", 0, 0);
        mb.push_i64(20); mb.push_i64(22); mb.hostcall(0, 2); mb.op(Op::Ret);
        mb.end_func();
        std::vector<u8> by = mb.finish();
        std::string err;
        auto m = vm::load(by.data(), by.size(), Limits{}, &err);
        CHECK(m != nullptr);
        auto v = vm::VM::create(Limits{});
        v->set_host_fns(descs, 3);
        i64 fi = vm::module_find_func(*m, "useh");
        i64 r = 0;
        CHECK(v->call(*m, static_cast<u32>(fi), nullptr, 0, &r) == Trap::Finished);
        CHECK(r == 42);
    }
    {   // arity mismatch (module says 3, registered says 2) -> BadCall
        vm::ModuleBuilder mb;
        mb.begin_func("bad", 0, 0);
        mb.push_i64(1); mb.push_i64(2); mb.push_i64(3);
        mb.hostcall(0, 3); mb.op(Op::Ret);
        mb.end_func();
        std::vector<u8> by = mb.finish();
        std::string err;
        auto m = vm::load(by.data(), by.size(), Limits{}, &err);
        CHECK(m != nullptr);                                // verifier ok (arity<=32)
        auto v = vm::VM::create(Limits{});
        v->set_host_fns(descs, 3);
        i64 fi = vm::module_find_func(*m, "bad");
        i64 r = 0;
        CHECK(v->call(*m, static_cast<u32>(fi), nullptr, 0, &r) == Trap::BadCall);
    }
    {   // host fn requesting an abort -> HostError
        vm::ModuleBuilder mb;
        mb.begin_func("er", 0, 0);
        mb.hostcall(1, 0); mb.op(Op::Ret);
        mb.end_func();
        std::vector<u8> by = mb.finish();
        std::string err;
        auto m = vm::load(by.data(), by.size(), Limits{}, &err);
        auto v = vm::VM::create(Limits{});
        v->set_host_fns(descs, 3);
        i64 fi = vm::module_find_func(*m, "er");
        i64 r = 0;
        CHECK(v->call(*m, static_cast<u32>(fi), nullptr, 0, &r) == Trap::HostError);
    }
    {   // set_user pointer reaches the host fn (capability via host only)
        vm::ModuleBuilder mb;
        mb.begin_func("bm", 0, 0);
        mb.hostcall(2, 0); mb.hostcall(2, 0); mb.op(Op::Ret);
        mb.end_func();
        std::vector<u8> by = mb.finish();
        std::string err;
        auto m = vm::load(by.data(), by.size(), Limits{}, &err);
        auto v = vm::VM::create(Limits{});
        v->set_host_fns(descs, 3);
        int counter = 0;
        v->set_user(&counter);
        i64 fi = vm::module_find_func(*m, "bm");
        i64 r = 0;
        CHECK(v->call(*m, static_cast<u32>(fi), nullptr, 0, &r) == Trap::Finished);
        CHECK(counter == 2);
    }
    {   // out-of-range host index -> BadCall
        vm::ModuleBuilder mb;
        mb.begin_func("hr", 0, 0);
        mb.hostcall(9, 0); mb.op(Op::Ret);
        mb.end_func();
        std::vector<u8> by = mb.finish();
        std::string err;
        auto m = vm::load(by.data(), by.size(), Limits{}, &err);
        auto v = vm::VM::create(Limits{});
        v->set_host_fns(descs, 3);
        i64 fi = vm::module_find_func(*m, "hr");
        i64 r = 0;
        CHECK(v->call(*m, static_cast<u32>(fi), nullptr, 0, &r) == Trap::BadCall);
    }
}

// ---- verifier rejections (the security boundary) ------------------------
void test_verifier_rejects() {
    // A known-good module to corrupt.
    vm::ModuleBuilder good;
    good.begin_func("ok", 0, 0);
    good.push_i64(1); good.op(Op::Ret);
    good.end_func();
    std::vector<u8> base = good.finish();
    CHECK(loads_ok(base));                                  // sanity

    {   // bad magic
        std::vector<u8> b = base; b[0] ^= 0xFFu;
        CHECK(!loads_ok(b));
    }
    {   // bad version
        std::vector<u8> b = base; b[4] = 9; b[5] = 9;
        CHECK(!loads_ok(b));
    }
    {   // too small
        std::vector<u8> b(10, 0u);
        CHECK(!loads_ok(b));
    }
    {   // size mismatch (truncate one byte)
        std::vector<u8> b = base; b.pop_back();
        CHECK(!loads_ok(b));
    }
    {   // unknown opcode
        CHECK(!loads_ok(raw_mod(0, 0, 0, "x", { 0xEEu })));
    }
    {   // truncated immediate (PushI64 with no payload)
        CHECK(!loads_ok(raw_mod(0, 0, 0, "x",
              { static_cast<u8>(Op::PushI64) })));
    }
    {   // local index out of range (num_locals = 1, slot 3)
        vm::ModuleBuilder mb;
        mb.begin_func("li", 0, 1);
        mb.load_local(3); mb.op(Op::Ret);
        mb.end_func();
        CHECK(!loads_ok(mb.finish()));
    }
    {   // call index out of range (only 1 func, call #7)
        vm::ModuleBuilder mb;
        mb.begin_func("ci", 0, 0);
        mb.call(7); mb.op(Op::Ret);
        mb.end_func();
        CHECK(!loads_ok(mb.finish()));
    }
    {   // operand-stack underflow (Add with empty stack)
        vm::ModuleBuilder mb;
        mb.begin_func("su", 0, 0);
        mb.op(Op::Add); mb.op(Op::Ret);
        mb.end_func();
        CHECK(!loads_ok(mb.finish()));
    }
    {   // operand-stack overflow vs the verify-time limit
        vm::ModuleBuilder mb;
        mb.begin_func("so", 0, 0);
        for (int i = 0; i < 40; ++i) mb.push_i64(i);
        mb.op(Op::Ret);
        mb.end_func();
        std::vector<u8> by = mb.finish();
        std::string err;
        Limits lim; lim.max_stack = 8;                      // 40 pushes > 8
        CHECK(vm::load(by.data(), by.size(), lim, &err) == nullptr);
    }
    {   // jump target out of range
        std::vector<u8> code = { static_cast<u8>(Op::Jmp),
                                 0xFFu, 0xFFu, 0xFFu, 0x3Fu };  // rel huge
        CHECK(!loads_ok(raw_mod(0, 0, 0, "j", code)));
    }
    {   // jump into the middle of a PushI64 immediate
        std::vector<u8> code;
        code.push_back(static_cast<u8>(Op::PushI64));       // [0..8]
        for (int i = 0; i < 8; ++i) code.push_back(0u);
        code.push_back(static_cast<u8>(Op::Jmp));           // [9..13]
        const i32 rel = -11;                                // target = 14-11 = 3
        for (int i = 0; i < 4; ++i)
            code.push_back(static_cast<u8>((static_cast<u32>(rel) >> (8 * i)) & 0xFFu));
        CHECK(!loads_ok(raw_mod(0, 0, 0, "jm", code)));
    }
    {   // mem_pages over the limit
        std::vector<u8> b = raw_mod(9999, 0, 0, "m",
                            { static_cast<u8>(Op::Ret) });
        std::string err;
        Limits lim; lim.max_mem_pages = 4;
        CHECK(vm::load(b.data(), b.size(), lim, &err) == nullptr);
    }
    {   // code_len over kMaxCodeLen — rejected at the descriptor scan,
        // before any N-sized allocation. Closes the verifier u32-overflow
        // class (ip+1+imm and the vector(N+1) height map can't wrap once
        // N is bounded). code_len is at byte offset 28 (hdr 20 + the
        // num_params/num_locals descriptor words).
        auto patch_clen = [&](u32 v) {
            std::vector<u8> b = base;
            for (usize i = 0; i < 4u; ++i)
                b[28u + i] = static_cast<u8>(
                    (v >> (8u * static_cast<u32>(i))) & 0xFFu);
            return b;
        };
        CHECK(!loads_ok(patch_clen(vm::kMaxCodeLen + 1u)));
        CHECK(!loads_ok(patch_clen(0xFFFFFFFFu)));
        // Exactly at the cap passes the cap check (then fails the
        // exact-size check) — proves the bound is inclusive.
        std::string e;
        std::vector<u8> at = patch_clen(vm::kMaxCodeLen);
        CHECK(vm::load(at.data(), at.size(), Limits{}, &e) == nullptr);
        CHECK(e == "size mismatch");
    }
}

// ---- trap_name table ----------------------------------------------------
void test_trap_names() {
    CHECK(std::string(vm::trap_name(Trap::None))            == "None");
    CHECK(std::string(vm::trap_name(Trap::BadModule))       == "BadModule");
    CHECK(std::string(vm::trap_name(Trap::OutOfBoundsMemory))== "OutOfBoundsMemory");
    CHECK(std::string(vm::trap_name(Trap::BudgetExhausted)) == "BudgetExhausted");
    CHECK(std::string(vm::trap_name(Trap::Halted))          == "Halted");
    CHECK(std::string(vm::trap_name(Trap::Finished))        == "Finished");
}

}  // namespace

int main() {
    test_basic();
    test_loop();
    test_recursion();
    test_memory();
    test_float();
    test_int_traps();
    test_limits_and_terminators();
    test_hostcall();
    test_verifier_rejects();
    test_trap_names();

    if (g_fail == 0) {
        cardinal::log::infof("vmtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("vmtest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
