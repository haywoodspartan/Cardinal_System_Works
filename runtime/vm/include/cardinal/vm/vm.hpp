#pragma once

// =============================================================================
// Cardinal — sandboxed bytecode virtual machine.
//
// A small, deterministic STACK machine used to run untrusted "script" code
// with *capability* sandboxing — strictly stronger than the process-only
// isolation in cardinal::sandbox:
//
//   * No ambient authority. Bytecode has NO syscalls. The only way it can
//     affect the outside world is HOSTCALL into a host-registered, fixed
//     allowlist of functions. Remove a host fn -> that capability is gone.
//   * Bounded memory. A single fixed linear-memory arena (N * 64 KiB pages,
//     capped). Every load/store is bounds-checked. No raw pointers escape.
//   * Bounded compute. A per-call instruction budget ("gas"); operand-stack
//     and call-stack depth caps. Exceeding any of them traps cleanly.
//   * Verified up front. load() statically rejects malformed modules:
//     out-of-range offsets, bad opcodes, jumps that don't land on an
//     instruction boundary, local/func indices out of range, and operand
//     stack underflow/overflow/merge-mismatch (abstract height pass).
//   * Deterministic. Same module + same inputs + same host fns => identical
//     result and identical instruction count, on any machine. No time, no
//     RNG, no threads unless a host fn introduces them.
//
// This is the execution core. The cardinal::sandbox runner hosts it inside
// the existing child process (OP_ATTACH_VM), so a VM bug only kills the
// child while the capability model contains the script regardless.
//
// C/C++ FRONTEND SEAM
// -------------------
// Producing this bytecode FROM C/C++ is a separate (large) compiler-backend
// effort and is intentionally NOT in this module. ModuleBuilder below is the
// stable target API a frontend (or the test suite, or a hand assembler)
// emits into; load() is the contract it must satisfy. cppscript could grow a
// "-emit-cardinal-vm" path that drives ModuleBuilder instead of producing a
// native DLL.
//
// BINARY MODULE FORMAT  (all integers little-endian)
// --------------------------------------------------
//   Header
//     u32 magic   = 0x314D5643   ('C','V','M','1')
//     u32 version = 1
//     u32 flags   = 0            (reserved)
//     u32 mem_pages              (linear memory size, in 64 KiB pages)
//     u32 num_funcs
//   Func descriptor table  [num_funcs]
//     u32 num_params             (params occupy local slots 0..num_params-1)
//     u32 num_locals             (TOTAL locals, includes params; >= num_params)
//     u32 code_len               (bytes)
//     u32 name_len               (bytes; 0 = anonymous)
//   Body  (concatenated, in func order)
//     for each func:  name[name_len]  then  code[code_len]
//   Total size must equal header + table + sum(name_len + code_len).
//
// INSTRUCTION SET  (1-byte opcode, then LE immediates; see Op below)
// Cells are untyped 64-bit; opcodes choose the interpretation. Signed
// integer ops are two's-complement i64; float ops are IEEE-754 f64.
// =============================================================================

#include <cardinal/core/types.hpp>       // u8..i64, cardinal::string/unique_ptr
#include <cardinal/core/containers.hpp>  // cardinal::vector

namespace cardinal::vm {

inline constexpr u32 kMagic       = 0x314D5643u;   // "CVM1"
inline constexpr u32 kVersion     = 1u;
inline constexpr u32 kPageSize    = 65536u;        // 64 KiB linear-memory page
inline constexpr u32 kMaxFuncs    = 4096u;
// Hard per-function bytecode cap. Far larger than any real function, yet
// ~256x below 2^32 so every N-based offset computation in the verifier
// (ip + 1 + imm, N + 1, jump targets) is provably free of u32 overflow.
inline constexpr u32 kMaxCodeLen  = 16u * 1024u * 1024u;   // 16 MiB

// ---- Opcodes --------------------------------------------------------------
enum class Op : u8 {
    Nop        = 0x00,
    Halt       = 0x01,             // stop; result = top-of-stack (or 0)
    Drop       = 0x02,
    Dup        = 0x03,
    PushI64    = 0x04,             // imm i64
    PushF64    = 0x05,             // imm f64-bits (u64)
    LoadLocal  = 0x06,             // imm u16
    StoreLocal = 0x07,             // imm u16
    Add        = 0x08, Sub = 0x09, Mul = 0x0A, Div = 0x0B, Mod = 0x0C,
    Neg        = 0x0D,
    And        = 0x0E, Or  = 0x0F, Xor = 0x10, Shl = 0x11, Shr = 0x12,
    Not        = 0x13,
    Eq         = 0x14, Ne  = 0x15, Lt  = 0x16, Le  = 0x17, Gt  = 0x18, Ge = 0x19,
    FAdd       = 0x1A, FSub = 0x1B, FMul = 0x1C, FDiv = 0x1D,
    I2F        = 0x1E, F2I  = 0x1F,
    Jmp        = 0x20,             // imm i32 (rel to end of immediate)
    Jz         = 0x21,             // imm i32; pop cond, branch if == 0
    Jnz        = 0x22,             // imm i32; pop cond, branch if != 0
    Call       = 0x23,             // imm u32 func index
    Ret        = 0x24,             // return; result = top-of-stack (or 0)
    MLoad8     = 0x25, MLoad16 = 0x26, MLoad32 = 0x27, MLoad64 = 0x28,
    MStore8    = 0x29, MStore16 = 0x2A, MStore32 = 0x2B, MStore64 = 0x2C,
    HostCall   = 0x2D,             // imm u32 host index, then u8 arity
                                   // (arity is verified statically and must
                                   //  equal the registered HostFnDesc::arity)
    MemSize    = 0x2E,             // push linear-memory size in bytes
};

// ---- Traps (clean, non-crashing failure modes) ----------------------------
enum class Trap : u32 {
    None = 0,
    BadModule,          // load()/verify rejected the module
    BadOpcode,          // unknown opcode reached at runtime (shouldn't if verified)
    StackOverflow,      // operand stack exceeded Limits::max_stack
    StackUnderflow,     // pop on empty operand stack
    CallDepth,          // call stack exceeded Limits::max_call_depth
    OutOfBoundsMemory,  // load/store outside linear memory
    DivByZero,          // integer Div/Mod by zero
    BadCall,            // call/hostcall index out of range
    BudgetExhausted,    // instruction budget hit zero
    HostError,          // a host fn returned an error sentinel request
    Halted,             // explicit Halt — NOT an error; result is valid
    Finished,           // top-level RET / fell off end — result is valid
};

const char* trap_name(Trap t) noexcept;

// ---- Host call ABI --------------------------------------------------------
// The single, narrow door to the outside world. A host fn gets its popped
// args (already i64 cells; reinterpret f64 bits yourself if needed) and
// returns one i64 cell pushed back. Set ctx.error to abort the VM with
// Trap::HostError. ctx.user is the embedder pointer from VM::set_user.
class VM;
struct HostContext {
    VM*   vm{nullptr};
    void* user{nullptr};
    bool  error{false};      // set true to trap the VM (Trap::HostError)
};
using HostFn = i64 (*)(HostContext& ctx, const i64* args, u32 nargs) noexcept;

struct HostFnDesc {
    const char* name{nullptr};   // for diagnostics only
    HostFn      fn{nullptr};
    u32         arity{0};        // exact # of args popped / supplied
};

// ---- Resource limits (the sandbox bounds) ---------------------------------
struct Limits {
    u64 instruction_budget{1'000'000};   // per top-level call()
    u32 max_stack{1024};                 // operand-stack cells
    u32 max_call_depth{64};
    u32 max_mem_pages{16};               // hard cap regardless of module ask
};

// ---- Verified module ------------------------------------------------------
// Opaque by design — its layout is an implementation detail. The custom
// deleter is defined in the .cpp (where Module is complete) so callers can
// hold a ModulePtr without seeing the definition.
struct Module;
struct ModuleDeleter { void operator()(Module*) const noexcept; };
using  ModulePtr = cardinal::unique_ptr<Module, ModuleDeleter>;

// Verify + parse. On success returns a Module; on failure returns nullptr and
// (if out_err) a human-readable reason. Never executes anything.
ModulePtr load(const u8* bytes, usize len,
               const Limits& limits, cardinal::string* out_err);

u32         module_func_count(const Module& m) noexcept;
// -1 if not found.
i64         module_find_func(const Module& m, const char* name) noexcept;

// ---- The VM ---------------------------------------------------------------
class VM {
public:
    static cardinal::unique_ptr<VM> create(const Limits& limits);
    virtual ~VM() = default;
    VM(const VM&)            = delete;
    VM& operator=(const VM&) = delete;

    // Host fn table is borrowed (must outlive the VM). Indices are positional.
    virtual void set_host_fns(const HostFnDesc* fns, u32 count) noexcept = 0;
    virtual void set_user(void* user) noexcept = 0;

    // Run module function `func_index` with `args` (copied into locals
    // 0..nargs-1; nargs must be <= that function's num_locals and >=
    // num_params is enforced by clamping/zero-fill of remaining locals).
    // Returns the trap reason; Halted/Finished are SUCCESS and *out_ret (if
    // non-null) holds the result cell. Budget is refreshed each call().
    virtual Trap call(const Module& m, u32 func_index,
                      const i64* args, u32 nargs, i64* out_ret) noexcept = 0;

    // Linear memory view (for host fns + tests). Size is mem_pages*64KiB.
    virtual u8*   memory()      noexcept = 0;
    virtual usize memory_size() const noexcept = 0;

    // Instructions executed by the most recent call() (determinism probe).
    virtual u64 instructions_executed() const noexcept = 0;

protected:
    VM() = default;
};

// ---------------------------------------------------------------------------
// ModuleBuilder — emit the binary format without hand-encoding bytes. This is
// the seam a C/C++ frontend / assembler / the tests target.
// ---------------------------------------------------------------------------
class ModuleBuilder {
public:
    ModuleBuilder();
    ~ModuleBuilder();
    ModuleBuilder(const ModuleBuilder&)            = delete;
    ModuleBuilder& operator=(const ModuleBuilder&) = delete;

    void set_mem_pages(u32 pages) noexcept;

    // Begin a function; returns its index. End with end_func().
    u32  begin_func(const char* name, u32 num_params, u32 num_locals) noexcept;
    void end_func() noexcept;

    // --- emit helpers (operate on the currently-open function) ---
    void op(Op o) noexcept;                        // operand-less opcodes
    void push_i64(i64 v) noexcept;
    void push_f64(f64 v) noexcept;
    void load_local(u16 slot) noexcept;
    void store_local(u16 slot) noexcept;
    void call(u32 func_index) noexcept;
    void hostcall(u32 host_index, u8 arity) noexcept;

    // Branches use named labels resolved by finish().
    u32  make_label() noexcept;                    // returns label id
    void place_label(u32 label) noexcept;          // bind it at the current ip
    void jmp(u32 label) noexcept;
    void jz(u32 label) noexcept;
    void jnz(u32 label) noexcept;

    // Serialise everything to the binary module format.
    cardinal::vector<u8> finish();

private:
    struct Impl;
    cardinal::unique_ptr<Impl> p_;
};

}  // namespace cardinal::vm
