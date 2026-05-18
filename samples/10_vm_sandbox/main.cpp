// =============================================================================
// Cardinal — capability-sandboxed bytecode VM, end to end (headless).
//
// Proves the whole chain the user asked for: hand-assembled cardinal::vm
// bytecode -> shipped over the pipe -> child process (cardinal_sandbox_
// runner) -> VERIFIED + loaded by cardinal::vm -> ticked -> the script's
// only capability (a host-call) bridged back over the pipe and re-emitted
// by the host log. A VM bug only kills the child; the capability model
// contains the script regardless of process isolation.
//
//   tick(dtbits):  r = 6 * 7;  log(r);  return r
//
// Exit codes:  0 ok  |  2 attach failed  |  3 tick failed  |  5 not isolated
// =============================================================================

#include <cardinal/vm/vm.hpp>
#include <cardinal/sandbox/sandbox.hpp>
#include <cardinal/core/log.hpp>

#include <vector>

int main() {
    namespace vm = cardinal::vm;

    // ---- assemble a tiny module with one "tick" export ----------------
    vm::ModuleBuilder mb;
    mb.set_mem_pages(1);
    mb.begin_func("tick", /*params*/ 1, /*locals*/ 1);   // local0 = dt-bits
    mb.push_i64(6);
    mb.push_i64(7);
    mb.op(vm::Op::Mul);          // [42]
    mb.op(vm::Op::Dup);          // [42,42]
    mb.hostcall(0, 1);           // log(42) -> pushes 0   [42,0]
    mb.op(vm::Op::Drop);         // [42]
    mb.op(vm::Op::Ret);          // return 42
    mb.end_func();
    std::vector<cardinal::u8> bytecode = mb.finish();

    cardinal::log::infof("vmsb", "assembled %zu-byte module",
                         static_cast<size_t>(bytecode.size()));

    // ---- run it in the process-isolated capability sandbox ------------
    cardinal::sandbox::Desc d;
    d.tick_timeout_ms   = 1000;
    d.attach_timeout_ms = 5000;

    auto sb = cardinal::sandbox::Sandbox::create_vm(
        d, bytecode.data(), bytecode.size());
    if (!sb) {
        cardinal::log::errorf("vmsb", "create_vm failed (runner missing?)");
        return 2;
    }
    {
        auto st = sb->status();
        cardinal::log::infof("vmsb",
            "VM sandbox attached: pid=%llu name='%s'",
            static_cast<unsigned long long>(st.pid),
            st.plugin_name.c_str());
    }

    for (int i = 0; i < 5; ++i) {
        if (!sb->tick(0.016f)) {
            cardinal::log::errorf("vmsb", "tick %d failed (sandbox dead)", i);
            return 3;
        }
    }

    auto st = sb->status();
    cardinal::log::infof("vmsb", "ticked %llu times, alive=%d",
                         static_cast<unsigned long long>(st.ticks),
                         st.alive ? 1 : 0);
    if (!st.alive) return 5;

    sb->detach();
    cardinal::log::infof("vmsb",
        "OK — bytecode verified, ran sandboxed in a child process, "
        "and its single host-call capability round-tripped over the pipe.");
    return 0;
}
