# Deterministic sandboxed-bytecode-VM regression suite. cardinal::vm is the
# capability sandbox under in-engine scripting: a verified stack machine
# with bounded memory/compute and a host-call allowlist. A regression here
# either (a) lets a malformed/hostile module past the verifier (a sandbox
# escape) or (b) drifts a deterministic result. This suite pins the
# ModuleBuilder->load->call round-trip: arithmetic, locals, control flow,
# recursion, linear-memory bounds, every Trap (DivByZero / OOB / budget /
# stack / call-depth / bad-call), the verifier's rejections (bad magic /
# version / size / opcode / jump / index / stack-imbalance), host-call ABI
# (incl. arity-mismatch and host-error), and bit-exact determinism +
# identical instruction counts across VM instances. Exit 0 = all pass.
add_executable(Cardinal_Test_VM main.cpp)
target_link_libraries(Cardinal_Test_VM PRIVATE
    cardinal::vm
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_VM PROPERTIES FOLDER tests)

add_test(NAME vm COMMAND Cardinal_Test_VM)
