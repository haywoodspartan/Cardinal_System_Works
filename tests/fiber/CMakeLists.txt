# cardinal::fiber — the hand-rolled x86-64 context switch under the job
# system. Exercised directly, without the scheduler: stack allocation with
# its guard page, a fresh context's first entry, ping-pong switches, nested
# contexts, deep recursion on a fiber stack (stack probes against the
# installed TIB bounds), a C++ exception caught on a fiber stack, and
# floating-point register state across a switch. Each stage logs before it
# runs so a crash names the stage. SKIPs (exit 0) where CARDINAL_FIBER_ASM
# is 0 (non-x86-64). Exit 0 = pass.
add_executable(Cardinal_Test_Fiber main.cpp)
target_link_libraries(Cardinal_Test_Fiber PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Fiber PROPERTIES FOLDER tests)

add_test(NAME fiber COMMAND Cardinal_Test_Fiber)
