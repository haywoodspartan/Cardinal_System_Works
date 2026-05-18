# JobSystem regression suite. The work-stealing job system is the
# parallel-execution substrate under net async loads, pack streaming,
# and the sim's parallel tick-group dispatch. A regression that DROPS
# or DOUBLE-RUNS a task silently corrupts those subsystems. Thread
# scheduling is nondeterministic, so every assertion here is a
# schedule-INDEPENDENT invariant (exactly-once execution, aggregate
# determinism across runs, wait() truly blocks, stats cross-check) —
# the same robust approach as the IdGen concurrency test.
add_executable(Cardinal_Test_Jobs main.cpp)
target_link_libraries(Cardinal_Test_Jobs PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Jobs PROPERTIES FOLDER tests)

add_test(NAME jobs COMMAND Cardinal_Test_Jobs)
