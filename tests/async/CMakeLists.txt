# cardinal::async regression suite — the ergonomic future/continuation
# + parallel_for/invoke layer over JobSystem that pack::Archive::
# load_async and other streaming/per-frame paths build on. A regression
# in value propagation, exactly-once range coverage, or continuation
# firing silently breaks async asset loads + parallel phases. Schedule-
# independent invariants only (same robust approach as the jobs/idgen
# suites); the no-pool fallback path is fully deterministic. Phase C
# pins the then()-vs-resolve attach contract (the former load_async
# lost-wakeup): the continuation fires exactly once with the right
# value whichever side wins the race — 2000-iteration stress.
add_executable(Cardinal_Test_Async main.cpp)
target_link_libraries(Cardinal_Test_Async PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Async PROPERTIES FOLDER tests)

add_test(NAME async COMMAND Cardinal_Test_Async)
