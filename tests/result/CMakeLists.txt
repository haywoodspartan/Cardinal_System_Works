# Deterministic Result<T,E> regression suite. Result is the in-house,
# exception-free value/error type returned across net/cook/serial/pack/
# import and more. Its T-xor-E storage is a manually-managed union, so
# the highest-stakes property is LIFETIME correctness: every contained
# value constructed exactly once and destroyed exactly once across
# copy / move / assign / scope-exit. A regression there silently leaks
# or double-frees on every error path. Pure + header-only; an
# instrumented payload type makes the accounting exact.
add_executable(Cardinal_Test_Result main.cpp)
target_link_libraries(Cardinal_Test_Result PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Result PROPERTIES FOLDER tests)

add_test(NAME result COMMAND Cardinal_Test_Result)
