# Blueprint round-trip suite — graph<->source codegen + parser, nesting
# flatten, idempotent canonicalize, literal fidelity, error reporting. Headless.
add_executable(Cardinal_Test_Blueprint main.cpp)
target_link_libraries(Cardinal_Test_Blueprint PRIVATE
    cardinal::blueprint
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Blueprint PROPERTIES FOLDER tests)

add_test(NAME blueprint COMMAND Cardinal_Test_Blueprint)
