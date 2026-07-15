# cardinal::ice regression suite — pure CPU, deterministic. Covers the node
# math against hand-computed references, the compile-time validation (arity /
# type / cycle), and the operator-dependency-graph contract: incremental
# re-evaluation counts after const edits / input touches / point-count changes.
add_executable(Cardinal_Test_Ice main.cpp)
target_link_libraries(Cardinal_Test_Ice PRIVATE
    cardinal::ice
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Ice PROPERTIES FOLDER tests)

add_test(NAME ice COMMAND Cardinal_Test_Ice)
