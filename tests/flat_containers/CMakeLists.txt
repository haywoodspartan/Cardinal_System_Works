# Cardinal — flat_set / flat_map regression suite. Exercises the cache-tight
# associative containers (sorted-vector + std::lower_bound lookup), the
# bulk_insert merge path, the C++17/20 try_emplace / insert_or_assign /
# contains additions, and the stored-comparator (EBO) path.
add_executable(Cardinal_Test_FlatContainers main.cpp)
target_link_libraries(Cardinal_Test_FlatContainers PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_FlatContainers PROPERTIES FOLDER tests)

add_test(NAME flat_containers COMMAND Cardinal_Test_FlatContainers)
