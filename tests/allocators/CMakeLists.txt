# Cardinal — allocator regression suite. Exercises LinearAllocator (atomic
# bump + ScopedArenaMarker + make<T>/make_array) and PoolAllocator<T>
# (slab + intrusive free list + make/destroy), including a concurrent
# alloc stress run for both.
add_executable(Cardinal_Test_Allocators main.cpp)
target_link_libraries(Cardinal_Test_Allocators PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Allocators PROPERTIES FOLDER tests)

add_test(NAME allocators COMMAND Cardinal_Test_Allocators)
