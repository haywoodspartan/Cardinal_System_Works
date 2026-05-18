# Deterministic ChunkCoord / ChunkCoordHash / ChunkSet regression
# suite. This is the spatial-hash keystone keyed on by scene
# visible-chunk culling AND world-partition streaming. A hash or
# equality regression silently mis-buckets chunks — objects vanish
# (cull) or chunks never stream/evict — invisibly until shipped. The
# header explicitly fixed an old "u64 packing" bug where same-xz /
# different-y chunks collided; this suite pins that contract. Pure +
# header-only + deterministic.
add_executable(Cardinal_Test_Spatial main.cpp)
target_link_libraries(Cardinal_Test_Spatial PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Spatial PROPERTIES FOLDER tests)

add_test(NAME spatial COMMAND Cardinal_Test_Spatial)
