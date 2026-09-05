# AEGIS raster benchmark + at-scale GPU equivalence. Times the brute per-pixel
# triangle-loop V-Buffer kernel against the tile-binned TriangleBinPass +
# binned kernel on a REAL device (both backends), and proves the two GPU paths
# produce identical prim / depth / material channels on a 960×540, 20k-tri
# scene (the 32×32 gpu_compute harness can't show the scaling). SKIPs (exit 0)
# without a capable GPU or async-compute queue so the GPU-less CI stays green.
add_executable(Cardinal_Test_AegisBench main.cpp)
target_link_libraries(Cardinal_Test_AegisBench PRIVATE
    cardinal::render
    cardinal::rhi
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_AegisBench PROPERTIES FOLDER tests)

add_test(NAME aegis_bench COMMAND Cardinal_Test_AegisBench)
