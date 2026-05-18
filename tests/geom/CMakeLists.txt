# Deterministic geometry regression suite. core::geom is the CPU
# spatial toolbox shared by physics broadphase, renderer frustum
# culling, partition, navmesh and AI. A silent regression here makes
# visible objects vanish (cull), collisions miss (BVH/overlap), or
# raycasts lie. Pure + deterministic, built on the now-locked core
# math. geom is merged into cardinal::core, so that's the only link
# dep; /W4 /WX via compile_flags.
add_executable(Cardinal_Test_Geom main.cpp)
target_link_libraries(Cardinal_Test_Geom PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Geom PROPERTIES FOLDER tests)

add_test(NAME geom COMMAND Cardinal_Test_Geom)
