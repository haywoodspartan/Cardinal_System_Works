# Deterministic world-partition regression suite. WorldPartition::tick is
# the cell-streaming brain: it unions every active viewer's needs, applies
# load/unload hysteresis (load_radius < unload_radius so cells don't churn
# at the boundary), honours Always cells, and enforces a soft resident cap
# with lowest-priority LRU eviction. A regression silently breaks world
# streaming — cells never load (holes in the world), never evict (VRAM
# blowup), or thrash at the radius seam. The decision path is pure CPU and
# synchronous (tick has no threads). Vision/Frustum coupling is owned by
# the geom suite and scoped out here (only no-viewer / distance-triggered
# DistanceOrVision paths, which need no frustum semantics, are pinned).
# Point-AABBs make every closest-distance an exact perfect square. Exit
# 0 = all pass.
add_executable(Cardinal_Test_Partition main.cpp)
target_link_libraries(Cardinal_Test_Partition PRIVATE
    cardinal::partition
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Partition PROPERTIES FOLDER tests)

add_test(NAME partition COMMAND Cardinal_Test_Partition)
