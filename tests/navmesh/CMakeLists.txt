# Deterministic polygon-navmesh regression suite. navmesh is the richer
# AI-movement backbone (complements the grid nav suite): triangle soup →
# auto edge-shared adjacency → polygon A* over centroids → funnel
# string-pulling to straight waypoints. A regression silently breaks
# agent navigation — wrong adjacency (no path), bad centroid cost
# (suboptimal route), or a funnel that drops the start/goal. Pure CPU,
# fully deterministic. The suite pins construction-GUARANTEED invariants
# (exact shared-edge neighbour indices, centroids, centroid-nearest
# nearest_poly, A* found / cost² / poly_visited on hand-traceable tiny
# meshes, and the funnel's universal endpoint + finiteness + count
# contract) rather than fragile hand-predicted waypoint coordinates.
# Exit 0 = all pass.
add_executable(Cardinal_Test_Navmesh main.cpp)
target_link_libraries(Cardinal_Test_Navmesh PRIVATE
    cardinal::navmesh
    cardinal::scene
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Navmesh PROPERTIES FOLDER tests)

add_test(NAME navmesh COMMAND Cardinal_Test_Navmesh)
