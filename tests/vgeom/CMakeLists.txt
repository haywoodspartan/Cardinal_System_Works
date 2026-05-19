# Deterministic virtualized-geometry (vgeom) regression suite.
# cook() builds the LOD cluster Hierarchy a mesh renders through; select()
# picks the per-frame cut. A regression silently corrupts every vgeom mesh
# — broken parent/child links, out-of-bounds vertex slices, a non-
# deterministic or mis-accounted cut. Pure CPU, single-threaded, fully
# deterministic. Exit 0 = all pass.
add_executable(Cardinal_Test_VGeom main.cpp)
target_link_libraries(Cardinal_Test_VGeom PRIVATE
    cardinal::vgeom
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_VGeom PROPERTIES FOLDER tests)

add_test(NAME vgeom COMMAND Cardinal_Test_VGeom)
