# Deterministic CPU mesh-authoring regression suite. edit::mesh_ops is the
# level editor's quick-edit toolbox: primitive generators (box / plane /
# sphere / cylinder / cone / torus / disk) plus pure topology ops
# (subdivide, mirror, decimate-cluster, Laplacian smooth, recompute
# normals, bake-transform, tint, append, stats). Every op returns a FRESH
# vector and never mutates its input — a regression there silently
# corrupts an artist's mesh or leaks an edit back into the source asset.
# All ops are pure CPU float math: vertex counts, AABBs, surface areas,
# midpoint splits, winding reversal and the no-mutation contract are all
# exactly pinned (counts are integer-exact; floats use an epsilon). The
# transcendental primitives (sphere/cylinder/cone/torus) are pinned by
# count + radius/extent invariants rather than per-vertex trig. Exit 0 =
# all pass.
add_executable(Cardinal_Test_MeshOps main.cpp)
target_link_libraries(Cardinal_Test_MeshOps PRIVATE
    cardinal::edit
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_MeshOps PROPERTIES FOLDER tests)

add_test(NAME mesh_ops COMMAND Cardinal_Test_MeshOps)
