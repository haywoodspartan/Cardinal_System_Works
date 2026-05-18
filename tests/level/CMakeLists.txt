# Deterministic level/HLOD regression suite. LevelManager spawns a
# reusable LevelInstance into an actor::World, composing the placement
# transform onto each template (translation + at*scale, rot + at.rot,
# scale * at.scale) — a regression mis-places every placed sub-level.
# despawn is deferred (marks actors dead via World::destroy, drops the
# placement immediately). build_hlod is a deterministic bottom-up greedy
# spatial clustering (collinear inputs make the tree hand-exact); a
# regression breaks LOD baking. select_hlod is a distance-band BFS that
# emits proxy vs leaf vs culled — a regression pops/over-draws geometry.
# Pure CPU, headless, fully deterministic. Exit 0 = all pass.
add_executable(Cardinal_Test_Level main.cpp)
target_link_libraries(Cardinal_Test_Level PRIVATE
    cardinal::level
    cardinal::actor
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Level PROPERTIES FOLDER tests)

add_test(NAME level COMMAND Cardinal_Test_Level)
