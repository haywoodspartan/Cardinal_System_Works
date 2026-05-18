# Deterministic world grid + chunk-streamer regression suite. WorldGrid
# is pure config + math (world↔chunk via floor division, AABB helpers,
# half-open in_world_bounds, cylindrical compute_visible_set). WorldStreamer
# diffs the visible set frame-over-frame and fires load/unload, with a
# camera-chunk no-move short-circuit, invalidate, and evict_all. A
# regression silently breaks open-world streaming — holes (chunks never
# load), leaks (never unload), or boundary thrash. Pure CPU, single-
# threaded, fully deterministic. (chunk_of uses floor(w*(1/size)); the
# suite asserts only mid-chunk coords so the 1/size float error can't
# flip a floor at an exact chunk boundary.) Exit 0 = all pass.
add_executable(Cardinal_Test_World main.cpp)
target_link_libraries(Cardinal_Test_World PRIVATE
    cardinal::world
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_World PROPERTIES FOLDER tests)

add_test(NAME world COMMAND Cardinal_Test_World)
