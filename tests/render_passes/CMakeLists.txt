# XSI-style Render Passes & Partitions — pure-CPU regression suite: disjoint
# partition assignment, non-destructive apply/restore against a live
# scene::Scene, current-pass semantics, and the text serialize round-trip.
add_executable(Cardinal_Test_RenderPasses main.cpp)
target_link_libraries(Cardinal_Test_RenderPasses PRIVATE
    cardinal::render
    cardinal::scene
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_RenderPasses PROPERTIES FOLDER tests)

add_test(NAME render_passes COMMAND Cardinal_Test_RenderPasses)
