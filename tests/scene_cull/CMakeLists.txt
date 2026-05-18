# Deterministic scene chunk-visibility regression suite. assign_chunks
# bins every entity into a world chunk (floor division — correct for
# negative coords, unlike naive truncation) and is_entity_visible gates
# the renderer's streaming cull on the visible-chunk set. A regression
# here silently culls visible entities or fails to cull off-region ones
# — invisible until it ships as popping / wasted draws. Pure CPU, no
# Mesh / rhi::Device (the cull predicate path), so fully headless +
# deterministic; builds on the already-locked ChunkCoord/ChunkSet.
add_executable(Cardinal_Test_SceneCull main.cpp)
target_link_libraries(Cardinal_Test_SceneCull PRIVATE
    cardinal::scene
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_SceneCull PROPERTIES FOLDER tests)

add_test(NAME scene_cull COMMAND Cardinal_Test_SceneCull)
