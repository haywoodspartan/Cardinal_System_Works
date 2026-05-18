# Deterministic scene-graph hierarchy regression suite. Scene's
# add/remove/find + parent/children + cycle-prevention back the editor
# outliner tree, drag-to-reparent, and every hierarchy walk. A
# cycle-detection regression hangs/corrupts traversal; a remove that
# fails to reparent children leaves dangling parent_ids. These ops are
# pure CPU (no Mesh/rhi::Device), so the suite is fully headless +
# deterministic. pick_entity / entity_world_aabb need a GPU Mesh and
# are out of scope here by design.
add_executable(Cardinal_Test_SceneGraph main.cpp)
target_link_libraries(Cardinal_Test_SceneGraph PRIVATE
    cardinal::scene
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_SceneGraph PROPERTIES FOLDER tests)

add_test(NAME scene_graph COMMAND Cardinal_Test_SceneGraph)
