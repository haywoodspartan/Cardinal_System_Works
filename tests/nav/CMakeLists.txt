# Deterministic navigation/pathfinding regression suite. nav::Grid +
# PathQuery (A* octile) + compute_flow_field are the AI-movement
# backbone — a regression silently breaks all navigation (agents can't
# path, take wrong/longer routes, or the flow field stops pointing at
# the goal). Pure, headless, fully deterministic given grid + start +
# goal + diagonal flag; the Studio nav panel drives the same code. Exact
# small-grid paths/costs are pinned, plus determinism + scratch reuse.
add_executable(Cardinal_Test_Nav main.cpp)
target_link_libraries(Cardinal_Test_Nav PRIVATE
    cardinal::nav
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Nav PROPERTIES FOLDER tests)

add_test(NAME nav COMMAND Cardinal_Test_Nav)
