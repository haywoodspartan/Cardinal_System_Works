# Gameplay kit regression suite: quaternion transform ops (actor::xf),
# core slerp / look_rotation, the TimerSet scheduler, and every Kit
# GameActor (Rotator / Orbiter / LookAt / WaypointMover / Lifetime /
# Spawner) driven through a real Game with the inspector's reflected-
# property path. Pure CPU, deterministic, headless.
add_executable(Cardinal_Test_GameKit main.cpp)
target_link_libraries(Cardinal_Test_GameKit PRIVATE
    cardinal::game
    cardinal::actor
    cardinal::sim
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_GameKit PROPERTIES FOLDER tests)

add_test(NAME game_kit COMMAND Cardinal_Test_GameKit)
