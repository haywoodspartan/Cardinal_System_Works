# Deterministic game-framework regression suite. Covers the two pure
# surfaces the editor + Play button depend on: the reflection
# ClassRegistry / PROP_* descriptors (inspect/save/load every authored
# actor) and the Game Stopped/Playing/Paused lifecycle + spawn_class
# begin_play/end_play broadcast. A regression silently breaks the
# inspector or the Play button. Pure CPU, deterministic. Exit 0 = pass.
add_executable(Cardinal_Test_Game main.cpp)
target_link_libraries(Cardinal_Test_Game PRIVATE
    cardinal::game
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Game PROPERTIES FOLDER tests)

add_test(NAME game COMMAND Cardinal_Test_Game)
