add_library(cardinal_game STATIC
    src/reflection.cpp
    src/game.cpp
    src/timer.cpp
    src/kit.cpp
    include/cardinal/game/timer.hpp
    include/cardinal/game/kit.hpp
    include/cardinal/game/reflection.hpp
    include/cardinal/game/game_actor.hpp
    include/cardinal/game/game.hpp)
add_library(cardinal::game ALIAS cardinal_game)

target_include_directories(cardinal_game PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_game
    PUBLIC  cardinal::core cardinal::actor cardinal::sim cardinal::scene
            cardinal::compile_flags)

set_target_properties(cardinal_game PROPERTIES FOLDER runtime)
