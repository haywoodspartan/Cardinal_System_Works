add_library(cardinal_level STATIC
    src/level.cpp
    include/cardinal/level/level.hpp)
add_library(cardinal::level ALIAS cardinal_level)

target_include_directories(cardinal_level PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_level
    PUBLIC  cardinal::core cardinal::actor cardinal::scene
            cardinal::compile_flags)

set_target_properties(cardinal_level PROPERTIES FOLDER runtime)
