add_library(cardinal_world STATIC
    src/world.cpp
    include/cardinal/world/world.hpp)
add_library(cardinal::world ALIAS cardinal_world)

target_include_directories(cardinal_world PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_world
    PUBLIC  cardinal::core cardinal::compile_flags)

set_target_properties(cardinal_world PROPERTIES FOLDER runtime)
