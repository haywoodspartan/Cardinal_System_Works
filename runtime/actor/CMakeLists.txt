add_library(cardinal_actor STATIC
    src/component.cpp
    src/actor.cpp
    src/world.cpp
    include/cardinal/actor/component.hpp
    include/cardinal/actor/actor.hpp
    include/cardinal/actor/world.hpp)
add_library(cardinal::actor ALIAS cardinal_actor)

target_include_directories(cardinal_actor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_actor
    PUBLIC  cardinal::core cardinal::scene cardinal::compile_flags)

set_target_properties(cardinal_actor PROPERTIES FOLDER runtime)
