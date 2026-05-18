add_library(cardinal_physics STATIC
    src/physics.cpp
    src/exotic.cpp
    include/cardinal/physics/physics.hpp
    include/cardinal/physics/exotic.hpp)
add_library(cardinal::physics ALIAS cardinal_physics)

target_include_directories(cardinal_physics PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_physics
    PUBLIC  cardinal::core cardinal::compile_flags)

set_target_properties(cardinal_physics PROPERTIES FOLDER runtime)
