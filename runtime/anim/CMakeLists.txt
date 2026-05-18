add_library(cardinal_anim STATIC
    src/anim.cpp
    include/cardinal/anim/anim.hpp)
add_library(cardinal::anim ALIAS cardinal_anim)

target_include_directories(cardinal_anim PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_anim
    PUBLIC  cardinal::core cardinal::scene cardinal::compile_flags)

set_target_properties(cardinal_anim PROPERTIES FOLDER runtime)
