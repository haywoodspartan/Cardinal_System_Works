add_library(cardinal_cine STATIC
    src/cine.cpp
    include/cardinal/cine/cine.hpp)
add_library(cardinal::cine ALIAS cardinal_cine)

target_include_directories(cardinal_cine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_cine
    PUBLIC  cardinal::core cardinal::anim cardinal::audio cardinal::compile_flags)

set_target_properties(cardinal_cine PROPERTIES FOLDER runtime)
