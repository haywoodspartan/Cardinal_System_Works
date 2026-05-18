add_library(cardinal_nav STATIC
    src/nav.cpp
    include/cardinal/nav/nav.hpp)
add_library(cardinal::nav ALIAS cardinal_nav)

target_include_directories(cardinal_nav PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_nav
    PUBLIC  cardinal::core cardinal::compile_flags)

set_target_properties(cardinal_nav PROPERTIES FOLDER runtime)
