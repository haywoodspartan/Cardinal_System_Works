add_library(cardinal_project STATIC
    src/project.cpp
    include/cardinal/project/project.hpp)
add_library(cardinal::project ALIAS cardinal_project)

target_include_directories(cardinal_project PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_project
    PUBLIC  cardinal::core cardinal::compile_flags)

set_target_properties(cardinal_project PROPERTIES FOLDER runtime)
